"""
fd_gpt2.py
==========
Forward-Only PACE Gradient Attribution (FD variant) for decoder-only
GPT-2 — vectorized, gradient-free, drop-in compatible with the
GPT-2 evaluation pipeline (xai_metrics_gpt2.py and run_eval_pg_gpt2.py).

Mirrors `fd.py` (the encoder FD variant) in structure and `paceg_gpt2.py`
in interface, so `run_eval_fd_gpt2.py` is structurally identical to
`run_eval_pg_gpt2.py`.

Algorithm
---------
For an input prompt Q (length Lq) and an answer A (length La, either
gold or greedy-generated), the full sequence has length T = Lq + La.

We define a *gate vector* g in [0, 1]^T, one entry per position:
    X_gated[i] = g_i * X[i] + (1 - g_i) * X_baseline[i]

  - Q positions are FREE (gates move from 0 to 1 along the path).
  - A positions are PINNED to g=1 (their embeddings stay identical to
    the original full-sequence embedding, because we are measuring
    p(A | Q_perturbed) and must not perturb the conditioning prefix).

The path is g^{(t)} = (t / (steps - 1)) * 1 over the free positions,
with pinned positions held at 1 throughout.

At each integration step t = 1..(steps-1), at base g^{(t-1)}:
    u_i^{(t)} = y_hat( g^{(t-1)} + h * e_i ) - y_hat( g^{(t-1)} )
where h = 1 / (steps - 1) is the natural step size.

Then:
    alpha_i^{(t)} = u_i^{(t)} / sum_j u_j^{(t)}
    PACE-FD_i     = sum_t alpha_i^{(t)} * Delta y_hat^{(t)}
    Delta y_hat^{(t)} = y_hat(g^{(t)}) - y_hat(g^{(t-1)})

The y_hat scalar is the joint log-probability of the answer:
    y_hat(g) = sum_{i=1..La} log p(a_i | x_gated at position Lq+i-1)

This preserves exact path-completeness:
    sum_i PACE-FD_i = sum_t Delta y_hat^{(t)} = y_hat(1) - y_hat(0)

Vectorization
-------------
All (steps - 1) integration steps are assembled into a single tensor of
shape (T_steps, Lq + 2, T) — for each step, one row for the base gate,
Lq rows for the probes (one per free Q position), one row for the joint
gate — then flattened to (T_steps * (Lq + 2), T) and pushed through the
model in chunks bounded by `chunk_size`.

Return dict
-----------
Keys match `paceg_gpt2.pace_gradient_gpt2` exactly:
    tokens, q_len, answer_positions, answer_ids,
    attributions, input_embed, base_embed, logits_full,
    predicted_answer, model, tokenizer, time
"""

import time
import random
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Module-level model cache (mirrors paceg_gpt2._CACHE)
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def get_model_tokenizer(model_name: str = "gpt2", device: str = "cpu"):
    """Return (model, tokenizer), loading from HuggingFace only once."""
    device = str(torch.device(device))
    key = (model_name, device)
    if key not in _CACHE:
        tok = GPT2TokenizerFast.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        mdl = GPT2LMHeadModel.from_pretrained(model_name)
        mdl.eval().to(device)
        _CACHE[key] = (mdl, tok)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Baseline embedding factory (matches paceg_gpt2._build_base_embed,
# returns tensor on `device` for FD's batched forward path)
# ---------------------------------------------------------------------------

def _build_base_embed(
    embed_layer: torch.nn.Embedding,
    input_embed: torch.Tensor,        # (1, T, D) reference for shape/dtype
    baseline: str,
    eos_token_id: int,
    device: str,
) -> torch.Tensor:
    """
    Build a baseline embedding shaped like `input_embed`, on `device`.
    GPT-2 has no [MASK] — supported baselines are zero, pad (EOS), mean.
    """
    embed_device = next(embed_layer.parameters()).device

    if baseline == "zero":
        return torch.zeros_like(input_embed).to(device)

    elif baseline == "pad":
        pad_id = torch.tensor([[eos_token_id]], device=embed_device)
        with torch.no_grad():
            pad_vec = embed_layer(pad_id).detach()          # (1, 1, D)
        return pad_vec.expand_as(input_embed).clone().to(device)

    elif baseline == "mean":
        with torch.no_grad():
            mean_vec = embed_layer.weight.mean(dim=0, keepdim=True).detach()
        return mean_vec.unsqueeze(0).expand_as(input_embed).clone().to(device)

    else:
        raise ValueError(
            f"Unknown baseline '{baseline}'. Choose: zero | pad | mean"
        )


# ---------------------------------------------------------------------------
# Chunked forward: returns joint log-prob y_hat(g) per sample
# ---------------------------------------------------------------------------

def _chunked_forward_logprob(
    model,
    X: torch.Tensor,              # (1, T, D)
    X_baseline: torch.Tensor,     # (1, T, D)
    gate_batch: torch.Tensor,     # (B, T) in [0, 1]
    answer_positions: list,
    answer_ids: torch.Tensor,     # (La,)
    chunk_size: int,
    device: str,
) -> torch.Tensor:
    """
    For each gate vector g in `gate_batch`, build the gated embedding
        X_gated = g * X + (1 - g) * X_baseline
    and return y_hat(g) = sum_i log p(a_i | x_gated at answer_positions[i]).

    Returns Tensor of shape (B,) on `device`.
    """
    B, T = gate_batch.shape
    La   = len(answer_positions)

    ans_pos_tensor = torch.tensor(answer_positions, device=device, dtype=torch.long)
    ans_ids_dev    = answer_ids.to(device).long()

    X_sq    = X.squeeze(0).to(device)            # (T, D)
    Xref_sq = X_baseline.squeeze(0).to(device)   # (T, D)

    y_chunks = []

    for i in range(0, B, chunk_size):
        j = min(i + chunk_size, B)
        g_chunk = gate_batch[i:j].to(device=device, dtype=X.dtype)   # (b, T)
        g_exp   = g_chunk.unsqueeze(-1)                              # (b, T, 1)
        X_gated = X_sq * g_exp + Xref_sq * (1.0 - g_exp)             # (b, T, D)

        with torch.no_grad():
            logits = model(inputs_embeds=X_gated).logits             # (b, T, V)

        log_probs = F.log_softmax(logits, dim=-1)                    # (b, T, V)
        gathered  = log_probs[:, ans_pos_tensor, :]                  # (b, La, V)
        token_lp  = gathered.gather(
            dim=-1,
            index=ans_ids_dev.view(1, La, 1).expand(j - i, La, 1),
        ).squeeze(-1)                                                # (b, La)
        y_chunks.append(token_lp.sum(dim=-1))                        # (b,)

    return torch.cat(y_chunks, dim=0)                                # (B,)


# ---------------------------------------------------------------------------
# Build the full (T_steps, Lq+2, T) probe block
# ---------------------------------------------------------------------------

def _build_gate_block(
    g_path: torch.Tensor,      # (steps, T)  path with pinned positions held at 1
    free_idx: torch.Tensor,    # (Lq,) indices of free positions (Q tokens)
    h: float,
    T: int,
) -> torch.Tensor:
    """
    For each active step t in 1..(steps-1), build (Lq + 2) gate vectors:
        row 0       : base   = g_path[t-1]                        in (T,)
        rows 1..Lq  : probes = g_path[t-1] + h * e_{free_idx[i]}  (clamped)
        row Lq+1    : joint  = g_path[t]                          in (T,)

    Returns shape (T_steps, Lq + 2, T).
    """
    T_steps = g_path.shape[0] - 1   # active steps
    Lq      = int(free_idx.numel())

    g_prev_all = g_path[:-1]                                  # (T_steps, T)
    g_curr_all = g_path[1:]                                   # (T_steps, T)

    # Build per-step probe matrix: shape (T_steps, Lq, T)
    # probe[t, i, :] = g_path[t-1] + h * e_{free_idx[i]}
    probes_all = g_prev_all.unsqueeze(1).expand(T_steps, Lq, T).clone()
    # Add h at column free_idx[i] in row i, for every step t
    # scatter_add_ along the last dim
    idx = free_idx.view(1, Lq, 1).expand(T_steps, Lq, 1)
    probes_all.scatter_add_(
        dim=2,
        index=idx,
        src=torch.full((T_steps, Lq, 1), h, device=g_path.device, dtype=g_path.dtype),
    )
    probes_all = probes_all.clamp(max=1.0)                    # (T_steps, Lq, T)

    gate_block = torch.cat(
        [
            g_prev_all.unsqueeze(1),                          # (T_steps, 1, T)
            probes_all,                                       # (T_steps, Lq, T)
            g_curr_all.unsqueeze(1),                          # (T_steps, 1, T)
        ],
        dim=1,
    )                                                         # (T_steps, Lq+2, T)
    return gate_block


# ---------------------------------------------------------------------------
# Public API: FD-PACE attribution for GPT-2
# ---------------------------------------------------------------------------

def pace_gradient_fd_gpt2(
    question: str,
    model_name: str = "gpt2",
    device: str = "cpu",
    steps: int = 50,
    a: float = 0.0,
    b: float = 1.0,
    chunk_size: int = 32,
    max_new_tokens: int = 30,
    gold_answer: Optional[str] = None,
    baseline: str = "zero",
) -> Dict[str, Any]:
    """
    Run gradient-free FD-PACE attribution for one (Q, A) pair on GPT-2.

    Parameters
    ----------
    question       : Full prompt string ("narrative ... Why did ...?")
    model_name     : GPT-2 variant.
    device         : 'cpu' or 'cuda'.
    steps          : Number of points on the gate path [a, b].
                     The number of *active* integration steps is steps-1.
    a, b           : Path endpoints (default 0 -> 1).
    chunk_size     : Max batch size for a single forward pass.
                     Total forwards: (steps-1) * (Lq + 2), chunked.
    max_new_tokens : Greedy generation budget when `gold_answer` is None.
    gold_answer    : If provided, skip generation and use as A.
    baseline       : 'zero' | 'pad' | 'mean' — see _build_base_embed.

    Returns
    -------
    dict with keys matching paceg_gpt2.pace_gradient_gpt2.
    """
    t0 = time.time()
    model, tokenizer = get_model_tokenizer(model_name, device)

    # ------------------------------------------------------------------
    # 1. Tokenise Q
    # ------------------------------------------------------------------
    q_enc       = tokenizer(question, return_tensors="pt", add_special_tokens=True)
    q_ids       = q_enc["input_ids"].to(device)               # [1, Lq]
    q_attn_mask = q_enc["attention_mask"].to(device)
    q_len       = q_ids.shape[1]

    # ------------------------------------------------------------------
    # 2. Get answer ids (gold or greedy)
    # ------------------------------------------------------------------
    if gold_answer is not None:
        a_enc    = tokenizer(gold_answer, return_tensors="pt",
                             add_special_tokens=False)
        a_ids_d  = a_enc["input_ids"].to(device)
        full_ids = torch.cat([q_ids, a_ids_d], dim=1)
    else:
        with torch.no_grad():
            full_ids = model.generate(
                q_ids,
                attention_mask=q_attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    T  = full_ids.shape[1]
    La = T - q_len
    if La == 0:
        raise ValueError(
            "Answer is empty — increase max_new_tokens or supply gold_answer."
        )
    answer_positions = list(range(q_len, T))
    answer_ids       = full_ids[0, q_len:].cpu()              # [La]

    # ------------------------------------------------------------------
    # 3. Token embeddings (wte output, same as PACE-G / LIME)
    # ------------------------------------------------------------------
    embed_layer = model.transformer.wte
    with torch.no_grad():
        input_embed = embed_layer(full_ids).detach()          # [1, T, D] on device

    # ------------------------------------------------------------------
    # 4. Baseline embedding
    # ------------------------------------------------------------------
    base_embed = _build_base_embed(
        embed_layer, input_embed,
        baseline, tokenizer.eos_token_id, device,
    )                                                         # [1, T, D] on device

    # ------------------------------------------------------------------
    # 5. Identify free positions:
    #    Q tokens are FREE (gates move).
    #    A tokens are PINNED at gate=1 (must not perturb the prefix
    #    that later answer tokens condition on).
    # ------------------------------------------------------------------
    free_mask = torch.zeros(T, dtype=torch.bool, device=device)
    free_mask[:q_len] = True
    free_idx  = torch.nonzero(free_mask, as_tuple=False).squeeze(-1)
    Lq_free   = int(free_idx.numel())

    # ------------------------------------------------------------------
    # 6. Reference logits (for compatibility with downstream eval)
    # ------------------------------------------------------------------
    with torch.no_grad():
        logits_full = model(inputs_embeds=input_embed).logits[0].detach().cpu()

    # ------------------------------------------------------------------
    # 7. Build the gate path.
    #    g_path[t, i] = t/(steps-1)   for i in free positions
    #    g_path[t, i] = 1.0           for i in pinned (A) positions
    # ------------------------------------------------------------------
    start_time = time.perf_counter()

    if Lq_free == 0 or steps < 2:
        attributions = torch.zeros(T, dtype=input_embed.dtype)
    else:
        t_vals = torch.linspace(a, b, steps, device=device, dtype=input_embed.dtype)
        g_path = torch.ones(steps, T, device=device, dtype=input_embed.dtype)
        # Free positions move from a to b; pinned stay at 1
        g_path[:, free_idx] = t_vals.unsqueeze(1).expand(steps, Lq_free)

        h = float((b - a) / max(steps - 1, 1))

        # ----- Assemble all probe vectors at once -----
        gate_block = _build_gate_block(g_path, free_idx, h, T)       # (T_steps, Lq+2, T)
        T_steps    = gate_block.shape[0]
        gate_flat  = gate_block.reshape(T_steps * (Lq_free + 2), T)

        # ----- Single chunked forward pass over the whole block -----
        y_flat = _chunked_forward_logprob(
            model            = model,
            X                = input_embed,
            X_baseline       = base_embed,
            gate_batch       = gate_flat,
            answer_positions = answer_positions,
            answer_ids       = answer_ids,
            chunk_size       = chunk_size,
            device           = device,
        )                                                            # (T_steps*(Lq+2),)

        # ----- Reshape and compute PACE-FD attributions -----
        y_target = y_flat.reshape(T_steps, Lq_free + 2)              # (T_steps, Lq+2)

        y_base  = y_target[:, 0:1]                                   # (T_steps, 1)
        y_probe = y_target[:, 1:1 + Lq_free]                         # (T_steps, Lq)
        y_joint = y_target[:, 1 + Lq_free]                           # (T_steps,)

        # u_i^{(t)} = y_hat(g^{(t-1)} + h e_i) - y_hat(g^{(t-1)})
        u = y_probe - y_base                                         # (T_steps, Lq)

        # Per-step normalization (signed; matches paper formulation)
        alpha = u / (u.sum(dim=1, keepdim=True) + 1e-10)             # (T_steps, Lq)

        # True joint output change per step (preserves exact completeness)
        delta_y = (y_joint - y_base.squeeze(1)).unsqueeze(1)         # (T_steps, 1)

        # PACE-FD attribution over the free positions
        attr_free = (alpha * delta_y).sum(dim=0)                     # (Lq_free,)

        # Scatter back into full-length attribution vector (pinned = 0)
        attributions = torch.zeros(T, dtype=input_embed.dtype, device=device)
        attributions[free_idx] = attr_free.to(attributions.dtype)
        attributions = attributions.cpu()

    end_time = time.perf_counter()

    # ------------------------------------------------------------------
    # 8. Sign convention
    # ------------------------------------------------------------------
    # PACE-FD attributions are signed:
    #   POSITIVE -> token raises log p(A|Q)  (supports the answer)
    #   NEGATIVE -> token suppresses log p(A|Q)
    #
    # xai_metrics_gpt2.py uses min-max normalization and expects
    # non-negative magnitudes as importance scores. We provide both:
    #   attributions          : |attr|   (drop-in for the eval pipeline)
    #   attributions_signed   : attr     (for analysis / debugging)
    attributions_signed = attributions.clone()
    attributions        = attributions.abs()

    # ------------------------------------------------------------------
    # 9. Package output (keys match paceg_gpt2.pace_gradient_gpt2)
    # ------------------------------------------------------------------
    tokens = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())

    return {
        "tokens":              tokens,
        "q_len":               q_len,
        "answer_positions":    answer_positions,
        "answer_ids":          answer_ids,                    # [La] CPU
        "attributions":        attributions,                  # [T]  CPU, |attr|
        "attributions_signed": attributions_signed,           # [T]  CPU, signed
        "input_embed":         input_embed.cpu(),             # [1,T,D] CPU
        "base_embed":          base_embed.cpu(),              # [1,T,D] CPU
        "logits_full":         logits_full,                   # [T,V]   CPU
        "predicted_answer":    tokenizer.decode(
                                   answer_ids.tolist(),
                                   skip_special_tokens=True),
        "model":               model,
        "tokenizer":           tokenizer,
        "time":                end_time - start_time,
        "fd_wall_time":        time.time() - t0,
    }