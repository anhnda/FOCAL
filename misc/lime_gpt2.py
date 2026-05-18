"""
lime_gpt2.py
============
LIME (Local Interpretable Model-agnostic Explanations) for decoder-only
GPT-2 token attribution.

Adapts Ribeiro et al. (KDD 2016) to autoregressive generation. The
interface mirrors `paceg_gpt2.py` so that `run_eval_pg_gpt2.py` and
`xai_metrics_gpt2.py` can consume the output without changes — i.e. the
return dict carries the same keys (`tokens`, `q_len`, `attributions`,
`input_embed`, `base_embed`, `answer_ids`, `answer_positions`,
`predicted_answer`, `model`, `tokenizer`, `time`, ...).

Algorithm
---------
Given a question Q (length Lq) and an answer A (length La, either gold
or greedy-generated), LIME for decoder-only models:

  1. Pins all Lq+La positions whose values must stay constant: the
     ANSWER tokens are pinned (we are explaining p(A|Q), so we cannot
     perturb A), and any special tokens in Q if present. Only the
     remaining Q-positions are "free" and perturbable.

  2. Draws N binary masks z^{(n)} in {0,1}^Lq over the free positions.
     Pinned positions are forced to 1. Free positions are Bernoulli with
     keep-probability p_keep (default 0.5).

  3. For each mask z, constructs a perturbed embedding sequence
         X_pert[i] = X[i]           if z_i = 1
                   = X_baseline[i]  if z_i = 0
     for i in [0, Lq), with X[i] kept exactly as-is for i in [Lq, Lq+La).
     One forward pass under torch.no_grad() yields the logits at the
     answer positions, from which the per-sample target

         y^{(n)} = sum_{i=1..La} log p(a_i | x_pert at position Lq+i-1)

     is computed.

  4. Each sample is weighted by an exponential kernel on cosine distance
     between z^{(n)} and the all-ones mask (over free positions only):

         pi(z) = exp( -D(z, 1)^2 / sigma^2 ),  D = 1 - cos_sim

  5. Solve weighted ridge regression of y^{(n)} on z^{(n)} (over free
     positions). The coefficients ARE the per-token attributions.
     Pinned positions receive 0 (non-identifiable in the regression).

Notes
-----
- LIME is stochastic. We seed torch's RNG at import for reproducibility
  within a process and expose `seed` per-call.
- All forward passes are batched and chunked. Total cost is N forwards
  (vs. PACE-G's `steps` forwards + backwards).
- The fitted intercept absorbs the baseline; we report only the per-
  token weights — i.e. relative contribution above the baseline.
- The attribution vector has length T = Lq + La with answer positions
  carrying 0 (they are pinned and constant). This matches the shape
  convention used by `xai_metrics_gpt2.py`.
"""

import time
import random
from typing import Optional, Dict, Any

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
# Baseline embedding factory (matches paceg_gpt2._build_base_embed)
# ---------------------------------------------------------------------------

def _build_base_embed(
    embed_layer: torch.nn.Embedding,
    input_embed: torch.Tensor,        # (1, T, D) shape reference
    baseline: str,
    eos_token_id: int,
    device: str,
) -> torch.Tensor:
    """
    Build a baseline embedding of shape `input_embed.shape`, on CPU.
    GPT-2 has no [MASK] — supported baselines are zero, pad (EOS), mean.
    """
    embed_device = next(embed_layer.parameters()).device

    if baseline == "zero":
        return torch.zeros_like(input_embed).cpu()

    elif baseline == "pad":
        pad_id  = torch.tensor([[eos_token_id]], device=embed_device)
        with torch.no_grad():
            pad_vec = embed_layer(pad_id).detach().cpu()       # (1, 1, D)
        return pad_vec.expand_as(input_embed).clone()

    elif baseline == "mean":
        with torch.no_grad():
            mean_vec = embed_layer.weight.mean(dim=0, keepdim=True).detach().cpu()
        return mean_vec.unsqueeze(0).expand_as(input_embed).clone()

    else:
        raise ValueError(
            f"Unknown baseline '{baseline}'. Choose: zero | pad | mean"
        )


# ---------------------------------------------------------------------------
# LIME kernel + weighted ridge regression (identical to lime.py)
# ---------------------------------------------------------------------------

def _lime_kernel_weights(
    Z: torch.Tensor,           # (N, L_free) in {0, 1}
    sigma: float,
) -> torch.Tensor:
    """
    Exponential kernel weight on cosine distance to the all-ones vector,
    computed over the *free* (non-pinned) positions.
    """
    N, L_free = Z.shape
    if L_free == 0:
        return torch.ones(N, device=Z.device, dtype=Z.dtype)

    z_norm    = Z.norm(dim=1)                          # (N,)
    ones_norm = float(L_free) ** 0.5
    dot       = Z.sum(dim=1)                           # (N,)
    denom     = (z_norm * ones_norm).clamp(min=1e-12)
    cos_sim   = dot / denom
    cos_sim[z_norm == 0] = 0.0                         # all-zero -> dist 1
    dist      = 1.0 - cos_sim
    return torch.exp(-(dist ** 2) / (sigma ** 2 + 1e-12))


def _weighted_ridge(
    Z: torch.Tensor,    # (N, L_free)
    y: torch.Tensor,    # (N,)
    w: torch.Tensor,    # (N,)
    lam: float,
) -> torch.Tensor:
    """
    Solve  argmin_beta  sum_n w_n (y_n - z_n . beta - beta_0)^2 + lam ||beta||^2
    No L2 penalty applied to the intercept. Returns the L_free-dim
    coefficient vector (intercept dropped).
    """
    N, L_free = Z.shape
    if L_free == 0:
        return torch.zeros(0, device=Z.device, dtype=Z.dtype)

    ones  = torch.ones(N, 1, device=Z.device, dtype=Z.dtype)
    Z_aug = torch.cat([ones, Z.to(Z.dtype)], dim=1)             # (N, 1+L_free)

    W = w.unsqueeze(1)                                          # (N, 1)
    A = Z_aug.transpose(0, 1) @ (W * Z_aug)                     # (1+L_free, 1+L_free)
    b = Z_aug.transpose(0, 1) @ (W.squeeze(1) * y).unsqueeze(1) # (1+L_free, 1)

    reg = torch.eye(L_free + 1, device=Z.device, dtype=Z.dtype) * lam
    reg[0, 0] = 0.0
    A = A + reg

    try:
        beta_full = torch.linalg.solve(A, b).squeeze(1)
    except RuntimeError:
        beta_full = torch.linalg.lstsq(A, b).solution.squeeze(1)

    return beta_full[1:]   # drop intercept


# ---------------------------------------------------------------------------
# Mask sampling over the free (perturbable) positions
# ---------------------------------------------------------------------------

def _sample_masks(
    L: int,                            # full sequence length (Lq + La)
    free_idx: torch.Tensor,            # (L_free,) indices into [0, L)
    n_samples: int,
    p_keep: float,
    device: str,
    dtype: torch.dtype,
    seed: Optional[int],
    include_full: bool = True,
) -> torch.Tensor:
    """
    Build (N, L) binary masks. Free positions are i.i.d. Bernoulli(p_keep);
    pinned positions are set to 1. Optionally inject the all-ones mask as
    the first sample (anchor the surrogate on the original input).
    """
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
    else:
        g = None

    L_free = free_idx.numel()
    N      = n_samples

    Z_full = torch.ones(N, L, device=device, dtype=dtype)
    if L_free > 0:
        if g is not None:
            rand = torch.rand(N, L_free, device=device, generator=g)
        else:
            rand = torch.rand(N, L_free, device=device)
        bits = (rand < p_keep).to(dtype)
        Z_full[:, free_idx] = bits

    if include_full and N > 0:
        Z_full[0, :] = 1.0

    return Z_full


# ---------------------------------------------------------------------------
# Chunked forward pass: returns per-sample target log p(A | Q_perturbed)
# ---------------------------------------------------------------------------

def _chunked_forward_logprob(
    model,
    X: torch.Tensor,              # (1, T, D)
    X_baseline: torch.Tensor,     # (1, T, D)
    mask_batch: torch.Tensor,     # (N, T) in {0, 1}
    answer_positions: list,       # list of indices into [0, T)
    answer_ids: torch.Tensor,     # (La,) on CPU or device
    chunk_size: int,
    device: str,
) -> torch.Tensor:
    """
    For each mask z in mask_batch, build a perturbed embedding
        X_pert = z * X + (1 - z) * X_baseline
    and return y = sum_i log p(a_i | x_pert at answer_positions[i]).

    Causality: positions in `answer_positions` are guaranteed to be
    pinned to 1 in `mask_batch` by the caller, so their embeddings stay
    exactly equal to X[answer_positions[i]] — preserving the original
    answer context for the prefix at those positions.

    Returns
    -------
    Tensor of shape (N,) — the per-sample joint log-probability target.
    """
    N, T = mask_batch.shape
    La   = len(answer_positions)

    answer_ids_dev = answer_ids.to(device).long()
    ans_pos_tensor = torch.tensor(answer_positions, device=device, dtype=torch.long)

    X_sq    = X.squeeze(0).to(device)            # (T, D)
    Xref_sq = X_baseline.squeeze(0).to(device)   # (T, D)

    y_chunks = []

    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        z_chunk = mask_batch[i:j].to(device=device, dtype=X.dtype)   # (b, T)
        z_exp   = z_chunk.unsqueeze(-1)                              # (b, T, 1)
        X_pert  = X_sq * z_exp + Xref_sq * (1.0 - z_exp)             # (b, T, D)

        with torch.no_grad():
            logits = model(inputs_embeds=X_pert).logits              # (b, T, V)

        # Gather log p(a_i | ...) at each answer position
        log_probs   = F.log_softmax(logits, dim=-1)                  # (b, T, V)
        # log_probs at answer_positions[i] for token answer_ids[i]
        # -> (b, La)
        gathered = log_probs[:, ans_pos_tensor, :]                   # (b, La, V)
        token_lp = gathered.gather(
            dim=-1,
            index=answer_ids_dev.view(1, La, 1).expand(j - i, La, 1),
        ).squeeze(-1)                                                # (b, La)
        y_chunk = token_lp.sum(dim=-1)                               # (b,)
        y_chunks.append(y_chunk.detach().cpu())

    return torch.cat(y_chunks, dim=0)                                # (N,)


# ---------------------------------------------------------------------------
# Public API: LIME attribution for GPT-2
# ---------------------------------------------------------------------------

def lime_gpt2(
    question: str,
    model_name: str = "gpt2",
    device: str = "cpu",
    n_samples: int = 1000,
    p_keep: float = 0.5,
    sigma: float = 0.25,
    ridge_lambda: float = 1.0,
    chunk_size: int = 32,
    max_new_tokens: int = 30,
    gold_answer: Optional[str] = None,
    baseline: str = "zero",
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Run LIME attribution for one (Q, A) pair on GPT-2.

    Parameters
    ----------
    question       : Full prompt string ("narrative ... Why did ...?")
    model_name     : GPT-2 variant.
    device         : 'cpu' or 'cuda'.
    n_samples      : Number of binary masks (LIME's N).
    p_keep         : Bernoulli keep-probability for free (Q) positions.
    sigma          : Kernel bandwidth (cosine-distance space).
    ridge_lambda   : Ridge L2 penalty for the surrogate model.
    chunk_size     : Mini-batch size for the masked forward passes.
    max_new_tokens : Greedy generation budget when `gold_answer` is None.
    gold_answer    : If provided, skip generation and use this as A.
    baseline       : 'zero' | 'pad' | 'mean' — see _build_base_embed.
    seed           : RNG seed for mask sampling.

    Returns
    -------
    dict with keys matching paceg_gpt2.pace_gradient_gpt2 — drop-in
    compatible with run_eval_pg_gpt2.py and xai_metrics_gpt2.py.
    """
    t0 = time.time()
    model, tokenizer = get_model_tokenizer(model_name, device)

    # ------------------------------------------------------------------
    # 1. Tokenise Q
    # ------------------------------------------------------------------
    q_enc       = tokenizer(question, return_tensors="pt", add_special_tokens=True)
    q_ids       = q_enc["input_ids"].to(device)            # [1, Lq]
    q_attn_mask = q_enc["attention_mask"].to(device)
    q_len       = q_ids.shape[1]

    # ------------------------------------------------------------------
    # 2. Get answer ids (gold or greedy)
    # ------------------------------------------------------------------
    if gold_answer is not None:
        a_enc    = tokenizer(gold_answer, return_tensors="pt",
                             add_special_tokens=False)
        a_ids_d  = a_enc["input_ids"].to(device)            # [1, La]
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
    answer_ids       = full_ids[0, q_len:].cpu()            # [La]

    # ------------------------------------------------------------------
    # 3. Build token embeddings (operate on wte output, like PACE-G)
    # ------------------------------------------------------------------
    embed_layer = model.transformer.wte
    with torch.no_grad():
        input_embed = embed_layer(full_ids).detach()        # [1, T, D] on device

    # ------------------------------------------------------------------
    # 4. Baseline embedding
    # ------------------------------------------------------------------
    base_embed_cpu = _build_base_embed(
        embed_layer, input_embed.cpu(),
        baseline, tokenizer.eos_token_id, device,
    )                                                       # [1, T, D] on CPU
    base_embed = base_embed_cpu.to(device)

    # ------------------------------------------------------------------
    # 5. Identify free (perturbable) positions:
    #    only the Q tokens are free; A tokens are pinned because we are
    #    measuring p(A | Q_perturbed) and must keep the answer context
    #    identical to the original sequence.
    # ------------------------------------------------------------------
    free_mask = torch.zeros(T, dtype=torch.bool, device=device)
    free_mask[:q_len] = True
    free_idx  = torch.nonzero(free_mask, as_tuple=False).squeeze(-1)
    L_free    = int(free_idx.numel())

    # ------------------------------------------------------------------
    # 6. Reference logits (for compatibility with the eval pipeline)
    # ------------------------------------------------------------------
    with torch.no_grad():
        logits_full = model(inputs_embeds=input_embed).logits[0].detach().cpu()

    # ------------------------------------------------------------------
    # 7. LIME core: sample, forward, kernel weights, ridge
    # ------------------------------------------------------------------
    if L_free == 0 or n_samples <= 0:
        attributions = torch.zeros(T, dtype=input_embed.dtype)
    else:
        Z = _sample_masks(
            L=T,
            free_idx=free_idx,
            n_samples=n_samples,
            p_keep=p_keep,
            device=device,
            dtype=input_embed.dtype,
            seed=seed,
            include_full=True,
        )                                                   # (N, T)

        y = _chunked_forward_logprob(
            model=model,
            X=input_embed,
            X_baseline=base_embed,
            mask_batch=Z,
            answer_positions=answer_positions,
            answer_ids=answer_ids,
            chunk_size=chunk_size,
            device=device,
        )                                                   # (N,) on CPU

        Z_free = Z[:, free_idx].cpu()                       # (N, L_free) on CPU
        w      = _lime_kernel_weights(Z_free, sigma=sigma)  # (N,)

        beta_free = _weighted_ridge(
            Z=Z_free, y=y, w=w, lam=ridge_lambda,
        )                                                   # (L_free,)

        # Scatter back into full-length attribution vector
        attributions = torch.zeros(T, dtype=input_embed.dtype)
        attributions[free_idx.cpu()] = beta_free.to(attributions.dtype)

    # ------------------------------------------------------------------
    # 8. Sign convention
    # ------------------------------------------------------------------
    # The raw ridge coefficient beta_i is the surrogate's estimate of
    # "how much does keeping token i present raise log p(A|Q)?".
    # POSITIVE beta_i  -> token supports the answer (important).
    # NEGATIVE beta_i  -> token suppresses the answer.
    #
    # For compatibility with xai_metrics_gpt2.py (which uses Soft-NC/NS
    # with min-max normalization on attributions, and assumes non-
    # negative magnitudes are "importance"), we take the absolute value.
    # The signed version is preserved in `attributions_signed`.
    attributions_signed = attributions.clone()
    attributions        = attributions.abs()

    # ------------------------------------------------------------------
    # 9. Decode tokens, package output
    # ------------------------------------------------------------------
    tokens = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())

    return {
        "tokens":             tokens,
        "q_len":              q_len,
        "answer_positions":   answer_positions,
        "answer_ids":         answer_ids,                   # [La]   CPU
        "attributions":       attributions,                 # [T]    CPU, |beta|
        "attributions_signed":attributions_signed,          # [T]    CPU, beta
        "input_embed":        input_embed.cpu(),            # [1,T,D] CPU
        "base_embed":         base_embed.cpu(),             # [1,T,D] CPU
        "logits_full":        logits_full,                  # [T,V]   CPU
        "predicted_answer":   tokenizer.decode(
                                  answer_ids.tolist(),
                                  skip_special_tokens=True),
        "model":              model,
        "tokenizer":          tokenizer,
        "time":               time.time() - t0,
    }