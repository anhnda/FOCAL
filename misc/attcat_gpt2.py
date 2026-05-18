"""
attcat_gpt2.py
==============
AttCAT (Attentive Class Activation Tokens) attribution for decoder-only
GPT-2 models, following the same pipeline and return-key contract as
paceg_gpt2.py so it is a drop-in replacement in run_eval_pg_gpt2.py.

AttCAT paper: "AttCAT: Explaining Transformers via Attentive Class
Activation Tokens", Qiang et al., NeurIPS 2022.
https://github.com/qiangyao1988/AttCAT

Algorithm (per token i, summed over all L transformer blocks):
  1. CAT_i^l  = grad(h_i^l) ⊙ h_i^l           (Hadamard product, no ReLU)
  2. AttCAT_i^l = mean_over_heads( alpha_i^l @ CAT_i^l )
  3. score_i  = sum_l  sum_d  AttCAT_i^l        (scalar per token)

GPT-2 specifics vs. encoder models
-------------------------------------
- Architecture  : decoder-only, causal self-attention.
- Block output  : hidden state captured from each GPT2Block via forward hook.
- Attn weights  : from GPT2Attention (out[2] when output_attentions=True).
- Target scalar : sum of answer-token logits at their sequence positions —
                  identical to paceg_gpt2.py, so metrics stay comparable.
- Baseline      : zero / pad / mean, same choices as paceg_gpt2.py.
- No *_helper   : GPT-2 has no position_embed / type_embed helpers.

Return keys  (identical to paceg_gpt2.py — xai_metrics_gpt2 works unchanged)
-------------------------------------------------------------------------------
    tokens           : list[str]      -- all tokens in [Q | A]
    q_len            : int            -- number of Q tokens
    answer_positions : list[int]      -- indices of A tokens in full sequence
    answer_ids       : Tensor [La]    -- answer token ids (CPU)
    attributions     : Tensor [T]     -- AttCAT score per token (CPU, raw)
    input_embed      : Tensor[1,T,D]  -- original embedding (CPU, detached)
    base_embed       : Tensor[1,T,D]  -- baseline embedding (CPU)
    logits_full      : Tensor[T,V]    -- reference logits (CPU)
    predicted_answer : str            -- decoded answer string
    model            : GPT2LMHeadModel
    tokenizer        : GPT2TokenizerFast
    time             : float          -- wall-clock seconds
"""

import time
from typing import Dict, List, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ---------------------------------------------------------------------------
# Module-level model cache  (identical pattern to paceg_gpt2.py)
# ---------------------------------------------------------------------------
_CACHE: dict = {}

def get_model_tokenizer(
    model_name: str = "gpt2",
    device: str = "cpu",
):
    """Return (model, tokenizer), loading from HuggingFace only once."""
    device = str(torch.device(device))   # normalizes "cuda" -> "cuda:0"
    key = (model_name, device)           # no `type` — GPT-2 cache needs only these two
    if key not in _CACHE:
        tok = GPT2TokenizerFast.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        mdl = GPT2LMHeadModel.from_pretrained(model_name)
        mdl.eval().to(device)
        _CACHE[key] = (mdl, tok)
    return _CACHE[key]

# ---------------------------------------------------------------------------
# Architecture helpers — GPT-2 specific
# ---------------------------------------------------------------------------

def _get_gpt2_blocks(model: GPT2LMHeadModel) -> List[torch.nn.Module]:
    """Return the list of GPT-2 transformer blocks (model.transformer.h)."""
    return list(model.transformer.h)


def _get_gpt2_attn(block: torch.nn.Module) -> torch.nn.Module:
    """Return the GPT2Attention submodule from a GPT2Block."""
    return block.attn


# ---------------------------------------------------------------------------
# Core AttCAT score computation
# ---------------------------------------------------------------------------

def _compute_attcat_scores(
    target_scalar: torch.Tensor,
    hidden_states_list: List[torch.Tensor],
    attn_weights_list: List[torch.Tensor],
    seq_len: int,
    device: str,
) -> torch.Tensor:
    """
    Compute raw AttCAT token scores for one scalar target.

    All tensors in hidden_states_list must still be in the autograd graph
    (no detach before this call).

    Args:
        target_scalar     : differentiable scalar (sum of answer logits).
        hidden_states_list: list of [1, T, d] block-output tensors (in graph).
        attn_weights_list : list of [1, H, T, T] attention weight tensors.
        seq_len           : full sequence length T.
        device            : torch device string.

    Returns:
        attcat_scores : [T] float tensor on `device`.
    """
    n_layers      = len(hidden_states_list)
    attcat_scores = torch.zeros(seq_len, device=device)

    for l_idx in range(n_layers):
        h_l = hidden_states_list[l_idx]   # [1, T, d] — must stay in graph

        try:
            (grad_h_l,) = torch.autograd.grad(
                target_scalar, h_l,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
        except RuntimeError:
            continue
        if grad_h_l is None:
            continue

        # CAT^l = grad ⊙ h  (no ReLU — preserve sign)
        cat_l = (grad_h_l * h_l.detach()).squeeze(0)   # [T, d]

        if l_idx < len(attn_weights_list):
            # alpha_l: [1, H, T_q, T_k] → squeeze → [H, T_q, T_k]
            # Causal mask makes alpha lower-triangular, so the einsum
            # naturally aggregates only from past / current tokens.
            alpha_l  = attn_weights_list[l_idx].squeeze(0)
            attcat_l = torch.einsum(
                "hij,jd->hid", alpha_l, cat_l
            ).mean(dim=0)                              # [T, d]
        else:
            attcat_l = cat_l                           # plain CAT fallback

        attcat_scores = attcat_scores + attcat_l.sum(dim=-1)   # [T]

    return attcat_scores


# ---------------------------------------------------------------------------
# Main attribution function
# ---------------------------------------------------------------------------

def attcat_gpt2(
    question: str,
    model_name: str = "gpt2",
    device: str = "cpu",
    max_new_tokens: int = 30,
    gold_answer: Optional[str] = None,
    baseline: str = "zero",
) -> Dict:
    """
    Run AttCAT attribution for one (question, answer) pair on GPT-2.

    Drop-in replacement for pace_gradient_gpt2() — identical parameters
    except `steps` / `a` / `b` (integration params) which AttCAT does not
    need (single forward pass instead of a Riemann sum).

    Parameters
    ----------
    question       : Full prompt string — "narrative  Why did ...?"
    model_name     : HuggingFace GPT-2 identifier.
    device         : 'cpu' or 'cuda'.
    max_new_tokens : Tokens to generate when gold_answer is None.
    gold_answer    : If provided, skip generation and use this string as A.
    baseline       : Reference embedding used by xai_metrics_gpt2 masking:
                       'zero' — all-zero embedding (default)
                       'pad'  — EOS token embedding broadcast across sequence
                       'mean' — mean of the full vocabulary embedding matrix

    Returns
    -------
    dict — see module docstring for the complete key list.
    """
    t0 = time.time()
    model, tokenizer = get_model_tokenizer(model_name, device)

    # ── 1. Tokenise Q ─────────────────────────────────────────────────────────
    q_enc       = tokenizer(question, return_tensors="pt", add_special_tokens=True)
    q_ids       = q_enc["input_ids"].to(device)        # [1, Lq]
    q_attn_mask = q_enc["attention_mask"].to(device)   # [1, Lq]
    q_len       = q_ids.shape[1]

    # ── 2. Answer token ids — generate or use gold ────────────────────────────
    if gold_answer is not None:
        a_enc    = tokenizer(gold_answer, return_tensors="pt",
                             add_special_tokens=False)
        a_ids_d  = a_enc["input_ids"].to(device)
        full_ids = torch.cat([q_ids, a_ids_d], dim=1)  # [1, T]
    else:
        with torch.no_grad():
            full_ids = model.generate(
                q_ids,
                attention_mask=q_attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )                                           # [1, T]

    T  = full_ids.shape[1]
    La = T - q_len

    if La == 0:
        raise ValueError(
            "Answer is empty — increase max_new_tokens or supply gold_answer."
        )

    answer_positions = list(range(q_len, T))
    answer_ids       = full_ids[0, q_len:].cpu()        # [La]

    # ── 3. Token embeddings & baseline ───────────────────────────────────────
    embed_layer = model.transformer.wte                 # nn.Embedding [V, D]

    with torch.no_grad():
        input_embed = embed_layer(full_ids).detach()    # [1, T, D]

    if baseline == "zero":
        base_embed = torch.zeros_like(input_embed)
    elif baseline == "pad":
        pad_id     = torch.tensor([[tokenizer.eos_token_id]], device=device)
        pad_vec    = embed_layer(pad_id).detach()       # [1, 1, D]
        base_embed = pad_vec.expand_as(input_embed).clone()
    elif baseline == "mean":
        mean_vec   = embed_layer.weight.mean(dim=0, keepdim=True)   # [1, D]
        base_embed = mean_vec.unsqueeze(0).expand_as(input_embed).clone()
    else:
        raise ValueError(
            f"Unknown baseline '{baseline}'. Choose: zero | pad | mean"
        )
    base_embed = base_embed.detach()

    # ── 4. Reference logits (no grad) ─────────────────────────────────────────
    with torch.no_grad():
        logits_full = model(
            inputs_embeds=input_embed
        ).logits[0].detach()                            # [T, V]

    # ── 5. Forward pass with hooks to capture h_l and alpha_l ────────────────
    # CRITICAL: h_l tensors must stay in the autograd graph — NO detach here.
    hidden_states_list: List[torch.Tensor] = []
    attn_weights_list:  List[torch.Tensor] = []
    hooks: List = []

    gpt2_blocks = _get_gpt2_blocks(model)

    def make_block_hook(idx: int):
        def fn(module, inp, out):
            # GPT2Block output: (hidden_state [1,T,d], present, ...)
            h = out[0] if isinstance(out, tuple) else out
            hidden_states_list.append(h)               # keep in graph
        return fn

    def make_attn_hook(idx: int):
        def fn(module, inp, out):
            # GPT2Attention output (output_attentions=True):
            #   (attn_output, present, attn_weights [1,H,T,T])
            if isinstance(out, tuple) and len(out) >= 3 and out[2] is not None:
                w = out[2]
                if w.dim() == 4:
                    attn_weights_list.append(w.detach())
        return fn

    for idx, block in enumerate(gpt2_blocks):
        hooks.append(block.register_forward_hook(make_block_hook(idx)))
        hooks.append(
            _get_gpt2_attn(block).register_forward_hook(make_attn_hook(idx))
        )

    with torch.enable_grad():
        outputs = model(
            inputs_embeds=input_embed,
            output_attentions=True,
            output_hidden_states=True,
        )

    for h in hooks:
        h.remove()

    # ── 6. Scalar target = sum of answer-token logits ─────────────────────────
    logits = outputs.logits                             # [1, T, V]
    target = sum(
        logits[0, answer_positions[i], answer_ids[i].to(device)]
        for i in range(La)
    )

    # Fallbacks if hooks captured nothing (non-standard model variants)
    if len(hidden_states_list) == 0 and outputs.hidden_states is not None:
        hidden_states_list = list(outputs.hidden_states[1:])
    if len(attn_weights_list) == 0 and outputs.attentions is not None:
        attn_weights_list = [
            a.detach() for a in outputs.attentions if a is not None
        ]

    # ── 7. AttCAT scores ──────────────────────────────────────────────────────
    attcat_scores = _compute_attcat_scores(
        target, hidden_states_list, attn_weights_list, T, device
    )

    # ── 8. Decode tokens ──────────────────────────────────────────────────────
    tokens = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())

    return {
        "tokens":           tokens,
        "q_len":            q_len,
        "answer_positions": answer_positions,
        "answer_ids":       answer_ids,                     # [La] CPU
        "attributions":     attcat_scores.detach().cpu(),   # [T]  CPU
        "input_embed":      input_embed.detach().cpu(),     # [1,T,D] CPU
        "base_embed":       base_embed.cpu(),               # [1,T,D] CPU
        "logits_full":      logits_full.cpu(),              # [T,V]  CPU
        "predicted_answer": tokenizer.decode(
                                answer_ids.tolist(),
                                skip_special_tokens=True),
        "model":            model,
        "tokenizer":        tokenizer,
        "time":             time.time() - t0,
    }