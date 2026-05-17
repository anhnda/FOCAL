"""
pace_gradients_gnorm.py

PACE Gradient Classification Attribution with configurable scalar-gate
gradient normalisation strategies.

The scalar gate ex yields grad_ex of shape (steps, L) after squeezing.
All four norms operate *across the token dimension* (dim=1), controlling
how each step's gate gradients are redistributed before delta-weighting.

Normalisation modes
-------------------
sign_norm    (default, identical to original pace_gradient_classification)
                 grad_ex / (grad_ex.sum(dim=1, keepdim=True) + 1e-10)
                 Divides by the signed sum of gate gradients across tokens.

sign_magl2   L2-magnitude normalisation across tokens:
                 grad_ex / (||grad_ex||_2  + 1e-10)   (per-step L2 norm over L)

sign_magl1   L1-magnitude normalisation across tokens:
                 grad_ex / (||grad_ex||_1  + 1e-10)   (per-step L1 norm over L)

safe_norm    Zero-safe signed-sum normalisation:
                 grad_ex / (0 if sum==0 → eps, else sum)
                 Equivalent to sign_norm but with an explicit zero guard instead
                 of the additive eps, preventing any bias on all-zero steps.

square_norm  Squared-gradient softmax-like normalisation:
                 grad_ex**2 / (grad_ex**2).sum(dim=1, keepdim=True) + 1e-10)
                 Attributions are non-negative and sum to ~1 per step.
                 Amplifies large-magnitude gate gradients and suppresses small
                 ones, acting as a soft selection over tokens.
"""

import time
import inspect
import torch
import numpy as np
import random
from typing import Dict, Any, Literal
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pace_gradients import get_baseline_embedding

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

_GNORM_MODES = ("sign_norm", "sign_magl2", "sign_magl1", "safe_norm", "square_norm")

_cache: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Internal helpers (same pattern as pace_gradient_ablations.py)
# ---------------------------------------------------------------------------

def _load(model_name: str, device: str):
    if model_name not in _cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()
        _cache[model_name] = {"model": model, "tokenizer": tokenizer}
    return _cache[model_name]["model"], _cache[model_name]["tokenizer"]


def _get_helper(model_name: str):
    if "distilbert" in model_name:
        from distilbert_helper import get_inputs, nn_forward_func
    elif "roberta" in model_name:
        from roberta_helper import get_inputs, nn_forward_func
    elif "bert" in model_name:
        from bert_helper import get_inputs, nn_forward_func
    else:
        raise NotImplementedError(f"No helper module for model: {model_name}")
    return get_inputs, nn_forward_func


def _normalise(grad_ex: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Normalise a (steps, L) scalar-gate gradient tensor across the token
    dimension (dim=1) according to `mode`.

    Parameters
    ----------
    grad_ex : (steps, L)  — raw gate gradients for every step
    mode    : one of _GNORM_MODES

    Returns
    -------
    (steps, L) normalised tensor
    """
    if mode == "sign_norm":
        # Original: divide by signed sum; eps prevents exact-zero denominator
        denom = grad_ex.sum(dim=1, keepdim=True) + 1e-10
        return grad_ex / denom

    elif mode == "sign_magl2":
        # L2 norm of the gate-gradient vector across tokens (per step)
        denom = grad_ex.norm(p=2, dim=1, keepdim=True) + 1e-10
        return grad_ex / denom

    elif mode == "sign_magl1":
        # L1 norm of the gate-gradient vector across tokens (per step)
        denom = grad_ex.norm(p=1, dim=1, keepdim=True) + 1e-10
        return grad_ex / denom

    elif mode == "safe_norm":
        row_sum = grad_ex.sum(dim=1, keepdim=True)
        denom   = torch.where(row_sum == 0,
                            torch.full_like(row_sum, 1e-10),
                            row_sum)
        grad_ex = grad_ex / denom

        # Keep sign, clip magnitude to threshold
        threshold = 1000.0
        clipped_mask = grad_ex.abs() >= threshold          # (steps, tokens)
        grad_ex = torch.where(
            clipped_mask,
            grad_ex.sign() * threshold,
            grad_ex
        )

        # Re-normalize only for rows that had at least one clipped value
        rows_clipped = clipped_mask.any(dim=1, keepdim=True)  # (steps, 1)
        if rows_clipped.any():
            row_sum2 = grad_ex.sum(dim=1, keepdim=True)
            denom2   = torch.where(row_sum2 == 0,
                                torch.full_like(row_sum2, 1e-10),
                                row_sum2)
            grad_ex  = torch.where(rows_clipped, grad_ex / denom2, grad_ex)
        return grad_ex
    elif mode == "square_norm":
        # Squared-gradient normalisation: grad^2 / sum(grad^2)
        # Non-negative outputs that sum to ~1 per step; amplifies large grads
        grad_sq = grad_ex ** 2                               # (steps, L)
        denom   = grad_sq.sum(dim=1, keepdim=True) + 1e-10  # (steps, 1)
        return grad_sq / denom

    else:
        raise ValueError(
            f"Unknown gnorm mode '{mode}'. Choose from: {_GNORM_MODES}"
        )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def pace_gradient_gnorm(
    sentence: str,
    a: float = 0.0,
    b: float = 1.0,
    steps: int = 100,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
    baseline: str = "mask",
    gnorm: Literal["sign_norm", "sign_magl2", "sign_magl1", "safe_norm", "square_norm"] = "sign_norm",
) -> Dict[str, Any]:
    """
    PACE classification attribution with a configurable scalar-gate gradient
    normalisation strategy.

    The integration path and scalar gate construction are identical to the
    original pace_gradient_classification. Only the line:

        grad_norm = grad_ex / (grad_ex.sum(dim=1, keepdim=True) + 1e-10)

    is replaced by the chosen `gnorm` variant (see module docstring).

    Parameters
    ----------
    sentence  : input text
    a, b      : interpolation path endpoints  [a=0, b=1] → baseline→input
    steps     : number of interpolation steps
    model_name: HuggingFace model identifier
    device    : 'cuda' or 'cpu'
    show_special_tokens: include CLS/SEP in returned tokens/attributions
    baseline  : baseline embedding strategy (mask | pad | zero | mean | random)
    gnorm     : gradient normalisation mode
                  'sign_norm'  — original signed-sum  (default)
                  'sign_magl2' — L2-norm across tokens
                  'sign_magl1' — L1-norm across tokens
                  'safe_norm'  — zero-safe signed-sum

    Returns
    -------
    dict with same keys as pace_gradient_classification, plus:
        'gnorm' : str — the normalisation mode used
    """
    if gnorm not in _GNORM_MODES:
        raise ValueError(f"gnorm must be one of {_GNORM_MODES}, got '{gnorm}'")

    get_inputs, nn_forward_func = _get_helper(model_name)
    model, tokenizer = _load(model_name, device)

    # ── Tokenise ─────────────────────────────────────────────────────────────
    enc = tokenizer(sentence, return_tensors="pt", truncation=True,
                    return_special_tokens_mask=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    fwd_params = inspect.signature(model.forward).parameters
    extra_kwargs = {}
    if "token_type_ids" in fwd_params and token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    # ── Embed & predict ───────────────────────────────────────────────────────
    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)                                    # (1, L, d)
        logits0 = model(inputs_embeds=X, attention_mask=attention_mask,
                        **extra_kwargs).logits[0]
    pred_id = int(logits0.argmax().item())

    L, d = X.shape[1], X.shape[2]

    # ── Baseline ──────────────────────────────────────────────────────────────
    X_baseline = get_baseline_embedding(baseline, embed, tokenizer, X, device)  # (1, L, d)

    # ── Build interpolation path via scalar gate (identical to original) ──────
    t_vals     = torch.linspace(a, b, steps, device=device, dtype=X.dtype)   # (steps,)
    coefs_base = t_vals.unsqueeze(1).expand(steps, L).clone()                 # (steps, L)
    # Fix CLS (pos 0) and SEP (last non-pad pos, conservatively pos -1)
    coefs_base[:, 0]  = 1.0
    coefs_base[:, -1] = 1.0

    # ex is the learnable perturbation on the coef; grad w.r.t. ex is our gate gradient
    ex           = torch.zeros(steps, L, 1, device=device, dtype=X.dtype,
                               requires_grad=True)
    itepolated_o = coefs_base.unsqueeze(-1) + ex                              # (steps, L, 1)
    iterpolated  = itepolated_o.tile((1, 1, d))                               # (steps, L, d)

    X_inter = (X.squeeze(0) * iterpolated
               + X_baseline.squeeze(0) * (1 - iterpolated))                  # (steps, L, d)

    attn_batch  = attention_mask.expand(steps, -1)
    extra_batch = {}
    if "token_type_ids" in extra_kwargs:
        extra_batch["token_type_ids"] = extra_kwargs["token_type_ids"].expand(steps, -1)

    # ── Forward + backward ────────────────────────────────────────────────────
    start_time = time.perf_counter()

    out          = model(inputs_embeds=X_inter, attention_mask=attn_batch, **extra_batch)
    logits_batch = out.logits[:, pred_id]                                     # (steps,)

    delta    = logits_batch - torch.cat([logits_batch[:1], logits_batch[:-1]])
    delta[0] = 0.0                                                            # first step has no predecessor

    (grad_ex_raw,) = torch.autograd.grad(logits_batch.sum(), ex)
    grad_ex = grad_ex_raw.squeeze(-1)                                         # (steps, L)

    end_time = time.perf_counter()

    # ── Normalise with chosen strategy ────────────────────────────────────────
    grad_normed = _normalise(grad_ex, gnorm)                                  # (steps, L)

    # Attribution = sum over steps of (normalised gate grad × step delta)
    attr = (grad_normed * delta.unsqueeze(1)).sum(dim=0)                      # (L,)

    # ── Position / type embeddings for metric helpers ─────────────────────────
    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

    attr_full = attr.detach().clone()   # (L,) unfiltered, on device

    # ── Filter special tokens ─────────────────────────────────────────────────
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    if not show_special_tokens:
        special_ids_set = set(tokenizer.all_special_ids)
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist())
                    if tid not in special_ids_set]
        tokens = [tokens[i] for i in keep_idx]
        attr   = attr[keep_idx]

    return {
        "tokens":          tokens,
        "attributions":    attr.detach().cpu(),
        "time":            end_time - start_time,
        "predicted_label": pred_id,
        "gnorm":           gnorm,
        # raw tensors for metric functions
        "model":           model,
        "nn_forward_func": nn_forward_func,
        "input_embed":     X,
        "attention_mask":  attention_mask,
        "position_embed":  position_embed,
        "type_embed":      type_embed,
        "attr_full":       attr_full,
    }