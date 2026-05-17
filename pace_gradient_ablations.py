"""
PACE Gradient Ablation Variants for Sentiment Classification

Two ablation families derived from the original PACE scalar-gate method:

1. pace_gradient_l12  — replaces the scalar gate with the L1 or L2 norm of
                        the raw embedding-space gradient vector for each token,
                        but still weights each step by the output delta.

2. pace_gradient_scalar — keeps the scalar gate exactly as in the original but
                          drops the delta weighting: attribution = mean of
                          (normalised scalar-gate gradient) over the path,
                          with NO multiplication by the step-wise logit delta.

Both functions share the same signature contract as pace_gradient_classification
and return the same dict keys so they are drop-in replacements in the eval loop.
"""

import time
import inspect
import torch
import numpy as np
import random
from typing import Dict, Any, Literal
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pace_gradients import get_baseline_embedding      # reuse factory

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

_cache: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load(model_name: str, device: str):
    """Cache model + tokenizer by model_name."""
    if model_name not in _cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()
        _cache[model_name] = {"model": model, "tokenizer": tokenizer}
    return _cache[model_name]["model"], _cache[model_name]["tokenizer"]


def _get_helper(model_name: str):
    """Return (get_inputs, nn_forward_func) from the right helper module."""
    if "distilbert" in model_name:
        from distilbert_helper import get_inputs, nn_forward_func
    elif "roberta" in model_name:
        from roberta_helper import get_inputs, nn_forward_func
    elif "bert" in model_name:
        from bert_helper import get_inputs, nn_forward_func
    else:
        raise NotImplementedError(f"No helper for {model_name}")
    return get_inputs, nn_forward_func


def _common_setup(sentence, model_name, device, baseline):
    """
    Tokenise, embed, build baseline, compute predicted class.
    Returns a bundle dict used by both ablation functions.
    """
    model, tokenizer = _load(model_name, device)
    get_inputs, nn_forward_func = _get_helper(model_name)

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

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)          # (1, L, d)
        logits0 = model(inputs_embeds=X, attention_mask=attention_mask,
                        **extra_kwargs).logits[0]
    pred_id = int(logits0.argmax().item())

    X_baseline = get_baseline_embedding(baseline, embed, tokenizer, X, device)  # (1, L, d)

    L, d = X.shape[1], X.shape[2]

    # position / type embeddings for metric helpers
    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return dict(
        model=model, tokenizer=tokenizer,
        nn_forward_func=nn_forward_func,
        embed=embed,
        input_ids=input_ids,
        attention_mask=attention_mask,
        extra_kwargs=extra_kwargs,
        X=X, X_baseline=X_baseline,
        pred_id=pred_id,
        L=L, d=d,
        tokens=tokens,
        position_embed=position_embed,
        type_embed=type_embed,
        device=device,
    )


def _pack_result(bundle, attr, attr_full, show_special_tokens, elapsed):
    """Filter special tokens and assemble the output dict."""
    model      = bundle["model"]
    tokenizer  = bundle["tokenizer"]
    input_ids  = bundle["input_ids"]
    tokens     = bundle["tokens"]

    if not show_special_tokens:
        special_ids_set = set(tokenizer.all_special_ids)
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist())
                    if tid not in special_ids_set]
        tokens = [tokens[i] for i in keep_idx]
        attr   = attr[keep_idx]

    return {
        "tokens":          tokens,
        "attributions":    attr.detach().cpu(),
        "time":            elapsed,
        "predicted_label": bundle["pred_id"],
        # raw tensors for metric functions
        "model":           model,
        "nn_forward_func": bundle["nn_forward_func"],
        "input_embed":     bundle["X"],
        "attention_mask":  bundle["attention_mask"],
        "position_embed":  bundle["position_embed"],
        "type_embed":      bundle["type_embed"],
        "attr_full":       attr_full,
    }


# ---------------------------------------------------------------------------
# Ablation 1: L1 / L2 norm gate  (still delta-weighted)
# ---------------------------------------------------------------------------

def pace_gradient_l12(
    sentence: str,
    a: float = 0.0,
    b: float = 1.0,
    steps: int = 100,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
    baseline: str = "mask",
    norm: Literal["l1", "l2"] = "l2",
) -> Dict[str, Any]:
    """
    PACE variant: attribution gate = Lp-norm of the embedding-space gradient
    vector per token (shape (d,)), aggregated with step delta weighting.

    For step t with interpolated embedding X_t:
        g_t[i]  = dL/dX_t[i]            shape (d,)   — raw gradient in embed space
        w_t[i]  = ||g_t[i]||_p          scalar per token
        w_t_n   = w_t / sum(w_t)        normalised across tokens
        attr   += w_t_n * delta_t       accumulated over steps

    delta_t = logit(X_t)[pred] − logit(X_{t−1})[pred]  (same as original PACE)

    Args:
        norm: 'l1' or 'l2'
    """
    bundle = _common_setup(sentence, model_name, device, baseline)
    model, X, X_baseline = bundle["model"], bundle["X"], bundle["X_baseline"]
    attention_mask, extra_kwargs = bundle["attention_mask"], bundle["extra_kwargs"]
    L, d, pred_id = bundle["L"], bundle["d"], bundle["pred_id"]

    t_vals = torch.linspace(a, b, steps, device=device, dtype=X.dtype)    # (steps,)
    coefs  = t_vals.unsqueeze(1).expand(steps, L).clone()                 # (steps, L)
    # Fix CLS / SEP to coef=1 (same convention as original)
    coefs[:, 0]  = 1.0
    coefs[:, -1] = 1.0

    # We need gradients w.r.t. X_inter directly (shape steps × L × d)
    X_inter = (X.squeeze(0) * coefs.unsqueeze(-1)
               + X_baseline.squeeze(0) * (1 - coefs.unsqueeze(-1)))
    X_inter = X_inter.detach().requires_grad_(True)   # (steps, L, d)

    attn_batch  = attention_mask.expand(steps, -1)
    extra_batch = {k: v.expand(steps, -1) for k, v in extra_kwargs.items()
                   if k == "token_type_ids"}

    start_time = time.perf_counter()

    out          = model(inputs_embeds=X_inter, attention_mask=attn_batch, **extra_batch)
    logits_batch = out.logits[:, pred_id]              # (steps,)

    delta    = logits_batch - torch.cat([logits_batch[:1], logits_batch[:-1]])
    delta[0] = 0.0

    # Gradient w.r.t. full embedding tensor — shape (steps, L, d)
    (grad_emb,) = torch.autograd.grad(logits_batch.sum(), X_inter)

    end_time = time.perf_counter()

    # Lp norm over embedding dimension → (steps, L)
    if norm == "l2":
        grad_norm_vec = grad_emb.norm(p=2, dim=-1)   # (steps, L)
    elif norm == "l1":
        grad_norm_vec = grad_emb.norm(p=1, dim=-1)   # (steps, L)
    else:
        raise ValueError(f"norm must be 'l1' or 'l2', got '{norm}'")

    # Normalise across tokens at each step, then weight by delta
    denom     = grad_norm_vec.sum(dim=1, keepdim=True) + 1e-10
    grad_n    = grad_norm_vec / denom                  # (steps, L)
    attr      = (grad_n * delta.unsqueeze(1)).sum(dim=0)  # (L,)

    attr_full = attr.detach().clone()
    return _pack_result(bundle, attr, attr_full, show_special_tokens,
                        end_time - start_time)


# ---------------------------------------------------------------------------
# Ablation 2: Scalar gate only, NO delta weighting
# ---------------------------------------------------------------------------

def pace_gradient_scalar(
    sentence: str,
    a: float = 0.0,
    b: float = 1.0,
    steps: int = 100,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
    baseline: str = "mask",
) -> Dict[str, Any]:
    """
    PACE variant: scalar gate (same as original), but WITHOUT delta weighting.

    The scalar gate ex is the same learnable per-token scalar added to the
    interpolation coefficient (as in the original classification code).
    Gradient w.r.t. ex is normalised across tokens at each step, then simply
    *averaged* over steps — no multiplication by logit delta.

    attr = (1/steps) * Σ_t  ( grad_ex_t / ||grad_ex_t||_1 )

    This isolates the contribution of the delta weighting by removing it.
    """
    bundle = _common_setup(sentence, model_name, device, baseline)
    model, X, X_baseline = bundle["model"], bundle["X"], bundle["X_baseline"]
    attention_mask, extra_kwargs = bundle["attention_mask"], bundle["extra_kwargs"]
    L, d, pred_id = bundle["L"], bundle["d"], bundle["pred_id"]

    t_vals     = torch.linspace(a, b, steps, device=device, dtype=X.dtype)
    coefs_base = t_vals.unsqueeze(1).expand(steps, L).clone()
    coefs_base[:, 0]  = 1.0
    coefs_base[:, -1] = 1.0

    # Scalar gate — same construction as original pace_gradient_classification
    ex           = torch.zeros(steps, L, 1, device=device, dtype=X.dtype,
                               requires_grad=True)
    itepolated_o = coefs_base.unsqueeze(-1) + ex         # (steps, L, 1)
    iterpolated  = itepolated_o.tile((1, 1, d))          # (steps, L, d)

    X_inter = (X.squeeze(0) * iterpolated
               + X_baseline.squeeze(0) * (1 - iterpolated))

    attn_batch  = attention_mask.expand(steps, -1)
    extra_batch = {k: v.expand(steps, -1) for k, v in extra_kwargs.items()
                   if k == "token_type_ids"}

    start_time = time.perf_counter()

    out          = model(inputs_embeds=X_inter, attention_mask=attn_batch, **extra_batch)
    logits_batch = out.logits[:, pred_id]                # (steps,)

    # ── KEY DIFFERENCE: no delta; gradient only ──────────────────────────────
    (grad_ex,) = torch.autograd.grad(logits_batch.sum(), ex)
    grad_ex     = grad_ex.squeeze(-1)                    # (steps, L)

    end_time = time.perf_counter()

    # Normalise across tokens per step, then average over steps (no delta weight)
    denom    = grad_ex.sum(dim=1, keepdim=True) + 1e-10
    grad_n   = grad_ex / denom                           # (steps, L)
    attr     = grad_n.mean(dim=0)                        # (L,)  ← average, not delta-sum

    attr_full = attr.detach().clone()
    return _pack_result(bundle, attr, attr_full, show_special_tokens,
                        end_time - start_time)