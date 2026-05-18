"""
LIME (Local Interpretable Model-agnostic Explanations) for Transformer
Token Attribution — forward-only, vectorized
====================================================================

A reference implementation of LIME [Ribeiro et al., KDD 2016] adapted to
token-level attribution for Transformer classifiers and extractive QA
models. The interface mirrors fd.py (the FOCAL implementation) so the
eval script can swap in this module by changing a single import.

Algorithm
---------
For an input sentence tokenized to L tokens (with special tokens CLS/SEP
pinned as always-present), LIME:

  1. Draws N binary masks z^{(n)} in {0, 1}^L. Pinned positions are
     forced to 1; the remaining positions are sampled i.i.d. Bernoulli
     with probability p_keep (default 0.5).
  2. For each mask, constructs a perturbed embedding sequence
         X_pert[i] = X[i]           if z_i = 1
                   = X_baseline[i]  if z_i = 0
     and runs a single forward pass under torch.no_grad() to obtain the
     target-class probability f(z^{(n)}).
  3. Weighs each sample by an exponential kernel on cosine distance
     between z^{(n)} and the all-ones mask z* = 1:
         pi(z) = exp( -D(z, z*)^2 / sigma^2 ),
         D(z, z*) = 1 - cosine_similarity(z, z*).
  4. Fits weighted ridge regression of f(z^{(n)}) on z^{(n)} (over the
     non-pinned positions). The learned coefficients are the per-token
     attributions.

This is faithful to the original LIME formulation but specialized to
token features (binary present/absent) rather than tabular features.
Pinned positions receive zero attribution since they are constant across
all samples and therefore non-identifiable in the regression.

Notes
-----
- LIME is *not* deterministic: attributions depend on the random masks.
  We seed torch's RNG at import for reproducibility within a process,
  but expose `seed` per-call for finer control.
- All forward passes are batched and chunked for memory bounds. Total
  forward cost: N forwards (vs. FOCAL's (steps-1)*(L+2)).
- The fitted intercept absorbs the "all tokens removed" baseline; we
  report only the per-token weights.
"""
import time
import torch
import random
import inspect
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

cache = {}


# ---------------------------------------------------------------------------
# Baseline embedding factory (identical to fd.py for parity in evaluation)
# ---------------------------------------------------------------------------

def get_baseline_embedding(
    baseline: str,
    embed: torch.nn.Embedding,
    tokenizer,
    X: torch.Tensor,   # (1, L, d)
    device: str,
) -> torch.Tensor:
    """Return a baseline embedding of shape (1, L, d), detached."""
    L, d = X.shape[1], X.shape[2]

    if baseline == "mask":
        token_id = tokenizer.mask_token_id or tokenizer.pad_token_id
        with torch.no_grad():
            base_emb = embed(torch.tensor([[token_id]], device=device))
        return base_emb.expand(1, L, d).clone()

    elif baseline == "pad":
        token_id = tokenizer.pad_token_id
        with torch.no_grad():
            base_emb = embed(torch.tensor([[token_id]], device=device))
        return base_emb.expand(1, L, d).clone()

    elif baseline == "zero":
        return torch.zeros(1, L, d, device=device, dtype=X.dtype)

    elif baseline == "mean":
        with torch.no_grad():
            mean_vec = embed.weight.mean(dim=0)
        return mean_vec.view(1, 1, d).expand(1, L, d).clone()

    elif baseline == "random":
        vocab_size = embed.weight.shape[0]
        rand_id = torch.randint(0, vocab_size, (1,), device=device)
        with torch.no_grad():
            base_emb = embed(rand_id.unsqueeze(0))
        return base_emb.expand(1, L, d).clone()

    else:
        raise ValueError(
            f"Unknown baseline '{baseline}'. "
            "Choose from: mask, pad, zero, mean, random"
        )


# ---------------------------------------------------------------------------
# Model / tokenizer cache (identical to fd.py)
# ---------------------------------------------------------------------------

def get_model_tokenizer(model_name: str, device: str, type: str):
    key = (model_name, device, type)
    if key in cache:
        return cache[key]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if type == "qa":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    elif type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    else:
        raise ValueError(f"Unknown model type: {type}")
    model.eval()
    cache[key] = (model, tokenizer)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Internal helper: chunked forward over a large batch of masked embeddings
# ---------------------------------------------------------------------------

def _chunked_forward_logits_lime(
    model,
    X: torch.Tensor,            # (1, L, d)
    X_baseline: torch.Tensor,   # (1, L, d)
    mask_batch: torch.Tensor,   # (N, L) in {0, 1}, possibly float
    attention_mask: torch.Tensor,
    extra_kwargs: Dict[str, torch.Tensor],
    chunk_size: int,
    output_kind: str,           # 'classification' | 'qa'
):
    """
    Run N masked forward passes under no_grad in chunks.

    For a mask z in {0,1}^L, the perturbed input is
        X_pert = z * X + (1 - z) * X_baseline   (broadcast over d).

    Returns:
        classification: logits tensor of shape (N, num_classes)
        qa:             (start_logits (N, L), end_logits (N, L))
    """
    N, L = mask_batch.shape

    if output_kind == "classification":
        out_chunks = []
    else:
        start_chunks, end_chunks = [], []

    X_sq    = X.squeeze(0)            # (L, d)
    Xref_sq = X_baseline.squeeze(0)   # (L, d)

    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        z_chunk = mask_batch[i:j].to(X.dtype)               # (b, L)
        z_exp   = z_chunk.unsqueeze(-1)                     # (b, L, 1)
        X_pert  = X_sq * z_exp + Xref_sq * (1.0 - z_exp)    # (b, L, d)

        attn_chunk = attention_mask.expand(j - i, -1)
        extra_chunk = {}
        for k, v in extra_kwargs.items():
            extra_chunk[k] = v.expand(j - i, -1)

        with torch.no_grad():
            out = model(
                inputs_embeds=X_pert,
                attention_mask=attn_chunk,
                **extra_chunk,
            )

        if output_kind == "classification":
            out_chunks.append(out.logits)
        else:
            start_chunks.append(out.start_logits)
            end_chunks.append(out.end_logits)

    if output_kind == "classification":
        return torch.cat(out_chunks, dim=0)
    else:
        return torch.cat(start_chunks, dim=0), torch.cat(end_chunks, dim=0)


# ---------------------------------------------------------------------------
# LIME kernel + weighted ridge regression
# ---------------------------------------------------------------------------

def _lime_kernel_weights(
    Z: torch.Tensor,           # (N, L_free) in {0, 1}
    sigma: float,
) -> torch.Tensor:
    """
    Exponential kernel weight on cosine distance to the all-ones vector,
    computed over the *free* (non-pinned) positions.

    pi(z) = exp( -D(z, 1)^2 / sigma^2 ),
    D(z, 1) = 1 - cos_sim(z, 1) = 1 - sum(z) / sqrt(L_free * ||z||).

    For an all-zero mask we set the cosine similarity to 0 (distance 1),
    matching LIME's convention.
    """
    N, L_free = Z.shape
    if L_free == 0:
        return torch.ones(N, device=Z.device, dtype=Z.dtype)

    z_norm   = Z.norm(dim=1)                        # (N,)
    ones_norm = float(L_free) ** 0.5
    dot       = Z.sum(dim=1)                        # (N,)
    denom     = (z_norm * ones_norm).clamp(min=1e-12)
    cos_sim   = dot / denom
    cos_sim[z_norm == 0] = 0.0                      # all-zero -> dist 1
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

    Returns the L_free-dim coefficient vector (intercept absorbed).

    Implementation: concatenate a constant column for the intercept,
    apply no penalty to the intercept row, solve via normal equations.
    """
    N, L_free = Z.shape
    if L_free == 0:
        return torch.zeros(0, device=Z.device, dtype=Z.dtype)

    # Augment with intercept column
    ones = torch.ones(N, 1, device=Z.device, dtype=Z.dtype)
    Z_aug = torch.cat([ones, Z.to(Z.dtype)], dim=1)        # (N, 1 + L_free)

    W = w.unsqueeze(1)                                     # (N, 1)
    A = Z_aug.transpose(0, 1) @ (W * Z_aug)                # (1+L_free, 1+L_free)
    b = Z_aug.transpose(0, 1) @ (W.squeeze(1) * y).unsqueeze(1)   # (1+L_free, 1)

    # Ridge penalty on coefficients only, not intercept
    reg = torch.eye(L_free + 1, device=Z.device, dtype=Z.dtype) * lam
    reg[0, 0] = 0.0
    A = A + reg

    # Solve A beta = b
    try:
        beta_full = torch.linalg.solve(A, b).squeeze(1)    # (1 + L_free,)
    except RuntimeError:
        # Fall back to lstsq for ill-conditioned systems
        beta_full = torch.linalg.lstsq(A, b).solution.squeeze(1)

    return beta_full[1:]    # drop intercept


# ---------------------------------------------------------------------------
# Internal helper: sample masks for the free positions
# ---------------------------------------------------------------------------

def _sample_masks(
    L: int,
    free_idx: torch.Tensor,    # (L_free,) long indices into [0, L)
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
    the first sample (LIME convention: anchor on the original input).
    """
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
    else:
        g = None

    L_free = free_idx.numel()
    N = n_samples

    Z_full = torch.ones(N, L, device=device, dtype=dtype)
    if L_free > 0:
        if g is not None:
            rand = torch.rand(N, L_free, device=device, generator=g)
        else:
            rand = torch.rand(N, L_free, device=device)
        bits = (rand < p_keep).to(dtype)
        Z_full[:, free_idx] = bits

    if include_full and N > 0:
        # Force the first row to be the all-ones (original-input) mask
        Z_full[0, :] = 1.0

    return Z_full


# ---------------------------------------------------------------------------
# Classification attribution
# ---------------------------------------------------------------------------

def lime_classification(
    sentence: str,
    n_samples: int = 1000,
    p_keep: float = 0.5,
    sigma: float = 0.25,
    ridge_lambda: float = 1.0,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
    baseline: str = "mask",
    chunk_size: int = 64,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    LIME attribution for sentence classification.

    Mirrors fd.pace_gradient_classification's return-dict keys exactly so
    that eval scripts can swap implementations cleanly.

    Parameters
    ----------
    n_samples : int
        Number of binary masks sampled around the input (LIME's N).
    p_keep : float
        Bernoulli probability of keeping each non-special token in a
        sample. 0.5 is LIME's default.
    sigma : float
        Bandwidth of the exponential kernel in cosine-distance space.
        0.25 is a common choice for text LIME.
    ridge_lambda : float
        L2 penalty for the surrogate ridge regression.
    seed : Optional[int]
        Seed for the mask sampler. None -> non-deterministic.
    """
    global cache

    if "distilbert" in model_name:
        from distilbert_helper import get_inputs, nn_forward_func
    elif "roberta" in model_name:
        from roberta_helper import get_inputs, nn_forward_func
    elif "bert" in model_name:
        from bert_helper import get_inputs, nn_forward_func
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    if cache.get(model_name) is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model     = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        cache[model_name] = {"model": model, "tokenizer": tokenizer}

    tokenizer = cache[model_name]["tokenizer"]
    model     = cache[model_name]["model"]
    model.eval()

    enc = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        return_special_tokens_mask=True,
    )
    enc            = {k: v.to(device) for k, v in enc.items()}
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    special_tokens_mask = enc.get(
        "special_tokens_mask", torch.zeros_like(input_ids)
    ).to(device)
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    fwd_params = inspect.signature(model.forward).parameters
    extra_kwargs = {}
    if "token_type_ids" in fwd_params and token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)
    L, d = X.shape[1], X.shape[2]

    X_baseline = get_baseline_embedding(baseline, embed, tokenizer, X, device)

    # Predicted class on the original input
    with torch.no_grad():
        logits0 = model(
            inputs_embeds=X,
            attention_mask=attention_mask,
            **extra_kwargs,
        ).logits[0]
    pred_id = int(logits0.argmax().item())

    # ------------------------------------------------------------------
    # Identify free (perturbable) positions: everything that is NOT a
    # special token. Matches fd.py's pinning of CLS / final-special.
    # ------------------------------------------------------------------
    is_special = special_tokens_mask[0].bool()                  # (L,)
    is_pad     = (attention_mask[0] == 0)
    fixed      = (is_special | is_pad)                          # (L,)
    free_idx   = torch.nonzero(~fixed, as_tuple=False).squeeze(-1)
    L_free     = int(free_idx.numel())

    start_time = time.perf_counter()

    if L_free == 0 or n_samples <= 0:
        attr = torch.zeros(L, device=device, dtype=X.dtype)
    else:
        # 1) Sample masks
        Z = _sample_masks(
            L=L,
            free_idx=free_idx,
            n_samples=n_samples,
            p_keep=p_keep,
            device=device,
            dtype=X.dtype,
            seed=seed,
            include_full=True,
        )                                                       # (N, L)

        # 2) Batched forward over masked embeddings
        logits_flat = _chunked_forward_logits_lime(
            model=model,
            X=X,
            X_baseline=X_baseline,
            mask_batch=Z,
            attention_mask=attention_mask,
            extra_kwargs=extra_kwargs,
            chunk_size=chunk_size,
            output_kind="classification",
        )                                                       # (N, C)
        probs = F.softmax(logits_flat, dim=-1)
        y     = probs[:, pred_id]                               # (N,)

        # 3) Kernel weights over free positions
        Z_free = Z[:, free_idx]                                 # (N, L_free)
        w      = _lime_kernel_weights(Z_free, sigma=sigma)      # (N,)

        # 4) Weighted ridge regression
        beta_free = _weighted_ridge(
            Z=Z_free, y=y, w=w, lam=ridge_lambda
        )                                                       # (L_free,)

        # Scatter into a full-length attribution vector; pinned positions = 0
        attr = torch.zeros(L, device=device, dtype=X.dtype)
        attr[free_idx] = beta_free.to(X.dtype)

    end_time = time.perf_counter()

    # Helpers for the eval-script's metric calls
    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

    attr_full = attr.detach().clone()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    if not show_special_tokens:
        special_ids_set = set(tokenizer.all_special_ids)
        keep_idx = [
            i for i, tid in enumerate(input_ids[0].tolist())
            if tid not in special_ids_set
        ]
        tokens = [tokens[i] for i in keep_idx]
        attr   = attr[keep_idx]

    return {
        "tokens":          tokens,
        "attributions":    attr.detach().cpu(),
        "time":            end_time - start_time,
        "predicted_label": pred_id,
        # raw tensors for eval-script metric calls (same keys as fd.py)
        "model":           model,
        "nn_forward_func": nn_forward_func,
        "input_embed":     X,
        "attention_mask":  attention_mask,
        "position_embed":  position_embed,
        "type_embed":      type_embed,
        "attr_full":       attr_full,
    }


# ---------------------------------------------------------------------------
# QA attribution
# ---------------------------------------------------------------------------

def lime_qa(
    question: str,
    context: str,
    n_samples: int = 1000,
    p_keep: float = 0.5,
    sigma: float = 0.25,
    ridge_lambda: float = 1.0,
    model_name: str = "deepset/bert-base-cased-squad2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
    baseline: str = "mask",
    chunk_size: int = 64,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    LIME attribution for extractive QA.

    Computes two separate ridge regressions — one for the start-token
    probability at the predicted start index, and one for the end-token
    probability at the predicted end index — sharing the same mask batch.
    Mirrors fd.pace_gradient_qa's return-dict keys.
    """
    model, tokenizer = get_model_tokenizer(model_name, device, type="qa")
    model.eval()

    enc = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
    )
    input_ids           = enc["input_ids"].to(device)
    attention_mask      = enc["attention_mask"].to(device)
    token_type_ids      = enc.get("token_type_ids", None)
    special_tokens_mask = enc.get(
        "special_tokens_mask", torch.zeros_like(input_ids)
    ).to(device)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    fwd_params = inspect.signature(model.forward).parameters
    extra_kwargs = {}
    if "token_type_ids" in fwd_params and token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)
        outputs0      = model(
            inputs_embeds=X,
            attention_mask=attention_mask,
            **extra_kwargs,
        )
        start_logits0 = outputs0.start_logits[0]
        end_logits0   = outputs0.end_logits[0]

    L, d = X.shape[1], X.shape[2]
    start_idx = int(start_logits0.argmax().item())
    end_idx   = int(end_logits0.argmax().item())
    if end_idx < start_idx:
        end_idx = start_idx

    start_prob = F.softmax(start_logits0, dim=0)[start_idx]
    end_prob   = F.softmax(end_logits0,   dim=0)[end_idx]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    pred_answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1])

    X_baseline = get_baseline_embedding(baseline, embed, tokenizer, X, device)

    # Pinned positions: CLS, SEP, PAD, all specials
    ids     = input_ids[0]
    cls_id  = tokenizer.cls_token_id
    sep_id  = tokenizer.sep_token_id
    is_special = special_tokens_mask[0].bool()
    is_pad     = (attention_mask[0] == 0)
    is_cls = (ids == cls_id) if cls_id is not None else torch.zeros(L, dtype=torch.bool, device=device)
    is_sep = (ids == sep_id) if sep_id is not None else torch.zeros(L, dtype=torch.bool, device=device)
    fixed  = (is_special | is_pad | is_cls | is_sep)            # (L,)
    free_idx = torch.nonzero(~fixed, as_tuple=False).squeeze(-1)
    L_free   = int(free_idx.numel())

    start_time = time.perf_counter()

    if L_free == 0 or n_samples <= 0:
        attr_start = torch.zeros(L, device=device, dtype=X.dtype)
        attr_end   = torch.zeros(L, device=device, dtype=X.dtype)
    else:
        Z = _sample_masks(
            L=L,
            free_idx=free_idx,
            n_samples=n_samples,
            p_keep=p_keep,
            device=device,
            dtype=X.dtype,
            seed=seed,
            include_full=True,
        )                                                       # (N, L)

        start_logits_flat, end_logits_flat = _chunked_forward_logits_lime(
            model=model,
            X=X,
            X_baseline=X_baseline,
            mask_batch=Z,
            attention_mask=attention_mask,
            extra_kwargs=extra_kwargs,
            chunk_size=chunk_size,
            output_kind="qa",
        )                                                       # each (N, L)
        start_probs = F.softmax(start_logits_flat, dim=-1)
        end_probs   = F.softmax(end_logits_flat,   dim=-1)
        y_s         = start_probs[:, start_idx]                 # (N,)
        y_e         = end_probs[:,   end_idx  ]                 # (N,)

        Z_free = Z[:, free_idx]                                 # (N, L_free)
        w      = _lime_kernel_weights(Z_free, sigma=sigma)      # (N,)

        beta_s = _weighted_ridge(Z=Z_free, y=y_s, w=w, lam=ridge_lambda)
        beta_e = _weighted_ridge(Z=Z_free, y=y_e, w=w, lam=ridge_lambda)

        attr_start = torch.zeros(L, device=device, dtype=X.dtype)
        attr_end   = torch.zeros(L, device=device, dtype=X.dtype)
        attr_start[free_idx] = beta_s.to(X.dtype)
        attr_end  [free_idx] = beta_e.to(X.dtype)

    end_time = time.perf_counter()

    base_token_emb          = X_baseline[0, 0:1, :]
    special_tokens_mask_out = fixed

    tokens_out     = tokens.copy()
    attr_start_out = attr_start.clone()
    attr_end_out   = attr_end.clone()

    if not show_special_tokens:
        special_ids_set = set(tokenizer.all_special_ids)
        keep_idx = [
            i for i, tid in enumerate(input_ids[0].tolist())
            if tid not in special_ids_set
        ]
        tokens_out     = [tokens[i] for i in keep_idx]
        attr_start_out = attr_start[keep_idx]
        attr_end_out   = attr_end[keep_idx]

    return {
        "tokens":              tokens_out,
        "attributions_start":  attr_start_out,
        "attributions_end":    attr_end_out,
        "time":                end_time - start_time,
        "predicted_answer":    pred_answer,
        "start_idx":           start_idx,
        "end_idx":             end_idx,
        "start_logit":         float(start_logits0[start_idx].item()),
        "end_logit":           float(end_logits0[end_idx].item()),
        "model":               model,
        "input_embed":         X,
        "attention_mask":      attention_mask,
        "token_type_ids":      token_type_ids,
        "base_token_emb":      base_token_emb,
        "special_tokens_mask": special_tokens_mask_out,
        "start_prob":          start_prob,
        "end_prob":            end_prob,
    }