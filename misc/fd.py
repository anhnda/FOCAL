"""
Forward-Only PACE Gradient Attribution (FD variant) — vectorized
====================================================================

A gradient-free variant of PACE-Grad that replaces autograd-based gate
sensitivities `v_i^{(t)} = d y_hat / d g_i` with finite-difference probes
along the same integration path. The perturbation size is the natural
step size of the path itself (h = 1/(steps-1)), so no extra hyperparameter
is introduced.

For each integration step t = 1..m, at base gate g^{(t-1)} = (t-1)/m * 1:
    u_i^{(t)} = y_hat( g^{(t-1)} + h * e_i ) - y_hat( g^{(t-1)} )

This costs (L+1) forward passes per step but is fully batched, runs under
torch.no_grad(), and deploys on quantized models where autograd is
unavailable. The attribution formula is otherwise identical to PACE-Grad:

    alpha_i^{(t)} = u_i^{(t)} / sum_j u_j^{(t)}
    PACE-FD_i     = sum_t alpha_i^{(t)} * delta_y_hat^{(t)}

where delta_y_hat^{(t)} is the *true* joint output change (preserves
exact completeness; see PACE-N analysis).

This version vectorizes the per-step loop into a single batched forward:
all `steps - 1` active steps are assembled into one tensor of shape
(T, L+2, L), flattened to (T*(L+2), L), and pushed through the model in
chunks. Output is numerically equivalent to the loop version (bit-exact
in fp64; within rounding in fp16/bf16).

Interface is intentionally identical to pace_gradients.py — same args,
same return dict keys — so eval scripts can swap implementations cleanly.
"""
import time
import torch
import random
import inspect
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any
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
# Baseline embedding factory (identical to pace_gradients.py)
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
# Model / tokenizer cache (identical to pace_gradients.py)
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
# Internal helper: chunked forward over a large batch of gated embeddings
# ---------------------------------------------------------------------------

def _chunked_forward_logits(
    model,
    X: torch.Tensor,            # (1, L, d)
    X_baseline: torch.Tensor,   # (1, L, d)
    gate_batch: torch.Tensor,   # (B, L) in [0, 1]
    attention_mask: torch.Tensor,
    extra_kwargs: Dict[str, torch.Tensor],
    chunk_size: int,
    output_kind: str,           # 'classification' | 'qa'
):
    """
    Run a (possibly large) batch of gated forward passes under no_grad,
    splitting into chunks to bound memory. Returns logits stacked along
    the batch axis.

    For 'classification': returns (B, num_classes)
    For 'qa'            : returns (start_logits (B, L), end_logits (B, L))
    """
    B, L = gate_batch.shape
    d = X.shape[2]

    if output_kind == "classification":
        out_chunks = []
    else:
        start_chunks, end_chunks = [], []

    X_sq    = X.squeeze(0)            # (L, d)
    Xref_sq = X_baseline.squeeze(0)   # (L, d)

    for i in range(0, B, chunk_size):
        j = min(i + chunk_size, B)
        g_chunk = gate_batch[i:j]                          # (b, L)
        g_exp   = g_chunk.unsqueeze(-1)                    # (b, L, 1)
        X_inter = X_sq * g_exp + Xref_sq * (1.0 - g_exp)   # (b, L, d)

        attn_chunk = attention_mask.expand(j - i, -1)
        extra_chunk = {}
        for k, v in extra_kwargs.items():
            extra_chunk[k] = v.expand(j - i, -1)

        with torch.no_grad():
            out = model(
                inputs_embeds=X_inter,
                attention_mask=attn_chunk,
                **extra_chunk,
            )

        if output_kind == "classification":
            out_chunks.append(out.logits)
        else:
            start_chunks.append(out.start_logits)
            end_chunks.append(out.end_logits)

    if output_kind == "classification":
        return torch.cat(out_chunks, dim=0)   # (B, num_classes)
    else:
        return torch.cat(start_chunks, dim=0), torch.cat(end_chunks, dim=0)


# ---------------------------------------------------------------------------
# Internal helper: assemble the full (T, L+2, L) gate block
# ---------------------------------------------------------------------------

def _build_gate_block(
    g_path: torch.Tensor,   # (steps, L)
    eye_L: torch.Tensor,    # (L, L)
    h: float,
) -> torch.Tensor:
    """
    Build the per-step probe batch for every active step in one shot.

    Returns
    -------
    gate_block : (T, L+2, L) where T = steps - 1
        row 0       : base   = g_path[t-1]
        rows 1..L   : probes = g_path[t-1] + h * e_i, clamped to <= 1
        row L+1     : joint  = g_path[t]
    """
    g_prev_all = g_path[:-1]                                  # (T, L)
    g_curr_all = g_path[1:]                                   # (T, L)

    # probes_all[t, i, :] = g_path[t] + h * e_i (clamped)
    probes_all = g_prev_all.unsqueeze(1) + h * eye_L.unsqueeze(0)
    probes_all = probes_all.clamp(max=1.0)                    # (T, L, L)

    gate_block = torch.cat(
        [
            g_prev_all.unsqueeze(1),                          # (T, 1, L)
            probes_all,                                       # (T, L, L)
            g_curr_all.unsqueeze(1),                          # (T, 1, L)
        ],
        dim=1,
    )                                                         # (T, L+2, L)
    return gate_block


# ---------------------------------------------------------------------------
# Classification attribution (forward-only, vectorized)
# ---------------------------------------------------------------------------

def pace_gradient_classification(
    sentence: str,
    a: float = 0.0,
    b: float = 1.0,
    steps: int = 100,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
    baseline: str = "mask",
    chunk_size: int = 64,
) -> Dict[str, Any]:
    """
    Forward-only PACE attribution for sentence classification.

    Mirrors pace_gradients.pace_gradient_classification's interface and
    return-dict keys. The only behavioural difference is that gate
    sensitivities are computed via finite differences along the path
    instead of autograd.

    Parameters
    ----------
    chunk_size : int
        Max batch size for a single forward pass. The full integration
        emits (steps - 1) * (L + 2) forwards; we chunk to bound GPU memory.
    """
    global cache

    # Per-model helpers (same as pace_gradients.py)
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
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    fwd_params = inspect.signature(model.forward).parameters
    extra_kwargs = {}
    if "token_type_ids" in fwd_params and token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)   # (1, L, d)

    L, d = X.shape[1], X.shape[2]

    X_RefMask = get_baseline_embedding(baseline, embed, tokenizer, X, device)

    # Predicted class on the original input
    with torch.no_grad():
        logits0 = model(
            inputs_embeds=X,
            attention_mask=attention_mask,
            **extra_kwargs,
        ).logits[0]
    pred_id = int(logits0.argmax().item())

    # ------------------------------------------------------------------
    # Build the path: g^{(t)} = t/(steps-1) * 1, CLS/SEP gates pinned.
    # ------------------------------------------------------------------
    t_vals = torch.linspace(a, b, steps, device=device, dtype=X.dtype)  # (steps,)
    g_path = t_vals.unsqueeze(1).expand(steps, L).clone()               # (steps, L)
    g_path[:, 0]  = 1.0   # CLS pinned (matches pace_gradients.py)
    g_path[:, -1] = 1.0   # final special token pinned

    # Step size for finite difference. Natural choice: h = (b - a) / (steps - 1).
    h = float((b - a) / max(steps - 1, 1))

    eye_L = torch.eye(L, device=device, dtype=X.dtype)   # (L, L)

    start_time = time.perf_counter()

    T = steps - 1   # number of active integration steps
    if T > 0:
        # (T, L+2, L) -> (T*(L+2), L)
        gate_block = _build_gate_block(g_path, eye_L, h)
        gate_flat  = gate_block.reshape(T * (L + 2), L)

        logits_flat = _chunked_forward_logits(
            model=model,
            X=X,
            X_baseline=X_RefMask,
            gate_batch=gate_flat,
            attention_mask=attention_mask,
            extra_kwargs=extra_kwargs,
            chunk_size=chunk_size,
            output_kind="classification",
        )                                                  # (T*(L+2), C)

        y_target = logits_flat[:, pred_id].reshape(T, L + 2)   # (T, L+2)

        y_base  = y_target[:, 0:1]                             # (T, 1)
        y_probe = y_target[:, 1:1 + L]                         # (T, L)
        y_joint = y_target[:, 1 + L]                           # (T,)

        # u_i^{(t)} = y_hat(base + h e_i) - y_hat(base)
        u = y_probe - y_base                                   # (T, L)

        # alpha normalized per step (signed; matches paper)
        alpha = u / (u.sum(dim=1, keepdim=True) + 1e-10)       # (T, L)

        # True joint output change per step (preserves exact completeness)
        delta_y = (y_joint - y_base.squeeze(1)).unsqueeze(1)   # (T, 1)

        attr = (alpha * delta_y).sum(dim=0)                    # (L,)
    else:
        attr = torch.zeros(L, device=device, dtype=X.dtype)

    end_time = time.perf_counter()

    # position/type embeddings for caller's metric calls
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
        # raw tensors for eval script to compute metrics
        "model":           model,
        "nn_forward_func": nn_forward_func,
        "input_embed":     X,
        "attention_mask":  attention_mask,
        "position_embed":  position_embed,
        "type_embed":      type_embed,
        "attr_full":       attr_full,
    }


# ---------------------------------------------------------------------------
# QA attribution (forward-only, vectorized)
# ---------------------------------------------------------------------------

def pace_gradient_qa(
    question: str,
    context: str,
    a: float = 0.0,
    b: float = 1.0,
    steps: int = 101,
    model_name: str = "deepset/bert-base-cased-squad2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
    baseline: str = "mask",
    chunk_size: int = 64,
) -> Dict[str, Any]:
    """
    Forward-only PACE attribution for extractive QA.

    Mirrors pace_gradients.pace_gradient_qa's interface and return-dict
    keys. Computes separate start/end attributions via the same FD path
    construction as the classification variant.
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

    # Pinned positions: CLS, SEP, PAD, and other specials -> gate=1
    ids     = input_ids[0]
    cls_id  = tokenizer.cls_token_id
    sep_id  = tokenizer.sep_token_id
    is_special = special_tokens_mask[0].bool()
    is_pad     = (attention_mask[0] == 0)
    is_cls = (ids == cls_id) if cls_id is not None else torch.zeros(L, dtype=torch.bool, device=device)
    is_sep = (ids == sep_id) if sep_id is not None else torch.zeros(L, dtype=torch.bool, device=device)
    fixed = (is_special | is_pad | is_cls | is_sep)   # (L,)

    # Path with pinned positions held at 1
    t_vals = torch.linspace(a, b, steps, device=device, dtype=X.dtype)
    g_path = t_vals.unsqueeze(1).expand(steps, L).clone()
    g_path[:, fixed] = 1.0

    h = float((b - a) / max(steps - 1, 1))
    eye_L = torch.eye(L, device=device, dtype=X.dtype)

    start_time = time.perf_counter()

    T = steps - 1
    if T > 0:
        gate_block = _build_gate_block(g_path, eye_L, h)
        gate_flat  = gate_block.reshape(T * (L + 2), L)

        start_logits_flat, end_logits_flat = _chunked_forward_logits(
            model=model,
            X=X,
            X_baseline=X_baseline,
            gate_batch=gate_flat,
            attention_mask=attention_mask,
            extra_kwargs=extra_kwargs,
            chunk_size=chunk_size,
            output_kind="qa",
        )                                                  # each (T*(L+2), L)

        ys = start_logits_flat[:, start_idx].reshape(T, L + 2)   # (T, L+2)
        ye = end_logits_flat[:,   end_idx  ].reshape(T, L + 2)   # (T, L+2)

        # --- start head ---
        ys_base  = ys[:, 0:1]                                # (T, 1)
        ys_probe = ys[:, 1:1 + L]                            # (T, L)
        ys_joint = ys[:, 1 + L]                              # (T,)
        u_s      = ys_probe - ys_base
        alpha_s  = u_s / (u_s.sum(dim=1, keepdim=True) + 1e-10)
        delta_s  = (ys_joint - ys_base.squeeze(1)).unsqueeze(1)
        attr_start = (alpha_s * delta_s).sum(dim=0)          # (L,)

        # --- end head ---
        ye_base  = ye[:, 0:1]
        ye_probe = ye[:, 1:1 + L]
        ye_joint = ye[:, 1 + L]
        u_e      = ye_probe - ye_base
        alpha_e  = u_e / (u_e.sum(dim=1, keepdim=True) + 1e-10)
        delta_e  = (ye_joint - ye_base.squeeze(1)).unsqueeze(1)
        attr_end = (alpha_e * delta_e).sum(dim=0)            # (L,)
    else:
        attr_start = torch.zeros(L, device=device, dtype=X.dtype)
        attr_end   = torch.zeros(L, device=device, dtype=X.dtype)

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