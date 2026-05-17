"""
PACE Gradient Attribution for Classification and QA Tasks
Supports configurable baselines: mask, pad, mean, zero, random
"""
import time
import torch
import random
import inspect
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from xai_metrics import *

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

cache = {}


# ---------------------------------------------------------------------------
# Baseline embedding factory
# ---------------------------------------------------------------------------

def get_baseline_embedding(
    baseline: str,
    embed: torch.nn.Embedding,
    tokenizer,
    X: torch.Tensor,   # (1, L, d)
    device: str,
) -> torch.Tensor:
    """
    Return a baseline embedding of shape (1, L, d).

    Args:
        baseline : 'mask' | 'pad' | 'zero' | 'mean' | 'random'
        embed    : model's input embedding layer
        tokenizer: needed for mask/pad token ids
        X        : original embeddings (1, L, d) — for shape/dtype reference
        device   : target device

    Returns:
        X_baseline (1, L, d), detached
    """
    L, d = X.shape[1], X.shape[2]

    if baseline == "mask":
        token_id = tokenizer.mask_token_id or tokenizer.pad_token_id
        with torch.no_grad():
            base_emb = embed(torch.tensor([[token_id]], device=device))  # (1, 1, d)
        return base_emb.expand(1, L, d).clone()

    elif baseline == "pad":
        token_id = tokenizer.pad_token_id
        with torch.no_grad():
            base_emb = embed(torch.tensor([[token_id]], device=device))  # (1, 1, d)
        return base_emb.expand(1, L, d).clone()

    elif baseline == "zero":
        return torch.zeros(1, L, d, device=device, dtype=X.dtype)

    elif baseline == "mean":
        with torch.no_grad():
            mean_vec = embed.weight.mean(dim=0)                   # (d,)
        return mean_vec.view(1, 1, d).expand(1, L, d).clone()

    elif baseline == "random":
        vocab_size = embed.weight.shape[0]
        rand_id = torch.randint(0, vocab_size, (1,), device=device)
        with torch.no_grad():
            base_emb = embed(rand_id.unsqueeze(0))                # (1, 1, d)
        return base_emb.expand(1, L, d).clone()

    else:
        raise ValueError(
            f"Unknown baseline '{baseline}'. "
            "Choose from: mask, pad, zero, mean, random"
        )


# ---------------------------------------------------------------------------
# Model / tokenizer cache
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
    cache[key] = (model, tokenizer)
    return model, tokenizer


# ---------------------------------------------------------------------------
# QA attribution
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
) -> Dict[str, Any]:
    model, tokenizer = get_model_tokenizer(model_name, device, type="qa")

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
    special_tokens_mask = enc.get("special_tokens_mask", torch.zeros_like(input_ids)).to(device)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    fwd_params = inspect.signature(model.forward).parameters
    extra_kwargs = {}
    if "token_type_ids" in fwd_params and token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)                          # (1, L, d)
        outputs0 = model(inputs_embeds=X, attention_mask=attention_mask, **extra_kwargs)
        start_logits0 = outputs0.start_logits[0]      # (L,)
        end_logits0   = outputs0.end_logits[0]        # (L,)

    L, d = X.shape[1], X.shape[2]
    start_idx = int(start_logits0.argmax().item())
    end_idx   = int(end_logits0.argmax().item())
    if end_idx < start_idx:
        end_idx = start_idx

    start_prob = F.softmax(start_logits0, dim=0)[start_idx]
    end_prob   = F.softmax(end_logits0,   dim=0)[end_idx]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    pred_answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1])

    # --- Baseline ---
    X_baseline = get_baseline_embedding(baseline, embed, tokenizer, X, device)  # (1, L, d)

    # Fixed positions (CLS, SEP, PAD) — keep coef=1
    ids = input_ids[0]
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    is_special = special_tokens_mask[0].bool()
    is_pad     = (attention_mask[0] == 0)
    is_cls     = (ids == cls_id) if cls_id is not None else torch.zeros(L, dtype=torch.bool, device=device)
    is_sep     = (ids == sep_id) if sep_id is not None else torch.zeros(L, dtype=torch.bool, device=device)
    fixed      = (is_special | is_pad | is_cls | is_sep)  # (L,)

    # === Batched integration ===
    t_vals     = torch.linspace(a, b, steps, device=device, dtype=X.dtype)  # (steps,)
    coefs_base = t_vals.unsqueeze(1).expand(steps, L).clone()
    coefs_base[:, fixed] = 1.0

    coefs2     = coefs_base.detach().requires_grad_(True)
    coefs2_exp = coefs2.unsqueeze(-1).expand(steps, L, d)
    X_inter2   = X.squeeze(0) * coefs2_exp + X_baseline.squeeze(0) * (1 - coefs2_exp)

    attn_batch = attention_mask.expand(steps, -1)
    extra_batch = {}
    if "token_type_ids" in extra_kwargs:
        extra_batch["token_type_ids"] = extra_kwargs["token_type_ids"].expand(steps, -1)

    start_time = time.perf_counter()

    out2          = model(inputs_embeds=X_inter2, attention_mask=attn_batch, **extra_batch)
    start_scores2 = out2.start_logits[:, start_idx]   # (steps,)
    end_scores2   = out2.end_logits[:,   end_idx]     # (steps,)

    delta_start2 = start_scores2 - torch.cat([start_scores2[:1], start_scores2[:-1]])
    delta_end2   = end_scores2   - torch.cat([end_scores2[:1],   end_scores2[:-1]])

    (grad_start2,) = torch.autograd.grad(start_scores2.sum(), coefs2, retain_graph=True)
    (grad_end2,)   = torch.autograd.grad(end_scores2.sum(),   coefs2, retain_graph=False)

    end_time = time.perf_counter()

    grad_start_n = grad_start2 / (grad_start2.sum(dim=1, keepdim=True) + 1e-10)
    grad_end_n   = grad_end2   / (grad_end2.sum(  dim=1, keepdim=True) + 1e-10)

    attr_start = (grad_start_n * delta_start2.unsqueeze(1)).sum(dim=0)  # (L,)
    attr_end   = (grad_end_n   * delta_end2.unsqueeze(1)  ).sum(dim=0)  # (L,)

    # For metrics: expose the single baseline token embedding (1, d)
    base_token_emb      = X_baseline[0, 0:1, :]     # (1, d) — representative token
    special_tokens_mask_out = fixed

    tokens_out     = tokens.copy()
    attr_start_out = attr_start.clone()
    attr_end_out   = attr_end.clone()

    if not show_special_tokens:
        special_ids_set = set(tokenizer.all_special_ids)
        keep_idx       = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special_ids_set]
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


# ---------------------------------------------------------------------------
# Classification attribution
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
) -> Dict[str, Any]:
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

    enc = tokenizer(sentence, return_tensors="pt", truncation=True,
                    return_special_tokens_mask=True)
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

    X_RefMask = get_baseline_embedding(baseline, embed, tokenizer, X, device)  # (1, L, d)

    with torch.no_grad():
        logits0 = model(inputs_embeds=X, attention_mask=attention_mask, **extra_kwargs).logits[0]
    pred_id = int(logits0.argmax().item())

    t_vals     = torch.linspace(a, b, steps, device=device, dtype=X.dtype)
    coefs_base = t_vals.unsqueeze(1).expand(steps, L).clone()
    coefs_base[:, 0]  = 1.0
    coefs_base[:, -1] = 1.0

    ex           = torch.zeros(steps, L, 1, device=device, dtype=X.dtype, requires_grad=True)
    itepolated_o = coefs_base.unsqueeze(-1) + ex
    iterpolated  = itepolated_o.tile((1, 1, d))

    X_inter = (X.squeeze(0) * iterpolated
               + X_RefMask.squeeze(0) * (1 - iterpolated))

    attn_batch  = attention_mask.expand(steps, -1)
    extra_batch = {}
    if "token_type_ids" in extra_kwargs:
        extra_batch["token_type_ids"] = extra_kwargs["token_type_ids"].expand(steps, -1)

    start_time = time.perf_counter()

    out          = model(inputs_embeds=X_inter, attention_mask=attn_batch, **extra_batch)
    logits_batch = out.logits[:, pred_id]

    delta    = logits_batch - torch.cat([logits_batch[:1], logits_batch[:-1]])
    delta[0] = 0.0

    (grad_ex,) = torch.autograd.grad(logits_batch.sum(), ex)
    grad_ex    = grad_ex.squeeze(-1)

    grad_norm = grad_ex / (grad_ex.sum(dim=1, keepdim=True) + 1e-10)
    attr      = (grad_norm * delta.unsqueeze(1)).sum(dim=0)  # (L,)

    end_time = time.perf_counter()

    # position/type embeddings for caller's metric calls
    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

    attr_full = attr.detach().clone()   # unfiltered, on device

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    if not show_special_tokens:
        special_ids_set = set(tokenizer.all_special_ids)
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special_ids_set]
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