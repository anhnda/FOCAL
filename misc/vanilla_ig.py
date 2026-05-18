"""
Integrated Gradients Attribution for Classification Tasks
- No delta weighting (pure IG: sum of gradients along path)
- Gradients taken w.r.t. interpolated embeddings directly (not coef scalars)
- Final attribution = L2 norm over embedding dim D per token
Supports configurable baselines: mask, pad, mean, zero, random
"""
import time
import torch
import random
import inspect
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pace_gradients import get_baseline_embedding

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

cache = {}
def ig_qa(
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
    from transformers import AutoModelForQuestionAnswering
    from pace_gradients import get_model_tokenizer

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
        start_logits0 = outputs0.start_logits[0]
        end_logits0   = outputs0.end_logits[0]

    L, d = X.shape[1], X.shape[2]
    start_idx = int(start_logits0.argmax().item())
    end_idx   = int(end_logits0.argmax().item())
    if end_idx < start_idx:
        end_idx = start_idx

    start_prob = F.softmax(start_logits0, dim=0)[start_idx]
    end_prob   = F.softmax(end_logits0,   dim=0)[end_idx]

    tokens      = tokenizer.convert_ids_to_tokens(input_ids[0])
    pred_answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1])

    X_baseline = get_baseline_embedding(baseline, embed, tokenizer, X, device)  # (1, L, d)

    # Fixed positions — keep at original embedding (coef=1 in PACE; here we
    # simply do not interpolate them — same effect via the fixed mask below)
    ids    = input

def ig_classification(
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

    X_baseline = get_baseline_embedding(baseline, embed, tokenizer, X, device)  # (1, L, d)

    with torch.no_grad():
        logits0 = model(inputs_embeds=X, attention_mask=attention_mask,
                        **extra_kwargs).logits[0]
    pred_id = int(logits0.argmax().item())

    # Interpolated embeddings: X_inter[i] = baseline + t_i * (X - baseline)
    # Shape: (steps, L, d) — grad leaf is X_inter directly
    t_vals = torch.linspace(a, b, steps, device=device, dtype=X.dtype)  # (steps,)

    # (steps, L, d): each step is a full embedding matrix
    X_inter = (
        X_baseline.squeeze(0).unsqueeze(0)                          # (1, L, d)
        + t_vals.view(steps, 1, 1)                                  # (steps, 1, 1)
        * (X.squeeze(0) - X_baseline.squeeze(0)).unsqueeze(0)      # (1, L, d)
    ).requires_grad_(True)   # (steps, L, d) — grad leaf

    attn_batch  = attention_mask.expand(steps, -1)
    extra_batch = {}
    if "token_type_ids" in extra_kwargs:
        extra_batch["token_type_ids"] = extra_kwargs["token_type_ids"].expand(steps, -1)

    start_time = time.perf_counter()

    out          = model(inputs_embeds=X_inter, attention_mask=attn_batch, **extra_batch)
    logits_batch = out.logits[:, pred_id]   # (steps,)

    # Gradient w.r.t. interpolated embeddings directly — shape (steps, L, d)
    (grad_embed,) = torch.autograd.grad(logits_batch.sum(), X_inter)

    end_time = time.perf_counter()

    # Pure IG: mean gradient over steps, no delta weighting
    # Shape: (L, d)
    mean_grad = grad_embed.mean(dim=0)   # (L, d)

    # Final attribution: L2 norm over embedding dim d -> (L,)
    attr_full = mean_grad.norm(dim=-1).detach()   # (L,)

    # position/type embeddings for caller's metric calls
    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    attr   = attr_full.clone()

    if not show_special_tokens:
        special_ids_set = set(tokenizer.all_special_ids)
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist())
                    if tid not in special_ids_set]
        tokens = [tokens[i] for i in keep_idx]
        attr   = attr_full[keep_idx]

    return {
        "tokens":          tokens,
        "attributions":    attr.cpu(),
        "time":            end_time - start_time,
        "predicted_label": pred_id,
        # raw tensors for eval script
        "model":           model,
        "nn_forward_func": nn_forward_func,
        "input_embed":     X,
        "attention_mask":  attention_mask,
        "position_embed":  position_embed,
        "type_embed":      type_embed,
        "attr_full":       attr_full,
    }