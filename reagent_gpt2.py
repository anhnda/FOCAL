"""
reagent_gpt2.py
===============
ReAGent-style occlusion attribution for decoder-only GPT-2 on TellMeWhy.
Supports configurable baselines: zero, pad, mean
"""

import time
import math
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

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


def _build_base_embed(
    embed_layer: torch.nn.Embedding,
    input_embed: torch.Tensor,   # (1, T, D) reference shape/dtype
    baseline: str,
    tokenizer: GPT2TokenizerFast,
    device: str,
) -> torch.Tensor:
    """
    Build baseline embedding of shape (1, T, D).

    Args:
        baseline : 'zero' | 'pad' | 'mean'
        zero : all-zero vector — standard IG
        pad  : EOS token embedding broadcast across all positions
        mean : mean of vocabulary embedding matrix
    """
    if baseline == "zero":
        return torch.zeros_like(input_embed)

    elif baseline == "pad":
        pad_id  = torch.tensor([[tokenizer.eos_token_id]], device=device)
        pad_vec = embed_layer(pad_id).detach()             # (1, 1, D)
        return pad_vec.expand_as(input_embed).clone()

    elif baseline == "mean":
        mean_vec = embed_layer.weight.mean(dim=0, keepdim=True)   # (1, D)
        return mean_vec.unsqueeze(0).expand_as(input_embed).clone()

    else:
        raise ValueError(f"Unknown baseline '{baseline}'. Choose: zero | pad | mean")


def _hellinger(P: torch.Tensor, Q: torch.Tensor) -> float:
    P = P.float().clamp(min=0.0)
    Q = Q.float().clamp(min=0.0)
    return (
        (1.0 / math.sqrt(2))
        * ((P.sqrt() - Q.sqrt()).pow(2).sum()).sqrt()
    ).item()


def _answer_dist(
    model: GPT2LMHeadModel,
    full_ids: torch.Tensor,
    answer_positions: list,
    device: str,
) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids=full_ids.to(device)).logits   # (1, T, V)
    dists = torch.stack([
        F.softmax(logits[0, pos, :], dim=-1)
        for pos in answer_positions
    ], dim=0)                                                   # (La, V)
    return dists.mean(dim=0).cpu()                              # (V,)


def _compute_importance(
    model:            GPT2LMHeadModel,
    tokenizer:        GPT2TokenizerFast,
    full_ids:         torch.Tensor,    # (1, T) CPU
    q_len:            int,
    answer_positions: list,
    p_orig:           torch.Tensor,    # (V,)
    device:           str,
) -> torch.Tensor:
    T           = full_ids.shape[1]
    importance  = torch.zeros(T)
    special_ids = set(tokenizer.all_special_ids)
    occ_id      = tokenizer.eos_token_id

    for i in range(q_len):
        tok_id = full_ids[0, i].item()
        if tok_id in special_ids:
            importance[i] = 0.0
            continue
        full_ids_occ       = full_ids.clone()
        full_ids_occ[0, i] = occ_id
        p_occ = _answer_dist(model, full_ids_occ, answer_positions, device)
        importance[i] = _hellinger(p_orig, p_occ)

    return importance   # (T,) CPU


def reagent_gpt2(
    question:       str,
    model_name:     str = "gpt2",
    device:         str = "cpu",
    max_new_tokens: int = 30,
    gold_answer:    Optional[str] = None,
    baseline:       str = "zero",
) -> dict:
    """
    Run ReAGent-style occlusion attribution for one TellMeWhy sample.

    Parameters
    ----------
    baseline : 'zero' | 'pad' | 'mean'
        Controls the base_embed returned for metric compatibility.
        Does NOT affect occlusion scores (which always use EOS replacement)
        but DOES affect faithfulness metrics if the caller uses base_embed
        as the replacement token in Soft-NC/NS/log-odds computations.
    """
    t0 = time.time()
    model, tokenizer = get_model_tokenizer(model_name, device)

    # 1. Tokenise Q
    q_enc       = tokenizer(question, return_tensors="pt", add_special_tokens=True)
    q_ids       = q_enc["input_ids"].to(device)
    q_attn_mask = q_enc["attention_mask"].to(device)
    q_len       = q_ids.shape[1]

    # 2. Get answer token ids
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

    full_ids         = full_ids.cpu()
    answer_positions = list(range(q_len, T))
    answer_ids       = full_ids[0, q_len:]   # (La,)

    # 3. Reference answer distribution
    p_orig = _answer_dist(model, full_ids, answer_positions, device)

    # 4. Occlusion importance scores
    attributions = _compute_importance(
        model, tokenizer,
        full_ids, q_len, answer_positions,
        p_orig, device,
    )   # (T,) CPU

    # 5. Embeddings for metric compatibility
    embed_layer = model.transformer.wte
    with torch.no_grad():
        input_embed = embed_layer(full_ids.to(device)).detach().cpu()   # (1, T, D)

    # base_embed built from --baseline, used by xai_metrics_gpt2
    base_embed = _build_base_embed(
        embed_layer, input_embed, baseline, tokenizer, device="cpu"
    ).detach()   # (1, T, D) CPU

    with torch.no_grad():
        logits_full = model(
            inputs_embeds=input_embed.to(device)
        ).logits[0].detach().cpu()   # (T, V)

    tokens = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())

    return {
        "tokens":           tokens,
        "q_len":            q_len,
        "answer_positions": answer_positions,
        "answer_ids":       answer_ids,
        "attributions":     attributions,
        "input_embed":      input_embed,
        "base_embed":       base_embed,
        "logits_full":      logits_full,
        "predicted_answer": tokenizer.decode(
                                answer_ids.tolist(),
                                skip_special_tokens=True),
        "model":            model,
        "tokenizer":        tokenizer,
        "time":             time.time() - t0,
    }