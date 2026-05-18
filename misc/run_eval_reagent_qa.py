"""
Benchmark Script for ReAGent Attribution on Question Answering

This script evaluates the ReAGent attribution method (adapted for encoder models)
on the SQuADv2 / SQuAD dataset, computing XAI metrics: log-odds, comprehensiveness,
and sufficiency.

Mirrors run_eval_pg_qa.py exactly — only the attribution backend changes from
pace_gradient_qa() to reagent_qa() (defined below).

Usage:
    python run_eval_reagent_qa.py --model_name deepset/bert-base-cased-squad2 --num_samples 1000
    python run_eval_reagent_qa.py --model_name cleandata/bert-finetuned-squad --num_samples 1000 --top_k 5
    python run_eval_reagent_qa.py --demo  # Run demo with a few examples

─────────────────────────────────────────────────────────────────────────────
ReAGent-QA algorithm (per token position i)
─────────────────────────────────────────────────────────────────────────────
1. Obtain (start_logits, end_logits) = QA_model(x)
   P_start_orig = softmax(start_logits)
   P_end_orig   = softmax(end_logits)

2. For each non-special token position i:
   a. Mask position i with [MASK], query MLM oracle → top-k replacements r_1…r_k.
   b. For each r_j: replace x[i] → x̃, compute P_start_j, P_end_j.
   c. importance_start[i] = mean_j  Hellinger(P_start_orig, P_start_j)
      importance_end[i]   = mean_j  Hellinger(P_end_orig,   P_end_j)

3. Special tokens get importance = 0.
"""

import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import traceback
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
)

from xai_metrics import (
    calculate_log_odds_qa,
    calculate_comprehensiveness_qa,
    calculate_sufficiency_qa,
)

# ── Reproducibility ────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ── Model caches ───────────────────────────────────────────────────────────
_qa_cache:  dict = {}   # model_name → {"model": ..., "tokenizer": ...}
_mlm_cache: dict = {}   # mlm_name   → (tokenizer, model)


# ── Loaders ────────────────────────────────────────────────────────────────

def _load_qa(model_name: str, device: str):
    if model_name not in _qa_cache:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        mdl = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
        mdl.eval()
        _qa_cache[model_name] = {"model": mdl, "tokenizer": tok}
    entry = _qa_cache[model_name]
    return entry["tokenizer"], entry["model"]


def _load_mlm(mlm_name: str, device: str):
    if mlm_name not in _mlm_cache:
        tok = AutoTokenizer.from_pretrained(mlm_name, use_fast=True)
        mdl = AutoModelForMaskedLM.from_pretrained(mlm_name).to(device)
        mdl.eval()
        _mlm_cache[mlm_name] = (tok, mdl)
    return _mlm_cache[mlm_name]


# ── Helpers ────────────────────────────────────────────────────────────────

def _hellinger(p: torch.Tensor, q: torch.Tensor) -> float:
    """Hellinger distance H(p,q) = (1/√2)‖√p − √q‖₂  ∈ [0, 1]."""
    p = p.float().clamp(min=0.0)
    q = q.float().clamp(min=0.0)
    return (0.5 * ((p.sqrt() - q.sqrt()) ** 2).sum()).sqrt().item()


def _get_qa_dists(qa_model, input_ids: torch.Tensor,
                  attention_mask: torch.Tensor,
                  token_type_ids: torch.Tensor | None,
                  device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (softmax start dist, softmax end dist), each shape (L,)."""
    kwargs = dict(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
    )
    if token_type_ids is not None:
        kwargs["token_type_ids"] = token_type_ids.to(device)
    with torch.no_grad():
        out = qa_model(**kwargs)
    return F.softmax(out.start_logits[0], dim=-1), F.softmax(out.end_logits[0], dim=-1)


def _get_top_k_replacements(
    qa_tokenizer,
    mlm_tokenizer,
    mlm_model,
    input_ids: torch.Tensor,   # (1, L)
    position: int,
    qa_tokens: list[str],
    top_k: int,
    device: str,
) -> list[int]:
    """
    Query the MLM oracle for top-k replacement token ids (in the QA tokenizer vocab)
    for position `position`.  Mirrors _get_top_k_replacements() from
    reagent_classification.py exactly.
    """
    qa_special  = set(qa_tokenizer.all_special_tokens)
    mlm_special = set(mlm_tokenizer.all_special_tokens)

    masked_tokens = []
    for i, tok in enumerate(qa_tokens):
        if tok in qa_special:
            continue
        masked_tokens.append(mlm_tokenizer.mask_token if i == position else tok)

    masked_text = mlm_tokenizer.convert_tokens_to_string(masked_tokens)
    mlm_enc = mlm_tokenizer(
        masked_text, return_tensors="pt", truncation=True
    ).to(device)

    mask_token_id = mlm_tokenizer.mask_token_id
    mask_positions = (mlm_enc["input_ids"][0] == mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return [input_ids[0, position].item()]

    mask_pos = mask_positions[0].item()
    with torch.no_grad():
        mlm_logits = mlm_model(**mlm_enc).logits[0]

    top_k_ids_mlm = mlm_logits[mask_pos].topk(top_k * 3).indices

    replacement_ids = []
    for mlm_id in top_k_ids_mlm.tolist():
        token_str = mlm_tokenizer.decode([mlm_id]).strip()
        if not token_str or token_str in mlm_special:
            continue
        qa_ids = qa_tokenizer.encode(token_str, add_special_tokens=False)
        if len(qa_ids) == 1:
            replacement_ids.append(qa_ids[0])
        if len(replacement_ids) >= top_k:
            break

    return replacement_ids[:top_k] if replacement_ids else [input_ids[0, position].item()]


def _compute_importance_scores(
    qa_tokenizer,
    qa_model,
    mlm_tokenizer,
    mlm_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor | None,
    qa_tokens: list[str],
    p_start_orig: torch.Tensor,
    p_end_orig: torch.Tensor,
    top_k: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-token Hellinger importance for start and end position distributions.
    Returns (importance_start, importance_end), each shape (L,).
    """
    L = input_ids.shape[1]
    special_ids = set(qa_tokenizer.all_special_ids)
    imp_start = np.zeros(L, dtype=np.float32)
    imp_end   = np.zeros(L, dtype=np.float32)

    for i in range(L):
        tok_id = input_ids[0, i].item()
        if tok_id in special_ids:
            continue

        replacement_ids = _get_top_k_replacements(
            qa_tokenizer, mlm_tokenizer, mlm_model,
            input_ids, i, qa_tokens, top_k, device,
        )

        scores_start, scores_end = [], []
        for rep_id in replacement_ids:
            x_rep = input_ids.clone()
            x_rep[0, i] = rep_id
            p_s, p_e = _get_qa_dists(qa_model, x_rep, attention_mask,
                                     token_type_ids, device)
            scores_start.append(_hellinger(p_start_orig, p_s))
            scores_end.append(_hellinger(p_end_orig, p_e))

        imp_start[i] = float(np.mean(scores_start)) if scores_start else 0.0
        imp_end[i]   = float(np.mean(scores_end))   if scores_end   else 0.0

    return imp_start, imp_end


# ── Helper dispatch (mirrors reagent_classification._get_helper_fns) ──────

def _get_helper_fns(model_name: str):
    if "distilbert" in model_name:
        from distilbert_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "roberta" in model_name:
        from roberta_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "bert" in model_name:
        from bert_helper import get_inputs, get_base_token_emb, nn_forward_func
    else:
        raise NotImplementedError(f"No helper module for model: {model_name}")
    return get_inputs, get_base_token_emb, nn_forward_func


# ── Public API ─────────────────────────────────────────────────────────────

def reagent_qa(
    question: str,
    context: str,
    model_name: str,
    top_k: int = 3,
    mlm_name: str | None = None,
    show_special_tokens: bool = False,
    device: str | None = None,
) -> dict:
    """
    Run ReAGent-style feature attribution on a BERT-based QA model.

    Args:
        question:            The question string.
        context:             The passage/context string.
        model_name:          HuggingFace QA model name/path.
        top_k:               MLM replacement candidates per token (paper default: 3).
        mlm_name:            MLM oracle; defaults to the QA checkpoint itself.
        show_special_tokens: Whether to include [CLS]/[SEP]/[PAD] in output.
        device:              Torch device; auto-detected if None.

    Returns dict with keys:
        tokens               — list[str]
        attributions_start   — list[float]   Hellinger importance (start dist)
        attributions_end     — list[float]   Hellinger importance (end dist)
        predicted_answer     — str
        start_idx, end_idx   — int
        start_logit, end_logit — float
        start_prob, end_prob — float
        input_embed          — torch.Tensor  (1, L, D)  for xai_metrics
        attention_mask       — torch.Tensor  (1, L)
        special_tokens_mask  — torch.Tensor  (1, L)
        token_type_ids       — torch.Tensor | None
        base_token_emb       — torch.Tensor
        model                — nn.Module
        time                 — float
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()

    if mlm_name is None:
        mlm_name = model_name

    qa_tokenizer, qa_model   = _load_qa(model_name, device)
    mlm_tokenizer, mlm_model = _load_mlm(mlm_name, device)

    # ── Tokenize (question + context) ──
    enc = qa_tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
    )
    input_ids           = enc["input_ids"]                # (1, L)
    attention_mask      = enc["attention_mask"]           # (1, L)
    special_tokens_mask = enc["special_tokens_mask"]      # (1, L)
    token_type_ids      = enc.get("token_type_ids", None) # (1, L) or None

    qa_tokens = qa_tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # ── Original QA distributions ──
    p_start_orig, p_end_orig = _get_qa_dists(
        qa_model, input_ids, attention_mask, token_type_ids, device
    )

    # Predicted span
    start_idx   = int(p_start_orig.argmax().item())
    end_idx     = int(p_end_orig.argmax().item())
    start_logit = p_start_orig[start_idx].item()
    end_logit   = p_end_orig[end_idx].item()
    # Keep as scalar Tensors for xai_metrics (torch.log requires Tensor input)
    start_prob_tensor = p_start_orig[start_idx].detach()
    end_prob_tensor   = p_end_orig[end_idx].detach()

    predicted_answer = qa_tokenizer.decode(
        input_ids[0, start_idx: end_idx + 1], skip_special_tokens=True
    )

    # ── Per-token Hellinger importance ──
    imp_start, imp_end = _compute_importance_scores(
        qa_tokenizer, qa_model,
        mlm_tokenizer, mlm_model,
        input_ids, attention_mask, token_type_ids,
        qa_tokens, p_start_orig, p_end_orig,
        top_k, device,
    )

    # ── Embedding tensors needed by xai_metrics ──
    embed = qa_model.get_input_embeddings()
    with torch.no_grad():
        input_embed = embed(input_ids.to(device))          # (1, L, D)

    # base_token_emb: embedding of [PAD] token (index 0), shape (1, 1, D)
    pad_id = qa_tokenizer.pad_token_id or 0
    pad_tensor = torch.tensor([[pad_id]], device=device)
    with torch.no_grad():
        base_token_emb = embed(pad_tensor)                 # (1, 1, D)

    t1 = time.perf_counter()

    # ── Build token-level output (optionally drop specials) ──
    special_ids = set(qa_tokenizer.all_special_ids)
    out_tokens, out_start, out_end = [], [], []
    for tok_str, tok_id, s, e in zip(
        qa_tokens, input_ids[0].tolist(),
        imp_start.tolist(), imp_end.tolist()
    ):
        if not show_special_tokens and tok_id in special_ids:
            continue
        out_tokens.append(tok_str)
        out_start.append(s)
        out_end.append(e)

    return {
        "tokens":              out_tokens,
        "attributions_start":  torch.tensor(imp_start, dtype=torch.float32, device=device),
        "attributions_end":    torch.tensor(imp_end,   dtype=torch.float32, device=device),
        "predicted_answer":    predicted_answer,
        "start_idx":           start_idx,
        "end_idx":             end_idx,
        "start_logit":         start_logit,
        "end_logit":           end_logit,
        "start_prob":          start_prob_tensor,   # Tensor — required by xai_metrics
        "end_prob":            end_prob_tensor,     # Tensor — required by xai_metrics
        # tensors for xai_metrics
        "model":               qa_model,
        "input_embed":         input_embed,
        "attention_mask":      attention_mask.to(device),
        "special_tokens_mask": special_tokens_mask.to(device),
        "token_type_ids":      token_type_ids.to(device) if token_type_ids is not None else None,
        "base_token_emb":      base_token_emb,
        "time":                t1 - t0,
    }


# ── Single-example runner (mirrors run_eval_pg_qa.run_single_example) ─────

def run_single_example(
    question: str,
    context: str,
    model_name: str,
    top_k: int,
    device: str,
    topk: int = 20,
) -> dict:
    """
    Run ReAGent-QA attribution on one example and compute faithfulness metrics.

    Args:
        question:   The question string.
        context:    The context/passage.
        model_name: HuggingFace QA model name.
        top_k:      MLM replacement candidates per token.
        device:     Computation device.
        topk:       Percentage of top tokens for metrics.

    Returns:
        Dictionary with attribution results and all six faithfulness metrics.
    """
    res = reagent_qa(
        question=question,
        context=context,
        model_name=model_name,
        top_k=top_k,
        device=device,
        show_special_tokens=True,
    )

    model               = res["model"]
    input_embed         = res["input_embed"]
    attention_mask      = res["attention_mask"]
    special_tokens_mask = res["special_tokens_mask"]
    token_type_ids      = res["token_type_ids"]
    base_token_emb      = res["base_token_emb"]
    attr_start          = res["attributions_start"]
    attr_end            = res["attributions_end"]
    start_idx           = res["start_idx"]
    end_idx             = res["end_idx"]
    prob_start_orig     = res["start_prob"]
    prob_end_orig       = res["end_prob"]

    # position_embed and type_embed: ReAGent works in embedding space directly,
    # so these are None (xai_metrics_qa handles None gracefully).
    position_embed = None
    type_embed     = None

    log_odd_start, log_odd_end = calculate_log_odds_qa(
        model, input_embed, attention_mask, special_tokens_mask, token_type_ids,
        base_token_emb, attr_start, attr_end, start_idx, end_idx,
        prob_start_orig, prob_end_orig,
        topk=topk,
    )
    comp_start, comp_end = calculate_comprehensiveness_qa(
        model, input_embed, attention_mask, special_tokens_mask, token_type_ids,
        base_token_emb, attr_start, attr_end, start_idx, end_idx,
        prob_start_orig, prob_end_orig,
        topk=topk,
    )
    suff_start, suff_end = calculate_sufficiency_qa(
        model, input_embed, attention_mask, special_tokens_mask, token_type_ids,
        base_token_emb, attr_start, attr_end, start_idx, end_idx,
        prob_start_orig, prob_end_orig,
        topk=topk,
    )

    return {
        "tokens":            res["tokens"],
        "attributions_start": attr_start,
        "attributions_end":   attr_end,
        "predicted_answer":   res["predicted_answer"],
        "start_idx":          start_idx,
        "end_idx":            end_idx,
        "start_logit":        res["start_logit"],
        "end_logit":          res["end_logit"],
        "time":               res["time"],
        # Metrics for start position
        "log_odd_start": log_odd_start,
        "comp_start":    comp_start,
        "suff_start":    suff_start,
        # Metrics for end position
        "log_odd_end": log_odd_end,
        "comp_end":    comp_end,
        "suff_end":    suff_end,
    }


# ── Benchmark loop (mirrors run_eval_pg_qa.run_benchmark) ─────────────────

def run_benchmark(args):
    """Run the full benchmark on SQuAD / SQuADv2 dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device      : {device}")
    print(f"Model             : {args.model_name}")
    print(f"MLM oracle        : {args.model_name}  (top_k={args.top_k})")
    print(f"Number of samples : {args.num_samples}")
    print(f"Top-k pct metrics : {args.topk}%")

    print(f"\nLoading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="validation")
    data = list(zip(
        dataset["question"],
        dataset["context"],
        dataset["answers"],
        dataset["id"],
    ))

    # Filter short examples (same as run_eval_pg_qa)
    upd_data = [
        row for row in data
        if len((row[0] + row[1]).split()) < 80
    ]
    print(f"Filtered samples  : {len(upd_data)}")

    answerable_data = [
        {"context": row[1], "question": row[0], "answers": row[2]}
        for row in upd_data
    ]

    if len(answerable_data) > args.num_samples:
        sampled_data = random.sample(answerable_data, args.num_samples)
    else:
        sampled_data = answerable_data
        print(f"Warning: Only {len(answerable_data)} samples available")

    print(f"Evaluation samples: {len(sampled_data)}")

    print("\nLoading model...")
    _load_qa(args.model_name, device)
    _load_mlm(args.model_name, device)
    print("Models loaded successfully")

    print("\nStarting ReAGent-QA attribution benchmark...")

    total_log_odds_start = total_log_odds_end = 0.0
    total_comp_start     = total_comp_end     = 0.0
    total_suff_start     = total_suff_end     = 0.0
    total_time = 0.0
    count = errors = 0

    for idx, example in enumerate(tqdm(sampled_data)):
        try:
            res = run_single_example(
                question=example["question"],
                context=example["context"],
                model_name=args.model_name,
                top_k=args.top_k,
                device=device,
                topk=args.topk,
            )

            total_log_odds_start += res["log_odd_start"]
            total_log_odds_end   += res["log_odd_end"]
            total_comp_start     += res["comp_start"]
            total_comp_end       += res["comp_end"]
            total_suff_start     += res["suff_start"]
            total_suff_end       += res["suff_end"]
            total_time           += res["time"]
            count += 1

            if count % args.print_step == 0:
                print(
                    f"\n[{count}/{len(sampled_data)}] Running averages:"
                    f"\n  Log-odds (start): {total_log_odds_start / count:.4f}"
                    f"\n  Log-odds (end):   {total_log_odds_end   / count:.4f}"
                    f"\n  Comp (start):     {total_comp_start     / count:.4f}"
                    f"\n  Comp (end):       {total_comp_end       / count:.4f}"
                    f"\n  Suff (start):     {total_suff_start     / count:.4f}"
                    f"\n  Suff (end):       {total_suff_end       / count:.4f}"
                    f"\n  Avg time:         {total_time / count:.4f}s"
                )

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\nError processing sample {idx}: {str(e)[:100]}")
                traceback.print_exc()
            continue

    if count > 0:
        print(f"\n{'─'*56}")
        print(f"ReAGent-QA (top_k={args.top_k})  |  {args.model_name}")
        print(f"  Log-odds (start):          {total_log_odds_start / count:.6f}")
        print(f"  Comprehensiveness (start): {total_comp_start     / count:.6f}")
        print(f"  Sufficiency (start):       {total_suff_start     / count:.6f}")
        print(f"  Log-odds (end):            {total_log_odds_end   / count:.6f}")
        print(f"  Comprehensiveness (end):   {total_comp_end       / count:.6f}")
        print(f"  Sufficiency (end):         {total_suff_end       / count:.6f}")
        print(f"  Avg time/sample:           {total_time / count:.4f}s")
        print(f"  Total time:                {total_time:.2f}s")
        print(f"  Evaluated: {count}  |  Errors: {errors}")
        print(f"{'─'*56}")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark ReAGent Attribution for Question Answering"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepset/bert-base-cased-squad2",
        help="QA model name (default: deepset/bert-base-cased-squad2)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad",
        help="Dataset name from HuggingFace datasets (default: squad)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="MLM replacement candidates per token (paper default: 3)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Percentage of top tokens for metrics calculation",
    )
    parser.add_argument(
        "--print_step",
        type=int,
        default=100,
        help="Print metrics every N samples",
    )
    args = parser.parse_args()
    run_benchmark(args)