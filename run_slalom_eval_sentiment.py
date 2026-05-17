"""
evaluate_slalom.py

Benchmark SLALOM explanations with the same interface and metrics
as the PACE gradient evaluation scripts (log-odds, comprehensiveness,
sufficiency).

Metrics are computed via xai_metrics to guarantee identical formulas:
  - deletion-based comprehensiveness (topk tokens removed, sequence shortened)
  - deletion-based sufficiency (only topk tokens kept, sequence shortened)
  - topk denominator = full sequence length L  (matches xai_metrics)

Usage:
    python evaluate_slalom.py --model distilbert --dataset sst2
    python evaluate_slalom.py --model bert --dataset imdb --num_samples 500
    python evaluate_slalom.py --model distilbert --dataset sst2 --attr_mode value
"""

import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from slalom_explanations import SLALOMLocalExplanantions
from xai_metrics import calculate_log_odds, calculate_comprehensiveness, calculate_sufficiency
from pace_gradients import get_baseline_embedding

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_NAMES = {
    ("distilbert", "sst2"):   "distilbert-base-uncased-finetuned-sst-2-english",
    ("distilbert", "imdb"):   "textattack/distilbert-base-uncased-imdb",
    ("distilbert", "rotten"): "textattack/distilbert-base-uncased-rotten-tomatoes",
    ("bert",       "sst2"):   "textattack/bert-base-uncased-SST-2",
    ("bert",       "imdb"):   "textattack/bert-base-uncased-imdb",
    ("bert",       "rotten"): "textattack/bert-base-uncased-rotten-tomatoes",
    ("roberta",    "sst2"):   "textattack/roberta-base-SST-2",
    ("roberta",    "imdb"):   "textattack/roberta-base-imdb",
    ("roberta",    "rotten"): "textattack/roberta-base-rotten-tomatoes",
}


# ── SLALOM output format detection (run once, cached) ─────────────────────
_SLALOM_FORMAT = None

def _detect_and_unpack(res):
    """
    Auto-detect SLALOM output format and return (tokens, values, imps).

    Observed formats across SLALOM versions:
      A) dict  with keys "tokens","value","imp"
      B) list  of (token_str, value_vec, imp_vec)        3-tuple per token
      C) list  of (token_str, stacked_array)             shape (num_modes, num_labels)
      D) list  of (token_str, score_vec)                 single mode
    """
    global _SLALOM_FORMAT

    if _SLALOM_FORMAT is None:
        if isinstance(res, dict):
            _SLALOM_FORMAT = "dict"
        elif isinstance(res, (list, tuple)) and len(res) > 0:
            elem = res[0]
            if isinstance(elem, (list, tuple)):
                n = len(elem)
                if n >= 3:
                    _SLALOM_FORMAT = "list_3tuple"
                elif n == 2:
                    v = np.array(elem[1])
                    _SLALOM_FORMAT = "list_2tuple_stacked" if v.ndim == 2 else "list_2tuple_single"
                else:
                    raise ValueError(f"Unexpected tuple length {n}: {elem}")
            else:
                raise ValueError(f"Unexpected element type {type(elem)}: {elem}")
        else:
            raise ValueError(f"Unexpected SLALOM output type {type(res)}: {res}")
        print(f"[SLALOM format detected] {_SLALOM_FORMAT}")

    def _to_1d(x):
        x = np.array(x, dtype=np.float32)
        if x.ndim == 1:
            return x
        elif x.shape[-1] == 1:
            return x[..., 0]
        else:
            return x[..., 1] - x[..., 0]

    if _SLALOM_FORMAT == "dict":
        tokens = res["tokens"]
        values = _to_1d(np.array(res["value"], dtype=np.float32))
        imps   = _to_1d(np.array(res["imp"],   dtype=np.float32))
    elif _SLALOM_FORMAT == "list_3tuple":
        tokens = [r[0] for r in res]
        values = _to_1d(np.stack([np.array(r[1], dtype=np.float32) for r in res]))
        imps   = _to_1d(np.stack([np.array(r[2], dtype=np.float32) for r in res]))
    elif _SLALOM_FORMAT == "list_2tuple_stacked":
        tokens  = [r[0] for r in res]
        stacked = np.stack([np.array(r[1], dtype=np.float32) for r in res])
        values  = _to_1d(stacked[:, 0, :])
        imps    = _to_1d(stacked[:, 1, :]) if stacked.shape[1] > 1 else np.zeros(len(tokens), dtype=np.float32)
    elif _SLALOM_FORMAT == "list_2tuple_single":
        tokens = [r[0] for r in res]
        values = _to_1d(np.stack([np.array(r[1], dtype=np.float32) for r in res]))
        imps   = np.zeros(len(tokens), dtype=np.float32)

    return tokens, values, imps


# ── Forward wrapper matching xai_metrics signature ────────────────────────
def _nn_forward_func(model, input_embed, attention_mask,
                     position_embed=None, type_embed=None,
                     return_all_logits=False):
    """
    Thin wrapper so calculate_* from xai_metrics can drive the SLALOM model.
    SLALOM models are plain AutoModelForSequenceClassification, so we call
    inputs_embeds directly.  position_embed / type_embed are unused here
    (DistilBERT has neither; BERT/RoBERTa use internal positional encodings
    when inputs_embeds is given).
    """
    with torch.no_grad():
        out = model(inputs_embeds=input_embed, attention_mask=attention_mask)
    return out.logits


# ── Single-sample wrapper ──────────────────────────────────────────────────
def slalom_explain_and_eval(
    text, slalom_explainer, model, tokenizer,
    device, base_token_emb, topk=20, attr_mode="lin",
):
    # ── Run SLALOM ──
    t0  = time.perf_counter()
    raw = slalom_explainer.tokenize_and_explain(text)
    t1  = time.perf_counter()

    tokens_out, values, imps = _detect_and_unpack(raw)

    # ── Build attribution vector ──
    if attr_mode == "value":
        attr_np = values
    elif attr_mode == "imp":
        attr_np = imps
    else:  # "lin"
        attr_np = values * np.exp(np.clip(imps, -20, 20))

    # ── Tokenise to get embeddings, matching length expected by xai_metrics ──
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    return_special_tokens_mask=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    L              = input_ids.shape[1]

    embed = model.get_input_embeddings()
    with torch.no_grad():
        input_embed = embed(input_ids)          # (1, L, d)

    # ── Align attribution vector to full length L ──
    attr = torch.zeros(L, dtype=torch.float32, device=device)
    if attr_np.shape[0] == L:
        attr = torch.tensor(attr_np, dtype=torch.float32, device=device)
    else:
        # SLALOM may strip special tokens; remap by position
        special_ids_set = set(tokenizer.all_special_ids)
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist())
                    if tid not in special_ids_set]
        if len(keep_idx) == attr_np.shape[0]:
            attr[keep_idx] = torch.tensor(attr_np, dtype=torch.float32, device=device)
        # else: leave zeros (rare edge case, logged below)

    # ── Compute metrics via xai_metrics (identical to PACE eval) ──
    # position_embed and type_embed are None; _nn_forward_func ignores them.
    log_odd, pred_id = calculate_log_odds(
        _nn_forward_func, model,
        input_embed, None, None,
        attention_mask, base_token_emb,
        attr, topk=topk,
    )
    comp = calculate_comprehensiveness(
        _nn_forward_func, model,
        input_embed, None, None,
        attention_mask, base_token_emb,
        attr, topk=topk,
    )
    suff = calculate_sufficiency(
        _nn_forward_func, model,
        input_embed, None, None,
        attention_mask, base_token_emb,
        attr, topk=topk,
    )

    return {
        "tokens":          tokens_out,
        "value":           values.tolist(),
        "imp":             imps.tolist(),
        "lin":             (values * np.exp(np.clip(imps, -20, 20))).tolist(),
        "predicted_label": pred_id,
        "time":            t1 - t0,
        "log_odd":         log_odd,
        "comp":            comp,
        "suff":            suff,
    }


# ── Benchmark loop ─────────────────────────────────────────────────────────
def run_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device}")
    print(f"Model         : {args.model} / {args.dataset}")
    print(f"SLALOM mode   : {args.attr_mode}")
    print(f"Top-k         : {args.topk}%")
    print(f"Eval baseline : {args.eval_baseline}")

    model_name = MODEL_NAMES[(args.model, args.dataset)]
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # Build base_token_emb — shape (1, d) to match xai_metrics expectation
    embed = model.get_input_embeddings()
    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X   = embed(dummy_ids)   # (1, 1, d)

    base_token_emb = get_baseline_embedding(
        args.eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0:1, :]   # (1, d) — matches xai_metrics assignment base_token_emb shape

    slalom_explainer = SLALOMLocalExplanantions(
        model, tokenizer, modes=["value", "imp"]
    )

    # ── Dataset loading — mirrors Doc 2 exactly ──
    if args.dataset == "imdb":
        dataset = load_dataset("imdb")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))
        data    = random.sample(data, min(args.num_samples, len(data)))
    elif args.dataset == "sst2":
        dataset = load_dataset("glue", "sst2")["test"]
        data    = list(zip(dataset["sentence"], dataset["label"]))
    elif args.dataset == "rotten":
        dataset = load_dataset("rotten_tomatoes")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))

    if len(data) > args.num_samples:
        data = random.sample(data, args.num_samples)
    print(f"Samples       : {len(data)}")

    total_log_odd = total_comp = total_suff = total_time = 0.0
    count = errors = 0

    for row in tqdm(data):
        text = row[0]
        try:
            res = slalom_explain_and_eval(
                text=text,
                slalom_explainer=slalom_explainer,
                model=model,
                tokenizer=tokenizer,
                device=device,
                base_token_emb=base_token_emb,
                topk=args.topk,
                attr_mode=args.attr_mode,
            )
            total_log_odd += res["log_odd"]
            total_comp    += res["comp"]
            total_suff    += res["suff"]
            total_time    += res["time"]
            count         += 1

            if count % args.print_step == 0:
                print(f"\n[{count}/{len(data)}]"
                      f"  log-odds={total_log_odd/count:.4f}"
                      f"  comp={total_comp/count:.4f}"
                      f"  suff={total_suff/count:.4f}"
                      f"  time={total_time/count:.4f}s")

        except Exception as e:
            errors += 1
            if errors <= 5:
                import traceback; traceback.print_exc()

    if count > 0:
        print(f"\n{'─'*52}")
        print(f"SLALOM ({args.attr_mode})  |  {args.model} / {args.dataset}")
        print(f"  Log-odds         : {total_log_odd/count:.6f}")
        print(f"  Comprehensiveness: {total_comp/count:.6f}")
        print(f"  Sufficiency      : {total_suff/count:.6f}")
        print(f"  Avg time/sample  : {total_time/count:.4f}s")
        print(f"  Evaluated        : {count}  |  Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        choices=["distilbert", "bert", "roberta"],
                        default="distilbert")
    parser.add_argument("--dataset",      choices=["sst2", "imdb", "rotten"],
                        default="sst2")
    parser.add_argument("--num_samples",  type=int, default=1000)
    parser.add_argument("--topk",         type=int, default=20)
    parser.add_argument("--attr_mode",    choices=["value", "imp", "lin"],
                        default="lin")
    parser.add_argument("--print_step",   type=int, default=100)
    parser.add_argument("--eval-baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Baseline embedding used to replace tokens in faithfulness metrics")
    args = parser.parse_args()
    run_benchmark(args)