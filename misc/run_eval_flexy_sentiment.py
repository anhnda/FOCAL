"""
run_eval_flexi_sentiment.py
===========================
Evaluation script for Flexible Instance-Specific Rationalization on sentiment
classification tasks (SST-2, IMDB, Rotten Tomatoes).

Uses PACE gradient attributions as the pool of feature scoring methods and
applies flexi.py to select — at instance level — the best combination of:
  (1) feature scoring method  [FEAT]
  (2) rationale length        [LEN]
  (3) rationale type          [TYPE]

Reports NormSuff, NormComp, and runtime, comparing:
  • FIXED   – single PACE method, fixed length N, fixed type TOPK
  • I-L FEAT – instance-level feature selection, fixed len + type
  • I-L LEN  – instance-level length selection, fixed method + type
  • I-L TYPE – instance-level type selection, fixed method + len
  • I-L ALL  – instance-level FEAT + LEN + TYPE

Usage
-----
python run_eval_flexi_sentiment.py \
    --model distilbert \
    --dataset sst2 \
    --steps 100 \
    --baseline mask \
    --eval-baseline mask \
    --delta jsd \
    --rationale-ratio 0.20 \
    --skip-rate 0.02

Requirements: transformers, datasets, torch, tqdm
              (pace_gradients.py, flexi.py, xai_metrics.py in same dir)
"""

import argparse
import random
import time
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pace_gradients import get_baseline_embedding, pace_gradient_classification
from flexi import (
    compute_norm_comp_suff,
    select_all,
    select_feature_scoring,
    select_rationale_length,
    select_rationale_type,
)

# ── reproducibility ─────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# ── helpers ──────────────────────────────────────────────────────────────────

MODEL_MAP = {
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

DATASET_RATIO = {          # fixed pre-defined rationale ratio N (Table 1)
    "sst2":   0.20,
    "imdb":   0.20,
    "rotten": 0.20,
}


def load_data(dataset_name: str, n_samples: int = 2000):
    if dataset_name == "imdb":
        ds = load_dataset("imdb")["test"]
        data = list(zip(ds["text"], ds["label"]))
        data = random.sample(data, min(n_samples, len(data)))
    elif dataset_name == "sst2":
        ds = load_dataset("glue", "sst2")["validation"]
        data = list(zip(ds["sentence"], ds["label"]))
    elif dataset_name == "rotten":
        ds = load_dataset("rotten_tomatoes")["test"]
        data = list(zip(ds["text"], ds["label"]))
        data = random.sample(data, min(n_samples, len(data)))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data


def accumulate(acc: dict, key: str, val: float):
    acc[key].append(val)


def mean(acc: dict, key: str) -> float:
    vals = acc[key]
    return float(np.mean(vals)) if vals else 0.0


def print_results(tag: str, acc: dict, count: int):
    print(
        f"[{tag:>12s}]  "
        f"NormSuff: {mean(acc, 'norm_suff'):.4f}  "
        f"NormComp: {mean(acc, 'norm_comp'):.4f}  "
        f"Time: {mean(acc, 'time'):.4f}s  "
        f"(n={count})"
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Flexible Instance-Specific Rationalization — Sentiment Eval"
    )
    parser.add_argument("--model",   type=str, default="distilbert",
                        choices=["distilbert", "bert", "roberta"])
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=["sst2", "imdb", "rotten"])
    parser.add_argument("--steps",   type=int, default=100,
                        help="PACE integration steps")
    parser.add_argument("--baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Baseline for PACE integration path")
    parser.add_argument("--eval-baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Token replacement baseline for faithfulness metrics")
    parser.add_argument("--delta", type=str, default="jsd",
                        choices=["jsd", "kl", "perplexity", "classdiff"],
                        help="Divergence function ∆ for rationale selection")
    parser.add_argument("--rationale-ratio", type=float, default=None,
                        help="Override fixed rationale ratio N (default: dataset default)")
    parser.add_argument("--skip-rate", type=float, default=0.02,
                        help="Token-skip fraction for length search (0=no skip)")
    parser.add_argument("--print-step", type=int, default=50,
                        help="Print running averages every N instances")
    parser.add_argument("--max-samples", type=int, default=2000)
    args = parser.parse_args()

    model_name = MODEL_MAP[(args.model, args.dataset)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ratio_N = args.rationale_ratio or DATASET_RATIO.get(args.dataset, 0.20)

    print("=" * 70)
    print(f"  Flexible Instance-Specific Rationalization — Sentiment")
    print("=" * 70)
    print(f"  Model         : {model_name}")
    print(f"  Dataset       : {args.dataset}")
    print(f"  Device        : {device}")
    print(f"  PACE baseline : {args.baseline}  steps={args.steps}")
    print(f"  Eval baseline : {args.eval_baseline}")
    print(f"  Delta fn (∆)  : {args.delta}")
    print(f"  Ratio N       : {ratio_N:.0%}")
    print(f"  Skip rate     : {args.skip_rate:.0%}")
    print("=" * 70)

    # ── build eval baseline token embedding ─────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    eval_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    eval_model.eval()
    embed = eval_model.get_input_embeddings()

    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X   = embed(dummy_ids)   # (1, 1, d)

    eval_base_token_emb = get_baseline_embedding(
        args.eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0:1, :]   # (1, d)

    # ── load data ────────────────────────────────────────────────────────────
    data = load_data(args.dataset, args.max_samples)
    print(f"  Loaded {len(data)} instances.\n")

    # ── accumulators for each evaluation mode ────────────────────────────────
    modes = ["fixed", "il_feat", "il_len", "il_type", "il_all"]
    accs  = {m: defaultdict(list) for m in modes}
    count = 0

    for text, _label in tqdm(data, desc="Evaluating"):
        # ── PACE attributions (single method = PACE gradient) ─────────────
        t0 = time.perf_counter()
        res = pace_gradient_classification(
            sentence=text,
            a=0.0, b=1.0,
            steps=args.steps,
            model_name=model_name,
            show_special_tokens=True,   # keep special tokens for masking
            baseline=args.baseline,
        )
        pace_time = time.perf_counter() - t0

        fwd_fn         = res["nn_forward_func"]
        model_obj      = res["model"]
        input_embed    = res["input_embed"]       # (1, L, d)
        position_embed = res["position_embed"]
        type_embed     = res["type_embed"]
        attention_mask = res["attention_mask"]
        attr_full      = res["attr_full"]         # (L,) on device

        L = input_embed.shape[1]
        N = max(1, int(L * ratio_N))              # fixed rationale length in tokens

        # Pool of attribution methods — here we have one (PACE).
        # Extend this dict with additional attribution outputs (IG, attention, etc.)
        # to enable the multi-method selection demonstrated in the paper.
        attr_dict = {
            "pace": attr_full,
        }

        # Helper: compute metrics for a given set of selected indices
        def get_metrics(indices):
            return compute_norm_comp_suff(
                fwd_fn, model_obj,
                input_embed, position_embed, type_embed,
                attention_mask, eval_base_token_emb,
                indices,
            )

        # ── FIXED: PACE, fixed length N, fixed type TOPK ─────────────────
        t0 = time.perf_counter()
        from flexi import _topk_indices
        fixed_indices = _topk_indices(attr_full, N)
        fixed_m = get_metrics(fixed_indices)
        fixed_time = pace_time + (time.perf_counter() - t0)
        accumulate(accs["fixed"], "norm_suff", fixed_m["norm_suff"])
        accumulate(accs["fixed"], "norm_comp", fixed_m["norm_comp"])
        accumulate(accs["fixed"], "time", fixed_time)

        # ── I-L FEAT: best attribution method, fixed length N, fixed TOPK ─
        t0 = time.perf_counter()
        r_feat = select_feature_scoring(
            fwd_fn, model_obj,
            input_embed, position_embed, type_embed,
            attention_mask, eval_base_token_emb,
            attr_dict, k=N, rationale_type="topk",
            delta_name=args.delta,
        )
        m_feat = get_metrics(r_feat["best_indices"])
        il_feat_time = pace_time + (time.perf_counter() - t0)
        accumulate(accs["il_feat"], "norm_suff", m_feat["norm_suff"])
        accumulate(accs["il_feat"], "norm_comp", m_feat["norm_comp"])
        accumulate(accs["il_feat"], "time", il_feat_time)

        # ── I-L LEN: fixed PACE, instance-level length, fixed TOPK ───────
        t0 = time.perf_counter()
        r_len = select_rationale_length(
            fwd_fn, model_obj,
            input_embed, position_embed, type_embed,
            attention_mask, eval_base_token_emb,
            attr_full, N=N, rationale_type="topk",
            delta_name=args.delta, skip_rate=args.skip_rate,
        )
        m_len = get_metrics(r_len["best_indices"])
        il_len_time = pace_time + (time.perf_counter() - t0)
        accumulate(accs["il_len"], "norm_suff", m_len["norm_suff"])
        accumulate(accs["il_len"], "norm_comp", m_len["norm_comp"])
        accumulate(accs["il_len"], "best_k",    r_len["best_k"])
        accumulate(accs["il_len"], "time",      il_len_time)

        # ── I-L TYPE: fixed PACE, fixed length N, instance-level type ─────
        t0 = time.perf_counter()
        r_type = select_rationale_type(
            fwd_fn, model_obj,
            input_embed, position_embed, type_embed,
            attention_mask, eval_base_token_emb,
            attr_full, k=N,
            delta_name=args.delta,
        )
        m_type = get_metrics(r_type["best_indices"])
        il_type_time = pace_time + (time.perf_counter() - t0)
        accumulate(accs["il_type"], "norm_suff", m_type["norm_suff"])
        accumulate(accs["il_type"], "norm_comp", m_type["norm_comp"])
        accumulate(accs["il_type"], "time",      il_type_time)

        # ── I-L ALL: instance-level FEAT + LEN + TYPE ─────────────────────
        t0 = time.perf_counter()
        r_all = select_all(
            fwd_fn, model_obj,
            input_embed, position_embed, type_embed,
            attention_mask, eval_base_token_emb,
            attr_dict, N=N,
            delta_name=args.delta, skip_rate=args.skip_rate,
        )
        m_all = get_metrics(r_all["best_indices"])
        il_all_time = pace_time + (time.perf_counter() - t0)
        accumulate(accs["il_all"], "norm_suff", m_all["norm_suff"])
        accumulate(accs["il_all"], "norm_comp", m_all["norm_comp"])
        accumulate(accs["il_all"], "best_k",    r_all["best_k"])
        accumulate(accs["il_all"], "time",      il_all_time)

        count += 1

        if count % args.print_step == 0:
            print(f"\n── Running averages @ {count} instances ──")
            print_results("FIXED",   accs["fixed"],   count)
            print_results("IL-FEAT", accs["il_feat"], count)
            print_results("IL-LEN",  accs["il_len"],  count)
            print_results("IL-TYPE", accs["il_type"], count)
            print_results("IL-ALL",  accs["il_all"],  count)
            avg_k_len = mean(accs["il_len"], "best_k")
            avg_k_all = mean(accs["il_all"], "best_k")
            print(f"  Avg selected k: IL-LEN={avg_k_len:.1f}  IL-ALL={avg_k_all:.1f}  "
                  f"(fixed N={N})")

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  FINAL RESULTS  ({count} instances)")
    print("=" * 70)

    header = f"{'Mode':>12s}  {'NormSuff':>10s}  {'NormComp':>10s}  {'Time(s)':>9s}"
    print(header)
    print("-" * len(header))

    for tag, mode in [
        ("FIXED",   "fixed"),
        ("IL-FEAT", "il_feat"),
        ("IL-LEN",  "il_len"),
        ("IL-TYPE", "il_type"),
        ("IL-ALL",  "il_all"),
    ]:
        ns = mean(accs[mode], "norm_suff")
        nc = mean(accs[mode], "norm_comp")
        t  = mean(accs[mode], "time")
        print(f"  {tag:>10s}  {ns:>10.4f}  {nc:>10.4f}  {t:>9.4f}")

    print("=" * 70)
    print(f"  Avg instance-level k (IL-LEN):  {mean(accs['il_len'], 'best_k'):.1f}  "
          f"(fixed N={N})")
    print(f"  Avg instance-level k (IL-ALL):  {mean(accs['il_all'], 'best_k'):.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()