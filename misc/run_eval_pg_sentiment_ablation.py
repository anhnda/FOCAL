"""
Eval runner for PACE-gradient ablation methods on sentiment classification.

Methods
-------
  l1     — pace_gradient_l12  with norm='l1'  (L1-norm gate, delta-weighted)
  l2     — pace_gradient_l12  with norm='l2'  (L2-norm gate, delta-weighted)
  scalar — pace_gradient_scalar               (scalar gate, NO delta weight)

Usage examples
--------------
  python run_eval_pg_ablation.py --method l2 --model distilbert --dataset sst2
  python run_eval_pg_ablation.py --method l1 --model bert        --dataset imdb --steps 50
  python run_eval_pg_ablation.py --method scalar --model roberta --dataset rotten \
         --baseline zero --eval-baseline pad
"""

import time
import random
import argparse

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from xai_metrics import (
    calculate_log_odds,
    calculate_comprehensiveness,
    calculate_sufficiency,
)
from pace_gradients import get_baseline_embedding
from pace_gradient_ablations import pace_gradient_l12, pace_gradient_scalar

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# ── Model name lookup ─────────────────────────────────────────────────────────
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


def load_data(dataset_name: str):
    if dataset_name == "imdb":
        dataset = load_dataset("imdb")["test"]
        data = list(zip(dataset["text"], dataset["label"]))
        data = random.sample(data, 2000)
    elif dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2")["test"]
        data = list(zip(dataset["sentence"], dataset["label"]))
    elif dataset_name == "rotten":
        dataset = load_dataset("rotten_tomatoes")["test"]
        data = list(zip(dataset["text"], dataset["label"]))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data


def make_attribution_fn(method: str, model_name: str, steps: int,
                        baseline: str, device: str):
    """
    Return a callable (text) → result-dict for the chosen ablation method.
    """
    if method == "l2":
        def fn(text):
            return pace_gradient_l12(
                sentence=text, steps=steps, model_name=model_name,
                show_special_tokens=False, baseline=baseline,
                device=device, norm="l2",
            )
    elif method == "l1":
        def fn(text):
            return pace_gradient_l12(
                sentence=text, steps=steps, model_name=model_name,
                show_special_tokens=False, baseline=baseline,
                device=device, norm="l1",
            )
    elif method == "scalar":
        def fn(text):
            return pace_gradient_scalar(
                sentence=text, steps=steps, model_name=model_name,
                show_special_tokens=False, baseline=baseline,
                device=device,
            )
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: l1, l2, scalar")
    return fn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PACE gradient ablation methods on sentiment classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["l1", "l2", "scalar"],
        help=(
            "Attribution method to evaluate:\n"
            "  l1     — L1-norm of embedding gradient, delta-weighted\n"
            "  l2     — L2-norm of embedding gradient, delta-weighted\n"
            "  scalar — scalar gate only, NO delta weighting"
        ),
    )
    parser.add_argument("--model",   type=str, default="distilbert",
                        choices=["distilbert", "bert", "roberta"])
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=["sst2", "imdb", "rotten"])
    parser.add_argument("--steps",   type=int, default=100)
    parser.add_argument(
        "--baseline", type=str, default="mask",
        choices=["mask", "pad", "zero", "mean", "random"],
        help="Baseline embedding for the integration path",
    )
    parser.add_argument(
        "--eval-baseline", type=str, default="mask",
        choices=["mask", "pad", "zero", "mean", "random"],
        help="Baseline embedding used to ablate tokens in faithfulness metrics",
    )
    parser.add_argument("--topk",        type=int, default=20,
                        help="Top-k tokens to ablate in faithfulness metrics")
    parser.add_argument("--print-every", type=int, default=100,
                        help="Print running averages every N samples")
    args = parser.parse_args()

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    model_name   = MODEL_NAMES[(args.model, args.dataset)]
    method_label = {
        "l1":     "L1-norm gate  (delta-weighted)",
        "l2":     "L2-norm gate  (delta-weighted)",
        "scalar": "Scalar gate   (no delta weight)",
    }[args.method]

    print("=" * 65)
    print(f"  PACE Gradient Ablation Evaluation")
    print("=" * 65)
    print(f"  Method        : {args.method}  [{method_label}]")
    print(f"  Model         : {model_name}")
    print(f"  Dataset       : {args.dataset}")
    print(f"  PG baseline   : {args.baseline}")
    print(f"  Eval baseline : {args.eval_baseline}")
    print(f"  Steps         : {args.steps}")
    print(f"  Top-k         : {args.topk}")
    print(f"  Device        : {device}")
    print("=" * 65)

    # ── Build eval_base_token_emb (used by all metric calls) ─────────────────
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    eval_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    eval_model.eval()
    embed = eval_model.get_input_embeddings()

    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X   = embed(dummy_ids)                          # (1, 1, d)

    eval_base_token_emb = get_baseline_embedding(
        args.eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0:1, :]                                              # (1, d)

    # ── Smoke test ────────────────────────────────────────────────────────────
    smoke_text = (
        "This is a really bad movie, although it has a promising start, "
        "it ended on a very low note."
    )
    attr_fn = make_attribution_fn(
        args.method, model_name, args.steps, args.baseline, device
    )
    res = attr_fn(smoke_text)
    print("\nSmoke test:")
    for tok, val in zip(res["tokens"], res["attributions"]):
        print(f"  {tok:>15s} : {val.item():+.6f}")

    # ── Dataset loop ──────────────────────────────────────────────────────────
    data = load_data(args.dataset)
    print(f"\nRunning on {len(data)} samples …\n")

    log_odds_sum = 0.0
    comps_sum    = 0.0
    suffs_sum    = 0.0
    total_time   = 0.0
    count        = 0

    for text, _label in tqdm(data):
        res = attr_fn(text)

        log_odd, _ = calculate_log_odds(
            res["nn_forward_func"], res["model"],
            res["input_embed"], res["position_embed"], res["type_embed"],
            res["attention_mask"], eval_base_token_emb,
            res["attr_full"], topk=args.topk,
        )
        comp = calculate_comprehensiveness(
            res["nn_forward_func"], res["model"],
            res["input_embed"], res["position_embed"], res["type_embed"],
            res["attention_mask"], eval_base_token_emb,
            res["attr_full"], topk=args.topk,
        )
        suff = calculate_sufficiency(
            res["nn_forward_func"], res["model"],
            res["input_embed"], res["position_embed"], res["type_embed"],
            res["attention_mask"], eval_base_token_emb,
            res["attr_full"], topk=args.topk,
        )

        log_odds_sum += log_odd
        comps_sum    += comp
        suffs_sum    += suff
        total_time   += res["time"]
        count        += 1

        if count % args.print_every == 0:
            print(
                f"  [{count:>5d}]  "
                f"Log-odds: {log_odds_sum / count:+.4f}  "
                f"Comp: {comps_sum / count:.4f}  "
                f"Suff: {suffs_sum / count:.4f}  "
                f"Time/sample: {total_time / count:.4f}s"
            )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  Final results  [{count} samples]")
    print("=" * 65)
    print(f"  Method        : {args.method}  [{method_label}]")
    print(f"  Log-odds      : {log_odds_sum / count:+.4f}")
    print(f"  Comprehensiveness : {comps_sum / count:.4f}")
    print(f"  Sufficiency   : {suffs_sum / count:.4f}")
    print(f"  Avg time/sample : {total_time / count:.4f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()