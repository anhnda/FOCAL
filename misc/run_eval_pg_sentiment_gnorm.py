"""
run_eval_pg_sentiment_gnorm.py

Evaluate PACE gradient classification attribution under different scalar-gate
gradient normalisation strategies.

Normalisation modes
-------------------
  sign_norm   — grad_ex / (grad_ex.sum(dim=1, keepdim=True) + 1e-10)
                Original PACE normalisation (signed sum + eps).
  sign_magl2  — grad_ex / (||grad_ex||_2 + 1e-10)
                L2-magnitude across tokens per step.
  sign_magl1  — grad_ex / (||grad_ex||_1 + 1e-10)
                L1-magnitude across tokens per step.
  safe_norm   — grad_ex / (0→eps, else signed sum)
                Zero-safe variant of sign_norm; no eps bias on non-zero steps.

Usage examples
--------------
  python run_eval_pg_sentiment_gnorm.py --gnorm sign_norm  --model distilbert --dataset sst2
  python run_eval_pg_sentiment_gnorm.py --gnorm sign_magl2 --model bert        --dataset imdb
  python run_eval_pg_sentiment_gnorm.py --gnorm safe_norm  --model roberta     --dataset rotten \\
         --baseline zero --eval-baseline pad --steps 50
"""

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
from pace_gradient_gnorm import pace_gradient_gnorm, _GNORM_MODES

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

# Human-readable descriptions shown in the header/summary
_GNORM_LABELS = {
    "sign_norm":   "signed-sum + eps        (original PACE)",
    "sign_magl2":  "L2-magnitude across tokens + eps",
    "sign_magl1":  "L1-magnitude across tokens + eps",
    "safe_norm":   "zero-safe signed-sum    (no eps bias)",
    "square_norm": "grad² / sum(grad²)      (non-negative, sums to ~1)",
}


# ── Data loader ───────────────────────────────────────────────────────────────
def load_data(dataset_name: str, num_samples: int):
    if dataset_name == "imdb":
        dataset = load_dataset("imdb")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))
        data    = random.sample(data, min(num_samples, len(data)))
    elif dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2")["test"]
        data    = list(zip(dataset["sentence"], dataset["label"]))
        if len(data) > num_samples:
            data = random.sample(data, num_samples)
    elif dataset_name == "rotten":
        dataset = load_dataset("rotten_tomatoes")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))
        if len(data) > num_samples:
            data = random.sample(data, num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PACE gradient classification under different scalar-gate "
            "gradient normalisation strategies."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--gnorm", type=str, required=True,
        choices=list(_GNORM_MODES),
        metavar="GNORM",
        help=(
            "Gradient normalisation mode:\n"
            "  sign_norm   — signed sum + eps  (original PACE default)\n"
            "  sign_magl2  — L2-norm across tokens + eps\n"
            "  sign_magl1  — L1-norm across tokens + eps\n"
            "  safe_norm   — zero-safe signed sum (no eps on non-zero steps)\n"
            "  square_norm — grad² / sum(grad²)  (non-negative, sums to ~1)"
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
        help="Baseline embedding for the integration path  (default: mask)",
    )
    parser.add_argument(
        "--eval-baseline", type=str, default="mask",
        choices=["mask", "pad", "zero", "mean", "random"],
        help="Baseline used to ablate tokens in faithfulness metrics  (default: mask)",
    )
    parser.add_argument("--topk",        type=int, default=20,
                        help="Top-k %% of tokens to ablate  (default: 20)")
    parser.add_argument("--num-samples", type=int, default=2000,
                        help="Max samples to evaluate  (default: 2000)")
    parser.add_argument("--print-every", type=int, default=100,
                        help="Print running averages every N samples  (default: 100)")
    args = parser.parse_args()

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = MODEL_NAMES[(args.model, args.dataset)]
    gnorm_desc = _GNORM_LABELS[args.gnorm]

    print("=" * 68)
    print("  PACE Gradient — Scalar-Gate Gradient Normalisation Ablation")
    print("=" * 68)
    print(f"  gnorm         : {args.gnorm}  [{gnorm_desc}]")
    print(f"  Model         : {model_name}")
    print(f"  Dataset       : {args.dataset}")
    print(f"  PG baseline   : {args.baseline}")
    print(f"  Eval baseline : {args.eval_baseline}")
    print(f"  Steps         : {args.steps}")
    print(f"  Top-k         : {args.topk}%")
    print(f"  Max samples   : {args.num_samples}")
    print(f"  Device        : {device}")
    print("=" * 68)

    # ── Build eval_base_token_emb once ────────────────────────────────────────
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    eval_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    eval_model.eval()
    embed = eval_model.get_input_embeddings()

    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X   = embed(dummy_ids)               # (1, 1, d)

    eval_base_token_emb = get_baseline_embedding(
        args.eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0:1, :]                                   # (1, d)

    # ── Smoke test ────────────────────────────────────────────────────────────
    smoke_text = (
        "This is a really bad movie, although it has a promising start, "
        "it ended on a very low note."
    )
    smoke = pace_gradient_gnorm(
        smoke_text, steps=args.steps, model_name=model_name,
        show_special_tokens=False, baseline=args.baseline,
        device=device, gnorm=args.gnorm,
    )
    print("\nSmoke test:")
    for tok, val in zip(smoke["tokens"], smoke["attributions"]):
        print(f"  {tok:>15s} : {val.item():+.6f}")

    # ── Dataset loop ──────────────────────────────────────────────────────────
    data = load_data(args.dataset, args.num_samples)
    print(f"\nRunning on {len(data)} samples …\n")

    log_odds_sum = 0.0
    comps_sum    = 0.0
    suffs_sum    = 0.0
    total_time   = 0.0
    count        = 0

    for text, _label in tqdm(data):
        res = pace_gradient_gnorm(
            sentence=text,
            steps=args.steps,
            model_name=model_name,
            show_special_tokens=False,
            baseline=args.baseline,
            device=device,
            gnorm=args.gnorm,
        )

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
    print("\n" + "=" * 68)
    print(f"  Final results  [{count} samples]")
    print("=" * 68)
    print(f"  gnorm             : {args.gnorm}  [{gnorm_desc}]")
    print(f"  Log-odds          : {log_odds_sum / count:+.4f}")
    print(f"  Comprehensiveness : {comps_sum / count:.4f}")
    print(f"  Sufficiency       : {suffs_sum / count:.4f}")
    print(f"  Avg time/sample   : {total_time / count:.4f}s")
    print("=" * 68)


if __name__ == "__main__":
    main()