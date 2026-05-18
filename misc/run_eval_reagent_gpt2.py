"""
run_eval_reagent_gpt2.py
========================
Benchmark ReAGent-style occlusion attribution on GPT-2 + TellMeWhy.
Supports --baseline (method) and --eval-baseline (faithfulness metrics).

Usage:
    python run_eval_reagent_gpt2.py --num_samples 200 --use_gold
    python run_eval_reagent_gpt2.py --baseline pad --eval-baseline mean
"""

import random
import argparse
import traceback

import numpy as np
import torch
from tqdm import tqdm

from reagent_gpt2 import reagent_gpt2, get_model_tokenizer, _build_base_embed
from xai_metrics_gpt2 import calculate_all_metrics_gpt2

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def load_tellmewhy_txt(path: str, num_samples: int, use_gold: bool) -> list:
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts    = line.split("\t")
            prompt   = parts[0].strip()
            gold_ans = parts[1].strip() if (use_gold and len(parts) > 1) else None
            if not prompt:
                continue
            samples.append({"question": prompt, "gold_answer": gold_ans})
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    print(f"Loaded {len(samples)} samples from {path}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Single-sample pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_single_example(
    question:        str,
    gold_answer,
    model_name:      str,
    device:          str,
    topk:            int,
    max_new_tokens:  int,
    n_samples:       int,
    baseline:        str,
    eval_base_embed: torch.Tensor,   # (1, T, D) or (1, 1, D) — broadcast by metrics
) -> dict:
    res = reagent_gpt2(
        question=question,
        model_name=model_name,
        device=device,
        max_new_tokens=max_new_tokens,
        gold_answer=gold_answer,
        baseline=baseline,
    )

    # Override base_embed with eval_base_embed for metric calls
    # Expand to match sequence length if needed
    T          = res["input_embed"].shape[1]
    eval_base  = eval_base_embed.expand(1, T, -1)   # (1, T, D)

    metrics = calculate_all_metrics_gpt2(
        model=res["model"],
        input_embed=res["input_embed"],
        base_embed=eval_base,
        attributions=res["attributions"],
        answer_ids=res["answer_ids"],
        answer_positions=res["answer_positions"],
        topk=topk,
        n_samples=n_samples,
        device=device,
    )

    return {
        "tokens":           res["tokens"],
        "q_len":            res["q_len"],
        "attributions":     res["attributions"],
        "predicted_answer": res["predicted_answer"],
        "time":             res["time"],
        "soft_nc":          metrics["soft_nc"].item(),
        "soft_ns":          metrics["soft_ns"].item(),
        "log_odds":         metrics["log_odds"].item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("ReAGent (occlusion)  |  GPT-2  |  TellMeWhy")
    print("=" * 60)
    print(f"Device        : {device}")
    print(f"Model         : {args.model_name}")
    print(f"Dataset       : {args.data_path}")
    print(f"Samples       : {args.num_samples}")
    print(f"Top-k %       : {args.topk}")
    print(f"Baseline      : {args.baseline}")
    print(f"Eval baseline : {args.eval_baseline}")
    print(f"Gold answer   : {args.use_gold}")
    print(f"MC samples    : {args.n_samples}")
    print("=" * 60)

    print("\nLoading model ...")
    model, tokenizer = get_model_tokenizer(args.model_name, device)
    print("Model loaded.\n")

    # Build eval_base_embed once — shape (1, 1, D), broadcast to (1, T, D) per sample
    embed_layer = model.transformer.wte
    with torch.no_grad():
        dummy_embed = embed_layer(
            torch.tensor([[tokenizer.eos_token_id]], device=device)
        ).detach().cpu()   # (1, 1, D) — shape reference

    eval_base_embed = _build_base_embed(
        embed_layer, dummy_embed, args.eval_baseline, tokenizer, device="cpu"
    )   # (1, 1, D)

    samples = load_tellmewhy_txt(
        args.data_path,
        num_samples=args.num_samples,
        use_gold=args.use_gold,
    )
    if not samples:
        print("No samples found — check --data_path.")
        return

    total_soft_nc  = 0.0
    total_soft_ns  = 0.0
    total_log_odds = 0.0
    total_time     = 0.0
    count          = 0
    errors         = 0

    for idx, sample in enumerate(tqdm(samples, desc="ReAGent")):
        try:
            res = run_single_example(
                question        = sample["question"],
                gold_answer     = sample["gold_answer"],
                model_name      = args.model_name,
                device          = device,
                topk            = args.topk,
                max_new_tokens  = args.max_new_tokens,
                n_samples       = args.n_samples,
                baseline        = args.baseline,
                eval_base_embed = eval_base_embed,
            )

            total_soft_nc  += res["soft_nc"]
            total_soft_ns  += res["soft_ns"]
            total_log_odds += res["log_odds"]
            total_time     += res["time"]
            count          += 1

            if args.verbose and count <= 3:
                _print_sample(sample["question"], res)

            if count % args.print_step == 0:
                _print_running(count, len(samples),
                               total_soft_nc, total_soft_ns,
                               total_log_odds, total_time)

        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"\n[Error sample {idx}]: {str(exc)[:120]}")
                traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("FINAL RESULTS  —  ReAGent / GPT-2 / TellMeWhy")
    print("=" * 60)
    if count > 0:
        print(f"  Soft-NC  (Comprehensiveness) : {total_soft_nc  / count:.6f}")
        print(f"  Soft-NS  (Sufficiency)       : {total_soft_ns  / count:.6f}")
        print(f"  Log-odds                     : {total_log_odds / count:.6f}")
        print(f"  Avg time / sample            : {total_time     / count:.4f}s")
        print(f"  Successful samples           : {count} / {len(samples)}")
        print(f"  Errors                       : {errors}")
    else:
        print("  No samples processed successfully.")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _print_running(count, total, snc, sns, lo, t):
    print(f"\n[{count}/{total}] Running averages:")
    print(f"  Soft-NC  : {snc / count:.4f}")
    print(f"  Soft-NS  : {sns / count:.4f}")
    print(f"  Log-odds : {lo  / count:.4f}")
    print(f"  Avg time : {t   / count:.4f}s")


def _print_sample(question: str, res: dict):
    tokens = res["tokens"]
    scores = res["attributions"].tolist()
    q_len  = res["q_len"]
    print(f"\n{'─' * 60}")
    print(f"Q : {question[:120]}")
    print(f"A : {res['predicted_answer']}")
    q_scores = sorted(zip(tokens[:q_len], scores[:q_len]),
                      key=lambda x: x[1], reverse=True)
    print("Top-5 Q tokens by ReAGent attribution:")
    for tok, sc in q_scores[:5]:
        print(f"    {tok!r:20s}  {sc:.4f}")
    print(f"Soft-NC={res['soft_nc']:.4f}  "
          f"Soft-NS={res['soft_ns']:.4f}  "
          f"Log-odds={res['log_odds']:.4f}  "
          f"Time={res['time']:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark ReAGent Attribution on GPT-2 + TellMeWhy"
    )
    parser.add_argument("--data_path",     type=str,
                        default="datasets2/tellmewhy2.txt")
    parser.add_argument("--model_name",    type=str, default="gpt2",
                        help="gpt2 | gpt2-medium | gpt2-large | ./local-model")
    parser.add_argument("--num_samples",   type=int, default=200)
    parser.add_argument("--topk",          type=int, default=20)
    parser.add_argument("--max_new_tokens",type=int, default=30)
    parser.add_argument("--n_samples",     type=int, default=10,
                        help="Monte-Carlo draws for Soft-NC/NS")
    parser.add_argument("--use_gold",      action="store_true")
    parser.add_argument("--print_step",    type=int, default=50)
    parser.add_argument("--verbose",       action="store_true")
    parser.add_argument("--baseline",      type=str, default="zero",
                        choices=["zero", "pad", "mean"],
                        help="Baseline for base_embed returned by reagent_gpt2 "
                             "(affects method's embedding reference)")
    parser.add_argument("--eval-baseline", type=str, default="zero",
                        choices=["zero", "pad", "mean"],
                        help="Baseline embedding used to replace tokens in "
                             "faithfulness metrics (Soft-NC, Soft-NS, log-odds)")
    args = parser.parse_args()
    run_benchmark(args)