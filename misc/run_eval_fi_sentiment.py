"""
run_fi_eval_sentiment.py

Functional Information (FI) attribution adapted for BERT-based sequence
classification (DistilBERT / BERT / RoBERTa + classification head).

FI variants supported (same as fi_gpt2.py):
  fi           -- E[(∇f)² / f]          (standard, default)
  smooth_grad  -- E[∇f]
  smooth_grad_sq -- E[(∇f)²]
  fi_cov       -- E[(Σ∇f)·∇f / f]

f(z) = softmax(classifier(z))[pred_class]   (scalar, always > 0)

Evaluation metrics: log-odds, comprehensiveness, sufficiency
(same xai_metrics pattern as attcat / PACE / SLALOM eval scripts).
"""

import time
import random
import argparse
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from xai_metrics import (
    calculate_log_odds,
    calculate_comprehensiveness,
    calculate_sufficiency,
)
from pace_gradients import get_baseline_embedding

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

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

_cache = {}


def _load_model(model_name: str, device: str):
    if model_name not in _cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model     = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()
        _cache[model_name] = {"model": model, "tokenizer": tokenizer}
    return _cache[model_name]["model"], _cache[model_name]["tokenizer"]


# ---------------------------------------------------------------------------
# Covariance helper (mirrors fi_gpt2.py)
# ---------------------------------------------------------------------------

def _estimate_covariance(embed: torch.Tensor, regularise: float = 1e-4) -> torch.Tensor:
    """
    Estimate a shared (D, D) covariance from embed [1, L, D].
    Falls back to diagonal variance for single-sample case.
    """
    flat = embed.squeeze(0)           # [L, D]
    B, D = flat.shape
    if B < 2:
        var = flat.var(dim=0).clamp(min=regularise)
        return torch.diag(var)
    xc  = flat - flat.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / (B - 1)
    cov += regularise * torch.eye(D, device=flat.device, dtype=flat.dtype)
    return cov


def _cholesky_safe(cov: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.cholesky(cov)
    except RuntimeError:
        d = torch.abs(torch.diag(cov)).clamp(min=1e-6)
        try:
            return torch.linalg.cholesky(cov + d.min() * torch.eye(cov.shape[0], device=cov.device))
        except RuntimeError:
            return torch.linalg.cholesky(cov + d.mean() * torch.eye(cov.shape[0], device=cov.device))


# ---------------------------------------------------------------------------
# Core FI classification attribution
# ---------------------------------------------------------------------------

def fi_classification(
    sentence: str,
    model_name: str,
    device: str = "cpu",
    n: int = 20,
    var_spread: float = 0.15,
    method: str = "fi",
    show_special_tokens: bool = False,
) -> dict:
    """
    Functional Information attribution for a BERT-based classifier.

    f(z) = softmax(classifier(z))[pred_class]  — scalar, always > 0.

    Args:
        sentence   : Input text.
        model_name : HuggingFace classifier model name.
        device     : Torch device.
        n          : Monte-Carlo perturbation draws.
        var_spread : Noise scale multiplier on per-dim embedding variance.
        method     : 'fi' | 'smooth_grad' | 'smooth_grad_sq' | 'fi_cov'
        show_special_tokens: Whether to include [CLS]/[SEP] in output.

    Returns dict with keys:
        tokens, attributions, predicted_label, time,
        model, nn_forward_func, input_embed, attention_mask,
        position_embed, type_embed, attr_full
    """
    t0 = time.perf_counter()

    model, tokenizer = _load_model(model_name, device)

    if "distilbert" in model_name:
        from distilbert_helper import get_inputs, nn_forward_func
    elif "roberta" in model_name:
        from roberta_helper import get_inputs, nn_forward_func
    elif "bert" in model_name:
        from bert_helper import get_inputs, nn_forward_func
    else:
        raise NotImplementedError(f"No helper for {model_name}")

    enc = tokenizer(sentence, return_tensors="pt", truncation=True,
                    return_special_tokens_mask=True)
    enc            = {k: v.to(device) for k, v in enc.items()}
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    fwd_params   = inspect.signature(model.forward).parameters
    extra_kwargs = {}
    if "token_type_ids" in fwd_params and "token_type_ids" in enc:
        extra_kwargs["token_type_ids"] = enc["token_type_ids"]

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)   # (1, L, D)

    L, D = X.shape[1], X.shape[2]

    # Predicted class (fixed across all perturbations)
    with torch.no_grad():
        logits0 = model(inputs_embeds=X, attention_mask=attention_mask,
                        **extra_kwargs).logits[0]
    pred_id = int(logits0.argmax().item())

    # Per-dim noise variance: var_spread * Var_over_tokens(embed)
    sigma2_vec = X.squeeze(0).var(dim=0) * var_spread   # (D,)

    # Covariance for fi_cov
    if method == "fi_cov":
        Sigma = _estimate_covariance(X) * var_spread    # (D, D)
        Sigma = Sigma.to(device)

    # Monte-Carlo accumulation
    accumulated = torch.zeros(1, L, D, device=device, dtype=X.dtype)

    for _ in range(n):
        noise = torch.randn_like(X) * sigma2_vec.sqrt()
        z     = (X + noise).requires_grad_(True)

        logits_z = model(inputs_embeds=z, attention_mask=attention_mask,
                         **extra_kwargs).logits[0]               # (C,)

        # f(z) = softmax probability of predicted class — always > 0
        f_val = F.softmax(logits_z, dim=-1)[pred_id].clamp(min=1e-8)

        model.zero_grad()
        f_val.backward()
        grad = z.grad.detach()   # (1, L, D)

        if method == "fi":
            accumulated = accumulated + (grad ** 2) / f_val.detach()

        elif method == "smooth_grad":
            accumulated = accumulated + grad

        elif method == "smooth_grad_sq":
            accumulated = accumulated + (grad ** 2)

        elif method == "fi_cov":
            g       = grad.squeeze(0)                                    # (L, D)
            sigma_g = (Sigma @ g.unsqueeze(-1)).squeeze(-1)              # (L, D)
            fi_cov  = (sigma_g * g) / f_val.detach()                    # (L, D)
            accumulated = accumulated + fi_cov.unsqueeze(0)

        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose: fi | fi_cov | smooth_grad | smooth_grad_sq"
            )

    mean_attr = accumulated / n                                          # (1, L, D)
    attr_full = mean_attr.norm(dim=-1).squeeze(0).detach()              # (L,)

    # position/type embeddings for metric calls
    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
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
        "predicted_label": pred_id,
        "time":            time.perf_counter() - t0,
        # raw tensors for eval script metric calls
        "model":           model,
        "nn_forward_func": nn_forward_func,
        "input_embed":     X,
        "attention_mask":  attention_mask,
        "position_embed":  position_embed,
        "type_embed":      type_embed,
        "attr_full":       attr_full,
    }


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(args):
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = MODEL_NAMES[(args.model, args.dataset)]

    print(f"Device        : {device}")
    print(f"Model         : {model_name}")
    print(f"Dataset       : {args.dataset}")
    print(f"FI method     : {args.method}")
    print(f"n draws       : {args.n}")
    print(f"var_spread    : {args.var_spread}")
    print(f"Eval baseline : {args.eval_baseline}")

    # Load model once to build eval_base_token_emb
    model, tokenizer = _load_model(model_name, device)
    embed = model.get_input_embeddings()
    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X   = embed(dummy_ids)   # (1, 1, D)

    eval_base_token_emb = get_baseline_embedding(
        args.eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0:1, :]   # (1, D)

    # Smoke test
    demo = ("This is a really bad movie, although it has a promising start, "
            "it ended on a very low note.")
    res_demo = fi_classification(
        demo, model_name=model_name, device=device,
        n=args.n, var_spread=args.var_spread, method=args.method,
    )
    print("\nSmoke test:")
    for tok, val in zip(res_demo["tokens"], res_demo["attributions"]):
        print(f"  {tok:>15s} : {val.item():+.6f}")

    # Dataset
    print("\nLoading dataset ...")
    if args.dataset == "imdb":
        dataset = load_dataset("imdb")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))
        data    = random.sample(data, min(args.num_samples, len(data)))
    elif args.dataset == "sst2":
        dataset = load_dataset("glue", "sst2")["validation"]
        data    = list(zip(dataset["sentence"], dataset["label"]))
    elif args.dataset == "rotten":
        dataset = load_dataset("rotten_tomatoes")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))
        data    = random.sample(data, min(args.num_samples, len(data)))

    print(f"Evaluating {len(data)} samples ...")

    log_odds_sum = comps_sum = suffs_sum = total_time = 0.0
    count = errors = 0

    for row in tqdm(data):
        text = row[0]
        try:
            res = fi_classification(
                text, model_name=model_name, device=device,
                n=args.n, var_spread=args.var_spread, method=args.method,
                show_special_tokens=False,
            )

            log_odd, _ = calculate_log_odds(
                res["nn_forward_func"], res["model"],
                res["input_embed"], res["position_embed"], res["type_embed"],
                res["attention_mask"], eval_base_token_emb,
                res["attr_full"], topk=20,
            )
            comp = calculate_comprehensiveness(
                res["nn_forward_func"], res["model"],
                res["input_embed"], res["position_embed"], res["type_embed"],
                res["attention_mask"], eval_base_token_emb,
                res["attr_full"], topk=20,
            )
            suff = calculate_sufficiency(
                res["nn_forward_func"], res["model"],
                res["input_embed"], res["position_embed"], res["type_embed"],
                res["attention_mask"], eval_base_token_emb,
                res["attr_full"], topk=20,
            )

            log_odds_sum += log_odd
            comps_sum    += comp
            suffs_sum    += suff
            total_time   += res["time"]
            count        += 1

        except Exception:
            errors += 1
            if errors <= 5:
                import traceback; traceback.print_exc()

        if count % args.print_step == 0 and count > 0:
            print(
                f"[{count}/{len(data)}]  "
                f"Log-odds: {log_odds_sum/count:.4f}  "
                f"Comp: {comps_sum/count:.4f}  "
                f"Suff: {suffs_sum/count:.4f}  "
                f"Time: {total_time/count:.4f}s"
            )

    print(f"\n{'─'*52}")
    print(f"FI-{args.method}  |  {args.model} / {args.dataset}")
    n = max(count, 1)
    print(f"  Log-odds         : {log_odds_sum/n:.6f}")
    print(f"  Comprehensiveness: {comps_sum/n:.6f}")
    print(f"  Sufficiency      : {suffs_sum/n:.6f}")
    print(f"  Avg time/sample  : {total_time/n:.4f}s")
    print(f"  Evaluated        : {count}  |  Errors: {errors}")
    print(f"{'─'*52}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        choices=["distilbert", "bert", "roberta"],
                        default="distilbert")
    parser.add_argument("--dataset",      choices=["sst2", "imdb", "rotten"],
                        required=True)
    parser.add_argument("--method",       choices=["fi", "smooth_grad", "smooth_grad_sq", "fi_cov"],
                        default="fi")
    parser.add_argument("--n",            type=int,   default=20,
                        help="Monte-Carlo perturbation draws")
    parser.add_argument("--var_spread",   type=float, default=0.15,
                        help="Noise scale multiplier on per-dim embedding variance")
    parser.add_argument("--num_samples",  type=int,   default=1000)
    parser.add_argument("--print_step",   type=int,   default=100)
    parser.add_argument("--eval-baseline", type=str,  default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Baseline embedding used to replace tokens in faithfulness metrics")
    args = parser.parse_args()
    run_benchmark(args)