"""
Benchmark Script for LIME Attribution on Sentiment Classification

Usage:
    python eval_lime_sentiment.py --model distilbert --dataset sst2 --n-samples 1000

Mirrors eval_fd_sentiment.py's CLI and reporting format so results are
directly comparable. The single behavioural difference is the method:
LIME (random-mask surrogate regression) instead of FOCAL/FD.
"""
import time
from tqdm import tqdm
import torch
import random
import argparse
import numpy as np
from datasets import load_dataset
from xai_metrics import *
from LIME import lime_classification, get_baseline_embedding
from transformers import AutoTokenizer, AutoModelForSequenceClassification

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
# import os
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str, default="distilbert",
                        choices=["distilbert", "bert", "roberta"])
    parser.add_argument("--dataset",       type=str, choices=["sst2", "imdb", "rotten"])
    parser.add_argument("--n-samples",     type=int, default=100,
                        help="Number of binary masks to draw per input (LIME's N)")
    parser.add_argument("--p-keep",        type=float, default=0.5,
                        help="Bernoulli keep-probability for non-special tokens")
    parser.add_argument("--sigma",         type=float, default=0.25,
                        help="Kernel bandwidth in cosine-distance space")
    parser.add_argument("--ridge-lambda",  type=float, default=1.0,
                        help="L2 penalty for the surrogate ridge regression")
    parser.add_argument("--baseline",      type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Baseline embedding used to replace masked tokens during LIME")
    parser.add_argument("--eval-baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Baseline embedding used to replace tokens in faithfulness metrics")
    parser.add_argument("--chunk-size",    type=int, default=64,
                        help="Max batch size per forward pass")
    parser.add_argument("--seed",          type=int, default=42,
                        help="Seed for LIME mask sampler (use -1 for non-deterministic)")
    args = parser.parse_args()

    n_samples     = args.n_samples
    p_keep        = args.p_keep
    sigma         = args.sigma
    ridge_lambda  = args.ridge_lambda
    model         = args.model
    dataset_name  = args.dataset
    baseline      = args.baseline
    eval_baseline = args.eval_baseline
    chunk_size    = args.chunk_size
    seed          = None if args.seed < 0 else args.seed

    if model == "distilbert":
        if dataset_name == "sst2":
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        elif dataset_name == "imdb":
            model_name = "textattack/distilbert-base-uncased-imdb"
        elif dataset_name == "rotten":
            model_name = "textattack/distilbert-base-uncased-rotten-tomatoes"
    elif model == "bert":
        if dataset_name == "sst2":
            model_name = "textattack/bert-base-uncased-SST-2"
        elif dataset_name == "imdb":
            model_name = "textattack/bert-base-uncased-imdb"
        elif dataset_name == "rotten":
            model_name = "textattack/bert-base-uncased-rotten-tomatoes"
    elif model == "roberta":
        if dataset_name == "sst2":
            model_name = "textattack/roberta-base-SST-2"
        elif dataset_name == "imdb":
            model_name = "textattack/roberta-base-imdb"
        elif dataset_name == "rotten":
            model_name = "textattack/roberta-base-rotten-tomatoes"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device}")
    print(f"Model         : {model_name}")
    print(f"Dataset       : {dataset_name}")
    print(f"LIME baseline : {baseline}")
    print(f"Eval baseline : {eval_baseline}")
    print(f"N samples     : {n_samples}  p_keep={p_keep}  sigma={sigma}  lambda={ridge_lambda}")
    print(f"Chunk size    : {chunk_size}  seed={seed}")
    print(f"Method        : LIME (forward-only, random-mask surrogate regression)")

    # Load model once to build eval_base_token_emb
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    eval_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    eval_model.eval()
    embed = eval_model.get_input_embeddings()

    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X   = embed(dummy_ids)

    eval_base_token_emb = get_baseline_embedding(
        eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0:1, :]

    # Smoke test
    text = "This is a really bad movie, although it has a promising start, it ended on a very low note."
    res  = lime_classification(
        sentence=text,
        n_samples=n_samples,
        p_keep=p_keep,
        sigma=sigma,
        ridge_lambda=ridge_lambda,
        model_name=model_name,
        show_special_tokens=False,
        baseline=baseline,
        chunk_size=chunk_size,
        seed=seed,
    )
    print("\nSmoke test:")
    for tok, val in zip(res["tokens"], res["attributions"]):
        print(f"{tok:>12s} : {val.item():+.6f}")
    print(f"Smoke test time: {res['time']:.4f}s")

    # Dataset
    if dataset_name == "imdb":
        dataset = load_dataset("imdb")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))
        data    = random.sample(data, 2000)
    elif dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2")["test"]
        data    = list(zip(dataset["sentence"], dataset["label"]))
    elif dataset_name == "rotten":
        dataset = load_dataset("rotten_tomatoes")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))

    log_odds, comps, suffs, count, total_time = 0, 0, 0, 0, 0
    print_step = 100
    print("\nStarting LIME attribution computation...")

    for row in tqdm(data):
        text = row[0]
        res  = lime_classification(
            sentence=text,
            n_samples=n_samples,
            p_keep=p_keep,
            sigma=sigma,
            ridge_lambda=ridge_lambda,
            model_name=model_name,
            show_special_tokens=False,
            baseline=baseline,
            chunk_size=chunk_size,
            seed=seed,
        )

        attr = res["attr_full"]

        log_odd, _ = calculate_log_odds(
            res["nn_forward_func"], res["model"],
            res["input_embed"], res["position_embed"], res["type_embed"],
            res["attention_mask"], eval_base_token_emb,
            attr, topk=20,
        )
        comp = calculate_comprehensiveness(
            res["nn_forward_func"], res["model"],
            res["input_embed"], res["position_embed"], res["type_embed"],
            res["attention_mask"], eval_base_token_emb,
            attr, topk=20,
        )
        suff = calculate_sufficiency(
            res["nn_forward_func"], res["model"],
            res["input_embed"], res["position_embed"], res["type_embed"],
            res["attention_mask"], eval_base_token_emb,
            attr, topk=20,
        )

        log_odds   += log_odd
        comps      += comp
        suffs      += suff
        total_time += res["time"]
        count      += 1

        if count % print_step == 0:
            print(
                f"[{count}]  "
                f"Log-odds: {log_odds/count:.4f}  "
                f"Comp: {comps/count:.4f}  "
                f"Suff: {suffs/count:.4f}  "
                f"Time: {total_time/count:.4f}s"
            )

    print(
        f"\nFinal [{count} samples]  "
        f"Log-odds: {log_odds/count:.4f}  "
        f"Comp: {comps/count:.4f}  "
        f"Suff: {suffs/count:.4f}  "
        f"Time: {total_time/count:.4f}s"
    )