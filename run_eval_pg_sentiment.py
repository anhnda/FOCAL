import time
from tqdm import tqdm
import torch
import random
import argparse
import numpy as np
from datasets import load_dataset
from xai_metrics import *
from pace_gradients import pace_gradient_classification, get_baseline_embedding
from transformers import AutoTokenizer, AutoModelForSequenceClassification

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str, default="distilbert",
                        choices=["distilbert", "bert", "roberta"])
    parser.add_argument("--dataset",       type=str, choices=["sst2", "imdb", "rotten"])
    parser.add_argument("--steps",         type=int, default=100)
    parser.add_argument("--baseline",      type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Baseline embedding for PACE integration path")
    parser.add_argument("--eval-baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Baseline embedding used to replace tokens in faithfulness metrics")
    args = parser.parse_args()

    a, b          = 0, 1
    steps         = args.steps
    model         = args.model
    dataset_name  = args.dataset
    baseline      = args.baseline
    eval_baseline = args.eval_baseline

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
    print(f"PG baseline   : {baseline}")
    print(f"Eval baseline : {eval_baseline}")
    print(f"Range         : [{a}, {b}]  steps={steps}")

    # Load model once to build eval_base_token_emb
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    eval_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    eval_model.eval()
    embed = eval_model.get_input_embeddings()

    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X   = embed(dummy_ids)   # (1, 1, d)

    # Computed once, reused for all metric calls
    eval_base_token_emb = get_baseline_embedding(
        eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0:1, :]   # (1, d)

    # Smoke test
    text = "This is a really bad movie, although it has a promising start, it ended on a very low note."
    res  = pace_gradient_classification(
        text, a=a, b=b, steps=steps,
        model_name=model_name,
        show_special_tokens=False,
        baseline=baseline,
    )
    print("\nSmoke test:")
    for tok, val in zip(res["tokens"], res["attributions"]):
        print(f"{tok:>12s} : {val.item():+.6f}")

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
    print("\nStarting PACE attribution computation...")

    for row in tqdm(data):
        text = row[0]
        res  = pace_gradient_classification(
            sentence=text, a=a, b=b, steps=steps,
            model_name=model_name,
            show_special_tokens=False,
            baseline=baseline,
        )

        # Use attr_full (unfiltered, full-length including special tokens)
        # to match Doc 6 behavior where metrics are computed inside the
        # function on the full attribution vector before special token removal
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