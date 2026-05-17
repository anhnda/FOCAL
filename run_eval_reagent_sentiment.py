"""
reagent_classification.py

ReAGent adapted for BERT-based sequence classification
(DistilBERT / BERT / RoBERTa + classification head).
"""

import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

try:
    from attcat_eval_sentiment import cache as _attcat_cache
except ImportError:
    _attcat_cache = {}

_clf_cache = _attcat_cache
_mlm_cache: dict = {}

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


def _hellinger(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.float().clamp(min=0.0)
    q = q.float().clamp(min=0.0)
    return (0.5 * ((p.sqrt() - q.sqrt()) ** 2).sum()).sqrt().item()


def _load_clf(model_name: str, device: str):
    if model_name not in _clf_cache:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        mdl.eval()
        _clf_cache[model_name] = {"model": mdl, "tokenizer": tok}
    entry = _clf_cache[model_name]
    return entry["tokenizer"], entry["model"]


def _load_mlm(mlm_name: str, device: str):
    if mlm_name not in _mlm_cache:
        tok = AutoTokenizer.from_pretrained(mlm_name, use_fast=True)
        mdl = AutoModelForMaskedLM.from_pretrained(mlm_name).to(device)
        mdl.eval()
        _mlm_cache[mlm_name] = (tok, mdl)
    return _mlm_cache[mlm_name]


def _get_label_dist(clf_model, clf_tokenizer, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor, device: str) -> torch.Tensor:
    with torch.no_grad():
        logits = clf_model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ).logits[0]
    return F.softmax(logits, dim=-1)


def _get_top_k_replacements(
    clf_tokenizer,
    mlm_tokenizer,
    mlm_model,
    input_ids_clf: torch.Tensor,
    position: int,
    clf_tokens: list[str],
    top_k: int,
    device: str,
) -> list[int]:
    clf_special_tokens = set(clf_tokenizer.all_special_tokens)
    mlm_special_tokens = set(mlm_tokenizer.all_special_tokens)

    masked_tokens = []
    for i, tok in enumerate(clf_tokens):
        if tok in clf_special_tokens:
            continue
        if i == position:
            masked_tokens.append(mlm_tokenizer.mask_token)
        else:
            masked_tokens.append(tok)

    masked_text = mlm_tokenizer.convert_tokens_to_string(masked_tokens)
    mlm_enc = mlm_tokenizer(masked_text, return_tensors="pt", truncation=True).to(device)

    mask_token_id = mlm_tokenizer.mask_token_id
    mask_positions = (mlm_enc["input_ids"][0] == mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return [input_ids_clf[0, position].item()]

    mask_pos = mask_positions[0].item()

    with torch.no_grad():
        mlm_logits = mlm_model(**mlm_enc).logits[0]

    top_k_ids_mlm = mlm_logits[mask_pos].topk(top_k * 3).indices

    replacement_ids = []
    for mlm_id in top_k_ids_mlm.tolist():
        token_str = mlm_tokenizer.decode([mlm_id]).strip()
        if not token_str or token_str in mlm_special_tokens:
            continue
        clf_ids = clf_tokenizer.encode(token_str, add_special_tokens=False)
        if len(clf_ids) == 1:
            replacement_ids.append(clf_ids[0])
        if len(replacement_ids) >= top_k:
            break

    if not replacement_ids:
        replacement_ids = [input_ids_clf[0, position].item()]

    return replacement_ids[:top_k]


def _compute_importance_scores(
    clf_tokenizer,
    clf_model,
    mlm_tokenizer,
    mlm_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    clf_tokens: list[str],
    p_orig: torch.Tensor,
    top_k: int,
    device: str,
) -> np.ndarray:
    L = input_ids.shape[1]
    special_ids = set(clf_tokenizer.all_special_ids)
    importance = np.zeros(L, dtype=np.float32)

    for i in range(L):
        tok_id = input_ids[0, i].item()
        if tok_id in special_ids:
            importance[i] = 0.0
            continue

        replacement_ids = _get_top_k_replacements(
            clf_tokenizer, mlm_tokenizer, mlm_model,
            input_ids, i, clf_tokens, top_k, device,
        )

        scores = []
        for rep_id in replacement_ids:
            x_rep = input_ids.clone()
            x_rep[0, i] = rep_id
            p_rep = _get_label_dist(clf_model, clf_tokenizer,
                                    x_rep, attention_mask, device)
            scores.append(_hellinger(p_orig, p_rep))

        importance[i] = float(np.mean(scores)) if scores else 0.0

    return importance


from xai_metrics import (
    calculate_log_odds,
    calculate_comprehensiveness,
    calculate_sufficiency,
)


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


def _compute_faithfulness_metrics(
    clf_model,
    clf_tokenizer,
    model_name: str,
    sentence: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    importance: np.ndarray,
    device: str,
    eval_base_token_emb: torch.Tensor,
    topk_pct: int = 20,
) -> tuple[float, float, float, int]:
    get_inputs, _, nn_forward_func = _get_helper_fns(model_name)

    embed = clf_model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids.to(device))
        logits0 = clf_model(
            inputs_embeds=X, attention_mask=attention_mask.to(device)
        ).logits[0]
    pred_id = int(logits0.argmax().item())

    inp = get_inputs(clf_model, clf_tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

    X              = X.to(device)
    position_embed = position_embed.to(device) if position_embed is not None else None
    type_embed     = type_embed.to(device)     if type_embed     is not None else None
    base_token_emb = eval_base_token_emb.to(device)

    attr_full        = torch.tensor(importance, dtype=torch.float32, device=device)
    attention_mask_d = attention_mask.to(device)

    log_odd, _ = calculate_log_odds(
        nn_forward_func, clf_model, X, position_embed, type_embed,
        attention_mask_d, base_token_emb, attr_full, topk=topk_pct,
    )
    comp = calculate_comprehensiveness(
        nn_forward_func, clf_model, X, position_embed, type_embed,
        attention_mask_d, base_token_emb, attr_full, topk=topk_pct,
    )
    suff = calculate_sufficiency(
        nn_forward_func, clf_model, X, position_embed, type_embed,
        attention_mask_d, base_token_emb, attr_full, topk=topk_pct,
    )

    return log_odd, comp, suff, pred_id


def reagent_classification(
    sentence: str,
    model_name: str,
    top_k: int = 3,
    topk_pct: int = 20,
    mlm_name: str | None = None,
    show_special_tokens: bool = False,
    device: str | None = None,
    eval_base_token_emb: torch.Tensor | None = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()

    if mlm_name is None:
        mlm_name = model_name

    clf_tokenizer, clf_model = _load_clf(model_name, device)
    mlm_tokenizer, mlm_model = _load_mlm(mlm_name, device)

    enc = clf_tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        return_special_tokens_mask=True,
    )
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    clf_tokens     = clf_tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    p_orig = _get_label_dist(clf_model, clf_tokenizer,
                             input_ids, attention_mask, device)

    importance = _compute_importance_scores(
        clf_tokenizer, clf_model,
        mlm_tokenizer, mlm_model,
        input_ids, attention_mask, clf_tokens,
        p_orig, top_k, device,
    )

    if eval_base_token_emb is None:
        _, get_base_token_emb, _ = _get_helper_fns(model_name)
        eval_base_token_emb = get_base_token_emb(clf_model, clf_tokenizer, device)

    log_odd, comp, suff, pred_id = _compute_faithfulness_metrics(
        clf_model, clf_tokenizer, model_name, sentence,
        input_ids, attention_mask,
        importance, device,
        eval_base_token_emb=eval_base_token_emb,
        topk_pct=topk_pct,
    )

    t1 = time.perf_counter()

    special_ids = set(clf_tokenizer.all_special_ids)
    out_tokens, out_attr = [], []
    for tok_str, tok_id, imp in zip(clf_tokens, input_ids[0].tolist(), importance.tolist()):
        if not show_special_tokens and tok_id in special_ids:
            continue
        out_tokens.append(tok_str)
        out_attr.append(imp)

    return {
        "tokens":          out_tokens,
        "attributions":    out_attr,
        "predicted_label": pred_id,
        "log_odd":         log_odd,
        "comp":            comp,
        "suff":            suff,
        "time":            t1 - t0,
    }


def run_benchmark(args):
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    model_name   = MODEL_NAMES[(args.model, args.dataset)]
    dataset_name = args.dataset

    # Print header in the same order as PACE / AttCAT
    print(f"Device        : {device}")
    print(f"Model         : {model_name}")
    print(f"Dataset       : {dataset_name}")
    print(f"MLM oracle    : {model_name}  (top_k={args.top_k})")
    print(f"Eval baseline : {args.eval_baseline}")

    _load_clf(model_name, device)
    _load_mlm(model_name, device)

    # Build eval_base_token_emb once — identical to PACE / AttCAT
    from pace_gradients import get_baseline_embedding
    clf_tokenizer = _clf_cache[model_name]["tokenizer"]
    clf_model     = _clf_cache[model_name]["model"]
    embed         = clf_model.get_input_embeddings()

    with torch.no_grad():
        dummy_ids = torch.tensor([[clf_tokenizer.cls_token_id or 0]], device=device)
        dummy_X   = embed(dummy_ids)   # (1, 1, d)

    # Computed once, reused for all metric calls
    eval_base_token_emb = get_baseline_embedding(
        args.eval_baseline, embed, clf_tokenizer, dummy_X, device
    )[0, 0:1, :]   # (1, d)

    # Smoke test — identical sentence and format to PACE / AttCAT
    text = "This is a really bad movie, although it has a promising start, it ended on a very low note."
    res  = reagent_classification(
        sentence=text,
        model_name=model_name,
        top_k=args.top_k,
        topk_pct=args.topk_pct,
        device=device,
        show_special_tokens=False,
        eval_base_token_emb=eval_base_token_emb,
    )
    print("\nSmoke test:")
    for tok, val in zip(res["tokens"], res["attributions"]):
        print(f"{tok:>12s} : {val:+.6f}")

    # Dataset — identical sampling logic to PACE / AttCAT
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
    print("\nStarting ReAGent attribution computation...")

    for row in tqdm(data):
        text = row[0]
        res  = reagent_classification(
            sentence=text,
            model_name=model_name,
            top_k=args.top_k,
            topk_pct=args.topk_pct,
            device=device,
            show_special_tokens=False,
            eval_base_token_emb=eval_base_token_emb,
        )

        log_odds   += res["log_odd"]
        comps      += res["comp"]
        suffs      += res["suff"]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         choices=["distilbert", "bert", "roberta"],
                        default="distilbert")
    parser.add_argument("--dataset",       choices=["sst2", "imdb", "rotten"],
                        default="sst2")
    parser.add_argument("--top_k",         type=int, default=3)
    parser.add_argument("--topk_pct",      type=int, default=20)
    parser.add_argument("--eval-baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"])
    args = parser.parse_args()
    run_benchmark(args)