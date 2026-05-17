"""
attcat_eval_qa.py

Evaluation of AttCAT (Attentive Class Activation Tokens) attributions on
Question Answering datasets (SQuAD / SQuADv2), following the structure of
run_eval_pg_qa.py.

AttCAT paper: "AttCAT: Explaining Transformers via Attentive Class Activation
Tokens", Qiang et al., NeurIPS 2022.
https://github.com/qiangyao1988/AttCAT

AttCAT Algorithm (per token i, summed over all L layers):
  1. CAT_i^l  = grad(h_i^l) ⊙ h_i^l           (Hadamard product, no ReLU)
  2. AttCAT_i^l = mean_over_heads( alpha_i^l @ CAT_i^l )
  3. score_i  = sum_l  sum_d  AttCAT_i^l        (scalar per token)

For QA, attribution is computed separately for the start and end logits.

Metrics (identical to run_eval_pg_qa.py):
  calculate_log_odds_qa / calculate_comprehensiveness_qa / calculate_sufficiency_qa
  from xai_metrics, using QA helper functions from *_helper.py.
"""

import time
import tqdm
import torch
import random
import argparse
import traceback
import numpy as np
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
from xai_metrics import (
    calculate_log_odds_qa,
    calculate_comprehensiveness_qa,
    calculate_sufficiency_qa,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ---------------------------------------------------------------------------
# Model / tokenizer cache
# ---------------------------------------------------------------------------
cache = {}


def _get_cached(model_name: str, device: str):
    if model_name not in cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True,
        ).to(device)
        model.eval()
        cache[model_name] = {"model": model, "tokenizer": tokenizer}
    return cache[model_name]["model"], cache[model_name]["tokenizer"]


# ---------------------------------------------------------------------------
# Architecture helpers  (shared with attcat_eval_sentiment.py)
# ---------------------------------------------------------------------------

def _get_encoder_layers(model):
    if hasattr(model, "bert"):
        return list(model.bert.encoder.layer)
    if hasattr(model, "distilbert"):
        return list(model.distilbert.transformer.layer)
    if hasattr(model, "roberta"):
        return list(model.roberta.encoder.layer)
    raise RuntimeError("Unsupported model architecture.")


def _get_attn_submodule(layer):
    if hasattr(layer, "attention"):
        return layer.attention
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    return None


# ---------------------------------------------------------------------------
# Core AttCAT computation for QA
# ---------------------------------------------------------------------------

def _compute_attcat_scores(
    target_scalar: torch.Tensor,
    hidden_states_list: List[torch.Tensor],
    attn_weights_list: List[torch.Tensor],
    seq_len: int,
    device: str,
) -> torch.Tensor:
    """
    Compute AttCAT token scores for a single scalar target (start or end logit).

    Returns:
        attcat_scores: [seq_len] tensor of raw attribution values.
    """
    n_layers = len(hidden_states_list)
    attcat_scores = torch.zeros(seq_len, device=device)

    for l_idx in range(n_layers):
        h_l = hidden_states_list[l_idx]   # [1, seq, d] — must be in autograd graph

        try:
            (grad_h_l,) = torch.autograd.grad(
                target_scalar, h_l,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
        except RuntimeError:
            continue
        if grad_h_l is None:
            continue

        # CAT^l = grad ⊙ h  (no ReLU — preserve directionality)
        cat_l = (grad_h_l * h_l.detach()).squeeze(0)   # [seq, d]

        if l_idx < len(attn_weights_list):
            # alpha_l: [1, H, seq_q, seq_k] → squeeze → [H, seq_q, seq_k]
            alpha_l = attn_weights_list[l_idx].squeeze(0)
            # AttCAT^l_i = mean_H( sum_j alpha_{i,j} * cat_j )
            attcat_l = torch.einsum("hij,jd->hid", alpha_l, cat_l).mean(dim=0)  # [seq, d]
        else:
            attcat_l = cat_l   # plain CAT fallback

        attcat_scores = attcat_scores + attcat_l.sum(dim=-1)   # [seq]

    return attcat_scores


def attcat_qa(
    question: str,
    context: str,
    model_name: str,
    show_special_tokens: bool = False,
    device: str = "cpu",
) -> Dict:
    """
    Compute AttCAT attributions for a QA model on (question, context).

    Returns a dict with tokens, attributions_{start,end}, predicted_answer,
    start/end indices and logits, plus data needed by xai_metrics QA functions.
    """
    t0 = time.perf_counter()
    model, tokenizer = _get_cached(model_name, device)

    # ── helper imports (same pattern as pace_gradients.py) ───────────────────
    if "distilbert" in model_name:
        from distilbert_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "roberta" in model_name:
        from roberta_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "bert" in model_name:
        from bert_helper import get_inputs, get_base_token_emb, nn_forward_func
    else:
        raise NotImplementedError(f"No helper for {model_name}")

    # ── tokenise ─────────────────────────────────────────────────────────────
    enc = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
    )
    input_ids           = enc["input_ids"].to(device)
    attention_mask      = enc["attention_mask"].to(device)
    token_type_ids      = enc.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
    special_tokens_mask = enc["special_tokens_mask"].to(device)
    seq_len             = input_ids.shape[1]

    # ── forward hooks ────────────────────────────────────────────────────────
    # CRITICAL: h_l tensors must remain in the autograd graph.
    hidden_states_list: List[torch.Tensor] = []
    attn_weights_list:  List[torch.Tensor] = []
    hooks = []

    encoder_layers = _get_encoder_layers(model)

    def make_layer_hook(idx: int):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                h = None
                for t in reversed(out):
                    if isinstance(t, torch.Tensor) and t.dim() == 3:
                        h = t
                        break
                if h is None:
                    h = out[0]
            else:
                h = out
            hidden_states_list.append(h)   # keep in graph — NO detach
        return fn

    def make_attn_hook(idx: int):
        def fn(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                if out[1].dim() == 4:          # [1, H, seq, seq]
                    attn_weights_list.append(out[1].detach())
        return fn

    for idx, layer in enumerate(encoder_layers):
        hooks.append(layer.register_forward_hook(make_layer_hook(idx)))
        attn_mod = _get_attn_submodule(layer)
        if attn_mod is not None:
            hooks.append(attn_mod.register_forward_hook(make_attn_hook(idx)))

    # ── forward pass ─────────────────────────────────────────────────────────
    with torch.enable_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    for h in hooks:
        h.remove()

    start_logits = outputs.start_logits   # [1, seq]
    end_logits   = outputs.end_logits     # [1, seq]

    start_idx = int(start_logits.argmax(dim=-1).item())
    end_idx   = int(end_logits.argmax(dim=-1).item())

    start_logit_scalar = start_logits[0, start_idx]
    end_logit_scalar   = end_logits[0, end_idx]

    # softmax probabilities for metric functions
    prob_start = torch.softmax(start_logits, dim=-1)[0, start_idx].item()
    prob_end   = torch.softmax(end_logits,   dim=-1)[0, end_idx].item()

    # fallbacks if hooks fired nothing
    if len(hidden_states_list) == 0 and outputs.hidden_states is not None:
        hidden_states_list = list(outputs.hidden_states[1:])
    if len(attn_weights_list) == 0 and outputs.attentions is not None:
        attn_weights_list = [a.detach() for a in outputs.attentions if a is not None]

    # ── AttCAT: separate passes for start and end logits ─────────────────────
    attr_start = _compute_attcat_scores(
        start_logit_scalar, hidden_states_list, attn_weights_list, seq_len, device
    )
    attr_end = _compute_attcat_scores(
        end_logit_scalar, hidden_states_list, attn_weights_list, seq_len, device
    )

    # ── predicted answer span ────────────────────────────────────────────────
    answer_start = max(start_idx, 0)
    answer_end   = max(end_idx, answer_start)
    predicted_answer = tokenizer.decode(
        input_ids[0, answer_start : answer_end + 1],
        skip_special_tokens=True,
    )

    # ── token filter ─────────────────────────────────────────────────────────
    tokens_raw = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    special_ids_set = set(tokenizer.all_special_ids)

    if show_special_tokens:
        tokens     = tokens_raw
        attr_s_out = attr_start
        attr_e_out = attr_end
    else:
        keep       = [i for i, tid in enumerate(input_ids[0].tolist())
                      if tid not in special_ids_set]
        tokens     = [tokens_raw[i] for i in keep]
        attr_s_out = attr_start[keep]
        attr_e_out = attr_end[keep]

    # ── inputs for xai_metrics (QA variant) ──────────────────────────────────
    embed = model.get_input_embeddings()
    with torch.no_grad():
        input_embed = embed(input_ids)   # [1, seq, d]

    base_token_emb = get_base_token_emb(model, tokenizer, device)

    return {
        # Attribution outputs
        "tokens":             tokens,
        "attributions_start": attr_s_out.detach().cpu(),
        "attributions_end":   attr_e_out.detach().cpu(),
        "predicted_answer":   predicted_answer,
        "start_idx":          start_idx,
        "end_idx":            end_idx,
        "start_logit":        start_logit_scalar.item(),
        "end_logit":          end_logit_scalar.item(),
        "start_prob":         prob_start,
        "end_prob":           prob_end,
        # Full-length (incl. special tokens) for metrics
        "attr_start_full":    attr_start.detach(),
        "attr_end_full":      attr_end.detach(),
        # Tensors required by xai_metrics QA functions
        "model":              model,
        "input_embed":        input_embed,
        "attention_mask":     attention_mask,
        "special_tokens_mask": special_tokens_mask,
        "token_type_ids":     token_type_ids,
        "base_token_emb":     base_token_emb,
        "time":               time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Single-example runner (mirrors run_single_example in run_eval_pg_qa.py)
# ---------------------------------------------------------------------------

def run_single_example(
    question: str,
    context: str,
    model_name: str,
    device: str,
    topk: int = 20,
) -> Dict:
    res = attcat_qa(
        question=question,
        context=context,
        model_name=model_name,
        show_special_tokens=True,
        device=device,
    )

    model               = res["model"]
    input_embed         = res["input_embed"]
    attention_mask      = res["attention_mask"]
    special_tokens_mask = res["special_tokens_mask"]
    token_type_ids      = res["token_type_ids"]
    base_token_emb      = res["base_token_emb"]
    attr_start          = res["attr_start_full"]
    attr_end            = res["attr_end_full"]
    start_idx           = res["start_idx"]
    end_idx             = res["end_idx"]
    prob_start_orig     = res["start_prob"]
    prob_end_orig       = res["end_prob"]

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
        "tokens":             res["tokens"],
        "attributions_start": res["attributions_start"],
        "attributions_end":   res["attributions_end"],
        "predicted_answer":   res["predicted_answer"],
        "start_idx":          start_idx,
        "end_idx":            end_idx,
        "start_logit":        res["start_logit"],
        "end_logit":          res["end_logit"],
        "time":               res["time"],
        "log_odd_start":      log_odd_start,
        "comp_start":         comp_start,
        "suff_start":         suff_start,
        "log_odd_end":        log_odd_end,
        "comp_end":           comp_end,
        "suff_end":           suff_end,
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device : {device}")
    print(f"Model        : {args.model_name}")
    print(f"Dataset      : {args.dataset}")
    print(f"Num samples  : {args.num_samples}")
    print(f"Top-k        : {args.topk}%")

    # ── dataset ──────────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.dataset} ...")
    dataset = load_dataset(args.dataset, split="validation")
    data = list(zip(
        dataset["question"], dataset["context"],
        dataset["answers"],  dataset["id"],
    ))

    # Keep only short examples (same filter as run_eval_pg_qa.py)
    upd_data = [
        (q, c, a, idx)
        for q, c, a, idx in data
        if len((q + c).split()) < 80
    ]
    print(f"Filtered samples : {len(upd_data)}")

    answerable_data = [
        {"context": c, "question": q, "answers": a}
        for q, c, a, _ in upd_data
    ]

    if len(answerable_data) > args.num_samples:
        sampled_data = random.sample(answerable_data, args.num_samples)
    else:
        sampled_data = answerable_data
        print(f"Warning: Only {len(answerable_data)} samples available.")

    print(f"Evaluating {len(sampled_data)} samples with AttCAT QA ...\n")

    # warm-up / cache model
    _get_cached(args.model_name, device)

    # ── evaluation loop ───────────────────────────────────────────────────────
    total_log_odds_start = total_log_odds_end = 0.0
    total_comp_start     = total_comp_end     = 0.0
    total_suff_start     = total_suff_end     = 0.0
    total_time = 0.0
    count = errors = 0

    for idx, example in enumerate(tqdm.tqdm(sampled_data)):
        try:
            res = run_single_example(
                question=example["question"],
                context=example["context"],
                model_name=args.model_name,
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
                print(f"\n[{count}/{len(sampled_data)}] Running averages:")
                print(f"  Log-odds (start): {total_log_odds_start / count:.4f}")
                print(f"  Log-odds (end):   {total_log_odds_end   / count:.4f}")
                print(f"  Comp (start):     {total_comp_start     / count:.4f}")
                print(f"  Comp (end):       {total_comp_end       / count:.4f}")
                print(f"  Suff (start):     {total_suff_start     / count:.4f}")
                print(f"  Suff (end):       {total_suff_end       / count:.4f}")
                print(f"  Avg time:         {total_time / count:.4f}s")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n[WARN] Error on sample {idx}: {str(e)[:120]}")
                traceback.print_exc()
            continue

    # ── final results ─────────────────────────────────────────────────────────
    n = max(count, 1)
    print("\n=== Final Results ===")
    print(f"  Log-odds (start):          {total_log_odds_start / n:.6f}")
    print(f"  Comprehensiveness (start): {total_comp_start     / n:.6f}")
    print(f"  Sufficiency (start):       {total_suff_start     / n:.6f}")
    print(f"  Log-odds (end):            {total_log_odds_end   / n:.6f}")
    print(f"  Comprehensiveness (end):   {total_comp_end       / n:.6f}")
    print(f"  Sufficiency (end):         {total_suff_end       / n:.6f}")
    print(f"  Average time/sample:       {total_time / n:.4f}s")
    print(f"  Total samples evaluated:   {count}")
    if errors:
        print(f"  Skipped (errors):          {errors}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AttCAT attributions on Question Answering datasets."
    )
    parser.add_argument(
        "--model_name", type=str,
        default="deepset/bert-base-cased-squad2",
        help="HuggingFace QA model name",
    )
    parser.add_argument(
        "--dataset", type=str, default="squad",
        choices=["squad", "squad_v2"],
        help="HuggingFace dataset to evaluate on",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--topk", type=int, default=50,
        help="Percentage of top tokens for metrics calculation",
    )
    parser.add_argument(
        "--print_step", type=int, default=100,
        help="Print running averages every N samples",
    )

    args = parser.parse_args()

    # ── quick demo ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo_q = "Who invented the telephone?"
    demo_c = (
        "Alexander Graham Bell is often credited with inventing the telephone "
        "in 1876, though Elisha Gray filed a patent caveat on the same day."
    )
    print("\n--- AttCAT QA demo attribution ---")
    demo = attcat_qa(demo_q, demo_c, model_name=args.model_name,
                     show_special_tokens=False, device=device)
    print(f"Predicted answer: {demo['predicted_answer']!r}")
    print(f"Start logit idx={demo['start_idx']}  End logit idx={demo['end_idx']}")
    print("\nToken attributions (start | end):")
    for tok, s, e in zip(
        demo["tokens"],
        demo["attributions_start"].tolist(),
        demo["attributions_end"].tolist(),
    ):
        print(f"  {tok:>20s} : start={s:+.5f}  end={e:+.5f}")

    print()
    run_benchmark(args)