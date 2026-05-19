"""
v8 — pure probability distributions. No thresholds, no labels, no classifier.

Hypothesis (two-sided):
  Some input tokens and some output tokens are "core"; the rest is filler.
  Removing a core input changes the output. A core output position has
  a peaked distribution; a filler position is diffuse.

This file just measures and prints. No decisions.

For every output position t and every mask S we record:
  log p(y_t | S)              -- prob of the originally-generated token
  log p(argmax | S)            -- prob of the winning token under S
  argmax token id under S
  top-K probs under S          -- shape of the distribution
  entropy of distribution      -- diffuseness

Then for each position t we print:
  - The full distribution of log p(y_t | S) across all k masks
    (min, p10, p25, median, p75, p90, max, mean, std).
  - Full-prompt: log p(y_t | full), top-K alternatives with probs, entropy.
  - Empty-prompt: same.
  - Each LOO row: which input was removed, log p(y_t | S), whether argmax
    flipped, and what it flipped to (if anything).
  - Sorted score for each input token i:
      delta_i = log p(y_t | full) - log p(y_t | LOO_i)
    (no threshold, just the number, sorted)

No filler/core/echo/etc. labels. You decide what the numbers mean.
"""

import math
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

MAX_NEW_TOKENS = 60
LN2 = math.log(2.0)

N_ABLATION_SAMPLES = 128
RNG_SEED = 0
TOP_K_ALTS = 8           # how many alternatives to print at full/empty


# ---------------------------------------------------------------------------
# Prompt structuring
# ---------------------------------------------------------------------------

def build_structured_prompt(tokenizer, parts):
    user_message = "".join(text for _, text in parts)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False, add_generation_prompt=True,
    )
    empty_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False, add_generation_prompt=True,
    )
    pre_len = 0
    while (pre_len < len(empty_text)
           and pre_len < len(full_text)
           and empty_text[pre_len] == full_text[pre_len]):
        pre_len += 1
    suf_len = 0
    while (suf_len < len(empty_text) - pre_len
           and suf_len < len(full_text) - pre_len
           and empty_text[-1 - suf_len] == full_text[-1 - suf_len]):
        suf_len += 1
    prefix_text = full_text[:pre_len]
    suffix_text = full_text[len(full_text) - suf_len:]
    assert prefix_text + user_message + suffix_text == full_text

    enc = tokenizer(full_text, return_offsets_mapping=True,
                    add_special_tokens=False)
    full_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    spans = [("__prefix__", 0, len(prefix_text))]
    cur = len(prefix_text)
    for name, text in parts:
        spans.append((name, cur, cur + len(text)))
        cur += len(text)
    spans.append(("__suffix__", cur, cur + len(suffix_text)))

    def span_of(off):
        a, b = off
        mid = (a + b) / 2.0
        for name, s, e in spans:
            if s <= mid < e or (mid == e and e == len(full_text)):
                return name
        return spans[-1][0]

    by_span = {name: [] for name, _, _ in spans}
    for tid, off in zip(full_ids, offsets):
        if off == (0, 0):
            by_span[spans[0][0]].append(tid)
            continue
        by_span[span_of(off)].append(tid)

    template_prefix_ids = torch.tensor(by_span["__prefix__"],
                                       dtype=torch.long, device=DEVICE)
    template_suffix_ids = torch.tensor(by_span["__suffix__"],
                                       dtype=torch.long, device=DEVICE)
    group_names = [name for name, _ in parts]
    group_token_ids = [torch.tensor(by_span[name], dtype=torch.long,
                                    device=DEVICE) for name in group_names]
    return (group_names, group_token_ids,
            template_prefix_ids, template_suffix_ids)


def get_stop_token_ids(tokenizer):
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for special in ["<|eot_id|>", "<|end_of_text|>"]:
        tid = tokenizer.convert_tokens_to_ids(special)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)
    return stop_ids


# ---------------------------------------------------------------------------
# Forward pass: extract full per-position info under one mask
# ---------------------------------------------------------------------------

def forward_under_mask(model, prompt_ids, answer_ids, top_k=TOP_K_ALTS):
    """
    Run one forward pass. Returns dict of per-position info, all in nats:
      lp_y      [T]    log p(y_t | S, y_<t)
      argmax    [T]    argmax token id at each position
      lp_argmax [T]    its log-prob
      entropy   [T]    H of distribution at each position
      topk_ids  [T,K]  top-K token ids
      topk_lp   [T,K]  their log-probs
    """
    full_ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    with torch.no_grad():
        logits = model(full_ids).logits[0]
    L = prompt_ids.numel()
    T = answer_ids.numel()
    pred = torch.arange(L - 1, L - 1 + T, device=DEVICE)
    log_probs = F.log_softmax(logits[pred], dim=-1)        # [T, V]
    probs = log_probs.exp()

    lp_y = log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)
    lp_argmax, argmax_id = log_probs.max(dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    topk_lp, topk_ids = log_probs.topk(top_k, dim=-1)
    return {
        "lp_y": lp_y,
        "argmax": argmax_id,
        "lp_argmax": lp_argmax,
        "entropy": entropy,
        "topk_ids": topk_ids,
        "topk_lp": topk_lp,
    }


def generate_answer(model, tokenizer, prompt_ids, stop_token_ids,
                    max_new_tokens=MAX_NEW_TOKENS):
    generated = []
    cur_ids = prompt_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(cur_ids.unsqueeze(0)).logits[0, -1]
            next_id = int(logits.argmax().item())
            if next_id in stop_token_ids:
                break
            generated.append(next_id)
            cur_ids = torch.cat([cur_ids,
                                 torch.tensor([next_id], device=DEVICE)])
    if not generated:
        return torch.tensor([], dtype=torch.long, device=DEVICE), ""
    a_ids = torch.tensor(generated, dtype=torch.long, device=DEVICE)
    return a_ids, tokenizer.decode(a_ids)


def assemble_from_mask(template_prefix_ids, template_suffix_ids,
                       flat_input_ids, mask):
    kept = flat_input_ids[mask.bool()]
    return torch.cat([template_prefix_ids, kept, template_suffix_ids])


# ---------------------------------------------------------------------------
# Mask designs
# ---------------------------------------------------------------------------

def stratified_masks(n_inputs, k, rng):
    """
    Row 0          : full
    Row 1          : empty
    Rows 2..2+n-1  : LOO (one input removed)
    Remaining      : random bernoulli with mixed keep-prob
    """
    masks = []
    masks.append([1] * n_inputs)
    masks.append([0] * n_inputs)
    for i in range(n_inputs):
        m = [1] * n_inputs
        m[i] = 0
        masks.append(m)
    remaining = max(0, k - len(masks))
    for _ in range(remaining):
        p = rng.choice([0.25, 0.5, 0.5, 0.75])
        masks.append([1 if rng.random() < p else 0 for _ in range(n_inputs)])
    masks = masks[:k] if len(masks) >= k else masks
    return torch.tensor(masks, dtype=torch.float32, device=DEVICE)


# ---------------------------------------------------------------------------
# Sweep: collect everything across masks
# ---------------------------------------------------------------------------

def sweep(model, template_prefix_ids, template_suffix_ids,
          flat_input_ids, answer_ids, masks, top_k=TOP_K_ALTS):
    """
    Returns:
      lp_y       [k, T]   log p(y_t | S)
      argmax     [k, T]   argmax token id
      lp_argmax  [k, T]   its log-prob
      entropy    [k, T]   entropy at each position
      topk_ids   [k, T, K]
      topk_lp    [k, T, K]
    """
    k = masks.shape[0]
    T = answer_ids.numel()
    lp_y = torch.zeros(k, T, device=DEVICE)
    argmax = torch.zeros(k, T, dtype=torch.long, device=DEVICE)
    lp_argmax = torch.zeros(k, T, device=DEVICE)
    entropy = torch.zeros(k, T, device=DEVICE)
    topk_ids = torch.zeros(k, T, top_k, dtype=torch.long, device=DEVICE)
    topk_lp = torch.zeros(k, T, top_k, device=DEVICE)

    for j in range(k):
        prompt_S = assemble_from_mask(template_prefix_ids,
                                       template_suffix_ids,
                                       flat_input_ids, masks[j])
        out = forward_under_mask(model, prompt_S, answer_ids, top_k=top_k)
        lp_y[j] = out["lp_y"]
        argmax[j] = out["argmax"]
        lp_argmax[j] = out["lp_argmax"]
        entropy[j] = out["entropy"]
        topk_ids[j] = out["topk_ids"]
        topk_lp[j] = out["topk_lp"]
    return lp_y, argmax, lp_argmax, entropy, topk_ids, topk_lp


# ---------------------------------------------------------------------------
# Reporting — pure dump, no thresholds
# ---------------------------------------------------------------------------

def _decode(tokenizer, ids):
    return [tokenizer.decode([t]) for t in ids.tolist()]


def _quantiles(x_bits):
    """x is a 1-D tensor in bits. Return min/p10/p25/med/p75/p90/max/mean/std."""
    x = x_bits.cpu().double().numpy()
    import numpy as np
    return {
        "min": float(np.min(x)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "med": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def explain_question(model, tokenizer, parts, stop_token_ids):
    user_message_preview = "".join(t for _, t in parts).replace("\n", " ")
    print("=" * 100)
    print(f"  USER: {user_message_preview}")
    print("=" * 100)

    (group_names, group_token_ids,
     tpl_pre, tpl_suf) = build_structured_prompt(tokenizer, parts)

    flat_input_ids = torch.cat(group_token_ids)
    group_name_per_input = []
    for gname, gids in zip(group_names, group_token_ids):
        group_name_per_input.extend([gname] * gids.numel())
    n_inputs = flat_input_ids.numel()

    full_prompt = torch.cat([tpl_pre, flat_input_ids, tpl_suf])
    a_ids, answer_text = generate_answer(model, tokenizer, full_prompt,
                                         stop_token_ids)
    if a_ids.numel() == 0:
        print("  Model produced no answer.")
        return
    print(f"  ASSISTANT: {answer_text!r}")
    print(f"  Inputs ({n_inputs} tokens):")
    in_tokens = _decode(tokenizer, flat_input_ids)
    for i, tok in enumerate(in_tokens):
        print(f"      [{i:>2}] {group_name_per_input[i]:<10} {tok!r}")
    out_tokens = _decode(tokenizer, a_ids)
    print(f"  Outputs ({a_ids.numel()} tokens):")
    for t, tok in enumerate(out_tokens):
        print(f"      [{t:>2}] {tok!r}")
    print()

    rng = random.Random(RNG_SEED)
    masks = stratified_masks(n_inputs, N_ABLATION_SAMPLES, rng)
    k = masks.shape[0]
    print(f"  Running {k} forward passes ...")
    lp_y, argmax, lp_argmax, entropy, topk_ids, topk_lp = sweep(
        model, tpl_pre, tpl_suf, flat_input_ids, a_ids, masks)
    print()

    # Indices into rows of `masks`
    full_idx = 0
    empty_idx = 1
    loo_start = 2
    loo_end = loo_start + n_inputs

    # Convert to bits for display
    lp_y_b = lp_y / LN2
    lp_argmax_b = lp_argmax / LN2
    entropy_b = entropy / LN2
    topk_lp_b = topk_lp / LN2

    # ----- per-output-position dump -----
    for t in range(a_ids.numel()):
        print("-" * 100)
        print(f"  OUTPUT POSITION t={t}  token={out_tokens[t]!r}  "
              f"(id={a_ids[t].item()})")
        print()

        # 1. Distribution of log p(y_t | S) across all k masks
        qs = _quantiles(lp_y_b[:, t])
        print(f"  log p(y_t | S) across k={k} masks   [bits]")
        print(f"      min={qs['min']:+.2f}  p10={qs['p10']:+.2f}  "
              f"p25={qs['p25']:+.2f}  med={qs['med']:+.2f}  "
              f"p75={qs['p75']:+.2f}  p90={qs['p90']:+.2f}  "
              f"max={qs['max']:+.2f}")
        print(f"      mean={qs['mean']:+.2f}  std={qs['std']:.2f}")
        print()

        # 2. Entropy distribution
        eqs = _quantiles(entropy_b[:, t])
        print(f"  H(distribution at t) across k masks   [bits]")
        print(f"      min={eqs['min']:.2f}  med={eqs['med']:.2f}  "
              f"max={eqs['max']:.2f}  mean={eqs['mean']:.2f}")
        print()

        # 3. Full-prompt details: top-K
        print(f"  Under FULL prompt at t={t}:")
        print(f"      log p(y_t)    = {lp_y_b[full_idx, t]:+.3f} bits  "
              f"(prob = {math.exp(lp_y[full_idx, t]):.4f})")
        print(f"      H             = {entropy_b[full_idx, t]:.3f} bits")
        print(f"      top-{TOP_K_ALTS} alternatives:")
        for r in range(TOP_K_ALTS):
            tid = int(topk_ids[full_idx, t, r].item())
            mark = "  <-- y_t" if tid == int(a_ids[t].item()) else ""
            print(f"          {tokenizer.decode([tid])!r:<16}  "
                  f"lp = {topk_lp_b[full_idx, t, r]:+.3f} b  "
                  f"p = {math.exp(topk_lp[full_idx, t, r]):.4f}{mark}")
        print()

        # 4. Empty-prompt details
        print(f"  Under EMPTY prompt at t={t}:")
        print(f"      log p(y_t)    = {lp_y_b[empty_idx, t]:+.3f} bits  "
              f"(prob = {math.exp(lp_y[empty_idx, t]):.4f})")
        print(f"      H             = {entropy_b[empty_idx, t]:.3f} bits")
        print(f"      argmax        = "
              f"{tokenizer.decode([int(argmax[empty_idx, t].item())])!r}  "
              f"lp = {lp_argmax_b[empty_idx, t]:+.3f} b")
        print()

        # 5. Per-input LOO sweep
        print(f"  Per-input LOO (sorted by delta = log p(y_t | full) - "
              f"log p(y_t | LOO_i)):")
        full_lp_b = lp_y_b[full_idx, t].item()
        deltas = []
        for i in range(n_inputs):
            row = loo_start + i
            lp_loo = lp_y_b[row, t].item()
            am_id = int(argmax[row, t].item())
            am_lp = lp_argmax_b[row, t].item()
            flipped = am_id != int(a_ids[t].item())
            deltas.append((i, full_lp_b - lp_loo, lp_loo, am_id, am_lp,
                           flipped))
        deltas.sort(key=lambda r: r[1], reverse=True)
        for i, delta, lp_loo, am_id, am_lp, flipped in deltas:
            flag = "  FLIPPED" if flipped else ""
            am_tok = tokenizer.decode([am_id])
            print(f"      remove [{i:>2}] {group_name_per_input[i]:<10}"
                  f"{in_tokens[i]!r:<14}  "
                  f"delta = {delta:+.3f} b   "
                  f"log p(y_t|LOO) = {lp_loo:+.3f} b   "
                  f"argmax = {am_tok!r} ({am_lp:+.2f} b){flag}")
        print()


# ---------------------------------------------------------------------------
# Question set
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE
    ).to(DEVICE).eval()
    stop_token_ids = get_stop_token_ids(tokenizer)

    questions = [
        [("passage",
          "Passage: The Compact of Yssaria was ratified in 1487 "
          "by the council of Mirentane.\n"),
         ("question",
          "Question: When was the Compact of Yssaria ratified? "
          "Answer in one short sentence.")],

        [("passage",
          "Passage: The Treaty of Tordesillas was signed in 1494, "
          "dividing the New World between Spain and Portugal.\n"),
         ("question",
          "Question: In what year was the Treaty of Tordesillas signed? "
          "Answer in one short sentence.")],

        [("question", "What is the capital of Vietnam?")],

        [("question", "What is the largest planet in our solar system?")],
    ]

    for q in questions:
        explain_question(model, tokenizer, q, stop_token_ids)


if __name__ == "__main__":
    main()