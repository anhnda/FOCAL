"""
Generate-then-analyze interface for the readout-point method.
v6 — lightweight ablation, joint output-importance + input-attribution.

Story so far
------------
v1: D_prefix exploded on long answers.
v2: slot_H collapsed on content tokens (entropy measured after commitment).
v3: Di_inst as primary score (full_lp - prior_lp).
v4: Group-level Owen φ, exact between groups, exact within groups (limited).
v5: Per-token Owen φ matrix via permutation sampling within each group.
    ~3500 forward passes per question, expensive.

v6 (this file)
--------------
Two key changes:

1. ONE set of forward passes serves BOTH analyses.
   For every sampled input mask S, the forward pass returns log p(y_t | S)
   for ALL output positions t simultaneously. From the resulting [k, T]
   matrix V we extract:
     - per-output-token importance (variance, mean-drop, LOO range)
     - per-input-token attribution (KernelSHAP regression on V)
   Cost: ~k forward passes total. Default k=128.

2. Mode A/B/Echo/Filler classifier on top of attribution.
   Each output token gets a label based on:
     - importance score (filter out filler)
     - whether answer string appears in any input group (echo vs retrieval)
     - φ-pattern shape (sharp vs diffuse)
     - which group the top-φ inputs belong to (context vs weight retrieval)

Caveat: KernelSHAP φ values are noisier than Owen permutation φ for the
same sample budget — they fit a linear model that misses higher-order
interactions. For top-k ranking this is usually fine; for exact φ
values, use v5.

The "lightweight" promise: ~128 forward passes vs v5's ~3500, with the
same downstream answers.
"""

import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

MAX_NEW_TOKENS = 60
LN2 = math.log(2.0)

# Sampling
N_ABLATION_SAMPLES = 128      # total k; the stratified mix below adds to this
RNG_SEED = 0

# Classifier thresholds (in bits)
IMPORTANCE_THRESHOLD = 1.0    # mean-drop above this -> not filler
PHI_CONCENTRATION_THRESHOLD = 0.4  # top-1 |phi| / sum |phi|; above -> sharp

TOP_INPUTS_PER_OUTPUT = 5


# ---------------------------------------------------------------------------
# Prompt structuring (same as v5)
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
# Forward-pass helpers
# ---------------------------------------------------------------------------

def answer_logprobs_under(model, prompt_ids, answer_ids):
    full_ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    with torch.no_grad():
        logits = model(full_ids).logits[0]
    L = prompt_ids.numel()
    pred = torch.arange(L - 1, L - 1 + answer_ids.numel(), device=DEVICE)
    log_probs = F.log_softmax(logits[pred], dim=-1)
    return log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)


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
    """Build prompt = prefix + kept input tokens + suffix."""
    kept = flat_input_ids[mask.bool()]
    return torch.cat([template_prefix_ids, kept, template_suffix_ids])


# ---------------------------------------------------------------------------
# Sampling scheme
# ---------------------------------------------------------------------------

def stratified_masks(n_inputs, k, rng):
    """
    Mix of mask designs:
      - 1 full-on (calibrates v_full)
      - 1 all-off (calibrates v_empty)
      - n_inputs leave-one-out (one per input token)
      - rest: bernoulli with varied keep-prob
    Returns Tensor[k, n_inputs] of 0/1.
    """
    masks = []
    masks.append([1] * n_inputs)                       # full
    masks.append([0] * n_inputs)                       # empty
    for i in range(n_inputs):                          # LOO
        m = [1] * n_inputs
        m[i] = 0
        masks.append(m)
    remaining = max(0, k - len(masks))
    for _ in range(remaining):
        p = rng.choice([0.25, 0.5, 0.5, 0.75])         # mostly p=0.5
        masks.append([1 if rng.random() < p else 0 for _ in range(n_inputs)])
    masks = masks[:k] if len(masks) >= k else masks    # never under-fill
    return torch.tensor(masks, dtype=torch.float32, device=DEVICE)


# ---------------------------------------------------------------------------
# Core: collect V matrix [k, T] of log p(y_t | S) for sampled masks S
# ---------------------------------------------------------------------------

def collect_ablation_matrix(model, template_prefix_ids, template_suffix_ids,
                             flat_input_ids, answer_ids, masks):
    """One forward pass per row of masks. Returns V: Tensor[k, T] in nats."""
    k = masks.shape[0]
    T = answer_ids.numel()
    V = torch.zeros(k, T, device=DEVICE)
    for j in range(k):
        prompt_S = assemble_from_mask(template_prefix_ids,
                                       template_suffix_ids,
                                       flat_input_ids, masks[j])
        V[j] = answer_logprobs_under(model, prompt_S, answer_ids)
    return V


# ---------------------------------------------------------------------------
# KernelSHAP regression — closed-form solve
# ---------------------------------------------------------------------------

def _kernel_shap_weights(masks):
    """Standard KernelSHAP weights pi(|S|) = (n-1) / [C(n, |S|) |S| (n-|S|)]."""
    k, n = masks.shape
    s = masks.sum(dim=1).long().cpu().numpy()         # coalition sizes
    w = np.zeros(k, dtype=np.float64)
    for j, sj in enumerate(s):
        if sj == 0 or sj == n:
            w[j] = 1e6                                # anchor full/empty
        else:
            denom = math.comb(n, sj) * sj * (n - sj)
            w[j] = (n - 1) / denom
    return torch.tensor(w, device=masks.device, dtype=torch.float32)


def kernel_shap_phi(masks, V):
    """
    Solve weighted least squares for each output column:
        phi[:, t] ≈ argmin_phi  Σ_j w_j ( V[j,t] - (intercept + masks[j] · phi) )^2
    Subject to efficiency: intercept + 1·phi = V[full], intercept = V[empty].
    Returns phi: Tensor[n_inputs, T] in nats.

    We solve it with the standard KernelSHAP closed form:
        Use augmented design [1, masks] and weighted normal equations.
    """
    k, n = masks.shape
    T = V.shape[1]
    w = _kernel_shap_weights(masks)             # [k]
    X = torch.cat([torch.ones(k, 1, device=masks.device), masks], dim=1)  # [k, n+1]
    Wx = X * w.unsqueeze(1)                     # [k, n+1]
    XtWX = X.t() @ Wx                           # [n+1, n+1]
    XtWX_inv = torch.linalg.pinv(XtWX)
    XtWV = X.t() @ (Wx[:, 0:1] * 0 + w.unsqueeze(1) * V)  # [n+1, T]
    beta = XtWX_inv @ XtWV                      # [n+1, T]
    phi = beta[1:]                              # [n, T]
    return phi


# ---------------------------------------------------------------------------
# Output-token importance scores derived from V
# ---------------------------------------------------------------------------

def importance_scores(V, masks):
    """
    Three scores per output token, all in bits.
      variance   : Var_S log p(y_t | S)
      mean_drop  : log p(y_t | full) - mean_S log p(y_t | S)
      loo_range  : max_S in LOO - min_S in LOO
    Returns dict of Tensor[T].
    """
    V_bits = V / LN2
    T = V_bits.shape[1]

    full_idx = 0      # by construction in stratified_masks
    empty_idx = 1
    loo_start = 2
    n_inputs = masks.shape[1]
    loo_end = loo_start + n_inputs

    variance = V_bits.var(dim=0)
    mean_drop = V_bits[full_idx] - V_bits.mean(dim=0)
    loo_block = V_bits[loo_start:loo_end]       # [n_inputs, T]
    loo_range = loo_block.max(dim=0).values - loo_block.min(dim=0).values

    return {
        "variance": variance,
        "mean_drop": mean_drop,
        "loo_range": loo_range,
        "full_lp_bits": V_bits[full_idx],
        "empty_lp_bits": V_bits[empty_idx],
    }


# ---------------------------------------------------------------------------
# Mode A / B / Echo / Filler classifier
# ---------------------------------------------------------------------------

def classify_output_token(t, out_tokens, importance, phi_bits,
                           group_name_per_input, group_names, parts_text):
    """Return one of: 'context_retrieval', 'weight_retrieval', 'echo',
    'filler', based on importance and phi pattern."""
    imp = importance["mean_drop"][t].item()
    if imp < IMPORTANCE_THRESHOLD:
        return "filler", {}

    # Where does the answer token's text appear in inputs?
    tok_text = out_tokens[t].strip()
    appears_in = set()
    if tok_text:
        for gname, text in parts_text.items():
            if tok_text and tok_text in text:
                appears_in.add(gname)

    # phi concentration: top-1 |phi| relative to sum
    abs_phi = phi_bits[:, t].abs()
    total = abs_phi.sum().item()
    if total == 0:
        concentration = 0.0
        top_input_group = None
    else:
        top_idx = int(abs_phi.argmax().item())
        concentration = abs_phi[top_idx].item() / total
        top_input_group = group_name_per_input[top_idx]

    info = {
        "imp_bits": imp,
        "concentration": concentration,
        "top_input_group": top_input_group,
        "appears_in_groups": sorted(appears_in),
    }

    has_question_group = "question" in group_names
    non_question_groups = [g for g in group_names if g != "question"]

    # Echo: token literally appears in both question and another group AND
    # phi splits roughly evenly between them.
    if has_question_group and non_question_groups:
        if "question" in appears_in and any(g in appears_in
                                            for g in non_question_groups):
            # Check phi symmetry between question and the other groups
            q_sum = sum(abs_phi[i].item()
                        for i, g in enumerate(group_name_per_input)
                        if g == "question")
            other_sum = sum(abs_phi[i].item()
                            for i, g in enumerate(group_name_per_input)
                            if g in non_question_groups)
            if min(q_sum, other_sum) / max(q_sum, other_sum + 1e-9) > 0.6:
                return "echo", info

    # Context retrieval: top phi is in a non-question group, sharp.
    if (top_input_group in non_question_groups
            and concentration > PHI_CONCENTRATION_THRESHOLD):
        return "context_retrieval", info

    # Weight retrieval: top phi is in question group, sharp, and the answer
    # token doesn't appear verbatim in any input.
    if (top_input_group == "question"
            and concentration > PHI_CONCENTRATION_THRESHOLD
            and not appears_in):
        return "weight_retrieval", info

    # Weight retrieval with multiple keying tokens: not concentrated on one,
    # but content-word concentration in the question group.
    if top_input_group == "question" and not appears_in:
        return "weight_retrieval", info

    # Fallback: high importance, ambiguous pattern.
    return "ambiguous", info


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _decode(tokenizer, ids):
    return [tokenizer.decode([t]) for t in ids.tolist()]


def explain_question(model, tokenizer, parts, stop_token_ids):
    user_message_preview = "".join(t for _, t in parts).replace("\n", " ")
    print("=" * 100)
    print(f"  USER: {user_message_preview}")
    print("=" * 100)

    (group_names, group_token_ids,
     tpl_pre, tpl_suf) = build_structured_prompt(tokenizer, parts)
    parts_text = {name: text for name, text in parts}

    # Flatten input tokens and remember group membership per input position.
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
    print(f"  Inputs: {n_inputs} tokens across groups {group_names}")
    print()

    rng = random.Random(RNG_SEED)
    masks = stratified_masks(n_inputs, N_ABLATION_SAMPLES, rng)
    print(f"  Running {masks.shape[0]} ablation forward passes ...")
    V = collect_ablation_matrix(model, tpl_pre, tpl_suf,
                                 flat_input_ids, a_ids, masks)

    importance = importance_scores(V, masks)
    phi = kernel_shap_phi(masks, V)                  # nats, [n_inputs, T]
    phi_bits = phi / LN2

    out_tokens = _decode(tokenizer, a_ids)
    in_tokens = _decode(tokenizer, flat_input_ids)

    # ----- per-output-token classification + top-K input attribution -----
    print()
    header = (f"{'t':>3}  {'token':<14}  "
              f"{'imp':>7}  {'var':>6}  {'LOO':>6}  {'conc':>5}  "
              f"{'label':<18}")
    print(header)
    print("-" * len(header))
    classifications = []
    for t, tok in enumerate(out_tokens):
        label, info = classify_output_token(
            t, out_tokens, importance, phi_bits,
            group_name_per_input, group_names, parts_text)
        classifications.append((t, label, info))
        print(f"{t:>3}  {repr(tok):<14}  "
              f"{importance['mean_drop'][t].item():>7.3f}  "
              f"{importance['variance'][t].item():>6.3f}  "
              f"{importance['loo_range'][t].item():>6.3f}  "
              f"{info.get('concentration', 0.0):>5.2f}  "
              f"{label:<18}")
    print()

    # ----- detailed attribution for non-filler tokens only -----
    print(f"  Detailed input attribution (top {TOP_INPUTS_PER_OUTPUT}) for "
          f"non-filler output tokens:")
    print()
    for t, label, info in classifications:
        if label == "filler":
            continue
        col = phi_bits[:, t]
        order = torch.argsort(col.abs(), descending=True).tolist()
        top = order[:TOP_INPUTS_PER_OUTPUT]
        print(f"  y[{t:>2}] = {out_tokens[t]!r:<14}  [{label}]  "
              f"imp = {info['imp_bits']:+.2f} bits")
        for i in top:
            print(f"      {group_name_per_input[i]:<10}[{i:>2}] "
                  f"{in_tokens[i]!r:<14}  phi = {col[i].item():+.4f}")
        print()


# ---------------------------------------------------------------------------
# Question set (same as v5)
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE
    ).to(DEVICE).eval()
    stop_token_ids = get_stop_token_ids(tokenizer)

    questions = [
        # Mode A: fictional passage QA. Answer "1487" must come from passage.
        [("passage",
          "Passage: The Compact of Yssaria was ratified in 1487 "
          "by the council of Mirentane.\n"),
         ("question",
          "Question: When was the Compact of Yssaria ratified? "
          "Answer in one short sentence.")],

        # Mode A again, real history.
        [("passage",
          "Passage: The Treaty of Tordesillas was signed in 1494, "
          "dividing the New World between Spain and Portugal.\n"),
         ("question",
          "Question: In what year was the Treaty of Tordesillas signed? "
          "Answer in one short sentence.")],

        # Mode B: closed-book, answer from weights.
        [("question", "What is the capital of Vietnam?")],

        # Mode B: closed-book.
        [("question", "What is the largest planet in our solar system?")],
    ]

    for q in questions:
        explain_question(model, tokenizer, q, stop_token_ids)


if __name__ == "__main__":
    main()