"""
Generate-then-analyze interface for the readout-point method.
v7 — two-sided importance: input-side phi + output-side margin/flip.

Story so far
------------
v6: one set of ~128 ablation forward passes gives us V[k, T] =
    log p(y_t | S) for all sampled masks S and all output positions t.
    From V we derive per-output-token importance (variance, mean_drop,
    loo_range) and per-input-token attribution (KernelSHAP phi).
    Hypothesis check was one-sided: only input-side ablation effects
    were measured.

v7 (this file)
--------------
The hypothesis is two-sided:
  - INPUT side: critical input kept => log p(y_t | S) stays high;
                critical input removed => it crashes.
  - OUTPUT side: at a critical output position, y_t is the argmax with
                 a large margin over runner-up; at filler positions,
                 many tokens are nearly tied.

v6 only saw the input side. v7 adds the output side by tracking, for
every (mask S, position t):
  - top-1 alternative token y'_{j,t} = argmax_{y != y_t} log p(y | S)
  - its log-prob
  - whether the argmax flipped away from y_t under this mask

This is essentially free: we already do the forward passes for V; we
just need to keep the runner-up logits per position instead of only
gathering y_t's value.

Position convention: we score at the CURRENT position t of the fixed
teacher-forced answer a_ids. This means later positions benefit from
the committed prefix a_ids[:t] — if critical information already
"baked in" at an earlier position, the argmax at t may not flip even
when the critical input is ablated. This is a feature, not a bug: it
tells us WHERE in the output the commitment happened. Multi-token
answers ("1487") should show the action concentrated in early
positions.

Outputs per output token t:
  imp_bits       : log p(y_t | full) - mean_S log p(y_t | S)   [v6]
  variance       : Var_S log p(y_t | S)                         [v6]
  loo_range      : range over LOO masks                          [v6]
  margin_full    : log p(y_t | full) - log p(y'^{full} | full)
  margin_mean    : mean_S [ log p(y_t | S) - log p(y'^S | S) ]
  flip_rate      : fraction of masks where argmax(log p|S) != y_t
  flip_alts      : set of alternative tokens that win when flipped
"""

import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

MAX_NEW_TOKENS = 60
LN2 = math.log(2.0)

N_ABLATION_SAMPLES = 128
RNG_SEED = 0

IMPORTANCE_THRESHOLD = 1.0
PHI_CONCENTRATION_THRESHOLD = 0.4
FLIP_RATE_CRITICAL = 0.10           # >=10% of masks flip => committed-late
MARGIN_FULL_CONFIDENT = 1.0         # bits; full-prompt margin to count as committed

TOP_INPUTS_PER_OUTPUT = 5
TOP_FLIP_ALTS = 3                   # how many alt tokens to show per position


# ---------------------------------------------------------------------------
# Prompt structuring (unchanged from v6)
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

def answer_logprobs_and_runnerup(model, prompt_ids, answer_ids):
    """
    Returns:
      lp_y     : Tensor[T]  log p(y_t | S, y_<t)
      lp_alt   : Tensor[T]  log p(argmax_{y != y_t} | S, y_<t)
      alt_id   : Tensor[T]  the alt token id at each position
      argmax_id: Tensor[T]  the overall argmax at each position (== y_t or alt)
    All in nats.
    """
    full_ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    with torch.no_grad():
        logits = model(full_ids).logits[0]
    L = prompt_ids.numel()
    T = answer_ids.numel()
    pred = torch.arange(L - 1, L - 1 + T, device=DEVICE)
    log_probs = F.log_softmax(logits[pred], dim=-1)        # [T, V]

    # log p(y_t)
    lp_y = log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)

    # Overall argmax per row
    argmax_id = log_probs.argmax(dim=-1)

    # Runner-up among y != y_t: mask out y_t and take argmax
    masked = log_probs.clone()
    masked.scatter_(1, answer_ids.unsqueeze(1), float("-inf"))
    alt_lp, alt_id = masked.max(dim=-1)

    return lp_y, alt_lp, alt_id, argmax_id


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
# Sampling scheme (unchanged from v6)
# ---------------------------------------------------------------------------

def stratified_masks(n_inputs, k, rng):
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
# Core: V matrix [k, T] of log p(y_t | S), plus alt tracking
# ---------------------------------------------------------------------------

def collect_ablation_matrices(model, template_prefix_ids, template_suffix_ids,
                               flat_input_ids, answer_ids, masks):
    """
    One forward pass per mask. Returns:
      V        : Tensor[k, T]  log p(y_t | S)         in nats
      V_alt    : Tensor[k, T]  log p(alt_{j,t} | S)   in nats
      alt_ids  : LongTensor[k, T]  alt token at each (j, t)
      flipped  : BoolTensor[k, T]  True iff argmax under S_j at pos t != y_t
    """
    k = masks.shape[0]
    T = answer_ids.numel()
    V = torch.zeros(k, T, device=DEVICE)
    V_alt = torch.zeros(k, T, device=DEVICE)
    alt_ids = torch.zeros(k, T, dtype=torch.long, device=DEVICE)
    flipped = torch.zeros(k, T, dtype=torch.bool, device=DEVICE)

    for j in range(k):
        prompt_S = assemble_from_mask(template_prefix_ids,
                                       template_suffix_ids,
                                       flat_input_ids, masks[j])
        lp_y, lp_alt, alt_id, argmax_id = answer_logprobs_and_runnerup(
            model, prompt_S, answer_ids)
        V[j] = lp_y
        V_alt[j] = lp_alt
        alt_ids[j] = alt_id
        flipped[j] = argmax_id != answer_ids
    return V, V_alt, alt_ids, flipped


# ---------------------------------------------------------------------------
# KernelSHAP regression (unchanged from v6)
# ---------------------------------------------------------------------------

def _kernel_shap_weights(masks):
    k, n = masks.shape
    s = masks.sum(dim=1).long().cpu().numpy()
    w = np.zeros(k, dtype=np.float64)
    for j, sj in enumerate(s):
        if sj == 0 or sj == n:
            w[j] = 1e6
        else:
            denom = math.comb(n, sj) * sj * (n - sj)
            w[j] = (n - 1) / denom
    return torch.tensor(w, device=masks.device, dtype=torch.float32)


def kernel_shap_phi(masks, V):
    k, n = masks.shape
    T = V.shape[1]
    w = _kernel_shap_weights(masks)
    X = torch.cat([torch.ones(k, 1, device=masks.device), masks], dim=1)
    Wx = X * w.unsqueeze(1)
    XtWX = X.t() @ Wx
    XtWX_inv = torch.linalg.pinv(XtWX)
    XtWV = X.t() @ (w.unsqueeze(1) * V)
    beta = XtWX_inv @ XtWV
    phi = beta[1:]
    return phi


# ---------------------------------------------------------------------------
# Importance scores (v6 + new output-side ones)
# ---------------------------------------------------------------------------

def importance_scores(V, V_alt, flipped, masks):
    """
    Per-output-token scores. Bits throughout.

    Input-side (from v6):
      variance, mean_drop, loo_range, full_lp_bits, empty_lp_bits
    Output-side (new):
      margin_full   : log p(y_t | full) - log p(alt | full)
      margin_mean   : mean_S [ log p(y_t | S) - log p(alt | S) ]
      margin_min    : min_S of the same                       (worst-case)
      flip_rate     : fraction of masks where argmax != y_t
    """
    V_bits = V / LN2
    V_alt_bits = V_alt / LN2
    margin_bits = V_bits - V_alt_bits                   # [k, T]

    full_idx = 0
    empty_idx = 1
    loo_start = 2
    n_inputs = masks.shape[1]
    loo_end = loo_start + n_inputs

    variance = V_bits.var(dim=0)
    mean_drop = V_bits[full_idx] - V_bits.mean(dim=0)
    loo_block = V_bits[loo_start:loo_end]
    loo_range = loo_block.max(dim=0).values - loo_block.min(dim=0).values

    margin_full = margin_bits[full_idx]
    margin_mean = margin_bits.mean(dim=0)
    margin_min = margin_bits.min(dim=0).values
    flip_rate = flipped.float().mean(dim=0)

    return {
        "variance": variance,
        "mean_drop": mean_drop,
        "loo_range": loo_range,
        "full_lp_bits": V_bits[full_idx],
        "empty_lp_bits": V_bits[empty_idx],
        "margin_full": margin_full,
        "margin_mean": margin_mean,
        "margin_min": margin_min,
        "flip_rate": flip_rate,
    }


def flip_alternatives(alt_ids, flipped, argmax_under_S, t, tokenizer):
    """
    For position t, find which tokens beat y_t under ablation, and how often.
    argmax_under_S[j, t] is the winning token id under mask j; flipped[j, t]
    says whether it differs from y_t. Returns Counter of top alt token texts.
    """
    flips = argmax_under_S[flipped[:, t], t].tolist()
    cnt = Counter(flips)
    decoded = Counter()
    for tid, c in cnt.items():
        decoded[tokenizer.decode([tid])] = c
    return decoded


# ---------------------------------------------------------------------------
# Mode A / B / Echo / Filler classifier (v7 refinement)
# ---------------------------------------------------------------------------

def classify_output_token(t, out_tokens, importance, phi_bits,
                           group_name_per_input, group_names, parts_text):
    """
    Augmented with output-side signals.

    Categories:
      - filler             : low input-side importance AND low output-side margin
      - committed_late     : low flip_rate BUT high mean_drop
                              (info was baked in by earlier output tokens;
                              this position is determined by the prefix)
      - context_retrieval  : flips under ablation, top-phi is in non-question group
      - weight_retrieval   : flips under ablation, top-phi is in question group
                              and answer not literally in inputs
      - echo               : flips under ablation, answer appears in BOTH question
                              and another group; phi split symmetrically
      - ambiguous          : high importance but pattern doesn't match the above
    """
    imp = importance["mean_drop"][t].item()
    margin_full = importance["margin_full"][t].item()
    flip_rate = importance["flip_rate"][t].item()

    # Filler: input ablation doesn't move it AND model wasn't very committed
    if imp < IMPORTANCE_THRESHOLD and margin_full < MARGIN_FULL_CONFIDENT:
        return "filler", {"imp_bits": imp, "margin_full": margin_full,
                          "flip_rate": flip_rate}

    # phi concentration
    abs_phi = phi_bits[:, t].abs()
    total = abs_phi.sum().item()
    if total == 0:
        concentration = 0.0
        top_input_group = None
    else:
        top_idx = int(abs_phi.argmax().item())
        concentration = abs_phi[top_idx].item() / total
        top_input_group = group_name_per_input[top_idx]

    # Where does this output token's text appear in inputs?
    tok_text = out_tokens[t].strip()
    appears_in = set()
    if tok_text:
        for gname, text in parts_text.items():
            if tok_text in text:
                appears_in.add(gname)

    info = {
        "imp_bits": imp,
        "margin_full": margin_full,
        "flip_rate": flip_rate,
        "concentration": concentration,
        "top_input_group": top_input_group,
        "appears_in_groups": sorted(appears_in),
    }

    # Committed-late: high mean_drop but argmax doesn't flip. Information was
    # already nailed down by the prefix a_ids[:t]; ablating inputs lowers
    # log p(y_t) but doesn't change the winner.
    if imp >= IMPORTANCE_THRESHOLD and flip_rate < FLIP_RATE_CRITICAL:
        return "committed_late", info

    has_question_group = "question" in group_names
    non_question_groups = [g for g in group_names if g != "question"]

    if has_question_group and non_question_groups:
        if "question" in appears_in and any(g in appears_in
                                            for g in non_question_groups):
            q_sum = sum(abs_phi[i].item()
                        for i, g in enumerate(group_name_per_input)
                        if g == "question")
            other_sum = sum(abs_phi[i].item()
                            for i, g in enumerate(group_name_per_input)
                            if g in non_question_groups)
            if min(q_sum, other_sum) / max(q_sum, other_sum + 1e-9) > 0.6:
                return "echo", info

    if (top_input_group in non_question_groups
            and concentration > PHI_CONCENTRATION_THRESHOLD):
        return "context_retrieval", info

    if (top_input_group == "question"
            and concentration > PHI_CONCENTRATION_THRESHOLD
            and not appears_in):
        return "weight_retrieval", info

    if top_input_group == "question" and not appears_in:
        return "weight_retrieval", info

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
    V, V_alt, alt_ids, flipped = collect_ablation_matrices(
        model, tpl_pre, tpl_suf, flat_input_ids, a_ids, masks)

    # argmax under each S: y_t when not flipped, alt_id when flipped
    argmax_under_S = torch.where(flipped, alt_ids,
                                  a_ids.unsqueeze(0).expand_as(alt_ids))

    importance = importance_scores(V, V_alt, flipped, masks)
    phi = kernel_shap_phi(masks, V)
    phi_bits = phi / LN2

    out_tokens = _decode(tokenizer, a_ids)
    in_tokens = _decode(tokenizer, flat_input_ids)

    # ----- per-output-token table -----
    print()
    header = (f"{'t':>3}  {'token':<14}  "
              f"{'imp':>6}  {'mFull':>6}  {'mMin':>6}  {'flip%':>6}  "
              f"{'conc':>5}  {'label':<18}")
    print(header)
    print("-" * len(header))
    classifications = []
    for t, tok in enumerate(out_tokens):
        label, info = classify_output_token(
            t, out_tokens, importance, phi_bits,
            group_name_per_input, group_names, parts_text)
        classifications.append((t, label, info))
        print(f"{t:>3}  {repr(tok):<14}  "
              f"{importance['mean_drop'][t].item():>6.2f}  "
              f"{importance['margin_full'][t].item():>6.2f}  "
              f"{importance['margin_min'][t].item():>6.2f}  "
              f"{importance['flip_rate'][t].item() * 100:>5.1f}%  "
              f"{info.get('concentration', 0.0):>5.2f}  "
              f"{label:<18}")
    print()
    print("  Legend: imp = mean log-prob drop in bits;   mFull = full-prompt "
          "argmax margin in bits;")
    print("          mMin = worst-case margin across masks (negative => some "
          "ablation flips it);")
    print("          flip% = fraction of ablations where argmax != y_t at "
          "this position;")
    print("          conc = top-1 |phi| / sum |phi| over input tokens.")
    print()

    # ----- detailed attribution for non-filler tokens -----
    print(f"  Detailed input attribution (top {TOP_INPUTS_PER_OUTPUT}) "
          f"and flip alternatives (top {TOP_FLIP_ALTS}):")
    print()
    for t, label, info in classifications:
        if label == "filler":
            continue
        col = phi_bits[:, t]
        order = torch.argsort(col.abs(), descending=True).tolist()
        top = order[:TOP_INPUTS_PER_OUTPUT]
        print(f"  y[{t:>2}] = {out_tokens[t]!r:<14}  [{label}]  "
              f"imp={info['imp_bits']:+.2f}b  "
              f"mFull={info['margin_full']:+.2f}b  "
              f"flip={info['flip_rate'] * 100:.1f}%")
        for i in top:
            print(f"      input  {group_name_per_input[i]:<10}[{i:>2}] "
                  f"{in_tokens[i]!r:<14}  phi = {col[i].item():+.4f}")
        # Flip alternatives at this position
        if info["flip_rate"] > 0:
            alts = flip_alternatives(alt_ids, flipped, argmax_under_S,
                                     t, tokenizer)
            top_alts = alts.most_common(TOP_FLIP_ALTS)
            alt_strs = ", ".join(f"{txt!r}x{c}" for txt, c in top_alts)
            print(f"      flips_to: {alt_strs}")
        print()


# ---------------------------------------------------------------------------
# Question set (same as v5/v6)
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