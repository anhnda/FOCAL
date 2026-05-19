"""
Generate-then-analyze interface for the readout-point method.
INSTRUCT MODEL VERSION — v4 (Owen-value SHAP attribution).

Story so far
------------
v1: D_prefix exploded on long answers, cumulative term inflated late tokens.
v2: Slot entropy fixed runaway-commas but collapsed on content tokens
    (entropy is measured at the prediction that produced the token —
    by then the model is already committed).
v3: Settled on Di_inst = log p(y_t | q, y_<t) - log p(y_t | y_<t) as the
    primary score, with slot_H as a soft tiebreaker:
        score = max(Di_inst, 0) * (1 + slot_H)
    This correctly ranked content tokens in the Yssaria / Tordesillas /
    Jupiter tests without flagging filler.

v4 (this file)
--------------
Di_inst is a *single scalar per answer token*: "how much did the entire
prompt help?" It can't tell us WHICH part of the prompt helped. For the
Tordesillas example, was it the year "1494" in the passage that drove
the model to emit "1494" in the answer, or the chat template, or the
word "Treaty" appearing earlier? We want per-input-group credit.

Exact Shapley over tokens is 2^n forward passes. Owen values are the
standard fix: partition inputs into G groups, do Shapley *between
groups* (2^G coalitions) and Shapley *within each group* (2^|g|).
For QA we use the natural partition:
    [chat_template_system]
    [passage]          (if present)
    [question]
    [trailing_template] (the "<|assistant|>" header part)
which is typically G=3 or G=4 — tractable.

The "off" state for a group is REMOVING those tokens from the prompt.
We keep generation fixed to the v3 greedy decode, then for each
already-generated token y_t we attribute log p(y_t | coalition, y_<t).
Owen group attribution φ_g satisfies the efficiency axiom:
    Σ_g φ_g(t)  =  log p(y_t | full prompt, y_<t)
                 - log p(y_t | empty prompt, y_<t)
The script prints the residual so you can verify.

Within-group (token-level) Shapley is optional and gated by
WITHIN_GROUP_SHAPLEY because |group| can be 20+ tokens. When enabled,
it runs only on the single highest-Di_inst answer token, on the group
the user picks (default: the one with the largest |φ_g|), and uses
sampling if the group is larger than EXACT_WITHIN_LIMIT.
"""

import math
import itertools
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

MAX_NEW_TOKENS = 60
LN2 = math.log(2.0)

# Owen / Shapley knobs
WITHIN_GROUP_SHAPLEY = True   # also attribute inside the winning group
EXACT_WITHIN_LIMIT = 12       # exact Shapley up to this group size; sample above
WITHIN_SHAPLEY_SAMPLES = 200  # permutation samples when group is too big
TOPK_TOKENS_TO_EXPLAIN = 3    # how many answer tokens to Owen-attribute


# ---------------------------------------------------------------------------
# Prompt construction. We now build the prompt as STRUCTURED PIECES so we can
# turn each piece on/off for Owen attribution.
# ---------------------------------------------------------------------------

def build_structured_prompt(tokenizer, parts):
    """
    parts: list of (group_name, text) pairs in the order they should appear
           in the user message. We always wrap them with the chat template.

    Returns:
      group_names:        list[str], length G
      group_token_ids:    list[Tensor], the token ids for each group's text
                          when tokenized in-context
      template_prefix_ids: Tensor, chat-template tokens BEFORE the user content
      template_suffix_ids: Tensor, chat-template tokens AFTER the user content
                           (i.e. the assistant header — this is what the model
                           reads right before generating)
      full_prompt_ids:    Tensor, the full prompt the model would see with all
                          groups present (== prefix + concat(groups) + suffix)
    """
    # Build full user message by concatenating part texts with single spaces /
    # newlines as appropriate. The caller controls separators inside `parts`.
    user_message = "".join(text for _, text in parts)

    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False, add_generation_prompt=True,
    )
    empty_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False, add_generation_prompt=True,
    )

    # Locate the user-content insertion point inside the template by diffing
    # the templated-with-content string against the templated-empty string.
    # We find the longest common prefix and suffix of full_text vs empty_text.
    # The middle of empty_text is "" (nothing between prefix and suffix), and
    # the middle of full_text is exactly user_message.
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
    # sanity: prefix + user_message + suffix == full_text
    assert prefix_text + user_message + suffix_text == full_text, \
        "Chat template split failed; check tokenizer template assumptions."

    # Tokenize each piece *in the context of the full prompt* using offset
    # mapping. This is the only way to get token boundaries that match how
    # the model actually sees the prompt (BPE merges respect neighbors).
    enc = tokenizer(full_text, return_offsets_mapping=True,
                    add_special_tokens=False)
    full_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # Compute character spans for prefix, each group, and suffix in full_text.
    spans = []
    spans.append(("__prefix__", 0, len(prefix_text)))
    cur = len(prefix_text)
    for name, text in parts:
        spans.append((name, cur, cur + len(text)))
        cur += len(text)
    spans.append(("__suffix__", cur, cur + len(suffix_text)))
    assert cur + len(suffix_text) == len(full_text)

    # Assign each token to whichever span contains its midpoint.
    def span_of(off):
        a, b = off
        mid = (a + b) / 2.0
        for name, s, e in spans:
            if s <= mid < e or (mid == e and e == len(full_text)):
                return name
        return spans[-1][0]

    by_span = {name: [] for name, _, _ in spans}
    for tid, off in zip(full_ids, offsets):
        if off == (0, 0):  # special tokens with no char span — keep in prefix
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
    full_prompt_ids = torch.tensor(full_ids, dtype=torch.long, device=DEVICE)

    return (group_names, group_token_ids,
            template_prefix_ids, template_suffix_ids, full_prompt_ids)


def assemble_prompt(template_prefix_ids, template_suffix_ids,
                    group_token_ids, mask):
    """Concatenate template + only those groups whose mask bit is 1."""
    pieces = [template_prefix_ids]
    for keep, g in zip(mask, group_token_ids):
        if keep:
            pieces.append(g)
    pieces.append(template_suffix_ids)
    return torch.cat(pieces)


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

def _entropy_bits(log_probs_row):
    p = log_probs_row.exp()
    nz = p > 0
    return -(p[nz] * log_probs_row[nz]).sum().item() / LN2


def generate_with_logprobs(model, tokenizer, prompt_ids,
                           stop_token_ids, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy decode from full prompt. Records full_lp and slot_H per token."""
    generated, full_lp_list, slot_H_list = [], [], []
    cur_ids = prompt_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(cur_ids.unsqueeze(0)).logits[0, -1]
            log_probs = F.log_softmax(logits, dim=-1)
            next_id = int(log_probs.argmax().item())
            if next_id in stop_token_ids:
                break
            generated.append(next_id)
            full_lp_list.append(log_probs[next_id].item())
            slot_H_list.append(_entropy_bits(log_probs))
            cur_ids = torch.cat([cur_ids,
                                 torch.tensor([next_id], device=DEVICE)])
    if not generated:
        empty_long = torch.tensor([], dtype=torch.long, device=DEVICE)
        empty_f = torch.tensor([], device=DEVICE)
        return empty_long, empty_f, empty_f, ""
    a_ids = torch.tensor(generated, dtype=torch.long, device=DEVICE)
    full_lp = torch.tensor(full_lp_list, device=DEVICE)
    slot_H = torch.tensor(slot_H_list, device=DEVICE)
    return a_ids, full_lp, slot_H, tokenizer.decode(a_ids)


def answer_logprobs_under(model, prompt_ids, answer_ids):
    """
    log p(y_t | prompt_ids, y_<t) for each t, as a Tensor of shape [T].
    Single forward pass over prompt_ids ++ answer_ids.
    """
    full_ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    with torch.no_grad():
        logits = model(full_ids).logits[0]
    L_prefix = prompt_ids.numel()
    pred_positions = torch.arange(L_prefix - 1,
                                  L_prefix - 1 + answer_ids.numel(),
                                  device=DEVICE)
    pred_logits = logits[pred_positions]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    return log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)


def prior_logprobs(model, tokenizer, answer_ids):
    """log p(y_t | y_<t) starting from BOS, for the v3 Di_inst baseline."""
    bos = torch.tensor([tokenizer.bos_token_id], device=DEVICE)
    return answer_logprobs_under(model, bos, answer_ids)


# ---------------------------------------------------------------------------
# Owen values: Shapley between groups
# ---------------------------------------------------------------------------

def _shapley_weight(coalition_size, total):
    """Standard Shapley weight |S|! * (n-|S|-1)! / n! ."""
    return (math.factorial(coalition_size)
            * math.factorial(total - coalition_size - 1)
            / math.factorial(total))


def owen_group_attribution(model, template_prefix_ids, template_suffix_ids,
                           group_token_ids, answer_ids):
    """
    Exact between-group Shapley. Returns:
        phi:   Tensor [G, T] — per (group, answer-token) attribution in nats
        empty_lp: Tensor [T] — log p(y_t | empty prompt, y_<t)
        full_lp:  Tensor [T] — log p(y_t | full prompt, y_<t)
    Efficiency: phi.sum(dim=0) ≈ full_lp - empty_lp.
    """
    G = len(group_token_ids)
    T = answer_ids.numel()
    n_coalitions = 2 ** G

    # Cache v(S) = log p(y | prompt(S), y_<t)  for every subset S of groups.
    # 2^G forward passes; each pass returns the full per-token vector.
    v = {}
    for bits in range(n_coalitions):
        mask = [(bits >> i) & 1 for i in range(G)]
        prompt_S = assemble_prompt(template_prefix_ids, template_suffix_ids,
                                   group_token_ids, mask)
        v[bits] = answer_logprobs_under(model, prompt_S, answer_ids)

    phi = torch.zeros(G, T, device=DEVICE)
    for i in range(G):
        for bits in range(n_coalitions):
            if (bits >> i) & 1:
                continue  # need coalitions WITHOUT i
            S_size = bin(bits).count("1")
            w = _shapley_weight(S_size, G)
            with_i = bits | (1 << i)
            phi[i] += w * (v[with_i] - v[bits])

    empty_lp = v[0]
    full_lp = v[n_coalitions - 1]
    return phi, empty_lp, full_lp


# ---------------------------------------------------------------------------
# Within-group Shapley (optional). Operates on a single answer token.
# ---------------------------------------------------------------------------

def within_group_shapley(model, template_prefix_ids, template_suffix_ids,
                          other_groups_ids, target_group_ids, answer_ids,
                          t_index):
    """
    Shapley over individual tokens inside one group, holding the rest of the
    prompt at "fully on". Returns phi_token: Tensor [|group|] in nats.
    For efficiency we evaluate only at the chosen answer position t_index,
    not all T positions.
    """
    n = target_group_ids.numel()
    target_list = target_group_ids.tolist()
    prefix_static = torch.cat([template_prefix_ids] + other_groups_ids[0])
    suffix_static = torch.cat(other_groups_ids[1] + [template_suffix_ids])

    def value_for_subset(subset_indices):
        kept = [target_list[i] for i in sorted(subset_indices)]
        kept_t = torch.tensor(kept, dtype=torch.long, device=DEVICE)
        prompt_S = torch.cat([prefix_static, kept_t, suffix_static])
        lp = answer_logprobs_under(model, prompt_S, answer_ids)
        return lp[t_index].item()

    phi = [0.0] * n

    if n <= EXACT_WITHIN_LIMIT:
        # Exact: 2^n subsets.
        all_idx = list(range(n))
        # Cache v(S)
        v = {}
        for r in range(n + 1):
            for combo in itertools.combinations(all_idx, r):
                v[frozenset(combo)] = value_for_subset(combo)
        for i in range(n):
            acc = 0.0
            for combo in v:
                if i in combo:
                    continue
                with_i = combo | {i}
                acc += _shapley_weight(len(combo), n) * (v[with_i] - v[combo])
            phi[i] = acc
    else:
        # Permutation sampling.
        rng = random.Random(0)
        for _ in range(WITHIN_SHAPLEY_SAMPLES):
            perm = list(range(n))
            rng.shuffle(perm)
            cur = []
            v_prev = value_for_subset(())
            for tok_idx in perm:
                cur.append(tok_idx)
                v_cur = value_for_subset(cur)
                phi[tok_idx] += (v_cur - v_prev) / WITHIN_SHAPLEY_SAMPLES
                v_prev = v_cur

    return torch.tensor(phi, device=DEVICE)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _decode_tokens(tokenizer, ids):
    return [tokenizer.decode([t]) for t in ids.tolist()]


def explain_question(model, tokenizer, parts, stop_token_ids):
    user_message_preview = "".join(t for _, t in parts).replace("\n", " ")
    print("=" * 100)
    print(f"  USER (groups={[n for n,_ in parts]}):")
    print(f"    {user_message_preview}")
    print("=" * 100)

    (group_names, group_token_ids,
     tpl_pre, tpl_suf, full_prompt_ids) = build_structured_prompt(
        tokenizer, parts)

    # Show how the prompt got split
    print("  Prompt split:")
    for name, gids in zip(group_names, group_token_ids):
        toks = _decode_tokens(tokenizer, gids)
        print(f"    [{name}] ({gids.numel()} toks): {''.join(toks)!r}")
    print()

    # 1) Greedy decode under the full prompt (this fixes y_<t for everything)
    a_ids, full_lp_gen, slot_H, answer_text = generate_with_logprobs(
        model, tokenizer, full_prompt_ids, stop_token_ids)
    if a_ids.numel() == 0:
        print("  Model produced no answer.")
        return
    print(f"  ASSISTANT: {answer_text!r}")
    print()

    # 2) v3 baseline: Di_inst against pure-LM prior
    pri_lp = prior_logprobs(model, tokenizer, a_ids)
    Di_inst_bits = (full_lp_gen - pri_lp) / LN2

    # 3) Owen between-group attribution
    phi_nats, empty_lp, full_lp_check = owen_group_attribution(
        model, tpl_pre, tpl_suf, group_token_ids, a_ids)
    phi_bits = phi_nats / LN2
    full_lp_bits = full_lp_check / LN2
    empty_lp_bits = empty_lp / LN2
    sum_phi_bits = phi_bits.sum(dim=0)
    residual_bits = full_lp_bits - empty_lp_bits - sum_phi_bits  # should be ~0

    tokens = _decode_tokens(tokenizer, a_ids)

    # Per-token table
    G = len(group_names)
    col_headers = (
        f"{'idx':>3}  {'token':<14}  "
        f"{'Di_inst':>8}  {'slotH':>6}  {'full_lp':>8}  {'empty_lp':>9}  "
        + "  ".join(f"phi[{n[:8]}]".rjust(13) for n in group_names)
        + "  " + f"{'Σφ':>8}  {'resid':>7}"
    )
    print(col_headers)
    print("-" * len(col_headers))
    for i, tok in enumerate(tokens):
        row = (
            f"{i:>3}  {repr(tok):<14}  "
            f"{Di_inst_bits[i].item():>8.3f}  "
            f"{slot_H[i].item():>6.3f}  "
            f"{full_lp_bits[i].item():>8.3f}  "
            f"{empty_lp_bits[i].item():>9.3f}  "
            + "  ".join(f"{phi_bits[g,i].item():>13.3f}" for g in range(G))
            + f"  {sum_phi_bits[i].item():>8.3f}"
            + f"  {residual_bits[i].item():>7.3f}"
        )
        print(row)
    print()
    print("  (All quantities in bits. resid = full_lp - empty_lp - Σφ "
          "should be ~0 by Shapley efficiency.)")
    print()

    # 4) Pick the top-K answer tokens by Di_inst and report which group
    #    explains them.
    k = min(TOPK_TOKENS_TO_EXPLAIN, len(tokens))
    top_t = torch.topk(Di_inst_bits, k=k).indices.tolist()
    print(f"  Top-{k} answer tokens by Di_inst — group attribution (bits):")
    for t in sorted(top_t):
        contribs = [(group_names[g], phi_bits[g, t].item()) for g in range(G)]
        contribs.sort(key=lambda x: -abs(x[1]))
        ranked = ", ".join(f"{n}={v:+.3f}" for n, v in contribs)
        print(f"    t={t:>2} {tokens[t]!r:<14} Di_inst={Di_inst_bits[t]:+.3f}"
              f"  ->  {ranked}")
    print()

    # 5) Optional within-group token-level Shapley on the single best token,
    #    inside its top-credit group.
    if WITHIN_GROUP_SHAPLEY and top_t:
        t_star = top_t[0]
        g_star = int(phi_bits[:, t_star].abs().argmax().item())
        target_group = group_token_ids[g_star]
        if target_group.numel() == 0:
            print(f"  (skip within-group: group {group_names[g_star]} is empty)")
            return
        # Partition the OTHER groups by whether they come before or after the
        # target group, so we can rebuild prompts correctly.
        before = group_token_ids[:g_star]
        after = group_token_ids[g_star+1:]
        phi_tok_nats = within_group_shapley(
            model, tpl_pre, tpl_suf,
            other_groups_ids=(before, after),
            target_group_ids=target_group,
            answer_ids=a_ids,
            t_index=t_star,
        )
        phi_tok_bits = phi_tok_nats / LN2
        group_toks = _decode_tokens(tokenizer, target_group)
        print(f"  Within-group Shapley on answer token "
              f"t={t_star} {tokens[t_star]!r} "
              f"inside group [{group_names[g_star]}] "
              f"({target_group.numel()} toks):")
        order = torch.argsort(phi_tok_bits.abs(), descending=True).tolist()
        for rank, i in enumerate(order[:8]):
            print(f"    {rank+1:>2}. {group_toks[i]!r:<14} "
                  f"phi={phi_tok_bits[i].item():+.4f} bits")
        print()


# ---------------------------------------------------------------------------
# Question set. Each question is now a LIST OF (group_name, text) pairs so
# Owen attribution has natural buckets to work with.
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE
    ).to(DEVICE).eval()
    stop_token_ids = get_stop_token_ids(tokenizer)

    questions = [
        # Simple one-group questions still work: G=1, Owen reduces to
        # "full vs empty" and phi == full_lp - empty_lp trivially.
        [("question", "What is the name of our planet?")],
        [("question", "What is the capital of Vietnam?")],
        [("question", "What is the largest planet in our solar system?")],

        # Passage QA: the interesting case for Owen — passage vs question.
        [("passage",
          "Passage: The Treaty of Tordesillas was signed in 1494, "
          "dividing the New World between Spain and Portugal.\n"),
         ("question",
          "Question: In what year was the Treaty of Tordesillas signed? "
          "Answer in one short sentence.")],

        [("question",
          "What is 23 plus 45? Answer with just the number.")],

        # Fictional passage QA. This is the diagnostic: the answer "1487"
        # cannot come from prior knowledge, so phi[passage] should
        # dominate phi[question] on the year token.
        [("passage",
          "Passage: The Compact of Yssaria was ratified in 1487 "
          "by the council of Mirentane.\n"),
         ("question",
          "Question: When was the Compact of Yssaria ratified? "
          "Answer in one short sentence.")],
    ]

    for q in questions:
        explain_question(model, tokenizer, q, stop_token_ids)


if __name__ == "__main__":
    main()