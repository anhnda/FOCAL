"""
Generate-then-analyze interface for the readout-point method.
v5 — full input-token -> output-token SHAP/Owen attribution matrix.

Difference from v4
------------------
v4 produced per-output-token *group-level* attribution (phi[passage],
phi[question]) and only ran token-level Shapley on a single chosen
output token. v5 produces the full matrix: for every output token y_t,
the SHAP value of every input token x_i.

How
---
We keep the v4 Owen structure:
  - Exact Shapley between groups (2^G forward passes, G is small).
  - Permutation-sampled Shapley within each group.

The trick that makes this tractable: every forward pass over
`prompt(S) ++ answer` returns log p(y_t | S) for ALL output positions t
in a single shot (teacher-forced). So one permutation sample yields
marginal contributions for every (input_token, output_token) pair
simultaneously. Cost is k * (n_input + 1) forward passes for the whole
matrix, not per output token.

Between-group Owen weight: when we sample a permutation of the tokens
inside group g, we evaluate v(S) at "all-of-other-groups present" —
this is the standard Owen-within-group definition and matches the
between-group phi we computed in v4 (efficiency holds at the group
level, not the token level, when within is sampled — but sums are
unbiased).

Output per question
-------------------
For each output token y_t, a sorted list of input tokens by |phi|, with
sign, showing which input tokens pushed log p(y_t) up or down.
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

# Sampling knobs
N_PERMUTATIONS = 80          # per group; raise for lower variance
TOP_INPUTS_PER_OUTPUT = 6    # how many input tokens to list per output token
RNG_SEED = 0


# ---------------------------------------------------------------------------
# Prompt construction (unchanged from v4)
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
    """log p(y_t | prompt_ids, y_<t) for all t. Tensor [T]."""
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


def assemble_from_pieces(template_prefix_ids, template_suffix_ids,
                          ordered_group_pieces):
    """ordered_group_pieces is a list of Tensors (some may be empty)."""
    return torch.cat([template_prefix_ids, *ordered_group_pieces,
                      template_suffix_ids])


# ---------------------------------------------------------------------------
# Within-group permutation Shapley, shared across all output positions
# ---------------------------------------------------------------------------

def within_group_shapley_matrix(model, template_prefix_ids,
                                 template_suffix_ids,
                                 groups_before, target_group, groups_after,
                                 answer_ids, n_permutations, rng):
    """
    Sample-based Shapley over tokens of `target_group`, returning a
    [n_target, T_out] matrix in nats. Other groups are held fully present.

    One permutation = (n_target + 1) forward passes; each pass yields a
    log-prob vector of length T_out, so all output positions get sampled
    in lockstep.
    """
    n = target_group.numel()
    T_out = answer_ids.numel()
    if n == 0:
        return torch.zeros(0, T_out, device=DEVICE)

    target_list = target_group.tolist()
    phi = torch.zeros(n, T_out, device=DEVICE)

    for _ in range(n_permutations):
        perm = list(range(n))
        rng.shuffle(perm)

        # Start with empty target group, build up.
        kept = []
        kept_tensor = torch.tensor(kept, dtype=torch.long, device=DEVICE)
        prompt_S = assemble_from_pieces(
            template_prefix_ids, template_suffix_ids,
            groups_before + [kept_tensor] + groups_after)
        v_prev = answer_logprobs_under(model, prompt_S, answer_ids)

        for tok_idx in perm:
            kept.append(target_list[tok_idx])
            # Preserve original-token-order within the target group when
            # rebuilding the prompt — sort by the original index.
            indices_kept = sorted(perm[:perm.index(tok_idx) + 1])
            kept_ordered = [target_list[i] for i in indices_kept]
            kept_tensor = torch.tensor(kept_ordered, dtype=torch.long,
                                        device=DEVICE)
            prompt_S = assemble_from_pieces(
                template_prefix_ids, template_suffix_ids,
                groups_before + [kept_tensor] + groups_after)
            v_cur = answer_logprobs_under(model, prompt_S, answer_ids)
            phi[tok_idx] += (v_cur - v_prev) / n_permutations
            v_prev = v_cur

    return phi


def owen_full_matrix(model, template_prefix_ids, template_suffix_ids,
                     group_token_ids, answer_ids,
                     n_permutations=N_PERMUTATIONS):
    """
    Returns phi_all: Tensor [n_input_total, T_out] in nats, and
    flat_input_ids: Tensor [n_input_total] giving the input token id for
    each row, plus group_slices: list of (group_name, slice) for display.
    """
    rng = random.Random(RNG_SEED)
    pieces = []
    flat_ids = []
    group_slices = []
    cursor = 0
    G = len(group_token_ids)

    for g in range(G):
        before = group_token_ids[:g]
        after = group_token_ids[g + 1:]
        phi_g = within_group_shapley_matrix(
            model, template_prefix_ids, template_suffix_ids,
            before, group_token_ids[g], after,
            answer_ids, n_permutations, rng,
        )
        pieces.append(phi_g)
        flat_ids.append(group_token_ids[g])
        group_slices.append((g, slice(cursor, cursor + phi_g.shape[0])))
        cursor += phi_g.shape[0]

    phi_all = torch.cat(pieces, dim=0) if pieces else torch.zeros(
        0, answer_ids.numel(), device=DEVICE)
    flat_input_ids = torch.cat(flat_ids) if flat_ids else torch.tensor(
        [], dtype=torch.long, device=DEVICE)
    return phi_all, flat_input_ids, group_slices


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

    full_prompt = assemble_from_pieces(tpl_pre, tpl_suf, group_token_ids)
    a_ids, answer_text = generate_answer(model, tokenizer, full_prompt,
                                         stop_token_ids)
    if a_ids.numel() == 0:
        print("  Model produced no answer.")
        return
    print(f"  ASSISTANT: {answer_text!r}")
    print()

    phi_all, flat_ids, group_slices = owen_full_matrix(
        model, tpl_pre, tpl_suf, group_token_ids, a_ids,
        n_permutations=N_PERMUTATIONS,
    )
    phi_bits = phi_all / LN2  # [n_input, T_out]

    in_tokens = _decode(tokenizer, flat_ids)
    out_tokens = _decode(tokenizer, a_ids)

    # Build (group_name, position_in_group) labels for each input row
    row_labels = []
    for (g_idx, sl) in group_slices:
        gname = group_names[g_idx]
        for local_i, tok in enumerate(in_tokens[sl]):
            row_labels.append(f"{gname}[{local_i:>2}] {tok!r}")

    print(f"  Input-token contributions per output token "
          f"(top {TOP_INPUTS_PER_OUTPUT}, bits, sampled with "
          f"{N_PERMUTATIONS} permutations per group):")
    print()

    for t, otok in enumerate(out_tokens):
        col = phi_bits[:, t]
        order = torch.argsort(col.abs(), descending=True).tolist()
        top = order[:TOP_INPUTS_PER_OUTPUT]
        total = col.sum().item()
        print(f"  y[{t:>2}] = {otok!r}    Σφ = {total:+.3f} bits")
        for i in top:
            print(f"      {row_labels[i]:<40}  phi = {col[i].item():+.4f}")
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

        [("question", "What is the largest planet in our solar system?")],
    ]

    for q in questions:
        explain_question(model, tokenizer, q, stop_token_ids)


if __name__ == "__main__":
    main()