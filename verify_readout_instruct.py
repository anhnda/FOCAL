"""
Generate-then-analyze interface for the readout-point method.
INSTRUCT MODEL VERSION — v2.

Changes from v1
---------------
We dropped two components that were causing the score to misbehave on
long and extractive answers:

  * D_prefix = full_lp - nopre_lp had two failure modes:
      - For tokens deep in the answer, nopre_lp asks "how likely is this
        mid-sentence token to come *immediately* after the question,
        with no answer-so-far?" That's a nonsensical distribution, so
        nopre_lp collapses and D_prefix explodes (30+ bits on filler
        commas in the Jupiter example).
      - Function words like " of" / " the" have the HIGHEST D_prefix
        because they're the most predictable from local context, which
        inverted the score: " of" between "council" and "Mirentane"
        outranked "1487" (the actual answer) in the Yssaria test.

  * D_input_cum is monotone non-decreasing, so as a multiplier it just
    inflates late tokens for free.

Replacements
------------
  * SLOT ENTROPY  H_t = H(p(. | question, y_<t))
        How open was the choice at this position? High = many alternatives
        were live (the model had a real decision to make); low = forced
        continuation (BPE tail, punctuation, function-word completion).
        This is computed during generation at no extra cost.

  * READOUT score:
        readout = max(Di_inst, 0) * conf * H_slot
        - max(Di_inst, 0): the question shifted this token up from prior.
        - conf: the model committed to it.
        - H_slot: the slot was non-trivial.
        A pure-filler BPE tail gets conf~1 but H_slot~0, so it scores 0.
        A function word gets H_slot moderate but Di_inst~0, so it scores 0.
        The answer-bearing token (e.g. "1487", "Hanoi", "Jupiter") gets
        all three nonzero.

  * SETUP score is unchanged in spirit but rewritten to not use D_prefix:
        setup = max(Di_inst, 0) * H_slot
        It still flags positions where the question genuinely shifted the
        distribution AND the slot was open.

  * combined = max(setup, readout). For factoid extraction these
    typically coincide on the answer token; the max just lets either
    interpretation win.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

MAX_NEW_TOKENS = 60
LN2 = float(torch.log(torch.tensor(2.0)))  # for converting nats -> bits


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_chat_prompt(tokenizer, user_message):
    messages = [{"role": "user", "content": user_message}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(prompt_text, return_tensors="pt",
                    add_special_tokens=False).input_ids[0].to(DEVICE)
    return ids, prompt_text


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
# Greedy generation, recording per-token full log-prob AND slot entropy.
# Both come from the same forward pass — no extra cost over v1.
# ---------------------------------------------------------------------------

def _entropy_bits(log_probs_row):
    """Shannon entropy of a log-probability row, in bits."""
    p = log_probs_row.exp()
    # Mask zeros to avoid 0 * -inf.
    nz = p > 0
    return -(p[nz] * log_probs_row[nz]).sum().item() / LN2


def generate_with_logprobs(model, tokenizer, prompt_ids,
                            stop_token_ids, max_new_tokens=MAX_NEW_TOKENS):
    """
    Greedy decode. Returns (answer_ids, full_lp, slot_entropy, answer_text).
    full_lp: log p(y_t | question, y_<t), in nats.
    slot_entropy: H(p(. | question, y_<t)), in bits.
    """
    generated, full_lp_list, slot_H_list = [], [], []
    cur_ids = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(cur_ids.unsqueeze(0)).logits[0, -1]
            log_probs = F.log_softmax(logits, dim=-1)
            next_id = int(log_probs.argmax().item())

            if next_id in stop_token_ids:
                break

            lp = log_probs[next_id].item()
            H = _entropy_bits(log_probs)

            generated.append(next_id)
            full_lp_list.append(lp)
            slot_H_list.append(H)
            cur_ids = torch.cat([cur_ids,
                                 torch.tensor([next_id], device=DEVICE)])

    if not generated:
        empty_long = torch.tensor([], dtype=torch.long, device=DEVICE)
        empty_f = torch.tensor([], device=DEVICE)
        return empty_long, empty_f, empty_f, ""

    a_ids = torch.tensor(generated, dtype=torch.long, device=DEVICE)
    full_lp = torch.tensor(full_lp_list, device=DEVICE)
    slot_H = torch.tensor(slot_H_list, device=DEVICE)
    answer_text = tokenizer.decode(a_ids)
    return a_ids, full_lp, slot_H, answer_text


# ---------------------------------------------------------------------------
# Prior pass: log p(y_t | y_<t) starting from BOS, no question, no template.
# This isolates "what the base LM thought of these tokens on their own".
# ---------------------------------------------------------------------------

def prior_logprobs(model, tokenizer, answer_ids):
    bos = torch.tensor([tokenizer.bos_token_id], device=DEVICE)
    full_ids = torch.cat([bos, answer_ids]).unsqueeze(0)
    with torch.no_grad():
        logits = model(full_ids).logits[0]
    L_prefix = bos.numel()
    pred_positions = torch.arange(L_prefix - 1,
                                  L_prefix - 1 + answer_ids.numel(),
                                  device=DEVICE)
    pred_logits = logits[pred_positions]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    return log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def generate_and_explain(model, tokenizer, user_message, stop_token_ids,
                         top_k=3):
    print("=" * 96)
    print(f"  USER: {user_message}")
    print("=" * 96)

    prompt_ids, _ = build_chat_prompt(tokenizer, user_message)

    # Pass 1: generation. Records full_lp and slot entropy per token.
    a_ids, full_lp, slot_H, answer_text = generate_with_logprobs(
        model, tokenizer, prompt_ids, stop_token_ids)
    if a_ids.numel() == 0:
        print("  Model produced no answer.")
        return

    print(f"  ASSISTANT: {answer_text!r}")
    print()

    # Pass 2: prior log-probs (BOS only, no question).
    pri_lp = prior_logprobs(model, tokenizer, a_ids)

    # Convert to bits for readability.
    full_lp_bits = full_lp / LN2
    pri_lp_bits = pri_lp / LN2

    # Per-token signals.
    Di_inst = full_lp_bits - pri_lp_bits        # bits the question added
    surprisal = -full_lp_bits                   # raw token surprisal, bits
    conf = full_lp.exp()                        # p(y_t | question, y_<t)
    # slot_H already in bits.

    # New scores.
    # setup:   question shifted the choice AND the slot was open.
    # readout: the question shifted it, the slot was open, AND the model
    #          committed (conf high).
    Di_pos = torch.clamp(Di_inst, min=0.0)
    setup = Di_pos * slot_H
    readout = Di_pos * slot_H * conf
    combined = torch.maximum(setup, readout)

    tokens = [tokenizer.decode([t]) for t in a_ids.tolist()]

    header = (f"{'idx':>3}  {'token':<14}  "
              f"{'Di_inst':>8}  {'surpr':>7}  {'slotH':>7}  "
              f"{'conf':>6}  {'setup':>8}  {'readout':>8}  {'combined':>9}")
    print(header)
    print("-" * len(header))
    for i, tok in enumerate(tokens):
        print(f"{i:>3}  {repr(tok):<14}  "
              f"{Di_inst[i].item():>8.3f}  "
              f"{surprisal[i].item():>7.3f}  "
              f"{slot_H[i].item():>7.3f}  "
              f"{conf[i].item():>6.3f}  "
              f"{setup[i].item():>8.3f}  "
              f"{readout[i].item():>8.3f}  "
              f"{combined[i].item():>9.3f}")

    k = min(top_k, len(tokens))
    top_idx = torch.topk(combined, k=k).indices.tolist()
    top_idx_sorted = sorted(top_idx)
    print()
    print(f"  Top-{k} important tokens (by combined score):")
    for i in top_idx_sorted:
        print(f"    idx {i:>2}  {tokens[i]!r}  "
              f"(combined={combined[i].item():.2f}, "
              f"Di_inst={Di_inst[i].item():.2f}b, "
              f"slotH={slot_H[i].item():.2f}b, "
              f"conf={conf[i].item():.2f})")

    print()
    print("  Highlighted answer:")
    flagged = set(top_idx)
    highlighted = "".join(
        f"[{tok.strip()}]" if i in flagged else tok
        for i, tok in enumerate(tokens)
    )
    print(f"    {highlighted}")
    print()


def main():
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE
    ).to(DEVICE).eval()
    stop_token_ids = get_stop_token_ids(tokenizer)

    questions = [
        "What is the name of our planet?",
        "What is the capital of Vietnam?",
        "What is the largest planet in our solar system?",

        "Passage: The Treaty of Tordesillas was signed in 1494, "
        "dividing the New World between Spain and Portugal.\n"
        "Question: In what year was the Treaty of Tordesillas signed? "
        "Answer in one short sentence.",

        "What is 23 plus 45? Answer with just the number.",

        "Passage: The Compact of Yssaria was ratified in 1487 "
        "by the council of Mirentane.\n"
        "Question: When was the Compact of Yssaria ratified? "
        "Answer in one short sentence.",
    ]

    for q in questions:
        generate_and_explain(model, tokenizer, q, stop_token_ids, top_k=3)


if __name__ == "__main__":
    main()