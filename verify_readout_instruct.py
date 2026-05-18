"""
Generate-then-analyze interface for the readout-point method.
INSTRUCT MODEL VERSION — v3.

Story so far
------------
v1: Used D_prefix (full_lp - nopre_lp) and a cumulative information term.
    Both pathologized on long answers: D_prefix exploded because asking
    "how likely is this mid-sentence token to come straight after the
    question, no prefix?" is a nonsensical query. The cumulative term
    inflated late tokens monotonically. Worst case: a comma at position
    28 of the Jupiter answer scored 400; the actual answer "Jupiter"
    scored 78.

v2: Replaced D_prefix with slot entropy H_t = H(p(. | question, y_<t)).
    Fixed the runaway-commas problem but introduced a new failure mode:
    slot entropy is measured AT the prediction that produced the token,
    and by that point the model is already committed. Result: slot_H
    collapses to ~0.01–0.2 bits on every content token, killing the
    score. The only positions with substantial slot_H (>1 bit) were
    structural choice points like " It" and " with" starting a new
    clause — not content tokens.

v3 (this file)
--------------
Conclusion from v2: Di_inst (information gained from the question, i.e.
log p(y_t | q, y_<t) - log p(y_t | y_<t)) is the cleanest signal we
have. It correctly ranked:
  - "Compact", "Y/ss/aria", "M/ire/nt/ane", "148"/"7" in the Yssaria test
  - "Treaty", "T" in the Tordesillas test
  - "largest", "planet" in the Jupiter test
without flagging filler punctuation or function words.

So v3 keeps Di_inst as the primary score and adds slot entropy only as
a *soft* tiebreaker (1 + slot_H), which can't annihilate the signal:

    score = max(Di_inst, 0) * (1 + slot_H)

We also print "Di_inst alone" rankings side-by-side so you can see
whether the slot-entropy tiebreaker is helping or hurting on each
example.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

MAX_NEW_TOKENS = 60
LN2 = float(torch.log(torch.tensor(2.0)))  # nats -> bits


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
# Greedy generation. Records full log-prob and slot entropy per token
# from the same forward pass.
# ---------------------------------------------------------------------------

def _entropy_bits(log_probs_row):
    p = log_probs_row.exp()
    nz = p > 0
    return -(p[nz] * log_probs_row[nz]).sum().item() / LN2


def generate_with_logprobs(model, tokenizer, prompt_ids,
                            stop_token_ids, max_new_tokens=MAX_NEW_TOKENS):
    """Returns (answer_ids, full_lp_nats, slot_entropy_bits, answer_text)."""
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
# Prior pass: log p(y_t | y_<t) starting from BOS — what the base LM
# thought of these tokens on their own, no question, no chat template.
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

def print_topk_summary(label, tokens, score_tensor, k):
    """Print top-k by a single score tensor."""
    k = min(k, len(tokens))
    top_idx = torch.topk(score_tensor, k=k).indices.tolist()
    print(f"  Top-{k} by {label}:")
    for i in sorted(top_idx):
        print(f"    idx {i:>2}  {tokens[i]!r:<14}  "
              f"score={score_tensor[i].item():.3f}")
    return set(top_idx)


def highlighted(tokens, flagged):
    return "".join(
        f"[{tok.strip()}]" if i in flagged else tok
        for i, tok in enumerate(tokens)
    )


def generate_and_explain(model, tokenizer, user_message, stop_token_ids,
                         top_k=3):
    print("=" * 96)
    print(f"  USER: {user_message}")
    print("=" * 96)

    prompt_ids, _ = build_chat_prompt(tokenizer, user_message)

    a_ids, full_lp, slot_H, answer_text = generate_with_logprobs(
        model, tokenizer, prompt_ids, stop_token_ids)
    if a_ids.numel() == 0:
        print("  Model produced no answer.")
        return

    print(f"  ASSISTANT: {answer_text!r}")
    print()

    pri_lp = prior_logprobs(model, tokenizer, a_ids)

    full_lp_bits = full_lp / LN2
    pri_lp_bits = pri_lp / LN2

    Di_inst = full_lp_bits - pri_lp_bits        # bits the question added
    surprisal = -full_lp_bits                   # raw surprisal, bits
    conf = full_lp.exp()
    # slot_H already in bits

    # v3 scores
    Di_pos = torch.clamp(Di_inst, min=0.0)
    di_only = Di_pos
    score = Di_pos * (1.0 + slot_H)             # soft tiebreaker

    tokens = [tokenizer.decode([t]) for t in a_ids.tolist()]

    header = (f"{'idx':>3}  {'token':<14}  "
              f"{'Di_inst':>8}  {'surpr':>7}  {'slotH':>7}  "
              f"{'conf':>6}  {'di_only':>8}  {'score':>8}")
    print(header)
    print("-" * len(header))
    for i, tok in enumerate(tokens):
        print(f"{i:>3}  {repr(tok):<14}  "
              f"{Di_inst[i].item():>8.3f}  "
              f"{surprisal[i].item():>7.3f}  "
              f"{slot_H[i].item():>7.3f}  "
              f"{conf[i].item():>6.3f}  "
              f"{di_only[i].item():>8.3f}  "
              f"{score[i].item():>8.3f}")

    print()
    flagged_di = print_topk_summary("Di_inst alone", tokens, di_only, top_k)
    print()
    flagged_score = print_topk_summary("score = Di_inst * (1 + slotH)",
                                        tokens, score, top_k)
    print()
    print("  Highlighted (Di_inst-alone):")
    print(f"    {highlighted(tokens, flagged_di)}")
    print("  Highlighted (score):")
    print(f"    {highlighted(tokens, flagged_score)}")
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