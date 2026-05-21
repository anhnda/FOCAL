"""
Two CLI modes for probing an INSTRUCT causal LM (chat-template wrapped).

  --a "<sentence>" one-shot: wrap sentence as a user message via the chat
                   template, greedy-generate an assistant reply, print it,
                   exit.

  --g "<golden>"   debug loop: golden answer fixed. Each iteration, read a
                   (possibly permuted) user message from stdin. Build:
                       chat_prefix + user_message + chat_suffix + golden
                   and run ONE forward pass. For each golden-token position
                   (1-shift aligned to predict that token) print:
                       - probability of the target golden token
                       - rank of the target token
                       - top-k highest-prob tokens at that position
                   Empty line or Ctrl-D exits.

Usage:
    pip install torch transformers accelerate
    huggingface-cli login
    python probe_cli_instruct.py --a "What is the capital of Vietnam?"
    python probe_cli_instruct.py --g "Hanoi is the capital of Vietnam."
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

TOPK = 10
MAX_NEW_TOKENS = 64


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    print(f"Loading {MODEL_NAME} on {DEVICE} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE
    ).to(DEVICE).eval()
    return tokenizer, model


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
# Build the templated prompt ids for a given user message.
# ---------------------------------------------------------------------------

def build_prompt_ids(tokenizer, user_message):
    """Return the token ids for: chat_prefix + user_message + chat_suffix."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt",
                    add_special_tokens=False).input_ids[0].to(DEVICE)
    return ids


# ---------------------------------------------------------------------------
# Mode --a : one-shot greedy generation
# ---------------------------------------------------------------------------

def mode_generate(tokenizer, model, user_message):
    if not user_message.strip():
        print("(empty input, exiting)")
        return

    prompt_ids = build_prompt_ids(tokenizer, user_message)
    stop_token_ids = get_stop_token_ids(tokenizer)

    generated = []
    cur_ids = prompt_ids.clone()
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits = model(cur_ids.unsqueeze(0)).logits[0, -1]
            next_id = int(logits.argmax().item())
            if next_id in stop_token_ids:
                break
            generated.append(next_id)
            cur_ids = torch.cat([cur_ids,
                                 torch.tensor([next_id], device=DEVICE)])

    answer = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"Answer: {answer}")


# ---------------------------------------------------------------------------
# Mode --g : permutation debug loop
# ---------------------------------------------------------------------------

def debug_one(tokenizer, model, user_message, golden):
    """One forward pass on (templated user_message + golden); per-position debug."""
    prompt_ids = build_prompt_ids(tokenizer, user_message)
    gold_ids = tokenizer(golden, return_tensors="pt",
                         add_special_tokens=False).input_ids[0].to(DEVICE)

    full_ids = torch.cat([prompt_ids, gold_ids]).unsqueeze(0)

    with torch.no_grad():
        logits = model(full_ids).logits[0]  # [T, V]

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    L_in = prompt_ids.numel()
    L_g = gold_ids.numel()

    # Position (L_in - 1 + k) predicts gold token at index k.
    pred_positions = torch.arange(L_in - 1, L_in - 1 + L_g, device=DEVICE)
    pred_probs = probs[pred_positions]                         # [L_g, V]

    target_probs = pred_probs.gather(1, gold_ids.unsqueeze(1)).squeeze(1)
    topk_vals, topk_idx = pred_probs.topk(TOPK, dim=-1)

    gold_tokens = [tokenizer.decode([t]) for t in gold_ids.tolist()]

    print(f"\nUser   : {user_message!r}")
    print(f"Golden : {golden!r}")
    print(f"  (templated prompt tokens: {L_in}, golden tokens: {L_g})\n")

    for k in range(L_g):
        tgt = gold_tokens[k]
        p = target_probs[k].item()
        rank = (pred_probs[k] > pred_probs[k, gold_ids[k]]).sum().item() + 1
        print(f"  [pos {k:>2}] target={tgt!r}  p={p:.4f}  rank={rank}")

        top_tokens = [tokenizer.decode([i]) for i in topk_idx[k].tolist()]
        top_pairs = [f"{tok!r}:{v.item():.3f}"
                     for tok, v in zip(top_tokens, topk_vals[k])]
        print(f"           top{TOPK}: {', '.join(top_pairs)}")
    print()


def mode_debug_loop(tokenizer, model, golden):
    print(f"Golden answer fixed as: {golden!r}")
    print("Enter user messages. Empty line or Ctrl-D to quit.\n")
    while True:
        try:
            user_message = input("User> ")
        except EOFError:
            print()
            return
        if not user_message.strip():
            return
        debug_one(tokenizer, model, user_message, golden)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=str, default=None, metavar="SENTENCE",
                        help="One-shot: feed SENTENCE as user message, "
                             "print generated assistant reply, exit.")
    parser.add_argument("--g", type=str, default=None, metavar="GOLDEN",
                        help="Debug loop: golden assistant answer string.")
    args = parser.parse_args()

    if args.a is None and args.g is None:
        parser.error("must pass either --a <sentence> or --g <golden>")
    if args.a is not None and args.g is not None:
        parser.error("--a and --g are mutually exclusive")

    tokenizer, model = load_model()

    if args.a is not None:
        mode_generate(tokenizer, model, args.a)
    else:
        mode_debug_loop(tokenizer, model, args.g)


if __name__ == "__main__":
    main()