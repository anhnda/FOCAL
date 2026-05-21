"""
Two CLI modes for probing a causal LM:

  --a              one-shot: read a sentence from stdin, model generates an
                   answer, print it, exit.

  --g "<golden>"   debug loop: each iteration, read a (possibly permuted)
                   input sentence from stdin. Concat (input + golden answer),
                   tokenize, run ONE forward pass, then for each answer-token
                   position (1-shift aligned to predict that token) print:
                       - the probability the model assigned to the target
                         golden token at that position
                       - the top-k tokens with highest probability at that
                         same position
                   Empty line or Ctrl-D exits.

Usage:
    pip install torch transformers accelerate
    huggingface-cli login   # gated model
    python probe_cli.py --a
    python probe_cli.py --g " Hanoi is the capital of Vietnam."
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B"
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


# ---------------------------------------------------------------------------
# Mode --a : one-shot generation
# ---------------------------------------------------------------------------

def mode_generate(tokenizer, model):
    try:
        prompt = input("Input: ")
    except EOFError:
        return
    if not prompt.strip():
        print("(empty input, exiting)")
        return

    input_ids = tokenizer(prompt, return_tensors="pt",
                          add_special_tokens=True).input_ids.to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0, input_ids.shape[1]:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"Answer: {answer}")


# ---------------------------------------------------------------------------
# Mode --g : permutation debug loop
# ---------------------------------------------------------------------------

def debug_one(tokenizer, model, user_input, golden):
    """One forward pass on (user_input + golden); print per-position debug."""
    # Tokenize separately so we know how many tokens belong to the golden
    # answer (those are the positions we want to debug).
    in_ids = tokenizer(user_input, return_tensors="pt",
                       add_special_tokens=True).input_ids[0].to(DEVICE)
    gold_ids = tokenizer(golden, return_tensors="pt",
                         add_special_tokens=False).input_ids[0].to(DEVICE)

    full_ids = torch.cat([in_ids, gold_ids]).unsqueeze(0)

    with torch.no_grad():
        logits = model(full_ids).logits[0]  # [T, V]

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    L_in = in_ids.numel()
    L_g = gold_ids.numel()

    # For predicting gold token at index k (0-based within the golden span),
    # the logits we need come from position (L_in - 1 + k) in the full seq,
    # because position t's logits predict the token at t+1.
    # k=0  -> uses logits at L_in-1 (last input token) to predict first gold tok
    # k=1  -> uses logits at L_in   to predict second gold tok
    # ...
    pred_positions = torch.arange(L_in - 1, L_in - 1 + L_g, device=DEVICE)
    pred_logprobs = log_probs[pred_positions]   # [L_g, V]
    pred_probs = probs[pred_positions]          # [L_g, V]

    # Probability assigned to the actual target token at each position.
    target_probs = pred_probs.gather(1, gold_ids.unsqueeze(1)).squeeze(1)

    # Top-k predictions at each position.
    topk_vals, topk_idx = pred_probs.topk(TOPK, dim=-1)

    gold_tokens = [tokenizer.decode([t]) for t in gold_ids.tolist()]

    print(f"\nInput  : {user_input!r}")
    print(f"Golden : {golden!r}")
    print(f"  (input tokens: {L_in}, golden tokens: {L_g})\n")

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
    print("Enter input sentences. Empty line or Ctrl-D to quit.\n")
    while True:
        try:
            user_input = input("Input> ")
        except EOFError:
            print()
            return
        if not user_input.strip():
            return
        debug_one(tokenizer, model, user_input, golden)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", action="store_true",
                        help="One-shot: read a sentence, print model answer, exit.")
    parser.add_argument("--g", type=str, default=None,
                        help="Debug loop: golden answer string (with leading "
                             "space if needed for proper tokenization, e.g. "
                             "' Hanoi is ...').")
    args = parser.parse_args()

    if not args.a and args.g is None:
        parser.error("must pass either --a or --g <golden>")
    if args.a and args.g is not None:
        parser.error("--a and --g are mutually exclusive")

    tokenizer, model = load_model()

    if args.a:
        mode_generate(tokenizer, model)
    else:
        mode_debug_loop(tokenizer, model, args.g)


if __name__ == "__main__":
    main()