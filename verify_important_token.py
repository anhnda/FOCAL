"""
Test the two-source contribution hypothesis.

Hypothesis (user's, restated):
    A token y_t is IMPORTANT iff BOTH the input x and the prefix y_<t
    contribute substantially to predicting it.
    A FILLER token is one the prefix alone determines, with the input
    adding little.

We compute two quantities per output token:
    Delta_input(t)  = log p(y_t | x, y_<t) - log p(y_t | y_<t)
                    = how much the input adds, given the prefix
                    = PMI(y_t ; x | y_<t)
    Delta_prefix(t) = log p(y_t | x, y_<t) - log p(y_t | x)
                    = how much the prefix adds, given the input
                    where p(y_t | x) treats y_t as if it were the first
                    answer token (no prefix).

A token is FLAGGED as important when both deltas are above their
within-sequence median (or, equivalently, when their product is large).

We test on three task types:
    1. Closed-book recall (model knows the answer from weights)
    2. Context-grounded extraction (answer is in the input)
    3. Multi-step reasoning (answer requires combining input + prefix)

Run:
    pip install torch transformers accelerate
    huggingface-cli login  # gated model
    python verify_two_sources.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Teacher-forced log-probs of a target answer given a prefix
# ---------------------------------------------------------------------------

def tf_logprobs(model, tokenizer, prefix_ids, answer_ids):
    """log p(answer_t | prefix, answer_<t) for each t."""
    if prefix_ids.numel() == 0:
        prefix_ids = torch.tensor([tokenizer.bos_token_id], device=DEVICE)

    full_ids = torch.cat([prefix_ids, answer_ids]).unsqueeze(0)
    with torch.no_grad():
        logits = model(full_ids).logits[0]

    L_prefix = prefix_ids.numel()
    pred_positions = torch.arange(L_prefix - 1,
                                  L_prefix - 1 + answer_ids.numel(),
                                  device=DEVICE)
    pred_logits = logits[pred_positions]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    return log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)


# ---------------------------------------------------------------------------
# The "no prefix" prediction: for every position t, compute p(y_t | x)
# treating y_t as if it were the immediate next token after the question.
#
# We do this with ONE forward pass on (x,) and read off the distribution at
# the final position of x. That gives us p(* | x). We then look up the
# probability of each specific y_t in that single distribution.
#
# Interpretation: "how likely is this exact token to appear right after the
# question, with no answer-so-far built up?"
# ---------------------------------------------------------------------------

def no_prefix_logprobs(model, tokenizer, question_ids, answer_ids):
    """log p(y_t | x) for each t, using a single forward pass on x."""
    with torch.no_grad():
        logits = model(question_ids.unsqueeze(0)).logits[0]
    # Distribution over the FIRST token after x.
    next_dist = F.log_softmax(logits[-1], dim=-1)  # [V]
    # Look up each answer token in that one distribution.
    return next_dist[answer_ids]  # [La]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(model, tokenizer, question, answer, label):
    print("=" * 78)
    print(f"  {label}")
    print(f"  Q: {question}")
    print(f"  A: {answer}")
    print("=" * 78)

    q_ids = tokenizer(question, return_tensors="pt",
                      add_special_tokens=False).input_ids[0].to(DEVICE)
    a_ids = tokenizer(" " + answer, return_tensors="pt",
                      add_special_tokens=False).input_ids[0].to(DEVICE)
    tokens = [tokenizer.decode([t]) for t in a_ids.tolist()]

    # Three forward passes.
    full_lp  = tf_logprobs(model, tokenizer, q_ids, a_ids)            # p(y|x,y<t)
    prior_lp = tf_logprobs(model, tokenizer,
                           torch.tensor([], dtype=torch.long, device=DEVICE),
                           a_ids)                                      # p(y|y<t)
    nopre_lp = no_prefix_logprobs(model, tokenizer, q_ids, a_ids)     # p(y|x)

    delta_input  = full_lp - prior_lp   # input's contribution given prefix
    delta_prefix = full_lp - nopre_lp   # prefix's contribution given input

    # Hypothesis: important iff BOTH deltas are large.
    # Use product of positive parts as the conjunction score.
    score = torch.clamp(delta_input, min=0) * torch.clamp(delta_prefix, min=0)

    # Header
    header = (f"{'idx':>3}  {'token':<14}  "
              f"{'D_input':>8}  {'D_prefix':>9}  {'AND-score':>10}  "
              f"{'full_lp':>8}")
    print(header)
    print("-" * len(header))
    for i, tok in enumerate(tokens):
        print(f"{i:>3}  {repr(tok):<14}  "
              f"{delta_input[i].item():>8.3f}  "
              f"{delta_prefix[i].item():>9.3f}  "
              f"{score[i].item():>10.3f}  "
              f"{full_lp[i].item():>8.3f}")

    # Tokens flagged by the conjunction (top-1 by score)
    top_idx = int(torch.argmax(score).item())
    print(f"\n  TOP token under conjunction score: idx {top_idx} = {tokens[top_idx]!r}")

    # Also flag ALL tokens above median on both axes simultaneously.
    di_med = delta_input.median()
    dp_med = delta_prefix.median()
    flagged = [(i, tokens[i]) for i in range(len(tokens))
               if delta_input[i] > di_med and delta_prefix[i] > dp_med]
    print(f"  Flagged (above-median on both axes): "
          f"{[t for _, t in flagged]}")
    print()


def main():
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    # ---- Closed-book recall (model has the fact memorized) ----
    analyze(model, tokenizer,
            "What is the capital of Vietnam?",
            "Hanoi is the capital of Vietnam.",
            "Closed-book recall: Hanoi (front-loaded)")
    analyze(model, tokenizer,
            "What is the capital of Vietnam?",
            "The capital of Vietnam is Hanoi.",
            "Closed-book recall: Hanoi (back-loaded)")

    # ---- Context-grounded extraction ----
    # The answer "1494" must come from the passage; the model is unlikely
    # to know this date from weights alone.
    ctx_q = ("Passage: The Treaty of Tordesillas was signed in 1494, "
             "dividing the New World between Spain and Portugal.\n"
             "Question: In what year was the Treaty of Tordesillas signed?")
    analyze(model, tokenizer, ctx_q,
            "1494 is the year it was signed.",
            "Context extraction: 1494 (front-loaded)")
    analyze(model, tokenizer, ctx_q,
            "The Treaty of Tordesillas was signed in 1494.",
            "Context extraction: 1494 (back-loaded)")

    # ---- Reasoning: answer requires combining input + prefix ----
    # Arithmetic where each digit token depends on both the input numbers
    # and the partial result so far.
    analyze(model, tokenizer,
            "What is 23 plus 45? Answer with just the number.",
            "23 plus 45 equals 68.",
            "Reasoning: arithmetic")


if __name__ == "__main__":
    main()