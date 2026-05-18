"""
Verification script for forward-only output-token importance on Llama-3.2-1B.

We compare three scoring methods on the example:
    Question: "What is the capital of Vietnam?"
    Answer A: "Hanoi is the capital of Vietnam."
    Answer B: "The capital of Vietnam is Hanoi."

Methods:
    1. Masked-baseline I(y_t)         -- from the original doc
    2. Prior surprisal  H_t (proxy)   -- single-pass entropy-like signal
    3. Conditional PMI                -- the corrected method
    4. Product score = H_t * max(0, PMI)

Each method should ideally rank `Hanoi` as the most important token in both
answer phrasings. We print a per-token table for each answer so you can
inspect the numbers directly.

Run:
    pip install torch transformers accelerate
    huggingface-cli login    # gated model; need a token with access
    python verify_importance.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # use fp32 on CPU for numerical stability of log-probs


# ---------------------------------------------------------------------------
# Core: teacher-forced log-probabilities of a target answer given a prefix
# ---------------------------------------------------------------------------

def teacher_forced_logprobs(model, tokenizer, prefix_ids, answer_ids):
    """
    Return log p(answer_t | prefix, answer_<t) for each t.

    prefix_ids: 1D LongTensor of token ids forming the conditioning prefix
                (e.g. the question, or empty for the prior pass).
    answer_ids: 1D LongTensor of token ids forming the answer to score.

    Returns a 1D tensor of length len(answer_ids) with log-probs.
    """
    if prefix_ids.numel() == 0:
        # The model still needs *some* input — use BOS alone for the prior pass.
        prefix_ids = torch.tensor([tokenizer.bos_token_id], device=DEVICE)

    full_ids = torch.cat([prefix_ids, answer_ids]).unsqueeze(0)  # [1, L]
    with torch.no_grad():
        out = model(full_ids)
        logits = out.logits[0]  # [L, V]

    # The logit at position i predicts token i+1. We want predictions for the
    # answer tokens, which occupy positions [len(prefix), len(full)-1].
    L_prefix = prefix_ids.numel()
    # Predict answer_ids[j] from logits at position (L_prefix - 1 + j).
    pred_positions = torch.arange(L_prefix - 1, L_prefix - 1 + answer_ids.numel(),
                                  device=DEVICE)
    pred_logits = logits[pred_positions]  # [La, V]
    log_probs_all = F.log_softmax(pred_logits, dim=-1)  # [La, V]
    target_logprobs = log_probs_all.gather(1, answer_ids.unsqueeze(1)).squeeze(1)

    # Also return per-position entropy of the full distribution (for H_t).
    probs = log_probs_all.exp()
    entropy = -(probs * log_probs_all).sum(dim=-1)  # [La]

    return target_logprobs, entropy, log_probs_all


# ---------------------------------------------------------------------------
# Masked-baseline pass: replace question tokens with a mask/pad token
# ---------------------------------------------------------------------------

def masked_input_logprobs(model, tokenizer, question_ids, answer_ids,
                          mask_token_id=None):
    """
    Compute log p(y_t | x_masked, y_<t) where x_masked is the question
    with every token replaced by `mask_token_id` (default: pad / eos).

    This is the baseline used by the original doc.
    """
    if mask_token_id is None:
        # Llama tokenizers don't have a true mask token; use the pad/eos.
        mask_token_id = tokenizer.pad_token_id
        if mask_token_id is None:
            mask_token_id = tokenizer.eos_token_id

    masked_ids = torch.full_like(question_ids, mask_token_id)
    logprobs, _, _ = teacher_forced_logprobs(model, tokenizer,
                                             masked_ids, answer_ids)
    return logprobs


# ---------------------------------------------------------------------------
# Main analysis for one (question, answer) pair
# ---------------------------------------------------------------------------

def analyze(model, tokenizer, question, answer, label):
    print("=" * 78)
    print(f"  {label}")
    print(f"  Q: {question}")
    print(f"  A: {answer}")
    print("=" * 78)

    # Tokenize. We add a leading space to the answer so its first token isn't
    # merged with the question's trailing punctuation.
    q_ids = tokenizer(question, return_tensors="pt",
                      add_special_tokens=False).input_ids[0].to(DEVICE)
    a_ids = tokenizer(" " + answer, return_tensors="pt",
                      add_special_tokens=False).input_ids[0].to(DEVICE)

    answer_tokens = [tokenizer.decode([t]) for t in a_ids.tolist()]

    # Pass 1: full conditioning -> log p(y_t | x, y_<t) and entropy H_t.
    full_lp, H_t, _ = teacher_forced_logprobs(model, tokenizer, q_ids, a_ids)

    # Pass 2: prior only (no question) -> log p(y_t | y_<t).
    prior_lp, _, _ = teacher_forced_logprobs(model, tokenizer,
                                             torch.tensor([], dtype=torch.long,
                                                          device=DEVICE),
                                             a_ids)

    # Pass 3: masked-baseline -> log p(y_t | x_masked, y_<t).  (Original doc.)
    masked_lp = masked_input_logprobs(model, tokenizer, q_ids, a_ids)

    # Scores
    pmi = full_lp - prior_lp                       # corrected method
    I_masked = full_lp - masked_lp                 # original doc's method
    prior_surprisal = -prior_lp                    # "prefix didn't know" signal
    product = prior_surprisal * torch.clamp(pmi, min=0.0)  # conjunction score

    # Pretty-print
    header = f"{'idx':>3}  {'token':<14}  {'H_t':>7}  {'-logP(prior)':>12}  " \
             f"{'PMI':>7}  {'I_masked':>9}  {'product':>8}"
    print(header)
    print("-" * len(header))
    for i, tok in enumerate(answer_tokens):
        print(f"{i:>3}  {repr(tok):<14}  "
              f"{H_t[i].item():>7.3f}  "
              f"{prior_surprisal[i].item():>12.3f}  "
              f"{pmi[i].item():>7.3f}  "
              f"{I_masked[i].item():>9.3f}  "
              f"{product[i].item():>8.3f}")

    # Winner under each scoring rule
    def winner(scores, name):
        idx = int(torch.argmax(scores).item())
        print(f"  argmax({name:<20}) -> idx {idx}, token {answer_tokens[idx]!r}")

    print()
    winner(H_t, "entropy H_t")
    winner(pmi, "PMI")
    winner(I_masked, "I_masked (original)")
    winner(product, "product (proposed)")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    question = "What is the capital of Vietnam?"
    analyze(model, tokenizer, question,
            "Hanoi is the capital of Vietnam.",
            "Answer A (front-loaded)")
    analyze(model, tokenizer, question,
            "The capital of Vietnam is Hanoi.",
            "Answer B (back-loaded)")

    # A second example, to check generalization.
    analyze(model, tokenizer,
            "What is the largest planet in our solar system?",
            "Jupiter is the largest planet.",
            "Sanity check: Jupiter (front-loaded)")
    analyze(model, tokenizer,
            "What is the largest planet in our solar system?",
            "The largest planet is Jupiter.",
            "Sanity check: Jupiter (back-loaded)")


if __name__ == "__main__":
    main()