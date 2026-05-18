"""
Test the READOUT hypothesis.

Reframing (user's insight):
    The input causes the prefix; the prefix causes the answer.
    So measuring D_input *only at the answer token* misses the chain —
    the input did its work earlier, and by the answer slot it's been
    fully absorbed into the prefix.

    The answer token is the READOUT point: where the cumulative input
    contribution stops growing AND the model is highly confident AND
    the specific prefix shaped this token.

Three per-token quantities:
    D_input_inst(t)  = log p(y_t | x, y_<t) - log p(y_t | y_<t)
                       instantaneous input contribution.
    D_input_cum(t)   = sum_{s<=t} D_input_inst(s)
                       total input contribution to the prefix-up-to-t.
                       (This equals log [ p(y_<=t | x) / p(y_<=t) ].)
    D_prefix(t)      = log p(y_t | x, y_<t) - log p(y_t | x)
                       contribution of the specific prefix.
    conf(t)          = exp(log p(y_t | x, y_<t))  -- the model's certainty.

Three derived signals:
    setup(t)   = D_input_inst(t) * D_prefix(t)
                 "both sources actively contributing at this step"
                 -- flags setup tokens and front-loaded answers.
    readout(t) = D_input_cum(t) * D_prefix(t) * conf(t) * decay(t)
                 where decay(t) = sigmoid(-D_input_inst(t)) shrinks the
                 score when the input is STILL adding info (we want it
                 to have stopped adding info, i.e. been absorbed).
                 -- flags back-loaded answers.
    combined(t) = max(setup(t), readout(t))
                 -- should flag both front- and back-loaded answers.

Run:
    pip install torch transformers accelerate
    huggingface-cli login
    python verify_readout.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Forward-pass primitives
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


def no_prefix_logprobs(model, tokenizer, question_ids, answer_ids):
    """log p(y_t | x) for each t -- treat y_t as the first token after x."""
    with torch.no_grad():
        logits = model(question_ids.unsqueeze(0)).logits[0]
    next_dist = F.log_softmax(logits[-1], dim=-1)
    return next_dist[answer_ids]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(model, tokenizer, question, answer, label):
    print("=" * 88)
    print(f"  {label}")
    print(f"  Q: {question}")
    print(f"  A: {answer}")
    print("=" * 88)

    q_ids = tokenizer(question, return_tensors="pt",
                      add_special_tokens=False).input_ids[0].to(DEVICE)
    a_ids = tokenizer(" " + answer, return_tensors="pt",
                      add_special_tokens=False).input_ids[0].to(DEVICE)
    tokens = [tokenizer.decode([t]) for t in a_ids.tolist()]

    # Three forward passes.
    full_lp  = tf_logprobs(model, tokenizer, q_ids, a_ids)
    prior_lp = tf_logprobs(model, tokenizer,
                           torch.tensor([], dtype=torch.long, device=DEVICE),
                           a_ids)
    nopre_lp = no_prefix_logprobs(model, tokenizer, q_ids, a_ids)

    # Per-token quantities.
    d_input_inst = full_lp - prior_lp                  # instantaneous PMI
    d_input_cum  = torch.cumsum(d_input_inst, dim=0)   # cumulative PMI
    d_prefix     = full_lp - nopre_lp                  # prefix contribution
    conf         = full_lp.exp()                        # local certainty (prob)

    # Setup score: both sources actively contributing right now.
    setup = (torch.clamp(d_input_inst, min=0)
             * torch.clamp(d_prefix, min=0))

    # Readout score: cumulative input is large, but instant has decayed,
    # the prefix shapes this token, and the model is confident.
    decay = torch.sigmoid(-d_input_inst)  # ~1 when instant PMI is small/negative
    readout = (torch.clamp(d_input_cum, min=0)
               * torch.clamp(d_prefix, min=0)
               * conf
               * decay)

    combined = torch.maximum(setup, readout)

    # Pretty-print
    header = (f"{'idx':>3}  {'token':<14}  "
              f"{'Di_inst':>8}  {'Di_cum':>7}  {'D_pref':>7}  "
              f"{'conf':>6}  {'setup':>8}  {'readout':>8}  {'combined':>8}")
    print(header)
    print("-" * len(header))
    for i, tok in enumerate(tokens):
        print(f"{i:>3}  {repr(tok):<14}  "
              f"{d_input_inst[i].item():>8.3f}  "
              f"{d_input_cum[i].item():>7.2f}  "
              f"{d_prefix[i].item():>7.2f}  "
              f"{conf[i].item():>6.3f}  "
              f"{setup[i].item():>8.3f}  "
              f"{readout[i].item():>8.3f}  "
              f"{combined[i].item():>8.3f}")

    def winner(scores, name):
        idx = int(torch.argmax(scores).item())
        print(f"  argmax({name:<10}) -> idx {idx}, token {tokens[idx]!r}")

    print()
    winner(setup, "setup")
    winner(readout, "readout")
    winner(combined, "combined")
    print()


def main():
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    # Closed-book recall, both phrasings.
    analyze(model, tokenizer,
            "What is the capital of Vietnam?",
            "Hanoi is the capital of Vietnam.",
            "Hanoi (front-loaded) -- expect setup to win at 'anoi'")
    analyze(model, tokenizer,
            "What is the capital of Vietnam?",
            "The capital of Vietnam is Hanoi.",
            "Hanoi (back-loaded) -- expect readout to win at 'anoi'")

    # Context extraction.
    ctx_q = ("Passage: The Treaty of Tordesillas was signed in 1494, "
             "dividing the New World between Spain and Portugal.\n"
             "Question: In what year was the Treaty of Tordesillas signed?")
    analyze(model, tokenizer, ctx_q,
            "1494 is the year it was signed.",
            "1494 (front-loaded) -- expect setup to win at '149'")
    analyze(model, tokenizer, ctx_q,
            "The Treaty of Tordesillas was signed in 1494.",
            "1494 (back-loaded) -- expect readout to win at '149' or '4'")

    # Arithmetic.
    analyze(model, tokenizer,
            "What is 23 plus 45? Answer with just the number.",
            "23 plus 45 equals 68.",
            "Arithmetic -- expect readout to win at '68'")

    # Sanity: another back-loaded factual.
    analyze(model, tokenizer,
            "What is the largest planet in our solar system?",
            "The largest planet is Jupiter.",
            "Jupiter (back-loaded) -- expect readout to win at 'Jupiter'")


if __name__ == "__main__":
    main()