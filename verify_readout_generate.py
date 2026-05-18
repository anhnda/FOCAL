"""
Generate-then-analyze interface for the readout-point method.

Workflow:
    1. User provides a question (no reference answer).
    2. Model generates an answer greedily.
    3. We run the three forward passes on (question, generated_answer)
       and compute setup / readout / combined scores per token.
    4. We print the result, highlighting which generated tokens were
       flagged as important.

The first forward pass (full conditioning) is folded into generation:
greedy decoding already gives us log p(y_t | x, y_<t) at every step,
so we record those instead of re-running the model.

This is the realistic edge-deployment scenario:
    - 1 generation pass (gives full_lp for free)
    - 1 forward pass on the generated answer with no prefix (for prior_lp)
    - 1 forward pass on the question alone (for nopre_lp)
    = 3 forward passes total, constant in length, no backwards.

Run:
    pip install torch transformers accelerate
    huggingface-cli login
    python verify_readout_generate.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Stop conditions for greedy generation.
MAX_NEW_TOKENS = 30


# ---------------------------------------------------------------------------
# Greedy generation that also records the log-prob of each emitted token
# under the FULL conditioning (question + answer-so-far). This is exactly
# full_lp(t) = log p(y_t | x, y_<t), so we get pass 1 for free.
# ---------------------------------------------------------------------------

def generate_with_logprobs(model, tokenizer, question, max_new_tokens=MAX_NEW_TOKENS):
    """
    Greedy-decode an answer to `question`. Return:
        question_ids : [Lq]
        answer_ids   : [La]
        full_lp      : [La]   log p(y_t | x, y_<t) for each generated token
        answer_text  : str
    Stops at EOS, newline, or max_new_tokens.
    """
    q_ids = tokenizer(question, return_tensors="pt",
                      add_special_tokens=False).input_ids[0].to(DEVICE)

    generated = []
    full_lp_list = []

    # Build the prefix incrementally. We could use KV-cache for speed; here
    # we re-run for clarity since the script is small.
    cur_ids = q_ids.clone()

    # Tokens that end the generation.
    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    # Newline often signals end of a one-line answer for base models.
    for nl in ["\n", "\n\n"]:
        nl_ids = tokenizer(nl, add_special_tokens=False).input_ids
        if len(nl_ids) == 1:
            stop_token_ids.add(nl_ids[0])

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(cur_ids.unsqueeze(0)).logits[0, -1]   # [V]
            log_probs = F.log_softmax(logits, dim=-1)
            next_id = int(log_probs.argmax().item())
            lp = log_probs[next_id].item()

            if next_id in stop_token_ids:
                break

            generated.append(next_id)
            full_lp_list.append(lp)
            cur_ids = torch.cat([cur_ids,
                                 torch.tensor([next_id], device=DEVICE)])

    if not generated:
        # Edge case: model immediately stopped. Return empty answer.
        return q_ids, torch.tensor([], dtype=torch.long, device=DEVICE), \
               torch.tensor([], device=DEVICE), ""

    a_ids = torch.tensor(generated, dtype=torch.long, device=DEVICE)
    full_lp = torch.tensor(full_lp_list, device=DEVICE)
    answer_text = tokenizer.decode(a_ids)
    return q_ids, a_ids, full_lp, answer_text


# ---------------------------------------------------------------------------
# The other two passes (unchanged from before).
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
    """log p(y_t | x) for each t, using one forward pass on the question."""
    with torch.no_grad():
        logits = model(question_ids.unsqueeze(0)).logits[0]
    next_dist = F.log_softmax(logits[-1], dim=-1)
    return next_dist[answer_ids]


# ---------------------------------------------------------------------------
# Top-level: generate, then trace back.
# ---------------------------------------------------------------------------

def generate_and_explain(model, tokenizer, question, top_k=3):
    """Generate an answer to `question` and identify the important tokens."""
    print("=" * 88)
    print(f"  Q: {question}")
    print("=" * 88)

    # Pass 1 (during generation): full_lp.
    q_ids, a_ids, full_lp, answer_text = generate_with_logprobs(model, tokenizer,
                                                                question)
    if a_ids.numel() == 0:
        print("  Model produced no answer.")
        return

    print(f"  Generated: {answer_text!r}")
    print()

    # Pass 2: prior_lp (no question).
    prior_lp = tf_logprobs(model, tokenizer,
                           torch.tensor([], dtype=torch.long, device=DEVICE),
                           a_ids)
    # Pass 3: nopre_lp (no answer-so-far).
    nopre_lp = no_prefix_logprobs(model, tokenizer, q_ids, a_ids)

    # Per-token quantities.
    d_input_inst = full_lp - prior_lp
    d_input_cum  = torch.cumsum(d_input_inst, dim=0)
    d_prefix     = full_lp - nopre_lp
    conf         = full_lp.exp()

    # Scores.
    setup = (torch.clamp(d_input_inst, min=0)
             * torch.clamp(d_prefix, min=0))
    decay = torch.sigmoid(-d_input_inst)
    readout = (torch.clamp(d_input_cum, min=0)
               * torch.clamp(d_prefix, min=0)
               * conf
               * decay)
    combined = torch.maximum(setup, readout)

    # Token strings.
    tokens = [tokenizer.decode([t]) for t in a_ids.tolist()]

    # Table.
    header = (f"{'idx':>3}  {'token':<14}  "
              f"{'Di_inst':>8}  {'Di_cum':>7}  {'D_pref':>7}  "
              f"{'conf':>6}  {'setup':>8}  {'readout':>8}  {'combined':>8}  "
              f"{'src':>5}")
    print(header)
    print("-" * len(header))
    for i, tok in enumerate(tokens):
        # Which score is dominant at this position?
        if setup[i].item() >= readout[i].item():
            src = "setup"
        else:
            src = "read"
        print(f"{i:>3}  {repr(tok):<14}  "
              f"{d_input_inst[i].item():>8.3f}  "
              f"{d_input_cum[i].item():>7.2f}  "
              f"{d_prefix[i].item():>7.2f}  "
              f"{conf[i].item():>6.3f}  "
              f"{setup[i].item():>8.3f}  "
              f"{readout[i].item():>8.3f}  "
              f"{combined[i].item():>8.3f}  "
              f"{src:>5}")

    # Top-k under the combined score.
    k = min(top_k, len(tokens))
    top_idx = torch.topk(combined, k=k).indices.tolist()
    top_idx_sorted = sorted(top_idx)
    print()
    print(f"  Top-{k} important tokens (by combined score):")
    for i in top_idx_sorted:
        print(f"    idx {i:>2}  {tokens[i]!r}  (combined={combined[i].item():.2f})")

    # Highlighted reconstruction.
    print()
    print("  Highlighted answer:")
    highlighted = ""
    flagged = set(top_idx)
    for i, tok in enumerate(tokens):
        if i in flagged:
            highlighted += f"[{tok.strip()}]"
        else:
            highlighted += tok
    print(f"    {highlighted}")
    print()


def main():
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    # Open-ended questions -- no reference answer provided.
    questions = [
        "What is the name of our planet?",
        "What is the capital of Vietnam?",
        "What is the largest planet in our solar system?",
        "Passage: The Treaty of Tordesillas was signed in 1494, "
        "dividing the New World between Spain and Portugal.\n"
        "Question: In what year was the Treaty of Tordesillas signed?\n"
        "Answer:",
        "What is 23 plus 45? Answer with just the number.",
        # A made-up fact -- model has no parametric knowledge.
        "Passage: The Compact of Yssaria was ratified in 1487 "
        "by the council of Mirentane.\n"
        "Question: When was the Compact of Yssaria ratified?\n"
        "Answer:",
    ]

    for q in questions:
        generate_and_explain(model, tokenizer, q, top_k=3)


if __name__ == "__main__":
    main()