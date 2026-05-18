"""
Generate-then-analyze interface for the readout-point method.
INSTRUCT MODEL VERSION.

Key differences from the base-model version:
    1. Uses meta-llama/Llama-3.2-1B-Instruct.
    2. Applies the chat template so the model receives the question as
       a chat turn rather than text to continue.
    3. Stops on the instruct-model's end-of-turn tokens (<|eot_id|>).

The XAI analysis is unchanged. The "question" for the analysis is the
full templated chat prompt up to where the assistant begins generating.
The "answer" is whatever the assistant generated up to <|eot_id|>.

Run:
    pip install torch transformers accelerate
    huggingface-cli login
    python verify_readout_instruct.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Instruct models can produce concise answers; allow some room but cap.
MAX_NEW_TOKENS = 60


# ---------------------------------------------------------------------------
# Build the chat prompt and identify stop tokens for an instruct model
# ---------------------------------------------------------------------------

def build_chat_prompt(tokenizer, user_message):
    """
    Apply the chat template and tokenize. Returns the input_ids
    representing the full prompt up to where the assistant should
    begin generating.
    """
    messages = [{"role": "user", "content": user_message}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(prompt_text, return_tensors="pt",
                    add_special_tokens=False).input_ids[0].to(DEVICE)
    return ids, prompt_text


def get_stop_token_ids(tokenizer):
    """Tokens that should terminate generation for the instruct model."""
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    # Llama-3 instruct uses <|eot_id|> to end a turn.
    for special in ["<|eot_id|>", "<|end_of_text|>"]:
        tid = tokenizer.convert_tokens_to_ids(special)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)
    return stop_ids


# ---------------------------------------------------------------------------
# Greedy generation, recording log p(y_t | x, y_<t) for each emitted token
# ---------------------------------------------------------------------------

def generate_with_logprobs(model, tokenizer, prompt_ids,
                            stop_token_ids, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy decode. Return (answer_ids, full_lp, answer_text)."""
    generated, full_lp_list = [], []
    cur_ids = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(cur_ids.unsqueeze(0)).logits[0, -1]
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
        return (torch.tensor([], dtype=torch.long, device=DEVICE),
                torch.tensor([], device=DEVICE), "")

    a_ids = torch.tensor(generated, dtype=torch.long, device=DEVICE)
    full_lp = torch.tensor(full_lp_list, device=DEVICE)
    answer_text = tokenizer.decode(a_ids)
    return a_ids, full_lp, answer_text


# ---------------------------------------------------------------------------
# Teacher-forced log-probs for the prior and no-prefix passes
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


def no_prefix_logprobs(model, tokenizer, prompt_ids, answer_ids):
    """log p(y_t | x) for each t -- single forward pass on prompt only."""
    with torch.no_grad():
        logits = model(prompt_ids.unsqueeze(0)).logits[0]
    next_dist = F.log_softmax(logits[-1], dim=-1)
    return next_dist[answer_ids]


# ---------------------------------------------------------------------------
# IMPORTANT: For the PRIOR pass, we want p(y_t | y_<t) without ANY
# conditioning on the question. For an instruct model, this should be
# an unconditional language-model pass starting from BOS. We do NOT
# wrap the answer in chat template tags, because we want the actual
# token sequence the model generated, scored under its base LM behavior.
# ---------------------------------------------------------------------------

def prior_logprobs(model, tokenizer, answer_ids):
    """log p(y_t | y_<t) starting from BOS, no chat template."""
    return tf_logprobs(model, tokenizer,
                       torch.tensor([], dtype=torch.long, device=DEVICE),
                       answer_ids)


# ---------------------------------------------------------------------------
# Top-level: generate, then analyze
# ---------------------------------------------------------------------------

def generate_and_explain(model, tokenizer, user_message, stop_token_ids,
                         top_k=3):
    print("=" * 88)
    print(f"  USER: {user_message}")
    print("=" * 88)

    prompt_ids, _ = build_chat_prompt(tokenizer, user_message)

    # Pass 1 (during generation): full_lp.
    a_ids, full_lp, answer_text = generate_with_logprobs(
        model, tokenizer, prompt_ids, stop_token_ids)
    if a_ids.numel() == 0:
        print("  Model produced no answer.")
        return

    print(f"  ASSISTANT: {answer_text!r}")
    print()

    # Pass 2: prior_lp (BOS only, no question, no chat template).
    pri_lp = prior_logprobs(model, tokenizer, a_ids)
    # Pass 3: nopre_lp (full templated prompt, no answer-so-far).
    nopre_lp = no_prefix_logprobs(model, tokenizer, prompt_ids, a_ids)

    # Per-token signals.
    d_input_inst = full_lp - pri_lp
    d_input_cum  = torch.cumsum(d_input_inst, dim=0)
    d_prefix     = full_lp - nopre_lp
    conf         = full_lp.exp()

    setup = (torch.clamp(d_input_inst, min=0)
             * torch.clamp(d_prefix, min=0))
    decay = torch.sigmoid(-d_input_inst)
    readout = (torch.clamp(d_input_cum, min=0)
               * torch.clamp(d_prefix, min=0)
               * conf
               * decay)
    combined = torch.maximum(setup, readout)

    tokens = [tokenizer.decode([t]) for t in a_ids.tolist()]

    header = (f"{'idx':>3}  {'token':<14}  "
              f"{'Di_inst':>8}  {'Di_cum':>7}  {'D_pref':>7}  "
              f"{'conf':>6}  {'setup':>8}  {'readout':>8}  {'combined':>8}  "
              f"{'src':>5}")
    print(header)
    print("-" * len(header))
    for i, tok in enumerate(tokens):
        src = "setup" if setup[i].item() >= readout[i].item() else "read"
        print(f"{i:>3}  {repr(tok):<14}  "
              f"{d_input_inst[i].item():>8.3f}  "
              f"{d_input_cum[i].item():>7.2f}  "
              f"{d_prefix[i].item():>7.2f}  "
              f"{conf[i].item():>6.3f}  "
              f"{setup[i].item():>8.3f}  "
              f"{readout[i].item():>8.3f}  "
              f"{combined[i].item():>8.3f}  "
              f"{src:>5}")

    k = min(top_k, len(tokens))
    top_idx = torch.topk(combined, k=k).indices.tolist()
    top_idx_sorted = sorted(top_idx)
    print()
    print(f"  Top-{k} important tokens (by combined score):")
    for i in top_idx_sorted:
        print(f"    idx {i:>2}  {tokens[i]!r}  (combined={combined[i].item():.2f})")

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
        # Open-ended factual.
        "What is the name of our planet?",
        "What is the capital of Vietnam?",
        "What is the largest planet in our solar system?",

        # Context extraction with a real fact (model may also know).
        "Passage: The Treaty of Tordesillas was signed in 1494, "
        "dividing the New World between Spain and Portugal.\n"
        "Question: In what year was the Treaty of Tordesillas signed? "
        "Answer in one short sentence.",

        # Arithmetic.
        "What is 23 plus 45? Answer with just the number.",

        # Made-up fact: the model has no parametric knowledge, so the
        # answer MUST come from the passage.
        "Passage: The Compact of Yssaria was ratified in 1487 "
        "by the council of Mirentane.\n"
        "Question: When was the Compact of Yssaria ratified? "
        "Answer in one short sentence.",
    ]

    for q in questions:
        generate_and_explain(model, tokenizer, q, stop_token_ids, top_k=3)


if __name__ == "__main__":
    main()