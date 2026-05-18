"""
ig_gpt2.py
==========
Vanilla Integrated Gradients attribution for decoder-only GPT-2 models.

Differences from paceg_gpt2.py (PACE):
  1. Gradient taken w.r.t. interpolated embeddings directly (not coef scalars)
  2. No delta weighting — pure mean gradient over path (no * delta term)
  3. Final attribution = L2 norm over embedding dim D per token

Pipeline
--------
1. Tokenise Q -> greedy-generate A (or use gold_answer).
2. Concatenate [Q | A] -> input_embed [1, T, D].
3. Build interpolation path: X_inter[s] = base + t_s * (X - base)
4. Single batched forward pass over all steps.
5. Grad w.r.t. X_inter directly — shape (steps, T, D).
6. Mean over steps -> (T, D); L2 norm over D -> (T,).

Return keys (compatible with xai_metrics_gpt2.py and run_eval_pg_gpt2.py)
--------------------------------------------------------------------------
    tokens, q_len, answer_positions, answer_ids,
    attributions, input_embed, base_embed,
    logits_full, predicted_answer, model, tokenizer, time
"""

import time
from typing import Optional

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

_CACHE: dict = {}

def get_model_tokenizer(
    model_name: str = "gpt2",
    device: str = "cpu",
):
    """Return (model, tokenizer), loading from HuggingFace only once."""
    device = str(torch.device(device))   # normalizes "cuda" -> "cuda:0"
    key = (model_name, device)           # no `type` — GPT-2 cache needs only these two
    if key not in _CACHE:
        tok = GPT2TokenizerFast.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        mdl = GPT2LMHeadModel.from_pretrained(model_name)
        mdl.eval().to(device)
        _CACHE[key] = (mdl, tok)
    return _CACHE[key]


def ig_gpt2(
    question: str,
    model_name: str = "gpt2",
    device: str = "cpu",
    steps: int = 100,
    a: float = 0.0,
    b: float = 1.0,
    max_new_tokens: int = 30,
    gold_answer: Optional[str] = None,
    baseline: str = "zero",
) -> dict:
    """
    Vanilla Integrated Gradients attribution for one (question, answer) pair.

    Parameters
    ----------
    question       : Full prompt string.
    model_name     : HuggingFace GPT-2 identifier.
    device         : 'cpu' or 'cuda'.
    steps          : Number of Riemann-sum steps.
    a, b           : Integration interval endpoints (default 0 -> 1).
    max_new_tokens : Tokens to generate when gold_answer is None.
    gold_answer    : If provided, skip generation and use this string as A.
    baseline       : IG baseline type. One of:
                       'zero' -- all-zero embedding vector (default)
                       'pad'  -- EOS/pad token embedding broadcast
                       'mean' -- mean of vocabulary embedding matrix

    Returns
    -------
    dict -- same keys as pace_gradient_gpt2 for drop-in metric compatibility.
    """
    t0 = time.time()
    model, tokenizer = get_model_tokenizer(model_name, device)

    # ------------------------------------------------------------------
    # 1. Tokenise Q
    # ------------------------------------------------------------------
    q_enc       = tokenizer(question, return_tensors="pt", add_special_tokens=True)
    q_ids       = q_enc["input_ids"].to(device)
    q_attn_mask = q_enc["attention_mask"].to(device)
    q_len       = q_ids.shape[1]

    # ------------------------------------------------------------------
    # 2. Get answer token ids
    # ------------------------------------------------------------------
    if gold_answer is not None:
        a_enc    = tokenizer(gold_answer, return_tensors="pt",
                             add_special_tokens=False)
        a_ids_d  = a_enc["input_ids"].to(device)
        full_ids = torch.cat([q_ids, a_ids_d], dim=1)
    else:
        with torch.no_grad():
            full_ids = model.generate(
                q_ids,
                attention_mask=q_attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    T  = full_ids.shape[1]
    La = T - q_len

    if La == 0:
        raise ValueError(
            "Answer is empty — increase max_new_tokens or supply gold_answer."
        )

    answer_positions = list(range(q_len, T))
    answer_ids       = full_ids[0, q_len:].cpu()   # [La]

    # ------------------------------------------------------------------
    # 3. Build token embeddings
    # ------------------------------------------------------------------
    embed_layer = model.transformer.wte            # nn.Embedding [V, D]

    with torch.no_grad():
        input_embed = embed_layer(full_ids).detach()   # [1, T, D]

    # ------------------------------------------------------------------
    # 4. Baseline embedding
    # ------------------------------------------------------------------
    if baseline == "zero":
        base_embed = torch.zeros_like(input_embed)
    elif baseline == "pad":
        pad_id     = torch.tensor([[tokenizer.eos_token_id]], device=device)
        pad_vec    = embed_layer(pad_id).detach()          # [1, 1, D]
        base_embed = pad_vec.expand_as(input_embed).clone()
    elif baseline == "mean":
        mean_vec   = embed_layer.weight.mean(dim=0, keepdim=True)   # [1, D]
        base_embed = mean_vec.unsqueeze(0).expand_as(input_embed).clone()
    else:
        raise ValueError(f"Unknown baseline '{baseline}'. Choose: zero | pad | mean")

    base_embed = base_embed.detach()

    # ------------------------------------------------------------------
    # 5. Reference logits
    # ------------------------------------------------------------------
    with torch.no_grad():
        logits_full = model(
            inputs_embeds=input_embed
        ).logits[0].detach()   # [T, V]

    # ------------------------------------------------------------------
    # 6. Batched interpolation — grad leaf is X_inter directly
    #
    #    X_inter[s] = base + t_s * (X - base)   shape (steps, T, D)
    #
    #    Score: sum of answer-token logits at their positions
    #    Grad:  d(score)/d(X_inter) — shape (steps, T, D)
    #    IG:    mean over steps (no delta multiplication)
    #    Attr:  L2 norm over D -> (T,)
    # ------------------------------------------------------------------
    t_vals  = torch.linspace(a, b, steps, device=device)   # (steps,)
    X_src   = input_embed.squeeze(0)    # (T, D)
    X_base  = base_embed.squeeze(0)     # (T, D)

    # (steps, T, D) — batched interpolation
    X_inter = (
        X_base.unsqueeze(0)
        + t_vals.view(steps, 1, 1) * (X_src - X_base).unsqueeze(0)
    ).requires_grad_(True)

    start_time = time.perf_counter()

    # GPT-2 expects (batch, seq, D); here batch=steps, seq=T
    logits_batch = model(inputs_embeds=X_inter).logits   # (steps, T, V)

    # Score: sum answer-token logits at their positions, summed over steps
    score = sum(
        logits_batch[:, answer_positions[i], answer_ids[i]].sum()
        for i in range(La)
    )

    (grad_embed,) = torch.autograd.grad(score, X_inter)   # (steps, T, D)

    end_time = time.perf_counter()

    # Pure IG: mean gradient over steps — no delta weighting
    mean_grad    = grad_embed.mean(dim=0)                  # (T, D)

    # L2 norm over embedding dim D -> scalar per token
    attributions = mean_grad.norm(dim=-1).detach().cpu()   # (T,)

    # ------------------------------------------------------------------
    # 7. Decode tokens
    # ------------------------------------------------------------------
    tokens = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())

    return {
        "tokens":           tokens,
        "q_len":            q_len,
        "answer_positions": answer_positions,
        "answer_ids":       answer_ids,
        "attributions":     attributions,
        "input_embed":      input_embed.cpu(),
        "base_embed":       base_embed.cpu(),
        "logits_full":      logits_full.cpu(),
        "predicted_answer": tokenizer.decode(
                                answer_ids.tolist(),
                                skip_special_tokens=True),
        "model":            model,
        "tokenizer":        tokenizer,
        "time":             end_time - start_time,
    }