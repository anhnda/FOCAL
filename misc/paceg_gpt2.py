"""
paceg_gpt2.py
=============
PACE Gradient Attribution for decoder-only GPT-2 models.

Pipeline (per TellMeWhy sample)
--------------------------------
1.  Tokenise Q  ->  greedy-generate A  (or use gold_answer directly).
2.  Concatenate [Q | A] into a single sequence — one forward pass only.
    GPT-2's causal mask is enforced internally; no extra mask needed.
3.  Integrate gradients of the *answer-token logit sum* w.r.t. the token
    embedding, along a linear path from a chosen baseline to the real input
    (zero | pad | mean — controlled by `baseline` parameter).
4.  L2-norm over the embedding dimension gives a scalar attribution per token.

Return keys (consumed by xai_metrics_gpt2.py and run_eval_pg_gpt2.py)
-----------------------------------------------------------------------
    tokens           : list[str]     -- all tokens in [Q | A]
    q_len            : int           -- number of Q tokens
    answer_positions : list[int]     -- indices of A tokens in full sequence
    answer_ids       : Tensor [La]   -- answer token ids (CPU)
    attributions     : Tensor [T]    -- L2-norm attribution per token (CPU)
    input_embed      : Tensor[1,T,D] -- original embedding (CPU, detached)
    base_embed       : Tensor[1,T,D] -- baseline embedding (CPU)
    logits_full      : Tensor[T,V]   -- reference logits (CPU)
    predicted_answer : str           -- decoded answer string
    model            : GPT2LMHeadModel
    tokenizer        : GPT2TokenizerFast
    time             : float         -- wall-clock seconds
"""

import time
from typing import Optional

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ---------------------------------------------------------------------------
# Module-level model cache  (load once per process)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Core attribution function
# ---------------------------------------------------------------------------
# in paceg_gpt2.py — add this function
def _build_base_embed(
    embed_layer: torch.nn.Embedding,
    input_embed: torch.Tensor,   # (1, T, D) or (1, 1, D) — shape reference
    baseline: str,
    eos_token_id: int,
    device: str,
) -> torch.Tensor:
    # embed_layer lives on the model's device, not necessarily `device` arg
    embed_device = next(embed_layer.parameters()).device

    if baseline == "zero":
        return torch.zeros_like(input_embed)
    elif baseline == "pad":
        pad_id  = torch.tensor([[eos_token_id]], device=embed_device)
        pad_vec = embed_layer(pad_id).detach().cpu()   # always return CPU
        return pad_vec.expand_as(input_embed).clone()
    elif baseline == "mean":
        mean_vec = embed_layer.weight.mean(dim=0, keepdim=True).detach().cpu()  # CPU
        return mean_vec.unsqueeze(0).expand_as(input_embed).clone()
    else:
        raise ValueError(f"Unknown baseline '{baseline}'. Choose: zero | pad | mean")
def pace_gradient_gpt2(
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
    Run PACE Gradient attribution for one (question, answer) pair.

    Parameters
    ----------
    question       : Full prompt string -- "narrative Why did ...?"
    model_name     : HuggingFace GPT-2 identifier.
    device         : 'cpu' or 'cuda'.
    steps          : Number of Riemann-sum steps.
    a, b           : Integration interval endpoints (default 0 -> 1).
    max_new_tokens : Tokens to generate when gold_answer is None.
    gold_answer    : If provided, skip generation and use this string as A.
    baseline       : IG baseline type. One of:
                       'zero' -- all-zero embedding vector (default, standard IG)
                       'pad'  -- EOS/pad token embedding repeated across sequence
                       'mean' -- mean of entire vocabulary embedding matrix

    Returns
    -------
    dict -- see module docstring for the full key list.
    """
    t0 = time.time()
    model, tokenizer = get_model_tokenizer(model_name, device)

    # ------------------------------------------------------------------
    # 1. Tokenise Q -- always build an explicit attention_mask.
    #    GPT-2 sets pad_token = eos_token, so HuggingFace cannot infer
    #    the mask automatically and emits a warning without it.
    # ------------------------------------------------------------------
    q_enc       = tokenizer(question, return_tensors="pt", add_special_tokens=True)
    q_ids       = q_enc["input_ids"].to(device)        # [1, Lq]
    q_attn_mask = q_enc["attention_mask"].to(device)   # [1, Lq]  all-ones
    q_len       = q_ids.shape[1]

    # ------------------------------------------------------------------
    # 2. Get answer token ids -- generate or use gold
    # ------------------------------------------------------------------
    if gold_answer is not None:
        # add_special_tokens=False avoids a spurious leading BOS token
        a_enc    = tokenizer(gold_answer, return_tensors="pt",
                             add_special_tokens=False)
        a_ids_d  = a_enc["input_ids"].to(device)       # [1, La]
        full_ids = torch.cat([q_ids, a_ids_d], dim=1)  # [1, Lq+La]
    else:
        with torch.no_grad():
            full_ids = model.generate(
                q_ids,
                attention_mask=q_attn_mask,            # suppresses the warning
                max_new_tokens=max_new_tokens,
                do_sample=False,                       # greedy -> deterministic
                pad_token_id=tokenizer.eos_token_id,
            )                                          # [1, Lq+La]

    T  = full_ids.shape[1]
    La = T - q_len

    if La == 0:
        raise ValueError(
            "Answer is empty -- increase max_new_tokens or supply gold_answer."
        )

    answer_positions = list(range(q_len, T))
    answer_ids       = full_ids[0, q_len:].cpu()       # [La]

    # ------------------------------------------------------------------
    # 3. Build token embeddings
    #    Hook point: model.transformer.wte (word-token embedding layer).
    #    GPT-2 adds positional embeddings *inside* its transformer blocks,
    #    so interpolating here operates in pure token-embedding space --
    #    the correct convention for IG / PACE.
    # ------------------------------------------------------------------
    embed_layer = model.transformer.wte               # nn.Embedding [V, D]

    with torch.no_grad():
        input_embed = embed_layer(full_ids).detach()  # [1, T, D]

    # ------------------------------------------------------------------
    # Baseline embedding  (IG reference point)
    #
    #   zero : all-zero vector — standard IG (Sundararajan et al. 2017)
    #          out-of-distribution for GPT-2 but mathematically clean
    #   pad  : EOS token embedding broadcast across all positions
    #          in-distribution; makes PACE/ReAGent perturbation spaces
    #          more comparable (both use EOS as "absent token")
    #   mean : mean of the full vocabulary embedding matrix
    #          semantically neutral, in-distribution, no token bias
    # ------------------------------------------------------------------
    if baseline == "zero":
        base_embed = torch.zeros_like(input_embed)
    elif baseline == "pad":
        pad_id     = torch.tensor([[tokenizer.eos_token_id]], device=device)
        pad_vec    = embed_layer(pad_id).detach()          # [1, 1, D]
        base_embed = pad_vec.expand_as(input_embed).clone()
    elif baseline == "mean":
        mean_vec   = embed_layer.weight.mean(dim=0, keepdim=True)  # [1, D]
        base_embed = mean_vec.unsqueeze(0).expand_as(input_embed).clone()
    else:
        raise ValueError(f"Unknown baseline '{baseline}'. Choose: zero | pad | mean")

    base_embed = base_embed.detach()
    delta      = input_embed - base_embed             # [1, T, D]

    # ------------------------------------------------------------------
    # 4. Reference logits (no grad -- for inspection / log-odds)
    # ------------------------------------------------------------------
    with torch.no_grad():
        logits_full = model(
            inputs_embeds=input_embed
        ).logits[0].detach()                          # [T, V]

    # ------------------------------------------------------------------
    # 5. PACE Riemann-sum integration
    #
    #    f(alpha) = sum_{i in answer_positions} logit[answer_ids[i]]( model(x_a) )
    #    x_a      = base_embed + alpha * delta
    #
    #    IG  = (sum_a  df/dx_a) / steps  *  delta        [1, T, D]
    # ------------------------------------------------------------------
    alphas            = torch.linspace(a, b, steps, device=device)
    accumulated_grads = torch.zeros_like(input_embed)  # [1, T, D]

    for alpha in alphas:
        interp = (base_embed + alpha * delta).requires_grad_(True)

        logits = model(inputs_embeds=interp).logits    # [1, T, V]

        # Scalar target: sum answer-token logits at their positions
        score = sum(
            logits[0, answer_positions[i], answer_ids[i]]
            for i in range(La)
        )

        model.zero_grad()
        score.backward()

        accumulated_grads = accumulated_grads + interp.grad.detach()

    # Integrated gradients  [1, T, D]
    ig = (accumulated_grads / steps) * delta

    # L2-norm over embedding dim D -> scalar per token  [T]
    attributions = ig.norm(dim=-1).squeeze(0).cpu()   # [T]

    # ------------------------------------------------------------------
    # 6. Decode tokens for readability
    # ------------------------------------------------------------------
    tokens = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())

    return {
        "tokens":           tokens,
        "q_len":            q_len,
        "answer_positions": answer_positions,
        "answer_ids":       answer_ids,               # [La] CPU
        "attributions":     attributions,             # [T]  CPU
        "input_embed":      input_embed.cpu(),        # [1,T,D] CPU
        "base_embed":       base_embed.cpu(),         # [1,T,D] CPU
        "logits_full":      logits_full.cpu(),        # [T,V]  CPU
        "predicted_answer": tokenizer.decode(
                                answer_ids.tolist(),
                                skip_special_tokens=True),
        "model":            model,
        "tokenizer":        tokenizer,
        "time":             time.time() - t0,
    }