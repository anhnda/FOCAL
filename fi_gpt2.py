"""
fi_gpt2.py
==========
Functional Information (FI) Attribution for decoder-only GPT-2 models.

Theory (Gat et al., ICML 2022)
-------------------------------
The Functional Fisher Information of feature x_i w.r.t. label y is:

    FI_i = E_{z ~ µ}[ (∇f_y(z)_i)^2 / f_y(z) ]                (standard)

With class covariance Σ (Theorem 4.1):

    FI_cov_i = E_{z ~ µ}[ (Σ∇f_y(z))_i · ∇f_y(z)_i / f_y(z) ] (correlated)

Where:
  - f_y(z) = softmax output for label y  (always > 0, so division is safe)
  - z ~ µ = N(x, σ²I)  (perturbed embedding; σ² controlled by `var_spread`)
  - gradient is taken w.r.t. token embeddings, L2-normed across the D dim

Decoder-only adaptation
-----------------------
Unlike classification, GPT-2 has no single f_y scalar. We define:

    f(z) = sum_{i in answer_positions} softmax_logit[answer_ids[i]]( model(z) )

This mirrors the PACE-grad target: the summed answer-token probabilities.
Using softmax probabilities (not logits) ensures f(z) > 0 for the FI
denominator. We use log-softmax + exp for numerical stability.

Pipeline
--------
1. Tokenise Q -> greedy-generate A (or use gold_answer).
2. Concatenate [Q | A] -> input_embed  [1, T, D].
3. For n perturbation draws:
   a. Sample z = input_embed + ε,  ε ~ N(0, σ²I)  (in embedding space)
   b. Forward pass -> softmax probabilities at answer positions
   c. f(z) = sum of p[answer_ids[i]] at answer_positions[i]
   d. Backward: grad = ∂f/∂z  [1, T, D]
   e. Accumulate  (grad² / f)  per token                 [standard FI]
      or          (Σ·grad · grad / f)                    [covariance FI]
4. Average over n draws.
5. L2-norm over embedding dim D -> scalar attribution per token  [T].

Return keys (compatible with xai_metrics_gpt2.py)
--------------------------------------------------
    tokens, q_len, answer_positions, answer_ids,
    attributions, input_embed, base_embed,
    logits_full, predicted_answer, model, tokenizer, time
"""

import time
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Reuse the model cache from paceg_gpt2 if co-located, otherwise define here
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
# Covariance helpers  (mirrors utilss.py from the FI paper)
# ---------------------------------------------------------------------------

def _estimate_covariance(
    embed: torch.Tensor,
    per_token: bool = False,
    regularise: float = 1e-4,
) -> torch.Tensor:
    """
    Estimate an embedding-space covariance matrix from a single sample
    (or a class of samples if called externally).

    For a single input we can only estimate an isotropic covariance
    (i.e., variance * I), since we have 1 sample and D dimensions.
    For a proper Σ, pass a batch of embeddings (B, T, D) and set
    per_token=True to get one D×D matrix per token position, or
    per_token=False for a single shared D×D matrix (the FI paper
    shares Σ across the embedding dimension for text).

    Parameters
    ----------
    embed      : Tensor  [B, T, D] or [T, D] or [D]
    per_token  : if True, return (T, D, D); else return (D, D)
    regularise : ridge added to diagonal for positive-definiteness

    Returns
    -------
    Σ          : Tensor  (T, D, D) or (D, D)
    """
    if embed.ndim == 1:
        embed = embed.unsqueeze(0).unsqueeze(0)          # [1,1,D]
    elif embed.ndim == 2:
        embed = embed.unsqueeze(0)                        # [1,T,D]
    # embed: [B, T, D]
    B, T, D = embed.shape

    def _cov(x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        if B < 2:
            # Fallback: identity scaled by per-dim variance
            var = x.var(dim=0).clamp(min=regularise)
            return torch.diag(var)
        xc = x - x.mean(dim=0, keepdim=True)
        cov = (xc.T @ xc) / (B - 1)
        cov += regularise * torch.eye(D, device=x.device, dtype=x.dtype)
        return cov

    if per_token:
        return torch.stack([_cov(embed[:, t, :]) for t in range(T)], dim=0)  # (T,D,D)
    else:
        flat = embed.reshape(B * T, D)
        return _cov(flat)   # (D, D)


def _cholesky_safe(cov: torch.Tensor) -> torch.Tensor:
    """Cholesky with automatic diagonal regularisation on failure."""
    try:
        return torch.linalg.cholesky(cov)
    except RuntimeError:
        d = torch.abs(torch.diag(cov)).clamp(min=1e-6)
        try:
            return torch.linalg.cholesky(cov + d.min() * torch.eye(cov.shape[0], device=cov.device))
        except RuntimeError:
            return torch.linalg.cholesky(cov + d.mean() * torch.eye(cov.shape[0], device=cov.device))


# ---------------------------------------------------------------------------
# Core FI attribution
# ---------------------------------------------------------------------------
def _build_base_embed(
    embed_layer: torch.nn.Embedding,
    input_embed: torch.Tensor,   # (1, T, D) — shape/dtype reference
    baseline: str,
    eos_token_id: int,
    device: str,
) -> torch.Tensor:
    """
    Build baseline embedding (1, T, D).
    baseline : 'zero' | 'pad' | 'mean'
    """
    if baseline == "zero":
        return torch.zeros_like(input_embed)
    elif baseline == "pad":
        pad_id  = torch.tensor([[eos_token_id]], device=device)
        pad_vec = embed_layer(pad_id).detach()             # (1, 1, D)
        return pad_vec.expand_as(input_embed).clone()
    elif baseline == "mean":
        mean_vec = embed_layer.weight.mean(dim=0, keepdim=True)   # (1, D)
        return mean_vec.unsqueeze(0).expand_as(input_embed).clone()
    else:
        raise ValueError(f"Unknown baseline '{baseline}'. Choose: zero | pad | mean")
def fi_gradient_gpt2(
    question: str,
    model_name: str = "gpt2",
    device: str = "cpu",
    n: int = 20,
    var_spread: float = 0.15,
    max_new_tokens: int = 30,
    gold_answer: Optional[str] = None,
    method: str = "fi",
    covariance: Optional[Union[torch.Tensor, np.ndarray]] = None,
    per_token_cov: bool = False,
    baseline: str = "zero",   
) -> dict:
    """
    Functional Information attribution for one (question, answer) pair on GPT-2.

    Parameters
    ----------
    question       : Full prompt string — "narrative Why did ...?"
    model_name     : HuggingFace GPT-2 identifier.
    device         : 'cpu' or 'cuda'.
    n              : Number of perturbation samples for Monte-Carlo estimate.
    var_spread     : Multiplier on estimated per-dim variance for noise σ².
                     (Mirrors the `var_spread` convention in the FI codebase.)
    max_new_tokens : Tokens to generate when gold_answer is None.
    gold_answer    : If provided, skip generation and use this string as A.
    method         : Attribution variant:
                       'fi'          -- standard FI: E[(∇f)²/f]           (default)
                       'smooth_grad' -- SmoothGrad:  E[∇f]
                       'smooth_grad_sq' -- SmoothGradSQ: E[(∇f)²]
                       'fi_cov'      -- covariance FI: E[(Σ∇f)·∇f/f]
    covariance     : Optional pre-computed covariance Σ for 'fi_cov'.
                     Shape (D, D) or (T, D, D).  If None and method='fi_cov',
                     Σ is estimated from the single input embedding.
    per_token_cov  : If True and estimating Σ, compute per-token Σ (T,D,D).

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
        a_enc    = tokenizer(gold_answer, return_tensors="pt", add_special_tokens=False)
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
    # 3. Build token embeddings  (same hook point as PACE-grad)
    # ------------------------------------------------------------------
    embed_layer = model.transformer.wte            # nn.Embedding [V, D]

    with torch.no_grad():
        input_embed = embed_layer(full_ids).detach()   # (1, T, D)

    # baseline embedding — replaces hardcoded zeros
    base_embed = _build_base_embed(
        embed_layer, input_embed, baseline,
        tokenizer.eos_token_id, device
    ).detach()

    # ------------------------------------------------------------------
    # 4. Reference logits (no grad)
    # ------------------------------------------------------------------
    with torch.no_grad():
        logits_full = model(inputs_embeds=input_embed).logits[0].detach()  # [T, V]

    # ------------------------------------------------------------------
    # 5. Estimate noise variance  σ² = var_spread * Var(embedding)
    #
    #    For a single sequence we estimate the variance across the T
    #    token positions (same spirit as the FI paper's var_spread trick).
    # ------------------------------------------------------------------
    D = input_embed.shape[-1]
    # Per-dimension variance across all token positions
    embed_flat = input_embed.squeeze(0)             # [T, D]
    sigma2_vec = embed_flat.var(dim=0) * var_spread # [D]  -- per-dim variance
    sigma2_vec = sigma2_vec.to(device)

    # ------------------------------------------------------------------
    # 5b. Covariance matrix for 'fi_cov'
    # ------------------------------------------------------------------
    if method == "fi_cov":
        if covariance is not None:
            if isinstance(covariance, np.ndarray):
                covariance = torch.tensor(covariance, dtype=input_embed.dtype)
            Sigma = covariance.to(device)
        else:
            # Estimate from this single input (rough but principled)
            Sigma = _estimate_covariance(
                input_embed.squeeze(0).unsqueeze(0),  # [1, T, D]
                per_token=per_token_cov,
            ).to(device)
            # Scale by var_spread (matches FI paper convention)
            Sigma = Sigma * var_spread

    # ------------------------------------------------------------------
    # 6. Monte-Carlo FI accumulation
    #
    #   For each draw:
    #     z = x + ε,  ε_d ~ N(0, σ²_d)   (independent per dimension)
    #     f(z) = Σ_{i in A} softmax(model(z))[position_i, answer_id_i]
    #     grad  = ∂f/∂z                   [1, T, D]
    #
    #   Accumulate:
    #     FI:          (grad² / f)         [1, T, D]
    #     FI_cov:      (Σ·grad · grad / f) [1, T, D]
    #     SmoothGrad:  grad                [1, T, D]
    #     SmoothGradSq:(grad²)             [1, T, D]
    # ------------------------------------------------------------------
    accumulated = torch.zeros_like(input_embed)   # [1, T, D]

    for _ in range(n):
        # Perturb in embedding space
        noise   = torch.randn_like(input_embed) * sigma2_vec.sqrt()
        z       = (input_embed + noise).requires_grad_(True)

        # Forward
        logits  = model(inputs_embeds=z).logits   # [1, T, V]

        # f(z): sum of softmax probabilities at answer token positions
        # Using log_softmax for stability; exp to recover probability
        log_probs = F.log_softmax(logits[0], dim=-1)   # [T, V]
        f_val = sum(
            log_probs[answer_positions[i], answer_ids[i]].exp()
            for i in range(La)
        )                                               # scalar >= 0

        # Guard against numerically-zero f (shouldn't happen for non-trivial answers)
        f_val = f_val.clamp(min=1e-8)

        model.zero_grad()
        f_val.backward()

        grad = z.grad.detach()                          # [1, T, D]

        if method == "fi":
            # E[(∇f)² / f]
            accumulated = accumulated + (grad ** 2) / f_val.detach()

        elif method == "smooth_grad":
            # E[∇f]
            accumulated = accumulated + grad

        elif method == "smooth_grad_sq":
            # E[(∇f)²]
            accumulated = accumulated + (grad ** 2)

        elif method == "fi_cov":
            # E[(Σ∇f)·∇f / f]
            # grad shape: [1, T, D]; Sigma shape: (D,D) or (T,D,D)
            g = grad.squeeze(0)   # [T, D]
            if Sigma.ndim == 2:
                # shared covariance: Σg_t for each token
                sigma_g = (Sigma @ g.unsqueeze(-1)).squeeze(-1)  # [T, D]
            else:
                # per-token covariance: Σ_t g_t
                sigma_g = torch.einsum("tij,tj->ti", Sigma, g)    # [T, D]
            fi_cov_t = (sigma_g * g) / f_val.detach()             # [T, D]
            accumulated = accumulated + fi_cov_t.unsqueeze(0)

        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose: fi | fi_cov | smooth_grad | smooth_grad_sq"
            )

    # Average over n draws
    mean_attr = accumulated / n                    # [1, T, D]

    # L2-norm over embedding dim D -> scalar per token  [T]
    attributions = mean_attr.norm(dim=-1).squeeze(0).cpu()   # [T]

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
        "time":             time.time() - t0,
    }