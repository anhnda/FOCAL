"""
soft_faith.py
=============
Soft Normalized Sufficiency (Soft-NS) and Soft Normalized Comprehensiveness
(Soft-NC) metrics from:

  Zhao & Aletras, "Incorporating Attribution Importance for Improving
  Faithfulness Metrics", ACL 2023.
  https://github.com/casszhao/SoftFaith

nn_forward_func signature (bert_helper / distilbert_helper / roberta_helper):
    nn_forward_func(model, input_embed, attention_mask=None,
                    position_embed=None, type_embed=None, ...)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_scores(attr: torch.Tensor) -> torch.Tensor:
    """Map attribution scores to [0, 1] via min-max normalisation."""
    a_min = attr.min()
    a_max = attr.max()
    if (a_max - a_min).abs() < 1e-9:
        return torch.full_like(attr, 0.5)
    return (attr - a_min) / (a_max - a_min)


def _call_forward(
    nn_forward_func: Callable,
    model,
    input_embed: torch.Tensor,
    position_embed: torch.Tensor,
    type_embed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Call nn_forward_func with keyword args matching bert_helper signature."""
    return nn_forward_func(
        model,
        input_embed,
        attention_mask=attention_mask,
        position_embed=position_embed,
        type_embed=type_embed,
    )


def _get_predicted_class_and_prob(
    nn_forward_func: Callable,
    model,
    input_embed: torch.Tensor,
    position_embed: torch.Tensor,
    type_embed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
):
    """Return (pred_class, p_full) on the unperturbed input."""
    with torch.no_grad():
        logits     = _call_forward(nn_forward_func, model, input_embed, position_embed, type_embed, attention_mask)
        probs      = F.softmax(logits, dim=-1)
        pred_class = int(probs.argmax(dim=-1).item())
        p_full     = float(probs[0, pred_class].item())
    return pred_class, p_full


def _get_prob_from_embed(
    nn_forward_func: Callable,
    model,
    perturbed_embed: torch.Tensor,
    position_embed: torch.Tensor,
    type_embed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    pred_class: int,
) -> float:
    """Return p(pred_class) given a perturbed token embedding."""
    with torch.no_grad():
        logits = _call_forward(nn_forward_func, model, perturbed_embed, position_embed, type_embed, attention_mask)
        probs  = F.softmax(logits, dim=-1)
        return float(probs[0, pred_class].item())


def _baseline_prob(
    nn_forward_func: Callable,
    model,
    input_embed: torch.Tensor,
    position_embed: torch.Tensor,
    type_embed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    base_token_emb: Optional[torch.Tensor],
    pred_class: int,
) -> float:
    """p(pred_class) on a zeroed / baseline sequence — S(X, ŷ, 0)."""
    seq_len = input_embed.shape[1]
    if base_token_emb is not None:
        # base_token_emb: (1, d) -> (1, seq_len, d)
        zero_embed = base_token_emb.unsqueeze(0).expand(
            1, seq_len, -1
        ).to(input_embed.device)
    else:
        zero_embed = torch.zeros_like(input_embed)

    return _get_prob_from_embed(
        nn_forward_func, model,
        zero_embed, position_embed, type_embed, attention_mask,
        pred_class,
    )


# ---------------------------------------------------------------------------
# Soft perturbation  (Eq. 3 in the paper)
# ---------------------------------------------------------------------------

def soft_input_perturbation(
    token_embeddings: torch.Tensor,
    attr_scores: torch.Tensor,
    mode: str = "sufficiency",
) -> torch.Tensor:
    """Apply per-token Bernoulli dropout to token embeddings.

    Parameters
    ----------
    token_embeddings : Tensor, shape (1, seq_len, hidden_dim)
    attr_scores      : Tensor, shape (seq_len,) — raw attribution scores
    mode             : "sufficiency"       -> q = a_i  (retain ∝ importance)
                       "comprehensiveness" -> q = 1-a_i (remove ∝ importance)

    Returns
    -------
    Tensor, shape (1, seq_len, hidden_dim)
    """
    assert mode in ("sufficiency", "comprehensiveness")

    scores = _normalize_scores(attr_scores.float())   # (seq_len,) in [0, 1]
    q      = scores if mode == "sufficiency" else 1.0 - scores

    device = token_embeddings.device
    # Bernoulli mask: (seq_len,) -> (1, seq_len, 1) broadcasts over hidden dim
    mask = torch.bernoulli(q.to(device))              # (seq_len,)
    mask = mask.unsqueeze(0).unsqueeze(-1)            # (1, seq_len, 1)

    return token_embeddings.detach() * mask


# ---------------------------------------------------------------------------
# Soft-NS
# ---------------------------------------------------------------------------

def calculate_soft_sufficiency(
    nn_forward_func: Callable,
    model,
    input_embed: torch.Tensor,
    position_embed: torch.Tensor,
    type_embed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    attr_full: torch.Tensor,
    base_token_emb: Optional[torch.Tensor] = None,
    n_samples: int = 10,
) -> float:
    """Soft Normalised Sufficiency (Soft-NS).  Eq. 4 of Zhao & Aletras 2023.

        Soft-S  = 1 - max(0, p(ŷ|X) - p(ŷ|X'))
        Soft-NS = (Soft-S - S(X,ŷ,0)) / (1 - S(X,ŷ,0))
    """
    pred_class, p_full = _get_predicted_class_and_prob(
        nn_forward_func, model,
        input_embed, position_embed, type_embed, attention_mask,
    )

    p_base = _baseline_prob(
        nn_forward_func, model,
        input_embed, position_embed, type_embed, attention_mask,
        base_token_emb, pred_class,
    )
    s_base = 1.0 - max(0.0, p_full - p_base)
    denom  = 1.0 - s_base
    if abs(denom) < 1e-9:
        return 0.0

    soft_s_vals = []
    for _ in range(n_samples):
        x_prime = soft_input_perturbation(input_embed, attr_full, mode="sufficiency")
        p_prime = _get_prob_from_embed(
            nn_forward_func, model,
            x_prime, position_embed, type_embed, attention_mask,
            pred_class,
        )
        soft_s_vals.append(1.0 - max(0.0, p_full - p_prime))

    soft_s = float(np.mean(soft_s_vals))
    return (soft_s - s_base) / denom


# ---------------------------------------------------------------------------
# Soft-NC
# ---------------------------------------------------------------------------

def calculate_soft_comprehensiveness(
    nn_forward_func: Callable,
    model,
    input_embed: torch.Tensor,
    position_embed: torch.Tensor,
    type_embed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    attr_full: torch.Tensor,
    base_token_emb: Optional[torch.Tensor] = None,
    n_samples: int = 10,
) -> float:
    """Soft Normalised Comprehensiveness (Soft-NC).  Eq. 5 of Zhao & Aletras 2023.

        Soft-C  = max(0, p(ŷ|X) - p(ŷ|X'))
        Soft-NC = Soft-C / (1 - S(X,ŷ,0))
    """
    pred_class, p_full = _get_predicted_class_and_prob(
        nn_forward_func, model,
        input_embed, position_embed, type_embed, attention_mask,
    )

    p_base = _baseline_prob(
        nn_forward_func, model,
        input_embed, position_embed, type_embed, attention_mask,
        base_token_emb, pred_class,
    )
    s_base = 1.0 - max(0.0, p_full - p_base)
    denom  = 1.0 - s_base
    if abs(denom) < 1e-9:
        return 0.0

    soft_c_vals = []
    for _ in range(n_samples):
        x_prime = soft_input_perturbation(input_embed, attr_full, mode="comprehensiveness")
        p_prime = _get_prob_from_embed(
            nn_forward_func, model,
            x_prime, position_embed, type_embed, attention_mask,
            pred_class,
        )
        soft_c_vals.append(max(0.0, p_full - p_prime))

    soft_c = float(np.mean(soft_c_vals))
    return soft_c / denom


# ---------------------------------------------------------------------------
# Soft log-odds
# ---------------------------------------------------------------------------

def calculate_soft_log_odds(
    nn_forward_func: Callable,
    model,
    input_embed: torch.Tensor,
    position_embed: torch.Tensor,
    type_embed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    attr_full: torch.Tensor,
    base_token_emb: Optional[torch.Tensor] = None,
    n_samples: int = 10,
) -> float:
    """Soft log-odds: mean drop in log-probability after soft erasure."""
    pred_class, p_full = _get_predicted_class_and_prob(
        nn_forward_func, model,
        input_embed, position_embed, type_embed, attention_mask,
    )

    eps = 1e-9
    log_odds_vals = []
    for _ in range(n_samples):
        x_prime = soft_input_perturbation(input_embed, attr_full, mode="comprehensiveness")
        p_prime = _get_prob_from_embed(
            nn_forward_func, model,
            x_prime, position_embed, type_embed, attention_mask,
            pred_class,
        )
        lo = (np.log((p_full  + eps) / (1 - p_full  + eps))
            - np.log((p_prime + eps) / (1 - p_prime + eps)))
        log_odds_vals.append(lo)

    return float(np.mean(log_odds_vals))