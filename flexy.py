"""
flexi.py — Flexible Instance-Specific Rationalization of NLP Models
====================================================================
Implements the method from:
  Chrysostomou & Aletras (AAAI-22)
  "Flexible Instance-Specific Rationalization of NLP Models"

Three orthogonal axes of flexibility (any combination can be used):
  1. Feature-scoring method  (FEAT)   – pick the best Ω per instance
  2. Rationale length        (LEN)    – pick the best k ∈ [1, N] per instance
  3. Rationale type          (TYPE)   – pick TOPK or CONTIGUOUS per instance

The selection criterion is the divergence δ between the model's output
distribution on the full input and on the masked input (rationale removed).
Supported divergence functions (∆):
  - 'jsd'       Jensen-Shannon Divergence          (default, per paper §5)
  - 'kl'        Kullback-Leibler Divergence
  - 'perplexity' Perplexity of masked output
  - 'classdiff' Predicted-class probability drop
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable


# ---------------------------------------------------------------------------
# Divergence / delta functions
# ---------------------------------------------------------------------------

def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits.float(), dim=-1)


def delta_jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon Divergence between two probability vectors."""
    p = p.double().clamp(1e-12, 1.0)
    q = q.double().clamp(1e-12, 1.0)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def delta_kl(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL-divergence KL(p || q)."""
    p = p.double().clamp(1e-12, 1.0)
    q = q.double().clamp(1e-12, 1.0)
    return float((p * (p / q).log()).sum())


def delta_perplexity(p: torch.Tensor, q: torch.Tensor) -> float:
    """Perplexity of q relative to p: exp(H(p, q))."""
    p = p.double().clamp(1e-12, 1.0)
    q = q.double().clamp(1e-12, 1.0)
    cross_entropy = -(p * q.log()).sum()
    return float(cross_entropy.exp())


def delta_classdiff(p: torch.Tensor, q: torch.Tensor) -> float:
    """Drop in predicted-class probability: p[argmax p] - q[argmax p]."""
    pred = int(p.argmax().item())
    return float(p[pred].item() - q[pred].item())


_DELTA_FNS: Dict[str, Callable] = {
    "jsd":        delta_jsd,
    "kl":         delta_kl,
    "perplexity": delta_perplexity,
    "classdiff":  delta_classdiff,
}


# ---------------------------------------------------------------------------
# Low-level masking helpers
# ---------------------------------------------------------------------------

def _topk_indices(attr: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of the k highest attribution scores."""
    k = max(1, min(k, attr.shape[0]))
    return torch.topk(attr, k, sorted=False).indices


def _contiguous_indices(attr: torch.Tensor, k: int) -> torch.Tensor:
    """Return the start..start+k slice with the highest summed attribution."""
    n = attr.shape[0]
    k = max(1, min(k, n))
    if k == n:
        return torch.arange(n, device=attr.device)
    best_score, best_start = float("-inf"), 0
    window = attr[:k].sum().item()
    cur = window
    if cur > best_score:
        best_score, best_start = cur, 0
    for i in range(1, n - k + 1):
        cur = cur - attr[i - 1].item() + attr[i + k - 1].item()
        if cur > best_score:
            best_score, best_start = cur, i
    return torch.arange(best_start, best_start + k, device=attr.device)


def _mask_and_forward(
    forward_fn: Callable,
    model: torch.nn.Module,
    input_embed: torch.Tensor,        # (1, L, d)
    position_embed: Optional[torch.Tensor],
    type_embed: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    base_token_emb: torch.Tensor,     # (1, d)
    indices: torch.Tensor,            # token positions to mask
) -> torch.Tensor:
    """Replace `indices` positions with base_token_emb and run the model.
    Returns logits (num_classes,)."""
    masked = input_embed.detach().clone()
    masked[0][indices] = base_token_emb
    logits = forward_fn(
        model, masked,
        attention_mask=attention_mask,
        position_embed=position_embed,
        type_embed=type_embed,
        return_all_logits=True,
    ).squeeze()
    return logits


# ---------------------------------------------------------------------------
# Core δ computation for a single (attr, k, rationale_type) combination
# ---------------------------------------------------------------------------

def compute_delta(
    forward_fn: Callable,
    model: torch.nn.Module,
    input_embed: torch.Tensor,
    position_embed: Optional[torch.Tensor],
    type_embed: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    base_token_emb: torch.Tensor,
    prob_full: torch.Tensor,          # (C,) softmax of full-input logits
    attr: torch.Tensor,               # (L,) attribution scores
    k: int,
    rationale_type: str,              # 'topk' | 'contiguous'
    delta_fn: Callable,
) -> Tuple[float, torch.Tensor]:
    """
    Compute divergence δ for one (attr, k, type) candidate.

    Returns
    -------
    delta_val : float
    indices   : torch.Tensor  — the selected token positions
    """
    if rationale_type == "topk":
        indices = _topk_indices(attr, k)
    elif rationale_type == "contiguous":
        indices = _contiguous_indices(attr, k)
    else:
        raise ValueError(f"Unknown rationale_type '{rationale_type}'")

    logits_masked = _mask_and_forward(
        forward_fn, model,
        input_embed, position_embed, type_embed, attention_mask,
        base_token_emb, indices,
    )
    prob_masked = _softmax(logits_masked)
    delta_val = delta_fn(prob_full, prob_masked)
    return delta_val, indices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_feature_scoring(
    forward_fn: Callable,
    model: torch.nn.Module,
    input_embed: torch.Tensor,
    position_embed: Optional[torch.Tensor],
    type_embed: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    base_token_emb: torch.Tensor,
    attr_dict: Dict[str, torch.Tensor],   # {name: attr_tensor (L,)}
    k: int,                               # fixed rationale length (# tokens)
    rationale_type: str = "topk",         # fixed type
    delta_name: str = "jsd",
) -> Dict[str, Any]:
    """
    Instance-level feature scoring method selection (§3, axis 1).

    For each Ωᵢ in `attr_dict`, compute δᵢ and return the one with δ_max.

    Parameters
    ----------
    attr_dict     : mapping from method name → attribution tensor (L,)
    k             : number of tokens in the rationale (fixed)
    rationale_type: 'topk' or 'contiguous' (fixed)
    delta_name    : divergence function name

    Returns
    -------
    dict with keys:
      best_method   str
      best_delta    float
      best_indices  torch.Tensor
      all_deltas    {name: float}
    """
    delta_fn = _DELTA_FNS[delta_name]

    with torch.no_grad():
        logits_full = forward_fn(
            model, input_embed,
            attention_mask=attention_mask,
            position_embed=position_embed,
            type_embed=type_embed,
            return_all_logits=True,
        ).squeeze()
    prob_full = _softmax(logits_full)

    best_name, best_delta, best_indices = None, float("-inf"), None
    all_deltas = {}

    with torch.no_grad():
        for name, attr in attr_dict.items():
            dval, idxs = compute_delta(
                forward_fn, model,
                input_embed, position_embed, type_embed, attention_mask,
                base_token_emb, prob_full, attr, k, rationale_type, delta_fn,
            )
            all_deltas[name] = dval
            if dval > best_delta:
                best_delta, best_name, best_indices = dval, name, idxs

    return {
        "best_method":  best_name,
        "best_delta":   best_delta,
        "best_indices": best_indices,
        "all_deltas":   all_deltas,
    }


def select_rationale_length(
    forward_fn: Callable,
    model: torch.nn.Module,
    input_embed: torch.Tensor,
    position_embed: Optional[torch.Tensor],
    type_embed: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    base_token_emb: torch.Tensor,
    attr: torch.Tensor,                   # (L,) fixed attribution
    N: int,                               # upper-bound length (fixed pre-defined ratio)
    rationale_type: str = "topk",         # fixed type
    delta_name: str = "jsd",
    skip_rate: float = 0.02,              # 2% skip for long sequences (§4 speed-up)
) -> Dict[str, Any]:
    """
    Instance-level rationale length selection (§3, axis 2).

    Iterates k ∈ range(1, N+1) (with optional skip) and picks δ_max.

    Parameters
    ----------
    N           : maximum / fixed rationale length in *tokens*
    skip_rate   : fraction of sequence to skip per step (0 = no skip)

    Returns
    -------
    dict with keys:
      best_k        int
      best_delta    float
      best_indices  torch.Tensor
      all_deltas    {k: float}
    """
    delta_fn = _DELTA_FNS[delta_name]
    L = attr.shape[0]

    with torch.no_grad():
        logits_full = forward_fn(
            model, input_embed,
            attention_mask=attention_mask,
            position_embed=position_embed,
            type_embed=type_embed,
            return_all_logits=True,
        ).squeeze()
    prob_full = _softmax(logits_full)

    # Build candidate k values with optional skip
    step = max(1, int(L * skip_rate)) if skip_rate > 0 else 1
    candidates = list(range(1, N + 1, step))
    if N not in candidates:
        candidates.append(N)

    best_k, best_delta, best_indices = 1, float("-inf"), None
    all_deltas: Dict[int, float] = {}

    with torch.no_grad():
        for k in candidates:
            dval, idxs = compute_delta(
                forward_fn, model,
                input_embed, position_embed, type_embed, attention_mask,
                base_token_emb, prob_full, attr, k, rationale_type, delta_fn,
            )
            all_deltas[k] = dval
            if dval > best_delta:
                best_delta, best_k, best_indices = dval, k, idxs

    return {
        "best_k":       best_k,
        "best_delta":   best_delta,
        "best_indices": best_indices,
        "all_deltas":   all_deltas,
    }


def select_rationale_type(
    forward_fn: Callable,
    model: torch.nn.Module,
    input_embed: torch.Tensor,
    position_embed: Optional[torch.Tensor],
    type_embed: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    base_token_emb: torch.Tensor,
    attr: torch.Tensor,                   # (L,) fixed attribution
    k: int,                               # fixed length
    delta_name: str = "jsd",
) -> Dict[str, Any]:
    """
    Instance-level rationale type selection (§3, axis 3): TOPK vs CONTIGUOUS.

    Returns
    -------
    dict with keys:
      best_type     str ('topk' | 'contiguous')
      best_delta    float
      best_indices  torch.Tensor
      all_deltas    {'topk': float, 'contiguous': float}
    """
    delta_fn = _DELTA_FNS[delta_name]

    with torch.no_grad():
        logits_full = forward_fn(
            model, input_embed,
            attention_mask=attention_mask,
            position_embed=position_embed,
            type_embed=type_embed,
            return_all_logits=True,
        ).squeeze()
    prob_full = _softmax(logits_full)

    best_type, best_delta, best_indices = None, float("-inf"), None
    all_deltas = {}

    with torch.no_grad():
        for rtype in ("topk", "contiguous"):
            dval, idxs = compute_delta(
                forward_fn, model,
                input_embed, position_embed, type_embed, attention_mask,
                base_token_emb, prob_full, attr, k, rtype, delta_fn,
            )
            all_deltas[rtype] = dval
            if dval > best_delta:
                best_delta, best_type, best_indices = dval, rtype, idxs

    return {
        "best_type":    best_type,
        "best_delta":   best_delta,
        "best_indices": best_indices,
        "all_deltas":   all_deltas,
    }


def select_all(
    forward_fn: Callable,
    model: torch.nn.Module,
    input_embed: torch.Tensor,
    position_embed: Optional[torch.Tensor],
    type_embed: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    base_token_emb: torch.Tensor,
    attr_dict: Dict[str, torch.Tensor],   # {name: attr (L,)}
    N: int,                               # upper-bound length in tokens
    delta_name: str = "jsd",
    skip_rate: float = 0.02,
) -> Dict[str, Any]:
    """
    Full instance-level selection of FEAT + LEN + TYPE simultaneously (§3).

    Iterates over all (Ωᵢ, k, type) combinations and picks the global δ_max.

    Returns
    -------
    dict with keys:
      best_method   str
      best_k        int
      best_type     str
      best_delta    float
      best_indices  torch.Tensor
    """
    delta_fn = _DELTA_FNS[delta_name]
    L = next(iter(attr_dict.values())).shape[0]

    with torch.no_grad():
        logits_full = forward_fn(
            model, input_embed,
            attention_mask=attention_mask,
            position_embed=position_embed,
            type_embed=type_embed,
            return_all_logits=True,
        ).squeeze()
    prob_full = _softmax(logits_full)

    step = max(1, int(L * skip_rate)) if skip_rate > 0 else 1
    candidates = list(range(1, N + 1, step))
    if N not in candidates:
        candidates.append(N)

    best = {"method": None, "k": 1, "type": None,
            "delta": float("-inf"), "indices": None}

    with torch.no_grad():
        for name, attr in attr_dict.items():
            for k in candidates:
                for rtype in ("topk", "contiguous"):
                    dval, idxs = compute_delta(
                        forward_fn, model,
                        input_embed, position_embed, type_embed, attention_mask,
                        base_token_emb, prob_full, attr, k, rtype, delta_fn,
                    )
                    if dval > best["delta"]:
                        best.update(
                            method=name, k=k, type=rtype,
                            delta=dval, indices=idxs
                        )

    return {
        "best_method":  best["method"],
        "best_k":       best["k"],
        "best_type":    best["type"],
        "best_delta":   best["delta"],
        "best_indices": best["indices"],
    }


# ---------------------------------------------------------------------------
# Convenience: compute NormComp and NormSuff from a result dict
# ---------------------------------------------------------------------------

def compute_norm_comp_suff(
    forward_fn: Callable,
    model: torch.nn.Module,
    input_embed: torch.Tensor,
    position_embed: Optional[torch.Tensor],
    type_embed: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    base_token_emb: torch.Tensor,        # (1, d) evaluation baseline token
    best_indices: torch.Tensor,          # selected rationale positions
) -> Dict[str, float]:
    """
    Compute NormComp and NormSuff (Carton et al. 2020 / DeYoung et al. 2020)
    for the selected rationale.

    NormSuff(x, ŷ, R) = [Suff(x,ŷ,R) - Suff(x,ŷ,0)] / [1 - Suff(x,ŷ,0)]
    NormComp(x, ŷ, R) = Comp(x,ŷ,R) / [1 - Suff(x,ŷ,0)]

    where
      Suff(x, ŷ, R) = 1 - max(0, p(ŷ|x) - p(ŷ|R))
      Comp(x, ŷ, R) = max(0, p(ŷ|x) - p(ŷ|x\R))
      Suff(x, ŷ, 0) = sufficiency of a zeroed baseline input
    """
    with torch.no_grad():
        # ---- full input ----
        logits_full = forward_fn(
            model, input_embed,
            attention_mask=attention_mask,
            position_embed=position_embed,
            type_embed=type_embed,
            return_all_logits=True,
        ).squeeze()
        pred = int(logits_full.argmax().item())
        prob_full_pred = float(_softmax(logits_full)[pred].item())

        # ---- zeroed baseline (Suff denominator) ----
        L, d = input_embed.shape[1], input_embed.shape[2]
        zero_embed = torch.zeros(1, L, d, device=input_embed.device, dtype=input_embed.dtype)
        logits_zero = forward_fn(
            model, zero_embed,
            attention_mask=attention_mask,
            position_embed=position_embed,
            type_embed=type_embed,
            return_all_logits=True,
        ).squeeze()
        prob_zero_pred = float(_softmax(logits_zero)[pred].item())

        # ---- rationale only (sufficiency) ----
        # Keep only rationale tokens; mask everything else
        rationale_only = input_embed.detach().clone()
        all_idx = torch.arange(L, device=input_embed.device)
        non_rationale = all_idx[~torch.isin(all_idx, best_indices)]
        rationale_only[0][non_rationale] = base_token_emb
        logits_rat = forward_fn(
            model, rationale_only,
            attention_mask=attention_mask,
            position_embed=position_embed,
            type_embed=type_embed,
            return_all_logits=True,
        ).squeeze()
        prob_rat_pred = float(_softmax(logits_rat)[pred].item())

        # ---- input minus rationale (comprehensiveness) ----
        masked_no_rat = input_embed.detach().clone()
        masked_no_rat[0][best_indices] = base_token_emb
        logits_no_rat = forward_fn(
            model, masked_no_rat,
            attention_mask=attention_mask,
            position_embed=position_embed,
            type_embed=type_embed,
            return_all_logits=True,
        ).squeeze()
        prob_no_rat_pred = float(_softmax(logits_no_rat)[pred].item())

    # Metrics
    suff_full = 1.0 - max(0.0, prob_full_pred - prob_rat_pred)
    suff_zero = 1.0 - max(0.0, prob_full_pred - prob_zero_pred)
    denom = 1.0 - suff_zero

    norm_suff = (suff_full - suff_zero) / denom if abs(denom) > 1e-9 else 0.0
    comp = max(0.0, prob_full_pred - prob_no_rat_pred)
    norm_comp = comp / denom if abs(denom) > 1e-9 else 0.0

    return {
        "norm_suff": norm_suff,
        "norm_comp": norm_comp,
        "comp":      comp,
        "suff":      suff_full,
        "f1_drop":   prob_full_pred - prob_no_rat_pred,   # sign is +ve when rationale matters
    }