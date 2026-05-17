"""
xai_metrics_gpt2.py
====================
Faithfulness metrics for decoder-only GPT-2 PACE Gradient attributions.

All three metrics follow Zhao & Shan (ReAGent, AAAI 2024):

    Soft-NC (Eq. 15) — Soft Normalised Comprehensiveness
        = ΔP_{X'\\R, t} / ΔP_{0,t}

    Soft-NS (Eq. 14) — Soft Normalised Sufficiency
        = max(0, ΔP_{0,t} − ΔP_{X',t}) / ΔP_{0,t}

    where ΔP is Hellinger distance over the full vocabulary distribution
    (Eq. 13), and ΔP_{0,t} is the Hellinger distance from the zero-input
    distribution to the full-input distribution (normalisation anchor).

    Perturbation uses soft Bernoulli masking (Eq. 11-12):
        x'_i = x_i ⊙ e_i,   e_i ~ Ber(q_i)
        q_i = 1 − s_i   (comprehensiveness — remove important tokens)
        q_i = s_i        (sufficiency      — retain important tokens)

    Log-odds — classic hard top-k masking on Q tokens for comparison
        = mean_t [ log p(a_t | full) − log p(a_t | Q_top-k zeroed) ]

Public API
----------
    calculate_all_metrics_gpt2(
        model, input_embed, base_embed, attributions,
        answer_ids, answer_positions,
        topk=20, n_samples=10, stride=1, device="cpu"
    ) → dict with keys:
            soft_nc  : torch.Tensor scalar   (call .item() for float)
            soft_ns  : torch.Tensor scalar
            log_odds : torch.Tensor scalar
"""

import math
from typing import Literal

import torch
import torch.nn.functional as F


# ============================================================
# Low-level helpers
# ============================================================

def _hellinger(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Hellinger distance between two probability distributions.

    H(P,Q) = (1/√2) √[ Σ_v (√p_v − √q_v)² ]

    Range: [0, 1].  Symmetric.  Both inputs must sum to 1.
    """
    P = P.clamp(min=0.0)
    Q = Q.clamp(min=0.0)
    return (1.0 / math.sqrt(2)) * ((P.sqrt() - Q.sqrt()).pow(2).sum().sqrt())


def _vocab_dist(
    model,
    embed: torch.Tensor,
    position: int,
    device: str,
) -> torch.Tensor:
    """
    One forward pass → softmax vocabulary distribution at `position`.

    Parameters
    ----------
    embed    : [1, T, D]  token embeddings (CPU, moved to device internally)
    position : sequence index to read the distribution from

    Returns
    -------
    Tensor [V] on CPU.
    """
    with torch.no_grad():
        logits = model(inputs_embeds=embed.to(device)).logits  # [1, T, V]
    return F.softmax(logits[0, position, :], dim=-1).cpu()


def _soft_mask(
    embed: torch.Tensor,
    attributions: torch.Tensor,
    mode: Literal["comprehensiveness", "sufficiency"],
    seed: int,
) -> torch.Tensor:
    """
    Soft Bernoulli perturbation (Eq. 11-12 of the ReAGent paper).

    Normalises attributions to [0,1], samples a binary mask per token,
    and returns embed ⊙ mask.

    Parameters
    ----------
    embed        : [1, T, D]  (CPU)
    attributions : [T]        (any non-negative values)
    mode         : 'comprehensiveness' → remove important tokens
                   'sufficiency'       → retain important tokens
    seed         : for reproducibility across Monte-Carlo draws

    Returns
    -------
    Perturbed embedding [1, T, D] on CPU.
    """
    T    = attributions.shape[0]
    # PACE IG attributions are L2-norms — always >= 0.
    # ReAGent occlusion scores are Hellinger distances — always >= 0.
    # Simple min-max normalisation is correct for both.
    amin = attributions.min()
    amax = attributions.max()

    if (amax - amin).abs() < 1e-8:
        s = torch.full((T,), 0.5)
    else:
        s = (attributions - amin) / (amax - amin)   # [T] in [0,1]

    # q_i = probability that token i is *kept* in the mask
    q = (1.0 - s) if mode == "comprehensiveness" else s  # [T]

    gen  = torch.Generator()
    gen.manual_seed(seed)
    mask = torch.bernoulli(q, generator=gen)              # [T]  1=keep

    # Broadcast over embedding dimension D  →  [1, T, D]
    mask = mask.unsqueeze(0).unsqueeze(-1).expand_as(embed)
    return embed * mask


# ============================================================
# Per-position metric computation
# ============================================================

def _delta_P(
    model,
    embed_orig: torch.Tensor,
    embed_pert: torch.Tensor,
    position: int,
    device: str,
) -> torch.Tensor:
    """Hellinger( P_orig_t , P_pert_t ) at sequence position t."""
    P_orig = _vocab_dist(model, embed_orig, position, device)
    P_pert = _vocab_dist(model, embed_pert, position, device)
    return _hellinger(P_orig, P_pert)

def _delta_P0(
    model,
    embed_orig: torch.Tensor,
    eval_base_embed: torch.Tensor,   # (1, T, D) or (1, 1, D) — replaces hardcoded zeros
    position: int,
    device: str,
) -> torch.Tensor:
    """ΔP_{0,t} = Hellinger( P_base_t , P_orig_t )"""
    P_base = _vocab_dist(model, eval_base_embed, position, device)
    P_orig = _vocab_dist(model, embed_orig,      position, device)
    return _hellinger(P_base, P_orig)


def _soft_nc_sequence(
    model,
    embed_orig: torch.Tensor,
    attributions: torch.Tensor,
    answer_positions: list,
    eval_base_embed: torch.Tensor,
    device: str,
    n_samples: int,
    stride: int,
) -> torch.Tensor:
    positions = answer_positions[::stride]
    if not positions:
        return torch.tensor(0.0)

    nc_total = torch.tensor(0.0)
    valid    = 0

    for pos in positions:
        dP0 = _delta_P0(model, embed_orig, eval_base_embed, pos, device)
        if dP0.item() < 1e-8:
            continue
        valid += 1

        sample_sum = torch.tensor(0.0)
        for k in range(n_samples):
            e_pert     = _soft_mask(embed_orig, attributions,
                                    mode="comprehensiveness", seed=k)
            dP_pert    = _delta_P(model, embed_orig, e_pert, pos, device)
            sample_sum = sample_sum + dP_pert / dP0

        nc_total = nc_total + sample_sum / n_samples

    return nc_total / valid if valid > 0 else torch.tensor(0.0)


def _soft_ns_sequence(
    model,
    embed_orig: torch.Tensor,
    attributions: torch.Tensor,
    answer_positions: list,
    eval_base_embed: torch.Tensor,
    device: str,
    n_samples: int,
    stride: int,
) -> torch.Tensor:
    positions = answer_positions[::stride]
    if not positions:
        return torch.tensor(0.0)

    ns_total = torch.tensor(0.0)
    valid    = 0

    for pos in positions:
        dP0 = _delta_P0(model, embed_orig, eval_base_embed, pos, device)
        if dP0.item() < 1e-8:
            continue
        valid += 1

        sample_sum = torch.tensor(0.0)
        for k in range(n_samples):
            e_pert     = _soft_mask(embed_orig, attributions,
                                    mode="sufficiency", seed=100 + k)
            dP_pert    = _delta_P(model, embed_orig, e_pert, pos, device)
            ns_val     = torch.clamp(dP0 - dP_pert, min=0.0) / dP0
            sample_sum = sample_sum + ns_val

        ns_total = ns_total + sample_sum / n_samples

    return ns_total / valid if valid > 0 else torch.tensor(0.0)


def _log_odds_sequence(
    model,
    embed_orig: torch.Tensor,
    attributions: torch.Tensor,
    answer_ids: torch.Tensor,
    answer_positions: list,
    device: str,
    topk: int,
) -> torch.Tensor:
    """
    Log-odds with hard top-k% masking on Q tokens.

    For each answer position t:
        log_odds_t = log p(a_t | full) − log p(a_t | Q_top-k zeroed)

    Averaged over all answer positions.
    """
    q_len   = answer_positions[0] if answer_positions else embed_orig.shape[1]
    q_attrs = attributions[:q_len]                         # [Lq]
    k_count = max(1, int(len(q_attrs) * topk / 100))

    # topk by value — both PACE (L2-norm) and ReAGent (Hellinger)
    # scores are always >= 0, so this is correct as-is.
    _, top_idx    = torch.topk(q_attrs, k=k_count)
    embed_masked  = embed_orig.clone()
    embed_masked[0, top_idx, :] = 0.0

    lo_total = torch.tensor(0.0)

    for i, pos in enumerate(answer_positions):
        tok_id = answer_ids[i].item()
        P_full = _vocab_dist(model, embed_orig,   pos, device)
        P_mask = _vocab_dist(model, embed_masked, pos, device)

        p_full = P_full[tok_id].clamp(min=1e-9)
        p_mask = P_mask[tok_id].clamp(min=1e-9)

        # Standard convention (matches BERT PACE / AttCAT xai_metrics.py):
        #   log_odds = log p(a_t | masked) - log p(a_t | full)
        #   NEGATIVE = masking hurt prediction = those tokens were important = GOOD
        #   POSITIVE = masking helped = tokens were suppressive = BAD attribution
        lo_total = lo_total + (p_mask.log() - p_full.log())

    n = len(answer_positions)
    return - torch.abs(lo_total / n )


# ============================================================
# Public entry point
# ============================================================

def calculate_all_metrics_gpt2(
    model,
    input_embed: torch.Tensor,
    base_embed: torch.Tensor,
    attributions: torch.Tensor,
    answer_ids: torch.Tensor,
    answer_positions: list,
    topk: int = 20,
    n_samples: int = 10,
    stride: int = 1,
    device: str = "cpu",
    eval_base_embed: torch.Tensor | None = None,   # ← added
) -> dict:
    """
    eval_base_embed : (1, T, D) or (1, 1, D) baseline used in ΔP_0
                      (the normalisation anchor for Soft-NC and Soft-NS).
                      If None, falls back to base_embed for backward
                      compatibility (which itself defaulted to zeros).
    """
    # Resolve eval baseline — prefer explicit eval_base_embed,
    # fall back to base_embed (preserves original behaviour when not set)
    anchor = eval_base_embed if eval_base_embed is not None else base_embed

    # Expand (1, 1, D) → (1, T, D) if needed
    T = input_embed.shape[1]
    if anchor.shape[1] == 1 and T > 1:
        anchor = anchor.expand(1, T, -1)

    soft_nc = _soft_nc_sequence(
        model, input_embed, attributions,
        answer_positions, anchor, device, n_samples, stride,
    )
    soft_ns = _soft_ns_sequence(
        model, input_embed, attributions,
        answer_positions, anchor, device, n_samples, stride,
    )
    log_odds = _log_odds_sequence(
        model, input_embed, attributions,
        answer_ids, answer_positions, device, topk,
    )

    return {
        "soft_nc":  soft_nc,
        "soft_ns":  soft_ns,
        "log_odds": log_odds,
    }
