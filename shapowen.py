"""
shapowen.py
===========
Owen-Value (Hierarchical / Partition Shapley) Attribution for decoder-only
GPT-2 — gradient-free, forward-only, drop-in compatible with the GPT-2
evaluation pipeline (xai_metrics_gpt2.py and run_eval_*_gpt2.py).

Mirrors `fd_gpt2.py` in interface, so `eval_shapowen.py` is structurally
identical to `run_eval_fd_gpt2.py`.

Algorithm
---------
For an input prompt Q (length Lq) and an answer A (length La, either
gold or greedy-generated), the full sequence has length T = Lq + La.

We treat the Q tokens as the *features* of a coalition game and the A
tokens as the *target* (pinned, never perturbed). The value function is

    v(S) = log p(A | Q_S)
         = sum_{i=1..La} log p(a_i | x_gated at position Lq+i-1)

where Q_S is the Q-prefix with positions outside S replaced by the
baseline embedding (zero / pad / mean — same baselines as FD-PACE).
This is the embedding-space analogue of the "..."  mask used by the
SHAP Text masker: positions not in S are erased to a neutral vector.

Instead of running full SHAP over 2^Lq coalitions, or sampling
permutations, we compute the *Owen value* on a binary hierarchy over
the Q tokens. The hierarchy is the balanced binary tree built bottom-up
over consecutive positions, which is the default produced by the SHAP
Text masker when no richer structure is available.

The recursion runs top-down. At each internal node with children L, R
and parent-context C (the set of Q positions kept in the coalition above
this node), we evaluate four coalitions:

    v(C),  v(C ∪ L),  v(C ∪ R),  v(C ∪ L ∪ R)

and split the parent's attribution between L and R via the closed-form
2-player Shapley values:

    φ_L = ½ [v(C∪L) − v(C)]  +  ½ [v(C∪L∪R) − v(C∪R)]
    φ_R = ½ [v(C∪R) − v(C)]  +  ½ [v(C∪L∪R) − v(C∪L)]

By construction φ_L + φ_R = v(C∪L∪R) − v(C), and the entire tree's
leaves sum to v([Lq]) − v(∅), so efficiency holds exactly.

This is Owen's algorithm on a fixed binary partition. It is exact for
the hierarchical game, NOT for the full Shapley game over 2^Lq
coalitions — that's the bias-for-variance trade we accepted.

Cost: O(Lq) internal nodes × 4 evaluations, with caching across siblings
brings it to roughly 2*Lq − 1 unique forward passes. All evaluations are
vectorized into a single chunked forward pass.

Return dict
-----------
Keys match `fd_gpt2.pace_gradient_fd_gpt2` exactly:
    tokens, q_len, answer_positions, answer_ids,
    attributions, attributions_signed,
    input_embed, base_embed, logits_full,
    predicted_answer, model, tokenizer, time
"""

import time
import random
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Module-level model cache (mirrors fd_gpt2._CACHE)
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def get_model_tokenizer(model_name: str = "gpt2", device: str = "cpu"):
    """Return (model, tokenizer), loading from HuggingFace only once."""
    device = str(torch.device(device))
    key = (model_name, device)
    if key not in _CACHE:
        tok = GPT2TokenizerFast.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        mdl = GPT2LMHeadModel.from_pretrained(model_name)
        mdl.eval().to(device)
        _CACHE[key] = (mdl, tok)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Baseline embedding factory (identical to fd_gpt2._build_base_embed)
# ---------------------------------------------------------------------------

def _build_base_embed(
    embed_layer: torch.nn.Embedding,
    input_embed: torch.Tensor,        # (1, T, D) reference for shape/dtype
    baseline: str,
    eos_token_id: int,
    device: str,
) -> torch.Tensor:
    """
    Build a baseline embedding shaped like `input_embed`, on `device`.
    GPT-2 has no [MASK] — supported baselines are zero, pad (EOS), mean.
    """
    embed_device = next(embed_layer.parameters()).device

    if baseline == "zero":
        return torch.zeros_like(input_embed).to(device)

    elif baseline == "pad":
        pad_id = torch.tensor([[eos_token_id]], device=embed_device)
        with torch.no_grad():
            pad_vec = embed_layer(pad_id).detach()          # (1, 1, D)
        return pad_vec.expand_as(input_embed).clone().to(device)

    elif baseline == "mean":
        with torch.no_grad():
            mean_vec = embed_layer.weight.mean(dim=0, keepdim=True).detach()
        return mean_vec.unsqueeze(0).expand_as(input_embed).clone().to(device)

    else:
        raise ValueError(
            f"Unknown baseline '{baseline}'. Choose: zero | pad | mean"
        )


# ---------------------------------------------------------------------------
# Chunked forward: returns v(S) = log p(A | Q_S) per coalition
# ---------------------------------------------------------------------------

def _chunked_forward_logprob(
    model,
    X: torch.Tensor,              # (1, T, D)
    X_baseline: torch.Tensor,     # (1, T, D)
    gate_batch: torch.Tensor,     # (B, T) in {0, 1} — coalition indicators
    answer_positions: list,
    answer_ids: torch.Tensor,     # (La,)
    chunk_size: int,
    device: str,
) -> torch.Tensor:
    """
    For each coalition vector g in `gate_batch`, build the gated embedding
        X_gated = g * X + (1 - g) * X_baseline
    and return v(g) = sum_i log p(a_i | x_gated at answer_positions[i]).

    Returns Tensor of shape (B,) on `device`.

    Identical signature/behaviour to fd_gpt2._chunked_forward_logprob,
    but here `gate_batch` carries discrete {0,1} coalition indicators
    rather than continuous gate values.
    """
    B, T = gate_batch.shape
    La   = len(answer_positions)

    ans_pos_tensor = torch.tensor(answer_positions, device=device, dtype=torch.long)
    ans_ids_dev    = answer_ids.to(device).long()

    X_sq    = X.squeeze(0).to(device)            # (T, D)
    Xref_sq = X_baseline.squeeze(0).to(device)   # (T, D)

    y_chunks = []

    for i in range(0, B, chunk_size):
        j = min(i + chunk_size, B)
        g_chunk = gate_batch[i:j].to(device=device, dtype=X.dtype)   # (b, T)
        g_exp   = g_chunk.unsqueeze(-1)                              # (b, T, 1)
        X_gated = X_sq * g_exp + Xref_sq * (1.0 - g_exp)             # (b, T, D)

        with torch.no_grad():
            logits = model(inputs_embeds=X_gated).logits             # (b, T, V)

        log_probs = F.log_softmax(logits, dim=-1)                    # (b, T, V)
        gathered  = log_probs[:, ans_pos_tensor, :]                  # (b, La, V)
        token_lp  = gathered.gather(
            dim=-1,
            index=ans_ids_dev.view(1, La, 1).expand(j - i, La, 1),
        ).squeeze(-1)                                                # (b, La)
        y_chunks.append(token_lp.sum(dim=-1))                        # (b,)

    return torch.cat(y_chunks, dim=0)                                # (B,)


# ---------------------------------------------------------------------------
# Hierarchy: balanced binary tree over consecutive Q positions
# ---------------------------------------------------------------------------

class _Node:
    """
    Lightweight binary-tree node.

    `span` is a tuple (lo, hi) representing the half-open interval of
    Q positions belonging to this subtree's coverage in the FULL sequence
    indexing, i.e. lo, hi ∈ [0, q_len). Leaves have hi == lo + 1.
    """
    __slots__ = ("span", "left", "right")

    def __init__(self, span: Tuple[int, int],
                 left: Optional["_Node"] = None,
                 right: Optional["_Node"] = None):
        self.span  = span
        self.left  = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _build_balanced_tree(lo: int, hi: int) -> _Node:
    """
    Build a balanced binary tree over the half-open interval [lo, hi).

    Splits at the midpoint. For an interval of size 1 we return a leaf.
    This is the default hierarchical structure used by SHAP's Text masker
    when no richer linguistic grouping is supplied — it captures locality
    (adjacent tokens cluster) without committing to syntax.
    """
    if hi - lo <= 1:
        return _Node((lo, hi))
    mid = (lo + hi) // 2
    return _Node(
        span  = (lo, hi),
        left  = _build_balanced_tree(lo, mid),
        right = _build_balanced_tree(mid, hi),
    )


# ---------------------------------------------------------------------------
# Coalition enumeration: walk the tree, collect unique masks to evaluate
# ---------------------------------------------------------------------------

def _enumerate_coalitions(
    root: _Node,
    q_len: int,
    T: int,
    device: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[Tuple[int, ...], int], List[Tuple[_Node, int, int, int, int]]]:
    """
    Walk the tree recursively. At each *internal* node with context C
    (the set of Q positions that are ON outside this subtree, inherited
    from ancestors), enqueue four coalition gate vectors:

        v(C),  v(C ∪ L),  v(C ∪ R),  v(C ∪ L ∪ R)

    The recursion descends into L with context C' = C ∪ R, and into R
    with context C' = C ∪ L. This is the standard top-down Owen
    expansion: at each level, the sibling that is NOT being further
    decomposed is held ON, so its leaves' attributions accumulate
    from joint interactions in the parent split rather than being
    double-counted in the child subtree's splits.

    Returns
    -------
    gate_batch : Tensor (N_unique, T)
        Unique coalition indicator vectors, A positions pinned to 1.
    index_map  : dict mapping coalition tuple -> row index in gate_batch
    node_calls : list of (node, idx_C, idx_CL, idx_CR, idx_CLR)
        For every internal node, the four row indices into gate_batch
        to look up for that node's 2-player Shapley split.
    """
    # Each "coalition" is a frozenset of Q positions that are ON.
    # We canonicalize as a sorted tuple for use as a dict key.
    index_map: Dict[Tuple[int, ...], int] = {}
    gate_rows: List[torch.Tensor] = []
    node_calls: List[Tuple[_Node, int, int, int, int]] = []

    def _coal_key(positions: frozenset) -> Tuple[int, ...]:
        return tuple(sorted(positions))

    def _get_or_add(positions: frozenset) -> int:
        key = _coal_key(positions)
        if key in index_map:
            return index_map[key]
        # Build the gate vector: 1 on Q positions in `positions`,
        # 0 on Q positions not in `positions`, 1 on all A positions.
        g = torch.zeros(T, device=device, dtype=dtype)
        if positions:
            idx = torch.tensor(sorted(positions), device=device, dtype=torch.long)
            g[idx] = 1.0
        if T > q_len:
            g[q_len:] = 1.0                       # answer positions pinned ON
        row_idx = len(gate_rows)
        gate_rows.append(g)
        index_map[key] = row_idx
        return row_idx

    def _leaves(node: _Node) -> frozenset:
        """Set of Q positions covered by the subtree rooted at node."""
        return frozenset(range(node.span[0], node.span[1]))

    def _recurse(node: _Node, context: frozenset):
        # Leaves: no further split needed. Their attribution is the
        # 2-player Shapley share computed at the *parent* node — handled
        # by the parent's call and pushed downward in _solve_owen.
        if node.is_leaf:
            return

        L_set = _leaves(node.left)
        R_set = _leaves(node.right)

        C   = context
        CL  = context | L_set
        CR  = context | R_set
        CLR = context | L_set | R_set

        idx_C   = _get_or_add(C)
        idx_CL  = _get_or_add(CL)
        idx_CR  = _get_or_add(CR)
        idx_CLR = _get_or_add(CLR)

        node_calls.append((node, idx_C, idx_CL, idx_CR, idx_CLR))

        # Descend. When recursing into L, sibling R is held ON in context.
        # When recursing into R, sibling L is held ON in context.
        _recurse(node.left,  context | R_set)
        _recurse(node.right, context | L_set)

    # Ensure the "all ON" and "all OFF" coalitions are present too — they
    # anchor efficiency reporting and are typically the root's CLR / C.
    _get_or_add(frozenset())                             # ∅
    _get_or_add(_leaves(root))                           # full Q
    _recurse(root, frozenset())

    gate_batch = torch.stack(gate_rows, dim=0)           # (N_unique, T)
    return gate_batch, index_map, node_calls


# ---------------------------------------------------------------------------
# Solve the Owen recursion given precomputed v(S) for every needed S
# ---------------------------------------------------------------------------

def _solve_owen(
    root: _Node,
    node_calls: List[Tuple[_Node, int, int, int, int]],
    v_values: torch.Tensor,         # (N_unique,)  v(S) for each unique coalition
    full_q_idx: int,                # index of v(full Q) in v_values
    empty_idx: int,                 # index of v(∅)     in v_values
    q_len: int,
) -> torch.Tensor:
    """
    Run the top-down Owen recursion using cached v(S) values.

    Each internal node receives a "parent attribution" — the share of
    v(root) − v(∅) that this subtree owns — and splits it between its
    children via 2-player Shapley. Leaves' attributions are the final
    per-Q-token Owen values.

    Returns Tensor (q_len,) on the same device as v_values.
    """
    attributions = torch.zeros(q_len, device=v_values.device, dtype=v_values.dtype)

    root_share = v_values[full_q_idx] - v_values[empty_idx]

    # Edge case: q_len == 1. The root IS a leaf — no internal-node splits
    # exist. The single Q token absorbs the entire v(full) − v(empty) gap.
    if root.is_leaf:
        attributions[root.span[0]] = root_share
        return attributions

    # Map from id(node) -> parent attribution to receive.
    parent_attr: Dict[int, torch.Tensor] = {}
    parent_attr[id(root)] = root_share

    # node_calls are in pre-order (parents before children). The recursion
    # in _enumerate_coalitions emits them that way, so walking the list
    # in order is safe — every node we encounter has its parent_attr set.
    for node, idx_C, idx_CL, idx_CR, idx_CLR in node_calls:
        v_C   = v_values[idx_C]
        v_CL  = v_values[idx_CL]
        v_CR  = v_values[idx_CR]
        v_CLR = v_values[idx_CLR]

        # 2-player Shapley split of (v_CLR − v_C) between L and R.
        phi_L = 0.5 * (v_CL  - v_C) + 0.5 * (v_CLR - v_CR)
        phi_R = 0.5 * (v_CR  - v_C) + 0.5 * (v_CLR - v_CL)

        # Renormalize so that φ_L + φ_R = parent_attr[id(node)] exactly.
        # The raw φ_L + φ_R already equals v_CLR − v_C; if that matches
        # the parent's allotted share (it does at the root and propagates
        # correctly down a consistent tree), no rescaling is needed.
        # In floating point we rescale defensively.
        share = parent_attr[id(node)]
        raw_sum = phi_L + phi_R
        if torch.abs(raw_sum) > 1e-12:
            scale = share / raw_sum
            phi_L = phi_L * scale
            phi_R = phi_R * scale
        else:
            # Degenerate: split parent's share equally.
            phi_L = share * 0.5
            phi_R = share * 0.5

        # Dispatch to children: if a child is a leaf, write into the
        # attribution vector at its position; otherwise stash for the
        # child's own internal-node processing.
        if node.left.is_leaf:
            pos = node.left.span[0]
            attributions[pos] = phi_L
        else:
            parent_attr[id(node.left)] = phi_L

        if node.right.is_leaf:
            pos = node.right.span[0]
            attributions[pos] = phi_R
        else:
            parent_attr[id(node.right)] = phi_R

    return attributions


# ---------------------------------------------------------------------------
# Public API: Owen-Shapley attribution for GPT-2
# ---------------------------------------------------------------------------

def shap_owen_gpt2(
    question: str,
    model_name: str = "gpt2",
    device: str = "cpu",
    chunk_size: int = 32,
    max_new_tokens: int = 30,
    gold_answer: Optional[str] = None,
    baseline: str = "zero",
    # The following two args are accepted for interface parity with
    # `pace_gradient_fd_gpt2` but are unused by the Owen algorithm:
    steps: Optional[int] = None,
    a: float = 0.0,
    b: float = 1.0,
) -> Dict[str, Any]:
    """
    Run Owen-value (hierarchical Shapley) attribution for one (Q, A)
    pair on GPT-2.

    Parameters
    ----------
    question       : Full prompt string ("narrative ... Why did ...?")
    model_name     : GPT-2 variant.
    device         : 'cpu' or 'cuda'.
    chunk_size     : Max batch size for a single forward pass.
                     Total forwards: O(Lq) (typically 2*Lq − 1), chunked.
    max_new_tokens : Greedy generation budget when `gold_answer` is None.
    gold_answer    : If provided, skip generation and use as A.
    baseline       : 'zero' | 'pad' | 'mean' — see _build_base_embed.
    steps, a, b    : Ignored. Present only for CLI parity with FD-PACE.

    Returns
    -------
    dict with keys matching fd_gpt2.pace_gradient_fd_gpt2.
    """
    t0 = time.time()
    model, tokenizer = get_model_tokenizer(model_name, device)

    # ------------------------------------------------------------------
    # 1. Tokenise Q
    # ------------------------------------------------------------------
    q_enc       = tokenizer(question, return_tensors="pt", add_special_tokens=True)
    q_ids       = q_enc["input_ids"].to(device)               # [1, Lq]
    q_attn_mask = q_enc["attention_mask"].to(device)
    q_len       = q_ids.shape[1]

    # ------------------------------------------------------------------
    # 2. Get answer ids (gold or greedy) — identical to fd_gpt2
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
    answer_ids       = full_ids[0, q_len:].cpu()              # [La]

    # ------------------------------------------------------------------
    # 3. Token embeddings (wte output)
    # ------------------------------------------------------------------
    embed_layer = model.transformer.wte
    with torch.no_grad():
        input_embed = embed_layer(full_ids).detach()          # [1, T, D] on device

    # ------------------------------------------------------------------
    # 4. Baseline embedding
    # ------------------------------------------------------------------
    base_embed = _build_base_embed(
        embed_layer, input_embed,
        baseline, tokenizer.eos_token_id, device,
    )                                                         # [1, T, D] on device

    # ------------------------------------------------------------------
    # 5. Reference logits (for compatibility with downstream eval)
    # ------------------------------------------------------------------
    with torch.no_grad():
        logits_full = model(inputs_embeds=input_embed).logits[0].detach().cpu()

    # ------------------------------------------------------------------
    # 6. Run Owen recursion
    # ------------------------------------------------------------------
    start_time = time.perf_counter()

    if q_len == 0:
        attributions = torch.zeros(T, dtype=input_embed.dtype)
    else:
        # Build hierarchy over Q positions [0, q_len)
        root = _build_balanced_tree(0, q_len)

        # Enumerate every unique coalition the tree's internal nodes need
        gate_batch, index_map, node_calls = _enumerate_coalitions(
            root   = root,
            q_len  = q_len,
            T      = T,
            device = device,
            dtype  = input_embed.dtype,
        )

        # Single chunked forward pass over all unique coalitions
        v_values = _chunked_forward_logprob(
            model            = model,
            X                = input_embed,
            X_baseline       = base_embed,
            gate_batch       = gate_batch,
            answer_positions = answer_positions,
            answer_ids       = answer_ids,
            chunk_size       = chunk_size,
            device           = device,
        )                                                     # (N_unique,)

        # Lookups for efficiency-anchor coalitions
        empty_idx  = index_map[tuple()]
        full_q_idx = index_map[tuple(range(q_len))]

        # Top-down Owen recursion: produces Owen value per Q position
        attr_q = _solve_owen(
            root        = root,
            node_calls  = node_calls,
            v_values    = v_values,
            full_q_idx  = full_q_idx,
            empty_idx   = empty_idx,
            q_len       = q_len,
        )                                                     # (q_len,)

        # Scatter into full-length attribution vector (A positions = 0)
        attributions = torch.zeros(T, dtype=input_embed.dtype, device=device)
        attributions[:q_len] = attr_q.to(attributions.dtype)
        attributions = attributions.cpu()

    end_time = time.perf_counter()

    # ------------------------------------------------------------------
    # 7. Sign convention (matches fd_gpt2)
    # ------------------------------------------------------------------
    # Owen attributions are signed:
    #   POSITIVE -> token raises log p(A|Q)  (supports the answer)
    #   NEGATIVE -> token suppresses log p(A|Q)
    #
    # xai_metrics_gpt2.py uses min-max normalization and expects
    # non-negative magnitudes as importance scores. We provide both:
    #   attributions          : |attr|   (drop-in for the eval pipeline)
    #   attributions_signed   : attr     (for analysis / debugging)
    attributions_signed = attributions.clone()
    attributions        = attributions.abs()

    # ------------------------------------------------------------------
    # 8. Package output (keys match fd_gpt2.pace_gradient_fd_gpt2)
    # ------------------------------------------------------------------
    tokens = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())

    return {
        "tokens":              tokens,
        "q_len":               q_len,
        "answer_positions":    answer_positions,
        "answer_ids":          answer_ids,                    # [La] CPU
        "attributions":        attributions,                  # [T]  CPU, |attr|
        "attributions_signed": attributions_signed,           # [T]  CPU, signed
        "input_embed":         input_embed.cpu(),             # [1,T,D] CPU
        "base_embed":          base_embed.cpu(),              # [1,T,D] CPU
        "logits_full":         logits_full,                   # [T,V]   CPU
        "predicted_answer":    tokenizer.decode(
                                   answer_ids.tolist(),
                                   skip_special_tokens=True),
        "model":               model,
        "tokenizer":           tokenizer,
        "time":                end_time - start_time,
        "owen_wall_time":      time.time() - t0,
    }