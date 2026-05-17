"""
gaf_eval_sentiment.py

Generalized Attention Flow (GAF) feature attribution for sentiment classification.

Paper: "Generalized Attention Flow: Feature Attribution for Transformer Models
via Maximum Flow", Azarkhalili & Libbrecht, arXiv:2502.15765, Feb 2025.

Three information tensor variants (Sec. 3.1):
  AF  : Ā = E_H(A)                      — raw attention weights
  GF  : Ā = E_H( relu(∇A) )             — gradient of attention weights
  AGF : Ā = E_H( relu(A ⊙ ∇A) )        — attention × gradient (best in paper)

where ∇A = ∂y_c/∂A, y_c = predicted-class logit before softmax.

Graph construction (Algorithm 1 — Backward Information Capacity):
  Nodes: 2 + t*(l+1)  (super-source ss, super-target st, t tokens × (l+1) layers)
  Edges encode the information tensor as upper-bound capacities.
  Lower bounds: 0 everywhere.

Optimization (Eq. 12):
  min_{B^T f = 0}  c^T f  −  μ Σ_e [log(f_e − l_e) + log(u_e − f_e)]
  solved via CVXPY (interior-point / barrier method).

Token attribution = total outflow of each node in the first (input) layer.

Usage:
  python gaf_eval_sentiment.py --model distilbert --dataset sst2 --variant AGF
  python gaf_eval_sentiment.py --model bert --dataset imdb --variant GF
"""

import time
import tqdm
import torch
import random
import argparse
import numpy as np
import cvxpy as cp
from typing import Dict, List, Tuple, Literal
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from xai_metrics import (
    calculate_log_odds,
    calculate_comprehensiveness,
    calculate_sufficiency,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ---------------------------------------------------------------------------
# Model / tokenizer cache
# ---------------------------------------------------------------------------
_cache: Dict[str, Dict] = {}


def _get_cached(model_name: str, device: str):
    if model_name not in _cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=False,
        ).to(device)
        model.eval()
        _cache[model_name] = {"model": model, "tokenizer": tokenizer}
    return _cache[model_name]["model"], _cache[model_name]["tokenizer"]


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

def _get_attn_modules(model) -> List[torch.nn.Module]:
    """Return list of self-attention sub-modules, one per encoder layer."""
    if hasattr(model, "bert"):
        return [layer.attention for layer in model.bert.encoder.layer]
    if hasattr(model, "distilbert"):
        return [layer.attention for layer in model.distilbert.transformer.layer]
    if hasattr(model, "roberta"):
        return [layer.attention for layer in model.roberta.encoder.layer]
    raise RuntimeError("Unsupported model architecture.")


# ---------------------------------------------------------------------------
# Information tensor construction
# ---------------------------------------------------------------------------

def _build_information_tensor(
    attn_weights: List[torch.Tensor],   # L × [1, H, t, t]
    attn_grads:   List[torch.Tensor],   # L × [1, H, t, t]  (may be empty for AF)
    variant: str,                        # "AF" | "GF" | "AGF"
) -> np.ndarray:
    """
    Build Ā ∈ R^{L × t × t} from collected attention weights and their gradients.

    AF  : Ā_l = mean_H(A_l)
    GF  : Ā_l = mean_H( relu(∇A_l) )
    AGF : Ā_l = mean_H( relu(A_l ⊙ ∇A_l) )
    """
    tensors = []
    for l_idx, aw in enumerate(attn_weights):
        A = aw.squeeze(0)   # [H, t, t]
        if variant == "AF":
            layer_tensor = A.mean(dim=0)                           # [t, t]
        elif variant == "GF":
            G = attn_grads[l_idx].squeeze(0)                       # [H, t, t]
            layer_tensor = torch.relu(G).mean(dim=0)
        elif variant == "AGF":
            G = attn_grads[l_idx].squeeze(0)
            layer_tensor = torch.relu(A * G).mean(dim=0)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        tensors.append(layer_tensor.detach().cpu().numpy())        # [t, t]
    return np.stack(tensors, axis=0)   # [L, t, t]


# ---------------------------------------------------------------------------
# Graph construction — Algorithm 1 (Backward Information Capacity)
# ---------------------------------------------------------------------------

def _get_backward_capacity(A_bar: np.ndarray) -> Dict:
    """
    Algorithm 1 from the paper.

    A_bar : [L, t, t]
    Returns dict with keys: u, l, ss, st, n_nodes
      u[i, j] = upper-bound capacity of edge i→j
      l[i, j] = lower-bound capacity (all zero)
      ss       = super-source node index
      st       = super-target node index
    """
    L, t, _ = A_bar.shape

    # Scaling factor γ (to convert to near-integers for numerical stability)
    pos_vals = A_bar[A_bar > 0]
    if len(pos_vals) == 0:
        pos_vals = np.array([1e-6])
    beta_min = pos_vals.min()
    beta     = -np.floor(np.log10(beta_min))
    gamma    = 10.0 ** beta

    Qtl = t * (L + 1) + 2
    u = np.zeros((Qtl, Qtl), dtype=np.float64)
    l = np.zeros((Qtl, Qtl), dtype=np.float64)

    u_inf = float(t)

    # super-source → first layer  (nodes 1 … t)
    for i in range(t):
        u[i + 1][0] = u_inf

    # last layer (nodes t*L+1 … t*L+t) → super-target (node -1 = Qtl-1)
    for i in range(t):
        u[Qtl - 1][-(i + 2)] = u_inf

    # inter-layer edges:  layer j+1 → layer j  (attention flows backward)
    for j in range(L):
        start = t * j + 1
        mid   = t * (j + 1) + 1
        end_  = t * (j + 2) + 1
        u[mid:end_, start:mid] = A_bar[j, :, :]

    ss = t * (L + 1) + 1   # super-source
    st = 0                  # super-target

    return {"u": u * gamma, "l": l, "ss": ss, "st": st, "Qtl": Qtl, "gamma": gamma}


# ---------------------------------------------------------------------------
# Log-barrier regularized min-cost circulation (Eq. 12) via CVXPY
# ---------------------------------------------------------------------------

def _solve_mcc_barrier(cap: Dict, mu: float = 1e-3) -> np.ndarray:
    """
    Solve the log-barrier regularized min-cost circulation problem (Eq. 12):

      min_{B^T f = 0}  c^T f  − μ Σ_e [log(f_e − l_e) + log(u_e − f_e)]

    Returns f_out : [Qtl,]  total outflow per node.
    """
    u_mat = cap["u"]    # [Qtl, Qtl]
    l_mat = cap["l"]    # [Qtl, Qtl]
    ss    = cap["ss"]
    st    = cap["st"]
    Qtl   = cap["Qtl"]

    # Identify edges (i→j) where u[i,j] > 0
    rows, cols = np.where(u_mat > 0)
    n_edges = len(rows)

    if n_edges == 0:
        return np.zeros(Qtl)

    u_e = u_mat[rows, cols]   # [m,]
    l_e = l_mat[rows, cols]   # [m,] (all zero)

    # Build edge-vertex incidence matrix B ∈ R^{m × Qtl}
    # B[e, tail] = -1,  B[e, head] = +1
    B = np.zeros((n_edges, Qtl), dtype=np.float64)
    for e, (i, j) in enumerate(zip(rows, cols)):
        B[e, i] = -1   # i is tail (source of edge)
        B[e, j] = +1   # j is head

    # Cost vector c: c_{t→s} = -1 (the back-edge from st to ss), else 0
    # We add the back-edge from st to ss with large capacity u_inf = ||u||_1
    u_inf_back = float(u_e.sum())
    c_vec = np.zeros(n_edges + 1)
    c_vec[-1] = -1.0   # back-edge cost

    # Extend B and capacity for the back-edge ss ← st  (st → ss direction)
    b_back = np.zeros((1, Qtl))
    b_back[0, st] = -1
    b_back[0, ss] = +1
    B_ext  = np.vstack([B, b_back])
    u_ext  = np.append(u_e, u_inf_back)
    l_ext  = np.append(l_e, 0.0)

    # CVXPY variable
    f = cp.Variable(n_edges + 1)

    # Objective: c^T f  − μ Σ log(f_e − l_e) + log(u_e − f_e)
    # CVXPY log-barrier: cp.sum(cp.log(f - l) + cp.log(u - f))
    barrier = cp.sum(cp.log(f - l_ext) + cp.log(u_ext - f))
    objective = cp.Minimize(c_vec @ f - mu * barrier)

    # Constraints: B^T f = 0  (flow conservation)
    constraints = [B_ext.T @ f == 0]

    prob = cp.Problem(objective, constraints)

    # Warm-start: initialise f at midpoint of [l, u]
    f.value = (l_ext + u_ext) / 2.0

    try:
        prob.solve(
            solver=cp.CLARABEL,
            warm_start=True,
            verbose=False,
        )
    except cp.SolverError:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            return np.zeros(Qtl)

    if f.value is None:
        return np.zeros(Qtl)

    f_val = f.value[:-1]   # drop the back-edge

    # Compute total outflow per node
    f_out = np.zeros(Qtl)
    for e, (i, _) in enumerate(zip(rows, cols)):
        if f_val[e] > 0:
            f_out[i] += f_val[e]

    return f_out


# ---------------------------------------------------------------------------
# Core GAF computation
# ---------------------------------------------------------------------------

def gaf_classification(
    sentence: str,
    model_name: str,
    variant: Literal["AF", "GF", "AGF"] = "AGF",
    mu: float = 1e-3,
    show_special_tokens: bool = False,
    device: str = "cpu",
) -> Dict:
    """
    Compute GAF token attributions for a single sentence.

    Parameters
    ----------
    sentence        : input text
    model_name      : HuggingFace model identifier
    variant         : "AF" | "GF" | "AGF"
    mu              : log-barrier weight (smaller → closer to true max-flow)
    show_special_tokens : include [CLS]/[SEP] in output
    device          : "cuda" or "cpu"

    Returns
    -------
    dict with keys: tokens, attributions, pred_class, log_odd, comp, suff, time
    """
    t0 = time.perf_counter()
    model, tokenizer = _get_cached(model_name, device)

    # ── helpers for metrics ──────────────────────────────────────────────────
    if "distilbert" in model_name:
        from distilbert_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "roberta" in model_name:
        from roberta_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "bert" in model_name:
        from bert_helper import get_inputs, get_base_token_emb, nn_forward_func
    else:
        raise NotImplementedError(f"No helper for {model_name}")

    # ── tokenise ─────────────────────────────────────────────────────────────
    enc = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    seq_len        = input_ids.shape[1]

    # ── collect attention weights via hooks ──────────────────────────────────
    # Keep attn weights IN graph for GF/AGF (need grad w.r.t. them).
    # For AF we can detach immediately.
    attn_weights_list: List[torch.Tensor] = []   # [1, H, t, t] each
    hooks = []
    attn_modules = _get_attn_modules(model)

    def make_attn_hook(idx: int):
        def fn(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                aw = out[1]
                if aw.dim() == 4:
                    attn_weights_list.append(aw)   # keep in graph
        return fn

    for idx, attn_mod in enumerate(attn_modules):
        hooks.append(attn_mod.register_forward_hook(make_attn_hook(idx)))

    # ── forward pass ─────────────────────────────────────────────────────────
    with torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    logits     = outputs.logits
    pred_class = int(logits.argmax(dim=-1).item())
    target     = logits[0, pred_class]

    # Fallback: use model output attentions
    if len(attn_weights_list) == 0 and outputs.attentions is not None:
        attn_weights_list = list(outputs.attentions)

    n_layers = len(attn_weights_list)

    # ── compute attention gradients (GF / AGF only) ──────────────────────────
    attn_grads_list: List[torch.Tensor] = []

    if variant in ("GF", "AGF"):
        # Grad of predicted logit w.r.t. each layer's attention weight matrix
        # All attn_weights_list entries share the same graph, so use retain_graph
        for l_idx, aw in enumerate(attn_weights_list):
            retain = (l_idx < n_layers - 1)
            try:
                (g,) = torch.autograd.grad(
                    target, aw,
                    retain_graph=True,   # always retain — multiple grads needed
                    create_graph=False,
                    allow_unused=False,
                )
                attn_grads_list.append(g.detach() if g is not None else torch.zeros_like(aw))
            except RuntimeError:
                attn_grads_list.append(torch.zeros_like(aw.detach()))

    # Detach attention weights after gradient computation
    attn_weights_det = [aw.detach() for aw in attn_weights_list]

    # ── build information tensor Ā ∈ [L, t, t] ──────────────────────────────
    A_bar = _build_information_tensor(attn_weights_det, attn_grads_list, variant)
    # A_bar: [L, t, t]

    # ── build graph capacity (Algorithm 1) ───────────────────────────────────
    cap = _get_backward_capacity(A_bar)

    # ── solve log-barrier min-cost circulation ────────────────────────────────
    f_out = _solve_mcc_barrier(cap, mu=mu)
    # f_out: [Qtl,]  total outflow per node

    # ── extract token attributions ────────────────────────────────────────────
    # Algorithm 1: super-source ss = t*(L+1)+1, super-target st = 0
    # Input-layer nodes (first transformer layer = last in backward graph) are
    # at positions t*L+1 … t*L+t  (1-indexed in the paper).
    # Their outflow toward super-target represents attribution.
    # Actually: the *first* layer of tokens (indices 1…t) feed into super-source.
    # Attributions at input tokens = outflow of nodes at layer 0:
    # nodes 1, 2, …, t  (0-indexed in our array).
    L  = n_layers
    t  = seq_len

    # Layer 0 nodes (input tokens) are at indices 1 … t in our matrix
    input_node_start = 1
    input_node_end   = t + 1   # exclusive

    scores_np = f_out[input_node_start:input_node_end]   # [t,]
    attcat_scores = torch.tensor(scores_np, dtype=torch.float32)

    # ── token filter ──────────────────────────────────────────────────────────
    tokens_raw      = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    special_ids_set = set(tokenizer.all_special_ids)

    if show_special_tokens:
        tokens = tokens_raw
        attr   = attcat_scores
    else:
        keep   = [i for i, tid in enumerate(input_ids[0].tolist())
                  if tid not in special_ids_set]
        tokens = [tokens_raw[i] for i in keep]
        attr   = attcat_scores[keep]

    # ── metrics (identical to pace_gradients.py call pattern) ────────────────
    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)   # [1, seq, d]

    base_token_emb = get_base_token_emb(model, tokenizer, device)
    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

    attr_full = attcat_scores   # full-length including special tokens

    log_odd, _ = calculate_log_odds(
        nn_forward_func, model, X, position_embed, type_embed,
        attention_mask, base_token_emb, attr_full, topk=20
    )
    comp = calculate_comprehensiveness(
        nn_forward_func, model, X, position_embed, type_embed,
        attention_mask, base_token_emb, attr_full, topk=20
    )
    suff = calculate_sufficiency(
        nn_forward_func, model, X, position_embed, type_embed,
        attention_mask, base_token_emb, attr_full, topk=20
    )

    return {
        "tokens":       tokens,
        "attributions": attr,
        "pred_class":   pred_class,
        "log_odd":      log_odd,
        "comp":         comp,
        "suff":         suff,
        "time":         time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate GAF attributions on sentiment datasets."
    )
    parser.add_argument("--model",   type=str, default="distilbert",
                        choices=["distilbert", "bert", "roberta"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["sst2", "imdb", "rotten"])
    parser.add_argument("--variant", type=str, default="AGF",
                        choices=["AF", "GF", "AGF"],
                        help="Information tensor variant (AF | GF | AGF)")
    parser.add_argument("--mu",         type=float, default=1e-3,
                        help="Log-barrier weight μ (smaller→closer to max-flow)")
    parser.add_argument("--n_samples",  type=int, default=2000)
    parser.add_argument("--print_step", type=int, default=100)
    args = parser.parse_args()

    MODEL_MAP = {
        "distilbert": {
            "sst2":   "distilbert-base-uncased-finetuned-sst-2-english",
            "imdb":   "textattack/distilbert-base-uncased-imdb",
            "rotten": "textattack/distilbert-base-uncased-rotten-tomatoes",
        },
        "bert": {
            "sst2":   "textattack/bert-base-uncased-SST-2",
            "imdb":   "textattack/bert-base-uncased-imdb",
            "rotten": "textattack/bert-base-uncased-rotten-tomatoes",
        },
        "roberta": {
            "sst2":   "textattack/roberta-base-SST-2",
            "imdb":   "textattack/roberta-base-imdb",
            "rotten": "textattack/roberta-base-rotten-tomatoes",
        },
    }
    model_name = MODEL_MAP[args.model][args.dataset]
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model   : {model_name}")
    print(f"Dataset : {args.dataset}")
    print(f"Variant : {args.variant}")
    print(f"μ       : {args.mu}")
    print(f"Device  : {device}")

    # ── demo ──────────────────────────────────────────────────────────────────
    demo_text = (
        "This is a really bad movie, although it has a promising start, "
        "it ended on a very low note."
    )
    print(f"\n--- GAF-{args.variant} demo attribution ---")
    res_demo = gaf_classification(
        demo_text, model_name=model_name,
        variant=args.variant, mu=args.mu,
        show_special_tokens=False, device=device,
    )
    for tok, val in zip(res_demo["tokens"], res_demo["attributions"]):
        print(f"  {tok:>15s} : {val.item():+.6f}")
    print(f"  log_odd={res_demo['log_odd']:.4f}  "
          f"comp={res_demo['comp']:.4f}  suff={res_demo['suff']:.4f}")

    # ── dataset ───────────────────────────────────────────────────────────────
    print("\nLoading dataset ...")
    if args.dataset == "imdb":
        ds   = load_dataset("imdb")["test"]
        data = list(zip(ds["text"], ds["label"]))
        data = random.sample(data, min(args.n_samples, len(data)))
    elif args.dataset == "sst2":
        ds   = load_dataset("glue", "sst2")["validation"]
        data = list(zip(ds["sentence"], ds["label"], ds["idx"]))
    elif args.dataset == "rotten":
        ds   = load_dataset("rotten_tomatoes")["test"]
        data = list(zip(ds["text"], ds["label"]))
        data = random.sample(data, min(args.n_samples, len(data)))

    print(f"Evaluating {len(data)} samples with GAF-{args.variant} ...\n")

    log_odds_sum = comps_sum = suffs_sum = total_time_sum = 0.0
    count = 0

    for row in tqdm.tqdm(data):
        text = row[0]
        try:
            res = gaf_classification(
                text, model_name=model_name,
                variant=args.variant, mu=args.mu,
                show_special_tokens=False, device=device,
            )
            log_odds_sum   += res["log_odd"]
            comps_sum      += res["comp"]
            suffs_sum      += res["suff"]
            total_time_sum += res["time"]
            count += 1
        except Exception as e:
            print(f"[WARN] skipped: {e}")
            continue

        if count % args.print_step == 0:
            print(
                f"[{count:>5d}]  "
                f"Log-odds: {log_odds_sum / count:.4f}  "
                f"Comp: {comps_sum / count:.4f}  "
                f"Suff: {suffs_sum / count:.4f}  "
                f"Time/sample: {total_time_sum / count:.4f}s"
            )

    print(f"\n=== Final Results — GAF-{args.variant} ===")
    n = max(count, 1)
    print(
        f"Log-odds         : {log_odds_sum / n:.4f}\n"
        f"Comprehensiveness: {comps_sum / n:.4f}\n"
        f"Sufficiency      : {suffs_sum / n:.4f}\n"
        f"Time/sample      : {total_time_sum / n:.4f}s\n"
        f"Total samples    : {count}"
    )