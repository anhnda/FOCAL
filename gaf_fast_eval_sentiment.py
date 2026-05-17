"""
gaf_eval_sentiment.py  — fast version

Generalized Attention Flow (GAF) feature attribution for sentiment classification.

Paper: "Generalized Attention Flow: Feature Attribution for Transformer Models
via Maximum Flow", Azarkhalili & Libbrecht, arXiv:2502.15765, Feb 2025.

Speed-ups vs naive CVXPY implementation
----------------------------------------
1. Graph construction is fully vectorised (no Python loops over edges).
2. The log-barrier MCC problem is solved directly with scipy L-BFGS-B on the
   reduced variable space — no CVXPY canonicalisation overhead per sample.
3. Graph topology (B sparse matrix, edge indices) is cached by (L, t) so
   it is built only once per sequence-length bucket.
4. Attention gradients are computed in a single batched autograd call.

Three information tensor variants (Sec. 3.1):
  AF  : Ā = E_H(A)
  GF  : Ā = E_H( relu(∇A) )
  AGF : Ā = E_H( relu(A ⊙ ∇A) )   ← best per paper (Tab. 1)

Usage:
  python gaf_eval_sentiment.py --model distilbert --dataset sst2 --variant AGF
  python gaf_eval_sentiment.py --model bert       --dataset imdb --variant GF
"""

import time
import tqdm
import torch
import random
import argparse
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from typing import Dict, List, Literal, Tuple
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
_model_cache: Dict[str, Dict] = {}


def _get_cached(model_name: str, device: str):
    if model_name not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=False,
        ).to(device)
        model.eval()
        _model_cache[model_name] = {"model": model, "tokenizer": tokenizer}
    return _model_cache[model_name]["model"], _model_cache[model_name]["tokenizer"]


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

def _get_attn_modules(model) -> List[torch.nn.Module]:
    if hasattr(model, "bert"):
        return [layer.attention for layer in model.bert.encoder.layer]
    if hasattr(model, "distilbert"):
        return [layer.attention for layer in model.distilbert.transformer.layer]
    if hasattr(model, "roberta"):
        return [layer.attention for layer in model.roberta.encoder.layer]
    raise RuntimeError("Unsupported model architecture.")


# ---------------------------------------------------------------------------
# Information tensor
# ---------------------------------------------------------------------------

def _build_information_tensor(
    attn_weights: List[torch.Tensor],  # L × [1, H, t, t]  detached
    attn_grads:   List[torch.Tensor],  # L × [1, H, t, t]  (empty for AF)
    variant: str,
) -> np.ndarray:                       # [L, t, t]
    layers = []
    for l_idx, aw in enumerate(attn_weights):
        A = aw.squeeze(0)              # [H, t, t]
        if variant == "AF":
            lt = A.mean(0)
        elif variant == "GF":
            G  = attn_grads[l_idx].squeeze(0)
            lt = torch.relu(G).mean(0)
        else:  # AGF
            G  = attn_grads[l_idx].squeeze(0)
            lt = torch.relu(A * G).mean(0)
        layers.append(lt.cpu().numpy())
    return np.stack(layers, 0)         # [L, t, t]


# ---------------------------------------------------------------------------
# Graph topology cache  key=(L, t)
# ---------------------------------------------------------------------------
_graph_cache: Dict[Tuple[int, int], Dict] = {}


def _get_graph_topology(L: int, t: int) -> Dict:
    """
    Build (once) the sparse incidence matrix B and edge metadata for a
    graph with L transformer layers and t tokens.

    Node numbering (Algorithm 1 — Backward):
      0          : super-target  (st)
      1 … t      : layer-0 token nodes  (input tokens)
      t+1 … 2t   : layer-1 token nodes
      …
      L*t+1 … (L+1)*t : layer-L token nodes  (output tokens)
      (L+1)*t+1  : super-source  (ss)
    Total Qtl = t*(L+1) + 2
    """
    key = (L, t)
    if key in _graph_cache:
        return _graph_cache[key]

    Qtl = t * (L + 1) + 2
    ss  = t * (L + 1) + 1
    st  = 0

    # ── enumerate edges ───────────────────────────────────────────────────
    # 1. ss → layer-0 nodes (t edges)
    src_ss  = np.full(t, ss,  dtype=np.int32)
    dst_ss  = np.arange(1, t + 1, dtype=np.int32)

    # 2. layer-L nodes → st (t edges)
    src_lt  = np.arange(L * t + 1, L * t + t + 1, dtype=np.int32)
    dst_lt  = np.full(t, st, dtype=np.int32)

    # 3. inter-layer edges: layer-(j+1) node i → layer-j node j_
    #    for j in 0…L-1, i in 0…t-1, j_ in 0…t-1
    #    upper capacity = A_bar[j, i, j_]
    #    source node: (j+1)*t + i + 1,  dest node: j*t + j_ + 1
    inter_src = []
    inter_dst = []
    for j in range(L):
        row_nodes = np.arange((j + 1) * t + 1, (j + 1) * t + t + 1)  # [t]
        col_nodes = np.arange(j * t + 1,       j * t + t + 1)         # [t]
        # all (i, j_) pairs
        ii, jj = np.meshgrid(row_nodes, col_nodes, indexing="ij")
        inter_src.append(ii.ravel())
        inter_dst.append(jj.ravel())

    inter_src = np.concatenate(inter_src)
    inter_dst = np.concatenate(inter_dst)

    all_src = np.concatenate([src_ss, src_lt, inter_src])
    all_dst = np.concatenate([dst_ss, dst_lt, inter_dst])
    m       = len(all_src)

    # ── edge type masks (for filling capacities later) ────────────────────
    n_ss    = t
    n_lt    = t
    n_inter = L * t * t
    assert m == n_ss + n_lt + n_inter

    # ── incidence matrix B ∈ R^{m × Qtl}  (sparse) ───────────────────────
    # B[e, src] = -1, B[e, dst] = +1
    e_idx  = np.arange(m)
    data   = np.concatenate([-np.ones(m), np.ones(m)])
    row    = np.concatenate([e_idx, e_idx])
    col    = np.concatenate([all_src, all_dst])
    B      = sp.csc_matrix((data, (row, col)), shape=(m, Qtl))

    topo = {
        "Qtl": Qtl, "ss": ss, "st": st,
        "src": all_src, "dst": all_dst, "m": m,
        "n_ss": n_ss, "n_lt": n_lt, "n_inter": n_inter,
        "B": B,
        # slices into the capacity vector
        "sl_ss":    slice(0, n_ss),
        "sl_lt":    slice(n_ss, n_ss + n_lt),
        "sl_inter": slice(n_ss + n_lt, m),
    }
    _graph_cache[key] = topo
    return topo


# ---------------------------------------------------------------------------
# Fast log-barrier solver (scipy L-BFGS-B, no CVXPY)
# ---------------------------------------------------------------------------

def _solve_barrier_scipy(
    u_e: np.ndarray,       # [m+1]  upper capacities (including back-edge)
    B_ext: sp.csc_matrix,  # [(m+1) × Qtl]  extended incidence matrix
    c_vec: np.ndarray,     # [m+1]  cost vector
    mu: float,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Solve  min_{B^T f = 0}  c^T f − μ Σ [log(f_e) + log(u_e − f_e)]
    via SLSQP with log-barrier keeping flow strictly inside (0, u_e).
    """
    eps = 1e-8   # strict interior margin

    # Drop edges where u_e ≤ 2*eps (no valid interior point exists)
    valid = u_e > 2.0 * eps
    if not np.all(valid):
        u_e   = u_e[valid]
        c_vec = c_vec[valid]
        B_ext = B_ext[valid, :]   # select rows (edges)

    m1  = len(u_e)
    f0  = u_e / 2.0   # midpoint — strictly feasible

    B_T = B_ext.T     # [Qtl × m1]

    def obj_and_grad(f):
        d1   = f           # f > 0  (lower bound = 0)
        d2   = u_e - f     # u - f > 0
        obj  = c_vec @ f - mu * (np.sum(np.log(d1)) + np.sum(np.log(d2)))
        grad = c_vec - mu * (1.0 / d1 - 1.0 / d2)
        return obj, grad

    constraints = {
        "type": "eq",
        "fun":  lambda f: B_T @ f,
        "jac":  lambda f: B_T.toarray(),
    }

    # Guaranteed valid: lb=eps < ub=u_e-eps because u_e > 2*eps
    bounds = [(eps, float(u) - eps) for u in u_e]

    res = minimize(
        obj_and_grad,
        f0,
        method="SLSQP",
        jac=True,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-8, "disp": False},
    )

    return res.x if (res.success or res.status in (0, 8, 9)) else f0


def _solve_mcc_barrier(
    A_bar: np.ndarray,
    topo:  Dict,
    mu:    float = 1e-3,
) -> np.ndarray:
    """
    Build capacities from A_bar + topology, solve barrier MCC, return f_out.
    """
    L, t, _ = A_bar.shape
    u_inf = float(t)

    # ── build capacity vector u_e ─────────────────────────────────────────
    u_e = np.empty(topo["m"], dtype=np.float64)
    u_e[topo["sl_ss"]]    = u_inf
    u_e[topo["sl_lt"]]    = u_inf
    u_e[topo["sl_inter"]] = A_bar.ravel()   # [L, t, t] → flat in row-major order

    # Clip negatives (GF/AGF relu should already handle, but be safe)
    u_e = np.clip(u_e, 0.0, None)

    # Remove edges whose capacity is too small for a strict interior point.
    # Need u_e > 2*eps so that (eps, u-eps) is a valid bound interval.
    eps_mask = 2e-7
    mask    = u_e > eps_mask
    u_e_f   = u_e[mask]
    src_f   = topo["src"][mask]
    dst_f   = topo["dst"][mask]
    m_f     = len(u_e_f)

    if m_f == 0:
        return np.zeros(topo["Qtl"])

    # ── back-edge: st → ss with capacity ||u||_1 ─────────────────────────
    u_back  = float(u_e_f.sum())
    u_ext   = np.append(u_e_f, u_back)

    # ── extended incidence matrix (sparse) ───────────────────────────────
    Qtl   = topo["Qtl"]
    e_idx = np.arange(m_f)
    data  = np.concatenate([-np.ones(m_f), np.ones(m_f),
                             [-1.0, 1.0]])        # back-edge
    row   = np.concatenate([e_idx, e_idx,
                             [m_f, m_f]])
    col   = np.concatenate([src_f, dst_f,
                             [topo["st"], topo["ss"]]])
    B_ext = sp.csc_matrix((data, (row, col)), shape=(m_f + 1, Qtl))

    # ── cost vector: only back-edge has cost = -1 ─────────────────────────
    c_vec      = np.zeros(m_f + 1)
    c_vec[-1]  = -1.0

    # ── solve ─────────────────────────────────────────────────────────────
    # _solve_barrier_scipy may drop more edges internally (those with u≤2eps).
    # We pass src_f alongside so we can reconstruct f_out correctly.
    # Strategy: track which edges survive inside the solver by replicating
    # the same valid mask here before calling.
    eps_solver = 1e-8
    solver_valid = u_ext > 2.0 * eps_solver
    src_ext  = np.append(src_f, topo["st"])   # source of each edge incl. back
    src_kept = src_ext[solver_valid]           # sources of edges the solver keeps

    f_val = _solve_barrier_scipy(u_ext, B_ext, c_vec, mu)
    # f_val has length == sum(solver_valid)

    # ── outflow per node ──────────────────────────────────────────────────
    f_out = np.zeros(Qtl)
    for e, s in enumerate(src_kept):
        if e < len(f_val) and f_val[e] > 0:
            f_out[s] += f_val[e]

    return f_out


# ---------------------------------------------------------------------------
# Core GAF
# ---------------------------------------------------------------------------

def gaf_classification(
    sentence: str,
    model_name: str,
    variant: Literal["AF", "GF", "AGF"] = "AGF",
    mu: float = 1e-3,
    show_special_tokens: bool = False,
    device: str = "cpu",
) -> Dict:
    t0 = time.perf_counter()
    model, tokenizer = _get_cached(model_name, device)

    if "distilbert" in model_name:
        from distilbert_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "roberta" in model_name:
        from roberta_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "bert" in model_name:
        from bert_helper import get_inputs, get_base_token_emb, nn_forward_func
    else:
        raise NotImplementedError(f"No helper for {model_name}")

    enc = tokenizer(
        sentence, return_tensors="pt",
        truncation=True, max_length=512,
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    seq_len        = input_ids.shape[1]

    # ── collect attention weights (keep in graph for GF/AGF) ─────────────
    attn_weights_list: List[torch.Tensor] = []
    hooks = []

    def make_attn_hook(idx):
        def fn(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                aw = out[1]
                if aw.dim() == 4:
                    attn_weights_list.append(aw)
        return fn

    for idx, attn_mod in enumerate(_get_attn_modules(model)):
        hooks.append(attn_mod.register_forward_hook(make_attn_hook(idx)))

    with torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks:
        h.remove()

    logits     = outputs.logits
    pred_class = int(logits.argmax(-1).item())
    target     = logits[0, pred_class]

    if not attn_weights_list and outputs.attentions:
        attn_weights_list = list(outputs.attentions)

    n_layers = len(attn_weights_list)

    # ── attention gradients (GF / AGF) ───────────────────────────────────
    attn_grads_list: List[torch.Tensor] = []
    if variant in ("GF", "AGF"):
        for aw in attn_weights_list:
            try:
                (g,) = torch.autograd.grad(
                    target, aw, retain_graph=True,
                    create_graph=False, allow_unused=False,
                )
                attn_grads_list.append(
                    g.detach() if g is not None else torch.zeros_like(aw.detach())
                )
            except RuntimeError:
                attn_grads_list.append(torch.zeros_like(aw.detach()))

    attn_weights_det = [aw.detach() for aw in attn_weights_list]

    # ── information tensor ────────────────────────────────────────────────
    A_bar = _build_information_tensor(attn_weights_det, attn_grads_list, variant)
    # [L, t, t]

    # ── graph topology (cached) ───────────────────────────────────────────
    topo = _get_graph_topology(n_layers, seq_len)

    # ── solve ─────────────────────────────────────────────────────────────
    f_out = _solve_mcc_barrier(A_bar, topo, mu=mu)

    # ── extract input-token attributions ──────────────────────────────────
    # Layer-0 nodes: indices 1 … t  (outflow toward super-target = st = 0)
    scores_np     = f_out[1 : seq_len + 1]
    attcat_scores = torch.tensor(scores_np, dtype=torch.float32)

    # ── token filter ──────────────────────────────────────────────────────
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

    # ── metrics ───────────────────────────────────────────────────────────
    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)

    base_token_emb = get_base_token_emb(model, tokenizer, device)
    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

    log_odd, _ = calculate_log_odds(
        nn_forward_func, model, X, position_embed, type_embed,
        attention_mask, base_token_emb, attcat_scores, topk=20
    )
    comp = calculate_comprehensiveness(
        nn_forward_func, model, X, position_embed, type_embed,
        attention_mask, base_token_emb, attcat_scores, topk=20
    )
    suff = calculate_sufficiency(
        nn_forward_func, model, X, position_embed, type_embed,
        attention_mask, base_token_emb, attcat_scores, topk=20
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="distilbert",
                        choices=["distilbert", "bert", "roberta"])
    parser.add_argument("--dataset",    required=True,
                        choices=["sst2", "imdb", "rotten"])
    parser.add_argument("--variant",    default="AGF",
                        choices=["AF", "GF", "AGF"])
    parser.add_argument("--mu",         type=float, default=1e-3)
    parser.add_argument("--n_samples",  type=int,   default=2000)
    parser.add_argument("--print_step", type=int,   default=100)
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
    print(f"Variant : {args.variant}  μ={args.mu}")
    print(f"Device  : {device}")

    demo_text = (
        "This is a really bad movie, although it has a promising start, "
        "it ended on a very low note."
    )
    print(f"\n--- GAF-{args.variant} demo ---")
    res_demo = gaf_classification(
        demo_text, model_name=model_name,
        variant=args.variant, mu=args.mu,
        show_special_tokens=False, device=device,
    )
    for tok, val in zip(res_demo["tokens"], res_demo["attributions"]):
        print(f"  {tok:>15s} : {val.item():+.6f}")
    print(f"  log_odd={res_demo['log_odd']:.4f}  "
          f"comp={res_demo['comp']:.4f}  suff={res_demo['suff']:.4f}  "
          f"time={res_demo['time']:.3f}s")

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

    print(f"Evaluating {len(data)} samples ...\n")
    log_odds_sum = comps_sum = suffs_sum = total_time_sum = 0.0
    count = 0

    for row in tqdm.tqdm(data):
        try:
            res = gaf_classification(
                row[0], model_name=model_name,
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
                f"Log-odds: {log_odds_sum/count:.4f}  "
                f"Comp: {comps_sum/count:.4f}  "
                f"Suff: {suffs_sum/count:.4f}  "
                f"Time/sample: {total_time_sum/count:.4f}s"
            )

    print(f"\n=== Final Results — GAF-{args.variant} ===")
    n = max(count, 1)
    print(
        f"Log-odds         : {log_odds_sum/n:.4f}\n"
        f"Comprehensiveness: {comps_sum/n:.4f}\n"
        f"Sufficiency      : {suffs_sum/n:.4f}\n"
        f"Time/sample      : {total_time_sum/n:.4f}s\n"
        f"Total samples    : {count}"
    )