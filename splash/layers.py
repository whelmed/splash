from __future__ import annotations
from typing import Sequence, Dict, List
import numpy as np

from .types import EvalKnobs, LayerCurve, CoherenceBands
from .tiling import coherence_map_for_sequence, iter_tiles
from .geometry import auto_knobs_to_scion

# SCION utilities
from scion.utils import (
    pairwise_diffs,
    gaussian_kernel_from_d2,
    row_normalize,
    sym_normalize,
    sinkhorn_rowcol_normalize,
)

def _pick_reference_N(Ns: Sequence[int]) -> int:
    arr = sorted(set(int(n) for n in Ns if n > 0))
    if not arr:
        return 8
    return arr[len(arr)//2]  # median-like

# --- helpers for new metrics -------------------------------------------------

def _kernel_row_probs(X: np.ndarray, *, reach: float, norm: str = "row", sym_blend: float = 0.0) -> np.ndarray:
    """
    Build SCION-consistent kernel and return per-row probability vectors (no self-mass).
    """
    d2, _ = pairwise_diffs(X)
    W = gaussian_kernel_from_d2(d2, reach=reach)
    if norm == "sym":
        Wn = sym_normalize(W)
    elif norm == "sinkhorn":
        Wn = sinkhorn_rowcol_normalize(W)
    else:
        Wr = row_normalize(W)
        if sym_blend and sym_blend > 0.0:
            Ws = sym_normalize(W)
            Wn = (1.0 - float(sym_blend)) * Wr + float(sym_blend) * Ws
        else:
            Wn = Wr
    # convert to per-row probabilities (guard rows)
    r = Wn.sum(axis=1, keepdims=True) + 1e-12
    P = Wn / r
    np.fill_diagonal(P, 0.0)
    return P

def _entropy_tile(X: np.ndarray, scion_reach: float, norm: str, sym_blend: float) -> float:
    """
    Coherence entropy ℋ for a tile = mean row entropy of the normalized kernel.
    """
    P = _kernel_row_probs(X, reach=scion_reach, norm=norm, sym_blend=sym_blend)
    eps = 1e-18
    H_rows = -np.sum(P * np.log(P + eps), axis=1)
    return float(np.mean(H_rows))

def _energy_flux_from_tiles(tiles) -> float:
    """
    Energy flux ΦE proxy = slope of ledger across tile index (signed).
    """
    y = [getattr(tm.measures, "ledger", 0.0) for tm in tiles]
    if len(y) <= 1:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    # robust slope via least squares
    slope = float(np.polyfit(x, np.asarray(y, dtype=np.float64), deg=1)[0])
    return slope

def _barrier_from_tiles(tiles, bands: CoherenceBands) -> float:
    """
    Barrier B >= 0 outside the safe set; negative or zero inside.
    Here we use alignment band as the barrier backbone: B = mean(pass - R).
    """
    gaps = [float(bands.align_pass - getattr(tm.measures, "alignment_score", 0.0)) for tm in tiles]
    return float(np.mean(gaps)) if gaps else 0.0

# --- main API ----------------------------------------------------------------

def layer_curves_from_hidden(
    hidden_layers: List[np.ndarray],   # each: (T, d)
    *,
    eval_knobs: EvalKnobs,
    bands: CoherenceBands,
) -> LayerCurve:
    """
    Compute per-layer trajectories for a set of metrics by tiling each layer's sequence of places.
    Existing metrics (averaged across tiles): alignment_score (R), tension (T), bend_spread (Q), asymmetry (|beta|).
    Added metrics:
        - entropy        (ℋ)      : mean row-entropy of SCION-normalized kernel per tile, averaged over tiles
        - energy_flux    (ΦE)     : slope of ledger across tiles
        - barrier        (B)      : mean(pass - R) across tiles (positive outside safe band)
    """
    refN = _pick_reference_N(eval_knobs.Ns)
    stride = max(1, int(round(refN * eval_knobs.stride_fraction)))

    # base knobs → single-N local eval for curves
    local_knobs = EvalKnobs(
        Ns=(refN,),
        stride_fraction=eval_knobs.stride_fraction,
        distance=eval_knobs.distance,
        normalize=eval_knobs.normalize,
        sym_blend=eval_knobs.sym_blend,
        target_degree=eval_knobs.target_degree,
        degree_tolerance=eval_knobs.degree_tolerance,
        k_neighbors=eval_knobs.k_neighbors,
        mutual_knn=eval_knobs.mutual_knn,
        ensure_connected=eval_knobs.ensure_connected,
        layer_combine=eval_knobs.layer_combine,
        record_tiles=True,
        max_tokens=eval_knobs.max_tokens,
    )

    # metrics to aggregate from SCION Measures
    base_metric_names = ("alignment_score", "tension", "bend_spread", "asymmetry")
    # new metrics we add
    extra_metric_names = ("entropy", "energy_flux", "barrier")
    all_metric_names = base_metric_names + extra_metric_names

    per_layer: Dict[str, List[float]] = {m: [] for m in all_metric_names}

    # SCION knob parameters needed for ℋ
    scion_knobs = auto_knobs_to_scion(local_knobs)
    k_norm = scion_knobs.norm if hasattr(scion_knobs, "norm") else "row"
    k_reach = float(getattr(scion_knobs, "reach", 1.5))
    k_sym_blend = float(getattr(scion_knobs, "sym_blend", 0.0))

    for Xi in hidden_layers:
        # 1) standard per-tile measures (alignment, tension, Q, asymmetry)
        res = coherence_map_for_sequence(Xi, eval_knobs=local_knobs, bands=bands)
        tiles = res.tiles[refN] if refN in res.tiles else []

        # aggregate base metrics
        for m in base_metric_names:
            vals = [float(getattr(tm.measures, m)) for tm in tiles if hasattr(tm.measures, m)]
            per_layer[m].append(float(np.mean(vals) if vals else 0.0))

        # 2) entropy ℋ per tile via SCION-consistent kernel, then average
        H_vals: List[float] = []
        # build tiles the same way as the coherence map did
        T = Xi.shape[0]
        specs = iter_tiles(T=T, N=refN, stride=stride)
        for spec in specs:
            Xtile = Xi[spec.start:spec.end]
            if Xtile.shape[0] >= 2:
                H_vals.append(_entropy_tile(Xtile, scion_reach=k_reach, norm=k_norm, sym_blend=k_sym_blend))
        per_layer["entropy"].append(float(np.mean(H_vals) if H_vals else 0.0))

        # 3) energy flux ΦE from per-tile ledger slope
        per_layer["energy_flux"].append(_energy_flux_from_tiles(tiles))

        # 4) barrier B from band gap (pass - R); >0 outside safe set
        per_layer["barrier"].append(_barrier_from_tiles(tiles, bands))

    return LayerCurve(
        metric_names=list(all_metric_names),
        per_layer=per_layer,
        layer_names=[f"layer_{i}" for i in range(len(hidden_layers))],
        meta={"reference_N": int(refN)},
    )


# from __future__ import annotations
# from typing import Sequence, Dict, List, Optional
# import numpy as np

# from .types import EvalKnobs, LayerCurve
# from .tiling import coherence_map_for_sequence

# def _pick_reference_N(Ns: Sequence[int]) -> int:
#     arr = sorted(set(int(n) for n in Ns if n > 0))
#     if not arr:
#         return 8
#     return arr[len(arr)//2]  # median-like

# def layer_curves_from_hidden(
#     hidden_layers: List[np.ndarray],   # each: (T, d)
#     *,
#     eval_knobs: EvalKnobs,
#     metric_names: Sequence[str] = ("alignment_score", "tension", "asymmetry"),
#     bands=None,  # optional CoherenceBands; only needed to build MapResult tiles (labels not used here)
# ) -> LayerCurve:
#     """
#     For each layer's hidden states:
#       - build a coherence map at a single reference N (median of eval_knobs.Ns)
#       - aggregate mean over tiles for requested metrics
#     """
#     from .types import CoherenceBands
#     if bands is None:
#         bands = CoherenceBands()

#     refN = _pick_reference_N(eval_knobs.Ns)
#     local_knobs = EvalKnobs(
#         Ns=(refN,),
#         stride_fraction=eval_knobs.stride_fraction,
#         distance=eval_knobs.distance,
#         normalize=eval_knobs.normalize,
#         sym_blend=eval_knobs.sym_blend,
#         target_degree=eval_knobs.target_degree,
#         degree_tolerance=eval_knobs.degree_tolerance,
#         k_neighbors=eval_knobs.k_neighbors,
#         mutual_knn=eval_knobs.mutual_knn,
#         ensure_connected=eval_knobs.ensure_connected,
#         layer_combine=eval_knobs.layer_combine,
#         record_tiles=True,
#         max_tokens=eval_knobs.max_tokens,
#     )

#     per_layer: Dict[str, List[float]] = {m: [] for m in metric_names}
#     for Xi in hidden_layers:
#         res = coherence_map_for_sequence(Xi, eval_knobs=local_knobs, bands=bands)
#         # since we used one N only, just average tile metrics
#         for m in metric_names:
#             vals = []
#             for tm in res.tiles[refN]:
#                 if hasattr(tm.measures, m):
#                     vals.append(getattr(tm.measures, m))
#             per_layer[m].append(float(np.mean(vals) if vals else 0.0))

#     return LayerCurve(
#         metric_names=list(metric_names),
#         per_layer=per_layer,
#         layer_names=[f"layer_{i}" for i in range(len(hidden_layers))],
#         meta={"reference_N": int(refN)},
#     )
