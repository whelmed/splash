from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional
import math
import numpy as np

from .. import EvalKnobs, CoherenceBands, evaluate_prompts
from ..types import EvalRun, LayerCurve, MapResult

# ---------- Sweep spec ----------

@dataclass
class SweepSpec:
    # Each element is a list of candidate values; Cartesian product drives the sweep
    Ns: List[int] = None                  # e.g., [24, 32]
    k_neighbors: List[int] = None         # e.g., [6, 8, 12]
    settle_steps: List[int] = None        # e.g., [12, 24]
    distance: List[str] = None            # ["l2", "cosine"]
    layer_combine: List[str] = None       # ["last", "mean_last_4"]
    # fixed “good practice” defaults (can be overridden in base_knobs)
    mutual_knn: bool = True
    ensure_connected: bool = True
    target_degree: int = 16
    diag_strength_cap: float = 64.0       # avoid robust cap binding during diagnostics
    # scoring weights
    w_contrast: float = 1.0
    w_stability: float = 0.5
    w_regime: float = 1.0

    def __post_init__(self):
        if self.Ns is None: self.Ns = [24, 32]
        if self.k_neighbors is None: self.k_neighbors = [6, 8, 12]
        if self.settle_steps is None: self.settle_steps = [12, 24]
        if self.distance is None: self.distance = ["l2"]
        if self.layer_combine is None: self.layer_combine = ["mean_last_4", "last"]

# ---------- Utilities ----------

def _refN(Ns: Iterable[int]) -> int:
    arr = sorted({int(n) for n in Ns if n > 0})
    return arr[len(arr)//2] if arr else 8

def _entropy_norm(H: float, k_eff: float) -> float:
    k_eff = max(2.0, float(k_eff))
    denom = math.log(k_eff)
    return float(H / denom) if denom > 0 else 0.0

def _agg_curve(curve: LayerCurve) -> Dict[str, float]:
    per = curve.per_layer
    def last(key): return float(per.get(key, [0.0])[-1]) if key in per else 0.0
    def mean(key):
        return float(np.mean(per.get(key, [0.0]))) if key in per else 0.0
    return {
        "R_last":      last("alignment_score"),
        "R_mean":      mean("alignment_score"),
        "H_mean":      mean("entropy"),
        "PhiE_mean":   mean("energy_flux"),
        "rho_phi_mean":mean("asymmetry") and float(np.mean([abs(x) for x in per["asymmetry"]])) or 0.0,
    }

def _tiles_deg_and_R(mres: MapResult, refN: int) -> Tuple[List[float], List[float]]:
    ts = mres.tiles.get(refN, [])
    deg = [float(tm.measures.effective_degree) for tm in ts]
    R   = [float(tm.measures.alignment_score) for tm in ts]
    return deg, R

def _score_config(rows: List[Dict[str, float]], deg_all: List[float], spec: SweepSpec, k: int, N: int) -> float:
    """
    Simple composite score:
      + contrast: std over prompts of R_last and ΦE_mean
      + stability: low tile-R std (within each prompt) but >0 degree CV (avoid perfect regularity)
      + regime: degree near k (or target) and entropy_norm in [0.6, 0.95]
    """
    # rows: one per prompt (aggregated)
    r_last = np.array([r["R_last"] for r in rows], dtype=float)
    phiE   = np.array([r["PhiE_mean"] for r in rows], dtype=float)

    contrast = float(np.std(r_last) + 0.2 * np.std(phiE))  # emphasize R

    # degree regularity (per-sweep across prompts)
    deg = np.array(deg_all, dtype=float)
    deg_mean = float(np.mean(deg)) if deg.size else 0.0
    deg_std  = float(np.std(deg))  if deg.size else 0.0
    cv_deg   = deg_std / (deg_mean + 1e-9)

    # “some” irregularity is good; penalize cv≈0 and huge cv
    irr = float(max(0.0, min(1.0, cv_deg / 0.25)))  # ~0.25 cv hits 1.0

    # entropy regime (use average across prompts if available)
    H = np.array([r.get("H_norm", 0.0) for r in rows], dtype=float)
    H_ok = float(np.mean(np.clip((H - 0.6) / 0.35, 0.0, 1.0)))  # 0.6..0.95

    # degree target vs achieved
    deg_target = min(spec.target_degree, N - 1)
    deg_ok = math.exp(-abs(deg_mean - deg_target) / 4.0)

    stability = float(1.0 - np.mean([r.get("tile_R_std", 0.0) for r in rows]))

    return (spec.w_contrast * contrast
            + spec.w_stability * stability
            + spec.w_regime * (0.6 * H_ok + 0.4 * irr + 0.4 * deg_ok))

# ---------- Calibrate ----------

@dataclass
class CalibResult:
    summary: List[Dict[str, Any]]     # one row per config
    best: Dict[str, Any]               # best row
    profile: "ModelProfile"            # filled profile

def calibrate_model(
    *,
    adapter,                          # Splash ModelAdapter
    prompts: List[str],
    base_knobs: Optional[EvalKnobs] = None,
    bands: Optional[CoherenceBands] = None,
    sweep: Optional[SweepSpec] = None,
) -> CalibResult:
    """
    Run a sweep over (N, k, settle_steps, distance, layer_combine) to select an
    Operating Point (OP) that maximizes discriminability × stability. Emits a
    ModelProfile that can be persisted and reloaded.

    Robustness notes:
    - Prefilters Ns by tokenized length so we don't consider tile sizes that
      cannot be formed for the current prompts.
    - Skips any configuration where the reference-N produced zero tiles for
      any prompt (non-comparable).
    - Avoids complete graphs by skipping k >= N.
    """
    # Lazy import to avoid circulars
    from splash.calibrate.profile import ModelProfile, ProfileStats

    if base_knobs is None:
        base_knobs = EvalKnobs()
    if bands is None:
        bands = CoherenceBands()
    if sweep is None:
        sweep = SweepSpec()

    # --- Preflight: get a safe cap on N from adapter.tokenize ---
    try:
        batch = adapter.tokenize(prompts, max_length=getattr(base_knobs, "max_tokens", None))
        T_cap = int(getattr(batch, "input_ids").shape[1])
    except Exception:
        T_cap = None  # adapter may not expose shapes; fall back to user Ns

    # Filter Ns that can actually tile these prompts
    Ns_filtered = list(sweep.Ns)
    if T_cap is not None:
        Ns_filtered = [n for n in sweep.Ns if n <= T_cap]
    if not Ns_filtered:
        # Fall back to a feasible single N (keeps tests stable)
        fallbackN = max(8, T_cap or 16)
        Ns_filtered = [fallbackN]

    rows: List[Dict[str, Any]] = []

    for N in Ns_filtered:
        for k in sweep.k_neighbors:
            if k >= N:
                # Avoid near-complete graphs; user can increase N if they need large k
                continue
            for settle in sweep.settle_steps:
                for dist in sweep.distance:
                    for lc in sweep.layer_combine:
                        ek = EvalKnobs(
                            Ns=(N,),
                            stride_fraction=base_knobs.stride_fraction,
                            distance=dist,
                            normalize=base_knobs.normalize,
                            sym_blend=base_knobs.sym_blend,
                            target_degree=sweep.target_degree,
                            degree_tolerance=base_knobs.degree_tolerance,
                            k_neighbors=k,
                            mutual_knn=sweep.mutual_knn,
                            ensure_connected=sweep.ensure_connected,
                            layer_combine=lc,
                            record_tiles=True,
                            max_tokens=base_knobs.max_tokens,
                            settle_steps=settle,
                            # diagnostics overrides so degree is not cap-bound
                            diag_strength_cap=sweep.diag_strength_cap,
                            scion_preset=getattr(base_knobs, "scion_preset", "robust"),
                        )

                        runs: List[EvalRun] = evaluate_prompts(
                            prompts, adapter=adapter, eval_knobs=ek, bands=bands
                        )

                        refN = _refN(ek.Ns)
                        deg_all: List[float] = []
                        prompt_rows: List[Dict[str, float]] = []
                        empty_any = False

                        for run in runs:
                            # If this refN produced no tiles, skip this config entirely
                            if not run.coherence_maps:
                                empty_any = True
                                break
                            mkey = next(iter(run.coherence_maps.keys()))
                            mres: MapResult = run.coherence_maps[mkey]
                            deg, tile_R = _tiles_deg_and_R(mres, refN)
                            if len(deg) == 0:
                                empty_any = True
                                break

                            agg = _agg_curve(run.layer_curves)
                            k_eff = float(np.mean(deg))
                            agg["H_norm"] = _entropy_norm(agg.get("H_mean", 0.0), k_eff)
                            agg["tile_R_std"] = float(np.std(tile_R)) if tile_R else 0.0

                            deg_all.extend(deg)
                            prompt_rows.append(agg)

                        if empty_any or not prompt_rows:
                            continue

                        score = _score_config(prompt_rows, deg_all, sweep, k=k, N=N)

                        row = {
                            "N": N,
                            "k": k,
                            "settle": settle,
                            "distance": dist,
                            "layer_combine": lc,
                            "deg_mean": float(np.mean(deg_all)) if deg_all else 0.0,
                            "deg_std": float(np.std(deg_all)) if deg_all else 0.0,
                            "H_norm_mean": float(np.mean([r["H_norm"] for r in prompt_rows])) if prompt_rows else 0.0,
                            "R_last_mean": float(np.mean([r["R_last"] for r in prompt_rows])) if prompt_rows else 0.0,
                            "R_last_std": float(np.std([r["R_last"] for r in prompt_rows])) if prompt_rows else 0.0,
                            "PhiE_mean": float(np.mean([r["PhiE_mean"] for r in prompt_rows])) if prompt_rows else 0.0,
                            "PhiE_std": float(np.std([r["PhiE_mean"] for r in prompt_rows])) if prompt_rows else 0.0,
                            "rho_phi_mean": float(np.mean([r["rho_phi_mean"] for r in prompt_rows])) if prompt_rows else 0.0,
                            "rho_phi_std": float(np.std([r["rho_phi_mean"] for r in prompt_rows])) if prompt_rows else 0.0,
                            "score": score,
                        }
                        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"calibrate_model: sweep produced no configurations; "
            f"consider smaller N or longer max_tokens (T_cap={T_cap}, Ns={list(sweep.Ns)})"
        )

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    best = rows_sorted[0]

    prof_stats = ProfileStats(
        degree_mean=best["deg_mean"],
        degree_std=best["deg_std"],
        entropy_norm_mean=best["H_norm_mean"],
        entropy_norm_std=0.0,
        R_last_mean=best["R_last_mean"],
        R_last_std=best["R_last_std"],
        PhiE_mean=best["PhiE_mean"],
        PhiE_std=best["PhiE_std"],
        rho_phi_mean=best["rho_phi_mean"],
        rho_phi_std=best["rho_phi_std"],
        score=best["score"],
    )

    profile = ModelProfile(
        adapter=getattr(adapter, "get_config", lambda: {"adapter": "unknown"})(),
        places={"distance": best["distance"]},
        op={
            "N": best["N"],
            "k": best["k"],
            "settle": best["settle"],
            "layer_combine": best["layer_combine"],
            "mutual_knn": sweep.mutual_knn,
            "ensure_connected": sweep.ensure_connected,
        },
        degree={"target": sweep.target_degree, "mean": best["deg_mean"], "std": best["deg_std"]},
        entropy_norm={"mean": best["H_norm_mean"]},
        thresholds={
            "R_warn": 0.65,
            "R_pass": 0.75,
            "PhiE_max": 0.0,     # inflation guard
            "rho_phi_max": 0.08, # fracture guard (tune per domain)
        },
        policy={
            "misalign": {"action": "align_apply", "window": best["N"]},
            "inflation": {"temp": 0.5, "top_p": 0.90, "duration": 3},
        },
        stats=prof_stats,
        meta={"sweep_size": len(rows_sorted), "refN": best["N"], "diag_strength_cap": sweep.diag_strength_cap},
    )

    return CalibResult(summary=rows_sorted, best=best, profile=profile)
