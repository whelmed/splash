from __future__ import annotations

"""
Adapters that convert Splash/SCION structures into dashboard-ready payloads.

- inv_grid_from_layer_curve:  map LayerCurve -> canonical invariant grid
- inv_grid_from_runs:         merge multiple LayerCurves into a single grid
- dislocation_matrix_from_tiles: build a simple ρφ-like matrix from MapResult tiles
- scg_trajectory_from_places: build SCG-3 coords (x=cosθ, y=sinθ, z=curvature proxy)
- scg_trajectory_from_hidden: convenience wrapper (hidden -> places -> trajectory)

Notes on SCG-3 mapping:
  We construct a World(places=X, knobs=...) and take the engine's phase θ per unit
  after a minimal settle (1 tick). We then compute per-unit local tension via
  World.local_tension() as a curvature-energy proxy. The dashboard trajectory uses:
      x_i = cos(θ_i), y_i = sin(θ_i), z_i = normalize(local_tension_i)
  Token "health" colors come from classifying a small window around each token
  using Splash's one_tick_measures + classify_measures.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

# Splash types & helpers
from ..types import EvalKnobs, CoherenceBands, LayerCurve, MapResult
from ..geometry import (
    to_places_from_hidden
)
from ..tiling import iter_tiles
from .dash import plot_scg_trajectory_with_overlay

# SCION public facade
from matplotlib.figure import Figure

from splash.scg.phase_frame import PhaseFrame
from splash.scg.projection import amplitude, scg3_from_A_theta_kappa
from splash.scg.curvature import kappa_knn
from splash.scg.dynamics import kuramoto_settle

# ---------------------------------------------------------------------------
# Invariant grid adapters
# ---------------------------------------------------------------------------

def inv_grid_from_layer_curve(layer_curve) -> Dict[str, Dict[str, List[float]]]:
    """
    Convert a Splash LayerCurve into the invariant grid dict expected by the dashboard.

    Maps LayerCurve.per_layer keys to canonical series:
      alignment_score  -> "alignment"
      bend_spread (Q)  -> "curvature"      (fallback: tension)
      asymmetry        -> "dislocation"    (use |asymmetry|)
      entropy          -> "entropy"
      energy_flux      -> "energy_flux"
      barrier          -> "barrier"

    Returns: {"<condition>": {<canonical_metric>: List[float], ...}}
    Missing series are simply omitted.
    """
    per = getattr(layer_curve, "per_layer", {}) or {}
    out: Dict[str, List[float]] = {}

    def _series(name: str) -> Optional[List[float]]:
        v = per.get(name)
        return list(v) if isinstance(v, (list, tuple)) and len(v) > 0 else None

    # Alignment (R)
    s = _series("alignment_score")
    if s is not None:
        out["alignment"] = s

    # Curvature proxy: prefer bend_spread (Q), else tension (T)
    s = _series("bend_spread")
    if s is None:
        s = _series("tension")
    if s is not None:
        out["curvature"] = s

    # Dislocation proxy from |asymmetry|
    s = _series("asymmetry")
    if s is not None:
        out["dislocation"] = [float(abs(x)) for x in s]

    # entropy (ℋ)
    s = _series("entropy")
    if s is not None:
        out["entropy"] = s

    # energy flux (ΦE)
    s = _series("energy_flux")
    if s is not None:
        out["energy_flux"] = s

    # barrier (B)
    s = _series("barrier")
    if s is not None:
        out["barrier"] = s

    # Legend/condition label
    condition = getattr(layer_curve, "meta", {}).get("label", "run")
    return {condition: out}


def inv_grid_from_runs(layer_curves: List[LayerCurve]) -> Dict[str, Dict[str, List[float]]]:
    """
    Merge several LayerCurves into one invariant grid dict with multiple conditions.
    """
    grid: Dict[str, Dict[str, List[float]]] = {}
    for lc in layer_curves:
        chunk = inv_grid_from_layer_curve(lc)
        grid.update(chunk)
    return grid


# ---------------------------------------------------------------------------
# Dislocation heatmap adapter
# ---------------------------------------------------------------------------

def dislocation_matrix_from_tiles(map_result: MapResult) -> np.ndarray:
    """
    Build a simple (1, T_tiles) dislocation-like matrix from a MapResult by:
      - choosing the median N,
      - extracting a per-tile value as dedicated 'dislocation' if present,
        otherwise |asymmetry| clamped to [0,1].

    Returns:
        np.ndarray shape (1, num_tiles) suitable for plot_dislocation_heatmap.
        Returns empty array if nothing sensible is available.
    """
    tiles_by_N = getattr(map_result, "tiles", {})
    if not tiles_by_N:
        return np.empty((0, 0))

    Ns = sorted(list(tiles_by_N.keys()))
    N_pick = Ns[len(Ns) // 2]
    tiles = tiles_by_N.get(N_pick, [])
    vals: List[float] = []

    for tm in tiles:
        m = getattr(tm, "measures", None)
        if m is None:
            continue
        v = getattr(m, "dislocation", None)
        if v is None:
            a = float(abs(getattr(m, "asymmetry", 0.0)))
            v = max(0.0, min(1.0, a))
        vals.append(float(v))

    if not vals:
        return np.empty((0, 0))
    return np.asarray(vals, dtype=float)[None, :]


# ---------------------------------------------------------------------------
# SCG-3 trajectory adapters
# ---------------------------------------------------------------------------

def _median_refN(eval_knobs: EvalKnobs) -> int:
    Ns = sorted(set(int(n) for n in eval_knobs.Ns if n > 0))
    if not Ns:
        return 8
    return Ns[len(Ns) // 2]


PHASE_FRAME: PhaseFrame | None = None

def set_phase_frame(frame: PhaseFrame) -> None:
    global PHASE_FRAME
    PHASE_FRAME = frame

def scg_trajectory_from_places(
    E_seq: np.ndarray,
    *,
    settle_steps: int = 16,
    k_neighbors: int = 8,
) -> Dict[str, Any]:
    """Build SCG trajectory for a short token window.
    E_seq: (T, D) embeddings for tokens in order
    Returns lists of (x,y,z) per token and per-token θ, A, κ.
    """
    assert PHASE_FRAME is not None, "Set PHASE_FRAME via set_phase_frame(...)"

    A = amplitude(E_seq)
    P2 = PHASE_FRAME.project2D(E_seq)
    theta0 = np.arctan2(P2[:, 1], P2[:, 0])
    theta = kuramoto_settle(theta0, steps=settle_steps)
    kappa = kappa_knn(E_seq, k=k_neighbors)

    xyz = []
    for i in range(1, len(A) + 1):
        agg = scg3_from_A_theta_kappa(A[:i], theta[:i], kappa[:i])
        xyz.append((agg["x"], agg["y"], agg["z"]))
    return {
        "xyz": np.asarray(xyz),  # (T,3)
        "theta": theta,
        "A": A,
        "kappa": kappa,
    }


def scg_trajectory_from_hidden(
    hidden_bt: np.ndarray,
    *,
    eval_knobs: EvalKnobs,
    bands: CoherenceBands,
    distance: Optional[str] = None,
    max_tokens: Optional[int] = None,
    return_modes: bool = False,                 # NEW (default keeps old behavior)
):
    if hidden_bt.ndim != 3 or hidden_bt.shape[0] < 1:
        raise ValueError("hidden_bt must be (B,T,d) with B>=1.")
    dist = distance or eval_knobs.distance
    Xs = to_places_from_hidden(hidden_bt[:1], distance=dist, max_tokens=max_tokens or eval_knobs.max_tokens)
    return scg_trajectory_from_places(Xs[0], eval_knobs=eval_knobs, bands=bands, return_modes=return_modes)


def _refN_eval(eval_knobs: "EvalKnobs") -> int:
    return _median_refN(eval_knobs)  # we already defined _median_refN earlier in this file

def _safe_tiles(T: int, refN: int, stride: int) -> List[Tuple[int,int]]:
    """
    Return a list of (start,end) windows. If T < refN, return a single tail window of size T.
    """
    if T <= 0:
        return []
    if T < refN:
        return [(max(0, T - refN), T)]
    return [(spec.start, spec.end) for spec in iter_tiles(T, refN, stride)]

def windowed_trajectories_from_places(
    X: "np.ndarray",                          # (T,d) places
    *,
    eval_knobs: "EvalKnobs",
    bands: "CoherenceBands",
    max_panels: int = 6,
    overlay_global: bool = True,
) -> List[Tuple["Figure", Dict[str,int]]]:
    """
    Build up to `max_panels` trajectory figures, one per sliding window at refN.
    Returns list of (Figure, meta) where meta={'start':s,'end':e,'N':refN}.
    """
    T = X.shape[0]
    refN = _refN_eval(eval_knobs)
    stride = max(1, int(round(refN * eval_knobs.stride_fraction)))
    windows = _safe_tiles(T, refN, stride)
    if not windows:
        return []

    # background coords (optional)
    if overlay_global:
        coords_all, _, _ = scg_trajectory_from_places(X, eval_knobs=eval_knobs, bands=bands, return_modes=True)
    figs: List[Tuple["Figure", Dict[str,int]]] = []

    # sample last K windows to reduce clutter
    if len(windows) > max_panels:
        windows = windows[-max_panels:]

    for (s,e) in windows:
        Xw = X[s:e]
        coords_w, flags_w, modes_w = scg_trajectory_from_places(
            Xw, eval_knobs=eval_knobs, bands=bands, return_modes=True
        )
        if overlay_global:
            fig = plot_scg_trajectory_with_overlay(
                coords_all=coords_all, coords_focus=coords_w,
                focus_flags=flags_w, mode_flags_focus=modes_w,
                title=f"SCG Trajectory (tokens {s}:{e})"
            )
        else:
            # fallback to the simpler single-path plot (already in dash)
            from .dash import plot_scg_trajectory
            fig = plot_scg_trajectory(coords_w, flags_w, title=f"SCG Trajectory (tokens {s}:{e})")
        figs.append((fig, {"start": s, "end": e, "N": refN}))
    return figs

def windowed_trajectories_from_hidden(
    hidden_bt: "np.ndarray",                  # (B,T,d)
    *,
    eval_knobs: "EvalKnobs",
    bands: "CoherenceBands",
    max_panels: int = 6,
    overlay_global: bool = True,
) -> List[Tuple["Figure", Dict[str,int]]]:
    """
    Convenience wrapper: convert hidden→places for the first sequence and call windowed trajectories.
    """
    Xs = to_places_from_hidden(hidden_bt[:1], distance=eval_knobs.distance, max_tokens=eval_knobs.max_tokens)
    return windowed_trajectories_from_places(
        Xs[0], eval_knobs=eval_knobs, bands=bands, max_panels=max_panels, overlay_global=overlay_global
    )

def trajectory_from_map_tile(
    X: "np.ndarray",               # (T,d) places for the sequence behind the map
    mres: "MapResult",
    *,
    eval_knobs: "EvalKnobs",
    bands: "CoherenceBands",
    N: Optional[int] = None,
    tile_index: int = 0,
    overlay_global: bool = True,
) -> "Figure":
    """
    Render trajectory for a specific tile chosen from a coherence MapResult (e.g., click on the heatmap).
    """
    refN = int(N if N is not None else _refN_eval(eval_knobs))
    tiles = mres.tiles.get(refN, [])
    if not tiles:
        raise ValueError(f"No tiles at N={refN} in MapResult.")
    tile_index = max(0, min(tile_index, len(tiles)-1))
    s, e = int(tiles[tile_index].spec.start), int(tiles[tile_index].spec.end)

    # background
    if overlay_global:
        coords_all, _, _ = scg_trajectory_from_places(X, eval_knobs=eval_knobs, bands=bands, return_modes=True)

    # focus
    coords_w, flags_w, modes_w = scg_trajectory_from_places(
        X[s:e], eval_knobs=eval_knobs, bands=bands, return_modes=True
    )

    if overlay_global:
        return plot_scg_trajectory_with_overlay(
            coords_all=coords_all, coords_focus=coords_w,
            focus_flags=flags_w, mode_flags_focus=modes_w,
            title=f"SCG Trajectory (tile {tile_index} @ N={refN}, {s}:{e})"
        )
    else:
        from .dash import plot_scg_trajectory
        return plot_scg_trajectory(coords_w, flags_w, title=f"SCG Trajectory (tile {tile_index} @ N={refN})")
