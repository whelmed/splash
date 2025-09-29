from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np

from splash.types import Measures, EvalKnobs
from splash.geometry import one_tick_measures
from splash.scg.invariants import energy_flux

@dataclass
class LayerCurves:
    alignment_score: np.ndarray
    alignment_xy:    np.ndarray
    kappa_var:       np.ndarray
    rho_phi:         np.ndarray
    entropy:         np.ndarray
    barrier:         np.ndarray
    energy_flux:     np.ndarray  # derived from per-layer energies

def curves_from_layers(
    layers: List[np.ndarray],
    *,
    eval_knobs: EvalKnobs,
) -> LayerCurves:
    ms: List[Measures] = []
    energies: List[float] = []
    for E in layers:
        m, e = one_tick_measures(E, eval_knobs=eval_knobs, return_energy=True)
        ms.append(m); energies.append(e)

    def arr(attr: str) -> np.ndarray:
        return np.array([getattr(m, attr) for m in ms], dtype=float)

    return LayerCurves(
        alignment_score = arr("alignment_score"),
        alignment_xy    = arr("alignment_xy"),
        kappa_var       = arr("kappa_var"),
        rho_phi         = arr("rho_phi"),
        entropy         = arr("entropy"),
        barrier         = arr("barrier"),
        energy_flux     = energy_flux(energies, beta=0.5),
    )

def tile_alignment_map(
    E: np.ndarray,
    *,
    eval_knobs: EvalKnobs,
    tile_N: int,
    stride_fraction: float = 0.5,
    alignment: Literal["score","xy"] = "score",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of size tile_N across tokens and compute alignment per tile.
    Returns (centers, values). Uses the same settle/k choices as curves.
    """
    stride = max(1, int(tile_N * stride_fraction))
    T = E.shape[0]
    centers, vals = [], []
    for start in range(0, max(1, T - tile_N + 1), stride):
        Xi = E[start : start + tile_N]
        if Xi.shape[0] < 2:  # skip degenerate tiles
            continue
        m = one_tick_measures(Xi, eval_knobs=eval_knobs)
        v = m.alignment_score if alignment == "score" else m.alignment_xy
        centers.append(start + Xi.shape[0] // 2)
        vals.append(v)
    return np.array(centers), np.array(vals, dtype=float)


def windowed_alignment_curve(
    layers: List[np.ndarray],
    *,
    eval_knobs: EvalKnobs,
    tile_N: int,
    stride_fraction: float = 0.5,
    alignment: Literal["score","xy"] = "score",
) -> np.ndarray:
    """
    For each layer, slide a tile of size N across tokens and average the chosen
    alignment metric over tiles. This makes tiles and 'curve' directly comparable.
    """
    vals = []
    for E in layers:
        centers, tiles = tile_alignment_map(
            E, eval_knobs=eval_knobs, tile_N=tile_N,
            stride_fraction=stride_fraction, alignment=alignment
        )
        v = float(tiles.mean()) if tiles.size else float("nan")
        vals.append(v)
    return np.asarray(vals, dtype=float)
