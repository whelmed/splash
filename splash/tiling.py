from __future__ import annotations
from typing import Sequence, Dict, List, Tuple
import numpy as np

from scion.api import Measures
from .types import EvalKnobs, CoherenceBands, TileSpec, TileMeasures, MapResult
from .geometry import one_tick_measures, classify_measures

def iter_tiles(T: int, N: int, stride: int) -> List[TileSpec]:
    """Yield TileSpec windows covering length T with given N and stride."""
    assert N > 0 and stride > 0
    tiles: List[TileSpec] = []
    i = 0
    while i + N <= T:
        tiles.append(TileSpec(start=i, end=i + N, N=N, layer=None))
        i += stride
    # ensure tail coverage if last window missed the end (optional)
    if tiles and tiles[-1].end < T and (T - N) >= 0:
        tiles.append(TileSpec(start=T - N, end=T, N=N, layer=None))
    elif not tiles and T >= N:
        tiles.append(TileSpec(start=0, end=N, N=N, layer=None))
    return tiles

def _global_means_from_tiles(tiles: List[TileMeasures]) -> Dict[str, float]:
    if not tiles:
        return {}
    keys = tiles[0].measures.__dict__.keys()
    out: Dict[str, float] = {}
    for k in keys:
        vals = [getattr(t.measures, k) for t in tiles if hasattr(t.measures, k)]
        if vals and isinstance(vals[0], (int, float)):
            out[k] = float(np.mean(vals))
    return out

def coherence_map_for_sequence(
    X: np.ndarray,                    # (T, d)
    *,
    eval_knobs: EvalKnobs,
    bands: CoherenceBands
) -> MapResult:
    """
    Compute per-tile Measures across Ns for a single sequence of places.
    - uses World(Knobs) internally
    - runs one-tick per tile (fast path)
    - returns tiles + global metric means
    """
    T = X.shape[0]
    tiles_by_N: Dict[int, List[TileMeasures]] = {}
    for N in eval_knobs.Ns:
        stride = max(1, int(round(N * eval_knobs.stride_fraction)))
        specs = iter_tiles(T, N, stride)
        tlist: List[TileMeasures] = []
        for spec in specs:
            Xi = X[spec.start:spec.end]
            m = one_tick_measures(Xi, eval_knobs=eval_knobs)


            label = classify_measures(m, bands)
            tlist.append(TileMeasures(spec=spec, measures=m, label=label))
        tiles_by_N[N] = tlist

    # global means over all tiles (pool across Ns)
    pooled = [tm for lst in tiles_by_N.values() for tm in lst]
    gmeans = _global_means_from_tiles(pooled)
    meta = {"T": int(T), "Ns": list(eval_knobs.Ns), "stride_fraction": float(eval_knobs.stride_fraction)}
    return MapResult(Ns=list(eval_knobs.Ns), tiles=tiles_by_N, global_means=gmeans, meta=meta)

def coherence_maps_for_batch(
    Xs: List[np.ndarray],
    *,
    eval_knobs: EvalKnobs,
    bands: CoherenceBands
) -> List[MapResult]:
    """Batch wrapper over coherence_map_for_sequence."""
    return [coherence_map_for_sequence(X, eval_knobs=eval_knobs, bands=bands) for X in Xs]
