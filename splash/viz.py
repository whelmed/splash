from __future__ import annotations
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

from .types import MapResult, LayerCurve

def plot_coherence_map(result: MapResult):
    """
    Heatmap over tiles (x-axis ~ token positions via tile centers; y-axis N)
    Uses alignment_score as primary; overlays tension as contours if available.
    """
    Ns = sorted(result.Ns)
    # build matrix: rows=N, cols=tile index (not aligned across Ns perfectly; we interpolate per N)
    mats = []
    cols = 0
    for N in Ns:
        vals = [tm.measures.alignment_score for tm in result.tiles.get(N, [])]
        mats.append(np.array(vals, dtype=np.float64)[None, :])
        cols = max(cols, len(vals))
    H = np.zeros((len(Ns), cols), dtype=np.float64)
    for i, row in enumerate(mats):
        H[i, :row.shape[1]] = row

    plt.figure()
    plt.imshow(H, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(Ns)), [f"N={n}" for n in Ns])
    plt.xlabel("tile index")
    plt.title("Alignment score heatmap (rows = N)")
    plt.colorbar(label="alignment_score")
    plt.show()

def plot_layer_curves(curves: LayerCurve):
    """
    Line plots of metric vs layer index.
    """
    x = np.arange(len(next(iter(curves.per_layer.values()))))
    for m, ys in curves.per_layer.items():
        plt.figure()
        plt.plot(x, ys)
        plt.xlabel("layer")
        plt.ylabel(m)
        plt.title(f"{m} vs layer")
        plt.show()

def plot_embedding_carpet(X: "np.ndarray", tokens: List[str]):
    """
    2D projection of SWG 'places' with token annotations.
    For v1, we simply use first two dimensions of X; callers can pass projected coordinates.
    """
    if X.shape[1] < 2:
        raise ValueError("X must have at least 2 dims for a 2D carpet")
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=20)
    if tokens:
        for i, t in enumerate(tokens[:len(X)]):
            plt.text(X[i, 0], X[i, 1], t, fontsize=8)
    plt.title("Embedding carpet (first 2 dims)")
    plt.show()
