from __future__ import annotations
import numpy as np
try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None

def kappa_knn(E: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Discrete Laplacian proxy: ||e_i - mean(e_neighbors)||.
    Robust for tiny N: if N<3, fallback to sequential curvature ||Δe||.
    """
    N, D = E.shape
    if N <= 1:
        return np.zeros(N, dtype=float)
    if N < 3:
        # sequential fallback
        d = np.linalg.norm(np.diff(E, axis=0), axis=1)
        # pad ends so shape matches N
        return np.concatenate([[d[0]], d])

    k = int(max(1, min(k, N - 1)))
    if NearestNeighbors is None:
        # brute force
        d2 = _cdist2(E, E)
        idx = np.argsort(d2, axis=1)[:, 1 : k + 1]
    else:
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(E)
        idx = nn.kneighbors(E, return_distance=False)[:, 1:]

    nn_mean = E[idx].mean(axis=1)                 # (N,D)
    kappa = np.linalg.norm(E - nn_mean, axis=1)   # (N,)
    return kappa.astype(float)

def kappa_var(kappa: np.ndarray) -> float:
    """
    Dimensionless curvature-variance in [0,1].
    Uses variance relative to mean-square scale; falls back to Δκ variance if flat.
    """
    x = np.asarray(kappa, float)
    n = x.size
    if n <= 1:
        return 0.0
    # basic variance & scale
    v = float(np.var(x))
    scale = float(np.mean(x * x)) + 1e-12
    if v < 1e-12 and n >= 3:
        v = float(np.var(np.diff(x)))  # fallback keeps tiny windows informative
    val = v / (v + scale)              # squashes to (0,1)
    return float(np.clip(val, 0.0, 1.0))

def _cdist2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    AA = np.sum(A * A, axis=1, keepdims=True)
    BB = np.sum(B * B, axis=1, keepdims=True).T
    return AA + BB - 2.0 * (A @ B.T)
