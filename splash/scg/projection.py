from __future__ import annotations
import numpy as np
from typing import Dict


def scg3_from_A_theta_kappa(A: np.ndarray, theta: np.ndarray, kappa: np.ndarray) -> Dict[str, float]:
    """SCG-3 aggregation with amplitude weights. Numerically clamps r∈[0,1]."""
    w = np.asarray(A, dtype=float)
    s = float(w.sum())
    if s <= 1e-12:
        # degenerate: uniform weights
        w = np.full_like(w, 1.0 / max(1, w.size), dtype=float)
    else:
        w = w / s

    # weighted phase plane
    x = float(np.sum(w * np.cos(theta)))
    y = float(np.sum(w * np.sin(theta)))
    # curvature channel
    z = float(np.sum(w * np.asarray(kappa, dtype=float)))

    r = float(np.hypot(x, y))
    # guard tiny overshoot (e.g., 1.00000003) and underflow
    r = float(np.clip(r, 0.0, 1.0))

    return {"x": x, "y": y, "z": z, "r": r, "w": w}


def amplitude(E: np.ndarray) -> np.ndarray:
    return np.linalg.norm(E, axis=1)  # (N,)
