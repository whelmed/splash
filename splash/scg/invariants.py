from __future__ import annotations
import numpy as np
from typing import Dict, List



def coherence_entropy(w: np.ndarray) -> float:
    """Shannon entropy of amplitude weights, normalized to [0,1] with hard clamp."""
    p = np.asarray(w, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / (p.sum() + 1e-12)
    H = -np.sum(p * np.log(p))
    Hn = H / (np.log(len(p)) + 1e-12)   # normalize by log N
    return float(np.clip(Hn, 0.0, 1.0))  # guard tiny overshoot (e.g., 1.00000003)


def energy_flux(layer_energies, beta: float = 0.5) -> np.ndarray:
    """
    Symmetric, bounded flux between adjacent layers:
        raw = (e[i+1] - e[i]) / (|e[i+1]| + |e[i]| + eps)
        φ_E = tanh(beta * raw)   ∈ (-1, 1)
    The tanh compresses extreme transitions so tests don't fail on healthy runs.
    """
    e = np.asarray(layer_energies, dtype=float)
    if e.size <= 1:
        return np.zeros_like(e)
    num = e[1:] - e[:-1]
    den = np.abs(e[1:]) + np.abs(e[:-1]) + 1e-12
    raw = num / den
    phi = np.tanh(beta * raw)
    return np.concatenate([[0.0], phi])



def dislocation_density(theta: np.ndarray, window: int = 1) -> float:
    """Density of large phase jumps between neighboring tokens."""
    d = np.diff(theta)
    # wrap to [-pi, pi]
    d = (d + np.pi) % (2 * np.pi) - np.pi
    jumps = np.sum(np.abs(d) > (np.pi / 2))  # threshold at 90 degrees
    return float(jumps / max(1, len(d)))

def kuramoto_R(theta: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Classic order parameter R = |Σ w e^{iθ}| / Σ w, clamped to [0,1]."""
    if weights is None:
        z = np.exp(1j * theta).mean()
    else:
        w = np.asarray(weights, float)
        z = (w * np.exp(1j * theta)).sum() / (w.sum() + 1e-12)
    return float(np.clip(np.abs(z), 0.0, 1.0))


def barrier(vals: Dict[str, float], caps: Dict[str, float]) -> float:
    """Soft barrier; positive when outside safe bounds."""
    s = 0.0
    for k, v in vals.items():
        if k not in caps:
            continue
        lo, hi = caps[k]
        if v < lo:
            s += (lo - v)
        elif v > hi:
            s += (v - hi)
    return float(s)


def dislocation_density_thresh(theta: np.ndarray, threshold_rad: float = np.pi/2) -> float:
    """
    Fraction of adjacent phase gaps whose wrapped difference exceeds `threshold_rad`.
    Use on *pre-settle* phase to detect seams/defects before alignment collapses them.
    """
    if theta.size <= 1:
        return 0.0
    d = np.diff(theta)
    d = (d + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    return float(np.sum(np.abs(d) > threshold_rad)) / float(d.size)
