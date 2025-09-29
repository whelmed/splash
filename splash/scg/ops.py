from __future__ import annotations
import numpy as np

def apply_barrier(kappa: np.ndarray, strength: float = 1e-2) -> np.ndarray:
    """Gentle curvature smoothing; caller decides when to use."""
    if kappa.size == 0:
        return kappa
    mu = np.median(kappa)
    return kappa - strength * (kappa - mu)

def apply_flux_budget(phi_layer: float, budget: float = 0.02) -> float:
    if phi_layer > budget:
        return budget
    if phi_layer < -budget:
        return -budget
    return phi_layer

def chi_anchor_pass(theta: np.ndarray, gain: float = 0.15) -> np.ndarray:
    """Light phase tightening when entering a 'decision token' window."""
    th = np.array(theta, copy=True)
    mean_angle = np.angle(np.exp(1j * th).mean())
    th += gain * np.sin(mean_angle - th)
    return th
