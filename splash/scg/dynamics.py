from __future__ import annotations
import numpy as np


def kuramoto_settle(theta: np.ndarray, K: float = 0.5, steps: int = 16) -> np.ndarray:
    """Small, deterministic settle on phases (no randomness, no refit)."""
    th = np.array(theta, dtype=float, copy=True)
    for _ in range(max(1, int(steps))):
        mean_angle = np.angle(np.exp(1j * th).mean())
        th += K * np.sin(mean_angle - th)
    return th
