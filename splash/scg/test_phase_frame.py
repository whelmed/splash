from __future__ import annotations
import numpy as np
import pytest

from splash.scg.phase_frame import fit_phase_frame, PhaseFrame

def test_phase_frame_handles_tiny_shapes():
    # (N,D) where min(N,D) = 1 would crash PCA(2) without guards
    E1 = np.linspace(0.0, 1.0, 5).reshape(-1, 1)  # (5,1)
    frame: PhaseFrame = fit_phase_frame(E1, method="pca")
    P2 = frame.project2D(E1)
    assert P2.shape == (5, 2)
    assert np.isfinite(P2).all()
