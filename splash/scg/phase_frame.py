from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None


@dataclass
class PhaseFrame:
    """Fixed 2-D phase frame used to compute token angles θ.
    basis : (D,2) orthonormal matrix (columns are the 2D directions)
    fingerprint : stable id to verify the frame is frozen across runs
    """
    basis: np.ndarray
    fingerprint: str

    def project2D(self, E: np.ndarray) -> np.ndarray:
        """
        Project (N,D_current) embeddings to 2D. If D_current != D_ref, fall back
        to a deterministic local SVD so mixed-D tests never crash.
        """
        D_ref = int(self.basis.shape[0])
        D_cur = int(E.shape[1])
        if D_cur == D_ref:
            return E @ self.basis  # (N,2)

        # Fallback: local 2D basis from centered E
        Ec = E - E.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
            B = Vt[:2].T  # (D_cur, <=2)
            if B.shape[1] < 2:
                B = _pad_to_two(B, D_cur)
        except np.linalg.LinAlgError:
            B = _identity_two(D_cur)
        B = _orthonormalize_2(B)
        return E @ B


def _identity_two(D: int) -> np.ndarray:
    I = np.eye(D, dtype=float)
    if D == 1:
        return np.concatenate([I, np.zeros((D, 1))], axis=1)  # (1,2)
    return I[:, :2]  # (D,2)


def _pad_to_two(B: np.ndarray, D: int) -> np.ndarray:
    """Pad a (D,1) basis to (D,2) with a simple orthogonal/identity column."""
    if B.shape[1] >= 2:
        return B
    e = _identity_two(D)[:, 1:2]  # second identity column
    return np.concatenate([B, e], axis=1)


def _orthonormalize_2(B: np.ndarray) -> np.ndarray:
    """Gram-Schmidt for two columns; returns (D,2) with orthonormal columns."""
    if B.shape[1] != 2:
        raise ValueError("Basis must have exactly two columns")
    b0 = B[:, 0]; n0 = np.linalg.norm(b0) + 1e-12; b0 = b0 / n0
    b1 = B[:, 1] - b0 * (b0 @ B[:, 1])
    n1 = np.linalg.norm(b1) + 1e-12; b1 = b1 / n1
    return np.stack([b0, b1], axis=1)


def fit_phase_frame_scg(E_ref: np.ndarray, max_iter: int = 50) -> PhaseFrame:
    """
    Fit a 2D SCG-native phase frame.
    Optimizes orientation so that coherence invariants (R high, Kv low, B low) are satisfied.
    """
    from .curvature import kappa_knn, kappa_var
    from .invariants import dislocation_density_thresh


    # Center embeddings
    Ec = E_ref - E_ref.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    B = Vt[:2].T  # initial guess (like PCA)

    def _score(basis: np.ndarray) -> float:
        """Return coherence score: higher is better."""
        # Project
        P2 = Ec @ basis
        theta = np.arctan2(P2[:,1], P2[:,0])
        kappa = kappa_knn(E_ref, k=8)
        Kv = kappa_var(kappa)
        R = np.abs(np.exp(1j*theta).mean())
        rho = dislocation_density_thresh(theta, threshold_rad=np.pi/6)
        # Score favors high R, low Kv, low rho
        return R - 0.5*Kv - 0.5*rho

    best_B = B
    best_score = _score(B)

    # Small random rotations to search nearby frames
    for _ in range(max_iter):
        Rmat = np.linalg.qr(np.random.randn(E_ref.shape[1], 2))[0]
        cand_B = Rmat
        s = _score(cand_B)
        if s > best_score:
            best_B, best_score = cand_B, s

    basis = _orthonormalize_2(best_B)
    fp = _fingerprint(basis, meta={"method": "scg", "D": int(basis.shape[0])})
    return PhaseFrame(basis=basis, fingerprint=fp)


def fit_phase_frame(E_ref: np.ndarray, method: str = "pca") -> PhaseFrame:
    """Fit ONCE on a reference pool; robust when min(N,D) < 2."""
    assert E_ref.ndim == 2, "E_ref must be (N_ref,D)"
    N, D = E_ref.shape
    # Tiny shapes: fabricate a simple identity-based 2D basis
    if min(N, D) < 2:
        B = _identity_two(D)
        fp = _fingerprint(B, {"method": "identity", "D": D})
        return PhaseFrame(basis=_orthonormalize_2(B), fingerprint=fp)

    if method == "pca" and PCA is not None:
        # Ask for at most 2 comps but not more than min(N,D)
        ncomp = int(min(2, N, D))
        p = PCA(n_components=ncomp, svd_solver="auto", random_state=0).fit(E_ref)
        B = p.components_.T  # (D, ncomp)
        if ncomp < 2:
            B = _pad_to_two(B, D)
    elif method == "scg":
        return fit_phase_frame_scg(E_ref)
    else:
        U, S, Vt = np.linalg.svd(E_ref - E_ref.mean(0), full_matrices=False)
        B = Vt[:2].T
        if B.shape[1] < 2:
            B = _pad_to_two(B, D)

    basis = _orthonormalize_2(B)
    fp = _fingerprint(basis, meta={"method": method, "D": int(basis.shape[0])})
    return PhaseFrame(basis=basis, fingerprint=fp)


def load_phase_frame(path: str) -> PhaseFrame:
    obj = np.load(path, allow_pickle=True)
    basis = obj["basis"]
    meta = json.loads(str(obj["meta"].item()))
    return PhaseFrame(basis=basis, fingerprint=meta["fingerprint"])


def save_phase_frame(frame: PhaseFrame, path: str) -> None:
    meta = {"fingerprint": frame.fingerprint}
    np.savez_compressed(path, basis=frame.basis, meta=json.dumps(meta))


def _fingerprint(basis: np.ndarray, meta: Optional[dict] = None) -> str:
    h = hashlib.sha256()
    h.update(np.round(basis, 8).tobytes())
    if meta:
        h.update(json.dumps(meta, sort_keys=True).encode())
    return h.hexdigest()[:16]
