from __future__ import annotations
from typing import Dict, Tuple, Any, Union, List, Optional, Literal
import numpy as np

from splash.types import Measures, EvalKnobs
from splash.scg.phase_frame import PhaseFrame, fit_phase_frame
from splash.scg.curvature import kappa_knn, kappa_var
from splash.scg.projection import amplitude, scg3_from_A_theta_kappa
from splash.scg.invariants import coherence_entropy, kuramoto_R, barrier, dislocation_density_thresh
from splash.scg.dynamics import kuramoto_settle
from splash.types import Measures, CoherenceBands, BandLabel  # ⬅ typed imports


# per-D PhaseFrame cache
_PHASE_FRAMES: dict[int, PhaseFrame] = {}

def set_phase_frame(frame: PhaseFrame) -> None:
    _PHASE_FRAMES[int(frame.basis.shape[0])] = frame

def _ensure_phase_frame(E: np.ndarray) -> PhaseFrame:
    D = int(E.shape[1])
    frame = _PHASE_FRAMES.get(D)
    if frame is None:
        frame = fit_phase_frame(E, method="pca")
        _PHASE_FRAMES[D] = frame
    return frame

def _resolve_k_neighbors(N: int, ek: EvalKnobs | None, k_neighbors: int | None) -> int:
    if k_neighbors is not None:
        return int(max(2, min(k_neighbors, max(2, N - 1))))
    if ek and ek.k_neighbors is not None:
        return int(max(2, min(ek.k_neighbors, max(2, N - 1))))
    tgt = (ek.target_degree if ek else 12.0)
    return int(max(2, min(round(tgt), max(2, N - 1))))

def _resolve_settle_steps(ek: EvalKnobs | None, settle_steps: int | None) -> int | None:
    if settle_steps is not None:
        return int(settle_steps)
    return int(ek.settle_steps) if ek else 16

def one_tick_measures(
    E: np.ndarray,
    *,
    eval_knobs: EvalKnobs | None = None,
    settle_steps: int | None = None,
    k_neighbors: int | None = None,
    barrier_caps: Dict[str, Tuple[float, float]] | None = None,
    return_energy: bool = False,
    phase_whiten: bool = True,
    rho_threshold_rad: float = np.pi / 2,
    settle_gain: float | None = None,
    **_legacy_ignored: Any,
) -> Union[Measures, Tuple[Measures, float]]:
    """
    Compute SCG measures for a single layer/window.
    If return_energy=True, returns (Measures, energy) where energy = sum(A^2).
    """
    assert E.ndim == 2 and E.shape[0] >= 1, "E must be (N,D) with N>=1"
    frame = _ensure_phase_frame(E)

    steps = _resolve_settle_steps(eval_knobs, settle_steps)
    k_eff = _resolve_k_neighbors(E.shape[0], eval_knobs, k_neighbors)

    # primitives
    A = amplitude(E)                          # (N,)
    energy = float(np.sum(A * A))

    P2 = frame.project2D(E)                   # (N,2)
    if phase_whiten:
        s = P2.std(axis=0, keepdims=True) + 1e-8
        P2 = (P2 - P2.mean(axis=0, keepdims=True)) / s

    theta_pre = np.arctan2(P2[:, 1], P2[:, 0])  # PRE-settle angles (for ρφ)
    # settle with a smaller default gain to avoid R→1 immediately
    K = 0.2 if settle_gain is None else float(settle_gain)
    theta = theta_pre if (steps is None or steps <= 0) else kuramoto_settle(theta_pre, K=K, steps=int(steps))

    kappa = kappa_knn(E, k=k_eff)              # (N,)

    # aggregates + invariants
    agg = scg3_from_A_theta_kappa(A, theta, kappa)  # {x,y,z,r,w}
    H   = coherence_entropy(agg["w"])
    R   = kuramoto_R(theta, weights=agg["w"])

    # ρφ on PRE-settle angles with a configurable threshold (reveals seams)
    rho = dislocation_density_thresh(theta_pre, threshold_rad=rho_threshold_rad)
    kv  = kappa_var(kappa)

    B = 0.0
    if barrier_caps:
        B = barrier(vals={"entropy": H, "r": agg["r"], "kappa_var": kv}, caps=barrier_caps)

    m = Measures(
        alignment_score=R,
        alignment_xy=agg["r"],
        kappa_var=kv,
        rho_phi=rho,
        entropy=H,
        barrier=B,
        energy_flux=0.0,  # curve builder computes φ_E from per-layer energies
        x=agg["x"], y=agg["y"], z=agg["z"],
        # legacy aliases
        tension=kv,
        asymmetry=rho,
        bend_spread=0.0,
        ledger=0.0,
        memory_mean=0.0,
        shed_rate=0.0,
        shed_total=0.0,
        effective_degree=float(min(k_eff, max(0, E.shape[0] - 1))),
        cap_fraction=0.0,
    )
    return (m, energy) if return_energy else m

# ---------------------------------------------------------------------------
# SCION compatibility shim
# ---------------------------------------------------------------------------
# splash/geometry.py  (append-only helper)
def auto_knobs_to_scion(eval_knobs: Any):
    """Deprecated: Splash runs without SCION. Returns eval_knobs unchanged."""
    return eval_knobs



# Optional alias if you want a clearer name internally
auto_knobs_to_engine = auto_knobs_to_scion


def select_layer_hidden(
    hidden_layers: List[np.ndarray] | np.ndarray,
    *,
    mode: Literal["last", "mean_last_4", "custom"] = "last",
    custom_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Collapse per-layer hidden states into a single (B, T, d) tensor.
    Accepts either a list[ndarray] or already-stacked ndarray (L,B,T,d).
    """
    if isinstance(hidden_layers, np.ndarray):
        # assume (L, B, T, d)
        LBTD = hidden_layers
        assert LBTD.ndim == 4, "stacked hidden must be (L,B,T,d)"
        layers = [LBTD[i] for i in range(LBTD.shape[0])]
    else:
        layers = hidden_layers
    assert len(layers) >= 1, "need at least one layer"
    if mode == "last":
        return layers[-1]
    if mode == "mean_last_4":
        k = min(4, len(layers))
        return np.mean(layers[-k:], axis=0)
    if mode == "custom":
        assert custom_weights is not None, "custom mode requires custom_weights (L,)"
        w = np.asarray(custom_weights, dtype=np.float64)
        w = w / (np.sum(w) + 1e-12)
        out = np.zeros_like(layers[0], dtype=np.float64)
        for ai, li in zip(w, layers):
            out += ai * li
        return out
    raise ValueError(f"unknown mode: {mode}")

def _row_norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n

def to_places_from_hidden(
    hidden_bt: np.ndarray,
    *,
    distance: Literal["cosine", "l2"] = "cosine",
    max_tokens: Optional[int] = None
) -> List[np.ndarray]:
    """
    Convert a (B,T,d) hidden tensor into per-sequence arrays of places X (T,d).
    For cosine distance, we L2-normalize the vectors; SCION then derives geometry.
    """
    assert hidden_bt.ndim == 3, "hidden must be (B,T,d)"
    B, T, d = hidden_bt.shape
    if max_tokens is not None and T > max_tokens:
        hidden_bt = hidden_bt[:, :max_tokens, :]
        T = max_tokens
    Xs: List[np.ndarray] = []
    if distance == "cosine":
        Xs = [ _row_norm(hidden_bt[b]) for b in range(B) ]
    elif distance == "l2":
        # center for stability
        Xs = [ hidden_bt[b] - np.mean(hidden_bt[b], axis=0, keepdims=True) for b in range(B) ]
    else:
        raise ValueError(f"unknown distance: {distance}")
    return Xs



def classify_measures(m: Measures, bands: CoherenceBands) -> BandLabel:
    """
    SCG-first classification using typed Measures and CoherenceBands.
    - alignment_score drives pass/warn
    - tension (≡ kappa_var) and asymmetry (≡ rho_phi) must be within limits
    - a small 'near' band catches close calls around pass thresholds
    """
    reasons: List[str] = []

    # PASS: alignment meets pass AND both structural constraints are ok
    if (m.alignment_score >= bands.align_pass and
        m.tension <= bands.tension_max_ok and
        abs(m.asymmetry) <= bands.asym_max_ok):
        return BandLabel("pass")

    # NEAR: within near_margin of pass AND at most one mild violation
    near = False
    if m.alignment_score >= bands.align_pass - bands.near_margin:
        near = True
        if m.tension > bands.tension_max_ok:
            reasons.append(f"tension={m.tension:.3f} > {bands.tension_max_ok:.3f}")
        if abs(m.asymmetry) > bands.asym_max_ok:
            reasons.append(f"|asym|={abs(m.asymmetry):.3f} > {bands.asym_max_ok:.3f}")
        if len(reasons) <= 1:
            return BandLabel("near", reasons=reasons or ["close to threshold"])

    # WARN: alignment above warn, or single constraint breach
    warn_reasons: List[str] = []
    if m.alignment_score >= bands.align_warn:
        # alignment okay-ish, but constraints may be off
        if m.tension > bands.tension_max_ok:
            warn_reasons.append(f"tension={m.tension:.3f} > {bands.tension_max_ok:.3f}")
        if abs(m.asymmetry) > bands.asym_max_ok:
            warn_reasons.append(f"|asym|={abs(m.asymmetry):.3f} > {bands.asym_max_ok:.3f}")
        if warn_reasons:
            return BandLabel("warn", reasons=warn_reasons)

    # FAIL: alignment below warn OR multiple strong breaches
    fail_reasons: List[str] = []
    if m.alignment_score < bands.align_warn:
        fail_reasons.append(f"alignment={m.alignment_score:.3f} < warn {bands.align_warn:.3f}")
    if m.tension > bands.tension_max_ok and abs(m.asymmetry) > bands.asym_max_ok:
        fail_reasons.append("both tension and asymmetry out of bounds")
    if not fail_reasons and not warn_reasons and not near:
        fail_reasons.append("outside bands")

    return BandLabel("fail", reasons=fail_reasons)
