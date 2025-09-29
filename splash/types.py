from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol, Sequence, Dict, Any, Tuple, List
import numpy as np

Scalar = float

# --- Core evaluation knobs (orthogonal to SCION's runtime knobs) ---
@dataclass
class EvalKnobs:
    """
    Evaluation-time controls. Orthogonal to scion.api.Knobs, which governs
    the SCION dynamics. These set how we SAMPLE, BUILD GEOMETRY, and REPORT.
    """
    Ns: Sequence[int] = (4, 8, 16, 32)
    stride_fraction: float = 0.5          # tile stride = max(1, int(N * stride_fraction))
    distance: Literal["cosine", "l2"] = "cosine"
    normalize: Literal["row", "sym", "ds"] = "row"
    sym_blend: float = 0.0
    target_degree: float = 12.0           # passed to auto-reach / weight scaling
    degree_tolerance: float = 1.0
    k_neighbors: Optional[int] = None     # if set, use sparse/gather path
    mutual_knn: bool = True
    ensure_connected: bool = True
    layer_combine: Literal["last", "mean_last_4", "custom"] = "last"
    record_tiles: bool = True             # keep per-tile measures
    max_tokens: Optional[int] = None      # truncate long sequences safely
    settle_steps: int = 1                 # small settle so R "feels" the geometry; default 1 keeps tests stable
    # in EvalKnobs (add these fields; defaults preserve current behavior)
    diag_strength_cap: Optional[float] = None   # if set, override SCION Knobs.strength_cap for measurement
    diag_strength_base: Optional[float] = None  # optional: override strength_base as well
    scion_preset: Literal["robust","balanced","fast","theorem"] = "robust"  # choose engine preset for measurement



# @dataclass
# class CoherenceBands:
#     """
#     Thresholds to classify 'coherence categories' from Measures.
#     Designed to be model/task agnostic but configurable in notebooks.
#     """
#     align_pass: float = 0.72              # alignment_score >= pass
#     align_warn: float = 0.65              # warn <= alignment_score < pass
#     tension_max_ok: float = 0.35          # tension <= ok
#     asym_max_ok: float = 0.20             # |asymmetry| <= ok

# @dataclass
# class BandLabel:
#     """Result of mapping measures to a category label."""
#     label: Literal["pass", "near", "warn", "fail"]
#     reasons: List[str] = field(default_factory=list)

# --- Embedding sources and payloads ---
@dataclass
class SequenceBatch:
    """Tokenized inputs and (optionally) decoded text for reference."""
    input_ids: np.ndarray               # (B, T)
    attention_mask: Optional[np.ndarray] = None  # (B, T)
    texts: Optional[List[str]] = None

@dataclass
class HiddenStates:
    """
    Per-layer hidden states captured at the end of a forward pass.
    shapes[layer] == (B, T, d_model)
    """
    layers: List[np.ndarray]
    final: np.ndarray                   # (B, T, d_model)
    layer_names: Optional[List[str]] = None

@dataclass
class AttentionMaps:
    """
    Optional attention maps per layer (averaged or head-wise).
    If head-wise: (B, num_heads, T, T); if averaged: (B, T, T)
    """
    per_layer: Optional[List[np.ndarray]] = None
    averaged: Optional[List[np.ndarray]] = None

@dataclass
class ModelOutputs:
    """Container returned by adapters after one forward pass."""
    batch: SequenceBatch
    hidden: HiddenStates
    attention: Optional[AttentionMaps] = None
    token_embeddings: Optional[np.ndarray] = None    # (V, d_model) if available
    pos_embeddings: Optional[np.ndarray] = None      # (T, d_model) for used positions

# --- Geometry & evaluation results ---
@dataclass
class TileSpec:
    start: int
    end: int                   # exclusive; size == N
    N: int
    layer: Optional[int] = None

@dataclass
class TileMeasures:
    """One tile's SCG measures (from scion.api.Measures) + label."""
    spec: TileSpec
    measures: Any       # scion.api.Measures (alignment_score, tension, ...)
    label: BandLabel

@dataclass
class MapResult:
    """
    Global/regional results for one sequence at one layer/source.
    """
    Ns: Sequence[int]
    tiles: Dict[int, List[TileMeasures]]      # key: N -> list of tiles (in order)
    global_means: Dict[str, float]            # mean per metric across all tiles
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LayerCurve:
    """
    Layer-wise trajectory of selected measures (e.g., alignment_score, tension).
    """
    metric_names: List[str]
    per_layer: Dict[str, List[float]]         # metric -> [layers]
    layer_names: Optional[List[str]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvalRun:
    """Top-level result bundle suitable for JSON/Report I/O."""
    coherence_maps: Dict[str, MapResult]      # key like 'combined:last' or 'layer:12'
    layer_curves: Optional[LayerCurve] = None
    config: Dict[str, Any] = field(default_factory=dict)



# -------------------------------
# SCG-First Measures (typed)
# -------------------------------
@dataclass
class Measures:
    """
    One-tick measures using the SCG lexicon, with legacy aliases preserved.
    The SCG-native fields (alignment_xy, kappa_var, rho_phi, entropy, barrier)
    are computed by Splash; legacy fields map to them for compatibility.
    """
    # --- Alignment ---
    alignment_score: float                 # Kuramoto-style order parameter (0..1)
    alignment_xy: float = 0.0             # SCG XY coherence radius (0..1)

    # --- Curvature / phase structure ---
    kappa_var: float = 0.0                # curvature variance proxy (0..1-ish)
    rho_phi: float = 0.0                  # phase dislocation density (0..1)
    entropy: float = 0.0                  # normalized coherence entropy (0..1)
    barrier: float = 0.0                  # soft barrier score (0 in-bounds)

    # --- Energy / extras (optional; set by callers that track them) ---
    energy_flux: float = 0.0
    x: float = 0.0                        # SCG-3 coords (aggregated)
    y: float = 0.0
    z: float = 0.0

    # --- Legacy compatibility (SCION-era names) ---
    tension: float = 0.0                  # ≡ kappa_var
    asymmetry: float = 0.0                # ≡ rho_phi
    bend_spread: float = 0.0
    ledger: float = 0.0
    memory_mean: float = 0.0
    shed_rate: float = 0.0
    shed_total: float = 0.0
    effective_degree: float = 0.0
    cap_fraction: float = 0.0

    # convenience: dict-like access for notebooks/tests that used item-style
    def __getitem__(self, key: str) -> float:
        return getattr(self, key)

    def get(self, key: str, default: float = 0.0) -> float:
        return getattr(self, key, default)


# -------------------------------
# Coherence bands / labels
# -------------------------------
@dataclass
class CoherenceBands:
    """
    Thresholds to classify 'coherence categories' from Measures.
    Your original four remain the source of truth; SCG code may
    consult the added optional fields if present.
    """
    align_pass: float = 0.72               # alignment_score >= pass
    align_warn: float = 0.65               # warn <= alignment_score < pass
    tension_max_ok: float = 0.35           # tension <= ok  (≡ kappa_var)
    asym_max_ok: float = 0.20              # |asymmetry| <= ok (≡ rho_phi)

    # Optional extras (safe defaults)
    near_margin: float = 0.03              # "near" window around pass thresholds


@dataclass
class BandLabel:
    """Result of mapping measures to a category label."""
    label: Literal["pass", "near", "warn", "fail"]
    reasons: List[str] = field(default_factory=list)


# @dataclass
# class Measures:
#     """One-tick measures using the SCG lexicon."""
#     alignment_score: float        # group in-step score (0..1)
#     tension: float                # disagreement load
#     bend_spread: float            # variance of swing gaps
#     ledger: float                 # motion + tension (+ on-site if present)
#     memory_mean: float            # mean memory across units
#     shed_rate: float              # instantaneous shed proxy (wash)
#     shed_total: float             # cumulative shed proxy over the run
#     asymmetry: float              # directional kernel asymmetry
#     effective_degree: float       # avg active neighbors per unit
#     cap_fraction: float           # fraction with coupling at cap

#     # defining __getitem__ to allow dict-like access
#     def __getitem__(self, key: str) -> float:
#         return getattr(self, key)

#     def get(self, key: str, default: float = 0.0) -> float:
#         return getattr(self, key, default)
