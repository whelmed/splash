from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Literal, Tuple
import numpy as np

from splash.types import BandLabel

# Minimal “invariant frame” built on SCION Measures + Splash’s banding
@dataclass
class InvariantFrame:
    t: int                               # decoding step or frame index
    window: Tuple[int, int]              # [start, end) token indices used for this measurement
    measures: Dict[str, float]           # from scion.api.Measures (as dict-like)
    label: BandLabel                     # pass/near/warn/fail (splash.geometry.classify_measures)
    meta: Dict[str, Any] = field(default_factory=dict)

# Policy decision (what to do given invariants)
@dataclass
class Decision:
    severity: Literal["none", "info", "warn", "fail"]
    op: Literal["none", "rotate", "align", "dilate", "couple_memory", "transport",
                "temp_clamp", "top_p", "top_k", "rep_penalty"]
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    policy_id: str = "balanced"

# A realized action trace (before/after verification)
@dataclass
class ActionTrace:
    before: InvariantFrame
    decision: Decision
    after: Optional[InvariantFrame] = None
    ok: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

@dataclass
class Episode:
    frames: List[InvariantFrame] = field(default_factory=list)
    actions: List[ActionTrace] = field(default_factory=list)
    checkpoints: List[int] = field(default_factory=list)  # frame indices for roll-back points
    verdict: Literal["ok", "degraded", "aborted", "unknown"] = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyConfig:
    # invariant thresholds; these align with splash.types.CoherenceBands semantics
    align_pass: float = 0.75
    align_warn: float = 0.65
    tension_max_ok: float = 0.35
    asym_max_ok: float = 0.20
    # decoding controls (bounds)
    temp_min: float = 0.1
    temp_max: float = 0.9
    top_p_min: float = 0.7
    top_p_max: float = 0.95
    top_k_max: int = 100
    rep_pen_max: float = 1.5
    # hysteresis/cooldowns
    cool_steps: int = 3
