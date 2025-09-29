from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from splash.control.types import InvariantFrame, Decision

@dataclass
class VerifyResult:
    ok: bool
    reason: str

class Verifier:
    """
    Post-action check: require improved alignment or reduced tension, with band not worse.
    """
    def __init__(self, min_delta_align: float = 0.01, max_delta_tension: float = -0.01):
        self.min_da = min_delta_align
        self.max_dt = max_delta_tension

    def verify(self, before: InvariantFrame, after: Optional[InvariantFrame], decision: Decision) -> VerifyResult:
        if after is None:
            # advisory or intent-only actions are OK by definition
            return VerifyResult(ok=True, reason="advisory/intent-only")
        dR = after.measures.get("alignment_score",0.0) - before.measures.get("alignment_score",0.0)
        dT = after.measures.get("tension",0.0) - before.measures.get("tension",0.0)
        if after.label.label in ("fail",) and before.label.label not in ("fail",):
            return VerifyResult(ok=False, reason=f"band worsened to {after.label.label}")
        if (dR >= self.min_da) or (dT <= self.max_dt):
            return VerifyResult(ok=True, reason=f"improved (ΔR={dR:.3f}, ΔT={dT:.3f})")
        return VerifyResult(ok=False, reason=f"no improvement (ΔR={dR:.3f}, ΔT={dT:.3f})")
