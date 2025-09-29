from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict

from splash.control.types import InvariantFrame, Decision, PolicyConfig

@dataclass
class PolicyPack:
    """Preset policy packs."""
    @staticmethod
    def conservative() -> PolicyConfig:
        return PolicyConfig(align_pass=0.80, align_warn=0.70, temp_max=0.7, top_p_max=0.9)
    @staticmethod
    def balanced() -> PolicyConfig:
        return PolicyConfig()
    @staticmethod
    def creative() -> PolicyConfig:
        return PolicyConfig(align_pass=0.72, align_warn=0.60, temp_max=1.0, top_p_max=0.98)

class RulePolicy:
    """
    Deterministic rules mapping invariant triggers → actions.
    Severity ordering: fail > warn > info > none.
    """
    def __init__(self, cfg: PolicyConfig | None = None, policy_id: str = "balanced"):
        self.cfg = cfg or PolicyPack.balanced()
        self.policy_id = policy_id

    def decide(self, frame: InvariantFrame) -> Decision:
        a = frame.measures.get
        # Simple triggers from Splash bands + SCION measures
        align = a("alignment_score", 0.0)
        tens  = a("tension", 1.0)
        asym  = abs(a("asymmetry", 0.0))
        # Fail (hard guard)
        if align < self.cfg.align_warn:
            return Decision(severity="fail", op="temp_clamp",
                            args={"temperature": max(self.cfg.temp_min, 0.2)},
                            reason=f"alignment {align:.2f} < warn {self.cfg.align_warn:.2f}",
                            policy_id=self.policy_id)
        # Warn: high tension or asymmetry
        if tens > self.cfg.tension_max_ok or asym > self.cfg.asym_max_ok:
            return Decision(severity="warn", op="top_p",
                            args={"top_p": min(self.cfg.top_p_max, 0.9)},
                            reason=f"tension/asymmetry exceed limits",
                            policy_id=self.policy_id)
        # Near: below pass but above warn → align nudge
        if align < self.cfg.align_pass:
            return Decision(severity="info", op="align",
                            args={"preview": True},  # SCION align preview (no mutations)
                            reason=f"alignment near threshold {align:.2f} < pass {self.cfg.align_pass:.2f}",
                            policy_id=self.policy_id)
        # Pass
        return Decision(severity="none", op="none", args={}, reason="stable", policy_id=self.policy_id)
