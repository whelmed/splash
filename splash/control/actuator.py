from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

from splash.types import EvalKnobs, CoherenceBands
from splash.geometry import one_tick_measures, classify_measures
from splash.control.types import InvariantFrame, Decision

@dataclass
class ActuatorResult:
    after: Optional[InvariantFrame]
    applied: bool
    notes: list

class Actuator:
    """
    v1 actuator supports two modes:
      (A) advisory/logit processors when logits are supplied by caller
      (B) SCION "align preview" (predictive) for places (no model mutation)
    """
    def __init__(self, eval_knobs: EvalKnobs, bands: CoherenceBands, *, adapter=None):
        self.eval_knobs = eval_knobs
        self.bands = bands
        self.adapter = adapter


    def enable_dynamic(self, dyn_manager):
        if hasattr(self.adapter, "set_dynamic_manager"):
            self.adapter.set_dynamic_manager(dyn_manager)
            return True
        return False

    # --- Logit processors (optional) ---
    @staticmethod
    def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
        if temperature <= 0: return logits
        return logits / float(temperature)

    @staticmethod
    def apply_top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
        # leave exact sampling to upstream decoder; we just tag intent here
        return logits  # placeholder: controller can pass this intent to the decoder

    @staticmethod
    def apply_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
        return logits  # placeholder intent

    @staticmethod
    def apply_repetition_penalty(logits: np.ndarray, token_ids: np.ndarray, rep_penalty: float) -> np.ndarray:
        return logits  # placeholder intent

    # --- SCION “align preview” over places (purely predictive) ---
    def scion_align_preview(self, X_window: np.ndarray, t: int, span: Tuple[int,int], meta: Dict[str,Any] | None=None) -> InvariantFrame:
        # We simply run one-tick measures again (preview is a no-op in v1).
        # In v2 we could simulate a slightly stronger "strength_base" by cloning knobs.
        m = one_tick_measures(X_window, eval_knobs=self.eval_knobs)
        label = classify_measures(m, self.bands)
        measures = {
            "alignment_score": m.alignment_score, "tension": m.tension,
            "bend_spread": m.bend_spread, "ledger": m.ledger,
            "memory_mean": m.memory_mean, "shed_rate": m.shed_rate, "shed_total": m.shed_total,
            "asymmetry": m.asymmetry, "effective_degree": m.effective_degree, "cap_fraction": m.cap_fraction,
        }
        return InvariantFrame(t=t, window=span, measures=measures, label=label, meta=meta or {"preview": True})

    def apply(self,
              decision: Decision,
              *,
              t: int,
              X_window: np.ndarray,
              span: Tuple[int,int],
              logits: Optional[np.ndarray] = None,
              meta: Optional[Dict[str,Any]] = None) -> ActuatorResult:
        notes = []
        if decision.op == "none":
            return ActuatorResult(after=None, applied=False, notes=["no-op"])
        if decision.op in ("temp_clamp", "top_p", "top_k", "rep_penalty"):
            if logits is None:
                # advisory only: record intent (controller can pass this to sampler)
                notes.append(f"advisory:{decision.op} -> {decision.args}")
                return ActuatorResult(after=None, applied=False, notes=notes)
            # If logits provided, apply transformation in-place (temperature only in v1)
            if decision.op == "temp_clamp":
                self.apply_temperature(logits, float(decision.args.get("temperature", 0.7)))
                notes.append(f"applied temperature={decision.args.get('temperature')}")
                return ActuatorResult(after=None, applied=True, notes=notes)
            # top_p/top_k/rep_penalty remain as intents in v1
            notes.append(f"intent:{decision.op} -> {decision.args}")
            return ActuatorResult(after=None, applied=True, notes=notes)

        if decision.op in ("align", "rotate", "dilate", "couple_memory", "transport"):
            # v1: preview only (no hidden-state mutation). We still return predicted frame.
            after = self.scion_align_preview(X_window, t=t, span=span, meta=meta)
            notes.append("preview-only; no state mutation")
            return ActuatorResult(after=after, applied=False, notes=notes)

        notes.append(f"unknown op {decision.op}")
        return ActuatorResult(after=None, applied=False, notes=notes)
