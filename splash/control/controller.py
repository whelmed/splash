from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from ..types import EvalKnobs, CoherenceBands
from ..geometry import to_places_from_hidden
from .types import InvariantFrame, Decision, ActionTrace, Episode
from .observer import measure_frame
from .policy import RulePolicy
from .actuator import Actuator
from .verifier import Verifier

class Controller:
    """
    v1 control-plane orchestrator.
    Typical usage:
        controller = Controller(eval_knobs, bands)
        for step in range(T):
            frame = controller.step(hidden_bt[:, :step+1, :], logits=None)  # teacher-forced or streaming
    """
    def __init__(self,
                 eval_knobs: EvalKnobs,
                 bands: CoherenceBands,
                 policy: RulePolicy | None = None,
                 verifier: Verifier | None = None):
        self.knobs = eval_knobs
        self.bands = bands
        self.policy = policy or RulePolicy(policy_id="balanced")
        self.verifier = verifier or Verifier()
        self.episode = Episode()

    def _prepare_window(self, hidden_bt: np.ndarray, window_size: Optional[int] = None) -> Tuple[np.ndarray, Tuple[int,int]]:
        """
        Convert hidden (B,T,d) → places (T,d) for first sequence. Slice last K as window.
        """
        assert hidden_bt.ndim == 3 and hidden_bt.shape[0] >= 1, "hidden must be (B,T,d)"
        Xs = to_places_from_hidden(hidden_bt[:1], distance=self.knobs.distance, max_tokens=self.knobs.max_tokens)
        X = Xs[0]  # (T, d)
        T = X.shape[0]
        K = int(window_size) if window_size else min(64, T)
        s = max(0, T - K); e = T
        return X[s:e], (s, e)

    def step(self, hidden_bt: np.ndarray, *, logits: Optional[np.ndarray] = None, window_size: Optional[int] = None) -> InvariantFrame:
        """
        One control step. Provide current hidden states (B,T,d). logits optional.
        Returns the *measured* frame (before any action).
        """
        Xw, span = self._prepare_window(hidden_bt, window_size=window_size)
        t = span[1] - 1  # last token index in the window as "time"
        # 1) Observe
        frame = measure_frame(t=t, X_window=Xw, eval_knobs=self.knobs, bands=self.bands, window_span=span)
        self.episode.frames.append(frame)
        # 2) Policy
        decision: Decision = self.policy.decide(frame)
        # 3) Actuate
        actuator = Actuator(self.knobs, self.bands)
        ar = actuator.apply(decision, t=t, X_window=Xw, span=span, logits=logits)
        # 4) Verify
        vr = self.verifier.verify(frame, ar.after, decision)
        self.episode.actions.append(ActionTrace(before=frame, decision=decision, after=ar.after, ok=vr.ok, notes=ar.notes+[vr.reason]))
        return frame

    def get_episode(self) -> Episode:
        return self.episode
