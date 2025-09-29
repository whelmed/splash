from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

from ..types import EvalKnobs, CoherenceBands, BandLabel
from ..geometry import one_tick_measures, classify_measures
from ..control.types import InvariantFrame

def measure_frame(
    *,
    t: int,
    X_window: np.ndarray,                 # (N, d) “places” for the current window
    eval_knobs: EvalKnobs,
    bands: CoherenceBands,
    meta: Dict[str, Any] | None = None,
    window_span: Tuple[int, int] = (0, 0),
) -> InvariantFrame:
    """
    Compute SCION one-tick measures on the given window and classify into a band.
    Uses Splash->SCION mapping already present in splash.geometry.
    """

    m = one_tick_measures(X_window, eval_knobs=eval_knobs)


    label: BandLabel = classify_measures(m, bands)
    measures = {
        "alignment_score": m.alignment_score,
        "tension": m.tension,
        "bend_spread": m.bend_spread,
        "ledger": m.ledger,
        "memory_mean": m.memory_mean,
        "shed_rate": m.shed_rate,
        "shed_total": m.shed_total,
        "asymmetry": m.asymmetry,
        "effective_degree": m.effective_degree,
        "cap_fraction": m.cap_fraction,
    }
    return InvariantFrame(
        t=t, window=window_span, measures=measures, label=label, meta=meta or {}
    )
