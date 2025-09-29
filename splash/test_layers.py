import numpy as np

from splash.layers import _pick_reference_N, layer_curves_from_hidden
from splash.types import EvalKnobs, LayerCurve, CoherenceBands

def test_pick_reference_N_various_inputs():
    assert _pick_reference_N([4, 8, 16]) == 8  # median
    assert _pick_reference_N([2, 4]) in (2, 4)
    assert _pick_reference_N([]) == 8  # default fallback

def test_layer_curves_from_hidden_basic():
    # 3 layers, each with (T=6, d=3)
    hidden_layers = [np.random.randn(6, 3) for _ in range(3)]
    knobs = EvalKnobs(Ns=(4,8), stride_fraction=0.5)
    bands = CoherenceBands()
    curves = layer_curves_from_hidden(hidden_layers, eval_knobs=knobs, bands=bands)

    assert isinstance(curves, LayerCurve)
    # layer count should match number of layers
    n_layers = len(hidden_layers)
    for metric, vals in curves.per_layer.items():
        assert len(vals) == n_layers
    # meta should include reference_N
    assert "reference_N" in curves.meta
