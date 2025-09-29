import numpy as np
import pytest

from splash.geometry import (
    select_layer_hidden,
    to_places_from_hidden,
    auto_knobs_to_scion,
    one_tick_measures,
    classify_measures,
)
from splash.types import EvalKnobs, CoherenceBands, BandLabel

def test_select_layer_hidden_last_and_mean():
    layers = [np.ones((2,4,3))*i for i in range(1,4)]  # 1, 2, 3
    out_last = select_layer_hidden(layers, mode="last")
    assert np.allclose(out_last, layers[-1])

    out_mean = select_layer_hidden(layers, mode="mean_last_4")
    # should average all layers because len(layers)=3 < 4
    expected = (layers[0] + layers[1] + layers[2]) / 3
    assert np.allclose(out_mean, expected)


def test_select_layer_hidden_custom_weights():
    layers = [np.ones((1,2,2)), np.ones((1,2,2))*2]
    weights = np.array([0.25,0.75])
    out = select_layer_hidden(layers, mode="custom", custom_weights=weights)
    expected = 0.25*layers[0] + 0.75*layers[1]
    assert np.allclose(out, expected)

def test_to_places_from_hidden_cosine_and_l2():
    hidden = np.random.randn(2,5,4)
    Xs_cos = to_places_from_hidden(hidden, distance="cosine")
    assert len(Xs_cos) == 2
    assert np.allclose(np.linalg.norm(Xs_cos[0],axis=1),1,atol=1e-6)

    Xs_l2 = to_places_from_hidden(hidden, distance="l2")
    assert len(Xs_l2) == 2
    # mean should be ~0 across tokens
    assert np.allclose(np.mean(Xs_l2[0],axis=0),0,atol=1e-6)

def test_auto_knobs_to_scion_mapping():
    ek = EvalKnobs(normalize="sym", sym_blend=0.5, k_neighbors=10, mutual_knn=False, ensure_connected=False)
    k = auto_knobs_to_scion(ek)
    assert isinstance(k, EvalKnobs)
    assert k.normalize == "sym"
    assert pytest.approx(k.sym_blend) == 0.5
    assert k.k_neighbors == 10
    assert k.mutual_knn is False
    assert k.ensure_connected is False

def test_one_tick_measures_and_classify():
    # simple 1D line of 5 points
    X = np.linspace(0,1,5)[:,None]
    ek = EvalKnobs(Ns=(4,))
    m = one_tick_measures(X, eval_knobs=ek)
    assert hasattr(m, "alignment_score")

    bands = CoherenceBands()
    label = classify_measures(m, bands)
    assert isinstance(label, BandLabel)
    assert label.label in ["pass","near","warn","fail"]
