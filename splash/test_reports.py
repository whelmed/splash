from splash.reports import to_json, from_json
from splash.types import EvalRun, MapResult, LayerCurve

def test_to_json_and_from_json_roundtrip():
    dummy_map = MapResult(
        Ns=[4],
        tiles={4: []},
        global_means={"alignment_score": 0.5},
        meta={"T": 10},
    )
    dummy_curve = LayerCurve(
        metric_names=["alignment_score"],
        per_layer={"alignment_score": [0.1, 0.2]},
        layer_names=["l0","l1"],
    )
    run = EvalRun(
        coherence_maps={"combined:last": dummy_map},
        layer_curves=dummy_curve,
        config={"adapter": {"name": "Dummy"}}
    )

    obj = to_json(run)
    assert isinstance(obj, dict)
    assert "coherence_maps" in obj
    assert "layer_curves" in obj

    run2 = from_json(obj)
    assert isinstance(run2, EvalRun)
    assert "combined:last" in run2.coherence_maps
    assert run2.config["adapter"]["name"] == "Dummy"
