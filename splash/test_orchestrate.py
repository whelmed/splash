import numpy as np

from splash.orchestrate import evaluate_prompts, evaluate_hidden
from splash.types import EvalKnobs, CoherenceBands, HiddenStates, SequenceBatch, ModelOutputs, EvalRun

class DummyAdapter:
    def tokenize(self, texts, *, max_length=None):
        arr = np.ones((len(texts), 5), dtype=np.int64)
        return SequenceBatch(input_ids=arr, texts=texts)

    def forward(self, batch, *, capture_attention=False):
        B, T = batch.input_ids.shape
        hs = [np.random.randn(B, T, 4) for _ in range(3)]
        hidden = HiddenStates(layers=hs, final=hs[-1])
        return ModelOutputs(batch=batch, hidden=hidden)

    def get_config(self):
        return {"adapter": "DummyAdapter"}

def test_evaluate_prompts_returns_evalrun():
    adapter = DummyAdapter()
    knobs = EvalKnobs(Ns=(4,))
    bands = CoherenceBands()
    runs = evaluate_prompts(["hello world"], adapter=adapter, eval_knobs=knobs, bands=bands)
    assert isinstance(runs, list)
    assert isinstance(runs[0], EvalRun)
    assert "combined:last" in runs[0].coherence_maps

def test_evaluate_hidden_direct():
    hs = [np.random.randn(1, 6, 4) for _ in range(2)]
    hidden = HiddenStates(layers=hs, final=hs[-1])
    knobs = EvalKnobs(Ns=(4,))
    bands = CoherenceBands()
    run = evaluate_hidden(hidden, eval_knobs=knobs, bands=bands)
    assert isinstance(run, EvalRun)
    assert "combined:last" in run.coherence_maps
