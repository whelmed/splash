import numpy as np
from . import calibrate_model, SweepSpec
from ..types import EvalKnobs, CoherenceBands, SequenceBatch, HiddenStates, ModelOutputs

class DummyAdapter:
    def __init__(self, d_model=16, seed=0):
        self.d = d_model
        self.rng = np.random.default_rng(seed)
    def tokenize(self, texts, *, max_length=None):
        T = max(16, max(len(t.split()) for t in texts))
        return SequenceBatch(input_ids=np.ones((len(texts), T), dtype=np.int64), texts=texts)
    def forward(self, batch, *, capture_attention=False):
        B, T = batch.input_ids.shape
        # two “layers” with slight structure
        base = self.rng.standard_normal((B, T, self.d))
        layer2 = base + 0.1 * np.cumsum(self.rng.standard_normal((B, T, self.d)), axis=1)
        hidden = HiddenStates(layers=[base, layer2], final=layer2)
        return ModelOutputs(batch=batch, hidden=hidden)
    def get_config(self):
        return {"adapter": "Dummy", "d_model": self.d}

def test_calibrate_minimal():
    adapter = DummyAdapter()
    prompts = [
        "Alpha beta gamma delta.",
        "Short prompt.",
        "Reason about numbers: 12 + 35 = ?",
    ]
    base = EvalKnobs(max_tokens=64)
    bands = CoherenceBands()
    sweep = SweepSpec(Ns=[24], k_neighbors=[6, 8], settle_steps=[12], distance=["l2"], layer_combine=["mean_last_4"])
    res = calibrate_model(adapter=adapter, prompts=prompts, base_knobs=base, bands=bands, sweep=sweep)
    assert isinstance(res.summary, list) and len(res.summary) >= 1
    assert "profile" in res.__dict__
    p = res.profile
    assert isinstance(p.op["N"], int) and isinstance(p.op["k"], int)
    assert 0.0 <= p.stats.R_last_mean <= 1.0
