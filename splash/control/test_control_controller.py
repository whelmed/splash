import numpy as np
from ..types import EvalKnobs, CoherenceBands, HiddenStates, SequenceBatch, ModelOutputs
from .controller import Controller

class DummyAdapter:
    def tokenize(self, texts, *, max_length=None):
        arr = np.ones((len(texts), 6), dtype=np.int64)
        return SequenceBatch(input_ids=arr, texts=texts)
    def forward(self, batch, *, capture_attention=False):
        B, T = batch.input_ids.shape
        hs = [np.random.randn(B, T, 8) for _ in range(2)]
        hidden = HiddenStates(layers=hs, final=hs[-1])
        return ModelOutputs(batch=batch, hidden=hidden)
    def get_config(self): return {"adapter":"dummy"}

def test_controller_step_teacher_forced():
    # Build synthetic hidden states (B=1,T=10,d=8)
    hidden_bt = np.random.randn(1, 10, 8)
    ctrl = Controller(EvalKnobs(Ns=(4,)), CoherenceBands())
    fr = ctrl.step(hidden_bt, logits=None, window_size=6)
    assert "alignment_score" in fr.measures
    ep = ctrl.get_episode()
    assert len(ep.frames) == 1
    assert len(ep.actions) == 1
