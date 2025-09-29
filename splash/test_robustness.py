import numpy as np

from splash.robustness import (
    synonym_swap, paraphrase, distractor_clause,
    perturb_and_compare, RobustnessReport
)
from splash.types import EvalKnobs, CoherenceBands, SequenceBatch, HiddenStates, ModelOutputs

class DummyAdapter:
    """Simple mock adapter returning random hidden states"""
    def __init__(self, d_model=4):
        self.d_model = d_model

    def tokenize(self, texts, *, max_length=None):
        arr = np.ones((len(texts), 5), dtype=np.int64)
        return SequenceBatch(input_ids=arr, texts=texts)

    def forward(self, batch, *, capture_attention=False):
        B, T = batch.input_ids.shape
        # 2 fake layers + final
        hs = [np.random.randn(B, T, self.d_model) for _ in range(2)]
        hidden = HiddenStates(layers=hs, final=hs[-1])
        return ModelOutputs(batch=batch, hidden=hidden)

    def get_config(self):
        return {"adapter": "DummyAdapter"}

def test_synonym_swap_and_paraphrase_and_distractor():
    texts = ["This is a quick test.", "A small example."]
    swapped = synonym_swap(texts, prob=1.0)  # force replacement
    assert swapped != texts
    paraphrased = paraphrase(["clause1, clause2"], strength=1.0)
    assert paraphrased[0] != "clause1, clause2"
    distracted = distractor_clause(["short sentence"], position="end")
    assert "unrelated background" in distracted[0]

def test_perturb_and_compare_with_dummy_adapter():
    texts = ["A simple sentence for testing."]
    adapter = DummyAdapter()
    knobs = EvalKnobs(Ns=(4,))
    bands = CoherenceBands()
    variants = {"swap": lambda ts: ["different sentence."]}
    report = perturb_and_compare(
        texts, adapter=adapter, eval_knobs=knobs, bands=bands, variants=variants
    )
    assert isinstance(report, RobustnessReport)
    assert "swap" in report.variants
    assert "alignment_score" in report.base.global_means
