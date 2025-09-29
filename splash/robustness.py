from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List
import numpy as np

from .types import EvalKnobs, CoherenceBands, MapResult
from .adapters import ModelAdapter
from .geometry import select_layer_hidden, to_places_from_hidden
from .tiling import coherence_map_for_sequence

from transformers import pipeline

# Simple perturbation type
Perturb = Callable[[List[str]], List[str]]


# cache pipelines globally so they load once
_fill_mask = None
_paraphrase_pipe = None

def _get_fill_mask():
    global _fill_mask
    if _fill_mask is None:
        _fill_mask = pipeline("fill-mask", model="bert-base-uncased")
    return _fill_mask

def _get_paraphraser():
    global _paraphrase_pipe
    if _paraphrase_pipe is None:
        _paraphrase_pipe = pipeline("text2text-generation", model="t5-small")
    return _paraphrase_pipe

def synonym_swap(texts, *, prob: float = 0.3):
    """
    Replace a random word in each text with a synonym predicted
    by a masked-language model. Deterministic by seeding numpy.
    """
    rng = np.random.default_rng(12345)
    fill_mask = _get_fill_mask()
    out = []
    for t in texts:
        words = t.split()
        if len(words) < 3:
            out.append(t)
            continue
        idx = rng.integers(1, len(words)-1)  # avoid first/last
        masked = words.copy()
        masked[idx] = fill_mask.tokenizer.mask_token
        masked_text = " ".join(masked)
        pred = fill_mask(masked_text, top_k=1)[0]["token_str"]
        words[idx] = pred
        out.append(" ".join(words))
    return out

def paraphrase(texts, *, strength: float = 0.5):
    """
    Generate paraphrases using T5-small.
    """
    para = _get_paraphraser()
    out = []
    for t in texts:
        prompt = "paraphrase: " + t
        res = para(prompt, max_length=64, num_return_sequences=1)
        out.append(res[0]["generated_text"])
    return out

def distractor_clause(texts, *, position: str = "end"):
    filler = "Note: unrelated background details."
    out = []
    for t in texts:
        if position == "start":
            out.append(filler + " " + t)
        elif position == "middle":
            words = t.split()
            mid = len(words)//2
            out.append(" ".join(words[:mid] + [filler] + words[mid:]))
        else:
            out.append(t + " " + filler)
    return out


# ---- robustness orchestration ----

@dataclass
class RobustnessReport:
    base: MapResult
    variants: Dict[str, MapResult]
    deltas: Dict[str, Dict[str, float]]  # variant -> metric -> delta

def _global_alignment(res: MapResult) -> float:
    # convenience: use global_means if present
    return float(res.global_means.get("alignment_score", 0.0))

def perturb_and_compare(
    texts: List[str],
    *,
    adapter: ModelAdapter,
    eval_knobs: EvalKnobs,
    bands: CoherenceBands,
    variants: Dict[str, Perturb]
) -> RobustnessReport:
    """
    Run baseline and perturbed prompts; return metric deltas (alignment_score and others)
    """
    # baseline
    batch = adapter.tokenize(texts, max_length=eval_knobs.max_tokens)
    outputs = adapter.forward(batch, capture_attention=False)
    H = outputs.hidden
    # choose combine per knobs
    comb = select_layer_hidden(H.layers, mode=eval_knobs.layer_combine)
    Xs = to_places_from_hidden(comb, distance=eval_knobs.distance, max_tokens=eval_knobs.max_tokens)
    # evaluate first sample only (v1)
    base = coherence_map_for_sequence(Xs[0], eval_knobs=eval_knobs, bands=bands)

    # variants
    variants_out: Dict[str, MapResult] = {}
    for name, fn in variants.items():
        t2 = fn(texts)
        b2 = adapter.tokenize(t2, max_length=eval_knobs.max_tokens)
        o2 = adapter.forward(b2, capture_attention=False)
        H2 = o2.hidden
        comb2 = select_layer_hidden(H2.layers, mode=eval_knobs.layer_combine)
        Xs2 = to_places_from_hidden(comb2, distance=eval_knobs.distance, max_tokens=eval_knobs.max_tokens)
        variants_out[name] = coherence_map_for_sequence(Xs2[0], eval_knobs=eval_knobs, bands=bands)

    # deltas (global means)
    deltas: Dict[str, Dict[str, float]] = {}
    base_means = base.global_means
    for name, res in variants_out.items():
        d: Dict[str, float] = {}
        for k in set(list(base_means.keys()) + list(res.global_means.keys())):
            d[k] = float(res.global_means.get(k, 0.0) - base_means.get(k, 0.0))
        deltas[name] = d

    return RobustnessReport(base=base, variants=variants_out, deltas=deltas)
