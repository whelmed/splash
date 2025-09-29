from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

from .types import EvalKnobs, CoherenceBands, EvalRun, MapResult, HiddenStates
from .adapters import ModelAdapter
from .geometry import select_layer_hidden, to_places_from_hidden
from .tiling import coherence_map_for_sequence
from .layers import layer_curves_from_hidden

def evaluate_prompts(
    texts: List[str],
    *,
    adapter: ModelAdapter,
    eval_knobs: EvalKnobs,
    bands: CoherenceBands,
    capture_attention: bool = False
) -> List[EvalRun]:
    """
    End-to-end for notebook usage:
      1) tokenize + forward (capture hidden states)
      2) choose layer combine
      3) build N-maps per sequence
      4) build layer curves per sequence
    Returns one EvalRun per input text.
    """
    batch = adapter.tokenize(texts, max_length=eval_knobs.max_tokens)
    outputs = adapter.forward(batch, capture_attention=capture_attention)
    H = outputs.hidden

    # combine layers according to knobs
    combined = select_layer_hidden(H.layers, mode=eval_knobs.layer_combine)
    Xs = to_places_from_hidden(combined, distance=eval_knobs.distance, max_tokens=eval_knobs.max_tokens)

    runs: List[EvalRun] = []
    for b in range(len(texts)):
        # coherence map for this sequence
        mres: MapResult = coherence_map_for_sequence(Xs[b], eval_knobs=eval_knobs, bands=bands)

        # layer curves using per-layer states, sequence-specific
        per_layer_X = [to_places_from_hidden(h[b:b+1], distance=eval_knobs.distance, max_tokens=eval_knobs.max_tokens)[0]
                       for h in H.layers]
        curves = layer_curves_from_hidden(per_layer_X, eval_knobs=eval_knobs, bands=bands)

        runs.append(EvalRun(
            coherence_maps={"combined:" + eval_knobs.layer_combine: mres},
            layer_curves=curves,
            config={
                "adapter": adapter.get_config(),
                "eval_knobs": {
                    "Ns": list(eval_knobs.Ns),
                    "stride_fraction": eval_knobs.stride_fraction,
                    "distance": eval_knobs.distance,
                    "normalize": eval_knobs.normalize,
                    "sym_blend": eval_knobs.sym_blend,
                    "target_degree": eval_knobs.target_degree,
                    "degree_tolerance": eval_knobs.degree_tolerance,
                    "k_neighbors": eval_knobs.k_neighbors,
                    "mutual_knn": eval_knobs.mutual_knn,
                    "ensure_connected": eval_knobs.ensure_connected,
                    "layer_combine": eval_knobs.layer_combine,
                    "max_tokens": eval_knobs.max_tokens,
                }
            }
        ))
    return runs

def evaluate_hidden(
    hidden: HiddenStates,
    *,
    eval_knobs: EvalKnobs,
    bands: CoherenceBands
) -> EvalRun:
    """
    Same as evaluate_prompts, but starts from already-captured HiddenStates.
    Useful for offline pipelines (e.g., cached activations).
    """
    combined = select_layer_hidden(hidden.layers, mode=eval_knobs.layer_combine)
    Xs = to_places_from_hidden(combined, distance=eval_knobs.distance, max_tokens=eval_knobs.max_tokens)
    # assume single sequence if batch dim == 1
    X0 = Xs[0]
    mres = coherence_map_for_sequence(X0, eval_knobs=eval_knobs, bands=bands)

    per_layer_X = [to_places_from_hidden(h[0:1], distance=eval_knobs.distance, max_tokens=eval_knobs.max_tokens)[0]
                   for h in hidden.layers]
    curves = layer_curves_from_hidden(per_layer_X, eval_knobs=eval_knobs, bands=bands)

    return EvalRun(
        coherence_maps={"combined:" + eval_knobs.layer_combine: mres},
        layer_curves=curves,
        config={"eval_knobs": {
            "Ns": list(eval_knobs.Ns),
            "stride_fraction": eval_knobs.stride_fraction,
            "distance": eval_knobs.distance,
            "normalize": eval_knobs.normalize,
            "sym_blend": eval_knobs.sym_blend,
            "target_degree": eval_knobs.target_degree,
            "degree_tolerance": eval_knobs.degree_tolerance,
            "k_neighbors": eval_knobs.k_neighbors,
            "mutual_knn": eval_knobs.mutual_knn,
            "ensure_connected": eval_knobs.ensure_connected,
            "layer_combine": eval_knobs.layer_combine,
            "max_tokens": eval_knobs.max_tokens,
        }}
    )
