from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Dict, Any

from .types import EvalRun, MapResult, TileMeasures, TileSpec, LayerCurve, BandLabel

def _to_basic(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_basic(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_basic(v) for v in obj]
    return obj

def to_json(run: EvalRun) -> Dict[str, Any]:
    """
    Convert EvalRun (which contains dataclasses and numpy scalars) into a JSON-serializable dict.
    Measures (from scion.api) are dataclasses too, so asdict() works.
    """
    return _to_basic(run)

def from_json(obj: Dict[str, Any]) -> EvalRun:
    """
    Best-effort round-trip reconstruction.
    For v1 simplicity, we keep it dict-like; notebooks can consume dicts directly.
    """
    # We intentionally return the dict as EvalRun-like by unpacking; strict rebuild could be added later.
    # To avoid surprise, we keep a minimal recreation that matches EvalRun shape.
    return EvalRun(
        coherence_maps=obj.get("coherence_maps", {}),
        layer_curves=obj.get("layer_curves", None),
        config=obj.get("config", {}),
    )
