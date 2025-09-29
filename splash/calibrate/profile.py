from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, Any
import json

@dataclass
class ProfileStats:
    # Key telemetry aggregated across prompts
    degree_mean: float
    degree_std: float
    entropy_norm_mean: float
    entropy_norm_std: float
    R_last_mean: float
    R_last_std: float
    PhiE_mean: float
    PhiE_std: float
    rho_phi_mean: float
    rho_phi_std: float
    score: float

@dataclass
class ModelProfile:
    adapter: Dict[str, Any]
    places: Dict[str, Any]      # e.g., {"distance": "l2"}
    op: Dict[str, Any]          # chosen operating point (N, k, settle, layer_combine)
    degree: Dict[str, float]    # {"target": 16, "mean": ..., "std": ...}
    entropy_norm: Dict[str, float]
    thresholds: Dict[str, float]    # recommended bands/guards
    policy: Dict[str, Any]          # suggested control actions per mode
    stats: ProfileStats
    meta: Dict[str, Any] = field(default_factory=dict)

def save_profile(profile: ModelProfile, path: str):
    obj = asdict(profile)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_profile(path: str) -> ModelProfile:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    stats = ProfileStats(**obj["stats"])
    obj["stats"] = stats
    return ModelProfile(**obj)
