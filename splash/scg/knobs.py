from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Pruning:
    k_neighbors: int = 8
    mutual: bool = True
    ensure_connected: bool = True
    target_degree: float = 12.0
    degree_tolerance: float = 1.0

@dataclass
class Knobs:
    """Lightweight, SCION-free equivalent used by Splash."""
    norm: str = "row"
    guard_tick: bool = True
    auto_reach: bool = True
    pruning: Pruning = field(default_factory=Pruning)

    @classmethod
    def preset(cls, name: str = "robust") -> "Knobs":
        name = (name or "robust").lower()
        if name == "fast":
            return cls(norm="row", guard_tick=True, auto_reach=True,
                       pruning=Pruning(k_neighbors=6, mutual=True,
                                       ensure_connected=True,
                                       target_degree=10.0, degree_tolerance=2.0))
        if name == "dev":
            return cls(norm="row", guard_tick=True, auto_reach=True,
                       pruning=Pruning(k_neighbors=8, mutual=False,
                                       ensure_connected=True,
                                       target_degree=8.0, degree_tolerance=3.0))
        # default: robust
        return cls(norm="row", guard_tick=True, auto_reach=True,
                   pruning=Pruning(k_neighbors=8, mutual=True,
                                   ensure_connected=True,
                                   target_degree=12.0, degree_tolerance=1.0))
