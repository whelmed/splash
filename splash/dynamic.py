from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List
import torch
import torch.nn as nn
from .sala import SALA, PositionalGate

def _device_auto():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

@dataclass
class LayerSpec:
    layer_id: int
    use_sala: bool = True
    use_pos: bool = False
    pos_mode: str = "diag"      # 'diag' | 'lowrank'
    pos_rank: int = 16
    pos_alpha: float = 0.2
    include_layer_id: bool = False

class DynamicManager(nn.Module):
    """
    Hosts per-layer SALA and/or PositionalGate modules and applies them
    to HF hidden states during adapter.forward() (optionally).

    Usage:
        mgr = DynamicManager(d_model=hidden_size)
        mgr.add_layer(12, use_sala=True, use_pos=True, pos_mode="diag")

        adapter.set_dynamic_manager(mgr)  # (HFAdapter hook)
    """
    def __init__(self, d_model: int, *, device: Optional[str] = None):
        super().__init__()
        self.d_model = int(d_model)
        self.device = torch.device(device or _device_auto())
        self.layer_sala = nn.ModuleDict()   # key: str(layer_id)
        self.layer_pos  = nn.ModuleDict()   # key: str(layer_id)

    def add_layer(self, layer_id: int, *, use_sala: bool = True,
                  use_pos: bool = False, pos_mode: str = "diag",
                  pos_rank: int = 16, pos_alpha: float = 0.2,
                  include_layer_id: bool = False):
        key = str(int(layer_id))
        if use_sala and key not in self.layer_sala:
            self.layer_sala[key] = SALA(self.d_model, m=4).to(self.device)
        if use_pos and key not in self.layer_pos:
            self.layer_pos[key] = PositionalGate(
                self.d_model, mode=pos_mode, rank=pos_rank,
                alpha=pos_alpha, include_layer_id=include_layer_id
            ).to(self.device)

    @torch.no_grad()
    def apply_layers(self, hidden_layers: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        hidden_layers: list of (B,T,d) torch tensors from HF with gradients detached.
        Contract: out.hidden_states usually includes embeddings at index 0; we skip index 0.
        """
        if not self.layer_sala and not self.layer_pos:
            return hidden_layers

        out: List[torch.Tensor] = []
        # Build positions once (assume same T for all layers)
        B, T, d = hidden_layers[-1].shape
        pos = torch.arange(T, device=self.device).view(1, T).repeat(B, 1)

        for i, h in enumerate(hidden_layers):
            # layer 0 is often embeddings; keep pass-through
            if i == 0:
                out.append(h)
                continue
            key = str(i)  # layer index (HF convention: embeddings=0, layers start at 1)
            y = h.to(self.device)
            if key in self.layer_sala:
                y = self.layer_sala[key](y)
            if key in self.layer_pos:
                y = self.layer_pos[key](y, pos=pos, layer_id=i)
            out.append(y)
        return out

    # ----- persistence -----
    def save(self, path: str):
        torch.save({
            "d_model": self.d_model,
            "state_dict": self.state_dict(),
        }, path)

    @staticmethod
    def load(path: str, *, device: Optional[str] = None) -> "DynamicManager":
        chk = torch.load(path, map_location=device or _device_auto())
        mgr = DynamicManager(d_model=int(chk["d_model"]), device=device)
        mgr.load_state_dict(chk["state_dict"])
        return mgr
