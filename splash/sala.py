# sala.py
# ------------------------------------------------------------
# SWG/SCG-aligned adapters: SALA (orthogonal rotations) + PositionalGate
# ------------------------------------------------------------
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _unit_vectors(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize last-dim vectors to unit length."""
    # W: (..., d)
    nrm = W.norm(dim=-1, keepdim=True).clamp_min(eps)
    return W / nrm


def _fourier_features(pos: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
    """
    Fourier time features in [0,1] → R^{2*n_freq}.
    pos: (B,T) or (T,) integer positions (0..L-1); we internally scale to [0,1].
    """
    if pos.dim() == 1:  # (T,)
        pos = pos[None, :]
    B, T = pos.shape
    device = pos.device
    # scale to [0,1]
    t = pos.float()
    L = (t.max(dim=1, keepdim=True).values.clamp_min(1.0))
    x = (t / L).unsqueeze(-1)  # (B,T,1)
    freqs = torch.exp2(torch.arange(n_freq, device=device).float()) * math.pi  # [pi, 2pi, 4pi, ...]
    ang = x * freqs  # (B,T,n_freq)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B,T,2*n_freq)


# ------------------------------------------------------------
# SALA: product of Householder reflections (orthogonal) as a small adapter.
# ------------------------------------------------------------

class SALA(nn.Module):
    """
    SWG-Aligned Rotations (SALA)
    ----------------------------
    Orthogonal adapter Q in O(d) parameterized as a product of m Householder
    reflections H(v) = I - 2 * v v^T / (v^T v). Orthogonality is exact; energy is preserved.

    Args:
      d_model:   feature dim
      m:         number of reflections (2..8 is typical). Larger m = more expressive.
      init_scale:initialize raw vectors small so Q ≈ I at start

    Forward:
      x: (..., d_model)  -> Q x  (applied on the last dimension)

    Notes:
      • Determinant = (-1)^m; if you require det(Q)=+1 (proper rotation), choose even m.
      • For multi-head blocks, you can share one SALA per layer or per subspace.
    """
    def __init__(self, d_model: int, m: int = 4, init_scale: float = 1e-3):
        super().__init__()
        assert m >= 1, "m >= 1"
        self.d = d_model
        self.m = m
        # Raw vectors; we normalize to unit vectors inside forward
        W = torch.randn(m, d_model) * init_scale
        self.W = nn.Parameter(W)

    @torch.no_grad()
    def orthogonality_error(self, n: int = 128) -> float:
        """Quick check: ||Q^T Q - I||_F on random batch vectors."""
        x = torch.randn(n, self.d, device=self.W.device)
        y = self(x)  # (n,d)
        G = y.T @ y / n
        err = torch.linalg.norm(G - torch.eye(self.d, device=G.device), ord="fro").item()
        return err

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        v = _unit_vectors(self.W)  # (m, d)
        y = x
        # Apply H_m ... H_1  (right-multiplication on feature dim)
        # Each reflection: y <- y - 2 * (y·v) v
        for i in range(self.m):
            vi = v[i]  # (d,)
            proj = torch.tensordot(y, vi, dims=([-1], [0]))  # (...,)
            y = y - 2.0 * proj.unsqueeze(-1) * vi
        return y


# ------------------------------------------------------------
# PositionalGate: small, stable gating as a function of position (and optionally layer).
# ------------------------------------------------------------
# --- in sala.py ---------------------------------------------------------------
def _fourier_features(pos: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
    """
    Fourier time features in [0,1] → R^{2*n_freq}.
    pos: (B,T) or (T,) integer positions (0..L-1). Returns (B,T,2*n_freq).
    """
    if pos.dim() == 1:                  # (T,) → (1,T)
        pos = pos[None, :]
    assert pos.dim() == 2, f"pos must be (B,T) or (T,), got shape={list(pos.shape)}"
    B, T = pos.shape
    device = pos.device
    t = pos.float()
    # scale to [0,1] per batch row to avoid blowup on very long sequences
    L = t.max(dim=1, keepdim=True).values.clamp_min(1.0)
    x = (t / L).unsqueeze(-1)           # (B,T,1)
    freqs = torch.exp2(torch.arange(n_freq, device=device).float()) * math.pi  # [pi,2pi,4pi,...]
    ang = x * freqs                     # (B,T,n_freq)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B,T,2*n_freq)




class PositionalGate(nn.Module):
    """
    PositionalGate
    --------------
    A tiny gate g(t) that rescales features to linearize/steady the SCG arc across time.
    Two modes:
      - 'diag'   : per-feature diagonal gating   (h * g(t))
      - 'lowrank': low-rank subspace gating      (h + U diag(g(t)) V^T h)

    The gate is centered at 1.0 (identity) and bounded via tanh:
      diag:    g = 1 + alpha * tanh(MLP(phi(t)))
      lowrank: project -> scale -> backproject

    Args:
      d_model:  feature dim
      mode:     'diag' | 'lowrank'
      rank:     low-rank dimension if mode='lowrank'
      n_freq:   Fourier features count (per sin/cos bank)
      alpha:    max fractional change (e.g., 0.2 means ±20%)
      include_layer_id: if True, append a scalar layer id feature (normalized)

    Forward:
      h:   (B,T,d)
      pos: (B,T) or (T,) integer token positions (0..L-1)
      layer_id: optional int or float for layer, used when include_layer_id=True
    """
    def __init__(
        self,
        d_model: int,
        mode: str = "diag",
        rank: int = 16,
        n_freq: int = 8,
        alpha: float = 0.2,
        include_layer_id: bool = False,
    ):
        super().__init__()
        assert mode in ("diag", "lowrank")
        self.d = d_model
        self.mode = mode
        self.rank = rank
        self.alpha = alpha
        self.n_freq = n_freq
        self.include_layer_id = include_layer_id

        in_dim = 2 * n_freq
        if include_layer_id:
            in_dim += 1

        hidden = max(32, 4 * n_freq)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, (d_model if mode == "diag" else rank)),
        )
        if mode == "lowrank":
            # Learn subspace (U,V) with orthonormal columns (optional QR in reset).
            self.U = nn.Parameter(torch.randn(d_model, rank) * (1.0 / math.sqrt(d_model)))
            self.V = nn.Parameter(torch.randn(d_model, rank) * (1.0 / math.sqrt(d_model)))

    @torch.no_grad()
    def reset_subspace_orthonormal(self):
        if self.mode == "lowrank":
            # QR for orthonormal columns
            qU, _ = torch.linalg.qr(self.U, mode="reduced")
            qV, _ = torch.linalg.qr(self.V, mode="reduced")
            self.U.copy_(qU)
            self.V.copy_(qV)


    def forward(self, h: torch.Tensor, pos: torch.Tensor, layer_id: Optional[float] = None) -> torch.Tensor:
        """
        h:   (B,T,d)
        pos: (B,T) or (T,)
        """
        # ---- shape & dtype guards ----
        assert h.dim() == 3, f"PositionalGate expects h=(B,T,d), got {list(h.shape)}"
        B, T, d = h.shape
        if pos.dim() == 1:
            pos = pos.unsqueeze(0).expand(B, -1)            # (B,T)
        elif pos.dim() == 2 and pos.size(0) != B:
            pos = pos.expand(B, -1)                         # broadcast batch if needed
        pos = pos.to(h.device)

        # ---- time features ----
        feat = _fourier_features(pos, self.n_freq)          # (B,T,2*n_freq)
        if self.include_layer_id:
            lid_val = float(layer_id if layer_id is not None else 0.0)
            lid = torch.full((B, T, 1), lid_val, device=h.device, dtype=feat.dtype)
            feat = torch.cat([feat, lid], dim=-1)           # (B,T,in_dim)

        # ---- gating ----
        g_raw = self.mlp(feat)                              # (B,T,d) or (B,T,rank)
        if self.mode == "diag":
            g = 1.0 + self.alpha * torch.tanh(g_raw)        # (B,T,d)
            return h * g
        else:
            # low-rank: h + U diag(g) V^T h
            # U,V: (d,r); g: (B,T,r)
            g = 1.0 + self.alpha * torch.tanh(g_raw)        # (B,T,rank)
            V = self.V                                      # (d,r)
            U = self.U                                      # (d,r)
            # project → scale → backproject
            hV = torch.einsum("btd,dr->btr", h, V)          # (B,T,r)
            hV = hV * g                                     # (B,T,r)
            Ug = torch.einsum("btr,rd->btd", hV, U.T)       # (B,T,d)  (note rd)
            return h + Ug

# ------------------------------------------------------------
# Integration helpers
# ------------------------------------------------------------

class LayerAdapter(nn.Module):
    """
    Convenience wrapper to attach SALA + PositionalGate to a layer.
    Use any subset (set enable flags).
    """
    def __init__(
        self,
        d_model: int,
        sala_m: int = 4,
        pos_mode: Optional[str] = "diag",
        pos_rank: int = 16,
        pos_alpha: float = 0.2,
        include_layer_id: bool = False,
    ):
        super().__init__()
        self.sala = SALA(d_model, m=sala_m) if sala_m > 0 else None
        self.pos = PositionalGate(
            d_model, mode=(pos_mode or "diag"), rank=pos_rank, alpha=pos_alpha, include_layer_id=include_layer_id
        ) if pos_mode else None

    def forward(self, h: torch.Tensor, pos: Optional[torch.Tensor] = None, layer_id: Optional[int] = None) -> torch.Tensor:
        # h: (B,T,d)
        if self.sala is not None:
            h = self.sala(h)
        if self.pos is not None:
            assert pos is not None, "PositionalGate requires pos (token indices)."
            h = self.pos(h, pos=pos, layer_id=layer_id)
        return h


# ------------------------------------------------------------
# Invariant loss scaffolding (attach to your ROI trainer)
# ------------------------------------------------------------

class InvariantLoss(nn.Module):
    """
    Composite invariant loss to train adapters on ROIs that improved during runtime control.
    L = wR*(R_target - R)^2 + wPhiE*relu(PhiE)^2 + wrho*(rho_phi)^2 + wQ*(Q)^2
    Supply invariants from your Splash/SCION measure call.

    Tip: set R_target per domain (e.g., 0.65 for warn threshold).
    """
    def __init__(self, wR=1.0, wPhiE=0.5, wrho=0.25, wQ=0.25, R_target=0.65):
        super().__init__()
        self.wR, self.wPhiE, self.wrho, self.wQ = wR, wPhiE, wrho, wQ
        self.R_target = R_target

    def forward(self, invariants: dict) -> torch.Tensor:
        R     = torch.as_tensor(invariants.get("R", 0.0), dtype=torch.float32)
        PhiE  = torch.as_tensor(invariants.get("PhiE", 0.0), dtype=torch.float32)
        rho   = torch.as_tensor(invariants.get("rho_phi", 0.0), dtype=torch.float32)
        Q     = torch.as_tensor(invariants.get("Q", 0.0), dtype=torch.float32)

        return (
            self.wR   * (self.R_target - R).pow(2)
          + self.wPhiE* F.relu(PhiE).pow(2)
          + self.wrho * rho.pow(2)
          + self.wQ   * Q.pow(2)
        )
