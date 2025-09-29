from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .roi_miner import ROITensor, monitor_invariants_np
from .dynamic import DynamicManager

# Differentiable invariants (SCION-diff)
from .diff_scion import unroll_measures_diff


# ------------------------------------------------------------------------------
# Proxy losses (kept for fallback / ablation)
# ------------------------------------------------------------------------------

def loss_smooth_align(h: torch.Tensor) -> torch.Tensor:
    """
    Encourage alignment of consecutive tokens: minimize (1 - cos(h_t, h_{t+1})).
    h: (B,T,d)
    """
    h0 = F.normalize(h[:, :-1, :], dim=-1)
    h1 = F.normalize(h[:, 1:,  :], dim=-1)
    cs = (h0 * h1).sum(dim=-1)                 # (B,T-1)
    return (1.0 - cs).mean()

def loss_curvature(h: torch.Tensor) -> torch.Tensor:
    """
    Penalize second difference ||h_{t+1} - 2h_t + h_{t-1}||^2
    """
    x_prev = h[:, :-2, :]
    x_curr = h[:, 1:-1, :]
    x_next = h[:, 2:,  :]
    dd = x_next - 2.0 * x_curr + x_prev
    return (dd.pow(2).mean())

def loss_small_change(h_old: torch.Tensor, h_new: torch.Tensor) -> torch.Tensor:
    """
    Regularize adapter strength: mean ||h_new - h_old||^2
    """
    return (h_new - h_old).pow(2).mean()


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------

class ROIDataset(Dataset):
    def __init__(self, rois: List[ROITensor], min_len: int = 6):
        self.items = [r for r in rois if r.h.shape[1] >= min_len]
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        r = self.items[idx]
        return {"h": r.h.clone(), "pos": r.pos.clone(), "layer_id": r.layer_id,
                "text": r.text, "span": r.span, "mode": r.mode, "meta": r.meta}


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Optim
    lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    epochs: int = 2
    grad_clip: float = 1.0

    # Proxy loss weights (fallback / regularizer)
    w_smooth: float = 1.0
    w_curve: float = 0.25
    w_small: float = 0.1

    # Use differentiable invariants?
    use_diff_scion: bool = True
    R_target: float = 0.65                # warn band
    wR: float = 1.0                       # (R_target - R)^2
    wQ: float = 0.25                      # curvature proxy
    wPhi: float = 0.5                     # ReLU(PhiE) penalty
    wrho: float = 0.25                    # dislocation proxy penalty

    # SCION-diff kernel/dynamics (with annealing)
    diff_k: int = 8
    diff_reach: float = 1.5
    diff_steps_start: int = 6
    diff_steps_end: int = 12
    diff_tau_start: float = 0.30          # soft-kNN temperature (start)
    diff_tau_end: float = 0.10            # anneal towards sharper selection
    diff_dt: float = 0.05

    # Oracle consistency (exact SCION vs diff)
    ci_weight: float = 0.10               # set 0 to disable
    monitor_every: int = 100              # SCION monitor cadence (steps)

    # Min ROI length
    min_roi_len: int = 6


# ------------------------------------------------------------------------------
# Loss through differentiable invariants
# ------------------------------------------------------------------------------

def invariant_loss_diff(inv: Dict[str, torch.Tensor], cfg: TrainConfig) -> torch.Tensor:
    """
    L = wR*(R_target - R)^2 + wQ*Q + wPhi*relu(PhiE) + wrho*rho_phi
    """
    R    = inv["R"]
    Q    = inv["Q"]
    PhiE = inv["PhiE"]
    rho  = inv["rho_phi"]
    return (
        cfg.wR   * (cfg.R_target - R).relu().pow(2)
      + cfg.wQ   * Q
      + cfg.wPhi * F.relu(PhiE)
      + cfg.wrho * rho
    )


def consistency_penalty(inv_diff: Dict[str, torch.Tensor], inv_exact: Dict[str, float]) -> torch.Tensor:
    """
    || inv_diff - inv_exact ||^2 over a few key terms.
    """
    loss = 0.0
    for k in ("R", "Q", "PhiE", "rho_phi"):
        v_diff = inv_diff[k]
        v_exact = torch.as_tensor(inv_exact.get(k, 0.0), dtype=v_diff.dtype, device=v_diff.device)
        loss = loss + (v_diff - v_exact).pow(2)
    return loss


# ------------------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------------------

def train_dynamic_adapters(
    *,
    dyn_manager: DynamicManager,        # contains SALA/PosGate nn.Modules
    rois: List[ROITensor],
    cfg: TrainConfig = TrainConfig(),
    monitor_eval_knobs = None,          # optional EvalKnobs for exact SCION monitoring
    monitor_bands = None,               # optional bands
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fine-tune SALA/PosGate directly through differentiable invariants (SCION-diff).
    Falls back to proxy losses if cfg.use_diff_scion=False.
    """
    # Filter too-short ROIs up front
    rois = [r for r in rois if r.h.shape[1] >= cfg.min_roi_len]
    if len(rois) == 0:
        return {"steps": 0, "loss": 0.0, "logs": []}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dyn_manager.to(device).train()
    params = [p for p in dyn_manager.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    ds = ROIDataset(rois, min_len=cfg.min_roi_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    total_steps = max(1, cfg.epochs * math.ceil(len(ds) / max(1, cfg.batch_size)))
    step = 0
    logs: List[Dict[str, Any]] = []

    for epoch in range(cfg.epochs):
        for batch in dl:
            h = batch["h"].to(device)         # (B,T,d)
            pos = batch["pos"].to(device)     # (B,T)
            h_old = h.clone()

            # Apply adapters once per pack (we apply all configured keys once; near-identity at init)
            y = h
            for key, mod in dyn_manager.layer_sala.items():
                y = mod(y)
            for key, mod in dyn_manager.layer_pos.items():
                y = mod(y, pos=pos, layer_id=int(key) if key.isdigit() else 0)

            if cfg.use_diff_scion:
                # Anneal soft-kNN temperature and unroll steps
                progress = step / max(1, total_steps - 1)
                tau = cfg.diff_tau_start + (cfg.diff_tau_end - cfg.diff_tau_start) * progress
                unroll_steps = int(round(cfg.diff_steps_start + (cfg.diff_steps_end - cfg.diff_steps_start) * progress))
                unroll_steps = max(1, unroll_steps)

                # Compute differentiable invariants per ROI in the batch and build loss
                L_batch = 0.0
                for b in range(y.size(0)):
                    x0 = y[b]                 # (T,d)
                    inv = unroll_measures_diff(
                        x0, k=cfg.diff_k, reach=cfg.diff_reach,
                        steps=unroll_steps, dt=cfg.diff_dt, tau=tau
                    )
                    L = invariant_loss_diff(inv, cfg)

                    # small-change regularizer (keep adapters gentle)
                    L = L + cfg.w_small * (y[b] - h_old[b]).pow(2).mean()

                    # optional consistency to exact SCION (oracle)
                    if cfg.ci_weight > 0.0 and (monitor_eval_knobs is not None):
                        inv_exact = monitor_invariants_np(y[b:b+1].detach(), monitor_eval_knobs, monitor_bands)
                        L = L + cfg.ci_weight * consistency_penalty(inv, inv_exact)

                    L_batch = L_batch + L
                loss = L_batch / y.size(0)

            else:
                # Fallback proxy objective (kept for ablation)
                L_s = loss_smooth_align(y) * cfg.w_smooth
                L_c = loss_curvature(y)   * cfg.w_curve
                L_m = loss_small_change(h_old, y) * cfg.w_small
                loss = L_s + L_c + L_m

            optim.zero_grad()
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optim.step()

            # Optional monitoring with exact SCION (no grad)
            if (step % cfg.monitor_every == 0):
                log_row: Dict[str, Any] = {"step": int(step), "loss": float(loss.item())}
                if monitor_eval_knobs is not None:
                    try:
                        inv_exact = monitor_invariants_np(y[:1].detach(), monitor_eval_knobs, monitor_bands)
                        log_row.update({k: float(v) for k, v in inv_exact.items()})
                    except Exception:
                        pass
                logs.append(log_row)

            step += 1

    return {"steps": step, "loss": float(loss.item()), "logs": logs}
