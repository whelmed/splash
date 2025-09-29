from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Optional, Tuple
import json, os
import numpy as np
import torch

from .types import EvalKnobs, CoherenceBands, HiddenStates, SequenceBatch, ModelOutputs
from .geometry import to_places_from_hidden, auto_knobs_to_scion, one_tick_measures

@dataclass
class ROISpec:
    text: str
    start: int
    end: int
    layer_id: int
    mode: str                 # 'misalign' | 'inflation' | 'fracture' | 'pass'
    delta_R: float
    delta_PhiE: float
    meta: Dict[str, Any]

def load_episode_jsonl(paths: Iterable[str]) -> List[Dict[str, Any]]:
    episodes = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    episodes.append(obj)
                except Exception:
                    continue
    return episodes

def mine_rois_from_episodes(
    episodes: List[Dict[str, Any]],
    *,
    min_delta_R: float = 0.01,
    max_delta_PhiE: float = 0.0,
    modes: Optional[List[str]] = None,
) -> List[ROISpec]:
    """
    Scan Episode logs with shape like:
      {"frames":[...], "actions":[{"before":Frame, "after":Frame, "ok":bool, "notes":[...]}], "text": "..."}
    and produce ROIs where the action improved invariants.
    """
    keep_modes = set(modes) if modes else None
    rois: List[ROISpec] = []
    for ep in episodes:
        text = ep.get("text") or ep.get("prompt") or ""
        actions = ep.get("actions", [])
        for act in actions:
            ok = act.get("ok", False)
            before = act.get("before", {})
            after  = act.get("after", {})
            meta   = {"reason": act.get("decision", {}).get("reason", "")}
            # window span (start,end) stored in Frame.window; fallback to meta
            win_b = before.get("window", [0,0]); win_a = after.get("window", win_b)
            s,e = int(win_a[0]), int(win_a[1]) if isinstance(win_a, (list,tuple)) and len(win_a)==2 else (0,0)
            measures_b = before.get("measures", {})
            measures_a = after.get("measures", {})
            Rb, Ra = float(measures_b.get("alignment_score", 0.0)), float(measures_a.get("alignment_score", 0.0))
            Eb, Ea = float(measures_b.get("ledger", 0.0)), float(measures_a.get("ledger", 0.0))
            dR  = Ra - Rb
            dPhi= Ea - Eb  # positive means inflation; improvement = negative
            label = before.get("label", {}).get("label") or before.get("label", "warn")
            mode  = act.get("mode") or _infer_mode_from_notes(act.get("notes", []), label)
            if keep_modes and mode not in keep_modes:
                continue
            improved = (dR >= min_delta_R) or (dPhi <= max_delta_PhiE)
            if ok or improved:
                rois.append(ROISpec(text=text, start=s, end=e, layer_id=int(before.get("t", 0)),
                                    mode=mode, delta_R=float(dR), delta_PhiE=float(dPhi), meta=meta))
    return rois

def _infer_mode_from_notes(notes: List[str], fallback: str) -> str:
    blob = " ".join(notes).lower()
    if "fracture" in blob: return "fracture"
    if "inflation" in blob or "phi" in blob: return "inflation"
    if "misalign" in blob or "align" in blob: return "misalign"
    return fallback

# --- tensor harvesting --------------------------------------------------------

@dataclass
class ROITensor:
    h: torch.Tensor        # (1, T_roi, d)
    pos: torch.Tensor      # (1, T_roi)
    layer_id: int
    text: str
    span: Tuple[int,int]
    mode: str
    meta: Dict[str, Any]

def harvest_roi_tensors(
    rois: List[ROISpec],
    *,
    adapter,                      # Splash HFAdapter (or compatible)
    layer_pick: str = "last",     # 'last' | 'mean_last_4' (we’ll slice a single layer for ROI)
    max_tokens: Optional[int] = None,
) -> List[ROITensor]:
    """
    Re-encode the text and extract hidden-state windows for each ROI span.
    """
    if not rois:
        return []
    texts = [r.text for r in rois]
    batch = adapter.tokenize(texts, max_length=max_tokens)
    out   = adapter.forward(batch, capture_attention=False)
    hs: HiddenStates = out.hidden
    B,T = batch.input_ids.shape
    # choose per-sample per-layer tensor set
    # We map: 'last' -> last layer; 'mean_last_4' -> average last 4 layers
    tensors: List[ROITensor] = []
    for i, r in enumerate(rois):
        s, e = max(0, r.start), min(T, r.end or T)
        if s >= e:  # skip invalid
            continue
        if layer_pick == "mean_last_4" and len(hs.layers) >= 5:
            stack = [torch.from_numpy(hs.layers[-j][i:i+1, s:e, :]) for j in range(1,5)]
            h_roi = torch.stack(stack, dim=0).mean(dim=0)  # (1, T_roi, d)
        else:
            h_roi = torch.from_numpy(hs.final[i:i+1, s:e, :])  # (1, T_roi, d)
        pos = torch.arange(s, e).view(1, -1)
        tensors.append(ROITensor(h=h_roi.float(), pos=pos.long(), layer_id=r.layer_id,
                                 text=r.text, span=(s,e), mode=r.mode, meta=r.meta))
    return tensors

# --- SCION monitor (optional; no autograd) -----------------------------------

def monitor_invariants_np(
    h_roi: torch.Tensor, eval_knobs: EvalKnobs, bands: CoherenceBands
) -> Dict[str, float]:
    """
    Compute R, PhiE, rho_phi, Q for a (1,T,d) roi using Splash->SCION path.
    No gradients; for logging only.
    """
    with torch.no_grad():
        X = h_roi.detach().cpu().numpy()  # (1,T,d)
    Xs = to_places_from_hidden(X, distance=eval_knobs.distance, max_tokens=eval_knobs.max_tokens)
    m = one_tick_measures(Xs[0], eval_knobs=eval_knobs)
    return {
        "R": float(m.alignment_score),
        "PhiE": float(m.ledger),
        "rho_phi": float(abs(m.asymmetry)),
        "Q": float(m.bend_spread),
    }
