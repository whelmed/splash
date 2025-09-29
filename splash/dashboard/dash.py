from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional
from scipy.interpolate import make_interp_spline

# ---- thresholds -------------------------------------------------------------

@dataclass
class DashboardThresholds:
    """Thresholds for overlay lines in invariant grid plots (UI-agnostic)."""
    entropy_min:        Optional[float] = 0.5      # ℋ low → collapse risk
    curvature_max:      Optional[float] = None     # σ²κ high → tension buildup (leave None if not known)
    alignment_min:      Optional[float] = 0.65     # R low → misalignment
    energy_flux_max:    Optional[float] = None     # ΦE > 0 sustained → inflation
    dislocation_max:    Optional[float] = None     # ρφ spikes → local fractures
    barrier_max:        Optional[float] = 0.0      # barrier B ≤ 0 inside safe set (if provided)

# ---- 1) SCG Trajectory (3D) ------------------------------------------------
# splash/dashboard/dash.py  (update signature + coloring logic)

def plot_scg_trajectory(
    coords: np.ndarray,
    health_flags: List[str],
    title: str = "SCG Trajectory",
    mode_flags: Optional[List[str]] = None,   # NEW: if provided, color by mode
) -> Figure:
    """
    If mode_flags is passed, color by failure mode:
      pass=green, near=yellow, misalign=orange, fracture=red, inflation=magenta
    Otherwise fall back to health_flags (green/yellow/red).
    """
    assert coords.ndim == 2 and coords.shape[1] == 3
    N = coords.shape[0]
    xs, ys, zs = coords[:,0], coords[:,1], coords[:,2]
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")

    # choose palette
    if mode_flags is not None:
        palette = {
            "pass": (0,0.6,0),
            "near": (1,0.7,0),
            "misalign": (1.0,0.55,0.0),  # orange
            "fracture": (0.85,0,0),      # red
            "inflation": (0.75,0,0.75),  # magenta
        }
        colors = [palette.get(m, (0.4,0.4,0.4)) for m in mode_flags]
    else:
        palette = {"green": (0,0.6,0), "yellow": (1,0.7,0), "red": (0.85,0,0)}
        colors = [palette.get(h, (0.4,0.4,0.4)) for h in health_flags]

    # smooth polyline
    if N >= 4:
        t = np.arange(N); t_new = np.linspace(0, N-1, 400)
        from scipy.interpolate import make_interp_spline
        sx, sy, sz = make_interp_spline(t, xs)(t_new), make_interp_spline(t, ys)(t_new), make_interp_spline(t, zs)(t_new)
        # interpolate colors
        def interp_color(idx):
            i0 = int(np.floor(idx)); i1 = min(i0+1, N-1); w = idx - i0
            c0, c1 = np.array(colors[i0]), np.array(colors[i1]); return (1-w)*c0 + w*c1
        for i in range(len(t_new)-1):
            ax.plot(sx[i:i+2], sy[i:i+2], sz[i:i+2], color=interp_color(t_new[i]), linewidth=2.0)
    else:
        ax.plot(xs, ys, zs, linewidth=2.0, color="0.4")

    # token markers
    for i, (x,y,z) in enumerate(zip(xs,ys,zs), start=1):
        ax.scatter(x,y,z, color=colors[i-1], s=60, edgecolor="k", zorder=5)
        ax.text(x,y,z, str(i), color="black")

    ax.set_title(title); ax.set_xlabel("SCG-X (phase cos)"); ax.set_ylabel("SCG-Y (phase sin)"); ax.set_zlabel("SCG-Z (curvature)")
    fig.tight_layout(); return fig

# ---- 2) Invariant grid (2x3) -----------------------------------------------

def plot_invariants_grid(
    inv_dict: Mapping[str, Mapping[str, List[float]]],
    thresholds: DashboardThresholds,
    title: str = "Invariant Dashboard",
) -> Figure:
    """
    inv_dict: {"Condition A": {"entropy": [...], "curvature": [...], "alignment": [...],
                               "energy_flux": [...], "dislocation": [...], "barrier": [...]} , ...}
    Missing series are allowed.
    """
    layout = [("entropy", "Coherence Entropy (ℋ)"),
              ("curvature", "Curvature Variance (σ²κ)"),
              ("alignment", "Alignment (R)"),
              ("energy_flux", "Energy Flux (ΦE)"),
              ("dislocation", "Phase Dislocation Density (ρφ)"),
              ("barrier", "Barrier Function (B)")]
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharex=False)
    axs = axs.ravel()

    def _overlay(ax, key: str):
        if key == "entropy" and thresholds.entropy_min is not None:
            ax.axhline(thresholds.entropy_min, color="r", linestyle="--", alpha=0.6)
        if key == "curvature" and thresholds.curvature_max is not None:
            ax.axhline(thresholds.curvature_max, color="r", linestyle="--", alpha=0.6)
        if key == "alignment" and thresholds.alignment_min is not None:
            ax.axhline(thresholds.alignment_min, color="r", linestyle="--", alpha=0.6)
        if key == "energy_flux" and thresholds.energy_flux_max is not None:
            ax.axhline(thresholds.energy_flux_max, color="r", linestyle="--", alpha=0.6)
        if key == "dislocation" and thresholds.dislocation_max is not None:
            ax.axhline(thresholds.dislocation_max, color="r", linestyle="--", alpha=0.6)
        if key == "barrier" and thresholds.barrier_max is not None:
            ax.axhline(thresholds.barrier_max, color="r", linestyle="--", alpha=0.6)

    for i, (key, label) in enumerate(layout):
        ax = axs[i]
        present = False
        for cond, series in inv_dict.items():
            y = series.get(key)
            if y is None:
                continue
            x = np.arange(len(y))
            ax.plot(x, y, label=cond, linewidth=2)
            present = True
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        if present:
            _overlay(ax, key)
        if i >= 3:
            ax.set_xlabel("Layer / Depth")
        if i in (0, 3):
            ax.set_ylabel("Value")

    # legend
    handles, labels = [], []
    for cond in inv_dict.keys():
        # get handle from first axis where plotted
        for line in axs[0].get_lines() + axs[1].get_lines() + axs[2].get_lines():
            # We will not try to deduplicate strictly; small legend top-right
            pass
    axs[0].legend(inv_dict.keys(), loc="best", fontsize=9)

    fig.suptitle(title)
    return fig

# ---- 3) Dislocation heatmap -------------------------------------------------

def plot_dislocation_heatmap(
    matrix: np.ndarray,                      # (layers, tokens) or (tokens, layers) — we auto-fix
    title: str = "Phase Dislocation Density Heatmap",
) -> Figure:
    mat = np.asarray(matrix)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D")
    # Prefer axes as (layers, tokens); if more columns than rows assume this already
    layers_tokens = mat if mat.shape[0] >= mat.shape[1] else mat.T

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(layers_tokens.T, aspect="auto", origin="lower", cmap="bwr", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Dislocation (ρφ)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Token")
    ax.set_title(title)
    return fig

# ---- 4) Summary table -------------------------------------------------------

def make_summary_table(summary_dict: Mapping[str, Mapping[str, float]]) -> pd.io.formats.style.Styler:
    """
    summary_dict: {"Run/Condition": {"H_mid": ..., "H_late": ..., "σ²κ_mid": ..., "R_mid": ..., "ΦE_mid": ..., "ρφ_mid": ...}, ...}
    Returns a pandas Styler you can display in notebooks or export_html.
    """
    df = pd.DataFrame(summary_dict).T
    # basic formatting
    return (df.style
            .format(precision=4)
            .background_gradient(cmap="RdYlGn_r", axis=None))

# ---- 5) One-call convenience for an EvalRun --------------------------------

def render_run_dashboard(
    *,
    inv_grid: Mapping[str, Mapping[str, List[float]]],
    thresholds: Optional[DashboardThresholds] = None,
    dislocation_matrix: Optional[np.ndarray] = None,
) -> Dict[str, Figure]:
    """
    Render the common figures for one run:
      - invariant grid (required)
      - dislocation heatmap (optional, if matrix supplied)
    Returns dict of figures.
    """
    figs: Dict[str, Figure] = {}
    th = thresholds or DashboardThresholds()
    figs["invariants"] = plot_invariants_grid(inv_grid, th, title="Invariant Dashboard")

    if dislocation_matrix is not None:
        figs["dislocation"] = plot_dislocation_heatmap(dislocation_matrix, title="Dislocation Density (ρφ)")
    return figs


# --- Focused trajectory with faint global overlay ----------------------------

def plot_scg_trajectory_with_overlay(
    coords_all: "np.ndarray",                 # (T,3) background path (faint)
    coords_focus: "np.ndarray",               # (t,3) focus window
    focus_flags: List[str],                   # health flags or mode colors for focus
    *,
    title: str = "SCG Trajectory (windowed)",
    mode_flags_focus: Optional[List[str]] = None,
) -> Figure:
    """
    Draw a faint global path (grey) for context and a colored, annotated focus window on top.
    focus_flags/mode_flags_focus follow the same semantics as plot_scg_trajectory.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Background (faint)
    xs, ys, zs = coords_all[:,0], coords_all[:,1], coords_all[:,2]
    ax.plot(xs, ys, zs, color=(0.6,0.6,0.6), alpha=0.25, linewidth=2.0, zorder=0)

    # Focus (colored)
    Nf = coords_focus.shape[0]
    # Choose palette
    if mode_flags_focus is not None:
        palette = {
            "pass": (0,0.6,0),
            "near": (1,0.7,0),
            "misalign": (1.0,0.55,0.0),
            "fracture": (0.85,0,0),
            "inflation": (0.75,0,0.75),
        }
        colors = [palette.get(m, (0.4,0.4,0.4)) for m in mode_flags_focus]
    else:
        palette = {"green": (0,0.6,0), "yellow": (1,0.7,0), "red": (0.85,0,0)}
        colors = [palette.get(h, (0.4,0.4,0.4)) for h in focus_flags]

    # Smooth focus polyline
    xf, yf, zf = coords_focus[:,0], coords_focus[:,1], coords_focus[:,2]
    if Nf >= 4:
        t = np.arange(Nf); t_new = np.linspace(0, Nf-1, 200)

        sx, sy, sz = make_interp_spline(t, xf)(t_new), make_interp_spline(t, yf)(t_new), make_interp_spline(t, zf)(t_new)

        def interp_color(idx):
            i0 = int(np.floor(idx)); i1 = min(i0+1, Nf-1); w = idx - i0
            c0, c1 = np.array(colors[i0]), np.array(colors[i1]); return (1-w)*c0 + w*c1

        for i in range(len(t_new)-1):
            ax.plot(sx[i:i+2], sy[i:i+2], sz[i:i+2], color=interp_color(t_new[i]), linewidth=2.5, zorder=2)
    else:
        ax.plot(xf, yf, zf, linewidth=2.5, color=colors[-1], zorder=2)

    # Token markers (sparser to reduce clutter)
    step = max(1, Nf // 8)
    for i in range(0, Nf, step):
        ax.scatter(xf[i], yf[i], zf[i], color=colors[i], s=60, edgecolor="k", zorder=3)
    # Always label very first and last in the focus window
    ax.text(xf[0], yf[0], zf[0], "s", color="k", zorder=3)
    ax.text(xf[-1], yf[-1], zf[-1], "e", color="k", zorder=3)

    ax.set_title(title)
    ax.set_xlabel("SCG-X (phase cos)"); ax.set_ylabel("SCG-Y (phase sin)"); ax.set_zlabel("SCG-Z (curvature)")
    fig.tight_layout()
    return fig
