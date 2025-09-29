import numpy as np
from matplotlib.figure import Figure
from . import (
    DashboardThresholds,
    plot_scg_trajectory,
    plot_invariants_grid,
    plot_dislocation_heatmap,
    make_summary_table,
    render_run_dashboard,
)

def test_plot_invariants_grid_smoke():
    inv = {
        "base": {
            "entropy":   [2.5, 1.6, 1.8, 2.1],
            "curvature": [1e4, 2e4, 1.2e4, 9e3],
            "alignment": [0.95, 0.75, 0.68, 0.82],
            "energy_flux": [0.0, 1.2e6, 2.0e5, -8.1e6],
            "dislocation": [0.01, 0.06, 0.06, 0.02],
        }
    }
    fig = plot_invariants_grid(inv, DashboardThresholds())
    assert isinstance(fig, Figure)

def test_plot_scg_trajectory_smoke():
    coords = np.cumsum(np.random.randn(12, 3) * 0.1, axis=0)
    flags  = ["green"] * 4 + ["yellow"] * 4 + ["red"] * 4
    fig = plot_scg_trajectory(coords, flags)
    assert isinstance(fig, Figure)

def test_plot_dislocation_heatmap_smoke():
    mat = np.random.rand(6, 10)
    fig = plot_dislocation_heatmap(mat)
    assert isinstance(fig, Figure)

def test_make_summary_table_smoke():
    tbl = make_summary_table({"RunA": {"H_mid": 1.2, "R_mid": 0.7}})
    # Styler object
    assert hasattr(tbl, "to_html")

def test_render_run_dashboard_smoke():
    inv = {"run": {"entropy":[1,2,3], "alignment":[.9,.7,.8]}}
    figs = render_run_dashboard(inv_grid=inv)
    assert "invariants" in figs and isinstance(figs["invariants"], Figure)
