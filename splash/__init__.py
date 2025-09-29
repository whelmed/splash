from .types import (
    EvalKnobs, CoherenceBands, BandLabel,
    SequenceBatch, HiddenStates, AttentionMaps, ModelOutputs,
    TileSpec, TileMeasures, MapResult, LayerCurve, EvalRun,
)
from .adapters import ModelAdapter, HFAdapter, APIAdapter
from .geometry import (
    select_layer_hidden, to_places_from_hidden,
    auto_knobs_to_scion, one_tick_measures, classify_measures,
)
from .tiling import iter_tiles, coherence_map_for_sequence, coherence_maps_for_batch
from .layers import layer_curves_from_hidden
from .robustness import (
    Perturb, RobustnessReport, synonym_swap, paraphrase, distractor_clause, perturb_and_compare
)
from .orchestrate import evaluate_prompts, evaluate_hidden
from .viz import plot_coherence_map, plot_layer_curves, plot_embedding_carpet
from .reports import to_json, from_json

__all__ = [
    # types
    "EvalKnobs", "CoherenceBands", "BandLabel",
    "SequenceBatch", "HiddenStates", "AttentionMaps", "ModelOutputs",
    "TileSpec", "TileMeasures", "MapResult", "LayerCurve", "EvalRun",
    # adapters
    "ModelAdapter", "HFAdapter", "APIAdapter",
    # geometry
    "select_layer_hidden", "to_places_from_hidden",
    "auto_knobs_to_scion", "one_tick_measures", "classify_measures",
    # tiling
    "iter_tiles", "coherence_map_for_sequence", "coherence_maps_for_batch",
    # layers
    "layer_curves_from_hidden",
    # robustness
    "Perturb", "RobustnessReport", "synonym_swap", "paraphrase", "distractor_clause", "perturb_and_compare",
    # orchestrate
    "evaluate_prompts", "evaluate_hidden",
    # viz
    "plot_coherence_map", "plot_layer_curves", "plot_embedding_carpet",
    # reports
    "to_json", "from_json",
]


# control-plane
from .control import (
    InvariantFrame, Decision, ActionTrace, Episode, PolicyConfig,
    measure_frame, RulePolicy, PolicyPack, Actuator, ActuatorResult,
    Verifier, VerifyResult, Controller
)
__all__ += [
    "InvariantFrame", "Decision", "ActionTrace", "Episode", "PolicyConfig",
    "measure_frame", "RulePolicy", "PolicyPack", "Actuator", "ActuatorResult",
    "Verifier", "VerifyResult", "Controller",
]

# dashboard
from .dashboard import (
    DashboardThresholds,
    plot_scg_trajectory,
    plot_invariants_grid,
    plot_dislocation_heatmap,
    make_summary_table,
    render_run_dashboard,
    inv_grid_from_layer_curve,
    inv_grid_from_runs,
    dislocation_matrix_from_tiles,
)

__all__ += [
    "DashboardThresholds",
    "plot_scg_trajectory",
    "plot_invariants_grid",
    "plot_dislocation_heatmap",
    "make_summary_table",
    "render_run_dashboard",
    "inv_grid_from_layer_curve",
    "inv_grid_from_runs",
    "dislocation_matrix_from_tiles",
]
