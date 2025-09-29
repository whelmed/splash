import numpy as np

from splash.tiling import iter_tiles, coherence_map_for_sequence, coherence_maps_for_batch
from splash.types import EvalKnobs, CoherenceBands, MapResult, TileMeasures

def test_iter_tiles_basic_and_tail():
    tiles = iter_tiles(T=10, N=4, stride=2)
    # Should cover positions 0..9
    starts = [t.start for t in tiles]
    ends = [t.end for t in tiles]
    assert starts[0] == 0
    assert all(e - s == 4 for s, e in zip(starts, ends))
    assert tiles[-1].end == 10  # should end at last token

def test_iter_tiles_no_full_window():
    # When T < N, expect empty list OR one tile adjusted
    tiles = iter_tiles(T=3, N=4, stride=1)
    assert tiles == [] or (len(tiles) == 1 and tiles[0].N == 4)

def test_coherence_map_for_sequence_returns_expected_types():
    X = np.random.randn(12, 3)  # 12 tokens, 3D
    knobs = EvalKnobs(Ns=(4,8), stride_fraction=0.5)
    bands = CoherenceBands()
    result = coherence_map_for_sequence(X, eval_knobs=knobs, bands=bands)
    assert isinstance(result, MapResult)
    # Check keys for Ns
    for N in (4,8):
        assert N in result.tiles
        assert all(isinstance(tm, TileMeasures) for tm in result.tiles[N])
    # Global means include alignment_score
    assert "alignment_score" in result.global_means

def test_coherence_maps_for_batch_multiple_sequences():
    Xs = [np.random.randn(10, 2), np.random.randn(15, 2)]
    knobs = EvalKnobs(Ns=(4,))
    bands = CoherenceBands()
    results = coherence_maps_for_batch(Xs, eval_knobs=knobs, bands=bands)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, MapResult) for r in results)
