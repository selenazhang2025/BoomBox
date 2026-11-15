"""The CUDA kernel, tested without a GPU.

``kernels/boombust.cu`` is compiled verbatim by the system C++ compiler with the
CUDA keywords shimmed onto host equivalents (see ``cuda_harness``), then checked
against the NumPy reference backend. Because both sides use the same
counter-based RNG, agreement is expected to be *exact*, not merely statistical.

Caveat worth stating plainly: this exercises the kernel's indexing, shared
memory layout, RNG and tallying, and -- with the multi-threaded launch -- its
barrier and atomics. It says nothing about warp behaviour, occupancy or real
device memory. Those need a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from boombust.sim.backends.numpy_backend import simulate as numpy_simulate

cuda_harness = pytest.importorskip("cuda_harness")

pytestmark = pytest.mark.skipif(
    cuda_harness.find_compiler() is None, reason="no C++ compiler available"
)


def make_params(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.uniform(0.6, 1.0, n), rng.uniform(1.0, 2.8, n), rng.uniform(0.3, 0.9, n)], axis=1
    ).astype(np.float32)


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(3)
    n_players, n_lineups, k = 12, 40, 6
    mine = make_params(n_players, 1)
    theirs = make_params(n_players, 2)
    lineups = np.stack(
        [rng.choice(n_players, k, replace=False) for _ in range(n_lineups)]
    ).astype(np.int32)
    opp_lineup = rng.choice(n_players, k, replace=False).astype(np.int32)
    return mine, theirs, lineups, opp_lineup


def test_kernel_matches_numpy_reference_exactly(case):
    mine, theirs, lineups, opp = case
    n_sims = 20_000
    wins, ties = cuda_harness.run_kernel(mine, theirs, lineups, opp, n_sims, seed=99, tile=16)
    reference = numpy_simulate(mine, theirs, lineups, opp, n_sims, seed=99, rng="counter")
    np.testing.assert_array_equal(wins, reference.wins)
    np.testing.assert_array_equal(ties, reference.ties)


def test_results_are_independent_of_thread_count(case):
    """Proves the barrier and the atomics, not just the arithmetic."""
    mine, theirs, lineups, opp = case
    serial = cuda_harness.run_kernel(mine, theirs, lineups, opp, 8_000, seed=5, tile=16, threads=1)
    parallel = cuda_harness.run_kernel(mine, theirs, lineups, opp, 8_000, seed=5, tile=16, threads=64)
    np.testing.assert_array_equal(serial[0], parallel[0])
    np.testing.assert_array_equal(serial[1], parallel[1])


@pytest.mark.parametrize("tile", [4, 16, 64])
def test_results_are_independent_of_tile_size(case, tile):
    """Tiling is a blocking strategy, so it must not change the answer."""
    mine, theirs, lineups, opp = case
    base = cuda_harness.run_kernel(mine, theirs, lineups, opp, 6_000, seed=11, tile=8)
    other = cuda_harness.run_kernel(mine, theirs, lineups, opp, 6_000, seed=11, tile=tile)
    np.testing.assert_array_equal(base[0], other[0])
    np.testing.assert_array_equal(base[1], other[1])


def test_handles_a_simulation_count_that_does_not_divide_the_tile(case):
    """The last block is partial; it must not count phantom simulations."""
    mine, theirs, lineups, opp = case
    n_sims = 1_003  # deliberately coprime with the tile
    wins, ties = cuda_harness.run_kernel(mine, theirs, lineups, opp, n_sims, seed=2, tile=16)
    assert (wins + ties <= n_sims).all()
    reference = numpy_simulate(mine, theirs, lineups, opp, n_sims, seed=2, rng="counter")
    np.testing.assert_array_equal(wins, reference.wins)


def test_a_dominated_lineup_never_wins_more_than_a_better_one():
    """Sanity check with hand-made params: a strictly better lineup must win more."""
    mine = np.array(
        [[1.0, np.log(20.0), 0.2], [1.0, np.log(5.0), 0.2], [1.0, np.log(12.0), 0.2]],
        dtype=np.float32,
    )
    theirs = np.array([[1.0, np.log(12.0), 0.3]], dtype=np.float32)
    lineups = np.array([[0], [1], [2]], dtype=np.int32)
    opp = np.array([0], dtype=np.int32)
    wins, _ = cuda_harness.run_kernel(mine, theirs, lineups, opp, 20_000, seed=1, tile=32)
    assert wins[0] > wins[2] > wins[1]


def test_certain_zero_player_never_scores():
    """p_active = 0 must mean a hard zero, which is how byes and OUT are modelled."""
    mine = np.array([[0.0, np.log(20.0), 0.5]], dtype=np.float32)
    theirs = np.array([[1.0, np.log(1.0), 0.1]], dtype=np.float32)
    lineups = np.array([[0]], dtype=np.int32)
    opp = np.array([0], dtype=np.int32)
    wins, ties = cuda_harness.run_kernel(mine, theirs, lineups, opp, 5_000, seed=4, tile=16)
    assert wins[0] == 0  # a zero can never beat a strictly positive opponent
