"""The CUDA kernel, tested without a GPU.

``kernels/boombust.cu`` is compiled verbatim by the system C++ compiler with the
CUDA keywords shimmed onto host equivalents (see ``cuda_harness``), then checked
against the NumPy reference backend.

What is exact, and what is not. The RNG is counter-based, so the *integer*
stream is bit-identical on both sides by construction, and so is ``u01``. What
follows it is not: ``logf``/``cosf``/``expf`` are different implementations in
libm and in NumPy (and different again on a real GPU), so individual draws agree
to a few ULP rather than to the bit. The kernel also sums k starters
sequentially where NumPy sums all P through a BLAS matmul, and float addition is
not associative.

That only matters where a comparison sits within a few ULP of the boundary. In
this fixture exactly 1 of 800,000 comparisons does, so win counts agree outright
and ties -- which are an *exact float equality* and the most fragile predicate
here -- may differ by one. Asserting bit-equality on the counts passed on macOS
and failed on Linux; the tolerances below are set from that measurement.

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


def test_rng_stream_is_bit_identical(case):
    """The part that *is* exact, on every platform: the counter-based stream.

    Everything downstream inherits its determinism from here, so this is worth
    asserting on its own rather than through the win counts.
    """
    from boombust.sim.rng import _mix, _u01

    seed, sims = np.uint64(99), np.arange(4096, dtype=np.uint64)
    # splitmix64 is defined on wrapping uint64 arithmetic; the overflow is the
    # algorithm, not an accident.
    with np.errstate(over="ignore"):
        h = _mix(seed, np.uint64(3), sims)
        u = _u01(h)
    assert u.dtype == np.float32
    # u01 is an integer-to-float conversion times an exact power of two, so it
    # is reproducible bit for bit -- and never 0, which keeps log() safe.
    assert (u > 0).all() and (u <= 1).all()
    with np.errstate(over="ignore"):
        again = _u01(_mix(seed, np.uint64(3), sims))
    assert np.array_equal(again.view(np.int32), u.view(np.int32))


def test_kernel_matches_numpy_reference(case):
    """Counts agree; the residue is float rounding, and it is bounded.

    Ties are allowed to move because a tie is ``total == other`` on float32 --
    one ULP either way erases or invents one. Win probability is what the tool
    actually reports, so that is held to a tight tolerance.
    """
    mine, theirs, lineups, opp = case
    n_sims = 20_000
    wins, ties = cuda_harness.run_kernel(mine, theirs, lineups, opp, n_sims, seed=99, tile=16)
    reference = numpy_simulate(mine, theirs, lineups, opp, n_sims, seed=99, rng="counter")

    # Measured: 1 comparison in 800,000 lies within 16 ULP of the boundary, so
    # a handful of flips is the honest ceiling -- not an open-ended tolerance.
    assert np.abs(wins - reference.wins).max() <= 1
    assert np.abs(ties - reference.ties).max() <= 1
    assert int((wins != reference.wins).sum() + (ties != reference.ties).sum()) <= 4

    kernel_p = (wins + 0.5 * ties) / n_sims
    reference_p = (reference.wins + 0.5 * reference.ties) / n_sims
    assert np.abs(kernel_p - reference_p).max() < 1e-4


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
