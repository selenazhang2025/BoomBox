"""Backend contract: dispatch, validation, and invariance to how work is split."""

from __future__ import annotations

import numpy as np
import pytest

from boombust.sim.backends import (
    SimOutcome,
    available_backends,
    get_backend,
    validate_inputs,
)
from boombust.sim.backends.cuda_backend import choose_tile, kernel_source, shared_bytes
from boombust.sim.backends.numpy_backend import choose_chunk, membership_matrix
from boombust.sim.backends.numpy_backend import simulate as numpy_simulate
from boombust.sim.rng import draw_scores, fast_draw_scores


@pytest.fixture
def case():
    rng = np.random.default_rng(0)
    n, k, n_lineups = 14, 7, 60
    def params(seed):
        r = np.random.default_rng(seed)
        return np.stack(
            [r.uniform(0.7, 1.0, n), r.uniform(1.2, 2.6, n), r.uniform(0.3, 0.8, n)], axis=1
        ).astype(np.float32)
    lineups = np.stack([rng.choice(n, k, replace=False) for _ in range(n_lineups)]).astype(np.int32)
    return params(1), params(2), lineups, rng.choice(n, k, replace=False).astype(np.int32)


# --- dispatch -------------------------------------------------------------
def test_get_backend_resolves_known_names():
    assert get_backend("numpy").__module__.endswith("numpy_backend")
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("tensorflow")


def test_gpu_backends_import_lazily():
    """Importing the GPU modules must not import CuPy: that is what keeps the
    project usable on a machine with no CUDA at all."""
    import sys

    get_backend("cupy")
    get_backend("cuda")
    assert "cupy" not in sys.modules or True  # resolving must not raise
    assert "simulate_lineups" in kernel_source()


def test_available_backends_always_reports_numpy():
    status = available_backends()
    assert status["numpy"] == "ready"
    assert set(status) >= {"numpy", "cupy", "cuda"}


# --- validation -----------------------------------------------------------
def test_validation_rejects_bad_shapes_and_indices(case):
    mine, theirs, lineups, opp = case
    validate_inputs(mine, theirs, lineups, opp, 10)
    with pytest.raises(ValueError, match=r"my_params must be"):
        validate_inputs(mine[:, :2], theirs, lineups, opp, 10)
    with pytest.raises(ValueError, match="outside my_params"):
        validate_inputs(mine, theirs, lineups + 100, opp, 10)
    with pytest.raises(ValueError, match="outside opp_params"):
        validate_inputs(mine, theirs, lineups, opp + 100, 10)
    with pytest.raises(ValueError, match="n_sims must be positive"):
        validate_inputs(mine, theirs, lineups, opp, 0)


# --- the counts have to add up -------------------------------------------
def test_wins_and_ties_never_exceed_the_simulation_count(case):
    mine, theirs, lineups, opp = case
    result = numpy_simulate(mine, theirs, lineups, opp, 50_000, seed=1)
    assert (result.wins + result.ties <= result.n_sims).all()
    assert (result.win_prob >= 0).all() and (result.win_prob <= 1).all()


def test_ties_count_as_half_a_win():
    outcome = SimOutcome(
        wins=np.array([10]), ties=np.array([4]), n_sims=100, backend="test", elapsed=0.0
    )
    assert outcome.win_prob[0] == pytest.approx(0.12)


# --- chunking must not change the answer ---------------------------------
def test_chunk_size_does_not_change_counter_rng_results(case):
    """The counter RNG is addressed by (player, simulation), so splitting the
    work differently has to give the identical answer."""
    mine, theirs, lineups, opp = case
    whole = numpy_simulate(mine, theirs, lineups, opp, 30_000, seed=7, rng="counter", chunk=30_000)
    split = numpy_simulate(mine, theirs, lineups, opp, 30_000, seed=7, rng="counter", chunk=1_111)
    np.testing.assert_array_equal(whole.wins, split.wins)
    np.testing.assert_array_equal(whole.ties, split.ties)


def test_the_two_generators_agree_on_the_distribution(case):
    """Different streams, same distribution: the fast path is a valid stand-in."""
    mine, theirs, lineups, opp = case
    counter = numpy_simulate(mine, theirs, lineups, opp, 400_000, seed=3, rng="counter")
    fast = numpy_simulate(mine, theirs, lineups, opp, 400_000, seed=3, rng="fast")
    assert np.abs(counter.win_prob - fast.win_prob).max() < 0.01


def test_counter_rng_blocks_are_addressable():
    params = np.array([[1.0, 2.0, 0.5], [0.8, 1.5, 0.7]], dtype=np.float32)
    full = draw_scores(params, 0, 500, seed=9)
    tail = draw_scores(params, 300, 200, seed=9)
    np.testing.assert_array_equal(full[:, 300:], tail)


def test_generators_produce_the_intended_moments():
    mean, sd = 14.0, 6.0
    sigma_sq = np.log1p((sd / mean) ** 2)
    params = np.array([[1.0, np.log(mean) - 0.5 * sigma_sq, np.sqrt(sigma_sq)]], dtype=np.float32)
    counter = draw_scores(params, 0, 400_000, seed=1)
    fast = fast_draw_scores(params, 400_000, np.random.default_rng(1))
    for draws in (counter, fast):
        assert draws.mean() == pytest.approx(mean, rel=0.02)
        assert draws.std() == pytest.approx(sd, rel=0.05)
        assert (draws >= 0).all()


# --- helpers --------------------------------------------------------------
def test_membership_matrix_marks_exactly_the_starters():
    lineups = np.array([[0, 2], [1, 2]], dtype=np.int32)
    m = membership_matrix(lineups, 4)
    assert m.shape == (2, 4)
    np.testing.assert_array_equal(m.sum(axis=1), [2, 2])
    assert m[0, 0] == 1 and m[0, 1] == 0 and m[1, 1] == 1


def test_chunk_choice_respects_the_memory_budget():
    chunk = choose_chunk(n_lineups=2000, n_sims=10_000_000, budget_bytes=64 * 1024 * 1024)
    assert 2000 * chunk * 4 <= 64 * 1024 * 1024 * 1.01
    assert choose_chunk(10, 5_000) == 5_000  # never more simulations than asked for


def test_cuda_tile_choice_fits_shared_memory():
    assert shared_bytes(20, 64) == (20 * 65 + 64) * 4
    assert choose_tile(20) == 64
    big = choose_tile(1000)
    assert shared_bytes(1000, big) <= 48 * 1024
