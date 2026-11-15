"""The claim the whole project rests on.

Ranking lineups by win probability is only worth doing if it can disagree with
ranking them by projected points. It can, and the direction is not arbitrary:

* a favourite should prefer the steadier lineup,
* a heavy underdog should prefer the volatile one,

because you win by clearing the opponent's total, not by averaging well. These
tests hold the mean fixed and vary only the spread, so the effect cannot be
confused with one lineup simply being better.
"""

from __future__ import annotations

import numpy as np
import pytest

from boombust.sim.backends.numpy_backend import simulate

N_SIMS = 2_000_000


def lognormal_params(mean: float, sd: float, p_active: float = 1.0) -> list[float]:
    """Moment-match the mixture params the simulator consumes."""
    sigma_sq = np.log1p((sd * sd) / (mean * mean))
    return [p_active, float(np.log(mean) - 0.5 * sigma_sq), float(np.sqrt(sigma_sq))]


@pytest.fixture(scope="module")
def roster():
    """Eight identical starters, plus a safe and a risky candidate of equal mean."""
    core = [lognormal_params(12.0, 5.0) for _ in range(8)]
    safe = lognormal_params(10.0, 3.0)
    risky = lognormal_params(10.0, 11.0)
    mine = np.array(core + [safe, risky], dtype=np.float32)
    lineups = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 9]], dtype=np.int32
    )
    return mine, lineups


def opponent(total_mean: float) -> tuple[np.ndarray, np.ndarray]:
    params = np.array(
        [lognormal_params(total_mean / 9.0, 5.0) for _ in range(9)], dtype=np.float32
    )
    return params, np.arange(9, dtype=np.int32)


def win_probs(roster, total_mean: float, seed: int = 42) -> tuple[float, float]:
    mine, lineups = roster
    theirs, opp_lineup = opponent(total_mean)
    result = simulate(mine, theirs, lineups, opp_lineup, N_SIMS, seed=seed, rng="counter")
    safe, risky = result.win_prob
    return float(safe), float(risky)


def test_both_lineups_project_the_same_points(roster):
    """Guards the premise: if the means differed, the rest proves nothing."""
    mine, lineups = roster
    p_active, ln_mu, ln_sigma = mine[:, 0], mine[:, 1], mine[:, 2]
    means = p_active * np.exp(ln_mu + 0.5 * ln_sigma ** 2)
    assert means[8] == pytest.approx(means[9], rel=1e-3)
    assert means[lineups[0]].sum() == pytest.approx(means[lineups[1]].sum(), rel=1e-3)


def test_favourite_prefers_the_steady_lineup(roster):
    safe, risky = win_probs(roster, total_mean=85.0)
    assert safe > 0.5, "the setup should make this side a clear favourite"
    assert safe > risky + 0.01, f"favourite should prefer safe: {safe:.4f} vs {risky:.4f}"


def test_heavy_underdog_prefers_the_volatile_lineup(roster):
    safe, risky = win_probs(roster, total_mean=130.0)
    assert safe < 0.25, "the setup should make this side a heavy underdog"
    assert risky > safe + 0.01, f"underdog should prefer risky: {risky:.4f} vs {safe:.4f}"


def test_the_preference_flips_exactly_once_as_the_matchup_hardens(roster):
    """A single crossover, not noise: the edge to the risky lineup is monotone
    in how far behind you are."""
    edges = []
    for total_mean in (70.0, 85.0, 95.0, 106.0, 115.0, 130.0, 150.0):
        safe, risky = win_probs(roster, total_mean)
        edges.append(risky - safe)
    # Negative while favoured, positive once a clear underdog, and the sign
    # changes exactly once across the sweep.
    assert edges[0] < 0 and edges[-1] > 0
    sign_changes = sum(1 for a, b in zip(edges, edges[1:]) if (a < 0) != (b < 0))
    assert sign_changes == 1, f"expected one crossover, got {sign_changes}: {edges}"


def test_projected_points_cannot_express_any_of_this(roster):
    """The control: a points-only ranking is flat across both lineups, so it has
    no way to pick differently as the matchup changes."""
    mine, lineups = roster
    means = mine[:, 0] * np.exp(mine[:, 1] + 0.5 * mine[:, 2] ** 2)
    totals = means[lineups].sum(axis=1)
    assert abs(totals[0] - totals[1]) < 0.05
