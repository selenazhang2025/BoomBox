"""The score model: recent form, matchup adjustment, and the zero spike."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boombust.model.projections import (
    Priors,
    fit_player,
    fit_priors,
    params_matrix,
    project_roster,
)
from boombust.sim.rng import draw_scores


def games(points: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"points": points})


def flat_priors(mean: float = 10.0, cv: float = 0.6, dud: float = 0.15) -> Priors:
    p = Priors()
    for position in ("QB", "RB", "WR", "TE", "K", "D/ST"):
        p.position_mean[position] = mean
        p.position_cv[position] = cv
        p.position_dud[position] = dud
    return p


# --- the distribution it produces ----------------------------------------
def test_the_fitted_params_reproduce_the_reported_mean_and_sd():
    """The reported moments must be the moments you actually get when you draw."""
    projection = fit_player(
        "p", "Player", "WR", games([18.0, 4.0, 22.0, 11.0, 0.0, 15.0, 9.0, 26.0]),
        flat_priors(), team="KC", opponent="DEN",
    )
    draws = draw_scores(params_matrix([projection]), 0, 400_000, seed=5)
    assert draws.mean() == pytest.approx(projection.mean, rel=0.02)
    assert draws.std() == pytest.approx(projection.sd, rel=0.05)


def test_the_distribution_is_right_skewed_with_an_atom_at_zero():
    projection = fit_player(
        "p", "Player", "RB", games([12.0, 0.0, 3.0, 25.0, 8.0, 0.0, 14.0]), flat_priors(),
        opponent="DEN",
    )
    draws = draw_scores(params_matrix([projection]), 0, 200_000, seed=6)[0]
    assert (draws == 0).mean() == pytest.approx(1 - projection.p_active, abs=0.01)
    live = draws[draws > 0]
    assert np.mean(live) > np.median(live), "a right-skewed component should have mean above median"


def test_quantiles_bracket_the_mean_and_respect_the_zero_spike():
    projection = fit_player("p", "P", "WR", games([10.0] * 6), flat_priors(), opponent="DEN")
    assert projection.quantile(0.0) == 0.0
    assert projection.quantile(0.5) > 0
    assert projection.quantile(0.9) > projection.quantile(0.1)
    heavy_dud = fit_player(
        "p", "P", "WR", games([0.0, 0.0, 0.0, 20.0]), flat_priors(), opponent="DEN"
    )
    assert heavy_dud.quantile(0.2) == 0.0  # more than 20% of the mass sits at zero


# --- recency and shrinkage ------------------------------------------------
def test_recent_games_count_for_more_than_old_ones():
    rising = fit_player(
        "p", "P", "WR", games([2.0, 2.0, 2.0, 2.0, 20.0, 20.0]), flat_priors(), opponent="DEN"
    )
    falling = fit_player(
        "p", "P", "WR", games([20.0, 20.0, 2.0, 2.0, 2.0, 2.0]), flat_priors(), opponent="DEN"
    )
    assert rising.mean > falling.mean


def test_a_player_with_no_history_falls_back_to_the_positional_prior():
    projection = fit_player(
        "p", "Rookie", "WR", games([]), flat_priors(mean=9.0), opponent="DEN"
    )
    assert projection.n_games == 0
    assert "prior" in projection.note
    assert projection.mean == pytest.approx(9.0 * (1 - 0.15), rel=0.05)


def test_a_short_history_is_pulled_toward_the_prior():
    """One huge game should not make someone a 40-point-a-week player."""
    prior = flat_priors(mean=10.0)
    one_game = fit_player("p", "P", "WR", games([40.0]), prior, opponent="DEN")
    many_games = fit_player("p", "P", "WR", games([40.0] * 10), prior, opponent="DEN")
    assert one_game.mean < many_games.mean
    assert one_game.mean < 40.0


# --- availability ---------------------------------------------------------
@pytest.mark.parametrize("status", ["OUT", "INJURY_RESERVE", "SUSPENSION"])
def test_ruled_out_players_score_exactly_zero(status):
    projection = fit_player("p", "P", "RB", games([15.0] * 6), flat_priors(), opponent="DEN", status=status)
    assert projection.p_active == 0.0
    assert projection.mean == 0.0 and projection.sd == 0.0
    assert draw_scores(params_matrix([projection]), 0, 1_000, seed=1).max() == 0.0


def test_a_bye_week_is_a_zero_even_for_a_healthy_player():
    projection = fit_player("p", "P", "RB", games([15.0] * 6), flat_priors(), opponent=None)
    assert projection.p_active == 0.0
    assert projection.note == "bye week"


def test_questionable_lowers_the_mean_without_touching_the_shape():
    healthy = fit_player("p", "P", "RB", games([15.0] * 6), flat_priors(), opponent="DEN")
    doubtful = fit_player(
        "p", "P", "RB", games([15.0] * 6), flat_priors(), opponent="DEN", status="QUESTIONABLE"
    )
    assert doubtful.mean < healthy.mean
    assert doubtful.ln_mu == pytest.approx(healthy.ln_mu)  # same distribution when he plays
    assert doubtful.p_active < healthy.p_active


# --- matchup adjustment ---------------------------------------------------
def test_a_generous_defence_raises_the_projection_and_a_stingy_one_lowers_it():
    priors = flat_priors()
    priors.defense[("SOFT", "WR")] = 1.25
    priors.defense[("HARD", "WR")] = 0.80
    soft = fit_player("p", "P", "WR", games([12.0] * 8), priors, opponent="SOFT")
    hard = fit_player("p", "P", "WR", games([12.0] * 8), priors, opponent="HARD")
    neutral = fit_player("p", "P", "WR", games([12.0] * 8), priors, opponent="OTHER")
    assert soft.mean > neutral.mean > hard.mean
    assert soft.def_factor == 1.25 and hard.def_factor == 0.80
    # Scaling the mean keeps the coefficient of variation, not the raw spread.
    assert soft.sd / soft.mean == pytest.approx(neutral.sd / neutral.mean, rel=1e-3)


def test_defence_factors_are_shrunk_and_clamped():
    """A defence with one freak game must not get an extreme factor."""
    history = pd.DataFrame(
        {
            "key": ["a", "b", "c", "d"],
            "position": ["WR"] * 4,
            "season": [2024] * 4,
            "week": [1, 1, 2, 2],
            "team": ["X", "Y", "X", "Y"],
            "opponent": ["SOFT", "HARD", "OTHER", "OTHER"],
            "points": [60.0, 1.0, 10.0, 10.0],
        }
    )
    priors = fit_priors(history)
    for (_team, _pos), factor in priors.defense.items():
        assert 0.75 <= factor <= 1.30


def test_priors_are_fitted_only_from_fantasy_relevant_games():
    history = pd.DataFrame(
        {
            "key": ["a", "b", "c"],
            "position": ["WR"] * 3,
            "season": [2024] * 3,
            "week": [1, 1, 1],
            "team": ["X", "Y", "Z"],
            "opponent": ["A", "B", "C"],
            "points": [20.0, 10.0, 0.0],
        }
    )
    priors = fit_priors(history)
    assert priors.position_mean["WR"] == pytest.approx(15.0)  # the zero is excluded
    assert priors.position_dud["WR"] == pytest.approx(1 / 3)  # but counted as a dud


# --- packing --------------------------------------------------------------
def test_params_matrix_is_the_shape_the_backends_expect():
    projections = [
        fit_player(f"p{i}", "P", "WR", games([10.0] * 5), flat_priors(), opponent="DEN")
        for i in range(4)
    ]
    matrix = params_matrix(projections)
    assert matrix.shape == (4, 3)
    assert matrix.dtype == np.float32
    assert (matrix[:, 0] >= 0).all() and (matrix[:, 0] <= 1).all()


def test_project_roster_handles_missing_history_and_fullback_positions():
    history = pd.DataFrame(columns=["key", "position", "season", "week", "team", "opponent", "points"])
    roster = [
        {"player_id": "x", "name": "A Back", "position": "FB", "pro_team": "KC", "injury_status": "ACTIVE"},
        {"player_id": "y", "name": "A Kicker", "position": "K", "pro_team": "ZZZ", "injury_status": "ACTIVE"},
    ]
    out = project_roster(history, Priors(), roster, {"KC": "DEN"})
    assert out[0].position == "RB"           # fullbacks are modelled as running backs
    assert out[1].p_active == 0.0            # ZZZ has no game this week: a bye
    assert out[1].note == "bye week"
