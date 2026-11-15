"""The reference simulator, in plain NumPy. Runs anywhere.

Every other backend is checked against this one. The shape of the computation:

  * draw a score for each player in each simulation      -> (P, S)
  * sum each candidate lineup's starters                 -> (L, S)
  * compare against the opponent's total                 -> (L, S) booleans
  * count the wins                                       -> (L,)

The lineup sums are a matrix multiply: a (L, P) 0/1 membership matrix against
the (P, S) draws. That is both the clearest way to say it and the fastest thing
NumPy can do with it.

The catch is (L, S). At 1,700 lineups and a million simulations that array is
6.8 GB in float32, so the work is split into chunks of simulations sized to a
memory budget. The CUDA backend exists because it never forms the array at all.
"""

from __future__ import annotations

import time

import numpy as np

from ..rng import draw_scores, fast_draw_scores
from . import SimOutcome, validate_inputs

DEFAULT_BUDGET_BYTES = 256 * 1024 * 1024


def membership_matrix(lineups: np.ndarray, n_players: int) -> np.ndarray:
    """(L, P) 0/1 matrix: which players start in which candidate lineup."""
    n_lineups, k = lineups.shape
    out = np.zeros((n_lineups, n_players), dtype=np.float32)
    rows = np.repeat(np.arange(n_lineups), k)
    out[rows, lineups.ravel()] = 1.0
    return out


def choose_chunk(n_lineups: int, n_sims: int, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> int:
    """Largest simulation chunk whose (L, chunk) totals fit the budget."""
    per_sim = max(n_lineups * 4, 1)
    chunk = max(int(budget_bytes // per_sim), 1024)
    return int(min(chunk, n_sims))


def simulate(
    my_params: np.ndarray,
    opp_params: np.ndarray,
    lineups: np.ndarray,
    opp_lineup: np.ndarray,
    n_sims: int,
    seed: int = 0,
    *,
    chunk: int | None = None,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
    rng: str = "fast",
    **_: object,
) -> SimOutcome:
    """Count, for every candidate lineup, how many simulations it wins.

    ``rng="fast"`` uses NumPy's own generator. ``rng="counter"`` uses the
    counter-based generator the CUDA kernel implements, which makes the two
    backends comparable draw for draw.
    """
    my_params = np.ascontiguousarray(my_params, dtype=np.float32)
    opp_params = np.ascontiguousarray(opp_params, dtype=np.float32)
    lineups = np.ascontiguousarray(lineups, dtype=np.int32)
    opp_lineup = np.ascontiguousarray(opp_lineup, dtype=np.int32)
    validate_inputs(my_params, opp_params, lineups, opp_lineup, n_sims)

    n_lineups = lineups.shape[0]
    chunk = int(chunk) if chunk else choose_chunk(n_lineups, n_sims, budget_bytes)
    membership = membership_matrix(lineups, my_params.shape[0])

    wins = np.zeros(n_lineups, dtype=np.int64)
    ties = np.zeros(n_lineups, dtype=np.int64)
    generator = np.random.default_rng(seed)

    peak = (
        membership.nbytes
        + n_lineups * chunk * 4
        + (my_params.shape[0] + opp_params.shape[0]) * chunk * 4
    )

    start = time.perf_counter()
    done = 0
    while done < n_sims:
        size = min(chunk, n_sims - done)
        if rng == "counter":
            mine = draw_scores(my_params, done, size, seed, side=0)
            theirs = draw_scores(opp_params, done, size, seed, side=1)
        else:
            mine = fast_draw_scores(my_params, size, generator)
            theirs = fast_draw_scores(opp_params, size, generator)

        opp_total = theirs[opp_lineup].sum(axis=0)       # (size,)
        totals = membership @ mine                        # (L, size)
        wins += (totals > opp_total).sum(axis=1)
        ties += (totals == opp_total).sum(axis=1)
        done += size
    elapsed = time.perf_counter() - start

    return SimOutcome(
        wins=wins, ties=ties, n_sims=int(n_sims), backend="numpy", elapsed=elapsed, peak_bytes=int(peak)
    )
