"""The same algorithm as the NumPy backend, moved onto the GPU with CuPy.

Deliberately a near-line-for-line port: ``cp`` for ``np``, and the draws and the
(L, chunk) totals live in device memory. It is the honest "array library on a
GPU" step, and it is much faster than NumPy -- but it still has to build the
(L, chunk) totals array, so it is still bounded by device memory rather than by
arithmetic. That bound is what the custom kernel removes.

CuPy is imported inside the functions, never at module scope, so importing
BoomBust on a machine without CUDA costs nothing.
"""

from __future__ import annotations

import time

import numpy as np

from ..rng import draw_scores, fast_draw_scores
from . import SimOutcome, validate_inputs

DEFAULT_BUDGET_BYTES = 1024 * 1024 * 1024


def choose_chunk(n_lineups: int, n_sims: int, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> int:
    per_sim = max(n_lineups * 4, 1)
    return int(min(max(int(budget_bytes // per_sim), 4096), n_sims))


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
    import cupy as cp

    my_params = np.ascontiguousarray(my_params, dtype=np.float32)
    opp_params = np.ascontiguousarray(opp_params, dtype=np.float32)
    lineups = np.ascontiguousarray(lineups, dtype=np.int32)
    opp_lineup = np.ascontiguousarray(opp_lineup, dtype=np.int32)
    validate_inputs(my_params, opp_params, lineups, opp_lineup, n_sims)

    n_lineups, k = lineups.shape
    chunk = int(chunk) if chunk else choose_chunk(n_lineups, n_sims, budget_bytes)

    d_my = cp.asarray(my_params)
    d_opp = cp.asarray(opp_params)
    d_lineups = cp.asarray(lineups)
    d_opp_lineup = cp.asarray(opp_lineup)

    # (L, P) 0/1 membership matrix, so a lineup's total is one matmul.
    membership = cp.zeros((n_lineups, my_params.shape[0]), dtype=cp.float32)
    rows = cp.repeat(cp.arange(n_lineups), k)
    membership[rows, d_lineups.ravel()] = 1.0

    wins = cp.zeros(n_lineups, dtype=cp.int64)
    ties = cp.zeros(n_lineups, dtype=cp.int64)
    generator = cp.random.default_rng(seed)

    peak = (
        membership.nbytes
        + n_lineups * chunk * 4
        + (my_params.shape[0] + opp_params.shape[0]) * chunk * 4
    )

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    done = 0
    while done < n_sims:
        size = min(chunk, n_sims - done)
        if rng == "counter":
            mine = draw_scores(d_my, done, size, seed, side=0, xp=cp)
            theirs = draw_scores(d_opp, done, size, seed, side=1, xp=cp)
        else:
            mine = fast_draw_scores(d_my, size, generator, xp=cp)
            theirs = fast_draw_scores(d_opp, size, generator, xp=cp)

        opp_total = theirs[d_opp_lineup].sum(axis=0)
        totals = membership @ mine
        wins += (totals > opp_total).sum(axis=1)
        ties += (totals == opp_total).sum(axis=1)
        done += size
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start

    return SimOutcome(
        wins=cp.asnumpy(wins),
        ties=cp.asnumpy(ties),
        n_sims=int(n_sims),
        backend="cupy",
        elapsed=elapsed,
        peak_bytes=int(peak),
    )
