"""The custom CUDA kernel, launched through CuPy's RawKernel.

What the kernel changes, compared to the CuPy backend:

* **Memory.** The array backends must build an (L, chunk) totals array, so they
  trade speed against device memory and have to loop over chunks. The kernel
  keeps a tile of simulations in shared memory and tallies into registers, so
  device memory is O(L + P) -- a few hundred kilobytes regardless of how many
  simulations are asked for.
* **Traffic.** Draws are generated where they are used and never written to
  global memory. The only global writes are the two win counters.
* **Passes.** One launch covers every simulation and every lineup; there is no
  chunk loop and no intermediate to read back.

See ``kernels/boombust.cu``. CuPy is imported inside the functions so that
importing BoomBust on a machine without CUDA costs nothing.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import SimOutcome, validate_inputs

KERNEL_PATH = Path(__file__).resolve().parents[3] / "kernels" / "boombust.cu"

DEFAULT_TILE = 64         # simulations held in shared memory per block
DEFAULT_BLOCK = 256       # threads per block
MAX_SHARED_BYTES = 48 * 1024


def kernel_source() -> str:
    return KERNEL_PATH.read_text()


@lru_cache(maxsize=1)
def _module():
    import cupy as cp

    return cp.RawModule(code=kernel_source(), backend="nvrtc", options=("-std=c++14",))


def _kernel():
    return _module().get_function("simulate_lineups")


def shared_bytes(n_players: int, tile: int) -> int:
    """Shared memory for one block: the padded draw tile plus opponent totals."""
    return (n_players * (tile + 1) + tile) * 4


def choose_tile(n_players: int, requested: int = DEFAULT_TILE) -> int:
    """Largest tile that fits in a block's shared memory, at most ``requested``."""
    tile = int(requested)
    while tile > 1 and shared_bytes(n_players, tile) > MAX_SHARED_BYTES:
        tile //= 2
    if shared_bytes(n_players, tile) > MAX_SHARED_BYTES:
        raise ValueError(f"roster of {n_players} players does not fit in shared memory")
    return tile


def simulate(
    my_params: np.ndarray,
    opp_params: np.ndarray,
    lineups: np.ndarray,
    opp_lineup: np.ndarray,
    n_sims: int,
    seed: int = 0,
    *,
    tile: int = DEFAULT_TILE,
    block: int = DEFAULT_BLOCK,
    **_: object,
) -> SimOutcome:
    import cupy as cp

    my_params = np.ascontiguousarray(my_params, dtype=np.float32)
    opp_params = np.ascontiguousarray(opp_params, dtype=np.float32)
    lineups = np.ascontiguousarray(lineups, dtype=np.int32)
    opp_lineup = np.ascontiguousarray(opp_lineup, dtype=np.int32)
    validate_inputs(my_params, opp_params, lineups, opp_lineup, n_sims)

    n_lineups, k = lineups.shape
    n_my = my_params.shape[0]
    n_opp = opp_params.shape[0]
    tile = choose_tile(n_my, tile)

    d_my = cp.asarray(my_params)
    d_opp = cp.asarray(opp_params)
    d_lineups = cp.asarray(lineups)
    d_opp_lineup = cp.asarray(opp_lineup)
    d_wins = cp.zeros(n_lineups, dtype=cp.uint64)
    d_ties = cp.zeros(n_lineups, dtype=cp.uint64)

    blocks = int((int(n_sims) + tile - 1) // tile)
    smem = shared_bytes(n_my, tile)
    kernel = _kernel()

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    kernel(
        (blocks,),
        (int(block),),
        (
            d_my, d_opp, d_lineups, d_opp_lineup, d_wins, d_ties,
            np.int32(n_my), np.int32(n_opp), np.int32(n_lineups),
            np.int32(k), np.int32(opp_lineup.size),
            np.int64(n_sims), np.uint64(seed), np.int32(tile),
        ),
        shared_mem=smem,
    )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start

    peak = (
        d_my.nbytes + d_opp.nbytes + d_lineups.nbytes + d_opp_lineup.nbytes
        + d_wins.nbytes + d_ties.nbytes
    )
    return SimOutcome(
        wins=cp.asnumpy(d_wins).astype(np.int64),
        ties=cp.asnumpy(d_ties).astype(np.int64),
        n_sims=int(n_sims),
        backend="cuda",
        elapsed=elapsed,
        peak_bytes=int(peak),
    )
