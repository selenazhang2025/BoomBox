"""A counter-based RNG, implemented identically in NumPy and in CUDA C.

Ordinary generators carry state, which makes a GPU kernel either serialise or
keep a per-thread state array. A counter-based generator instead *hashes* the
coordinates of the draw it wants:

    value = f(seed, player, sim)

so any thread can produce any draw with no state, no setup and no memory. That
is what lets the CUDA kernel hold nothing but the win counters, and it means the
NumPy reference and the kernel can be asked for the same draw and compared.

The mixer is SplitMix64. The NumPy code below is a line-for-line port of
``kernels/boombust.cu``; ``tests/test_backends.py`` checks they agree.
"""

from __future__ import annotations

import numpy as np

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
MIX_A = np.uint64(0xBF58476D1CE4E5B9)
MIX_B = np.uint64(0x94D049BB133111EB)
STREAM_SHIFT = np.uint64(0xD1B54A32D192ED03)
SIDE_STRIDE = np.uint64(1000003)

# 24 random bits mapped into (0, 1); never exactly 0, so log() is safe.
# Exactly 2^-24, which is representable in float32; 1/16777217 is not, and
# evaluating it in double then rounding lands 1 ULP away from what the kernel's
# float arithmetic produces. See the matching note in kernels/boombust.cu.
_U01_SCALE = np.float32(2.0) ** np.float32(-24)


def splitmix64(x: np.ndarray) -> np.ndarray:
    x = (x + GOLDEN).astype(np.uint64)
    x = ((x ^ (x >> np.uint64(30))) * MIX_A).astype(np.uint64)
    x = ((x ^ (x >> np.uint64(27))) * MIX_B).astype(np.uint64)
    return (x ^ (x >> np.uint64(31))).astype(np.uint64)


def _mix(seed: np.uint64, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return splitmix64(splitmix64(seed ^ (a * GOLDEN).astype(np.uint64)) ^ b)


def _u01(h: np.ndarray) -> np.ndarray:
    bits = (h >> np.uint64(40)) + np.uint64(1)
    return (bits.astype(np.float32) * _U01_SCALE).astype(np.float32)


TWO_PI = np.float32(2.0 * np.pi)


def draw_scores(
    params,
    sim_start: int,
    n_sims: int,
    seed: int,
    side: int = 0,
    xp=np,
):
    """Draw scores for every player over a contiguous block of simulations.

    ``params`` is (P, 3): p_active, ln_mu, ln_sigma. Returns (P, n_sims) float32.
    ``side`` separates the two rosters into independent streams, and
    ``sim_start`` makes any block of simulations reproducible on its own -- which
    is what lets the work be chunked, or spread over a CUDA grid, without
    changing the answer.

    Pass ``xp=cupy`` to run the identical arithmetic on a GPU.
    """
    params = xp.asarray(params, dtype=xp.float32)
    p_active = params[:, 0:1]
    ln_mu = params[:, 1:2]
    ln_sigma = params[:, 2:3]
    n_players = params.shape[0]

    idx = (xp.arange(n_players, dtype=xp.uint64) + np.uint64(side) * SIDE_STRIDE).reshape(-1, 1)
    sims = xp.arange(sim_start, sim_start + n_sims, dtype=xp.uint64).reshape(1, -1)

    h = _mix(np.uint64(seed), idx, sims)
    u_active = _u01(h)
    h2 = splitmix64(h ^ STREAM_SHIFT)
    u1 = _u01(h2)
    u2 = _u01(splitmix64(h2))

    # Box-Muller, using one of the two normals it produces. The kernel does the
    # same thing in the same order so the two agree draw for draw.
    # Every constant is float32: a bare -2.0 is a Python float, which promotes
    # the whole expression to float64 and rounds once at the end, where the
    # kernel rounds at every step.
    z = (xp.sqrt(np.float32(-2.0) * xp.log(u1)) * xp.cos(TWO_PI * u2)).astype(xp.float32)
    values = xp.exp(ln_mu + ln_sigma * z).astype(xp.float32)
    return xp.where(u_active < p_active, values, xp.float32(0.0)).astype(xp.float32)


def fast_draw_scores(params, n_sims: int, rng, xp=np):
    """The same distribution via NumPy's own generator: faster, not reproducible
    across backends. This is what the NumPy baseline benchmark uses, because it
    is what someone writing this in NumPy would actually reach for."""
    params = xp.asarray(params, dtype=xp.float32)
    p_active = params[:, 0:1]
    ln_mu = params[:, 1:2]
    ln_sigma = params[:, 2:3]
    shape = (params.shape[0], n_sims)
    z = rng.standard_normal(shape, dtype=xp.float32)
    values = xp.exp(ln_mu + ln_sigma * z).astype(xp.float32)
    live = rng.random(shape, dtype=xp.float32) < p_active
    return xp.where(live, values, xp.float32(0.0))
