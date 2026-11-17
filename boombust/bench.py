"""Benchmark the backends against each other on the same workload.

Wall-clock time alone is not comparable when the backends can run different
sized problems, so the headline number is **throughput**: lineup-simulations
per second, where one lineup-simulation is "sum one candidate lineup's starters
for one simulated week and compare it to the opponent". A week with 1,700
candidate lineups and a million simulations is 1.7 billion of them.

Peak memory is reported alongside, because it is the real difference between
the array backends and the custom kernel: the array backends must build an
(L, chunk) totals array and are bounded by memory, while the kernel tallies in
registers and is bounded only by arithmetic.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .sim.backends import available_backends, get_backend
from .sim.rng import draw_scores


@dataclass
class BenchRow:
    backend: str
    n_sims: int
    n_lineups: int
    n_starters: int
    seconds: float
    lineup_sims_per_sec: float
    peak_mb: float
    note: str = ""
    # Peak memory is only comparable across backends if every row is quoted at
    # the *same* workload. The pure-Python baseline runs a small slice of the
    # week (it would take a quarter of an hour otherwise), so its raw ``peak_mb``
    # is for that slice and reads as far leaner than it is. ``peak_mb_at_full``
    # restates it at ``n_sims_full`` -- the number the other rows actually ran --
    # and ``extrapolated`` marks the row whose rate came from the shorter run.
    peak_mb_at_full: float = float("nan")
    n_sims_full: int = 0
    extrapolated: bool = False


def synthetic_workload(
    n_players: int = 17, n_lineups: int = 1700, n_starters: int = 9, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A workload the size of a real ESPN week: one roster, every legal lineup."""
    rng = np.random.default_rng(seed)

    def params(n: int, offset: int) -> np.ndarray:
        r = np.random.default_rng(seed + offset)
        return np.stack(
            [r.uniform(0.7, 1.0, n), r.uniform(1.2, 2.8, n), r.uniform(0.3, 0.8, n)], axis=1
        ).astype(np.float32)

    lineups = np.stack(
        [rng.choice(n_players, n_starters, replace=False) for _ in range(n_lineups)]
    ).astype(np.int32)
    opp_lineup = rng.choice(n_players, n_starters, replace=False).astype(np.int32)
    return params(n_players, 1), params(n_players, 2), lineups, opp_lineup


# --- the deliberately slow baseline --------------------------------------
def pure_python_simulate(
    my_params: np.ndarray,
    opp_params: np.ndarray,
    lineups: np.ndarray,
    opp_lineup: np.ndarray,
    n_sims: int,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """The same algorithm written the obvious way: nested Python loops.

    This is the thing everything else is being compared against. It is far too
    slow to run a real week, so the benchmark runs it on a few thousand
    simulations and compares *rates*.
    """
    draws_mine = draw_scores(my_params, 0, n_sims, seed, side=0)
    draws_theirs = draw_scores(opp_params, 0, n_sims, seed, side=1)
    mine = draws_mine.tolist()
    theirs = draws_theirs.tolist()
    opp_idx = opp_lineup.tolist()
    rows = lineups.tolist()
    wins = [0] * len(rows)

    start = time.perf_counter()
    for sim in range(n_sims):
        opp_total = 0.0
        for q in opp_idx:
            opp_total += theirs[q][sim]
        for li, row in enumerate(rows):
            total = 0.0
            for p in row:
                total += mine[p][sim]
            if total > opp_total:
                wins[li] += 1
    elapsed = time.perf_counter() - start
    return np.array(wins, dtype=np.int64), elapsed


# --- the benchmark --------------------------------------------------------
def run(
    n_sims: int = 1_000_000,
    *,
    backends: tuple[str, ...] | None = None,
    n_lineups: int = 1700,
    n_players: int = 17,
    n_starters: int = 9,
    python_sims: int = 4_000,
    seed: int = 0,
    include_python: bool = True,
) -> list[BenchRow]:
    mine, theirs, lineups, opp = synthetic_workload(n_players, n_lineups, n_starters, seed)
    status = available_backends()
    wanted = backends or ("numpy", "cupy", "cuda")
    rows: list[BenchRow] = []

    if include_python:
        _, elapsed = pure_python_simulate(mine, theirs, lineups, opp, python_sims, seed)
        work = float(n_lineups) * python_sims
        python_peak = ((mine.shape[0] + theirs.shape[0]) * python_sims * 4) / 1e6
        rows.append(
            BenchRow(
                backend="python",
                n_sims=python_sims,
                n_lineups=n_lineups,
                n_starters=n_starters,
                seconds=elapsed,
                lineup_sims_per_sec=work / elapsed if elapsed else float("nan"),
                # The draw buffers, which is the honest floor: the nested-loop
                # version then copies them into Python lists, several times larger.
                peak_mb=python_peak,
                peak_mb_at_full=python_peak * (n_sims / python_sims),
                n_sims_full=n_sims,
                extrapolated=True,
                note=(
                    f"nested Python loops, measured on {python_sims:,} simulations "
                    f"and quoted as a rate; memory restated at {n_sims:,}"
                ),
            )
        )

    for name in wanted:
        if not status.get(name, "").startswith("ready"):
            rows.append(
                BenchRow(
                    backend=name, n_sims=0, n_lineups=n_lineups, n_starters=n_starters,
                    seconds=float("nan"), lineup_sims_per_sec=float("nan"), peak_mb=float("nan"),
                    note=status.get(name, "not available"),
                )
            )
            continue
        simulate = get_backend(name)
        # One short warm-up: JIT compilation and allocator setup are not what
        # we are trying to measure.
        simulate(mine, theirs, lineups, opp, min(20_000, n_sims), seed)
        outcome = simulate(mine, theirs, lineups, opp, n_sims, seed)
        work = float(n_lineups) * n_sims
        rows.append(
            BenchRow(
                backend=name,
                n_sims=n_sims,
                n_lineups=n_lineups,
                n_starters=n_starters,
                seconds=outcome.elapsed,
                lineup_sims_per_sec=work / outcome.elapsed if outcome.elapsed else float("nan"),
                peak_mb=outcome.peak_bytes / 1e6,
                peak_mb_at_full=outcome.peak_bytes / 1e6,
                n_sims_full=n_sims,
                extrapolated=False,
                note=status.get(name, ""),
            )
        )
    return rows


def host_description() -> str:
    return f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"


def device_description() -> str:
    """The CUDA device's name, or "" when there is no GPU to name.

    A benchmark that reports 22 billion anything is meaningless without the card
    it ran on, and the card is the one fact a reader cannot recover later. Asking
    CuPy costs nothing and keeps the saved run self-documenting.
    """
    try:
        import cupy  # noqa: PLC0415 -- lazy by design: CuPy will not install on macOS

        name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return ""


def format_table(rows: list[BenchRow]) -> str:
    baseline = next(
        (r.lineup_sims_per_sec for r in rows if r.backend == "python" and r.lineup_sims_per_sec == r.lineup_sims_per_sec),
        None,
    )
    head = f"{'backend':<9} {'sims':>11} {'seconds':>9} {'lineup-sims/s':>15} {'speedup':>9} {'peak MB':>9}"
    lines = [head, "-" * len(head)]
    any_extrapolated = False
    for row in rows:
        if row.seconds != row.seconds:  # NaN: backend unavailable
            lines.append(f"{row.backend:<9} {'-':>11} {'-':>9} {'-':>15} {'-':>9} {'-':>9}   {row.note}")
            continue
        speedup = (row.lineup_sims_per_sec / baseline) if baseline else float("nan")
        # Quote memory at the common workload so the column compares like with like.
        peak = row.peak_mb_at_full if row.peak_mb_at_full == row.peak_mb_at_full else row.peak_mb
        mark = "+" if row.extrapolated else " "
        any_extrapolated |= row.extrapolated
        lines.append(
            f"{row.backend:<9} {row.n_sims:>11,} {row.seconds:>9.2f} "
            f"{row.lineup_sims_per_sec:>15,.0f}{mark}{speedup:>8.0f}x {peak:>9.1f}"
        )
    if any_extrapolated:
        full = next((r.n_sims_full for r in rows if r.extrapolated and r.n_sims_full), 0)
        lines += [
            "",
            f"+ rate measured on a shorter run and quoted per second; peak MB restated at {full:,} sims.",
            "  Peak MB is the float32 draw buffer -- for the Python row, real lists are several times larger.",
        ]
    return "\n".join(lines)


def save(rows: list[BenchRow], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": host_description(),
        "device": device_description(),
        "rows": [asdict(r) for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
