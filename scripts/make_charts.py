#!/usr/bin/env python
"""Regenerate every figure in the README.

    python scripts/make_charts.py --all
    python scripts/make_charts.py --risk-tilt --benchmark

Each figure is written in a light and a dark variant so the README can serve
whichever the reader's theme asks for.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
IMG = REPO / "docs" / "img"


def make_risk_tilt(n_sims: int) -> None:
    """The crossover: when does a volatile lineup beat a steady one?"""
    from boombust.sim.backends.numpy_backend import simulate
    from boombust.viz import risk_tilt_chart

    def params(mean: float, sd: float) -> list[float]:
        sigma_sq = float(np.log1p((sd * sd) / (mean * mean)))
        return [1.0, float(np.log(mean) - 0.5 * sigma_sq), float(np.sqrt(sigma_sq))]

    core = [params(12.0, 5.0) for _ in range(8)]
    mine = np.array(core + [params(10.0, 3.0), params(10.0, 11.0)], dtype=np.float32)
    lineups = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 9]], dtype=np.int32
    )
    opp_lineup = np.arange(9, dtype=np.int32)

    totals = [70, 80, 90, 100, 106, 112, 120, 130, 140, 150]
    safe, risky = [], []
    for total in totals:
        theirs = np.array([params(total / 9.0, 5.0) for _ in range(9)], dtype=np.float32)
        result = simulate(mine, theirs, lineups, opp_lineup, n_sims, seed=42, rng="counter")
        safe.append(float(result.win_prob[0]))
        risky.append(float(result.win_prob[1]))

    paths = risk_tilt_chart(
        totals, safe, risky, IMG / "risk_tilt.png",
        subtitle="Both lineups project the same points; only the spread of one player differs",
    )
    print("risk tilt ->", ", ".join(str(p) for p in paths))


def make_benchmark(n_sims: int, n_lineups: int, python_sims: int) -> None:
    from boombust.bench import device_description, format_table, host_description, run, save
    from boombust.viz import benchmark_chart

    rows = run(n_sims=n_sims, n_lineups=n_lineups, python_sims=python_sims)
    print(format_table(rows))
    save(rows, REPO / "docs" / "benchmark.json")
    paths = benchmark_chart(
        [asdict(r) for r in rows], IMG / "benchmark.png",
        subtitle=(
            f"{n_lineups:,} candidate lineups x {n_sims:,} simulated weeks  ·  "
            + "  ·  ".join(x for x in (device_description(), host_description()) if x)
        ),
    )
    print("benchmark ->", ", ".join(str(p) for p in paths))


def redraw_benchmark(source: Path) -> None:
    """Redraw the benchmark chart from a saved run instead of measuring again.

    The GPU rows can only be measured on a CUDA machine, so the run that
    produced them is the artifact worth keeping. This redraws the chart from
    that stored JSON -- on any machine, without a GPU and without overwriting
    the numbers with whatever the local box happens to manage.
    """
    from boombust.viz import benchmark_chart

    payload = json.loads(Path(source).read_text())
    rows = payload["rows"]
    live = [r for r in rows if r.get("lineup_sims_per_sec") == r.get("lineup_sims_per_sec")]
    if not live:
        raise SystemExit(f"{source} has no usable rows")
    n_sims = max(r.get("n_sims_full") or r.get("n_sims", 0) for r in live)
    n_lineups = live[0].get("n_lineups", 0)
    where = "  ·  ".join(x for x in (payload.get("device", ""), payload.get("host", "")) if x)
    paths = benchmark_chart(
        rows, IMG / "benchmark.png",
        subtitle=f"{n_lineups:,} candidate lineups x {n_sims:,} simulated weeks  ·  {where}",
    )
    print("benchmark (redrawn from", f"{source})", "->", ", ".join(str(p) for p in paths))


def make_reliability(league_ids: list[str], seasons: list[int], weeks: range, n_sims: int) -> None:
    import pandas as pd

    from boombust.db import session
    from boombust.validate.backtest import run_backtest
    from boombust.validate.calibration import (
        apply_margin_logistic,
        fit_margin_logistic,
        reliability_table,
        skill_score,
        summarise,
    )
    from boombust.viz import reliability_chart

    frames = []
    with session() as conn:
        for league_id in league_ids:
            df = run_backtest(conn, league_id, seasons, weeks, n_sims=n_sims, progress=False)
            if not df.empty:
                frames.append(df)
            print(f"  {league_id}: {len(df)} matchups")
    if not frames:
        raise SystemExit("no matchups; run 'boombust synth' for these seasons first")
    df = pd.concat(frames, ignore_index=True)

    y = df["home_won"].to_numpy(float)
    p_model = df["p_home_model"].to_numpy(float)
    p_base = apply_margin_logistic(
        df["projected_margin"].to_numpy(float),
        fit_margin_logistic(df["projected_margin"].to_numpy(float), y),
    )
    coin = np.full_like(p_model, 0.5)

    bins = 8
    curves = [
        ("BoomBust simulator", reliability_table(p_model, y, bins), summarise(p_model, y)),
        ("projected points", reliability_table(p_base, y, bins), summarise(p_base, y)),
    ]
    paths = reliability_chart(
        curves, IMG / "calibration.png",
        subtitle=(
            f"{len(df):,} matchups, {min(seasons)}-{max(seasons)}  ·  "
            "each week modelled using only earlier games"
        ),
    )
    print("calibration ->", ", ".join(str(p) for p in paths))
    summary = {
        "n_matchups": int(len(df)),
        "seasons": seasons,
        "simulator": summarise(p_model, y),
        "points_baseline": summarise(p_base, y),
        "coin_flip": summarise(coin, y),
        "skill_vs_baseline": skill_score(p_model, y, p_base),
        "skill_vs_coin": skill_score(p_model, y, coin),
    }
    (REPO / "docs" / "calibration.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--risk-tilt", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--benchmark-from", metavar="JSON",
        help="redraw the benchmark chart from a saved run instead of measuring again",
    )
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--sims", type=int, default=2_000_000, help="simulations for the risk-tilt sweep")
    parser.add_argument("--bench-sims", type=int, default=1_000_000)
    parser.add_argument("--bench-lineups", type=int, default=1700)
    parser.add_argument("--python-sims", type=int, default=4000)
    parser.add_argument("--backtest-sims", type=int, default=100_000)
    parser.add_argument("--leagues", default="synthetic,synthetic-b,synthetic-c,synthetic-d")
    parser.add_argument("--seasons", default="2021,2022,2023,2024")
    args = parser.parse_args()

    IMG.mkdir(parents=True, exist_ok=True)
    if args.all or args.risk_tilt:
        make_risk_tilt(args.sims)
    if args.benchmark_from:
        redraw_benchmark(Path(args.benchmark_from))
    elif args.all or args.benchmark:
        make_benchmark(args.bench_sims, args.bench_lineups, args.python_sims)
    if args.all or args.calibration:
        make_reliability(
            [s.strip() for s in args.leagues.split(",") if s.strip()],
            [int(s) for s in args.seasons.split(",")],
            range(1, 15),
            args.backtest_sims,
        )


if __name__ == "__main__":
    main()
