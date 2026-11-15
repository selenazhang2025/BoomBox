"""Command line interface.

    boombust init                     create the database
    boombust ingest --seasons 2021-2024
    boombust synth  --season 2024     build a league from real players
    boombust sync                     pull a real ESPN league
    boombust lineup --week 10         rank your lineups by win probability
    boombust backends                 what can run here
    boombust benchmark                compare the backends
    boombust backtest                 replay past weeks and check calibration
    boombust decide                   did the win-probability pick win more?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .config import load_config
from .db import init_db, session
from .model.scoring import SLOT_LABELS

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _plain() -> bool:
    return not sys.stdout.isatty()


def b(text: str) -> str:
    return text if _plain() else f"{BOLD}{text}{RESET}"


def dim(text: str) -> str:
    return text if _plain() else f"{DIM}{text}{RESET}"


def parse_weeks(spec: str) -> list[int]:
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = (int(x) for x in chunk.split("-", 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


# --- commands -------------------------------------------------------------
def cmd_init(args, config) -> int:
    path = init_db(config.db_path)
    print(f"database ready at {path}")
    return 0


def cmd_ingest(args, config) -> int:
    from .data.nflverse import available_seasons, ingest, parse_seasons

    init_db(config.db_path)
    if args.list:
        print("nflverse has weekly player stats for:", ", ".join(str(s) for s in available_seasons(refresh=args.refresh)))
        return 0
    seasons = parse_seasons(args.seasons, refresh=args.refresh)
    if not seasons:
        print("nothing to ingest", file=sys.stderr)
        return 1
    print(f"ingesting {len(seasons)} season(s): {', '.join(str(s) for s in seasons)}")
    counts = ingest(seasons, config.db_path, refresh=args.refresh)
    print("  " + "  ".join(f"{k}={v:,}" for k, v in counts.items()))
    return 0


def cmd_synth(args, config) -> int:
    from .data.synthetic import build_league

    init_db(config.db_path)
    weeks = parse_weeks(args.weeks)
    counts = build_league(
        args.season,
        league_id=args.league_id,
        n_teams=args.teams,
        weeks=weeks,
        seed=args.seed,
        db_path=config.db_path,
    )
    print(
        f"built league {args.league_id!r} for {args.season}: "
        f"{counts['teams']} teams, {counts['matchups']} matchups, weeks {weeks[0]}-{weeks[-1]}"
    )
    print(dim("  players, stat lines and outcomes are real; only the league is invented"))
    return 0


def cmd_sync(args, config) -> int:
    from .data.espn import EspnClient, sync_league
    from .data.match import link_roster_ids

    league_id = args.league_id or config.league_id
    season = args.season or config.season
    if not league_id or not season:
        print("set BOOMBUST_LEAGUE_ID and BOOMBUST_SEASON in .env, or pass --league-id/--season", file=sys.stderr)
        return 2
    init_db(config.db_path)
    client = EspnClient(league_id, season, config.espn_s2, config.swid)
    weeks = parse_weeks(args.weeks)
    print(f"syncing ESPN league {league_id}, season {season}, weeks {weeks[0]}-{weeks[-1]}")
    counts = sync_league(client, weeks, config.db_path)
    print("  " + "  ".join(f"{k}={v:,}" for k, v in counts.items()))
    with session(config.db_path) as conn:
        matched, unmatched = link_roster_ids(conn, str(league_id), int(season))
    print(f"  matched {matched:,} roster entries to nflverse players")
    if unmatched:
        shown = ", ".join(sorted(set(unmatched))[:8])
        print(dim(f"  unmatched ({len(set(unmatched))}): {shown}"))
        print(dim("  unmatched players fall back to the positional prior"))
    return 0


def cmd_backends(args, config) -> int:
    from .sim.backends import available_backends

    print(b("backend     status"))
    for name, status in available_backends().items():
        print(f"{name:<11} {status}")
    return 0


def cmd_lineup(args, config) -> int:
    from .sim.engine import optimise_week

    league_id = args.league_id or config.league_id or "synthetic"
    with session(config.db_path) as conn:
        season = args.season or config.season or _latest_league_season(conn, league_id)
        team_id = args.team if args.team is not None else _first_team(conn, league_id, season)
        advice = optimise_week(
            conn, str(league_id), int(season), args.week, int(team_id),
            backend=args.backend, n_sims=args.sims, seed=args.seed,
            opponent_mode=args.opponent_lineup,
        )
    _print_advice(advice, top=args.top, show_all_swings=args.all_swings)
    if args.json:
        Path(args.json).write_text(json.dumps(_advice_payload(advice), indent=2))
        print(dim(f"\nwrote {args.json}"))
    return 0


def _latest_league_season(conn, league_id: str) -> int:
    row = conn.execute(
        "SELECT MAX(season) AS s FROM leagues WHERE league_id = ?", (str(league_id),)
    ).fetchone()
    if not row or row["s"] is None:
        raise SystemExit(f"no league {league_id!r} in the database; run 'boombust synth' or 'boombust sync' first")
    return int(row["s"])


def _first_team(conn, league_id: str, season: int) -> int:
    row = conn.execute(
        "SELECT MIN(team_id) AS t FROM fantasy_teams WHERE league_id = ? AND season = ?",
        (str(league_id), int(season)),
    ).fetchone()
    if not row or row["t"] is None:
        raise SystemExit("no teams found for that league and season")
    return int(row["t"])


def _print_advice(advice, *, top: int = 5, show_all_swings: bool = False) -> None:
    best = advice.best_by_win_prob
    print()
    print(b(f"{advice.team_name}  vs  {advice.opponent_name}") + dim(f"   {advice.league_name}, {advice.season} week {advice.week}"))
    print(
        dim(
            f"{len(advice.lineups):,} legal lineups  ·  {advice.outcome.n_sims:,} simulated weeks  ·  "
            f"{advice.outcome.backend} backend  ·  {advice.outcome.elapsed:.2f}s"
        )
    )
    if advice.dropped:
        names = ", ".join(advice.spec.names[i] for i in advice.dropped)
        print(dim(f"trimmed from the search to keep it tractable: {names}"))
    print()
    print(b(f"  best lineup — {100 * advice.win_prob[best]:.1f}% to win"))
    print(dim(f"  {'slot':<6} {'player':<26} {'proj':>6} {'10th–90th':>14}  note"))
    for slot, player in advice.slot_assignment(best):
        p = advice.projections[player]
        lo, hi = p.quantile(0.10), p.quantile(0.90)
        note = p.note or (f"vs {p.opponent}" if p.opponent else "")
        if p.def_factor and abs(p.def_factor - 1.0) > 0.06 and p.p_active > 0:
            note += f"  ({'+' if p.def_factor > 1 else ''}{100 * (p.def_factor - 1):.0f}% matchup)"
        print(
            f"  {SLOT_LABELS.get(slot, slot):<6} {advice.spec.names[player]:<26} "
            f"{p.mean:>6.1f} {lo:>6.1f}–{hi:<7.1f} {dim(note)}"
        )
    print(
        f"  {'':<6} {'total':<26} {advice.lineup_mean[best]:>6.1f}"
        + dim(f"      opponent {advice.opp_mean:.1f} ± {advice.opp_sd:.1f}")
    )

    if advice.they_disagree:
        pts = advice.best_by_points
        print()
        print(b("  the points-maximising lineup is a different lineup"))
        mine = set(advice.lineups[best].tolist())
        theirs = set(advice.lineups[pts].tolist())
        for out_i in sorted(theirs - mine):
            for in_i in sorted(mine - theirs):
                print(
                    f"    start {advice.spec.names[in_i]} over {advice.spec.names[out_i]}: "
                    f"{advice.lineup_mean[best] - advice.lineup_mean[pts]:+.1f} projected points, "
                    f"{100 * (advice.win_prob[best] - advice.win_prob[pts]):+.2f}pp win probability"
                )
                break
        print(
            dim(
                f"    you are {'an underdog' if advice.win_prob[best] < 0.5 else 'favoured'}, so the simulator "
                f"prefers the {'higher' if advice.lineup_sd[best] > advice.lineup_sd[pts] else 'lower'}-variance lineup"
            )
        )

    order = advice.order_by_win_prob()[:top]
    print()
    print(b(f"  top {len(order)} lineups"))
    print(dim(f"  {'#':>3} {'P(win)':>8} {'proj':>7} {'sd':>6}  differences from the best lineup"))
    best_set = set(advice.lineups[best].tolist())
    for rank, idx in enumerate(order, start=1):
        this = set(advice.lineups[idx].tolist())
        added = [advice.spec.names[i] for i in sorted(this - best_set)]
        removed = [advice.spec.names[i] for i in sorted(best_set - this)]
        if not added and not removed:
            delta = dim("—")
        else:
            delta = " / ".join(f"+{a}" for a in added) + ("  " if added and removed else "") + " / ".join(f"-{r}" for r in removed)
        print(
            f"  {rank:>3} {100 * advice.win_prob[idx]:>7.2f}% {advice.lineup_mean[idx]:>7.1f} "
            f"{advice.lineup_sd[idx]:>6.1f}  {delta}"
        )

    swings = advice.player_swings()
    if not show_all_swings:
        swings = swings[:8]
    print()
    print(b("  what each decision is worth"))
    print(dim("  the swing between forcing a player into the lineup and forcing them out"))
    print(dim(f"  {'player':<26} {'win prob':>10} {'points':>9}  verdict"))
    for i, swing, points_swing, starts in swings:
        verdict = "start" if starts else "bench"
        print(
            f"  {advice.spec.names[i]:<26} {100 * swing:>+9.2f}pp {points_swing:>+8.1f}  "
            f"{b(verdict) if starts else dim(verdict)}"
        )
    print()


def _advice_payload(advice) -> dict:
    best = advice.best_by_win_prob
    return {
        "league": advice.league_name,
        "season": advice.season,
        "week": advice.week,
        "team": advice.team_name,
        "opponent": advice.opponent_name,
        "backend": advice.outcome.backend,
        "n_sims": advice.outcome.n_sims,
        "seconds": advice.outcome.elapsed,
        "n_lineups": int(len(advice.lineups)),
        "best": {
            "win_probability": float(advice.win_prob[best]),
            "projected_points": float(advice.lineup_mean[best]),
            "projected_sd": float(advice.lineup_sd[best]),
            "starters": [
                {"slot": SLOT_LABELS.get(s, str(s)), "player": advice.spec.names[p],
                 "projected": float(advice.projections[p].mean)}
                for s, p in advice.slot_assignment(best)
            ],
        },
        "opponent_projection": {"mean": advice.opp_mean, "sd": advice.opp_sd},
        "points_pick_differs": bool(advice.they_disagree),
        "swings": [
            {"player": advice.spec.names[i], "win_prob_swing": s, "points_swing": p, "starts": st}
            for i, s, p, st in advice.player_swings()
        ],
    }


def cmd_benchmark(args, config) -> int:
    from .bench import format_table, host_description, run, save

    print(f"host: {host_description()}")
    print(dim("one lineup-simulation = sum one lineup's starters for one week and compare\n"))
    rows = run(
        n_sims=args.sims,
        n_lineups=args.lineups,
        python_sims=args.python_sims,
        include_python=not args.no_python,
    )
    print(format_table(rows))
    if args.out:
        save(rows, args.out)
        print(dim(f"\nwrote {args.out}"))
    if args.chart:
        from .viz import benchmark_chart
        from dataclasses import asdict

        paths = benchmark_chart(
            [asdict(r) for r in rows],
            args.chart,
            subtitle=f"{args.lineups:,} candidate lineups x {args.sims:,} simulated weeks · {host_description()}",
        )
        print(dim(f"wrote {', '.join(str(p) for p in paths)}"))
    return 0


def cmd_backtest(args, config) -> int:
    from .validate.backtest import run_backtest
    from .validate.calibration import (
        apply_margin_logistic,
        fit_margin_logistic,
        reliability_table,
        skill_score,
        summarise,
    )

    league_id = args.league_id or config.league_id or "synthetic"
    seasons = [int(s) for s in parse_weeks(args.seasons)]
    weeks = parse_weeks(args.weeks)
    with session(config.db_path) as conn:
        df = run_backtest(
            conn, str(league_id), seasons, weeks,
            backend=args.backend, n_sims=args.sims, seed=args.seed,
        )
    if df.empty:
        print("no matchups found to replay", file=sys.stderr)
        return 1

    y = df["home_won"].to_numpy(float)
    p_model = df["p_home_model"].to_numpy(float)
    margin = df["projected_margin"].to_numpy(float)
    coefficients = fit_margin_logistic(margin, y)
    p_base = apply_margin_logistic(margin, coefficients)
    coin = np.full_like(p_model, 0.5)

    print()
    print(b(f"  backtest: {len(df):,} matchups, seasons {min(seasons)}-{max(seasons)}"))
    print(dim("  every week modelled using only games played before it"))
    print()
    print(dim(f"  {'forecaster':<28} {'Brier':>8} {'log loss':>9} {'ECE':>7} {'accuracy':>9} {'sharpness':>10}"))
    rows = [
        ("BoomBust simulator", p_model),
        ("projected-points baseline", p_base),
        ("always 50%", coin),
    ]
    for name, p in rows:
        m = summarise(p, y)
        print(
            f"  {name:<28} {m['brier']:>8.4f} {m['log_loss']:>9.4f} {m['ece']:>7.3f} "
            f"{m['accuracy']:>9.3f} {m['sharpness']:>10.3f}"
        )
    print()
    print(f"  Brier skill vs the points baseline: {skill_score(p_model, y, p_base):+.4f}")
    print(f"  Brier skill vs a coin flip:         {skill_score(p_model, y, coin):+.4f}")
    print(dim("  (the baseline is fitted on these same outcomes; the simulator fits nothing)"))
    print()
    table = reliability_table(p_model, y, args.bins)
    print(b("  reliability"))
    print(dim(f"  {'stated':>10} {'observed':>10} {'n':>6}"))
    for _, r in table.iterrows():
        print(f"  {r['predicted']:>9.1%} {r['observed']:>10.1%} {int(r['n']):>6}")
    print()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(dim(f"wrote {args.out}"))
    if args.chart:
        from .viz import reliability_chart

        curves = [
            ("BoomBust simulator", table, summarise(p_model, y)),
            ("projected points", reliability_table(p_base, y, args.bins), summarise(p_base, y)),
        ]
        paths = reliability_chart(
            curves,
            args.chart,
            subtitle=f"{len(df):,} matchups, {min(seasons)}-{max(seasons)}; error bars are binomial standard errors",
        )
        print(dim(f"wrote {', '.join(str(p) for p in paths)}"))
    return 0


def cmd_decide(args, config) -> int:
    from .validate.backtest import decision_backtest

    league_id = args.league_id or config.league_id or "synthetic"
    seasons = [int(s) for s in parse_weeks(args.seasons)]
    weeks = parse_weeks(args.weeks)
    with session(config.db_path) as conn:
        df = decision_backtest(
            conn, str(league_id), seasons, weeks,
            backend=args.backend, n_sims=args.sims, seed=args.seed,
        )
    if df.empty:
        print("no decisions to score", file=sys.stderr)
        return 1
    differ = df[df["differs"]]
    print()
    print(b(f"  {len(df):,} lineup decisions replayed"))
    print(f"  the two rules picked the same lineup {100 * (1 - len(differ) / len(df)):.1f}% of the time")
    if differ.empty:
        print(dim("  they never disagreed, so there is nothing to score"))
        return 0
    print()
    plural = "decision" if len(differ) == 1 else "decisions"
    print(f"  where they disagreed ({len(differ):,} {plural}):")
    print(f"    win-probability pick won {differ['won_chosen'].sum():.1f} of {len(differ)}")
    print(f"    points pick won          {differ['won_points'].sum():.1f} of {len(differ)}")
    edge = (differ["won_chosen"] - differ["won_points"]).mean()
    print(f"    difference: {edge:+.4f} wins per decision ({edge * len(differ):+.1f} wins over the sample)")
    print(
        dim(
            f"    average claimed edge was {100 * (differ['p_win_chosen'] - differ['p_win_points']).mean():+.3f}pp, "
            f"at a cost of {(differ['points_chosen'] - differ['points_points']).mean():+.2f} projected points"
        )
    )
    print()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(dim(f"wrote {args.out}"))
    return 0


# --- argument parsing -----------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boombust", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"boombust {__version__}")
    parser.add_argument("--db", help="path to the sqlite database (default: data/boombust.sqlite)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the database").set_defaults(func=cmd_init)

    p = sub.add_parser("ingest", help="download nflverse weekly data")
    p.add_argument("--seasons", default=None, help="e.g. 2021-2024 or 2023,2024")
    p.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    p.add_argument("--list", action="store_true", help="list the seasons nflverse has")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("synth", help="build a synthetic league from real players")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--league-id", default="synthetic")
    p.add_argument("--teams", type=int, default=10)
    p.add_argument("--weeks", default="1-14")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("sync", help="pull a real ESPN league")
    p.add_argument("--league-id", default=None)
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--weeks", default="1-18")
    p.set_defaults(func=cmd_sync)

    sub.add_parser("backends", help="show which backends can run here").set_defaults(func=cmd_backends)

    p = sub.add_parser("lineup", help="rank your lineups by win probability")
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--team", type=int, default=None, help="fantasy team id")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--league-id", default=None)
    p.add_argument("--backend", default="numpy", help="numpy, cupy or cuda")
    p.add_argument("--sims", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--all-swings", action="store_true")
    p.add_argument("--opponent-lineup", choices=("best", "actual"), default="best")
    p.add_argument("--json", default=None, help="also write the result as json")
    p.set_defaults(func=cmd_lineup)

    p = sub.add_parser("benchmark", help="compare the backends")
    p.add_argument("--sims", type=int, default=1_000_000)
    p.add_argument("--lineups", type=int, default=1700)
    p.add_argument("--python-sims", type=int, default=4000)
    p.add_argument("--no-python", action="store_true", help="skip the pure-Python baseline")
    p.add_argument("--out", default=None, help="write the results as json")
    p.add_argument("--chart", default=None, help="write a chart to this path")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("backtest", help="replay past weeks and check calibration")
    p.add_argument("--league-id", default=None)
    p.add_argument("--seasons", default="2021-2024")
    p.add_argument("--weeks", default="1-14")
    p.add_argument("--backend", default="numpy")
    p.add_argument("--sims", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--out", default=None, help="write the per-matchup rows as csv")
    p.add_argument("--chart", default=None, help="write a reliability chart to this path")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("decide", help="did the win-probability pick actually win more?")
    p.add_argument("--league-id", default=None)
    p.add_argument("--seasons", default="2021-2024")
    p.add_argument("--weeks", default="1-14")
    p.add_argument("--backend", default="numpy")
    p.add_argument("--sims", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_decide)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    if args.db:
        config = type(config)(
            db_path=Path(args.db), league_id=config.league_id, season=config.season,
            espn_s2=config.espn_s2, swid=config.swid,
        )
    try:
        return args.func(args, config)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
