"""Charts: calibration, benchmark, and the risk/reward crossover.

Colours come from a validated categorical palette (slots blue / orange / aqua),
which clears the colour-blind separation and normal-vision floors on both
surfaces. Light-mode aqua sits under 3:1 against the surface, so every series
carries a visible direct label and identity is never colour alone. Each chart is
written out in a light and a dark variant so a README can serve whichever the
reader's theme asks for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str, str]
    positive: str
    negative: str


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    positive="#2a78d6",
    negative="#d03b3b",
)
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70"),
    positive="#3987e5",
    negative="#d03b3b",
)
THEMES = (LIGHT, DARK)


def _style(theme: Theme) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.ink,
            "axes.labelcolor": theme.ink_secondary,
            "axes.edgecolor": theme.axis,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
        }
    )


def _header(ax, theme: Theme, title: str, subtitle: str = "") -> None:
    """Title above subtitle, with enough room that they never collide."""
    ax.set_title(
        title, fontsize=13, fontweight="bold", color=theme.ink, loc="left",
        pad=32 if subtitle else 16,
    )
    if subtitle:
        ax.annotate(
            subtitle, xy=(0, 1), xytext=(0, 10), xycoords="axes fraction",
            textcoords="offset points", fontsize=9.5, color=theme.ink_secondary,
            ha="left", va="bottom",
        )


def _variants(path: Path | str) -> list[tuple[Theme, Path]]:
    """One output per theme: ``name.png`` and ``name-dark.png``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return [
        (LIGHT, path),
        (DARK, path.with_name(f"{path.stem}-dark{path.suffix}")),
    ]


# --- calibration ----------------------------------------------------------
def reliability_chart(
    curves: Sequence[tuple[str, pd.DataFrame, dict]],
    path: Path | str,
    *,
    title: str = "Does a stated probability hold up?",
    subtitle: str = "",
) -> list[Path]:
    """Observed win rate against stated win probability, per forecaster.

    ``curves`` is a list of (label, reliability table, metrics dict).
    """
    written = []
    for theme, out in _variants(path):
        _style(theme)
        fig, ax = plt.subplots(figsize=(7.2, 5.4))

        ax.plot([0, 1], [0, 1], linestyle=(0, (4, 4)), linewidth=1.5, color=theme.axis, zorder=1)
        # Placed below the diagonal in the gap the curves leave clear.
        ax.annotate(
            "perfectly calibrated", xy=(0.84, 0.28), color=theme.muted, fontsize=9,
            ha="center", va="center",
        )

        drawn = []
        for i, (label, table, metrics) in enumerate(curves):
            colour = theme.series[i % len(theme.series)]
            x = table["predicted"].to_numpy(float)
            y = table["observed"].to_numpy(float)
            err = table["se"].to_numpy(float)
            counts = table["n"].to_numpy(float)
            ax.errorbar(
                x, y, yerr=err, fmt="none", ecolor=colour, elinewidth=1.5, capsize=0, alpha=0.55, zorder=2
            )
            ax.plot(x, y, linewidth=2, color=colour, zorder=3)
            # Marker size carries the bin count. Without it a bin holding four
            # matchups reads exactly as confidently as one holding three hundred,
            # and the end bins -- always the thinnest -- dominate the eye.
            sizes = 8.0 + 10.0 * np.sqrt(counts / counts.max())
            for xi, yi, si in zip(x, y, sizes):
                ax.plot(
                    [xi], [yi], linestyle="none", marker="o", markersize=si, color=colour,
                    markeredgecolor=theme.surface, markeredgewidth=2, zorder=4,
                )
            drawn.append((label, x, y, colour))

        # Label each curve where the two are furthest apart, so the labels never
        # stack on each other or run off the right edge.
        if len(drawn) >= 2:
            x0, y0 = drawn[0][1], drawn[0][2]
            x1, y1 = drawn[1][1], drawn[1][2]
            gap = np.abs(y0 - np.interp(x0, x1, y1))
            # Only anchor on bins with enough matchups to be worth pointing at:
            # the thin end bins separate the most and mean the least.
            counts = curves[0][1]["n"].to_numpy(float)
            solid = counts >= 0.15 * counts.max()
            # np.interp clamps outside its range, which would invent a huge gap
            # at an x the other curve does not reach. Only compare where both
            # curves actually have bins.
            overlap = (x0 >= x1.min()) & (x0 <= x1.max())
            gap = np.where(solid & overlap, gap, -1.0)
            if not (solid & overlap).any():
                gap = np.where(overlap, np.abs(y0 - np.interp(x0, x1, y1)), -1.0)
            anchor_x = float(x0[int(np.argmax(gap))])
        else:
            anchor_x = float(drawn[0][1][len(drawn[0][1]) // 2])
        values = [float(np.interp(anchor_x, x, y)) for _label, x, y, _c in drawn]
        midpoint = sum(values) / len(values)
        for (label, _x, _y, _colour), value in zip(drawn, values):
            above = value >= midpoint
            ax.annotate(
                label, xy=(anchor_x, min(max(value, 0.03), 0.93)),
                xytext=(0, 14 if above else -14),
                textcoords="offset points", color=theme.ink, fontsize=10, fontweight="bold",
                ha="center", va="bottom" if above else "top", zorder=6,
                annotation_clip=False,
            )

        rows = []
        for label, _table, metrics in curves:
            rows.append(
                f"{label}:  Brier {metrics['brier']:.4f}   log loss {metrics['log_loss']:.4f}"
                f"   ECE {metrics['ece']:.3f}"
            )
        rows.append("marker size = matchups in the bin")
        ax.text(
            0.02, 0.98, "\n".join(rows), transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color=theme.ink_secondary, linespacing=1.6,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("stated win probability")
        ax.set_ylabel("observed win rate")
        _header(ax, theme, title, subtitle)
        ax.set_axisbelow(True)
        fig.savefig(out)
        plt.close(fig)
        written.append(out)
    return written


# --- benchmark ------------------------------------------------------------
def _ratio(x: float) -> str:
    """A speedup reads as 0.011x below 1 and 119x above it, never 0x."""
    return f"{x:,.0f}x" if x >= 10 else (f"{x:.1f}x" if x >= 1 else f"{x:.3f}x")


def benchmark_chart(
    rows: Sequence[dict],
    path: Path | str,
    *,
    title: str = "Lineup-simulations per second",
    subtitle: str = "",
) -> list[Path]:
    """One measure across backends: horizontal bars on a log scale."""
    usable = [r for r in rows if r.get("lineup_sims_per_sec") == r.get("lineup_sims_per_sec")]
    usable = sorted(usable, key=lambda r: r["lineup_sims_per_sec"])
    written = []
    for theme, out in _variants(path):
        _style(theme)
        fig, ax = plt.subplots(figsize=(7.8, 0.78 * max(len(usable), 1) + 2.2))
        labels = [r["backend"] for r in usable]
        values = np.array([r["lineup_sims_per_sec"] for r in usable], dtype=float)
        positions = np.arange(len(usable))
        colour = theme.series[0]

        # Throughput spans orders of magnitude, so the axis has to be
        # logarithmic -- and a bar's *length* on a log axis is not proportional
        # to its value, which would overstate the small backends. Encoding with
        # position instead keeps the comparison honest; the hairline is a guide
        # to the label, not a magnitude.
        floor = values.min() / 2.5 if len(values) else 1.0
        for pos, value in zip(positions, values):
            ax.plot([floor, value], [pos, pos], linewidth=1, color=theme.grid, zorder=2)
        ax.plot(
            values, positions, linestyle="none", marker="o", markersize=10, color=colour,
            markeredgecolor=theme.surface, markeredgewidth=2, zorder=4,
        )

        # NumPy, not the nested-loop baseline, when it is present: nobody would
        # write the slow one for real work, so measuring against it inflates the
        # GPU rows and says more about the interpreter than about the kernel.
        ref = next((r for r in usable if r["backend"] == "numpy"), None)
        baseline = ref["lineup_sims_per_sec"] if ref else (values[0] if len(values) else 1.0)
        against = "numpy" if ref else "baseline"
        extrapolated = False
        for pos, row, value in zip(positions, usable, values):
            # Peak memory only compares across rows if every row is quoted at the
            # same workload: a baseline that ran a short slice of the week would
            # otherwise look like the leanest backend on the chart.
            peak = row.get("peak_mb_at_full")
            if peak is None or peak != peak:
                peak = row["peak_mb"]
            peak_text = f"{peak:,.0f} MB" if peak >= 1 else f"{peak * 1000:,.0f} kB"
            mark = "*" if row.get("extrapolated") else ""
            extrapolated |= bool(row.get("extrapolated"))
            ax.annotate(
                f"{value:,.0f}/s{mark}    {_ratio(value / baseline)}    {peak_text} peak",
                xy=(value, pos), xytext=(14, 0), textcoords="offset points",
                va="center", fontsize=9, color=theme.ink_secondary, zorder=5,
            )
        ax.set_yticks(positions, labels, fontsize=11, color=theme.ink)
        ax.set_ylim(-0.6, len(usable) - 0.4)
        ax.set_xscale("log")
        if len(values):
            # Only enough headroom to lift the fastest dot off the spine. The
            # labels sit outside the axes and a tight bounding box keeps them --
            # stretching the axis to fit text instead would paint decade ticks
            # across empty space the data never reaches.
            ax.set_xlim(floor, values.max() * 2.2)
        ax.set_xlabel(f"lineup-simulations per second (log scale) · speedup shown vs {against}")
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        if extrapolated:
            ax.annotate(
                "* rate measured on a shorter run and quoted per second;"
                " peak memory restated at the full workload",
                xy=(0, 0), xytext=(0, -42), xycoords="axes fraction",
                textcoords="offset points", fontsize=8, color=theme.muted,
                ha="left", va="top",
            )
        _header(ax, theme, title, subtitle)
        fig.savefig(out)
        plt.close(fig)
        written.append(out)
    return written


# --- the risk/reward crossover -------------------------------------------
def risk_tilt_chart(
    opponent_means: Sequence[float],
    safe: Sequence[float],
    risky: Sequence[float],
    path: Path | str,
    *,
    title: str = "Why win probability and projected points disagree",
    subtitle: str = "",
) -> list[Path]:
    """Two panels: both lineups' win probability, then the edge to the risky one."""
    x = np.asarray(opponent_means, float)
    safe = np.asarray(safe, float)
    risky = np.asarray(risky, float)
    edge = risky - safe
    written = []
    for theme, out in _variants(path):
        _style(theme)
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(7.4, 6.6), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.18}
        )

        # The two curves nearly coincide, so labelling the line ends would
        # stack the labels on top of each other. Label where they are furthest
        # apart instead, one above the line and one below.
        split = int(np.argmax(np.abs(risky - safe)))
        for series, label, slot in ((safe, "steady lineup", 0), (risky, "volatile lineup", 1)):
            colour = theme.series[slot]
            top.plot(x, 100 * series, linewidth=2, color=colour, zorder=3)
            top.plot(
                x, 100 * series, linestyle="none", marker="o", markersize=8, color=colour,
                markeredgecolor=theme.surface, markeredgewidth=2, zorder=4,
            )
            above = series[split] >= (risky[split] + safe[split]) / 2
            top.annotate(
                label,
                xy=(x[split], 100 * series[split]),
                xytext=(0, 16 if above else -16), textcoords="offset points",
                color=theme.ink, fontsize=10, fontweight="bold",
                ha="center", va="bottom" if above else "top", zorder=5,
            )
        top.axhline(50, color=theme.axis, linewidth=1.2, linestyle=(0, (4, 4)), zorder=1)
        top.set_ylabel("win probability (%)")
        top.set_ylim(0, 100)
        top.set_axisbelow(True)
        _header(top, theme, title, subtitle)

        # Diverging: one hue each side of a neutral zero.
        bottom.axhline(0, color=theme.axis, linewidth=1.2, zorder=2)
        bottom.fill_between(
            x, 0, 100 * edge, where=(edge >= 0), color=theme.positive, alpha=0.75, zorder=3,
            interpolate=True,
        )
        bottom.fill_between(
            x, 0, 100 * edge, where=(edge < 0), color=theme.negative, alpha=0.75, zorder=3,
            interpolate=True,
        )
        bottom.plot(x, 100 * edge, linewidth=2, color=theme.ink_secondary, zorder=4)
        bottom.set_ylabel("edge to volatile (pp)")
        bottom.set_xlabel("opponent's projected total (points)")
        bottom.set_axisbelow(True)
        span = max(abs(100 * edge).max(), 0.5) * 1.5
        bottom.set_ylim(-span, span)
        bottom.annotate(
            "volatile wins more", xy=(0.02, 0.86), xycoords="axes fraction",
            fontsize=9, color=theme.ink_secondary,
        )
        bottom.annotate(
            "steady wins more", xy=(0.02, 0.08), xycoords="axes fraction",
            fontsize=9, color=theme.ink_secondary,
        )
        fig.savefig(out)
        plt.close(fig)
        written.append(out)
    return written
