"""Scoring rules for probability forecasts, and a reliability curve.

A simulator that says 70% is only useful if the 70% side wins about 70% of the
time. These are the tools for checking that, plus a baseline to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of the probabilities. Lower is better; 0.25 is a coin flip."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p >= 0.5) == (y >= 0.5)))


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed win rate against predicted probability, bin by bin."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        predicted = float(p[mask].mean())
        observed = float(y[mask].mean())
        # Binomial standard error on the observed rate, for an honest error bar.
        se = float(np.sqrt(max(observed * (1 - observed), EPS) / n))
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "n": n,
                "predicted": predicted,
                "observed": observed,
                "se": se,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Average gap between predicted and observed, weighted by bin size."""
    table = reliability_table(p, y, n_bins)
    if table.empty:
        return float("nan")
    weights = table["n"].to_numpy(float)
    gaps = np.abs(table["predicted"].to_numpy(float) - table["observed"].to_numpy(float))
    return float((weights * gaps).sum() / weights.sum())


def sharpness(p: np.ndarray) -> float:
    """How far from 50% the forecasts dare to go. A model can be perfectly
    calibrated and useless by always saying 50%, so this is reported alongside."""
    return float(np.mean(np.abs(np.asarray(p, float) - 0.5)))


# --- the baseline to beat -------------------------------------------------
def fit_margin_logistic(margin: np.ndarray, y: np.ndarray, *, iters: int = 50) -> tuple[float, float]:
    """Fit P(win) = sigmoid(a + b * projected margin) by Newton's method.

    This is the strongest thing a projected-points system can honestly do:
    projections give a margin, and a single fitted slope turns that margin into
    a probability. It is the baseline the simulator has to beat.
    """
    x = np.asarray(margin, float)
    y = np.asarray(y, float)
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), EPS, None)
        gradient = design.T @ (y - mu)
        hessian = design.T @ (design * w[:, None])
        try:
            step = np.linalg.solve(hessian + 1e-9 * np.eye(2), gradient)
        except np.linalg.LinAlgError:  # pragma: no cover
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def apply_margin_logistic(margin: np.ndarray, coefficients: tuple[float, float]) -> np.ndarray:
    a, b = coefficients
    return 1.0 / (1.0 + np.exp(-(a + b * np.asarray(margin, float))))


def summarise(p: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> dict[str, float]:
    return {
        "n": int(len(p)),
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
        "ece": expected_calibration_error(p, y, n_bins),
        "accuracy": accuracy(p, y),
        "sharpness": sharpness(p),
        "mean_prediction": float(np.mean(p)),
        "base_rate": float(np.mean(y)),
    }


def skill_score(p: np.ndarray, y: np.ndarray, reference: np.ndarray) -> float:
    """Brier skill score against a reference forecast. Positive means better."""
    ref = brier_score(reference, y)
    return float(1.0 - brier_score(p, y) / ref) if ref > 0 else float("nan")
