"""Backend registry.

Every backend implements the same ``simulate(...) -> SimOutcome`` call, so the
engine never knows which one it is talking to. CuPy is imported *inside* the GPU
backends, never at module scope: CuPy does not install on macOS, and a
top-level import would break the NumPy path on the very machine most of this
was written on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

BACKENDS = ("numpy", "cupy", "cuda")


@dataclass
class SimOutcome:
    """Raw counts from one simulation run."""

    wins: np.ndarray          # (L,) int64
    ties: np.ndarray          # (L,) int64
    n_sims: int
    backend: str
    elapsed: float
    peak_bytes: int = 0       # device/host bytes held for draws and totals

    @property
    def win_prob(self) -> np.ndarray:
        """Ties split the pot, so they count as half a win."""
        return (self.wins + 0.5 * self.ties) / float(self.n_sims)


class Backend(Protocol):
    def __call__(
        self,
        my_params: np.ndarray,
        opp_params: np.ndarray,
        lineups: np.ndarray,
        opp_lineup: np.ndarray,
        n_sims: int,
        seed: int,
        **kwargs: object,
    ) -> SimOutcome: ...


def get_backend(name: str) -> Backend:
    """Resolve a backend by name, importing it only when asked for."""
    name = (name or "numpy").lower()
    if name == "numpy":
        from .numpy_backend import simulate

        return simulate
    if name == "cupy":
        from .cupy_backend import simulate

        return simulate
    if name == "cuda":
        from .cuda_backend import simulate

        return simulate
    raise ValueError(f"unknown backend {name!r}; choose from {', '.join(BACKENDS)}")


def available_backends() -> dict[str, str]:
    """Which backends can actually run here, and why not if they cannot."""
    status = {"numpy": "ready"}
    try:
        import cupy  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the host
        reason = str(exc).split("\n")[0]
        status["cupy"] = f"unavailable ({reason})"
        status["cuda"] = f"unavailable ({reason})"
        return status
    try:
        dev = cupy.cuda.runtime.getDeviceProperties(0)
        gpu = dev["name"].decode() if isinstance(dev["name"], bytes) else str(dev["name"])
        status["cupy"] = f"ready ({gpu})"
        status["cuda"] = f"ready ({gpu})"
    except Exception as exc:  # pragma: no cover
        reason = str(exc).split("\n")[0]
        status["cupy"] = f"installed but no device ({reason})"
        status["cuda"] = f"installed but no device ({reason})"
    return status


def validate_inputs(
    my_params: np.ndarray,
    opp_params: np.ndarray,
    lineups: np.ndarray,
    opp_lineup: np.ndarray,
    n_sims: int,
) -> None:
    if my_params.ndim != 2 or my_params.shape[1] != 3:
        raise ValueError(f"my_params must be (P, 3), got {my_params.shape}")
    if opp_params.ndim != 2 or opp_params.shape[1] != 3:
        raise ValueError(f"opp_params must be (Q, 3), got {opp_params.shape}")
    if lineups.ndim != 2 or lineups.size == 0:
        raise ValueError(f"lineups must be a non-empty (L, K) array, got {lineups.shape}")
    if lineups.max(initial=0) >= my_params.shape[0] or lineups.min(initial=0) < 0:
        raise ValueError("lineups index players outside my_params")
    if opp_lineup.size == 0:
        raise ValueError("opp_lineup is empty")
    if opp_lineup.max(initial=0) >= opp_params.shape[0] or opp_lineup.min(initial=0) < 0:
        raise ValueError("opp_lineup indexes players outside opp_params")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
