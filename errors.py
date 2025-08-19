# errors.py
from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Any


# ---------------------------
# Base exception hierarchy
# ---------------------------

class MultiObjError(Exception):
    """Base class for all package-specific exceptions."""


class ConfigError(MultiObjError):
    """Invalid or inconsistent configuration options."""


class DataLoadError(MultiObjError):
    """Raised when a dataset/path cannot be loaded."""

    def __init__(self, path: str, message: str | None = None) -> None:
        msg = message or f"Failed to load data from path: {path!r}"
        super().__init__(msg)
        self.path = path


class DataShapeError(MultiObjError):
    """Unexpected array/dataframe shape or missing values."""

    def __init__(
        self,
        name: str,
        expected: Tuple[int | None, ...] | None,
        got: Tuple[int, ...] | None,
        message: str | None = None,
    ) -> None:
        exp = "any" if expected is None else expected
        message = message or f"Bad shape for {name!r}: expected {exp}, got {got}"
        super().__init__(message)
        self.name = name
        self.expected = expected
        self.got = got


class MetricError(MultiObjError):
    """Raised when a metric cannot be computed (e.g., empty front or ref set)."""


class RNGScopeError(MultiObjError):
    """Unknown RNG scope or invalid RNG usage."""


class CoverageError(MultiObjError):
    """No feasible provider within coverage radius for a consumer at time t."""

    def __init__(self, consumer_id: str, t: int, radius: float) -> None:
        super().__init__(
            f"No feasible provider within radius {radius:.3f} for consumer {consumer_id!r} at t={t}"
        )
        self.consumer_id = consumer_id
        self.t = t
        self.radius = radius


class InfeasibleAssignmentError(MultiObjError):
    """Raised when an assignment cannot satisfy hard constraints."""


class NotFittedError(MultiObjError):
    """Operation requires a fitted model/state, but it hasn't been initialized."""


class AlgorithmError(MultiObjError):
    """Generic algorithm runtime error (NSGA/MOPSO/MOGWO/etc.)."""


# ---------------------------
# Warnings (opt-in)
# ---------------------------

class ConvergenceWarning(UserWarning):
    """Algorithm stopped early or failed to converge meaningfully."""


class NumericalStabilityWarning(UserWarning):
    """Potential numerical issues (underflow/overflow/div-by-zero)."""


class ReproducibilityWarning(UserWarning):
    """Non-deterministic behavior detected when determinism was expected."""


# ---------------------------
# Small validation helpers
# ---------------------------

def require_nonempty(name: str, x: Iterable[Any]) -> None:
    """Raise DataShapeError if an iterable is empty."""
    try:
        iterator = iter(x)
    except TypeError as e:  # not iterable
        raise DataShapeError(name, expected=None, got=None, message=f"{name!r} is not iterable") from e
    if not any(True for _ in iterator):
        raise DataShapeError(name, expected=None, got=(0,), message=f"{name!r} cannot be empty")


def require_same_length(a: Sequence[Any], b: Sequence[Any], name_a: str = "a", name_b: str = "b") -> None:
    """Raise DataShapeError if two sequences do not have the same length."""
    if len(a) != len(b):
        raise DataShapeError(
            name=f"{name_a}/{name_b}",
            expected=(len(a), len(a)),
            got=(len(a), len(b)),
            message=f"Lengths differ: {name_a}={len(a)} vs {name_b}={len(b)}",
        )


def require_coverage(feasible_count: int, consumer_id: str, t: int, radius: float) -> None:
    """Raise CoverageError if no feasible providers exist for a consumer."""
    if feasible_count <= 0:
        raise CoverageError(consumer_id=consumer_id, t=t, radius=radius)


__all__ = [
    # exceptions
    "MultiObjError",
    "ConfigError",
    "DataLoadError",
    "DataShapeError",
    "MetricError",
    "RNGScopeError",
    "CoverageError",
    "InfeasibleAssignmentError",
    "NotFittedError",
    "AlgorithmError",
    # warnings
    "ConvergenceWarning",
    "NumericalStabilityWarning",
    "ReproducibilityWarning",
    # helpers
    "require_nonempty",
    "require_same_length",
    "require_coverage",
]
