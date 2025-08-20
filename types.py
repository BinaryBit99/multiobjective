from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Literal is 3.8+; fall back to typing_extensions when needed
try:  # pragma: no cover
    from typing import Literal
except ImportError:  # Python < 3.8
    from typing_extensions import Literal  # type: ignore


ErrorType = Literal["tp", "res", "rel"]


@dataclass
class ProviderRecord:
    """Typed data for a provider at a given time."""
    service_id: str
    timestamp: int
    response_time_ms: float
    throughput_kbps: float
    cost: float
    coords: Tuple[float, float]
    qos: str | None
    qos_prob: float
    qos_volatility: float


@dataclass
class ConsumerRecord:
    """Typed data for a consumer at a given time."""
    service_id: str
    timestamp: int
    response_time_ms: float
    throughput_kbps: float
    cost: float
    coords: Tuple[float, float]
    qos: str | None
    qos_prob: float
    qos_volatility: float


# provider index per consumer (length == number of consumers)
Assignment = List[int]

# A set of objective tuples (err, cost)
Front = List[Tuple[float, float]]

# (producers, consumers) at a time t
TimeSlice = Tuple[List[ProviderRecord], List[ConsumerRecord]]

# t -> (producers, consumers)
TimeSeries = Dict[int, TimeSlice]
