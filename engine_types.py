"""Data structures for the ramjet sizing model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal


@dataclass(frozen=True)
class GasProps:
    """Gas properties for 1-D compressible-flow relations."""
    gamma: float = 1.4
    R: float = 287.0


@dataclass
class Station:
    """One-dimensional flow station."""
    name: str
    M: float
    Tt: float
    Pt: float
    A: float

    mdot: Optional[float] = None

    T: Optional[float] = None
    P: Optional[float] = None

    notes: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class FlightState:
    """Ambient state used for net thrust."""
    Pa: float
    Vinf: float


@dataclass(frozen=True)
class ThrustBreakdown:
    """Gross thrust, ram drag, and net thrust."""
    Fg: float
    Fram: float
    Fnet: float
    mdot_air: float
    Vinf: float


@dataclass
class ComponentResult:
    """Output from a component calculation."""
    station_out: Station
    warnings: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class NozzleInputs:
    """Nozzle operating specification."""
    mode: Literal["fixed", "matched"] = "fixed"
    Ae_At_fixed: float = 1.0
    Pa: float = 101325.0

    Ae_max: Optional[float] = None

    capacity_tol: float = 0.01


@dataclass
class NozzleResult:
    """Nozzle state, capacity check, and gross thrust terms."""
    mode: Literal["fixed", "matched"]
    choked: bool

    At: float
    Ae: float
    Ae_At_used: float

    Me: float
    Te: float
    Pe: float
    Ve: float

    mdot_used_for_thrust: float
    mdot_cap_choked: Optional[float] = None
    rel_mdot_err: Optional[float] = None

    F_mom: float = 0.0
    F_press: float = 0.0
    Fg: float = 0.0


@dataclass
class EngineRunResult:
    """Stations, warnings, and scalar outputs from an engine run."""
    stations: List[Station]
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, float] = field(default_factory=dict)
