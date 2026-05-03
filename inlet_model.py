from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal, List

from engine_types import GasProps

InletType = Literal[
    "pitot",
    "nwc_low_cost_underslung",   # Johnson Ch.7 Case 6
]


@dataclass(frozen=True)
class InletInputs:
    M_inf: float
    P_inf: float      # Pa
    T_inf: float      # K

    inlet_type: InletType = "pitot"

    A_cowl: float = 0.01   # m^2

    M2: float = 0.30

    PM: float = 0.0   # pressure margin
    SM: float = 0.0   # spillage margin

    diffuser_pt_ratio: float = 1.0


@dataclass
class InletResult:
    inlet_type: str

    # Freestream
    M_inf: float
    P_inf: float
    T_inf: float
    V_inf: float
    rho_inf: float
    q_inf: float

    # Freestream stagnation
    Pt_inf: float
    Tt_inf: float

    # Critical performance
    PRCR: float
    ARCR: float
    CDadd: float

    # Actual off-critical performance
    PR: float
    AR: float

    # Delivered station 2
    M2: float
    Pt2: float
    Tt2: float

    # Trial geometry / drag
    A_cowl: float
    A_capture: float
    mdot_capture: float
    F_add: float

    warnings: List[str] = field(default_factory=list)


@dataclass
class InletSizingResult:
    """Inlet area required for a specified air mass flow."""
    mdot_air_required: float
    A_capture_required: float
    A_cowl_required: float
    F_add_required: float

def speed_of_sound(T: float, gas: GasProps) -> float:
    if T <= 0.0:
        raise ValueError(f"speed_of_sound: T must be > 0, got {T}")
    return math.sqrt(gas.gamma * gas.R * T)


def freestream_velocity(M: float, T: float, gas: GasProps) -> float:
    if M <= 0.0:
        raise ValueError(f"freestream_velocity: M must be > 0, got {M}")
    return M * speed_of_sound(T, gas)


def freestream_density(P: float, T: float, gas: GasProps) -> float:
    if P <= 0.0:
        raise ValueError(f"freestream_density: P must be > 0, got {P}")
    if T <= 0.0:
        raise ValueError(f"freestream_density: T must be > 0, got {T}")
    return P / (gas.R * T)


def stagnation_temperature(T: float, M: float, gas: GasProps) -> float:
    return T * (1.0 + 0.5 * (gas.gamma - 1.0) * M * M)


def stagnation_pressure(P: float, M: float, gas: GasProps) -> float:
    tt_t = 1.0 + 0.5 * (gas.gamma - 1.0) * M * M
    return P * tt_t ** (gas.gamma / (gas.gamma - 1.0))

def linear_interp_clamped(x: float, xp: list[float], fp: list[float]) -> float:
    if len(xp) != len(fp):
        raise ValueError("linear_interp_clamped: xp and fp must be same length")
    if len(xp) < 2:
        raise ValueError("linear_interp_clamped: need at least 2 points")
    if any(xp[i] >= xp[i + 1] for i in range(len(xp) - 1)):
        raise ValueError("linear_interp_clamped: xp must be strictly increasing")

    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]

    for i in range(len(xp) - 1):
        x0, x1 = xp[i], xp[i + 1]
        if x0 <= x <= x1:
            y0, y1 = fp[i], fp[i + 1]
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    return fp[-1]

def normal_shock_M2(M1: float, gamma: float = 1.4) -> float:
    if M1 <= 1.0:
        raise ValueError(f"normal_shock_M2: M1 must be > 1, got {M1}")
    num = 1.0 + 0.5 * (gamma - 1.0) * M1 * M1
    den = gamma * M1 * M1 - 0.5 * (gamma - 1.0)
    return math.sqrt(num / den)


def normal_shock_static_pressure_ratio(M1: float, gamma: float = 1.4) -> float:
    if M1 <= 1.0:
        raise ValueError(f"normal_shock_static_pressure_ratio: M1 must be > 1, got {M1}")
    return 1.0 + (2.0 * gamma / (gamma + 1.0)) * (M1 * M1 - 1.0)


def isentropic_Pt_over_P(M: float, gamma: float = 1.4) -> float:
    if M <= 0.0:
        raise ValueError(f"isentropic_Pt_over_P: M must be > 0, got {M}")
    return (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (gamma / (gamma - 1.0))


def pitot_critical_recovery(M_inf: float, gas: GasProps) -> float:
    """Critical pressure recovery for a pitot inlet with a normal shock."""
    if M_inf <= 0.0:
        raise ValueError(f"pitot_critical_recovery: M_inf must be > 0, got {M_inf}")

    if M_inf <= 1.0:
        return 1.0

    gamma = gas.gamma
    M2n = normal_shock_M2(M_inf, gamma=gamma)
    p2_p1 = normal_shock_static_pressure_ratio(M_inf, gamma=gamma)

    pt1_p1 = isentropic_Pt_over_P(M_inf, gamma=gamma)
    pt2_p2 = isentropic_Pt_over_P(M2n, gamma=gamma)

    return (pt2_p2 * p2_p1) / pt1_p1

# Johnson Ch.7 Case 6: NWC Low-Cost Underslung, MD = 3.23.
_CASE6_M_REC =    [2.00, 2.50, 3.00, 3.23, 3.50, 4.00, 4.50]
_CASE6_PRCR =     [0.86, 0.75, 0.53, 0.42, 0.33, 0.22, 0.16]
_CASE6_ARCR =     [0.51, 0.67, 0.83, 0.90, 0.908, 0.923, 0.93]

_CASE6_M_CD =     [1.83, 2.53, 2.73, 2.93, 3.23, 3.73, 4.23, 4.73, 5.13, 5.48, 5.73, 6.73]
_CASE6_CDADD =    [0.536, 0.322, 0.255, 0.193, 0.102, 0.083, 0.074, 0.068, 0.060, 0.056, 0.053, 0.050]


def nwc_low_cost_underslung_criticals(M_inf: float) -> tuple[float, float, float]:
    prcr = linear_interp_clamped(M_inf, _CASE6_M_REC, _CASE6_PRCR)
    arcr = linear_interp_clamped(M_inf, _CASE6_M_REC, _CASE6_ARCR)
    cdadd = linear_interp_clamped(M_inf, _CASE6_M_CD, _CASE6_CDADD)
    return prcr, arcr, cdadd

def apply_johnson_margins(PRCR: float, ARCR: float, PM: float, SM: float) -> tuple[float, float]:
    if PM < 0.0 or SM < 0.0:
        raise ValueError("apply_johnson_margins: PM and SM must be >= 0")
    if PM > 0.0 and SM > 0.0:
        raise ValueError("apply_johnson_margins: only one of PM or SM may be nonzero at a time")
    if PM >= 1.0:
        raise ValueError(f"apply_johnson_margins: PM must be < 1, got {PM}")
    if SM >= 1.0:
        raise ValueError(f"apply_johnson_margins: SM must be < 1, got {SM}")

    PR = PRCR * (1.0 - PM)
    AR = ARCR * (1.0 - SM)
    return PR, AR

def run_inlet(inp: InletInputs, gas: GasProps) -> InletResult:
    warnings: list[str] = []

    if inp.A_cowl <= 0.0:
        raise ValueError(f"run_inlet: A_cowl must be > 0, got {inp.A_cowl}")
    if inp.M2 <= 0.0:
        raise ValueError(f"run_inlet: M2 must be > 0, got {inp.M2}")
    if inp.diffuser_pt_ratio <= 0.0:
        raise ValueError(
            f"run_inlet: diffuser_pt_ratio must be > 0, got {inp.diffuser_pt_ratio}"
        )

    rho_inf = freestream_density(inp.P_inf, inp.T_inf, gas)
    V_inf = freestream_velocity(inp.M_inf, inp.T_inf, gas)
    q_inf = 0.5 * rho_inf * V_inf * V_inf

    Tt_inf = stagnation_temperature(inp.T_inf, inp.M_inf, gas)
    Pt_inf = stagnation_pressure(inp.P_inf, inp.M_inf, gas)

    if inp.inlet_type == "pitot":
        PRCR = pitot_critical_recovery(inp.M_inf, gas) * inp.diffuser_pt_ratio
        ARCR = 1.0
        CDadd = 0.0

        if inp.M_inf > 2.5:
            warnings.append(
                "Pitot inlet selected at high Mach number; total-pressure recovery will be poor."
            )

    elif inp.inlet_type == "nwc_low_cost_underslung":
        PRCR, ARCR, CDadd = nwc_low_cost_underslung_criticals(inp.M_inf)
        PRCR *= inp.diffuser_pt_ratio

        if inp.M_inf < 1.5:
            warnings.append(
                "High-speed external-compression inlet used below about Mach 1.5; attached-shock operation is not realistic there."
            )
        if inp.M_inf < _CASE6_M_REC[0] or inp.M_inf > _CASE6_M_REC[-1]:
            warnings.append(
                "Freestream Mach lies outside the tabulated Case 6 recovery-data range; PRCR/ARCR are being clamped."
            )
        if inp.M_inf < _CASE6_M_CD[0] or inp.M_inf > _CASE6_M_CD[-1]:
            warnings.append(
                "Freestream Mach lies outside the tabulated Case 6 additive-drag range; CDadd is being clamped."
            )

    else:
        raise ValueError(f"run_inlet: unsupported inlet_type '{inp.inlet_type}'")

    PR, AR = apply_johnson_margins(PRCR=PRCR, ARCR=ARCR, PM=inp.PM, SM=inp.SM)

    Pt2 = Pt_inf * PR
    Tt2 = Tt_inf

    A_capture = inp.A_cowl * AR
    mdot_capture = rho_inf * V_inf * A_capture

    F_add = CDadd * inp.A_cowl * q_inf

    if inp.M2 >= 1.0:
        warnings.append(
            "M2 >= 1 specified. For the present ramjet architecture, subsonic delivered M2 is usually intended."
        )
    if PR < 0.2:
        warnings.append(
            f"Low inlet pressure recovery predicted (PR={PR:.3f}); check inlet suitability."
        )
    if AR < 0.5:
        warnings.append(
            f"Low capture-area recovery predicted (AR={AR:.3f}); inlet may be badly off-design."
        )

    return InletResult(
        inlet_type=inp.inlet_type,
        M_inf=inp.M_inf,
        P_inf=inp.P_inf,
        T_inf=inp.T_inf,
        V_inf=V_inf,
        rho_inf=rho_inf,
        q_inf=q_inf,
        Pt_inf=Pt_inf,
        Tt_inf=Tt_inf,
        PRCR=PRCR,
        ARCR=ARCR,
        CDadd=CDadd,
        PR=PR,
        AR=AR,
        M2=inp.M2,
        Pt2=Pt2,
        Tt2=Tt2,
        A_cowl=inp.A_cowl,
        A_capture=A_capture,
        mdot_capture=mdot_capture,
        F_add=F_add,
        warnings=warnings,
    )

def size_inlet_for_required_airflow(inlet: InletResult, mdot_air_required: float) -> InletSizingResult:
    """Invert the inlet model for required air mass flow."""
    if mdot_air_required <= 0.0:
        raise ValueError(
            f"size_inlet_for_required_airflow: mdot_air_required must be > 0, got {mdot_air_required}"
        )

    rhoV = inlet.rho_inf * inlet.V_inf
    if rhoV <= 0.0:
        raise ValueError("size_inlet_for_required_airflow: rho_inf * V_inf must be > 0")
    if inlet.AR <= 0.0:
        raise ValueError("size_inlet_for_required_airflow: inlet AR must be > 0")

    A_capture_required = mdot_air_required / rhoV
    A_cowl_required = A_capture_required / inlet.AR
    F_add_required = inlet.CDadd * A_cowl_required * inlet.q_inf

    return InletSizingResult(
        mdot_air_required=mdot_air_required,
        A_capture_required=A_capture_required,
        A_cowl_required=A_cowl_required,
        F_add_required=F_add_required,
    )
