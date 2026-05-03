from __future__ import annotations

import math
from typing import Optional

import gasdyn
from engine_types import GasProps, Station, ComponentResult, NozzleInputs, NozzleResult

def fill_station_statics(st: Station, gas: GasProps) -> None:
    """Populate static properties from total state and Mach number."""
    gamma = gas.gamma
    R = gas.R

    st.T = st.Tt / gasdyn.Tt_over_T(st.M, gamma)
    st.P = st.Pt / gasdyn.Pt_over_P(st.M, gamma)
    st.rho = st.P / (R * st.T)
    st.a = math.sqrt(gamma * R * st.T)
    st.V = st.M * st.a


def mdot_from_totals(M: float, Pt: float, Tt: float, A: float, gas: GasProps) -> float:
    """Mass flow from total state, area, and Mach using the Johnson f5 form."""
    gamma = gas.gamma
    R = gas.R
    return (Pt * A / math.sqrt(Tt)) * math.sqrt(gamma / R) * gasdyn.f5(M, gamma)


def ensure_mdot(st: Station, gas: GasProps) -> None:
    """Populate station mass flow if it has not already been set."""
    if st.mdot is None:
        st.mdot = mdot_from_totals(st.M, st.Pt, st.Tt, st.A, gas)


def mdot_choked_capacity(Pt: float, Tt: float, At: float, gas: GasProps) -> float:
    """Choked mass-flow capacity at throat area At."""
    return mdot_from_totals(1.0, Pt, Tt, At, gas)

def duct_f5_step(
    st_in: Station,
    gas: GasProps,
    *,
    name_out: str,
    area_ratio: float = 1.0,
    pt_ratio: float = 1.0,
    branch: str = "subsonic",
    mdot_override: Optional[float] = None,
) -> ComponentResult:
    """Duct, diffuser, or loss element using Johnson f5."""
    warnings: list[str] = []

    if area_ratio <= 0.0:
        raise ValueError(f"duct_f5_step: area_ratio must be > 0, got {area_ratio}")
    if pt_ratio <= 0.0:
        raise ValueError(f"duct_f5_step: pt_ratio must be > 0, got {pt_ratio}")
    if branch not in ("subsonic", "supersonic"):
        raise ValueError("duct_f5_step: branch must be 'subsonic' or 'supersonic'")

    fill_station_statics(st_in, gas)
    ensure_mdot(st_in, gas)

    mdot_used = mdot_override if mdot_override is not None else st_in.mdot
    assert mdot_used is not None

    Tt_out = st_in.Tt
    Pt_out = st_in.Pt * pt_ratio
    A_out = st_in.A * area_ratio

    gamma = gas.gamma
    R = gas.R

    f5_target = mdot_used * math.sqrt(Tt_out) / (Pt_out * A_out) * math.sqrt(R / gamma)

    M_out = gasdyn.mach_from_f5(f5_target, gamma=gamma, branch=branch)

    st_out = Station(
        name=name_out,
        M=M_out,
        Tt=Tt_out,
        Pt=Pt_out,
        A=A_out,
        mdot=mdot_used,
        notes=[],
    )
    fill_station_statics(st_out, gas)

    if branch == "subsonic" and M_out > 0.95:
        warnings.append(f"{name_out}: Mach is close to 1 (M={M_out:.3f}); check choking risk.")
    if branch == "supersonic" and M_out < 1.05:
        warnings.append(
            f"{name_out}: supersonic branch selected but M={M_out:.3f} is near 1; verify branch choice."
        )
    if pt_ratio < 1.0:
        st_out.notes.append(f"Pt loss applied: Pt_out/Pt_in = {pt_ratio:.4f}")
    if area_ratio != 1.0:
        st_out.notes.append(f"Area change applied: A_out/A_in = {area_ratio:.4f}")

    return ComponentResult(station_out=st_out, warnings=warnings)

def combustor_max_Tt_out_f6(
    *,
    st_in: Station,
    gas: GasProps,
    fuel_air_ratio: float,
    branch: str = "subsonic",
) -> float:
    """Maximum Tt_out before the f6 inversion leaves the selected branch."""
    if branch not in ("subsonic", "supersonic"):
        raise ValueError("combustor_max_Tt_out_f6: branch must be 'subsonic' or 'supersonic'")
    if fuel_air_ratio < 0.0:
        raise ValueError("combustor_max_Tt_out_f6: fuel_air_ratio must be >= 0")

    fill_station_statics(st_in, gas)
    ensure_mdot(st_in, gas)

    gamma = gas.gamma
    f6_in = gasdyn.f6(st_in.M, gamma)
    denom = f6_in * (1.0 + fuel_air_ratio)

    if denom <= 0.0:
        return float("nan")

    if branch == "subsonic":
        bracket = (1e-8, 0.999999)
    else:
        bracket = (1.000001, 20.0)

    f6_min, f6_max = gasdyn._range_on_bracket(lambda M: gasdyn.f6(M, gamma), bracket)
    return st_in.Tt * (f6_max / denom) ** 2

def combustor_f6_step(
    st_in: Station,
    gas: GasProps,
    *,
    name_out: str,
    Tt_out: float,
    pt_ratio: float = 1.0,
    fuel_air_ratio: float = 0.0,
    area_ratio: float = 1.0,
    branch: str = "subsonic",
) -> ComponentResult:
    """Combustor step with prescribed Tt_out and f6-based Mach inversion."""
    warnings: list[str] = []

    if Tt_out <= 0.0:
        raise ValueError(f"combustor_f6_step: Tt_out must be > 0, got {Tt_out}")
    if pt_ratio <= 0.0:
        raise ValueError(f"combustor_f6_step: pt_ratio must be > 0, got {pt_ratio}")
    if area_ratio <= 0.0:
        raise ValueError(f"combustor_f6_step: area_ratio must be > 0, got {area_ratio}")
    if fuel_air_ratio < 0.0:
        raise ValueError(f"combustor_f6_step: fuel_air_ratio must be >= 0, got {fuel_air_ratio}")
    if branch not in ("subsonic", "supersonic"):
        raise ValueError("combustor_f6_step: branch must be 'subsonic' or 'supersonic'")

    if pt_ratio > 1.0:
        warnings.append(f"{name_out}: pt_ratio > 1 implies total-pressure gain; verify inputs.")
    if Tt_out < st_in.Tt:
        warnings.append(f"{name_out}: Tt_out < Tt_in (cooling step); verify intent.")
    if fuel_air_ratio > 0.3:
        warnings.append(f"{name_out}: fuel_air_ratio={fuel_air_ratio:.3f} is unusually high; verify fuel model.")

    fill_station_statics(st_in, gas)
    ensure_mdot(st_in, gas)
    mdot_air = st_in.mdot
    assert mdot_air is not None

    gamma = gas.gamma

    Pt_out = st_in.Pt * pt_ratio
    A_out = st_in.A * area_ratio
    mdot_out = mdot_air * (1.0 + fuel_air_ratio)

    f6_in = gasdyn.f6(st_in.M, gamma)
    f6_target = f6_in * (1.0 + fuel_air_ratio) * math.sqrt(Tt_out / st_in.Tt)

    if branch == "subsonic":
        bracket = (1e-8, 0.999999)
    else:
        bracket = (1.000001, 20.0)

    f6_min, f6_max = gasdyn._range_on_bracket(lambda M: gasdyn.f6(M, gamma), bracket)

    if not (f6_min <= f6_target <= f6_max):
        denom = f6_in * (1.0 + fuel_air_ratio)
        if denom > 0.0:
            Tt_out_max = st_in.Tt * (f6_max / denom) ** 2
            warnings.append(
                f"{name_out}: requested heat addition would choke flow on {branch} branch. "
                f"Max achievable Tt_out ≈ {Tt_out_max:.1f} K for current M_in and f/a."
            )
        raise ValueError(
            f"combustor_f6_step: f6_target={f6_target:.6g} outside {branch} range "
            f"[{f6_min:.6g}, {f6_max:.6g}]. Likely combustor choking / excessive heat input."
        )

    M_out = gasdyn.mach_from_f6(f6_target, gamma=gamma, branch=branch)

    st_out = Station(
        name=name_out,
        M=M_out,
        Tt=Tt_out,
        Pt=Pt_out,
        A=A_out,
        mdot=mdot_out,
        notes=[],
    )
    fill_station_statics(st_out, gas)

    if branch == "subsonic" and M_out > 0.95:
        warnings.append(f"{name_out}: Mach is close to 1 (M={M_out:.3f}); choking risk in combustor.")
    if branch == "supersonic" and M_out < 1.05:
        warnings.append(
            f"{name_out}: supersonic branch selected but M={M_out:.3f} is near 1; verify branch choice."
        )

    if pt_ratio < 1.0:
        st_out.notes.append(f"Pt loss applied: Pt_out/Pt_in = {pt_ratio:.4f}")
    if area_ratio != 1.0:
        st_out.notes.append(f"Area change applied: A_out/A_in = {area_ratio:.4f}")
    st_out.notes.append(
        f"Combustor: Tt_in={st_in.Tt:.1f} K -> Tt_out={Tt_out:.1f} K, f/a={fuel_air_ratio:.4f}, branch={branch}"
    )

    return ComponentResult(station_out=st_out, warnings=warnings)

def pstar_over_Pt(gamma: float) -> float:
    """Critical static-to-total pressure ratio at M=1."""
    return 1.0 / gasdyn.Pt_over_P(1.0, gamma)


def mach_from_Pt_over_P(Pt_over_P_target: float, *, gamma: float, branch: str) -> float:
    """Invert Pt/P on the selected Mach branch."""
    if Pt_over_P_target < 1.0:
        raise ValueError(f"Pt/P must be >= 1. Got {Pt_over_P_target}")

    if branch == "subsonic":
        bracket = (1e-8, 0.999999)
    elif branch == "supersonic":
        bracket = (1.000001, 20.0)
    else:
        raise ValueError("branch must be 'subsonic' or 'supersonic'")

    func = lambda M: gasdyn.Pt_over_P(M, gamma)
    return gasdyn.solve_bisection(func, Pt_over_P_target, bracket)

def nozzle_step(
    st_in: Station,
    gas: GasProps,
    *,
    name_out: str,
    At: float,
    spec: NozzleInputs,
    exit_branch_fixed: str = "supersonic",
) -> tuple[ComponentResult, NozzleResult]:
    """Nozzle step with choked and unchoked operating modes."""
    warnings: list[str] = []

    if At <= 0.0:
        raise ValueError(f"nozzle_step: At must be > 0, got {At}")
    if spec.Pa <= 0.0:
        warnings.append(f"{name_out}: Pa <= 0 is unusual; check ambient pressure input.")

    fill_station_statics(st_in, gas)
    ensure_mdot(st_in, gas)
    mdot_in = st_in.mdot
    assert mdot_in is not None

    gamma = gas.gamma
    R = gas.R
    Pt = st_in.Pt
    Tt = st_in.Tt
    Pa = spec.Pa

    pcrit = Pt * pstar_over_Pt(gamma)
    can_choke = Pa < pcrit
    choked = can_choke

    mdot_cap = mdot_choked_capacity(Pt=Pt, Tt=Tt, At=At, gas=gas)

    Ae: float
    Ae_At_used: float
    Me: float

    if choked:
        rel_err = (mdot_in - mdot_cap) / mdot_cap if mdot_cap != 0.0 else float("inf")
        if abs(rel_err) > spec.capacity_tol:
            if mdot_in > mdot_cap:
                warnings.append(
                    f"{name_out}: mdot_in ({mdot_in:.6g}) exceeds choked capacity ({mdot_cap:.6g}) "
                    f"by {rel_err*100:.2f}%. Increase At or adjust upstream totals."
                )
            else:
                warnings.append(
                    f"{name_out}: mdot_in ({mdot_in:.6g}) is below choked capacity ({mdot_cap:.6g}) "
                    f"by {abs(rel_err)*100:.2f}%. Nozzle is underfilled for this At."
                )

        if spec.mode == "fixed":
            Ae_At_used = spec.Ae_At_fixed
            if Ae_At_used <= 0.0:
                raise ValueError(f"{name_out}: Ae_At_fixed must be > 0, got {Ae_At_used}")

            Ae = At * Ae_At_used

            if spec.Ae_max is not None and Ae > spec.Ae_max:
                Ae = spec.Ae_max
                Ae_At_used = Ae / At
                warnings.append(f"{name_out}: Ae clamped by Ae_max; using Ae/At={Ae_At_used:.4f}")

            Me = gasdyn.mach_from_area(Ae_At_used, gamma=gamma, branch=exit_branch_fixed)

        elif spec.mode == "matched":
            Pt_over_Pa = Pt / Pa
            try:
                Me_match = mach_from_Pt_over_P(Pt_over_Pa, gamma=gamma, branch="supersonic")
            except ValueError:
                warnings.append(
                    f"{name_out}: cannot match Pe to Pa with supersonic expansion at this Pt/Pa; "
                    f"falling back to Me=1 (Ae/At=1)."
                )
                Me_match = 1.0

            Ae_At_used = gasdyn.A_over_Astar(Me_match, gamma=gamma)
            Ae = At * Ae_At_used

            if spec.Ae_max is not None and Ae > spec.Ae_max:
                Ae = spec.Ae_max
                Ae_At_used = Ae / At
                warnings.append(
                    f"{name_out}: matched nozzle limited by Ae_max; cannot fully match ambient. "
                    f"Using clamped Ae/At={Ae_At_used:.4f}"
                )
                Me = gasdyn.mach_from_area(Ae_At_used, gamma=gamma, branch="supersonic")
            else:
                Me = Me_match

        else:
            raise ValueError(f"{name_out}: unknown nozzle mode '{spec.mode}'")

    else:
        warnings.append(
            f"{name_out}: nozzle cannot choke because Pa >= P* (Pa={Pa:.3g}, P*={pcrit:.3g}). "
            "Treating as unchoked (convergent-like) with Pe=Pa."
        )

        Ae = At
        Ae_At_used = 1.0
        Pt_over_Pa = Pt / Pa
        Me = mach_from_Pt_over_P(Pt_over_Pa, gamma=gamma, branch="subsonic")

    Te = Tt / gasdyn.Tt_over_T(Me, gamma)
    Pe = Pt / gasdyn.Pt_over_P(Me, gamma)
    a_e = math.sqrt(gamma * R * Te)
    Ve = Me * a_e

    F_mom = mdot_in * Ve
    F_press = (Pe - Pa) * Ae
    Fg = F_mom + F_press

    st_out = Station(
        name=name_out,
        M=Me,
        Tt=Tt,
        Pt=Pt,
        A=Ae,
        mdot=mdot_in,
        notes=[],
    )
    fill_station_statics(st_out, gas)

    st_out.notes.append(f"Nozzle mode={spec.mode}, choked={choked}")
    st_out.notes.append(f"At={At:.6g} m^2, Ae={Ae:.6g} m^2, Ae/At={Ae_At_used:.6g}")
    st_out.notes.append(f"Exit: Me={Me:.6g}, Ve={Ve:.6g} m/s, Pe={Pe:.6g} Pa, Pa={Pa:.6g} Pa")
    st_out.notes.append(f"Gross thrust: Fg = mdot*Ve + (Pe-Pa)*Ae = {Fg:.6g} N")

    noz = NozzleResult(
        mode=spec.mode,
        choked=choked,
        At=At,
        Ae=Ae,
        Ae_At_used=Ae_At_used,
        Me=Me,
        Te=Te,
        Pe=Pe,
        Ve=Ve,
        mdot_used_for_thrust=mdot_in,
        mdot_cap_choked=(mdot_cap if choked else None),
        rel_mdot_err=((mdot_in - mdot_cap) / mdot_cap if (choked and mdot_cap != 0.0) else None),
        F_mom=F_mom,
        F_press=F_press,
        Fg=Fg,
    )

    return ComponentResult(station_out=st_out, warnings=warnings), noz
