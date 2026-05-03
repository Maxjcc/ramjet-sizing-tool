from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Literal, Tuple

from engine_types import GasProps, Station, EngineRunResult, NozzleInputs
from engine_core import (
    fill_station_statics,
    ensure_mdot,
    duct_f5_step,
    combustor_f6_step,
    combustor_max_Tt_out_f6,
    nozzle_step,
    mdot_choked_capacity,
)

def area_from_diameter(D: float) -> float:
    if D <= 0.0:
        raise ValueError(f"area_from_diameter: D must be > 0, got {D}")
    return math.pi * (D * D) / 4.0


def diameter_from_area(A: float) -> float:
    if A <= 0.0:
        raise ValueError(f"diameter_from_area: A must be > 0, got {A}")
    return math.sqrt(4.0 * A / math.pi)


def a2_bracket_from_combustor_area(
    A_comb: float,
    low_factor: float = 0.25,
    high_factor: float = 1.25,
    floor: float = 1.0e-8,
) -> tuple[float, float]:
    """Build the initial A2 search bracket from combustor area."""
    if A_comb <= 0.0:
        raise ValueError(f"a2_bracket_from_combustor_area: A_comb must be > 0, got {A_comb}")
    if low_factor <= 0.0 or high_factor <= 0.0:
        raise ValueError("a2_bracket_from_combustor_area: factors must be > 0")
    if high_factor <= low_factor:
        raise ValueError("a2_bracket_from_combustor_area: high_factor must exceed low_factor")

    A_ref = max(A_comb, floor)
    A2_low = max(floor, low_factor * A_ref)
    A2_high = max(A2_low * 1.01, high_factor * A_ref)
    return A2_low, A2_high

@dataclass(frozen=True)
class EngineSizingInputs:
    # Station-2 delivered totals (post-inlet)
    M2: float
    Tt2: float
    Pt2: float

    # Cycle / combustor
    fuel_air_ratio: float
    Tt4: float

    # Loss models
    pt_ratio_23: float
    pt_ratio_34: float
    pt_ratio_45: float

    # Geometry rules
    A5_A4: float
    Ae_A5_req: float

    # Packaging
    Dcomb_max: float
    De_max: float

    # Ambient / flight
    Pa: float
    nozzle_mode: Literal["fixed", "matched"] = "fixed"
    Vinf: float = 0.0

    # A2 inner loop controls
    A2_low: float = 1.0e-4
    A2_high: float = 1.0e-2
    A2_high_max: float = 0.25
    expand_factor: float = 1.5
    tol_rel: float = 0.001
    max_iter: int = 60

def ram_drag(mdot_air: float, Vinf: float) -> float:
    return mdot_air * Vinf

def size_ramjet_engine_1d(inp: EngineSizingInputs, gas: GasProps) -> EngineRunResult:
    """
    Size a 1-D ramjet by solving A2 against the fixed station-5 throat capacity.
    """

    warnings: list[str] = []
    iteration_count = 0

    A_comb = area_from_diameter(inp.Dcomb_max)
    A5_fixed = inp.A5_A4 * A_comb
    Ae_max = area_from_diameter(inp.De_max)

    def run_with_A2(A2: float) -> Tuple[Optional[EngineRunResult], Optional[float], Optional[float], str]:
        local_warnings: list[str] = []

        st2 = Station(name="2", M=inp.M2, Tt=inp.Tt2, Pt=inp.Pt2, A=A2, mdot=None, notes=[])
        fill_station_statics(st2, gas)
        ensure_mdot(st2, gas)

        area_ratio_23 = A_comb / A2
        try:
            res3 = duct_f5_step(
                st_in=st2,
                gas=gas,
                name_out="3",
                area_ratio=area_ratio_23,
                pt_ratio=inp.pt_ratio_23,
                branch="subsonic",
            )
        except ValueError as e:
            local_warnings.append(f"3: diffuser / duct infeasible at A2={A2:.6g}: {e}")
            return None, None, None, "duct_infeasible"

        st3 = res3.station_out
        st3.A = A_comb
        local_warnings += res3.warnings

        try:
            Tt4_max = combustor_max_Tt_out_f6(
                st_in=st3,
                gas=gas,
                fuel_air_ratio=inp.fuel_air_ratio,
                branch="subsonic",
            )
        except Exception:
            Tt4_max = float("nan")

        if Tt4_max == Tt4_max and inp.Tt4 > Tt4_max * (1.0 + 1e-9):
            local_warnings.append(
                f"4: requested Tt4={inp.Tt4:.1f} K exceeds combustor f6-limit "
                f"(Tt4_max≈{Tt4_max:.1f} K) for M3≈{st3.M:.3f}, f/a={inp.fuel_air_ratio:.4f}. "
                "Treating this A2 as infeasible."
            )
            return None, None, None, "combustor_infeasible"

        try:
            res4 = combustor_f6_step(
                st_in=st3,
                gas=gas,
                name_out="4",
                Tt_out=inp.Tt4,
                pt_ratio=inp.pt_ratio_34,
                fuel_air_ratio=inp.fuel_air_ratio,
                area_ratio=1.0,
                branch="subsonic",
            )
        except ValueError as e:
            local_warnings.append(f"4: combustor infeasible at A2={A2:.6g}: {e}")
            return None, None, None, "combustor_infeasible"

        st4 = res4.station_out
        st4.A = A_comb
        local_warnings += res4.warnings

        st5 = Station(
            name="5",
            M=1.0,
            Tt=st4.Tt,
            Pt=st4.Pt * inp.pt_ratio_45,
            A=A5_fixed,
            mdot=st4.mdot,
            notes=[],
        )
        fill_station_statics(st5, gas)

        mdot4 = st4.mdot
        assert mdot4 is not None

        mdot_cap_5 = mdot_choked_capacity(
            Pt=st5.Pt,
            Tt=st5.Tt,
            At=st5.A,
            gas=gas,
        )

        res = EngineRunResult(
            stations=[st2, st3, st4, st5],
            warnings=local_warnings,
            meta={"feasible": 1.0},
        )
        return res, mdot4, mdot_cap_5, "ok"

    def rel_err(mdot_target: float, cap: float) -> float:
        return (cap - mdot_target) / mdot_target

    best_trial: Optional[Tuple[EngineRunResult, float, float, float]] = None
    best_abs_err = float("inf")

    def update_best_trial(
        res: Optional[EngineRunResult],
        mdot4: Optional[float],
        cap: Optional[float],
        A2: float,
    ) -> None:
        nonlocal best_trial, best_abs_err

        if res is None or mdot4 is None or cap is None:
            return

        abs_err = abs(rel_err(mdot4, cap))
        if abs_err < best_abs_err:
            best_abs_err = abs_err
            best_trial = (res, mdot4, cap, A2)

    A2_lo = inp.A2_low
    A2_hi = inp.A2_high

    lo = run_with_A2(A2_lo)
    tries = 0
    while lo[0] is None and A2_lo < inp.A2_high_max and tries < 40:
        A2_lo *= inp.expand_factor
        lo = run_with_A2(A2_lo)
        tries += 1
    update_best_trial(lo[0], lo[1], lo[2], A2_lo)

    if lo[0] is None:
        return EngineRunResult(
            stations=[],
            warnings=[
                "No feasible low-A2 point found in search range. "
                "Likely diffuser or combustor infeasibility for the requested operating point."
            ] + warnings,
            meta={"feasible": 0.0},
        )

    res_lo, mdot4_lo, cap_lo = lo[0], lo[1], lo[2]
    assert res_lo is not None and mdot4_lo is not None and cap_lo is not None
    err_lo = rel_err(mdot4_lo, cap_lo)

    # Infeasible high-side trials are treated as beyond the usable A2 range.
    hi: Tuple[Optional[EngineRunResult], Optional[float], Optional[float], str] = (None, None, None, "")
    hi_found = False

    for _ in range(40):
        hi = run_with_A2(A2_hi)
        update_best_trial(hi[0], hi[1], hi[2], A2_hi)

        if hi[0] is None:
            A2_hi = 0.5 * (A2_lo + A2_hi)
        else:
            res_hi, mdot4_hi, cap_hi = hi[0], hi[1], hi[2]
            assert res_hi is not None and mdot4_hi is not None and cap_hi is not None

            err_hi = rel_err(mdot4_hi, cap_hi)
            if err_lo * err_hi <= 0.0:
                hi_found = True
                break

            if err_hi > 0.0:
                next_hi = A2_hi * inp.expand_factor
                if next_hi >= inp.A2_high_max:
                    break
                A2_hi = next_hi
            else:
                break

        if abs(A2_hi - A2_lo) <= max(A2_lo, 1.0e-12) * 1.0e-6:
            break

    if not hi_found or hi[0] is None:
        if best_trial is not None:
            res_sol, mdot4_sol, cap_sol, A2_sol = best_trial
            res_sol.warnings += [
                "Could not establish full A2 bracket; using best feasible trial found during search."
            ] + warnings
        else:
            return EngineRunResult(
                stations=[],
                warnings=[
                    "No feasible A2 point found in search range. "
                    "Likely diffuser or combustor infeasibility for the requested operating point."
                ] + warnings,
                meta={"feasible": 0.0},
            )
    else:
        res_lo, mdot4_lo, cap_lo = lo[0], lo[1], lo[2]
        res_hi, mdot4_hi, cap_hi = hi[0], hi[1], hi[2]

        assert res_lo is not None and mdot4_lo is not None and cap_lo is not None
        assert res_hi is not None and mdot4_hi is not None and cap_hi is not None

        err_lo = rel_err(mdot4_lo, cap_lo)
        err_hi = rel_err(mdot4_hi, cap_hi)

        update_best_trial(res_lo, mdot4_lo, cap_lo, A2_lo)
        update_best_trial(res_hi, mdot4_hi, cap_hi, A2_hi)

        expand_it = 0
        while err_lo * err_hi > 0.0 and A2_hi < inp.A2_high_max and expand_it < 40:
            A2_hi *= inp.expand_factor
            trial = run_with_A2(A2_hi)

            if trial[0] is None:
                break

            res_hi, mdot4_hi, cap_hi = trial[0], trial[1], trial[2]
            assert res_hi is not None and mdot4_hi is not None and cap_hi is not None

            update_best_trial(res_hi, mdot4_hi, cap_hi, A2_hi)

            err_hi = rel_err(mdot4_hi, cap_hi)
            expand_it += 1

        res_sol: Optional[EngineRunResult] = None
        mdot4_sol: Optional[float] = None
        cap_sol: Optional[float] = None
        A2_sol: Optional[float] = None

        if err_lo * err_hi <= 0.0:
            A2_low = A2_lo
            A2_high = A2_hi

            for it in range(inp.max_iter):
                iteration_count = it + 1
                A2_mid = 0.5 * (A2_low + A2_high)
                trial = run_with_A2(A2_mid)

                if trial[0] is None:
                    A2_high = A2_mid
                    continue

                res_mid, mdot4_mid, cap_mid = trial[0], trial[1], trial[2]
                assert res_mid is not None and mdot4_mid is not None and cap_mid is not None

                update_best_trial(res_mid, mdot4_mid, cap_mid, A2_mid)

                e_mid = rel_err(mdot4_mid, cap_mid)

                res_sol = res_mid
                mdot4_sol = mdot4_mid
                cap_sol = cap_mid
                A2_sol = A2_mid

                if abs(e_mid) < inp.tol_rel:
                    break

                if e_mid * err_lo > 0.0:
                    A2_low = A2_mid
                    err_lo = e_mid
                else:
                    A2_high = A2_mid
                    err_hi = e_mid

            if res_sol is None or mdot4_sol is None or cap_sol is None or A2_sol is None:
                if best_trial is not None:
                    res_sol, mdot4_sol, cap_sol, A2_sol = best_trial
                    iteration_count = 0
                    res_sol.warnings += [
                        "A2 solve did not converge cleanly; using best feasible trial by minimum massflow residual."
                    ] + warnings
                else:
                    return EngineRunResult(
                        stations=[],
                        warnings=["A2 solve failed unexpectedly."] + warnings,
                        meta={"feasible": 0.0},
                    )
        else:
            if best_trial is None:
                return EngineRunResult(
                    stations=[],
                    warnings=["Could not bracket A2 massflow match, and no feasible trial was saved."] + warnings,
                    meta={"feasible": 0.0},
                )

            res_sol, mdot4_sol, cap_sol, A2_sol = best_trial
            iteration_count = 0
            res_sol.warnings += [
                "Could not bracket A2 massflow match; using best feasible trial by minimum massflow residual."
            ] + warnings

    assert res_sol is not None and mdot4_sol is not None and cap_sol is not None and A2_sol is not None

    st2, st3, st4, st5 = res_sol.stations

    Ae_A5_req = inp.Ae_A5_req
    Ae_req = Ae_A5_req * A5_fixed
    if Ae_req > Ae_max:
        Ae_req = Ae_max
        warnings.append(
            f"Nozzle exit area clamped by De_max; using Ae/A5={Ae_req / A5_fixed:.4f}"
        )

    noz_spec = NozzleInputs(
        mode=inp.nozzle_mode,
        Ae_At_fixed=(Ae_req / A5_fixed) if inp.nozzle_mode == "fixed" else 1.0,
        Pa=inp.Pa,
        Ae_max=Ae_max,
        capacity_tol=0.01,
    )

    rese, noz = nozzle_step(
        st_in=st5,
        gas=gas,
        name_out="e",
        At=st5.A,
        spec=noz_spec,
        exit_branch_fixed="supersonic",
    )
    ste = rese.station_out
    fill_station_statics(ste, gas)
    ensure_mdot(ste, gas)

    mdot_air = st3.mdot
    assert mdot_air is not None

    Fram = ram_drag(mdot_air, inp.Vinf)
    Fnet = noz.Fg - Fram

    all_warnings = (res_sol.warnings or []) + (rese.warnings or []) + warnings

    meta = dict(res_sol.meta or {})
    meta.update({
        "feasible": 1.0,
        "A2_solution": float(A2_sol),
        "A2_internal": float(A2_sol),
        "A_comb": float(A_comb),
        "A5": float(A5_fixed),
        "Ae_max": float(Ae_max),
        "nozzle_choked": 1.0 if noz.choked else 0.0,
        "Ae_used": float(noz.Ae),
        "Ae_At_used": float(noz.Ae_At_used),
        "D2_equiv": float(diameter_from_area(A2_sol)),
        "D5_equiv": float(diameter_from_area(A5_fixed)),
        "De_used": float(diameter_from_area(noz.Ae)),
        "mdot_air_required": float(mdot_air),
        "mdot4": float(mdot4_sol),
        "mdot_cap": float(cap_sol),
        "Fg": float(noz.Fg),
        "Fram": float(Fram),
        "Fnet": float(Fnet),
        "iterations": float(iteration_count),
        "best_abs_rel_err": float(best_abs_err if best_abs_err < float("inf") else math.nan),
    })

    return EngineRunResult(
        stations=[st2, st3, st4, st5, ste],
        warnings=all_warnings,
        meta=meta,
    )
