from __future__ import annotations

from typing import Iterable

from engine_types import GasProps, Station
from engine_sizing import (
    EngineSizingInputs,
    size_ramjet_engine_1d,
    area_from_diameter,
    a2_bracket_from_combustor_area,
)
from inlet_model import (
    InletInputs,
    InletResult,
    InletSizingResult,
    run_inlet,
    size_inlet_for_required_airflow,
)


def fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "-"
    ax = abs(x)
    if (ax != 0.0) and (ax < 1e-3 or ax >= 1e6):
        return f"{x:.{nd}e}"
    return f"{x:.{nd}f}"


def print_inlet_summary(inlet: InletResult) -> None:
    print("\n================ INLET PERFORMANCE SUMMARY ================")
    print(f"  inlet_type   = {inlet.inlet_type}")
    print(f"  M_inf        = {inlet.M_inf}")
    print(f"  P_inf        = {inlet.P_inf} Pa")
    print(f"  T_inf        = {inlet.T_inf} K")
    print(f"  rho_inf      = {inlet.rho_inf} kg/m^3")
    print(f"  V_inf        = {inlet.V_inf} m/s")
    print(f"  q_inf        = {inlet.q_inf} Pa")
    print(f"  Pt_inf       = {inlet.Pt_inf} Pa")
    print(f"  Tt_inf       = {inlet.Tt_inf} K")
    print(f"  PRCR         = {inlet.PRCR}")
    print(f"  ARCR         = {inlet.ARCR}")
    print(f"  CDadd        = {inlet.CDadd}")
    print(f"  PR           = {inlet.PR}")
    print(f"  AR           = {inlet.AR}")
    print(f"  Pt2          = {inlet.Pt2} Pa")
    print(f"  Tt2          = {inlet.Tt2} K")
    print(f"  M2           = {inlet.M2}")
    print(f"  trial A_cowl = {inlet.A_cowl} m^2")
    print(f"  trial A_cap  = {inlet.A_capture} m^2")
    print(f"  trial mdot   = {inlet.mdot_capture} kg/s")
    print(f"  trial F_add  = {inlet.F_add} N")
    print("==========================================================\n")


def print_inputs(inp: EngineSizingInputs) -> None:
    print("\n================ ENGINE SIZING INPUTS ================")
    print("Station 2 (post-inlet delivery):")
    print(f"  M2   = {inp.M2}")
    print(f"  Tt2  = {inp.Tt2} K")
    print(f"  Pt2  = {inp.Pt2} Pa\n")

    print("Combustor / Cycle:")
    print(f"  Fuel-air ratio = {inp.fuel_air_ratio}")
    print(f"  Tt4            = {inp.Tt4} K\n")

    print("Total pressure ratios:")
    print(f"  Pt3/Pt2 = {inp.pt_ratio_23}")
    print(f"  Pt4/Pt3 = {inp.pt_ratio_34}")
    print(f"  Pt5/Pt4 = {inp.pt_ratio_45}\n")

    print("Geometry rules:")
    print(f"  A5/A4        = {inp.A5_A4}")
    print(f"  Ae/A5_req    = {inp.Ae_A5_req}")
    print(f"  nozzle_mode  = {inp.nozzle_mode}\n")

    print("Packaging limits:")
    print(f"  Dcomb_max  = {inp.Dcomb_max} m")
    print(f"  De_max     = {inp.De_max} m\n")

    print("Ambient / flight:")
    print(f"  Pa    = {inp.Pa} Pa")
    print(f"  Vinf  = {inp.Vinf} m/s")

    print("\nA2 search bracket:")
    print(f"  A2_low   = {inp.A2_low} m^2")
    print(f"  A2_high  = {inp.A2_high} m^2")
    print("=======================================================\n")


def print_station_table(stations: Iterable[Station]) -> None:
    print("=== Stations (2 → 3 → 4 → 5 → e) ===")

    header = (
        f"{'St':<3} | {'M':>8} | {'Tt[K]':>10} | {'Pt[Pa]':>12} | {'A[m^2]':>12} | {'mdot[kg/s]':>11} || "
        f"{'T[K]':>10} | {'P[Pa]':>12} | {'V[m/s]':>10}"
    )
    print(header)
    print("-" * len(header))

    for st in stations:
        row = (
            f"{st.name:<3} | "
            f"{fmt(st.M, 6):>8} | "
            f"{fmt(st.Tt, 2):>10} | "
            f"{fmt(st.Pt, 2):>12} | "
            f"{fmt(st.A, 6):>12} | "
            f"{fmt(st.mdot, 6):>11} || "
            f"{fmt(st.T, 2):>10} | "
            f"{fmt(st.P, 2):>12} | "
            f"{fmt(st.V, 2):>10}"
        )
        print(row)


def print_engine_summary(result) -> None:
    meta = result.meta

    def g(key: str) -> float | None:
        v = meta.get(key, None)
        return float(v) if v is not None else None

    print("\n=== Engine Performance Summary ===")
    print(f"  A2 (internal)       = {fmt(g('A2_internal'), 6)} m^2   (D2 equiv = {fmt(g('D2_equiv'), 4)} m)")
    print(f"  Acomb - A3&A4       = {fmt(g('A_comb'), 6)} m^2")
    print(f"  A5 (throat geom)    = {fmt(g('A5'), 6)} m^2   (D5 equiv = {fmt(g('D5_equiv'), 4)} m)")
    print(f"  Ae used             = {fmt(g('Ae_used'), 6)} m^2   (Ae/At used = {fmt(g('Ae_At_used'), 6)})")
    print(f"  Nozzle choked?      = {'YES' if g('nozzle_choked') == 1.0 else 'NO'}")
    print(f"  mdot_air_required   = {fmt(g('mdot_air_required'), 6)} kg/s")
    print(f"  mdot4 (incl fuel)   = {fmt(g('mdot4'), 6)} kg/s")
    print(f"  mdot_cap@throat     = {fmt(g('mdot_cap'), 6)} kg/s")

    if g("mdot4") is not None and g("mdot_cap") is not None and g("mdot4") != 0.0:
        mismatch = (g("mdot_cap") - g("mdot4")) / g("mdot4") * 100.0
        print(f"  throat mdot mismatch= {fmt(mismatch, 3)} % (cap - target)")

    print("\n  Thrust breakdown:")
    print(f"    Gross thrust Fg   = {fmt(g('Fg'), 3)} N")
    print(f"    Ram drag Fram     = {fmt(g('Fram'), 3)} N")
    print(f"    Net thrust Fnet   = {fmt(g('Fnet'), 3)} N")

    iters = g("iterations")
    if iters is not None:
        print(f"\n  A2 solve iterations = {int(round(iters))}")

    print("\n  Nozzle notes:")
    st_e = next((s for s in result.stations if s.name.lower() in ("e", "exit")), None)
    if st_e and st_e.notes:
        for note in st_e.notes:
            print(f"    - {note}")
    else:
        print("    - (none)")


def print_required_inlet_sizing(inlet: InletResult, sizing: InletSizingResult) -> None:
    capture_ratio = inlet.A_capture / sizing.A_capture_required
    cowl_ratio = inlet.A_cowl / sizing.A_cowl_required
    mdot_ratio = inlet.mdot_capture / sizing.mdot_air_required

    print("\n=== Required Inlet Sizing for Solved Engine ===")
    print(f"  required mdot_air   = {fmt(sizing.mdot_air_required, 6)} kg/s")
    print(f"  required A_capture  = {fmt(sizing.A_capture_required, 6)} m^2")
    print(f"  required A_cowl     = {fmt(sizing.A_cowl_required, 6)} m^2")
    print(f"  required F_add      = {fmt(sizing.F_add_required, 3)} N")

    print("\n  Trial inlet comparison:")
    print(f"    trial A_capture / required = {fmt(capture_ratio, 4)}")
    print(f"    trial A_cowl / required    = {fmt(cowl_ratio, 4)}")
    print(f"    trial mdot / required      = {fmt(mdot_ratio, 4)}")

    if capture_ratio < 1.0:
        print("    -> Trial inlet is undersized for the solved engine airflow.")
    elif capture_ratio > 1.05:
        print("    -> Trial inlet is oversized relative to the solved engine airflow.")
    else:
        print("    -> Trial inlet is close to the solved engine airflow requirement.")


def main() -> None:
    gas = GasProps(gamma=1.4, R=287.0)

    inlet_inp = InletInputs(
        M_inf=5,
        P_inf=2970,
        T_inf=220.6,

        inlet_type="nwc_low_cost_underslung",
        A_cowl=0.007,
        M2=0.35,

        PM=0.0,
        SM=0.0,

        diffuser_pt_ratio=0.98,
    )

    inlet_res = run_inlet(inlet_inp, gas)

    Dcomb_max = 0.1524
    A_comb_guess = area_from_diameter(Dcomb_max)
    A2_low, A2_high = a2_bracket_from_combustor_area(A_comb_guess)

    inp = EngineSizingInputs(
        M2=inlet_res.M2,
        Tt2=inlet_res.Tt2,
        Pt2=inlet_res.Pt2,

        fuel_air_ratio=0.04,
        Tt4=1950.0,

        pt_ratio_23=0.95,
        pt_ratio_34=0.95,
        pt_ratio_45=0.99,

        A5_A4=0.65,
        Ae_A5_req=1.05,
        nozzle_mode="matched",

        Dcomb_max=Dcomb_max,
        De_max=0.2,

        Pa=inlet_inp.P_inf,
        Vinf=inlet_res.V_inf,

        A2_low=A2_low,
        A2_high=A2_high,
    )

    print_inlet_summary(inlet_res)
    print_inputs(inp)

    result = size_ramjet_engine_1d(inp, gas=gas)
    sizing = size_inlet_for_required_airflow(
        inlet=inlet_res,
        mdot_air_required=float(result.meta["mdot_air_required"]),
    )

    print_station_table(result.stations)

    if inlet_res.warnings or result.warnings:
        print("\n=== WARNINGS ===")
        for w in inlet_res.warnings:
            print(" -", w)
        for w in result.warnings:
            print(" -", w)

    print_engine_summary(result)
    print_required_inlet_sizing(inlet_res, sizing)


if __name__ == "__main__":
    main()
