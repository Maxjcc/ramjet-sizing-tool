"""Compressible-flow relations used by the ramjet sizing model."""

from __future__ import annotations

import math
from typing import Callable, Tuple

def _check_gamma(gamma: float) -> None:
    if gamma <= 1.0:
        raise ValueError(f"gamma must be > 1.0, got {gamma}")


def _check_mach(M: float) -> None:
    if M <= 0.0:
        raise ValueError(f"Mach number must be > 0, got {M}")

def Tt_over_T(M: float, gamma: float = 1.4) -> float:
    """
    Total-to-static temperature ratio, Tt/T, for isentropic flow.

    Tt/T = 1 + (gamma - 1)/2 * M^2
    """
    _check_gamma(gamma)
    _check_mach(M)
    return 1.0 + 0.5 * (gamma - 1.0) * M * M


def Pt_over_P(M: float, gamma: float = 1.4) -> float:
    """
    Total-to-static pressure ratio, Pt/P, for isentropic flow.

    Pt/P = (Tt/T)^(gamma/(gamma-1))
    """
    tt_t = Tt_over_T(M, gamma)
    return tt_t ** (gamma / (gamma - 1.0))


def rho_t_over_rho(M: float, gamma: float = 1.4) -> float:
    """
    Total-to-static density ratio, rhot/rho, for isentropic flow.

    rhot/rho = (Tt/T)^(1/(gamma-1))
    """
    tt_t = Tt_over_T(M, gamma)
    return tt_t ** (1.0 / (gamma - 1.0))


def A_over_Astar(M: float, gamma: float = 1.4) -> float:
    """
    Area-Mach relation (forward direction): A/A* as a function of Mach.

    A/A* = (1/M) * [ (2/(gamma+1)) * (1 + (gamma-1)/2 * M^2) ]^((gamma+1)/(2*(gamma-1)))
    """
    _check_gamma(gamma)
    _check_mach(M)

    term1 = 1.0 / M
    term2 = (2.0 / (gamma + 1.0)) * Tt_over_T(M, gamma)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return term1 * (term2 ** exponent)


def mach_from_area(A_Astar: float, gamma: float = 1.4, branch: str = "subsonic") -> float:
    """Invert the area-Mach relation on the selected branch."""
    _check_gamma(gamma)
    if A_Astar < 1.0:
        raise ValueError(f"mach_from_area: A/A* must be >= 1. Got {A_Astar}")

    if branch == "subsonic":
        bracket = (1e-8, 0.999999)
    elif branch == "supersonic":
        bracket = (1.000001, 20.0)
    else:
        raise ValueError("mach_from_area: branch must be 'subsonic' or 'supersonic'.")

    func = lambda M: A_over_Astar(M, gamma)
    amin, amax = _range_on_bracket(func, bracket)

    if not (amin <= A_Astar <= amax):
        raise ValueError(
            "mach_from_area: no solution on requested branch.\n"
            f"  branch   = {branch}\n"
            f"  gamma    = {gamma}\n"
            f"  A/A*     = {A_Astar:.6e}\n"
            f"  A/A* range on branch = [{amin:.6e}, {amax:.6e}]"
        )

    return solve_bisection(func, A_Astar, bracket)

def f5(M: float, gamma: float = 1.4) -> float:
    """
    Johnson Ch.4 f5(M): massflow-conserving duct function.

    f5 = M * (Tt/T)^(0.5 - gamma/(gamma-1))

    Used for area changes and total-pressure losses without heat or mass addition.
    """
    _check_gamma(gamma)
    _check_mach(M)

    tt_t = Tt_over_T(M, gamma)
    exponent = 0.5 - gamma / (gamma - 1.0)
    return M * (tt_t ** exponent)


def f6(M: float, gamma: float = 1.4) -> float:
    """
    Johnson Ch.4 f6(M): momentum/stream-thrust conserving function.

    f6 = M * sqrt(gamma * (Tt/T)) / (1 + gamma*M^2)

    Used across the combustor with heat and optional mass addition.
    """
    _check_gamma(gamma)
    _check_mach(M)

    tt_t = Tt_over_T(M, gamma)
    numerator = M * math.sqrt(gamma * tt_t)
    denominator = 1.0 + gamma * M * M
    return numerator / denominator

def solve_bisection(
    func: Callable[[float], float],
    target: float,
    bracket: Tuple[float, float],
    tol: float = 1e-10,
    max_iter: int = 200
) -> float:
    """Solve func(x) = target on a sign-changing bracket."""
    a, b = bracket
    if a <= 0.0 or b <= 0.0:
        raise ValueError("Bracket Mach values must be > 0.")
    if a >= b:
        raise ValueError("Bracket must satisfy a < b.")

    fa = func(a) - target
    fb = func(b) - target

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError(
            "Bisection requires a sign change across the bracket.\n"
            f"At a={a}: func-target={fa}\n"
            f"At b={b}: func-target={fb}\n"
            "Choose a bracket that contains the root."
        )

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = func(c) - target

        if abs(fc) < tol or (b - a) < tol:
            return c

        if fa * fc < 0.0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return 0.5 * (a + b)

def _range_on_bracket(func: Callable[[float], float], bracket: Tuple[float, float]) -> Tuple[float, float]:
    """Return the endpoint range of a monotone function over a bracket."""
    a, b = bracket
    fa = func(a)
    fb = func(b)
    return (min(fa, fb), max(fa, fb))

def mach_from_f5(f_target: float, gamma: float = 1.4, branch: str = "subsonic") -> float:
    """Invert f5(M) on the selected branch."""
    _check_gamma(gamma)
    if f_target <= 0.0:
        raise ValueError("mach_from_f5: f_target must be > 0.")

    if branch == "subsonic":
        bracket = (1e-8, 0.999999)
    elif branch == "supersonic":
        bracket = (1.000001, 20.0)
    else:
        raise ValueError("mach_from_f5: branch must be 'subsonic' or 'supersonic'.")

    func = lambda M: f5(M, gamma)
    fmin, fmax = _range_on_bracket(func, bracket)

    if not (fmin <= f_target <= fmax):
        raise ValueError(
            "mach_from_f5: no solution on requested branch.\n"
            f"  branch   = {branch}\n"
            f"  gamma    = {gamma}\n"
            f"  f_target = {f_target:.6e}\n"
            f"  f5 range on branch = [{fmin:.6e}, {fmax:.6e}]"
        )

    return solve_bisection(func, f_target, bracket)


def mach_from_f6(f_target: float, gamma: float = 1.4, branch: str = "subsonic") -> float:
    """Invert f6(M) on the selected branch."""
    _check_gamma(gamma)
    if f_target <= 0.0:
        raise ValueError("mach_from_f6: f_target must be > 0.")

    if branch == "subsonic":
        bracket = (1e-8, 0.999999)
    elif branch == "supersonic":
        bracket = (1.000001, 20.0)
    else:
        raise ValueError("mach_from_f6: branch must be 'subsonic' or 'supersonic'.")

    func = lambda M: f6(M, gamma)
    fmin, fmax = _range_on_bracket(func, bracket)

    if not (fmin <= f_target <= fmax):
        raise ValueError(
            "mach_from_f6: no solution on requested branch.\n"
            f"  branch   = {branch}\n"
            f"  gamma    = {gamma}\n"
            f"  f_target = {f_target:.6e}\n"
            f"  f6 range on branch = [{fmin:.6e}, {fmax:.6e}]"
        )

    return solve_bisection(func, f_target, bracket)

def bisect_root(
    func: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = 1e-10,
    max_iter: int = 200
) -> float:
    """Solve func(x) = 0 over [lo, hi]."""
    return solve_bisection(func, 0.0, (lo, hi), tol=tol, max_iter=max_iter)
