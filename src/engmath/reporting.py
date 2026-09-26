"""Reporting results with uncertainty and significant figures, and first-order propagation.

Convention used: round the uncertainty to ONE significant figure, or TWO if its leading digit
is 1; then round the value to the same decimal place.

>>> from engmath.reporting import format_uncertainty
>>> format_uncertainty(2.29987, 0.08907)
'2.30 ± 0.09'
>>> format_uncertainty(10523.4, 156.7)
'10520 ± 160'
>>> format_uncertainty(0.000123456, 0.0000123, unit="g/mL")
'(1.23 ± 0.12)e-04 g/mL'
"""

from __future__ import annotations

import math

import numpy as np


def _decimals(u: float) -> int:
    """Decimal place (as used by round()) of the last significant digit of the rounded uncertainty."""
    exp = math.floor(math.log10(abs(u)))
    lead = int(abs(u) / 10**exp)
    sig = 2 if lead == 1 else 1
    d = -(exp - sig + 1)
    u_r = round(u, d)  # rounding can bump 0.096 -> 0.1, which changes the leading digit
    exp2 = math.floor(math.log10(abs(u_r)))
    lead2 = int(round(abs(u_r) / 10**exp2, 6))
    sig2 = 2 if lead2 == 1 else 1
    return -(exp2 - sig2 + 1)


def format_uncertainty(value: float, u: float, unit: str = "", sci: bool | None = None) -> str:
    """Format value ± u with consistent significant figures; scientific notation for very small
    or large numbers (automatic unless ``sci`` is given)."""
    if not u > 0:
        raise ValueError("The uncertainty must be positive.")
    if sci is None:
        sci = abs(value) < 1e-3 or abs(value) >= 1e6
    if sci:
        e = math.floor(math.log10(abs(value))) if value else math.floor(math.log10(u))
        body = format_uncertainty(value / 10**e, u / 10**e, sci=False)
        return f"({body})e{e:+03d}" + (f" {unit}" if unit else "")
    d = _decimals(u)
    v, uu = round(value, d), round(u, d)
    if d > 0:
        text = f"{v:.{d}f} ± {uu:.{d}f}"
    else:
        text = f"{int(v)} ± {int(uu)}"
    return text + (f" {unit}" if unit else "")


def propagate(func, values: dict, uncertainties: dict) -> tuple[float, float, dict]:
    """First-order propagation for independent inputs: u_y^2 = sum (df/dx_i u_i)^2.
    Returns (y, u_y, contributions) where contributions are the fractions of u_y^2."""
    names = list(values)
    x = np.array([values[k] for k in names], float)
    y = float(func(**values))
    terms = {}
    for i, k in enumerate(names):
        h = 1e-6 * max(abs(x[i]), 1e-12)
        up, dn = dict(values), dict(values)
        up[k], dn[k] = x[i] + h, x[i] - h
        c = (func(**up) - func(**dn)) / (2 * h)
        terms[k] = (c * uncertainties.get(k, 0.0)) ** 2
    var = sum(terms.values())
    return y, float(np.sqrt(var)), {k: v / var if var else 0.0 for k, v in terms.items()}
