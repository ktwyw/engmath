"""Numerical integration of data and of functions.

>>> import numpy as np
>>> from engmath import integrate
>>> abs(integrate.gauss_legendre(np.exp, 0, 1, 5) - (np.e - 1)) < 2e-12   # theory: error ~ 1.1e-12
True
"""

from __future__ import annotations

import numpy as np


def trapezoid(y, x=None, dx: float = 1.0) -> float:
    """Trapezoidal rule for (possibly unevenly spaced) data. Error O(h^2)."""
    y = np.asarray(y, float)
    if x is None:
        return float(dx * (y.sum() - 0.5 * (y[0] + y[-1])))
    x = np.asarray(x, float)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def cumulative_trapezoid(y, x, initial: float = 0.0) -> np.ndarray:
    """Running integral (same length as y), e.g. concentration from a measured rate."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return initial + np.r_[0.0, np.cumsum(np.diff(x) * (y[1:] + y[:-1]) / 2)]


def simpson(y, x=None, dx: float = 1.0) -> float:
    """Composite Simpson's rule (error O(h^4)). Unequal spacing is handled pairwise; with an odd
    number of intervals the last interval uses the 3-point parabola through the final points."""
    y = np.asarray(y, float)
    x = np.arange(y.size) * dx if x is None else np.asarray(x, float)
    if y.size < 3:
        raise ValueError("Simpson's rule needs at least 3 points.")
    h = np.diff(x)
    total = 0.0
    n_int = h.size
    last = n_int - 1 if n_int % 2 else n_int
    for i in range(0, last, 2):
        h0, h1 = h[i], h[i + 1]
        total += (h0 + h1) / 6 * ((2 - h1 / h0) * y[i] + (h0 + h1) ** 2 / (h0 * h1) * y[i + 1]
                                  + (2 - h0 / h1) * y[i + 2])
    if n_int % 2:  # integrate the last interval with the parabola through the last three points
        h0, h1 = h[-2], h[-1]
        a = (2 * h1**2 + 3 * h0 * h1) / (6 * (h0 + h1))
        b = (h1**2 + 3 * h0 * h1) / (6 * h0)
        c = -h1**3 / (6 * h0 * (h0 + h1))
        total += a * y[-1] + b * y[-2] + c * y[-3]
    return float(total)


def composite(f, a: float, b: float, n: int, rule: str = "trapezoid") -> float:
    """Integrate a function with n equal intervals - for convergence studies."""
    x = np.linspace(a, b, n + 1)
    y = f(x)
    if rule == "trapezoid":
        return trapezoid(y, x)
    if rule == "simpson":
        if n % 2:
            raise ValueError("Simpson needs an even number of intervals.")
        return simpson(y, x)
    if rule == "midpoint":
        xm = 0.5 * (x[1:] + x[:-1])
        return float(np.sum(f(xm)) * (b - a) / n)
    raise ValueError("rule must be 'trapezoid', 'simpson' or 'midpoint'.")


def romberg(f, a: float, b: float, levels: int = 6) -> np.ndarray:
    """Romberg table: trapezoid results with Richardson extrapolation; R[-1, -1] is the best."""
    R = np.zeros((levels, levels))
    for i in range(levels):
        R[i, 0] = composite(f, a, b, 2**i, "trapezoid")
        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)
    return R


def gauss_legendre(f, a: float, b: float, n: int = 5) -> float:
    """n-point Gauss-Legendre quadrature: exact for polynomials up to degree 2n - 1."""
    t, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * t + 0.5 * (a + b)
    return float(0.5 * (b - a) * np.sum(w * f(x)))
