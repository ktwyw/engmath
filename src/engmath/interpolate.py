"""Interpolation: linear, natural cubic spline (built from scratch) and polynomial (Lagrange).

>>> from engmath import interpolate
>>> s = interpolate.CubicSpline([0, 1, 2, 3], [0, 1, 8, 27])
>>> round(float(s(1.5)), 4)          # natural end conditions: not exactly 1.5**3 = 3.375
3.15
"""

from __future__ import annotations

import numpy as np

from .linalg import thomas


def linear(x, y, xq, extrapolate: bool = False):
    """Piecewise-linear interpolation; refuses to extrapolate unless asked (a common silent error)."""
    x, y, xq = np.asarray(x, float), np.asarray(y, float), np.asarray(xq, float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing.")
    if not extrapolate and (np.min(xq) < x[0] or np.max(xq) > x[-1]):
        raise ValueError("Query points outside the data range; pass extrapolate=True if intended.")
    if extrapolate:
        i = np.clip(np.searchsorted(x, xq) - 1, 0, x.size - 2)
        return y[i] + (y[i + 1] - y[i]) * (xq - x[i]) / (x[i + 1] - x[i])
    return np.interp(xq, x, y)


class CubicSpline:
    """Natural cubic spline (zero second derivative at both ends).

    Second derivatives M_i solve the tridiagonal system
    h_{i-1} M_{i-1} + 2 (h_{i-1} + h_i) M_i + h_i M_{i+1} = 6 (d_i - d_{i-1}),  d_i = (y_{i+1} - y_i)/h_i,
    solved in O(n) with :func:`engmath.linalg.thomas`.
    """

    def __init__(self, x, y):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        if self.x.size < 3 or np.any(np.diff(self.x) <= 0):
            raise ValueError("Need >= 3 points with strictly increasing x.")
        h = np.diff(self.x)
        d = np.diff(self.y) / h
        n = self.x.size
        M = np.zeros(n)
        if n > 2:
            M[1:-1] = thomas(np.r_[0, h[1:-1]], 2 * (h[:-1] + h[1:]), np.r_[h[1:-1], 0], 6 * np.diff(d))
        self.h, self.M = h, M

    def _locate(self, xq):
        return np.clip(np.searchsorted(self.x, xq) - 1, 0, self.x.size - 2)

    def __call__(self, xq):
        xq = np.asarray(xq, float)
        i = self._locate(xq)
        h, M, x, y = self.h[i], self.M, self.x, self.y
        a, b = x[i + 1] - xq, xq - x[i]
        cubic = (M[i] * a**3 + M[i + 1] * b**3) / (6 * h)
        return cubic + (y[i] / h - M[i] * h / 6) * a + (y[i + 1] / h - M[i + 1] * h / 6) * b

    def derivative(self, xq):
        xq = np.asarray(xq, float)
        i = self._locate(xq)
        h, M, x, y = self.h[i], self.M, self.x, self.y
        a, b = x[i + 1] - xq, xq - x[i]
        return (-M[i] * a**2 + M[i + 1] * b**2) / (2 * h) + (y[i + 1] - y[i]) / h - (M[i + 1] - M[i]) * h / 6

    def integral(self) -> float:
        """Exact integral of the spline over the data range."""
        h, M, y = self.h, self.M, self.y
        return float(np.sum(h * (y[:-1] + y[1:]) / 2 - h**3 * (M[:-1] + M[1:]) / 24))


def lagrange(x, y, xq):
    """Polynomial through all points (degree n - 1). Beware Runge oscillations for many
    equally spaced points - the reason splines exist."""
    x, y, xq = np.asarray(x, float), np.asarray(y, float), np.asarray(xq, float)
    out = np.zeros_like(xq)
    for j in range(x.size):
        L = np.ones_like(xq)
        for m in range(x.size):
            if m != j:
                L *= (xq - x[m]) / (x[j] - x[m])
        out += y[j] * L
    return out
