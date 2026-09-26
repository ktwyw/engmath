"""Numerical differentiation of functions and of (unevenly spaced) data.

>>> import numpy as np
>>> from engmath import differentiate as d
>>> x = np.array([0.0, 0.5, 1.5, 3.0]); y = x**2
>>> d.gradient(x, y).round(12).tolist()     # exact for quadratics, even with uneven spacing
[0.0, 1.0, 3.0, 6.0]
"""

from __future__ import annotations

import numpy as np


def forward(f, x, h: float = 1e-6):
    """Forward difference (f(x+h) - f(x))/h: first-order accurate, O(h)."""
    return (f(x + h) - f(x)) / h


def central(f, x, h: float = 1e-5):
    """Central difference (f(x+h) - f(x-h))/2h: second-order accurate, O(h^2)."""
    return (f(x + h) - f(x - h)) / (2 * h)


def richardson(f, x, h: float = 0.1, levels: int = 4) -> float:
    """Central differences at h, h/2, h/4 ... combined by Richardson extrapolation (O(h^(2*levels)))."""
    D = np.zeros((levels, levels))
    for i in range(levels):
        D[i, 0] = central(f, x, h / 2**i)
        for j in range(1, i + 1):
            D[i, j] = D[i, j - 1] + (D[i, j - 1] - D[i - 1, j - 1]) / (4**j - 1)
    return float(D[-1, -1])


def gradient(x, y) -> np.ndarray:
    """First derivative of data with UNEQUAL spacing, second-order accurate everywhere
    (3-point Lagrange formulas; one-sided at the ends). Exact for quadratics."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        raise ValueError("Need at least 3 points.")
    h = np.diff(x)
    d = np.empty_like(y)
    h1, h2 = h[0], h[1]
    d[0] = -(2 * h1 + h2) / (h1 * (h1 + h2)) * y[0] + (h1 + h2) / (h1 * h2) * y[1] - h1 / (h2 * (h1 + h2)) * y[2]
    hm, hp = h[:-1], h[1:]  # interior points (vectorised: no Python loop)
    d[1:-1] = (-hp / (hm * (hm + hp)) * y[:-2] - (hm - hp) / (hm * hp) * y[1:-1]
               + hm / (hp * (hm + hp)) * y[2:])
    h1, h2 = h[-2], h[-1]
    d[-1] = h2 / (h1 * (h1 + h2)) * y[-3] - (h1 + h2) / (h1 * h2) * y[-2] + (h1 + 2 * h2) / (h2 * (h1 + h2)) * y[-1]
    return d


def second_derivative(x, y) -> np.ndarray:
    """Second derivative at interior points of unevenly spaced data (ends copied from neighbours)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    hm, hp = np.diff(x)[:-1], np.diff(x)[1:]
    inner = 2 * (hp * y[:-2] - (hm + hp) * y[1:-1] + hm * y[2:]) / (hm * hp * (hm + hp))
    return np.r_[inner[0], inner, inner[-1]]
