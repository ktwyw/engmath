"""Smoothing noisy data: moving average, Savitzky-Golay and LOWESS (all from scratch).

>>> import numpy as np
>>> from engmath import smoothing
>>> smoothing.moving_average([1, 2, 3, 4, 5], 3).tolist()
[1.5, 2.0, 3.0, 4.0, 4.5]
"""

from __future__ import annotations

import numpy as np


def moving_average(y, window: int = 5) -> np.ndarray:
    """Centred moving average (odd window); the window shrinks at the ends to keep the length."""
    y = np.asarray(y, float)
    if window % 2 == 0 or window < 1:
        raise ValueError("window must be a positive odd integer.")
    k = window // 2
    return np.array([y[max(0, i - k) : i + k + 1].mean() for i in range(y.size)])


def savitzky_golay(y, half_window: int = 5, order: int = 2, deriv: int = 0, dx: float = 1.0) -> np.ndarray:
    """Savitzky-Golay filter for EQUALLY spaced data: fit a polynomial of ``order`` by least
    squares in each window of 2*half_window + 1 points and take its value (deriv=0) or its
    derivative (deriv=1, 2). Edges use the fit of the first/last full window."""
    y = np.asarray(y, float)
    m = half_window
    if 2 * m + 1 > y.size or order >= 2 * m + 1:
        raise ValueError("Window too large for the data, or order too high for the window.")
    t = np.arange(-m, m + 1)
    V = np.vander(t, order + 1, increasing=True)
    C = np.linalg.pinv(V)  # rows give polynomial coefficients from window values
    fact = np.prod(np.arange(1, deriv + 1)) if deriv else 1
    out = np.empty_like(y)
    coef_row = C[deriv] * fact / dx**deriv
    out[m:-m] = np.convolve(y, coef_row[::-1], mode="valid")
    poly = np.polynomial.Polynomial
    n = y.size
    left = poly(C @ y[: 2 * m + 1])
    right = poly(C @ y[-(2 * m + 1):])
    if deriv:
        left, right = left.deriv(deriv), right.deriv(deriv)
    for i in range(m):  # edges: evaluate the end-window polynomial away from its centre
        out[i] = left(i - m) / dx**deriv
        out[n - m + i] = right(i + 1) / dx**deriv
    return out


def lowess(x, y, frac: float = 0.3, degree: int = 1, robust_iters: int = 0) -> np.ndarray:
    """LOWESS/LOESS: at each x_i fit a weighted polynomial to the nearest frac*n points with
    tricube weights. ``robust_iters`` > 0 adds bisquare down-weighting of outliers."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = x.size
    r = max(degree + 2, int(np.ceil(frac * n)))
    robust = np.ones(n)
    fit = np.empty(n)
    for _ in range(robust_iters + 1):
        for i in range(n):
            dist = np.abs(x - x[i])
            idx = np.argsort(dist)[:r]
            hmax = dist[idx].max() * 1.0001 or 1.0
            w = (1 - (dist[idx] / hmax) ** 3) ** 3 * robust[idx]
            V = np.vander(x[idx] - x[i], degree + 1, increasing=True)
            sw = np.sqrt(w)
            coef = np.linalg.lstsq(V * sw[:, None], y[idx] * sw, rcond=None)[0]
            fit[i] = coef[0]
        resid = y - fit
        s = np.median(np.abs(resid)) or 1e-12
        u = resid / (6 * s)
        robust = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
    return fit
