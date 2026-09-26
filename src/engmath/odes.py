"""Ordinary differential equations dy/dt = f(t, y): Euler, classical RK4, adaptive RK23
(Bogacki-Shampine) and backward (implicit) Euler for stiff problems.

>>> import numpy as np
>>> from engmath import odes
>>> t, y = odes.rk4(lambda t, y: -2 * y, [1.0], 0, 1, 20)
>>> bool(abs(y[-1, 0] - np.exp(-2)) < 1e-6)
True
"""

from __future__ import annotations

import numpy as np


def _f(fun, t, y):
    return np.atleast_1d(np.asarray(fun(t, y), float))


def euler(fun, y0, t0: float, t1: float, n: int):
    """Explicit Euler, n steps: first order, and unstable for stiff problems if h is too large."""
    t = np.linspace(t0, t1, n + 1)
    y = np.empty((n + 1, np.size(y0)))
    y[0] = y0
    for k in range(n):
        y[k + 1] = y[k] + (t[k + 1] - t[k]) * _f(fun, t[k], y[k])
    return t, y


def rk4(fun, y0, t0: float, t1: float, n: int):
    """Classical fourth-order Runge-Kutta, n equal steps (global error O(h^4))."""
    t = np.linspace(t0, t1, n + 1)
    y = np.empty((n + 1, np.size(y0)))
    y[0] = y0
    for k in range(n):
        h, tk, yk = t[k + 1] - t[k], t[k], y[k]
        k1 = _f(fun, tk, yk)
        k2 = _f(fun, tk + h / 2, yk + h / 2 * k1)
        k3 = _f(fun, tk + h / 2, yk + h / 2 * k2)
        k4 = _f(fun, tk + h, yk + h * k3)
        y[k + 1] = yk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


def rk23(fun, y0, t0: float, t1: float, rtol: float = 1e-6, atol: float = 1e-9, h0: float | None = None,
         max_steps: int = 100000):
    """Adaptive Bogacki-Shampine 3(2) pair: the step size follows the local error estimate,
    small where the solution changes fast, large where it is smooth."""
    t, y = t0, np.atleast_1d(np.asarray(y0, float))
    h = h0 or (t1 - t0) / 100
    ts, ys = [t], [y.copy()]
    k1 = _f(fun, t, y)
    for _ in range(max_steps):
        if t >= t1:
            break
        h = min(h, t1 - t)
        k2 = _f(fun, t + h / 2, y + h / 2 * k1)
        k3 = _f(fun, t + 3 * h / 4, y + 3 * h / 4 * k2)
        y3 = y + h * (2 / 9 * k1 + 1 / 3 * k2 + 4 / 9 * k3)
        k4 = _f(fun, t + h, y3)
        y2 = y + h * (7 / 24 * k1 + 1 / 4 * k2 + 1 / 3 * k3 + 1 / 8 * k4)
        err = np.max(np.abs(y3 - y2) / (atol + rtol * np.maximum(np.abs(y), np.abs(y3))))
        if err <= 1:
            t, y, k1 = t + h, y3, k4  # first-same-as-last: k4 is the next k1
            ts.append(t)
            ys.append(y.copy())
        h *= min(5.0, max(0.2, 0.9 * err ** (-1 / 3))) if err > 0 else 5.0
    return np.array(ts), np.array(ys)


def backward_euler(fun, y0, t0: float, t1: float, n: int, newton_iters: int = 20):
    """Implicit Euler y_{k+1} = y_k + h f(t_{k+1}, y_{k+1}), solved by Newton with a numerical
    Jacobian. First order but unconditionally stable - the simplest stiff solver."""
    t = np.linspace(t0, t1, n + 1)
    m = np.size(y0)
    y = np.empty((n + 1, m))
    y[0] = y0
    for k in range(n):
        h = t[k + 1] - t[k]
        z = y[k] + h * _f(fun, t[k], y[k])  # explicit predictor
        for _ in range(newton_iters):
            g = z - y[k] - h * _f(fun, t[k + 1], z)
            J = np.eye(m)
            for j in range(m):
                dz = 1e-7 * max(1.0, abs(z[j]))
                zz = z.copy()
                zz[j] += dz
                J[:, j] -= h * (_f(fun, t[k + 1], zz) - _f(fun, t[k + 1], z)) / dz
            step = np.linalg.solve(J, g)
            z -= step
            if np.max(np.abs(step)) < 1e-12 * (1 + np.max(np.abs(z))):
                break
        y[k + 1] = z
    return t, y
