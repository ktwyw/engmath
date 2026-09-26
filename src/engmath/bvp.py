"""Boundary-value problems for second-order ODEs: the shooting method and finite differences.

Boundary conditions are given as ``("dirichlet", value)`` or ``("robin", alpha, beta, gamma)``,
meaning alpha * y' + beta * y = gamma (Neumann: beta = 0; convection: y' + (h/k) y = (h/k) T_inf).

>>> import numpy as np
>>> from engmath import bvp
>>> x, y = bvp.finite_difference(lambda x: 0*x, lambda x: 0*x, lambda x: -2 + 0*x, 0, 1, 50,
...                              ("dirichlet", 0.0), ("dirichlet", 0.0))    # y'' = -2  ->  y = x(1 - x)
>>> float(np.max(np.abs(y - x*(1 - x)))) < 1e-12
True
"""

from __future__ import annotations

import numpy as np

from .linalg import thomas
from .odes import rk4
from .roots import secant


def _residual(bc, y, yp):
    if bc[0] == "dirichlet":
        return y - bc[1]
    if bc[0] == "robin":
        _, alpha, beta, gamma = bc
        return alpha * yp + beta * y - gamma
    raise ValueError("Boundary condition must be ('dirichlet', value) or ('robin', alpha, beta, gamma).")


def shooting(f, a: float, b: float, ya: float, right, slopes=(0.0, 1.0), n: int = 400):
    """Shooting method for y'' = f(x, y, y') with y(a) = ya and a right boundary condition ``right``.

    Guess the unknown initial slope s, integrate the initial-value problem with RK4, and adjust s by
    the secant method until the right-hand condition is met. Returns (x, y, y', s)."""
    def end_miss(s):
        _, sol = rk4(lambda x, u: [u[1], f(x, u[0], u[1])], [ya, s], a, b, n)
        return _residual(right, sol[-1, 0], sol[-1, 1])

    s = secant(end_miss, *slopes, tol=1e-12).root
    x, sol = rk4(lambda x, u: [u[1], f(x, u[0], u[1])], [ya, s], a, b, n)
    return x, sol[:, 0], sol[:, 1], s


def finite_difference(p, q, r, a: float, b: float, n: int, left, right):
    """Linear BVP y'' + p(x) y' + q(x) y = r(x) on [a, b] with n intervals (second-order accurate).

    Central differences give a tridiagonal system (solved by the Thomas algorithm); Robin conditions
    use a ghost node, which keeps second-order accuracy. Returns (x, y)."""
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    P, Q, R = (np.broadcast_to(np.asarray(fn(x), float), x.shape).copy() for fn in (p, q, r))
    lo = 1 / h**2 - P / (2 * h)  # coefficient of y[i-1]
    di = -2 / h**2 + Q  # coefficient of y[i]
    up = 1 / h**2 + P / (2 * h)  # coefficient of y[i+1]
    rhs = R.copy()
    lower, diag, upper = lo.copy(), di.copy(), up.copy()
    if left[0] == "dirichlet":
        lower[0], diag[0], upper[0], rhs[0] = 0.0, 1.0, 0.0, left[1]
    else:  # ghost node y[-1] = y[1] - 2h (gamma - beta y[0]) / alpha
        _, al, be, ga = left
        diag[0] = di[0] + lo[0] * 2 * h * be / al
        upper[0] = up[0] + lo[0]
        rhs[0] = R[0] + lo[0] * 2 * h * ga / al
    if right[0] == "dirichlet":
        lower[-1], diag[-1], upper[-1], rhs[-1] = 0.0, 1.0, 0.0, right[1]
    else:  # ghost node y[n+1] = y[n-1] + 2h (gamma - beta y[n]) / alpha
        _, al, be, ga = right
        lower[-1] = lo[-1] + up[-1]
        diag[-1] = di[-1] - up[-1] * 2 * h * be / al
        rhs[-1] = R[-1] - up[-1] * 2 * h * ga / al
    return x, thomas(lower, diag, upper, rhs)
