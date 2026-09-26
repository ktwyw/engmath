"""1D transient diffusion/heat conduction dC/dt = D d2C/dx2 by finite differences:
explicit FTCS (stable only for D dt/dx^2 <= 1/2) and Crank-Nicolson (unconditionally stable,
second order in time; each step is one tridiagonal solve).

Boundary conditions: fixed value at x = 0 (``c_left``) and fixed value at x = L (``c_right``).
"""

from __future__ import annotations

import numpy as np

from .linalg import thomas


def ftcs(c0, D: float, dx: float, dt: float, steps: int, c_left: float, c_right: float) -> np.ndarray:
    """Forward-time central-space scheme. Raises if the stability number r = D dt/dx^2 > 0.5."""
    r = D * dt / dx**2
    if r > 0.5:
        raise ValueError(f"Unstable: r = D dt/dx^2 = {r:.3g} > 0.5. Reduce dt or use crank_nicolson().")
    c = np.array(c0, float)
    c[0], c[-1] = c_left, c_right
    for _ in range(steps):
        c[1:-1] = c[1:-1] + r * (c[2:] - 2 * c[1:-1] + c[:-2])
    return c


def crank_nicolson(c0, D: float, dx: float, dt: float, steps: int, c_left: float, c_right: float) -> np.ndarray:
    """Crank-Nicolson scheme: average of explicit and implicit Laplacians."""
    c = np.array(c0, float)
    c[0], c[-1] = c_left, c_right
    n = c.size - 2  # interior unknowns
    r = D * dt / dx**2
    lower = np.full(n, -r / 2)
    diag = np.full(n, 1 + r)
    upper = np.full(n, -r / 2)
    for _ in range(steps):
        rhs = r / 2 * c[:-2] + (1 - r) * c[1:-1] + r / 2 * c[2:]
        rhs[0] += r / 2 * c_left  # new-time boundary values
        rhs[-1] += r / 2 * c_right
        c[1:-1] = thomas(lower, diag, upper, rhs)
    return c
