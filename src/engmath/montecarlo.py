"""Monte Carlo methods: integration in many dimensions and random walks.

>>> from engmath import montecarlo
>>> est, se = montecarlo.integrate(lambda p: (p**2).sum(axis=1), [(0, 1)] * 3, 200_000, seed=1)
>>> abs(est - 1.0) < 4 * se          # exact value: 3 * 1/3 = 1
True
"""

from __future__ import annotations

import numpy as np


def integrate(f, bounds, n: int = 100_000, seed: int | None = 0):
    """Integral of f over a box by plain Monte Carlo: volume x mean of f at n random points.

    ``f`` receives an (n, d) array of points. Returns (estimate, standard error). The error shrinks
    as 1/sqrt(n) WHATEVER the dimension d - unlike grid-based quadrature."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    pts = lo + (hi - lo) * rng.random((n, lo.size))
    vals = np.asarray(f(pts), float)
    vol = float(np.prod(hi - lo))
    return float(vol * vals.mean()), float(vol * vals.std(ddof=1) / np.sqrt(n))


def random_walk(n_steps: int, n_walkers: int, dim: int = 1, step: float = 1.0, seed: int | None = 0) -> np.ndarray:
    """Lattice random walks: at each step every walker moves +/-step along one randomly chosen axis.

    Returns positions of shape (n_steps + 1, n_walkers, dim), starting at the origin. The mean squared
    displacement grows linearly, <r^2> = n_steps * step^2: the microscopic picture of diffusion."""
    rng = np.random.default_rng(seed)
    axis = rng.integers(0, dim, size=(n_steps, n_walkers))
    sign = rng.choice([-1.0, 1.0], size=(n_steps, n_walkers))
    moves = np.zeros((n_steps, n_walkers, dim))
    np.put_along_axis(moves, axis[..., None], (sign * step)[..., None], axis=2)
    return np.concatenate([np.zeros((1, n_walkers, dim)), np.cumsum(moves, axis=0)])
