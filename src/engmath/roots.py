"""Root finding for f(x) = 0: bisection, Newton-Raphson, secant and Brent's method.

Every solver returns a :class:`RootResult` holding the full iteration history, so convergence
can be plotted and compared (linear for bisection, superlinear for secant, quadratic for Newton).

>>> from engmath import roots
>>> r = roots.newton(lambda x: x**2 - 2, 1.0)
>>> round(r.root, 12), r.converged
(1.414213562373, True)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RootResult:
    root: float
    converged: bool
    iterations: int
    history: list = field(default_factory=list)  # successive estimates
    method: str = ""

    def __str__(self) -> str:
        status = "converged" if self.converged else "NOT converged"
        return f"{self.method}: root = {self.root:.12g} ({status} in {self.iterations} iterations)"


def bisection(f, a: float, b: float, tol: float = 1e-12, maxiter: int = 200) -> RootResult:
    """Bisection: halve a bracket [a, b] with f(a) f(b) < 0. Slow (1 bit per step) but guaranteed."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs (the root must be bracketed).")
    hist = []
    for k in range(1, maxiter + 1):
        m = 0.5 * (a + b)
        fm = f(m)
        hist.append(m)
        if fm == 0 or 0.5 * (b - a) < tol:
            return RootResult(m, True, k, hist, "bisection")
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return RootResult(0.5 * (a + b), False, maxiter, hist, "bisection")


def derivative(f, x: float, h: float | None = None) -> float:
    """Central-difference derivative with a step scaled to x (used when no derivative is given)."""
    h = h or 1e-6 * max(1.0, abs(x))
    return (f(x + h) - f(x - h)) / (2 * h)


def newton(f, x0: float, df=None, tol: float = 1e-12, maxiter: int = 50) -> RootResult:
    """Newton-Raphson x_{k+1} = x_k - f(x_k)/f'(x_k). Quadratic convergence near a simple root,
    but it can diverge from a poor starting point. ``df`` defaults to a numerical derivative."""
    x = float(x0)
    hist = [x]
    for k in range(1, maxiter + 1):
        d = df(x) if df else derivative(f, x)
        if d == 0:
            return RootResult(x, False, k, hist, "Newton")
        step = f(x) / d
        x -= step
        hist.append(x)
        if not np.isfinite(x):
            return RootResult(x, False, k, hist, "Newton")
        if abs(step) < tol * max(1.0, abs(x)):
            return RootResult(x, True, k, hist, "Newton")
    return RootResult(x, False, maxiter, hist, "Newton")


def secant(f, x0: float, x1: float, tol: float = 1e-12, maxiter: int = 100) -> RootResult:
    """Secant method: Newton with the derivative replaced by a finite-difference slope."""
    f0, f1 = f(x0), f(x1)
    hist = [x0, x1]
    for k in range(1, maxiter + 1):
        if f1 == f0:
            return RootResult(x1, False, k, hist, "secant")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        hist.append(x2)
        if abs(x2 - x1) < tol * max(1.0, abs(x2)):
            return RootResult(x2, True, k, hist, "secant")
        x0, f0, x1, f1 = x1, f1, x2, f(x2)
    return RootResult(x1, False, maxiter, hist, "secant")


def brent(f, a: float, b: float, tol: float = 1e-12) -> RootResult:
    """Brent's method via SciPy: bracketing safety with superlinear speed - the practical default."""
    from scipy.optimize import brentq

    hist = []

    def g(x):
        hist.append(x)
        return f(x)

    root, info = brentq(g, a, b, xtol=tol, full_output=True)
    return RootResult(float(root), bool(info.converged), int(info.iterations), hist, "Brent")


def convergence_order(history, root: float) -> float:
    """Estimate the order p from errors e_k: p ~ log(e_{k+1}/e_k) / log(e_k/e_{k-1}).

    Errors below ~1e-10 relative are dropped: there the reference root and round-off dominate and the
    ratios become meaningless. ``root`` should be known much more precisely than the iterates."""
    e = np.abs(np.asarray(history, dtype=float) - root)
    e = e[e > 1e-10 * (abs(root) if root else 1.0)]
    if e.size < 3:
        return float("nan")
    p = np.log(e[2:] / e[1:-1]) / np.log(e[1:-1] / e[:-2])
    return float(p[-1])
