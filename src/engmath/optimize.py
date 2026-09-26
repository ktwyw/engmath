"""Minimisation without derivatives: golden-section search (1D) and Nelder-Mead simplex (nD).

>>> from engmath import optimize
>>> r = optimize.nelder_mead(lambda p: (p[0] - 1)**2 + 10 * (p[1] + 2)**2, [0, 0])
>>> r.x.round(5).tolist()
[1.0, -2.0]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MinResult:
    x: np.ndarray
    fun: float
    iterations: int
    converged: bool
    history: list = field(default_factory=list)  # best objective value per iteration


def golden_section(f, a: float, b: float, tol: float = 1e-10, maxiter: int = 500) -> MinResult:
    """Minimum of a unimodal f on [a, b]; the bracket shrinks by 0.618 each step."""
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    hist = []
    for k in range(1, maxiter + 1):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
        hist.append(min(fc, fd))
        if b - a < tol * max(1.0, abs(a) + abs(b)):
            x = 0.5 * (a + b)
            return MinResult(np.array([x]), float(f(x)), k, True, hist)
    x = 0.5 * (a + b)
    return MinResult(np.array([x]), float(f(x)), maxiter, False, hist)


def nelder_mead(f, x0, step: float = 0.1, tol: float = 1e-10, maxiter: int = 5000) -> MinResult:
    """Nelder-Mead downhill simplex (standard coefficients: reflection 1, expansion 2,
    contraction 0.5, shrink 0.5). ``step`` is the relative size of the initial simplex."""
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    simplex = [x0]
    for i in range(n):
        v = x0.copy()
        v[i] = v[i] * (1 + step) if v[i] != 0 else step
        simplex.append(v)
    simplex = np.array(simplex)
    fs = np.array([f(v) for v in simplex])
    hist = []
    for k in range(1, maxiter + 1):
        order = np.argsort(fs)
        simplex, fs = simplex[order], fs[order]
        hist.append(fs[0])
        if abs(fs[-1] - fs[0]) <= tol * (abs(fs[0]) + tol) and np.max(np.abs(simplex[1:] - simplex[0])) <= tol * (
                1 + np.max(np.abs(simplex[0]))):
            return MinResult(simplex[0], float(fs[0]), k, True, hist)
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        if fr < fs[0]:
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = f(xe)
            simplex[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            simplex[-1], fs[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid) if fr >= fs[-1] else centroid + 0.5 * (xr - centroid)
            fc = f(xc)
            if fc < min(fr, fs[-1]):
                simplex[-1], fs[-1] = xc, fc
            else:  # shrink towards the best vertex
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                fs[1:] = [f(v) for v in simplex[1:]]
    order = np.argsort(fs)
    return MinResult(simplex[order[0]], float(fs[order[0]]), maxiter, False, hist)


# ---------------------------------------------------------------------------- linear programming
@dataclass
class LPResult:
    x: np.ndarray
    objective: float
    shadow_prices: np.ndarray  # value of one more unit of each resource (dual values)
    iterations: int
    vertices: list = field(default_factory=list)  # corner points visited by the simplex method


def simplex_lp(c, A_ub, b_ub, tol: float = 1e-12, maxiter: int = 1000) -> LPResult:
    """Maximise c^T x subject to A_ub x <= b_ub and x >= 0 (with b_ub >= 0, so x = 0 is feasible),
    by the tableau simplex method. The method walks from corner to corner of the feasible region,
    always improving the objective. Shadow prices are read from the final tableau."""
    c, A, b = np.asarray(c, float), np.atleast_2d(np.asarray(A_ub, float)), np.asarray(b_ub, float)
    m, n = A.shape
    if np.any(b < 0):
        raise ValueError("This teaching simplex needs b_ub >= 0 (the origin must be feasible).")
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n], T[:m, n : n + m], T[:m, -1] = A, np.eye(m), b
    T[-1, :n] = -c
    basis = list(range(n, n + m))

    def vertex():
        x = np.zeros(n + m)
        x[basis] = T[:m, -1]
        return x[:n].copy()

    visited = [vertex()]
    for it in range(1, maxiter + 1):
        col = int(np.argmin(T[-1, :-1]))
        if T[-1, col] >= -tol:  # no improving direction: optimal
            return LPResult(vertex(), float(T[-1, -1]), T[-1, n : n + m].copy(), it - 1, visited)
        pos = T[:m, col] > tol
        if not pos.any():
            raise ValueError("The problem is unbounded: the objective can grow without limit.")
        ratios = np.full(m, np.inf)
        ratios[pos] = T[:m, -1][pos] / T[:m, col][pos]
        row = int(np.argmin(ratios))  # ratio test: the first constraint to become binding
        T[row] /= T[row, col]
        for i in range(m + 1):
            if i != row:
                T[i] -= T[i, col] * T[row]
        basis[row] = col
        visited.append(vertex())
    raise RuntimeError("Simplex did not terminate (cycling?).")


# ---------------------------------------------------------------------------- constrained minimisation
def penalty_minimize(f, x0, ineq=(), eq=(), mu0: float = 10.0, growth: float = 10.0, rounds: int = 8,
                     tol: float = 1e-10) -> MinResult:
    """Minimise f(x) subject to g(x) <= 0 for g in ``ineq`` and h(x) = 0 for h in ``eq`` by the
    quadratic penalty method: minimise f + mu * (sum max(0, g)^2 + sum h^2) for increasing mu, each
    time starting from the previous solution. ``history`` holds x after each round.

    Scale matters: ``mu0`` times a typical constraint violation should be comparable to the objective.
    If mu0 is too small, the first round happily violates the constraints (e.g. shrinks a tank to zero
    volume) and later rounds may never recover. Also make sure the objective is bounded below in the
    infeasible region - optimise log(x) for quantities that must stay positive."""
    x = np.asarray(x0, float)
    mu = mu0
    hist = []
    for _ in range(rounds):
        def P(v, mu=mu):
            viol = sum(max(0.0, g(v)) ** 2 for g in ineq) + sum(h(v) ** 2 for h in eq)
            return f(v) + mu * viol
        x = nelder_mead(P, x, step=0.05, tol=tol).x
        hist.append(x.copy())
        mu *= growth
    return MinResult(x, float(f(x)), rounds, True, hist)
