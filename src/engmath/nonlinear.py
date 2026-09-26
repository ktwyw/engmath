"""Systems of nonlinear equations F(x) = 0: Newton's method with a backtracking line search.

>>> import numpy as np
>>> from engmath import nonlinear
>>> F = lambda v: [v[0]**2 + v[1]**2 - 4, v[0] - v[1]]       # circle meets the line y = x
>>> r = nonlinear.newton_system(F, [1.0, 0.5])
>>> r.x.round(10).tolist(), r.converged
([1.4142135624, 1.4142135624], True)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SystemResult:
    x: np.ndarray
    converged: bool
    iterations: int
    residual_history: list = field(default_factory=list)  # ||F(x_k)|| per iteration
    x_history: list = field(default_factory=list)

    def __str__(self) -> str:
        status = "converged" if self.converged else "NOT converged"
        return f"x = {np.array2string(self.x, precision=8)} ({status} in {self.iterations} iterations)"


def jacobian(F, x, rel_step: float = 1e-7) -> np.ndarray:
    """Numerical Jacobian J_ij = dF_i/dx_j by forward differences."""
    x = np.asarray(x, float)
    f0 = np.asarray(F(x), float)
    J = np.empty((f0.size, x.size))
    for j in range(x.size):
        h = rel_step * max(abs(x[j]), 1.0)
        xp = x.copy()
        xp[j] += h
        J[:, j] = (np.asarray(F(xp), float) - f0) / h
    return J


def newton_system(F, x0, J=None, tol: float = 1e-10, maxiter: int = 50, line_search: bool = True) -> SystemResult:
    """Newton's method for F(x) = 0: solve J(x_k) dx = -F(x_k), then x_{k+1} = x_k + t dx.

    ``J`` is the Jacobian function (default: numerical). With ``line_search`` the step length t is
    halved until ||F|| decreases sufficiently (Armijo condition) - this globalises Newton's method
    and rescues many poor starting points. Quadratic convergence near a regular root."""
    x = np.asarray(x0, float).copy()
    f = np.asarray(F(x), float)
    norm = float(np.linalg.norm(f))
    res_hist, x_hist = [norm], [x.copy()]
    for k in range(1, maxiter + 1):
        Jk = np.asarray(J(x), float) if J is not None else jacobian(F, x)
        try:
            dx = np.linalg.solve(Jk, -f)
        except np.linalg.LinAlgError:
            return SystemResult(x, False, k, res_hist, x_hist)
        t = 1.0
        while True:
            x_new = x + t * dx
            f_new = np.asarray(F(x_new), float)
            new_norm = float(np.linalg.norm(f_new))
            if not line_search or (np.isfinite(new_norm) and new_norm <= (1 - 1e-4 * t) * norm) or t < 1e-4:
                break
            t /= 2
        x, f, norm = x_new, f_new, new_norm
        res_hist.append(norm)
        x_hist.append(x.copy())
        if norm < tol or np.linalg.norm(t * dx) < tol * (1 + np.linalg.norm(x)):
            return SystemResult(x, bool(norm < max(tol, 1e-8)), k, res_hist, x_hist)
    return SystemResult(x, False, maxiter, res_hist, x_hist)
