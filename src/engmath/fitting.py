"""Least-squares fitting from scratch: linear models with standard errors, robust (Huber IRLS)
regression, and Levenberg-Marquardt for nonlinear models.

For a full statistical treatment (prediction bands, diagnostics, calibration) see the companion
library engstat; here the emphasis is on HOW the algorithms work.

>>> import numpy as np
>>> from engmath import fitting
>>> x = np.arange(6.0); y = 2 + 3 * x
>>> fitting.linear_lstsq(np.column_stack([np.ones(6), x]), y).coef.round(10).tolist()
[2.0, 3.0]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class FitResult:
    coef: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    residuals: np.ndarray
    sigma: float  # residual standard deviation
    dof: int
    iterations: int = 0
    history: list = field(default_factory=list)
    weights: np.ndarray | None = None

    def conf_int(self, conf: float = 0.95) -> np.ndarray:
        t = stats.t.ppf(0.5 + conf / 2, self.dof)
        return np.column_stack([self.coef - t * self.se, self.coef + t * self.se])

    @property
    def correlation(self) -> np.ndarray:
        """Parameter correlation matrix: |r| near 1 means parameters cannot be told apart."""
        d = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(d, d)


def linear_lstsq(X, y, weights=None) -> FitResult:
    """Solve min ||W^(1/2)(y - X b)||^2 with QR (numerically safer than the normal equations)."""
    X, y = np.atleast_2d(np.asarray(X, float)), np.asarray(y, float)
    w = np.ones(y.size) if weights is None else np.asarray(weights, float)
    sw = np.sqrt(w)
    Q, R = np.linalg.qr(X * sw[:, None])
    coef = np.linalg.solve(R, Q.T @ (y * sw))
    resid = y - X @ coef
    dof = y.size - X.shape[1]
    sigma = float(np.sqrt(np.sum(w * resid**2) / dof))
    Rinv = np.linalg.inv(R)
    cov = sigma**2 * Rinv @ Rinv.T
    return FitResult(coef, np.sqrt(np.diag(cov)), cov, resid, sigma, dof)


def huber_irls(X, y, c: float = 1.345, tol: float = 1e-10, maxiter: int = 100) -> FitResult:
    """Robust regression by iteratively reweighted least squares with Huber weights
    w = min(1, c / |r/s|), s = MAD/0.6745. Outliers get weights < 1 instead of dominating."""
    X, y = np.atleast_2d(np.asarray(X, float)), np.asarray(y, float)
    fit = linear_lstsq(X, y)
    hist = [fit.coef.copy()]
    w = np.ones(y.size)
    for k in range(1, maxiter + 1):
        r = y - X @ fit.coef
        s = np.median(np.abs(r - np.median(r))) / 0.6745 or 1e-12
        u = np.abs(r / s)
        w = np.where(u <= c, 1.0, c / np.maximum(u, 1e-12))
        new = linear_lstsq(X, y, w)
        hist.append(new.coef.copy())
        if np.max(np.abs(new.coef - fit.coef)) < tol * (1 + np.max(np.abs(fit.coef))):
            new.iterations, new.history, new.weights = k, hist, w
            return new
        fit = new
    fit.iterations, fit.history, fit.weights = maxiter, hist, w
    return fit


def jacobian(func, x, p, rel_step: float = 1e-7) -> np.ndarray:
    """Numerical Jacobian d func(x, p) / d p by forward differences."""
    p = np.asarray(p, float)
    f0 = np.asarray(func(x, *p), float)
    J = np.empty((f0.size, p.size))
    for j in range(p.size):
        dp = rel_step * max(abs(p[j]), 1e-8)
        pp = p.copy()
        pp[j] += dp
        J[:, j] = (np.asarray(func(x, *pp), float) - f0) / dp
    return J


def levenberg_marquardt(func, x, y, p0, tol: float = 1e-12, maxiter: int = 500, lam0: float = 1e-3) -> FitResult:
    """Nonlinear least squares min sum (y - func(x, p))^2 by Levenberg-Marquardt.

    Each step solves (J^T J + lam diag(J^T J)) dp = J^T r. Small lam -> Gauss-Newton (fast near the
    optimum); large lam -> gradient descent (robust far away). lam adapts to success/failure."""
    y = np.asarray(y, float)
    p = np.asarray(p0, float).copy()
    r = y - func(x, *p)
    rss = float(r @ r)
    if not np.isfinite(rss):
        raise ValueError("The model is not finite at the starting values (e.g. a singularity inside the data "
                         "range); choose better starting values.")
    lam = lam0
    hist = [rss]
    k = 0
    for k in range(1, maxiter + 1):  # noqa: B007 - k is reported as the iteration count
        J = jacobian(func, x, p)
        A = J.T @ J
        g = J.T @ r
        dp = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-300), g)
        p_new = p + dp
        r_new = y - func(x, *p_new)
        rss_new = float(r_new @ r_new)
        if np.isfinite(rss_new) and rss_new < rss:
            converged = (rss - rss_new) <= tol * rss or np.max(np.abs(dp) / (np.abs(p) + 1e-12)) < 1e-10
            p, r, rss, lam = p_new, r_new, rss_new, max(lam / 10, 1e-12)
            hist.append(rss)
            if converged:
                break
        else:
            lam *= 10
            if lam > 1e12:
                break
    dof = y.size - p.size
    sigma = float(np.sqrt(rss / dof))
    J = jacobian(func, x, p)
    cov = sigma**2 * np.linalg.inv(J.T @ J)
    return FitResult(p, np.sqrt(np.diag(cov)), cov, r, sigma, dof, k, hist)
