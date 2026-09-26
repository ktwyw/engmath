"""Linear systems and eigenvalues: Gaussian elimination with partial pivoting, the Thomas
algorithm for tridiagonal systems, Jacobi and Gauss-Seidel iteration, and power iteration.

>>> import numpy as np
>>> from engmath import linalg
>>> A = np.array([[4., -1, 0], [-1, 4, -1], [0, -1, 4]]); b = np.array([2., 4, 10])
>>> linalg.gauss_solve(A, b).round(10).tolist()
[1.0, 2.0, 3.0]
"""

from __future__ import annotations

import numpy as np


def gauss_solve(A, b) -> np.ndarray:
    """Solve A x = b by Gaussian elimination with partial pivoting and back substitution."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).ravel()
    n = b.size
    if A.shape != (n, n):
        raise ValueError("A must be square and match the length of b.")
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(A[k:, k])))  # partial pivoting: largest pivot in the column
        if A[p, k] == 0:
            raise np.linalg.LinAlgError("Matrix is singular.")
        if p != k:
            A[[k, p]], b[[k, p]] = A[[p, k]], b[[p, k]]
        m = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k:] -= np.outer(m, A[k, k:])
        b[k + 1 :] -= m * b[k]
    if A[-1, -1] == 0:
        raise np.linalg.LinAlgError("Matrix is singular.")
    x = np.empty(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1 :] @ x[i + 1 :]) / A[i, i]
    return x


def thomas(lower, diag, upper, rhs) -> np.ndarray:
    """Tridiagonal solver in O(n): lower[i] multiplies x[i-1] (lower[0] unused), upper[i]
    multiplies x[i+1] (upper[-1] unused). Arises in 1D heat conduction, splines, implicit PDEs."""
    a, b, c, d = (np.array(v, dtype=float) for v in (lower, diag, upper, rhs))
    n = d.size
    cp, dp = np.empty(n), np.empty(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        den = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def iterative_solve(A, b, method: str = "gauss-seidel", tol: float = 1e-10, maxiter: int = 10000, x0=None):
    """Jacobi or Gauss-Seidel iteration. Converges for diagonally dominant A.
    Returns (x, residual_history)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    D = np.diag(A)
    if np.any(D == 0):
        raise ValueError("Zero on the diagonal: reorder the equations.")
    hist = []
    for _ in range(maxiter):
        if method == "jacobi":
            x = (b - (A @ x - D * x)) / D
        elif method == "gauss-seidel":
            for i in range(b.size):
                x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i + 1 :] @ x[i + 1 :]) / A[i, i]
        else:
            raise ValueError("method must be 'jacobi' or 'gauss-seidel'.")
        r = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
        hist.append(r)
        if r < tol:
            break
    return x, hist


def power_iteration(A, tol: float = 1e-12, maxiter: int = 10000, seed: int = 0):
    """Largest-magnitude eigenvalue and its eigenvector by repeated multiplication.
    Inverse iteration (power iteration on A^-1) gives the smallest one - e.g. the fundamental
    frequency of a vibrating structure."""
    A = np.asarray(A, dtype=float)
    v = np.random.default_rng(seed).standard_normal(A.shape[0])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(maxiter):
        w = A @ v
        lam_new = float(v @ w)
        v = w / np.linalg.norm(w)
        if abs(lam_new - lam) < tol * max(1.0, abs(lam_new)):
            return lam_new, v
        lam = lam_new
    return lam, v
