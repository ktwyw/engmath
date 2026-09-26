"""Sparse matrices for 2D problems: the five-point Laplacian built with Kronecker products, a Poisson
solver for rectangles, and the conjugate-gradient method written from scratch.

>>> import numpy as np
>>> from engmath import sparse
>>> X, Y, U = sparse.poisson_dirichlet(lambda x, y: -4 + 0*x, lambda x, y: x**2 + y**2, 1, 1, 20, 20)
>>> float(np.max(np.abs(U - (X**2 + Y**2)))) < 1e-10      # quadratics are exact for 5 points
True
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def laplacian_1d(n: int, h: float) -> sp.csr_matrix:
    """-d2/dx2 on n interior nodes (Dirichlet ends): tridiag(-1, 2, -1) / h^2."""
    return sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csr") / h**2


def laplacian_2d(nx: int, ny: int, dx: float, dy: float) -> sp.csr_matrix:
    """Five-point -Laplacian on an nx-by-ny grid of interior nodes (x index fastest):
    A = I_y (x) T_x + T_y (x) I_x, where (x) is the Kronecker product."""
    return (sp.kron(sp.identity(ny), laplacian_1d(nx, dx)) + sp.kron(laplacian_1d(ny, dy), sp.identity(nx))).tocsr()


def poisson_dirichlet(f, g, Lx: float, Ly: float, nx: int, ny: int, solver: str = "direct"):
    """Solve -Laplacian(u) = f on [0, Lx] x [0, Ly] with u = g on the boundary.

    nx, ny are numbers of interior nodes. Returns full grids (X, Y, U) including the boundary.
    ``solver`` = 'direct' (sparse LU, scipy.sparse.linalg.spsolve) or 'cg' (conjugate gradient below)."""
    dx, dy = Lx / (nx + 1), Ly / (ny + 1)
    x, y = np.linspace(0, Lx, nx + 2), np.linspace(0, Ly, ny + 2)
    X, Y = np.meshgrid(x, y)  # shape (ny + 2, nx + 2)
    U = np.asarray(g(X, Y), float) * np.ones_like(X)
    rhs = np.asarray(f(X[1:-1, 1:-1], Y[1:-1, 1:-1]), float) * np.ones((ny, nx))
    rhs[:, 0] += U[1:-1, 0] / dx**2  # known boundary values move to the right-hand side
    rhs[:, -1] += U[1:-1, -1] / dx**2
    rhs[0, :] += U[0, 1:-1] / dy**2
    rhs[-1, :] += U[-1, 1:-1] / dy**2
    A = laplacian_2d(nx, ny, dx, dy)
    if solver == "direct":
        u = spla.spsolve(A.tocsc(), rhs.ravel())
    elif solver == "cg":
        u = conjugate_gradient(A, rhs.ravel(), tol=1e-12)[0]
    else:
        raise ValueError("solver must be 'direct' or 'cg'.")
    U[1:-1, 1:-1] = u.reshape(ny, nx)
    return X, Y, U


def conjugate_gradient(A, b, tol: float = 1e-10, maxiter: int | None = None, x0=None):
    """Conjugate gradient for symmetric positive-definite A. Needs only products A @ p, so A may be
    sparse (or any object with @). Returns (x, relative-residual history)."""
    b = np.asarray(b, float)
    x = np.zeros_like(b) if x0 is None else np.array(x0, float)
    r = b - A @ x
    p = r.copy()
    rr = r @ r
    bnorm = np.linalg.norm(b) or 1.0
    hist = [np.sqrt(rr) / bnorm]
    for _ in range(maxiter or 10 * b.size):
        Ap = A @ p
        alpha = rr / (p @ Ap)  # exact line search along p
        x += alpha * p
        r -= alpha * Ap
        rr_new = r @ r
        hist.append(np.sqrt(rr_new) / bnorm)
        if hist[-1] < tol:
            break
        p = r + (rr_new / rr) * p  # new direction, A-conjugate to all previous ones
        rr = rr_new
    return x, hist
