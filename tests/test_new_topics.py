import numpy as np
import pytest

from engmath import bvp, fourier, montecarlo, nonlinear, optimize, sparse


# ---------------------------------------------------------------- nonlinear systems
def test_newton_system_with_analytic_jacobian():
    F = lambda v: [v[0] ** 2 - v[1] - 1, v[0] + v[1] ** 2 - 7]  # noqa: E731
    J = lambda v: [[2 * v[0], -1], [1, 2 * v[1]]]  # noqa: E731
    a = nonlinear.newton_system(F, [1.0, 1.0], J=J)
    b = nonlinear.newton_system(F, [1.0, 1.0])
    assert a.converged and np.allclose(a.x, b.x, atol=1e-8) and np.allclose(F(a.x), 0, atol=1e-10)
    assert "converged" in str(a)


def test_line_search_rescues_poor_start():
    F = lambda v: [np.arctan(v[0]), v[1] - 1]  # noqa: E731  plain Newton diverges from x = 2
    assert nonlinear.newton_system(F, [2.0, 0.0], line_search=True).converged
    assert not nonlinear.newton_system(F, [2.0, 0.0], line_search=False, maxiter=15).converged


def test_singular_jacobian_reported():
    r = nonlinear.newton_system(lambda v: [v[0] ** 2, v[1] ** 2 + 1], [0.0, 0.0])
    assert not r.converged


# ---------------------------------------------------------------- boundary-value problems
def test_shooting_dirichlet_matches_fd():
    xs, ys, _, _ = bvp.shooting(lambda x, y, yp: -y, 0, 1, 0.0, ("dirichlet", 1.0), n=200)
    exact = np.sin(xs) / np.sin(1)
    assert np.max(np.abs(ys - exact)) < 1e-9
    xf, yf = bvp.finite_difference(lambda x: 0 * x, lambda x: 1 + 0 * x, lambda x: 0 * x, 0, 1, 200,
                                   ("dirichlet", 0.0), ("dirichlet", 1.0))
    assert np.max(np.abs(yf - np.sin(xf) / np.sin(1))) < 1e-5


def test_bad_boundary_condition():
    with pytest.raises(ValueError):
        bvp.shooting(lambda x, y, yp: 0 * y, 0, 1, 0.0, ("neumann", 1.0))


# ---------------------------------------------------------------- Fourier
def test_fft_requires_power_of_two_and_detects_frequency():
    with pytest.raises(ValueError):
        fourier.fft(np.ones(12))
    fs = 512.0
    t = np.arange(512) / fs
    f, a = fourier.spectrum(np.sin(2 * np.pi * 37 * t), fs)
    assert f[np.argmax(a)] == pytest.approx(37)
    with pytest.raises(ValueError):
        fourier.spectrum(np.ones(8), 1.0, window="kaiser")


def test_hann_reduces_leakage():
    fs, t = 1000.0, np.arange(1000) / 1000.0
    x = np.sin(2 * np.pi * 50.5 * t)  # between two bins: leakage
    f, rect = fourier.spectrum(x, fs, window=None)
    _, hann = fourier.spectrum(x, fs, window="hann")
    far = np.abs(f - 50.5) > 20
    assert hann[far].max() < rect[far].max() / 10


def test_bandpass_filter():
    fs, t = 1000.0, np.arange(2000) / 1000.0
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 300 * t)
    y = fourier.fft_filter(x, fs, low=50, high=200)
    assert np.allclose(y, np.sin(2 * np.pi * 100 * t), atol=1e-9)


# ---------------------------------------------------------------- optimisation
def test_simplex_unbounded_and_infeasible_origin():
    with pytest.raises(ValueError, match="unbounded"):
        optimize.simplex_lp([1, 1], [[1, -1]], [1])
    with pytest.raises(ValueError):
        optimize.simplex_lp([1, 1], [[1, 1]], [-1])


def test_simplex_visits_improving_vertices():
    lp = optimize.simplex_lp([3, 5], [[1, 0], [0, 2], [3, 2]], [4, 12, 18])  # classic textbook LP
    assert lp.objective == pytest.approx(36) and np.allclose(lp.x, [2, 6])
    profits = [3 * v[0] + 5 * v[1] for v in lp.vertices]
    assert all(b >= a for a, b in zip(profits, profits[1:]))


def test_penalty_simple_constraint():
    r = optimize.penalty_minimize(lambda v: (v[0] - 3) ** 2 + (v[1] - 3) ** 2, [0.0, 0.0],
                                  ineq=[lambda v: v[0] + v[1] - 2])
    assert np.allclose(r.x, [1, 1], atol=1e-3) and len(r.history) == 8


# ---------------------------------------------------------------- Monte Carlo
def test_mc_error_shrinks_with_n():
    f = lambda p: np.exp(-np.sum(p ** 2, axis=1))  # noqa: E731
    _, se1 = montecarlo.integrate(f, [(0, 1)] * 4, 10_000)
    _, se2 = montecarlo.integrate(f, [(0, 1)] * 4, 160_000)
    assert se2 == pytest.approx(se1 / 4, rel=0.1)


def test_random_walk_shape_and_steps():
    w = montecarlo.random_walk(50, 10, dim=3, step=0.5, seed=1)
    assert w.shape == (51, 10, 3) and np.allclose(w[0], 0)
    steps = np.abs(np.diff(w, axis=0)).sum(axis=2)
    assert np.allclose(steps, 0.5)


# ---------------------------------------------------------------- sparse
def test_laplacian_properties():
    A = sparse.laplacian_2d(10, 8, 0.1, 0.1)
    assert A.shape == (80, 80) and A.nnz <= 5 * 80
    assert abs(A - A.T).max() < 1e-12
    assert np.all(np.linalg.eigvalsh(A.toarray()) > 0)


def test_poisson_solvers_agree_and_errors():
    f = lambda x, y: 1 + 0 * x  # noqa: E731
    g = lambda x, y: 0 * x  # noqa: E731
    _, _, U1 = sparse.poisson_dirichlet(f, g, 1, 1, 25, 25)
    _, _, U2 = sparse.poisson_dirichlet(f, g, 1, 1, 25, 25, solver="cg")
    assert np.max(np.abs(U1 - U2)) < 1e-9 and U1.max() == pytest.approx(0.0737, abs=1e-3)
    with pytest.raises(ValueError):
        sparse.poisson_dirichlet(f, g, 1, 1, 5, 5, solver="magic")


def test_cg_converges_in_at_most_n_steps():
    A = np.diag([1.0, 2, 3, 4, 5])
    _, hist = sparse.conjugate_gradient(A, np.ones(5), tol=1e-12)
    assert len(hist) - 1 <= 5
