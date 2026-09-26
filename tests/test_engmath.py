import numpy as np
import pytest

from engmath import (
    differentiate,
    fitting,
    integrate,
    interpolate,
    linalg,
    odes,
    optimize,
    pdes,
    reporting,
    roots,
    smoothing,
)


# ---------------------------------------------------------------- roots
def test_bisection_requires_bracket():
    with pytest.raises(ValueError, match="bracketed"):
        roots.bisection(lambda x: x**2 + 1, -1, 1)


def test_newton_divergence_is_reported():
    r = roots.newton(lambda x: np.arctan(x), 3.0, maxiter=30)  # classic divergent start
    assert not r.converged


def test_newton_zero_derivative():
    r = roots.newton(lambda x: x**2 - 1, 0.0, df=lambda x: 2 * x)
    assert not r.converged


def test_all_methods_agree():
    f = lambda x: np.exp(-x) - x  # noqa: E731
    vals = [roots.bisection(f, 0, 1).root, roots.newton(f, 0.5).root, roots.secant(f, 0, 1).root,
            roots.brent(f, 0, 1).root]
    assert np.ptp(vals) < 1e-10 and "converged" in str(roots.newton(f, 0.5))


def test_newton_needs_fewer_iterations_than_bisection():
    f = lambda x: x**3 - 2  # noqa: E731
    assert roots.newton(f, 1.0).iterations < roots.bisection(f, 0, 2).iterations / 4


# ---------------------------------------------------------------- optimisation
def test_golden_section_quadratic():
    r = optimize.golden_section(lambda x: (x - 1.3) ** 2, -5, 5)
    assert r.converged and r.x[0] == pytest.approx(1.3, abs=1e-8)


def test_nelder_mead_history_decreases():
    r = optimize.nelder_mead(lambda p: np.sum((p - 3) ** 2), [0.0, 0.0, 0.0])
    assert r.converged and np.all(np.diff(r.history) <= 1e-15) and np.allclose(r.x, 3, atol=1e-5)


# ---------------------------------------------------------------- linear algebra
def test_gauss_errors():
    with pytest.raises(ValueError):
        linalg.gauss_solve(np.ones((2, 3)), [1, 2])
    with pytest.raises(np.linalg.LinAlgError):
        linalg.gauss_solve([[1, 2], [2, 4]], [1, 2])


def test_iterative_errors_and_history():
    with pytest.raises(ValueError):
        linalg.iterative_solve([[0, 1], [1, 0]], [1, 1])
    with pytest.raises(ValueError):
        linalg.iterative_solve([[4, 1], [1, 4]], [1, 1], method="sor")
    A = np.array([[4.0, 1], [1, 3]])
    _, hj = linalg.iterative_solve(A, [1, 2], "jacobi")
    _, hg = linalg.iterative_solve(A, [1, 2], "gauss-seidel")
    assert len(hg) < len(hj)  # Gauss-Seidel converges faster here


def test_power_iteration_symmetric():
    A = np.array([[2.0, 1], [1, 2]])
    lam, v = linalg.power_iteration(A)
    assert lam == pytest.approx(3) and abs(abs(v[0]) - abs(v[1])) < 1e-6


# ---------------------------------------------------------------- interpolation
def test_linear_refuses_extrapolation():
    with pytest.raises(ValueError, match="outside"):
        interpolate.linear([0, 1, 2], [0, 1, 4], [3])
    assert interpolate.linear([0, 1, 2], [0, 1, 4], [3], extrapolate=True)[0] == pytest.approx(7)
    with pytest.raises(ValueError):
        interpolate.linear([0, 2, 1], [0, 1, 2], [0.5])


def test_spline_passes_through_points_and_validates():
    x = np.array([0, 1, 2.5, 4])
    y = np.array([1, 3, 2, 5.0])
    s = interpolate.CubicSpline(x, y)
    assert np.allclose(s(x), y)
    with pytest.raises(ValueError):
        interpolate.CubicSpline([0, 1], [0, 1])


def test_runge_phenomenon():
    xn = np.linspace(-1, 1, 15)
    f = lambda x: 1 / (1 + 25 * x**2)  # noqa: E731
    xq = np.linspace(-1, 1, 500)
    err_poly = np.max(np.abs(interpolate.lagrange(xn, f(xn), xq) - f(xq)))
    err_spline = np.max(np.abs(interpolate.CubicSpline(xn, f(xn))(xq) - f(xq)))
    assert err_poly > 1 and err_spline < 0.05


# ---------------------------------------------------------------- integration
def test_integration_rules_and_errors():
    assert integrate.trapezoid([1, 1, 1], dx=0.5) == pytest.approx(1.0)
    assert integrate.simpson([0, 1, 4], dx=1) == pytest.approx(8 / 3)
    with pytest.raises(ValueError):
        integrate.simpson([1, 2])
    with pytest.raises(ValueError):
        integrate.composite(np.sin, 0, 1, 3, "simpson")
    with pytest.raises(ValueError):
        integrate.composite(np.sin, 0, 1, 4, "boole")


def test_romberg_table_improves():
    R = integrate.romberg(np.sin, 0, np.pi, 5)
    assert abs(R[-1, -1] - 2) < abs(R[-1, 0] - 2) < abs(R[0, 0] - 2)


# ---------------------------------------------------------------- differentiation
def test_difference_accuracy():
    err_central = abs(differentiate.central(np.sin, 1.0) - np.cos(1))
    assert err_central < abs(differentiate.forward(np.sin, 1.0, 1e-5) - np.cos(1))
    with pytest.raises(ValueError):
        differentiate.gradient([0, 1], [0, 1])


# ---------------------------------------------------------------- smoothing
def test_smoothing_errors_and_shapes():
    with pytest.raises(ValueError):
        smoothing.moving_average([1, 2, 3], 2)
    with pytest.raises(ValueError):
        smoothing.savitzky_golay(np.ones(5), 3, 2)
    y = np.random.default_rng(0).normal(size=50)
    assert smoothing.savitzky_golay(y, 4, 2).shape == y.shape and smoothing.lowess(np.arange(50.0), y).shape == y.shape


def test_savgol_preserves_polynomials():
    x = np.arange(30.0)
    y = 0.5 * x**2 - x
    assert np.allclose(smoothing.savitzky_golay(y, 3, 2), y)
    assert np.allclose(smoothing.savitzky_golay(y, 3, 2, deriv=1), x - 1)


def test_lowess_robust_iterations_resist_outlier():
    x = np.arange(40.0)
    y = x.copy()
    y[20] = 100
    plain = smoothing.lowess(x, y, 0.3)
    robust = smoothing.lowess(x, y, 0.3, robust_iters=2)
    assert abs(robust[20] - 20) < abs(plain[20] - 20)


# ---------------------------------------------------------------- fitting
def test_fit_result_helpers():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 30)
    f = fitting.linear_lstsq(np.column_stack([np.ones(30), x]), 1 + 2 * x + rng.normal(0, 0.05, 30))
    ci = f.conf_int()
    assert ci[1, 0] < 2 < ci[1, 1] and f.correlation.shape == (2, 2) and f.correlation[0, 0] == pytest.approx(1)


def test_huber_weights_flag_outlier():
    x = np.arange(10.0)
    y = x.copy()
    y[9] = 40
    h = fitting.huber_irls(np.column_stack([np.ones(10), x]), y)
    assert h.weights[9] < 0.2 and h.iterations > 1


def test_levenberg_marquardt_exponential():
    x = np.linspace(0, 4, 25)
    y = 3 * np.exp(-0.7 * x) + np.random.default_rng(2).normal(0, 0.01, 25)
    r = fitting.levenberg_marquardt(lambda x, a, k: a * np.exp(-k * x), x, y, [1, 1])
    assert r.coef == pytest.approx([3, 0.7], rel=0.02) and r.history[-1] < r.history[0]


# ---------------------------------------------------------------- ODEs and PDEs
def test_ode_system_shapes():
    t, y = odes.rk4(lambda t, y: [y[1], -y[0]], [1.0, 0.0], 0, np.pi, 200)
    assert y.shape == (201, 2) and y[-1, 0] == pytest.approx(-1, abs=1e-6)


def test_rk23_adapts_step():
    t, y = odes.rk23(lambda t, y: -50 * (y - np.sin(t)), [0.0], 0, 3)
    steps = np.diff(t)
    assert steps.max() / steps.min() > 3


def test_ftcs_and_cn_agree():
    c0 = np.zeros(51)
    a = pdes.ftcs(c0, 1.0, 0.02, 0.0001, 500, 1.0, 0.0)
    b = pdes.crank_nicolson(c0, 1.0, 0.02, 0.0005, 100, 1.0, 0.0)
    assert np.max(np.abs(a - b)) < 5e-3


# ---------------------------------------------------------------- reporting
@pytest.mark.parametrize("value,u,text", [(2.29987, 0.08907, "2.30 ± 0.09"), (0.348, 0.0081, "0.348 ± 0.008"),
                                          (10523.4, 156.7, "10520 ± 160"), (9.87654, 0.0963, "9.88 ± 0.10"),
                                          (51.66, 1.23, "51.7 ± 1.2")])
def test_format_uncertainty(value, u, text):
    assert reporting.format_uncertainty(value, u) == text


def test_format_uncertainty_units_and_errors():
    assert reporting.format_uncertainty(1.2e-4, 1.2e-5, unit="g/mL").endswith("e-04 g/mL")
    with pytest.raises(ValueError):
        reporting.format_uncertainty(1.0, 0.0)


def test_propagate_contributions_sum_to_one():
    y, u, c = reporting.propagate(lambda a, b: a * b, {"a": 2.0, "b": 5.0}, {"a": 0.1, "b": 0.1})
    assert y == 10 and sum(c.values()) == pytest.approx(1) and c["a"] > c["b"]
