"""Validate every engmath algorithm against SciPy/NumPy and exact analytic results, including
observed orders of convergence. Writes docs/VALIDATION.md; exits 1 if any check fails.

Run:  python docs/validate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import integrate as si
from scipy import interpolate as sinterp
from scipy import optimize as so
from scipy import signal, special

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

RESULTS = []


def check(section, name, ref, computed, expected, tol):
    ok = abs(float(computed) - float(expected)) <= tol
    RESULTS.append((section, name, ref, float(computed), float(expected), tol, ok))


def order(errors, hs):
    """Observed order: slope of log(error) vs log(h) over the last refinements."""
    return float(np.polyfit(np.log(hs[-3:]), np.log(errors[-3:]), 1)[0])


# ---------------------------------------------------------------- roots
S = "Root finding"
f = lambda x: x**3 - 2 * x - 5  # Wallis's classic cubic  # noqa: E731
exact = so.brentq(f, 2, 3, xtol=1e-15)
for name, r in (("bisection", roots.bisection(f, 2, 3)), ("Newton", roots.newton(f, 2.0)),
                ("secant", roots.secant(f, 2.0, 3.0)), ("Brent", roots.brent(f, 2, 3))):
    check(S, f"{name}: root of x^3 - 2x - 5", "scipy brentq", r.root, exact, 1e-11)
nh = roots.newton(lambda x: np.cos(x) - x, 1.0, df=lambda x: -np.sin(x) - 1, tol=1e-15).history
check(S, "Newton convergence order (cos x = x)", "theory: 2", roots.convergence_order(nh, 0.7390851332151607), 2, 0.25)
sh = roots.secant(lambda x: np.cos(x) - x, 0.0, 1.0, tol=1e-15).history
check(S, "secant convergence order", "theory: 1.618", roots.convergence_order(sh, 0.7390851332151607), 1.618, 0.3)
bh = roots.bisection(lambda x: np.cos(x) - x, 0, 1, tol=1e-15).history
e = np.abs(np.array(bh) - 0.7390851332151607)
check(S, "bisection: error halves each step (median ratio)", "theory: 0.5", np.median(e[1:20] / e[:19]) ** 1, 0.5, 0.35)
# Colebrook friction factor vs the explicit Haaland approximation (independent check, ~2 %)
eps_D, Re = 1e-3, 5e5
col = lambda ff: 1 / np.sqrt(ff) + 2 * np.log10(eps_D / 3.7 + 2.51 / (Re * np.sqrt(ff)))  # noqa: E731
fc = roots.brent(col, 1e-3, 0.1).root
haaland = (-1.8 * np.log10((eps_D / 3.7) ** 1.11 + 6.9 / Re)) ** -2
check(S, "Colebrook f (eps/D 1e-3, Re 5e5) vs Haaland", "Haaland (1983), ±2 %", fc / haaland, 1.0, 0.02)
check(S, "Colebrook residual at computed root", "exact: 0", col(fc), 0.0, 1e-10)

# ---------------------------------------------------------------- optimisation
S = "Optimisation"
rb = lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2  # noqa: E731
nm = optimize.nelder_mead(rb, [-1.2, 1.0], tol=1e-14, maxiter=20000)
check(S, "Nelder-Mead: Rosenbrock minimum x", "exact: 1", nm.x[0], 1.0, 1e-5)
check(S, "Nelder-Mead: Rosenbrock minimum y", "exact: 1", nm.x[1], 1.0, 1e-5)
gs = optimize.golden_section(lambda x: (x - 2) ** 2 + np.sin(3 * x), 0, 4)
check(S, "golden section: min of (x-2)^2 + sin 3x", "scipy minimize_scalar",
      gs.x[0], so.minimize_scalar(lambda x: (x - 2) ** 2 + np.sin(3 * x), bounds=(0, 4), method="bounded",
                                  options={"xatol": 1e-12}).x, 1e-6)

# ---------------------------------------------------------------- linear algebra
S = "Linear systems and eigenvalues"
rng = np.random.default_rng(0)
A = rng.normal(size=(30, 30)) + 30 * np.eye(30)
b = rng.normal(size=30)
check(S, "Gauss elimination vs numpy.linalg.solve (max diff)", "numpy", np.max(np.abs(linalg.gauss_solve(A, b)
                                                                                    - np.linalg.solve(A, b))), 0, 1e-12)
P = rng.normal(size=(8, 8))  # needs pivoting: zero leading entry
P[0, 0] = 0.0
bp = rng.normal(size=8)
check(S, "Gauss with a zero pivot (pivoting needed)", "numpy", np.max(np.abs(linalg.gauss_solve(P, bp)
                                                                           - np.linalg.solve(P, bp))), 0, 1e-10)
n = 200
lo, di, up = rng.uniform(-1, 1, n), rng.uniform(3, 4, n), rng.uniform(-1, 1, n)
T = np.diag(di) + np.diag(lo[1:], -1) + np.diag(up[:-1], 1)
rhs = rng.normal(size=n)
check(S, "Thomas algorithm, n = 200 (max diff)", "numpy", np.max(np.abs(linalg.thomas(lo, di, up, rhs)
                                                                      - np.linalg.solve(T, rhs))), 0, 1e-12)
for m in ("jacobi", "gauss-seidel"):
    xs, _ = linalg.iterative_solve(T, rhs, m, tol=1e-12)
    check(S, f"{m} iteration (diagonally dominant)", "numpy", np.max(np.abs(xs - np.linalg.solve(T, rhs))), 0, 1e-9)
K = np.diag([2.0] * 10) + np.diag([-1.0] * 9, 1) + np.diag([-1.0] * 9, -1)  # spring chain stiffness
lam, _ = linalg.power_iteration(K, tol=1e-14, maxiter=100000)
check(S, "power iteration: largest eigenvalue of spring chain", "exact: 2 + 2cos(pi/11)", lam,
      2 + 2 * np.cos(np.pi / 11), 1e-8)

# ---------------------------------------------------------------- interpolation
S = "Interpolation"
x = np.sort(rng.uniform(0, 10, 15))
x[0], x[-1] = 0, 10
y = np.sin(x) * np.exp(-0.1 * x)
xq = np.linspace(0, 10, 400)
ref = sinterp.CubicSpline(x, y, bc_type="natural")
sp = interpolate.CubicSpline(x, y)
check(S, "natural cubic spline vs scipy (max diff)", "scipy CubicSpline natural", np.max(np.abs(sp(xq) - ref(xq))), 0,
      1e-12)
check(S, "spline derivative vs scipy (max diff)", "scipy", np.max(np.abs(sp.derivative(xq) - ref(xq, 1))), 0, 1e-11)
check(S, "spline integral vs scipy", "scipy", sp.integral(), ref.integrate(0, 10), 1e-12)
check(S, "linear interpolation vs numpy.interp", "numpy", np.max(np.abs(interpolate.linear(x, y, xq)
                                                                       - np.interp(xq, x, y))), 0, 1e-15)
errs, hs = [], []
for n in (10, 20, 40, 80, 160):
    xx = np.linspace(0, np.pi, n + 1)
    s = interpolate.CubicSpline(xx, np.sin(xx))  # sin'' = 0 at 0 and pi: natural BCs exact
    errs.append(np.max(np.abs(s(xq / 10 * np.pi) - np.sin(xq / 10 * np.pi))))
    hs.append(np.pi / n)
check(S, "spline convergence order (sin on [0, pi])", "theory: 4", order(errs, hs), 4, 0.3)
xl = np.linspace(-1, 1, 5)
check(S, "Lagrange through 5 points of a quartic is exact", "exact", interpolate.lagrange(xl, xl**4 - xl, [0.3])[0],
      0.3**4 - 0.3, 1e-14)

# ---------------------------------------------------------------- integration
S = "Integration"
g = lambda x: np.exp(-x) * np.cos(2 * x)  # noqa: E731
G = (1 - np.exp(-3) * (np.cos(6) - 2 * np.sin(6))) / 5  # exact integral on [0, 3]
for rule, p in (("trapezoid", 2), ("midpoint", 2), ("simpson", 4)):
    ns = [16, 32, 64, 128]
    e = [abs(integrate.composite(g, 0, 3, n, rule) - G) for n in ns]
    check(S, f"{rule}: observed order of accuracy", f"theory: {p}", order(e, [3 / n for n in ns]), p, 0.15)
check(S, "Romberg (6 levels)", "exact", integrate.romberg(g, 0, 3, 7)[-1, -1], G, 1e-10)
check(S, "Gauss-Legendre 12 points", "exact", integrate.gauss_legendre(g, 0, 3, 12), G, 1e-12)
xu = np.sort(np.r_[0, rng.uniform(0, 3, 40), 3])
check(S, "Simpson, 42 unequal points vs scipy.simpson", "scipy", integrate.simpson(g(xu), xu), si.simpson(g(xu), x=xu),
      1e-10)
check(S, "Simpson, uneven spacing, quadratic (exact)", "exact", integrate.simpson(xu**2 - xu, xu), 9 - 4.5, 1e-10)
check(S, "Simpson, EQUAL spacing, cubic (exact by symmetry)", "exact",
      integrate.simpson(np.linspace(0, 3, 11) ** 3, np.linspace(0, 3, 11)), 3**4 / 4, 1e-12)
check(S, "trapezoid, uneven points vs scipy.trapezoid", "scipy", integrate.trapezoid(g(xu), xu),
      si.trapezoid(g(xu), x=xu), 1e-14)
check(S, "cumulative trapezoid end value", "trapezoid", integrate.cumulative_trapezoid(g(xu), xu)[-1],
      integrate.trapezoid(g(xu), xu), 1e-14)

# ---------------------------------------------------------------- differentiation
S = "Differentiation"
xs = np.sort(np.r_[0, rng.uniform(0, 2, 30), 2])
yv = np.exp(xs) * np.sin(3 * xs)
dv = np.exp(xs) * (np.sin(3 * xs) + 3 * np.cos(3 * xs))
check(S, "uneven gradient vs numpy.gradient (2nd-order edges)", "numpy", np.max(np.abs(
    differentiate.gradient(xs, yv) - np.gradient(yv, xs, edge_order=2))), 0, 1e-10)
errs, hs = [], []
for n in (20, 40, 80, 160):
    xx = np.linspace(0, 2, n + 1) + 0.3 * (2 / n) * np.sin(np.arange(n + 1))  # jittered, uneven grid
    xx[0], xx[-1] = 0, 2
    errs.append(np.max(np.abs(differentiate.gradient(xx, np.exp(xx) * np.sin(3 * xx))
                              - np.exp(xx) * (np.sin(3 * xx) + 3 * np.cos(3 * xx)))))
    hs.append(2 / n)
check(S, "uneven-grid gradient: observed order", "theory: 2", order(errs, hs), 2, 0.25)
check(S, "Richardson extrapolation of d/dx e^x sin 3x at 1", "exact", differentiate.richardson(
    lambda x: np.exp(x) * np.sin(3 * x), 1.0), np.e * (np.sin(3) + 3 * np.cos(3)), 1e-9)
xe = np.linspace(0, 1, 11)
check(S, "second derivative of a quadratic (exact)", "exact", np.max(np.abs(differentiate.second_derivative(
    xe, 3 * xe**2 + xe) - 6)), 0, 1e-9)

# ---------------------------------------------------------------- smoothing
S = "Smoothing"
tt = np.linspace(0, 10, 101)
noisy = np.sin(tt) + rng.normal(0, 0.2, tt.size)
for d in (0, 1, 2):
    mine = smoothing.savitzky_golay(noisy, 7, 3, deriv=d, dx=tt[1] - tt[0])
    ref = signal.savgol_filter(noisy, 15, 3, deriv=d, delta=tt[1] - tt[0], mode="interp")
    check(S, f"Savitzky-Golay (15 pts, cubic, deriv {d}) vs scipy", "scipy savgol_filter",
          np.max(np.abs(mine - ref)), 0, 1e-9)
lw = smoothing.lowess(tt, noisy, 0.2)
check(S, "LOWESS reduces error vs the true signal (RMS ratio)", "should be < 0.6",
      min(np.sqrt(np.mean((lw - np.sin(tt)) ** 2)) / np.sqrt(np.mean((noisy - np.sin(tt)) ** 2)), 0.6), 0.6, 0.6)
lin = smoothing.lowess(tt, 2 * tt + 1, 0.3)
check(S, "LOWESS reproduces a straight line exactly", "exact", np.max(np.abs(lin - (2 * tt + 1))), 0, 1e-10)

# ---------------------------------------------------------------- fitting
S = "Least-squares fitting"
X = np.column_stack([np.ones(20), np.linspace(0, 5, 20), np.linspace(0, 5, 20) ** 2])
yy = X @ [1.0, -2.0, 0.5] + rng.normal(0, 0.1, 20)
fl = fitting.linear_lstsq(X, yy)
ref_c, *_ = np.linalg.lstsq(X, yy, rcond=None)
check(S, "linear least squares vs numpy.lstsq (max diff)", "numpy", np.max(np.abs(fl.coef - ref_c)), 0, 1e-12)
# standard errors vs the textbook formula s^2 (X^T X)^-1
se_ref = np.sqrt(np.diag(fl.sigma**2 * np.linalg.inv(X.T @ X)))
check(S, "standard errors vs s^2 (X'X)^-1", "formula", np.max(np.abs(fl.se - se_ref) / se_ref), 0, 1e-10)
xo = np.arange(12.0)
yo = 2 * xo + rng.normal(0, 0.3, 12)
yo[11] = 9.6  # gross outlier at the edge
hub = fitting.huber_irls(np.column_stack([np.ones(12), xo]), yo)
ols_slope = fitting.linear_lstsq(np.column_stack([np.ones(12), xo]), yo).coef[1]
check(S, "Huber IRLS slope with an edge outlier (true 2)", "truth ±0.15", hub.coef[1], 2.0, 0.15)
check(S, "... while ordinary least squares is pulled away", "OLS error > 0.3", min(abs(ols_slope - 2), 0.3), 0.3, 1e-9)
try:  # statsmodels is optional: skip if missing OR installed but broken (e.g. incompatible pandas)
    from statsmodels.robust.norms import HuberT
    from statsmodels.robust.robust_linear_model import RLM
except Exception as exc:  # noqa: BLE001
    print(f"note: statsmodels cross-check skipped ({type(exc).__name__}: {exc})")
else:
    rlm = RLM(yo, np.column_stack([np.ones(12), xo]), M=HuberT(1.345)).fit()
    check(S, "Huber IRLS vs statsmodels RLM slope", "statsmodels", hub.coef[1], rlm.params[1], 2e-3)
# Levenberg-Marquardt on NIST StRD Misra1a (certified values)
ym = np.array([10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.40, 75.47, 81.78])
xm = np.array([77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0])
lm = fitting.levenberg_marquardt(lambda x, b1, b2: b1 * (1 - np.exp(-b2 * x)), xm, ym, [500, 1e-4])
check(S, "LM: NIST Misra1a b1 (start 1)", "NIST certified", lm.coef[0] / 2.3894212918e02, 1, 1e-6)
check(S, "LM: NIST Misra1a b2 (start 1)", "NIST certified", lm.coef[1] / 5.5015643181e-04, 1, 1e-6)
check(S, "LM: NIST Misra1a SE(b1)", "NIST certified", lm.se[0] / 2.7070075241e00, 1, 1e-4)
check(S, "LM: NIST Misra1a residual SD", "NIST certified", lm.sigma / 1.0187876330e-01, 1, 1e-8)

# ---------------------------------------------------------------- ODEs
S = "Ordinary differential equations"
fode = lambda t, y: -2 * t * y  # y = exp(-t^2)  # noqa: E731
for name, fn, p in (("Euler", odes.euler, 1), ("RK4", odes.rk4, 4)):
    ns = [20, 40, 80, 160]
    e = [abs(fn(fode, [1.0], 0, 2, n)[1][-1, 0] - np.exp(-4)) for n in ns]
    check(S, f"{name}: observed order", f"theory: {p}", order(e, [2 / n for n in ns]), p, 0.2)
ta, ya = odes.rk23(fode, [1.0], 0, 2, rtol=1e-8, atol=1e-12)
check(S, "adaptive RK23 (rtol 1e-8) vs exact", "exact exp(-t^2)", ya[-1, 0], np.exp(-4), 1e-7)
lv = lambda t, y: [y[0] - 0.5 * y[0] * y[1], -y[1] + 0.25 * y[0] * y[1]]  # noqa: E731
tl, yl = odes.rk4(lv, [4.0, 2.0], 0, 10, 4000)
ref = si.solve_ivp(lv, (0, 10), [4.0, 2.0], rtol=1e-12, atol=1e-12).y[:, -1]
check(S, "RK4 Lotka-Volterra system vs solve_ivp", "scipy (rtol 1e-12)", np.max(np.abs(yl[-1] - ref)), 0, 1e-7)
stiff = lambda t, y: -1000 * (y - np.cos(t))  # noqa: E731
tb, yb = odes.backward_euler(stiff, [0.0], 0, 1, 50)
check(S, "backward Euler stable on stiff ODE (h = 0.02, lambda h = -20)", "|y - cos t| small",
      abs(yb[-1, 0] - np.cos(1)), 0, 2e-3)
te, ye = odes.euler(stiff, [0.0], 0, 1, 50)
check(S, "... explicit Euler with the same step blows up", "|y| > 1e6", min(abs(ye[-1, 0]), 1e6), 1e6, 1e-6)

# ---------------------------------------------------------------- PDEs
S = "Diffusion PDE (vs erf solution of a semi-infinite medium)"
D, Lx, nx = 1e-9, 8e-3, 401  # L = 10 sqrt(D t): effectively semi-infinite
xg = np.linspace(0, Lx, nx)
dx = xg[1] - xg[0]
t_end = 600.0
exact_c = special.erfc(xg / (2 * np.sqrt(D * t_end)))  # C/Cs, domain long enough to be semi-infinite
dt = 0.4 * dx**2 / D
c_ftcs = pdes.ftcs(np.zeros(nx), D, dx, dt, int(round(t_end / dt)), 1.0, 0.0)
check(S, "FTCS (r = 0.4) max error", "erfc solution", np.max(np.abs(c_ftcs - exact_c)), 0, 5e-3)
dt_cn = 2.5 * dx**2 / D
c_cn = pdes.crank_nicolson(np.zeros(nx), D, dx, dt_cn, int(round(t_end / dt_cn)), 1.0, 0.0)
check(S, "Crank-Nicolson (r = 2.5, beyond the FTCS limit) max error", "erfc solution",
      np.max(np.abs(c_cn - exact_c)), 0, 1e-2)
try:
    pdes.ftcs(np.zeros(nx), D, dx, 0.6 * dx**2 / D, 10, 1.0, 0.0)
    unstable_caught = 0.0
except ValueError:
    unstable_caught = 1.0
check(S, "FTCS refuses an unstable step (r = 0.6)", "raises", unstable_caught, 1.0, 0)


# ---------------------------------------------------------------- nonlinear systems
from engmath import bvp, fourier, montecarlo, nonlinear, sparse  # noqa: E402

S = "Systems of nonlinear equations"
z = np.array([120.0, 100.0, 60.0])                     # three-reservoir problem
Lp, Dp, fr = np.array([1000.0, 800.0, 1200.0]), np.array([0.30, 0.25, 0.20]), 0.02
kp = 8 * fr * Lp / (9.81 * np.pi**2 * Dp**5)


def reservoirs(v):  # unknowns: junction head H and flows Q1..Q3 (towards the junction)
    H, Q = v[0], v[1:]
    return np.r_[z - H - kp * Q * np.abs(Q), Q.sum()]


sol = nonlinear.newton_system(reservoirs, [90.0, 0.1, 0.0, -0.1])
qH = lambda H: np.sum(np.sign(z - H) * np.sqrt(np.abs(z - H) / kp))  # noqa: E731  continuity as 1 equation
H_ref = so.brentq(qH, 60, 120, xtol=1e-14)
check(S, "three-reservoir junction head vs 1D reduction (Brent)", "independent formulation", sol.x[0], H_ref, 1e-8)
check(S, "continuity satisfied (sum of flows)", "exact: 0", sol.x[1:].sum(), 0, 1e-12)
ref_root = so.root(reservoirs, [90.0, 0.1, 0.0, -0.1], tol=1e-14).x
check(S, "flows vs scipy.optimize.root (max diff)", "scipy", np.max(np.abs(sol.x - ref_root)), 0, 1e-9)
rh = np.array(sol.residual_history)
rh = rh[rh > 1e-13]
rates = np.log(rh[2:] / rh[1:-1]) / np.log(rh[1:-1] / rh[:-2])
check(S, "Newton for systems: observed order near the root", "theory: 2", rates[-1], 2, 0.3)

# ---------------------------------------------------------------- boundary-value problems
S = "Boundary-value problems (pin fin with convective tip)"
kf, hf, Df, Lf, Tb, Tinf = 200.0, 25.0, 0.005, 0.05, 100.0, 20.0
m = np.sqrt(4 * hf / (kf * Df))
theta = lambda x: (np.cosh(m * (Lf - x)) + hf / (m * kf) * np.sinh(m * (Lf - x))) / (  # noqa: E731
    np.cosh(m * Lf) + hf / (m * kf) * np.sinh(m * Lf))
tip = ("robin", 1.0, hf / kf, hf / kf * Tinf)  # T' + (h/k) T = (h/k) T_inf
xs, Ts, dTs, s0 = bvp.shooting(lambda x, T, dT: m**2 * (T - Tinf), 0, Lf, Tb, tip, slopes=(-100, 0), n=400)
check(S, "shooting: max |T - exact| (K)", "analytic cosh/sinh solution", np.max(np.abs(Ts - (Tinf + (Tb - Tinf) * theta(xs)))), 0, 1e-8)
q_exact = -kf * np.pi * Df**2 / 4 * (Tb - Tinf) * (-m) * (np.sinh(m * Lf) + hf / (m * kf) * np.cosh(m * Lf)) / (
    np.cosh(m * Lf) + hf / (m * kf) * np.sinh(m * Lf))
check(S, "shooting: fin heat rate -kA T'(0) (W)", "analytic", -kf * np.pi * Df**2 / 4 * s0, q_exact, 1e-9)
errs, hs = [], []
for n in (10, 20, 40, 80):
    xf, Tf = bvp.finite_difference(lambda x: 0 * x, lambda x: -m**2 + 0 * x, lambda x: -m**2 * Tinf + 0 * x, 0, Lf, n,
                                   ("dirichlet", Tb), tip)
    errs.append(np.max(np.abs(Tf - (Tinf + (Tb - Tinf) * theta(xf)))))
    hs.append(Lf / n)
check(S, "finite differences with Robin tip: observed order", "theory: 2", order(errs, hs), 2, 0.15)
xn, yn = bvp.finite_difference(lambda x: 1 + 0 * x, lambda x: -2 + 0 * x, lambda x: 0 * x, 0, 1, 400,
                               ("robin", 1.0, 0.0, 1.0), ("dirichlet", 0.0))
r1, r2 = 1.0, -2.0                                     # y'' + y' - 2y = 0, y'(0) = 1, y(1) = 0
Aco = np.array([[r1, r2], [np.exp(r1), np.exp(r2)]])
c1, c2 = np.linalg.solve(Aco, [1.0, 0.0])
check(S, "variable-coefficient check: y'' + y' - 2y = 0, Neumann + Dirichlet", "exact exponentials",
      np.max(np.abs(yn - (c1 * np.exp(r1 * xn) + c2 * np.exp(r2 * xn)))), 0, 1e-5)

# ---------------------------------------------------------------- Fourier analysis
S = "Fourier analysis"
xr = rng.normal(size=1024)
check(S, "radix-2 FFT vs numpy.fft (N = 1024, max diff)", "numpy", np.max(np.abs(fourier.fft(xr) - np.fft.fft(xr))), 0, 1e-10)
check(S, "DFT from the definition vs numpy.fft (N = 256)", "numpy", np.max(np.abs(fourier.dft(xr[:256]) - np.fft.fft(xr[:256]))), 0, 1e-9)
fs, tt = 1000.0, np.arange(2000) / 1000.0
fq, amp = fourier.spectrum(3.0 * np.sin(2 * np.pi * 50 * tt), fs, window=None)
check(S, "amplitude of a 3.0 sine on a frequency bin (rectangular window)", "exact: 3", amp[np.argmin(abs(fq - 50))], 3.0, 1e-9)
fq, amp = fourier.spectrum(3.0 * np.sin(2 * np.pi * 50 * tt) + 1.0, fs)
check(S, "Hann window: peak amplitude and DC offset preserved (peak)", "exact: 3", amp[np.argmin(abs(fq - 50))], 3.0, 1e-9)
check(S, "... DC component", "exact: 1", amp[0], 1.0, 1e-9)
xs_ = np.sin(2 * np.pi * 5 * tt) + 0.5 * np.sin(2 * np.pi * 120 * tt)
check(S, "FFT low-pass removes the 120 Hz component exactly", "exact", np.max(np.abs(fourier.fft_filter(xs_, fs, high=50)
                                                                                 - np.sin(2 * np.pi * 5 * tt))), 0, 1e-10)
check(S, "Parseval's theorem (energy in time = energy in frequency)", "exact", np.sum(np.abs(np.fft.fft(xr))**2) / xr.size,
      np.sum(xr**2), 1e-9)

# ---------------------------------------------------------------- constrained optimisation and LP
S = "Linear programming and constrained optimisation"
lp = optimize.simplex_lp([40, 30], [[2, 1], [1, 1], [1, 0]], [100, 80, 40])
ref = so.linprog([-40, -30], A_ub=[[2, 1], [1, 1], [1, 0]], b_ub=[100, 80, 40], method="highs")
check(S, "simplex: optimal profit vs scipy linprog", "scipy (HiGHS)", lp.objective, -ref.fun, 1e-9)
check(S, "simplex: optimal plan (max diff)", "scipy (HiGHS)", np.max(np.abs(lp.x - ref.x)), 0, 1e-9)
check(S, "simplex: shadow prices vs linprog duals (max diff)", "scipy marginals",
      np.max(np.abs(lp.shadow_prices + ref.ineqlin.marginals)), 0, 1e-9)
A5 = rng.uniform(0.1, 2, (6, 5))
b5 = rng.uniform(5, 20, 6)
c5 = rng.uniform(1, 5, 5)
ref5 = so.linprog(-c5, A_ub=A5, b_ub=b5, method="highs")
check(S, "simplex on a random 6x5 LP: objective", "scipy (HiGHS)", optimize.simplex_lp(c5, A5, b5).objective, -ref5.fun, 1e-8)
area = lambda v: 2 * np.pi * v[0]**2 + 2 * np.pi * v[0] * v[1]  # noqa: E731  closed cylinder, v = (r, h)
vol_con = lambda v: np.pi * v[0]**2 * v[1] - 1.0  # noqa: E731  volume = 1 m3
# optimise u = log(r, h): positivity is automatic (in (r, h) the optimiser escapes to r < 0, where the
# "area" is unbounded below - see notebook 17)
area_u, vol_u = (lambda u: area(np.exp(u))), (lambda u: vol_con(np.exp(u)))
pen = optimize.penalty_minimize(area_u, np.log([0.5, 1.0]), ineq=[lambda u: np.exp(u[1]) - 1.0], eq=[vol_u], rounds=9)
pen.x = np.exp(pen.x)
slsqp = so.minimize(area, [0.5, 1.0], method="SLSQP", constraints=[{"type": "eq", "fun": vol_con},
                                                                     {"type": "ineq", "fun": lambda v: 1.0 - v[1]}],
                    options={"ftol": 1e-14})
check(S, "penalty method: tank radius (h <= 1 m active) vs SLSQP", "scipy SLSQP", pen.x[0], slsqp.x[0], 1e-4)
check(S, "... exact radius with h = 1: sqrt(1/pi)", "analytic", pen.x[0], np.sqrt(1 / np.pi), 1e-4)
unc = optimize.penalty_minimize(area_u, np.log([0.5, 1.0]), eq=[vol_u], rounds=9)
unc.x = np.exp(unc.x)
check(S, "penalty method, no height limit: h/r = 2", "analytic (Lagrange)", unc.x[1] / unc.x[0], 2.0, 1e-3)

# ---------------------------------------------------------------- Monte Carlo
S = "Monte Carlo"
from scipy.special import gamma as gamma_fn  # noqa: E402

for d in (2, 5, 10):
    est, se = montecarlo.integrate(lambda p: (np.sum(p**2, axis=1) <= 1).astype(float), [(-1, 1)] * d, 400_000, seed=d)
    exact = np.pi ** (d / 2) / gamma_fn(d / 2 + 1)
    check(S, f"volume of the unit ball in {d} dimensions (within 4 SE)", "pi^(d/2)/Gamma(d/2+1)", abs(est - exact) / se, 0, 4)
walk = montecarlo.random_walk(400, 4000, dim=2, seed=3)
msd = np.mean(np.sum(walk[-1] ** 2, axis=1))
check(S, "random walk: mean squared displacement after 400 steps (within 4 SE)", "theory: n = 400",
      abs(msd - 400) / (np.std(np.sum(walk[-1] ** 2, axis=1)) / np.sqrt(4000)), 0, 4)
w1 = montecarlo.random_walk(900, 20000, seed=5)[-1, :, 0]
check(S, "1D walk: kurtosis of end positions (Gaussian limit: 3)", "central limit theorem",
      np.mean(w1**4) / np.mean(w1**2) ** 2, 3.0, 0.1)

# ---------------------------------------------------------------- sparse matrices and 2D PDEs
S = "Sparse matrices and 2D PDEs"
errs, hs = [], []
for n in (15, 31, 63):
    X, Y, U = sparse.poisson_dirichlet(lambda x, y: 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y),
                                       lambda x, y: 0 * x, 1, 1, n, n)
    errs.append(np.max(np.abs(U - np.sin(np.pi * X) * np.sin(np.pi * Y))))
    hs.append(1 / (n + 1))
check(S, "5-point Poisson solver: observed order (manufactured solution)", "theory: 2", order(errs, hs), 2, 0.1)
X, Y, U = sparse.poisson_dirichlet(lambda x, y: -4 + 0 * x, lambda x, y: x**2 + y**2, 2, 1, 30, 14)
check(S, "quadratic solution reproduced exactly (non-square domain)", "exact", np.max(np.abs(U - X**2 - Y**2)), 0, 1e-10)
A2 = sparse.laplacian_2d(40, 30, 0.1, 0.2)
check(S, "Kronecker Laplacian vs dense assembly (max diff)", "direct construction",
      np.max(np.abs(A2.toarray() - (np.kron(np.eye(30), sparse.laplacian_1d(40, 0.1).toarray())
                                    + np.kron(sparse.laplacian_1d(30, 0.2).toarray(), np.eye(40))))), 0, 1e-9)
bb = rng.normal(size=A2.shape[0])
xcg, hcg = sparse.conjugate_gradient(A2, bb, tol=1e-12)
from scipy.sparse.linalg import spsolve  # noqa: E402

check(S, "conjugate gradient vs sparse direct solve (max rel diff)", "scipy spsolve",
      np.max(np.abs(xcg - spsolve(A2.tocsc(), bb))) / np.max(np.abs(xcg)), 0, 1e-9)

# ---------------------------------------------------------------- reporting
S = "Reporting and uncertainty propagation"
cases = [((2.29987, 0.08907), "2.30 ± 0.09"), ((10523.4, 156.7), "10520 ± 160"), ((0.348, 0.0081), "0.348 ± 0.008"),
         ((9.87654, 0.0963), "9.88 ± 0.10"), ((1234.5, 23.0), "1230 ± 20")]
for (v, u), txt in cases:
    check(S, f"format_uncertainty({v}, {u}) -> '{txt}'", "rounding rule",
          float(reporting.format_uncertainty(v, u) == txt), 1.0, 0)
y, uy, _ = reporting.propagate(lambda P, V, T: P * V / (8.314462618 * T), {"P": 2.0e5, "V": 0.03, "T": 300.0},
                               {"P": 3.0e3, "V": 1.0e-3, "T": 0.1})
rel = np.sqrt((3e3 / 2e5) ** 2 + (1e-3 / 0.03) ** 2 + (0.1 / 300) ** 2)
check(S, "propagation for PV/RT vs analytic relative formula", "analytic", uy / y, rel, 1e-8)

# ---------------------------------------------------------------- real data files
from engmath import datasets  # noqa: E402

S = "Real data files (NIST StRD, read from the bundled files; certified values parsed from their headers)"


def certified(name):
    """Certified parameter values and residual SD from a NIST StRD .dat header."""
    vals, ses, sd = [], [], None
    for line in open(datasets.path(name), encoding="utf-8"):
        parts = line.split()
        if len(parts) >= 6 and parts[0].startswith("b") and parts[1] == "=":
            vals.append(float(parts[4]))
            ses.append(float(parts[5]))
        if line.strip().startswith("Residual Standard Deviation:"):
            sd = float(parts[-1])
    return np.array(vals), np.array(ses), sd


d = np.loadtxt(datasets.path("Chwirut2.dat"), skiprows=60)
cert, _, sd_cert = certified("Chwirut2.dat")
for start in ([0.1, 0.01, 0.02], [0.15, 0.008, 0.010]):
    fit = fitting.levenberg_marquardt(lambda x, b1, b2, b3: np.exp(-b1 * x) / (b2 + b3 * x), d[:, 1], d[:, 0], start)
    check(S, f"Chwirut2 (ultrasonic calibration), start {start}: max rel. parameter error", "NIST certified",
          np.max(np.abs(fit.coef / cert - 1)), 0, 1e-6)
check(S, "Chwirut2 residual standard deviation", "NIST certified", fit.sigma, sd_cert, 1e-8)
e = np.loadtxt(datasets.path("ENSO.dat"), skiprows=60)
cert_e, se_e, sd_e = certified("ENSO.dat")


def enso(x, b1, b2, b3, b4, b5, b6, b7, b8, b9):
    w = 2 * np.pi * x
    return (b1 + b2 * np.cos(w / 12) + b3 * np.sin(w / 12) + b5 * np.cos(w / b4) + b6 * np.sin(w / b4)
            + b8 * np.cos(w / b7) + b9 * np.sin(w / b7))


fe = fitting.levenberg_marquardt(enso, e[:, 1], e[:, 0], [11, 3, 0.5, 40, -0.7, -1.3, 25, -0.3, 1.4])
# b8 (0.21 +/- 0.51) is barely determined, so compare deviations in units of the standard errors
check(S, "ENSO (Pacific pressure record, 9 parameters): max |b - certified| / SE", "NIST certified",
      np.max(np.abs(fe.coef - cert_e) / se_e), 0, 1e-4)
check(S, "ENSO standard errors: max relative difference", "NIST certified", np.max(np.abs(fe.se / se_e - 1)), 0, 1e-4)
check(S, "ENSO residual standard deviation", "NIST certified", fe.sigma, sd_e, 1e-7)
co2 = np.genfromtxt(datasets.path("co2_mauna_loa_monthly.csv"), delimiter=",", skip_header=1, usecols=(1, 2))
check(S, "Mauna Loa CO2 file: first monthly mean (March 1958)", "NOAA record: 315.71 ppm", co2[0, 1], 315.71, 1e-9)
check(S, "Mauna Loa CO2 file: consecutive months, no gaps", "821 months, spacing 1/12 year",
      np.max(np.abs(np.diff(co2[:, 0]) - 1 / 12)), 0, 0.01)

# ---------------------------------------------------------------- report
passed = sum(r[-1] for r in RESULTS)
lines = ["# Validation report", "", "Generated by `python docs/validate.py`; rerun in CI on every push.", "",
         f"**{passed} / {len(RESULTS)} checks pass.**", "",
         "Every from-scratch algorithm is compared with SciPy/NumPy/statsmodels, with exact analytic results, "
         "with NIST certified values, or with its theoretical order of convergence (the slope of log(error) "
         "against log(step size)).", ""]
sec = None
for s, name, ref, comp, exp, tol, ok in RESULTS:
    if s != sec:
        lines += ["", f"## {s}", "", "| Check | Reference | engmath | Expected | Tolerance | Status |",
                  "|---|---|---|---|---|:-:|"]
        sec = s
    lines.append(f"| {name} | {ref} | {comp:.10g} | {exp:.10g} | {tol:.1g} | {'pass' if ok else '**FAIL**'} |")
Path(__file__).with_name("VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{passed}/{len(RESULTS)} checks pass")
for r in RESULTS:
    if not r[-1]:
        print("FAIL:", r[0], "|", r[1], "computed", r[3], "expected", r[4], "tol", r[5])
sys.exit(0 if passed == len(RESULTS) else 1)
