"""Build the worked solutions to every exercise of the course, as executed notebooks.

The exercise texts are taken from ../notebooks/build_notebooks.py, so questions and solutions cannot
drift apart; the build fails if a notebook's number of solutions differs from its number of exercises.

    python solutions/build_solutions.py            # build and execute all solution notebooks
    python solutions/build_solutions.py 05 12      # only those whose names start with 05 or 12
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("course", HERE.parent / "notebooks" / "build_notebooks.py")
course = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(course)
md, code, SETUP = course.md, course.code, course.SETUP
SOLUTIONS: dict[str, dict] = {}


def solutions(name, setup=""):
    """Register the solutions of one course notebook: a function returning one list of cells per exercise."""
    def deco(fn):
        SOLUTIONS[name] = {"setup": setup, "answers": fn()}
        return fn
    return deco


def exercises(name) -> list[str]:
    cells = course.apply_extras(name, list(course.NOTEBOOKS[name]))
    text = [c.source for c in cells if c.cell_type == "markdown" and "## Exercises" in c.source][0]
    items = re.split(r"\n(?=\d+\. )", text.split("## Exercises")[1].strip())
    return [re.sub(r"^\d+\.\s*", "", it).strip() for it in items]


def notebook_cells(name) -> list:
    title = re.sub(r"^#\s*", "", course.NOTEBOOKS[name][0].source.splitlines()[0])
    entry = SOLUTIONS[name]
    ex = exercises(name)
    if len(ex) != len(entry["answers"]):
        raise ValueError(f"{name}: {len(ex)} exercises but {len(entry['answers'])} solutions")
    cells = [md(f"""
# Solutions · {title}

Worked solutions to the exercises of [notebook {name[:2]}](../notebooks/{name}.ipynb). Try each exercise
yourself before reading its solution - the learning happens in the attempt. Solutions are one way to
answer each question; other correct approaches exist.
"""), code(SETUP + ("\n" + entry["setup"] if entry["setup"] else ""))]
    for i, (text, answer) in enumerate(zip(ex, entry["answers"]), 1):
        cells.append(md(f"## Exercise {i}\n\n{text}"))
        cells.extend(answer)
    return cells


def write_all(only=()):
    import nbclient

    for name in SOLUTIONS:
        if only and not name.startswith(tuple(only)):
            continue
        nb = new_notebook(cells=notebook_cells(name),
                          metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                    "language_info": {"name": "python"}})
        nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                resources={"metadata": {"path": str(HERE)}}).execute()
        path = HERE / f"{name[:2]}_solutions.ipynb"
        nbformat.write(nb, path)
        print("wrote", path.name)


# =====================================================================================================
@solutions("00_python_numpy_primer", setup="R, u_max = 0.01, 0.5\nfrom engmath import integrate")
def _():
    return [
        [code(r"""Q_exact = 49/60 * np.pi * R**2 * u_max
errors = []
for n in (11, 101, 1001, 10001):
    r = np.linspace(0, R, n)
    Q = integrate.trapezoid(u_max * (1 - r/R)**(1/7) * 2*np.pi*r, r)
    errors.append(abs(Q - Q_exact) / Q_exact)
    print(f"n = {n:>5}: relative error {errors[-1]:.2e}")
print(f"error reduction per 10x more points: {errors[-2]/errors[-1]:.1f}x (a second-order method would give 100x)")"""),
         md(r"""
The error falls far more slowly than the 100x per decade seen for the laminar profile. The reason is at the
wall: $u \propto (1 - r/R)^{1/7}$ has an infinite slope at $r = R$, so near the wall the integrand is not
well approximated by straight lines, however fine the grid. The trapezoidal rule's $h^2$ error law
assumes a smooth integrand. (Clustering points near the wall, or a substitution that removes the
singular slope, restores fast convergence.)
""")],
        [code(r"""def reynolds(rho, u, D, mu):
    # np.asarray lets the same code accept a number, a list or an array
    return np.asarray(rho) * np.asarray(u) * np.asarray(D) / np.asarray(mu)

print("scalar:", reynolds(998.0, 1.0, 0.025, 1.0e-3))
print("array: ", reynolds(998.0, [0.05, 0.5, 2.0], 0.025, 1.0e-3))
u = np.linspace(0.01, 3, 300)
plt.plot(u, reynolds(998.0, u, 0.025, 1.0e-3))
plt.axhline(2300, color="C1", ls="--", label="Re = 2300 (end of laminar flow)")
plt.axhline(4000, color="C3", ls="--", label="Re = 4000 (fully turbulent)")
plt.xlabel("mean velocity (m/s)"); plt.ylabel("Reynolds number"); plt.yscale("log"); plt.legend(); plt.show()"""),
         md(r"""
NumPy's arithmetic works element by element, so the same formula handles scalars and arrays. In a 25 mm
water pipe the flow is laminar only below about 0.1 m/s; typical velocities of 1-2 m/s are well into
the turbulent range.
""")],
        [code(r"""import time

def midpoint_loop(f, a, b, n):
    h = (b - a) / n
    total = 0.0
    for i in range(n):
        total += f(a + (i + 0.5) * h)
    return total * h

def midpoint_vec(f, a, b, n):
    h = (b - a) / n
    return h * np.sum(f(a + (np.arange(n) + 0.5) * h))

integrand = lambda r: u_max * (1 - r**2/R**2) * 2*np.pi*r
for fn in (midpoint_loop, midpoint_vec):
    t0 = time.perf_counter(); fn(integrand, 0, R, 10**6); print(f"{fn.__name__}: {time.perf_counter() - t0:.3f} s")
Q_exact = np.pi * R**2 * u_max / 2
n = 20
err_mid = midpoint_vec(integrand, 0, R, n) - Q_exact
err_trap = integrate.trapezoid(integrand(np.linspace(0, R, n + 1)), np.linspace(0, R, n + 1)) - Q_exact
print(f"n = {n}: midpoint error {err_mid:+.3e}, trapezoid error {err_trap:+.3e}, ratio {err_mid/err_trap:.2f}")"""),
         md(r"""
The vectorised version is much faster, because the loop runs in NumPy's compiled code instead of the
Python interpreter. The midpoint rule's error is roughly **half** that of the trapezoidal rule and of
**opposite sign** (both are second order). That is why their weighted average, (2 × midpoint +
trapezoid)/3, cancels the leading error: it is exactly Simpson's rule.
""")],
    ]


# =====================================================================================================
@solutions("01_errors_and_uncertainty", setup=r"""from engmath import reporting
def efficiency(Q, H, P, rho=998.0, g=9.81):
    return rho * g * Q * H / P
values = {"Q": 0.0125, "H": 32.0, "P": 5200.0}
unc = {"Q": 0.0002, "H": 0.4, "P": 60.0}""")
def _():
    return [
        [code(r"""x = 1e-12
direct, stable = np.log(1 + x), np.log1p(x)
print(f"log(1 + x) = {direct:.16e}\nlog1p(x)   = {stable:.16e}\nrelative error of the direct form: {abs(direct/stable - 1):.1e}")
print(f"spacing of doubles near 1: {np.spacing(1.0):.2e}")"""),
         md(r"""
$1 + 10^{-12}$ must be rounded to the nearest double, and near 1 the doubles are $2.2\times10^{-16}$ apart. The
rounding changes the *small part* $x$ by up to about $10^{-4}$ of itself, and the logarithm faithfully returns
that wrong value. `np.log1p` computes $\ln(1 + x)$ without ever forming $1 + x$. The same problem - and the
same kind of cure (`np.expm1`, `np.hypot`) - appears whenever a small quantity is added to a large one.
""")],
        [code(r"""dT, u_dT = 24.0 - 20.0, np.hypot(0.2, 0.2)
print(f"temperature rise {dT:.1f} ± {u_dT:.2f} K  ->  relative uncertainty {u_dT/dT:.1%}")
for u_diff in (0.05, 0.02):
    print(f"with a differential sensor measuring the rise directly (±{u_diff} K): {u_diff/dT:.2%}")
print(f"or double the temperature rise (halve the flow): {np.hypot(0.2, 0.2)/8:.1%}")"""),
         md(r"""
The duty depends on the **difference** of two similar temperatures. Their absolute uncertainties add (in
quadrature), while the difference itself is small, so the relative uncertainty is large - the same
cancellation as in the quadratic-formula example. Remedies: measure the difference directly (a
differential thermocouple or thermopile, whose calibration errors largely cancel), or design the test
for a larger temperature rise.
""")],
        [code(r"""from scipy import stats
rng = np.random.default_rng(3)
N = 1_000_000
eta = efficiency(rng.normal(values["Q"], unc["Q"], N), rng.normal(values["H"], unc["H"], N),
                 rng.normal(values["P"], 0.20 * values["P"], N))
lin_value, lin_u, _ = reporting.propagate(efficiency, values, {**unc, "P": 0.20 * values["P"]})
lo, med, hi = np.percentile(eta, [2.5, 50, 97.5])
print(f"linear propagation: {lin_value:.3f} ± {lin_u:.3f}")
print(f"Monte Carlo: mean {eta.mean():.3f}, SD {eta.std():.3f}, skewness {stats.skew(eta):.2f}")
print(f"95 % interval [{lo:.3f}, {hi:.3f}]: {med - lo:.3f} below the median, {hi - med:.3f} above")
plt.hist(eta, bins=200, density=True); plt.axvline(lin_value, color="k"); plt.xlabel("efficiency"); plt.show()"""),
         md(r"""
The distribution is clearly **skewed to the right**, and its mean lies above the nominal value. The
efficiency depends on $1/P$, a curved function: low power readings raise $\eta$ more than high readings
lower it. With 20 % input uncertainty the linear formula - which assumes a straight-line dependence -
no longer describes the shape, and even the standard deviation differs. Report the Monte Carlo interval.
""")],
        [code(r"""def propagate_correlated(func, values, unc, R):
    names = list(values)
    cu = np.empty(len(names))
    for i, name in enumerate(names):
        h = 1e-6 * abs(values[name])
        up, down = {**values, name: values[name] + h}, {**values, name: values[name] - h}
        cu[i] = (func(**up) - func(**down)) / (2*h) * unc[name]
    return func(**values), float(np.sqrt(cu @ np.asarray(R) @ cu))

for rho_QH in (0.0, 0.5, -0.5):
    R = np.eye(3); R[0, 1] = R[1, 0] = rho_QH
    eta0, u = propagate_correlated(efficiency, values, unc, R)
    print(f"correlation(Q, H) = {rho_QH:+.1f}: u(eta) = {u:.4f}")"""),
         md(r"""
$\eta$ increases with both $Q$ and $H$, so their sensitivity coefficients have the same sign. A positive
correlation (e.g. both read by instruments sharing a drifting reference) makes their errors reinforce
each other and increases $u(\eta)$; a negative correlation partly cancels them. Ignoring correlation can
therefore understate *or* overstate an uncertainty.
""")],
    ]


# =====================================================================================================
@solutions("02_root_finding", setup=r"""from engmath import roots
eps, D, Re = 0.045e-3, 0.05, 2e5
def colebrook(f):
    return 1/np.sqrt(f) + 2*np.log10(eps/D/3.7 + 2.51/(Re*np.sqrt(f)))""")
def _():
    return [
        [code(r"""a, b, R = 0.364, 4.27e-5, 8.314
def vdw_roots(P, T):
    # (P + a/V^2)(V - b) = RT  <=>  P V^3 - (P b + R T) V^2 + a V - a b = 0
    r = np.roots([P, -(P*b + R*T), a, -a*b])
    return np.sort(r[np.abs(r.imag) < 1e-12].real)
for T in (300.0, 280.0):
    Vs = vdw_roots(50e5, T)
    print(f"T = {T:.0f} K, 50 bar: {len(Vs)} real root(s), V_m = {np.round(Vs*1e6, 1)} cm³/mol "
          f"(ideal gas {R*T/50e5*1e6:.0f} cm³/mol)")
V = np.linspace(6e-5, 8e-4, 500)
for T in (300.0, 280.0):
    plt.plot(V*1e6, (R*T/(V - b) - a/V**2)/1e5, label=f"{T:.0f} K")
plt.axhline(50, color="k", ls="--", lw=1); plt.ylim(0, 120)
plt.xlabel("molar volume (cm³/mol)"); plt.ylabel("pressure (bar)"); plt.legend(); plt.show()"""),
         md(r"""
At 300 K the isotherm crosses 50 bar once: one real root, a gas-like molar volume about a quarter below
the ideal-gas value. At 280 K - below the van der Waals critical temperature (about 304 K) - the isotherm has
a loop, and 50 bar cuts it three times: the smallest root is liquid-like, the largest gas-like, and the
middle one is physically unstable. Plotting first shows how many roots to look for, and where to
bracket each one.
""")],
        [code(r"""def g(f):                       # fixed-point form: f = g(f)
    return (-2*np.log10(eps/D/3.7 + 2.51/(Re*np.sqrt(f))))**-2
f_star = roots.brent(colebrook, 0.005, 0.08, tol=1e-15).root
f, errs = 0.02, []
for k in range(30):
    f = g(f); errs.append(abs(f - f_star))
    if errs[-1] < 1e-15:
        break
ratios = np.array(errs[1:6]) / np.array(errs[:5])
print(f"fixed point: {len(errs)} iterations; error ratios {np.round(ratios, 4)}")
print(f"Newton from the same start: {roots.newton(colebrook, 0.02, tol=1e-15).iterations} iterations")"""),
         md(r"""
The error shrinks by a roughly **constant factor** each iteration: linear convergence (order 1). Here the
factor is very small, because $g$ depends only weakly on $f$ - so fixed-point iteration is a perfectly
good method for Colebrook, and simple to program. Newton converges quadratically but needs derivatives;
for this particular equation the difference in iteration count is modest.
""")],
        [code(r"""def haaland(Re, rel):
    return (-1.8*np.log10((rel/3.7)**1.11 + 6.9/Re))**-2
Re_range = np.logspace(3.7, 8, 100)
fig, ax = plt.subplots()
worst = 0
for rel in [1e-5, 1e-4, 1e-3, 1e-2, 5e-2]:
    fc = np.array([roots.brent(lambda f: 1/np.sqrt(f) + 2*np.log10(rel/3.7 + 2.51/(R_*np.sqrt(f))), 1e-4, 1).root
                   for R_ in Re_range])
    err = 100 * (haaland(Re_range, rel) / fc - 1)
    worst = max(worst, np.max(np.abs(err)))
    ax.semilogx(Re_range, err, label=f"ε/D = {rel:g}")
ax.set(xlabel="Reynolds number", ylabel="Haaland error vs Colebrook (%)"); ax.legend(fontsize=8); plt.show()
print(f"largest deviation: {worst:.2f} %")"""),
         md(r"""
Haaland's explicit formula stays within about 1.5 % of Colebrook over the whole turbulent range - smaller
than the uncertainty of Colebrook itself (roughly 5-15 % against measurements). For hand calculations
the explicit formula is fine; in software, solving Colebrook costs almost nothing.
""")],
        [code(r"""def regula_falsi(g, a, b, tol=1e-12, maxiter=500):
    ga, gb = g(a), g(b)
    moved = {"a": 0, "b": 0}
    for k in range(1, maxiter + 1):
        c = b - gb * (b - a) / (gb - ga)       # where the chord crosses zero
        gc = g(c)
        if abs(gc) < tol or min(abs(c - a), abs(b - c)) < tol:
            return c, k, moved
        if ga * gc < 0:
            b, gb = c, gc; moved["b"] += 1
        else:
            a, ga = c, gc; moved["a"] += 1
    return c, maxiter, moved

root, its, moved = regula_falsi(colebrook, 0.005, 0.08)
print(f"regula falsi: f = {root:.12f} after {its} iterations; endpoint updates {moved}")
print(f"bisection:    {roots.bisection(colebrook, 0.005, 0.08).iterations} iterations")"""),
         md(r"""
Look at which endpoint moved: only one. $g$ is convex on this bracket, so every chord crosses zero on the
same side of the root and the other endpoint is never replaced. The bracket then shrinks from one side
only and convergence becomes linear - here slower than halving the bracket. Modified versions (the
Illinois method, which halves the stuck endpoint's function value) and Brent's method fix exactly this.
""")],
    ]


# =====================================================================================================
@solutions("03_linear_systems", setup=r"""from engmath import linalg
L, k, q, T_left, T_right = 0.1, 15.0, 4e5, 20.0, 60.0""")
def _():
    return [
        [code(r"""h, T_inf, n = 25.0, 20.0, 50                    # n intervals; unknowns T_1 ... T_n (T_0 fixed)
x = np.linspace(0, L, n + 1); dx = x[1] - x[0]
lower, diag, upper = np.ones(n), -2*np.ones(n), np.ones(n)
rhs = -q*dx**2/k * np.ones(n)
rhs[0] -= T_left
# last node: ghost node T_{n+1} = T_{n-1} - 2 dx h/k (T_n - T_inf) from -k T' = h (T - T_inf)
lower[-1] = 2.0
diag[-1] = -2 - 2*dx*h/k
rhs[-1] = -q*dx**2/k - 2*dx*h/k*T_inf
T = np.r_[T_left, linalg.thomas(lower, diag, upper, rhs)]
C1 = q*L*(1 + h*L/(2*k)) / (k + h*L)                # exact: T = T_left + C1 x - q x^2/(2k)
T_exact = T_left + C1*x - q*x**2/(2*k)
print(f"max error {np.max(np.abs(T - T_exact)):.1e} K; surface T {T[-1]:.1f} °C; maximum {T.max():.1f} °C at x = {x[T.argmax()]*100:.1f} cm")"""),
         md(r"""
Only the **last row** changes: its coefficients and right-hand side now contain the convection
coefficient. The ghost-node treatment keeps second-order accuracy (here the profile is quadratic, so the
result is exact to rounding). With the weak convection ($h$ = 25 W/(m²·K)) the right face runs far
hotter than the 60 °C it was held at before - convection removes the generated heat much less
effectively than a fixed-temperature wall.
""")],
        [code(r"""N = 31
def solve_plate(method, tol=1e-7):
    T = np.zeros((N, N)); T[0, :] = 100.0
    for sweep in range(1, 20000):
        old = T.copy()
        if method == "jacobi":             # every node from the OLD values (vectorised)
            T[1:-1, 1:-1] = 0.25*(old[2:, 1:-1] + old[:-2, 1:-1] + old[1:-1, 2:] + old[1:-1, :-2])
        else:                              # Gauss-Seidel: newest values as soon as available
            for i in range(1, N-1):
                for j in range(1, N-1):
                    T[i, j] = 0.25*(T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1])
        if np.max(np.abs(T - old)) < tol:
            return sweep, T[N//2, N//2]
for m in ("jacobi", "gauss-seidel"):
    sweeps, centre = solve_plate(m)
    print(f"{m:<13}: {sweeps:5d} sweeps, centre {centre:.4f} °C")"""),
         md(r"""
Jacobi needs about **twice** as many sweeps as Gauss-Seidel - the classical result for this problem,
because Gauss-Seidel uses each updated value immediately. Jacobi has one practical advantage: every
node can be updated independently, so it vectorises (as here) and parallelises perfectly.
""")],
        [code(r"""import scipy.linalg as sla
m_, ks, nm = 2.0, 1.5e4, 5
K = ks*(2*np.eye(nm) - np.eye(nm, k=1) - np.eye(nm, k=-1))
for label, masses in (("equal masses", [m_]*5), ("middle mass doubled", [m_, m_, 2*m_, m_, m_])):
    lam, modes = sla.eigh(K, np.diag(masses))       # K v = omega^2 M v
    print(f"{label:<20}: {np.round(np.sqrt(lam)/(2*np.pi), 2)} Hz")"""),
         md(r"""
The heavier middle mass lowers the 1st, 3rd and 5th frequencies but leaves the **2nd and 4th unchanged**.
Those modes are antisymmetric: the middle mass sits at a node and does not move, so its mass cannot
matter. A general rule: added mass lowers a natural frequency only in proportion to how much the mass
moves in that mode - which is why engineers look at mode shapes, not just frequencies.
""")],
        [code(r"""Ns = 21
def sor_sweeps(omega, tol=1e-6):
    T = np.zeros((Ns, Ns)); T[0, :] = 100.0
    for sweep in range(1, 5000):
        change = 0.0
        for i in range(1, Ns-1):
            for j in range(1, Ns-1):
                gs = 0.25*(T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1])
                new = T[i, j] + omega*(gs - T[i, j])
                change = max(change, abs(new - T[i, j])); T[i, j] = new
        if change < tol:
            return sweep
omegas = np.arange(1.0, 1.96, 0.05)
counts = [sor_sweeps(w) for w in omegas]
best = omegas[int(np.argmin(counts))]
theory = 2 / (1 + np.sin(np.pi / (Ns - 1)))
plt.plot(omegas, counts, "o-"); plt.axvline(theory, color="k", ls="--", label=f"theory {theory:.3f}")
plt.xlabel("relaxation factor ω"); plt.ylabel("sweeps to converge"); plt.legend(); plt.show()
print(f"Gauss-Seidel (ω = 1): {counts[0]} sweeps; best ω = {best:.2f}: {min(counts)} sweeps")"""),
         md(r"""
Over-relaxation cuts the number of sweeps several-fold. The best factor agrees with the theoretical
optimum $\omega = 2/(1 + \sin(\pi/(N-1)))$ for this model problem. Too large an $\omega$ makes the iteration
oscillate, and at $\omega \ge 2$ it diverges.
""")],
    ]


# =====================================================================================================
@solutions("04_interpolation", setup=r"""from engmath import interpolate, roots
def p_antoine(T):          # kPa
    return 10**(8.07131 - 1730.63/(233.426 + T)) * 0.133322""")
def _():
    return [
        [code(r"""T_fine = np.linspace(20, 100, 800)
p_fine = p_antoine(T_fine)
interior = (T_fine > 40) & (T_fine < 80)
print(f"{'spacing':>8}{'linear':>10}{'spline (all)':>14}{'spline (40-80 °C)':>19}")
res = []
for dT in (20, 10, 5):
    Tt = np.arange(20.0, 100.1, dT)
    e_lin = np.max(np.abs(interpolate.linear(Tt, p_antoine(Tt), T_fine)/p_fine - 1))
    s = interpolate.CubicSpline(Tt, p_antoine(Tt))(T_fine)/p_fine - 1
    res.append((e_lin, np.max(np.abs(s)), np.max(np.abs(s[interior]))))
    print(f"{dT:>6} °C{res[-1][0]:>10.2e}{res[-1][1]:>14.2e}{res[-1][2]:>19.2e}")
for i in range(2):
    print(f"halving the spacing reduces the errors by: linear {res[i][0]/res[i+1][0]:.1f}x, "
          f"spline {res[i][1]/res[i+1][1]:.1f}x, spline interior {res[i][2]/res[i+1][2]:.1f}x")"""),
         md(r"""
Linear interpolation behaves as theory predicts: halving the spacing cuts the error about 4x ($h^2$). The
natural spline falls short of the promised 16x ($h^4$) over the whole table, because its **end condition**
(zero curvature at both ends) is wrong for this curve, which is strongly curved at 100 °C. That error is
largest near the ends and converges only like $h^2$; away from the ends the spline does much better.
Exercise 4 removes the problem with a better end condition.
""")],
        [code(r"""Tt = np.arange(10.0, 101.0, 10.0)
s = interpolate.CubicSpline(Tt, np.log(p_antoine(Tt)))
T_boil = roots.brent(lambda T: s(T) - np.log(50.0), 10, 100).root
T_exact = 1730.63/(8.07131 - np.log10(50.0/0.133322)) - 233.426
print(f"spline of ln p: {T_boil:.4f} °C;  Antoine equation inverted: {T_exact:.4f} °C")"""),
         md(r"""
The boiling point at 50 kPa follows from a root search on the interpolant: about 81.4 °C, agreeing with
the Antoine equation (inverted exactly) to a few thousandths of a degree. Interpolating $\ln p$ makes the
curve nearly linear, which is why so few table points suffice.
""")],
        [code(r"""f = lambda x: 1/(1 + 25*x**2)
xq = np.linspace(-1, 1, 1000)
for n in (10, 20, 40):
    x_eq = np.linspace(-1, 1, n + 1)
    x_ch = np.cos(np.arange(n + 1) * np.pi / n)
    e_eq = np.max(np.abs(interpolate.lagrange(x_eq, f(x_eq), xq) - f(xq)))
    e_ch = np.max(np.abs(interpolate.lagrange(x_ch, f(x_ch), xq) - f(xq)))
    print(f"degree {n:>2}: equally spaced error {e_eq:9.2e}   Chebyshev points error {e_ch:.2e}")"""),
         md(r"""
With Chebyshev points - clustered towards the ends of the interval - the polynomial error **decreases**
steadily with the degree, while with equally spaced points it explodes. Runge's phenomenon is caused by
the choice of points, not by polynomials themselves; Chebyshev-based methods are the basis of highly
accurate numerical libraries.
""")],
        [code(r"""def spline_clamped(x, y, xq, d0, dn):
    h = np.diff(x); n = len(x)
    A, rhs = np.zeros((n, n)), np.zeros(n)
    A[0, 0], A[0, 1] = 2*h[0], h[0]                       # clamped start: slope d0
    rhs[0] = 6*((y[1] - y[0])/h[0] - d0)
    A[-1, -2], A[-1, -1] = h[-1], 2*h[-1]                 # clamped end: slope dn
    rhs[-1] = 6*(dn - (y[-1] - y[-2])/h[-1])
    for i in range(1, n - 1):
        A[i, i-1], A[i, i], A[i, i+1] = h[i-1], 2*(h[i-1] + h[i]), h[i]
        rhs[i] = 6*((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])
    M = np.linalg.solve(A, rhs)
    i = np.clip(np.searchsorted(x, xq) - 1, 0, n - 2)
    a, b, hi = x[i+1] - xq, xq - x[i], h[i]
    return (M[i]*a**3 + M[i+1]*b**3)/(6*hi) + (y[i]/hi - M[i]*hi/6)*a + (y[i+1]/hi - M[i+1]*hi/6)*b

dpdT = lambda T: p_antoine(T) * np.log(10) * 1730.63 / (233.426 + T)**2
Tt = np.arange(10.0, 101.0, 10.0); pt = p_antoine(Tt)
T_fine = np.linspace(10, 100, 900); p_fine = p_antoine(T_fine)
e_nat = np.abs(interpolate.CubicSpline(Tt, pt)(T_fine)/p_fine - 1)
e_cla = np.abs(spline_clamped(Tt, pt, T_fine, dpdT(Tt[0]), dpdT(Tt[-1]))/p_fine - 1)
ends = (T_fine < 20) | (T_fine > 90)
print(f"max error near the ends: natural {e_nat[ends].max():.2e}, clamped {e_cla[ends].max():.2e}")
print(f"max error in the middle: natural {e_nat[~ends].max():.2e}, clamped {e_cla[~ends].max():.2e}")"""),
         md(r"""
With the correct end slopes the error near the ends of the table drops dramatically, and the whole
interpolant improves. The first and last rows are the only change: they replace "zero curvature" by the
slope condition obtained by differentiating the end cubics.
""")],
    ]


# =====================================================================================================
@solutions("05_integration", setup=r"""from engmath import integrate
from scipy.integrate import quad
def power(t):
    return 40 + 25*np.sin(np.pi*t/8)**2 + 30*np.exp(-(t-0.5)**2/0.08) - 20*np.exp(-(t-4)**2/0.1)
E_exact = quad(power, 0, 8, epsabs=1e-13, limit=200)[0]""")
def _():
    return [
        [code(r"""def error_for(minutes, offset):
    # logging every `minutes`, first sample `offset` minutes after the start (plus the start and end)
    t = np.unique(np.r_[0.0, np.arange(offset/60, 8, minutes/60), 8.0])
    return integrate.trapezoid(power(t), t) / E_exact - 1

for minutes in (60, 30, 20, 15, 10, 5):
    errs = [error_for(minutes, off) for off in np.arange(0, minutes, 1.0)]
    worst = max(errs, key=abs)
    print(f"every {minutes:>2} min: error {errs[0]*100:+.3f} % on the hour grid, worst case over all "
          f"schedule offsets {worst*100:+.3f} %{'   <- always within 0.5 %' if abs(worst) < 0.005 else ''}")"""),
         md(r"""
On the regular grid starting at the full hour, 30-minute logging happens to pass - but only because one
sample falls exactly on the peak of the start-up surge at $t$ = 0.5 h. Shift the schedule and the error
exceeds 0.5 %. The honest answer looks at the **worst case over all schedule offsets**: logging every
20 minutes or faster is needed. Choose a logging interval from the fastest event you need to capture,
not from the integration formula.
""")],
        [code(r"""t_eq = np.linspace(0, 3, 11)
rng = np.random.default_rng(0)
t_uneven = np.sort(np.r_[0, rng.uniform(0, 3, 9), 3])
exact = 3**4 / 4
print(f"equal spacing:   Simpson error for t^3 = {integrate.simpson(t_eq**3, t_eq) - exact:+.2e}")
print(f"unequal spacing: Simpson error for t^3 = {integrate.simpson(t_uneven**3, t_uneven) - exact:+.2e}")
print(f"unequal spacing: Simpson error for t^2 = {integrate.simpson(t_uneven**2, t_uneven) - 3**3/3:+.2e}")"""),
         md(r"""
On equal spacing Simpson's rule integrates cubics exactly: the parabola's error for the cubic term is an
odd function about the middle point and cancels. With unequal spacing the symmetry is lost and a small
error remains, while quadratics are still exact (the parabola *is* the function).
""")],
        [code(r"""t_log = np.arange(0, 8.01, 0.5)
P_log = power(t_log)
rng = np.random.default_rng(5)
E_noisy = np.array([integrate.trapezoid(P_log + rng.normal(0, 1.0, P_log.size), t_log) for _ in range(20000)])
w = np.full(t_log.size, 0.5); w[[0, -1]] = 0.25                    # trapezoid weights (h = 0.5 h)
print(f"noise-induced uncertainty: Monte Carlo {E_noisy.std():.3f} kWh, formula sigma*sqrt(sum w²) = {np.sqrt(np.sum(w**2)):.3f} kWh")
print(f"sampling (discretisation) error of the same data: {integrate.trapezoid(P_log, t_log) - E_exact:+.3f} kWh")"""),
         md(r"""
The random noise contributes an uncertainty of about 2 kWh - **larger** than the error from sampling only
every 30 minutes. Because the energy is a weighted sum of readings, the Monte Carlo result can be
checked exactly: $\sigma_E = \sigma\sqrt{\sum w_i^2}$. Averaging over many readings reduces the effect of
noise, but it never disappears; a faster logger would reduce both errors.
""")],
        [code(r"""def boole(f, a, b, n):
    if n % 4:
        raise ValueError("n must be a multiple of 4")
    x = np.linspace(a, b, n + 1); y = f(x); h = (b - a) / n
    w = np.tile([14, 32, 12, 32], n // 4).astype(float); w = np.r_[w, 7.0]; w[0] = 7.0
    return 2*h/45 * np.sum(w * y)

ns = np.array([16, 32, 64, 128, 256, 512, 1024])
errs = np.array([abs(boole(power, 0, 8, n) - E_exact) for n in ns])
print("error reduction per halving of h:", np.round(errs[:-1] / errs[1:], 1), " (order 6 means 2^6 = 64)")
print(f"order fitted on the finest grids: {np.polyfit(np.log(8/ns[-3:]), np.log(errs[-3:]), 1)[0]:.2f}")
print(f"30-minute data (n = 16): Boole error {boole(power, 0, 8, 16) - E_exact:+.2f} kWh, "
      f"trapezoid {integrate.composite(power, 0, 8, 16, 'trapezoid') - E_exact:+.2f} kWh")"""),
         md(r"""
Where two groups of four intervals meet, their end weights (7 + 7) add to 14 - hence the repeating
pattern 14, 32, 12, 32. On fine grids the error falls by the factor 64 per halving that order 6 predicts.
On coarse grids it behaves irregularly - improving by factors of about 11 and 47, then by almost 2000 at
once as the grid starts to resolve the surge: the *pre-asymptotic* regime, where order-of-accuracy
statements do not yet apply. On the 30-minute data Boole's rule is far **worse** than the trapezoid: it fits
high-degree polynomials through the samples, which overshoot around an event narrower than the sample
spacing. More accuracy requires more data, not a cleverer formula.
""")],
    ]


# =====================================================================================================
@solutions("06_differentiation", setup=r"""from engmath import differentiate, fitting, smoothing
T_inf, dT0, C, n_true = 20.0, 480.0, 5.0e-4, 1.25
t = np.r_[np.arange(0, 60, 5), np.arange(60, 300, 20), np.arange(300, 1801, 100)].astype(float)
dT = (dT0**-0.25 + C*t/4)**-4
T = T_inf + dT
T_noisy = T + np.random.default_rng(3).normal(0, 0.5, T.size)

def fit_n(temps, derivs):
    ok = (derivs < 0) & (temps > T_inf)
    f = fitting.linear_lstsq(np.column_stack([np.ones(ok.sum()), np.log(temps[ok] - T_inf)]), np.log(-derivs[ok]))
    return f.coef[1], f.conf_int()[1]""")
def _():
    return [
        [code(r"""for label, temps in (("noise-free", T), ("noisy", T_noisy)):
    for tmax, part in ((1800, "all data"), (600, "first 10 min")):
        m = t <= tmax
        n_est, ci = fit_n(temps[m], differentiate.gradient(t[m], temps[m]))
        print(f"{label:<10} {part:<12} ({m.sum():2d} points): n = {n_est:.3f}, 95 % CI [{ci[0]:.3f}, {ci[1]:.3f}]")"""),
         md(r"""
For the noisy data the confidence interval **widens** when only the first ten minutes are used: fewer
points, and - more importantly - a narrower range of temperature differences, so the slope of
$\ln(-dT/dt)$ against $\ln(T - T_\infty)$ is less well determined. A wide range of the independent variable
is what makes a slope precise. (For noise-free data both intervals are tiny: the only error left is the
small truncation error of the differences.)
""")],
        [code(r"""fwd = np.r_[np.diff(T) / np.diff(t), np.nan]            # forward differences (last point has none)
n_fwd, _ = fit_n(T[:-1], fwd[:-1])
n_cen, _ = fit_n(T, differentiate.gradient(t, T))
print(f"forward differences: n = {n_fwd:.4f};  second-order formulas: n = {n_cen:.4f};  true 1.25")"""),
         md(r"""
The forward difference estimates the slope at the *middle* of each interval but assigns it to the start,
where the true slope is steeper. The error is largest where the curve bends most - early on - and it biases
the fitted exponent. The second-order formulas remove almost all of this bias.
""")],
        [code(r"""for frac in (0.15, 0.25, 0.4):
    T_smooth = smoothing.lowess(t, T_noisy, frac=frac, degree=2)
    n_s, ci = fit_n(T_smooth, differentiate.gradient(t, T_smooth))
    print(f"LOWESS (frac {frac}): n = {n_s:.3f}, 95 % CI [{ci[0]:.3f}, {ci[1]:.3f}]")
n_raw, ci = fit_n(T_noisy, differentiate.gradient(t, T_noisy))
print(f"no smoothing:       n = {n_raw:.3f}, 95 % CI [{ci[0]:.3f}, {ci[1]:.3f}]")"""),
         md(r"""
Moderate smoothing (frac 0.25) narrows the confidence interval and keeps the estimate close to 1.25. Too
much smoothing (frac 0.4) distorts the curve: the estimate moves well away from 1.25 and its interval
**excludes the true value**. The interval is computed as if the smoothed points were independent
measurements, which they are not, and it ignores the bias that smoothing introduces - so it can be badly
misleading. Smoothing trades noise for bias. Fitting the integrated model directly
(notebook 09, exercise 3) avoids differentiation altogether.
""")],
        [code(r"""import sympy as sp
s, h1, h2 = sp.symbols("s h1 h2", positive=True)
f0, f1, f2 = sp.symbols("f0 f1 f2")
parabola = sp.interpolate([(0, f0), (h1, f1), (h1 + h2, f2)], s)
start = sp.expand(sp.diff(parabola, s).subs(s, 0))
for fi in (f0, f1, f2):
    print(fi, ":", sp.factor(start.coeff(fi)))
formula = sp.lambdify((f0, f1, f2, h1, h2), start)
print("formula:", formula(T[0], T[1], T[2], t[1] - t[0], t[2] - t[1]), " library:", differentiate.gradient(t, T)[0])"""),
         md(r"""
Differentiating the same parabola at its first point gives
$f'(t_0) \approx -\frac{2h_1 + h_2}{h_1(h_1 + h_2)}f_0 + \frac{h_1 + h_2}{h_1h_2}f_1 - \frac{h_1}{h_2(h_1 + h_2)}f_2$, which is exactly what
`differentiate.gradient` uses at the first point. It is second order too, but with a larger error constant
than the central formula, because it extrapolates rather than interpolates.
""")],
    ]


# =====================================================================================================
@solutions("07_smoothing", setup=r"""from engmath import smoothing
x = np.linspace(0, 20, 401); dx = x[1] - x[0]
def peak(x, h, c, w): return h*np.exp(-0.5*((x - c)/w)**2)
true = peak(x, 1.0, 8.0, 0.8) + peak(x, 0.6, 10.5, 1.0) + 0.05
y = true + np.random.default_rng(7).normal(0, 0.05, x.size)""")
def _():
    return [
        [code(r"""y_spike = y.copy(); y_spike[300] += 1.5                     # glitch at x = 15
plain = smoothing.lowess(x, y_spike, 0.04)
robust = smoothing.lowess(x, y_spike, 0.04, robust_iters=2)
near = np.abs(x - 15) < 1.0
print(f"max error near the spike: plain LOWESS {np.max(np.abs(plain - true)[near]):.3f}, "
      f"robust LOWESS {np.max(np.abs(robust - true)[near]):.3f}")
plt.plot(x, y_spike, ".", ms=2, color="gray"); plt.plot(x, plain, label="LOWESS")
plt.plot(x, robust, label="robust LOWESS (2 iterations)"); plt.xlim(12, 18); plt.ylim(-0.1, 0.6); plt.legend(); plt.show()"""),
         md(r"""
Plain LOWESS spreads the glitch into a bump several points wide. The robust iterations give the spike a
weight near zero after the first pass, so the smoothed curve ignores it - robust methods flag outliers
instead of averaging them in.
""")],
        [code(r"""fig, ax = plt.subplots()
for m in (5, 10, 20):
    d2 = smoothing.savitzky_golay(y, m, 3, deriv=2, dx=dx)
    ax.plot(x, d2, label=f"window {2*m + 1} points")
ax.plot(x, np.gradient(np.gradient(true, dx), dx), "k--", label="true second derivative")
ax.set(xlim=(5, 14), ylim=(-3, 2), xlabel="x", ylabel="second derivative"); ax.legend(fontsize=8); plt.show()"""),
         md(r"""
A peak shows up as a **minimum** of the second derivative; the shoulder of the smaller peak becomes a
separate minimum near $x$ = 10.5 even though the signal itself has no separate maximum there. With an
11-point window the noise swamps everything; with 41 points both minima survive but become shallower
and broader (the main one loses about a third of its depth) - larger windows still would merge them.
Each differentiation
multiplies the noise by roughly $1/h$ - the second derivative by $1/h^2$ - so it needs more smoothing,
and that smoothing distorts the very features we look for.
""")],
        [code(r"""xs = np.arange(30.0)
quadratic = 0.3*xs**2 - 2*xs + 7
print("max |SG(quadratic) - quadratic| =", np.max(np.abs(smoothing.savitzky_golay(quadratic, 2, 2) - quadratic)))"""),
         md(r"""
The filter fits a quadratic by least squares in each 5-point window; if the data *are* a quadratic, the
fit is exact and returns the data unchanged. Smooth physical signals look locally like low-order
polynomials, so the filter passes them almost untouched while averaging away noise - unlike a moving
average, which fits a constant and flattens every peak.
""")],
        [code(r"""def exponential_smoothing(y, a):
    s = np.empty_like(y); s[0] = y[0]
    for i in range(1, y.size):
        s[i] = a*y[i] + (1 - a)*s[i-1]
    return s

x_peak = x[np.argmax(true)]
print(f"true peak at x = {x_peak:.2f}")
for a in (0.5, 0.2, 0.1, 0.05):
    s = exponential_smoothing(true, a)            # noise-free signal: the delay is not blurred by noise
    print(f"a = {a:<4}: peak at x = {x[np.argmax(s)]:.2f}, delay {x[np.argmax(s)] - x_peak:.2f} "
          f"(theory (1-a)/a samples = {(1 - a)/a*dx:.2f}), height {s.max():.3f} (true {true.max():.3f})")
plt.plot(x, y, ".", ms=2, color="gray"); plt.plot(x, true, "k", label="true")
plt.plot(x, exponential_smoothing(y, 0.1), label="exponential smoothing, a = 0.1")
plt.xlim(5, 14); plt.legend(); plt.show()"""),
         md(r"""
Exponential smoothing uses only past values - it can run in real time, sample by sample, which is why
instruments use it. The price is a **delay**: the output lags the signal by about $(1 - a)/a$ samples, so
peaks appear later (and lower). Centred filters (moving average, Savitzky-Golay, LOWESS) need future
samples and have no lag, which is fine for recorded data. (For strong smoothing of a narrow peak the
measured delay is somewhat below $(1 - a)/a$, a rule derived for slowly varying signals - and the peak is
also much lower.)
""")],
    ]


# =====================================================================================================
@solutions("08_linear_regression", setup=r"""from engmath import datasets, fitting
cal = np.loadtxt(datasets.path("thermocouple_calibration.csv"), delimiter=",", skiprows=1)
T_ref, E = cal[:, 0], cal[:, 1]
X2 = np.column_stack([np.ones_like(T_ref), T_ref, T_ref**2])
quad = fitting.linear_lstsq(X2, E)""")
def _():
    return [
        [code(r"""X3 = np.column_stack([X2, T_ref**3])
cub = fitting.linear_lstsq(X3, E)
ci3 = cub.conf_int()
print(f"cubic term {cub.coef[3]:.2e}, 95 % CI [{ci3[3,0]:.2e}, {ci3[3,1]:.2e}]")
for i, name in enumerate(["a0", "a1", "a2"]):
    w2 = np.ptp(quad.conf_int()[i]); w3 = np.ptp(ci3[i])
    print(f"{name}: CI width quadratic fit {w2:.2e}, cubic fit {w3:.2e} ({w3/w2:.1f}x wider)")"""),
         md(r"""
The confidence interval of the cubic term contains zero: it is not supported by the data. Adding it also
**widens** the intervals of the other coefficients, because $T^2$ and $T^3$ are strongly correlated over
0-400 °C - the data cannot tell their effects apart, so each becomes less certain. Unnecessary terms
cost precision; keep the simplest model the data support.
""")],
        [code(r"""low = T_ref <= 200
q_low = fitting.linear_lstsq(X2[low], E[low])
x400 = np.array([1, 400, 400**2])
print(f"voltage at 400 °C: measured {E[-1]:.4f} mV, full fit {x400 @ quad.coef:.4f} mV, "
      f"fit to T <= 200 °C only {x400 @ q_low.coef:.4f} mV")
print(f"extrapolation error {abs(x400 @ q_low.coef - E[-1])*1000:.0f} µV vs residual SD of the fit {q_low.sigma*1000:.0f} µV")"""),
         md(r"""
Within its own range the restricted fit is as good as the full one, but at 400 °C it misses by many times
its residual scatter. The curvature coefficient is poorly determined from the lower half alone, and its
error grows like $T^2$ outside that range. Extrapolation amplifies uncertainty; calibrate over the full
range of use.
""")],
        [code(r"""R = 8.314
T = np.linspace(300, 400, 11)
k_true = 1e7 * np.exp(-50000/(R*T))
sigma = 0.02 * k_true.max()                        # constant ABSOLUTE error (e.g. detector noise)
rng = np.random.default_rng(8)
estimates = {"ln k, equal weights": [], "ln k, weights k²/σ²": [], "nonlinear fit to k": []}
for _ in range(500):
    k = k_true + rng.normal(0, sigma, T.size)
    if np.any(k <= 0):
        continue
    X = np.column_stack([np.ones_like(T), -1/(R*T)])
    estimates["ln k, equal weights"].append(fitting.linear_lstsq(X, np.log(k)).coef[1])
    estimates["ln k, weights k²/σ²"].append(fitting.linear_lstsq(X, np.log(k), weights=k**2).coef[1])
    estimates["nonlinear fit to k"].append(fitting.levenberg_marquardt(
        lambda T, lnA, Ea: np.exp(lnA - Ea/(R*T)), T, k, [16, 50000]).coef[1])
for name, v in estimates.items():
    print(f"{name:<22}: Ea = {np.mean(v)/1000:6.2f} ± {np.std(v)/1000:.2f} kJ/mol (true 50.00)")"""),
         md(r"""
Taking logarithms changes the error structure: a constant absolute error $\sigma$ in $k$ becomes an error
$\sigma/k$ in $\ln k$ - large for the small rate constants at low temperature. Fitting $\ln k$ with equal
weights therefore trusts the noisiest points as much as the best ones, and the estimate of $E_a$ becomes
much less precise. Weights $k^2/\sigma^2$ (the inverse variance of $\ln k$) restore the balance, and the
nonlinear fit to $k$ itself does so automatically. Linearise for starting values; fit the original model.
""")],
        [code(r"""E_bad = E.copy(); E_bad[-1] -= 0.8
coef = np.linalg.lstsq(X2, E_bad, rcond=None)[0]
for it in range(1, 100):
    r = E_bad - X2 @ coef
    s = np.median(np.abs(r - np.median(r))) / 0.6745
    w = np.minimum(1.0, 1.345 / np.maximum(np.abs(r / s), 1e-12))
    sw = np.sqrt(w)
    new = np.linalg.lstsq(X2 * sw[:, None], E_bad * sw, rcond=None)[0]
    if np.max(np.abs(new - coef)) < 1e-12 * (1 + np.max(np.abs(coef))):
        coef = new; break
    coef = new
lib = fitting.huber_irls(X2, E_bad)
print(f"converged after {it} iterations; max difference from the library {np.max(np.abs(coef - lib.coef)):.1e}")
print("weights:", np.round(w, 2))"""),
         md(r"""
Each pass computes residuals from the current fit, turns them into weights (full weight within 1.345
robust standard deviations, decreasing beyond), and refits by weighted least squares. The faulty reading
ends with a weight near zero - the same result as the library function.
""")],
    ]


# =====================================================================================================
@solutions("09_nonlinear_regression", setup=r"""from engmath import fitting
A_t, B_t, C_t = 8.20417, 1642.89, 230.300
T = np.linspace(20, 90, 12)
p = 10**(A_t - B_t/(C_t + T)) * (1 + 0.01*np.random.default_rng(5).standard_normal(T.size))
def antoine_log(T, A, B, C):
    return A - B/(C + T)
fit_log = fitting.levenberg_marquardt(antoine_log, T, np.log10(p), p0=[8.0, 1500.0, 220.0])""")
def _():
    return [
        [code(r"""fit_p = fitting.levenberg_marquardt(lambda T, A, B, C: 10**(A - B/(C + T)), T, p, p0=[8.0, 1500.0, 220.0])
for name, f in (("fit to log10 p", fit_log), ("fit to p      ", fit_p)):
    rel = 100 * (10**antoine_log(T, *f.coef) / p - 1)
    print(f"{name}: A, B, C = {np.round(f.coef, 3)};  relative residuals at 20 °C {rel[0]:+.2f} %, at 90 °C {rel[-1]:+.2f} %")"""),
         md(r"""
With equal weights on $p$, the high-temperature points - whose pressures are twenty times larger - dominate
the sum of squares, and the fit sacrifices the low-temperature end. Because the measurement errors are
*relative* (1 % of $p$), fitting $\log p$ weights all points equally in the right sense: its residuals
stay within about 1 % everywhere, while the fit to $p$ is off by more than 5 % at 20 °C. The two fits
also give very different parameters - with strongly correlated parameters, how the data are weighted
changes the estimates a lot.
""")],
        [code(r"""fitC = fitting.linear_lstsq(np.column_stack([np.ones(T.size), -1/(230.3 + T)]), np.log10(p))
print(f"C fixed at 230.3: A = {fitC.coef[0]:.4f} ± {fitC.se[0]:.4f},  B = {fitC.coef[1]:.2f} ± {fitC.se[1]:.2f}")
print(f"all three free:   A = {fit_log.coef[0]:.4f} ± {fit_log.se[0]:.4f},  B = {fit_log.coef[1]:.2f} ± {fit_log.se[1]:.2f}")"""),
         md(r"""
Fixing $C$ removes the parameter that $A$ and $B$ were trading off against, and their standard errors shrink
by more than an order of magnitude. This is why tabulated Antoine constants often fix $C$, or quote all
three with the understanding that they belong together.
""")],
        [code(r"""T_inf, dT0, C, n_true = 20.0, 480.0, 5.0e-4, 1.25
t = np.r_[np.arange(0, 60, 5), np.arange(60, 300, 20), np.arange(300, 1801, 100)].astype(float)
T_noisy = T_inf + (dT0**-0.25 + C*t/4)**-4 + np.random.default_rng(3).normal(0, 0.5, t.size)
def cooling(t, dT_0, C_, n):          # solution of dT/dt = -C (T - T_inf)^n for n != 1
    return T_inf + (dT_0**(1 - n) + (n - 1)*C_*t)**(1/(1 - n))
fit = fitting.levenberg_marquardt(cooling, t, T_noisy, [450.0, 1e-3, 1.1])
print(f"integrated model: n = {fit.coef[2]:.4f} ± {fit.se[2]:.4f}   (true 1.25)")
print("differential method on the same data (notebook 06): n = 1.249 ± 0.019")"""),
         md(r"""
Fitting the integrated model uses the temperatures directly, without amplifying their noise by
differentiation, and determines $n$ far more precisely than the differential method. The differential
method remains useful for *discovering* the form of a rate law; for estimating its parameters, fit the
integrated model.
""")],
        [code(r"""def jac(f, x, params, rel=1e-7):
    J = np.empty((len(x), len(params)))
    for j in range(len(params)):
        dp = rel * max(abs(params[j]), 1e-8); sh = np.array(params, float); sh[j] += dp
        J[:, j] = (f(x, *sh) - f(x, *params)) / dp
    return J

def lm_by_hand(start, lam0, iters=40, damping=True):
    y_obs = np.log10(p); params = np.array(start, float); lam = lam0
    rss = np.sum((y_obs - antoine_log(T, *params))**2); rejected = 0
    for _ in range(iters):
        J = jac(antoine_log, T, params); A = J.T @ J
        damp = lam * np.diag(np.diag(A)) if damping else 0
        try:
            step = np.linalg.solve(A + damp, J.T @ (y_obs - antoine_log(T, *params)))
        except np.linalg.LinAlgError:
            return params, rss, rejected
        new = np.sum((y_obs - antoine_log(T, *(params + step)))**2)
        if damping:
            if new < rss: params, rss, lam = params + step, new, lam/10
            else: lam *= 10; rejected += 1
        else:
            params, rss = params + step, new          # Gauss-Newton accepts every step
    return params, rss, rejected

print(f"optimum RSS (library): {np.sum(fit_log.residuals**2):.6e}")
p_lm, rss_lm, rej = lm_by_hand([7.5, 1300, 180], 1e-3)
print(f"LM from (7.5, 1300, 180), 40 iterations: RSS {rss_lm:.6e}, {rej} rejected steps, A, B, C = {np.round(p_lm, 2)}")
with np.errstate(all="ignore"):
    for start in ([8, 1500, 220], [7.5, 1300, 180], [5, 500, 100], [10, 3000, 300]):
        p_gn, rss_gn, _ = lm_by_hand(start, 0.0, iters=15, damping=False)
        print(f"pure Gauss-Newton from {start}, 15 iterations: RSS {rss_gn:.6e}, A, B, C = {np.round(p_gn, 2)}")"""),
         md(r"""
With correlations near 0.999 the RSS surface is a long, narrow valley. Levenberg-Marquardt insists that
every step lowers the RSS: steps that overshoot the valley floor are rejected, $\lambda$ grows, and progress
becomes a crawl (here it gets there, but needs 40 iterations). Pure Gauss-Newton simply takes every
step - and on this problem it converges quickly from **all** starting points, even very poor ones. The
reason: the Antoine model is linear in $A$ and $B$, and only $C$ enters nonlinearly, so the linearisation
is good almost everywhere, and Gauss-Newton recovers from a temporary overshoot. Damping is insurance:
it costs speed on well-behaved problems like this one, and pays off on strongly nonlinear models or
starts far from the optimum, where undamped steps can diverge.
""")],
    ]


# =====================================================================================================
@solutions("10_implicit_models_optimisation", setup=r"""from engmath import datasets, optimize, roots
rho, mu, L, D = 998.0, 1.0e-3, 20.0, 0.05
meas = np.loadtxt(datasets.path("pipe_pressure_drop.csv"), delimiter=",", skiprows=1)
Q, dp = meas[:, 0], meas[:, 1]
eps_true, k_true = 0.15e-3, 1.03
def colebrook_f(Re, eps):
    return roots.brent(lambda f: 1/np.sqrt(f) + 2*np.log10(eps/D/3.7 + 2.51/(Re*np.sqrt(f))), 1e-4, 0.2).root
def dp_model(Q, eps, k=1.0, friction=colebrook_f):
    v = 4*k*np.atleast_1d(Q)/(np.pi*D**2)
    Re = rho*v*D/mu
    return np.array([friction(R, eps) for R in Re]) * L/D * rho*v**2/2
def fit(objective):
    r = optimize.nelder_mead(lambda p: objective(p[0], p[1]), [-4.0, 1.0], step=0.05, tol=1e-12)
    return 10**r.x[0], r.x[1]
rel_rss = lambda le, k: np.sum(np.log(dp_model(Q, 10**le, k) / dp)**2)
eps_rel, k_rel = fit(rel_rss)""")
def _():
    return [
        [code(r"""abs_rss = lambda le, k: np.sum((dp_model(Q, 10**le, k) - dp)**2)
eps_abs, k_abs = fit(abs_rss)
print(f"relative residuals: eps = {eps_rel*1000:.3f} mm, k = {k_rel:.4f}")
print(f"absolute residuals: eps = {eps_abs*1000:.3f} mm, k = {k_abs:.4f}   (truth {eps_true*1000} mm, {k_true})")
print(f"share of the absolute RSS scale: largest ΔP² / sum ΔP² = {dp.max()**2 / np.sum(dp**2):.0%}")"""),
         md(r"""
With absolute residuals the two highest flows carry most of the weight (the largest pressure drop alone
accounts for over half of $\sum \Delta P^2$), so the estimates are driven by very few points and change
noticeably. With a *relative* measurement error (2 % here), relative - logarithmic - residuals give every
measurement its fair weight.
""")],
        [code(r"""nu = mu / rho
Q_lam = np.array([2e-5, 4e-5, 6e-5])                                  # Re < 2000
v_lam = 4*k_true*Q_lam/(np.pi*D**2)
dp_lam = 64/(v_lam*D/nu) * L/D * rho*v_lam**2/2 * (1 + 0.02*np.random.default_rng(9).standard_normal(3))
Q_all, dp_all = np.r_[Q_lam, Q], np.r_[dp_lam, dp]
def friction_any(Re, eps):
    return 64/Re if Re < 2000 else colebrook_f(Re, eps)
rss_all = lambda le, k: np.sum(np.log(dp_model(Q_all, 10**le, k, friction_any) / dp_all)**2)
from engmath import fitting
def log_dp(Qs, log_eps, k):          # log of the model pressure drop: the residual we minimise
    return np.log(dp_model(Qs, 10**log_eps, k, friction_any))
for label, Qs, dps in (("turbulent data only", Q, dp), ("with laminar points", Q_all, dp_all)):
    f = fitting.levenberg_marquardt(log_dp, Qs, np.log(dps), [-3.8, 1.0])
    print(f"{label:<20}: log10 eps = {f.coef[0]:.3f} ± {f.se[0]:.3f} (eps {10**f.coef[0]*1000:.3f} mm), "
          f"k = {f.coef[1]:.4f} ± {f.se[1]:.4f}, correlation {f.correlation[0, 1]:+.3f}")
print(f"truth: eps = {eps_true*1000} mm, k = {k_true}")
le = np.linspace(-4.5, -3.2, 45); kk = np.linspace(0.97, 1.09, 45)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, obj, title in ((axs[0], rel_rss, "turbulent data only"), (axs[1], rss_all, "with 3 laminar points")):
    Z = np.array([[obj(a, b) for a in le] for b in kk])
    ax.contourf(10**le*1000, kk, np.log10(Z), levels=25, cmap="viridis")
    ax.plot(eps_true*1000, k_true, "wx", ms=10, mew=2)
    ax.set(xscale="log", xlabel="ε (mm)", ylabel="k", title=title)
plt.show()"""),
         md(r"""
In laminar flow $f = 64/Re$ does not depend on roughness at all, so laminar points pin down the meter
factor $k$ on their own. The valley becomes shorter, and the **standard errors** of both parameters fall
by a factor of about 2.5, with a weaker correlation between them. Judge an experimental design by these
uncertainties, not by whether one noisy data set happens to land closer to the truth: in both fits the
true values lie within about 1.2 standard errors. A few well-chosen extra measurements do more than
many more of the same kind: **design experiments to break parameter correlations.**
""")],
        [code(r"""def haaland_f(Re, eps):
    return (-1.8*np.log10((eps/D/3.7)**1.11 + 6.9/Re))**-2
hl_rss = lambda le, k: np.sum(np.log(dp_model(Q, 10**le, k, haaland_f) / dp)**2)
eps_h, k_h = fit(hl_rss)
print(f"Colebrook model: eps = {eps_rel*1000:.3f} mm, k = {k_rel:.4f}")
print(f"Haaland model:   eps = {eps_h*1000:.3f} mm, k = {k_h:.4f}")
from engmath import fitting
f = fitting.levenberg_marquardt(lambda Qs, le, k: np.log(dp_model(Qs, 10**le, k)), Q, np.log(dp), [-3.8, 1.0])
se_eps = 10**f.coef[0] * np.log(10) * f.se[0]
print(f"model bias in eps: {abs(eps_h - eps_rel)*1000:.3f} mm;  statistical standard error of eps: {se_eps*1000:.3f} mm")"""),
         md(r"""
The bias caused by the approximate friction formula is about a quarter of the statistical standard error
of the roughness, which comes from the measurement noise and the parameter correlation. Here the cheaper
explicit formula would be acceptable - but that is a conclusion to *check*, not to assume.
""")],
        [code(r"""span, tol = 3.5, 1e-6
gs = optimize.golden_section(lambda le: rel_rss(le, 1.0), -6, -2.5, tol=tol)
print(f"golden-section search: {gs.iterations + 2} evaluations")
print(f"grid search to the same precision: {int(span/tol) + 1:,} evaluations")
for n in (36, 351, 3501):
    grid = np.linspace(-6, -2.5, n)
    best = grid[np.argmin([rel_rss(g, 1.0) for g in grid])] if n <= 351 else None
    if best is not None:
        print(f"grid of {n:>4} points: eps = {10**best*1000:.4f} mm (spacing {span/(n-1):.3f} in log10)")
print(f"golden section:       eps = {10**gs.x[0]*1000:.4f} mm")"""),
         md(r"""
Each golden-section step shrinks the bracket by 0.618, so reaching a precision of $10^{-6}$ in
$\log_{10}\varepsilon$ from a bracket of width 3.5 takes about 30 evaluations; a uniform grid needs
3.5 million. For expensive models - here every evaluation solves eight Colebrook equations - the choice of
algorithm matters enormously. Grids remain useful for a first look at the landscape.
""")],
    ]


# =====================================================================================================
@solutions("11_odes", setup=r"""from engmath import odes, optimize
k1, k2, CA0 = 0.5, 0.2, 2.0""")
def _():
    return [
        [code(r"""A_tank, a_orifice, Cd, g, h0 = 1.0, 1.0e-3, 0.62, 9.81, 2.0
t_drain = A_tank / (Cd * a_orifice) * np.sqrt(2 * h0 / g)
rhs = lambda t, h: [-Cd * a_orifice / A_tank * np.sqrt(2 * g * max(h[0], 0.0))]
h_exact = lambda t: (np.sqrt(h0) - Cd * a_orifice / A_tank * np.sqrt(g / 2) * t)**2
print(f"exact draining time {t_drain:.1f} s")
for n in (20, 80, 320):
    t, h = odes.rk4(rhs, [h0], 0, t_drain, n)
    err = np.abs(h[:, 0] - h_exact(t))
    print(f"{n:>3} steps: error at t/2 {err[n//2]:.1e} m, at the end {err[-1]:.1e} m")
t, h = odes.rk4(rhs, [h0], 0, t_drain, 80)
plt.semilogy(t, np.abs(h[:, 0] - h_exact(t)) + 1e-18); plt.xlabel("time (s)"); plt.ylabel("|error| (m)")
plt.title("RK4 error grows as the tank empties"); plt.show()"""),
         md(r"""
RK4 is extremely accurate for most of the draining, but the error grows sharply as $h \to 0$. There the
right-hand side $\propto \sqrt h$ has an infinite derivative, so the smoothness that RK4's error formula
assumes is lost - the same reason the turbulent profile of notebook 00 converged slowly. The error at the
end also converges much more slowly with the step size than the error at mid-time. (Near-singular
behaviour like this is where adaptive step control helps most.)
""")],
        [code(r"""def CB_at_3(k1_):
    if abs(k1_ - k2) < 1e-9:                                  # limit k1 -> k2: C_B = CA0 k t e^(-k t)
        return CA0 * k2 * 3 * np.exp(-k2 * 3)
    return CA0 * k1_ / (k2 - k1_) * (np.exp(-k1_ * 3) - np.exp(-k2 * 3))
res = optimize.golden_section(lambda kk: -CB_at_3(kk), 0.01, 5.0)
k_opt = res.x[0]
_, c = odes.rk4(lambda t, y: [-k_opt*y[0], k_opt*y[0] - k2*y[1]], [CA0, 0.0], 0, 3, 300)
print(f"optimal k1 = {k_opt:.4f} 1/min; C_B(3 min) = {-res.fun:.4f} mol/L (RK4 check {c[-1, 1]:.4f})")
kk = np.linspace(0.05, 3, 200)
plt.plot(kk, [CB_at_3(v) for v in kk]); plt.axvline(k_opt, color="k", ls="--")
plt.xlabel("k1 (1/min)"); plt.ylabel("C_B after 3 min (mol/L)"); plt.show()"""),
         md(r"""
There is an optimum: with a small $k_1$, too little B has formed after 3 minutes; with a very large $k_1$,
B forms almost instantly and then has 3 minutes to decay. In practice $k_1$ is adjusted through the
temperature or the catalyst - this is how a reactor operating point is chosen. Wrapping an optimiser
around a model (here even an ODE solver) is a very common pattern.
""")],
        [code(r"""k2_fast = 1000.0
stiff = lambda t, c: [-k1*c[0], k1*c[0] - k2_fast*c[1]]
for h in (0.0020, 0.0025, 0.0027, 0.0028, 0.0029, 0.0030):
    with np.errstate(all="ignore"):
        _, c = odes.rk4(stiff, [CA0, 0.0], 0, 2.0, int(round(2.0/h)))
    print(f"h = {h:.4f} (k2 h = {k2_fast*h:.2f}): C_B(2) = {c[-1, 1]:.4e}  {'stable' if abs(c[-1, 1]) < 1 else 'UNSTABLE'}")
print(f"Euler limit 2/k2 = {2/k2_fast:.4f};  RK4 theory (|lambda h| < 2.785) = {2.785/k2_fast:.4f}")"""),
         md(r"""
RK4 stays stable up to $k_2 h \approx 2.79$, only about 40 % more than Euler's limit of 2. Higher order
buys accuracy, not stability: every explicit method has a bounded stability region, so for stiff
problems the step is limited by the fastest time scale, however smooth the solution. Implicit methods
remove that limit.
""")],
        [code(r"""def heun(fun, y0, t0, t1, n):
    t = np.linspace(t0, t1, n + 1); y = np.empty((n + 1, len(y0))); y[0] = y0
    for i in range(n):
        h = t[i+1] - t[i]
        k_start = np.array(fun(t[i], y[i]))
        y_pred = y[i] + h * k_start                       # Euler predictor
        y[i+1] = y[i] + h/2 * (k_start + np.array(fun(t[i+1], y_pred)))   # trapezoidal corrector
    return t, y

rhs = lambda t, c: [-k1*c[0], k1*c[0] - k2*c[1], k2*c[1]]
CB_exact = CA0*k1/(k2 - k1)*(np.exp(-k1*10) - np.exp(-k2*10))
ns = np.array([80, 160, 320, 640])              # fine enough to be in the asymptotic range
for name, solver in (("Euler", odes.euler), ("Heun", heun), ("RK4", odes.rk4)):
    errs = [abs(solver(rhs, [CA0, 0, 0], 0, 10, n)[1][-1, 1] - CB_exact) for n in ns]
    print(f"{name:<6}: observed order {np.polyfit(np.log(10/ns), np.log(errs), 1)[0]:.2f}")"""),
         md(r"""
Heun's method is second order: two slope evaluations per step, between Euler (one evaluation, first
order) and RK4 (four evaluations, fourth order). For smooth problems a higher order pays off quickly -
halving the step reduces Heun's error 4-fold and RK4's 16-fold.
""")],
    ]


# =====================================================================================================
@solutions("12_diffusion_pde", setup=r"""from engmath import datasets, optimize, pdes, linalg
from scipy.special import erf, erfc
alpha, T0, Ts = 1.2e-5, 20.0, 320.0
tc = np.loadtxt(datasets.path("thermocouple_10mm.csv"), delimiter=",", skiprows=1)
t_meas, T_meas = tc[:, 0], tc[:, 1]
def fit_alpha(x_tc):
    rss = lambda la: np.sum((Ts + (T0 - Ts)*erf(x_tc/(2*np.sqrt(10**la*t_meas))) - T_meas)**2)
    return 10**optimize.golden_section(rss, -6, -4).x[0]""")
def _():
    return [
        [code(r"""a_nom = fit_alpha(0.010)
for x_tc in (0.0095, 0.0105):
    a = fit_alpha(x_tc)
    print(f"thermocouple at {x_tc*1000:.1f} mm: alpha = {a*1e6:.2f} mm²/s ({(a/a_nom - 1)*100:+.1f} %)")
print(f"nominal 10.0 mm: alpha = {a_nom*1e6:.2f} mm²/s; statistical standard error about 0.05 mm²/s (notebook 12)")"""),
         md(r"""
The data only determine the combination $x/\sqrt{\alpha t}$, so $\alpha \propto x^2$ and a 5 % position error
becomes a 10 % error in $\alpha$ - twenty times the statistical uncertainty from the temperature noise. The
thermocouple position dominates the uncertainty budget. A better experiment measures the position
precisely (e.g. by X-ray) or uses two thermocouples, so that only their *distance* matters.
""")],
        [code(r"""L, nx = 0.25, 501
x = np.linspace(0, L, nx); dx = x[1] - x[0]
h, k = 500.0, 45.0                                      # W/(m2 K), W/(m K): hot gas on a steel surface
T_gas = Ts
def cn_convective(t_end, dt):
    r = alpha * dt / dx**2
    Bi = 2 * dx * h / k                                  # ghost node: T_-1 = T_1 + (2 dx h/k)(T_gas - T_0)
    n = nx - 1                                           # unknowns T_0 ... T_{nx-2}; far end fixed at T0
    lower, diag, upper = np.full(n, -r/2), np.full(n, 1 + r), np.full(n, -r/2)
    diag[0] = 1 + r + r/2*Bi; upper[0] = -r              # first row: the only entries that change
    T = np.full(nx, T0)
    for _ in range(int(round(t_end/dt))):
        rhs = r/2*np.r_[T[1], T[:-2]] + (1 - r)*T[:-1] + r/2*T[1:]
        rhs[0] = (1 - r - r/2*Bi)*T[0] + r*T[1] + r*Bi*T_gas
        rhs[-1] += r/2*T0
        T[:-1] = linalg.thomas(lower, diag, upper, rhs)
    return T
def exact_conv(x, t):                                   # semi-infinite solid with surface convection
    s = np.sqrt(alpha*t)
    theta = erfc(x/(2*s)) - np.exp(h*x/k + h**2*alpha*t/k**2) * erfc(x/(2*s) + h*s/k)
    return T0 + (T_gas - T0) * theta
T60 = cn_convective(60.0, 0.05)
print(f"surface temperature after 60 s: CN {T60[0]:.3f} °C, exact {exact_conv(0.0, 60.0):.3f} °C; "
      f"max error {np.max(np.abs(T60 - exact_conv(x, 60.0))):.3f} K")"""),
         md(r"""
With convection $-k\,\partial T/\partial x = h(T_{gas} - T)$ at the surface, the surface temperature becomes an
unknown. A ghost node eliminates the derivative, and only the **first row** changes - its diagonal and
upper entries, and a new term $r\,Bi\,T_{gas}$ on the right-hand side - so the system stays tridiagonal. The
result agrees with the classical analytical solution for a semi-infinite solid with surface convection.
""")],
        [code(r"""from engmath import fitting
Cs, D_true = 200.0, 1.0e-12                         # mol/m3, m2/s
t = np.linspace(60, 3600, 20)
rng = np.random.default_rng(12)
M = 2*Cs*np.sqrt(D_true*t/np.pi) * (1 + 0.03*rng.standard_normal(t.size))
f = fitting.linear_lstsq(np.sqrt(t)[:, None], M)       # M = slope * sqrt(t): a line through the origin
slope, se = f.coef[0], f.se[0]
D_est = np.pi * (slope/(2*Cs))**2
print(f"D = {D_est:.3e} ± {2*se/slope*D_est:.1e} m²/s  (true {D_true:.1e})")
plt.plot(np.sqrt(t), M*1e6, "o"); plt.plot(np.sqrt(t), slope*np.sqrt(t)*1e6)
plt.xlabel("√t (√s)"); plt.ylabel("uptake (µmol/m²)"); plt.show()"""),
         md(r"""
$M_t = \left(2C_s\sqrt{D/\pi}\right)\sqrt t$ is a straight line through the origin in $\sqrt t$, so ordinary linear least
squares gives the slope, and $D = \pi\,(\text{slope}/2C_s)^2$. Because $D$ depends on the *square* of the slope,
its relative uncertainty is twice that of the slope. The early-time formula only holds while the
penetration depth is small compared with the film thickness; later points bend away from the line.
""")],
        [code(r"""L2, nx2 = 0.25, 501
x2 = np.linspace(0, L2, nx2); dx2 = x2[1] - x2[0]
exact = lambda x, t: Ts + (T0 - Ts)*erf(x/(2*np.sqrt(alpha*t)))
def implicit_euler(t_end, r):
    dt = r*dx2**2/alpha; n = nx2 - 2
    lower, diag, upper = np.full(n, -r), np.full(n, 1 + 2*r), np.full(n, -r)
    T = np.full(nx2, T0); T[0] = Ts
    for _ in range(max(1, int(round(t_end/dt)))):
        rhs = T[1:-1].copy(); rhs[0] += r*Ts; rhs[-1] += r*T0
        T[1:-1] = linalg.thomas(lower, diag, upper, rhs)
    return T
for t_end in (1.0, 60.0):
    for r in (10.0, 100.0):
        n_steps = max(1, int(round(t_end/(r*dx2**2/alpha))))
        dt = t_end / n_steps
        e_ie = np.max(np.abs(implicit_euler(t_end, r) - exact(x2, t_end)))
        e_cn = np.max(np.abs(pdes.crank_nicolson(np.full(nx2, T0), alpha, dx2, dt, n_steps, Ts, T0) - exact(x2, t_end)))
        print(f"t = {t_end:4.0f} s, r = {r:5.0f}: implicit Euler error {e_ie:7.3f} K, Crank-Nicolson {e_cn:7.3f} K")"""),
         md(r"""
Implicit Euler is stable for any $r$, like Crank-Nicolson. At early times with large steps it is the more
accurate of the two here: it strongly damps the short-wavelength components excited by the sudden
surface jump, which Crank-Nicolson lets oscillate. At later times, with a moderate step ($r$ = 10),
Crank-Nicolson's second-order accuracy in time wins clearly - but with $r$ = 100 its oscillations have still
not died out after 60 s. Practical codes therefore often start with a few implicit Euler steps (or small
steps) and then switch to Crank-Nicolson.
""")],
    ]


# =====================================================================================================
@solutions("13_symbolic_sympy", setup="import sympy as sp")
def _():
    return [
        [code(r"""x, L, EI, w = sp.symbols("x L EI w", positive=True)
v = sp.Function("v")
bcs = {v(0): 0, v(x).diff(x, 2).subs(x, 0): 0, v(L): 0, v(x).diff(x, 2).subs(x, L): 0}
sol = sp.factor(sp.dsolve(sp.Eq(EI*v(x).diff(x, 4), w), v(x), ics=bcs).rhs)
print("v(x) =", sol)
print("midspan deflection =", sp.simplify(sol.subs(x, L/2)))
print("slope at the support =", sp.simplify(sol.diff(x).subs(x, 0)))"""),
         md(r"""
The maximum deflection (at midspan, by symmetry) is $5wL^4/(384EI)$ - only $5/48 \approx 0.10$ of the
cantilever's $wL^4/(8EI)$ for the same span and load, because both ends are supported.
""")],
        [code(r"""h = sp.Symbol("h", positive=True)
F = sp.symbols("F0:8")
taylor = lambda s: sum(F[j]*s**j/sp.factorial(j) for j in range(8))
forward = (taylor(h) - taylor(0)) / h - F[1]
five_point = (taylor(-2*h) - 8*taylor(-h) + 8*taylor(h) - taylor(2*h)) / (12*h) - F[1]
print("forward difference error:   ", sp.expand(forward).as_leading_term(h))
print("5-point central error:      ", sp.expand(five_point).as_leading_term(h))"""),
         md(r"""
The forward difference has leading error $h f''/2$: first order. The 5-point central formula cancels all
terms up to $h^4$: its leading error is $-h^4 f^{(5)}/30$, fourth order. More points buy higher order, at the
cost of a wider stencil (awkward near boundaries) and more sensitivity to noise.
""")],
        [code(r"""x, m, Tb, Tinf, L = sp.symbols("x m T_b T_inf L", positive=True)
T = sp.Function("T")
sol = sp.dsolve(sp.Eq(T(x).diff(x, 2), m**2*(T(x) - Tinf)), T(x),
                ics={T(0): Tb, T(x).diff(x).subs(x, L): 0}).rhs
print("T(x) =", sol)
textbook = Tinf + (Tb - Tinf)*sp.cosh(m*(L - x))/sp.cosh(m*L)
print("equal to the textbook form:", sp.simplify((sol - textbook).rewrite(sp.exp)) == 0)
from engmath import bvp
vals = {m: 10.0, Tb: 100.0, Tinf: 20.0, L: 0.05}
T_fun = sp.lambdify(x, sol.subs(vals), "numpy")
xf, Tf = bvp.finite_difference(lambda s: 0*s, lambda s: -100.0 + 0*s, lambda s: -100.0*20.0 + 0*s, 0, 0.05, 100,
                               ("dirichlet", 100.0), ("robin", 1.0, 0.0, 0.0))
print(f"max |finite differences - SymPy| = {np.max(np.abs(Tf - T_fun(xf))):.1e} K")"""),
         md(r"""
SymPy returns the solution with exponentials; the symbolic check confirms it equals the textbook form
$T - T_\infty = (T_b - T_\infty)\cosh(m(L - x))/\cosh(mL)$. The finite-difference solution with an insulated
(Neumann) tip agrees to within the expected second-order discretisation error.
""")],
        [code(r"""h1, h2, t = sp.symbols("h1 h2 t", positive=True)
f0, f1, f2 = sp.symbols("f0 f1 f2")
F = sp.symbols("F0:6")
second = sp.expand(sp.diff(sp.interpolate([(-h1, f0), (0, f1), (h2, f2)], t), t, 2))
print("f'' ≈", sp.factor(second))
taylor = lambda s: sum(F[j]*s**j/sp.factorial(j) for j in range(6))
err = sp.expand(second.subs({f0: taylor(-h1), f1: taylor(0), f2: taylor(h2)}) - F[2])
print("leading error:", sp.factor(err.coeff(F[3])*F[3]))
print("equal spacing:", sp.factor(err.subs({h1: sp.Symbol("h"), h2: sp.Symbol("h")})))
from engmath import differentiate
xs = np.array([0.0, 0.3, 0.5, 1.0, 1.2]); ys = np.exp(xs)
fun = sp.lambdify((f0, f1, f2, h1, h2), second)
print("formula:", fun(ys[1], ys[2], ys[3], xs[2]-xs[1], xs[3]-xs[2]), " library:", differentiate.second_derivative(xs, ys)[2])"""),
         md(r"""
On an uneven grid the second-derivative formula is only **first order**: its leading error is
$(h_2 - h_1)f'''/3$. The first-order term cancels only when $h_1 = h_2$, leaving the familiar second-order
error $h^2 f^{(4)}/12$. Uneven grids must therefore change smoothly (neighbouring spacings nearly equal) to
keep second-order accuracy for second derivatives.
""")],
    ]


# =====================================================================================================
@solutions("14_nonlinear_systems", setup=r"""from engmath import nonlinear, roots
z = np.array([120.0, 100.0, 60.0])
Lp = np.array([1000.0, 800.0, 1200.0]); Dp = np.array([0.30, 0.25, 0.20])
g = 9.81
k_const = 8 * 0.02 * Lp / (g * np.pi**2 * Dp**5)
def F_const(v):
    H, Q = v[0], v[1:]
    return np.r_[z - H - k_const * Q * np.abs(Q), Q.sum()]
sol_const = nonlinear.newton_system(F_const, [90.0, 0.1, 0.0, -0.1])""")
def _():
    return [
        [code(r"""eps, nu = 0.045e-3, 1.0e-6
def friction(Q, D):
    Re = max(4*abs(Q)/(np.pi*D*nu), 4000.0)            # guard: Colebrook needs turbulent flow
    return roots.brent(lambda f: 1/np.sqrt(f) + 2*np.log10(eps/D/3.7 + 2.51/(Re*np.sqrt(f))), 1e-4, 0.1).root
def F_cole(v):
    H, Q = v[0], v[1:]
    f = np.array([friction(q, d) for q, d in zip(Q, Dp)])
    k = 8 * f * Lp / (g * np.pi**2 * Dp**5)
    return np.r_[z - H - k * Q * np.abs(Q), Q.sum()]
sol = nonlinear.newton_system(F_cole, sol_const.x)
print(f"junction head: constant f {sol_const.x[0]:.2f} m, Colebrook {sol.x[0]:.2f} m")
for i in range(3):
    print(f"pipe {i+1}: {sol_const.x[i+1]*1000:7.1f} L/s -> {sol.x[i+1]*1000:7.1f} L/s, "
          f"friction factor {friction(sol.x[i+1], Dp[i]):.4f}")"""),
         md(r"""
With Colebrook the friction factors fall below the assumed 0.02 for these smooth steel pipes at high
Reynolds numbers, so all flows increase. The solution method is unchanged: $F$ simply contains inner root
solves. Starting from the constant-$f$ solution (a *continuation* strategy) gives Newton a good start.
""")],
        [code(r"""z4 = np.r_[z, 90.0]; L4 = np.r_[Lp, 900.0]; D4 = np.r_[Dp, 0.25]
k4 = 8 * 0.02 * L4 / (g * np.pi**2 * D4**5)
F4 = lambda v: np.r_[z4 - v[0] - k4 * v[1:] * np.abs(v[1:]), v[1:].sum()]
sol4 = nonlinear.newton_system(F4, [90.0, 0.1, 0.0, -0.1, 0.0])
print(f"junction head {sol4.x[0]:.2f} m (converged: {sol4.converged})")
for i, q in enumerate(sol4.x[1:], start=1):
    print(f"reservoir {i} ({z4[i-1]:.0f} m): {'supplies' if q > 0 else 'receives'} {abs(q)*1000:.1f} L/s")"""),
         md(r"""
The rule is simple once $H$ is known: reservoirs **above** the junction head supply water, those below
receive it. Adding the 90 m reservoir lowers the junction head (from 105.2 m to 100.1 m), because it
offers another outlet. The flow into the 100 m reservoir drops to a trickle: the junction head is now
barely above it, and a slightly larger fourth pipe would reverse that flow.
""")],
        [code(r"""for ls in (False, True):
    r = nonlinear.newton_system(F_const, [150.0, 0.1, 0.0, -0.1], line_search=ls, maxiter=50)
    print(f"line search {ls!s:5}: converged {r.converged} in {r.iterations} iterations, H = {r.x[0]:.2f} m; "
          f"residuals {np.array2string(np.array(r.residual_history[:6]), precision=1)}")"""),
         md(r"""
Both runs are identical: from 150 m every full Newton step already reduces $\|F\|$ enough, so the line
search never intervenes, and Newton converges quadratically to the (single, physical) solution. This
problem is well behaved. The line search only acts when a full step would *increase* the residual - on
harder problems (several solutions, sharper nonlinearity) that is what separates convergence from
divergence, as the `arctan` example in the notebook shows.
""")],
        [code(r"""def broyden(F, x, tol=1e-10, maxiter=100):
    x = np.array(x, float)
    f = np.asarray(F(x)); evals = 1
    J = nonlinear.jacobian(F, x); evals += len(x)          # one finite-difference Jacobian to start
    for it in range(maxiter):
        dx = np.linalg.solve(J, -f)
        x_new = x + dx
        f_new = np.asarray(F(x_new)); evals += 1
        J = J + np.outer(f_new - f - J @ dx, dx) / (dx @ dx)   # rank-one (Broyden) update
        x, f = x_new, f_new
        if np.linalg.norm(f) < tol:
            return x, it + 1, evals
    return x, maxiter, evals

xb, its, ev = broyden(F_const, [90.0, 0.1, 0.0, -0.1])
newton_evals = (sol_const.iterations) * (len(xb) + 1) + 1
print(f"Broyden: {its} iterations, {ev} evaluations of F;  Newton: {sol_const.iterations} iterations, about {newton_evals} evaluations")
print("same solution:", np.allclose(xb, sol_const.x, atol=1e-8))"""),
         md(r"""
Broyden's method needs more iterations - its Jacobian is only an approximation that improves as it goes
(superlinear rather than quadratic convergence) - but each iteration costs a single evaluation of $F$
instead of $n + 1$. When evaluating $F$ is expensive (a simulation, nested root solves) and $n$ is large,
that trade is usually worth it.
""")],
    ]


# =====================================================================================================
@solutions("15_boundary_value_problems", setup=r"""from engmath import bvp, linalg
import time
D = 0.005; L = 0.05; Ac = np.pi*D**2/4""")
def _():
    return [
        [code(r"""ks, hs, eps, sigma = 40.0, 10.0, 0.9, 5.670374419e-8
Tb2, Tinf2 = 673.15, 293.15
m2sq, rad = 4*hs/(ks*D), eps*sigma*4/(ks*D)
f = lambda x, T, dT: m2sq*(T - Tinf2) + rad*(T**4 - Tinf2**4)
xs, Ts, dTs, s = bvp.shooting(f, 0, L, Tb2, ("robin", 1.0, 0.0, 0.0), slopes=(-8000.0, -4000.0), n=400)
print(f"shooting: T'(0) = {s:.2f} K/m, heat rate {-ks*Ac*s:.3f} W, tip temperature {Ts[-1] - 273.15:.1f} °C")
print(f"tip slope T'(L) = {dTs[-1]:.2e} K/m (should be 0)")
print("finite differences + Newton (notebook 15): 6.01 W")"""),
         md(r"""
Shooting gives the same heat rate as the finite-difference/Newton solution of the notebook (the small
remaining difference is the finite-difference discretisation error). For a nonlinear ODE the tip miss is
no longer a straight-line function of the initial slope, so the secant method needs several iterations
instead of one - it still converges quickly because the miss is a smooth function of the slope.
""")],
        [code(r"""E, I = 200e9, 0.05*0.10**3/12
w, Lb = 2000.0, 4.0
EI = E*I
exact_mid = 5*w*Lb**4/(384*EI)
for n in (10, 20, 40, 80):
    # v'' = -M(x)/EI  with  M = w x (L - x) / 2   (v downwards, sagging moment positive)
    xb, vb = bvp.finite_difference(lambda x: 0*x, lambda x: 0*x, lambda x: -w*x*(Lb - x)/(2*EI),
                                   0, Lb, n, ("dirichlet", 0.0), ("dirichlet", 0.0))
    print(f"n = {n:>2}: midspan deflection {vb[n//2]*1000:.5f} mm (exact {exact_mid*1000:.5f} mm)")"""),
         md(r"""
The midspan deflection converges to the handbook value $5wL^4/(384EI)$ with errors 0.064, 0.016, 0.004 and
0.001 mm: each halving of the spacing divides the error by exactly 4, the signature of second-order
accuracy. It is *exactly* 4 here because the exact deflection is a quartic, whose fourth derivative - the
quantity in the leading error term $h^2 v''''/12$ - is constant. Ten intervals already give 0.8 % accuracy.
""")],
        [code(r"""k, h, Tb, Tinf = 200.0, 25.0, 100.0, 20.0
m = np.sqrt(4*h/(k*D)); Bi = h/(m*k)
lengths = np.linspace(0.005, 0.40, 200)
q = k*Ac*m*(Tb - Tinf)*(np.sinh(m*lengths) + Bi*np.cosh(m*lengths))/(np.cosh(m*lengths) + Bi*np.sinh(m*lengths))
area = np.pi*D*lengths + Ac
eff = q/(h*area*(Tb - Tinf))
q_inf = np.sqrt(h*np.pi*D*k*Ac)*(Tb - Tinf)            # infinitely long fin
L95 = lengths[np.argmax(q >= 0.95*q_inf)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.plot(lengths*1000, eff); a1.set(xlabel="fin length (mm)", ylabel="fin efficiency")
a2.plot(lengths*1000, q); a2.axhline(q_inf, color="k", ls="--", label="infinitely long fin")
a2.set(xlabel="fin length (mm)", ylabel="heat rate (W)"); a2.legend(); plt.show()
print(f"50 mm fin: efficiency {np.interp(0.05, lengths, eff):.2f}; 95 % of the maximum heat rate at L = {L95*1000:.0f} mm (mL = {m*L95:.2f})")"""),
         md(r"""
Efficiency falls steadily with length, because the far parts of a long fin are barely warmer than the air.
The heat rate levels off: beyond $mL \approx 1.8$ (here about 180 mm) the fin already delivers 95 % of what
an infinitely long fin could, so extra length adds material, weight and cost for almost nothing. The
50 mm fin, at $mL$ = 0.5, is short and efficient.
""")],
        [code(r"""def fd_dense(r, a, b, n, ya, yb):
    x = np.linspace(a, b, n + 1); hx = x[1] - x[0]
    A = np.zeros((n + 1, n + 1)); rhs = r(x) * hx**2
    A[0, 0] = A[-1, -1] = 1.0; rhs[0], rhs[-1] = ya, yb
    for i in range(1, n):
        A[i, i-1], A[i, i], A[i, i+1] = 1.0, -2.0, 1.0
    return x, np.linalg.solve(A, rhs)

r = lambda x: -np.pi**2*np.sin(np.pi*x)                 # exact y = sin(pi x)
for n in (500, 1000, 2000):
    t0 = time.perf_counter(); xd, yd = fd_dense(r, 0, 1, n, 0, 0); t_dense = time.perf_counter() - t0
    t0 = time.perf_counter(); xt, yt = bvp.finite_difference(lambda x: 0*x, lambda x: 0*x, r, 0, 1, n,
                                                             ("dirichlet", 0.0), ("dirichlet", 0.0)); t_thomas = time.perf_counter() - t0
    print(f"n = {n}: dense {t_dense*1000:7.1f} ms, Thomas {t_thomas*1000:5.2f} ms, same answer: {np.allclose(yd, yt)}")"""),
         md(r"""
Both give the same answer, but the dense solver's cost grows much faster: here about 4-5 times per
doubling of $n$, heading towards the factor 8 of its $n^3$ operation count as $n$ grows (at these sizes
fixed overheads still matter), and its memory grows like $n^2$. The Thomas algorithm grows like $n$ -
roughly doubling. For 1D
problems the tridiagonal structure is free performance; in 2D and 3D, sparse matrices (notebook 19) play
the same role.
""")],
    ]


# =====================================================================================================
@solutions("16_fourier_analysis", setup=r"""from engmath import datasets, fourier
data = np.loadtxt(datasets.path("pump_vibration.csv"), delimiter=",", skiprows=1)
t, signal = data[:, 0], data[:, 1]
fs = 1/(t[1] - t[0])
f_shaft = 1480/60""")
def _():
    return [
        [code(r"""short = signal[: int(0.25*fs)]
f, a = fourier.spectrum(short, fs)
print(f"record 0.25 s: frequency resolution {fs/short.size:.1f} Hz")
for target in (f_shaft, 2*f_shaft, 137.3):
    i = np.argmin(abs(f - target))
    print(f"  near {target:6.2f} Hz: amplitude {a[i]:.3f} mm/s")
print(f"  median amplitude 60-250 Hz (noise level): {np.median(a[(f > 60) & (f < 250)]):.3f} mm/s")
plt.plot(f, a); plt.xlim(0, 250); plt.xlabel("frequency (Hz)"); plt.ylabel("amplitude (mm/s)"); plt.show()"""),
         md(r"""
The resolution is now 4 Hz. The shaft peak (24.7 Hz) and its harmonic (49.3 Hz) are about 25 Hz apart, so
they remain clearly separated. What suffers is the weak defect component: with a shorter record the
noise level in the spectrum rises (0.042 mm/s instead of 0.018 mm/s with the full 2 s), so the defect
peak stands out only about 4.5 times above the noise instead of 8 times - and its height is less
accurate. Longer records improve both resolution and sensitivity.
""")],
        [code(r"""fs_t = 1000.0
def amp_spectrum(x, n_fft):
    # Hann window over the RECORDED samples, then zero-pad to n_fft points
    w = 0.5 - 0.5*np.cos(2*np.pi*np.arange(x.size)/x.size)
    X = np.fft.rfft(x*w, n=n_fft)
    return np.fft.rfftfreq(n_fft, 1/fs_t), 2*np.abs(X)/w.sum()
two_tones = lambda T: np.sin(2*np.pi*50*np.arange(int(T*fs_t))/fs_t) + np.sin(2*np.pi*52*np.arange(int(T*fs_t))/fs_t)
f1, a1 = amp_spectrum(two_tones(0.25), 250)          # 0.25 s record as it is
f2, a2 = amp_spectrum(two_tones(0.25), 2000)         # same record, zero-padded to 2 s
f3, a3 = amp_spectrum(two_tones(2.0), 2000)          # a genuine 2 s record
plt.plot(f1, a1, "o-", ms=4, label="0.25 s record")
plt.plot(f2, a2, label="0.25 s record zero-padded to 2 s")
plt.plot(f3, a3, label="genuine 2 s record")
plt.xlim(40, 62); plt.xlabel("frequency (Hz)"); plt.ylabel("amplitude"); plt.legend(fontsize=8); plt.show()"""),
         md(r"""
Zero-padding interpolates the same spectrum on a finer frequency grid: the curve becomes smooth, but the
two tones 2 Hz apart still appear as a **single** broad peak, because the 0.25 s record contains no
information to separate them. Only the genuinely longer record resolves them. (Apply the window to the
recorded samples *before* padding - windowing the padded array would suppress the data themselves.)
Resolution comes from measurement time, not from computation.
""")],
        [code(r"""N = signal.size
X = np.fft.rfft(signal)
mean_square = (np.abs(X[0])**2 + 2*np.sum(np.abs(X[1:-1])**2) + np.abs(X[-1])**2) / N**2
print(f"RMS from the spectrum (Parseval): {np.sqrt(mean_square):.4f} mm/s")
print(f"RMS in the time domain:           {np.sqrt(np.mean(signal**2)):.4f} mm/s")
print(f"sine components only: sqrt(sum A²/2) = {np.sqrt((1.0**2 + 0.4**2 + 0.15**2)/2):.4f} mm/s; noise alone 0.5 mm/s")"""),
         md(r"""
Parseval's theorem: the energy (mean square) is the same in time and in frequency. The overall RMS
combines the sinusoids ($A^2/2$ each) and the noise variance ($0.5^2$). Vibration-severity standards
(e.g. ISO 10816/20816) grade machines by this broadband RMS velocity - but, as here, a harmful defect can
hide under a healthy-looking RMS value. The spectrum shows *what* vibrates.
""")],
        [code(r"""kernel = np.zeros(1024); kernel[:5] = 1/5
H = np.abs(np.fft.rfft(kernel))
f_norm = np.fft.rfftfreq(1024)                        # in units of the sampling frequency
plt.plot(f_norm, H); plt.xlabel("frequency / sampling frequency"); plt.ylabel("|gain|")
plt.title("5-point moving average as a filter"); plt.show()
zeros = f_norm[1:][np.where((H[1:-1] < H[:-2]) & (H[1:-1] < H[2:]))[0] + 1][:2]
side = H[(f_norm > 0.2) & (f_norm < 0.4)].max()
print(f"zeros of the response near {np.round(zeros, 3)} fs (theory: 0.2 and 0.4); largest sidelobe gain {side:.2f}")"""),
         md(r"""
The moving average suppresses frequencies that fit a whole number of cycles into its 5-point window
(0.2 and 0.4 of the sampling frequency), but between those zeros high frequencies pass with gains up to
about 0.25 (the sidelobes), and the pass band droops gradually. A good low-pass filter should pass low
frequencies flat and block high ones strongly - the moving average does neither well, which is why
Savitzky-Golay or properly designed filters are preferred.
""")],
    ]


# =====================================================================================================
@solutions("17_optimisation_constraints_lp", setup=r"""from engmath import optimize
from scipy.optimize import linprog, milp, LinearConstraint, Bounds, minimize
c = [400, 300]
A = [[2, 1], [1, 1], [1, 0]]
b = [100, 80, 40]
base = optimize.simplex_lp(c, A, b)""")
def _():
    return [
        [code(r"""print(f"shadow price of assembly: EUR {base.shadow_prices[1]:.0f} per hour; cost EUR 150 per hour")
for extra in (0, 5, 10, 20, 25):
    lp = optimize.simplex_lp(c, A, [100, 80 + extra, 40])
    print(f"+{extra:>2} h: plan {np.round(lp.x, 1)}, profit EUR {lp.objective:,.0f} "
          f"(+{lp.objective - base.objective:,.0f}; minus overtime cost {lp.objective - base.objective - 150*extra:+,.0f})")"""),
         md(r"""
Each extra assembly hour earns EUR 200 and costs EUR 150, so buying all 10 hours adds EUR 500 net. The
shadow price is valid only while the same constraints stay binding: the plan shifts towards heavy-duty
pumps, and after 20 extra hours no standard pumps are left - beyond that, another hour is worth less.
Re-solving confirms the range in which the shadow price holds.
""")],
        [code(r"""A2 = np.array([[2.4, 1], [1, 1], [1, 0]]); b2 = [105, 80, 40]
lp = linprog([-400, -300], A_ub=A2, b_ub=b2, method="highs")
print(f"LP optimum: {np.round(lp.x, 3)}, profit EUR {-lp.fun:,.1f}")
for label, plan in (("rounded to nearest", np.round(lp.x)), ("rounded down", np.floor(lp.x))):
    feasible = np.all(A2 @ plan <= np.array(b2) + 1e-9)
    print(f"{label:<19}: {plan} feasible: {feasible}" + (f", profit EUR {400*plan[0] + 300*plan[1]:,.0f}" if feasible else
          f" (machining {A2[0] @ plan:.1f} h > 105 h)"))
ip = milp([-400, -300], constraints=LinearConstraint(A2, -np.inf, b2), integrality=[1, 1], bounds=Bounds(0, np.inf))
print(f"integer optimum (milp): {ip.x}, profit EUR {-ip.fun:,.0f}")"""),
         md(r"""
Rounding to the nearest integers **violates** the machining limit, and rounding down is feasible but gives
up EUR 300 of profit compared with the true integer optimum (17, 63), which neither rounding finds.
Integer programs need integer methods (branch and bound, as in `milp`), especially when quantities are
small; for large quantities rounding is usually harmless.
""")],
        [code(r"""import sympy as sp
r, h, lam, V = sp.symbols("r h lambda V", positive=True)
material = 2*(2*sp.pi*r**2) + 2*sp.pi*r*h            # caps twice as thick (wall thickness t factored out)
sol = sp.solve([sp.diff(material - lam*(sp.pi*r**2*h - V), v) for v in (r, h, lam)], [r, h, lam], dict=True)[0]
print("h/r =", sp.simplify(sol[h]/sol[r]))
num = minimize(lambda v: 4*np.pi*v[0]**2 + 2*np.pi*v[0]*v[1], [0.5, 1.0], method="SLSQP",
               bounds=[(1e-3, None)]*2, constraints=[{"type": "eq", "fun": lambda v: np.pi*v[0]**2*v[1] - 1}])
print(f"numerical check: r = {num.x[0]:.4f} m, h = {num.x[1]:.4f} m, h/r = {num.x[1]/num.x[0]:.3f}")"""),
         md(r"""
With end caps twice as thick, material in the caps is more expensive, so the optimal tank becomes taller
and slimmer: $h = 4r$ instead of $h = 2r$. In general, if the caps cost $c$ times as much per unit area as
the wall, $h/r = 2c$ - a result worth deriving once symbolically rather than re-optimising every case.
""")],
        [code(r"""area_r = lambda r_: 2*np.pi*r_**2 + 2/r_            # volume constraint eliminated: h = 1/(pi r^2)
g = lambda r_: 1/(np.pi*r_**2) - 1.0                # h <= 1  <=>  g(r) <= 0
r_cur = 1.0                                          # a strictly feasible start (h = 0.32 m)
for mu in 10.0**-np.arange(0, 9):
    barrier = lambda v, mu=mu: area_r(v[0]) - mu*np.log(-g(v[0])) if g(v[0]) < 0 else np.inf
    r_cur = optimize.nelder_mead(barrier, [r_cur], step=0.05, tol=1e-14).x[0]
    if mu in (1.0, 1e-3, 1e-8):
        print(f"mu = {mu:.0e}: r = {r_cur:.8f} m, h = {1/(np.pi*r_cur**2):.8f} m")
print(f"exact: r = {np.sqrt(1/np.pi):.8f} m, h = 1")"""),
         md(r"""
The barrier $-\mu\ln(-g)$ grows to infinity at the constraint boundary, so every iterate stays strictly
feasible (the tank is never taller than 1 m), and the optimum is approached from *inside* as $\mu \to 0$. The
penalty method approaches from *outside*, through infeasible designs. Interior-point methods built on
this idea are the workhorses of modern large-scale optimisation. Eliminating the equality constraint
first (using $h = 1/(\pi r^2)$) turned the problem into a one-variable one.
""")],
    ]


# =====================================================================================================
@solutions("18_monte_carlo", setup=r"""from engmath import montecarlo
from scipy import stats""")
def _():
    return [
        [code(r"""rng = np.random.default_rng(18)
ell, d = 1.0, 2.0
N = 1_000_000
centre = rng.uniform(0, d/2, N)                     # distance of the needle centre to the nearest line
angle = rng.uniform(0, np.pi/2, N)
p_hat = np.mean(centre <= ell/2*np.sin(angle))
pi_hat = 2*ell/(d*p_hat)
se = pi_hat*np.sqrt((1 - p_hat)/(p_hat*N))          # relative error of pi = relative error of p
print(f"pi ≈ {pi_hat:.4f} ± {se:.4f} from {N:,} needles")
N3 = (1 - p_hat)/p_hat * (np.pi/0.0005)**2
print(f"needles for ±0.0005 (three decimals, one standard error): about {N3:,.0f}")"""),
         md(r"""
Buffon's needle is a beautiful experiment but a terrible way to compute $\pi$: the error shrinks like
$1/\sqrt N$, so each extra correct digit costs 100 times more throws - three decimal places need tens of
millions of needles. (The simulation also uses $\pi$ to draw the random angles; a physical experiment
would not.)
""")],
        [code(r"""N = 100_000
rng = np.random.default_rng(1)
pf_exact = stats.norm.cdf(-(400 - 300)/np.hypot(25, 30))
L_plain = rng.normal(300, 30, N); S = rng.normal(400, 25, N)
fails = L_plain > S
p_plain, se_plain = fails.mean(), np.sqrt(fails.mean()*(1 - fails.mean())/N)
L_is = rng.normal(360, 30, N); S = rng.normal(400, 25, N)
w = stats.norm.pdf(L_is, 300, 30) / stats.norm.pdf(L_is, 360, 30)   # likelihood-ratio weights
vals = (L_is > S) * w
p_is, se_is = vals.mean(), vals.std(ddof=1)/np.sqrt(N)
print(f"exact {pf_exact:.5f}")
print(f"plain Monte Carlo:   {p_plain:.5f} ± {se_plain:.5f}")
print(f"importance sampling, load shifted: {p_is:.5f} ± {se_is:.5f}  ({se_plain/se_is:.1f}x smaller standard error)")
# shift BOTH variables towards the most likely failure point ("design point", about 359 MPa for both)
L2, S2 = rng.normal(360, 30, N), rng.normal(359, 25, N)
w2 = stats.norm.pdf(L2, 300, 30)/stats.norm.pdf(L2, 360, 30) * stats.norm.pdf(S2, 400, 25)/stats.norm.pdf(S2, 359, 25)
vals2 = (L2 > S2) * w2
se2 = vals2.std(ddof=1)/np.sqrt(N)
print(f"importance sampling, both shifted: {vals2.mean():.5f} ± {se2:.5f}  ({se_plain/se2:.1f}x smaller standard error, "
      f"about {(se_plain/se2)**2:.0f}x fewer samples for the same accuracy)")"""),
         md(r"""
Sampling the load from a distribution shifted towards failure produces many more failures to count; the
weights correct for sampling "the wrong" distribution, so the estimate stays unbiased. Shifting only the
load helps modestly (failures still need a low strength, which remains rare). Shifting **both** variables
towards the most likely failure point - where load and strength meet, near 359 MPa - cuts the standard
error about eightfold: some 70 times fewer samples for the same accuracy. Importance sampling is the standard tool for
the very small failure probabilities of structural reliability ($10^{-4}$ to $10^{-7}$), where plain Monte
Carlo would need billions of samples.
""")],
        [code(r"""walk = montecarlo.random_walk(900, 4000, dim=3, seed=7)
msd = np.mean(np.sum(walk**2, axis=2), axis=1)
steps = np.arange(msd.size)
slope = np.polyfit(steps, msd, 1)[0]
print(f"fitted <r²> per step: {slope:.3f}  (theory 1 for unit steps)")
plt.plot(steps, msd, label="3D walks (4000 walkers)"); plt.plot(steps, steps, "k--", label="<r²> = n")
plt.xlabel("steps n"); plt.ylabel("mean squared displacement"); plt.legend(); plt.show()
step_len, dt = 1e-10, 1e-12                          # molecular scale: 0.1 nm steps every picosecond
print(f"with {step_len*1e9:.1f} nm steps every {dt*1e12:.0f} ps: D = step²/(2 d dt) = {step_len**2/(2*3*dt):.2e} m²/s")"""),
         md(r"""
In three dimensions too, $\langle r^2\rangle = n\,\ell^2$ for steps of length $\ell$. After time $t = n\,\Delta t$, comparing with
the diffusion result $\langle r^2\rangle = 2dDt$ ($d$ = 3 dimensions) gives $D = \ell^2/(2d\,\Delta t)$. With molecular step
lengths and times this reproduces the order of magnitude of diffusion coefficients in liquids,
$10^{-9}$ to $10^{-10}$ m²/s: the macroscopic diffusion equation emerges from microscopic randomness.
""")],
        [code(r"""rng = np.random.default_rng(12)
N = 2_000_000
sums = rng.random((N, 12), dtype=np.float32).sum(axis=1)
p = np.mean(sums > 8)
print(f"Monte Carlo: P(sum > 8) = {p:.5f} ± {np.sqrt(p*(1 - p)/N):.5f}")
print(f"normal approximation (mean 6, variance 1): {stats.norm.sf(8, 6, 1):.5f}")
from math import comb, factorial                     # exact (Irwin-Hall distribution)
cdf8 = sum((-1)**k * comb(12, k) * (8 - k)**12 for k in range(9)) / factorial(12)
print(f"exact (Irwin-Hall): {1 - cdf8:.5f}")
cdf10 = sum((-1)**k * comb(12, k) * (10 - k)**12 for k in range(11)) / factorial(12)
print(f"further out, P(sum > 10): exact {1 - cdf10:.2e}, normal approximation {stats.norm.sf(10, 6, 1):.2e} "
      f"({stats.norm.sf(10, 6, 1)/(1 - cdf10):.1f}x too large)")"""),
         md(r"""
The sum of 12 uniforms has mean 6 and variance 1, and its distribution is close to normal - an old way of
generating normal random numbers. But in the **tails** the difference shows: the sum can never exceed 12,
and its tails are lighter than the normal distribution's. At a sum of 8 (two standard deviations) the
normal approximation is only about 2 % too high, but further out the error grows quickly: at 10 it
overestimates the probability almost fourfold. The central limit theorem describes the middle of a
distribution far better than its tails - exactly where failure probabilities live.
""")],
    ]


# =====================================================================================================
@solutions("19_sparse_2d_pdes", setup=r"""import time
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from engmath import sparse
Lx, Ly, thick, kb, P = 0.100, 0.060, 1.6e-3, 20.0, 2.0
q_vol = P/(0.02*0.02*thick)
def source(x, y):
    return np.where((np.abs(x - Lx/2) <= 0.01) & (np.abs(y - Ly/2) <= 0.01), q_vol/kb, 0.0)
def grid(nx, ny):
    X, Y = np.meshgrid(np.linspace(0, Lx, nx + 2)[1:-1], np.linspace(0, Ly, ny + 2)[1:-1])
    return source(X, Y).ravel()""")
def _():
    return [
        [code(r"""nx, ny = 199, 119
A = sparse.laplacian_2d(nx, ny, Lx/(nx + 1), Ly/(ny + 1))
f = grid(nx, ny)
theta_no = spla.spsolve(A.tocsc(), f)
h_conv = 10.0
A_conv = A + (2*h_conv/(kb*thick)) * sps.identity(nx*ny)
theta_conv = spla.spsolve(A_conv.tocsc(), f)
print(f"peak temperature: edges only {25 + theta_no.max():.1f} °C, with face convection {25 + theta_conv.max():.1f} °C")
print(f"heat leaving through the faces: {h_conv*2*np.sum(theta_conv)*(Lx/(nx+1))*(Ly/(ny+1)):.2f} W of {P} W")"""),
         md(r"""
Convection from both faces removes heat everywhere, not only at the clamped edges, and lowers the peak
temperature. The term simply adds a constant to the diagonal - the matrix stays sparse, symmetric and
positive definite, so the same solvers apply. The energy balance shows how much of the 2 W now leaves
through the faces instead of the frame.
""")],
        [code(r"""sizes, times = [], []
for n in (100, 200, 400):
    A = sparse.laplacian_2d(n, n, 1/(n + 1), 1/(n + 1)).tocsc()
    b = np.ones(n*n)
    t0 = time.perf_counter(); spla.spsolve(A, b); dt = time.perf_counter() - t0
    sizes.append(n*n); times.append(dt)
    print(f"{n}² = {n*n:>7,} unknowns: {dt:6.2f} s")
print(f"time grows like N^{np.polyfit(np.log(sizes), np.log(times), 1)[0]:.2f}")"""),
         md(r"""
The sparse direct solver's time grows faster than linearly in the number of unknowns $N$ (here about
$N^{1.3}$; theory for large 2D grids with a good ordering is about $N^{1.5}$), because factorisation creates
*fill-in*: new nonzeros
inside the band. It stays practical up to hundreds of thousands of 2D unknowns; for very large or 3D
problems, iterative methods with good preconditioners (conjugate gradient + multigrid) scale better.
An 800 × 800 grid (640,000 unknowns) is left to try on your own machine.
""")],
        [code(r"""rho_c = 1900*1100; alpha = kb/rho_c
nxt, nyt = 99, 59
At = sparse.laplacian_2d(nxt, nyt, Lx/(nxt + 1), Ly/(nyt + 1))
ft = grid(nxt, nyt)
theta_ss = spla.spsolve(At.tocsc(), ft).max()
def transient(dt, t_end=60.0):
    lu = spla.splu((sps.identity(nxt*nyt, format="csc") + dt*alpha*At).tocsc())
    theta = np.zeros(nxt*nyt); times, peak = [0.0], [0.0]
    for step in range(int(round(t_end/dt))):
        theta = lu.solve(theta + dt*alpha*ft)
        times.append((step + 1)*dt); peak.append(theta.max())
    times, peak = np.array(times), np.array(peak)
    t63 = np.interp(0.632*theta_ss, peak, times)        # interpolate between steps
    return t63, np.interp(10.0, times, peak)
ref = transient(0.05)
for dt in (2.0, 1.0, 0.5):
    t63, p10 = transient(dt)
    print(f"dt = {dt:3.1f} s: 63 % time {t63:6.2f} s (error {t63 - ref[0]:+.2f} s); rise after 10 s {p10:.3f} K (error {p10 - ref[1]:+.3f} K)")
print(f"reference (dt = 0.05 s): 63 % time {ref[0]:.2f} s")"""),
         md(r"""
Backward Euler is first order in time: halving the step roughly halves the error. The errors are also
systematic - backward Euler is overly damped, so the board appears to heat up *more slowly* than it
really does. Reading the 63 % time off the step grid adds a further error of up to one step, which is why
it is interpolated here.
""")],
        [code(r"""def pcg(A, b, M_inv_diag, tol=1e-10):
    x = np.zeros_like(b); r = b.copy(); z = M_inv_diag*r; p = z.copy(); rz = r @ z
    for it in range(1, 10*b.size):
        Ap = A @ p; a = rz/(p @ Ap); x += a*p; r -= a*Ap
        if np.linalg.norm(r)/np.linalg.norm(b) < tol:
            return x, it
        z = M_inv_diag*r; rz_new = r @ z; p = z + (rz_new/rz)*p; rz = rz_new
nx, ny = 299, 179
A = sparse.laplacian_2d(nx, ny, Lx/(nx + 1), Ly/(ny + 1))
b = grid(nx, ny)
_, hist = sparse.conjugate_gradient(A, b, tol=1e-10)
_, it_j = pcg(A, b, 1/A.diagonal())
print(f"plain CG: {len(hist) - 1} iterations; Jacobi-preconditioned CG: {it_j} iterations")
print(f"diagonal of A: min {A.diagonal().min():.4g}, max {A.diagonal().max():.4g}")"""),
         md(r"""
No saving at all: for this matrix every diagonal entry is the same, so the Jacobi preconditioner merely
rescales the problem by a constant and CG takes the same steps. Jacobi helps when the diagonal varies a
lot (e.g. strongly varying conductivity or cell sizes). For the Laplacian itself, effective
preconditioners must capture the coupling between nodes - incomplete Cholesky factorisation or multigrid.
""")],
    ]


# =====================================================================================================
@solutions("20_working_with_data_files", setup=r"""import pandas as pd
from engmath import datasets, fitting, fourier""")
def _():
    return [
        [code(r"""d = np.loadtxt(datasets.path("Hahn1.dat"), skiprows=60)
y, x = d[:, 0], d[:, 1]
def rational(x, b1, b2, b3, b4, b5, b6, b7):
    return (b1 + b2*x + b3*x**2 + b4*x**3) / (1 + b5*x + b6*x**2 + b7*x**3)
start1 = [10, -1, 0.05, -1e-5, -0.05, 0.001, -1e-6]              # NIST start 1 (file header)
fit = fitting.levenberg_marquardt(rational, x, y, start1)
cert = np.array([1.0776351733E+00, -1.2269296921E-01, 4.0863750610E-03, -1.4262662514E-06,
                 -5.7609940901E-03, 2.4053735503E-04, -1.2314450199E-07])   # certified (file header)
print("max relative difference from the certified values:", f"{np.max(np.abs(fit.coef/cert - 1)):.1e}")
print(f"residual SD {fit.sigma:.6f} (certified 8.1803852243E-02)")
xx = np.linspace(x.min(), x.max(), 400)
plt.plot(x, y, ".", ms=4, label="NIST measurements"); plt.plot(xx, rational(xx, *fit.coef), label="fitted rational model")
plt.xlabel("temperature (K)"); plt.ylabel("coefficient of thermal expansion"); plt.legend(); plt.show()"""),
         md(r"""
Copper's expansion coefficient rises steeply from near zero at low temperature and levels off above room
temperature - the shape the rational function captures. Starting from NIST's first starting values,
Levenberg-Marquardt reproduces the certified parameters of this "average difficulty" problem. With
seven highly correlated parameters, success depends on reasonable starting values - a good reason why
NIST publishes them.
""")],
        [code(r"""e = np.loadtxt(datasets.path("ENSO.dat"), skiprows=60)
pressure = e[:, 0] - e[:, 0].mean()
f_h, a_h = fourier.spectrum(pressure, fs=1.0)                        # Hann window; 1 sample per month
X = np.fft.rfft(pressure, n=4096)                                   # rectangular window, zero-padded:
f_z, a_z = np.fft.rfftfreq(4096), 2*np.abs(X)/pressure.size         # the spectrum on a much finer grid
def peaks(f, a, longest=80, shortest=10):
    idx = [i for i in range(1, len(a) - 1) if a[i] > a[i-1] and a[i] > a[i+1] and 1/longest < f[i] < 1/shortest]
    return sorted(idx, key=lambda i: -a[i])[:3]
for name, f, a in (("Hann window", f_h, a_h), ("rectangular, zero-padded", f_z, a_z)):
    print(f"{name:<25}: strongest peaks at periods {[round(float(1/f[i]), 1) for i in peaks(f, a)]} months")
print("certified model: 12 months, b4 = 44.3 months, b7 = 26.9 months;  frequency resolution 1/168 per month")
plt.plot(f_h, a_h, "o-", ms=3, label="Hann window")
plt.plot(f_z, a_z, label="rectangular window, zero-padded")
for period in (44.3, 26.9, 12.0):
    plt.axvline(1/period, color="k", ls=":", lw=1)
plt.xlim(0, 0.15); plt.xlabel("frequency (cycles per month)"); plt.ylabel("amplitude"); plt.legend(); plt.show()"""),
         md(r"""
The annual cycle (12 months) is the strongest feature. The two El Niño cycles, which NIST's certified model
places at 44.3 and 26.9 months, are only a few frequency bins apart in a 168-month record. With the
**Hann window** - whose wider main lobe suppresses leakage but blurs detail - the 27-month cycle merges into
a shoulder of the 44-month peak and does not appear as a separate maximum. With the **rectangular window**
(sharper main lobe, more leakage) and zero-padding to locate the maxima between bins, both cycles appear
as separate peaks close to the certified periods. Leakage versus resolution is a real trade-off, and on a
short record it decides what you can see; spurious small peaks from noise also appear, so interpret a
spectrum together with physical knowledge. The spectrum finds the cycles and provides starting values; the
nonlinear fit (as in the validation of this dataset) then determines them precisely.
""")],
        [code(r"""cols = ["date", "decimal_year", "co2_ppm", "deseasonalised_ppm", "days", "sd_days", "unc_mean"]
co2 = pd.read_csv(datasets.path("co2_mauna_loa_monthly.csv"), header=0, names=cols)
t, c = co2.decimal_year.to_numpy(), co2.co2_ppm.to_numpy()
def design(t):
    tau = (t - 1990)/10
    return np.column_stack([np.ones_like(t), tau, tau**2] + [f(2*np.pi*k*t) for k in (1, 2) for f in (np.sin, np.cos)])
def trend_450(mask):
    coef = fitting.linear_lstsq(design(t[mask]), c[mask]).coef
    years = np.linspace(1958, 2100, 14201)
    trend = design(years)[:, :3] @ coef[:3]
    return years[np.argmax(trend >= 450)], design(np.array([2026.0]))[:, :3] @ coef[:3], coef
y_all, _, _ = trend_450(np.ones_like(t, bool))
y_2000, pred_2026, _ = trend_450(t < 2000)
actual_2026 = co2.deseasonalised_ppm[co2.decimal_year > 2026].mean()
print(f"trend fitted to all data reaches 450 ppm in {y_all:.1f}")
print(f"trend fitted to 1958-1999 reaches 450 ppm in {y_2000:.1f}")
print(f"forecast for 2026 from the pre-2000 fit: {pred_2026[0]:.1f} ppm; observed (de-seasonalised, 2026): {actual_2026:.1f} ppm")"""),
         md(r"""
The fit to all data puts the 450 ppm crossing in the mid-2030s. The forecast made from data up to 2000
undershoots the 2026 level by almost 3 ppm, because the growth rate kept increasing faster than the
pre-2000 quadratic implied. Extrapolated trends carry the assumptions of their functional form
into the future, and their error grows with the extrapolation distance: treat them as scenarios, not
predictions - and state the uncertainty.
""")],
        [code(r"""def read_nist(name):
    lines = open(datasets.path(name), encoding="utf-8").read().splitlines()
    # "Data:" occurs twice: in the description block and as the marker of the data section - take the LAST
    start = max(i for i, line in enumerate(lines) if line.strip().startswith("Data:"))
    data = np.array([[float(v) for v in line.split()] for line in lines[start + 1:] if line.strip()])
    cert, sd = [], []
    for line in lines[:start]:
        parts = line.split()
        if len(parts) >= 6 and parts[0].startswith("b") and parts[1] == "=":
            cert.append(float(parts[4])); sd.append(float(parts[5]))
    return data, np.array(cert), np.array(sd)

for name in ("Chwirut2.dat", "Hahn1.dat", "ENSO.dat"):
    data, cert, sd = read_nist(name)
    same = np.array_equal(data, np.loadtxt(datasets.path(name), skiprows=60))
    print(f"{name:<13}: data {data.shape}, {len(cert)} certified parameters, identical to loadtxt: {same}")"""),
         md(r"""
The parser finds the data by its `Data:` marker and the certified values by their `b1 = ...` pattern,
instead of trusting a hard-coded line number, so it works for every file in the NIST collection and keeps
working if a header changes length. One trap: the word `Data:` also appears in the description block
("Data: 1 Response ..."), so a first version that took the *first* match failed - the marker of the data
section is the **last** match. Robust file readers look for structure, not for positions, and are tested
on every file they must read.
""")],
    ]


if __name__ == "__main__":
    write_all(only=[a for a in sys.argv[1:] if not a.startswith("--")])
