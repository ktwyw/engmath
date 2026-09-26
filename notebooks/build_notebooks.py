"""Generate the course notebooks from this script (single source of truth), then execute them.

    python notebooks/build_notebooks.py            # write + execute all notebooks
    python notebooks/build_notebooks.py --no-run   # write only
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
NOTEBOOKS: dict[str, list] = {}

SETUP = """try:                      # on Google Colab (or anywhere engmath is missing): install it from GitHub
    import engmath
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/ktwyw/engmath"], check=True)
import numpy as np
import matplotlib.pyplot as plt
import engmath
plt.rcParams.update({"figure.figsize": (7, 4), "figure.dpi": 90, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

import inspect
from IPython.display import Code

def show_source(obj):
    \"\"\"Display the source code of a library function or class.\"\"\"
    return Code(inspect.getsource(obj), language="python")"""


def md(text):
    return new_markdown_cell(text.strip("\n"))


def code(text):
    return new_code_cell(text.strip("\n"))


def notebook(name):
    def deco(fn):
        NOTEBOOKS[name] = fn()
        return fn
    return deco


# =====================================================================================================
@notebook("00_python_numpy_primer")
def _():
    return [
        md("""
# 00 · Python and NumPy for engineering calculations

**Goal:** the handful of Python ideas used in every later notebook - arrays, vectorisation,
functions, and plots - through one engineering problem.

**Problem.** Laminar flow in a pipe of radius $R$ has the parabolic velocity profile
$$u(r) = u_{\\max}\\left(1 - \\frac{r^2}{R^2}\\right).$$
What is the volumetric flow rate $Q = \\int_0^R u(r)\\,2\\pi r\\,dr$, and how does it compare with the
exact result $Q = \\pi R^2 u_{\\max}/2$?
"""),
        code(SETUP),
        md("## Arrays instead of loops\nA NumPy array holds many numbers; arithmetic applies element by element."),
        code("""R, u_max = 0.01, 0.5            # m, m/s
r = np.linspace(0, R, 11)        # 11 radial positions
u = u_max * (1 - r**2 / R**2)    # the whole profile in one line - no loop
print(np.round(u, 4))"""),
        md("""
## Vectorised code is also faster
The same calculation with a Python loop and with NumPy, on a million points:
"""),
        code("""import time
r_big = np.linspace(0, R, 1_000_000)

t0 = time.perf_counter()
u_loop = [u_max * (1 - ri**2 / R**2) for ri in r_big]
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
u_vec = u_max * (1 - r_big**2 / R**2)
t_vec = time.perf_counter() - t0
print(f"loop {t_loop*1e3:.1f} ms, vectorised {t_vec*1e3:.2f} ms  ->  {t_loop/t_vec:.0f}x faster")"""),
        md("## Functions make calculations reusable"),
        code("""def flow_rate(R, u_max, n=101):
    \"\"\"Flow rate by the trapezoidal rule on n radial points.\"\"\"
    r = np.linspace(0, R, n)
    integrand = u_max * (1 - r**2 / R**2) * 2 * np.pi * r
    return engmath.integrate.trapezoid(integrand, r)

exact = np.pi * R**2 * u_max / 2
for n in (5, 11, 101, 1001):
    Q = flow_rate(R, u_max, n)
    print(f"n = {n:>5}: Q = {Q*1e6:.6f} mL/s, relative error {abs(Q-exact)/exact:.1e}")"""),
        md("""
The error falls by about 100x for every 10x more points - the signature of a **second-order**
method (error $\\propto h^2$). Notebook 05 explains why.

## Plotting
"""),
        code("""r = np.linspace(-R, R, 200)
fig, ax = plt.subplots()
ax.plot(u_max * (1 - r**2 / R**2), r * 1000, lw=2)
ax.set(xlabel="velocity u (m/s)", ylabel="radial position (mm)", title="Laminar (Poiseuille) velocity profile")
plt.show()"""),
        md("""
## Exercises
1. Turbulent profiles are flatter: $u = u_{\\max}(1 - r/R)^{1/7}$. Compute $Q$ numerically and compare
   with the exact value $Q = \\tfrac{49}{60}\\pi R^2 u_{\\max}$. Why does the trapezoidal rule converge
   more slowly here? (Hint: look at the derivative at $r = R$.)
2. Write a function `reynolds(rho, u, D, mu)` that works for scalars *and* arrays, and use it to plot
   Re against velocity for water in a 25 mm pipe.
"""),
    ]


# =====================================================================================================
@notebook("01_errors_and_uncertainty")
def _():
    return [
        md("""
# 01 · Numbers, errors and reporting results

Every computed engineering result carries three kinds of error:

| Error | Source | Controlled by |
|---|---|---|
| **Round-off** | computers store ~16 significant digits | algorithm design |
| **Truncation** | approximating a limit, integral or series | step size / number of terms |
| **Measurement uncertainty** | the input data themselves | instruments, repetition |

This notebook shows all three, then how to *report* a result honestly.
"""),
        code(SETUP),
        md("## 1. Round-off: floating-point numbers are not real numbers"),
        code("""print(0.1 + 0.2 == 0.3, 0.1 + 0.2)
print(np.finfo(float).eps, "= machine epsilon (relative spacing of doubles)")"""),
        md("""
**Catastrophic cancellation.** The quadratic formula $x = (-b + \\sqrt{b^2 - 4ac})/2a$ subtracts two
nearly equal numbers when $b^2 \\gg 4ac$. The algebraically identical form
$x = 2c/(-b - \\sqrt{b^2-4ac})$ avoids the subtraction.
"""),
        code("""a, b, c = 1.0, 1e8, 1.0      # small root is -1e-8 (to 16 digits: -1.0000000000000000e-08)
naive = (-b + np.sqrt(b*b - 4*a*c)) / (2*a)
stable = 2*c / (-b - np.sqrt(b*b - 4*a*c))
print(f"naive  {naive:.16e}\\nstable {stable:.16e}")"""),
        md("""
The naive result is wrong already in its first digit - an error of 25 %, from a formula that is
algebraically exact. Rearranging formulas to avoid subtracting nearly
equal quantities is a core numerical skill.

## 2. Truncation versus round-off: the finite-difference step
A derivative approximated by $[f(x+h) - f(x)]/h$ has truncation error $\\propto h$ but round-off
error $\\propto \\varepsilon/h$. Too large a step is inaccurate; too small a step is *also* inaccurate.
"""),
        code("""h = np.logspace(-16, 0, 161)
x0 = 1.0
err_fwd = np.abs(engmath.differentiate.forward(np.exp, x0, h) - np.exp(x0))
err_cen = np.abs(engmath.differentiate.central(np.exp, x0, h) - np.exp(x0))
fig, ax = plt.subplots()
ax.loglog(h, err_fwd, label="forward difference, O(h)")
ax.loglog(h, err_cen, label="central difference, O(h²)")
ax.set(xlabel="step h", ylabel="error in d/dx eˣ at x = 1", title="The optimal step balances truncation and round-off")
ax.legend(); plt.show()
print(f"best forward step ~ {h[np.nanargmin(err_fwd)]:.0e} (theory ~ sqrt(eps) = {np.sqrt(np.finfo(float).eps):.0e})")
print(f"best central step ~ {h[np.nanargmin(err_cen)]:.0e} (theory ~ eps^(1/3) = {np.finfo(float).eps**(1/3):.0e})")"""),
        md("""
## 3. Measurement uncertainty and its propagation

**Problem.** The hydraulic efficiency of a pump is
$$\\eta = \\frac{\\rho g Q H}{P}$$
from measured flow $Q$, head $H$ and shaft power $P$. With independent inputs, first-order
propagation gives $u^2(\\eta) = \\sum_i (\\partial\\eta/\\partial x_i)^2 u^2(x_i)$.
"""),
        code("""def efficiency(Q, H, P, rho=998.0, g=9.81):
    return rho * g * Q * H / P

values = {"Q": 0.0125, "H": 32.0, "P": 5200.0}     # m3/s, m, W
unc    = {"Q": 0.0002, "H": 0.4,  "P": 60.0}       # standard uncertainties
eta, u_eta, share = engmath.reporting.propagate(efficiency, values, unc)
print(f"efficiency = {eta:.4f}, u = {u_eta:.4f}")
for k, s in share.items():
    print(f"  {k}: {100*s:4.1f} % of the variance")"""),
        md("""
The flow measurement dominates: even a perfect power meter would only reduce $u(\\eta)$ from 0.018 to
0.015, whereas halving the flow-meter uncertainty would do more. **Uncertainty budgets
tell you where to spend money.**

### Checking the linear approximation with Monte Carlo
Draw many random input sets, compute $\\eta$ for each, and look at the spread.
"""),
        code("""rng = np.random.default_rng(1)
N = 200_000
samples = efficiency(rng.normal(values["Q"], unc["Q"], N),
                     rng.normal(values["H"], unc["H"], N),
                     rng.normal(values["P"], unc["P"], N))
print(f"Monte Carlo: mean {samples.mean():.4f}, standard deviation {samples.std(ddof=1):.4f}")
plt.hist(samples, bins=100, density=True, alpha=0.7)
plt.xlabel("efficiency"); plt.title("Distribution of the computed efficiency"); plt.show()"""),
        md("""
Both methods agree because the relative uncertainties are small. When they are large, or the model
is strongly nonlinear, Monte Carlo is the safer choice (see also the companion library
[engstat](https://github.com/ktwyw/engstat) for GUM budgets with degrees of freedom).

## 4. Reporting: significant figures follow the uncertainty
Round the uncertainty to one significant figure (two if it starts with 1), then round the value to
the same decimal place.
"""),
        code("""from engmath.reporting import format_uncertainty
print("pump efficiency:", format_uncertainty(eta, u_eta))
for v, u in [(2.29987, 0.08907), (10523.4, 156.7), (0.0001234, 0.0000123), (51.66, 1.23)]:
    print(f"{v} ± {u}  ->  {format_uncertainty(v, u)}")"""),
        md("""
## Exercises
1. Compute $\\ln(1+x)$ for $x = 10^{-12}$ directly and with `np.log1p`. Explain the difference.
2. A heat exchanger duty is $\\dot{Q} = \\dot{m} c_p (T_{out} - T_{in})$ with $T_{in} = 20.0 \\pm 0.2$ °C and
   $T_{out} = 24.0 \\pm 0.2$ °C. Why is the relative uncertainty of $\\dot{Q}$ so large? Redesign the
   measurement to reduce it.
3. Use Monte Carlo to find the uncertainty of $\\eta$ if the power meter uncertainty were 20 %. Is the
   distribution still symmetric?
"""),
    ]


# =====================================================================================================
@notebook("02_root_finding")
def _():
    return [
        md("""
# 02 · Root finding: solving $f(x) = 0$

**Problem 1 - pipe friction.** For turbulent flow the Darcy friction factor $f$ satisfies the
implicit Colebrook equation
$$\\frac{1}{\\sqrt f} = -2\\log_{10}\\!\\left(\\frac{\\varepsilon/D}{3.7} + \\frac{2.51}{Re\\sqrt f}\\right),$$
which cannot be rearranged for $f$. We need it for a commercial-steel pipe ($\\varepsilon$ = 0.045 mm,
$D$ = 50 mm) at $Re = 2\\times10^5$.

**Problem 2 - particle settling.** How fast does a sand grain settle in water, when the drag
coefficient itself depends on the (unknown) velocity?
"""),
        code(SETUP + "\nfrom engmath import roots"),
        md("## Write the problem as g(f) = 0 and look at it first"),
        code("""eps, D, Re = 0.045e-3, 0.05, 2e5

def colebrook(f):
    return 1/np.sqrt(f) + 2*np.log10(eps/D/3.7 + 2.51/(Re*np.sqrt(f)))

f = np.linspace(0.005, 0.08, 300)
plt.plot(f, colebrook(f)); plt.axhline(0, color="k", lw=0.8)
plt.xlabel("friction factor f"); plt.ylabel("g(f)"); plt.title("One root, near f = 0.02"); plt.show()"""),
        md("""
## Four methods
| Method | Idea | Needs | Convergence |
|---|---|---|---|
| Bisection | halve a bracket | sign change | linear (1 bit/step) |
| Newton | follow the tangent | $f'$ and a good start | quadratic |
| Secant | tangent from two points | two starts | order 1.618 |
| Brent | bisection + interpolation | bracket | superlinear, guaranteed |
"""),
        code("""results = [roots.bisection(colebrook, 0.005, 0.08), roots.newton(colebrook, 0.02, tol=1e-15),
           roots.secant(colebrook, 0.01, 0.03, tol=1e-15), roots.brent(colebrook, 0.005, 0.08)]
for r in results:
    print(r)
from scipy.optimize import brentq
f_star = brentq(colebrook, 0.005, 0.08, xtol=1e-18)   # reference root, far more precise than the iterates"""),
        code("""fig, ax = plt.subplots()
for r in results:
    err = np.abs(np.array(r.history) - f_star)
    ax.semilogy(np.arange(1, err.size + 1), np.maximum(err, 1e-17), "o-", label=r.method)
ax.set(xlabel="iteration", ylabel="|error|", title="Convergence: Newton doubles the correct digits each step")
ax.legend(); plt.show()
for r in results[1:3]:
    print(f"{r.method:<8} observed order {roots.convergence_order(r.history, f_star):.2f}")
e = np.abs(np.array(results[0].history) - f_star)
print(f"bisection: error shrinks by a factor ~{np.median(e[1:25]/e[:24]):.2f} per step (linear convergence, ratio 1/2)")"""),
        md("""
**Newton's weakness:** it can wander off from a bad start. Try it on $\\arctan(x) = 0$ from $x_0 = 1.5$:
"""),
        code("""print(roots.newton(np.arctan, 1.3))    # converges
print(roots.newton(np.arctan, 1.5))    # diverges: the tangent overshoots further each step"""),
        md("""
In practice: **bracket first, then use Brent** (`scipy.optimize.brentq`), or Newton only with a
good starting guess.

## The Moody diagram from a root finder
Solving Colebrook in a loop over $Re$ and $\\varepsilon/D$ reproduces the turbulent part of the
Moody chart.
"""),
        code("""Re_range = np.logspace(3.6, 8, 120)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for rel in [0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
    ff = [roots.brent(lambda f: 1/np.sqrt(f) + 2*np.log10(rel/3.7 + 2.51/(Rei*np.sqrt(f))), 1e-4, 1).root
          for Rei in Re_range]
    ax.loglog(Re_range, ff, label=f"ε/D = {rel:g}")
ax.set(xlabel="Reynolds number", ylabel="Darcy friction factor", title="Moody diagram (turbulent region)")
ax.legend(fontsize=7, ncol=2); plt.show()"""),
        md("""
## Problem 2: settling velocity of a sand grain
Force balance at terminal velocity:
$$\\tfrac{1}{2}\\rho_f v^2 C_D(Re)\\,\\tfrac{\\pi d^2}{4} = (\\rho_p - \\rho_f)\\,g\\,\\tfrac{\\pi d^3}{6},
\\qquad C_D = \\frac{24}{Re}\\left(1 + 0.15 Re^{0.687}\\right)\\ (Re < 800,\\ \\text{Schiller-Naumann}),$$
with $Re = \\rho_f v d/\\mu$. The unknown $v$ appears inside $C_D$: another implicit equation.
"""),
        code("""rho_f, mu, rho_p, g = 998.0, 1.0e-3, 2650.0, 9.81

def force_balance(v, d):
    Re = rho_f * v * d / mu
    Cd = 24/Re * (1 + 0.15*Re**0.687)
    return 0.5*rho_f*v**2*Cd*np.pi*d**2/4 - (rho_p - rho_f)*g*np.pi*d**3/6

for d_mm in (0.05, 0.1, 0.2, 0.5):
    d = d_mm * 1e-3
    v = roots.brent(lambda v: force_balance(v, d), 1e-6, 1.0).root
    v_stokes = (rho_p - rho_f)*g*d**2/(18*mu)
    print(f"d = {d_mm:4.2f} mm: v = {v*1000:6.2f} mm/s (Re = {rho_f*v*d/mu:5.2f}); Stokes' law would give {v_stokes*1000:6.2f} mm/s")"""),
        md("""
Stokes' law ($Re \\ll 1$) is fine for fine silt but overpredicts badly for coarse sand, which is
exactly where the iterative solution is needed.

## Exercises
1. Solve the van der Waals equation $(P + a/V_m^2)(V_m - b) = RT$ for the molar volume of CO₂ at
   50 bar and 300 K ($a$ = 0.364 Pa·m⁶/mol², $b$ = 4.27×10⁻⁵ m³/mol). How many real roots are there
   at 280 K and 50 bar? Plot the cubic to find out.
2. Implement the fixed-point iteration $f_{k+1} = [-2\\log_{10}(\\ldots f_k\\ldots)]^{-2}$ for Colebrook.
   Estimate its convergence order and compare with Newton.
3. Add Haaland's explicit approximation to the Moody diagram and plot its relative error.
"""),
    ]


# =====================================================================================================
@notebook("03_linear_systems")
def _():
    return [
        md("""
# 03 · Linear systems and eigenvalues

Discretising almost any engineering field problem - heat conduction, structures, flow networks -
produces a linear system $A\\mathbf{x} = \\mathbf{b}$.

**Problem.** A 0.1 m thick wall generates heat uniformly ($\\dot q$ = 400 kW/m³, e.g. an electrically
heated element) with $k$ = 15 W/(m·K), faces held at 20 °C and 60 °C. Find the temperature profile
from the finite-difference form of $k\\,T'' + \\dot q = 0$:
$$T_{i-1} - 2T_i + T_{i+1} = -\\frac{\\dot q\\,\\Delta x^2}{k}.$$
"""),
        code(SETUP + "\nfrom engmath import linalg"),
        code("""L, k, q, T_left, T_right = 0.1, 15.0, 4e5, 20.0, 60.0
n = 49                                  # interior nodes
x = np.linspace(0, L, n + 2); dx = x[1] - x[0]
lower, diag, upper = np.ones(n), -2*np.ones(n), np.ones(n)
rhs = -q*dx**2/k * np.ones(n)
rhs[0] -= T_left; rhs[-1] -= T_right     # known boundary temperatures move to the right-hand side
T_inner = linalg.thomas(lower, diag, upper, rhs)
T = np.r_[T_left, T_inner, T_right]

T_exact = T_left + (T_right - T_left)*x/L + q/(2*k)*x*(L - x)
print(f"max |error| = {np.max(np.abs(T - T_exact)):.2e} K  (exact for a quadratic profile)")
plt.plot(x*100, T, "o", ms=3, label="finite differences"); plt.plot(x*100, T_exact, label="exact")
plt.xlabel("x (cm)"); plt.ylabel("T (°C)"); plt.legend(); plt.title("Wall with heat generation"); plt.show()
print(f"maximum temperature {T.max():.1f} °C at x = {x[T.argmax()]*100:.1f} cm")"""),
        md("""
## Why the Thomas algorithm matters
The matrix is **tridiagonal**. General Gaussian elimination costs $O(n^3)$ operations; the Thomas
algorithm exploits the structure and costs $O(n)$.
"""),
        code("""import time
for n in (100, 200, 400):
    A = np.diag(-2*np.ones(n)) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
    b = np.ones(n)
    t0 = time.perf_counter(); linalg.gauss_solve(A, b); t_g = time.perf_counter() - t0
    t0 = time.perf_counter(); linalg.thomas(np.ones(n), -2*np.ones(n), np.ones(n), b); t_t = time.perf_counter() - t0
    print(f"n = {n:>4}: Gauss {t_g*1e3:7.1f} ms   Thomas {t_t*1e3:5.2f} ms")"""),
        md("""
## 2D conduction by Gauss-Seidel iteration
For a square plate, direct solution gets expensive; iterative methods repeatedly replace each node
by the average of its neighbours (Laplace's equation) until nothing changes.
"""),
        code("""N = 31                                       # odd, so a true centre node exists
Tp = np.zeros((N, N)); Tp[0, :] = 100.0      # hot top edge, other edges at 0 °C
history, centre_early = [], None
for sweep in range(5000):
    old = Tp.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):
            Tp[i, j] = 0.25*(Tp[i+1, j] + Tp[i-1, j] + Tp[i, j+1] + Tp[i, j-1])
    change = np.max(np.abs(Tp - old)); history.append(change)
    if change < 1e-7:
        break
    if sweep == 199:
        centre_early = Tp[N//2, N//2]
print(f"after 200 sweeps the change per sweep is {history[199]:.1e} but the centre is only {centre_early:.2f} °C")
print(f"converged after {sweep + 1} sweeps; centre temperature {Tp[N//2, N//2]:.4f} °C (exact 25 °C by symmetry)")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
im = a1.imshow(Tp, cmap="inferno"); fig.colorbar(im, ax=a1, label="T (°C)"); a1.set_title("Plate temperature")
a2.semilogy(history); a2.set(xlabel="sweep", ylabel="max change", title="Slow convergence of Gauss-Seidel")
plt.show()"""),
        md("""
By symmetry (superpose the four rotations of the problem) the centre must be exactly 25 °C, even on
the discrete grid. Two lessons: Gauss-Seidel needs many sweeps, and **a small change per sweep is
not the same as a small error** - after 200 sweeps each sweep changes the solution only slightly, yet
the centre is still well below 25 °C. Faster alternatives (SOR, multigrid, sparse direct solvers)
exist for large problems.

## Eigenvalues: natural frequencies of a structure
Five equal masses $m$ connected by springs $k$ (ends fixed) vibrate as $M\\ddot{\\mathbf x} + K\\mathbf x = 0$.
Natural frequencies are $\\omega = \\sqrt{\\lambda}$ with $\\lambda$ the eigenvalues of $M^{-1}K$.
"""),
        code("""m, ks, nm = 2.0, 1.5e4, 5
K = ks*(2*np.eye(nm) - np.eye(nm, k=1) - np.eye(nm, k=-1))
A = K/m
lam_max, _ = linalg.power_iteration(A)                       # highest mode
lam_min, mode1 = linalg.power_iteration(np.linalg.inv(A))    # inverse iteration -> lowest mode
lam_min = 1/lam_min
all_lam = np.sort(np.linalg.eigvalsh(A))
print(f"lowest  frequency {np.sqrt(lam_min)/(2*np.pi):6.2f} Hz  (numpy: {np.sqrt(all_lam[0])/(2*np.pi):6.2f} Hz)")
print(f"highest frequency {np.sqrt(lam_max)/(2*np.pi):6.2f} Hz  (numpy: {np.sqrt(all_lam[-1])/(2*np.pi):6.2f} Hz)")
plt.plot(range(1, nm+1), mode1/np.max(np.abs(mode1)), "o-"); plt.xlabel("mass number"); plt.ylabel("relative amplitude")
plt.title("Fundamental mode shape"); plt.show()"""),
        md("""
## Exercises
1. Change the right boundary of the wall to convection, $-k\\,T' = h(T - T_\\infty)$, with
   $h$ = 25 W/(m²·K) and $T_\\infty$ = 20 °C. Only the last equation of the system changes - which one?
2. Solve the 2D plate with Jacobi iteration and compare the number of sweeps with Gauss-Seidel.
3. For the mass-spring chain, how do the frequencies change if the middle mass is doubled?
"""),
    ]


# =====================================================================================================
@notebook("04_interpolation")
def _():
    return [
        md("""
# 04 · Interpolation: reading between the lines of a table

**Problem.** Property tables list the vapour pressure of water every 10 °C. We need it at
35.4 °C, and at many temperatures for a simulation. The "true" values here come from the Antoine
equation for water, $\\log_{10} p[\\text{mmHg}] = 8.07131 - 1730.63/(233.426 + T[°C])$ (valid 1-100 °C),
so every interpolation error can be measured exactly.
"""),
        code(SETUP + "\nfrom engmath import interpolate"),
        code("""def p_antoine(T):          # kPa
    return 10**(8.07131 - 1730.63/(233.426 + T)) * 0.133322

T_tab = np.arange(10.0, 101.0, 10.0)
p_tab = p_antoine(T_tab)
for T, p in zip(T_tab, p_tab):
    print(f"{T:5.0f} °C  {p:8.3f} kPa")"""),
        md("## Linear vs cubic spline at 35.4 °C"),
        code("""T_q = 35.4
p_true = p_antoine(T_q)
p_lin = interpolate.linear(T_tab, p_tab, T_q)
p_spl = interpolate.CubicSpline(T_tab, p_tab)(T_q)
print(f"true   {p_true:.4f} kPa")
print(f"linear {p_lin:.4f} kPa  (error {100*(p_lin/p_true-1):+.2f} %)")
print(f"spline {p_spl:.4f} kPa  (error {100*(p_spl/p_true-1):+.2f} %)")"""),
        md("""
Linear interpolation always *overestimates* a convex curve like this one (the chord lies above the
curve). The cubic spline is far better.

## A better idea: interpolate a smoother quantity
Vapour pressure is nearly exponential in temperature, so $\\ln p$ is almost linear in $T$ (and even
more so in $1/T$ - the Clausius-Clapeyron equation). Interpolating the smoother quantity and
transforming back is a powerful trick.
"""),
        code("""T_fine = np.linspace(10, 100, 500)
p_fine = p_antoine(T_fine)
methods = {
    "linear in p": interpolate.linear(T_tab, p_tab, T_fine),
    "spline in p": interpolate.CubicSpline(T_tab, p_tab)(T_fine),
    "linear in ln p vs 1/T": np.exp(interpolate.linear(1/(T_tab[::-1]+273.15), np.log(p_tab[::-1]), 1/(T_fine+273.15))),
    "spline in ln p": np.exp(interpolate.CubicSpline(T_tab, np.log(p_tab))(T_fine)),
}
fig, ax = plt.subplots()
for name, p in methods.items():
    err = 100*np.abs(p/p_fine - 1)
    ax.semilogy(T_fine, np.maximum(err, 1e-6), label=f"{name} (max {err.max():.3f} %)")
ax.set(xlabel="T (°C)", ylabel="|relative error| (%)", title="Interpolation error across the table")
ax.legend(fontsize=8); plt.show()"""),
        md("""
Choosing *what* to interpolate matters as much as choosing the method. The spline of $\\ln p$ is about
six times more accurate than the spline of $p$ - and even *linear* interpolation of $\\ln p$ against
$1/T$, the form suggested by the Clausius-Clapeyron equation, beats both. A little physics beats a
higher polynomial degree.

## A warning: high-degree polynomials oscillate (Runge's phenomenon)
Passing one polynomial through many equally spaced points is tempting and wrong:
"""),
        code("""f = lambda x: 1/(1 + 25*x**2)
x_fine = np.linspace(-1, 1, 400)
fig, ax = plt.subplots()
ax.plot(x_fine, f(x_fine), "k", lw=2, label="function")
for n in (7, 13):
    xn = np.linspace(-1, 1, n)
    ax.plot(x_fine, interpolate.lagrange(xn, f(xn), x_fine), label=f"polynomial through {n} points")
xn = np.linspace(-1, 1, 13)
ax.plot(x_fine, interpolate.CubicSpline(xn, f(xn))(x_fine), "--", label="cubic spline, 13 points")
ax.set_ylim(-0.5, 1.5); ax.legend(fontsize=8); ax.set_title("More points make the polynomial WORSE"); plt.show()"""),
        md("""
## And a second warning: never extrapolate silently
`engmath.interpolate.linear` refuses by default:
"""),
        code("""try:
    interpolate.linear(T_tab, p_tab, 120.0)
except ValueError as e:
    print("ValueError:", e)"""),
        md("""
## Exercises
1. Repeat the error study with the table every 20 °C. How does the maximum error of each method
   scale with the spacing? (Theory: linear $\\propto h^2$, cubic spline $\\propto h^4$.)
2. Use a spline of $\\ln p$ against $T$ to find the boiling temperature at 50 kPa (combine with a
   root finder from notebook 02). Compare with the Antoine equation solved directly.
3. Replace the equally spaced points in the Runge example by Chebyshev points
   $x_j = \\cos(j\\pi/n)$. What happens to the polynomial?
"""),
    ]


# =====================================================================================================
@notebook("05_integration")
def _():
    return [
        md("""
# 05 · Numerical integration

**Problem.** A machine's power draw during an 8-hour shift is logged every 30 minutes. How much
energy did it use, and how much does the answer depend on the logging interval and the rule used?

To measure errors exactly we use a known smooth profile (start-up surge, production cycle,
lunch break):
$$P(t) = 40 + 25\\sin^2(\\pi t/8) + 30\\,e^{-(t-0.5)^2/0.08} - 20\\,e^{-(t-4)^2/0.1}\\quad\\text{kW}.$$
"""),
        code(SETUP + "\nfrom engmath import integrate\nfrom scipy.integrate import quad"),
        code("""def power(t):
    return 40 + 25*np.sin(np.pi*t/8)**2 + 30*np.exp(-(t-0.5)**2/0.08) - 20*np.exp(-(t-4)**2/0.1)

E_exact = quad(power, 0, 8, epsabs=1e-13, limit=200)[0]     # kWh, reference
t_log = np.arange(0, 8.01, 0.5)
P_log = power(t_log)
tt = np.linspace(0, 8, 800)
plt.plot(tt, power(tt), "k", lw=1, label="true profile")
plt.plot(t_log, P_log, "o", label="logged every 30 min")
plt.fill_between(t_log, P_log, alpha=0.15, step=None)
plt.xlabel("time (h)"); plt.ylabel("power (kW)"); plt.legend(); plt.show()
print(f"exact energy {E_exact:.3f} kWh")"""),
        code("""from engmath.interpolate import CubicSpline
rules = {"trapezoid": integrate.trapezoid(P_log, t_log),
         "Simpson":   integrate.simpson(P_log, t_log),
         "spline":    CubicSpline(t_log, P_log).integral()}
for k, v in rules.items():
    print(f"{k:<10} {v:8.3f} kWh   error {v - E_exact:+.3f} kWh ({100*(v/E_exact-1):+.2f} %)")"""),
        md("""
Surprise: Simpson's rule, "more accurate" in textbooks, is the *worst* here. Higher-order rules assume
the function is smooth on the scale of the sampling interval, but the start-up surge lasts only
about half an hour and is sampled by a single point - the parabolas then over- or undershoot. **The
sampling interval, not the integration rule, limits the accuracy here.** Integration rules cannot
recover information that was never measured; order-of-accuracy statements only hold once the
data resolve the signal.

## Order of accuracy
For a smooth integrand the error of a rule behaves as $C h^p$. On a log-log plot the slope is the
order $p$: trapezoid and midpoint 2, Simpson 4, while Gauss-Legendre converges faster than any
power.
"""),
        code("""ns = 2**np.arange(2, 11)
fig, ax = plt.subplots()
for rule in ("trapezoid", "midpoint", "simpson"):
    err = [abs(integrate.composite(power, 0, 8, n, rule) - E_exact) for n in ns]
    slope = np.polyfit(np.log(8/ns[-4:]), np.log(err[-4:]), 1)[0]
    ax.loglog(8/ns, err, "o-", label=f"{rule} (slope {slope:.2f})")
gl = [abs(integrate.gauss_legendre(power, 0, 8, n) - E_exact) for n in ns]
ax.loglog(8/ns, np.maximum(gl, 1e-15), "s-", label="Gauss-Legendre with n nodes")
ax.set(xlabel="step h (h)", ylabel="|error| (kWh)", title="Convergence of integration rules")
ax.legend(fontsize=8); plt.show()"""),
        md("""
**Romberg integration** extrapolates trapezoid results at $h, h/2, h/4, \\ldots$ to $h \\to 0$:
"""),
        code("""R = integrate.romberg(power, 0, 8, 8)
print("Romberg diagonal:", np.round(np.diag(R), 8))
print(f"error of best estimate: {abs(R[-1,-1] - E_exact):.1e} kWh")"""),
        md("""
## Running totals: cumulative integration
A flow meter reports the filling rate of a tank; the volume is the running integral.
"""),
        code("""t_min = np.array([0, 2, 5, 7, 10, 15, 20, 24, 30])          # uneven logging times
Q_Lmin = np.array([0, 35, 60, 72, 80, 82, 60, 30, 0])       # L/min
V = integrate.cumulative_trapezoid(Q_Lmin, t_min)
plt.plot(t_min, V, "o-"); plt.xlabel("time (min)"); plt.ylabel("volume (L)"); plt.title("Tank volume"); plt.show()
print(f"total volume {V[-1]:.0f} L")"""),
        md("""
## Exercises
1. How often must the power be logged for the trapezoidal energy estimate to be within 0.5 %?
2. Show that Simpson's rule on *equally* spaced points is exact for cubics, but not on unequal
   spacing. (Integrate $t^3$ on both kinds of grid.)
3. The logged power has random measurement noise of ±1 kW (standard deviation). Use Monte Carlo to
   find the resulting uncertainty in the energy. Is it larger or smaller than the sampling error?
"""),
    ]


# =====================================================================================================
@notebook("06_differentiation")
def _():
    return [
        md("""
# 06 · Differentiation of data: identifying a cooling law

**Problem.** A hot forging cools in still air. A data logger records its temperature, fast at first
and slower later (uneven time steps). Newton's law of cooling assumes
$dT/dt = -C\\,(T - T_\\infty)$, but natural convection suggests $h \\propto \\Delta T^{1/4}$, i.e.
$$\\frac{dT}{dt} = -C\\,(T - T_\\infty)^{n},\\qquad n = 1.25 .$$
Which is right? The **differential method**: estimate $dT/dt$ from the data, then fit
$\\ln(-dT/dt) = \\ln C + n\\ln(T - T_\\infty)$ - a straight line whose slope is $n$.
"""),
        code(SETUP + "\nfrom engmath import differentiate, fitting"),
        md("""
The data are generated from the exact solution of the $n = 1.25$ model,
$\\Delta T(t) = \\left[\\Delta T_0^{-1/4} + C t/4\\right]^{-4}$, so we know the right answer.
"""),
        code("""T_inf, dT0, C, n_true = 20.0, 480.0, 5.0e-4, 1.25
t = np.r_[np.arange(0, 60, 5), np.arange(60, 300, 20), np.arange(300, 1801, 100)].astype(float)   # s, uneven
dT = (dT0**-0.25 + C*t/4)**-4
T = T_inf + dT
plt.plot(t/60, T, "o"); plt.xlabel("time (min)"); plt.ylabel("T (°C)"); plt.title("Cooling curve (uneven sampling)")
plt.show()"""),
        md("""
## Derivatives on an uneven grid
Standard formulas assume equal spacing. `engmath.differentiate.gradient` uses 3-point Lagrange
formulas valid for **any** spacing (second-order accurate, exact for quadratics).
"""),
        code("""dTdt = differentiate.gradient(t, T)
exact = -C*dT**n_true
print(f"max relative error of the numerical derivative: {np.max(np.abs(dTdt/exact - 1))*100:.2f} %")
fit = fitting.linear_lstsq(np.column_stack([np.ones(t.size), np.log(dT)]), np.log(-dTdt))
lo, hi = fit.conf_int()[1]
print(f"estimated exponent n = {fit.coef[1]:.3f}  (95 % CI {lo:.3f} to {hi:.3f}; true 1.25)")
plt.plot(np.log(dT), np.log(-dTdt), "o"); plt.plot(np.log(dT), fit.coef[0] + fit.coef[1]*np.log(dT))
plt.xlabel("ln(T - T∞)"); plt.ylabel("ln(-dT/dt)"); plt.title("Differential method: slope = n"); plt.show()"""),
        md("""
Newton's law ($n = 1$) is clearly rejected for this data set.

## The catch: differentiation amplifies noise
Add a realistic ±0.5 °C thermocouple noise and repeat:
"""),
        code("""rng = np.random.default_rng(3)
T_noisy = T + rng.normal(0, 0.5, T.size)
d_noisy = differentiate.gradient(t, T_noisy)
ok = (d_noisy < 0) & (T_noisy > T_inf)             # logarithms need positive arguments
fit_n = fitting.linear_lstsq(np.column_stack([np.ones(ok.sum()), np.log(T_noisy[ok]-T_inf)]), np.log(-d_noisy[ok]))
rel = np.abs(d_noisy/exact - 1)
print(f"temperature noise: {100*0.5/np.median(T):.2f} % of T;  derivative errors: median {100*np.median(rel):.1f} %, max {100*rel.max():.0f} %")
print(f"noisy data: n = {fit_n.coef[1]:.3f} ± {fit_n.se[1]:.3f}   (noise-free: ± {fit.se[1]:.3f})")
plt.plot(t/60, -dTdt, "k-", label="exact data")
plt.plot(t/60, -d_noisy, "o", ms=4, label="0.5 °C noise")
plt.yscale("log"); plt.xlabel("time (min)"); plt.ylabel("-dT/dt (K/s)"); plt.legend(); plt.show()"""),
        md("""
Noise of a fraction of a percent in temperature becomes errors of tens of percent in individual
derivatives - worst at late times, where $T$ barely changes between readings. The estimate of $n$
survives here because many points average out, but its uncertainty grows about fifteen-fold, and
with fewer points or more noise the method fails outright. Two cures: **smooth before differentiating** (notebook 07) or **avoid
differentiation altogether** by fitting the integrated model directly (notebook 09).

## Exercises
1. Refit using only the first 10 minutes of data. How does the confidence interval of $n$ change?
2. Use `differentiate.forward` differences instead (first order). How large is the bias in $n$
   for the noise-free data?
3. Estimate $n$ from the noisy data by smoothing first with `engmath.smoothing.lowess`. Does the
   estimate improve?
"""),
    ]


# =====================================================================================================
@notebook("07_smoothing")
def _():
    return [
        md("""
# 07 · Smoothing noisy signals

**Problem.** A detector records two overlapping peaks (e.g. a chromatogram or a spectrum) with
noise. We want the peak positions and heights - which requires smoothing, and often derivatives.
Smoothing always trades **noise reduction** against **distortion** (peaks get lower and wider).
"""),
        code(SETUP + "\nfrom engmath import smoothing"),
        code("""x = np.linspace(0, 20, 401); dx = x[1] - x[0]
def peak(x, h, c, w): return h*np.exp(-0.5*((x - c)/w)**2)
true = peak(x, 1.0, 8.0, 0.8) + peak(x, 0.6, 10.5, 1.0) + 0.05
rng = np.random.default_rng(7)
y = true + rng.normal(0, 0.05, x.size)
plt.plot(x, y, ".", ms=2, color="gray", label="measured"); plt.plot(x, true, "k", label="true signal")
plt.xlabel("time (min)"); plt.ylabel("signal"); plt.legend(); plt.show()"""),
        code("""sm = {"moving average (15)": smoothing.moving_average(y, 15),
      "Savitzky-Golay (15, cubic)": smoothing.savitzky_golay(y, 7, 3),
      "LOWESS (frac 0.04)": smoothing.lowess(x, y, 0.04)}
fig, ax = plt.subplots()
ax.plot(x, true, "k", lw=2, label="true")
for k, v in sm.items():
    rms = np.sqrt(np.mean((v - true)**2))
    ax.plot(x, v, label=f"{k}: RMS error {rms:.4f}, peak {v.max():.3f}")
ax.set_xlim(5, 14); ax.legend(fontsize=8); ax.set_title(f"raw RMS error {np.sqrt(np.mean((y-true)**2)):.4f}, true peak {true.max():.3f}")
plt.show()"""),
        md("""
The moving average flattens the peak most: it fits a *constant* in each window. Savitzky-Golay fits
a *polynomial* in each window, which follows curvature and preserves the peak height much better - but
for the same window it removes less noise, so its overall RMS error is not the smallest. No filter wins
on every criterion: choose by what matters for your analysis (peak heights, positions or noise level).

## The window-size trade-off
"""),
        code("""halves = np.arange(2, 30)
rms_ma = [np.sqrt(np.mean((smoothing.moving_average(y, 2*m+1) - true)**2)) for m in halves]
rms_sg = [np.sqrt(np.mean((smoothing.savitzky_golay(y, m, 3) - true)**2)) for m in halves]
plt.plot(2*halves+1, rms_ma, "o-", label="moving average"); plt.plot(2*halves+1, rms_sg, "s-", label="Savitzky-Golay (cubic)")
plt.xlabel("window (points)"); plt.ylabel("RMS error vs true signal"); plt.legend()
plt.title("Too small: noise remains. Too large: peaks distort."); plt.show()"""),
        md("""
## Derivatives of smoothed data locate peaks
A maximum is where the first derivative crosses zero going downwards. Savitzky-Golay gives smoothed
derivatives directly (`deriv=1`).
"""),
        code("""d1 = smoothing.savitzky_golay(y, 10, 3, deriv=1, dx=dx)
d1_raw = np.gradient(y, dx)
def downward_crossings(d):
    return np.where((d[:-1] > 0) & (d[1:] <= 0))[0]
smooth = smoothing.savitzky_golay(y, 10, 3)
print(f"downward zero crossings: raw derivative {downward_crossings(d1_raw).size}, smoothed {downward_crossings(d1).size}")
real = [i for i in downward_crossings(d1) if smooth[i] > 0.3]      # keep only crossings on a real peak
peaks = [x[i] - d1[i]*(x[i+1]-x[i])/(d1[i+1]-d1[i]) for i in real]
print("peaks (signal above 0.3):", np.round(peaks, 2), "  - true centres 8.0 and 10.5")
fig, ax = plt.subplots()
ax.plot(x, d1_raw, color="gray", alpha=0.5, label="derivative of raw data")
ax.plot(x, d1, lw=2, label="Savitzky-Golay derivative"); ax.axhline(0, color="k", lw=0.8)
ax.set(xlim=(4, 16), ylim=(-2, 2), xlabel="time (min)", ylabel="dy/dx"); ax.legend(); plt.show()"""),
        md("""
Smoothing cuts the false zero crossings drastically, but the flat noisy baseline still produces some:
a **minimum peak height** is needed as well, which is what practical peak-picking software does.
The second peak's maximum appears below 10.5 because it sits on the tail of the first peak - the
maximum of a *sum* of peaks is not the centre of either (curve fitting, notebook 09, separates them).

## Exercises
1. Add a single spike (an electrical glitch) to the signal. Compare `lowess` with and without
   `robust_iters=2`.
2. Use the second derivative (`deriv=2`) to detect the shoulder of the smaller peak. Why is the
   second derivative even more sensitive to the window size?
3. Show that a Savitzky-Golay filter of order 2 with a 5-point window reproduces any quadratic
   exactly. Why does that make it suitable for smooth physical signals?
"""),
    ]


# =====================================================================================================
@notebook("08_linear_regression")
def _():
    return [
        md("""
# 08 · Linear least squares: calibrating a thermocouple

**Problem.** A thermocouple is calibrated against a reference thermometer. We need the relation
between its voltage (mV) and temperature, how good it is, and how to use it in reverse. The
calibration readings are in the file `thermocouple_calibration.csv` (synthetic data resembling a
type-K thermocouple), loaded with `np.loadtxt`.

"Linear" least squares means linear in the **coefficients**: $E = a_0 + a_1 T + a_2 T^2$ is a linear
model even though it is curved in $T$.
"""),
        code(SETUP + "\nfrom engmath import fitting"),
        code("""from engmath import datasets
cal = np.loadtxt(datasets.path("thermocouple_calibration.csv"), delimiter=",", skiprows=1)
T_ref, E = cal[:, 0], cal[:, 1]                       # reference temperature (°C), voltage (mV)
X1 = np.column_stack([np.ones_like(T_ref), T_ref])
X2 = np.column_stack([np.ones_like(T_ref), T_ref, T_ref**2])
lin, quad = fitting.linear_lstsq(X1, E), fitting.linear_lstsq(X2, E)
for name, f in (("linear", lin), ("quadratic", quad)):
    ci = f.conf_int()
    print(f"{name:<9} coefficients {np.array2string(f.coef, precision=6)}  residual SD {f.sigma*1000:.1f} µV")
print(f"quadratic term: {quad.coef[2]:.2e}, 95 % CI [{quad.conf_int()[2,0]:.2e}, {quad.conf_int()[2,1]:.2e}]")"""),
        md("""
The residual standard deviation drops sharply and the confidence interval of the quadratic term
excludes zero: the curvature is real. **Residual plots** show it directly:
"""),
        code("""fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, f, name in ((a1, lin, "linear"), (a2, quad, "quadratic")):
    ax.plot(T_ref, f.residuals*1000, "o-"); ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="T (°C)", ylabel="residual (µV)", title=f"{name} model")
plt.show()"""),
        md("""
A systematic arch in the residuals means the model is missing a term; random scatter means it
captures the physics.

## How the fit is computed - and why not the textbook formula
The normal equations $\\mathbf{b} = (X^TX)^{-1}X^T\\mathbf{y}$ square the condition number of $X$.
`engmath` uses a QR factorisation instead. With raw temperatures the columns $1, T, T^2$ differ by
five orders of magnitude:
"""),
        code("""for name, X in (("1, T, T²", X2), ("scaled: 1, t, t² with t = T/400", np.column_stack([np.ones(9), T_ref/400, (T_ref/400)**2]))):
    print(f"{name:<32} cond(X) = {np.linalg.cond(X):9.3g}   cond(XᵀX) = {np.linalg.cond(X.T @ X):9.3g}")"""),
        md("""
Each factor of 10 in the condition number can cost a digit of accuracy. Scaling or centring
predictors is cheap insurance.

## Using the calibration in reverse
In service we measure $E$ and need $T$: solve the fitted quadratic for $T$.
"""),
        code("""from engmath import roots
E_meas = 9.100
b0, b1, b2 = quad.coef
T_est = roots.brent(lambda T: b0 + b1*T + b2*T**2 - E_meas, 0, 450).root
print(f"E = {E_meas} mV  ->  T = {T_est:.2f} °C")"""),
        md("""
For the uncertainty of such inverse predictions (and prediction bands), see
`engstat.regression.Calibration` in the companion statistics library.

## Robust fitting: one bad reading
A loose connector produces one wrong reading at 400 °C. Ordinary least squares lets it pull the
whole line; Huber's robust regression down-weights it automatically.
"""),
        code("""E_bad = E.copy(); E_bad[-1] -= 0.8
ols = fitting.linear_lstsq(X2, E_bad)
rob = fitting.huber_irls(X2, E_bad)
TT = np.linspace(0, 400, 200); XX = np.column_stack([np.ones(200), TT, TT**2])
plt.plot(T_ref, E_bad, "o", label="data (last point faulty)")
plt.plot(TT, XX @ ols.coef, label="least squares"); plt.plot(TT, XX @ rob.coef, "--", label="Huber robust")
plt.xlabel("T (°C)"); plt.ylabel("E (mV)"); plt.legend(); plt.show()
print("robust weights:", np.round(rob.weights, 2))
print(f"error at 300 °C: OLS {abs(XX[150]@ols.coef - XX[150]@quad.coef)*1000:.0f} µV, robust {abs(XX[150]@rob.coef - XX[150]@quad.coef)*1000:.0f} µV")"""),
        md("""
Robust methods *flag* suspicious points (weight ≪ 1) - they do not excuse you from investigating
them.

## Exercises
1. Fit a cubic. Is the cubic term significant? What happens to the confidence intervals of the other
   coefficients, and why?
2. Refit the quadratic with only the points at or below 200 °C and use it to predict the voltage at
   400 °C. Compare with the full fit: the danger of extrapolation.
3. Linearise $k = A e^{-E_a/RT}$ as $\\ln k$ vs $1/T$ and fit it with `linear_lstsq`. Why do the
   weights of the points change when you transform the data? (Preview of notebook 09.)
"""),
    ]


# =====================================================================================================
@notebook("09_nonlinear_regression")
def _():
    return [
        md("""
# 09 · Nonlinear regression: fitting the Antoine equation

**Problem.** Vapour-pressure measurements of a solvent are to be fitted with the Antoine equation
$$\\log_{10} p = A - \\frac{B}{C + T},$$
which is nonlinear in $C$. Data are generated from Antoine constants commonly tabulated for ethanol
($A$ = 8.20417, $B$ = 1642.89, $C$ = 230.300; $p$ in mmHg, $T$ in °C) with 1 % random error.
"""),
        code(SETUP + "\nfrom engmath import fitting\nfrom scipy.optimize import curve_fit"),
        code("""A_t, B_t, C_t = 8.20417, 1642.89, 230.300
T = np.linspace(20, 90, 12)
rng = np.random.default_rng(5)
p = 10**(A_t - B_t/(C_t + T)) * (1 + 0.01*rng.standard_normal(T.size))

def antoine_log(T, A, B, C):
    return A - B/(C + T)"""),
        md("""
## Fitting with Levenberg-Marquardt
`engmath.fitting.levenberg_marquardt` blends Gauss-Newton (fast near the optimum) with gradient
descent (safe far away). The errors are proportional to $p$, so we fit $\\log_{10} p$, which makes them
roughly equal in size (the correct weighting).
"""),
        code("""fit = fitting.levenberg_marquardt(antoine_log, T, np.log10(p), p0=[8.0, 1500.0, 220.0])
popt, pcov = curve_fit(antoine_log, T, np.log10(p), p0=[8.0, 1500.0, 220.0])
for name, v, s, ref in zip("ABC", fit.coef, fit.se, popt):
    print(f"{name} = {v:10.4f} ± {s:8.4f}   (scipy curve_fit {ref:10.4f})")
print(f"iterations {fit.iterations}, residual SD {fit.sigma:.2e} in log10 p")
plt.semilogy(fit.history, "o-"); plt.xlabel("accepted iteration"); plt.ylabel("residual sum of squares")
plt.title("Levenberg-Marquardt convergence"); plt.show()"""),
        md("""
## The parameters are nearly impossible to separate
The standard errors are large compared with the precision of the fit. The **correlation matrix**
explains why:
"""),
        code("""print(np.round(fit.correlation, 4))"""),
        md("""
Correlations above 0.99 mean that a change in $B$ can be almost perfectly compensated by changes in
$A$ and $C$: many parameter sets describe the data equally well. The *fitted curve* is precise, but
individual parameters are not - so never extrapolate far, and never compare parameters from
different papers one at a time.

This also explains why the fitted $A$, $B$ and $C$ are all about 1.5 standard errors *below* the values
used to generate the data. With correlations this close to 1 the three errors are not independent:
they move together along a valley of almost equally good fits, so one unlucky set of noise shifts all
three at once. Judge a nonlinear fit by the fitted curve and its residuals, not by any single parameter.

## Starting values matter
"""),
        code("""for p0 in ([8, 1500, 220], [5, 500, 100], [8, 1500, -20]):
    try:
        np.seterr(divide="ignore")                   # the last start divides by zero - on purpose
        r = fitting.levenberg_marquardt(antoine_log, T, np.log10(p), p0=p0)
        print(f"start {p0}: A, B, C = {np.round(r.coef, 3)}, RSS = {r.history[-1]:.2e}")
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"start {p0}: refused - {e}")
np.seterr(divide="warn")"""),
        md("""
A start with $C = -20$ puts the singularity $C + T = 0$ exactly on the first data point (20 °C): the
model is infinite there and the fit cannot even begin. Poor but finite starts (the second line) are
usually rescued by Levenberg-Marquardt's damping. Good starting values come from physics or from a simpler linearised fit - e.g. the
two-parameter Clausius-Clapeyron form $\\log_{10} p = a - b/(T + 273.15)$, which is *linear* in $a, b$.

## Is the third parameter worth it?
"""),
        code("""cc = fitting.linear_lstsq(np.column_stack([np.ones(T.size), -1/(T + 273.15)]), np.log10(p))
T_f = np.linspace(20, 90, 200)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
a1.semilogy(T, p, "o", label="data"); a1.semilogy(T_f, 10**antoine_log(T_f, *fit.coef), label="Antoine (3 par.)")
a1.semilogy(T_f, 10**(cc.coef[0] - cc.coef[1]/(T_f + 273.15)), "--", label="Clausius-Clapeyron (2 par.)")
a1.set(xlabel="T (°C)", ylabel="p (mmHg)"); a1.legend(fontsize=8)
a2.plot(T, 100*fit.residuals*np.log(10), "o-", label=f"Antoine, SD {fit.sigma:.1e}")
a2.plot(T, 100*cc.residuals*np.log(10), "s-", label=f"Clausius-Clapeyron, SD {cc.sigma:.1e}")
a2.axhline(0, color="k", lw=0.8); a2.set(xlabel="T (°C)", ylabel="residual (%)"); a2.legend(fontsize=8)
plt.show()"""),
        md("""
The two-parameter model leaves a systematic pattern in the residuals: the extra parameter is
justified by the physics (temperature-dependent heat of vaporisation), not just by a smaller error.

## Exercises
1. Fit $p$ directly (not $\\log p$) with equal weights. Which data points dominate the fit, and how do
   the parameters change?
2. Fix $C$ at 230.3 and fit only $A$ and $B$ - a linear problem. Compare the standard errors with the
   three-parameter fit.
3. Fit the cooling data of notebook 06 by integrating the model instead of differentiating the data:
   fit $\\Delta T(t) = [\\Delta T_0^{-1/4} + Ct/4]^{-4}$ (generalised to exponent $n$) to the *noisy*
   temperatures. Compare the precision of $n$ with the differential method.
"""),
    ]


# =====================================================================================================
@notebook("10_implicit_models_optimisation")
def _():
    return [
        md("""
# 10 · Fitting implicit models: how rough is this pipe?

**Problem.** Pressure drops were measured over 20 m of a 50 mm pipe at several flow rates. The
pipe's roughness $\\varepsilon$ is unknown, and the flow meter may read a few percent low (unknown
factor $k$). The model is
$$\\Delta P = f\\,\\frac{L}{D}\\,\\frac{\\rho v^2}{2},\\qquad v = \\frac{4kQ}{\\pi D^2},$$
where $f$ solves the **implicit** Colebrook equation. Each model prediction therefore requires a
root-finding problem (notebook 02) *inside* the least-squares problem - a common situation with
equilibrium, phase and flow models. The measurements are in `pipe_pressure_drop.csv` (synthetic data
with 2 % measurement error, generated with known "true" values so that the fit can be checked).
"""),
        code(SETUP + "\nfrom engmath import optimize, roots"),
        code("""rho, mu, L, D = 998.0, 1.0e-3, 20.0, 0.05

def friction(Re, eps):
    g = lambda f: 1/np.sqrt(f) + 2*np.log10(eps/D/3.7 + 2.51/(Re*np.sqrt(f)))
    return roots.brent(g, 1e-4, 0.2).root

def dp_model(Q, eps, k=1.0):
    v = 4*k*Q/(np.pi*D**2)
    Re = rho*v*D/mu
    return np.array([friction(Ri, eps) for Ri in np.atleast_1d(Re)]) * L/D * rho*v**2/2

from engmath import datasets
meas = np.loadtxt(datasets.path("pipe_pressure_drop.csv"), delimiter=",", skiprows=1)
Q, dp = meas[:, 0], meas[:, 1]                                # m3/s, Pa
eps_true, k_true = 0.15e-3, 1.03          # used to synthesise the file (unknown in a real test) - for checking only
for q, d in zip(Q, dp):
    print(f"Q = {q*1000:4.1f} L/s   ΔP = {d/1000:8.2f} kPa")"""),
        md("""
## One parameter: golden-section search
Assume the flow meter is right ($k$ = 1) and search for the roughness that minimises the sum of
squared **relative** residuals. The pressure drops span two orders of magnitude, so absolute residuals
would let the highest flow dominate completely (compare exercise 1).
"""),
        code("""def rss(log_eps, k=1.0):
    return np.sum(np.log(dp_model(Q, 10**log_eps, k) / dp)**2)

r1 = optimize.golden_section(rss, -6, -2.5)
print(f"roughness (k fixed at 1): {10**r1.x[0]*1000:.3f} mm   (true {eps_true*1000} mm)")"""),
        md("""
Wrong! Forcing $k = 1$ makes the fit blame the low flow reading on the pipe. Both unknowns must be
estimated together.

## Two parameters: the Nelder-Mead simplex
With no derivatives available (each evaluation hides eight root solves), a derivative-free method
is natural. Optimising $\\log_{10}\\varepsilon$ rather than $\\varepsilon$ keeps the parameters on similar
scales and keeps $\\varepsilon$ positive.
"""),
        code("""r2 = optimize.nelder_mead(lambda p: rss(p[0], p[1]), [-4.0, 1.0], step=0.05, tol=1e-12)
eps_fit, k_fit = 10**r2.x[0], r2.x[1]
print(f"roughness {eps_fit*1000:.3f} mm (true {eps_true*1000}),  meter factor {k_fit:.4f} (true {k_true})")
print(f"{r2.iterations} simplex iterations, converged: {r2.converged}")
print(f"RSS at the fit {r2.fun:.5f}  vs  RSS at the TRUE parameters {rss(np.log10(eps_true), k_true):.5f}")"""),
        md("""
Much better than fixing $k = 1$, yet not equal to the truth - and the fit is **not** wrong: its RSS is
*lower* than at the true parameters. With 2 % noise these data genuinely cannot pin down both
parameters more precisely. Why?

## Looking at the objective function explains a lot
"""),
        code("""le = np.linspace(-4.5, -3.2, 60); kk = np.linspace(0.97, 1.09, 60)
Z = np.array([[rss(a, b) for a in le] for b in kk])
fig, ax = plt.subplots(figsize=(6.5, 4.5))
cs = ax.contourf(10**le*1000, kk, np.log10(Z), levels=25, cmap="viridis")
fig.colorbar(cs, label="log10 RSS")
ax.plot(eps_fit*1000, k_fit, "r*", ms=15, label="fit"); ax.plot(eps_true*1000, k_true, "wx", ms=10, mew=2, label="truth")
ax.set(xscale="log", xlabel="roughness ε (mm)", ylabel="flow-meter factor k", title="A long, curved valley")
ax.legend(); plt.show()"""),
        md("""
The valley is long and narrow: roughness and meter factor partly compensate each other, so they are
correlated. Better-designed experiments (e.g. adding laminar-flow points, where $f = 64/Re$ does not
depend on $\\varepsilon$ at all) would pin $k$ down independently - **experimental design and
numerical methods go together.**

## Exercises
1. Repeat the two-parameter fit with absolute residuals $\\sum(\\Delta P_{model} - \\Delta P)^2$. How much
   do the estimates change, and why?
2. Add three laminar points ($Re$ < 2000, use $f = 64/Re$) to the data and refit. How does the shape
   of the valley change?
3. Replace Colebrook with the explicit Haaland approximation inside the model. How large is the
   resulting bias in $\\varepsilon$, and is it smaller than the statistical uncertainty?
"""),
    ]


# =====================================================================================================
@notebook("11_odes")
def _():
    return [
        md("""
# 11 · Ordinary differential equations: a batch reactor

**Problem.** In a batch reactor the desired product B is formed from A and then degrades to C:
$$A \\xrightarrow{k_1} B \\xrightarrow{k_2} C,\\qquad
\\frac{dC_A}{dt} = -k_1 C_A,\\quad \\frac{dC_B}{dt} = k_1 C_A - k_2 C_B,\\quad \\frac{dC_C}{dt} = k_2 C_B .$$
When should the reaction be stopped to maximise B? The exact solution exists here, so every
numerical method can be checked.
"""),
        code(SETUP + "\nfrom engmath import odes, optimize"),
        code("""k1, k2, CA0 = 0.5, 0.2, 2.0          # 1/min, 1/min, mol/L
def rhs(t, c):
    return [-k1*c[0], k1*c[0] - k2*c[1], k2*c[1]]

def CB_exact(t):
    return CA0*k1/(k2 - k1)*(np.exp(-k1*t) - np.exp(-k2*t))

t, c = odes.rk4(rhs, [CA0, 0, 0], 0, 20, 200)
for i, name in enumerate("ABC"):
    plt.plot(t, c[:, i], label=name)
plt.plot(t[::10], CB_exact(t[::10]), "k.", label="B exact")
plt.xlabel("time (min)"); plt.ylabel("concentration (mol/L)"); plt.legend(); plt.show()
print(f"mass balance check: max |A+B+C - CA0| = {np.max(np.abs(c.sum(axis=1) - CA0)):.1e}")"""),
        md("## Optimal batch time: maximise $C_B(t)$"),
        code("""def neg_CB(t_end):
    n = max(10, int(t_end*20))
    return -odes.rk4(rhs, [CA0, 0, 0], 0, t_end, n)[1][-1, 1]

best = optimize.golden_section(neg_CB, 0.5, 10)
t_opt_exact = np.log(k1/k2)/(k1 - k2)
print(f"optimal time {best.x[0]:.4f} min (exact ln(k1/k2)/(k1-k2) = {t_opt_exact:.4f}), max C_B = {-best.fun:.4f} mol/L")"""),
        md("""
## Accuracy: Euler vs RK4
The global error of Euler's method is proportional to the step $h$; RK4's to $h^4$. Halving the step
halves Euler's error but reduces RK4's 16-fold.
"""),
        code("""ns = np.array([10, 20, 40, 80, 160, 320])
fig, ax = plt.subplots()
for name, solver in (("Euler", odes.euler), ("RK4", odes.rk4)):
    err = [abs(solver(rhs, [CA0, 0, 0], 0, 10, n)[1][-1, 1] - CB_exact(10)) for n in ns]
    slope = np.polyfit(np.log(10/ns), np.log(err), 1)[0]
    ax.loglog(10/ns, err, "o-", label=f"{name}: slope {slope:.2f}")
ax.set(xlabel="step h (min)", ylabel="|error in C_B(10)|"); ax.legend(); plt.show()"""),
        md("""
## Adaptive step size
`odes.rk23` estimates its own error at each step and adjusts $h$: small steps where the solution
changes fast, large where it is smooth.
"""),
        code("""ta, ca = odes.rk23(rhs, [CA0, 0, 0], 0, 20, rtol=1e-6)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
a1.plot(ta, ca[:, 1], "o-", ms=3); a1.set(xlabel="time (min)", ylabel="C_B", title=f"{ta.size - 1} adaptive steps")
a2.plot(ta[1:], np.diff(ta), "o-", ms=3); a2.set(xlabel="time (min)", ylabel="step size (min)", title="Step size grows as things settle")
plt.show()
print(f"error at t = 20 min: {abs(ca[-1, 1] - CB_exact(20)):.1e}")"""),
        md("""
## Stiffness: when explicit methods fail
If B is an unstable intermediate ($k_2$ = 1000 1/min), the fast time scale forces explicit methods
to take tiny steps **for stability, not accuracy** - even after B has reached its quasi-steady level.
Implicit methods such as backward Euler remain stable with large steps.
"""),
        code("""k2_fast = 1000.0
stiff = lambda t, c: [-k1*c[0], k1*c[0] - k2_fast*c[1]]
exact_B = lambda t: CA0*k1/(k2_fast - k1)*(np.exp(-k1*t) - np.exp(-k2_fast*t))
for n in (100, 1000, 10000):
    h = 10/n
    tb, cb = odes.backward_euler(stiff, [CA0, 0], 0, 10, n)
    with np.errstate(over="ignore", invalid="ignore"):     # the explicit run is EXPECTED to overflow
        te, ce = odes.euler(stiff, [CA0, 0], 0, 10, n)
    print(f"h = {h:7.4f} min (k2 h = {k2_fast*h:6.1f}):  explicit Euler C_B(10) = {ce[-1, 1]:11.3e}   "
          f"backward Euler = {cb[-1, 1]:.4e}   exact = {exact_B(10):.4e}")"""),
        md("""
Explicit Euler is stable only for $k_2 h < 2$. In practice use SciPy's `solve_ivp(method="BDF")` or
`"Radau"` for stiff systems - the ideas are exactly those shown here.

## Exercises
1. A tank drains through an orifice: $A\\,dh/dt = -C_d a\\sqrt{2gh}$. Solve with RK4 and compare with
   the exact draining time $t = (A/C_d a)\\sqrt{2h_0/g}$. Why does the error grow near $h = 0$?
2. For the series reaction, find the $k_1$ that maximises $C_B$ at a fixed batch time of 3 min
   (an optimisation wrapped around an ODE solver).
3. Estimate the largest stable step for RK4 on the stiff system by trial. Compare with the Euler
   limit $h < 2/k_2$.
"""),
    ]


# =====================================================================================================
@notebook("12_diffusion_pde")
def _():
    return [
        md("""
# 12 · Partial differential equations: heat penetration and diffusion

**Problem.** The surface of a thick steel block is suddenly raised from 20 °C to 320 °C. How does
heat penetrate, and can we measure the thermal diffusivity $\\alpha$ from a thermocouple buried
10 mm below the surface?

The temperature obeys the diffusion equation $\\partial T/\\partial t = \\alpha\\,\\partial^2 T/\\partial x^2$
(identical in form to mass diffusion with $D$). For a semi-infinite solid the exact solution is
$$\\frac{T - T_s}{T_0 - T_s} = \\operatorname{erf}\\!\\left(\\frac{x}{2\\sqrt{\\alpha t}}\\right).$$
"""),
        code(SETUP + "\nfrom engmath import pdes, optimize\nfrom scipy.special import erf"),
        code("""alpha, T0, Ts = 1.2e-5, 20.0, 320.0            # m2/s (typical carbon steel), °C
L, nx = 0.25, 501                               # >= 8 sqrt(alpha t) at 60 s: effectively semi-infinite
x = np.linspace(0, L, nx); dx = x[1] - x[0]
exact = lambda x, t: Ts + (T0 - Ts)*erf(x/(2*np.sqrt(alpha*t)))

fig, ax = plt.subplots()
for t_end in (5, 20, 60):
    dt = 0.4*dx**2/alpha
    T = pdes.ftcs(np.full(nx, T0), alpha, dx, dt, int(round(t_end/dt)), Ts, T0)
    ax.plot(x*1000, T, label=f"FTCS, t = {t_end} s")
    ax.plot(x[::8]*1000, exact(x[::8], t_end), "k.", ms=4)
ax.set(xlim=(0, 60), xlabel="depth (mm)", ylabel="T (°C)", title="Heat penetration (dots: exact erf solution)")
ax.legend(); plt.show()"""),
        md("""
## Stability of the explicit scheme
FTCS updates each node from its neighbours. It is stable only if $r = \\alpha\\Delta t/\\Delta x^2 \\le 1/2$.
Beyond that, errors grow exponentially. (`pdes.ftcs` refuses unstable steps, so the unstable run is
coded by hand here.)
"""),
        code("""def ftcs_unchecked(T, r, steps):
    T = T.copy()
    for _ in range(steps):
        T[1:-1] = T[1:-1] + r*(T[2:] - 2*T[1:-1] + T[:-2])
    return T

T_init = np.full(nx, T0); T_init[0] = Ts
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for r, col in ((0.45, "C0"), (0.52, "C1")):
    a1.plot(x*1000, ftcs_unchecked(T_init, r, 60), color=col, label=f"r = {r}, after 60 steps")
    overshoot = [np.max(ftcs_unchecked(T_init, r, k)) - Ts for k in range(1, 301, 10)]
    a2.semilogy(range(1, 301, 10), np.maximum(overshoot, 1e-3), "o-", color=col, label=f"r = {r}")
a1.set(xlim=(0, 8), xlabel="depth (mm)", ylabel="T (°C)", title="A sawtooth appears near the surface")
a2.set(xlabel="time step", ylabel="max T above the surface value (K)", title="...and grows exponentially")
a1.legend(); a2.legend(); plt.show()"""),
        md("""
## Crank-Nicolson: stable with large steps
Averaging the explicit and implicit Laplacians gives a scheme that is unconditionally stable and
second order in time; each step solves one tridiagonal system (Thomas algorithm, notebook 03).
"""),
        code("""for t_end in (1.0, 60.0):
    for r in (0.4, 10.0, 100.0):
        n = max(1, int(round(t_end/(r*dx**2/alpha))))
        T_cn = pdes.crank_nicolson(np.full(nx, T0), alpha, dx, t_end/n, n, Ts, T0)
        print(f"t = {t_end:4.0f} s, r = {r:5.1f}: {n:5d} steps, max error {np.max(np.abs(T_cn - exact(x, t_end))):8.3f} K")"""),
        md("""
Every run is stable - nothing explodes - but **stability is not accuracy**. The sudden surface jump
excites short-wavelength components that Crank-Nicolson damps only weakly when $r$ is large; they
appear as oscillations near the surface, large at early times and still visible after 60 s for
$r = 100$. Small steps at the start (or a few implicit Euler steps first) cure this.

## Inverse problem: measuring the diffusivity
A thermocouple 10 mm deep records the temperature; its readings are in `thermocouple_10mm.csv`
(synthetic, with 1 K noise). Find the $\\alpha$ that best fits,
by minimising the sum of squared residuals over $\\log_{10}\\alpha$ with golden-section search.
"""),
        code("""from engmath import datasets
x_tc = 0.010
tc = np.loadtxt(datasets.path("thermocouple_10mm.csv"), delimiter=",", skiprows=1)
t_meas, T_meas = tc[:, 0], tc[:, 1]

def rss(log_a):
    a = 10**log_a
    return np.sum((Ts + (T0 - Ts)*erf(x_tc/(2*np.sqrt(a*t_meas))) - T_meas)**2)

res = optimize.golden_section(rss, -6, -4)
a_fit = 10**res.x[0]
# approximate standard error from the curvature of the RSS (Gauss-Newton)
h = 1e-3*a_fit
Jcol = (erf(x_tc/(2*np.sqrt((a_fit+h)*t_meas))) - erf(x_tc/(2*np.sqrt((a_fit-h)*t_meas))))*(T0 - Ts)/(2*h)
s = np.sqrt(res.fun/(t_meas.size - 1))
se = s/np.sqrt(Jcol @ Jcol)
from engmath.reporting import format_uncertainty
print(f"alpha = {format_uncertainty(a_fit*1e6, se*1e6)} mm²/s  (true {alpha*1e6} mm²/s)")
plt.plot(t_meas, T_meas, "o", label="thermocouple at 10 mm")
tt = np.linspace(0.5, 60, 200)
plt.plot(tt, Ts + (T0 - Ts)*erf(x_tc/(2*np.sqrt(a_fit*tt))), label="fitted model")
plt.xlabel("time (s)"); plt.ylabel("T (°C)"); plt.legend(); plt.show()"""),
        md("""
## Exercises
1. Repeat the inverse problem with the thermocouple position uncertain by ±0.5 mm. How does that
   uncertainty propagate into $\\alpha$? (Hint: $\\alpha$ enters only through $x/\\sqrt{\\alpha t}$.)
2. Replace the fixed surface temperature by convection (a Robin boundary condition) in the
   Crank-Nicolson scheme: which entries of the tridiagonal system change?
3. For mass transfer into a polymer film, the uptake per unit area is
   $M_t = 2C_s\\sqrt{Dt/\\pi}$ at early times. Show that fitting $M_t$ against $\\sqrt t$ is a *linear*
   regression, and use it to estimate $D$ from simulated data with 3 % noise.
"""),
    ]


# =====================================================================================================
@notebook("13_symbolic_sympy")
def _():
    return [
        md(r"""
# 13 · Symbolic mathematics with SymPy

Numerical methods give numbers; **symbolic** mathematics gives formulas. SymPy does algebra and
calculus exactly, like a tireless and careful colleague with pencil and paper. Engineers use it to

- derive formulas (and avoid algebra slips),
- derive and analyse *numerical methods* - e.g. the error of a finite-difference formula,
- produce exact reference results to test numerical code,
- generate derivatives (Jacobians) automatically and turn them into fast NumPy functions.

**Problem.** Derive the deflection of a cantilever beam under a uniform load, check the classic
handbook result, and then use SymPy to derive - and check - numerical methods from earlier notebooks.
"""),
        code(SETUP + "\nimport sympy as sp"),
        md("## Symbols, expressions and calculus"),
        code(r"""x, t = sp.symbols("x t")
expr = sp.exp(x) * sp.sin(x)
print("derivative:        ", sp.diff(expr, x))
print("antiderivative:    ", sp.integrate(expr, x))
print("Taylor series:     ", sp.series(expr, x, 0, 5))
print("Gaussian integral: ", sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo)))
print("integral of x^2 sin x on [0, pi]:", sp.integrate(x**2 * sp.sin(x), (x, 0, sp.pi)))"""),
        md(r"""
Results are exact: $\sqrt{\pi}$ stays `sqrt(pi)`, not 1.7724538509.

## Deriving a formula: deflection of a cantilever
A beam clamped at $x = 0$ and free at $x = L$ carries a uniform load $w$ (N/m). Euler-Bernoulli beam
theory gives $EI\,v^{(4)} = w$, with $v = v' = 0$ at the clamp and zero bending moment and shear force at
the free end, $v''(L) = v^{(3)}(L) = 0$.
"""),
        code(r"""L, EI, w = sp.symbols("L EI w", positive=True)
v = sp.Function("v")
bcs = {v(0): 0, v(x).diff(x).subs(x, 0): 0, v(x).diff(x, 2).subs(x, L): 0, v(x).diff(x, 3).subs(x, L): 0}
deflection = sp.factor(sp.dsolve(sp.Eq(EI * v(x).diff(x, 4), w), v(x), ics=bcs).rhs)
tip = sp.simplify(deflection.subs(x, L))
print("v(x) =", deflection)
print("tip deflection =", tip)"""),
        md(r"""
SymPy reproduces the handbook formula $v_{tip} = wL^4/(8EI)$. `lambdify` turns the symbolic result into
an ordinary NumPy function - here for a 2 m steel cantilever (rectangular 50 × 100 mm section) carrying
2 kN/m:
"""),
        code(r"""v_num = sp.lambdify((x, L, EI, w), deflection, "numpy")
E, b, hgt = 200e9, 0.05, 0.10
I = b * hgt**3 / 12
xx = np.linspace(0, 2, 100)
plt.plot(xx, -v_num(xx, 2.0, E*I, 2000.0) * 1000)
plt.xlabel("x (m)"); plt.ylabel("deflection (mm)"); plt.title("Cantilever under uniform load"); plt.show()
print(f"tip deflection {v_num(2.0, 2.0, E*I, 2000.0)*1000:.3f} mm")"""),
        md(r"""
## Deriving numerical methods: finite differences and their errors
Notebook 06 used a 3-point derivative formula for **unequal** spacing. Where does it come from, and how
accurate is it? Fit the parabola through three points and differentiate it at the middle one:
"""),
        code(r"""h1, h2 = sp.symbols("h1 h2", positive=True)
f0, f1, f2 = sp.symbols("f_0 f_1 f_2")
parabola = sp.interpolate([(-h1, f0), (0, f1), (h2, f2)], t)
derivative_formula = sp.expand(sp.diff(parabola, t).subs(t, 0))
for fi in (f0, f1, f2):
    print(fi, "coefficient:", sp.factor(derivative_formula.coeff(fi)))"""),
        md(r"""
These are exactly the coefficients used in notebook 06. Now the **error**: replace $f_0, f_1, f_2$ by
Taylor series of a smooth function about the middle point, written with its derivatives
$F_0 = f,\ F_1 = f',\ F_2 = f'', \ldots$
"""),
        code(r"""F = sp.symbols("F0:6")          # F0 = f, F1 = f', F2 = f'', ...
taylor = lambda s: sum(F[j] * s**j / sp.factorial(j) for j in range(6))
error = sp.expand(derivative_formula.subs({f0: taylor(-h1), f1: taylor(0), f2: taylor(h2)}) - F[1])
leading = sp.factor(error.coeff(F[3]) * F[3])
print("leading error term:", leading)
print("with equal spacing:", sp.factor(leading.subs({h1: sp.Symbol("h"), h2: sp.Symbol("h")})))"""),
        md(r"""
The error is proportional to $h_1 h_2\, F_3$ (the third derivative): **second order** for any spacing,
as the convergence study of notebook 06 found. For equal spacing it becomes the familiar $h^2 F_3/6$:
the central difference overestimates the slope where the third derivative is positive.

The same trick gives Simpson's rule: integrate the parabola through three equally spaced points.
"""),
        code(r"""h = sp.Symbol("h", positive=True)
simpson = sp.integrate(sp.interpolate([(-h, f0), (0, f1), (h, f2)], t), (t, -h, h))
print("Simpson's rule:", sp.factor(simpson))"""),
        md(r"""
## Symbolic Jacobians for Newton's method
Newton's method for systems (notebook 14) needs the Jacobian matrix. SymPy computes it exactly and
`lambdify` turns it into a fast function - no finite-difference errors.
"""),
        code(r"""X, Y = sp.symbols("X Y")
Fsym = sp.Matrix([X**2 + Y**2 - 4, sp.exp(X) + Y - 1])
Jsym = Fsym.jacobian([X, Y])
print("J =", Jsym)
F_fun = sp.lambdify([(X, Y)], list(Fsym), "numpy")
J_fun = sp.lambdify([(X, Y)], Jsym, "numpy")
from engmath import nonlinear
sol = nonlinear.newton_system(F_fun, [1.0, -1.5], J=J_fun)
print(sol, "  residual:", np.round(F_fun(sol.x), 14))"""),
        md(r"""
## Exact results as test cases for numerical code
A symbolic result is a perfect reference. How fast does Simpson's rule converge on
$\int_0^\pi x^2\sin x\,dx$, whose exact value SymPy gave above as $\pi^2 - 4$?
"""),
        code(r"""from engmath import integrate
exact = float(sp.integrate(x**2 * sp.sin(x), (x, 0, sp.pi)))
g = lambda s: s**2 * np.sin(s)
for n in (4, 8, 16, 32):
    err = abs(integrate.composite(g, 0, np.pi, n, "simpson") - exact)
    print(f"n = {n:>2}: error {err:.2e}")"""),
        md(r"""
Each doubling of $n$ cuts the error about 16-fold: fourth order, as theory predicts.

## Limits of symbolic computation
Not everything has a formula. Polynomials of degree five and higher generally have **no** solution in
radicals (Abel-Ruffini theorem); SymPy then returns implicit roots, and numbers must come from a
numerical method:
"""),
        code(r"""quintic = x**5 - x - 1
print(sp.solve(quintic, x)[:2], "...")
print("numerical root:", sp.nsolve(quintic, x, 1.2))"""),
        md(r"""
Symbolic expressions can also grow explosively (try the determinant of a general 6 × 6 symbolic
matrix). The practical pattern: **derive with SymPy, compute with NumPy** - and use each to check the
other.

## Exercises
1. Derive the deflection of a *simply supported* beam ($v = v'' = 0$ at both ends) under a uniform load
   and confirm the maximum deflection $5wL^4/(384EI)$.
2. Derive the leading error term of the forward difference $(f_1 - f_0)/h$ and of the 5-point central
   difference $(f_{-2} - 8f_{-1} + 8f_1 - f_2)/(12h)$. What are their orders?
3. Use `sp.dsolve` to solve the pin-fin equation of notebook 15, $T'' = m^2 (T - T_\infty)$, with
   $T(0) = T_b$ and an insulated tip $T'(L) = 0$. Check the formula against `bvp.finite_difference`.
"""),
    ]


# =====================================================================================================
@notebook("14_nonlinear_systems")
def _():
    return [
        md(r"""
# 14 · Systems of nonlinear equations

**Problem - the three-reservoir problem.** Three reservoirs at different levels are connected by pipes
to a common junction. What is the pressure head at the junction, and how much water flows in each pipe -
and in which direction? Each pipe obeys the energy equation (head loss $\propto Q|Q|$), and the flows
must balance at the junction:
$$z_i - H = k_i\,Q_i\,|Q_i|\quad (i = 1, 2, 3),\qquad Q_1 + Q_2 + Q_3 = 0,\qquad k_i = \frac{8 f L_i}{g\pi^2 D_i^5},$$
with $Q_i$ the flow from reservoir $i$ towards the junction. Four nonlinear equations in four unknowns
$(H, Q_1, Q_2, Q_3)$.
"""),
        code(SETUP + "\nfrom engmath import nonlinear, roots"),
        code(r"""z = np.array([120.0, 100.0, 60.0])              # reservoir levels (m)
Lp = np.array([1000.0, 800.0, 1200.0])            # pipe lengths (m)
Dp = np.array([0.30, 0.25, 0.20])                 # diameters (m)
f_darcy, g = 0.02, 9.81
k = 8 * f_darcy * Lp / (g * np.pi**2 * Dp**5)

def F(v):
    H, Q = v[0], v[1:]
    return np.r_[z - H - k * Q * np.abs(Q), Q.sum()]

sol = nonlinear.newton_system(F, [90.0, 0.1, 0.0, -0.1])
print(sol)
H, Q = sol.x[0], sol.x[1:]
print(f"junction head H = {H:.2f} m")
for i, q in enumerate(Q, start=1):
    direction = "from the reservoir into the junction" if q > 0 else "from the junction into the reservoir"
    print(f"pipe {i}: {abs(q)*1000:6.1f} L/s {direction}")"""),
        md(r"""
Water can flow either way in each pipe, which is why the flows are unknowns with a sign: the solution
tells us which reservoirs supply the network and which are being filled.

## Newton's method for systems
Linearise all equations at once: $F(\mathbf x + \Delta\mathbf x) \approx F(\mathbf x) + J\,\Delta\mathbf x = \mathbf 0$, where
$J_{ij} = \partial F_i/\partial x_j$ is the **Jacobian**. Each iteration solves one linear system
(notebook 03). Near the solution the error is roughly squared at every step:
"""),
        code(r"""plt.semilogy(sol.residual_history, "o-")
plt.xlabel("iteration"); plt.ylabel("||F(x)||"); plt.title("Quadratic convergence of Newton's method"); plt.show()"""),
        md(r"""
## Checking the answer independently
Here the flows can be eliminated: $Q_i = \mathrm{sign}(z_i - H)\sqrt{|z_i - H|/k_i}$, leaving one equation for
$H$ (continuity), which Brent's method (notebook 02) solves reliably.
"""),
        code(r"""continuity = lambda H: np.sum(np.sign(z - H) * np.sqrt(np.abs(z - H) / k))
H_check = roots.brent(continuity, z.min(), z.max()).root
print(f"Newton (4 unknowns): H = {H:.10f} m\nBrent  (1 unknown):  H = {H_check:.10f} m")"""),
        md(r"""
Reducing a system to fewer unknowns is always worth trying, but real networks (dozens of pipes,
loops, pumps) cannot be reduced by hand - Newton's method for systems is the workhorse.

## Starting values and basins of attraction
A nonlinear system may have several solutions; which one Newton finds depends on the start. The
system $x^2 + y^2 = 4$, $xy = 1$ has four solutions (a circle meets a hyperbola). Colouring each starting
point by the solution it converges to reveals the **basins of attraction**:
"""),
        code(r"""def newton_grid(x, y, iters=30):
    # vectorised Newton on a whole grid of starting points (analytic Jacobian, no line search)
    with np.errstate(all="ignore"):                  # some starts blow up - expected
        for _ in range(iters):
            f1, f2 = x**2 + y**2 - 4, x*y - 1
            a, b, c, d = 2*x, 2*y, y, x              # Jacobian [[a, b], [c, d]]
            det = a*d - b*c
            x, y = x - (d*f1 - b*f2)/det, y - (-c*f1 + a*f2)/det
    return x, y

g1 = np.linspace(-3, 3, 400)
X0, Y0 = np.meshgrid(g1, g1)
XE, YE = newton_grid(X0, Y0)
r1, r2 = np.sqrt(2 + np.sqrt(3)), np.sqrt(2 - np.sqrt(3))       # exact solutions (±r1, ±r2), (±r2, ±r1)
sols = np.array([[r1, r2], [r2, r1], [-r1, -r2], [-r2, -r1]])
dist = np.stack([np.hypot(XE - sx, YE - sy) for sx, sy in sols])
dist = np.nan_to_num(dist, nan=np.inf)
basin = np.where(dist.min(axis=0) < 1e-6, dist.argmin(axis=0), -1)
plt.figure(figsize=(5.5, 5))
plt.imshow(basin, extent=[-3, 3, -3, 3], origin="lower", cmap="Set2")
plt.plot(sols[:, 0], sols[:, 1], "k*", ms=12)
plt.xlabel("starting x"); plt.ylabel("starting y"); plt.title("Which solution does Newton find?")
plt.show()
print(f"{np.mean(basin == -1)*100:.1f} % of the starting points do not converge within 30 iterations")"""),
        md(r"""
Near the lines $y = \pm x$, where the Jacobian is singular, the basins interleave in intricate patterns:
a tiny change of starting point can lead to a different solution. **Always check that the solution
found is the physically meaningful one**, and take starting values from physics (here: the junction
head must lie between the lowest and the highest reservoir).

## Robustness: the line search
Plain Newton can overshoot wildly. `newton_system` halves the step until the residual really decreases
(a *line search*), which rescues many poor starting points:
"""),
        code(r"""Fbad = lambda v: [np.arctan(v[0]), v[1] - 0.1]
for ls in (False, True):
    r = nonlinear.newton_system(Fbad, [3.0, 1.0], line_search=ls, maxiter=30)
    print(f"line search {ls!s:5}: {r}")"""),
        md(r"""
## In practice
`scipy.optimize.root` offers several robust solvers (a hybrid Powell method by default):
"""),
        code(r"""from scipy.optimize import root
ref = root(F, [90.0, 0.1, 0.0, -0.1])
print("scipy:", np.round(ref.x, 10), "\nours: ", np.round(sol.x, 10))"""),
        md(r"""
## Exercises
1. Replace the constant friction factor by the Colebrook equation (notebook 02) for each pipe
   (commercial steel, $\varepsilon$ = 0.045 mm; water $\nu$ = 1.0 × 10⁻⁶ m²/s), which puts a root solve
   inside every evaluation of $F$. How much do the flows change?
2. Add a fourth reservoir at 90 m, connected by a 900 m, 0.25 m pipe. Which reservoirs now supply water?
3. Start from $H = 150$ m (above every reservoir). Does Newton still find the solution, with and without
   the line search? Look at the residual history to explain.
"""),
    ]


# =====================================================================================================
@notebook("15_boundary_value_problems")
def _():
    return [
        md(r"""
# 15 · Boundary-value problems: heat transfer in a fin

**Problem.** A pin fin (an aluminium rod, 5 mm diameter, 50 mm long) cools a device whose surface is at
100 °C, in air at 20 °C with $h$ = 25 W/(m²·K). How does the temperature vary along the fin, and how
much heat does it remove? The energy balance gives
$$\frac{d^2T}{dx^2} = m^2\,(T - T_\infty),\qquad m^2 = \frac{hP}{kA} = \frac{4h}{kD},$$
with $T(0) = T_b$ at the base and convection at the tip, $-k\,T'(L) = h\,(T(L) - T_\infty)$.
Unlike notebook 11, conditions are given at **both ends**: a *boundary-value* problem.
"""),
        code(SETUP + "\nfrom engmath import bvp, nonlinear, odes"),
        code(r"""k, h, D, L, Tb, Tinf = 200.0, 25.0, 0.005, 0.05, 100.0, 20.0
m = np.sqrt(4 * h / (k * D))
Ac = np.pi * D**2 / 4
Bi = h / (m * k)
def T_exact(x):
    return Tinf + (Tb - Tinf) * (np.cosh(m*(L - x)) + Bi*np.sinh(m*(L - x))) / (np.cosh(m*L) + Bi*np.sinh(m*L))
q_exact = k * Ac * m * (Tb - Tinf) * (np.sinh(m*L) + Bi*np.cosh(m*L)) / (np.cosh(m*L) + Bi*np.sinh(m*L))
print(f"m = {m:.2f} 1/m, mL = {m*L:.2f};  exact heat rate {q_exact:.4f} W")"""),
        md(r"""
## Method 1: shooting
Turn the BVP into an initial-value problem by *guessing* the unknown slope $T'(0) = s$, integrate to the
tip (RK4, notebook 11), and measure how badly the tip condition is missed. Adjusting $s$ until the miss
is zero is a root-finding problem (notebook 02).
"""),
        code(r"""rhs = lambda x, u: [u[1], m**2 * (u[0] - Tinf)]
tip_miss = lambda T_end, dT_end: dT_end + h/k * (T_end - Tinf)     # zero when the tip condition holds
fig, ax = plt.subplots()
for s in (-2000, -1000):
    xg, u = odes.rk4(rhs, [Tb, s], 0, L, 200)
    ax.plot(xg*1000, u[:, 0], "--", label=f"guess T'(0) = {s} K/m: tip miss {tip_miss(u[-1, 0], u[-1, 1]):+.0f} K/m")
tip = ("robin", 1.0, h/k, h/k*Tinf)                                  # T' + (h/k) T = (h/k) T_inf
xs, Ts, dTs, s_star = bvp.shooting(lambda x, T, dT: m**2 * (T - Tinf), 0, L, Tb, tip, slopes=(-2000, -1000))
ax.plot(xs*1000, Ts, "k", lw=2, label=f"shooting solution: T'(0) = {s_star:.1f} K/m")
ax.set(xlabel="x (mm)", ylabel="T (°C)"); ax.legend(fontsize=8); plt.show()
print(f"max error vs exact {np.max(np.abs(Ts - T_exact(xs))):.1e} K;  heat rate -kA T'(0) = {-k*Ac*s_star:.4f} W")"""),
        md(r"""
## Method 2: finite differences
Replace $T''$ by the central difference at every node (notebook 06). All the equations are then solved
**together** - a tridiagonal linear system (notebook 03). The convective tip uses a *ghost node* beyond
the end, which keeps second-order accuracy.
"""),
        code(r"""errs, ns = [], [5, 10, 20, 40, 80]
for n in ns:
    xf, Tf = bvp.finite_difference(lambda x: 0*x, lambda x: -m**2 + 0*x, lambda x: -m**2*Tinf + 0*x,
                                   0, L, n, ("dirichlet", Tb), tip)
    errs.append(np.max(np.abs(Tf - T_exact(xf))))
slope = np.polyfit(np.log(L/np.array(ns)), np.log(errs), 1)[0]
plt.loglog(L/np.array(ns)*1000, errs, "o-")
plt.xlabel("grid spacing (mm)"); plt.ylabel("max error (K)"); plt.title(f"Second-order accuracy: slope {slope:.2f}")
plt.show()"""),
        md(r"""
Shooting reuses an accurate ODE integrator and is easy to set up; finite differences handle many
unknowns and variable coefficients naturally, and extend to 2D and 3D (notebook 19).

## A nonlinear fin: adding radiation
A hot steel fin (400 °C base, $k$ = 40 W/(m·K), weak natural convection $h$ = 10 W/(m²·K), emissivity
0.9) also loses heat by radiation. With absolute temperatures the equation becomes nonlinear:
$$T'' = m^2 (T - T_\infty) + \frac{\varepsilon\sigma P}{kA}\,\left(T^4 - T_{sur}^4\right),$$
with an insulated tip for simplicity. Finite differences now give a **nonlinear system**, solved with
Newton's method (notebook 14):
"""),
        code(r"""ks, hs, eps, sigma = 40.0, 10.0, 0.9, 5.670374419e-8
Tb2, Tinf2 = 673.15, 293.15
m2sq = 4 * hs / (ks * D)
rad = eps * sigma * 4 / (ks * D)
n = 100
xg = np.linspace(0, L, n + 1); dx = xg[1] - xg[0]

def residual(T_in, with_radiation=True):
    T = np.r_[Tb2, T_in]
    Tg = np.r_[T, T[-2]]                                   # ghost node: insulated tip, T'(L) = 0
    source = m2sq*(T - Tinf2) + (rad*(T**4 - Tinf2**4) if with_radiation else 0.0)
    return (Tg[2:] - 2*Tg[1:-1] + Tg[:-2]) / dx**2 - source[1:]

for with_rad in (False, True):
    sol = nonlinear.newton_system(lambda v: residual(v, with_rad), np.full(n, Tb2))
    T = np.r_[Tb2, sol.x]
    q = -ks * Ac * (-3*T[0] + 4*T[1] - T[2]) / (2*dx)      # one-sided 2nd-order derivative at the base
    label = "convection + radiation" if with_rad else "convection only"
    plt.plot(xg*1000, T - 273.15, label=f"{label}: {q:.2f} W ({sol.iterations} Newton iterations)")
plt.xlabel("x (mm)"); plt.ylabel("T (°C)"); plt.legend(fontsize=8); plt.title("Radiation matters for hot fins"); plt.show()"""),
        md(r"""
## Exercises
1. Solve the radiating fin by shooting instead (tip condition $T'(L) = 0$). The ODE is nonlinear, so the
   secant iteration no longer converges in one step. Compare the heat rates.
2. **Beam deflection.** A simply supported beam under uniform load satisfies $EI\,v'' = -M(x)$ with
   the sagging bending moment $M(x) = wx(L - x)/2$, $v(0) = v(L) = 0$ and $v$ the deflection measured
   downwards. Solve it with
   `bvp.finite_difference` and compare the midspan deflection with $5wL^4/(384EI)$ (notebook 13).
3. Fin **efficiency** is the actual heat rate divided by the rate if the whole fin were at $T_b$. Plot it
   against fin length for the aluminium fin. Beyond which length does extra length hardly help?
"""),
    ]


# =====================================================================================================
@notebook("16_fourier_analysis")
def _():
    return [
        md(r"""
# 16 · Fourier analysis: diagnosing a machine from its vibration

**Problem.** An accelerometer on a pump bearing records vibration velocity at 2048 samples per second.
The shaft turns at 1480 rpm. Is the bearing healthy? In the time signal everything is mixed together
with noise; the **frequency spectrum** separates the contributions: shaft rotation, its harmonics, and -
if present - a bearing-defect frequency. The record is in `pump_vibration.csv` (synthetic, so that we
can check the analysis against what was put into it).
"""),
        code(SETUP + "\nfrom engmath import fourier"),
        code(r"""from engmath import datasets
data = np.loadtxt(datasets.path("pump_vibration.csv"), delimiter=",", skiprows=1)
t, signal = data[:, 0], data[:, 1]                    # time (s), vibration velocity (mm/s)
fs = 1 / (t[1] - t[0])                                # sampling rate from the time column: 2048 Hz
duration = t.size / fs
f_shaft = 1480 / 60                                   # shaft speed 1480 rpm = 24.67 Hz
plt.plot(t[:512], signal[:512]); plt.xlabel("time (s)"); plt.ylabel("velocity (mm/s)")
plt.title("A quarter of a second of vibration data: is there a defect?"); plt.show()"""),
        md(r"""
## The discrete Fourier transform
The DFT writes $N$ samples as a sum of sinusoids at the frequencies $f_k = k\,f_s/N$:
$$X_k = \sum_{n=0}^{N-1} x_n\,e^{-2\pi i k n/N}.$$
Computed from the definition this costs $N^2$ operations. The **fast Fourier transform** (FFT) splits the
sum into even and odd samples recursively and needs only about $N\log_2 N$:
"""),
        code(r"""import time
for N in (256, 1024, 4096):
    x = signal[:N]
    t0 = time.perf_counter(); Xd = fourier.dft(x); td = time.perf_counter() - t0
    t0 = time.perf_counter(); Xf = fourier.fft(x); tf = time.perf_counter() - t0
    t0 = time.perf_counter(); Xn = np.fft.fft(x); tn = time.perf_counter() - t0
    print(f"N = {N:5d}: DFT {td*1e3:8.1f} ms, our FFT {tf*1e3:6.1f} ms, numpy {tn*1e3:5.2f} ms; "
          f"max difference {max(np.max(np.abs(Xd - Xn)), np.max(np.abs(Xf - Xn))):.1e}")"""),
        md(r"""
## The amplitude spectrum
`fourier.spectrum` returns frequencies and amplitudes scaled so that a sine of amplitude $A$ appears as
a peak of height about $A$.
"""),
        code(r"""f, amp = fourier.spectrum(signal, fs, window="hann")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(f, amp)
for fr, name in ((f_shaft, "1x shaft"), (2*f_shaft, "2x shaft"), (137.3, "bearing defect")):
    i = np.argmin(abs(f - fr))
    ax.annotate(f"{name}\n{f[i]:.1f} Hz", (f[i], amp[i]), textcoords="offset points", xytext=(8, 8), fontsize=8)
ax.set(xlim=(0, 250), xlabel="frequency (Hz)", ylabel="amplitude (mm/s)", title="Vibration spectrum")
plt.show()
i_def = np.argmin(abs(f - 137.3))
noise_floor = np.median(amp[(f > 60) & (f < 250)])
print(f"frequency resolution fs/N = {fs/t.size:.2f} Hz")
print(f"defect peak {amp[i_def]:.3f} mm/s vs median noise level {noise_floor:.3f} mm/s")"""),
        md(r"""
The bearing defect, hidden in the time signal, stands out in the spectrum: the noise is spread over all
frequencies, while a periodic component concentrates in one.

## Leakage and windowing
The shaft frequency (24.67 Hz) falls *between* two frequency bins, which are 0.5 Hz apart. Its energy
then leaks into neighbouring bins. Tapering the record with a **Hann window** reduces the leakage:
"""),
        code(r"""f_r, a_r = fourier.spectrum(signal, fs, window=None)
f_h, a_h = fourier.spectrum(signal, fs, window="hann")
plt.semilogy(f_r, a_r, label="rectangular (no window)", alpha=0.8)
plt.semilogy(f_h, a_h, label="Hann window", alpha=0.8)
plt.xlim(0, 80); plt.ylim(1e-3, 2); plt.xlabel("frequency (Hz)"); plt.ylabel("amplitude (mm/s)")
plt.legend(); plt.title("Leakage around the shaft peak"); plt.show()"""),
        md(r"""
## Resolution and aliasing: two ways to be fooled
**Resolution.** Two frequencies closer than about $f_s/N = 1/T_{record}$ cannot be separated - only a
*longer record* helps. **Aliasing.** A frequency above the Nyquist frequency $f_s/2$ is indistinguishable
from a lower one. Sampling the defect vibration at only 200 Hz:
"""),
        code(r"""fs_low = 200.0
t_low = np.arange(int(fs_low * duration)) / fs_low
f_al, a_al = fourier.spectrum(np.sin(2*np.pi*137.3*t_low), fs_low)
print(f"a 137.3 Hz sine sampled at {fs_low:.0f} Hz appears at 200 - 137.3 = 62.7 Hz "
      f"(spectrum peak at {f_al[np.argmax(a_al)]:.1f} Hz, the nearest frequency bin)")"""),
        md(r"""
Once sampled, an aliased signal cannot be repaired: data-acquisition systems remove everything above
$f_s/2$ **before** sampling (anti-aliasing filters).

## Filtering in the frequency domain
Keep only a band around the defect frequency and transform back. Because the data file is synthetic, we
know the defect component it contains (0.15 mm/s at 137.3 Hz) and can check the result. How wide should
the band be?
"""),
        code(r"""defect_true = 0.15 * np.sin(2*np.pi*137.3*t + 2)
for lo, hi in ((130, 145), (135, 140), (136, 138.5), (136.8, 137.8)):
    est = fourier.fft_filter(signal, fs, low=lo, high=hi)
    print(f"band {lo}-{hi} Hz (width {hi - lo:4.1f} Hz): RMS error {np.sqrt(np.mean((est - defect_true)**2)):.3f} mm/s")
defect_est = fourier.fft_filter(signal, fs, low=136, high=138.5)
plt.plot(t[:300], defect_true[:300], "k", lw=2, label="true defect component")
plt.plot(t[:300], defect_est[:300], label="band-pass filtered signal")
plt.xlabel("time (s)"); plt.ylabel("mm/s"); plt.legend(); plt.show()
print(f"defect RMS {0.15/np.sqrt(2):.3f} mm/s, noise RMS 0.5 mm/s")"""),
        md(r"""
A narrower band lets through less noise - but a band that is *too* narrow also cuts off part of the
defect's own energy, which leaks into neighbouring bins because 137.3 Hz is not exactly on a bin. Even
at best, the noise sharing the band remains, so the reconstructed waveform is only approximate. For
**detecting** a fault the spectrum is the better tool; filtering is for separating components whose
frequencies are well apart."""),
        md(r"""
## Exercises
1. Use only the first 0.25 s of data. Can you still separate the shaft peak from its second harmonic?
   What is the frequency resolution now?
2. Zero-pad the 0.25 s record to 2 s (append zeros) and recompute the spectrum. Does that improve the
   *resolution*, or only make the curve smoother?
3. By Parseval's theorem the mean square of the signal equals the sum of squared Fourier amplitudes
   (suitably scaled). Compute the RMS vibration velocity from the spectrum and compare with
   `np.sqrt(np.mean(signal**2))` - the quantity used in vibration-severity standards.
"""),
    ]


# =====================================================================================================
@notebook("17_optimisation_constraints_lp")
def _():
    return [
        md(r"""
# 17 · Constrained optimisation and linear programming

Engineering design is optimisation *under constraints*: limited machine time, a required volume, a
maximum height. Two very common cases:

- **Linear programming (LP)** - linear objective, linear constraints. Production planning, blending,
  scheduling, transport.
- **Nonlinear constrained optimisation** - e.g. the cheapest vessel of a given volume.

**Problem 1 - production planning.** A workshop builds two pump models. A standard pump earns €400 and
needs 2 h of machining and 1 h of assembly; a heavy-duty pump earns €300 and needs 1 h of machining and
1 h of assembly. Each week there are 100 machining hours and 80 assembly hours, and castings for at
most 40 standard pumps. How many of each should be built?
"""),
        code(SETUP + "\nfrom engmath import optimize\nfrom scipy.optimize import linprog, minimize"),
        code(r"""c = [400, 300]                       # profit per pump (EUR)
A = [[2, 1],                         # machining hours
     [1, 1],                         # assembly hours
     [1, 0]]                         # castings for standard pumps
b = [100, 80, 40]
lp = optimize.simplex_lp(c, A, b)
print(f"optimal plan: {lp.x[0]:.0f} standard + {lp.x[1]:.0f} heavy-duty pumps, profit EUR {lp.objective:,.0f}")
print("corners visited by the simplex method:", [(int(round(v[0])), int(round(v[1]))) for v in lp.vertices])"""),
        md(r"""
## Why the answer is at a corner
Each constraint cuts the plane in half; together they form a convex polygon of feasible plans. Lines of
equal profit are parallel, so the best plan is where the highest profit line last touches the polygon -
**always at a corner**. The simplex method walks from corner to corner, improving the profit each time.
"""),
        code(r"""x1 = np.linspace(0, 60, 300)
fig, ax = plt.subplots(figsize=(6, 5))
ax.fill_between(x1, 0, np.minimum(100 - 2*x1, 80 - x1), where=(x1 <= 40) & (np.minimum(100 - 2*x1, 80 - x1) >= 0),
                alpha=0.25, label="feasible plans")
ax.plot(x1, 100 - 2*x1, label="machining: 2x₁ + x₂ = 100")
ax.plot(x1, 80 - x1, label="assembly: x₁ + x₂ = 80")
ax.axvline(40, color="C3", label="castings: x₁ = 40")
ax.plot(x1, (lp.objective - 400*x1)/300, "k--", lw=1, label=f"profit = EUR {lp.objective:,.0f}")
path = np.array(lp.vertices)
ax.plot(path[:, 0], path[:, 1], "ko-", ms=7, label="simplex path")
ax.set(xlim=(0, 60), ylim=(0, 100), xlabel="standard pumps x₁", ylabel="heavy-duty pumps x₂")
ax.legend(fontsize=8); plt.show()"""),
        md(r"""
## What are the resources worth?
The **shadow price** of a constraint is the extra profit from one more unit of that resource - the most
you should pay for it. The simplex method delivers these as a by-product:
"""),
        code(r"""for name, price in zip(["machining hour", "assembly hour", "casting"], lp.shadow_prices):
    print(f"one more {name:<15}: + EUR {price:6.1f}")
more_assembly = optimize.simplex_lp(c, A, [100, 81, 40])
print(f"check: with 81 assembly hours the profit is EUR {more_assembly.objective:,.0f} "
      f"(+{more_assembly.objective - lp.objective:.0f})")"""),
        md(r"""
Castings have a shadow price of zero: the constraint is not binding (only 20 of 40 are used), so more
castings are worthless. An extra assembly hour is worth more than an extra machining hour - useful
information for deciding on overtime.

For real problems (thousands of variables) use `scipy.optimize.linprog`; integer requirements (whole
pumps!) need `scipy.optimize.milp`.
"""),
        code(r"""ref = linprog([-ci for ci in c], A_ub=A, b_ub=b, method="highs")
print("linprog:", ref.x, " profit", -ref.fun, " shadow prices", -ref.ineqlin.marginals)"""),
        md(r"""
## Problem 2 - a nonlinear design: the minimum-material tank
A closed cylindrical tank must hold 1 m³. Which radius $r$ and height $h$ use the least sheet metal
(surface area $2\pi r^2 + 2\pi r h$)? The volume $\pi r^2 h = 1$ is an **equality constraint**. Without
further limits the classic answer - derived with a Lagrange multiplier in SymPy (notebook 13) - is:
"""),
        code(r"""import sympy as sp
r_, h_, lam, V = sp.symbols("r h lambda V", positive=True)
Lagr = 2*sp.pi*r_**2 + 2*sp.pi*r_*h_ - lam*(sp.pi*r_**2*h_ - V)
sol = sp.solve([sp.diff(Lagr, v) for v in (r_, h_, lam)], [r_, h_, lam], dict=True)[0]
print("optimum:  r =", sol[r_], "  h =", sol[h_], "  h/r =", sp.simplify(sol[h_]/sol[r_]))"""),
        md(r"""
**Height equals diameter.** Now add a practical limit - the tank must fit under a 1 m ceiling, $h \le 1$ -
and solve numerically with the **penalty method**: add $\mu\,(\text{violation})^2$ to the objective and
increase $\mu$ step by step, starting from $\mu_0$. This is where formulation mistakes surface.

First, a loophole. The formulas accept *any* numbers, including a negative radius:
"""),
        code(r"""area = lambda v: 2*np.pi*v[0]**2 + 2*np.pi*v[0]*v[1]           # v = (r, h)
volume = lambda v: np.pi*v[0]**2*v[1] - 1.0
bad = [-np.sqrt(1/np.pi), 1.0]
print(f"r = {bad[0]:.4f} m, h = 1 m: volume constraint {volume(bad):+.1e}, 'area' = {area(bad):.3f} m²")
for mu0 in (10.0, 1.0):
    with np.errstate(all="ignore"):
        naive = optimize.penalty_minimize(area, [0.5, 1.0], ineq=[lambda v: v[1] - 1.0], eq=[volume], mu0=mu0)
    print(f"penalty method, mu0 = {mu0:4}: r = {naive.x[0]:.4f} m, h = {naive.x[1]:.4f} m, 'area' = {naive.fun:.3f} m²")"""),
        md(r"""
A tank of radius −0.56 m satisfies both constraints with a *negative* "area" - better than any real tank.
With $\mu_0 = 10$ the optimiser happens to stay in the physical region and finds the right answer; with
$\mu_0 = 1$ it wanders across $r = 0$ and exploits the loophole. **Optimisers exploit every loophole in a
formulation**, and whether they find it is a matter of luck. Close it: optimise $\ln r$ and $\ln h$, so
positivity is automatic (or use bounds, as below).

Even then, the penalty weight must be sensible. If $\mu$ starts too small, violating the volume constraint
is cheaper than building a tank:
"""),
        code(r"""area_u = lambda u: area(np.exp(u))
vol_u = lambda u: volume(np.exp(u))
h_limit = lambda u: np.exp(u[1]) - 1.0
for mu0 in (1.0, 10.0):
    res = optimize.penalty_minimize(area_u, np.log([0.5, 1.0]), ineq=[h_limit], eq=[vol_u], mu0=mu0)
    r_opt, h_opt = np.exp(res.x)
    print(f"mu0 = {mu0:4}: r = {r_opt:.6f} m, h = {h_opt:.6f} m, volume = {np.pi*r_opt**2*h_opt:.6f} m³")"""),
        md(r"""
With $\mu_0 = 1$ the first round shrinks the tank to nothing (the penalty for zero volume, 1, is less than
the area of a real tank, about 5.5 m²) and later rounds cannot recover. **Scale the penalty to the
objective.** With a sensible $\mu_0$ the answer is correct: the height limit is active ($h = 1$ m), so
$r = \sqrt{1/\pi} = 0.5642$ m. SciPy's SLSQP solver handles constraints directly:
"""),
        code(r"""slsqp = minimize(area, [0.5, 1.0], method="SLSQP", bounds=[(1e-3, None), (1e-3, None)],
                 constraints=[{"type": "eq", "fun": volume}, {"type": "ineq", "fun": lambda v: 1.0 - v[1]}])
print(f"SLSQP: r = {slsqp.x[0]:.6f} m, h = {slsqp.x[1]:.6f} m, area {slsqp.fun:.4f} m²  (exact r = {np.sqrt(1/np.pi):.6f})")"""),
        md(r"""
## Exercises
1. The workshop can buy up to 10 extra assembly hours at EUR 150 per hour. Should it? Use the shadow
   price, then confirm by re-solving.
2. **Whole pumps.** A redesigned standard pump needs 2.4 h of machining, and 105 machining hours are
   available. Solve the LP, then round its solution to whole pumps (to the nearest integer, and
   downwards). Are the rounded plans feasible? Optimal? Compare with `scipy.optimize.milp`, which
   requires whole numbers.
3. The tank's end caps must be twice as thick as its wall. Minimise the material *volume*
   ($2\cdot 2\pi r^2 t + 2\pi r h t$). How does the optimal $h/r$ change? Derive it with SymPy.
"""),
    ]


# =====================================================================================================
@notebook("18_monte_carlo")
def _():
    return [
        md(r"""
# 18 · Monte Carlo simulation

Monte Carlo methods answer questions with random numbers: simulate many random cases and average.
They shine where other methods fail - integrals in many dimensions, random processes, and
probabilities of rare failures. The price: the error decreases only as $1/\sqrt N$.

Three engineering uses:
1. integration in many dimensions,
2. random walks - the microscopic picture of diffusion,
3. the probability that a structure fails.
"""),
        code(SETUP + "\nfrom engmath import montecarlo\nfrom scipy import stats\nfrom scipy.special import erf, gamma as gamma_fn"),
        md(r"""
## Why Monte Carlo wins in high dimensions
Estimate the volume of the unit ball in $d$ dimensions (exact: $\pi^{d/2}/\Gamma(d/2+1)$) with the same
budget of about one million function evaluations: a grid (midpoint rule, $m$ points per axis, $m^d$ in
total) versus random points.
"""),
        code(r"""budget = 1_000_000
inside = lambda p: (np.sum(p**2, axis=1) <= 1.0).astype(float)
print(f"{'d':>3}{'points/axis':>12}{'grid error':>12}{'MC error':>11}{'MC std. error':>15}")
for d in (2, 3, 5, 8, 10):
    exact = np.pi**(d/2) / gamma_fn(d/2 + 1)
    m = int(round(budget ** (1/d)))
    axis = -1 + (np.arange(m) + 0.5) * 2/m                  # midpoints of m cells on [-1, 1]
    grid = np.stack(np.meshgrid(*([axis] * d), indexing="ij"), axis=-1).reshape(-1, d)
    grid_est = inside(grid).mean() * 2**d
    mc_est, mc_se = montecarlo.integrate(inside, [(-1, 1)] * d, m**d, seed=d)
    print(f"{d:>3}{m:>12}{abs(grid_est/exact - 1):>11.2%}{abs(mc_est/exact - 1):>11.2%}{mc_se/exact:>14.2%}")"""),
        md(r"""
In two and three dimensions the grid is at least as accurate as Monte Carlo (compare with the Monte
Carlo *standard error*, its typical error; a single run can be luckier). By five dimensions the grid falls
behind, and in ten dimensions - only 4 points per axis - its error is huge, while Monte Carlo's error
barely depends on the dimension. This is why Monte Carlo is used for high-dimensional
problems - uncertainty propagation with many inputs (notebook 01), statistical physics, finance.

## The $1/\sqrt N$ law
"""),
        code(r"""f5 = lambda p: np.exp(-np.sum(p**2, axis=1))            # integral over [0, 1]^5 is known exactly
exact5 = (np.sqrt(np.pi)/2 * erf(1))**5
Ns = np.logspace(2, 6, 9).astype(int)
errs = [abs(montecarlo.integrate(f5, [(0, 1)]*5, N, seed=1)[0] - exact5) for N in Ns]
ses = [montecarlo.integrate(f5, [(0, 1)]*5, N, seed=1)[1] for N in Ns]
plt.loglog(Ns, errs, "o", label="actual error")
plt.loglog(Ns, ses, "-", label="estimated standard error")
plt.loglog(Ns, ses[0]*np.sqrt(Ns[0]/Ns), "k:", label="∝ 1/√N")
plt.xlabel("number of samples N"); plt.ylabel("error"); plt.legend(); plt.show()"""),
        md(r"""
One more correct digit costs **100 times** more samples. The standard error, estimated from the samples
themselves, tells you how far to trust the result.

## Random walks and diffusion
A molecule in a liquid is knocked about randomly. Model it as a walker taking unit steps in random
directions. After $n$ steps its mean squared displacement is $\langle r^2\rangle = n$ - it grows linearly
in time, the signature of diffusion ($\langle r^2\rangle = 2dDt$ in $d$ dimensions).
"""),
        code(r"""paths = montecarlo.random_walk(1000, 5, dim=2, seed=1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
for w in range(5):
    a1.plot(paths[:, w, 0], paths[:, w, 1], lw=0.8)
a1.set(aspect="equal", title="Five random walks of 1000 steps")
walk = montecarlo.random_walk(1000, 3000, dim=2, seed=2)
msd = np.mean(np.sum(walk**2, axis=2), axis=1)
a2.plot(msd, label="simulated ⟨r²⟩ (3000 walkers)"); a2.plot([0, 1000], [0, 1000], "k--", label="theory ⟨r²⟩ = n")
a2.set(xlabel="steps n", ylabel="mean squared displacement", title="Diffusion: ⟨r²⟩ grows linearly"); a2.legend()
plt.show()"""),
        md(r"""
The positions of many 1D walkers after $n$ steps approach a Gaussian of variance $n$ (central limit
theorem) - exactly the solution of the diffusion equation from a point source, whose error-function
form appeared in notebook 12. Random walks and the diffusion PDE are two views of one phenomenon.
"""),
        code(r"""ends = montecarlo.random_walk(400, 20000, seed=3)[-1, :, 0]
xs = np.arange(-80, 81, 2)
plt.hist(ends, bins=np.arange(-81, 82, 2), density=True, alpha=0.6, label="20 000 walkers after 400 steps")
plt.plot(xs, stats.norm.pdf(xs, 0, np.sqrt(400)), "k", label="Gaussian, variance 400 (diffusion solution)")
plt.xlabel("position"); plt.ylabel("probability density"); plt.legend(); plt.show()"""),
        md(r"""
## Probability of failure
A steel member has a strength $S$ and carries a load effect $L$, both uncertain: $S \sim N(400, 25)$ MPa
and $L \sim N(300, 30)$ MPa. It fails when $L > S$. For normal variables the answer is exact,
$P_f = \Phi\!\left(-\frac{\mu_S - \mu_L}{\sqrt{\sigma_S^2 + \sigma_L^2}}\right)$; Monte Carlo just counts failures:
"""),
        code(r"""pf_exact = stats.norm.cdf(-(400 - 300) / np.hypot(25, 30))
rng = np.random.default_rng(18)
for N in (10_000, 1_000_000):
    fails = rng.normal(300, 30, N) > rng.normal(400, 25, N)
    p = fails.mean()
    print(f"N = {N:>9,}: P_f = {p:.5f} ± {np.sqrt(p*(1-p)/N):.5f}   (exact {pf_exact:.5f})")
print(f"samples needed for ±10 % relative error: about {int((1 - pf_exact)/(pf_exact*0.1**2)):,}")"""),
        md(r"""
Rare events are expensive: the number of samples needed grows like $1/P_f$. Monte Carlo's strength is that
**nothing** in the method assumed normal distributions. Suppose the load is better described by a
skewed lognormal distribution with the same mean and standard deviation - no simple formula exists,
but the simulation is unchanged:
"""),
        code(r"""sig_ln = np.sqrt(np.log(1 + (30/300)**2)); mu_ln = np.log(300) - sig_ln**2/2
N = 2_000_000
load = rng.lognormal(mu_ln, sig_ln, N)
p_ln = np.mean(load > rng.normal(400, 25, N))
print(f"lognormal load: P_f = {p_ln:.5f} ± {np.sqrt(p_ln*(1 - p_ln)/N):.5f}  vs normal load {pf_exact:.5f}")"""),
        md(r"""
The heavier upper tail of the lognormal load raises the failure probability noticeably: in reliability,
the **tails** of the distributions decide the answer.

## Exercises
1. Estimate $\pi$ with Buffon's needle: drop needles of length $\ell$ on lines spaced $d \ge \ell$ apart;
   $P(\text{cross}) = 2\ell/(\pi d)$. How many throws for three correct digits?
2. **Importance sampling.** Draw loads from $N(360, 30)$ instead of $N(300, 30)$ (more failures) and
   correct each sample with the weight $\phi_{300}(L)/\phi_{360}(L)$. How much smaller is the standard
   error for the same N?
3. Simulate 3D walks and confirm $\langle r^2\rangle = n$. Use the result to relate the step length and time
   step of a walker to a diffusion coefficient $D$.
"""),
    ]


# =====================================================================================================
@notebook("19_sparse_2d_pdes")
def _():
    return [
        md(r"""
# 19 · Sparse matrices and 2D PDEs at realistic size

**Problem.** A 2 W chip (20 × 20 mm) sits at the centre of a 100 × 60 mm circuit board, 1.6 mm thick,
whose edges are clamped to a heat-sinking frame at 25 °C. Copper planes give the board an effective
in-plane conductivity of about 20 W/(m·K). How hot does the board get? Steady conduction with a heat
source is the **Poisson equation**
$$-\nabla^2 T = \frac{\dot q}{k},\qquad \dot q = \frac{P}{A_{chip}\,t}\ \text{under the chip, 0 elsewhere}.$$
A grid fine enough to resolve the chip has tens of thousands of unknowns - far too many for dense
matrices, easy with **sparse** ones.
"""),
        code(SETUP + "\nimport time\nimport scipy.sparse as sps\nimport scipy.sparse.linalg as spla\nfrom engmath import sparse"),
        code(r"""Lx, Ly, thick, kb, P, T_edge = 0.100, 0.060, 1.6e-3, 20.0, 2.0, 25.0
q_vol = P / (0.02 * 0.02 * thick)                            # W/m3 under the chip
def source(x, y):                                            # right-hand side q/k
    chip = (np.abs(x - Lx/2) <= 0.01) & (np.abs(y - Ly/2) <= 0.01)
    return np.where(chip, q_vol / kb, 0.0)

nx, ny = 299, 179                                            # interior nodes: dx = dy = 1/3 mm
N = nx * ny
A = sparse.laplacian_2d(nx, ny, Lx/(nx + 1), Ly/(ny + 1))
print(f"unknowns: {N:,}")
print(f"nonzeros: {A.nnz:,} ({A.nnz/N:.2f} per row)")
print(f"memory as a sparse matrix: {(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)/1e6:.1f} MB")
print(f"memory as a dense matrix:  {N**2 * 8 / 1e9:.1f} GB")"""),
        md(r"""
Each row has at most five nonzeros (a node and its four neighbours), so storing only those - a
**sparse** matrix - needs a few megabytes instead of tens of gigabytes. The pattern for a tiny 6 × 4 grid:
"""),
        code(r"""small = sparse.laplacian_2d(6, 4, 1.0, 1.0)
plt.figure(figsize=(4, 4)); plt.spy(small, markersize=5)
plt.title("Nonzero pattern: 5-point Laplacian, 6 × 4 grid"); plt.show()"""),
        md(r"""
## Solving at realistic size
"""),
        code(r"""t0 = time.perf_counter()
X, Y, T = sparse.poisson_dirichlet(source, lambda x, y: T_edge + 0*x, Lx, Ly, nx, ny)
t_direct = time.perf_counter() - t0
t0 = time.perf_counter()
_, _, T_cg = sparse.poisson_dirichlet(source, lambda x, y: T_edge + 0*x, Lx, Ly, nx, ny, solver="cg")
t_cg = time.perf_counter() - t0
print(f"sparse direct (LU): {t_direct:.2f} s;  conjugate gradient: {t_cg:.2f} s;  "
      f"max difference {np.max(np.abs(T - T_cg)):.1e} K")
print(f"maximum board temperature: {T.max():.1f} °C")
fig, ax = plt.subplots(figsize=(8, 4.2))
cs = ax.contourf(X*1000, Y*1000, T, levels=20, cmap="inferno")
fig.colorbar(cs, label="T (°C)")
ax.add_patch(plt.Rectangle((40, 20), 20, 20, fill=False, ec="w", ls="--"))
ax.set(aspect="equal", xlabel="x (mm)", ylabel="y (mm)", title="Steady board temperature (dashed: chip)")
plt.show()"""),
        md(r"""
How does conjugate gradient converge? Each iteration needs only one sparse matrix-vector product, but
the number of iterations grows with the grid size:
"""),
        code(r"""rhs = source(X[1:-1, 1:-1], Y[1:-1, 1:-1]).ravel()
_, hist = sparse.conjugate_gradient(A, rhs, tol=1e-10)
plt.semilogy(hist); plt.xlabel("iteration"); plt.ylabel("relative residual")
plt.title(f"Conjugate gradient: {len(hist) - 1} iterations for {N:,} unknowns"); plt.show()"""),
        md(r"""
For comparison, Gauss-Seidel (notebook 03) needs a number of sweeps that grows with the *square* of the
number of nodes across the board - hundreds of thousands of sweeps here. Sparse direct solvers and
Krylov methods such as CG (usually with a *preconditioner*) are what engineering software uses.

## Is the grid fine enough?
"""),
        code(r"""for n_x, n_y in ((99, 59), (199, 119), (299, 179)):
    _, _, Tg = sparse.poisson_dirichlet(source, lambda x, y: T_edge + 0*x, Lx, Ly, n_x, n_y)
    print(f"grid {n_x + 1:>3} × {n_y + 1:<3} (spacing {Lx/(n_x + 1)*1000:.2f} mm): max T = {Tg.max():.3f} °C")"""),
        md(r"""
The maximum temperature changes little between the two finer grids: the answer is grid-independent to
the accuracy an engineer needs.

## Transient heating after switch-on
With heat capacity, $\rho c\,\partial T/\partial t = k\nabla^2 T + \dot q$. Implicit (backward) Euler in time -
unconditionally stable (notebooks 11 and 12) - needs one sparse solve per step with the *same*
matrix $I + \Delta t\,\alpha A$, so it is factorised once and reused. With $\theta = T - 25$ °C the edges are
at $\theta = 0$:
"""),
        code(r"""rho_c = 1900 * 1100                                         # J/(m3 K), typical board
alpha = kb / rho_c
nxt, nyt = 99, 59
At = sparse.laplacian_2d(nxt, nyt, Lx/(nxt + 1), Ly/(nyt + 1))
Xt, Yt = np.meshgrid(np.linspace(0, Lx, nxt + 2)[1:-1], np.linspace(0, Ly, nyt + 2)[1:-1])
ft = source(Xt, Yt).ravel()
dt, t_end = 1.0, 300.0
lu = spla.splu((sps.identity(nxt*nyt, format="csc") + dt * alpha * At).tocsc())   # factorise once
theta = np.zeros(nxt * nyt)
times, Tmax = [0.0], [T_edge]
for step in range(int(t_end / dt)):
    theta = lu.solve(theta + dt * alpha * ft)
    times.append((step + 1) * dt); Tmax.append(T_edge + theta.max())
theta_ss = spla.spsolve(At.tocsc(), ft)
t63 = times[np.argmax(np.array(Tmax) - T_edge >= 0.632 * theta_ss.max())]
plt.plot(times, Tmax, label="maximum temperature")
plt.axhline(T_edge + theta_ss.max(), color="k", ls="--", label="steady state")
plt.xlabel("time (s)"); plt.ylabel("T (°C)"); plt.legend(); plt.title(f"63 % of the final rise after about {t63:.0f} s")
plt.show()"""),
        md(r"""
## Exercises
1. The board also loses heat from both faces by convection, $h$ = 10 W/(m²·K). This adds a term
   $(2h/(k\,t))\,\theta$ to the equation, i.e. $A \to A + \frac{2h}{k t} I$. How much does the peak temperature drop?
2. Time the direct solver for grids of 100², 200², 400² and 800² nodes (if memory allows). How does the
   time grow with the number of unknowns?
3. Is the 63 % time accurate? Repeat the transient with time steps of 2 s and 0.5 s. How large is the
   time-step error of backward Euler here, and how does it scale with the step?
"""),
    ]


# =====================================================================================================
@notebook("20_working_with_data_files")
def _():
    return [
        md(r"""
# 20 · Working with real data files

In the other notebooks the data were generated in the code. Real work starts with a **file** - from a
data logger, an instrument, a colleague or a public archive - and files have headers, units, missing
values, error codes, gaps and inconsistencies. This notebook works through three files:

1. a NIST reference measurement (**real** ultrasonic-calibration data with certified results),
2. the Mauna Loa CO₂ record (**real**, 1958-2026, with a trap in its header),
3. a messy process-logger file (synthetic, so that the cleaning can be checked).

All three ship with `engmath`: `datasets.path(name)` gives a file's location on your computer, and from
there you read it exactly as you would read your own files.
"""),
        code(SETUP + "\nimport pandas as pd\nfrom engmath import datasets, fitting, fourier"),
        md(r"""
## Rule 1: look at the file before loading it
"""),
        code(r"""path = datasets.path("Chwirut2.dat")
with open(path, encoding="utf-8") as fh:
    lines = fh.readlines()
print(f"{len(lines)} lines. The start of the file:")
print("".join(lines[:16]))
print("... and the data section:")
print("".join(lines[58:64]))"""),
        md(r"""
The file documents itself: 60 lines of description (including NIST's certified answer), then two
columns - response first, then the predictor. Knowing that, `np.loadtxt` reads it in one line:
"""),
        code(r"""data = np.loadtxt(path, skiprows=60)
y, x = data[:, 0], data[:, 1]                           # ultrasonic response, metal distance
print(f"shape {data.shape}; x from {x.min()} to {x.max()}; y from {y.min()} to {y.max()}")
fit = fitting.levenberg_marquardt(lambda x, b1, b2, b3: np.exp(-b1*x) / (b2 + b3*x), x, y, [0.1, 0.01, 0.02])
certified = [1.6657666537e-01, 5.1653291286e-03, 1.2150007096e-02]    # copied from lines 41-43 of the file
for name, b, c in zip(("b1", "b2", "b3"), fit.coef, certified):
    print(f"{name} = {b:.10e}   NIST certified {c:.10e}")
xx = np.linspace(x.min(), x.max(), 200)
plt.plot(x, y, "o", ms=4, label="NIST measurements")
plt.plot(xx, np.exp(-fit.coef[0]*xx) / (fit.coef[1] + fit.coef[2]*xx), label="fitted model")
plt.xlabel("metal distance"); plt.ylabel("ultrasonic response"); plt.legend(); plt.show()"""),
        md(r"""
Reading the real file and fitting it reproduces NIST's certified parameters to about eight significant
digits - far more than the measurement precision can justify, and a good check that both the file
reading and the fitting code are right.

## Rule 2: never trust a header blindly
The Mauna Loa CO₂ record is one of the most important measurement series in science. Load it the
obvious way:
"""),
        code(r"""co2_path = datasets.path("co2_mauna_loa_monthly.csv")
naive = pd.read_csv(co2_path)
print(naive.head(3))"""),
        md(r"""
It looks plausible - but look closely. The header names **6** columns, while every row holds **7**
values. pandas silently turned the first value (the date) into the row *index* and shifted every label
by one column: the column called "Average" really holds the *de-seasonalised* values, and "Decimal Date"
holds the monthly means. No error, no warning - just wrong data under plausible names.
"""),
        code(r"""with open(co2_path, encoding="utf-8") as fh:
    header, first = fh.readline().strip(), fh.readline().strip()
print("header:", header.split(","), f"({len(header.split(','))} names)")
print("row:   ", first.split(","), f"({len(first.split(','))} values)")"""),
        md(r"""
Comparing with the documentation of NOAA's monthly file, the seven values are: date, decimal date,
monthly mean, de-seasonalised mean, number of measurement days, standard deviation of the days and
uncertainty of the monthly mean, with -1, -9.99 and -0.99 marking values that were not recorded. So we
name the columns ourselves:
"""),
        code(r"""cols = ["date", "decimal_year", "co2_ppm", "deseasonalised_ppm", "days", "sd_days", "unc_mean"]
co2 = pd.read_csv(co2_path, header=0, names=cols)
print(co2.head(3)); print("...")
print(co2.tail(2))
print(f"\n{len(co2)} months; months without a day count (-1): {(co2.days < 0).sum()}; "
      f"missing monthly means: {co2.co2_ppm.isna().sum()}")
print("consecutive months, no gaps:", np.allclose(np.diff(co2.decimal_year), 1/12, atol=0.01))"""),
        md(r"""
## Analysing the CO₂ record: trend, growth rate and seasonal cycle
Model the record as a smooth trend plus a yearly cycle (with a second harmonic):
$$c(t) = a_0 + a_1\tau + a_2\tau^2 + \sum_{k=1}^{2}\left[s_k\sin(2\pi k t) + c_k\cos(2\pi k t)\right],\qquad \tau = \frac{t - 1990}{10}.$$
It is linear in the coefficients, so linear least squares does it (notebook 08; the scaled time $\tau$
keeps the problem well conditioned).
"""),
        code(r"""t, c = co2.decimal_year.to_numpy(), co2.co2_ppm.to_numpy()
tau = (t - 1990) / 10
X = np.column_stack([np.ones_like(t), tau, tau**2] + [f(2*np.pi*k*t) for k in (1, 2) for f in (np.sin, np.cos)])
fit = fitting.linear_lstsq(X, c)
trend = X[:, :3] @ fit.coef[:3]
seasonal = X[:, 3:] @ fit.coef[3:]
growth = lambda year: (fit.coef[1] + 2*fit.coef[2]*(year - 1990)/10) / 10      # d(trend)/dt in ppm per year
print(f"growth rate of the trend: {growth(1960):.2f} ppm/yr in 1960, {growth(2025):.2f} ppm/yr in 2025")
print(f"seasonal cycle: peak-to-peak {seasonal.max() - seasonal.min():.1f} ppm")
print(f"residual standard deviation: {fit.sigma:.2f} ppm")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(t, c, lw=0.8, label="monthly mean"); a1.plot(t, trend, "k", label="quadratic trend")
a1.set(xlabel="year", ylabel="CO₂ (ppm)", title="Mauna Loa CO₂"); a1.legend()
year_frac = np.linspace(0, 1, 200)
one_year = np.column_stack([f(2*np.pi*k*year_frac) for k in (1, 2) for f in (np.sin, np.cos)]) @ fit.coef[3:]
a2.plot(12 * year_frac, one_year)
a2.set(xlabel="month of the year", ylabel="seasonal deviation (ppm)", title="The yearly cycle")
plt.show()"""),
        md(r"""
The growth rate has roughly tripled since 1960, and CO₂ swings by several ppm each year as the northern
hemisphere's vegetation grows in summer and decays in winter.

Is the model adequate? The spectrum of the residuals (notebook 16; sampling frequency 12 per year)
checks whether any periodic signal was missed:
"""),
        code(r"""resid = c - X @ fit.coef
f, amp = fourier.spectrum(resid, fs=12.0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
a1.plot(t, resid, lw=0.8); a1.axhline(0, color="k", lw=0.8)
a1.set(xlabel="year", ylabel="residual (ppm)", title="What the model misses")
a2.plot(f, amp); a2.set(xlabel="frequency (cycles per year)", ylabel="amplitude (ppm)", xlim=(0, 3),
                        title="Residual spectrum")
plt.show()
print(f"largest residual amplitude at {f[np.argmax(amp)]:.2f} cycles per year (period {1/f[np.argmax(amp)]:.0f} years)")
print(f"residual amplitude at 1 cycle per year: {amp[np.argmin(abs(f - 1))]:.3f} ppm "
      f"(the fitted seasonal amplitude was about {(seasonal.max() - seasonal.min())/2:.1f} ppm)")"""),
        md(r"""
No yearly peak remains - the seasonal terms captured the cycle. What is left is slow wandering over
years to decades: the quadratic trend is only an approximation to the true growth curve. **Residual
plots tell you where a model is wrong; they are not an afterthought.**

## Rule 3: a messy file needs a documented cleaning procedure
Now a file of the kind data loggers really produce. Read the header lines first - they carry information
the table does not:
"""),
        code(r"""log_path = datasets.path("process_log_messy.csv")
with open(log_path, encoding="utf-8") as fh:
    for line in fh:
        if line.startswith("#"):
            print(line.rstrip())
raw = pd.read_csv(log_path, comment="#", parse_dates=["timestamp"])
print(f"\n{len(raw)} rows; column types:\n{raw.dtypes}")
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(raw.timestamp, raw.temperature, ".", ms=3); ax.set(ylabel="temperature (as logged)", title="Raw logger data")
plt.show()"""),
        md(r"""
The raw plot shows everything that is wrong: a jump at 10:00 (the unit change announced in the header),
error codes at -999, a few spikes, and a gap. Clean step by step, and **count what each step changes**.
Note how the spike threshold is chosen: the genuine spikes lie over a hundred robust standard deviations
from the running median, while ordinary points - even the steep start of the heat-up, where a running
median at the edge of the data is biased - stay well below 20. Look at the numbers before choosing a
threshold; a threshold that is too low deletes real data.
"""),
        code(r"""df = raw.copy()
log = []

n = len(df); df = df.drop_duplicates()
log.append(f"removed {n - len(df)} duplicated rows (retransmissions)")

bad = df.temperature == -999
df.loc[bad, "temperature"] = np.nan                        # BEFORE converting units: -999 degF is not a temperature
log.append(f"replaced {bad.sum()} sensor-fault codes (-999) by NaN")

before_10 = df.timestamp < pd.Timestamp("2026-03-14 10:00")
df.loc[before_10, "temperature"] = (df.loc[before_10, "temperature"] - 32) * 5 / 9
log.append(f"converted {before_10.sum()} readings from degF to degC")
df = df.rename(columns={"temperature": "temperature_degC", "pressure": "pressure_bar"})

med = df.temperature_degC.rolling(7, center=True, min_periods=3).median()
dev = df.temperature_degC - med
mad = 1.4826 * np.nanmedian(np.abs(dev))                   # robust estimate of the noise level
z = np.abs(dev) / mad
print("largest deviations, in robust standard deviations:", np.round(np.sort(z.dropna())[-4:], 1))
spikes = z > 20
df.loc[spikes, "temperature_degC"] = np.nan
log.append(f"removed {spikes.sum()} spikes (more than 20 robust standard deviations from the running median)")

def fill_short_gaps(s, max_len=3):
    # interpolate runs of at most max_len missing values; leave longer gaps empty
    missing = s.isna()
    run_id = missing.ne(missing.shift()).cumsum()
    run_length = missing.groupby(run_id).transform("sum")
    return s.where(~(missing & (run_length <= max_len)), s.interpolate(limit_area="inside"))

df = df.set_index("timestamp").resample("1min").mean()     # regular 1-minute grid; the gap becomes NaN rows
empty_before = df.temperature_degC.isna().sum()
df = df.apply(fill_short_gaps)
log.append(f"placed data on a 1-minute grid ({len(df)} rows); interpolated gaps of at most 3 minutes; "
           f"{df.temperature_degC.isna().sum()} of {empty_before} empty temperature slots left empty "
           f"(the 12 minutes the logger was offline)")
print("\n".join(f"{i}. {entry}" for i, entry in enumerate(log, 1)))"""),
        md(r"""
Two principles: **keep the raw file untouched** (all changes happen in code, which documents them), and
**never invent data across a long gap** - the 12 minutes the logger was offline stay empty.

Because this file is synthetic we can check the cleaning: it was generated from a heat-up curve
$T(t) = T_\infty - (T_\infty - T_0)\,e^{-t/\tau}$ with $T_0$ = 20 °C, $T_\infty$ = 80 °C and $\tau$ = 45 min.
Fitting that model (notebook 09) to raw and cleaned data:
"""),
        code(r"""model = lambda tm, T0, Tinf, tau: Tinf - (Tinf - T0) * np.exp(-tm / tau)
ok = df.temperature_degC.notna()
minutes = (df.index[ok] - df.index[0]).total_seconds().to_numpy() / 60
clean_fit = fitting.levenberg_marquardt(model, minutes, df.temperature_degC[ok].to_numpy(), [25, 75, 30])
raw_ok = raw.temperature.notna()
raw_minutes = (raw.timestamp[raw_ok] - raw.timestamp.iloc[0]).dt.total_seconds().to_numpy() / 60
raw_fit = fitting.levenberg_marquardt(model, raw_minutes, raw.temperature[raw_ok].to_numpy(), [25, 75, 30])
for name, fr in (("raw data", raw_fit), ("cleaned data", clean_fit)):
    T0f, Tinff, tauf = fr.coef
    print(f"{name:<13}: T0 = {T0f:7.2f} °C, T_inf = {Tinff:6.2f} °C, tau = {tauf:6.2f} min, residual SD {fr.sigma:6.2f} K")
print("true values  : T0 =   20.00 °C, T_inf =  80.00 °C, tau =  45.00 min, noise SD 0.30 K")
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(df.index, df.temperature_degC, ".", ms=3, label="cleaned")
ax.plot(df.index[ok], model(minutes, *clean_fit.coef), "k", lw=1, label="fitted heat-up curve")
ax.set(ylabel="temperature (°C)"); ax.legend(); plt.show()"""),
        md(r"""
Without cleaning, the fitted parameters are meaningless. After cleaning they agree with the values used
to generate the file, and the residual standard deviation equals the sensor noise.

## Saving results
Put units in column names, write the cleaning log next to the data, and keep the raw file as it was:
"""),
        code(r"""import tempfile
from pathlib import Path
out_dir = Path(tempfile.gettempdir())
df.to_csv(out_dir / "process_log_clean.csv", float_format="%.3f")
(out_dir / "process_log_clean_README.txt").write_text("Cleaning steps:\n" + "\n".join(log), encoding="utf-8")
print(open(out_dir / "process_log_clean.csv", encoding="utf-8").read()[:230])"""),
        md(r"""
## Checklist for any data file
1. Open it as text and read the header, comments and documentation.
2. Check shapes, column types, ranges and units - and whether labels match the values.
3. Plot the raw data before doing anything else.
4. Clean in code, step by step, recording what each step changed. Never edit the raw file.
5. Treat missing values explicitly; do not fill long gaps.
6. Plot again, and check residuals of any model you fit.

## Exercises
1. **Copper's thermal expansion.** Read `Hahn1.dat` (NIST, real). Plot the coefficient of thermal
   expansion against temperature. Fit the certified 7-parameter rational model
   $y = (b_1 + b_2x + b_3x^2 + b_4x^3)/(1 + b_5x + b_6x^2 + b_7x^3)$ from NIST's first starting values (in
   the file) and compare with the certified values.
2. **El Niño.** Read `ENSO.dat` (NIST, real: monthly pressure differences between Easter Island and
   Darwin). Compute its spectrum. Besides the annual cycle, at which periods (in months) are there
   peaks? Compare with the certified periods $b_4$ and $b_7$ in the file header.
3. Extrapolate the CO₂ trend to find when the trend first exceeds 450 ppm. Refit using only data up to
   2000 and extrapolate again: how far off would that forecast have been for 2026? What does this say
   about extrapolating fitted trends?
"""),
    ]


# =====================================================================================================
# Teaching layer added to every notebook: learning objectives, prerequisites and time (after the title);
# an "Inside the algorithm" section (inserted before the cell containing `anchor`); and an
# "Implement it yourself" exercise appended to the exercises.
EXTRAS = {
    "00_python_numpy_primer": dict(
        objectives=["create NumPy arrays and compute with them without loops (vectorisation)",
                    "write reusable functions with docstrings",
                    "integrate sampled data with the trapezoidal rule and check the result against an exact value",
                    "make a labelled plot"],
        prereq="basic Python syntax (variables, lists, `for` loops - e.g. the official Python tutorial); integrals",
        time="45 min", anchor="## Plotting",
        inside=[md(r"""
## Inside the algorithm: the trapezoidal rule
The area under sampled data is approximated by trapezoids,
$\int f\,dx \approx \sum_i \tfrac12 (f_i + f_{i+1})(x_{i+1} - x_i)$. Here it is twice - as a loop (how
you would write it on paper) and vectorised (how NumPy wants it):
"""), code(r"""
def trapezoid_loop(y, x):
    total = 0.0
    for i in range(len(x) - 1):
        total += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    return total

def trapezoid_vec(y, x):
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))

r = np.linspace(0, R, 101)
integrand = u_max * (1 - r**2 / R**2) * 2 * np.pi * r
print(trapezoid_loop(integrand, r), trapezoid_vec(integrand, r), engmath.integrate.trapezoid(integrand, r))
"""), md(r"""
All three agree. `y[1:]` and `y[:-1]` are the array shifted by one element - the standard NumPy
idiom for "next value" and "current value". The library function is the same idea:
"""), code("show_source(engmath.integrate.trapezoid)")],
        exercise="**Implement it yourself:** write `midpoint(f, a, b, n)` for the midpoint rule, first with a "
                 "loop and then vectorised. Time both for n = 10⁶ and compare their accuracy with the "
                 "trapezoidal rule on the laminar profile."),

    "01_errors_and_uncertainty": dict(
        objectives=["distinguish round-off, truncation and measurement errors",
                    "choose a sensible finite-difference step",
                    "propagate uncertainties linearly and by Monte Carlo, and read an uncertainty budget",
                    "report a result with significant figures consistent with its uncertainty"],
        prereq="notebook 00; partial derivatives; mean and standard deviation", time="60 min",
        anchor="### Checking the linear approximation with Monte Carlo",
        inside=[md(r"""
## Inside the algorithm: first-order propagation
Each sensitivity coefficient $c_i = \partial\eta/\partial x_i$ is estimated by a central difference, and the
contributions $(c_i u_i)^2$ are added:
"""), code(r"""
def propagate_by_hand(func, values, unc):
    y0 = func(**values)
    variance = 0.0
    for name, x in values.items():
        h = 1e-6 * abs(x)
        up = {**values, name: x + h}
        down = {**values, name: x - h}
        c = (func(**up) - func(**down)) / (2 * h)      # sensitivity coefficient
        variance += (c * unc[name])**2
    return y0, float(np.sqrt(variance))

print("by hand:", propagate_by_hand(efficiency, values, unc))
print("library:", engmath.reporting.propagate(efficiency, values, unc)[:2])
""")],
        exercise="**Implement it yourself:** extend `propagate_by_hand` to correlated inputs, "
                 "$u^2 = \\sum_i\\sum_j c_i u_i\\, R_{ij}\\, c_j u_j$ with a correlation matrix $R$. How does "
                 "$u(\\eta)$ change if the errors in $Q$ and $H$ have correlation $+0.5$? And $-0.5$?"),

    "02_root_finding": dict(
        objectives=["rewrite an implicit engineering equation as g(x) = 0 and plot it before solving",
                    "implement Newton's method and bisection, and choose between root finders",
                    "measure the order of convergence from the iteration history",
                    "recognise and avoid divergence"],
        prereq="notebook 00; derivatives; logarithms", time="60-75 min", anchor="## Four methods",
        inside=[md(r"""
## Inside the algorithms
Newton's method replaces the curve by its tangent: $x_{k+1} = x_k - g(x_k)/g'(x_k)$. Bisection keeps
halving a bracket that contains a sign change. Both fit in a few lines:
"""), code(r"""
def newton_by_hand(g, x, tol=1e-12, h=1e-8):
    for k in range(50):
        slope = (g(x + h) - g(x - h)) / (2 * h)     # numerical derivative
        step = g(x) / slope
        x = x - step
        print(f"iteration {k+1}: x = {x:.15f}")
        if abs(step) < tol:
            return x
    raise RuntimeError("Newton did not converge")

def bisection_by_hand(g, a, b, tol=1e-12):
    while b - a > tol:
        m = (a + b) / 2
        if g(a) * g(m) <= 0:
            b = m          # the sign change is in the left half
        else:
            a = m          # ... or in the right half
    return (a + b) / 2

newton_by_hand(colebrook, 0.02)
print("bisection:", bisection_by_hand(colebrook, 0.005, 0.08))
"""), md(r"""
Watch the digits: each Newton iteration roughly doubles the number of correct ones. The library
versions add safeguards (zero slopes, non-finite values) and record the history:
"""), code("show_source(roots.newton)")],
        exercise="**Implement it yourself:** write the *regula falsi* (false position) method: bisection, but "
                 "split the bracket where the straight line through $(a, g(a))$ and $(b, g(b))$ crosses zero. "
                 "Compare its iteration count with bisection on the Colebrook equation. Why can it be slow when "
                 "one end of the bracket never moves?"),

    "03_linear_systems": dict(
        objectives=["discretise a 1D boundary-value problem into a linear system",
                    "implement and exploit the tridiagonal (Thomas) algorithm",
                    "solve a 2D problem iteratively and judge convergence correctly",
                    "compute natural frequencies as eigenvalues"],
        prereq="notebook 00; matrices and Gaussian elimination; second derivatives", time="60-75 min",
        anchor="## Why the Thomas algorithm matters",
        inside=[md(r"""
## Inside the algorithm: the Thomas algorithm
Gaussian elimination on a tridiagonal matrix only ever touches three diagonals. A forward sweep
eliminates the lower diagonal; back substitution then solves from the last unknown upwards:
"""), code(r"""
def thomas_by_hand(a, b, c, d):
    # a: lower, b: main, c: upper diagonal (a[0] and c[-1] unused); d: right-hand side
    n = len(d)
    c2, d2 = np.zeros(n), np.zeros(n)
    c2[0], d2[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):                          # forward sweep
        denom = b[i] - a[i] * c2[i-1]
        c2[i] = c[i] / denom if i < n - 1 else 0.0
        d2[i] = (d[i] - a[i] * d2[i-1]) / denom
    x = np.zeros(n)
    x[-1] = d2[-1]
    for i in range(n - 2, -1, -1):                 # back substitution
        x[i] = d2[i] - c2[i] * x[i+1]
    return x

print("max difference from the library:", np.max(np.abs(thomas_by_hand(lower, diag, upper, rhs) - T_inner)))
""")],
        exercise="**Implement it yourself:** add successive over-relaxation (SOR) to the 2D plate loop: "
                 "compute the Gauss-Seidel value $T_{GS}$, then set $T \\leftarrow T + \\omega(T_{GS} - T)$. "
                 "Find the $\\omega$ between 1.0 and 1.95 that needs the fewest sweeps."),

    "04_interpolation": dict(
        objectives=["interpolate tabulated data linearly and with cubic splines",
                    "build a natural cubic spline from its defining equations",
                    "improve accuracy by interpolating a transformed variable",
                    "recognise Runge oscillations and the danger of extrapolation"],
        prereq="notebooks 00 and 03 (tridiagonal systems); polynomials", time="45-60 min",
        anchor="## A better idea: interpolate a smoother quantity",
        inside=[md(r"""
## Inside the algorithm: a natural cubic spline
Between neighbouring points the spline is a cubic. Requiring continuous first and second derivatives
gives one equation per interior point for the second derivatives $M_i$:
$$h_{i-1}M_{i-1} + 2(h_{i-1}+h_i)M_i + h_iM_{i+1} = 6\left(\frac{y_{i+1}-y_i}{h_i} - \frac{y_i-y_{i-1}}{h_{i-1}}\right),$$
with $M_0 = M_n = 0$ at the ends (the "natural" spline). Another tridiagonal system (notebook 03):
"""), code(r"""
def spline_by_hand(x, y, xq):
    h = np.diff(x)
    n = len(x)
    A, rhs = np.zeros((n, n)), np.zeros(n)
    A[0, 0] = A[-1, -1] = 1.0                      # natural ends: M_0 = M_n = 0
    for i in range(1, n - 1):
        A[i, i-1], A[i, i], A[i, i+1] = h[i-1], 2 * (h[i-1] + h[i]), h[i]
        rhs[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    M = np.linalg.solve(A, rhs)
    i = np.clip(np.searchsorted(x, xq) - 1, 0, n - 2)   # interval containing each query point
    a, b, hi = x[i+1] - xq, xq - x[i], h[i]
    return ((M[i] * a**3 + M[i+1] * b**3) / (6 * hi)
            + (y[i] / hi - M[i] * hi / 6) * a + (y[i+1] / hi - M[i+1] * hi / 6) * b)

print(f"by hand {float(spline_by_hand(T_tab, p_tab, T_q)):.6f} kPa,  library {float(p_spl):.6f} kPa")
""")],
        exercise="**Implement it yourself:** turn `spline_by_hand` into a *clamped* spline with prescribed end "
                 "slopes $y'_0$ and $y'_n$ (only the first and last rows of the system change - derive them). "
                 "Use slopes from the Antoine equation: does the error near the ends of the table fall?"),

    "05_integration": dict(
        objectives=["integrate sampled data with the trapezoid rule, Simpson's rule and splines",
                    "implement composite Simpson's rule",
                    "measure an order of accuracy from a log-log plot",
                    "recognise when the sampling interval, not the rule, limits the accuracy"],
        prereq="notebooks 00 and 04; Taylor series (for why errors scale with h)", time="45-60 min",
        anchor="## Order of accuracy",
        inside=[md(r"""
## Inside the algorithm: Simpson's rule
Simpson's rule fits a parabola through each consecutive triple of equally spaced points; integrating the
parabola gives the weights $\tfrac{h}{3}(1, 4, 1)$. Added up over the whole range, the weights become
$\tfrac{h}{3}(1, 4, 2, 4, \ldots, 2, 4, 1)$:
"""), code(r"""
def simpson_by_hand(f, a, b, n):
    # composite Simpson's rule with n (even) equal intervals
    x = np.linspace(a, b, n + 1)
    w = np.ones(n + 1)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    return (b - a) / n / 3 * np.sum(w * f(x))

for n in (16, 64):
    print(f"n = {n}: by hand {simpson_by_hand(power, 0, 8, n):.10f}, library {integrate.composite(power, 0, 8, n, 'simpson'):.10f}")
"""), md("The library version also handles *unevenly* spaced data, giving each pair of intervals its own parabola:"),
            code("show_source(integrate.simpson)")],
        exercise="**Implement it yourself:** Boole's rule applies the weights $\\tfrac{2h}{45}(7, 32, 12, 32, 7)$ "
                 "to groups of four intervals. Implement it, measure its order on the power profile (theory: 6), "
                 "and explain why it still cannot capture the start-up surge from 30-minute data."),

    "06_differentiation": dict(
        objectives=["differentiate unevenly spaced data to second-order accuracy",
                    "derive a finite-difference formula from an interpolating parabola",
                    "estimate a rate-law exponent by the differential method",
                    "quantify how differentiation amplifies measurement noise"],
        prereq="notebook 00; Taylor series; a straight-line fit (explained fully in notebook 08)", time="45 min",
        anchor="## The catch: differentiation amplifies noise",
        inside=[md(r"""
## Inside the algorithm: derivatives on an uneven grid
Fit a parabola through three neighbouring points with spacings $h_1 = t_i - t_{i-1}$ and
$h_2 = t_{i+1} - t_i$, and differentiate it at $t_i$:
$$T'(t_i) \approx -\frac{h_2}{h_1(h_1+h_2)}T_{i-1} + \frac{h_2 - h_1}{h_1 h_2}T_i + \frac{h_1}{h_2(h_1+h_2)}T_{i+1}.$$
For equal spacing ($h_1 = h_2 = h$) this reduces to the familiar central difference $(T_{i+1} - T_{i-1})/2h$.
"""), code(r"""
def interior_derivative(t, T):
    d = np.full(len(t), np.nan)
    for i in range(1, len(t) - 1):
        h1, h2 = t[i] - t[i-1], t[i+1] - t[i]
        d[i] = (-h2 / (h1 * (h1 + h2)) * T[i-1] + (h2 - h1) / (h1 * h2) * T[i]
                + h1 / (h2 * (h1 + h2)) * T[i+1])
    return d

print("max difference from the library at interior points:", np.nanmax(np.abs(interior_derivative(t, T) - dTdt)))
""")],
        exercise="**Implement it yourself:** using the same parabola, derive the one-sided formula for the "
                 "*first* point (differentiate at $t_0$ instead of $t_1$). Implement it and check it against "
                 "`differentiate.gradient` at $t = 0$."),

    "07_smoothing": dict(
        objectives=["compare moving-average, Savitzky-Golay and LOWESS smoothing",
                    "derive Savitzky-Golay weights from local least squares",
                    "choose a window size by balancing noise against distortion",
                    "locate peaks from smoothed derivatives"],
        prereq="notebook 00; least squares (notebook 08 is helpful)", time="45 min",
        anchor="## The window-size trade-off",
        inside=[md(r"""
## Inside the algorithm: where Savitzky-Golay weights come from
Fit a polynomial by least squares to the $2m + 1$ points of a window and evaluate it at the centre.
Because the fit is linear in the data, the result is a *fixed weighted average* of the window - a row of
the pseudo-inverse of the Vandermonde matrix. For 5 points and a quadratic, this gives the classic
weights $(-3, 12, 17, 12, -3)/35$:
"""), code(r"""
m, order = 2, 2
t_win = np.arange(-m, m + 1)
V = np.vander(t_win, order + 1, increasing=True)    # columns: 1, t, t^2
P = np.linalg.pinv(V)
print("smoothing weights x 35:", np.round(P[0] * 35, 6) + 0.0)          # + 0.0 turns -0.0 into 0.0
print("first-derivative weights x 10:", np.round(P[1] * 10, 6) + 0.0)
"""), md(r"""
Smoothing is then a convolution with these weights, and the derivative uses the second row - exactly
what `smoothing.savitzky_golay` does:
"""), code("show_source(smoothing.savitzky_golay)")],
        exercise="**Implement it yourself:** write `exponential_smoothing(y, a)` with "
                 "$s_i = a\\,y_i + (1 - a)\\,s_{i-1}$, the filter used in many real-time instruments. Apply it to "
                 "the peaks. Why does it shift them to later times, unlike the centred filters?"),

    "08_linear_regression": dict(
        objectives=["fit models that are linear in their coefficients",
                    "compute least squares via the normal equations and via QR, with standard errors",
                    "judge a model from residual plots and coefficient confidence intervals",
                    "use robust regression to flag outliers"],
        prereq="notebooks 00 and 03; standard deviation and confidence intervals", time="60 min",
        anchor="## How the fit is computed",
        inside=[md(r"""
## Inside the algorithm: least squares in a few lines
Minimising $\lVert\mathbf y - X\mathbf b\rVert^2$ leads to the normal equations $X^TX\,\mathbf b = X^T\mathbf y$.
A QR factorisation $X = QR$ solves the same problem as $R\,\mathbf b = Q^T\mathbf y$ without forming
$X^TX$. The standard errors follow from $s^2 (X^TX)^{-1}$, where $s$ is the residual standard deviation:
"""), code(r"""
b_normal = np.linalg.solve(X2.T @ X2, X2.T @ E)          # normal equations
Q, Rm = np.linalg.qr(X2)
b_qr = np.linalg.solve(Rm, Q.T @ E)                      # QR factorisation
s = np.sqrt(np.sum((E - X2 @ b_qr)**2) / (len(E) - 3))   # residual standard deviation, n - p dof
se = s * np.sqrt(np.diag(np.linalg.inv(X2.T @ X2)))
print("normal equations:", b_normal)
print("QR:              ", b_qr)
print("library:         ", quad.coef)
print("standard errors: ", se, "  library:", quad.se)
""")],
        exercise="**Implement it yourself:** write the Huber IRLS loop: compute residuals $r$, a robust scale "
                 "$s = \\mathrm{MAD}/0.6745$, weights $w = \\min(1, 1.345/|r/s|)$; refit by weighted least squares; "
                 "repeat until the coefficients stop changing. Compare with `fitting.huber_irls`."),

    "09_nonlinear_regression": dict(
        objectives=["fit a nonlinear model and obtain standard errors of its parameters",
                    "follow Levenberg-Marquardt iterations and the role of damping",
                    "interpret parameter correlations and choose starting values",
                    "decide whether an extra parameter is justified"],
        prereq="notebook 08; the idea of a Jacobian (matrix of partial derivatives)", time="60 min",
        anchor="## The parameters are nearly impossible to separate",
        inside=[md(r"""
## Inside the algorithm: Levenberg-Marquardt, one iteration at a time
Linearise the model about the current parameters, $f(\mathbf p + \boldsymbol\delta) \approx f(\mathbf p) + J\boldsymbol\delta$,
with the Jacobian $J_{ij} = \partial f(x_i)/\partial p_j$. Least squares on the linearised model gives the
Gauss-Newton step; Levenberg-Marquardt adds damping $\lambda$:
$$\left(J^TJ + \lambda\,\mathrm{diag}(J^TJ)\right)\boldsymbol\delta = J^T\mathbf r .$$
If a step lowers the RSS it is accepted and $\lambda$ is reduced (trust the linearisation more);
otherwise it is rejected and $\lambda$ is increased (shorter, steepest-descent-like steps).
"""), code(r"""
def jacobian_by_hand(f, x, params, rel=1e-7):
    J = np.empty((len(x), len(params)))
    for j in range(len(params)):
        dp = rel * max(abs(params[j]), 1e-8)
        shifted = np.array(params, float)
        shifted[j] += dp
        J[:, j] = (f(x, *shifted) - f(x, *params)) / dp
    return J

y_obs = np.log10(p)
params, lam = np.array([8.0, 1500.0, 220.0]), 1e-3
rss = np.sum((y_obs - antoine_log(T, *params))**2)
for it in range(1, 11):
    J = jacobian_by_hand(antoine_log, T, params)
    A = J.T @ J
    step = np.linalg.solve(A + lam * np.diag(np.diag(A)), J.T @ (y_obs - antoine_log(T, *params)))
    new_rss = np.sum((y_obs - antoine_log(T, *(params + step)))**2)
    if new_rss < rss:
        params, rss, lam, verdict = params + step, new_rss, lam / 10, "accept"
    else:
        lam, verdict = lam * 10, "reject"
    print(f"{it:2d} {verdict}  lambda = {lam:7.1e}  RSS = {rss:.6e}  A, B, C = {np.round(params, 3)}")
"""), md(r"""
Each accepted step lowers the RSS and shrinks $\lambda$, so the method turns into fast Gauss-Newton near
the optimum. Once the RSS cannot improve any further (within rounding error), steps are rejected - the
signal to stop. This loop is the whole algorithm; the library version adds a convergence test, a
guard against non-finite values and the covariance matrix for the standard errors.
""")],
        exercise="**Implement it yourself:** run the hand-written loop for 40 iterations from $(7.5, 1300, 180)$ "
                 "and count the rejected steps - why does it crawl, given the parameter correlations? Then remove "
                 "the damping ($\\lambda = 0$, pure Gauss-Newton) and try both starting points. What does the "
                 "damping buy you?"),

    "10_implicit_models_optimisation": dict(
        objectives=["fit a model that needs a root solve inside every evaluation",
                    "minimise one parameter by golden-section search and several by Nelder-Mead",
                    "choose relative or absolute residuals",
                    "read an objective-function landscape to diagnose correlated parameters"],
        prereq="notebooks 02 (Colebrook equation) and 09", time="60 min",
        anchor="## Two parameters: the Nelder-Mead simplex",
        inside=[md(r"""
## Inside the algorithm: golden-section search
Keep two interior points that divide the bracket in the golden ratio. Compare their function values and
discard the part of the bracket that cannot contain the minimum. One old point is always reused, so each
step costs only **one** new evaluation - important when every evaluation hides eight Colebrook solves:
"""), code(r"""
def golden_by_hand(f, a, b, tol=1e-6):
    g = (np.sqrt(5) - 1) / 2                   # 0.618...
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    evaluations = 2
    while b - a > tol:
        if fc < fd:                            # the minimum lies in [a, d]
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:                                  # the minimum lies in [c, b]
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
        evaluations += 1
    return (a + b) / 2, evaluations

x_min, n_eval = golden_by_hand(rss, -6, -2.5)
print(f"roughness {10**x_min*1000:.3f} mm after {n_eval} evaluations (library: {10**r1.x[0]*1000:.3f} mm)")
""")],
        exercise="**Implement it yourself:** a grid search over $\\log_{10}\\varepsilon \\in [-6, -2.5]$ that reaches "
                 "the same final precision needs how many evaluations? Implement it and compare with golden-section "
                 "search."),

    "11_odes": dict(
        objectives=["solve systems of ODEs with Euler and Runge-Kutta methods",
                    "implement a Runge-Kutta 4 step and verify its order against an exact solution",
                    "use adaptive step-size control",
                    "recognise stiffness and use an implicit method"],
        prereq="notebook 00 and 10 (golden-section search); first-order reaction kinetics", time="60-75 min",
        anchor="## Optimal batch time",
        inside=[md(r"""
## Inside the algorithm: one Runge-Kutta 4 step
RK4 samples the slope four times per step - at the start, twice at the midpoint and at the end - and
averages them with weights 1, 2, 2, 1:
"""), code(r"""
def rk4_step(f, t, y, h):
    k1 = np.array(f(t, y))
    k2 = np.array(f(t + h/2, y + h/2 * k1))
    k3 = np.array(f(t + h/2, y + h/2 * k2))
    k4 = np.array(f(t + h, y + h * k3))
    return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

y, h = np.array([CA0, 0.0, 0.0]), 0.1
for step in range(200):
    y = rk4_step(rhs, step * h, y, h)
print("by hand:", y)
print("library:", c[-1])
""")],
        exercise="**Implement it yourself:** Heun's method predicts with an Euler step and corrects with the "
                 "average of the start and end slopes. Implement it, measure its order on the series reaction, "
                 "and place it between Euler and RK4."),

    "12_diffusion_pde": dict(
        objectives=["discretise the diffusion equation with explicit and Crank-Nicolson schemes",
                    "explain and demonstrate the stability limit r <= 1/2",
                    "distinguish stability from accuracy",
                    "estimate a material property from measurements (an inverse problem)"],
        prereq="notebooks 03 (tridiagonal systems), 06 (finite differences) and 10 (golden section)",
        time="60-75 min", anchor="## Stability of the explicit scheme",
        inside=[md(r"""
## Inside the algorithm: the explicit (FTCS) scheme
Replace $\partial T/\partial t$ by a forward difference and $\partial^2T/\partial x^2$ by a central difference:
$$T_i^{n+1} = T_i^n + r\,(T_{i+1}^n - 2T_i^n + T_{i-1}^n), \qquad r = \frac{\alpha\,\Delta t}{\Delta x^2}.$$
In NumPy the update of all interior nodes is a single line:
"""), code(r"""
def ftcs_by_hand(T, r, steps):
    T = T.copy()
    for _ in range(steps):
        T[1:-1] = T[1:-1] + r * (T[2:] - 2 * T[1:-1] + T[:-2])
    return T

dt = 0.4 * dx**2 / alpha
steps = int(round(20 / dt))
T_start = np.full(nx, T0)
T_start[0] = Ts
diff = np.max(np.abs(ftcs_by_hand(T_start, 0.4, steps) - pdes.ftcs(np.full(nx, T0), alpha, dx, dt, steps, Ts, T0)))
print(f"difference from the library: {diff:.1e} K")
"""), md(r"""
**Why $r \le 1/2$?** Rewrite the update as $T_i^{n+1} = r\,T_{i-1}^n + (1 - 2r)\,T_i^n + r\,T_{i+1}^n$. For
$r \le 1/2$ all three weights are non-negative and sum to 1, so each new temperature is a weighted
*average* of old ones and can never overshoot. For $r > 1/2$ the middle weight is negative - and a
growing sawtooth appears, as the next section shows.
""")],
        exercise="**Implement it yourself:** implicit (backward) Euler for the heat equation solves "
                 "$(1 + 2r)T_i^{n+1} - rT_{i-1}^{n+1} - rT_{i+1}^{n+1} = T_i^n$ - one Thomas solve per step. Show "
                 "that it is stable at $r = 100$ and compare its early-time error with Crank-Nicolson."),
}


def apply_extras(name, cells):
    """Insert the teaching layer into a notebook's cells."""
    ex = EXTRAS[name]
    objectives = "\n".join(f"- {o}" for o in ex["objectives"])
    header = md(f"""
**What you will learn**

{objectives}

**Before you start:** {ex['prereq']}  ·  **Time:** about {ex['time']}

Run the cells in order (Shift + Enter). Then change the numbers and run them again - that is how the
ideas sink in.
""")
    cells = [cells[0], header, *cells[1:]]
    idx = next(i for i, c in enumerate(cells) if c.cell_type == "markdown" and ex["anchor"] in c.source)
    cells[idx:idx] = ex["inside"]
    ex_cells = [c for c in cells if c.cell_type == "markdown" and "## Exercises" in c.source]
    if len(ex_cells) != 1:
        raise ValueError(f"{name}: expected exactly one '## Exercises' section, found {len(ex_cells)}")
    c = ex_cells[0]  # the exercises are always the last section of their cell
    n = len(re.findall(r"^\d+\.", c.source.split("## Exercises")[1], re.M))
    c.source = c.source.rstrip() + f"\n{n + 1}. {ex['exercise']}"
    return cells


# teaching layer for notebooks 13-19
EXTRAS.update({
    "13_symbolic_sympy": dict(
        objectives=["compute derivatives, integrals, series and exact solutions of ODEs with SymPy",
                    "derive an engineering formula (beam deflection) symbolically",
                    "derive finite-difference and quadrature formulas together with their error terms",
                    "turn symbolic results into NumPy functions and use them to test numerical code"],
        prereq="notebooks 00, 05 and 06; Taylor series", time="60 min", anchor="## Deriving numerical methods",
        inside=[md(r"""
## Inside the algorithm: what `lambdify` produces
`lambdify` writes an ordinary Python function whose body is the formula in NumPy syntax - you can read
the generated code:
"""), code(r"""import inspect
print(inspect.getsource(v_num))""")],
        exercise="**Implement it yourself:** use the Taylor-series technique to derive the 3-point formula for the "
                 "*second* derivative on an uneven grid and its leading error term. Is it still second order? "
                 "Compare with `differentiate.second_derivative`."),
    "14_nonlinear_systems": dict(
        objectives=["formulate an engineering network as a system of nonlinear equations",
                    "implement Newton's method for systems and observe quadratic convergence",
                    "check a solution by an independent formulation",
                    "understand basins of attraction and why line searches help"],
        prereq="notebooks 02 (root finding) and 03 (linear systems); partial derivatives", time="60 min",
        anchor="## Checking the answer independently",
        inside=[md(r"""
## Inside the algorithm: Newton's method for systems
Each iteration: evaluate $F$, build the Jacobian column by column with finite differences, solve
$J\,\Delta x = -F$, update:
"""), code(r"""def newton_by_hand(F, x, tol=1e-10):
    x = np.array(x, float)
    for it in range(1, 30):
        f = F(x)
        J = np.empty((len(f), len(x)))
        for j in range(len(x)):                        # finite-difference Jacobian, one column at a time
            dx = 1e-7 * max(abs(x[j]), 1.0)
            xp = x.copy(); xp[j] += dx
            J[:, j] = (F(xp) - f) / dx
        x = x + np.linalg.solve(J, -f)
        print(f"iteration {it}: ||F|| = {np.linalg.norm(F(x)):.2e}")
        if np.linalg.norm(F(x)) < tol:
            return x

x_hand = newton_by_hand(F, [90.0, 0.1, 0.0, -0.1])
print("max difference from the library:", np.max(np.abs(x_hand - sol.x)))""")],
        exercise="**Implement it yourself:** Broyden's method avoids recomputing the Jacobian: after each step "
                 "update it with $J \\leftarrow J + \\frac{(\\Delta F - J\\Delta x)\\Delta x^T}{\\Delta x^T\\Delta x}$. Implement it for the "
                 "reservoir problem and compare the number of function evaluations with Newton's method."),
    "15_boundary_value_problems": dict(
        objectives=["distinguish boundary-value from initial-value problems",
                    "solve a BVP by shooting and by finite differences, including a convective boundary",
                    "verify second-order accuracy against an exact solution",
                    "solve a nonlinear BVP as a nonlinear system"],
        prereq="notebooks 02, 03, 06, 11 and 14; fin heat transfer (helpful)", time="60-75 min",
        anchor="## Method 2: finite differences",
        inside=[md(r"""
## Inside the algorithm: why linear shooting needs only two shots
For a *linear* ODE the tip miss is a linear function of the guessed slope $s$. Two shots therefore
determine it exactly, and a straight line through them gives the correct slope - which is why the secant
iteration above converged in one step:
"""), code(r"""def miss(s):
    _, u = odes.rk4(rhs, [Tb, s], 0, L, 400)
    return tip_miss(u[-1, 0], u[-1, 1])

s0, s1 = -2000.0, -1000.0
m0, m1 = miss(s0), miss(s1)
s_linear = s0 - m0 * (s1 - s0) / (m1 - m0)            # where the straight line through both misses is zero
print(f"two shots + interpolation: T'(0) = {s_linear:.6f} K/m;  library: {s_star:.6f} K/m")""")],
        exercise="**Implement it yourself:** write a finite-difference solver for $y'' = r(x)$ with Dirichlet ends that "
                 "builds the full (dense) matrix and uses `np.linalg.solve`. Compare its time with "
                 "`bvp.finite_difference` (Thomas algorithm) for n = 500, 1000 and 2000."),
    "16_fourier_analysis": dict(
        objectives=["compute a DFT from its definition and understand how the FFT speeds it up",
                    "read an amplitude spectrum to identify machine faults",
                    "explain leakage, windowing, frequency resolution and aliasing",
                    "filter a signal in the frequency domain"],
        prereq="notebook 00; complex exponentials $e^{i\\theta} = \\cos\\theta + i\\sin\\theta$", time="60 min",
        anchor="## The amplitude spectrum",
        inside=[md(r"""
## Inside the algorithm: the DFT as a matrix, and the FFT butterfly
The DFT is a matrix-vector product with $W_{kn} = e^{-2\pi i kn/N}$. The FFT computes the same product by
splitting the samples into even and odd ones, transforming each half (recursively), and combining the
two halves with the *twiddle factors* $e^{-2\pi i k/N}$:
"""), code(r"""N = 8
n = np.arange(N)
W = np.exp(-2j * np.pi * np.outer(n, n) / N)            # the DFT matrix
x8 = signal[:N]
print("DFT matrix product matches numpy:", np.allclose(W @ x8, np.fft.fft(x8)))
show_source(fourier.fft)""")],
        exercise="**Implement it yourself:** compute the frequency response of a 5-point moving average (notebook 07) "
                 "by taking the FFT of its kernel zero-padded to 1024 points. Which frequencies does it suppress, "
                 "and why is it a poor low-pass filter?"),
    "17_optimisation_constraints_lp": dict(
        objectives=["formulate a production-planning problem as a linear program and solve it by the simplex method",
                    "interpret shadow prices",
                    "solve a nonlinear constrained design problem with the penalty method",
                    "recognise and fix formulation pitfalls (unphysical regions, badly scaled penalties)"],
        prereq="notebooks 10 (Nelder-Mead) and 13 (SymPy); systems of linear inequalities", time="60-75 min",
        anchor="## What are the resources worth?",
        inside=[md(r"""
## Inside the algorithm: one simplex step
The simplex tableau holds the constraints (with slack variables) and the objective row. The *entering*
variable is the one with the most negative objective coefficient (the steepest profit increase); the
*leaving* variable follows from the ratio test - the first constraint to become binding:
"""), code(r"""T = np.zeros((4, 6))
T[:3, :2], T[:3, 2:5], T[:3, -1] = A, np.eye(3), b
T[-1, :2] = [-ci for ci in c]
print("initial tableau (columns x1, x2, s1, s2, s3 | rhs):\n", T)
col = int(np.argmin(T[-1, :-1]))
ratios = np.where(T[:3, col] > 0, T[:3, -1] / np.where(T[:3, col] > 0, T[:3, col], 1), np.inf)
print(f"entering variable: x{col + 1};  ratios {ratios}  ->  constraint {int(np.argmin(ratios)) + 1} becomes binding first")""")],
        exercise="**Implement it yourself:** the *log-barrier* (interior-point) method minimises "
                 "$f(x) - \\mu\\sum\\ln(-g_i(x))$ for decreasing $\\mu$, staying strictly feasible. Implement it with "
                 "`optimize.nelder_mead` for the tank with $h \\le 1$ (eliminate the volume constraint first) and "
                 "compare with the penalty method."),
    "18_monte_carlo": dict(
        objectives=["estimate integrals by Monte Carlo and quantify their standard error",
                    "explain why Monte Carlo beats grids in high dimensions, and its 1/sqrt(N) cost",
                    "connect random walks with diffusion",
                    "estimate a structural failure probability and understand the cost of rare events"],
        prereq="notebook 01 (Monte Carlo propagation); normal distribution", time="60 min",
        anchor="## Why Monte Carlo wins in high dimensions",
        inside=[md(r"""
## Inside the algorithm: hit or miss
Throw random points into the square $[-1, 1]^2$; the fraction landing inside the unit circle estimates
$\pi/4$. The standard error follows from the binomial distribution:
"""), code(r"""rng_demo = np.random.default_rng(0)
pts = rng_demo.uniform(-1, 1, size=(100_000, 2))
hits = np.sum(pts**2, axis=1) <= 1
p_hit = hits.mean()
print(f"pi ≈ {4*p_hit:.4f} ± {4*np.sqrt(p_hit*(1 - p_hit)/hits.size):.4f}")""")],
        exercise="**Implement it yourself:** write a Monte Carlo estimate of the probability that the sum of 12 "
                 "uniform(0, 1) numbers exceeds 8, with its standard error. Compare with the normal approximation "
                 "(mean 6, variance 1) and explain the difference."),
    "19_sparse_2d_pdes": dict(
        objectives=["discretise a 2D Poisson problem with the 5-point stencil",
                    "build and store sparse matrices with Kronecker products",
                    "solve large systems with sparse direct solvers and conjugate gradient",
                    "check grid independence and simulate a transient with a reused factorisation"],
        prereq="notebooks 03, 12 and 15; partial derivatives", time="60-75 min",
        anchor="## Solving at realistic size",
        inside=[md(r"""
## Inside the algorithm: the 2D Laplacian from Kronecker products
Number the grid nodes row by row. The 2D operator is the 1D operator acting along $x$ in every row plus
the 1D operator acting along $y$ in every column: $A = I_y \otimes T_x + T_y \otimes I_x$. For a 3 × 2 grid:
"""), code(r"""Tx = sparse.laplacian_1d(3, 1.0).toarray()
Ty = sparse.laplacian_1d(2, 1.0).toarray()
print((np.kron(np.eye(2), Tx) + np.kron(Ty, np.eye(3))).astype(int))""")],
        exercise="**Implement it yourself:** add a Jacobi preconditioner to `conjugate_gradient` (divide the residual "
                 "by the diagonal of $A$ when forming the search direction). How many iterations does it save for "
                 "the board problem? Why so few, for this particular matrix?"),
})


EXTRAS["20_working_with_data_files"] = dict(
    objectives=["inspect a data file before loading it, and read it with NumPy and pandas",
                "detect silent problems such as mismatched headers, unit changes and error codes",
                "clean a messy file step by step with a record of every change",
                "analyse real records: reproduce a certified NIST fit; trend and seasonal cycle of CO₂"],
    prereq="notebooks 00, 08, 09 and 16; basic pandas is introduced here", time="75-90 min",
    anchor="Two principles: **keep the raw file untouched**",
    inside=[md(r"""
## Inside the algorithm: robust spike detection
A spike is a value far from its neighbours. The running **median** ignores a single wild value (a mean
would be dragged towards it), and the median absolute deviation (MAD, scaled by 1.4826 to match a
standard deviation for normal noise) measures the typical scatter without being inflated by the
spikes themselves:
"""), code(r"""values = np.array([50.1, 50.3, 49.9, 50.2, 75.0, 50.4, 50.1, 50.0])   # one spike
running_median = np.array([np.median(values[max(0, i-3):i+4]) for i in range(values.size)])
dev = values - running_median
robust_sd = 1.4826 * np.median(np.abs(dev))
print("deviation / robust SD:", np.round(dev / robust_sd, 1))
print(f"plain standard deviation {np.std(dev):.2f} (inflated by the spike) vs robust {robust_sd:.2f}")""")],
    exercise="**Implement it yourself:** write `read_nist(name)` that returns the data, the certified values and "
             "their standard deviations for any NIST StRD file, by finding the line that starts with `Data:` and "
             "the lines containing `b1 =`, `b2 =`, ... instead of hard-coding `skiprows=60`. Test it on all three "
             "NIST files.",
)


# =====================================================================================================
def write_all(run: bool = True, only=()):
    import nbclient

    for name, cells in NOTEBOOKS.items():
        if only and not name.startswith(tuple(only)):
            continue
        cells = apply_extras(name, cells)
        nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                "language": "python"},
                                                 "language_info": {"name": "python"}})
        path = HERE / f"{name}.ipynb"
        if run:
            client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                                             resources={"metadata": {"path": str(HERE)}})
            client.execute()
        nbformat.write(nb, path)
        print("wrote", path.name)


GALLERY = {  # README figures: (notebook, figure index, file name)
    "convergence.png": ("02_root_finding", 0 + 1),
    "moody.png": ("02_root_finding", 2),
    "valley.png": ("10_implicit_models_optimisation", 0),
    "instability.png": ("12_diffusion_pde", 1),
    "basins.png": ("14_nonlinear_systems", 1),
    "spectrum.png": ("16_fourier_analysis", 1),
    "board.png": ("19_sparse_2d_pdes", 1),
    "runge.png": ("04_interpolation", 1),
    "adaptive_steps.png": ("11_odes", 2),
}


def export_gallery():
    import base64

    out = HERE.parent / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    for fname, (nbname, idx) in GALLERY.items():
        nb = nbformat.read(HERE / f"{nbname}.ipynb", 4)
        pngs = [o["data"]["image/png"] for c in nb.cells for o in c.get("outputs", [])
                if "data" in o and "image/png" in o["data"]]
        (out / fname).write_bytes(base64.b64decode(pngs[idx]))
        print("gallery:", fname)


if __name__ == "__main__":
    run = "--no-run" not in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    write_all(run=run, only=only)
    if run and not only:
        export_gallery()
