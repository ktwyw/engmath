"""engmath - engineering mathematics with Python.

Readable from-scratch implementations of the numerical methods engineers use, each checked
against SciPy and exact results:

roots          bisection, Newton, secant, Brent; convergence order
optimize       golden-section search, Nelder-Mead simplex
linalg         Gaussian elimination, Thomas (tridiagonal), Jacobi/Gauss-Seidel, power iteration
interpolate    linear, natural cubic spline, Lagrange polynomial
integrate      trapezoid, Simpson (uneven spacing), Romberg, Gauss-Legendre
differentiate  finite differences, uneven-spacing gradient, Richardson extrapolation
smoothing      moving average, Savitzky-Golay, LOWESS
fitting        linear least squares, robust Huber IRLS, Levenberg-Marquardt
odes           Euler, RK4, adaptive RK23, backward Euler (stiff)
pdes           1D diffusion: explicit FTCS and Crank-Nicolson
reporting      value ± uncertainty with correct significant figures; error propagation
nonlinear      systems of nonlinear equations: Newton's method with line search
bvp            boundary-value problems: shooting and finite differences (Dirichlet/Robin)
fourier        DFT, radix-2 FFT, amplitude spectra with windowing, FFT filtering
montecarlo     Monte Carlo integration in many dimensions, random walks
sparse         sparse 2D Laplacian, Poisson solver, conjugate gradient
datasets       bundled data files: real NIST and NOAA measurements, course data
optimize       (also) simplex linear programming with shadow prices, penalty method
"""

from . import differentiate, fitting, integrate, interpolate, linalg, odes, optimize, pdes, reporting, roots, smoothing

__version__ = "0.1.0"
__all__ = ["bvp", "datasets", "differentiate", "fitting", "fourier", "integrate", "interpolate", "linalg", "montecarlo",
           "nonlinear", "odes", "optimize", "pdes", "reporting", "roots", "smoothing", "sparse", "__version__"]
