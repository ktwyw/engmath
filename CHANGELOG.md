# Changelog

## 0.1.0 - 2026-09-26

First public release.

- **Library** (`engmath`): readable from-scratch implementations in 16 modules - root finding,
  optimisation (golden section, Nelder-Mead, simplex linear programming with shadow prices, penalty
  method), linear systems and eigenvalues, interpolation, integration, differentiation, smoothing,
  least-squares fitting (QR, Huber, Levenberg-Marquardt), nonlinear systems, ODEs, boundary-value
  problems, diffusion PDEs, Fourier analysis, Monte Carlo methods, sparse 2D solvers and result
  reporting - plus `datasets` with bundled data files.
- **Course**: 21 executed Jupyter notebooks (about 21 hours of study) built on original engineering
  problems, each with learning objectives, an "Inside the algorithm" section and exercises including an
  "Implement it yourself" task.
- **Solutions**: worked, executed solutions to all 83 exercises.
- **Data**: real NIST StRD reference measurements (Chwirut2, Hahn1, ENSO) and the NOAA Mauna Loa CO₂
  record (public domain), and course measurement files including a deliberately messy logger file.
- **Validation**: 106 checks against SciPy/NumPy/statsmodels, exact solutions, NIST certified values
  (including results computed from the bundled real data files) and theoretical orders of
  convergence; 64 unit tests and doctests. Requires Python >= 3.10, NumPy >= 1.24, SciPy >= 1.11
  (tested down to Python 3.10 with NumPy 1.24 and SciPy 1.11).
