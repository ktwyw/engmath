"""Bundled data files - real measurements and course data - for practising reading files.

Use :func:`path` to get a file's location and read it with NumPy, pandas or plain Python, exactly as
you would read your own laboratory files; :func:`info` describes the source, units and licence.

>>> from engmath import datasets
>>> "Chwirut2.dat" in datasets.available()
True
>>> import numpy as np
>>> data = np.loadtxt(datasets.path("Chwirut2.dat"), skiprows=60)   # NIST header block: 60 lines
>>> data.shape
(54, 2)
"""

from __future__ import annotations

from importlib import resources

INFO = {
    "Chwirut2.dat": "REAL. NIST StRD nonlinear regression dataset Chwirut2: ultrasonic calibration (ultrasonic "
                    "response y vs metal distance x), 54 observations, with certified parameter values in the "
                    "header. Model y = exp(-b1 x)/(b2 + b3 x). Public domain (US government work). File copied "
                    "unchanged from the lmfit-py mirror of the NIST archive.",
    "Hahn1.dat": "REAL. NIST StRD dataset Hahn1: thermal expansion of copper (coefficient of thermal expansion y "
                 "vs temperature x in kelvin), 236 observations, certified values for a 7-parameter rational "
                 "model in the header. Public domain.",
    "ENSO.dat": "REAL. NIST StRD dataset ENSO: monthly averaged atmospheric pressure differences between Easter "
                "Island and Darwin, Australia (168 months). Contains an annual cycle and two longer cycles "
                "(El Nino). Certified values for a 9-parameter sinusoidal model in the header. Public domain.",
    "co2_mauna_loa_monthly.csv": "REAL. Monthly mean atmospheric CO2 (ppm) at Mauna Loa, Hawaii, March 1958 - "
                                 "July 2026 (NOAA Global Monitoring Laboratory, via github.com/datasets/co2-ppm, "
                                 "ODC-PDDL public-domain dedication). Downloaded 2026-09-26 and kept UNCHANGED, "
                                 "including its quirks: the header names 6 columns but rows have 7 values "
                                 "(date, decimal date, monthly mean, de-seasonalised mean, number of days, "
                                 "std. dev. of days, uncertainty of the mean; -1/-9.99/-0.99 mark missing values).",
    "thermocouple_calibration.csv": "Course data (synthetic, type-K-like): reference temperature (degC) and "
                                    "thermocouple voltage (mV). Notebook 08.",
    "pipe_pressure_drop.csv": "Course data (synthetic): measured flow rate (m3/s) and pressure drop (Pa) over "
                              "20 m of 50 mm pipe. Notebook 10.",
    "thermocouple_10mm.csv": "Course data (synthetic): temperature (degC) of a thermocouple 10 mm below a "
                             "suddenly heated steel surface. Notebook 12.",
    "pump_vibration.csv": "Course data (synthetic): pump-bearing vibration velocity (mm/s) sampled at 2048 Hz "
                          "for 2 s. Notebook 16.",
    "process_log_messy.csv": "Course data (synthetic, deliberately messy): a process data-logger file with a "
                             "metadata header, a mid-file change of temperature units, a gap, duplicated rows, "
                             "error codes and spikes. Notebook 20.",
}


def available() -> list[str]:
    """Names of all bundled data files."""
    return sorted(INFO)


def path(name: str) -> str:
    """Filesystem path of a bundled data file (works when engmath is installed, including on Colab)."""
    if name not in INFO:
        raise KeyError(f"Unknown data file {name!r}. Available: {', '.join(available())}")
    return str(resources.files("engmath.data").joinpath(name))


def info(name: str) -> str:
    """Source, content, units and licence of a data file."""
    if name not in INFO:
        raise KeyError(f"Unknown data file {name!r}.")
    return INFO[name]
