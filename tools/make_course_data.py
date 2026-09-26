"""Write the synthetic course data files into src/engmath/data (reproducible: fixed random seeds).

The generating code is identical to what the notebooks used before the data moved to files, so all
notebook results are unchanged. Run from the repository root:  python tools/make_course_data.py
"""
from pathlib import Path

import numpy as np
from scipy.special import erf

OUT = Path(__file__).resolve().parents[1] / "src" / "engmath" / "data"


def write(name, header, columns, fmt="%.17g"):
    np.savetxt(OUT / name, np.column_stack(columns), delimiter=",", header=header, comments="", fmt=fmt)
    print("wrote", name)


# notebook 08: thermocouple calibration
T_ref = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400.0])
rng = np.random.default_rng(11)
E = 0.0395 * T_ref + 1.5e-6 * T_ref**2 + rng.normal(0, 0.012, T_ref.size)
write("thermocouple_calibration.csv", "T_ref_degC,E_mV", [T_ref, E])

# notebook 10: pressure drops (Colebrook model with roughness 0.15 mm, meter factor 1.03, 2 % noise)
rho, mu, L, D = 998.0, 1.0e-3, 20.0, 0.05


def friction(Re, eps):
    from scipy.optimize import brentq
    return brentq(lambda f: 1 / np.sqrt(f) + 2 * np.log10(eps / D / 3.7 + 2.51 / (Re * np.sqrt(f))), 1e-4, 0.2,
                  xtol=1e-14)


def dp_model(Q, eps, k=1.0):
    v = 4 * k * Q / (np.pi * D**2)
    return np.array([friction(R, eps) for R in rho * v * D / mu]) * L / D * rho * v**2 / 2


Q = np.array([0.5, 1, 2, 3, 4, 5, 6, 8]) * 1e-3
rng = np.random.default_rng(2)
dp = dp_model(Q, 0.15e-3, 1.03) * (1 + 0.02 * rng.standard_normal(Q.size))
write("pipe_pressure_drop.csv", "Q_m3_per_s,dP_Pa", [Q, dp])

# notebook 12: thermocouple 10 mm below a surface raised from 20 to 320 degC (alpha = 1.2e-5 m2/s)
alpha, T0, Ts, x_tc = 1.2e-5, 20.0, 320.0, 0.010
t_meas = np.arange(2, 61, 2.0)
rng = np.random.default_rng(4)
T_meas = Ts + (T0 - Ts) * erf(x_tc / (2 * np.sqrt(alpha * t_meas))) + rng.normal(0, 1.0, t_meas.size)
write("thermocouple_10mm.csv", "t_s,T_degC", [t_meas, T_meas])

# notebook 16: pump vibration
fs, duration = 2048.0, 2.0
t = np.arange(int(fs * duration)) / fs
rng = np.random.default_rng(16)
f_shaft = 1480 / 60
signal = (1.00 * np.sin(2 * np.pi * f_shaft * t) + 0.40 * np.sin(2 * np.pi * 2 * f_shaft * t + 1)
          + 0.15 * np.sin(2 * np.pi * 137.3 * t + 2) + rng.normal(0, 0.5, t.size))
write("pump_vibration.csv", "t_s,velocity_mm_per_s", [t, signal])

# notebook 20: a deliberately messy process-logger file (truth known, so cleaning can be checked)
rng = np.random.default_rng(20)
minutes = np.arange(0, 360)                                   # 6 h at 1-minute intervals from 08:00
T_true = 20 + 60 * (1 - np.exp(-minutes / 45))                # heat-up towards 80 degC
P_true = 2.0 + 0.004 * (T_true - 20)                          # bar
T_meas = T_true + rng.normal(0, 0.3, minutes.size)
P_meas = P_true + rng.normal(0, 0.01, minutes.size)
lines = ["# Data logger DL-7 / reactor R2 / exported 2026-03-14 14:05",
         "# Columns: timestamp, temperature, pressure",
         "# Temperature unit: degF until 10:00, then logger reconfigured to degC",
         "# Pressure unit: bar. Error code -999 = sensor fault. Empty field = no reading.",
         "timestamp,temperature,pressure"]
offline = (minutes >= 190) & (minutes < 202)                  # logger offline 11:10-11:21
for i, m in enumerate(minutes):
    if offline[i]:
        continue
    ts = f"2026-03-14 {8 + m // 60:02d}:{m % 60:02d}:00"
    T = T_meas[i] * 9 / 5 + 32 if m < 120 else T_meas[i]      # degF before 10:00
    if m in (75, 250):
        T += 25.0 if m == 250 else 45.0                       # electrical spikes
    T_txt = "-999" if m in (140, 141) else f"{T:.2f}"         # sensor fault
    P_txt = "" if m in (33, 260, 261, 262) else f"{P_meas[i]:.3f}"
    row = f"{ts},{T_txt},{P_txt}"
    lines.append(row)
    if m in (50, 51, 300):                                    # retransmitted rows (duplicates)
        lines.append(row)
(OUT / "process_log_messy.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
np.savetxt(OUT.parents[2] / "tools" / "process_log_truth.csv", np.column_stack([minutes, T_true, P_true]),
           delimiter=",", header="minute,T_true_degC,P_true_bar", comments="", fmt="%.6f")
print("wrote process_log_messy.csv (and tools/process_log_truth.csv for checking)")
