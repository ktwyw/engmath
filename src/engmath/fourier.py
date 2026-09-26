"""Fourier analysis: the discrete Fourier transform from its definition, a radix-2 FFT written from
scratch, amplitude spectra with windowing, and FFT filtering.

>>> import numpy as np
>>> from engmath import fourier
>>> x = np.random.default_rng(0).normal(size=64)
>>> bool(np.allclose(fourier.fft(x), np.fft.fft(x)) and np.allclose(fourier.dft(x), np.fft.fft(x)))
True
"""

from __future__ import annotations

import numpy as np


def dft(x) -> np.ndarray:
    """X_k = sum_n x_n exp(-2 pi i k n / N), straight from the definition: O(N^2) operations."""
    x = np.asarray(x, complex)
    n = np.arange(x.size)
    return np.exp(-2j * np.pi * np.outer(n, n) / x.size) @ x


def fft(x) -> np.ndarray:
    """Recursive radix-2 Cooley-Tukey FFT: split into even and odd samples, transform each half, and
    combine with 'twiddle factors'. O(N log N). N must be a power of two."""
    x = np.asarray(x, complex)
    N = x.size
    if N == 1:
        return x
    if N & (N - 1):
        raise ValueError("This teaching FFT needs a power-of-two length; use numpy.fft for any length.")
    even, odd = fft(x[0::2]), fft(x[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd
    return np.concatenate([even + twiddle, even - twiddle])


def spectrum(signal, fs: float, window: str | None = "hann"):
    """One-sided amplitude spectrum: a sine of amplitude A appears as a peak of height ~A.

    ``window='hann'`` reduces spectral leakage (at the cost of slightly wider peaks); ``None`` uses
    the rectangular window. Returns (frequencies in Hz, amplitudes)."""
    x = np.asarray(signal, float)
    N = x.size
    # periodic Hann window (the standard for spectral analysis; np.hanning is the symmetric form)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N) if window == "hann" else np.ones(N)
    if window not in ("hann", None):
        raise ValueError("window must be 'hann' or None.")
    X = np.fft.rfft(x * w)
    amp = 2 * np.abs(X) / w.sum()
    amp[0] /= 2
    if N % 2 == 0:
        amp[-1] /= 2
    return np.fft.rfftfreq(N, 1 / fs), amp


def fft_filter(signal, fs: float, low: float | None = None, high: float | None = None) -> np.ndarray:
    """Ideal band-pass filter: zero all Fourier coefficients outside [low, high] Hz and transform back.
    (Sharp cut-offs can cause ringing; practical filters roll off gradually.)"""
    x = np.asarray(signal, float)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1 / fs)
    keep = np.ones_like(f, bool)
    if low is not None:
        keep &= f >= low
    if high is not None:
        keep &= f <= high
    return np.fft.irfft(X * keep, n=x.size)
