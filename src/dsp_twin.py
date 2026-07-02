#!/usr/bin/env python3
"""
dsp_twin.py — Python twin of the backend's V2 Phase 2 per-RX DSP layer (Seam A).

╔════════════════════════════════════════════════════════════════════════════╗
║ TRAIN/SERVE PARITY. This module MUST reproduce, bit-close, the C# DSP in      ║
║   wifi-csi-backend/CsiRadar.Backend/Application/Processing/Dsp/               ║
║     DspContract.cs · CsiDsp.cs · StftProcessor.cs                             ║
║ It is the #1 project invariant: every transform the model consumes must be    ║
║ identical in serve (C#) and train (Python). The golden parity test            ║
║ (tests/test_dsp_parity.py) enforces it against a fixture the backend dumps    ║
║ (tools/parity/dsp_golden.json). Change one side → change the other and        ║
║ regenerate the golden, or the test fails loudly.                              ║
║                                                                              ║
║ "Bit-for-bit" is realised as "within float32 numerical noise": amplitude is  ║
║ exact; phase (atan2) and STFT (cos/sin) differ only by sub-ULP libm noise.    ║
╚════════════════════════════════════════════════════════════════════════════╝

Raw layout: interleaved int8 [imag0, real0, imag1, real1, ...] (ESP-IDF order).
"""

from __future__ import annotations

import numpy as np

# ── Pinned contract constants — mirror DspContract.cs exactly ──
SUBCARRIERS = 64
STFT_WINDOW_SIZE = 64
STFT_HOP_SIZE = 16
STFT_BINS = STFT_WINDOW_SIZE // 2 + 1

# Symmetric Hann, identical to numpy.hanning(W) and to C# DspContract.HannWindow.
HANN = np.hanning(STFT_WINDOW_SIZE)


def amplitude(raw_iq) -> np.ndarray:
    """|CSI|_k = sqrt(imag^2 + real^2) per subcarrier, float32 (bit-exact vs C#)."""
    raw = np.asarray(raw_iq, dtype=np.int64)
    imag = raw[0::2]
    real = raw[1::2]
    return np.sqrt((imag * imag + real * real).astype(np.float32))


def sanitized_phase(raw_iq) -> np.ndarray:
    """
    Single-antenna sanitized phase: atan2 -> unwrap across subcarriers -> remove the
    least-squares linear trend (STO slope + constant offset). NOT the dual-antenna
    conjugate method. Matches CsiDsp.SanitizedPhase.
    """
    raw = np.asarray(raw_iq, dtype=np.float64)
    imag = raw[0::2]
    real = raw[1::2]
    phase = np.arctan2(imag, real)          # raw phase (double)
    phase = np.unwrap(phase)                 # discont=pi, period=2pi
    phase = _detrend_least_squares(phase)    # subtract slope*k + intercept
    return phase.astype(np.float32)


def _detrend_least_squares(p: np.ndarray) -> np.ndarray:
    """
    Closed-form OLS detrend over integer index k = 0..N-1 — identical maths to
    CsiDsp.DetrendLeastSquaresInPlace (mean-centred, not polyfit's SVD path, so the
    numerical route matches the backend).
    """
    n = p.shape[0]
    if n < 2:
        return p.copy()
    k = np.arange(n, dtype=np.float64)
    mean_k = (n - 1) / 2.0
    mean_p = p.mean()
    dk = k - mean_k
    var_k = np.dot(dk, dk)
    slope = np.dot(dk, p - mean_p) / var_k if var_k > 0 else 0.0
    intercept = mean_p - slope * mean_k
    return p - (slope * k + intercept)


def magnitude_column(window: np.ndarray) -> np.ndarray:
    """
    One STFT magnitude column via a direct real DFT of the Hann-windowed length-W
    signal (DC..Nyquist). Matches StftProcessor.MagnitudeColumn: float64 accumulation,
    float32 output. A direct DFT (not np.fft) keeps the rounding path identical to C#.
    """
    w = STFT_WINDOW_SIZE
    xn = np.asarray(window, dtype=np.float64) * HANN
    out = np.empty(STFT_BINS, dtype=np.float32)
    n = np.arange(w, dtype=np.float64)
    for m in range(STFT_BINS):
        omega = 2.0 * np.pi * m / w
        re = np.dot(xn, np.cos(omega * n))
        im = -np.dot(xn, np.sin(omega * n))
        out[m] = np.sqrt(re * re + im * im)
    return out


def spectrogram(series) -> np.ndarray:
    """
    Full STFT of a 1-D series -> [num_frames, STFT_BINS] (frame-major, time down rows).
    Matches StftProcessor.Spectrogram.
    """
    s = np.asarray(series, dtype=np.float64)
    w, hop = STFT_WINDOW_SIZE, STFT_HOP_SIZE
    num_frames = 0 if s.shape[0] < w else (s.shape[0] - w) // hop + 1
    out = np.empty((num_frames, STFT_BINS), dtype=np.float32)
    for f in range(num_frames):
        out[f] = magnitude_column(s[f * hop: f * hop + w])
    return out


# ── Deterministic golden inputs — MUST match DspGoldenParityTests.BuildRawIq /
#    BuildStftSeries in the backend so both sides start from identical integers. ──

def build_raw_iq() -> np.ndarray:
    k = np.arange(SUBCARRIERS)
    raw = np.empty(2 * SUBCARRIERS, dtype=np.int64)
    raw[0::2] = ((k * 7 + 3) % 61) - 30    # imag
    raw[1::2] = ((k * 13 + 11) % 59) - 29  # real
    return raw


def build_stft_series() -> np.ndarray:
    n = np.arange(128)
    return ((n * 9 + 5) % 127) - 63
