#!/usr/bin/env python3
"""
window_twin.py — Python twin of the backend's V2 Phase 3 fusion + windowing stage (Seam A).

╔════════════════════════════════════════════════════════════════════════════╗
║ TRAIN/SERVE PARITY. This module MUST reproduce, bit-close, the C# fusion in    ║
║   wifi-csi-backend/CsiRadar.Backend/Application/Processing/Windowing/           ║
║     WindowContract.cs · WindowAssembler.cs                                     ║
║ It is the #1 project invariant. The fused-window LAYOUT / axis order is the     ║
║ thing most likely to diverge silently (a transpose feeds the model garbage),    ║
║ so the golden parity test (tests/test_window_parity.py) enforces it against a   ║
║ fixture the backend dumps (tools/parity/window_golden.json). Change one side →  ║
║ change the other and regenerate the golden, or the test fails loudly.           ║
╚════════════════════════════════════════════════════════════════════════════╝

The fused window carries two flat, row-major tensors with pinned axis order:

  • dense   [rx, modality, frame, subcarrier]  = [2, 2, WINDOW_FRAMES, 64]
            modality 0 = amplitude, 1 = sanitized phase.
  • doppler [rx, subcarrier, stftFrame, bin]   = [2, 64, STFT_FRAMES, STFT_BINS]
            STFT (pinned Phase 2 geometry) of each subcarrier's amplitude series
            over the window.

No normalization here — that is baked into the ONNX graph in a later phase. Depends on
dsp_twin for the (already parity-tested) per-frame amplitude / sanitized-phase / STFT.
"""

from __future__ import annotations

import numpy as np

import dsp_twin

# ── Pinned contract constants — mirror WindowContract.cs exactly ──
RX_COUNT = 2
MODALITIES = 2
MODALITY_AMPLITUDE = 0
MODALITY_PHASE = 1

SUBCARRIERS = dsp_twin.SUBCARRIERS            # 64
WINDOW_FRAMES = 256                            # 2.56 s @ 100 Hz (≈2–3 stride cycles)
WINDOW_SLIDE = 128                             # 1.28 s (50% overlap)

STFT_WINDOW_SIZE = dsp_twin.STFT_WINDOW_SIZE   # 64
STFT_HOP_SIZE = dsp_twin.STFT_HOP_SIZE         # 16
STFT_BINS = dsp_twin.STFT_BINS                 # 33
STFT_FRAMES = (WINDOW_FRAMES - STFT_WINDOW_SIZE) // STFT_HOP_SIZE + 1  # 13


def fuse_window(raw_iq):
    """
    Fuse one window of raw I/Q into the pinned (dense, doppler) tensors.

    raw_iq: int array shaped [RX_COUNT, WINDOW_FRAMES, 2*SUBCARRIERS] — interleaved
            [imag, real] per subcarrier, per frame, per RX (matches WindowAssembler input:
            each frame's per-RX amplitude/phase are derived from this raw via dsp_twin).

    Returns (dense, doppler) as float32 ndarrays with the axis order above; ravel() them
    (C-order) to get the flat layout the backend dumps.
    """
    raw = np.asarray(raw_iq)
    assert raw.shape == (RX_COUNT, WINDOW_FRAMES, 2 * SUBCARRIERS), \
        f"raw_iq shape {raw.shape} != {(RX_COUNT, WINDOW_FRAMES, 2 * SUBCARRIERS)}"

    dense = np.zeros((RX_COUNT, MODALITIES, WINDOW_FRAMES, SUBCARRIERS), dtype=np.float32)
    doppler = np.zeros((RX_COUNT, SUBCARRIERS, STFT_FRAMES, STFT_BINS), dtype=np.float32)

    # ── Per-frame dense modalities (amplitude + sanitized phase), derived per RX ──
    amp = np.zeros((RX_COUNT, WINDOW_FRAMES, SUBCARRIERS), dtype=np.float32)
    for rx in range(RX_COUNT):
        for f in range(WINDOW_FRAMES):
            a = dsp_twin.amplitude(raw[rx, f])
            p = dsp_twin.sanitized_phase(raw[rx, f])
            amp[rx, f] = a
            dense[rx, MODALITY_AMPLITUDE, f] = a
            dense[rx, MODALITY_PHASE, f] = p

    # ── Windowed per-subcarrier Doppler: STFT of each subcarrier's amplitude series ──
    for rx in range(RX_COUNT):
        for k in range(SUBCARRIERS):
            series = amp[rx, :, k].astype(np.float64)
            spec = dsp_twin.spectrogram(series)      # [STFT_FRAMES, STFT_BINS]
            assert spec.shape == (STFT_FRAMES, STFT_BINS)
            doppler[rx, k] = spec

    return dense, doppler
