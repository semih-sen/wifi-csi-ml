#!/usr/bin/env python3
"""
test_window_parity.py — the mandatory Phase 3 golden cross-language parity gate.

The backend (serve) is the source of truth. Its WindowGoldenParityTests dumps
`tools/parity/window_golden.json`: fixed integer raw-I/Q inputs (per RX per frame) + the
C# fused DENSE and DOPPLER tensors. This test loads that fixture, recomputes the fused
window with the Python twin (src/window_twin.py, the train side), and asserts it
reproduces both tensors within a float32 tolerance.

The fused-window LAYOUT / axis order is the failure mode this gate exists to catch: a
silent transpose (dense [rx, modality, frame, subcarrier] or doppler
[rx, subcarrier, stftFrame, bin]) feeds the model garbage. Integer inputs ⇒ no
float-input divergence; only the fusion/windowing transform + layout is under test.

Run:  python tests/test_window_parity.py        (or via pytest)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import window_twin  # noqa: E402

# Same tolerance as the DSP gate: amplitude is exact; phase (atan2) and STFT (cos/sin)
# carry only sub-ULP cross-libm noise. Tight enough to catch any structural/layout drift.
ATOL = 1e-4
RTOL = 1e-4


def _find_golden() -> str:
    candidates = [
        os.environ.get("CSI_WINDOW_GOLDEN"),
        os.path.join(_HERE, "golden", "window_golden.json"),                     # vendored copy
        os.path.join(_ROOT, os.pardir, "wifi-csi-backend", "tools", "parity",    # sibling repo
                     "window_golden.json"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "window_golden.json not found. Generate it by running the backend test "
        "WindowGoldenParityTests (dotnet test --filter WindowGolden), or set "
        f"CSI_WINDOW_GOLDEN.\nLooked in: {candidates}"
    )


def _load_golden() -> dict:
    with open(_find_golden(), "r", encoding="utf-8") as f:
        return json.load(f)


def _report(name: str, got: np.ndarray, expected: np.ndarray) -> None:
    got = np.asarray(got, dtype=np.float64).ravel()
    expected = np.asarray(expected, dtype=np.float64).ravel()
    assert got.shape == expected.shape, f"{name}: shape {got.shape} != {expected.shape}"
    max_abs = float(np.max(np.abs(got - expected))) if got.size else 0.0
    print(f"  {name:16s} n={got.size:6d}  max_absdiff={max_abs:.3e}")
    assert np.allclose(got, expected, atol=ATOL, rtol=RTOL), \
        f"{name}: parity FAILED (max_absdiff={max_abs:.3e} > atol={ATOL})"


def test_constants_match_backend():
    g = _load_golden()
    assert g["rxCount"] == window_twin.RX_COUNT
    assert g["modalities"] == window_twin.MODALITIES
    assert g["subcarriers"] == window_twin.SUBCARRIERS
    assert g["windowFrames"] == window_twin.WINDOW_FRAMES
    assert g["windowSlide"] == window_twin.WINDOW_SLIDE
    assert g["stftWindowSize"] == window_twin.STFT_WINDOW_SIZE
    assert g["stftHopSize"] == window_twin.STFT_HOP_SIZE
    assert g["stftFrames"] == window_twin.STFT_FRAMES
    assert g["stftBins"] == window_twin.STFT_BINS
    assert g["denseShape"] == [window_twin.RX_COUNT, window_twin.MODALITIES,
                               window_twin.WINDOW_FRAMES, window_twin.SUBCARRIERS]
    assert g["dopplerShape"] == [window_twin.RX_COUNT, window_twin.SUBCARRIERS,
                                 window_twin.STFT_FRAMES, window_twin.STFT_BINS]


def _fuse_from_golden(g: dict):
    raw = np.asarray(g["rawIq"], dtype=np.int64).reshape(
        window_twin.RX_COUNT, window_twin.WINDOW_FRAMES, 2 * window_twin.SUBCARRIERS)
    return window_twin.fuse_window(raw)


def test_dense_parity():
    g = _load_golden()
    dense, _ = _fuse_from_golden(g)
    assert dense.shape == tuple(g["denseShape"]), \
        f"dense shape {dense.shape} != {tuple(g['denseShape'])}"
    _report("dense", dense.ravel(), g["dense"])


def test_doppler_parity():
    g = _load_golden()
    _, doppler = _fuse_from_golden(g)
    assert doppler.shape == tuple(g["dopplerShape"]), \
        f"doppler shape {doppler.shape} != {tuple(g['dopplerShape'])}"
    _report("doppler", doppler.ravel(), g["doppler"])


def test_amplitude_slice_is_bit_exact():
    """
    The dense amplitude modality is integer-derived (int^2 -> float32 sqrt, correctly
    rounded), so it must be BIT-EXACT vs the backend — a stronger check than allclose.
    """
    g = _load_golden()
    dense, _ = _fuse_from_golden(g)
    expected = np.asarray(g["dense"], dtype=np.float32).reshape(tuple(g["denseShape"]))
    got_amp = dense[:, window_twin.MODALITY_AMPLITUDE]
    exp_amp = expected[:, window_twin.MODALITY_AMPLITUDE]
    assert np.array_equal(got_amp, exp_amp), \
        "dense amplitude modality must be bit-exact, not merely close"


def _main() -> int:
    print(f"[parity] golden: {_find_golden()}")
    tests = [
        test_constants_match_backend,
        test_dense_parity,
        test_doppler_parity,
        test_amplitude_slice_is_bit_exact,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n[parity] {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
