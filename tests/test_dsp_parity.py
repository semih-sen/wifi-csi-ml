#!/usr/bin/env python3
"""
test_dsp_parity.py — the mandatory Phase 2 golden cross-language parity gate.

The backend (serve) is the source of truth. Its DspGoldenParityTests dumps
`tools/parity/dsp_golden.json`: fixed integer inputs + the C# amplitude / sanitized
phase / STFT outputs. This test loads that fixture, recomputes every stage with the
Python twin (src/dsp_twin.py, the train side), and asserts reproduction within a
float32 tolerance. If either side's transform drifts structurally (wrong unwrap,
wrong detrend, wrong STFT window/hop/axis/bins), this fails loudly — that is the
train/serve invariant enforced.

Run:  python tests/test_dsp_parity.py        (or via pytest)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import dsp_twin  # noqa: E402

# Tolerances: amplitude is exact; phase/STFT carry sub-ULP libm noise only. Tight
# enough that any structural divergence is caught, loose enough for cross-libm atan2/
# cos/sin last-bit differences on float32-scale values.
ATOL = 1e-4
RTOL = 1e-4


def _find_golden() -> str:
    candidates = [
        os.environ.get("CSI_DSP_GOLDEN"),
        os.path.join(_HERE, "golden", "dsp_golden.json"),                       # vendored copy
        os.path.join(_ROOT, os.pardir, "wifi-csi-backend", "tools", "parity",    # sibling repo
                     "dsp_golden.json"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "dsp_golden.json not found. Generate it by running the backend test "
        "DspGoldenParityTests (dotnet test --filter DspGolden), or set CSI_DSP_GOLDEN.\n"
        f"Looked in: {candidates}"
    )


def _load_golden() -> dict:
    with open(_find_golden(), "r", encoding="utf-8") as f:
        return json.load(f)


def _report(name: str, got: np.ndarray, expected: np.ndarray) -> None:
    got = np.asarray(got, dtype=np.float64).ravel()
    expected = np.asarray(expected, dtype=np.float64).ravel()
    assert got.shape == expected.shape, f"{name}: shape {got.shape} != {expected.shape}"
    max_abs = float(np.max(np.abs(got - expected))) if got.size else 0.0
    print(f"  {name:16s} n={got.size:4d}  max_absdiff={max_abs:.3e}")
    assert np.allclose(got, expected, atol=ATOL, rtol=RTOL), \
        f"{name}: parity FAILED (max_absdiff={max_abs:.3e} > atol={ATOL})"


def test_inputs_match_backend_formulas():
    """The twin's deterministic inputs must equal the integers the backend dumped."""
    g = _load_golden()
    assert np.array_equal(dsp_twin.build_raw_iq(), np.asarray(g["rawIq"], dtype=np.int64))
    assert np.array_equal(dsp_twin.build_stft_series(), np.asarray(g["stftSeries"], dtype=np.int64))
    assert g["subcarriers"] == dsp_twin.SUBCARRIERS
    assert g["stftWindowSize"] == dsp_twin.STFT_WINDOW_SIZE
    assert g["stftHopSize"] == dsp_twin.STFT_HOP_SIZE
    assert g["stftBins"] == dsp_twin.STFT_BINS


def test_amplitude_parity():
    g = _load_golden()
    got = dsp_twin.amplitude(g["rawIq"])
    _report("amplitude", got, g["amplitude"])
    # Amplitude must be EXACT (integer^2 -> float32 sqrt is correctly rounded).
    assert np.array_equal(got, np.asarray(g["amplitude"], dtype=np.float32)), \
        "amplitude must be bit-exact, not merely close"


def test_sanitized_phase_parity():
    g = _load_golden()
    got = dsp_twin.sanitized_phase(g["rawIq"])
    _report("sanit.phase", got, g["sanitizedPhase"])


def test_doppler_stft_parity():
    g = _load_golden()
    spec = dsp_twin.spectrogram(g["stftSeries"])
    assert spec.shape == (g["stftFrames"], g["stftBins"]), \
        f"spectrogram shape {spec.shape} != ({g['stftFrames']}, {g['stftBins']})"
    _report("doppler-stft", spec.ravel(), g["stft"])


def test_stft_direct_matches_numpy_rfft():
    """
    Sanity/compat: the direct-DFT twin agrees with numpy.fft.rfft magnitude, so a
    training pipeline using rfft stays inside the same contract (looser tol — rfft is a
    different rounding path than the direct DFT).
    """
    g = _load_golden()
    series = np.asarray(g["stftSeries"], dtype=np.float64)
    w, hop = dsp_twin.STFT_WINDOW_SIZE, dsp_twin.STFT_HOP_SIZE
    frames = g["stftFrames"]
    rfft_spec = np.empty((frames, dsp_twin.STFT_BINS), dtype=np.float64)
    for f in range(frames):
        seg = series[f * hop: f * hop + w] * dsp_twin.HANN
        rfft_spec[f] = np.abs(np.fft.rfft(seg))
    direct = dsp_twin.spectrogram(series).astype(np.float64)
    max_abs = float(np.max(np.abs(direct - rfft_spec)))
    print(f"  {'stft vs rfft':16s} max_absdiff={max_abs:.3e}")
    assert np.allclose(direct, rfft_spec, atol=1e-3, rtol=1e-3), \
        f"direct DFT vs numpy.rfft diverged (max_absdiff={max_abs:.3e})"


def _main() -> int:
    print(f"[parity] golden: {_find_golden()}")
    tests = [
        test_inputs_match_backend_formulas,
        test_amplitude_parity,
        test_sanitized_phase_parity,
        test_doppler_stft_parity,
        test_stft_direct_matches_numpy_rfft,
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
