#!/usr/bin/env python3
"""
infer.py — single-window ONNX inference over a recorded session (Seam A).

Validates the model contract with ZERO backend dependency, and doubles as the
C#-parity reference: the same window fed to the backend's OnnxModelEvaluator must
produce the same probabilities (the graph bakes in normalization + softmax, so this
runs raw filtered windows straight through onnxruntime).

Run:
    python src/infer.py data/recordings/<session>.json
    python src/infer.py <session>.csibin --window 3
    python src/infer.py <session>.json --all      # mean probabilities over all windows

Contract (see /CONTRACTS.md, Seam A):
    input  "input"  : float32 [batch, 64, 100]  subcarrier-major, RAW filtered window
    output "output" : float32 [batch, num_classes]  softmax probabilities
    labels: artifacts/labels.json (output-channel order)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import onnxruntime as ort

import read_csibin
from dataset import ROOT

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
ONNX_PATH = os.path.join(ARTIFACTS_DIR, "model.onnx")
LABELS_PATH = os.path.join(ARTIFACTS_DIR, "labels.json")

# Must match the exported model input + the backend's OnnxInput contract.
EXPECTED_SUBCARRIERS = 64
EXPECTED_WINDOW = 100


def load_labels() -> list[str]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{LABELS_PATH} is not a non-empty JSON array")
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description="Single-window ONNX inference (Seam A).")
    ap.add_argument("session", help="path to a .json manifest or .csibin payload")
    ap.add_argument("--window", type=int, default=0, help="window index to classify (default 0)")
    ap.add_argument("--all", action="store_true", help="report mean probabilities over all windows")
    args = ap.parse_args()

    if not os.path.exists(ONNX_PATH):
        print(f"[infer] no model at {ONNX_PATH}. Run: python src/export_onnx.py")
        return 2

    labels = load_labels()

    sess = read_csibin.load_session(args.session)
    if sess.subcarrier_count != EXPECTED_SUBCARRIERS:
        print(f"[infer] subcarrierCount={sess.subcarrier_count} != {EXPECTED_SUBCARRIERS}; "
              "this model cannot classify this session.")
        return 2

    # Window exactly as the backend does (subcarrier-major) — same code path as training.
    windows = read_csibin.window_stream(sess.amplitudes, EXPECTED_WINDOW, EXPECTED_WINDOW)
    if windows.shape[0] == 0:
        print(f"[infer] session has too few frames ({sess.amplitudes.shape[0]}) "
              f"for a {EXPECTED_WINDOW}-frame window.")
        return 2
    print(f"[infer] session label={sess.label!r} subject={sess.subject!r} "
          f"frames={sess.amplitudes.shape[0]} windows={windows.shape[0]} "
          f"(each {windows.shape[1]}×{windows.shape[2]})")

    ort_sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name = ort_sess.get_inputs()[0].name
    out_name = ort_sess.get_outputs()[0].name

    if args.all:
        probs = ort_sess.run([out_name], {in_name: windows.astype(np.float32)})[0]
        mean = probs.mean(axis=0)
        report("mean over all windows", mean, labels)
    else:
        idx = args.window
        if not (0 <= idx < windows.shape[0]):
            print(f"[infer] --window {idx} out of range [0, {windows.shape[0]})")
            return 2
        x = windows[idx:idx + 1].astype(np.float32)  # [1, 64, 100]
        prob = ort_sess.run([out_name], {in_name: x})[0][0]
        report(f"window {idx}", prob, labels)

    return 0


def report(title: str, prob: np.ndarray, labels: list[str]) -> None:
    if len(prob) != len(labels):
        raise ValueError(
            f"model output dim {len(prob)} != label count {len(labels)} — "
            "labels.json does not match this model (re-export both)."
        )
    order = np.argsort(prob)[::-1]
    top = int(order[0])
    print(f"[infer] {title}: predicted={labels[top]!r} confidence={prob[top]:.4f}")
    for i in order:
        print(f"        {labels[i]:<16} {prob[i]:.4f}")
    s = float(prob.sum())
    assert abs(s - 1.0) < 1e-3, f"probabilities do not sum to 1 (got {s})"


if __name__ == "__main__":
    raise SystemExit(main())
