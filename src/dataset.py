"""
dataset.py — build a windowed, session-grouped training set from .csibin recordings.

Responsibilities (README §3, §4):
  1. Glob *.json manifests under data/recordings/.
  2. Integrity-gate every manifest BEFORE loading its payload.
  3. Load surviving sessions via the canonical read_csibin.load_session().
  4. Window each session with read_csibin.window_stream() -> [num_windows, 64, time],
     subcarrier-major, bit-for-bit identical to the backend's SnapshotSubcarrierMajor.
  5. Assign integer labels in a single, deterministic class order and carry each
     window's source sessionId as the group key for the session-level split.

Nothing here computes normalization stats — those are train-split-only (README §6)
and so live in train.py, downstream of the session split.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np

import read_csibin

# Project root = parent of this src/ directory.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_SRC_DIR)
RECORDINGS_DIR = os.path.join(ROOT, "data", "recordings")

# Hard contract from the binary format / backend (README §3).
REQUIRED_SUBCARRIERS = 64


@dataclass
class Dataset:
    X: np.ndarray            # float32 [N, 64, time]  RAW filtered windows (NOT normalized)
    y: np.ndarray            # int64   [N]            label index into `classes`
    groups: np.ndarray       # int64   [N]            source sessionId (split group key)
    classes: list[str]       # output-index -> class name (the single source of truth)
    window_size: int
    slide_step: int
    sample_rate_hz: float
    baseline_applied: bool


def gather_manifests(recordings_dir: str = RECORDINGS_DIR) -> list[str]:
    """Glob manifests (NOT binaries), sorted for determinism."""
    return sorted(glob.glob(os.path.join(recordings_dir, "*.json")))


def _gate(meta: dict, ref: dict | None) -> tuple[bool, str]:
    """
    Return (keep, reason). `ref` holds the first accepted session's consistency
    params (or None for the first one). Reason is a human-readable drop note.
    """
    if not meta.get("complete", False):
        return False, "complete != true (dropped/skipped frames or interrupted)"
    if meta.get("subcarrierCount") != REQUIRED_SUBCARRIERS:
        return False, f"subcarrierCount={meta.get('subcarrierCount')} != {REQUIRED_SUBCARRIERS}"

    if ref is not None:
        for key in ("baselineApplied", "sampleRateHz", "windowSize", "slideStep"):
            if meta.get(key) != ref[key]:
                return False, f"{key}={meta.get(key)} inconsistent with dataset ({ref[key]})"

    return True, "ok"


def build_dataset(recordings_dir: str = RECORDINGS_DIR,
                  classes: list[str] | None = None) -> Dataset:
    """
    Build the full windowed dataset.

    `classes`: if given, pins the output-index order (use the list saved in the
    checkpoint so train and export never disagree — README §7). If None, the order
    is derived deterministically as sorted(unique labels found).
    """
    manifests = gather_manifests(recordings_dir)
    if not manifests:
        raise FileNotFoundError(
            f"No manifests (*.json) found in {recordings_dir}. "
            "Copy the backend's Recordings/ contents into data/recordings/ (README §8)."
        )

    ref: dict | None = None
    X_parts, labels, groups = [], [], []
    kept, dropped = [], []

    for mpath in manifests:
        with open(mpath, "r", encoding="utf-8") as f:
            meta = json.load(f)

        keep, reason = _gate(meta, ref)
        if not keep:
            dropped.append((os.path.basename(mpath), reason))
            continue

        if ref is None:
            ref = {k: meta[k] for k in ("baselineApplied", "sampleRateHz", "windowSize", "slideStep")}

        sess = read_csibin.load_session(mpath)
        windows = read_csibin.window_stream(sess.amplitudes, ref["windowSize"], ref["slideStep"])
        if windows.shape[0] == 0:
            dropped.append((os.path.basename(mpath),
                            f"too few frames ({sess.amplitudes.shape[0]}) for windowSize {ref['windowSize']}"))
            continue

        X_parts.append(windows.astype(np.float32, copy=False))
        labels.extend([meta["label"]] * windows.shape[0])
        groups.extend([sess.session_id] * windows.shape[0])
        kept.append((os.path.basename(mpath), meta["label"], sess.session_id, windows.shape[0]))

    if not X_parts:
        raise RuntimeError(
            "Every manifest was gated out. Dropped:\n  " +
            "\n  ".join(f"{n}: {r}" for n, r in dropped)
        )

    X = np.concatenate(X_parts, axis=0)
    labels = np.asarray(labels)
    groups = np.asarray(groups, dtype=np.int64)

    if classes is None:
        classes = sorted(set(labels.tolist()))
    else:
        unknown = set(labels.tolist()) - set(classes)
        if unknown:
            raise ValueError(f"Labels {sorted(unknown)} not present in pinned class order {classes}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.asarray([class_to_idx[l] for l in labels], dtype=np.int64)

    # Report what survived gating — this matters more than accuracy at smoke-test time.
    print(f"[dataset] kept {len(kept)} session(s), dropped {len(dropped)}")
    for name, label, sid, n in kept:
        print(f"    keep  {name}  label={label!r} session={sid} windows={n}")
    for name, reason in dropped:
        print(f"    DROP  {name}  -> {reason}")
    print(f"[dataset] X={X.shape} classes={classes} baselineApplied={ref['baselineApplied']}")

    return Dataset(
        X=X, y=y, groups=groups, classes=classes,
        window_size=ref["windowSize"], slide_step=ref["slideStep"],
        sample_rate_hz=ref["sampleRateHz"], baseline_applied=ref["baselineApplied"],
    )


if __name__ == "__main__":
    ds = build_dataset()
    uniq, counts = np.unique(ds.y, return_counts=True)
    print("[dataset] per-class window counts:",
          {ds.classes[i]: int(c) for i, c in zip(uniq, counts)})
    print("[dataset] sessions:", np.unique(ds.groups).tolist())
