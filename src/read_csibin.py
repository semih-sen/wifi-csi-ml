#!/usr/bin/env python3
"""
read_csibin.py — loader for CsiRadar `.csibin` recording sessions.

╔════════════════════════════════════════════════════════════════════════════╗
║ PROVENANCE (Seam A). This is a VENDORED COPY of                              ║
║   wifi-csi-backend/tools/read_csibin.py                                      ║
║ The two files MUST stay byte-identical. The binary layout below is the       ║
║ contract of record in /CONTRACTS.md (Seam A). `_FORMAT_VERSION` is asserted  ║
║ on load: bump the backend writer's format version and this side fails loudly ║
║ instead of silently misreading.                                              ║
║                                                                              ║
║ `window_stream()` MUST reproduce the backend's                              ║
║ CsiRingBuffer.SnapshotSubcarrierMajor bit-for-bit (subcarrier-major          ║
║ [subcarrier, time]). That identity is the train/serve invariant — do not     ║
║ change the windowing here without changing it there.                         ║
╚════════════════════════════════════════════════════════════════════════════╝

A session is two files written by the backend RecordingBackgroundService:
  <stem>.csibin   little-endian binary payload (header + frames)
  <stem>.json     manifest (label, frame count, filter params, integrity flags)

Typical use (assemble a labelled training set):

    import glob, json
    from read_csibin import load_session

    X, y = [], []
    for manifest_path in glob.glob("Recordings/*.json"):
        meta = json.load(open(manifest_path))
        if not meta["complete"]:
            continue  # skip sessions with dropped/skipped frames or interruptions
        sess = load_session(manifest_path)
        # sess.amplitudes: float32 [frame_count, subcarrier_count]  (the model-facing stream)
        # window it offline with meta["windowSize"] / meta["slideStep"] to match inference 1:1
        ...

Binary layout (little-endian), mirrors CsiRecordingFileWriter:

  HEADER
    magic            4  bytes  = b"CSI1"
    formatVersion    int32
    subcarrierCount  int32
    sampleRateHz     float64
    captureRaw       uint8     (0/1)
    labelLength      int32
    label            labelLength UTF-8 bytes
    sessionId        int64
    startedAtUnixMs  int64

  FRAME (repeated)
    timestampMs      int64
    rssi             int32
    amplitudes       subcarrierCount × float32
    [ if captureRaw ]
      rawLength      int32
      raw            rawLength × int8
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

import numpy as np

_MAGIC = b"CSI1"
# Binary format version the backend writer stamps (CsiRecordingFileWriter). Asserted
# on load so a format bump (csibin-v2) breaks here loudly — see PROVENANCE above.
_FORMAT_VERSION = 1


@dataclass
class Session:
    label: str
    session_id: int
    subcarrier_count: int
    sample_rate_hz: float
    capture_raw: bool
    started_at_unix_ms: int
    timestamps_ms: np.ndarray   # int64  [N]
    rssi: np.ndarray            # int32  [N]
    amplitudes: np.ndarray      # float32 [N, subcarrier_count]
    raw: list[np.ndarray] | None  # list of int8 arrays, only when capture_raw


def _bin_path_for(manifest_path: str) -> str:
    meta = json.load(open(manifest_path, "r", encoding="utf-8"))
    return os.path.join(os.path.dirname(manifest_path), meta["binaryFile"])


def load_session(path: str) -> Session:
    """Load a session given either its .json manifest or its .csibin payload path."""
    bin_path = _bin_path_for(path) if path.endswith(".json") else path

    with open(bin_path, "rb") as f:
        buf = f.read()

    off = 0

    def take(fmt: str):
        nonlocal off
        size = struct.calcsize(fmt)
        vals = struct.unpack_from("<" + fmt, buf, off)
        off += size
        return vals

    magic = buf[off:off + 4]; off += 4
    if magic != _MAGIC:
        raise ValueError(f"bad magic {magic!r}; not a csibin file")

    (version,) = take("i")
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"unsupported .csibin format version {version} (expected {_FORMAT_VERSION}). "
            "This reader is vendored from the backend; re-sync read_csibin.py."
        )

    (sc,) = take("i")
    (fs,) = take("d")
    (capture_raw,) = take("B")
    capture_raw = bool(capture_raw)
    (label_len,) = take("i")
    label = buf[off:off + label_len].decode("utf-8"); off += label_len
    (session_id,) = take("q")
    (started_ms,) = take("q")

    ts_list, rssi_list, amp_list, raw_list = [], [], [], ([] if capture_raw else None)

    if not capture_raw:
        # Fixed stride → one vectorized read via a packed structured dtype.
        rec = np.dtype([("ts", "<i8"), ("rssi", "<i4"), ("amp", "<f4", (sc,))])
        body = np.frombuffer(buf, dtype=rec, offset=off)
        return Session(
            label=label, session_id=session_id, subcarrier_count=sc,
            sample_rate_hz=fs, capture_raw=False, started_at_unix_ms=started_ms,
            timestamps_ms=body["ts"].copy(),
            rssi=body["rssi"].copy(),
            amplitudes=body["amp"].copy(),
            raw=None,
        )

    # Variable stride (raw present) → frame loop.
    n = len(buf)
    while off < n:
        (ts,) = take("q")
        (rssi,) = take("i")
        amp = np.frombuffer(buf, dtype="<f4", count=sc, offset=off).copy(); off += 4 * sc
        (raw_len,) = take("i")
        raw = np.frombuffer(buf, dtype="<i1", count=raw_len, offset=off).copy(); off += raw_len
        ts_list.append(ts); rssi_list.append(rssi); amp_list.append(amp); raw_list.append(raw)

    return Session(
        label=label, session_id=session_id, subcarrier_count=sc,
        sample_rate_hz=fs, capture_raw=True, started_at_unix_ms=started_ms,
        timestamps_ms=np.asarray(ts_list, dtype=np.int64),
        rssi=np.asarray(rssi_list, dtype=np.int32),
        amplitudes=np.asarray(amp_list, dtype=np.float32),
        raw=raw_list,
    )


def window_stream(amplitudes: np.ndarray, window_size: int, slide_step: int) -> np.ndarray:
    """
    Reproduce the backend's windowing offline. Returns [num_windows, subcarrier, time]
    (subcarrier-major, matching SnapshotSubcarrierMajor / the model input layout).
    """
    n = amplitudes.shape[0]
    starts = range(0, n - window_size + 1, slide_step)
    out = [amplitudes[s:s + window_size].T for s in starts]  # T → [subcarrier, time]
    return np.stack(out) if out else np.empty((0, amplitudes.shape[1], window_size), np.float32)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: read_csibin.py <session.json | session.csibin>")
        raise SystemExit(2)
    s = load_session(sys.argv[1])
    print(f"label={s.label!r} session={s.session_id} frames={s.amplitudes.shape[0]} "
          f"subcarriers={s.subcarrier_count} fs={s.sample_rate_hz}Hz raw={s.capture_raw}")