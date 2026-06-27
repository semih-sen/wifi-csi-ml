# wifi-csi-ml

PyTorch training pipeline for the Wi-Fi CSI radar. Consumes the `.csibin` + `.json`
recordings produced by the C# backend and emits a contract-compliant `model.onnx`
that the backend loads via ML.NET `PredictionEnginePool`.

## Purpose of this milestone — a smoke test, not a model

Before collecting hours of data, we cut a small swatch to check the machine's stitch
settings: with **1–2 minute sample recordings of a couple of classes** (e.g.
`EmptyRoom`, `Walking`) we validate the **entire vertical slice** end to end —
reader → dataset → model → ONNX export → C# load → inference — and prove the ONNX
contract matches the backend.

**PASS criteria for this milestone (explicitly NOT accuracy):**

1. The pipeline runs end to end without errors and exports `model.onnx`.
2. The exported graph, run under `onnxruntime` in Python, matches the PyTorch
   model output within tolerance (export parity).
3. The C# backend loads `model.onnx`, feeds a real window, and gets back a
   sensible `InferenceResultDto` (`predictedLabel`, `confidence`, per-class `scores`).

What this milestone deliberately does **not** validate: classification accuracy,
generalization, or debounce tuning. Those require the full dataset and are a later
milestone. Do not read anything into the metrics here — with a handful of correlated
sessions they are meaningless by construction.

---

## 1. Project layout (isolated from the backend)

This is a **standalone Python project**, separate from `wifi-csi-backend`. No shared
virtualenv, no shared dependencies.

```
wifi-csi-ml/
├── README.md
├── requirements.txt
├── .gitignore                 # venv/, data/, artifacts/, checkpoints/, __pycache__/
├── data/
│   └── recordings/            # copy (or symlink) of the backend's Recordings/ folder
│                              #   each session = <stem>.csibin + <stem>.json
├── src/
│   ├── read_csibin.py         # COPIED VERBATIM from backend tools/ — the canonical reader
│   ├── dataset.py             # manifest globbing, integrity gating, windowing, session grouping
│   ├── model.py               # Csi1DCNN core + ExportWrapper (normalization + softmax baked in)
│   ├── train.py               # session-level split, class-weighted training loop, checkpointing
│   └── export_onnx.py         # bakes train mean/std, exports the contract, runs parity check
├── artifacts/
│   ├── model.onnx             # ← the deliverable handed to the C# backend
│   └── labels.json            # output-index → class-name map (handoff to backend)
└── checkpoints/
    └── best.pt
```

`read_csibin.py` is **copied, not reimplemented**. It is the single source of truth
for the binary layout and, critically, its `window_stream()` is the offline twin of
the backend's `SnapshotSubcarrierMajor`. Reimplementing it risks a silent train/serve
divergence — exactly what this whole architecture exists to prevent.

---

## 2. Setup (virtualenv + requirements)

Target: Python 3.11+, NVIDIA RTX 3050 Mobile (Ampere, CUDA-capable).

```bash
cd wifi-csi-ml
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
python -m pip install --upgrade pip
```

**Install PyTorch with CUDA separately** (the CUDA wheels come from PyTorch's own
index, not PyPI). Pick the build matching your driver from https://pytorch.org —
for this GPU a CUDA 12.x wheel is appropriate, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then the rest:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy>=1.26
scikit-learn>=1.4
onnx>=1.16
onnxruntime>=1.18
tqdm>=4.66
matplotlib>=3.8
```

Verify the GPU is visible before training (CPU works too — it's just slower, and you
said long runs are fine):

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

With 4 GB VRAM, use a modest batch size (32–64) and enable AMP (mixed precision) in
the training loop to fit and speed up. The 1D-CNN here is tiny, so this is generous.

---

## 3. Reading the data (`.csibin` + `.json`)

Each session is two files: a `.csibin` binary payload and a `.json` manifest. Workflow
in `dataset.py`:

1. **Glob manifests**, not binaries: iterate `data/recordings/*.json`.
2. **Integrity-gate every manifest before loading its payload:**

   | Manifest field | Gate |
   |---|---|
   | `complete` | must be `true` — drop incomplete sessions (dropped/skipped frames or interrupted) |
   | `baselineApplied` | must be **consistent across the whole dataset** (ideally all `true`); mixing baseline-corrected and uncorrected sessions mixes distributions |
   | `subcarrierCount` | must equal **64** — discard any stray 52-wide or legacy sessions |
   | `sampleRateHz`, `windowSize`, `slideStep` | must be consistent across sessions |

3. **Load** the surviving sessions with `read_csibin.load_session(manifest_path)` →
   `session.amplitudes` is `float32 [frame_count, 64]` (the model-facing filtered
   stream). Note: amplitudes are **baseline-subtracted and filtered, so they can be
   negative** — never assume non-negativity anywhere in the pipeline.
4. **Window each session** with `read_csibin.window_stream(amplitudes, windowSize,
   slideStep)` → `[num_windows, 64, 100]`, subcarrier-major. This reproduces the
   backend's window layout **bit for bit**; do not roll your own windowing.
5. **Assign labels** from `manifest["label"]` via a **fixed, explicit class order**
   (see §7) — and carry each window's **source `sessionId`** alongside it. That group
   key is mandatory for the split in §4.

---

## 4. ⚠️ CRITICAL RULE — split by SESSION, never by window

This is the single rule that, if broken, silently invalidates the entire project.

Because `slideStep < windowSize`, **consecutive windows overlap** (~50% at
`slideStep=50`). Two neighbouring windows are almost the same data. If you split
windows randomly into train/test, a window lands in train and its overlapping
neighbour lands in test → **data leakage** → test accuracy looks great and collapses
in the field. The metric becomes a lie.

**Rule:** an entire session goes wholly into train **or** wholly into test — never
split a session across both. Use the per-window `sessionId` as the group key:

```python
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=session_ids))
```

For the smoke test specifically: record **at least 2 short sessions per class** so the
session-level split path is actually exercised (1 train + 1 test session per class
minimum). If you only have 1 session per class, a real split is impossible — in that
degenerate case train on everything purely to validate export, and treat metrics as
non-existent. Either way, **build the session-grouped split into the code from day
one**, before real data arrives, so the habit and the infrastructure are correct.

(When you scale up: also keep sessions from the same physical setup — same day, same
position — on the same side of the split, and move to grouped k-fold.)

---

## 5. Model — 1D-CNN

Input tensor: `[batch, 64, 100]` — **channels = 64 subcarriers, length = 100 time
steps**. The subcarrier-major window maps onto this directly; no transpose needed.

We start with a compact 1D-CNN rather than an LSTM, deliberately: cleaner ONNX export
(no LSTM opset/op headaches on the ML.NET OnnxRuntime side), low inference latency for
the real-time chain, and a small graph. CSI window dynamics for presence/activity are
well within a CNN's reach. Keep an LSTM/GRU variant in reserve only if the CNN baseline
plateaus on real data.

Suggested `Csi1DCNN` core (defined in `model.py`):

| Stage | Spec |
|---|---|
| Block 1 | `Conv1d(64 → 64, k=5, pad=2)` → `BatchNorm1d` → `ReLU` → `MaxPool1d(2)` |
| Block 2 | `Conv1d(64 → 128, k=3, pad=1)` → `BatchNorm1d` → `ReLU` → `MaxPool1d(2)` |
| Block 3 | `Conv1d(128 → 128, k=3, pad=1)` → `BatchNorm1d` → `ReLU` |
| Head | `AdaptiveAvgPool1d(1)` → `Flatten` → `Dropout(0.3)` → `Linear(128 → num_classes)` |

The core returns **raw logits** (no softmax). Softmax and input normalization live in
the export wrapper (§6, §7) so they ship inside the ONNX graph — not in the training
loss, and not in C#.

Training (`train.py`): cross-entropy with **class weights** (the dataset is heavily
imbalanced toward `EmptyRoom`; compute weights from train-split counts), Adam/AdamW,
AMP, early stopping on validation loss, checkpoint best to `checkpoints/best.pt`.
Evaluate with a confusion matrix and per-class precision/recall — but remember the
smoke-test caveat: with tiny data these numbers are for plumbing validation only.

---

## 6. Normalization — baked INTO the ONNX graph

The second-biggest source of train/serve skew (after windowing) is normalization. We
eliminate it by computing standardization stats **from the training split only** and
embedding them as constants at the front of the exported graph. The backend then feeds
**raw filtered windows** and the graph normalizes internally — **zero extra code on
the C# side**, and skew is impossible.

- Compute **per-subcarrier** mean/std over the **training split only** (never the test
  split — that leaks), across all `(window, time)` samples → shape `[64]`, broadcast as
  `[1, 64, 1]` over `[B, 64, 100]`. Per-subcarrier (not a global scalar) because
  subcarriers carry different scales even after baseline subtraction.
- Wrap the trained core so the graph is `normalize → core → softmax`:

```python
class ExportWrapper(nn.Module):
    def __init__(self, core, mean, std):           # mean,std: tensors shaped [1,64,1]
        super().__init__()
        self.core = core
        self.register_buffer("mean", mean)          # become Constants in the ONNX graph
        self.register_buffer("std",  std)
    def forward(self, x):                            # x: [B,64,100] RAW filtered window
        x = (x - self.mean) / self.std
        return torch.softmax(self.core(x), dim=1)   # [B, num_classes] probabilities
```

Export the **wrapper**, never the bare core. The mean/std are frozen into the graph as
constants — the backend cannot accidentally diverge from them.

---

## 7. ONNX export contract

`export_onnx.py` must produce a graph the backend can bind without changes. The
backend's (currently commented) registration is:

```
.FromOnnxModel(modelFilePath: ..., inputColumnName: "input", outputColumnName: "output")
```

so the names are non-negotiable.

| Requirement | Value / rule |
|---|---|
| Input name | **`"input"`** (must match `inputColumnName`) |
| Output name | **`"output"`** (must match `outputColumnName`) |
| Input shape | `[batch, 64, 100]`, **dynamic batch axis** (backend feeds batch=1) |
| Output | **probabilities** (softmax in-graph) → `[batch, num_classes]` |
| Opset | a version ML.NET's OnnxRuntime supports (12–17 is safe); avoid exotic ops |

```python
torch.onnx.export(
    wrapper, torch.randn(1, 64, 100),
    "artifacts/model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
```

**Parity check (mandatory gate).** Immediately after export, run the same input
through `onnxruntime` and compare against the PyTorch wrapper output; assert they match
within tolerance (e.g. `atol=1e-4`). If they diverge, the export is broken — stop and
fix before involving the backend.

**Label map handoff.** Write `artifacts/labels.json` as the **ordered** class list, e.g.
`["EmptyRoom", "Walking"]`. The output channel order **is** this order: the backend
does `argmax → label` and `scores = {label: prob}` using it. The dataset must assign
integer labels in this exact same fixed order. A mismatch here is silent and vicious —
the model runs fine but every label is wrong. Pin the order once, share it via
`labels.json`, and never let the dataset and the export disagree.

**Input tensor shape note for the backend:** the backend's window snapshot is a flat,
subcarrier-major buffer (row-major `[64, 100]`), so reshaping it to `[1, 64, 100]` is
contiguous and correct. Confirm the C# `OnnxInput` declares the `[64, 100]` shape under
the `"input"` column before wiring inference.

---

## 8. Smoke-test runbook

End-to-end sequence for this milestone:

1. **Record** ≥2 short (1–2 min) sessions each for `EmptyRoom` and `Walking` via the
   backend's recorder. Confirm each manifest shows `complete: true`,
   `subcarrierCount: 64`, and a consistent `baselineApplied`.
2. **Stage data:** copy/symlink the backend `Recordings/` into `data/recordings/`.
3. **Train:** `python src/train.py` — verify the session-level split runs, loss
   decreases, and `checkpoints/best.pt` is written.
4. **Export:** `python src/export_onnx.py` — produces `artifacts/model.onnx` +
   `artifacts/labels.json`, and the parity check passes.
5. **Hand off:** give `model.onnx` + `labels.json` to the backend. Enable the
   `AddPredictionEnginePool` block and the commented inference line in
   `CsiProcessingBackgroundService`, with `OnnxInput`/`OnnxOutput` shaped to the
   `[64,100]` / `"input"` / `"output"` contract.
6. **Verify in C#:** feed a live (or recorded) window and confirm a coherent
   `ReceiveInference` payload reaches the frontend.

If all six pass, the stitch settings are correct and we can cut the real cloth: collect
the full, diverse, multi-session dataset and move to the accuracy milestone (with proper
metrics, debounce simulation, and threshold calibration).

---

## 9. Pitfalls checklist

- [ ] `read_csibin.py` is **copied** from the backend, not reimplemented.
- [ ] Split is **session-grouped** (`sessionId` group key) — never window-random.
- [ ] `complete == true` and `subcarrierCount == 64` gating applied before load.
- [ ] `baselineApplied` consistent across the whole dataset.
- [ ] Normalization stats computed on the **train split only**, baked into the graph.
- [ ] Graph is `normalize → core → softmax`; the wrapper is exported, not the core.
- [ ] Input/output named exactly `"input"` / `"output"`; batch axis dynamic.
- [ ] Export parity check vs PyTorch passes before backend handoff.
- [ ] `labels.json` order == output channel order == dataset label encoding.
- [ ] Negative amplitudes tolerated throughout (baseline-subtracted signal).
