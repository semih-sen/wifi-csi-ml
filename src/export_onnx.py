"""
export_onnx.py — bake train stats, export the ONNX contract, run the parity gate (README §6, §7).

Run:  python src/export_onnx.py

Produces:
  artifacts/model.onnx   — graph: normalize -> core -> softmax, input "input", output "output"
  artifacts/labels.json  — ordered class list; output channel order IS this order

The parity check is a MANDATORY gate: if onnxruntime and the PyTorch wrapper disagree
beyond tolerance, the export is broken — we stop before involving the backend.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from dataset import ROOT
from model import Csi1DCNN, ExportWrapper

CKPT_PATH = os.path.join(ROOT, "checkpoints", "best.pt")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
ONNX_PATH = os.path.join(ARTIFACTS_DIR, "model.onnx")
LABELS_PATH = os.path.join(ARTIFACTS_DIR, "labels.json")

OPSET = 17
PARITY_ATOL = 1e-4


def main():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"No checkpoint at {CKPT_PATH}. Run: python src/train.py")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    in_channels = ckpt["in_channels"]
    window_size = ckpt["window_size"]
    print(f"[export] classes={classes} in_channels={in_channels} window_size={window_size}")

    # Rebuild the trained core, then wrap with frozen train-split normalization stats.
    core = Csi1DCNN(num_classes=len(classes), in_channels=in_channels)
    core.load_state_dict(ckpt["core_state_dict"])
    core.eval()

    wrapper = ExportWrapper(core, ckpt["mean"].float(), ckpt["std"].float())
    wrapper.eval()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Export the WRAPPER (normalize -> core -> softmax), never the bare core.
    dummy = torch.randn(1, in_channels, window_size)
    torch.onnx.export(
        wrapper, dummy, ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,  # legacy TorchScript exporter: classic, ML.NET-friendly graph
    )
    print(f"[export] wrote {ONNX_PATH}")

    # ---- mandatory parity gate: onnxruntime vs PyTorch wrapper -------------------
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(ONNX_PATH))

    # Use a batch > 1 to also exercise the dynamic batch axis.
    test_in = torch.randn(4, in_channels, window_size)
    with torch.no_grad():
        torch_out = wrapper(test_in).numpy()

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    ort_out = sess.run(["output"], {"input": test_in.numpy()})[0]

    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    print(f"[parity] max |torch - onnx| = {max_diff:.2e} (atol={PARITY_ATOL:.0e})")
    if max_diff > PARITY_ATOL:
        raise SystemExit(
            f"[parity] FAILED: divergence {max_diff:.2e} > {PARITY_ATOL:.0e}. "
            "Export is broken — fix before backend handoff."
        )

    # Sanity: each row is a probability distribution.
    sums = ort_out.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4), f"softmax rows do not sum to 1: {sums}"
    print(f"[parity] PASSED — output rows are probability distributions (sum~1).")

    # ---- label map handoff: order == output channel order == dataset encoding ----
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)
    print(f"[export] wrote {LABELS_PATH}: {classes}")

    print("\n[export] DONE. Contract: input='input' [batch,64,100], "
          "output='output' [batch,num_classes] probabilities.")


if __name__ == "__main__":
    main()
