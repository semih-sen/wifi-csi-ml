"""
train.py — session-level split, class-weighted training, checkpointing (README §4, §5).

Run:  python src/train.py

Writes checkpoints/best.pt holding everything export_onnx.py needs:
  - core state_dict
  - per-subcarrier train-split mean/std  (shape [1, 64, 1])  -> baked into ONNX later
  - classes (output-index order)         -> the single source of truth for labels

CRITICAL: the split is by SESSION, never by window (overlapping windows would leak
across train/test and turn every metric into a lie). With <2 sessions per class a
real split is impossible — we fall back to train-on-everything purely to validate the
export path, and metrics are treated as non-existent (README §4 degenerate case).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from dataset import ROOT, build_dataset
from model import Csi1DCNN

# ---- config -------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.3
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 15           # early stopping on validation loss
LR = 1e-3
WEIGHT_DECAY = 1e-4
CKPT_PATH = os.path.join(ROOT, "checkpoints", "best.pt")
STD_EPS = 1e-6          # guard against zero-variance subcarriers


def session_split(y: np.ndarray, groups: np.ndarray):
    """
    Group-aware train/val split by sessionId. Returns (train_idx, val_idx).
    val_idx is empty in the degenerate case (too few sessions to split safely).
    """
    n_sessions = len(np.unique(groups))
    if n_sessions < 2:
        print(f"[split] only {n_sessions} session(s) -> DEGENERATE: train on all, no validation.")
        return np.arange(len(y)), np.empty(0, dtype=int)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, val_idx = next(gss.split(np.zeros(len(y)), y, groups=groups))

    # Guard: training split must contain every class, else weights/classifier are invalid.
    if set(y[train_idx].tolist()) != set(y.tolist()):
        print("[split] split left a class out of training -> DEGENERATE fallback: train on all.")
        return np.arange(len(y)), np.empty(0, dtype=int)

    print(f"[split] {n_sessions} sessions -> train sessions={np.unique(groups[train_idx]).tolist()} "
          f"val sessions={np.unique(groups[val_idx]).tolist()}")
    return train_idx, val_idx


def compute_norm_stats(X_train: np.ndarray):
    """Per-subcarrier mean/std over (window, time) of the TRAIN split only -> [1, 64, 1]."""
    mean = X_train.mean(axis=(0, 2)).astype(np.float32)   # [64]
    std = X_train.std(axis=(0, 2)).astype(np.float32)     # [64]
    std = np.maximum(std, STD_EPS)
    return mean.reshape(1, -1, 1), std.reshape(1, -1, 1)


def make_loader(X, y, batch_size, shuffle):
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X), torch.from_numpy(y)
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(core, loader, criterion, device):
    core.eval()
    total_loss, n = 0.0, 0
    preds, trues = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = core(xb)
        total_loss += criterion(logits, yb).item() * xb.size(0)
        n += xb.size(0)
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(yb.cpu().numpy())
    return total_loss / max(n, 1), np.concatenate(preds), np.concatenate(trues)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[train] device={device} amp={use_amp}")

    ds = build_dataset()
    train_idx, val_idx = session_split(ds.y, ds.groups)
    has_val = len(val_idx) > 0

    X_train, y_train = ds.X[train_idx], ds.y[train_idx]
    mean, std = compute_norm_stats(X_train)

    # Train the core on NORMALIZED inputs; ExportWrapper re-applies the same stats at
    # serve time, so the core sees the same distribution in both worlds.
    Xn_train = (X_train - mean) / std
    train_loader = make_loader(Xn_train.astype(np.float32), y_train, BATCH_SIZE, shuffle=True)

    if has_val:
        Xn_val = ((ds.X[val_idx] - mean) / std).astype(np.float32)
        val_loader = make_loader(Xn_val, ds.y[val_idx], BATCH_SIZE, shuffle=False)

    # Class weights from TRAIN-split counts (dataset skews heavily to EmptyRoom).
    counts = np.bincount(y_train, minlength=len(ds.classes)).astype(np.float32)
    weights = np.where(counts > 0, counts.sum() / (len(ds.classes) * counts), 0.0)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[train] class counts={counts.tolist()} weights={weights.round(3).tolist()}")

    core = Csi1DCNN(num_classes=len(ds.classes), in_channels=ds.X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(core.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        core.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(core(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / max(n, 1)

        # Monitor val loss when available, else train loss (degenerate case).
        if has_val:
            val_loss, _, _ = evaluate(core, val_loader, criterion, device)
            monitor = val_loss
            print(f"[epoch {epoch:3d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        else:
            monitor = train_loss
            print(f"[epoch {epoch:3d}] train_loss={train_loss:.4f} (no val)")

        if monitor < best_loss - 1e-5:
            best_loss = monitor
            best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[train] early stopping at epoch {epoch} (no improvement for {PATIENCE}).")
                break

    assert best_state is not None
    core.load_state_dict(best_state)

    # Smoke-test metrics — plumbing validation only, NOT accuracy (README §1).
    if has_val:
        _, preds, trues = evaluate(core, val_loader, criterion, device)
        labels_idx = list(range(len(ds.classes)))
        print("\n[eval] confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(trues, preds, labels=labels_idx))
        print("\n[eval] per-class report:")
        print(classification_report(trues, preds, labels=labels_idx,
                                    target_names=ds.classes, zero_division=0))
    else:
        print("\n[eval] skipped — degenerate single-session run, metrics are non-existent.")

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save({
        "core_state_dict": core.state_dict(),
        "classes": ds.classes,
        "mean": torch.from_numpy(mean),   # [1, 64, 1]
        "std": torch.from_numpy(std),     # [1, 64, 1]
        "window_size": ds.window_size,
        "slide_step": ds.slide_step,
        "sample_rate_hz": ds.sample_rate_hz,
        "baseline_applied": ds.baseline_applied,
        "in_channels": ds.X.shape[1],
        "best_monitor_loss": best_loss,
        "has_validation": has_val,
    }, CKPT_PATH)
    print(f"\n[train] saved best checkpoint -> {CKPT_PATH}")


if __name__ == "__main__":
    main()
