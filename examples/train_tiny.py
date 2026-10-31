#!/usr/bin/env python3
"""Train a tiny model on one frozen label regime and write everything the harness needs.

Steps: load the frozen regime ids -> read their inputs through the official reader -> train a few
epochs with early stopping on a fixed validation subset -> save weights -> write training_ids.npz
(every event seen + every labeled event) -> score the full test split with make_predictions ->
evaluate with the gates. Output tree: runs/tiny/<dataset>/L<fraction>/s<seed>/.

Usage: python examples/train_tiny.py --dataset mjd --label-fraction 0.01 --seed 42 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
from validation.readers import get_reader  # noqa: E402
from validation.freeze_splits import labeled_subset_positions  # noqa: E402
import tiny_models  # noqa: E402

N_VALIDATION = 4000


def read_inputs(reader, split, table, positions, batch_size):
    out = [None] * len(positions)
    where = {p: i for i, p in enumerate(positions)}
    for pos, inputs in reader.iter_inputs(split, batch_size=batch_size, indices=positions, table=table):
        if isinstance(inputs, list):
            for p, x in zip(pos, inputs):
                out[where[p]] = x
        else:
            for p, x in zip(pos, inputs):
                out[where[p]] = x
    return out


def batches(dataset, inputs, labels, batch_size, rng=None):
    idx = np.arange(len(inputs)) if rng is None else rng.permutation(len(inputs))
    for s in range(0, len(idx), batch_size):
        sel = idx[s:s + batch_size]
        xs = [inputs[i] for i in sel]
        tensors = tiny_models.to_tensors(dataset, np.stack(xs) if dataset in ("mjd", "exo200") else xs)
        yield [t.to(tiny_models.DEVICE) for t in tensors], torch.as_tensor(labels[sel], dtype=torch.float32, device=tiny_models.DEVICE)


@torch.no_grad()
def auc(model, dataset, inputs, labels, batch_size):
    model.eval()
    s = np.concatenate([model(*x).float().cpu().numpy() for x, _ in batches(dataset, inputs, labels, batch_size)])
    y = labels.astype(bool)
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    run = ROOT / "runs" / "tiny" / a.dataset / f"L{a.label_fraction}" / f"s{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    reader = get_reader(a.dataset)
    t0 = time.time()
    train_t = reader.table("train"); val_t = reader.table("validation")
    manifest = json.loads((ROOT / "validation" / "manifests" / f"{a.dataset}.json").read_text())
    pos = labeled_subset_positions(len(train_t["event_id"]), a.label_fraction, manifest["label_regimes"]["seed"])
    if a.max_train is not None and len(pos) > a.max_train:  # smoke-test cap: NOT a frozen regime
        pos = np.sort(np.random.default_rng(a.seed).choice(pos, a.max_train, replace=False))
    val_pos = np.sort(np.random.default_rng(0).choice(len(val_t["event_id"]), min(N_VALIDATION, len(val_t["event_id"])), replace=False))
    print(f"[{a.dataset}] tables in {time.time() - t0:.0f} s; {len(pos)} labeled train events, {len(val_pos)} validation events", flush=True)
    t0 = time.time()
    x_train = read_inputs(reader, "train", train_t, pos, a.batch_size); y_train = train_t["label"][pos].astype(np.float32)
    x_val = read_inputs(reader, "validation", val_t, val_pos, a.batch_size); y_val = val_t["label"][val_pos].astype(np.float32)
    print(f"[{a.dataset}] inputs read in {time.time() - t0:.0f} s", flush=True)
    # training ids: everything whose inputs were used (train regime + validation subset), labels used: train regime only
    np.savez_compressed(run / "training_ids.npz", event_id=np.concatenate([train_t["event_id"][pos], val_t["event_id"][val_pos]]),
                        labeled_event_id=train_t["event_id"][pos])
    model = tiny_models.build(a.dataset).to(tiny_models.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(a.seed)
    best, best_state, history = -1.0, None, []
    for epoch in range(a.epochs):
        model.train(); t0 = time.time(); total = 0.0; n = 0
        for x, y in batches(a.dataset, x_train, y_train, a.batch_size, rng):
            opt.zero_grad(); loss = loss_fn(model(*x), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach()) * len(y); n += len(y)
        v = auc(model, a.dataset, x_val, y_val, a.batch_size)
        history.append({"epoch": epoch + 1, "train_loss": total / n, "validation_auc": v, "seconds": round(time.time() - t0, 1)})
        print(f"[{a.dataset}] epoch {epoch + 1}: loss {total / n:.4f} val AUC {v:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if v > best:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    weights_dir = run / "weights"; weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, weights_dir / f"{a.dataset}.pt")
    os.environ["TINY_WEIGHTS_DIR"] = str(weights_dir)  # make_predictions (subprocess) loads this run's weights
    (run / "training_log.json").write_text(json.dumps({"args": vars(a), "history": history, "best_validation_auc": best,
                                                       "device": tiny_models.DEVICE}, indent=2) + "\n")
    py = sys.executable
    for mode in (("coarse", "full") if a.mode == "both" else (a.mode,)):
        out_dir = run / "coarse" if mode == "coarse" else run
        cmd = [py, "-B", "-m", "validation.make_predictions", "--dataset", a.dataset, "--model-module", str(ROOT / "examples" / "tiny_models.py"),
               "--output", str(out_dir / "predictions.npz"), "--batch-size", str(a.batch_size)]
        if mode == "coarse":
            cmd += ["--subset", "coarse"]
        if a.limit_test:
            cmd += ["--limit", str(a.limit_test)]
        subprocess.run(cmd, cwd=ROOT, check=True)
        cmd = [py, "-B", "-m", "validation.evaluate_predictions", "--dataset", a.dataset, "--mode", mode, "--predictions", str(out_dir / "predictions.npz"),
               "--training-ids", str(run / "training_ids.npz"), "--seed", str(a.seed), "--model", "tiny", "--output", str(out_dir)]
        if a.max_train is None:
            cmd += ["--label-fraction", str(a.label_fraction)]
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        print(f"[{a.dataset}] {mode} evaluation exit code {rc} (0 = all gates passed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--label-fraction", type=float, default=0.01, choices=(0.01, 0.1, 1.0))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--mode", default="both", choices=("coarse", "full", "both"), help="coarse = frozen validation subset, full = test split")
    ap.add_argument("--max-train", type=int, default=None, help="smoke tests only: cap the labeled set (breaks the regime gate on purpose)")
    ap.add_argument("--limit-test", type=int, default=None, help="smoke tests only: score the first N test events")
    main(ap.parse_args())
