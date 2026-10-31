#!/usr/bin/env python3
"""Small, fast model variants to probe the metric, each evaluated in both modes:

  feat_logreg   logistic regression on a dozen handcrafted shape features (energy-free inputs)
  random_init   the tiny network with random weights, no training (expect ~0.5 in both metrics)
  cheat_energy  the tiny network PLUS the true event energy as an input (deliberate violation of
                rule 5, as a metric test): inclusive AUC should jump, matched AUC should not reward it

Predictions are written directly (own code path, as the skill allows) because cheat_energy needs
the energy per event, which the make_predictions contract never provides.
Usage: python examples/train_small_variants.py --dataset mjd --label-fraction 0.1 --variants feat_logreg,random_init,cheat_energy
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "examples"))
from validation.readers import get_reader  # noqa: E402
from validation.readers.common import MANIFESTS  # noqa: E402
from validation.freeze_splits import labeled_subset_positions, coarse_positions  # noqa: E402
import tiny_models  # noqa: E402
from train_tiny import read_inputs, batches, auc  # noqa: E402

DEV = tiny_models.DEVICE


def features(dataset, inputs):
    """Handcrafted, energy-free shape features (per event)."""
    if dataset == "mjd":
        x = np.asarray(inputs)[:, 0, :]
        t = np.arange(x.shape[1]) / x.shape[1]
        f = [(x > 0.1).argmax(1) / x.shape[1], (x > 0.5).argmax(1) / x.shape[1], (x > 0.9).argmax(1) / x.shape[1],
             (x > 0.5).mean(1), (x > 0.9).mean(1), x[:, -500:].mean(1), np.abs(np.diff(x, axis=1)).max(1),
             x.std(1), x[:, :200].std(1), (x * t).sum(1) / np.maximum(x.sum(1), 1e-6)]
    elif dataset == "exo200":
        x = np.asarray(inputs); blocks = [(0, 38), (38, 76), (76, 114), (114, 152), (152, 189), (189, 226)]
        f = []
        for a, b in blocks:
            xb = np.abs(x[:, a:b, :])
            f += [xb.max((1, 2)), xb.mean((1, 2)), (xb.max(2) > 0.2).sum(1)]
        f += [np.abs(x).reshape(len(x), -1).argmax(1) % 300 / 300, (np.abs(x).max(2) > 0.2).sum(1)]
    else:
        f = []
        for h in inputs:
            h = np.asarray(h); xyz = h[:, :3]; n = len(h)
            ext = xyz.std(0) if n > 1 else np.zeros(3); diag = np.linalg.norm(xyz.max(0) - xyz.min(0)) if n else 0.0
            if dataset == "next":
                e = h[:, 3]; top = np.sort(e)[::-1][: max(1, n // 10)].sum() / max(e.sum(), 1e-12)
                f.append([np.log1p(n), *ext / 1000, diag / 1000, top, (e > 2 * e.mean()).mean() if n else 0.0])
            else:
                r = h[:, 3]; v = np.isfinite(r)
                f.append([np.log1p(n), *ext / 1000, diag / 1000, v.mean() if n else 0.0, np.nan_to_num(r[v]).mean() / 24 if v.any() else 0.0])
        return np.asarray(f, dtype=np.float32)
    return np.stack(f, 1).astype(np.float32)


class CheatNet(nn.Module):
    def __init__(self, backbone):
        super().__init__(); self.backbone = backbone; self.head = nn.Linear(2, 1)
        nn.init.constant_(self.head.weight, 1.0); nn.init.zeros_(self.head.bias)

    def forward(self, *x_and_z):
        *x, z = x_and_z
        return self.head(torch.stack([self.backbone(*x), z], 1)).squeeze(1)


def to_dev(dataset, xs):
    return [t.to(DEV) for t in tiny_models.to_tensors(dataset, np.stack(xs) if dataset in ("mjd", "exo200") else xs)]


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    reader = get_reader(a.dataset)
    train_t, val_t, test_t = reader.table("train"), reader.table("validation"), reader.table("test")
    man = json.loads((MANIFESTS / f"{a.dataset}.json").read_text())
    pos = labeled_subset_positions(len(train_t["event_id"]), a.label_fraction, man["label_regimes"]["seed"])
    val_pos = np.sort(np.random.default_rng(0).choice(len(val_t["event_id"]), min(4000, len(val_t["event_id"])), replace=False))
    cpos = coarse_positions(len(val_t["event_id"]), man["coarse"]["fraction"], man["coarse"]["seed"])
    t0 = time.time()
    x_tr = read_inputs(reader, "train", train_t, pos, 256); y_tr = train_t["label"][pos].astype(np.float32); e_tr = train_t["energy_keV"][pos]
    x_va = read_inputs(reader, "validation", val_t, val_pos, 256); y_va = val_t["label"][val_pos].astype(np.float32); e_va = val_t["energy_keV"][val_pos]
    print(f"[{a.dataset}] {len(pos)} train / {len(val_pos)} val inputs read in {time.time() - t0:.0f} s", flush=True)
    e_mu, e_sd = np.nanmean(e_tr), np.nanstd(e_tr)
    zf = lambda e: np.nan_to_num((e - e_mu) / e_sd).astype(np.float32)
    for variant in a.variants.split(","):
        run = ROOT / "runs" / variant / a.dataset / f"L{a.label_fraction}" / f"s{a.seed}"; run.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(run / "training_ids.npz", event_id=np.concatenate([train_t["event_id"][pos], val_t["event_id"][val_pos]]), labeled_event_id=train_t["event_id"][pos])
        t0 = time.time()
        if variant == "feat_logreg":
            F = features(a.dataset, x_tr); mu, sd = F.mean(0), F.std(0) + 1e-6
            lin = nn.Linear(F.shape[1], 1).to(DEV); opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-4)
            Ft = torch.from_numpy((F - mu) / sd).to(DEV); yt = torch.from_numpy(y_tr).to(DEV)
            for _ in range(300):
                opt.zero_grad(); nn.functional.binary_cross_entropy_with_logits(lin(Ft).squeeze(1), yt).backward(); opt.step()
            scorer = lambda xs, es: lin(torch.from_numpy((features(a.dataset, xs) - mu) / sd).to(DEV)).squeeze(1).detach().cpu().numpy()
        elif variant == "random_init":
            net = tiny_models.build(a.dataset).to(DEV).eval()
            scorer = lambda xs, es: net(*to_dev(a.dataset, xs)).detach().cpu().numpy()
        elif variant == "cheat_energy":
            net = CheatNet(tiny_models.build(a.dataset)).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            rng = np.random.default_rng(a.seed); best, best_state = -1, None
            for epoch in range(a.epochs):
                net.train(); idx = rng.permutation(len(x_tr))
                for s in range(0, len(idx), 64):
                    sel = idx[s:s + 64]; xs = [x_tr[i] for i in sel]
                    z = torch.from_numpy(zf(e_tr[sel])).to(DEV); yt = torch.from_numpy(y_tr[sel]).to(DEV)
                    opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(net(*to_dev(a.dataset, xs), z), yt); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                net.eval()
                with torch.no_grad():
                    sv = np.concatenate([net(*to_dev(a.dataset, x_va[s:s + 256]), torch.from_numpy(zf(e_va[s:s + 256])).to(DEV)).cpu().numpy() for s in range(0, len(x_va), 256)])
                v = _rank_auc(sv, y_va)
                print(f"[{a.dataset}] cheat_energy epoch {epoch + 1}: val AUC {v:.4f}", flush=True)
                if v > best: best, best_state = v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
            net.load_state_dict(best_state); net.eval()
            scorer = lambda xs, es: net(*to_dev(a.dataset, xs), torch.from_numpy(zf(es)).to(DEV)).detach().cpu().numpy()
        else:
            raise ValueError(variant)
        print(f"[{a.dataset}] {variant}: trained in {time.time() - t0:.0f} s", flush=True)
        for mode, split, table, positions, out in (("coarse", "validation", val_t, cpos, run / "coarse"), ("full", "test", test_t, None, run)):
            out.mkdir(exist_ok=True)
            n = len(table["event_id"]) if positions is None else len(positions)
            scores = np.full(len(table["event_id"]), np.nan)
            with torch.no_grad():
                for p, xs in reader.iter_inputs(split, batch_size=256, indices=positions, table=table):
                    scores[p] = scorer(xs, table["energy_keV"][p])
            keep = np.arange(len(scores)) if positions is None else positions
            np.savez_compressed(out / "predictions.npz", event_id=table["event_id"][keep], score=scores[keep])
            cmd = [sys.executable, "-B", "-m", "validation.evaluate_predictions", "--dataset", a.dataset, "--mode", mode, "--predictions", str(out / "predictions.npz"),
                   "--training-ids", str(run / "training_ids.npz"), "--label-fraction", str(a.label_fraction), "--seed", str(a.seed), "--model", variant, "--output", str(out)]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); line = json.loads(r.stdout.strip().splitlines()[-1])
            m = json.loads((out / "metrics.json").read_text())
            print(f"[{a.dataset}] {variant} {mode}: matched {line['matched_auc']:.4f} inclusive {m['metrics']['inclusive_auc']['value']:.4f} valid={line['benchmark_valid']}", flush=True)


def _rank_auc(s, y):
    y = y.astype(bool); order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True); ap.add_argument("--label-fraction", type=float, default=0.01)
    ap.add_argument("--variants", default="feat_logreg,random_init,cheat_energy"); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
