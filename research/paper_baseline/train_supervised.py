#!/usr/bin/env python3
"""Supervised (non-foundation-model) baseline under the evaluation skill: train a complete classifier from scratch on
one frozen label regime, select the epoch on the frozen coarse subset, score the test split once, evaluate with the
gates (validation.evaluate_predictions). The model module exposes

    NAME: str
    build(dataset) -> nn.Module            # forward(*tensors) -> one logit per event
    to_tensors(dataset, inputs) -> tuple   # reader inputs -> model input tensors (CPU)

Recipe = the paper's recipe for these runs: AdamW lr 5e-4, weight decay 1e-4, batch 64, CosineAnnealingLR(T_max = 50)
stepped once per epoch (no warm-up), gradient clipping 1, BCE without class weights, at most 50 epochs, seed 42; the
module may set per-dataset AMP (bf16 autocast), early-stopping patience (default 5) and selection metric (default
plain AUC, or the energy-matched AUC). Two data-regime rules of the frozen fine-tuning recipe make the baseline
comparable with a foundation model: selection on the frozen coarse subset of the validation split, and small label sets
cycled so that early stopping is armed only after 3000 optimizer steps.
Datasets listed in the module's TOKENIZED (with tokenize(dataset, item) / collate(tokens); point-cloud detectors) are tokenized once per regime into an
in-memory token bank (as the paper's NEXT token cache), with a process pool.

Output: runs/<NAME>/<dataset>/L<fraction>/s<seed>/ (train.json, model.pt, training_ids.npz, coarse/ and full
predictions.npz + metrics.json).

Usage: python research/paper_baseline/train_supervised.py --model-module research/paper_baseline/region_mlp.py \
           --dataset mjd --label-fraction 0.01 [--seed 42] [--max-hours 8]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from validation import coarse_cache, input_cache  # noqa: E402
from validation.finetune import RAM_BUDGET_GB, estimate_bytes, load_module, memory_batches, rank_auc, stream  # noqa: E402
from validation.freeze_splits import coarse_positions, labeled_subset_positions  # noqa: E402
from validation.readers import get_reader  # noqa: E402
from validation.readers.common import BENCHMARK, MANIFESTS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RECIPE = {"optimizer": "AdamW", "lr": 5e-4, "weight_decay": 1e-4, "betas": [0.9, 0.999], "batch_size": 64,
          "schedule": "CosineAnnealingLR(T_max=50) stepped per epoch, no warm-up", "cosine_t_max_epochs": 50,
          "gradient_clip_norm": 1.0, "epochs_max": 50, "early_stopping_patience_default": 5, "min_steps": 3000,
          "selection_default": "AUC on the frozen coarse subset of the validation split", "loss": "BCE with logits, no class weights",
          "precision_default": "fp32"}


class TokenBank:
    """Flat token arrays of a fixed list of events (coords [T, 3], features [T, f], offsets [N + 1])."""

    def __init__(self, coords, feats, offsets):
        self.coords, self.feats, self.offsets = coords, feats, offsets

    def __len__(self):
        return len(self.offsets) - 1

    def get(self, idx):
        return [(self.coords[self.offsets[i]:self.offsets[i + 1]], self.feats[self.offsets[i]:self.offsets[i + 1]]) for idx_ in [idx] for i in idx_]


_TOK = None


def _tok_init(module_path):
    global _TOK
    _TOK = load_module(module_path)[0]


def _tok_chunk(args):
    dataset, pos, items = args
    return pos, [_TOK.tokenize(dataset, x) for x in items]


def token_bank(mod_path, dataset, pairs, n_events, workers):
    """Tokenize the events of an iterator of (positions, items) with a process pool.
    Returns the bank (rows in iterator order) and the positions of its rows."""
    import multiprocessing as mp
    toks, order = [], []; t0 = time.time()
    with mp.get_context("fork").Pool(workers, initializer=_tok_init, initargs=(mod_path,)) as pool:
        def chunks():
            for pos, items in pairs:
                items = items if isinstance(items, list) else list(items)
                for s_ in range(0, len(items), 512):
                    yield dataset, np.asarray(pos[s_:s_ + 512]), items[s_:s_ + 512]
        for pos, out in pool.imap(_tok_chunk, chunks(), chunksize=1):
            toks.extend(out); order.append(pos)
            if len(toks) % 200000 < len(out):
                print(f"[{dataset}]   tokenized {len(toks)}/{n_events} ({time.time() - t0:.0f} s)", flush=True)
    assert len(toks) == n_events, (len(toks), n_events)
    off = np.r_[0, np.cumsum([len(c) for c, _ in toks])]
    return TokenBank(np.concatenate([c for c, _ in toks]), np.concatenate([f for _, f in toks]), off), np.concatenate(order)


def pack(mod, dataset, items):
    return [t.to(DEVICE) for t in mod.to_tensors(dataset, np.stack(items) if dataset in ("mjd", "exo200") else items)]


def autocast(on):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(on) and DEVICE == "cuda")


@torch.no_grad()
def score_items(model, mod, dataset, it, amp):
    model.eval(); out = []
    for x in it:
        with autocast(amp):
            out.append(model(*pack(mod, dataset, x if isinstance(x, list) else list(x))).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def score_bank(model, mod, bank, amp):
    model.eval(); out = []
    for s_ in range(0, len(bank), 512):
        x = [t.to(DEVICE) for t in mod.collate(bank.get(range(s_, min(len(bank), s_ + 512))))]
        with autocast(amp):
            out.append(model(*x).float().cpu().numpy())
    return np.concatenate(out)


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    mod, mod_sha = load_module(a.model_module)
    reader = get_reader(a.dataset)
    manifest = json.loads((MANIFESTS / f"{a.dataset}.json").read_text())
    train_t = reader.table("train")
    pos = labeled_subset_positions(len(train_t["event_id"]), a.label_fraction, manifest["label_regimes"]["seed"])
    runs_root = Path(os.environ.get("NLDBD_RUNS_ROOT", ROOT / "runs"))
    run = runs_root / mod.NAME / a.dataset / f"L{a.label_fraction}" / f"s{a.seed}"; run.mkdir(parents=True, exist_ok=True)
    if not coarse_cache.is_valid(a.dataset):
        val_t = reader.table("validation")
        coarse_cache.build(a.dataset, reader, val_t, coarse_positions(len(val_t["event_id"]), manifest["coarse"]["fraction"], manifest["coarse"]["seed"]))
    ids = np.load(MANIFESTS / manifest["ids_npz"], allow_pickle=False)
    coarse_ids, coarse_y = ids["coarse_event_id"].astype(str), ids["coarse_label"].astype(np.float32)
    amp = getattr(mod, "AMP", {}).get(a.dataset, False); patience = getattr(mod, "PATIENCE", {}).get(a.dataset, 5)
    selection = getattr(mod, "SELECTION", {}).get(a.dataset, "auc"); tokenized = a.dataset in getattr(mod, "TOKENIZED", ())
    def select_value(cs):
        if selection == "matched_auc":
            from validation.metrics import MetricContext, compute_all
            r = compute_all(["matched_auc"], MetricContext(label=ids["coarse_label"].astype(np.int8), score=cs.astype(np.float64),
                                                         energy_keV=ids["coarse_energy_keV"].astype(np.float64), group=ids["coarse_group"].astype(str)))
            v = r["matched_auc"].get("value")
            return float(v) if v is not None else -1.0
        return rank_auc(cs, coarse_y)
    assert np.array_equal(coarse_cache.cached_event_ids(a.dataset), coarse_ids)
    np.savez_compressed(run / "training_ids.npz", event_id=np.unique(np.concatenate([train_t["event_id"][pos], coarse_ids])),
                        labeled_event_id=train_t["event_id"][pos])
    model = mod.build(a.dataset).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"], betas=tuple(RECIPE["betas"]))
    bs = RECIPE["batch_size"]; steps_per_pass = math.ceil(len(pos) / bs)
    passes = max(1, math.ceil(RECIPE["min_steps"] / (patience * steps_per_pass)))
    planned = RECIPE["epochs_max"] * passes * steps_per_pass
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=RECIPE["cosine_t_max_epochs"])
    loss_fn = nn.BCEWithLogitsLoss(); rng = np.random.default_rng(a.seed)
    n_params = sum(p.numel() for p in model.parameters())
    est = estimate_bytes(reader, "train", train_t, pos); bank = None; tbank = cbank = None
    workers = int(os.environ.get("NLDBD_TOKENIZE_WORKERS", "12"))
    if tokenized:
        t0 = time.time()
        tbank, torder = token_bank(a.model_module, a.dataset, reader.iter_inputs("train", batch_size=4096, indices=pos, table=train_t), len(pos), workers)
        y_bank = train_t["label"][torder].astype(np.float32)      # iter_inputs yields in file order
        cbank, corder = token_bank(a.model_module, a.dataset, coarse_cache.iter_cached(a.dataset, batch_size=4096), len(coarse_ids), workers)
        assert np.array_equal(corder, np.arange(len(coarse_ids))), "coarse cache positions must be 0..n-1 in order"
        print(f"[{a.dataset}] token banks: {len(tbank)} train + {len(cbank)} coarse events, {tbank.coords.nbytes / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)
    elif est <= RAM_BUDGET_GB * 1e9:
        regime = f"L{a.label_fraction:g}"
        if not input_cache.is_valid(a.dataset, regime, train_t["event_id"][pos]):
            input_cache.build(a.dataset, regime, reader, "train", train_t, pos)
        items, cids = input_cache.load_all(a.dataset, regime)
        if not np.array_equal(cids, train_t["event_id"][pos]):
            raise RuntimeError("regime input cache order does not match the frozen regime")
        bank = (items, train_t["label"][pos])
    print(f"[{a.dataset}] {mod.NAME} L{a.label_fraction} s{a.seed}: {n_params} params, {len(pos)} labeled events "
          f"(~{est / 1e9:.1f} GB, {"token bank" if tokenized else ("in memory" if bank else "streamed")}), {passes} pass(es)/epoch, {planned} planned steps", flush=True)
    def coarse_scores():
        if cbank is not None:
            return score_bank(model, mod, cbank, amp)
        return score_items(model, mod, a.dataset, (x for _, x in coarse_cache.iter_cached(a.dataset, batch_size=256)), amp)

    def train_batches():
        if tbank is not None:
            order = rng.permutation(len(tbank))
            for s_ in range(0, len(order), bs):
                sel = order[s_:s_ + bs]
                yield [t.to(DEVICE) for t in mod.collate(tbank.get(sel))], y_bank[sel]
        else:
            for items_, y in (memory_batches(*bank, bs, rng) if bank else stream(reader, "train", train_t, pos, bs, rng)):
                yield pack(mod, a.dataset, items_), y

    best, best_state, bad, step, history, t_start, best_epoch = -1.0, None, 0, 0, [], time.time(), 0
    for epoch in range(RECIPE["epochs_max"]):
        model.train(); t0 = time.time(); total = 0.0; n = 0
        for _ in range(passes):
            for x, y in train_batches():
                yt = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
                opt.zero_grad()
                with autocast(amp):
                    loss = loss_fn(model(*x).float(), yt)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), RECIPE["gradient_clip_norm"]); opt.step()
                step += 1; total += float(loss.detach()) * len(y); n += len(y)
        sched.step()
        cs = coarse_scores()
        if not np.isfinite(cs).all():
            sys.exit(f"[{a.dataset}] non-finite outputs on {int((~np.isfinite(cs)).sum())} coarse events (epoch {epoch + 1})")
        v = select_value(cs); auc = rank_auc(cs, coarse_y)
        history.append({"epoch": epoch + 1, "steps": step, "train_loss": total / n, "coarse_auc": auc, "selection_value": v,
                        "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)})
        print(f"[{a.dataset}] epoch {epoch + 1} ({step} steps): loss {total / n:.4f} coarse AUC {auc:.4f} "
              f"selection ({selection}) {v:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if v > best:
            best, bad, best_epoch = v, 0, epoch + 1; best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        elif step >= RECIPE["min_steps"]:
            bad += 1
            if bad >= patience:
                print(f"[{a.dataset}] early stop (patience {patience})", flush=True); break
        if (time.time() - t_start) / 3600 > a.max_hours:
            print(f"[{a.dataset}] wall-clock cap reached", flush=True); break
    model.load_state_dict(best_state); torch.save(best_state, run / "model.pt")
    (run / "coarse").mkdir(exist_ok=True)
    np.savez_compressed(run / "coarse" / "predictions.npz", event_id=coarse_ids, score=coarse_scores().astype(np.float64))
    tbank = None
    test_t = reader.table("test"); scores = np.full(len(test_t["event_id"]), np.nan); model.eval()
    if tokenized:
        tb, tord = token_bank(a.model_module, a.dataset, reader.iter_inputs("test", batch_size=4096, table=test_t), len(test_t["event_id"]), workers)
        scores[tord] = score_bank(model, mod, tb, amp); del tb
    else:
        scores = np.full(len(test_t["event_id"]), np.nan)
        with torch.no_grad():
            for p, x in reader.iter_inputs("test", batch_size=256, table=test_t):
                with autocast(amp):
                    scores[p] = model(*pack(mod, a.dataset, x if isinstance(x, list) else list(x))).float().cpu().numpy()
    np.savez_compressed(run / "predictions.npz", event_id=test_t["event_id"], score=scores)
    (run / "train.json").write_text(json.dumps({
        "model": mod.NAME, "model_module": str(a.model_module), "model_module_sha256": mod_sha, "model_info": getattr(mod, "MODEL_INFO", None),
        "parameters": n_params, "recipe": RECIPE, "benchmark_version": BENCHMARK["version"], "dataset": a.dataset,
        "label_fraction": a.label_fraction, "seed": a.seed, "labeled_events": int(len(pos)), "amp_bf16": bool(amp),
        "early_stopping_patience": patience, "selection": selection, "passes_per_epoch": passes, "history": history,
        "selected_epoch": best_epoch, "best_coarse_auc": best, "hours": round((time.time() - t_start) / 3600, 3), "device": DEVICE,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2, default=str) + "\n")
    for mode, out in (("coarse", run / "coarse"), ("full", run)):
        cmd = [sys.executable, "-B", "-m", "validation.evaluate_predictions", "--dataset", a.dataset, "--mode", mode,
               "--predictions", str(out / "predictions.npz"), "--training-ids", str(run / "training_ids.npz"),
               "--label-fraction", str(a.label_fraction), "--seed", str(a.seed), "--model", mod.NAME, "--output", str(out)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        sys.stdout.write(r.stdout); sys.stderr.write(r.stderr[-2000:])
        print(f"[{a.dataset}] {mode} evaluation exit {r.returncode} (0 = all gates passed)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-module", required=True); ap.add_argument("--dataset", required=True, choices=tuple(BENCHMARK["datasets"]))
    ap.add_argument("--label-fraction", type=float, required=True, choices=(0.01, 0.1, 1.0))
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--max-hours", type=float, default=8.0)
    main(ap.parse_args())
