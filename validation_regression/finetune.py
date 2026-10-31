#!/usr/bin/env python3
"""Train an energy regressor on a foundation model (validation/fm_contract.md, unchanged) under the frozen
regression recipe (recipe.json) for one dataset and label regime, predict the test split, and evaluate it.

Output: runs_regression/<name>__<mode>/<dataset>/L<fraction>/s<seed>/ with finetune.json (recipe stamp, history,
selected epoch), training_ids.npz, regressor.pt, predictions.npz (test) and metrics.json.

Usage:
  python -m validation_regression.finetune --fm-module my_fm.py --name myfm --dataset mjd \
      --label-fraction 0.1 --mode probe [--seed 42] [--max-hours 3] [--pretraining-ids ids.npz]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.finetune import Head, load_module, pack  # noqa: E402
from validation.freeze_splits import labeled_subset_positions  # noqa: E402
from validation.readers.common import BENCHMARK as CLS_BENCHMARK  # noqa: E402
from validation_regression.evaluate import evaluate, headline_line  # noqa: E402
from validation_regression.views import BENCHMARK, get_view  # noqa: E402

RECIPE = json.loads((Path(__file__).resolve().parent / "recipe.json").read_text())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BUFFER = 8192
RAM_BUDGET_GB = float(os.environ.get("NLDBD_FINETUNE_RAM_GB", "8"))
RUNS = Path(os.environ.get("NLDBD_REGRESSION_RUNS", ROOT / "runs_regression"))
SCALE = 0.01   # loss unit: 1 % relative error


def scale_header(x, log_scale):
    """Research option --scale-header (models of branch R/raw-amplitude): write the amplitude scale into the SIDERIUS
    event-buffer header, header[6] = log scale and header[7] = 1, exactly as in that branch's training pools."""
    t = x[0]
    if not (t.dim() == 2 and t.shape[1] == 67808):
        raise ValueError("--scale-header needs the SIDERIUS event-buffer layout [B, 67808]")
    t = t.clone(); t[:, 6] = log_scale.to(t.dtype); t[:, 7] = 1.0
    return [t, *x[1:]]


def rawify(x, factor):
    """Undo the amplitude normalisation of the encoder input (research option --raw-input): multiply the waveform
    payload by factor = exp(log_scale - ref) (ref = median labeled log_scale, so values stay O(1)). Works on the
    reader layout ([B, 1, 3800] / [B, 226, 300]) and on the SIDERIUS event buffer ([B, 67808], 8 header slots)."""
    t = x[0]
    if t.dim() == 2 and t.shape[1] == 67808:
        t = torch.cat([t[:, :8], t[:, 8:] * factor[:, None]], 1)
    else:
        t = t * factor.view(-1, *([1] * (t.dim() - 1)))
    return [t, *x[1:]]


class Regressor(nn.Module):
    def __init__(self, encoder, dim, n_det, ls_mean, ls_std, c0, a0, raw=False, raw_ref=0.0, header=False):
        super().__init__()
        self.raw = raw; self.header = header; self.register_buffer("raw_ref", torch.tensor(float(raw_ref)))
        self.encoder = encoder; self.norm = nn.LayerNorm(dim); self.det = nn.Embedding(n_det, 8)
        self.mlp = nn.Sequential(nn.Linear(dim + 9, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.mlp[2].weight); nn.init.zeros_(self.mlp[2].bias)
        self.c = nn.Parameter(torch.as_tensor(c0, dtype=torch.float32)); self.a = nn.Parameter(torch.tensor(float(a0)))
        self.register_buffer("ls_mean", torch.tensor(float(ls_mean))); self.register_buffer("ls_std", torch.tensor(float(ls_std)))

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def from_embedding(self, emb, log_scale, det):
        z = torch.cat([self.norm(emb), ((log_scale - self.ls_mean) / self.ls_std)[:, None], self.det(det)], 1)
        return self.c[det] + self.a * log_scale + self.mlp(z).squeeze(1)

    def embed(self, x, log_scale=None):
        if self.raw:
            x = rawify(x, torch.exp(log_scale - self.raw_ref))
        if self.header and log_scale is not None:
            x = scale_header(x, log_scale)
        return Head.pool(self.encoder(*x))

    def forward(self, x, log_scale, det):
        return self.from_embedding(self.embed(x, log_scale), log_scale, det)


def calibration(log_scale, det, log_e, n_det, use_scale):
    """Least-squares start: log E = c[det] + a log_scale (a = 0 when the dataset has no amplitude scale)."""
    if not use_scale:
        return np.full(n_det, float(np.mean(log_e))), 0.0
    D = np.eye(n_det)[det]; X = np.c_[D, log_scale]
    w = np.linalg.lstsq(X, log_e, rcond=None)[0]
    c = w[:n_det]; seen = np.bincount(det, minlength=n_det) > 0; c[~seen] = c[seen].mean()
    return c, float(w[-1])


def collect(view, split, table, positions, keep_inputs):
    """One pass over the view: (inputs or None, log_scale, detector), aligned with ``positions``."""
    where = {p: i for i, p in enumerate(positions)}; n = len(positions)
    items = [None] * n if keep_inputs else None; ls = np.zeros(n, np.float32); det = np.zeros(n, np.int64)
    for pos, x, aux in view.iter_regression(split, indices=positions, table=table):
        idx = [where[p] for p in pos]
        ls[idx] = aux["log_scale"]; det[idx] = aux["detector"]
        if keep_inputs:
            for i, it in zip(idx, (x if isinstance(x, list) else list(x))):
                items[i] = it
    return items, ls, det


def stream(view, split, table, positions, target, batch_size, rng):
    """(items, log_scale, det, target) batches in file order, shuffled inside a buffer."""
    where = {p: i for i, p in enumerate(positions)}; buf = []
    def flush():
        order = rng.permutation(len(buf))
        for s in range(0, len(order), batch_size):
            sel = [buf[i] for i in order[s:s + batch_size]]
            yield [b[0] for b in sel], np.array([b[1] for b in sel], np.float32), np.array([b[2] for b in sel]), np.array([b[3] for b in sel], np.float32)
    for pos, x, aux in view.iter_regression(split, indices=positions, table=table):
        for k, it in enumerate(x if isinstance(x, list) else list(x)):
            buf.append((it, aux["log_scale"][k], aux["detector"][k], target[where[pos[k]]]))
        if len(buf) >= BUFFER:
            yield from flush(); buf = []
    if buf:
        yield from flush()


def loss_fn(pred, y):
    return nn.functional.huber_loss((pred - y) / SCALE, torch.zeros_like(pred), delta=1.0)


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    fm, fm_sha = load_module(a.fm_module)
    view = get_view(a.dataset); spec = BENCHMARK["datasets"][a.dataset]
    n_det = len(spec.get("detectors", [0])); use_scale = a.dataset != "supernemo"
    train_t = view.table("train"); val_t = view.table("validation")
    pos = labeled_subset_positions(len(train_t["event_id"]), a.label_fraction, CLS_BENCHMARK["label_regimes"]["seed"])
    pos = pos[train_t["energy_keV"][pos] > 1.0]
    if a.train_window_keV:   # research option (user 2026-10-25): train and select only on events whose reference energy is in a window
        lo, hi = a.train_window_keV
        pos = pos[(train_t["energy_keV"][pos] >= lo) & (train_t["energy_keV"][pos] <= hi)]
    sel_cfg = RECIPE["selection_subset"]; n_sel = min(sel_cfg["events"][a.dataset], len(val_t["event_id"]))
    spos = np.sort(np.random.default_rng(sel_cfg["seed"]).permutation(len(val_t["event_id"]))[:n_sel])
    spos = spos[val_t["energy_keV"][spos] > 1.0]
    if a.train_window_keV:
        spos = spos[(val_t["energy_keV"][spos] >= a.train_window_keV[0]) & (val_t["energy_keV"][spos] <= a.train_window_keV[1])]
    run = RUNS / f"{a.name}__{a.mode}" / a.dataset / f"L{a.label_fraction}" / f"s{a.seed}"; run.mkdir(parents=True, exist_ok=True)
    pre = np.load(a.pretraining_ids, allow_pickle=False)["event_id"].astype(str) if a.pretraining_ids else np.array([], dtype=str)
    np.savez_compressed(run / "training_ids.npz", event_id=np.unique(np.concatenate([train_t["event_id"][pos], val_t["event_id"][spos], pre])),
                        labeled_event_id=train_t["event_id"][pos])
    y_tr = np.log(train_t["energy_keV"][pos]).astype(np.float32); y_sel = np.log(val_t["energy_keV"][spos]).astype(np.float32)

    # inputs: the selection subset always in memory; labeled inputs in memory when they fit the RAM budget
    t0 = time.time()
    p64, x64, _ = next(view.iter_regression("train", indices=pos[:64], table=train_t))
    per = (sum(np.asarray(i).nbytes for i in x64) if isinstance(x64, list) else np.asarray(x64).nbytes) / len(p64)
    in_memory = per * len(pos) <= RAM_BUDGET_GB * 1e9
    tr_items, tr_ls, tr_det = collect(view, "train", train_t, pos, keep_inputs=in_memory)
    sel_items, sel_ls, sel_det = collect(view, "validation", val_t, spos, keep_inputs=True)
    print(f"[{a.dataset}] {len(pos)} labeled events ({per * len(pos) / 1e9:.2f} GB, {'in memory' if in_memory else 'streamed'}), "
          f"{len(spos)} selection events; read in {time.time() - t0:.0f} s", flush=True)
    c0, a0 = calibration(tr_ls.astype(np.float64), tr_det, y_tr.astype(np.float64), n_det, use_scale)
    ls_mean, ls_std = float(tr_ls.mean()), float(tr_ls.std() + 1e-6)

    encoder = fm.load_encoder(a.dataset, pretrained=(a.mode != "scratch"))
    dim = fm.EMBED_DIM[a.dataset] if isinstance(fm.EMBED_DIM, dict) else int(fm.EMBED_DIM)
    if (a.raw_input or a.scale_header) and (a.mode == "probe" or not use_scale):
        sys.exit("--raw-input / --scale-header need an amplitude scale (MJD, EXO-200) and an encoder that trains (full or scratch mode)")
    model = Regressor(encoder, dim, n_det, ls_mean, ls_std, c0, a0, raw=a.raw_input, raw_ref=float(np.median(tr_ls)),
                      header=a.scale_header).to(DEVICE)
    opt_cfg = RECIPE["optimizer"]
    groups = [{"params": model.head_params(), "lr": opt_cfg["lr_head"]}]
    if a.mode == "probe":
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    else:
        groups.append({"params": model.encoder.parameters(), "lr": opt_cfg["lr_encoder_full"] if a.mode == "full" else opt_cfg["lr_encoder_scratch"]})
    opt = torch.optim.AdamW(groups, weight_decay=opt_cfg["weight_decay"], betas=tuple(opt_cfg["betas"]))
    bs = RECIPE["batch_size"]; steps_per_pass = math.ceil(len(pos) / bs)
    passes = max(1, math.ceil(RECIPE["min_steps"] / (RECIPE["early_stopping"]["patience"] * steps_per_pass)))
    planned = RECIPE["epochs_max"] * passes * steps_per_pass; warm = int(RECIPE["schedule"]["warmup_fraction"] * planned)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, planned - warm)))))
    rng = np.random.default_rng(a.seed)
    n_params = sum(p.numel() for p in model.parameters()); n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{a.dataset}] {a.name}__{a.mode} L{a.label_fraction} s{a.seed}: {n_params} params ({n_train} trainable), "
          f"calibration start a={a0:.4f}, {passes} pass(es)/epoch", flush=True)

    T = lambda v, dt=torch.float32: torch.as_tensor(v, dtype=dt, device=DEVICE)
    def embed_all(items):
        with torch.no_grad():
            return torch.cat([model.embed(pack(fm, a.dataset, items[s:s + 256])).float() for s in range(0, len(items), 256)])
    emb_tr = emb_sel = None
    if a.mode == "probe":
        t0 = time.time(); model.eval()
        if in_memory:
            emb_tr = embed_all(tr_items)
        else:
            e = torch.zeros(len(pos), dim, device=DEVICE); where = {p: i for i, p in enumerate(pos)}
            with torch.no_grad():
                for p_, x_, _ in view.iter_regression("train", batch_size=256, indices=pos, table=train_t):
                    e[[where[q] for q in p_]] = model.embed(pack(fm, a.dataset, x_ if isinstance(x_, list) else list(x_))).float()
            emb_tr = e
        emb_sel = embed_all(sel_items); tr_items = None
        print(f"[{a.dataset}] probe: {len(emb_tr)} + {len(emb_sel)} embeddings cached in {time.time() - t0:.0f} s", flush=True)

    def selection_loss():
        model.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(spos), 256):
                ls_, dt_ = T(sel_ls[s:s + 256]), T(sel_det[s:s + 256], torch.long)
                emb = emb_sel[s:s + 256] if emb_sel is not None else model.embed(pack(fm, a.dataset, sel_items[s:s + 256]), ls_)
                out.append(model.from_embedding(emb, ls_, dt_).float())
        p = torch.cat(out); y = T(y_sel)
        rel = (p - y).cpu().numpy()
        return float(loss_fn(p, y)), float(1.482602 * np.median(np.abs(rel - np.median(rel)))), bool(torch.isfinite(p).all())

    def train_batches():
        if emb_tr is not None or in_memory:
            order = rng.permutation(len(pos))
            for s in range(0, len(order), bs):
                i = order[s:s + bs]
                x = emb_tr[i] if emb_tr is not None else [tr_items[k] for k in i]
                yield x, tr_ls[i], tr_det[i], y_tr[i]
        else:
            yield from stream(view, "train", train_t, pos, y_tr, bs, rng)

    best, best_state, bad, step, history, t_start, best_epoch = float("inf"), None, 0, 0, [], time.time(), 0
    start_epoch, hours_done, finished = 0, 0.0, False
    key = {"fm_module_sha256": fm_sha, "dataset": a.dataset, "label_fraction": a.label_fraction, "mode": a.mode, "seed": a.seed,
           "recipe_version": RECIPE["version"]}
    state_file = run / "resume_state.pt"
    if a.resume and state_file.exists():   # continue a run interrupted at an allocation boundary (epoch checkpoints)
        st = torch.load(state_file, map_location="cpu", weights_only=False)
        if st["key"] != key:
            sys.exit(f"{state_file} belongs to a different run; delete it to start over")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["np_rng"]; torch.set_rng_state(st["torch_rng"])
        if DEVICE == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        best, best_state, bad, step, history = st["best"], st["best_state"], st["bad"], st["step"], st["history"]
        best_epoch, start_epoch, hours_done, finished = st["best_epoch"], st["epoch"], st["hours"], st["finished"]
        print(f"[{a.dataset}] resumed after epoch {start_epoch} ({hours_done:.2f} h done)", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(start_epoch, RECIPE["epochs_max"]):
        if finished:
            break
        model.train()
        if a.mode == "probe":
            model.encoder.eval()
        t0 = time.time(); total = 0.0; n = 0
        for _ in range(passes):
            for x, ls_, dt_, y in train_batches():
                ls_, dt_, y = T(ls_), T(dt_, torch.long), T(y)
                pred = model.from_embedding(x, ls_, dt_) if emb_tr is not None else model(pack(fm, a.dataset, x), ls_, dt_)
                opt.zero_grad(); loss = loss_fn(pred, y); loss.backward()
                nn.utils.clip_grad_norm_(params, opt_cfg["gradient_clip_norm"]); opt.step(); sched.step()
                step += 1; total += float(loss.detach()) * len(y); n += len(y)
        v, rel_sigma, finite = selection_loss()
        if not finite:
            sys.exit(f"[{a.dataset}] non-finite predictions on the selection subset (epoch {epoch + 1})")
        history.append({"epoch": epoch + 1, "steps": step, "train_loss": total / n, "selection_loss": v,
                        "selection_rel_sigma": rel_sigma, "seconds": round(time.time() - t0, 1)})
        print(f"[{a.dataset}] epoch {epoch + 1} ({step} steps): loss {total / n:.4f} selection {v:.4f} "
              f"(robust rel. sigma {100 * rel_sigma:.3f} %) ({time.time() - t0:.0f} s)", flush=True)
        if v < best - RECIPE["early_stopping"]["min_delta"]:
            best, bad, best_epoch = v, 0, epoch + 1; best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        elif step >= RECIPE["min_steps"]:
            bad += 1
            if bad >= RECIPE["early_stopping"]["patience"]:
                print(f"[{a.dataset}] early stop", flush=True); finished = True
        hours = hours_done + (time.time() - t_start) / 3600
        if hours > a.max_hours and not finished:
            print(f"[{a.dataset}] wall-clock cap reached", flush=True); finished = True
        if a.resume:
            torch.save({"key": key, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if DEVICE == "cuda" else None, "best": best,
                        "best_state": best_state, "bad": bad, "step": step, "history": history, "best_epoch": best_epoch,
                        "epoch": epoch + 1, "hours": hours, "finished": finished}, state_file.with_suffix(".tmp"))
            os.replace(state_file.with_suffix(".tmp"), state_file)
        if finished:
            break
    model.load_state_dict(best_state); model.eval(); torch.save(best_state, run / "regressor.pt")
    test_t = view.table("test"); e_pred = np.full(len(test_t["event_id"]), np.nan)
    with torch.no_grad():
        for p_, x_, aux in view.iter_regression("test", batch_size=256, table=test_t):
            e_pred[p_] = torch.exp(model(pack(fm, a.dataset, x_ if isinstance(x_, list) else list(x_)), T(aux["log_scale"]),
                                         T(aux["detector"], torch.long))).double().cpu().numpy()
    np.savez_compressed(run / "predictions.npz", event_id=test_t["event_id"], energy_keV=e_pred)
    m = evaluate(a.dataset, "test", run / "predictions.npz", run / "training_ids.npz", f"{a.name}__{a.mode}", table=test_t)
    (run / "metrics.json").write_text(json.dumps(m, indent=2, default=float) + "\n")
    (run / "finetune.json").write_text(json.dumps({
        "recipe_version": RECIPE["version"], "recipe": RECIPE, "benchmark": f"{BENCHMARK['name']} {BENCHMARK['version']}", "mode": a.mode,
        "name": a.name, "fm_module": str(a.fm_module), "fm_module_sha256": fm_sha, "fm_info": getattr(fm, "MODEL_INFO", None), "embed_dim": dim,
        "parameters": n_params, "trainable_parameters": n_train, "dataset": a.dataset, "label_fraction": a.label_fraction, "seed": a.seed,
        "labeled_events": int(len(pos)), "selection_events": int(len(spos)), "calibration_start": {"a": a0, "c": list(map(float, c0))},
        "n_pretraining_ids": int(len(pre)), "history": history, "selected_epoch": best_epoch, "best_selection_loss": best,
        "hours": round(hours_done + (time.time() - t_start) / 3600, 3), "resumable": bool(a.resume), "train_window_keV": a.train_window_keV,
        "raw_input": bool(a.raw_input), "scale_header": bool(a.scale_header), "device": DEVICE,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2, default=str) + "\n")
    print(headline_line(m), flush=True)
    (run / "resume_state.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fm-module", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", required=True, choices=tuple(BENCHMARK["datasets"]))
    ap.add_argument("--label-fraction", type=float, required=True, choices=tuple(RECIPE["label_regimes"]))
    ap.add_argument("--mode", required=True, choices=("probe", "full", "scratch"))
    ap.add_argument("--seed", type=int, default=RECIPE["seeds"]["default"])
    ap.add_argument("--max-hours", type=float, default=RECIPE["wall_clock_cap_hours_default"])
    ap.add_argument("--pretraining-ids", type=Path, default=None)
    ap.add_argument("--train-window-keV", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="research option: train and select only on events with reference energy in [LO, HI] keV (not the frozen recipe)")
    ap.add_argument("--raw-input", action="store_true",
                    help="research option: the encoder sees the un-normalised amplitude (waveform x scale), not only the head")
    ap.add_argument("--scale-header", action="store_true",
                    help="research option for R/raw-amplitude models: header[6] = log amplitude scale, header[7] = 1 in the event buffer")
    ap.add_argument("--resume", action="store_true", help="epoch checkpoints in <run>/resume_state.pt; continue an interrupted run")
    main(ap.parse_args())
