#!/usr/bin/env python3
"""Fine-tune a foundation model (fm_contract.md) under the frozen recipe (finetune_recipe.json)
on one dataset and label regime, then score and evaluate in both modes.

Output: runs/<name>__<mode>/<dataset>/L<fraction>/s<seed>/ with finetune.json (recipe stamp,
history, selected epoch), training_ids.npz, coarse/ and full predictions + metrics.json.

Usage:
  python -m validation.finetune --fm-module my_fm.py --name myfm --dataset mjd \
      --label-fraction 0.1 --mode full [--seed 42] [--max-hours 2] [--pretraining-ids ids.npz]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.readers import get_reader  # noqa: E402
from validation.readers.common import BENCHMARK, MANIFESTS  # noqa: E402
from validation.freeze_splits import labeled_subset_positions  # noqa: E402
from validation import coarse_cache  # noqa: E402

RECIPE = json.loads((ROOT / "validation" / "finetune_recipe.json").read_text())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BUFFER = 8192


def load_module(spec: str):
    path = Path(spec)
    if path.suffix == ".py" and path.exists():
        s = importlib.util.spec_from_file_location(path.stem, path); m = importlib.util.module_from_spec(s)
        sys.modules[path.stem] = m; s.loader.exec_module(m)
        return m, hashlib.sha256(path.read_bytes()).hexdigest()
    m = importlib.import_module(spec)
    return m, hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()


class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim); self.net = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1))

    @staticmethod
    def pool(out):
        """Parameter-free masked mean pooling over tokens (an event embedding passes through unchanged)."""
        if isinstance(out, (tuple, list)):
            tokens, mask = out; m = mask.float()[..., None]
            return (tokens * m).sum(1) / m.sum(1).clamp(min=1.0)
        return out

    def from_embedding(self, emb):
        return self.net(self.norm(emb)).squeeze(1)

    def forward(self, out):
        return self.from_embedding(self.pool(out))


class Classifier(nn.Module):
    def __init__(self, encoder, dim):
        super().__init__(); self.encoder = encoder; self.head = Head(dim)

    def forward(self, *x):
        return self.head(self.encoder(*x))


def rank_auc(s, y):
    y = y.astype(bool); order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def stream(reader, split, table, positions, batch_size, rng):
    """(inputs, labels) batches: read in file order, shuffle inside a buffer."""
    bx, by = [], []
    def flush():
        X = bx if isinstance(bx[0], list) else bx
        items = [i for b in X for i in (b if isinstance(b, list) else list(b))]
        Y = np.concatenate(by); order = rng.permutation(len(Y)) if rng is not None else np.arange(len(Y))
        for s in range(0, len(order), batch_size):
            sel = order[s:s + batch_size]
            yield [items[i] for i in sel], Y[sel]
    n = 0
    for pos, x in reader.iter_inputs(split, batch_size=256, indices=positions, table=table):
        bx.append(x if isinstance(x, list) else list(x)); by.append(table["label"][pos]); n += len(pos)
        if n >= BUFFER:
            yield from flush(); bx, by, n = [], [], 0
    if bx:
        yield from flush()


RAM_BUDGET_GB = float(__import__("os").environ.get("NLDBD_FINETUNE_RAM_GB", "8"))


def load_in_memory(reader, split, table, positions):
    """Read the labeled inputs once (used when they fit the RAM budget); returns (items, labels)."""
    items = [None] * len(positions); where = {p: i for i, p in enumerate(positions)}
    for pos, x in reader.iter_inputs(split, batch_size=256, indices=positions, table=table):
        for p, item in zip(pos, (x if isinstance(x, list) else list(x))):
            items[where[p]] = item
    return items, table["label"][positions]


def memory_batches(items, labels, batch_size, rng):
    order = rng.permutation(len(items))
    for s in range(0, len(order), batch_size):
        sel = order[s:s + batch_size]
        yield [items[i] for i in sel], labels[sel]


def estimate_bytes(reader, split, table, positions, n_probe=64):
    pos, x = next(reader.iter_inputs(split, batch_size=n_probe, indices=positions[:n_probe], table=table))
    per = (sum(np.asarray(i).nbytes for i in x) if isinstance(x, list) else np.asarray(x).nbytes) / max(1, len(pos))
    return per * len(positions)


def pack(fm, dataset, items):
    return [t.to(DEVICE) for t in fm.to_tensors(dataset, np.stack(items) if dataset in ("mjd", "exo200") else items)]


@torch.no_grad()
def score_coarse(model, fm, dataset):
    model.eval(); out = []
    for _, x in coarse_cache.iter_cached(dataset, batch_size=256):
        out.append(model(*pack(fm, dataset, x if isinstance(x, list) else list(x))).float().cpu().numpy())
    return np.concatenate(out)


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    fm, fm_sha = load_module(a.fm_module)
    reader = get_reader(a.dataset)
    manifest = json.loads((MANIFESTS / f"{a.dataset}.json").read_text())
    train_t = reader.table("train")
    pos = labeled_subset_positions(len(train_t["event_id"]), a.label_fraction, manifest["label_regimes"]["seed"])
    runs_root = Path(__import__("os").environ.get("NLDBD_RUNS_ROOT", ROOT / "runs"))
    run = runs_root / f"{a.name}__{a.mode}" / a.dataset / f"L{a.label_fraction}" / f"s{a.seed}"; run.mkdir(parents=True, exist_ok=True)
    # coarse subset inputs (cached) and labels, for model selection only
    if not coarse_cache.is_valid(a.dataset):
        val_t = reader.table("validation")
        from validation.freeze_splits import coarse_positions
        cpos = coarse_positions(len(val_t["event_id"]), manifest["coarse"]["fraction"], manifest["coarse"]["seed"])
        coarse_cache.build(a.dataset, reader, val_t, cpos)
    ids = np.load(MANIFESTS / manifest["ids_npz"], allow_pickle=False)
    coarse_ids, coarse_y = ids["coarse_event_id"].astype(str), ids["coarse_label"].astype(np.float32)
    assert np.array_equal(coarse_cache.cached_event_ids(a.dataset), coarse_ids)
    pre = np.load(a.pretraining_ids, allow_pickle=False)["event_id"].astype(str) if a.pretraining_ids else np.array([], dtype=str)
    np.savez_compressed(run / "training_ids.npz", event_id=np.unique(np.concatenate([train_t["event_id"][pos], coarse_ids, pre])),
                        labeled_event_id=train_t["event_id"][pos])
    encoder = fm.load_encoder(a.dataset, pretrained=(a.mode != "scratch"))
    dim = fm.EMBED_DIM[a.dataset] if isinstance(fm.EMBED_DIM, dict) else int(fm.EMBED_DIM)
    model = Classifier(encoder, dim).to(DEVICE)
    if a.mode == "probe":
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        groups = [{"params": model.head.parameters(), "lr": RECIPE["optimizer"]["lr_head"]}]
    else:
        lr_enc = RECIPE["optimizer"]["lr_encoder_full"] if a.mode == "full" else RECIPE["optimizer"]["lr_encoder_scratch"]
        groups = [{"params": model.head.parameters(), "lr": RECIPE["optimizer"]["lr_head"]},
                  {"params": model.encoder.parameters(), "lr": lr_enc}]
    opt = torch.optim.AdamW(groups, weight_decay=RECIPE["optimizer"]["weight_decay"], betas=tuple(RECIPE["optimizer"]["betas"]))
    bs = RECIPE["batch_size"]; steps_per_pass = math.ceil(len(pos) / bs)
    passes_per_epoch = max(1, math.ceil(RECIPE["min_steps"] / (RECIPE["early_stopping"]["patience"] * steps_per_pass)))  # cycle small sets
    planned = RECIPE["epochs_max"] * passes_per_epoch * steps_per_pass
    warm = int(RECIPE["schedule"]["warmup_fraction"] * planned)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, planned - warm)))))
    loss_fn = nn.BCEWithLogitsLoss(); rng = np.random.default_rng(a.seed)
    n_params = sum(p.numel() for p in model.parameters()); n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{a.dataset}] {a.name}__{a.mode} L{a.label_fraction} s{a.seed}: {n_params} params ({n_train} trainable), {len(pos)} labeled events, "
          f"{passes_per_epoch} pass(es)/epoch, early stopping armed after {RECIPE['min_steps']} steps", flush=True)
    est = estimate_bytes(reader, "train", train_t, pos)
    bank = None
    if est <= RAM_BUDGET_GB * 1e9:
        from validation import input_cache
        regime = f"L{a.label_fraction:g}"
        if not input_cache.is_valid(a.dataset, regime, train_t["event_id"][pos]):
            print(f"[{a.dataset}] building the {regime} input cache at {input_cache.path_for(a.dataset, regime)} (one time)", flush=True)
            input_cache.build(a.dataset, regime, reader, "train", train_t, pos)
        items, ids = input_cache.load_all(a.dataset, regime)
        if not np.array_equal(ids, train_t["event_id"][pos]):
            raise RuntimeError("regime input cache order does not match the frozen regime; delete the cache file")
        bank = (items, train_t["label"][pos])
    print(f"[{a.dataset}] labeled inputs ~{est / 1e9:.2f} GB -> {'held in memory (from the regime cache)' if bank else 'streamed from disk each pass'}", flush=True)
    emb_train = emb_coarse = None
    if a.mode == "probe":
        # The encoder is frozen: its (eval-mode) embeddings never change, so every input is encoded exactly once and the
        # head trains on cached pooled embeddings. Identical arithmetic to re-encoding each pass, at a fraction of the time.
        t0 = time.time(); model.eval()
        with torch.no_grad():
            if bank:
                items, labels = bank
                emb_train = torch.cat([model.head.pool(model.encoder(*pack(fm, a.dataset, items[s:s + 256]))).float().cpu()
                                       for s in range(0, len(items), 256)]); y_train = np.asarray(labels, dtype=np.float32)
            else:
                e, ys = [], []
                for items, y in stream(reader, "train", train_t, pos, 256, None):
                    e.append(model.head.pool(model.encoder(*pack(fm, a.dataset, items))).float().cpu()); ys.append(y)
                emb_train = torch.cat(e); y_train = np.concatenate(ys).astype(np.float32)
            emb_coarse = torch.cat([model.head.pool(model.encoder(*pack(fm, a.dataset, x if isinstance(x, list) else list(x)))).float().cpu()
                                    for _, x in coarse_cache.iter_cached(a.dataset, batch_size=256)]).to(DEVICE)
        print(f"[{a.dataset}] probe: {len(emb_train)} labeled + {len(emb_coarse)} coarse embeddings cached in {time.time() - t0:.0f} s", flush=True)

    def coarse_scores():
        if emb_coarse is not None:
            model.eval()
            with torch.no_grad():
                return model.head.from_embedding(emb_coarse).float().cpu().numpy()
        return score_coarse(model, fm, a.dataset)

    best, best_state, bad, step, history, t_start = -1.0, None, 0, 0, [], time.time()
    for epoch in range(RECIPE["epochs_max"]):
        model.train(); t0 = time.time(); total = 0.0; n = 0
        if emb_train is not None:
            model.encoder.eval()
            for _ in range(passes_per_epoch):
                order = rng.permutation(len(emb_train))
                for s_ in range(0, len(order), bs):
                    sel = order[s_:s_ + bs]; yt = torch.as_tensor(y_train[sel], dtype=torch.float32, device=DEVICE)
                    opt.zero_grad(); loss = loss_fn(model.head.from_embedding(emb_train[sel].to(DEVICE)), yt); loss.backward()
                    nn.utils.clip_grad_norm_(model.head.parameters(), RECIPE["optimizer"]["gradient_clip_norm"])
                    opt.step(); sched.step(); step += 1; total += float(loss.detach()) * len(sel); n += len(sel)
        else:
            for _ in range(passes_per_epoch):
                for items, y in (memory_batches(*bank, bs, rng) if bank else stream(reader, "train", train_t, pos, bs, rng)):
                    x = pack(fm, a.dataset, items); yt = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
                    opt.zero_grad(); loss = loss_fn(model(*x), yt); loss.backward()
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], RECIPE["optimizer"]["gradient_clip_norm"])
                    opt.step(); sched.step(); step += 1; total += float(loss.detach()) * len(y); n += len(y)
        cs = coarse_scores()
        if not np.isfinite(cs).all():   # NaN/inf from the encoder: no epoch can be selected, report it explicitly
            sys.exit(f"[{a.dataset}] the encoder produced non-finite outputs for {int((~np.isfinite(cs)).sum())} of {len(cs)} "
                     f"coarse events (epoch {epoch + 1}); check NaN handling in the model's tokeniser for {a.dataset}")
        v = rank_auc(cs, coarse_y)
        history.append({"epoch": epoch + 1, "steps": step, "train_loss": total / n, "coarse_auc": v, "seconds": round(time.time() - t0, 1)})
        print(f"[{a.dataset}] epoch {epoch + 1} ({step} steps): loss {total / n:.4f} coarse AUC {v:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if v > best + RECIPE["early_stopping"]["min_delta"]:
            best, bad, best_epoch = v, 0, epoch + 1; best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        elif step >= RECIPE["min_steps"]:
            bad += 1
            if bad >= RECIPE["early_stopping"]["patience"]:
                print(f"[{a.dataset}] early stop", flush=True); break
        if (time.time() - t_start) / 3600 > a.max_hours:
            print(f"[{a.dataset}] wall-clock cap reached", flush=True); break
    model.load_state_dict(best_state); model.eval()
    torch.save(best_state, run / "classifier.pt")
    # coarse predictions from the cache, full predictions through the reader (test split touched once)
    (run / "coarse").mkdir(exist_ok=True)
    np.savez_compressed(run / "coarse" / "predictions.npz",
                        event_id=coarse_ids, score=coarse_scores().astype(np.float64))
    if "full" in a.eval_modes:  # the test split is touched only when a full evaluation is requested
        test_t = reader.table("test"); scores = np.full(len(test_t["event_id"]), np.nan)
        with torch.no_grad():
            for p, x in reader.iter_inputs("test", batch_size=256, table=test_t):
                scores[p] = model(*pack(fm, a.dataset, x if isinstance(x, list) else list(x))).float().cpu().numpy()
        np.savez_compressed(run / "predictions.npz", event_id=test_t["event_id"], score=scores)
    (run / "finetune.json").write_text(json.dumps({
        "recipe_version": RECIPE["version"], "recipe": RECIPE, "benchmark_version": BENCHMARK["version"], "mode": a.mode, "name": a.name,
        "fm_module": str(a.fm_module), "fm_module_sha256": fm_sha, "fm_info": getattr(fm, "MODEL_INFO", None), "embed_dim": dim,
        "parameters": n_params, "trainable_parameters": n_train, "dataset": a.dataset, "label_fraction": a.label_fraction, "seed": a.seed,
        "labeled_events": int(len(pos)), "eval_modes": list(a.eval_modes), "pretraining_ids_file": str(a.pretraining_ids) if a.pretraining_ids else None, "n_pretraining_ids": int(len(pre)),
        "history": history, "selected_epoch": best_epoch, "best_coarse_auc": best, "hours": round((time.time() - t_start) / 3600, 3),
        "device": DEVICE, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2, default=str) + "\n")
    for mode, out in (("coarse", run / "coarse"), ("full", run)):
        if mode not in a.eval_modes:
            continue
        cmd = [sys.executable, "-B", "-m", "validation.evaluate_predictions", "--dataset", a.dataset, "--mode", mode, "--predictions", str(out / "predictions.npz"),
               "--training-ids", str(run / "training_ids.npz"), "--label-fraction", str(a.label_fraction), "--seed", str(a.seed),
               "--model", f"{a.name}__{a.mode}", "--output", str(out)]
        (out / "metrics.json").unlink(missing_ok=True)
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
        if not (out / "metrics.json").exists():   # the evaluator crashed: fail loudly instead of dropping the dataset
            n_bad = int((~np.isfinite(np.load(out / "predictions.npz")["score"])).sum())
            sys.exit(f"[{a.dataset}] evaluation ({mode}) wrote no metrics (exit {r.returncode}); non-finite scores: {n_bad}; "
                     f"evaluator stderr tail: {r.stderr[-800:]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fm-module", required=True); ap.add_argument("--name", required=True, help="foundation-model name used in runs/<name>__<mode>")
    ap.add_argument("--dataset", required=True, choices=tuple(BENCHMARK["datasets"]))
    ap.add_argument("--label-fraction", type=float, required=True, choices=tuple(RECIPE["label_regimes"]))
    ap.add_argument("--mode", required=True, choices=("probe", "full", "scratch"))
    ap.add_argument("--seed", type=int, default=RECIPE["seeds"]["default"])
    ap.add_argument("--max-hours", type=float, default=RECIPE["wall_clock_cap_hours_default"])
    ap.add_argument("--pretraining-ids", type=Path, default=None)
    ap.add_argument("--eval-modes", default="coarse,full", help="'coarse' alone keeps the test split untouched (trial rounds)")
    args = ap.parse_args(); args.eval_modes = tuple(args.eval_modes.split(","))
    main(args)
