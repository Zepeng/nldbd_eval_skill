"""Tiny reference models, one per modality, used only to exercise the validation harness.

They implement the ``make_predictions`` contract (``load_model``, ``score``) and are trained by
``train_tiny.py``. They are deliberately small; their numbers are not a baseline.
Weights are read from ``$TINY_WEIGHTS_DIR/<dataset>.pt`` (train_tiny.py sets it to the run's weights/ folder).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL_INFO = {"family": "tiny reference models for harness tests", "file": __file__}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WaveformCNN(nn.Module):  # MJD: [B, 1, 3800]
    def __init__(self, width=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, width, 15, stride=4, padding=7), nn.GELU(),
            nn.Conv1d(width, 2 * width, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(2 * width, 4 * width, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(4 * width, 4 * width, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(4 * width, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class SensorCNN(nn.Module):  # EXO-200: [B, 226, 300] -> treat as one image
    def __init__(self, width=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, width, (5, 7), stride=(2, 3), padding=(2, 3)), nn.GELU(),
            nn.Conv2d(width, 2 * width, (5, 5), stride=(2, 2), padding=2), nn.GELU(),
            nn.Conv2d(2 * width, 4 * width, (3, 5), stride=(2, 2), padding=(1, 2)), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * width, 1))

    def forward(self, x):
        return self.net(x[:, None]).squeeze(1)


class DeepSets(nn.Module):  # NEXT / SuperNEMO: padded [B, N, F] + mask [B, N]
    def __init__(self, n_features, width=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(n_features, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, x, mask):
        h = self.phi(x) * mask[..., None]
        n = mask.sum(1, keepdim=True).clamp(min=1)
        mean = h.sum(1) / n
        mx = (h + (mask[..., None] - 1) * 1e4).max(1).values
        return self.rho(torch.cat([mean, mx], 1)).squeeze(1)


def build(dataset: str) -> nn.Module:
    return {"mjd": lambda: WaveformCNN(), "exo200": lambda: SensorCNN(),
            "next": lambda: DeepSets(4), "supernemo": lambda: DeepSets(5)}[dataset]()


def hits_to_padded(dataset: str, batch):
    """Per-hit features from the reader's raw hit arrays. Centering removes absolute position;
    NEXT energies become fractions of the event total; SuperNEMO tR NaN -> 0 plus a validity flag."""
    feats = []
    for h in batch:
        h = np.asarray(h, dtype=np.float32)
        xyz = (h[:, :3] - h[:, :3].mean(0)) / 1000.0
        if dataset == "next":
            e = h[:, 3:4] / max(float(h[:, 3].sum()), 1e-12)
            feats.append(np.concatenate([xyz, e], 1))
        else:
            r = h[:, 3:4]
            valid = np.isfinite(r).astype(np.float32)
            feats.append(np.concatenate([xyz, np.nan_to_num(r / 24.0), valid], 1))
    n = max(len(f) for f in feats)
    x = np.zeros((len(feats), n, feats[0].shape[1]), np.float32)
    m = np.zeros((len(feats), n), np.float32)
    for i, f in enumerate(feats):
        x[i, :len(f)] = f; m[i, :len(f)] = 1
    return torch.from_numpy(x), torch.from_numpy(m)


def to_tensors(dataset: str, inputs):
    if dataset in ("mjd", "exo200"):
        return (torch.from_numpy(np.ascontiguousarray(inputs)),)
    return hits_to_padded(dataset, inputs)


def load_model(dataset: str):
    weights = Path(os.environ.get("TINY_WEIGHTS_DIR", Path(__file__).resolve().parent / "weights")) / f"{dataset}.pt"
    model = build(dataset)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(DEVICE).eval()


@torch.no_grad()
def score(model, inputs):
    tensors = [t.to(DEVICE) for t in to_tensors(model.dataset if hasattr(model, "dataset") else _infer(model), inputs)]
    return model(*tensors).float().cpu().numpy()


def _infer(model):
    return {"WaveformCNN": "mjd", "SensorCNN": "exo200"}.get(type(model).__name__) or \
        ("next" if model.phi[0].in_features == 4 else "supernemo")
