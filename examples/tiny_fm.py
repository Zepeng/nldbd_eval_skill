"""Reference foundation model for the fine-tuning protocol (validation/fm_contract.md).

Encoders are the trunks of the tiny reference networks; "pretrained" weights are the trunks of the
tiny classifiers trained by train_tiny.py (runs/tiny/<ds>/L0.01/s42/weights/<ds>.pt), i.e. a
supervised-pretrained encoder, which is enough to exercise probe / full / scratch end to end.
A real foundation model replaces load_encoder with its own pretrained weights.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_models  # noqa: E402

EMBED_DIM = {"mjd": 64, "exo200": 64, "next": 128, "supernemo": 128}
MODEL_INFO = {"family": "tiny trunks (1-D CNN, 2-D CNN, DeepSets phi+pool) used as encoders; pretrained = trunk of the tiny classifier"}
WEIGHTS = Path(os.environ.get("TINY_FM_WEIGHTS", Path(__file__).resolve().parents[1] / "runs" / "tiny"))


class TrunkCNN(nn.Module):  # everything of WaveformCNN / SensorCNN before the final Linear -> [B, 64]
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, x):
        return self.net(x)


class TrunkSets(nn.Module):  # DeepSets phi + masked mean/max pooling -> [B, 128]
    def __init__(self, phi):
        super().__init__(); self.phi = phi

    def forward(self, x, mask):
        h = self.phi(x) * mask[..., None]; n = mask.sum(1, keepdim=True).clamp(min=1)
        return torch.cat([h.sum(1) / n, (h + (mask[..., None] - 1) * 1e4).max(1).values], 1)


def to_tensors(dataset, inputs):
    return tiny_models.to_tensors(dataset, inputs)


def load_encoder(dataset, pretrained=True):
    full = tiny_models.build(dataset)
    if pretrained:
        w = WEIGHTS / dataset / "L0.01" / "s42" / "weights" / f"{dataset}.pt"
        full.load_state_dict(torch.load(w, map_location="cpu"))
    if dataset in ("mjd", "exo200"):
        net = full.net
        if dataset == "exo200":
            class Wrapped(nn.Module):
                def __init__(self, inner):
                    super().__init__(); self.inner = inner
                def forward(self, x):
                    return self.inner(x[:, None])
            return TrunkCNN(Wrapped(net[:-1]))
        return TrunkCNN(net[:-1])
    return TrunkSets(full.phi)
