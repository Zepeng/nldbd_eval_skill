"""The paper's Region + MLP Transformer as an encoder (foundation-model contract, validation/fm_contract.md) for the
energy-regression baseline: the exact paper backbone of region_mlp.py (tokenizer, content and coordinate-MLP
embeddings, 2 pre-norm layers, masked mean pooling, output LayerNorm) with the paper's scalar head removed, so the
frozen regression recipe (validation_regression/recipe.json) attaches its own head exactly as for a foundation model.
Used with --mode scratch only (random initialisation; there are no pretrained weights).
SuperNEMO events are tokenized once and memoised by content hash (the paper's tokenizer is CPU-bound)."""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import region_mlp  # noqa: E402

EMBED_DIM = region_mlp.ARCH["d_model"]
MODEL_INFO = {"paper_model": "Region + MLP Transformer backbone (paper head removed)", **region_mlp.MODEL_INFO}
HEADS = ("task_head", "classification_head", "scalar_head")
_memo: dict = {}


def load_encoder(dataset, pretrained=True):
    net = region_mlp.build(dataset)
    inner = net.net if isinstance(net, region_mlp._Dict) else net
    for name in HEADS:
        if hasattr(inner, name):
            setattr(inner, name, torch.nn.Identity())     # forward now returns output_norm(pooled): [B, 64]
            break
    else:
        raise RuntimeError(f"{dataset}: no paper head found on {type(inner).__name__}")
    return net


def _tok(dataset, item):
    a = np.ascontiguousarray(item, dtype=np.float32)
    key = (dataset, a.shape, hashlib.blake2b(a.tobytes(), digest_size=16).digest())
    t = _memo.get(key)
    if t is None:
        t = _memo[key] = region_mlp.tokenize(dataset, a)
    return t


def to_tensors(dataset, inputs):
    if dataset in region_mlp.TOKENIZED:
        return region_mlp.collate([_tok(dataset, x) for x in inputs])
    return region_mlp.to_tensors(dataset, inputs)
