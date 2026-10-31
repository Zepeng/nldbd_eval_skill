"""The paper's Region + MLP Transformer (EnergyBench Table 1 "Region + MLP"), built from the paper's own code (imported
read-only, never copied or modified) with the exact configurations of the paper runs
(provenance/transformer_model_mapping.csv of the reproducibility bundle):

* MJD       mjd_transformer.MJDTransformer, raw_patches (patch 20 -> 190 tokens of 3800 samples) + coordinate_mlp, 112,257 params
* EXO-200   exo_transformer.EXOTransformerClassifier, raw_patches (6 blocks x 7 channel x 12 time regions) + coordinate_mlp, 120,769 params
* NEXT      next_transformer.NEXTTransformerClassifier, voxel 15 mm (<= 512 tokens) + coordinate_mlp, 111,105 params
* SuperNEMO published_supernemo TrackerHitTransformer, voxel 60 mm (<= 128 tokens) + coordinate_mlp, 111,233 params
All: d_model 64, 4 heads, 2 pre-norm GELU layers, feed-forward 256, dropout 0.1, masked mean pooling, the paper's own head.
Inputs are exactly the evaluation readers' (identical preprocessing to the paper workflows for MJD / EXO-200; NEXT and
SuperNEMO hits are tokenized per event by the paper's tokenizers). Labels follow the harness (1 = signal-like), so the
logit is signal-like for every detector (the paper's EXO-200 logit is background-like; BCE is symmetric).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

PAPER = os.environ.get("PAPER_TRANSFORMER_APPROACH", "/home/klz/Data/zeronu_benchmark/Transformer_Approach")
SUPERNEMO_PUBLISHED = os.environ.get("PAPER_PUBLISHED_SUPERNEMO", "/home/wenyu/iclr final paper/EnergyBench_reproducibility_20260923/code/transformers/published_supernemo")
for p in (f"{PAPER}/mjd_detector", f"{PAPER}/exo200_detector", f"{PAPER}/next_detector", f"{PAPER}/evalutaions_workflow", SUPERNEMO_PUBLISHED):
    if p not in sys.path:
        sys.path.append(p)

import mjd_transformer  # noqa: E402
import exo_transformer  # noqa: E402
import next_transformer  # noqa: E402
from architectures.trf_001_tracker_transformer import TrackerHitTransformer  # noqa: E402
from supernemobench.tokenization import SuperNEMOTrackerTokenizationConfig, tokenize_tracker_event  # noqa: E402

NAME = "paper_region_mlp"
ARCH = dict(d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, num_frequencies=6)
# paper training differences per detector (run_config.json of each paper run)
TOKENIZED = ("next", "supernemo")                   # tokenized once per regime into a token bank (paper: NEXT token cache)
AMP = {"next": True, "supernemo": True}                 # bf16 autocast; MJD / EXO-200 trained in fp32
PATIENCE = {"supernemo": 12}                             # others: 5
SELECTION = {"supernemo": "matched_auc"}                 # others: plain validation AUC
NEXT_TOK = next_transformer.TokenizationConfig(tokenization="voxel", max_tokens=512, voxel_size=15.0, coordinate_scale=1000.0,
                                               center_coordinates=True, voxel_truncation="occupancy", seed=42)
SN_TOK = SuperNEMOTrackerTokenizationConfig(tokenization="voxel", max_tokens=128, coordinate_scale_mm=1000.0, center_coordinates=True,
                                            voxel_size_mm=60.0, summary_morton_bits=10, tracker_radius_scale_mm=24.0, seed=42)
_next_builder = next_transformer.NEXTTokenBuilder(NEXT_TOK)
MODEL_INFO = {"paper_model": "Region + MLP Transformer", "arch": ARCH, "next_tokenization": NEXT_TOK.to_dict() if hasattr(NEXT_TOK, "to_dict") else str(NEXT_TOK),
              "supernemo_tokenization": SN_TOK.to_dict(), "amp": AMP, "patience": PATIENCE, "selection": SELECTION}


class _Dict(torch.nn.Module):
    """forward(coords, features, mask) for the token models."""

    def __init__(self, net, as_mapping):
        super().__init__(); self.net = net; self.as_mapping = as_mapping

    def forward(self, coords, features, mask):
        if self.as_mapping:
            return self.net({"coords": coords, "features": features, "mask": mask})
        return self.net(coords, features, mask)


def build(dataset):
    torch.manual_seed(42)
    if dataset == "mjd":
        return mjd_transformer.MJDTransformer(task="classification", position_encoding="coordinate_mlp",
                                              tokenization_config=mjd_transformer.TokenizationConfig(
                                                  tokenization="raw_patches", token_count=500, patch_size=20,
                                                  uniform_entity_fraction=0.5, entity_context_size=9), **ARCH)
    if dataset == "exo200":
        return exo_transformer.EXOTransformerClassifier(position_encoding="coordinate_mlp",
                                                        tokenization_config=exo_transformer.TokenizationConfig(
                                                            tokenization="raw_patches", channel_regions=7, time_regions=12,
                                                            uniform_entity_fraction=0.5, entity_context_size=9), **ARCH)
    if dataset == "next":
        return _Dict(next_transformer.NEXTTransformerClassifier(position_encoding="coordinate_mlp", feature_dim=2, **ARCH), True)
    if dataset == "supernemo":
        return _Dict(TrackerHitTransformer(position_encoding="coordinate_mlp", feature_dim=4, pooling="masked_mean", **ARCH), False)
    raise ValueError(dataset)


def tokenize(dataset, item):
    """One event -> unpadded (coords [n, 3], features [n, f]) float32 (NEXT, SuperNEMO)."""
    a = np.asarray(item, dtype=np.float32)
    if dataset == "next":
        inputs, _ = _next_builder(a[:, :3], a[:, 3], event_id="harness")
        n = int(inputs["mask"].sum())
        return inputs["coords"][:n].astype(np.float32), inputs["features"][:n].astype(np.float32)
    inputs, _ = tokenize_tracker_event(a[:, :3], a[:, 3], sampling_key="harness", config=SN_TOK)
    return inputs["coords"].astype(np.float32), inputs["features"].astype(np.float32)


def collate(tokens):
    n = max(len(c) for c, _ in tokens); B = len(tokens); f = tokens[0][1].shape[1]
    C = np.zeros((B, n, 3), np.float32); F = np.zeros((B, n, f), np.float32); M = np.zeros((B, n), bool)
    for i, (c, ft) in enumerate(tokens):
        C[i, :len(c)] = c; F[i, :len(c)] = ft; M[i, :len(c)] = True
    return torch.from_numpy(C), torch.from_numpy(F), torch.from_numpy(M)


def to_tensors(dataset, inputs):
    if dataset in ("mjd", "exo200"):
        return (torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32)),)
    return collate([tokenize(dataset, x) for x in inputs])
