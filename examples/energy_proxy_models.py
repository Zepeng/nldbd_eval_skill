"""Energy-proxy "models": no training, they score an event by a statistic known to track energy.

Purpose: show what the benchmark does with an energy shortcut. Inclusive AUC should be far from
0.5 while the energy-matched AUC should sit near 0.5 (the benchmark metric removes it).
MJD: minus the RMS of the first 200 normalised samples (baseline noise shrinks with energy after
max-abs normalisation; the paper reports Spearman -0.83 between noise and energy).
EXO-200: minus the mean absolute amplitude of the normalised event (same idea).
NEXT: number of hits. SuperNEMO: number of tracker hits.
"""
from __future__ import annotations

import numpy as np

MODEL_INFO = {"family": "energy-proxy statistics, no training"}


def load_model(dataset: str):
    return dataset




def score(model, inputs):
    dataset = model
    if dataset == "mjd":
        x = np.asarray(inputs)[:, 0, :200]
        return -np.sqrt((x ** 2).mean(axis=1))
    if dataset == "exo200":
        x = np.asarray(inputs)
        return -np.abs(x).mean(axis=(1, 2))
    return np.array([len(h) for h in inputs], dtype=np.float64)
