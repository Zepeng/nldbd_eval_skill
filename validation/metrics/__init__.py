"""Metric registry. A metric is a function ``compute(ctx) -> {"value", "status", "detail"}``.

``ctx`` is a ``MetricContext`` with the event-aligned arrays (``label`` 0/1 with 1 = positive,
``score`` higher = positive, ``energy_keV``, ``group``, ``weight``) and ``ctx.energybench``, the
frozen EnergyBench record for the declared profile (computed once, shared by all metrics).
``status`` is "ok" when the value is reportable, otherwise a reason and ``value`` is None.

To add a metric: write ``compute`` here (or in a module imported here), register it with
``@metric("name", description=...)``, list it in benchmark.json (primary_metric, secondary_metrics,
and per dataset), bump the benchmark version, document it in validation_skill.md (metric card).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..energybench.metrics import evaluate, load_config

REGISTRY: dict[str, dict] = {}


def metric(name: str, description: str):
    def register(fn: Callable):
        REGISTRY[name] = {"name": name, "description": description, "compute": fn}
        return fn
    return register


@dataclass
class MetricContext:
    label: np.ndarray
    score: np.ndarray
    energy_keV: np.ndarray
    group: np.ndarray
    weight: np.ndarray | None = None
    profile: str = "strict600"
    _record: dict | None = field(default=None, repr=False)

    @property
    def config(self):
        return load_config(self.profile)

    @property
    def energybench(self) -> dict:
        if self._record is None:
            self._record = evaluate(self.label, self.score, self.energy_keV, group=self.group, weight=self.weight,
                                    energy_unit="keV", config=self.config)
        return self._record


@metric("matched_auc", "Energy-matched AUC (EnergyBench strict600): pooled weighted pairwise AUC after reweighting both classes "
                       "to a common energy spectrum in 5 keV bins; reportable only when the matching gates pass.")
def matched_auc(ctx: MetricContext) -> dict:
    m = ctx.energybench["matching"]
    return {"value": m["matched_auc"], "status": m["status"],
            "detail": {"valid_bins": m["valid_bin_count"], "common_support_keV": m["common_support_keV"],
                       "coverage": [m["classes"][c]["matched_fraction_original_finite"] for c in ("0", "1")],
                       "matched_ess": [m["classes"][c]["matched_ess"] for c in ("0", "1")]}}


@metric("inclusive_auc", "Plain AUC on all finite scores, independent of energy (diagnostic: shows how much energy "
                         "alone separates the classes when compared with matched_auc).")
def inclusive_auc(ctx: MetricContext) -> dict:
    v = ctx.energybench["inclusive_auc"]
    return {"value": v, "status": "ok" if v is not None else "not_estimable", "detail": {}}


def compute_all(names, ctx: MetricContext) -> dict:
    out = {}
    for name in names:
        if name not in REGISTRY:
            raise KeyError(f"metric {name!r} is not registered; known: {sorted(REGISTRY)}")
        out[name] = REGISTRY[name]["compute"](ctx)
    return out
