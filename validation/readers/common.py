from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANIFESTS = HERE.parent / "manifests"
BENCHMARK_FILE = HERE.parent / "benchmark.json"
BENCHMARK = json.loads(BENCHMARK_FILE.read_text())
DATASETS = tuple(BENCHMARK["datasets"])
SPLITS = ("train", "validation", "test")


def dataset_spec(dataset: str) -> dict:
    try:
        return BENCHMARK["datasets"][dataset]
    except KeyError:
        raise ValueError(f"unknown dataset {dataset!r}; registered in benchmark.json: {DATASETS}") from None


def data_root(dataset: str) -> Path:
    """Dataset folder: ``NLDBD_<DATASET>_DATA`` overrides ``NLDBD_DATA_ROOT/<folder from benchmark.json>``."""
    specific = os.environ.get(f"NLDBD_{dataset.upper()}_DATA")
    if specific:
        return Path(specific)
    root = Path(os.environ.get("NLDBD_DATA_ROOT", "/home/klz/Data/zeronu_benchmark"))
    return root / dataset_spec(dataset)["folder"]


def sha256_lines(values) -> str:
    """SHA-256 of ``"\\n".join(values) + "\\n"``; the identity hash used in every manifest."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode())
        digest.update(b"\n")
    return digest.hexdigest()


class Reader:
    name: str = ""
    signal_group: str = ""
    background_group: str = ""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else data_root(self.name)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.name}: data folder {self.root} does not exist")

    # -- contract -------------------------------------------------------------------------
    def table(self, split: str) -> dict:
        raise NotImplementedError

    def iter_inputs(self, split: str, batch_size: int = 256, indices=None):
        """Yield ``(positions, inputs)``: positions index ``table(split)``; inputs are
        preprocessed exactly as in the paper and contain no energy, label or category."""
        raise NotImplementedError

    # -- helpers --------------------------------------------------------------------------
    @staticmethod
    def check_table(t: dict) -> dict:
        n = len(t["event_id"])
        for key in ("event_id", "label", "energy_keV", "group"):
            if key not in t or len(t[key]) != n:
                raise ValueError(f"table field {key} missing or misaligned")
        ids = np.asarray(t["event_id"]).astype(str)
        if len(np.unique(ids)) != n:
            raise ValueError("event_id values are not unique")
        if not np.all(np.isin(t["label"], (0, 1))):
            raise ValueError("labels must be 0/1")
        t["event_id"] = ids
        t["label"] = np.asarray(t["label"], dtype=np.int8)
        t["energy_keV"] = np.asarray(t["energy_keV"], dtype=np.float64)
        t["group"] = np.asarray(t["group"]).astype(str)
        return t

    def summary(self, split: str, t: dict | None = None) -> dict:
        t = self.table(split) if t is None else t
        ids = t["event_id"]
        e = t["energy_keV"]
        finite = np.isfinite(e)
        return {
            "n": int(len(ids)),
            "label_counts": {"0": int((t["label"] == 0).sum()), "1": int((t["label"] == 1).sum())},
            "group_counts": {g: int(c) for g, c in zip(*np.unique(t["group"], return_counts=True))},
            "ids_sha256_sorted": sha256_lines(np.sort(ids)),
            "ids_sha256_canonical_order": sha256_lines(ids),
            "labels_sha256_canonical_order": sha256_lines(t["label"].tolist()),
            "energy_keV": {
                "min": float(np.nanmin(e)) if finite.any() else None,
                "max": float(np.nanmax(e)) if finite.any() else None,
                "n_nonfinite": int((~finite).sum()),
                "n_below_0": int((e[finite] < 0).sum()),
                "n_above_3000": int((e[finite] > 3000).sum()),
                "sha256_repr": sha256_lines(np.array2string(x, precision=17) if False else repr(float(x)) for x in e),
            },
        }


def get_reader(dataset: str, root: Path | None = None) -> Reader:
    """Instantiate the reader registered for ``dataset`` in benchmark.json (``module:Class``)."""
    import importlib
    module_name, class_name = dataset_spec(dataset)["reader"].split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(root)


def load_json(path: Path):
    return json.loads(Path(path).read_text())
