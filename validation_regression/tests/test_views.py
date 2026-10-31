"""The regression view leaves the classification inputs untouched and adds only the scale / detector:
1. inputs are bit-identical to the classification reader's for the same events (MJD, EXO-200; SuperNEMO 2nubb/Bi214);
2. MJD/EXO-200: exp(log_scale) * input reproduces the baseline-subtracted raw waveform (lossless);
3. MJD: the detector index maps back to the stored detector id;
4. SuperNEMO regression test counts match the manifest; 0nubb events never appear in the classification tables.
Run: python -m validation_regression.tests.test_views"""
import sys

import h5py
import numpy as np

from validation.readers import get_reader
from validation_regression.views import BENCHMARK, get_view


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True); return bool(cond)


def main():
    ok = True
    for ds in ("mjd", "exo200"):
        cls, reg = get_reader(ds), get_view(ds); t = reg.table("test"); idx = np.arange(0, len(t["event_id"]), 997)[:40]
        p1, x1 = next(cls.iter_inputs("test", batch_size=64, indices=idx))
        p2, x2, aux = next(reg.iter_regression("test", batch_size=64, indices=idx))
        ok &= check(np.array_equal(p1, p2) and np.array_equal(x1, x2), f"{ds} inputs identical to the classification view")
        pos = int(p2[0])
        if ds == "mjd":
            with h5py.File(cls.test_files()[t["file_index"][pos]], "r") as f:
                raw = np.asarray(f["raw_waveform"][t["row"][pos]], dtype=np.float32); det_id = int(f["detector"][t["row"][pos]])
            raw = raw - raw[:200].mean()
            ok &= check(np.allclose(x2[0, 0] * np.exp(aux["log_scale"][0]), raw, rtol=1e-4, atol=1e-2), "mjd scale is lossless")
            ok &= check(BENCHMARK["datasets"]["mjd"]["detectors"][aux["detector"][0]] == det_id, "mjd detector index maps to the stored id")
        else:
            with h5py.File(cls.files[t["file_index"][pos]], "r") as f:
                raw = np.asarray(f["Waveforms"][str(t["row"][pos])][()], dtype=np.float32)
            raw = raw - raw[:, :200].mean(axis=1, keepdims=True)
            ok &= check(np.allclose(x2[0] * np.exp(aux["log_scale"][0]), raw, rtol=1e-4, atol=1e-2), "exo200 scale is lossless")
    cls, reg = get_reader("supernemo"), get_view("supernemo")
    ct, rt = cls.table("test"), reg.table("test")
    ok &= check(not np.isin(rt["event_id"][rt["source_key"] == "0nubb"], ct["event_id"]).any(), "supernemo 0nubb absent from classification test")
    for split in ("train", "validation"):
        c = cls.table(split)
        ok &= check(not any(i.startswith("SuperNEMO::0nubb::") for i in c["event_id"][:: max(1, len(c["event_id"]) // 100000)]),
                    f"supernemo 0nubb absent from classification {split}")
    shared = np.flatnonzero(np.isin(rt["event_id"], ct["event_id"]))[:30]
    p2, x2, _ = next(reg.iter_regression("test", indices=shared, table=rt))
    lookup = {e: i for i, e in enumerate(ct["event_id"])}
    p1, x1 = next(cls.iter_inputs("test", indices=np.array([lookup[e] for e in rt["event_id"][p2]]), table=ct))
    m1 = {ct["event_id"][p]: x for p, x in zip(p1, x1)}
    ok &= check(all(np.array_equal(m1[rt["event_id"][p]], x, equal_nan=True) for p, x in zip(p2, x2)), "supernemo hits identical to the classification view")
    return ok


if __name__ == "__main__":
    r = main(); print("ALL PASS" if r else "SOME FAILED"); sys.exit(0 if r else 1)
