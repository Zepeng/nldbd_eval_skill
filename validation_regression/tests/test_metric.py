"""Metric self-tests on the frozen test tables (reference energies only, no model):
1. the reference energy reproduces the reference resolution (local gain exactly 1);
2. Gaussian smearing of known width is recovered: FWHM_obs^2 ~ FWHM_ref^2 + (2.3548 s)^2 (within 5 % + 3 err);
3. a compressed energy scale around the line (E' = line + k (E - line), k = 0.7) is caught by the local-gain gate;
4. shuffled predictions are not reportable;
5. SuperNEMO: the reference gives residual FWHM 0; a constant prediction equals the trivial floor.
Run: python -m validation_regression.tests.test_metric [mjd exo200 supernemo]"""
import sys

import numpy as np

from validation_regression.resolution import FWHM_PER_SIGMA
from validation_regression.score import score
from validation_regression.views import BENCHMARK, get_view


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    return bool(cond)


def run(ds):
    t = get_view(ds).table("test"); e = t["energy_keV"]; rng = np.random.default_rng(0); ok = True
    ref = score(ds, e, t)
    if ds == "supernemo":
        h = ref["headline"]
        ok &= check(h["value"] == 0.0 and h["reportable"], f"{ds} reference residual FWHM 0 ({h['value']})")
        z = t["source_key"] == "0nubb"; c = np.full(len(e), np.median(e[z]))
        r = score(ds, c, t)["headline"]
        ok &= check(abs(r["value"] - h["trivial_floor_keV"]) < 1e-6, f"{ds} constant prediction = floor {r['value']:.1f} keV")
        s = 50.0; r = score(ds, e + rng.normal(0, s, len(e)), t)["headline"]
        ok &= check(abs(r["value"] / (FWHM_PER_SIGMA * s) - 1) < 0.03, f"{ds} smearing {s} keV recovered: {r['value']:.1f} vs {FWHM_PER_SIGMA * s:.1f}")
        return ok
    lines = BENCHMARK["datasets"][ds]["lines"]
    ok &= check(ref["headline"]["reportable"], f"{ds} reference headline reportable: {ref['headline']['value']}")
    ok &= check(all(abs(l["local_gain"] - 1) < 1e-9 for l in ref["lines"].values()), f"{ds} reference local gain = 1")
    for frac in (0.5, 2.0):
        for ln in lines:
            if ln["role"] == "secondary":
                continue
            r0 = ref["lines"][ln["name"]]; s = frac * r0["fwhm_keV"] / FWHM_PER_SIGMA
            r = score(ds, e + rng.normal(0, s, len(e)), t)["lines"][ln["name"]]
            expect = np.hypot(r0["fwhm_keV"], FWHM_PER_SIGMA * s)
            good = r["reportable"] and abs(r["fwhm_keV"] - expect) < 0.05 * expect + 3 * r["fwhm_err_keV"]
            ok &= check(good, f"{ds} {ln['name']} smear x{frac}: {r.get('fwhm_keV', float('nan')):.3f} vs {expect:.3f} keV "
                              f"(gain {r['local_gain']:.3f}, {r['reasons']})")
    head = next(l for l in lines if l["name"] == "Tl208_2615")
    comp = e.copy(); near = np.abs(e - head["keV"]) < 30 * head["sigma_ref_keV"]
    comp[near] = head["keV"] + 0.7 * (e[near] - head["keV"])
    r = score(ds, comp, t)["lines"][head["name"]]
    ok &= check(not r["reportable"] and abs(r["local_gain"] - 0.7) < 0.03, f"{ds} compressed scale caught (gain {r['local_gain']:.3f}, {r['reasons']})")
    r = score(ds, rng.permutation(e), t)
    ok &= check(not r["headline"]["reportable"], f"{ds} shuffled predictions not reportable")
    return ok


if __name__ == "__main__":
    res = [run(ds) for ds in (sys.argv[1:] or ("mjd", "exo200", "supernemo"))]
    print("ALL PASS" if all(res) else "SOME FAILED"); sys.exit(0 if all(res) else 1)
