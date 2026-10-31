"""Score predicted energies of one split against the reference reconstruction (benchmark.json definitions)."""
from __future__ import annotations

import numpy as np

from .resolution import FWHM_PER_SIGMA, fit_line, halfmax_fwhm, resolution_curve, residual_summary, robust_sigma
from .views import BENCHMARK

GATES = BENCHMARK["gates"]
TH228 = BENCHMARK["th228_lines_keV"]


def _initial_sigma(e_pred, e_ref, line, sigma_ref):
    """Starting width for the window iteration: reference width (+) residual width of the line events."""
    m = np.isfinite(e_pred) & (np.abs(e_ref - line) < 2 * sigma_ref)
    r = robust_sigma(e_pred[m] - e_ref[m]) if m.sum() >= 30 else sigma_ref
    return float(np.hypot(sigma_ref, r if np.isfinite(r) else 0.0))


def score(dataset, e_pred, table):
    spec = BENCHMARK["datasets"][dataset]
    e_pred = np.asarray(e_pred, dtype=np.float64); e_ref = np.asarray(table["energy_keV"], dtype=np.float64)
    n_bad = int((~np.isfinite(e_pred)).sum())
    out = {"dataset": dataset, "n_events": int(len(e_pred)), "n_nonfinite": n_bad, "lines": {}}
    ok = np.isfinite(e_pred) & np.isfinite(e_ref) & (e_ref > 0)
    out["relative_residual_sigma"] = robust_sigma((e_pred[ok] - e_ref[ok]) / e_ref[ok])
    if dataset == "supernemo":
        zero = table["source_key"] == "0nubb"
        res = residual_summary(e_pred[zero], e_ref[zero])
        floor = residual_summary(np.full(zero.sum(), np.median(e_ref[zero])), e_ref[zero])
        out["headline"] = {"name": "0nubb residual FWHM (keV)", "value": res["fwhm_keV"], "err": res["fwhm_err_keV"],
                           "bias_keV": res["bias_keV"], "relative_to_q_bb": res["fwhm_keV"] / spec["q_bb_keV"],
                           "trivial_floor_keV": floor["fwhm_keV"], "reportable": bool(n_bad == 0 and res["n"] > 0),
                           "n_events": res["n"]}
        out["secondary"] = {"predicted_0nubb_halfmax_fwhm_keV": halfmax_fwhm(e_pred[zero]),
                            "reference_0nubb_halfmax_fwhm_keV": halfmax_fwhm(e_ref[zero]),
                            "pearson_r_all": float(np.corrcoef(e_pred[ok], e_ref[ok])[0, 1]) if np.std(e_pred[ok]) > 0 else 0.0,
                            "by_source_residual_fwhm_keV": {s: residual_summary(e_pred[table["source_key"] == s], e_ref[table["source_key"] == s], n_boot=0)["fwhm_keV"]
                                                            for s in np.unique(table["source_key"])}}
        return out
    for ln in spec["lines"]:
        s0 = _initial_sigma(e_pred, e_ref, ln["keV"], ln["sigma_ref_keV"])
        r = fit_line(e_pred, ln["keV"], s0, spec["peak_model"], all_lines=TH228, e_ref=e_ref, sigma_ref=ln["sigma_ref_keV"],
                     min_window_keV=spec["min_window_keV"], window_sigma=spec["window_sigma"], gates=GATES)
        r["role"] = ln["role"]; out["lines"][ln["name"]] = r
    if dataset == "mjd":
        # Q_bb = 2039 keV lies between DEP 1592.5 and SEP 2103.5: the curve needs >= 4 reportable lines with at least one
        # above 1.5 MeV (DEP, 1620.5 or 2614.5); 2614.5 is optional because the spectrum ends just above it and a
        # regressor's predictions are often cut off there. SEP (Doppler-broadened) is reported directly, never fitted.
        pts = [(r["line_keV"], r["fwhm_keV"], r["fwhm_err_keV"]) for r in out["lines"].values() if r["role"] == "curve" and r["reportable"]]
        high = any(e >= 1500 for e, _, _ in pts)
        if len(pts) >= 4 and high:
            c = resolution_curve(pts, spec["q_bb_keV"])
            out["headline"] = {"name": "FWHM at Q_bb 2039 keV (keV)", "value": c["fwhm_keV"], "err": c["fwhm_err_keV"], "curve": c,
                               "reportable": True}
        else:
            out["headline"] = {"name": "FWHM at Q_bb 2039 keV (keV)", "value": None, "reportable": False,
                               "reasons": [f"{len(pts)} reportable curve lines (need >= 4)" + ("" if high else "; none above 1.5 MeV")]}
        out["secondary"] = {}
        for name, key in (("Tl208_DEP_1593", "fwhm_dep_1593"), ("Tl208_SEP_2104", "fwhm_sep_2104"), ("Tl208_2615", "fwhm_2615")):
            r = out["lines"][name]
            out["secondary"][f"{key}_keV"] = r["fwhm_keV"] if r["reportable"] else None
            out["secondary"][f"{key}_err_keV"] = r["fwhm_err_keV"] if r["reportable"] else None
    else:
        head = [r for r in out["lines"].values() if r["role"] == "headline"][0]
        out["headline"] = {"name": "sigma/E at 2614.5 keV", "value": head["fwhm_rel"] / FWHM_PER_SIGMA if head["reportable"] else None,
                           "err": head["fwhm_err_keV"] / FWHM_PER_SIGMA / head["line_keV"] if head["reportable"] else None,
                           "reportable": head["reportable"], "reasons": head["reasons"]}
    return out
