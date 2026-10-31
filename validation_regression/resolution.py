"""Physics resolution of a predicted energy spectrum.

Peak lines (MJD, EXO-200): binned Poisson-likelihood fit of the predicted-energy spectrum around each line,
iterating the window to the fitted width. Signal shapes: ``gauss``; ``gauss_step`` (Gaussian + erfc step, EXO-200);
``gauss_tail_step`` (Gaussian + low-energy exponential tail + erfc step, the standard HPGe shape; MJD and EXO-200). Background is linear;
known Th-228 lines inside the window are fitted as extra components with the same shape (free amplitudes).

The fitted FWHM is the half-maximum width of the full signal shape. It is divided by the LOCAL GAIN, the slope of
E_pred against E_ref measured on events whose reference energy lies in the sidebands line +-[4 s, 4 s + w]
(s = max(sigma_ref, sigma_pred), w = 8-12 s, widened until each band holds 200 events; two low bands when the spectrum
ends above the line): a regressor that narrows a peak by compressing the energy scale around it (or
shrinking nearby continuum onto the line) is charged for it. Reportability gates are in benchmark.json.

SuperNEMO (no monochromatic line): FWHM of the residual E_pred - E_ref over the 0nubb events, with the trivial floor
of a constant prediction, and the half-maximum FWHM of the predicted spectrum as a diagnostic.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, special

SQRT2 = math.sqrt(2.0)
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
MAD_TO_SIGMA = 1.482602218505602


def robust_sigma(x):
    x = np.asarray(x, dtype=np.float64)
    return float(MAD_TO_SIGMA * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


# ---------------------------------------------------------------------------------------------- signal shapes
def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))


def _low_tail(x, mu, sig, tau):
    """Gaussian-convolved exponential on the low-energy side, unit area."""
    z = (x - mu) / (SQRT2 * sig) + sig / (SQRT2 * tau)
    arg = (x - mu) / tau + sig ** 2 / (2 * tau ** 2)
    return np.exp(np.minimum(arg, 700.0)) * special.erfc(z) / (2 * tau)


def _step(x, mu, sig):
    return 0.5 * special.erfc((x - mu) / (SQRT2 * sig))


class PeakModel:
    """Parameter vector: N, mu, sig, [f_tail, tau, f_high, tau_high], [h_step], b0, b1, A_1..A_k (neighbour areas)."""

    def __init__(self, kind, line, neighbours, x_center):
        self.kind, self.line, self.offsets, self.xc = kind, line, np.asarray(neighbours, dtype=np.float64) - line, x_center
        self.tail = kind == "gauss_tail_step"
        self.has_step = kind in ("gauss_tail_step", "gauss_step")
        if kind not in ("gauss", "gauss_step", "gauss_tail_step"):
            raise ValueError(f"unknown peak model {kind!r}")

    def names(self):
        n = ["N", "mu", "sig"] + (["f_tail", "tau", "f_high", "tau_high"] if self.tail else []) + (["h_step"] if self.has_step else [])
        return n + ["b0", "b1"] + [f"A{k}" for k in range(len(self.offsets))]

    def unpack(self, p):
        return dict(zip(self.names(), p))

    def shape(self, x, q, mu=None):
        mu = q["mu"] if mu is None else mu
        g = _gauss(x, mu, q["sig"])
        if self.tail:
            g = (1 - q["f_tail"] - q["f_high"]) * g + q["f_tail"] * _low_tail(x, mu, q["sig"], q["tau"]) \
                + q["f_high"] * _low_tail(2 * mu - x, mu, q["sig"], q["tau_high"])
        return g

    def density(self, x, p):
        q = self.unpack(p)
        d = q["N"] * self.shape(x, q) + q["b0"] + q["b1"] * (x - self.xc)
        if self.has_step:
            d = d + q["h_step"] * _step(x, q["mu"], q["sig"])
        for k, off in enumerate(self.offsets):
            d = d + q[f"A{k}"] * self.shape(x, q, q["mu"] + off)
        return d

    def fwhm(self, p):
        q = self.unpack(p)
        if not self.tail:
            return FWHM_PER_SIGMA * q["sig"]
        s = q["sig"]; x = np.linspace(q["mu"] - 6 * s - 6 * q["tau"], q["mu"] + 6 * s + 6 * q["tau_high"], 6001); y = self.shape(x, q)
        i = int(np.argmax(y)); half = y[i] / 2
        lo = np.flatnonzero(y[:i] < half); hi = np.flatnonzero(y[i:] < half)
        if not len(lo) or not len(hi):
            return float("nan")
        a, b = lo[-1], i + hi[0]
        xl = np.interp(half, [y[a], y[a + 1]], [x[a], x[a + 1]]); xr = np.interp(half, [y[b], y[b - 1]], [x[b], x[b - 1]])
        return float(xr - xl)


def _binned_fit(model, edges, counts, p0, lo, hi, sub=5):
    w = np.diff(edges)
    xs = (edges[:-1, None] + w[:, None] * (np.arange(sub) + 0.5)[None, :] / sub)
    def expected(p):
        return np.maximum(model.density(xs, p).mean(axis=1) * w, 1e-12)
    def resid(p):
        nu = expected(p); n = counts
        dev = 2 * (nu - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / nu), 0.0))
        return np.sign(n - nu) * np.sqrt(np.maximum(dev, 0.0))
    r = optimize.least_squares(resid, p0, bounds=(lo, hi), x_scale="jac", method="trf", max_nfev=4000)
    return r, float((r.fun ** 2).sum())


def fit_line(e_pred, line, sigma_init, kind, all_lines=(), e_ref=None, sigma_ref=None, min_window_keV=0.0,
             window_sigma=8.0, gates=None, max_iter=6):
    """Fit one line in the predicted spectrum; returns a dict with FWHM (keV, gain-corrected), centroid, signal,
    chi2/ndf, local gain, reportable flag and reasons."""
    e = np.asarray(e_pred, dtype=np.float64); e = e[np.isfinite(e)]
    mu, sig = float(line), float(sigma_init)
    out = {"line_keV": float(line), "reportable": False, "reasons": []}
    for it in range(max_iter):
        W = min(max(window_sigma * sig, min_window_keV), 0.25 * line)
        a, b = mu - W, mu + W
        sel = e[(e >= a) & (e < b)]
        nb = int(np.clip(round(2 * W / (sig / 4)), 40, 400))
        edges = np.linspace(a, b, nb + 1); counts = np.histogram(sel, edges)[0].astype(np.float64)
        neigh = [l for l in all_lines if abs(l - line) > 1e-6 and a - 2 * sig < l < b + 2 * sig]
        model = PeakModel(kind, line, neigh, mu)
        side = np.r_[counts[: nb // 10], counts[-(nb // 10):]].mean() / (2 * W / nb)
        n_sig = max(counts.sum() - side * 2 * W, 10.0)
        p0 = [n_sig, mu, sig]; lo = [0.0, a + 0.1 * W, sig / 20]; hi = [np.inf, b - 0.1 * W, W / 2]
        if model.tail:
            p0 += [0.1, sig, 0.02, sig]; lo += [0.0, sig / 20, 0.0, sig / 20]; hi += [0.45, 10 * sig + 1e-3, 0.45, 10 * sig + 1e-3]
        if model.has_step:
            p0 += [max(side * 0.05, 1e-3)]; lo += [0.0]; hi += [np.inf]
        p0 += [max(side, 1e-3), 0.0]; lo += [-np.inf, -np.inf]; hi += [np.inf, np.inf]
        p0 += [n_sig * 0.1] * len(neigh); lo += [0.0] * len(neigh); hi += [np.inf] * len(neigh)
        lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        p0 = np.clip(np.asarray(p0, dtype=np.float64), np.where(np.isfinite(lo), lo, -1e300) + 1e-9 * (1 + np.abs(np.where(np.isfinite(lo), lo, 0))), hi)
        try:
            r, dev = _binned_fit(model, edges, counts, p0, lo, hi)
        except Exception as exc:  # noqa: BLE001 - a failed fit is "not reportable", never a crash
            out["reasons"].append(f"fit error: {exc}"); out["local_gain"] = float("nan"); return out
        q = model.unpack(r.x)
        fwhm = model.fwhm(r.x); new_sig = fwhm / FWHM_PER_SIGMA
        done = abs(new_sig - sig) < 0.05 * sig and abs(q["mu"] - mu) < 0.5 * sig
        mu, sig = float(q["mu"]), float(new_sig)
        if done:
            break
    ndf = max(1, nb - len(r.x))
    # uncertainty of the FWHM from the Fisher information of the deviance residuals
    try:
        J = r.jac; cov = np.linalg.pinv(J.T @ J)
        g = np.zeros(len(r.x))
        for k in range(len(r.x)):
            h = 1e-5 * max(abs(r.x[k]), 1e-3); p1 = r.x.copy(); p1[k] += h
            g[k] = (model.fwhm(p1) - fwhm) / h
        fwhm_err = float(math.sqrt(max(g @ cov @ g, 0.0)))
    except Exception:  # noqa: BLE001
        fwhm_err = float("nan")
    out.update({"centroid_keV": mu, "fwhm_raw_keV": float(fwhm), "fwhm_raw_err_keV": fwhm_err, "signal": float(q["N"]),
                "chi2_ndf": dev / ndf, "window_keV": [float(a), float(b)], "n_bins": nb, "iterations": it + 1,
                "converged": bool(r.success), "neighbours_keV": [float(x) for x in neigh],
                "params": {k: float(v) for k, v in q.items()}})
    # local gain from the sidebands (needs the reference energies)
    gain = 1.0
    if e_ref is not None:
        ep, er = np.asarray(e_pred, dtype=np.float64), np.asarray(e_ref, dtype=np.float64)
        s = max(sigma_ref or 0.0, sig)
        fin = np.isfinite(ep)
        def band(a_, b_):
            return (er > a_) & (er < b_) & fin
        width = 8 * s                                   # bands [4 s, 4 s + width]; widened until populated
        while True:
            low, high = band(line - 4 * s - width, line - 4 * s), band(line + 4 * s, line + 4 * s + width)
            far = band(line - 4 * s - 2 * width, line - 4 * s - width)
            pair = ("low", "high") if high.sum() >= 200 else ("far_low", "low")   # spectra cut just above the line
            A, B = {"low": low, "high": high, "far_low": far}[pair[0]], {"low": low, "high": high}[pair[1]]
            if (A.sum() >= 200 and B.sum() >= 200) or width * 1.5 > 12 * s:   # stay local: bands within 28 s
                break
            width *= 1.5
        out["sideband_width_keV"] = float(width)
        if A.sum() >= 50 and B.sum() >= 50:   # never below 50 events
            gain = float((np.median(ep[B]) - np.median(ep[A])) / (np.median(er[B]) - np.median(er[A])))
            out["sidebands"] = {pair[0]: int(A.sum()), pair[1]: int(B.sum())}
        else:
            out["reasons"].append("too few sideband events for the local gain"); gain = float("nan")
        line_ev = np.abs(er - line) < 2 * (sigma_ref or sig)
        out["residual_fwhm_keV"] = FWHM_PER_SIGMA * robust_sigma(ep[line_ev & np.isfinite(ep)] - er[line_ev & np.isfinite(ep)])
    out["local_gain"] = gain
    out["fwhm_keV"] = float(fwhm / gain) if np.isfinite(gain) and gain > 0 else float("nan")
    out["fwhm_err_keV"] = float(fwhm_err / gain) if np.isfinite(gain) and gain > 0 else float("nan")
    out["fwhm_rel"] = out["fwhm_keV"] / line
    gates = gates or {}
    if not r.success:
        out["reasons"].append("fit did not converge")
    if q["N"] < gates.get("min_signal_events", 300):
        out["reasons"].append(f"signal {q['N']:.0f} < {gates.get('min_signal_events', 300)}")
    if out["chi2_ndf"] >= gates.get("max_chi2_ndf", 3.0):
        out["reasons"].append(f"chi2/ndf {out['chi2_ndf']:.2f} >= {gates.get('max_chi2_ndf', 3.0)}")
    if abs(mu - line) >= gates.get("max_centroid_offset_fwhm", 3.0) * fwhm:
        out["reasons"].append(f"centroid {mu:.2f} more than {gates.get('max_centroid_offset_fwhm', 3.0)} FWHM from {line}")
    if (b - a) / 2 < gates.get("min_window_half_width_sigma", 4.0) * sig:
        out["reasons"].append("fit window narrower than the peak (window capped at 25 % of the line energy)")
    if np.isfinite(gain) and abs(gain - 1) > gates.get("max_local_gain_deviation", 0.2):
        out["reasons"].append(f"local gain {gain:.3f} deviates from 1 by more than {gates.get('max_local_gain_deviation', 0.1)}")
    out["reportable"] = not out["reasons"] and bool(np.isfinite(out["fwhm_keV"]))
    return out


def resolution_curve(points, at_keV):
    """Fit FWHM(E) = sqrt(a + b E + c E^2), a, b, c >= 0, through (E, FWHM, err) points; FWHM at ``at_keV``."""
    E = np.array([p[0] for p in points]); F = np.array([p[1] for p in points]); S = np.array([p[2] for p in points])
    S = np.where(np.isfinite(S) & (S > 0), S, np.nanmax(F) * 0.05)
    S = np.sqrt(S ** 2 + (0.01 * F) ** 2)          # 1 % floor: line-to-line systematic of the shape model
    f = lambda p, x: np.sqrt(np.maximum(p[0] + p[1] * x + p[2] * x ** 2, 1e-12))
    p0 = np.array([max(F.min() ** 2, 1e-3), 1e-3, 1e-7])
    r = optimize.least_squares(lambda p: (f(p, E) - F) / S, p0, bounds=([0, 0, 0], [np.inf] * 3), x_scale="jac")
    cov = np.linalg.pinv(r.jac.T @ r.jac) * max(1.0, (r.fun ** 2).sum() / max(1, len(E) - 3))
    val = float(f(r.x, at_keV)); g = np.array([1, at_keV, at_keV ** 2]) / (2 * val)
    return {"a": float(r.x[0]), "b": float(r.x[1]), "c": float(r.x[2]), "fwhm_keV": val,
            "fwhm_err_keV": float(math.sqrt(max(g @ cov @ g, 0.0))), "chi2_ndf": float((r.fun ** 2).sum() / max(1, len(E) - 3)),
            "lines_keV": E.tolist()}


def halfmax_fwhm(e, smooth_bins=3):
    """Half-maximum width of a (possibly asymmetric) spectrum: fine histogram, Gaussian smoothing, linear
    interpolation of the crossings."""
    e = np.asarray(e, dtype=np.float64); e = e[np.isfinite(e)]
    s = robust_sigma(e); med = np.median(e)
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    edges = np.arange(med - 8 * s, med + 8 * s + s / 20, s / 20); h = np.histogram(e, edges)[0].astype(np.float64)
    k = np.exp(-0.5 * (np.arange(-4 * smooth_bins, 4 * smooth_bins + 1) / smooth_bins) ** 2); h = np.convolve(h, k / k.sum(), "same")
    x = 0.5 * (edges[1:] + edges[:-1]); i = int(np.argmax(h)); half = h[i] / 2
    lo = np.flatnonzero(h[:i] < half); hi = np.flatnonzero(h[i:] < half)
    if not len(lo) or not len(hi):
        return float("nan")
    a, b = lo[-1], i + hi[0]
    return float(np.interp(half, [h[b], h[b - 1]], [x[b], x[b - 1]]) - np.interp(half, [h[a], h[a + 1]], [x[a], x[a + 1]]))


def residual_summary(e_pred, e_ref, n_boot=100, seed=0):
    d = np.asarray(e_pred, dtype=np.float64) - np.asarray(e_ref, dtype=np.float64); d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed); m = min(len(d), 50000)
    boots = [robust_sigma(rng.choice(d, m)) for _ in range(n_boot)] if len(d) else []
    s = robust_sigma(d)
    return {"n": int(len(d)), "fwhm_keV": FWHM_PER_SIGMA * s, "fwhm_err_keV": FWHM_PER_SIGMA * float(np.std(boots)) * math.sqrt(m / max(1, len(d))) if boots else float("nan"),
            "bias_keV": float(np.median(d)) if len(d) else float("nan")}
