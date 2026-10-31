"""EnergyBench paper metrics; NumPy only, no model inference.

All masks are aligned to the original supplied test population. In particular
the matched reporting denominator precedes both lower and upper range cuts.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

PROFILES = ('strict600', 'overflow601')

def load_config(profile='strict600'):
    if profile not in PROFILES:
        raise ValueError(f'Unknown publication profile: {profile}')
    return json.loads((Path(__file__).resolve().parent / 'configs' / (profile + '.json')).read_text())

def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def to_keV(energy, unit):
    if unit not in ('keV', 'MeV'):
        raise ValueError('Energy unit must be explicitly keV or MeV')
    original = np.asarray(energy, dtype=np.float64)
    if unit == 'keV':
        return original
    result = original * 1000.
    # Recognize only the exact floating-point representation of a canonical
    # MeV edge, not arbitrary nearby observations. Multiplication alone maps
    # e.g. 1.005 MeV just below 1005 keV because of binary rounding.
    finite = np.isfinite(original)
    candidate = np.zeros_like(result)
    candidate[finite] = np.rint(result[finite]/5.)*5.
    canonical = finite & (candidate >= 0) & (candidate <= 3000) & (original == candidate/1000.)
    return np.where(canonical,candidate,result)

def energy_bin_indices(energy_keV, include_overflow=False):
    e = np.asarray(energy_keV, dtype=float)
    good = np.isfinite(e) & (e >= 0.)
    if not include_overflow:
        good &= e <= 3000.
    indices = np.full(e.shape, -1, dtype=np.int64)
    indices[good] = np.searchsorted(np.arange(601, dtype=float)*5., e[good], side='right') - 1
    indices[good & (e == 3000.)] = 599
    return indices

def weighted_quantile(x, q, w):
    order = np.argsort(x, kind='stable')
    x, w = np.asarray(x)[order], np.asarray(w)[order]
    return np.interp(q, (np.cumsum(w)-.5*w)/w.sum(), x, left=x[0], right=x[-1])

def weighted_auc(label, score, weight=None):
    """P(s_positive>s_negative)+0.5 P(tie), retaining cross-bin pairs."""
    y, s = np.asarray(label), np.asarray(score, dtype=float)
    w = np.ones(s.size) if weight is None else np.asarray(weight, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or w.ndim != 1 or not (y.size == s.size == w.size):
        raise ValueError('unaligned one-dimensional ROC inputs')
    if not np.all(np.isin(y, (0,1))) or np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError('invalid labels or weights')
    m = np.isfinite(s) & (w > 0)
    y, s, w = y[m], s[m], w[m]
    if not np.any(y == 0) or not np.any(y == 1):
        return None
    order = np.argsort(s, kind='stable')
    y, s, w = y[order], s[order], w[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1])+1]
    pos = np.add.reduceat(w*(y == 1), starts)
    neg = np.add.reduceat(w*(y == 0), starts)
    below = np.cumsum(neg)-neg
    return float(np.clip(np.sum(pos*(below+.5*neg))/(pos.sum()*neg.sum()),0.,1.))

def ess(w):
    return float(w.sum()**2 / np.dot(w,w)) if np.any(w > 0) else 0.

def fraction(a, b):
    return float(a/b) if b > 0 else None

def score_edges(s, w, n):
    edges = np.unique(weighted_quantile(s, np.linspace(0,1,n+1), w))
    if edges.size < 2:
        span = max(abs(float(s[0]))*1e-9,1e-12)
        return np.array([s[0]-span,s[0]+span])
    edges[0] = np.nextafter(edges[0],-np.inf)
    edges[-1] = np.nextafter(edges[-1],np.inf)
    return edges

def histogram(s, w, edges):
    h = np.histogram(s, bins=edges, weights=w)[0].astype(float)
    return h / h.sum()

def js_nats(p,q):
    p, q = p/p.sum(), q/q.sum()
    m = .5*(p+q)
    a, b = p>0, q>0
    return max(0., float(.5*np.sum(p[a]*np.log(p[a]/m[a])) + .5*np.sum(q[b]*np.log(q[b]/m[b]))))

def evaluate(label, score, energy, group=None, weight=None, energy_unit='keV', config=None, return_arrays=False):
    config = load_config() if config is None else config
    # Grid is deliberately immutable: alternate grids require a new implementation/version.
    assert config['energy']['edges_keV'] == list(range(0,3001,5))
    include_overflow = 'overflow_bin' in config['energy']
    n_bins = 601 if include_overflow else 600
    assert config['energy']['n_bins'] == n_bins
    if include_overflow:
        assert config['energy']['overflow_bin']['index'] == 600
    y, s, e = np.asarray(label), np.asarray(score,dtype=float), to_keV(energy,energy_unit)
    w = np.ones(s.size) if weight is None else np.asarray(weight,dtype=float)
    g = y.astype(str) if group is None else np.asarray(group).astype(str)
    if any(a.ndim != 1 or a.size != s.size for a in (y,s,e,w,g)):
        raise ValueError('inputs must be aligned one-dimensional arrays')
    if not np.all(np.isin(y,(0,1))):
        raise ValueError('labels must be 0/1 with score increasing toward 1')
    if not np.all(np.isfinite(w)) or np.any(w<0):
        raise ValueError('base weights must be finite nonnegative')
    if np.any(np.isin(g, ('','None','nan'))):
        raise ValueError('missing category; supply explicit binary-label fallback')
    inc = np.isfinite(s) & (w > 0)
    finite = inc & np.isfinite(e)
    bins = energy_bin_indices(e, include_overflow=include_overflow)
    energy_valid = finite & (bins >= 0)
    result = {'protocol_version': config['protocol_version'], 'protocol_sha256': fingerprint(config),
              'n_input': int(s.size), 'inclusive_auc': weighted_auc(y,s,w),
              'n_nonfinite_score': int((~np.isfinite(s)).sum()),
              'n_nonfinite_energy': int((~np.isfinite(e)).sum()),
              'n_zero_weight':int((w == 0).sum()),
              'n_above_range':int((finite & (e>3000)).sum()) if not include_overflow else 0,
              'n_overflow':int((finite & (e>3000)).sum()) if include_overflow else 0,
              'n_below_range':int((finite & (e<0)).sum()), 'energy_bin_count':n_bins}
    def pop(mask):
        return {'n':int(mask.sum()),'base_mass':float(w[mask].sum())}
    # I: retained-event score quantiles and pooled histogram; pre-sparse admissible group masses.
    groups = {}
    minimum = config['independence']['min_events_per_group_energy_bin']
    for name in sorted(np.unique(g)):
        raw = (g == name) & finite
        eligible = (g == name) & energy_valid
        counts = np.bincount(bins[eligible], minlength=n_bins)
        valid = counts >= minimum
        retained = eligible.copy()
        retained[eligible] &= valid[bins[eligible]]
        out = {'original_finite':pop(raw),'energy_population':pop(eligible),'retained':pop(retained),
               'range_excluded':pop(raw & ~energy_valid),
               'below_range':pop(raw & (e<0)), 'above_range':pop(raw & (e>3000)),
               'group_aggregation_mass':float(w[eligible].sum()),
               'valid_bin_count':int(valid.sum()),'valid_bin_indices':np.flatnonzero(valid).tolist(),
               'bin_counts':counts.tolist(),'retained_fraction_energy_population':fraction(w[retained].sum(),w[eligible].sum()),
               'retained_fraction_original_finite':fraction(w[retained].sum(),w[raw].sum()),
               'sparse_excluded':pop(eligible & ~retained),'I_g':None,'D_g_nats':None}
        if not np.any(retained):
            out['status'] = 'not_estimable_no_eligible_energy_bin'
        else:
            edges = score_edges(s[retained],w[retained],config['independence']['score_quantile_bins'])
            pooled = histogram(s[retained],w[retained],edges)
            local_js, local_mass, local_hist = [], [], []
            for b in np.flatnonzero(valid):
                sel = eligible & (bins == b)
                p = histogram(s[sel],w[sel],edges)
                local_js.append(js_nats(p,pooled)); local_mass.append(float(w[sel].sum())); local_hist.append(p.tolist())
            probabilities = np.array(local_mass)/sum(local_mass)
            d = float(np.sum(probabilities*np.array(local_js)))
            out.update(status='ok',I_g=float(np.clip(1-np.sqrt(d/np.log(2)),0,1)),D_g_nats=d,
                       score_edges=edges.tolist(),pooled_histogram=pooled.tolist(),bin_histograms=local_hist,
                       bin_probability=probabilities.tolist(),bin_JS_nats=local_js)
            if valid.sum() == 1:
                out['interpretation'] = 'one eligible energy bin: defined but uninformative about dependence'
        groups[str(name)] = out
    good = [x for x in groups.values() if x['I_g'] is not None]
    result['independence'] = {'groups':groups,'I':float(np.average([x['I_g'] for x in good],weights=[x['group_aggregation_mass'] for x in good])) if good else None,
        'min_group_I':min(x['I_g'] for x in good) if good else None,
        'estimable_groups':len(good),'total_groups':len(groups),
        'status':'ok' if len(good)==len(groups) and good else ('partial_groups' if good else 'not_estimable')}
    # Estimate common support on the population admitted by the explicit profile.
    common, retained = np.zeros(s.size,dtype=bool), np.zeros(s.size,dtype=bool)
    mw = np.zeros(s.size)
    counts, masses, target = np.zeros((2,n_bins),dtype=int),np.zeros((2,n_bins)),np.zeros(n_bins)
    support, formal, diagnostic, reason = None,None,None,None
    status = 'not_estimable_missing_energy_population_class'
    if all(np.any(energy_valid & (y == c)) for c in (0,1)):
        alpha = config['matching']['support_trim_quantile']
        bounds = [weighted_quantile(e[energy_valid & (y==c)],[alpha,1-alpha],w[energy_valid & (y==c)]) for c in (0,1)]
        lo, hi = max(x[0] for x in bounds),min(x[1] for x in bounds)
        support = [float(lo),float(hi)]
        if hi < lo:
            status = 'not_estimable_disjoint_support'
        else:
            common = energy_valid & (e>=lo) & (e<=hi)
            for c in (0,1):
                sel = common & (y==c)
                counts[c] = np.bincount(bins[sel],minlength=n_bins)
                masses[c] = np.bincount(bins[sel],weights=w[sel],minlength=n_bins)
            valid = np.all(counts >= config['matching']['min_events_per_class_bin'],axis=0)
            if np.any(valid):
                probabilities = masses/masses.sum(axis=1)[:,None]
                target = np.minimum(probabilities[0],probabilities[1]); target[~valid] = 0.; target /= target.sum()
                for c in (0,1):
                    sel = common & (y==c)
                    mw[sel] = w[sel]*target[bins[sel]]/masses[c,bins[sel]]
                retained = mw>0
                diagnostic = weighted_auc(y,s,mw)
                cov = [fraction(w[retained & (y==c)].sum(),w[finite & (y==c)].sum()) for c in (0,1)]
                if min(cov) < config['matching']['min_coverage_original_finite']:
                    status = 'not_estimable_low_coverage'
                elif valid.sum() < config['matching']['min_valid_bins'] and lo != hi:
                    status = 'not_estimable_too_few_bins'
                else:
                    status, formal = 'ok',diagnostic
            else:
                status = 'not_estimable_no_eligible_bins'
    valid = np.all(counts >= config['matching']['min_events_per_class_bin'],axis=0)
    classes = {}
    for c in (0,1):
        sel = y==c
        stages = {'input':sel,'inclusive':sel&inc,'original_finite':sel&finite,
                  'energy_population':sel&energy_valid,'common_support':sel&common,'matched':sel&retained,
                  'range_excluded':sel&finite&~energy_valid,
                  'below_range':sel&finite&(e<0),'above_range':sel&finite&(e>3000),
                  'support_excluded_after_range':sel&energy_valid&~common,
                  'sparse_excluded_after_support':sel&common&~retained}
        out = {k:pop(v) for k,v in stages.items()}
        for stage in ('energy_population','common_support','matched'):
            out[stage+'_fraction_original_finite'] = fraction(w[stages[stage]].sum(),w[sel&finite].sum())
        out['matched_fraction_energy_population'] = fraction(w[sel&retained].sum(),w[sel&energy_valid].sum())
        out['matched_ess'] = ess(mw[sel]); out['matched_weight_sum'] = float(mw[sel].sum())
        classes[str(c)] = out
    result['matching'] = {'status':status,'matched_auc':formal,'diagnostic_auc_not_for_reporting':diagnostic,
        'common_support_keV':support,'valid_bin_count':int(valid.sum()),'valid_bin_indices':np.flatnonzero(valid).tolist(),
        'classes':classes,'bin_counts':counts.tolist(),'bin_base_mass':masses.tolist(),'target_mass':target.tolist(),
        'common_support_auc':weighted_auc(y[common],s[common],w[common]),
        'retained_unweighted_auc':weighted_auc(y[retained],s[retained],w[retained])}
    if return_arrays:
        return result,{'matched_weight':mw,'energy_population':energy_valid,'common_support':common,'retained':retained,'energy_bin':bins}
    return result
