"""Deterministic numerical protocol checks; run as a script, no pytest needed."""
from pathlib import Path
import importlib.util
import json
import sys
import types
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from metrics import evaluate, energy_bin_indices, to_keV, weighted_auc, load_config

def example():
    rng = np.random.default_rng(708)
    y = np.repeat([0,1],2000)
    e = np.tile(np.repeat([101.,106.,111.,116.],500),2)
    s = rng.normal(size=y.size)+.3*y+.01*e
    w = rng.uniform(.2,2.,y.size)
    return y,s,e,w

class ProtocolTests(unittest.TestCase):
    def test_units(self):
        y,s,e,w = example()
        a = evaluate(y,s,e,weight=w)
        b = evaluate(y,s,e/1000,weight=w,energy_unit='MeV')
        self.assertEqual(a,b)

    def test_edges_and_range(self):
        np.testing.assert_array_equal(energy_bin_indices([-1,0,4.999,5,10,2999.999,3000,3000.001,np.nan,np.inf]),[-1,0,0,1,2,599,599,-1,-1,-1])
        np.testing.assert_array_equal(energy_bin_indices(to_keV([0,.005,.01,3],'MeV')),[0,1,2,599])

    def test_all_601_unit_boundaries(self):
        edges=np.arange(601)*5.
        np.testing.assert_array_equal(to_keV(edges/1000.,'MeV'),edges)
        middle=edges[1:-1]
        before=to_keV(np.nextafter(middle/1000.,-np.inf),'MeV')
        after=to_keV(np.nextafter(middle/1000.,np.inf),'MeV')
        self.assertTrue(np.all(before<middle));self.assertTrue(np.all(after>middle))

    def test_constant_score(self):
        y,s,e,w = example(); r = evaluate(y,np.ones(y.size),e,weight=w)
        self.assertAlmostEqual(r['inclusive_auc'],.5)
        self.assertAlmostEqual(r['matching']['matched_auc'],.5)
        self.assertEqual(r['independence']['I'],1.)

    def test_bin_only_score(self):
        y,s,e,w = example(); r = evaluate(y,energy_bin_indices(e),e,weight=w)
        self.assertAlmostEqual(r['matching']['matched_auc'],.5,places=14)

    def test_auc_direct_enumeration_ties(self):
        rng = np.random.default_rng(88)
        for _ in range(30):
            y = np.r_[np.zeros(13,dtype=int),np.ones(17,dtype=int)]
            s = rng.integers(-3,4,size=y.size);w = rng.uniform(.01,3,size=y.size)
            diff = s[y==1,None]-s[None,y==0]
            pair = w[y==1,None]*w[None,y==0]
            direct = np.sum(pair*((diff>0)+.5*(diff==0)))/pair.sum()
            self.assertAlmostEqual(weighted_auc(y,s,w),direct,places=14)

    def test_sparse_empty_and_unestimable(self):
        y = np.repeat([0,1],19);e = np.ones(38)*100
        r=evaluate(y,y,e)
        self.assertIsNone(r['independence']['I']);self.assertIsNone(r['matching']['matched_auc'])
        self.assertEqual(r['matching']['valid_bin_count'],0)
        r=evaluate(np.zeros(40),np.ones(40),np.ones(40))
        self.assertIsNone(r['inclusive_auc']);self.assertIsNone(r['matching']['matched_auc'])
        r=evaluate(np.array([],int),np.array([]),np.array([]))
        self.assertIsNone(r['independence']['I']);self.assertIsNone(r['inclusive_auc'])

    def test_one_bin_and_constant_exception(self):
        y=np.repeat([0,1],100);e=np.tile(np.linspace(100.1,104.9,100),2)
        r=evaluate(y,np.ones(200),e)
        self.assertEqual(r['matching']['status'],'not_estimable_too_few_bins')
        self.assertEqual(r['independence']['I'],1.)
        r=evaluate(y,np.ones(200),np.ones(200)*100)
        self.assertEqual(r['matching']['matched_auc'],.5)

    def test_missing_range_and_original_denominator(self):
        y,s,e,w=example();e[:1100]=-2.;e[2000:3100]=-1
        r=evaluate(y,s,e,weight=w)
        self.assertEqual(r['inclusive_auc'],weighted_auc(y,s,w))
        self.assertEqual(r['matching']['status'],'not_estimable_low_coverage')
        for c in ('0','1'):
            x=r['matching']['classes'][c]
            self.assertEqual(x['range_excluded']['n'],1100)
            self.assertGreater(x['matched_fraction_energy_population'],x['matched_fraction_original_finite'])
        e[0]=np.nan;s[1]=np.nan;w[2]=0
        r=evaluate(y,s,e,weight=w)
        self.assertEqual(r['n_nonfinite_energy'],1);self.assertEqual(r['n_nonfinite_score'],1)
        self.assertEqual(r['matching']['classes']['0']['original_finite']['n'],1997)

    def test_upper_range_and_exact_endpoint(self):
        y=np.repeat([0,1],200)
        e=np.tile(np.r_[np.repeat(1.,50),np.repeat(6.,50),np.repeat(3000.,50),np.repeat(3001.,50)],2)
        r,arrays=evaluate(y,np.ones(400),e,return_arrays=True)
        self.assertEqual(r['n_above_range'],100)
        self.assertEqual(r['inclusive_auc'],.5)
        self.assertEqual(r['matching']['matched_auc'],.5)
        self.assertEqual(r['independence']['I'],1.)
        self.assertEqual(r['matching']['valid_bin_indices'],[0,1,599])
        self.assertEqual(len(r['matching']['target_mass']),600)
        for c in ('0','1'):
            x=r['matching']['classes'][c]
            self.assertEqual(x['range_excluded']['n'],50)
            self.assertEqual(x['above_range']['n'],50)
            self.assertEqual(x['matched_fraction_original_finite'],.75)
            self.assertEqual(len(r['independence']['groups'][c]['bin_counts']),600)
        self.assertTrue(np.all(arrays['energy_bin'][e==3000]==599))
        self.assertTrue(np.all(arrays['energy_bin'][e>3000]==-1))
        self.assertTrue(np.all(arrays['matched_weight'][e>3000]==0))

    def test_constant_support_still_requires_original_coverage(self):
        y=np.repeat([0,1],100);e=np.tile(np.r_[np.repeat(100.,20),np.repeat(4000.,80)],2)
        r=evaluate(y,np.ones(200),e)
        self.assertEqual(r['matching']['common_support_keV'],[100.,100.])
        self.assertEqual(r['matching']['status'],'not_estimable_low_coverage')
        self.assertIsNone(r['matching']['matched_auc'])
        self.assertEqual(r['matching']['diagnostic_auc_not_for_reporting'],.5)
        self.assertEqual(r['independence']['I'],1.)
        r=evaluate(y,np.ones(200),np.repeat(3001.,200))
        self.assertEqual(r['inclusive_auc'],.5)
        self.assertIsNone(r['independence']['I'])
        self.assertIsNone(r['matching']['matched_auc'])

    def test_reject_inconsistent_bin_configuration(self):
        y,s,e,w=example();cfg=load_config();cfg['energy']['n_bins']=601
        with self.assertRaises(AssertionError):evaluate(y,s,e,config=cfg)

    def test_invalid_weights_labels(self):
        y,s,e,w=example()
        for value in (-1.,np.nan,np.inf):
            ww=w.copy();ww[0]=value
            with self.assertRaises(ValueError):evaluate(y,s,e,weight=ww)
        y[0]=2
        with self.assertRaises(ValueError):evaluate(y,s,e,weight=w)

    def test_profile_agreement_on_in_range_population(self):
        y,s,e,w=example()
        a=evaluate(y,s,e,weight=w);b=evaluate(y,s,e,weight=w,config=load_config("overflow601"))
        self.assertEqual(a['inclusive_auc'],b['inclusive_auc'])
        self.assertEqual(a['matching']['matched_auc'],b['matching']['matched_auc'])
        self.assertEqual(a['independence']['I'],b['independence']['I'])
        for g in ('0','1'):
            self.assertEqual(a['independence']['groups'][g]['I_g'],b['independence']['groups'][g]['I_g'])

if __name__=='__main__':
    unittest.main(verbosity=2)
