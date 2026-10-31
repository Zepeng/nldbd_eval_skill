#!/usr/bin/env python3
"""Evaluate event-level predictions with the recorded EnergyBench paper profile."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

sys.dont_write_bytecode = True
from metrics import PROFILES, evaluate, fingerprint, load_config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def evaluate_file(path, profile, energy_key='energy_keV', energy_unit='keV', expected=None):
    with np.load(path, allow_pickle=False) as archive:
        required = {'score', 'label', 'event_id', energy_key}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f'Missing event arrays: {sorted(missing)}')
        data = {key: np.array(archive[key]) for key in archive.files}
    size = len(data['score'])
    for key in required | ({'weight', 'group'} & set(data)):
        if data[key].ndim != 1 or len(data[key]) != size:
            raise ValueError(f'{key} must be a one-dimensional event-aligned array')
    ids = data['event_id'].astype(str)
    if len(np.unique(ids)) != size or np.any(np.isin(ids, ['', 'None', 'nan'])):
        raise ValueError('event_id must be unique and nonempty')
    config = load_config(profile)
    result = evaluate(data['label'], data['score'], data[energy_key],
                      group=data.get('group'), weight=data.get('weight'),
                      energy_unit=energy_unit, config=config)
    output = {'input_sha256': sha256(path), 'profile': profile, 'metrics': result}
    if expected is not None:
        observed = {'I': result['independence']['I'],
                    'matched_auc': result['matching']['matched_auc'],
                    'inclusive_auc': result['inclusive_auc'], 'n_events': result['n_input']}
        for key, value in expected.items():
            if value is None:
                equal = observed[key] is None
            else:
                equal = observed[key] is not None and abs(observed[key] - value) <= 2e-12
            if not equal:
                raise AssertionError(f'{key}: evaluated {observed[key]}, expected {value}')
        output['matches_paper_result'] = True
    return output


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='One standardized prediction NPZ')
    source.add_argument('--manifest', type=Path, help='Published result/input manifest')
    parser.add_argument('--data-root', type=Path, help='Directory containing the manifest inputs')
    parser.add_argument('--profile', choices=PROFILES, help='Required for a single input')
    parser.add_argument('--dataset', action='append', help='Exact dataset filter; repeat as needed')
    parser.add_argument('--energy-key', default='energy_keV')
    parser.add_argument('--energy-unit', choices=('keV', 'MeV'), default='keV')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.input:
        if not args.profile:
            parser.error('--input requires --profile; use strict600 for the 0–3000 keV protocol')
        if args.dataset or args.data_root:
            parser.error('--dataset and --data-root apply only to manifest mode')
        result = evaluate_file(args.input, args.profile, args.energy_key, args.energy_unit)
        save(args.output, result)
        print(json.dumps({'I': result['metrics']['independence']['I'],
                          'matched_auc': result['metrics']['matching']['matched_auc'],
                          'output': str(args.output)}))
        return
    if args.profile or not args.data_root:
        parser.error('Manifest mode requires --data-root and selects the recorded profile per row')
    if args.energy_key != 'energy_keV' or args.energy_unit != 'keV':
        parser.error('Manifest inputs are already standardized to energy_keV')
    rows = json.loads(args.manifest.read_text())['records']
    if args.dataset:
        rows = [row for row in rows if row['dataset'] in args.dataset]
    if not rows:
        parser.error('No matching result records')
    missing = [str(args.data_root / row['relative_input']) for row in rows
               if not (args.data_root / row['relative_input']).is_file()]
    if missing:
        raise FileNotFoundError('Required event inputs are unavailable:\n' + '\n'.join(missing))
    receipt = []
    for row in rows:
        path = args.data_root / row['relative_input']
        if sha256(path) != row['input_sha256']:
            raise ValueError('Input SHA-256 mismatch: ' + row['id'])
        config = load_config(row['profile'])
        if fingerprint(config) != row['protocol_sha256']:
            raise ValueError('Protocol SHA-256 mismatch: ' + row['id'])
        result = evaluate_file(path, row['profile'], expected=row['expected'])
        save(args.output / (row['id'] + '.json'), result)
        receipt.append({'id': row['id'], 'profile': row['profile'], 'matches_paper_result': True})
        print(row['id'], 'PASS', flush=True)
    save(args.output / 'receipt.json', {'records': receipt, 'all_pass': True})


if __name__ == '__main__':
    main()
