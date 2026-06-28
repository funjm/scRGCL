#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch run ScCCL on a fixed set of datasets.

Usage examples:
  python run_scCCL_datasets.py --datasets all
  python run_scCCL_datasets.py --datasets 0,9,11
  python run_scCCL_datasets.py --datasets Quake_10x_Bladder,Pollen
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATA_DICT = {
    0: 'Quake_10x_Bladder',
    1: 'Quake_Smart-seq2_Diaphragm',
    2: 'Quake_Smart-seq2_Limb_Muscle',
    3: 'Romanov',
    4: 'Muraro',
    5: 'Klein',
    6: 'Quake_Smart-seq2_Trachea',
    7: 'Pollen',
    8: 'Chung',
    9: 'Baron1',
    10: 'Baron2',
    11: 'Baron3',
    12: 'Baron4',
    13: 'Quake_10x_Limb_Muscle',
    14: 'Quake_10x_Spleen',
}


def parse_dataset_list(spec: str):
    spec = spec.strip()
    if spec.lower() == 'all':
        return [v for _, v in sorted(DATA_DICT.items())]

    if '-' in spec and all(p.strip().isdigit() for p in spec.split('-', 1)):
        start, end = [int(p.strip()) for p in spec.split('-', 1)]
        if start > end:
            raise ValueError(f'Invalid range: {spec}')
        selected = []
        for idx in range(start, end + 1):
            if idx not in DATA_DICT:
                raise ValueError(f'Unknown dataset index in range: {idx}')
            selected.append(DATA_DICT[idx])
        return selected

    parts = [p.strip() for p in spec.split(',') if p.strip()]
    result = []
    for p in parts:
        if p.isdigit():
            idx = int(p)
            if idx in DATA_DICT:
                result.append(DATA_DICT[idx])
            else:
                raise ValueError(f'Unknown dataset index: {idx}')
        else:
            result.append(p)
    return result


def main():
    parser = argparse.ArgumentParser(description='Batch run ScCCL on multiple datasets')
    parser.add_argument('--datasets', default='all', help='Comma-separated dataset indices/names, a range like 0-14, or "all"')
    parser.add_argument('--python', default=sys.executable, help='Python executable to use')
    parser.add_argument('--script', default='/disk/fanjunming/home/workspace/bioinfo/scRGCL/baseline_methods/ScCCL/main.py', help='Path to the ScCCL entry script')
    parser.add_argument('--out-dir', default='/disk/fanjunming/home/workspace/bioinfo/scRGCL/baseline/results/ScCCL', help='Directory for per-run metrics, predictions, embeddings, and summary files')
    parser.add_argument('--extra-args', default='', help='Extra arguments to forward to the script')
    args = parser.parse_args()

    try:
        datasets = parse_dataset_list(args.datasets)
    except ValueError as exc:
        print(f'Error parsing datasets: {exc}', file=sys.stderr)
        sys.exit(1)

    script_path = Path(args.script)
    if not script_path.exists():
        print(f'Required script not found: {script_path}', file=sys.stderr)
        sys.exit(1)

    print(f'Running ScCCL for {len(datasets)} dataset(s): {datasets}')
    failed_datasets = []
    for ds in datasets:
        cmd = [args.python, str(script_path), '--name', ds, '--out-dir', args.out_dir]
        if args.extra_args:
            cmd.extend(args.extra_args.split())

        print('\n---')
        print('Running:', ' '.join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            failed_datasets.append(ds)
            print(f'Dataset {ds} failed with exit code {proc.returncode}. Continuing to next dataset.', file=sys.stderr)
            continue

    if failed_datasets:
        print(f'\nBatch completed with failures: {failed_datasets}', file=sys.stderr)
        sys.exit(1)

    print('\nBatch completed successfully.')


if __name__ == '__main__':
    main()
