#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量在多个数据集上运行超参数敏感度分析的简单包装脚本。

用法示例：
  python scripts/run_on_datasets.py --datasets 9,11 --param k
  python scripts/run_on_datasets.py --datasets all --param temperature

该脚本会按顺序调用 scripts/hyperparameter_sensitivity.py。
"""
import argparse
import subprocess
import sys
from pathlib import Path

# 与 main.py 中相同的数据集映射
DATA_DICT = {
    0: 'Quake_10x_Bladder', 1: 'Quake_10x_Limb_Muscle', 2: 'Quake_10x_Spleen',
    3: 'Quake_Smart-seq2_Diaphragm', 4: 'Quake_Smart-seq2_Limb_Muscle',
    5: 'Romanov', 6: 'Muraro', 7: 'Klein', 8: 'Quake_Smart-seq2_Trachea',
    9: 'Pollen', 10: 'Chung', 11: 'Baron1', 12: 'Baron2', 13: 'Baron3', 14: 'Baron4', 15: 'merged_annotated_cells'
}


def parse_dataset_list(spec: str):
    spec = spec.strip()
    if spec.lower() == 'all':
        return [v for _, v in sorted(DATA_DICT.items())]

    parts = [p.strip() for p in spec.split(',') if p.strip()]
    result = []
    for p in parts:
        if p.isdigit():
            idx = int(p)
            if idx in DATA_DICT:
                result.append(DATA_DICT[idx])
            else:
                raise ValueError(f"Unknown dataset index: {idx}")
        else:
            # assume dataset name
            result.append(p)
    return result


def main():
    parser = argparse.ArgumentParser(description='Batch run sensitivity analysis on multiple datasets')
    parser.add_argument('--datasets', default='9', help='Comma-separated dataset indices or names, or "all"')
    parser.add_argument('--param', default='k', help='Hyperparameter name to analyze (passed to hyperparameter_sensitivity.py)')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--python', default=sys.executable, help='Python executable to use')
    args = parser.parse_args()

    try:
        datasets = parse_dataset_list(args.datasets)
    except ValueError as e:
        print(f"Error parsing datasets: {e}")
        return

    script_path = Path(__file__).parent / 'hyperparameter_sensitivity.py'
    if not script_path.exists():
        print(f"Required script not found: {script_path}")
        return

    print(f"Running sensitivity analysis for {len(datasets)} dataset(s): {datasets}")
    for ds in datasets:
        cmd = [args.python, str(script_path), '--sensitivity_dataset', ds, '--param', args.param, '--seed', str(args.seed)]
        print('\n---')
        print('Running:', ' '.join(cmd))
        # run and stream output
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"Dataset {ds} failed with exit code {proc.returncode}. Stopping batch.")
            return

    print('\nBatch completed successfully.')


if __name__ == '__main__':
    main()
# 使用方法示例：
# 单个索引：python run_on_datasets.py --datasets 9 --param k
# 多个索引/名字：python run_on_datasets.py --datasets 9,11,15 --param k
# 全部：python run_on_datasets.py --datasets all --param temperature