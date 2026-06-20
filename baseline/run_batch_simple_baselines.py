import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime


DATA_DICT = {
    0: 'Quake_10x_Bladder',
    1: 'Quake_10x_Limb_Muscle',
    2: 'Quake_10x_Spleen',
    3: 'Quake_Smart-seq2_Diaphragm',
    4: 'Quake_Smart-seq2_Limb_Muscle',
    5: 'Romanov',
    6: 'Muraro',
    7: 'Klein',
    8: 'Quake_Smart-seq2_Trachea',
    9: 'Pollen',
    10: 'Chung',
    11: 'Baron1',
    12: 'Baron2',
    13: 'Baron3',
    14: 'Baron4',
    15: 'merged_annotated_cells',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch-run simple clustering baselines on multiple datasets.')
    parser.add_argument(
        '--python',
        default='/disk/fanjunming/home/miniconda3/envs/CAKE/bin/python',
        help='Python executable used to launch the per-dataset baseline script.',
    )
    parser.add_argument(
        '--script',
        default='/disk/fanjunming/home/workspace/bioinfo/scRGCL/baseline/run_pollen_simple_baselines.py',
        help='Per-dataset baseline script path.',
    )
    parser.add_argument(
        '--base-out-dir',
        default='/disk/fanjunming/home/workspace/bioinfo/scRGCL/baseline/results/batch_simple_baselines',
        help='Base output directory for all datasets.',
    )
    parser.add_argument('--n-pcs', type=int, default=30)
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--resolution', type=float, default=1.0)
    parser.add_argument('--n-top-genes', type=int, default=2000)
    parser.add_argument('--target-sum', type=float, default=1e4)
    parser.add_argument('--scale-max-value', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--dataset-ids',
        nargs='*',
        type=int,
        default=sorted(DATA_DICT),
        help='Optional subset of dataset IDs to run. Default: all IDs in DATA_DICT.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # ！！！
    args.dataset_ids = [10]
    os.makedirs(args.base_out_dir, exist_ok=True)
    summary_rows = []

    run_started = datetime.now().isoformat(timespec='seconds')
    print(f'Batch run started at {run_started}')

    for dataset_id in args.dataset_ids:
        if dataset_id not in DATA_DICT:
            summary_rows.append({
                'dataset_id': dataset_id,
                'dataset': f'UNKNOWN_ID_{dataset_id}',
                'status': 'skipped',
                'returncode': '',
                'metrics_path': '',
                'stdout_log': '',
                'stderr_log': '',
                'error': 'dataset_id not found in DATA_DICT',
            })
            continue

        dataset_name = DATA_DICT[dataset_id]
        dataset_out_dir = os.path.join(args.base_out_dir, dataset_name)
        os.makedirs(dataset_out_dir, exist_ok=True)
        stdout_log = os.path.join(dataset_out_dir, 'stdout.log')
        stderr_log = os.path.join(dataset_out_dir, 'stderr.log')

        cmd = [
            args.python,
            args.script,
            '--dataset', dataset_name,
            '--out-dir', dataset_out_dir,
            '--n-pcs', str(args.n_pcs),
            '--n-neighbors', str(args.n_neighbors),
            '--resolution', str(args.resolution),
            '--n-top-genes', str(args.n_top_genes),
            '--target-sum', str(args.target_sum),
            '--scale-max-value', str(args.scale_max_value),
            '--seed', str(args.seed),
        ]

        print(f'\n[{dataset_id}] Running {dataset_name}')
        print(' '.join(cmd))

        with open(stdout_log, 'w', encoding='utf-8') as out_f, open(stderr_log, 'w', encoding='utf-8') as err_f:
            proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, text=True)

        metrics_path = os.path.join(dataset_out_dir, 'metrics.csv')
        status = 'success' if proc.returncode == 0 and os.path.exists(metrics_path) else 'failed'
        error = '' if status == 'success' else f'Process exited with code {proc.returncode}'

        summary_rows.append({
            'dataset_id': dataset_id,
            'dataset': dataset_name,
            'status': status,
            'returncode': proc.returncode,
            'metrics_path': metrics_path if os.path.exists(metrics_path) else '',
            'stdout_log': stdout_log,
            'stderr_log': stderr_log,
            'error': error,
        })
        print(f'[{dataset_id}] {dataset_name} -> {status}')

    summary_csv = os.path.join(args.base_out_dir, 'batch_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['dataset_id', 'dataset', 'status', 'returncode', 'metrics_path', 'stdout_log', 'stderr_log', 'error'],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    success_count = sum(row['status'] == 'success' for row in summary_rows)
    failed_count = sum(row['status'] == 'failed' for row in summary_rows)
    skipped_count = sum(row['status'] == 'skipped' for row in summary_rows)
    print(f'\nBatch run finished. success={success_count}, failed={failed_count}, skipped={skipped_count}')
    print(f'Summary written to: {summary_csv}')

    if failed_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
