import multiprocessing as mp
import subprocess, shlex, sys
import os

# ALL_DATA = [
#     'Quake_10x_Bladder', 'Quake_10x_Spleen', 'Quake_Smart-seq2_Diaphragm',
#     'Quake_Smart-seq2_Limb_Muscle', 'Romanov', 'Klein', 'Quake_Smart-seq2_Trachea',
#     'Pollen', 'Chung', 'Baron1', 'Baron2', 'Baron3', 'Baron4', 'Muraro'
# ]
ALL_DATA = [
    'Chung', 'Baron1', 'Baron2', 'Klein',
    'Baron3', 'Baron4'
    ]
ALL_DATA = [
    'Klein',
    'Baron3', 'Quake_10x_Limb_Muscle'
    ]
ALL_DATA = [
    'Klein'
    ]
# ALL_DATA = [
#     'Quake_10x_Bladder', 'Quake_10x_Spleen', 'Quake_Smart-seq2_Diaphragm',
#     'Quake_Smart-seq2_Limb_Muscle', 'Romanov', 'Klein', 'Quake_Smart-seq2_Trachea',
#      'Chung', 'Baron1', 'Baron2', 'Baron3', 'Baron4', 'Muraro'
# ]
ABLATIONS = [0, 1, 2, 3, 4]   # 0 表示不消融，1–4 对应不同消融

def run_dataset(task):
    """在一个进程里顺序跑一个数据集的所有 ablation"""
    dataset, gpu_id = task
    for test in ABLATIONS:
        cmd = f'{sys.executable} main_test.py --name {dataset} --test {test}'
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU{gpu_id}] Running:", cmd, flush=True)
        subprocess.run(shlex.split(cmd), env=env, check=True)
        print(f"[GPU{gpu_id}] Finish: {dataset}, test={test}", flush=True)

if __name__ == '__main__':
    n_gpus = 2   # 你机器有两个 GPU: 0 和 1
    tasks = [(dataset, i % n_gpus) for i, dataset in enumerate(ALL_DATA)]

    with mp.Pool(processes=min(len(tasks), mp.cpu_count())) as pool:
        pool.map(run_dataset, tasks)
