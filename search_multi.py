import multiprocessing as mp
from opt import parser   # 你的参数解析器

ALL_DATA = ['Klein','Baron3', 
            'Quake_Smart-seq2_Limb_Muscle']

def run_one(name):
    import subprocess, shlex
    cmd = f'python search.py --name {name}'
    subprocess.run(shlex.split(cmd))      # 阻塞直到跑完

if __name__ == '__main__':
    with mp.Pool(processes=min(len(ALL_DATA), mp.cpu_count())) as pool:
        pool.map(run_one, ALL_DATA)