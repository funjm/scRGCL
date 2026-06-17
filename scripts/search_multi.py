import multiprocessing as mp
import sys
import os

# 1. 获取当前 main.py 所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取上一级目录（也就是项目根目录）的路径
parent_dir = os.path.dirname(current_dir)

# 3. 将根目录加入到系统路径中
sys.path.append(parent_dir)
from config.opt import parser   # 你的参数解析器

ALL_DATA = ['merged_annotated_cells']

def run_one(name):
    import subprocess, shlex
    cmd = f'python search.py --name {name}'
    subprocess.run(shlex.split(cmd))      # 阻塞直到跑完

if __name__ == '__main__':
    with mp.Pool(processes=min(len(ALL_DATA), mp.cpu_count())) as pool:
        pool.map(run_one, ALL_DATA)