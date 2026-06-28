import sys
from pathlib import Path
import json
from datetime import datetime

import torch
import opt
import numpy as np
import pandas as pd
import train

from utils import preprocess

import subprocess, time, os
def select_best_gpu(
    min_free_memory_mb=1000,
    set_visible_devices=True,
    sample_times=5,
    interval=1.0,
):
    """
    查询当前所有 GPU 的使用情况，选择负载最小的一张 GPU。

    选择规则：
    - 连续采样多次，避免 GPU-Util 的瞬时波动影响判断
    - 先过滤掉剩余显存小于 min_free_memory_mb 的 GPU
    - 再根据“平均 GPU 利用率 + 显存占用率”的综合分数选最小值

    参数：
        min_free_memory_mb: 最少需要的空闲显存，单位 MB
        set_visible_devices: 是否自动设置环境变量 CUDA_VISIBLE_DEVICES
        sample_times: 采样次数
        interval: 每次采样之间的间隔时间，单位秒

    返回：
        best_gpu: 选中的 GPU 编号
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]

    gpu_records = {}

    for sample_idx in range(sample_times):
        # 调用 nvidia-smi 获取 GPU 信息
        result = subprocess.check_output(cmd, encoding="utf-8")

        for line in result.strip().split("\n"):
            idx, util, mem_used, mem_total = [x.strip() for x in line.split(",")]

            idx = int(idx)
            util = float(util)
            mem_used = float(mem_used)
            mem_total = float(mem_total)

            if idx not in gpu_records:
                gpu_records[idx] = {
                    "utils": [],
                    "mem_used": mem_used,
                    "mem_total": mem_total,
                }

            gpu_records[idx]["utils"].append(util)
            # 显存使用取最近一次即可，因为通常比 util 更稳定
            gpu_records[idx]["mem_used"] = mem_used
            gpu_records[idx]["mem_total"] = mem_total

        if sample_idx < sample_times - 1:
            time.sleep(interval)

    candidates = []
    for idx, record in gpu_records.items():
        avg_util = sum(record["utils"]) / len(record["utils"])
        mem_used = record["mem_used"]
        mem_total = record["mem_total"]
        mem_free = mem_total - mem_used

        # 如果空闲显存不足，直接跳过
        if mem_free < min_free_memory_mb:
            continue

        # 显存占用率
        mem_ratio = mem_used / mem_total if mem_total > 0 else 1.0

        # 综合分数，越小表示当前越空闲
        # 这里让显存占用率权重大一些，平均 GPU 利用率权重小一些
        score = 0.2 * (avg_util / 100.0) + 0.8 * mem_ratio

        candidates.append((idx, score, avg_util, mem_used, mem_total))

    if not candidates:
        raise RuntimeError("没有找到满足空闲显存要求的 GPU")

    # 选择综合分数最低的 GPU
    best_gpu, _, _, _, _ = min(candidates, key=lambda x: x[1])

    # 如果需要，自动设置 CUDA_VISIBLE_DEVICES
    if set_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

    print(f"选择的 GPU: {best_gpu}，综合分数最低，平均 GPU 利用率: {avg_util:.2f}%，显存使用: {mem_used:.2f}MB / {mem_total:.2f}MB")

    return best_gpu


def save_baseline_results(out_dir, dataset, args, embedding, y_true, y_pred, metrics):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"{dataset}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    pred_df = pd.DataFrame({"pred_label": np.asarray(y_pred, dtype=int)})
    if y_true is not None:
        pred_df["label"] = np.asarray(y_true, dtype=object)
    pred_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
    pd.DataFrame(embedding).to_csv(os.path.join(run_dir, "embedding.csv"), index=False)
    summary = {
        "method": "ScCCL",
        "dataset": dataset,
        "args": vars(args),
        "metrics": metrics,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved ScCCL run outputs to: {run_dir}")


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.utils import get_dataset

    getdevice = select_best_gpu(min_free_memory_mb=1000, set_visible_devices=True, sample_times=5, interval=1.0)

    gene_exp, real_label = get_dataset(opt.args.name)
    gene_exp = preprocess(gene_exp, opt.args.select_gene)

    print(f"The gene expression matrix shape is {gene_exp.shape}...")
    cluster_number = np.unique(real_label).shape[0]
    print(f"The real clustering num is {cluster_number}...")

    results = train.run(gene_exp=gene_exp, cluster_number=cluster_number, dataset=opt.args.name,
                        real_label=real_label, epochs=opt.args.epoch, lr=opt.args.lr,
                        temperature=opt.args.temperature, dropout=opt.args.dropout,
                        layers=[opt.args.enc_1, opt.args.enc_2, opt.args.enc_3, opt.args.mlp_dim],
                        save_pred=True, cluster_methods=opt.args.cluster_methods, batch_size=opt.args.batch_size,
                        m=opt.args.m, noise=opt.args.noise)

    print("ARI:    " + str(results["ari"]))
    print("NMI:    " + str(results["nmi"]))
    print("Time:   " + str(results['time']))

    save_baseline_results(
        opt.args.out_dir,
        opt.args.name,
        opt.args,
        results.get("features", np.empty((0, 0))),
        real_label,
        results.get("pred", np.asarray([])),
        {
            "ari": float(results.get("ari", np.nan)),
            "nmi": float(results.get("nmi", np.nan)),
            "duration_sec": float(results.get("time", np.nan)),
            "n_clusters": int(len(np.unique(results.get("pred", np.asarray([]))))),
        },
    )
