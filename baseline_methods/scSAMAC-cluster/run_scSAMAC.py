from time import time
from pathlib import Path
import math, os
import sys
import json
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from openTSNE import TSNE
from scSAMAC import scSAMAC
from single_cell_tools import *
import numpy as np
import pandas as pd
import collections
from sklearn import metrics
from datetime import datetime
import scanpy as sc
from preprocess import read_dataset, normalize

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
        "method": "scSAMAC",
        "dataset": dataset,
        "args": vars(args),
        "metrics": metrics,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved scSAMAC run outputs to: {run_dir}")


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.utils import get_dataset

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='scSAMAC: model-based deep embedding clustering for single-cell RNA-seq data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=0, type=int, 
                        help='number of clusters, 0 means estimating by the Louvain algorithm')
    parser.add_argument('--knn', default=20, type=int, 
                        help='number of nearest neighbors, used by the Louvain algorithm')
    parser.add_argument('--resolution', default=.8, type=float,
                        help='resolution parameter, used by the Louvain algorithm, larger value for more number of clusters')
    parser.add_argument('--select_genes', default=0, type=int, 
                        help='number of selected genes, 0 means using all genes')
    parser.add_argument('--batch_size', default=256, type=int)

    parser.add_argument('--data_file', default='Pollen')

    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--sigma', default=2.5, type=float,
                        help='coefficient of random noise')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float,
                        help='tolerance for delta clustering labels to terminate training stage')

    parser.add_argument('--ae_weights', default=None,
                        help='file to pretrained weights, None for a new pretraining')

    parser.add_argument('--save_dir', default='results/scDeepCluster/',
                        help='directory to save model weights during the training stage')


    parser.add_argument('--ae_weight_file', default='AE_weights.pth.tar',
                        help='file name to save model weights after the pretraining stage')
    parser.add_argument('--final_latent_file', default='final_latent_file.txt',
                        help='file name to save final latent representations')
    parser.add_argument('--predict_label_file', default='pred_labels.txt',
                        help='file name to save final clustering labels')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-dir', default='/home/fanjunming/workspace/bioinfo/scRGCL/baseline/results/scSAMAC-cluster')

    args = parser.parse_args()

    select_best_gpu(min_free_memory_mb=1000, set_visible_devices=True, sample_times=5, interval=1.0)
    start_time = time.time()

    dataset_name = os.path.splitext(args.data_file)[0]
    x, y = get_dataset(dataset_name)
    y = pd.factorize(y)[0]

    if args.select_genes > 0:
        importantGenes = geneSelection(x, n=args.select_genes, plot=False)
        x = x[:, importantGenes]

    adata = sc.AnnData(x, dtype="float64")
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    input_size = adata.n_vars

    print(args)

    print(adata.X.shape)
    if y is not None:
        print(y.shape)

#    x_sd = adata.X.std(0)
#    x_sd_median = np.median(x_sd)
#    print("median of gene sd: %.5f" % x_sd_median)


    model = scSAMAC(input_dim=adata.n_vars, z_dim=32,
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=args.sigma, gamma=args.gamma, device=args.device)
    
    print(str(model))

    t0 = time.time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    
    print('Pretraining time: %d seconds.' % int(time.time() - t0))

    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.n_clusters > 0:
        y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, n_clusters=args.n_clusters, init_centroid=None, 
                    y_pred_init=None, y=None, batch_size=args.batch_size, num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
    else:
        import torch
        ### estimate number of clusters by Louvain algorithm on the autoencoder latent representations
        pretrain_latent = model.encodeBatch(torch.tensor(adata.X, dtype=torch.float64)).cpu().numpy()
        adata_latent = sc.AnnData(pretrain_latent)
        sc.pp.neighbors(adata_latent, n_neighbors=args.knn, use_rep="X")
        sc.tl.leiden(adata_latent, resolution=args.resolution)
        y_pred_init = np.asarray(adata_latent.obs['leiden'],dtype=int)
        features = pd.DataFrame(adata_latent.X,index=np.arange(0,adata_latent.n_obs))
        Group = pd.Series(y_pred_init,index=np.arange(0,adata_latent.n_obs),name="Group")
        Mergefeature = pd.concat([features,Group],axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
        n_clusters = cluster_centers.shape[0]
        print('Estimated number of clusters: ', n_clusters)
        y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, n_clusters=n_clusters, init_centroid=cluster_centers, 
                    y_pred_init=y_pred_init, y=y, batch_size=args.batch_size, num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)


    print('Total time: %d seconds.' % int(time.time() - t0))


    if y is not None:
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        print('Evaluating cells: NMI= %.4f, ARI= %.4f, ACC= %.4f' % (nmi, ari, acc))

    final_latent = model.encodeBatch(torch.tensor(adata.X, dtype=torch.float64)).cpu().numpy()
    np.savetxt(args.final_latent_file, final_latent, delimiter=",")
    np.savetxt(args.predict_label_file, y_pred, delimiter=",", fmt="%i")
    save_baseline_results(
        args.out_dir,
        dataset_name,
        args,
        final_latent,
        y,
        y_pred,
        {
            "acc": float(acc) if y is not None else None,
            "nmi": float(nmi) if y is not None else None,
            "ari": float(ari) if y is not None else None,
            "duration_sec": float(time.time() - start_time),
            "n_clusters": int(len(np.unique(y_pred))),
        },
    )

    # tsne_embedding = TSNE(
    #     perplexity=30,
    #     initialization="pca",
    #     metric="euclidean",
    #     n_jobs=8,
    #     random_state=42,
    # )
    # latent_tsne_2 = tsne_embedding.fit(final_latent)
    # np.savetxt("tsne_2D.txt", latent_tsne_2, delimiter=",")
