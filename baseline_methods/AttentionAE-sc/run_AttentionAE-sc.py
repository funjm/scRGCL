# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:34:50 2022

@author: LSH
"""
import torch
from warnings import simplefilter 
import argparse
import json
from sklearn import preprocessing
import random
import numpy as np
import pandas as pd
import utils
import scipy
from model import AttentionAE
from train import train, clustering, loss_func
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score
from datetime import datetime


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
        "method": "AttentionAE-sc",
        "dataset": dataset,
        "args": vars(args),
        "metrics": metrics,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved AttentionAE-sc run outputs to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001,help='learning rate, default:1e-3')
    parser.add_argument('--n_z', type=int, default=16, 
                        help='the number of dimension of latent vectors for each cell, default:16')
    parser.add_argument('--n_heads', type=int, default=8, 
                        help='the number of dattention heads, default:8')
    parser.add_argument('--n_hvg', type=int, default=2500, 
                        help='the number of the highly variable genes, default:2500')
    parser.add_argument('--training_epoch', type=int, default=200,
                        help='epoch of train stage, default:200')
    parser.add_argument('--clustering_epoch', type=int, default=100,
                        help='epoch of clustering stage, default:100')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='''the resolution of Leiden. The smaller the settings to get the more clusters
                        , advised to 0.1-1.0, default:1.0''')
    parser.add_argument('--connectivity_methods', type=str, default='gauss',
                        help='method for constructing the cell connectivity ("gauss" or "umap"), default:gauss')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='''The size of local neighborhood (in terms of number of neighboring data points) used 
                        for manifold approximation. Larger values result in more global views of the manifold, while 
                        smaller values result in more local data being preserved. In general values should be in the 
                        range 2 to 100. default:15''')
    parser.add_argument('--knn', type=int, default=False,
                        help='''If True, use a hard threshold to restrict the number of neighbors to n_neighbors, 
                        that is, consider a knn graph. Otherwise, use a Gaussian Kernel to assign low weights to
                        neighbors more distant than the n_neighbors nearest neighbor. default:False''')
    parser.add_argument('--name', type=str, default='Pollen',
                        help='name of input file(a h5ad file: Contains the raw count matrix "X")')
    parser.add_argument('--celltype', type=str, default='known',
                        help='the true labels of datasets are placed in adata.obs["celltype"]')
    parser.add_argument('--save_pred_label', type=str, default=False,
                        help='To choose whether saves the pred_label to the dict "./pred_label"')
    parser.add_argument('--save_model_para', type=str, default=False,
                        help='To choose whether saves the model parameters to the dict "./model_save"')
    parser.add_argument('--save_embedding', type=str, default=False,
                        help='To choose whether saves the cell embedding to the dict "./embedding"')
    parser.add_argument('--save_umap', type=str, default=False,
                        help='To choose whether saves the visualization to the dict "./umap_figure"')
    parser.add_argument('--out-dir', type=str, default='/home/fanjunming/workspace/bioinfo/scRGCL/baseline/results/AttentionAE-sc',
                        help='output directory for run metrics, predictions, embeddings, and summary')
    parser.add_argument('--max_num_cell', type=int, default=10000,
                        help='''a maximum threshold about the number of cells use in the model building, 
                        4,000 is the maximum cells that a GPU owning 8 GB memory can handle. 
                        More cells will bemploy the down-sampling straegy, 
                        which has been shown to be equally effective,
                        but it's recommended to process data with less than 24,000 cells at a time''')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='use GPU, or else use cpu (setting as "False")')
    args = parser.parse_args()
    select_best_gpu(min_free_memory_mb=1000, set_visible_devices=True, sample_times=5, interval=1.0)
    device = torch.device("cuda" if args.cuda else "cpu")
    simplefilter(action='ignore', category=FutureWarning)
    start_time = time.time()
    
    random.seed(1)
    if args.save_umap is True:
        umap_save_path = ['./umap_figure/%s_pred_label.png'%(args.name),'./umap_figure/%s_true_label.png'%(args.name)]
    else:
        umap_save_path = [None, None]
        
    adata, rawData, dataset, adj, r_adj = utils.load_data('./Data/AnnData/{}'.format(args.name),args=args)
    
    if args.celltype == "known":  
        celltype = adata.obs['celltype'].tolist()
    else:
        celltype = None
        
    if adata.shape[0] < args.max_num_cell:
        size_factor = adata.obs['size_factors'].values
        Zscore_data = preprocessing.scale(dataset)
        
        args.n_input = dataset.shape[1]
        print(args)
        init_model = AttentionAE(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, heads=args.n_heads, device=device)
        pretrain_model, _ = train(init_model, Zscore_data, rawData, adj, r_adj, size_factor, device, args)
        metric, pred_label, _, model, _ = clustering(pretrain_model, Zscore_data, rawData, celltype, 
                                                  adj, r_adj, size_factor, device, args)
        asw = metric[0]
        db  = metric[1]
        if celltype is not None:
            nmi = np.round(normalized_mutual_info_score(celltype, pred_label), 3)
            ari = np.round(adjusted_rand_score(celltype, pred_label), 3)
            print("Final ASW %.3f, DB %.3f, ARI %.3f, NMI %.3f"% (asw, db, ari, nmi))
            
        else:
            print("Final ASW %.3f, DB %.3f"% (asw, db))
        data = torch.Tensor(Zscore_data).to(device)
        if type(adj) == scipy.sparse._csr.csr_matrix:
            adj = utils.sparse_mx_to_torch_sparse_tnsor(adj).to(device)
        else:
            adj = torch.Tensor(adj).to(device)
        with torch.no_grad():
            z, _, _, _, _  = model(data, adj)
            if args.save_umap is True:
                utils.umap_visual(z.detach().cpu().numpy(), 
                                  label = pred_label, 
                                  title='predicted label', 
                                  save_path = umap_save_path[0],
                                  asw_used=True)
                if args.celltype == "known":  
                    utils.umap_visual(z.detach().cpu().numpy(), 
                                      label = celltype, 
                                      title='true label', 
                                      save_path = umap_save_path[1])
        if args.save_embedding is True:
            np.savetxt('./embedding/%s.csv'%(args.name), z.detach().cpu().numpy())
        if args.save_pred_label is True:
            np.savetxt('./pred_label/%s.csv'%(args.name),pred_label)
        if args.save_model_para is True:
            torch.save(model.state_dict(), './model_save/%s.pkl'%(args.name))
        elapsed = time.time() - start_time
        save_baseline_results(
            args.out_dir,
            args.name,
            args,
            z.detach().cpu().numpy(),
            celltype,
            pred_label,
            {
                "asw": float(asw),
                "db": float(db),
                "ari": float(ari) if celltype is not None else None,
                "nmi": float(nmi) if celltype is not None else None,
                "duration_sec": float(elapsed),
                "n_clusters": int(len(np.unique(pred_label))),
            },
        )
        
        # output predicted labels
        # np.savetxt('./results/%s_predicted_label.csv'%(args.name),pred_label)

    #down-sampling input
    else:
        new_adata = utils.random_downsimpling(adata, args.max_num_cell)
        new_adj, new_r_adj = utils.adata_knn(new_adata, method = args.connectivity_methods, 
                                             knn=args.knn, n_neighbors=args.n_neighbors)
        try: 
            new_Zscore_data = preprocessing.scale(new_adata.X.toarray())
            new_rawData = new_adata.raw[:, adata.raw.var['highly_variable']].X.toarray()
        except:
            new_Zscore_data = preprocessing.scale(new_adata.X)
            new_rawData = new_adata.raw[:, adata.raw.var['highly_variable']].X
            
        
        
        size_factor = new_adata.obs['size_factors'].values
        try: 
            Zscore_data = preprocessing.scale(dataset.toarray())
            
        except:
            Zscore_data = preprocessing.scale(dataset)
            
        
        new_celltype = new_adata.obs['celltype']
        args.n_input = dataset.shape[1]
        print(args)
        init_model = AttentionAE(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, heads=args.n_heads,device=device)
        pretrain_model, train_elapsed_time  = train(init_model, new_Zscore_data, new_rawData,
                                                    new_adj, new_r_adj, size_factor, device, args)
        _, _, cluster_layer, model, _ = clustering(pretrain_model, new_Zscore_data, new_rawData, 
                                                   new_celltype, new_adj, new_r_adj, size_factor, device, args)
        
        copy_model = AttentionAE(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, heads=args.n_heads, device=torch.device('cpu'))
        copy_model.load_state_dict(model.state_dict())
        
        data = torch.Tensor(Zscore_data).cpu()
        if type(adj) == scipy.sparse._csr.csr_matrix:
            adj = utils.sparse_mx_to_torch_sparse_tnsor(adj).to(device)
        else:
            adj = torch.Tensor(adj).cpu()
            
        with torch.no_grad():
            z, _, _, _, _  = copy_model(data,adj)
            _, p = loss_func(z, cluster_layer.cpu())
            pred_label = utils.dist_2_label(p)
            
            if args.celltype == "known":  
                asw = np.round(silhouette_score(z.detach().cpu().numpy(), pred_label), 3)
                db = np.round(davies_bouldin_score(z.detach().cpu().numpy(), pred_label), 3)
                nmi = np.round(normalized_mutual_info_score(celltype, pred_label), 3)
                ari = np.round(adjusted_rand_score(celltype, pred_label), 3)
                print("Final ASW %.3f, DB %.3f, ARI %.3f, NMI %.3f"% (asw, db, ari, nmi))
            else:
                asw = np.round(silhouette_score(z.detach().cpu().numpy(), pred_label), 3)
                db = np.round(davies_bouldin_score(z.detach().cpu().numpy(), pred_label), 3)
                print("Final ASW %.3f, DB %.3f"% (asw,db))
                
            if args.save_umap is True:
                utils.umap_visual(z.detach().cpu().numpy(), 
                                  label = pred_label, 
                                  title='predicted label', 
                                  save_path = umap_save_path[0],
                                  asw_used=True)
                if args.celltype == "known":  
                    utils.umap_visual(z.detach().cpu().numpy(), 
                                      label = celltype, 
                                      title='true label', 
                                      save_path = umap_save_path[1])
        if args.save_embedding is True:
            np.savetxt('./embedding/%s.csv'%(args.name), z.detach().cpu().numpy())
        if args.save_pred_label is True:
            np.savetxt('./pred_label/%s.csv'%(args.name), pred_label)
        if args.save_model_para is True:
            torch.save(model.state_dict(), './model_save/%s.pkl'%(args.name))
