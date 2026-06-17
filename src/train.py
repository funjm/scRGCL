import copy
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src import ScRGCL
from src import st_loss
from src.st_loss import cosine_sim, compute_knn
from src.clustering import clustering
from src.evaluation import evaluate
from src.utils import (get_device,
                       adjust_learning_rate,
                       save_model,
                       cluster_embedding,
                       high_confidence_adj,
                       evaluate_soft,
                       plot_learning_curves,
                       show_tsne,
                       align_cluster_labels,
                       select_best_gpu)

from torch.utils.tensorboard import SummaryWriter

# 单gpu
gpu_id = select_best_gpu()
print(f"选择的 GPU 是: {gpu_id}")
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def save_training_outputs(out_dir, dataset, gene_exp, features, labels, predictions, metrics, train_config):
    os.makedirs(out_dir, exist_ok=True)

    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(out_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    pred_df = pd.DataFrame({"label": labels})
    for name, values in predictions.items():
        pred_df[name] = values
    pred_path = os.path.join(out_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    embedding_path = os.path.join(out_dir, "embeddings.csv")
    pd.DataFrame(features).to_csv(embedding_path, index=False)

    summary = {
        "dataset": dataset,
        "n_cells": int(gene_exp.shape[0]),
        "n_input_features": int(gene_exp.shape[1]),
        "n_label_classes": int(len(np.unique(labels))),
        "args": train_config,
        "results": [metrics],
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {pred_path}")
    print(f"Saved embeddings to: {embedding_path}")
    print(f"Saved summary to: {summary_path}")



def resolve_dropout_rates(dropout, dropout_weak=None, dropout_strong=None):
    resolved_dropout_weak = dropout if dropout_weak is None else dropout_weak
    resolved_dropout_strong = dropout if dropout_strong is None else dropout_strong

    return resolved_dropout_weak, resolved_dropout_strong



def run(gene_exp, cluster_number, dataset, real_label, epochs, lr, temperature, dropout, layers, batch_size, m,
        save_pred=True, noise=None, use_cpu=None, cluster_methods=None, logger=None,
        dropout_weak=None, dropout_strong=None):
    if cluster_methods is None:
        cluster_methods = []
    results = {}

    start = time.time()
    embedding, best_model = train_model(gene_exp=gene_exp, cluster_number=cluster_number, real_label=real_label,
                                        epochs=epochs, lr=lr, temperature=temperature,
                                        dropout=dropout, dropout_weak=dropout_weak,
                                        dropout_strong=dropout_strong,
                                        layers=layers, batch_size=batch_size,
                                        m=m, save_pred=save_pred, noise=noise, use_cpu=use_cpu, logger=logger,
                                        dataset=dataset)  # 新增 dataset 参数

    if save_pred:
        results[f"features"] = embedding
        results[f"max_epoch"] = best_model
    elapsed = time.time() - start
    res_eval = cluster_embedding(embedding, cluster_number, real_label, save_pred=save_pred,
                                 cluster_methods=cluster_methods, random_state=20)
    results = {**results, **res_eval, "dataset": dataset, "time": elapsed}

    return results


def train_model(gene_exp, cluster_number, real_label, epochs, lr,
                temperature, dropout, layers, batch_size, m, lambda_i, lambda_c, lambda_p,
                k, n_neighbors, save_pred=True, noise=None, use_cpu=None, evaluate_training=True,
                eval_on_cpu=False, logger=None, dataset=None, save_fig_flag=False, log_name=None,
                dropout_weak=None, dropout_strong=None):
    device = get_device(use_cpu)
    resolved_dropout_weak, resolved_dropout_strong = resolve_dropout_rates(
        dropout,
        dropout_weak=dropout_weak,
        dropout_strong=dropout_strong,
    )

    # 初始化TensorBoard写入器
    tb_log_dir = os.path.join("runs", log_name if log_name else f"{dataset}_{time.strftime('%Y%m%d%H%M%S')}")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    dims = np.concatenate([[gene_exp.shape[1]], layers])
    data_aug_model_weak = ScRGCL.DataAug(dropout=resolved_dropout_weak)
    data_aug_model_strong = ScRGCL.DataAug(dropout=resolved_dropout_strong)
    if logger is not None:
        logger.write(
            f"Resolved dropout rates -> weak: {resolved_dropout_weak}, strong: {resolved_dropout_strong}"
        )
    encoder_q = ScRGCL.BaseEncoder(dims)
    encoder_k = ScRGCL.BaseEncoder(dims)
    instance_projector = ScRGCL.MLP(layers[2], layers[2] + layers[3], layers[2] + layers[3])
    cluster_projector = ScRGCL.MLP(layers[2], layers[3], cluster_number)
    instance_dim = layers[2] + layers[3]
    model = ScRGCL.ScRGCL(encoder_q, encoder_k, instance_projector, cluster_projector, cluster_number, instance_dim, m=m)
    # 数据增强和输入张量保持在 CPU，避免 DataParallel 前将整个 batch 复制到主卡
    model.to(device)

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    cpu_eval_model = None
    cpu_eval_criterion_rgc = None
    if eval_on_cpu and device.type == 'cuda':
        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        cpu_eval_model = copy.deepcopy(base_model).cpu()
        cpu_eval_model.eval()
        cpu_eval_criterion_rgc = st_loss.RGCLoss(temperature=temperature, device=torch.device('cpu'))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion_rgc = st_loss.RGCLoss(temperature=temperature, device=device)
    criterion_cluster = st_loss.ClusterLoss(cluster_number, temperature=temperature)
    criterion_prototype = st_loss.PrototypeLoss(temperature=temperature)

    max_value_q, max_value_k, max_value_fu = -1, -1, -1
    best_acc, best_nmi, best_ari, best_f1, best_epoch = 0.0, 0.0, float('-inf'), 0.0, 0
    soft_res_best = None

    best_features = None
    best_label_pred = None
    best_label_pred_aligned = None
    best_results = {}
    score_dict = {'loss': [], 'nmi_q': [], 'ari_q': []}

    idx = np.arange(len(gene_exp))
    save_adj_flag = 0 # 只在第一个epoch保存邻接矩阵
    for epoch in range(epochs):
        model.train()
        adjust_learning_rate(optimizer, epoch, lr)
        np.random.shuffle(idx)
        loss_instance_ = 0
        loss_prototype_ = 0
        loss_cluster_ = 0
        loss_sum = 0
        batch_count = 0
        
        for pre_index in range(len(gene_exp) // batch_size + 1):
            c_idx = np.arange(pre_index * batch_size,
                              min(len(gene_exp), (pre_index + 1) * batch_size))
            if len(c_idx) == 0:
                continue
            c_idx = idx[c_idx]
            c_inp = gene_exp[c_idx]
            input1 = data_aug_model_weak(torch.FloatTensor(c_inp))
            input2 = data_aug_model_strong(torch.FloatTensor(c_inp))

            feat_real_label = real_label[c_idx]

            if noise is None or noise == 0:
                input1 = torch.FloatTensor(input1).to(device)
                input2 = torch.FloatTensor(input2).to(device)
            else:
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input1.shape))
                input1 = torch.FloatTensor(input1 + noise_vec).to(device)
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input2.shape))
                input2 = torch.FloatTensor(input2 + noise_vec).to(device)

            q_instance, q_cluster, k_instance, k_cluster = model(input1, input2)

            pseudo_label, centers, dis = clustering(feature=q_instance.detach(),
                                                  cluster_num=cluster_number,
                                                  device=device)
            kmeans_pseudo_adj = high_confidence_adj(pseudo_label, dis, k, device)

            loss_instance, graph, knn_graph = criterion_rgc(q_instance,
                                     k_instance,
                                     n_neighbors=n_neighbors,
                                     kmeans_pseudo_adj=kmeans_pseudo_adj,
                                     temperature=temperature,
                                     dataset_name=log_name,
                                     epoch=epoch,
                                     logger=logger,
                                    save_adj_flag=save_adj_flag)
            save_adj_flag = 0
            
            loss_instance_ += loss_instance.item()

            features_cluster = torch.cat(
                [q_cluster.t().unsqueeze(1),
                 k_cluster.t().unsqueeze(1)],
                dim=1)
            loss_cluster = criterion_cluster(features_cluster)
            loss_cluster_ += loss_cluster.item()

            q_cluster = q_cluster.to(device)
            q_instance = q_instance.to(device)
            k_cluster = k_cluster.to(device)
            k_instance = k_instance.to(device)
            q_prototype = F.normalize(torch.mm(q_cluster.t(), q_instance), dim=1, p=2)
            k_prototype = F.normalize(torch.mm(k_cluster.t(), k_instance), dim=1, p=2)
            features_prototype = torch.cat([q_prototype.unsqueeze(1),
                                            k_prototype.unsqueeze(1)],
                                           dim=1)
            loss_prototype = criterion_prototype(features_prototype)
            loss_prototype_ += loss_prototype.item()

            loss = 0
            if lambda_i > 0:
                loss += lambda_i * loss_instance
            if lambda_c > 0:
                loss += lambda_c * loss_cluster
            if lambda_p > 0:
                loss += lambda_p * loss_prototype

            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1

        
        if evaluate_training and real_label is not None:
            if len(real_label) > 20 and epoch%20 != 0: # 前20轮不评估，快速迭代
                continue

            model.eval()
            if eval_on_cpu and cpu_eval_model is not None:
                eval_device = torch.device('cpu')
                eval_model = cpu_eval_model
                eval_criterion = cpu_eval_criterion_rgc
                # 将最新权重同步到 CPU 评估模型
                base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                cpu_eval_model.load_state_dict(base_model.state_dict())
            else:
                eval_device = torch.device('cuda')
                eval_model = model.to(eval_device)
                eval_criterion = st_loss.RGCLoss(temperature=temperature, device=eval_device)

            if noise is None or noise == 0:
                input1 = torch.FloatTensor(gene_exp).to(eval_device)
                input2 = torch.FloatTensor(gene_exp).to(eval_device)
            else:
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=gene_exp.shape)).to(eval_device)
                input1 = torch.FloatTensor(gene_exp + noise_vec).to(eval_device)
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=gene_exp.shape)).to(eval_device)
                input2 = torch.FloatTensor(gene_exp + noise_vec).to(eval_device)
                       
            with torch.no_grad():
                q_instance, q_cluster, k_instance, k_cluster = eval_model(input1, input2)

            label_pred, centers, dis = clustering(feature=q_instance,
                                          cluster_num=cluster_number,
                                          device=eval_device)
            kmeans_pseudo_adj = high_confidence_adj(label_pred, dis, k, eval_device)
            if eval_on_cpu and device.type == 'cuda':
                kmeans_pseudo_adj = kmeans_pseudo_adj.to(torch.device('cpu'))

            eval_criterion(q_instance,
                        k_instance,
                        n_neighbors=n_neighbors,
                        kmeans_pseudo_adj=kmeans_pseudo_adj,
                        temperature=temperature,
                        dataset_name=log_name,
                        epoch=epoch,
                        save_adj_flag=1,
                        logger=logger
                        )
            
            acc, nmi, ari, f1 = evaluate(y_true=real_label, y_pred=label_pred)
            label_pred_aligned = align_cluster_labels(real_label, label_pred)

            # 记录TensorBoard标量：损失与指标（每个epoch）
            writer.add_scalar("Loss/total", loss_sum, epoch + 1)
            writer.add_scalar("Loss/instance", loss_instance_, epoch + 1)
            writer.add_scalar("Loss/cluster", loss_cluster_, epoch + 1)
            writer.add_scalar("Loss/prototype", loss_prototype_, epoch + 1)
            writer.add_scalar("Metrics/acc", acc, epoch + 1)
            writer.add_scalar("Metrics/nmi", nmi, epoch + 1)
            writer.add_scalar("Metrics/ari", ari, epoch + 1)
            writer.add_scalar("Metrics/f1", f1, epoch + 1)

            features_np = q_instance.cpu().numpy()


            if epoch == 0 or (epoch + 1) % 20 == 0:
                logger.write(f"Epoch {epoch + 1}: Loss: {loss_sum:.5f}, ins: {loss_instance_:.5f}, clu: {loss_cluster_:.5f}, pro: {loss_prototype_:.5f}")
                logger.write(f"     acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, f1: {f1:.4f}")

            if save_fig_flag and (epoch == 0 or (epoch + 1) % 200 == 0):
                print("**********************save tsne figure**********************")
                show_tsne(features_np, label_pred_aligned, dataset, epoch + 1, tsne_perplexity=100, title='pred_label', scores=[acc, nmi, ari, f1])
                show_tsne(features_np, real_label, dataset, epoch + 1, tsne_perplexity=100, title='real_label', scores=[acc, nmi, ari, f1])

            if ari >= best_ari:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1
                best_epoch = epoch

                # Save best model parameters
                save_dir = os.path.join(os.getcwd(), "saved_models", dataset)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "best_model.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'data_aug_state_dict': data_aug_model_weak.state_dict(),
                    'encoder_q_state_dict': encoder_q.state_dict(),
                    'encoder_k_state_dict': encoder_k.state_dict(),
                    'instance_projector_state_dict': instance_projector.state_dict(),
                    'cluster_projector_state_dict': cluster_projector.state_dict(),
                    'epoch': epoch,
                    'ari': ari,
                    'acc': acc,
                    'nmi': nmi,
                    'f1': f1,
                    'm': m,
                    'dropout': dropout,
                    'dropout_weak': resolved_dropout_weak,
                    'dropout_strong': resolved_dropout_strong,
                }, save_path)

                best_features = features_np.copy()
                best_label_pred = np.asarray(label_pred).copy()
                best_label_pred_aligned = np.asarray(label_pred_aligned).copy()

                if save_fig_flag:
                    show_tsne(features_np, label_pred_aligned, dataset, 999, tsne_perplexity=100, title='pred_label', scores=[acc, nmi, ari, f1])
                    show_tsne(features_np, real_label, dataset, 999, tsne_perplexity=100, title='real_label', scores=[acc, nmi, ari, f1])

            
            if epoch == epochs - 1 and save_fig_flag:
                print("**********************save tsne figure**********************")
                show_tsne(features_np, label_pred_aligned, dataset, epoch + 1, tsne_perplexity=100, title='pred_label', scores=[acc, nmi, ari, f1])
                show_tsne(features_np, real_label, dataset, epoch + 1, tsne_perplexity=100, title='real_label', scores=[acc, nmi, ari, f1])

                print("**********************save umap figure**********************")
                from src import utils
                utils.umap_visual(features_np, 
                    label = label_pred_aligned,
                    title='scRGCL pred label', 
                    save_path = os.path.join(os.getcwd(), "umap_figure", dataset+"_pred_label.png"),
                    asw_used=True)
                
                utils.umap_visual(features_np, 
                    label = real_label, 
                    title='scRGCL real label', 
                    save_path = os.path.join(os.getcwd(), "umap_figure", dataset+"_real_label.png"),
                    asw_used=True)

    best_results['accq'] = best_acc
    best_results['nmiq'] = best_nmi
    best_results['ariq'] = best_ari
    best_results['f1q'] = best_f1
    best_results['epochq'] = best_epoch

    if best_features is not None and dataset is not None and real_label is not None:
        out_dir = os.path.join(os.getcwd(), "training_results", dataset, log_name if log_name else dataset)
        metrics = {
            'method': 'scRGCL',
            'n_clusters': int(len(np.unique(best_label_pred))),
            'acc': float(best_acc),
            'nmi': float(best_nmi),
            'ari': float(best_ari),
        }
        predictions = {
            'scrgcl_pred': best_label_pred,
            'scrgcl_pred_aligned': best_label_pred_aligned,
        }
        train_config = {
            'dataset': dataset,
            'epochs': epochs,
            'lr': lr,
            'temperature': temperature,
            'dropout': dropout,
            'dropout_weak': resolved_dropout_weak,
            'dropout_strong': resolved_dropout_strong,
            'layers': list(layers),
            'batch_size': batch_size,
            'm': m,
            'lambda_i': lambda_i,
            'lambda_c': lambda_c,
            'lambda_p': lambda_p,
            'k': k,
            'n_neighbors': n_neighbors,
            'noise': noise,
            'log_name': log_name,
        }
        save_training_outputs(
            out_dir=out_dir,
            dataset=dataset,
            gene_exp=gene_exp,
            features=best_features,
            labels=np.asarray(real_label),
            predictions=predictions,
            metrics=metrics,
            train_config=train_config,
        )
        if logger is not None:
            logger.write(f"Saved training outputs to {out_dir}")

    # 关闭TensorBoard写入器
    writer.close()

    return best_results
