import os
import time
import numpy as np
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
                       plot_learning_curves)
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from sklearn.manifold import TSNE


def show_tsne(features, labels, dataset_name, epoch, tsne_perplexity=30, title=None, scores=None):
    """
    绘制并保存 t-SNE 可视化图
    :param features: 特征矩阵，shape=(n_samples, n_features)
    :param labels: 真实标签或聚类标签，shape=(n_samples,)
    :param dataset_name: 数据集名称，用于命名图片
    :param epoch: 当前 epoch，用于命名图片
    :param save_dir: 图片保存目录
    """
    # os.makedirs(save_dir, exist_ok=True)
    # tsne = TSNE(n_components=2, random_state=42)
    tsne = TSNE(
        n_components=2, 
        perplexity=tsne_perplexity,        # 建议：调整为 50, 80 或 100。越大，簇分得越开，但也越慢。
        early_exaggeration=22,# 默认是12，如果觉得簇分得不够开，可以尝试改大到 20
        learning_rate='auto', # 自动学习率，通常效果最好
        # n_iter=1000,          # 迭代次数，确保收敛
        init='pca',           # 关键：使用 PCA 初始化，保持全局结构，避免“C型”扭曲
        random_state=42,
    )
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', s=15)
    plt.colorbar(scatter)
    # socre
    if scores is not None:
        acc, nmi, ari, f1 = scores
        plt.title(f"ari:{ari:.4f} nmi:{nmi:.4f}")
    else:
        plt.title(f"{dataset_name}")

    # Use relative path based on current working directory
    save_dir = os.path.join("visualization", "figs", "tsne", dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_epoch{epoch}_perplexity{tsne_perplexity}_{title}.png")
    # save_path = os.path.join(save_dir, "tmp-perplexity100")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[t-SNE] 图片已保存至 {save_path}")


def run(gene_exp, cluster_number, dataset, real_label, epochs, lr, temperature, dropout, layers, batch_size, m,
        save_pred=True, noise=None, use_cpu=None, cluster_methods=None, logger=None):
    if cluster_methods is None:
        cluster_methods = []
    results = {}

    start = time.time()
    embedding, best_model = train_model(gene_exp=gene_exp, cluster_number=cluster_number, real_label=real_label,
                                        epochs=epochs, lr=lr, temperature=temperature,
                                        dropout=dropout, layers=layers, batch_size=batch_size,
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
                 logger=None, dataset=None, save_fig_flag=False, log_name=None):
    device = get_device(use_cpu)

    # 初始化TensorBoard写入器
    tb_log_dir = os.path.join("runs", log_name if log_name else f"{dataset}_{time.strftime('%Y%m%d%H%M%S')}")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    dims = np.concatenate([[gene_exp.shape[1]], layers])
    data_aug_model = ScRGCL.DataAug(dropout=dropout)
    encoder_q = ScRGCL.BaseEncoder(dims)
    encoder_k = ScRGCL.BaseEncoder(dims)
    instance_projector = ScRGCL.MLP(layers[2], layers[2] + layers[3], layers[2] + layers[3])
    cluster_projector = ScRGCL.MLP(layers[2], layers[3], cluster_number)
    instance_dim = layers[2] + layers[3]
    model = ScRGCL.ScRGCL(encoder_q, encoder_k, instance_projector, cluster_projector, cluster_number, instance_dim, m=m)
    data_aug_model.to(device)
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion_rgc = st_loss.RGCLoss(temperature=temperature, device=device)
    criterion_cluster = st_loss.ClusterLoss(cluster_number, temperature=temperature)
    criterion_prototype = st_loss.PrototypeLoss(temperature=temperature)

    max_value_q, max_value_k, max_value_fu = -1, -1, -1
    best_acc, best_nmi, best_ari, best_f1, best_epoch = 0.0, 0.0, 0.0, 0.0, 0
    soft_res_best = None

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
            input1 = data_aug_model(torch.FloatTensor(c_inp))
            input2 = data_aug_model(torch.FloatTensor(c_inp))

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
            model.eval()
            # data_aug_model.eval()
            
            if noise is None or noise == 0:
                input1 = torch.FloatTensor(gene_exp).to(device)
                input2 = torch.FloatTensor(gene_exp).to(device)
            else:
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input1.shape))
                input1 = torch.FloatTensor(input1 + noise_vec).to(device)
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input2.shape))
                input2 = torch.FloatTensor(input2 + noise_vec).to(device)
                       
            with torch.no_grad():
                q_instance, q_cluster, k_instance, k_cluster = model(input1, input2)

                # q_instance, _, k_instance, _ = model(torch.FloatTensor(gene_exp).to(device), None)
            label_pred, centers, dis = clustering(feature=q_instance,
                                          cluster_num=cluster_number,
                                          device=device)
            kmeans_pseudo_adj = high_confidence_adj(label_pred, dis, k, device)
            criterion_rgc(q_instance,
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
                show_tsne(features_np, label_pred, dataset, epoch + 1, tsne_perplexity=100, title='pred_label', scores=[acc, nmi, ari, f1])
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
                    'data_aug_state_dict': data_aug_model.state_dict(),
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
                }, save_path)

                if save_fig_flag:
                    print("**********************save umap figure**********************")
                    from src import utils
                    utils.umap_visual(features_np,
                        label = label_pred,
                        title='scRGCL pred label',
                        save_path = os.path.join(os.getcwd(), "umap_figure", dataset+"_pred_label.png"),
                        asw_used=True)

                    utils.umap_visual(features_np,
                        label = real_label,
                        title='scRGCL real label',
                        save_path = os.path.join(os.getcwd(), "umap_figure", dataset+"_real_label.png"),
                        asw_used=True)
            
            if epoch == epochs - 1 and save_fig_flag:
                print("**********************save tsne figure**********************")
                show_tsne(features_np, label_pred, dataset, epoch + 1, tsne_perplexity=100, title='pred_label', scores=[acc, nmi, ari, f1])
                show_tsne(features_np, real_label, dataset, epoch + 1, tsne_perplexity=100, title='real_label', scores=[acc, nmi, ari, f1])

                print("**********************save umap figure**********************")
                from src import utils
                utils.umap_visual(features_np, 
                    label = label_pred, 
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

    # 关闭TensorBoard写入器
    writer.close()

    return best_results