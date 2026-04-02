
import torch
import numpy as np
import scanpy as sc
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random
import os
import sys
import subprocess

import time
import logging
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import h5py
import anndata as ad
import scipy.io as sio
import pandas as pd
from config.opt import args

import umap.umap_ as umap
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

def align_cluster_labels(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {pred: true for true, pred in zip(row_ind, col_ind)}

    return np.array([mapping.get(label, label) for label in y_pred])


def show_tsne(features, labels, dataset_name, epoch, tsne_perplexity=30, title=None, scores=None):
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        early_exaggeration=22,
        learning_rate='auto',
        init='pca',
        random_state=42,
    )
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(6, 5))
    unique_labels = np.unique(labels)
    color_list = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    cmap = ListedColormap(color_list[:max(len(unique_labels), 1)])
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap=cmap,
        s=15,
        vmin=np.min(unique_labels),
        vmax=np.max(unique_labels) if len(unique_labels) > 1 else np.min(unique_labels) + 1,
    )
    plt.colorbar(scatter)
    if scores is not None:
        acc, nmi, ari, f1 = scores
        plt.title(f"ari:{ari:.4f} nmi:{nmi:.4f}")
    else:
        plt.title(f"{dataset_name}")

    save_dir = os.path.join("visualization", "figs", "tsne", dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_epoch{epoch}_perplexity{tsne_perplexity}_{title}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[t-SNE] 图片已保存至 {save_path}")


h5_datasets = []
h5ad_datasets = ['Romanov', 'Pancreas_mouse', '10X_PBMC', 'Quake_10x_Bladder', 'Quake_10x_Spleen',
                 "Quake_Smart-seq2_Lung", "Quake_Smart-seq2_Diaphragm", "Quake_Smart-seq2_Trachea",
                 'Quake_Smart-seq2_Limb_Muscle', 'Quake_10x_Limb_Muscle']
ziscDesk_datasets = ['Muraro', 'Pollen', 'Adam', 'Baron1', 'Baron2', 'Baron3', 'Baron4', 'Baron_mouse1', 'Chung']
merged_dataset = ["human_brain", "Klein", "filtered_mPCTC"]
ca_datasets = ['small_size_4000_Data_Dong2020_Prostate', 'Data_He2021_Prostate', 'small_size_4000_Data_Song2022_Prostate', 'small_size_4000_Data_Chen2021_Prostate']

# Dataset path is at the same level as scRGCL (../dataset)
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'dataset')


def get_dataset(dataset):
    path = DATASET_PATH

    if dataset in h5_datasets:
        data_h5 = h5py.File(f"{path}/h5ad_datasets/{args.name}.h5ad", 'r')
        gene_exp = np.array(data_h5.get('X')) # (4271, 16653) max:777 min:0
        real_label = np.array(data_h5.get('Y')).reshape(-1)# (4271,) [2, 2, 8 ,3, 1]
        gene_exp = preprocess(gene_exp, args.select_gene)
    elif dataset in merged_dataset:
        mat_data = sio.loadmat(f"{path}/merged_dataset/{args.name}.mat")
        gene_exp = mat_data['fea']
        real_label = mat_data['label'].ravel()
        gene_exp = preprocess(gene_exp, args.select_gene)
    elif dataset in h5ad_datasets:
        adata = ad.read_h5ad(f"{path}/h5ad_datasets/{args.name}.h5ad")
        if dataset == "Baron_human":
            celltype = adata.obs['cell_type1']
            adata.X = adata.X.toarray()
        else:
            celltype = adata.obs['celltype']
        if dataset in ["Pancreas_human1", "Pancreas_human2", "Pancreas_human3", "Pancreas_human4", "Pancreas_mouse",
                        "Baron_human"]:
            real_label = celltype.values.codes
        else:
            real_label = celltype.values
        gene_exp = preprocess_h5ad(adata, args.select_gene)
    elif dataset in ziscDesk_datasets:
        ziscDesk_base = os.path.join(DATASET_PATH, 'ziscDesk-single-cell-data-20241011T104023Z-001', 'ziscDesk-single-cell-data')

        gene_exp = np.load(os.path.join(ziscDesk_base, dataset, 'expr.npz'))['arr_0']
        real_label = pd.read_csv(os.path.join(ziscDesk_base, dataset, 'Cells.csv'), index_col=0)['labels'].to_numpy()
        if gene_exp.shape[0] != len(real_label):
            gene_exp = gene_exp.T
        gene_exp = preprocess(gene_exp, args.select_gene)
    elif dataset in ca_datasets:
        ca_base = os.path.join(DATASET_PATH, '3ca', 'dataset', 'prostate', 'Data_Prostate')
        gene_exp = np.load(os.path.join(path, dataset, 'expr.npy'))
        real_label = pd.read_csv(os.path.join(path, dataset, 'y.csv'), index_col=0)['labels'].to_numpy()
        if gene_exp.shape[0] != len(real_label):
            gene_exp = gene_exp.T
        gene_exp = preprocess(gene_exp, args.select_gene)
        

    return gene_exp, real_label


def preprocess(gene_exp, select_genes):
    X = np.ceil(gene_exp).astype(np.float64)
    count_X = X
    print(X.shape, count_X.shape, f"keeping {select_genes} genes")
    adata = sc.AnnData(X)

    
    adata = counts_normalize(adata,
                      copy=True,
                      highly_genes=select_genes,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    X = adata.X.astype(np.float32)
    return X


def preprocess_h5ad(adata, select_genes):
    # X = np.ceil(gene_exp).astype(np.float64)
    # count_X = X
    # print(X.shape, count_X.shape, f"keeping {select_genes} genes")
    # adata = sc.AnnData(X)
    adata.X = np.ceil(adata.X).astype(np.float64)
    adata = counts_normalize(adata,
                             copy=True,
                             highly_genes=select_genes,
                             size_factors=True,
                             normalize_input=True,
                             logtrans_input=True)
    X = adata.X.astype(np.float32)
    return X


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def counts_normalize(adata, copy=True, highly_genes=None, filter_min_counts=True,
              size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    adata.X = np.clip(adata.X, 0, None)  # 将负值裁剪为 0
    
    if adata.X.size < 50e6:
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:

        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)

    print("过滤后数据:", adata.shape)

    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)

    return adata



def get_device(use_cpu):
    if use_cpu is None:
        if os.environ.get('CUDA_VISIBLE_DEVICES', '') == '' or not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


# 动态学习率调整
def adjust_learning_rate(optimizer, epoch, lr):
    p = {
        'epochs': 500,
        'optimizer': 'sgd',
        'optimizer_kwargs':
            {'nesterov': False,
             'weight_decay': 0.0001,
             'momentum': 0.9,
             },
        'scheduler': 'cosine',
        'scheduler_kwargs': {'lr_decay_rate': 0.1},
    }

    new_lr = None

    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        new_lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            new_lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        new_lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return lr


def save_model(name, model, optimizer, current_epoch, pre_epoch):
    # 设置保存目录
    save_dir = os.path.join(os.getcwd(), "save", name)
    
    # 检查并创建保存目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 删除前一个模型文件（如果 pre_epoch 不等于 -1）
    if pre_epoch != -1:
        # pre_path = os.path.join(save_dir, f"checkpoint_{pre_epoch}.tar")
        pre_path = os.path.join(save_dir, f"checkpoint.tar")
        if os.path.exists(pre_path):
            os.remove(pre_path)

    # 设置当前模型保存路径
    # cur_path = os.path.join(save_dir, f"checkpoint_{current_epoch}.tar")
    cur_path = os.path.join(save_dir, f"checkpoint.tar")
    
    # 保存模型状态
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, cur_path)


def cluster_embedding(embedding, cluster_number, real_label, save_pred=True, cluster_methods=None,
                      random_state=20):
    if cluster_methods is None:
        cluster_methods = ["KMeans"]
    result = {"t_clust": time.time()}
    if "KMeans" in cluster_methods:
        # 源代码
        # kmeans = KMeans(n_clusters=cluster_number, init="k-means++", random_state=20)
        # 固定聚类中心
        kmeans = KMeans(n_clusters=cluster_number, init="k-means++", random_state=20)
        
        pred = kmeans.fit_predict(embedding)
        result["centers"] = kmeans.cluster_centers_
        if real_label is not None:
            result[f"ari"] = round(adjusted_rand_score(real_label, pred), 4)
            result[f"nmi"] = round(normalized_mutual_info_score(real_label, pred), 4)
        result["t_k"] = time.time()
        if save_pred:
            result[f"pred"] = pred

    return result


def euclidean_distance(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1 = data1.to(device)
    data2 = data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)
    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # N*N matrix for pairwise euclidean distance
    dis = dis.sum(dim=-1).squeeze()

    return dis

def pseudo_graph(label_pred, device):
    # 根据伪标签生成邻居矩阵
    # 统一数据类型为PyTorch张量，并迁移到指定设备（如GPU）
    if isinstance(label_pred, torch.Tensor):
        pseudo_label = label_pred.clone().to(device)
    elif isinstance(label_pred, np.ndarray):
        pseudo_label = torch.tensor(label_pred.copy()).to(device)
        
    # 构建伪标签邻接矩阵：相同标签的节点间有边（值为1），不同标签为0
    # pseudo_label.unsqueeze(1)将形状从(n,)变为(n,1)，通过广播与原张量比较，得到(n,n)的布尔矩阵
    pseudo_g = (pseudo_label == pseudo_label.unsqueeze(1)).float().to(device)
    # 移除自环（对角线元素设为0）
    diag = torch.diag(pseudo_g) # 提取对角线元素（每个节点与自身的连接，值为1）
    pseudo_g = pseudo_g - torch.diag_embed(diag) # 用对角线元素构造对角矩阵并减去，使对角线为0

    return pseudo_g

def high_confidence_adj(label_pred, dis, k=0.5, device=torch.device('cuda:0')):
    # dis = euclidean_distance(feature, centers, device=device)
    # 取每个样本到最近类中心的距离（形状从(n, cluster_num)变为(n,)）
    dis = torch.min(dis, dim=-1).values
    
    # indices of non-high-confidence label
    # 例如k=0.3时，取70%距离最大的样本为低置信度
    values, indices = torch.topk(dis, int(len(dis) * (1 - k)), largest=True) # 注意！！！实际上可能是39.999，取int变为39
    
    # 先通过伪标签构建基础邻接矩阵
    pseudo_adj = pseudo_graph(label_pred, device)
    # 移除到类中心过远的低置信度样本的所有连接（包括入边和出边）
    pseudo_adj[:, indices] = 0 # 所有节点到低置信度样本的边设为0
    pseudo_adj[indices, :] = 0 # 低置信度样本到所有节点的边设为0

    return pseudo_adj


def show_heat_map(matrix, label=None, title=None, save_path='heatmap.jpg'):
    """
    Show heat map of matrix with dimension of N x N.

    Parameters
    - matrix: N x N matrix.
    - label: 1 x N matrix, for sorting 'matrix' on rows and columns
    - title: title of the heat map.
    - save_path: the save path of the heat map.
    """
    plt.rc('font', family='Times New Roman')

    if isinstance(matrix, torch.Tensor):
        if matrix.device == torch.device('cuda:0'):
            matrix = matrix.detach().cpu()
        matrix = matrix.numpy()
    data = matrix

    # # 将对角线元素置为 0
    # np.fill_diagonal(data, 0)

    # Sorting in x axis and y axis.
    if label is not None:
        idx_sort = np.argsort(label)
        data = data[idx_sort, :]
        data = data[:, idx_sort]

        np.fill_diagonal(data, 0)

    # 设置自定义双色colormap
    # custom_cmap = sns.color_palette(["#f7f6f6", "#67001f"], as_cmap=True)       # 酒红色
    # custom_cmap = sns.color_palette(["#F5F5F5", "#41b9bf"], as_cmap=True)       # 碧蓝色
    # custom_cmap = sns.color_palette(["#F5F5F5", "#3CA9AE"], as_cmap=True)       # 翡翠色

    # 创建自定义渐变颜色 colormap, 均匀分布颜色过渡
    colors = ["#F5F5F5", "#3CA9AE"]
    norm = plt.Normalize(0, 1)
    custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)

    plt.figure(dpi=120)
    sns.heatmap(data=data,
                cmap=custom_cmap,
                norm=norm)  # 矩阵数据集，数据的index和columns分别为heatmap的y轴方向和x轴方向标签

    if title is not None:
        plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    # plt.show()

def evaluate_soft(real_label, label_soft):
    result = {}
    result["soft_ari"] = round(adjusted_rand_score(real_label, label_soft), 4)
    result["soft_nmi"] = round(normalized_mutual_info_score(real_label, label_soft), 4)
    return result


def plot_learning_curves(data_dict, epochs, data_name=None, save_path='./visualization/figs/fig_learning_curves.png'):
    loss_list = data_dict['loss']
    nmi_list = data_dict['nmi']
    ari_list = data_dict['ari']
    x = np.linspace(1, epochs+1, epochs)
    linewidth = 4
    # color_list = ['#1a74b2', '#d31718', '#ff7903', '#942bc7', '#1f9b20']      # blue red orange purple green
    color_list = ['#d31718', '#1a74b2', '#ff7903', '#942bc7', '#9cce00']        # red orange blue purple green

    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(111)

    line_loss, = ax1.plot(x, loss_list, '-', label='Loss', color=color_list[0], linewidth=linewidth)
    ax2 = ax1.twinx()
    line_nmi, = ax2.plot(x, nmi_list, '-', label='NMI', color=color_list[2], linewidth=linewidth)
    line_ari, = ax2.plot(x, ari_list, '-', label='ARI', color=color_list[3], linewidth=linewidth)

    lines = [line_nmi, line_ari]
    ax2.spines['right'].set_color('black')
    ax2.tick_params(axis='y', color='black', labelcolor='black')

    # if data_name == 'acm':
    #     legend_position = (0.975, 0.35)
    # elif data_name == 'dblp':
    #     legend_position = (0.975, 0.28)
    # else:
    #     legend_position = (0.975, 0.30)
    # added these three lines
    line1, label1 = ax1.get_legend_handles_labels()
    line2, label2 = ax2.get_legend_handles_labels()
    ax1.legend(line1 + line2, label1 + label2,
               bbox_to_anchor=(0.95, 0.35),         # 3: (0.95, 0.35)  1: (0.95, 0.32)
               prop={'family': 'Times New Roman', 'size': 40},
               ncol = 2,
               loc='center right'
               )
    ax1.grid(True, linestyle='--', linewidth = 2)
    # ax1.grid(False)
    ax1.set_xlabel("Epoch", fontdict={'family': 'Times New Roman', 'size': 60})
    ax1.set_ylabel("Loss Value", fontdict={'family': 'Times New Roman', 'size': 60})
    ax2.set_ylabel("Score", fontdict={'family': 'Times New Roman', 'size': 60})

    ax1.tick_params(axis='x', labelsize=40)     # 设置 ax1 的 xticks 和 yticks 字体大小
    ax1.tick_params(axis='y', labelsize=40)
    ax2.tick_params(axis='y', labelsize=40)     # 设置 ax2 的 yticks 字体大小

    ax2.set_ylim([0, 1.0])

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)

    if data_name is not None:
        plt.title(data_name)

    save_name = f"{save_path}"
    plt.savefig(save_name, bbox_inches='tight')
    # plt.close()
    plt.show()

def get_logger(root = './training_logs', filename = None):
    """
    Get logger.

    Parameters
    - root: str. Root directory of log files.
    - filename: str, Optional. The name of log files.

    return
    - logger: Logger
    """
    # logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt = '[%(asctime)s]%(levelname)s: %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S')

    if filename is not None:
        """Save logs as files"""
        if not os.path.exists(root):
            os.makedirs(root)

        # mode = 'w', overwriting the previous content, 'a', appended to previous file.
        fh = logging.FileHandler(os.path.join(root, filename), "a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    """Print logs at terminal"""
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class NoNewlineStreamHandler(logging.StreamHandler):
    """不自动加换行的终端 Handler"""
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

class NoNewlineFileHandler(logging.FileHandler):
    """不自动加换行的文件 Handler"""
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

class DualLogger:
    def __init__(self, root='logs', filename=None, show_time=True, time_format="%Y-%m-%d %H:%M:%S"):
        """
        :param filename: 日志文件路径
        :param show_time: 是否显示时间戳
        :param time_format: 时间戳格式
        """
        self.show_time = show_time
        self.time_format = time_format
        self.new_line = True  # 记录是否处于新行开头

        self.logger = logging.getLogger("mylogger")
        self.logger.propagate = False  # 禁止日志冒泡, 只打印自己的日志
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            # 只显示 message，时间戳我们自己加
            formatter = logging.Formatter('%(message)s')

            # 终端输出
            ch = NoNewlineStreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            # 文件输出（追加模式）
            if filename is not None:
                if not os.path.exists(root):
                    os.makedirs(root)
                fh = NoNewlineFileHandler(os.path.join(root, filename),
                                          mode="a",
                                          encoding="utf-8")
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

    def _add_timestamp_if_needed(self, message):
        if self.show_time and self.new_line:
            ts = datetime.now().strftime(self.time_format)
            return f"[{ts}] {message}"
        return message

    def write(self, message="", end="\n"):
        msg = str(message)
        msg = self._add_timestamp_if_needed(msg)
        self.logger.info(msg + end)
        self.new_line = (end == "\n")  # 如果结束符是换行，下一次输出才加时间戳

    def flush(self):
        """兼容 print 的 file 接口"""
        pass

def umap_visual(data, title=None, save_path=None, label=None, asw_used=None):
    reducer = umap.UMAP(random_state=4132231)
    embedding = reducer.fit_transform(data)
    n_lables = len(set(label)) + 1
    mean_silhouette_score = silhouette_score(data, label)
    # ARI = calcu_ARI(label, true_label)
    # NMI = normalized_mutual_info_score(true_label, label)
    xlim_l = int(embedding[:, 0].min()) - 2
    xlim_r = int(embedding[:, 0].max()) + 2
    ylim_d = int(embedding[:, 1].min()) - 2
    ylim_u = int(embedding[:, 1].max()) + 2
    plt.figure(figsize = (6,4), dpi=200)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(n_lables)).set_ticks(np.arange(n_lables))
    plt.xlim((xlim_l, xlim_r))
    plt.ylim((ylim_d, ylim_u))
    plt.title('UMAP projection of the {0}'.format(title))
    if asw_used is not None:
        plt.text(xlim_r-2, ylim_d+1.5, "ASW=%.3f"%(mean_silhouette_score),
                  ha="right",)
    plt.grid(False)
    plt.show()
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

import subprocess
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

    return best_gpu