import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

EPS = 1e-8


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, features_cluster):
        c_i, c_j = torch.unbind(features_cluster, dim=1)
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        N = 2 * self.class_num
        c = features_cluster
        cluster_feature = torch.cat(torch.unbind(c, dim=1), dim=0)
        sim = torch.matmul(cluster_feature, cluster_feature.T) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss


class InstanceLoss(nn.Module):
    def __init__(self, temperature):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, features):

        device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)  # (200, 2, 100)

        self.batch_size = features.shape[0]
        self.mask = self.mask_correlated_samples(self.batch_size).to(device)  # (400, 400)
        N = 2 * self.batch_size
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (400, 100)

        sim = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature  # (400, 400)

        sim_i_j = torch.diag(sim, self.batch_size)  # (200, )
        sim_j_i = torch.diag(sim, -self.batch_size)  # (200, )
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  # (400, 1)
        negative_samples = sim[self.mask].reshape(N, -1)  # (400, 398)

        labels = torch.zeros(N).to(positive_samples.device).long()  # (400, )
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # (400, 399)

        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class PrototypeLoss(nn.Module):
    def __init__(self, temperature):
        super(PrototypeLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, features):

        device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        self.batch_size = features.shape[0]     # 4
        self.mask = self.mask_correlated_samples(self.batch_size).to(device)    # (8, 8)
        N = 2 * self.batch_size     # 8
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)      #(8, 100)

        sim = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature     # (8, 8)

        sim_i_j = torch.diag(sim, self.batch_size)      # (4,)
        sim_j_i = torch.diag(sim, -self.batch_size)     # (4,)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)   # (8, 1)
        negative_samples = sim[self.mask].reshape(N, -1)        # (8, 6)

        labels = torch.zeros(N).to(positive_samples.device).long()      # (8,)
        logits = torch.cat((positive_samples, negative_samples), dim=1) # (8, 7)

        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class RGCLoss(nn.Module):
    def __init__(self, temperature, device=torch.device('cuda')):
        super(RGCLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def cosine_sim(self, data1, data2):
        if data1.device is not torch.device('cuda'):
            data1, data2 = data1.to(self.device), data2.to(self.device)
        data1_norm = data1 / torch.norm(data1, dim=1, keepdim=True)
        data2_norm = data2 / torch.norm(data2, dim=1, keepdim=True)
        cos_sim = torch.mm(data1_norm, data2_norm.t())

        return cos_sim

    def compute_knn(self, sim_matrix, n_neighbors=3):
        diag = torch.diag(sim_matrix)# 提取相似度矩阵的对角线元素（每个样本与自身的相似度）
        sim_matrix = sim_matrix - torch.diag_embed(diag)# 用对角线元素构造对角矩阵，从原矩阵中减去，即把对角线清零
        values, indices = torch.topk(sim_matrix, n_neighbors, dim=-1, largest=True)
        adj = torch.zeros(sim_matrix.size(), device=self.device)
        # adj[indices] = 1
        adj = adj.scatter_(1, indices, 1)  # 在每行中按照列索引将值 scatter 到目标张量中
        adj = adj + adj.t()
        adj = torch.clamp(adj, max=1)

        return adj

    def compute_laplacian(self, adj):
        degree_arr = torch.sum(adj, dim=1)
        degree_arr = 1 / degree_arr.pow(0.5)
        degree_matrix = torch.diag(degree_arr)
        L = torch.eye(adj.size()[0]).to(self.device) - torch.mm(torch.mm(degree_matrix, adj), degree_matrix)

        return L

    def forward(self, q_instance, k_instance, n_neighbors, kmeans_pseudo_adj, temperature=1.0, dataset_name=None, epoch=None, save_adj_flag=0, logger=None):
        sim_matrix = self.cosine_sim(q_instance, k_instance)
        topk_sim_matrix = self.compute_knn(sim_matrix.clone().detach(), n_neighbors)
        graph = topk_sim_matrix + kmeans_pseudo_adj
        # 保存邻居信息
        # if save_adj_flag == 1:  
                          # 1
            # save_dir = os.path.join("adj", dataset_name)
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = os.path.join(save_dir, f"{dataset_name}_topk_sim_matrix_epoch_{epoch}.npy")
            # np.save(save_path, topk_sim_matrix.cpu().numpy())
            
            # save_path = os.path.join(save_dir, f"{dataset_name}_kmeans_pseudo_adj_epoch_{epoch}.npy")
            # np.save(save_path, kmeans_pseudo_adj.cpu().numpy())
            

        L = self.compute_laplacian(graph)
        L = - (L - torch.diag_embed(torch.diag(L)))
        # L.fill_diagonal_(0)  # 原地操作，直接修改 data1

        sim_matrix_exp = torch.exp(sim_matrix / temperature)
        pos_sim = torch.diag(sim_matrix_exp)
        # (P + L*M) / N
        loss = (pos_sim + torch.sum(sim_matrix_exp * L, dim=-1)) / (torch.sum(sim_matrix_exp, dim=-1))
        loss = - torch.log(loss).mean()

        # return loss
        return loss, graph, topk_sim_matrix




def clustering_loss(emb1, emb2, emb_fu, centers, alpha=1.0, device = torch.device('cuda')):

    y_wave_1 = soft_assignment(emb1, centers, alpha=alpha)
    y_wave_2 = soft_assignment(emb2, centers, alpha=alpha)
    y_wave_fu = soft_assignment(emb_fu, centers, alpha=alpha)
    target_fu = target_distribution(y_wave_fu)

    loss_clustering = F.kl_div(y_wave_1.log(), target_fu, reduction='batchmean')
    loss_clustering += F.kl_div(y_wave_2.log(), target_fu, reduction='batchmean')


    return loss_clustering

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def soft_assignment(inputs, centers, alpha=1.0):
    # q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(inputs, axis=1) - centers), axis=2) / alpha))
    # q **= (alpha + 1.0) / 2.0
    # q = np.transpose(np.transpose(q) / np.sum(q, axis=1))

    q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(inputs, dim=1) - centers), dim=2) / alpha))
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, dim=1)).t()

    return q




def NCL_loss(sim_inter_matrix, z1, z2, adj, tau=0.1, device=torch.device('cuda:0')):
    sim_intra_z1 = torch.exp(cosine_sim(z1, z1, device) / tau)
    sim_intra_z2 = torch.exp(cosine_sim(z2, z2, device) / tau)
    diag1 = torch.diag(sim_intra_z1)
    sim_intra_z1 = sim_intra_z1 - torch.diag_embed(diag1)
    diag2 = torch.diag(sim_intra_z2)
    sim_intra_z2 = sim_intra_z2 - torch.diag_embed(diag2)
    sim_inter_view = torch.exp(sim_inter_matrix / tau)

    pos_sim = torch.diag(sim_inter_view) + torch.sum(sim_inter_view * adj, dim=-1) \
              + torch.sum(sim_intra_z1 * adj, dim=-1) + torch.sum(sim_intra_z2 * adj, dim=-1)
    # (P + nu * M) / (N + M)
    loss_hsc = pos_sim / (
                torch.sum(sim_inter_view, dim=-1) + torch.sum(sim_intra_z1, dim=-1) + torch.sum(sim_intra_z2, dim=-1))
    # loss_hsc = (pos_sim) / (torch.sum(sim_inter_view, dim=-1))      # setting1
    # loss_hsc = (pos_sim + torch.sum(sim_inter_view * adj, dim=-1)) / (torch.sum(sim_inter_view, dim=-1))  # setting2

    loss_hsc = - torch.log(loss_hsc).mean()

    return loss_hsc


def MSE_loss(sim_matrix, adj):
    mse_loss = F.mse_loss(adj, sim_matrix)

    return mse_loss


def cosine_sim(data1, data2, device=torch.device('cuda:0')):
    A, B = data1.to(device), data2.to(device)
    # unitization eg.[3, 4] -> [3/sqrt(9 + 16), 4/sqrt(9 + 16)] = [3/5, 4/5]
    A_norm = A / torch.norm(A, dim=1, keepdim=True)
    B_norm = B / torch.norm(B, dim=1, keepdim=True)
    cos_sim = torch.mm(A_norm, B_norm.t())

    return cos_sim


def compute_knn(sim_matrix, n_neighbors=3, device=torch.device('cuda')):
    diag = torch.diag(sim_matrix)
    sim_matrix = sim_matrix - torch.diag_embed(diag)
    values, indices = torch.topk(sim_matrix, n_neighbors, dim=-1, largest=True)
    adj = torch.zeros(sim_matrix.size(), device=device)
    # adj[indices] = 1
    adj = adj.scatter_(1, indices, 1)  # 标记 “某样本是另一样本的 K 近邻”。
    adj = adj + adj.t()
    adj = torch.clamp(adj, max=1)

    return adj


def compute_laplacian(adj, device):
    degree_arr = torch.sum(adj, dim=1)
    degree_arr = 1 / degree_arr.pow(0.5)
    degree_matrix = torch.diag(degree_arr)
    L = torch.eye(adj.size()[0]).to(device) - torch.mm(torch.mm(degree_matrix, adj), degree_matrix)

    return L


def compute_homo_ratio(edge_index, label):
    homo_info = {}
    # if issparse(adj):
    #     adj = adj.toarray()
    # if edge_index is None:
    #     edge_index = np.where(adj > 0)
    #     edge_index = np.concatenate((np.expand_dims(edge_index[0], axis=0),
    #                                  np.expand_dims(edge_index[1], axis=0)), axis=0)
    homo_info['n_edges'] = edge_index.shape[1]
    # homo_info['adj_sum'] = np.sum(adj)

    same_label = 0
    for i in range(edge_index.shape[1]):
        if label[edge_index[0, i]] == label[edge_index[1, i]]:
            same_label += 1
    homo_ratio = same_label / edge_index.shape[1]
    homo_info['homo_ratio'] = homo_ratio

    edge_list = []
    true_neighbor_list = []
    ratio_list = []
    last_node = edge_index[0, 0]
    true_nb_num = 0
    n_edge = 0
    for i in range(edge_index.shape[1]):
        curr_node = edge_index[0, i]
        if curr_node != last_node:
            ratio = true_nb_num / n_edge
            edge_list.append(n_edge)
            true_neighbor_list.append(true_nb_num)
            ratio_list.append(ratio)
            n_edge = 0
            true_nb_num = 0
            last_node = curr_node
            # for the last node, e.g. [[ ... 8 8 9], [ ... 7 9 10]]
            if i == edge_index.shape[1] - 1:
                n_edge += 1
                curr_neighbor = edge_index[1, i]
                if label[curr_node] == label[curr_neighbor]:
                    true_nb_num += 1
                ratio = true_nb_num / n_edge
                ratio_list.append(ratio)
                edge_list.append(n_edge)
                true_neighbor_list.append(true_nb_num)
                break
        # for each node
        n_edge += 1
        curr_neighbor = edge_index[1, i]
        if label[curr_node] == label[curr_neighbor]:
            true_nb_num += 1
        # for the last node, e.g. [[ ... 9 9 9], [ ... 7 8 10]]
        if curr_node == last_node and i == edge_index.shape[1] - 1:
            ratio = true_nb_num / n_edge
            ratio_list.append(ratio)
            edge_list.append(n_edge)
            true_neighbor_list.append(true_nb_num)

    neighbor_homo_ratio = np.mean(ratio_list)
    homo_info['neighbor_homo_ratio'] = neighbor_homo_ratio
    homo_info['n_edge_list'] = edge_list
    homo_info['true_neighbor_list'] = true_neighbor_list
    homo_info['ratio_list'] = ratio_list

    return homo_info
