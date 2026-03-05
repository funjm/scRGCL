
import numpy
import torch
import random


random.seed(100)
torch.manual_seed(42)


adj = torch.randint(0, 2, (5, 5))
print(adj)
mask = adj == 0
print(mask)
adj1 = torch.where(mask, adj, 2)
print(adj1)
print(mask.float())

pos = torch.tensor([10, 20, 30, 40, 50])
neg = torch.tensor([5, 4, 6, 10, 25])
print(pos / neg)            # tensor([2., 5., 5., 4., 2.])
print((pos / neg).mean())   # tensor(3.6000)


# def compute_knn(sim_matrix, k=3, device=torch.device('cuda')):
#     diag = torch.diag(sim_matrix)
#     sim_matrix = sim_matrix - torch.diag_embed(diag)
#     values, indices = torch.topk(sim_matrix, k, dim=-1, largest=True)
#     adj = torch.zeros(sim_matrix.size(), device=device)
#     # adj[indices] = 1
#     adj = adj.scatter_(1, indices, 1)  # 在每行中按照列索引将值 scatter 到目标张量中
#     adj = adj + adj.t()
#     adj = torch.clamp(adj, max=1)
#
#     return adj
#
# def compute_laplacian(adj, device):
#     degree_arr = torch.sum(adj, dim=-1)     # tensor([6., 6., 5., 6., 8., 7., 6., 8., 6., 4.])
#     degree_arr = 1 / degree_arr.pow(0.5)
#     degree_matrx = torch.diag(degree_arr).to(device)
#     # degree_matrx_hat = 1 / degree_matrx.pow(0.5)
#     L = torch.eye(adj.size()[0]).to(device) - torch.mm(torch.mm(degree_matrx, adj), degree_matrx)
#
#     return L

# device = torch.device('cpu')
# matrix1 = torch.randint(0, 9, (10, 10))
# print(f"matrix1: {matrix1.size()}\n{matrix1}")
# knn_graph = compute_knn(matrix1, k=4, device=device)
# print(f"KNN: {knn_graph.size()}\n{knn_graph}")
# lap = compute_laplacian(knn_graph, device)
# print(f"lap: {lap.size()}\n{lap}")
