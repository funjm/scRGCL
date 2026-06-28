import os
import ScCCL
import opt
import time
import numpy as np
import torch
import st_loss
from utils import get_device, adjust_learning_rate, save_model, cluster_embedding


def run(gene_exp, cluster_number, dataset, real_label, epochs, lr, temperature, dropout, layers, batch_size, m,
        save_pred=True, noise=None, use_cpu=None, cluster_methods=None):
    if cluster_methods is None:
        cluster_methods = []
    results = {}

    start = time.time()
    embedding, best_model = train_model(gene_exp=gene_exp, cluster_number=cluster_number, real_label=real_label,
                                        epochs=epochs, lr=lr, temperature=temperature,
                                        dropout=dropout, layers=layers, batch_size=batch_size,
                                        m=m, save_pred=save_pred, noise=noise, use_cpu=use_cpu)

    if save_pred:
        results[f"features"] = embedding
        results[f"max_epoch"] = best_model
    elapsed = time.time() - start
    res_eval = cluster_embedding(embedding, cluster_number, real_label, save_pred=save_pred,
                                 cluster_methods=cluster_methods)
    results = {**results, **res_eval, "dataset": dataset, "time": elapsed}

    return results


def train_model(gene_exp, cluster_number, real_label, epochs, lr,
                temperature, dropout, layers, batch_size, m,
                save_pred=False, noise=None, use_cpu=None, evaluate_training=True):
    device = get_device(use_cpu)

    dims = np.concatenate([[gene_exp.shape[1]], layers])
    data_aug_model = ScCCL.DataAug(dropout=dropout)
    encoder_q = ScCCL.BaseEncoder(dims)
    encoder_k = ScCCL.BaseEncoder(dims)
    instance_projector = ScCCL.MLP(layers[2], layers[2] + layers[3], layers[2] + layers[3])
    cluster_projector = ScCCL.MLP(layers[2], layers[3], cluster_number)
    model = ScCCL.ScCCL(encoder_q, encoder_k, instance_projector, cluster_projector, cluster_number, m=m)
    data_aug_model.to(device)
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion_instance = st_loss.InstanceLoss(temperature=temperature)
    criterion_cluster = st_loss.ClusterLoss(cluster_number, temperature=temperature)

    max_value, best_model = -1, -1

    idx = np.arange(len(gene_exp))
    for epoch in range(epochs):
        model.train()
        adjust_learning_rate(optimizer, epoch, lr)
        np.random.shuffle(idx)
        loss_instance_ = 0
        loss_cluster_ = 0
        for pre_index in range(len(gene_exp) // batch_size + 1):
            c_idx = np.arange(pre_index * batch_size,
                              min(len(gene_exp), (pre_index + 1) * batch_size))
            if len(c_idx) == 0:
                continue
            c_idx = idx[c_idx]
            c_inp = gene_exp[c_idx]
            input1 = data_aug_model(torch.FloatTensor(c_inp))
            input2 = data_aug_model(torch.FloatTensor(c_inp))

            if noise is None or noise == 0:
                input1 = torch.FloatTensor(input1).to(device)
                input2 = torch.FloatTensor(input2).to(device)
            else:
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input1.shape))
                input1 = torch.FloatTensor(input1 + noise_vec).to(device)
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input2.shape))
                input2 = torch.FloatTensor(input2 + noise_vec).to(device)
            q_instance, q_cluster, k_instance, k_cluster = model(input1, input2)

            features_instance = torch.cat(
                [q_instance.unsqueeze(1),
                 k_instance.unsqueeze(1)],
                dim=1)
            features_cluster = torch.cat(
                [q_cluster.t().unsqueeze(1),
                 k_cluster.t().unsqueeze(1)],
                dim=1)
            loss_instance = criterion_instance(features_instance)
            loss_cluster = criterion_cluster(features_cluster)
            loss = loss_instance + loss_cluster
            loss_instance_ += loss_instance.item()
            loss_cluster_ += loss_cluster.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if evaluate_training and real_label is not None:
            model.eval()
            with torch.no_grad():
                q_instance, _, _, _ = model(torch.FloatTensor(gene_exp).to(device), None)
                features = q_instance.detach().cpu().numpy()
            res = cluster_embedding(features, cluster_number, real_label, save_pred=save_pred)
            print(
                f"Epoch {epoch}: Loss: {loss_instance_ + loss_cluster_}, ARI: {res['ari']}, "
                f"NMI: {res['nmi']} "
            )

            if res['ari'] + res['nmi'] >= max_value:
                max_value = res['ari'] + res['nmi']
                save_model(opt.args.name, model, optimizer, epoch, best_model)
                best_model = epoch

    model.eval()
    PWD = '/disk/fanjunming/home/workspace/bioinfo/scRGCL/baseline_methods/ScCCL'

    model_fp = os.path.join(PWD, "save", opt.args.name, "checkpoint_{}.tar".format(best_model))
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)
    with torch.no_grad():
        q_instance, _, _, _ = model(torch.FloatTensor(gene_exp).to(device), None)
        features = q_instance.detach().cpu().numpy()

    return features, best_model