#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test.py: Load saved model and evaluate on a dataset
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import ScRGCL
import st_loss
from clustering import clustering
from evaluation import evaluate
from utils import get_device, high_confidence_adj, get_dataset, set_random_seed


def load_model(model_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)

    # Extract dimensions from saved state dict
    # Infer dimensions from cluster_projector layer
    cluster_proj_weights = checkpoint['cluster_projector_state_dict']['encoder.2.weight']
    cluster_number = cluster_proj_weights.shape[0]
    mlp_dim = cluster_proj_weights.shape[1]

    instance_proj_weights = checkpoint['instance_projector_state_dict']['encoder.2.weight']
    instance_dim = instance_proj_weights.shape[1] - instance_proj_weights.shape[0]  # contrastive_dim -> mlp_dim -> dim

    encoder_q_weights = checkpoint['encoder_q_state_dict']['encoder.2.weight']
    input_dim = encoder_q_weights.shape[1]
    enc_dims = [
        input_dim,
        checkpoint['encoder_q_state_dict']['encoder.0.weight'].shape[0],
        checkpoint['encoder_q_state_dict']['encoder.3.weight'].shape[0],
        checkpoint['encoder_q_state_dict']['encoder.6.weight'].shape[0],
    ]

    return checkpoint, cluster_number, mlp_dim, instance_dim, enc_dims


def test_model(dataset, model_path=None, use_cpu=None, temperature=0.07, dropout=0.9,
               layers=None, cluster_number=None, m=0.5, k=0.5, n_neighbors=3,
               seed=100):
    """
    Load a saved model and evaluate on the given dataset.

    Parameters:
    -----------
    dataset : str
        Name of the dataset to test on
    model_path : str, optional
        Path to saved model checkpoint. If None, uses default path: saved_models/{dataset}/best_model.pt
    use_cpu : bool, optional
        Force CPU usage
    temperature : float
        Temperature parameter for loss
    dropout : float
        Dropout rate for data augmentation
    layers : list, optional
        Network architecture [enc_1, enc_2, enc_3, mlp_dim]
    cluster_number : int, optional
        Number of clusters. If None, inferred from data
    m : float
        Momentum coefficient
    k : float
        High-confidence adjacency threshold
    n_neighbors : int
        Number of k-NN neighbors
    seed : int
        Random seed

    Returns:
    --------
    dict : Evaluation metrics (acc, nmi, ari, f1)
    """
    device = get_device(use_cpu)
    set_random_seed(seed)

    # Load data
    gene_exp, real_label = get_dataset(dataset)
    if cluster_number is None:
        cluster_number = np.unique(real_label).shape[0]

    if layers is None:
        layers = [200, 40, 60, 40]  # default architecture

    if model_path is None:
        model_path = os.path.join(os.getcwd(), "saved_models", dataset, "best_model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Use m from checkpoint if available
    m = checkpoint.get('m', m)

    # Build model architecture (same as training)
    dims = np.concatenate([[gene_exp.shape[1]], layers])
    data_aug_model = ScRGCL.DataAug(dropout=dropout)
    encoder_q = ScRGCL.BaseEncoder(dims)
    encoder_k = ScRGCL.BaseEncoder(dims)
    instance_projector = ScRGCL.MLP(layers[2], layers[2] + layers[3], layers[2] + layers[3])
    cluster_projector = ScRGCL.MLP(layers[2], layers[3], cluster_number)
    instance_dim = layers[2] + layers[3]
    model = ScRGCL.ScRGCL(encoder_q, encoder_k, instance_projector, cluster_projector,
                          cluster_number, instance_dim, m=m)

    # Load saved state
    data_aug_model.load_state_dict(checkpoint['data_aug_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])

    data_aug_model.to(device)
    model.to(device)
    data_aug_model.eval()
    model.eval()

    # Evaluate
    input1 = torch.FloatTensor(gene_exp).to(device)
    input2 = torch.FloatTensor(gene_exp).to(device)

    with torch.no_grad():
        q_instance, q_cluster, k_instance, k_cluster = model(input1, input2)

    # Clustering
    label_pred, centers, dis = clustering(feature=q_instance, cluster_num=cluster_number, device=device)

    # Evaluate
    acc, nmi, ari, f1 = evaluate(y_true=real_label, y_pred=label_pred)

    results = {
        'acc': acc,
        'nmi': nmi,
        'ari': ari,
        'f1': f1,
        'dataset': dataset,
        'model_epoch': checkpoint.get('epoch', -1),
        'model_ari': checkpoint.get('ari', -1),
    }

    # Print results
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset}")
    print(f"Model trained for {checkpoint.get('epoch', -1)} epochs (ARI: {checkpoint.get('ari', -1):.4f})")
    print(f"{'='*50}")
    print(f"ACC: {acc:.4f}")
    print(f"NMI: {nmi:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"F1:  {f1:.4f}")
    print(f"{'='*50}")

    return results


def main():
    parser = argparse.ArgumentParser(description='ScRGCL Test', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, required=True, help='Dataset name')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA')
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature')
    parser.add_argument('--dropout', type=float, default=0.9, help='Dropout rate')
    parser.add_argument('--m', type=float, default=0.5, help='Momentum coefficient')
    parser.add_argument('--k', type=float, default=0.5, help='High-confidence threshold')
    parser.add_argument('--n_neighbors', type=int, default=3, help='Number of neighbors')
    parser.add_argument('--cluster_number', type=int, default=None, help='Number of clusters')

    args = parser.parse_args()

    test_model(
        dataset=args.name,
        model_path=args.model_path,
        use_cpu=not args.cuda,
        temperature=args.temperature,
        dropout=args.dropout,
        m=args.m,
        k=args.k,
        n_neighbors=args.n_neighbors,
        cluster_number=args.cluster_number,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
