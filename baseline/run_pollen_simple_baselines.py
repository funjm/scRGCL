import argparse
import json
import os
from dataclasses import asdict, dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from load_datamat import _encode_labels, _load_raw_dataset, _to_dense_array
import sys
from pathlib import Path

current_path = os.path.dirname(os.path.abspath(__file__))
# 上级目录
parent_path = os.path.dirname(current_path)
print(f"当前路径: {parent_path}")
sys.path.append(parent_path)

from src.utils import get_dataset



@dataclass
class ClusteringResult:
    method: str
    n_clusters: int
    ari: float
    nmi: float


def load_pollen_anndata(dataset_name: str = "Pollen") -> tuple[ad.AnnData, np.ndarray]:
    source_info = None
    features, labels, source_info = _load_raw_dataset(dataset_name)

    features = _to_dense_array(features)
    labels = _encode_labels(labels).astype(int)

    if features.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={features.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Cell/label mismatch: features.shape={features.shape}, labels.shape={labels.shape}"
        )

    adata = ad.AnnData(X=features)
    adata.obs["label"] = pd.Categorical(labels.astype(str))
    adata.uns["source_info"] = source_info
    return adata, labels


def preprocess_for_pca(
    adata: ad.AnnData,
    target_sum: float = 1e4,
    n_top_genes: int | None = 2000,
    scale_max_value: float = 10.0,
) -> ad.AnnData:
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    if n_top_genes is not None and adata.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
        if "highly_variable" in adata.var:
            adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=scale_max_value)
    return adata


def run_pca(
    adata: ad.AnnData,
    n_pcs: int,
    seed: int,
) -> ad.AnnData:
    adata = adata.copy()
    max_valid_pcs = max(1, min(n_pcs, adata.n_obs - 1, adata.n_vars))
    sc.pp.pca(adata, n_comps=max_valid_pcs, svd_solver="arpack", random_state=seed)
    return adata


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> ClusteringResult:
    return ClusteringResult(
        method=method,
        n_clusters=int(len(np.unique(y_pred))),
        ari=float(adjusted_rand_score(y_true, y_pred)),
        nmi=float(normalized_mutual_info_score(y_true, y_pred)),
    )


def infer_resolution_from_n_classes(n_classes: int) -> float:
    return float(np.clip(0.4 + 0.2 * np.log2(max(n_classes, 2)), 0.4, 2.0))


def run_leiden(
    adata_pca: ad.AnnData,
    y_true: np.ndarray,
    n_neighbors: int,
    n_pcs: int,
    resolution: float,
    seed: int,
) -> tuple[ClusteringResult, np.ndarray]:
    adata = adata_pca.copy()
    use_n_pcs = min(n_pcs, adata.obsm["X_pca"].shape[1])
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=use_n_pcs)
    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=seed,
        key_added="pred",
    )
    pred = adata.obs["pred"].astype(str).astype(int).to_numpy()
    return evaluate(y_true, pred, "PCA+Leiden"), pred


def run_louvain(
    adata_pca: ad.AnnData,
    y_true: np.ndarray,
    n_neighbors: int,
    n_pcs: int,
    resolution: float,
    seed: int,
) -> tuple[ClusteringResult, np.ndarray]:
    adata = adata_pca.copy()
    use_n_pcs = min(n_pcs, adata.obsm["X_pca"].shape[1])
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=use_n_pcs)
    try:
        sc.tl.louvain(
            adata,
            resolution=resolution,
            random_state=seed,
            flavor="vtraag",
            key_added="pred",
        )
    except ModuleNotFoundError as exc:
        if exc.name != "pkg_resources":
            raise
        print(
            "Warning: scanpy.tl.louvain with flavor='vtraag' requires the legacy "
            "'louvain' package, which imports pkg_resources and is incompatible in this "
            "environment. Falling back to flavor='igraph' for a valid Louvain baseline."
        )
        sc.tl.louvain(
            adata,
            random_state=seed,
            flavor="igraph",
            directed=False,
            key_added="pred",
        )
    pred = adata.obs["pred"].astype(str).astype(int).to_numpy()
    return evaluate(y_true, pred, "PCA+Louvain"), pred


def run_kmeans(
    adata_pca: ad.AnnData,
    y_true: np.ndarray,
    n_pcs: int,
    n_clusters: int,
    seed: int,
) -> tuple[ClusteringResult, np.ndarray]:
    x_pca = adata_pca.obsm["X_pca"][:, : min(n_pcs, adata_pca.obsm["X_pca"].shape[1])]
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    pred = model.fit_predict(x_pca)
    return evaluate(y_true, pred, "PCA+KMeans"), pred


def save_outputs(
    out_dir: str,
    adata_pca: ad.AnnData,
    labels: np.ndarray,
    results: list[ClusteringResult],
    predictions: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    results_df = pd.DataFrame([asdict(item) for item in results])
    results_path = os.path.join(out_dir, "metrics.csv")
    results_df.to_csv(results_path, index=False)

    pred_df = pd.DataFrame({"label": labels})
    for name, values in predictions.items():
        pred_df[name] = values
    pred_path = os.path.join(out_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    pca_path = os.path.join(out_dir, "pca_embeddings.csv")
    pd.DataFrame(adata_pca.obsm["X_pca"]).to_csv(pca_path, index=False)

    summary = {
        "dataset": args.dataset,
        "n_cells": int(adata_pca.n_obs),
        "n_genes_after_preprocess": int(adata_pca.n_vars),
        "n_label_classes": int(len(np.unique(labels))),
        "source_info": adata_pca.uns.get("source_info", ""),
        "args": vars(args),
        "results": [asdict(item) for item in results],
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics to: {results_path}")
    print(f"Saved predictions to: {pred_path}")
    print(f"Saved PCA embeddings to: {pca_path}")
    print(f"Saved summary to: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple clustering baselines on the Pollen dataset.")
    parser.add_argument("--dataset", default="merged_annotated_cells", help="Dataset name registered in load_datamat.py")
    parser.add_argument("--out-dir", default="/disk/fanjunming/home/workspace/bioinfo/baseline/results/pollen_simple_baselines")
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Manual resolution override. If omitted, infer it from the dataset class count.",
    )
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--scale-max-value", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # !!!
    # args.dataset = 'Pollen'  # 强制使用pollen数据集，忽略命令行输入

    sc.settings.verbosity = 2
    sc.set_figure_params(dpi=100, facecolor="white")

    adata, labels = load_pollen_anndata(args.dataset)
    print(f"Loaded {args.dataset}: n_cells={adata.n_obs}, n_genes={adata.n_vars}, source={adata.uns['source_info']}")
    
    print(f"Number of NaN values in expression matrix: {np.isnan(adata.X.data).sum()}")
    
    # adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=10, neginf=0)


    adata = preprocess_for_pca(
        adata,
        target_sum=args.target_sum,
        n_top_genes=args.n_top_genes,
        scale_max_value=args.scale_max_value,
    )
    adata_pca = run_pca(adata, n_pcs=args.n_pcs, seed=args.seed)
    adata_pca.uns["source_info"] = adata.uns.get("source_info", "")

    n_clusters = len(np.unique(labels))
    manual_resolution = args.resolution
    effective_resolution = (
        manual_resolution
        if manual_resolution is not None
        else infer_resolution_from_n_classes(n_clusters)
    )
    args.resolution = effective_resolution
    resolution_source = "manual override" if manual_resolution is not None else "auto"
    print(
        f"Using resolution={effective_resolution:.3f} for {n_clusters} label classes ({resolution_source})"
    )
    results: list[ClusteringResult] = []
    predictions: dict[str, np.ndarray] = {}

    leiden_result, leiden_pred = run_leiden(
        adata_pca,
        labels,
        n_neighbors=args.n_neighbors,
        n_pcs=args.n_pcs,
        resolution=args.resolution,
        seed=args.seed,
    )
    results.append(leiden_result)
    predictions["pca_leiden"] = leiden_pred

    louvain_result, louvain_pred = run_louvain(
        adata_pca,
        labels,
        n_neighbors=args.n_neighbors,
        n_pcs=args.n_pcs,
        resolution=args.resolution,
        seed=args.seed,
    )
    results.append(louvain_result)
    predictions["pca_louvain"] = louvain_pred

    kmeans_result, kmeans_pred = run_kmeans(
        adata_pca,
        labels,
        n_pcs=args.n_pcs,
        n_clusters=n_clusters,
        seed=args.seed,
    )
    results.append(kmeans_result)
    predictions["pca_kmeans"] = kmeans_pred

    metrics_df = pd.DataFrame([asdict(item) for item in results]).sort_values(["ari", "nmi"], ascending=False)
    print("\n=== Clustering metrics ===")
    print(metrics_df.to_string(index=False))

    save_outputs(args.out_dir, adata_pca, labels, results, predictions, args)


if __name__ == "__main__":
    main()
