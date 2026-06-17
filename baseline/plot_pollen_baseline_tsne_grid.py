import argparse
import json
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a shared t-SNE grid for baseline clustering results."
    )
    parser.add_argument(
        "--result-dir",
        default="/disk/fanjunming/home/workspace/bioinfo/scRGCL/baseline/results/pollen_simple_baselines",
        help="Directory containing pca_embeddings.csv, predictions.csv, metrics.csv, and optional summary.json.",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Optional output path for the comparison figure. Defaults to <result-dir>/tsne_method_grid.png.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        default=None,
        help="Optional comma-separated subset of method columns from predictions.csv.",
    )
    parser.add_argument("--point-size", type=float, default=18.0)
    return parser.parse_args()


def normalize_method_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def prettify_method_name(name: str) -> str:
    return name.replace("_", " ").replace("+", " ").title()


def align_cluster_labels_local(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {pred: true for true, pred in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, label) for label in y_pred])


def load_summary(summary_path: str) -> dict:
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_inputs(result_dir: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict]:
    pca_path = os.path.join(result_dir, "pca_embeddings.csv")
    pred_path = os.path.join(result_dir, "predictions.csv")
    metrics_path = os.path.join(result_dir, "metrics.csv")
    summary_path = os.path.join(result_dir, "summary.json")

    embeddings_df = pd.read_csv(pca_path)
    predictions_df = pd.read_csv(pred_path)
    metrics_df = pd.read_csv(metrics_path)
    summary = load_summary(summary_path)

    if "label" not in predictions_df.columns:
        raise ValueError(f"predictions.csv must contain a 'label' column: {pred_path}")

    required_metric_columns = {"method", "n_clusters", "ari", "nmi"}
    missing_metric_columns = required_metric_columns - set(metrics_df.columns)
    if missing_metric_columns:
        raise ValueError(
            f"metrics.csv is missing required columns {sorted(missing_metric_columns)}: {metrics_path}"
        )

    if len(embeddings_df) != len(predictions_df):
        raise ValueError(
            "Row count mismatch between pca_embeddings.csv and predictions.csv: "
            f"{len(embeddings_df)} vs {len(predictions_df)}"
        )

    method_columns = [col for col in predictions_df.columns if col != "label"]
    if not method_columns:
        raise ValueError(f"No method columns found in predictions.csv: {pred_path}")

    return embeddings_df.to_numpy(), predictions_df, metrics_df, summary


def select_method_columns(predictions_df: pd.DataFrame, methods_arg: str | None) -> list[str]:
    available_columns = [col for col in predictions_df.columns if col != "label"]
    if not methods_arg:
        return available_columns

    requested_columns = [item.strip() for item in methods_arg.split(",") if item.strip()]
    missing_columns = [col for col in requested_columns if col not in available_columns]
    if missing_columns:
        raise ValueError(
            f"Requested methods not found in predictions.csv: {missing_columns}. "
            f"Available: {available_columns}"
        )
    return requested_columns


def build_method_specs(
    method_columns: list[str],
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    y_true: np.ndarray,
) -> list[dict]:
    metrics_map = {
        normalize_method_name(row["method"]): row for _, row in metrics_df.iterrows()
    }

    specs = []
    for column in method_columns:
        metric_row = metrics_map.get(normalize_method_name(column))
        aligned_pred = align_cluster_labels_local(y_true, predictions_df[column].to_numpy())
        specs.append(
            {
                "column": column,
                "display_name": metric_row["method"] if metric_row is not None else prettify_method_name(column),
                "aligned_pred": aligned_pred,
                "metric_row": metric_row,
            }
        )
    return specs


def compute_tsne(features: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=22,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    return tsne.fit_transform(features)


def build_shared_cmap(max_label: int) -> ListedColormap:
    base_colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors)
    if max_label + 1 <= len(base_colors):
        return ListedColormap(base_colors[: max_label + 1])

    repeats = math.ceil((max_label + 1) / len(base_colors))
    return ListedColormap((base_colors * repeats)[: max_label + 1])


def make_title(display_name: str, metric_row) -> str:
    if metric_row is None:
        return display_name
    return (
        f"{display_name}\n"
        f"ARI={metric_row['ari']:.4f}  NMI={metric_row['nmi']:.4f}  K={int(metric_row['n_clusters'])}"
    )


def plot_tsne_grid(
    tsne_xy: np.ndarray,
    y_true: np.ndarray,
    method_specs: list[dict],
    dataset_name: str,
    out_path: str,
    point_size: float,
) -> None:
    panel_count = 1 + len(method_specs)
    n_cols = min(3, panel_count)
    n_rows = math.ceil(panel_count / n_cols)

    global_max = int(np.max(y_true))
    for spec in method_specs:
        global_max = max(global_max, int(np.max(spec["aligned_pred"])))

    cmap = build_shared_cmap(global_max)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows), dpi=150)
    axes = np.atleast_1d(axes).ravel()

    panels = [("True Label", y_true, None)] + [
        (make_title(spec["display_name"], spec["metric_row"]), spec["aligned_pred"], spec["column"])
        for spec in method_specs
    ]

    for ax, (title, labels, _) in zip(axes, panels):
        ax.scatter(
            tsne_xy[:, 0],
            tsne_xy[:, 1],
            c=labels,
            cmap=cmap,
            s=point_size,
            vmin=0,
            vmax=max(global_max, 1),
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(f"{dataset_name} PCA→t-SNE clustering comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    result_dir = os.path.abspath(args.result_dir)
    out_path = args.out_path or os.path.join(result_dir, "tsne_method_grid.png")

    features, predictions_df, metrics_df, summary = load_inputs(result_dir)
    y_true = predictions_df["label"].to_numpy()
    method_columns = select_method_columns(predictions_df, args.methods)
    method_specs = build_method_specs(method_columns, predictions_df, metrics_df, y_true)

    dataset_name = summary.get("dataset") or os.path.basename(result_dir.rstrip(os.sep))
    tsne_xy = compute_tsne(features, perplexity=args.perplexity, seed=args.seed)
    plot_tsne_grid(tsne_xy, y_true, method_specs, dataset_name, out_path, args.point_size)

    print(f"Loaded result directory: {result_dir}")
    print(f"Methods plotted: {', '.join(method_columns)}")
    print(f"Saved t-SNE comparison figure to: {out_path}")


if __name__ == "__main__":
    main()
