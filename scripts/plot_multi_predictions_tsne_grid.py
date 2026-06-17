import argparse
import json
import math
import os
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

import sys
from pathlib import Path

# 用于在子图内创建内嵌子轴
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 自动把项目根目录加入路径（不用手动改路径）
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import get_ground_truth_labels


PROJECT_ROOT = "/disk/fanjunming/home/workspace/bioinfo/scRGCL"



# tsne图的形状
PCA_BASELINE_BATCH_ROOT = os.path.join(PROJECT_ROOT, "baseline", "results", "batch_simple_baselines")
# 主方法，只需要预测结果填色
SCRGCL_TRAINING_RESULTS_ROOT = os.path.join(PROJECT_ROOT, "training_results")

default_dataset = "Pollen"
DEFAULT_EMBEDDING_CSV = os.path.join(PCA_BASELINE_BATCH_ROOT, default_dataset, "pca_embeddings.csv")
DEFAULT_LABEL_CSV = os.path.join(PCA_BASELINE_BATCH_ROOT, default_dataset, "predictions.csv")
BASELINE_DEFAULT_COLUMNS = ["pca_leiden", "pca_louvain", "pca_kmeans"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Plot a shared t-SNE grid from explicit or auto-discovered predictions.csv sources."
    )
    parser.add_argument("--embedding-csv", default=DEFAULT_EMBEDDING_CSV)
    parser.add_argument("--label-csv", default=DEFAULT_LABEL_CSV)
    parser.add_argument(
        "--source",
        action="append",
        help="Prediction source in the form PATH:col1,col2,... Can be repeated.",
    )
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--batch", action="store_true")
    return parser.parse_args()


def normalize_method_name(name: str) -> str:
    """规范化方法名，便于查找。"""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def prettify_method_name(name: str) -> str:
    """将内部名称转换为展示名称。"""
    return name.replace("_", " ").replace("+", " ").title()


def align_cluster_labels_local(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """将聚类预测标签对齐到真实标签。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {pred: true for true, pred in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, label) for label in y_pred])


def build_shared_cmap(max_label: int) -> ListedColormap:
    """构建共享的离散色图。"""
    base_colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors)
    if max_label + 1 <= len(base_colors):
        return ListedColormap(base_colors[: max_label + 1])

    repeats = math.ceil((max_label + 1) / len(base_colors))
    return ListedColormap((base_colors * repeats)[: max_label + 1])


def compute_tsne(pca_features: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    """对特征矩阵计算二维 t-SNE。"""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=22,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    return tsne.fit_transform(pca_features)


def parse_source_spec(source_spec: str) -> tuple[str, list[str]]:
    """解析 PATH:col1,col2,... 形式的来源配置。"""
    if ":" not in source_spec:
        raise ValueError(
            f"Invalid --source value '{source_spec}'. Expected format PATH:col1,col2,..."
        )
    path, column_text = source_spec.rsplit(":", 1)
    columns = [item.strip() for item in column_text.split(",") if item.strip()]
    if not path or not columns:
        raise ValueError(
            f"Invalid --source value '{source_spec}'. Expected format PATH:col1,col2,..."
        )
    return os.path.abspath(path), columns


def load_canonical_inputs(embedding_csv: str, label_csv: str) -> tuple[np.ndarray, np.ndarray]:
    """加载基准 embedding 和统一标签。"""
    embedding_df = pd.read_csv(embedding_csv)
    label_df = pd.read_csv(label_csv)

    if "label" not in label_df.columns:
        raise ValueError(f"Label source must contain a 'label' column: {label_csv}")
    if len(embedding_df) != len(label_df):
        raise ValueError(
            f"Embedding/label row mismatch: {len(embedding_df)} vs {len(label_df)}"
        )

    return embedding_df.to_numpy(), label_df["label"].to_numpy()


def load_metrics_map(predictions_path: str) -> dict:
    """读取预测文件旁边的 metrics.csv。"""
    metrics_path = os.path.join(os.path.dirname(predictions_path), "metrics.csv")
    if not os.path.exists(metrics_path):
        return {}

    metrics_df = pd.read_csv(metrics_path)
    if "method" not in metrics_df.columns:
        return {}

    return {
        normalize_method_name(str(row["method"])): row
        for _, row in metrics_df.iterrows()
    }


def validate_prediction_source(
    predictions_path: str,
    requested_columns: list[str],
    canonical_labels: np.ndarray,
) -> pd.DataFrame:
    """校验预测文件是否与统一标签一致。"""
    df = pd.read_csv(predictions_path)

    if "label" not in df.columns:
        raise ValueError(f"Predictions file must contain a 'label' column: {predictions_path}")
    if len(df) != len(canonical_labels):
        raise ValueError(
            f"Row mismatch for {predictions_path}: {len(df)} vs canonical {len(canonical_labels)}"
        )
    if not np.array_equal(df["label"].to_numpy(), canonical_labels):
        raise ValueError(f"Label order/content mismatch for predictions file: {predictions_path}")

    missing_columns = [col for col in requested_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing requested columns {missing_columns} in {predictions_path}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def build_metric_text(metric_row) -> str:
    """将可用指标格式化为面板标题文本。"""
    if metric_row is None:
        return ""

    parts = []
    if "ari" in metric_row.index:
        parts.append(f"ARI={float(metric_row['ari']):.4f}")
    if "nmi" in metric_row.index:
        parts.append(f"NMI={float(metric_row['nmi']):.4f}")
    if "acc" in metric_row.index:
        parts.append(f"ACC={float(metric_row['acc']):.4f}")
    if "f1" in metric_row.index:
        parts.append(f"F1={float(metric_row['f1']):.4f}")
    if "n_clusters" in metric_row.index:
        parts.append(f"K={int(metric_row['n_clusters'])}")
    if "epoch" in metric_row.index:
        parts.append(f"Epoch={int(metric_row['epoch'])}")
    return "  ".join(parts)


def read_summary(summary_path: str) -> dict:
    """读取 summary.json。"""
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cell_type_map(dataset_name: str) -> dict[int, str]:
    """加载数据集的标签到细胞类型映射。"""
    try:
        df = get_ground_truth_labels(dataset_name)
    except Exception:
        return {}

    if "labels" not in df.columns or "cell_type" not in df.columns:
        return {}

    print("======"*10)
    print(dataset_name, df['cell_type'].unique())
    cell_type = df["cell_type"].astype(str)
    labels = df["labels"].astype(int)
    label_cell_type_df = pd.DataFrame({"label": labels, "cell_type": cell_type})
    return (
        label_cell_type_df.drop_duplicates(subset=["label"])
        .sort_values("label")
        .set_index("label")["cell_type"]
        .to_dict()
    )


def infer_dataset_from_explicit_sources(source_specs: list[str], label_csv: str, dataset_name: Optional[str]) -> str:
    """根据显式来源推断数据集名称。"""
    if dataset_name:
        return dataset_name

    label_parent = os.path.basename(os.path.dirname(label_csv))
    if label_parent and label_parent != "batch_simple_baselines":
        return label_parent

    for source_spec in source_specs:
        predictions_path, _ = parse_source_spec(source_spec)
        summary = read_summary(os.path.join(os.path.dirname(predictions_path), "summary.json"))
        if summary.get("dataset"):
            return summary["dataset"]

        run_dir = os.path.dirname(predictions_path)
        dataset_dir = os.path.basename(os.path.dirname(run_dir))
        if dataset_dir:
            return dataset_dir

    return "comparison"


def default_out_path_for_dataset(dataset_name: str) -> str:
    """生成数据集的默认输出路径。"""
    return os.path.join(SCRGCL_TRAINING_RESULTS_ROOT, dataset_name, f"{dataset_name}_multi_method_tsne.png")


def resolve_training_column(predictions_path: str) -> Optional[str]:
    """选择训练结果中要绘制的预测列。"""
    df = pd.read_csv(predictions_path, nrows=1)
    if "scrgcl_pred_aligned" in df.columns:
        return "scrgcl_pred_aligned"
    if "scrgcl_pred" in df.columns:
        return "scrgcl_pred"
    return None


def resolve_baseline_columns(predictions_path: str) -> list[str]:
    """选择基线结果中要绘制的预测列。"""
    df = pd.read_csv(predictions_path, nrows=1)
    return [col for col in BASELINE_DEFAULT_COLUMNS if col in df.columns]


def discover_dataset_runs(method_name: str, dataset_name: str) -> list[str]:
    """查找某个数据集下有效的训练运行目录。"""
    dataset_dir = os.path.join(method_name, dataset_name)
    if not os.path.isdir(dataset_dir):
        return []

    run_dirs = []
    for entry in sorted(os.listdir(dataset_dir)):
        run_dir = os.path.join(dataset_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        summary = read_summary(os.path.join(run_dir, "summary.json"))
        if summary.get("dataset") != dataset_name:
            continue
        if 'main' not in run_dir:
            continue
        predictions_path = os.path.join(run_dir, "predictions.csv")
        if os.path.exists(predictions_path):
            run_dirs.append(run_dir)
    return run_dirs


def build_auto_sources_for_dataset(dataset_name: str) -> tuple[str, str, list[str]]:
    """为数据集自动组装输入文件和来源。"""
    embedding_csv = os.path.join(PCA_BASELINE_BATCH_ROOT, dataset_name, "pca_embeddings.csv")
    label_csv = os.path.join(PCA_BASELINE_BATCH_ROOT, dataset_name, "predictions.csv")
    baseline_predictions = os.path.join(PCA_BASELINE_BATCH_ROOT, dataset_name, "predictions.csv")

    if not os.path.exists(embedding_csv) or not os.path.exists(label_csv):
        raise FileNotFoundError(f"Missing baseline embedding or label source for dataset {dataset_name}")

    source_specs = []
    baseline_columns = resolve_baseline_columns(baseline_predictions)
    if baseline_columns:
        source_specs.append(f"{baseline_predictions}:{','.join(baseline_columns)}")


    # run_dirs获取对比方法的predictions，如scRGCL
    method_name = SCRGCL_TRAINING_RESULTS_ROOT
    run_dirs = discover_dataset_runs(method_name, dataset_name)

    # # !!!todo 多数据集比较
    # method_name = 'scccl'
    # run_dirs = discover_dataset_runs(method_name, dataset_name)
    # run_dirs合并

    if not run_dirs:
        raise FileNotFoundError(f"No valid training runs found for dataset {dataset_name}")

    for run_dir in run_dirs:
        predictions_path = os.path.join(run_dir, "predictions.csv")
        column = resolve_training_column(predictions_path)
        if column is not None:
            source_specs.append(f"{predictions_path}:{column}")

    if len(source_specs) <= 1:
        raise FileNotFoundError(f"No valid training prediction columns found for dataset {dataset_name}")

    return embedding_csv, label_csv, source_specs


def build_panel_specs(source_specs: list[str], canonical_labels: np.ndarray) -> list[dict]:
    """为每个预测列构建绘图所需的面板信息。"""
    panel_specs = []
    display_name_counts = {}

    for source_spec in source_specs:
        predictions_path, requested_columns = parse_source_spec(source_spec)
        predictions_df = validate_prediction_source(predictions_path, requested_columns, canonical_labels)
        metrics_map = load_metrics_map(predictions_path)
        source_alias = os.path.basename(os.path.dirname(predictions_path))

        for column in requested_columns:
            metric_row = metrics_map.get(normalize_method_name(column))
            if metric_row is None and normalize_method_name(column).startswith("scrgcl"):
                metric_row = metrics_map.get(normalize_method_name("scRGCL"))
            aligned_pred = align_cluster_labels_local(canonical_labels, predictions_df[column].to_numpy())
            display_name = metric_row["method"] if metric_row is not None else prettify_method_name(column)
            display_name_counts[display_name] = display_name_counts.get(display_name, 0) + 1
            panel_specs.append(
                {
                    "source_alias": source_alias,
                    "column": column,
                    "display_name": display_name,
                    "aligned_pred": aligned_pred,
                    "metric_row": metric_row[['ari', 'nmi', 'n_clusters']],
                    "predictions_path": predictions_path,
                }
            )

    for spec in panel_specs:
        if display_name_counts[spec["display_name"]] > 1:
            spec["display_name"] = f"{spec['source_alias']} / {spec['display_name']}"

    return panel_specs


def make_title(display_name: str, metric_row) -> str:
    """组合展示名称和指标文本。"""
    metric_text = build_metric_text(metric_row)
    if not metric_text:
        return display_name
    return f"{display_name}\n{metric_text}"


def render_cell_type_legend(
    legend_ax,
    cell_type_map: dict[int, str],
    cmap: ListedColormap,
    norm: Normalize,
    font_size: Optional[int] = 9,
) -> None:
    """在右侧绘制颜色到细胞类型的图例。"""
    entries = sorted(cell_type_map.items())
    if not entries:
        return

    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    # legend_ax.text(0.0, 1, "cell type", va="top", ha="left", fontsize=10, fontweight="bold")


    top = 0.88
    bottom = 0.06
    if len(entries) == 1:
        y_positions = [top]
    else:
        step = (top - bottom) / (len(entries) - 1)
        y_positions = [top - idx * step for idx in range(len(entries))]

    for (label, cell_type), y in zip(entries, y_positions):
        legend_ax.scatter(0.06, y, s=28, color=cmap(norm(label)), clip_on=False)
        legend_ax.text(0.14, y, cell_type, va="center", ha="left", fontsize=font_size)


def plot_tsne_grid(
    tsne_xy: np.ndarray,
    canonical_labels: np.ndarray,
    panel_specs: list[dict],
    dataset_name: str,
    out_path: str,
    point_size: float,
    cell_type_map: Optional[dict[int, str]] = None,
) -> None:
    """绘制并保存共享坐标系的 t-SNE 对比图。"""
    # 面板总数 = 1 个真实标签面板 + 若干个预测面板。
    panel_count = 1 + len(panel_specs)
    # 每行最多放 3 个面板。
    n_cols = min(3, panel_count)
    # 按总面板数和列数计算行数。
    n_rows = math.ceil(panel_count / n_cols)

    # 从真实标签开始计算全局最大类别值。
    global_max = int(np.max(canonical_labels))
    # 把所有预测面板里的标签也纳入最大值，保证颜色映射一致。
    for spec in panel_specs:
        global_max = max(global_max, int(np.max(spec["aligned_pred"])))

    # 构建一个共享色图，让所有子图使用同一套颜色。
    cmap = build_shared_cmap(global_max)
    norm = Normalize(vmin=0, vmax=max(global_max, 1))
    # 创建子图网格，大小随列数和行数变化。
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows), dpi=150)
    # 保证 axes 始终是一维数组，方便统一遍历。
    axes = np.atleast_1d(axes).ravel()

    # 第一个面板固定画真实标签，后面依次画各个方法的预测标签。
    panels = [("True Label", canonical_labels)] + [
        (make_title(spec["display_name"], spec["metric_row"]), spec["aligned_pred"])
        for spec in panel_specs[::-1]
    ]

    # 逐个子图绘制 t-SNE 散点。
    for ax, (title, labels) in zip(axes, panels):
        ax.scatter(
            tsne_xy[:, 0],
            tsne_xy[:, 1],
            c=labels,
            cmap=cmap,
            norm=norm,
            s=point_size,
        )
        # 设置子图标题。
        ax.set_title(title)
        # 隐藏 x 轴刻度。
        ax.set_xticks([])
        # 隐藏 y 轴刻度。
        ax.set_yticks([])

    # 如果子图数量不够，把多出来的格子关掉。
    for ax in axes[len(panels):]:
        ax.axis("off")

    # 设置整张图的总标题。
    fig.suptitle(f"{dataset_name} fixed-PCA t-SNE clustering comparison", fontsize=16)
    # 调整子图布局，给右侧注释栏留空间。
    # fig.tight_layout(rect=[0, 0, 0.96, 0.96])
    fig.tight_layout()
    # 右侧图例栏优先显示颜色到细胞类型的映射，没有映射时保留原始标签说明。
    
    # 在第一个子图（True Label）内部的右上角创建一个内嵌子轴，用于显示细胞类型图例，避免覆盖整张图。
    inset_ax = inset_axes(axes[0], width="28%", height="20%", loc="upper left", borderpad=0.6)
    render_cell_type_legend(inset_ax, cell_type_map, cmap, norm)
    
    # 全局细胞类型图例int版，global_max是标签的最大值，cell_type_map是标签到细胞类型的映射，cmap和norm用于颜色映射。
    

    legend_height = global_max * 0.02 + 0.1
    bottom_margin = 0.6 - legend_height / 2
    legend_ax = fig.add_axes([0.999, bottom_margin, 0.05, legend_height]) # [左边距, 底边距, 宽度, 高度]
    int_cell_type_map = {i: str(i) for i in range(global_max + 1)}
    render_cell_type_legend(legend_ax, int_cell_type_map, cmap, norm, font_size=13)
    fig.show()


    # 保存图片到目标路径。
    fig.savefig(out_path, bbox_inches="tight")
    # 关闭 figure，释放内存。
    plt.close(fig)


def run_one_plot(
    dataset_name: str,
    embedding_csv: str,
    label_csv: str,
    source_specs: list[str],
    out_path: str,
    perplexity: float,
    seed: int,
    point_size: float,
    cell_type_map: Optional[dict[int, str]] = None,
) -> None:
    """生成单个数据集的 t-SNE 对比图。"""
    pca_features, canonical_labels = load_canonical_inputs(embedding_csv, label_csv)# 获取各方法的制图信息，display_name，aligned_pred
    panel_specs = build_panel_specs(source_specs, canonical_labels)
    tsne_xy = compute_tsne(pca_features, perplexity=perplexity, seed=seed)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plot_tsne_grid(
        tsne_xy,
        canonical_labels,
        panel_specs,
        dataset_name,
        out_path,
        point_size,
        cell_type_map=cell_type_map,
    )

    print(f"Embedding source: {embedding_csv}")
    print(f"Canonical label source: {label_csv}")
    for spec in panel_specs:
        print(f"Plotted {spec['column']} from {spec['predictions_path']}")
    print(f"Saved t-SNE comparison figure to: {out_path}")


def run_batch(args: argparse.Namespace) -> None:
    """批量生成所有数据集的 t-SNE 对比图。"""
    dataset_dirs = [
        name for name in sorted(os.listdir(SCRGCL_TRAINING_RESULTS_ROOT))
        if os.path.isdir(os.path.join(SCRGCL_TRAINING_RESULTS_ROOT, name))
    ]
    if args.dataset_name:
        dataset_dirs = [name for name in dataset_dirs if name == args.dataset_name]

    for dataset_name in dataset_dirs:
        try:
            embedding_csv, label_csv, source_specs = build_auto_sources_for_dataset(dataset_name)
            cell_type_map = load_cell_type_map(dataset_name)
            out_path = os.path.abspath(args.out_path) if args.out_path else default_out_path_for_dataset(dataset_name)
            if args.out_path and os.path.isdir(args.out_path):
                out_path = os.path.join(os.path.abspath(args.out_path), f"{dataset_name}_multi_method_tsne.png")
            run_one_plot(
                dataset_name=dataset_name,
                embedding_csv=embedding_csv,
                label_csv=label_csv,
                source_specs=source_specs,
                out_path=out_path,
                perplexity=args.perplexity,
                seed=args.seed,
                point_size=args.point_size,
                cell_type_map=cell_type_map,
            )
        except Exception as exc:
            print(f"Warning: skipped dataset {dataset_name}: {exc}")


def main() -> None:
    """CLI 入口。"""
    args = parse_args()
    args.batch = 1
    if args.batch:
        run_batch(args)
        return

    if args.source:
        dataset_name = infer_dataset_from_explicit_sources(args.source, args.label_csv, args.dataset_name)
        embedding_csv = os.path.abspath(args.embedding_csv)
        label_csv = os.path.abspath(args.label_csv)
        out_path = os.path.abspath(args.out_path) if args.out_path else default_out_path_for_dataset(dataset_name)
        source_specs = args.source
    else:
        dataset_name = args.dataset_name or "Pollen"
        # 获取pca_embedding，真实标签，各方法预测标签的路径
        embedding_csv, label_csv, source_specs = build_auto_sources_for_dataset(dataset_name)
        out_path = os.path.abspath(args.out_path) if args.out_path else default_out_path_for_dataset(dataset_name)
    # 获取映射字典，{0: 'blood', 1: 'dermal', 2: 'neural', 3: 'pluripotent'}
    cell_type_map = load_cell_type_map(dataset_name) if dataset_name != "comparison" else {}

    run_one_plot(
        dataset_name=dataset_name,
        embedding_csv=embedding_csv,
        label_csv=label_csv,
        source_specs=source_specs,
        out_path=out_path,
        perplexity=args.perplexity,
        seed=args.seed,
        point_size=args.point_size,
        cell_type_map=cell_type_map,
    )


if __name__ == "__main__":
    main()
