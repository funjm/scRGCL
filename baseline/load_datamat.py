import math
import os
import scipy.io as sio
import scipy.sparse as sp
import numpy as np
try:
    import torch
except ImportError:  # optional for non-PyTorch utilities
    torch = None
import pandas as pd
from sklearn.preprocessing import normalize

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DATASET_DIR = "/disk/fanjunming/home/workspace/bioinfo/dataset/merged_dataset"
LARGE_DATASET_DIR = "/disk/fanjunming/home/workspace/bioinfo/dataset/large_dataset"
ZISCDESK_DATASET_DIR = (
    "/disk/fanjunming/home/workspace/bioinfo/dataset/"
    "ziscDesk-single-cell-data-20241011T104023Z-001/ziscDesk-single-cell-data"
)
# DEFAULT_DATASET_NAME = "human_ESC"
DEFAULT_DATASET_NAME = "Pollen"
# DEFAULT_DATASET_NAME = "Romanov"

FEATURE_KEY_CANDIDATES = ("fea", "feature", "features", "X", "data")
LABEL_KEY_CANDIDATES = ("label", "labels", "y", "Y", "gt")
ZISCDESK_LABEL_CANDIDATES = ("labels", "cell_type")
ZISCDESK_H5_LABEL_CANDIDATES = ("labels", "cell_type", "celltype", "cell_type1", "cell_ontology_class")
ZISCDESK_MATRIX_FILENAMES = ("expr.npz", "data.h5", "expr.mtx")


def _built_in_dataset_path(filename):
    return os.path.join(PROJECT_ROOT, "datasets", filename)


def _external_dataset_path(filename):
    return os.path.join(EXTERNAL_DATASET_DIR, filename)


def _dataset_entry(
    path,
    source_type="mat",
    feature_keys=FEATURE_KEY_CANDIDATES,
    label_keys=LABEL_KEY_CANDIDATES,
    label_columns=ZISCDESK_LABEL_CANDIDATES,
    h5_label_columns=ZISCDESK_H5_LABEL_CANDIDATES,
):
    return {
        "path": path,
        "source_type": source_type,
        "feature_keys": tuple(feature_keys),
        "label_keys": tuple(label_keys),
        "label_columns": tuple(label_columns),
        "h5_label_columns": tuple(h5_label_columns),
    }


BASE_DATASET_REGISTRY = {
    "human_ESC": _dataset_entry(_built_in_dataset_path("human_ESC.mat")),
    "human_brain": _dataset_entry(_built_in_dataset_path("human_brain.mat")),
    "time_course": _dataset_entry(_built_in_dataset_path("time_course.mat")),
    "Klein": _dataset_entry(_external_dataset_path("Klein.mat")),
    "PBMC16635_sort": _dataset_entry(_external_dataset_path("PBMC16635_sort.mat")),
    "Petropoulos_sort": _dataset_entry(_external_dataset_path("Petropoulos_sort.mat")),
    "YAN": _dataset_entry(_external_dataset_path("YAN.mat")),
    "Mouse_sort": _dataset_entry(_external_dataset_path("Mouse_sort.mat")),
    "filtered_mESC": _dataset_entry(_external_dataset_path("filtered_mESC.mat")),
    "filtered_mPCTC": _dataset_entry(_external_dataset_path("filtered_mPCTC.mat")),
    "filtered_time_course": _dataset_entry(_external_dataset_path("filtered_time_course.mat")),
    "merged_human_ESC": _dataset_entry(_external_dataset_path("human_ESC.mat")),
    "merged_human_brain": _dataset_entry(_external_dataset_path("human_brain.mat")),
    "leukemia1": _dataset_entry(_external_dataset_path("leukemia1.mat")),
    "li_NM1": _dataset_entry(_external_dataset_path("li_NM1.mat")),
    "li_cell1": _dataset_entry(_external_dataset_path("li_cell1.mat")),
    "mBladder_sort": _dataset_entry(_external_dataset_path("mBladder_sort.mat")),
    "mESC": _dataset_entry(_external_dataset_path("mESC.mat")),
    "mPCTC": _dataset_entry(_external_dataset_path("mPCTC.mat")),
    "new_Chen_sort": _dataset_entry(_external_dataset_path("new_Chen_sort.mat")),
    "new_Human_k": _dataset_entry(_external_dataset_path("new_Human_k.mat")),
    "new_Human_k_sort": _dataset_entry(_external_dataset_path("new_Human_k_sort.mat")),
    "new_Human_k_sortfilter": _dataset_entry(_external_dataset_path("new_Human_k_sortfilter.mat")),
    "new_Human_p_sortdelect": _dataset_entry(_external_dataset_path("new_Human_p_sortdelect.mat")),
    "new_Mouse_sort": _dataset_entry(_external_dataset_path("new_Mouse_sort.mat")),
    "new_Quake_10x_Spleen_sort": _dataset_entry(_external_dataset_path("new_Quake_10x_Spleen_sort.mat")),
    "new_Shekhar_mouse_retina_raw_data": _dataset_entry(
        _external_dataset_path("new_Shekhar_mouse_retina_raw_data.mat")
    ),
    "new_Zeisel": _dataset_entry(_external_dataset_path("new_Zeisel.mat")),
    "merged_time_course": _dataset_entry(_external_dataset_path("time_course.mat")),
    "merged_annotated_cells": _dataset_entry(os.path.join(LARGE_DATASET_DIR, "merged_annotated_cells"), source_type="large_dataset"),
}



def _is_ziscdesk_dataset_dir(path):
    if not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, "Cells.csv")):
        return False
    return any(os.path.exists(os.path.join(path, filename)) for filename in ZISCDESK_MATRIX_FILENAMES)



def _build_dataset_registry():
    registry = dict(BASE_DATASET_REGISTRY)
    if not os.path.isdir(ZISCDESK_DATASET_DIR):
        return registry

    for entry in sorted(os.scandir(ZISCDESK_DATASET_DIR), key=lambda item: item.name):
        if not entry.is_dir() or not _is_ziscdesk_dataset_dir(entry.path):
            continue
        if entry.name in registry:
            continue
        registry[entry.name] = _dataset_entry(entry.path, source_type="ziscdesk")
    return registry


DATASET_REGISTRY = _build_dataset_registry()



def get_dataset_names():
    return tuple(DATASET_REGISTRY.keys())



def _get_dataset_entry(data_name):
    if data_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{data_name}'. Available datasets: {', '.join(get_dataset_names())}")
    return DATASET_REGISTRY[data_name]



def _select_key(data, candidates, key_type, data_name, filename):
    for candidate in candidates:
        if candidate in data:
            return candidate
    available_keys = [key for key in data.keys() if not key.startswith("__")]
    raise KeyError(
        f"Dataset '{data_name}' from '{filename}' is missing a {key_type} key. "
        f"Tried {list(candidates)}. Available keys: {available_keys}"
    )



def _to_dense_array(values):
    if sp.issparse(values):
        return values.toarray()
    return np.asarray(values)



def _encode_labels(labels):
    labels = _to_dense_array(labels)
    labels = np.squeeze(labels)
    if np.issubdtype(labels.dtype, np.number):
        labels = labels.astype(np.float64).reshape(-1)
        return np.squeeze(labels - np.min(labels))

    flat_labels = np.asarray(labels).reshape(-1)
    normalized_labels = []
    for value in flat_labels:
        if isinstance(value, np.ndarray):
            normalized_labels.append(str(value.item() if value.size == 1 else value.tolist()))
        elif isinstance(value, (bytes, np.bytes_)):
            normalized_labels.append(value.decode("utf-8"))
        else:
            normalized_labels.append(str(value))
    _, encoded_labels = np.unique(np.asarray(normalized_labels, dtype=str), return_inverse=True)
    return encoded_labels



def _flatten_labels(labels):
    return np.asarray(_to_dense_array(labels)).reshape(-1)



def _load_raw_mat_dataset(dataset_entry, data_name):
    filename = dataset_entry["path"]
    data = sio.loadmat(filename)
    feature_key = _select_key(data, dataset_entry["feature_keys"], "feature", data_name, filename)
    label_key = _select_key(data, dataset_entry["label_keys"], "label", data_name, filename)
    return data[feature_key], data[label_key], f"feature_key={feature_key}, label_key={label_key}"



def _load_npz_matrix(filename, data_name):
    with np.load(filename, allow_pickle=False) as data:
        files = tuple(data.files)
        if len(files) == 1 or len(files) == 3:
            return data[files[0]], os.path.basename(filename)
        required_fields = {"data", "indices", "indptr", "shape", "format"}
        if required_fields.issubset(files):
            return sp.load_npz(filename), os.path.basename(filename)
    raise ValueError(
        f"Dataset '{data_name}' has an unsupported NPZ layout in '{filename}'. "
        f"Found keys: {list(files)}"
    )



def _load_h5_matrix(filename, data_name):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            f"Dataset '{data_name}' requires h5py to read fallback file '{filename}'."
        ) from exc

    with h5py.File(filename, "r") as handle:
        if "exprs" in handle:
            exprs = handle["exprs"]
            required_fields = ("data", "indices", "indptr", "shape")
            if all(field in exprs for field in required_fields):
                shape = tuple(int(value) for value in np.asarray(exprs["shape"]).reshape(-1))
                matrix = sp.csr_matrix(
                    (
                        np.asarray(exprs["data"]),
                        np.asarray(exprs["indices"]),
                        np.asarray(exprs["indptr"]),
                    ),
                    shape=shape,
                )
                return matrix, os.path.basename(filename)
        if "X" in handle:
            return np.asarray(handle["X"]), os.path.basename(filename)

    raise KeyError(f"Dataset '{data_name}' is missing a supported expression matrix in '{filename}'.")



def _load_mtx_matrix(filename, data_name):
    matrix = sio.mmread(filename)
    if matrix is None:
        raise ValueError(f"Dataset '{data_name}' could not be read from '{filename}'.")
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
    return matrix, os.path.basename(filename)



def _load_ziscdesk_expression(dataset_dir, data_name):
    loaders = (
        ("expr.npz", _load_npz_matrix),
        ("data.h5", _load_h5_matrix),
        ("expr.mtx", _load_mtx_matrix),
    )
    errors = []
    for filename, loader in loaders:
        full_path = os.path.join(dataset_dir, filename)
        if not os.path.exists(full_path):
            continue
        try:
            return loader(full_path, data_name)
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
    if errors:
        raise ValueError(f"Dataset '{data_name}' could not load an expression matrix. {'; '.join(errors)}")
    raise FileNotFoundError(
        f"Dataset '{data_name}' is missing all supported matrix files: {', '.join(ZISCDESK_MATRIX_FILENAMES)}"
    )



def _clean_metadata_columns(frame):
    columns = [column for column in frame.columns if not str(column).startswith("Unnamed:")]
    return frame.loc[:, columns]



def _load_labels_from_cells_csv(filename, data_name, label_columns):
    frame = _clean_metadata_columns(pd.read_csv(filename))
    for column in label_columns:
        if column in frame and not frame[column].isna().all():
            return frame[column].to_numpy(), f"{os.path.basename(filename)}:{column}"
    available_columns = list(frame.columns)
    raise KeyError(
        f"Dataset '{data_name}' is missing a supported label column in '{filename}'. "
        f"Tried {list(label_columns)}. Available columns: {available_columns}"
    )



def _load_labels_from_h5(filename, data_name, label_columns):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            f"Dataset '{data_name}' requires h5py to read fallback labels from '{filename}'."
        ) from exc

    with h5py.File(filename, "r") as handle:
        if "obs" not in handle:
            raise KeyError(f"Dataset '{data_name}' is missing 'obs' in '{filename}'.")
        obs = handle["obs"]
        for column in label_columns:
            if column in obs:
                return np.asarray(obs[column]), f"{os.path.basename(filename)}:{column}"

    raise KeyError(
        f"Dataset '{data_name}' is missing a supported HDF5 label column in '{filename}'. "
        f"Tried {list(label_columns)}"
    )



def _load_ziscdesk_labels(dataset_entry, data_name):
    dataset_dir = dataset_entry["path"]
    cells_path = os.path.join(dataset_dir, "Cells.csv")
    if os.path.exists(cells_path):
        try:
            return _load_labels_from_cells_csv(cells_path, data_name, dataset_entry["label_columns"])
        except Exception as cells_error:
            h5_path = os.path.join(dataset_dir, "data.h5")
            if os.path.exists(h5_path):
                try:
                    return _load_labels_from_h5(h5_path, data_name, dataset_entry["h5_label_columns"])
                except Exception as h5_error:
                    raise KeyError(f"{cells_error}; {h5_error}") from h5_error
            raise

    h5_path = os.path.join(dataset_dir, "data.h5")
    if os.path.exists(h5_path):
        return _load_labels_from_h5(h5_path, data_name, dataset_entry["h5_label_columns"])
    raise FileNotFoundError(f"Dataset '{data_name}' is missing both 'Cells.csv' and 'data.h5' label sources.")


def _load_large_dataset_expression(dataset_dir, data_name):
    matrix_path = os.path.join(dataset_dir, "matrix.mtx")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Dataset '{data_name}' is missing matrix file '{matrix_path}'.")
    return _load_mtx_matrix(matrix_path, data_name)


def _load_large_dataset_labels(dataset_dir, data_name):
    metadata_path = os.path.join(dataset_dir, "cell_metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Dataset '{data_name}' is missing label metadata file '{metadata_path}'.")

    frame = _clean_metadata_columns(pd.read_csv(metadata_path, index_col=0))
    for column in ("free_annotation", "cell_ontology_class", "cluster.ids", "labels"):
        if column in frame and not frame[column].isna().all():
            return frame[column].to_numpy(), f"{os.path.basename(metadata_path)}:{column}"

    available_columns = list(frame.columns)
    raise KeyError(
        f"Dataset '{data_name}' is missing a supported label column in '{metadata_path}'. "
        f"Tried ['free_annotation', 'cell_ontology_class', 'cluster.ids', 'labels']. Available columns: {available_columns}"
    )


def _load_raw_large_dataset(dataset_entry, data_name):
    dataset_dir = dataset_entry["path"]
    features, feature_source = _load_large_dataset_expression(dataset_dir, data_name)
    labels, label_source = _load_large_dataset_labels(dataset_dir, data_name)
    labels = _flatten_labels(labels)
    features = _align_features_with_labels(features, labels, data_name, feature_source)
    return features, labels, f"feature_source={feature_source}, label_source={label_source}"


def _align_features_with_labels(features, labels, data_name, feature_source):
    if len(features.shape) != 2:
        raise ValueError(
            f"Dataset '{data_name}' from '{feature_source}' must have a 2D expression matrix, got shape {features.shape}."
        )

    label_count = labels.shape[0]
    if features.shape[0] == label_count:
        return features
    if features.shape[1] == label_count:
        return features.T
    raise ValueError(
        f"Dataset '{data_name}' has {label_count} labels but expression matrix shape {features.shape} from '{feature_source}'."
    )



def _load_raw_ziscdesk_dataset(dataset_entry, data_name):
    features, feature_source = _load_ziscdesk_expression(dataset_entry["path"], data_name)
    labels, label_source = _load_ziscdesk_labels(dataset_entry, data_name)
    labels = _flatten_labels(labels)
    features = _align_features_with_labels(features, labels, data_name, feature_source)
    return features, labels, f"feature_source={feature_source}, label_source={label_source}"



def _load_raw_dataset(data_name):
    dataset_entry = _get_dataset_entry(data_name)
    if dataset_entry["source_type"] == "mat":
        return _load_raw_mat_dataset(dataset_entry, data_name)
    if dataset_entry["source_type"] == "ziscdesk":
        return _load_raw_ziscdesk_dataset(dataset_entry, data_name)
    if dataset_entry["source_type"] == "large_dataset":
        return _load_raw_large_dataset(dataset_entry, data_name)
    raise ValueError(f"Unsupported dataset source type '{dataset_entry['source_type']}' for dataset '{data_name}'.")



def get_datamat(data_name, shape, device):
    features, labels, source_info = _load_raw_dataset(data_name)

    features = _to_dense_array(features)
    labels = _encode_labels(labels)
    feature_dim = features.shape[1]
    shape = math.ceil(np.sqrt(feature_dim))
    if shape % 2 != 0:
        shape += 1

    print(f"Dataset {data_name}: {source_info}, feature_dim={feature_dim}, shape={shape}")
    padded_features = np.pad(features, ((0, 0), (0, shape * shape - feature_dim)), mode='constant') # 填充0(N, D)-->(N, ceil(sqrt(D))²)
    # 排序，活跃基因放前面
    idx = np.argsort(padded_features.std(0))[::-1][:shape * shape]
    features = padded_features[:, idx]
    ori_matrix = padded_features[:, idx]
    features = normalize(features, norm='l1', axis=1)
    # 二维矩阵表示样本-基因表达矩阵 (N, 1, ceil(sqrt(D)), ceil(sqrt(D)))
    features = features.reshape((-1, 1, shape, shape))
    
    # dropout_matrix 用于表示每个样本的 dropout 位置
    # 其中：
    # - 0：原始数据为 0 的位置（应该被mask）
    # - 1：原始数据非 0 的位置，或 padding 的 0（不应该被mask）
    dropout_matrix = np.ones_like(ori_matrix) # 初始化为1
    dropout_matrix[ori_matrix == 0] = 0 # 如果原始值是0，则标记为0
    idx = np.argwhere(np.all(dropout_matrix[..., :] == 0, axis=0)) # 不表达的基因（列的所有值都是0）
    dropout_matrix[:, idx] = 1 # 将全0列改为1
    dropout_matrix = dropout_matrix.reshape((-1, 1, shape, shape)) # reshape成4D
    print('to gpu')
    features = torch.from_numpy(features).float().to(device)
    dropout_matrix = dropout_matrix.astype(np.float64)
    dropout_matrix = torch.from_numpy(dropout_matrix).float().to(device)
    print('before return', features.shape)
    return features, labels, dropout_matrix
