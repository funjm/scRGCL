## scRGCL: Neighbor-Aware Graph Contrastive Learning for Robust Single-Cell Clustering
[![Static Badge](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)

scRGCL is a self-supervised deep learning method that learns a regularized representation guided by contrastive learning for single-cell clustering. Specifically, scRGCL captures the cell-type-associated expression structure by clustering similar cells together while ensuring consistency. For each sample, the model performs negative sampling by selecting cells from distinct clusters, thereby ensuring semantic dissimilarity between the target cell and its negative pairs. Moreover, scRGCL introduces a neighbor-aware re-weighting strategy that increases the contribution of samples from clusters closely related to the target, effectively preserving intra-cluster compactness.

![scRGCL architecture overview](https://github.com/user-attachments/assets/placeholder)

## Repository Structure
```bash
scRGCL/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── .gitignore
├── src/                # Core library
│   ├── __init__.py
│   ├── ScRGCL.py       # Core model definitions (encoder, projector, ScRGCL class)
│   ├── train.py        # Main training loop and evaluation
│   ├── clustering.py    # K-means clustering with cosine distance
│   ├── evaluation.py    # Clustering metrics (ACC, NMI, ARI, F1)
│   ├── st_loss.py       # Loss functions (RGCLoss, ClusterLoss, InstanceLoss, PrototypeLoss)
│   └── utils.py         # Data preprocessing, utilities, visualization helpers
├── scripts/             # Executable scripts
│   ├── __init__.py
│   ├── main.py          # Entry point for single-dataset experiments
│   ├── test.py          # Load saved model and evaluate
│   ├── search.py        # Hyperparameter optimization with Optuna
│   ├── search_multi.py  # Multi-dataset hyperparameter search
│   ├── hyperparameter_sensitivity.py  # Parameter sensitivity analysis
│   └── shap_main.py     # SHAP analysis script
└── config/              # Configuration
    ├── __init__.py
    └── opt.py           # Command-line argument parser and dataset-specific hyperparameters
```

## Pre-requisites
* Linux / macOS
* NVIDIA GPU (recommended for large datasets)
* Python 3.8+
* PyTorch (2.0+)
* scanpy, scikit-learn, scipy, numpy
* optuna (for hyperparameter search)
* umap-learn, matplotlib, seaborn (for visualization)

## Quick Start

### 1. Training on a single dataset
```bash
python main.py --name Quake_Smart-seq2_Trachea
```

### 2. Available datasets
Datasets are automatically loaded from configured paths. Supported datasets include:
| Dataset | Description |
|---------|-------------|
| `Quake_10x_Bladder` | 10x Genomics bladder cells |
| `Quake_10x_Limb_Muscle` | 10x Genomics limb muscle |
| `Quake_10x_Spleen` | 10x Genomics spleen |
| `Quake_Smart-seq2_Trachea` | Smart-seq2 trachea |
| `Quake_Smart-seq2_Limb_Muscle` | Smart-seq2 limb muscle |
| `Quake_Smart-seq2_Diaphragm` | Smart-seq2 diaphragm |
| `Romanov` | Romanov dataset |
| `Klein` | Klein dataset |
| `Muraro` | Muraro pancreatic dataset |
| `Pollen` | Pollen dataset |
| `Chung` | Chung dataset |
| `Baron1-Baron4` | Baron human pancreatic datasets |

### 3. Testing with saved models
After training, the best model is automatically saved to `saved_models/{dataset}/best_model.pt` when ARI improves.

```bash
python test.py --name Quake_Smart-seq2_Trachea
```

You can also specify a custom model path:
```bash
python test.py --name Quake_Smart-seq2_Trachea 
```

### 4. Key hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epoch` | 400 | Number of training epochs |
| `--lr` | 0.003 | Learning rate |
| `--temperature` | 0.07 | Contrastive learning temperature |
| `--dropout` | 0.9 | Data augmentation dropout rate |
| `--batch_size` | 10000 | Batch size |
| `--select_gene` | 2000 | Number of highly variable genes to select |
| `--enc_1/2/3` | 200/40/60 | Encoder layer dimensions |
| `--mlp_dim` | 40 | MLP projection dimension |
| `--m` | 0.5 | Momentum coefficient for key encoder |
| `--k` | 0.5 | High-confidence adjacency threshold |
| `--n_neighbors` | 3 | Number of k-NN neighbors |




## Citation
If you find ScRGCL useful for your research, please cite:

```bibtex
@article{scRGCL2024,
  title={ScRGCL: A graph contrastive learning framework for single-cell RNA-seq clustering},
  author={},
  journal={Bioinformatics},
  year={2024}
}
```

## License
This code is available for non-commercial academic purposes.
