## ScRGCL: Single-cell Graph Contrastive Learning for Clustering
[![Static Badge](https://img.shields.io/badge/JOURNAL-bioinformatics-blue)](https://doi.org/10.1093/bioinformatics/btaf444)
[![Static Badge](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)

ScRGCL is a self-supervised deep learning method for single-cell RNA-seq clustering. By integrating graph contrastive learning with momentum contrastive learning, ScRGCL learns discriminative cell embeddings without requiring labeled data. The model employs three complementary loss functions—instance-level, cluster-level, and prototype-level contrastive losses—to capture multi-scale similarity structures in single-cell expression data.
![scRGCL architecture overview]<img width="11073" height="4551" alt="framework5" src="https://github.com/user-attachments/assets/e29a833a-485a-44d2-8f66-2048016ea6ac" />


## Repository Structure
```bash
scRGCL/
├── ScRGCL.py           # Core model definitions (encoder, projector, ScRGCL class)
├── train.py            # Main training loop and evaluation
├── clustering.py       # K-means clustering with cosine distance
├── evaluation.py       # Clustering metrics (ACC, NMI, ARI, F1)
├── st_loss.py          # Loss functions (RGCLoss, ClusterLoss, InstanceLoss, PrototypeLoss)
├── utils.py            # Data preprocessing, utilities, visualization helpers
├── opt.py              # Command-line argument parser and dataset-specific hyperparameters
├── main.py             # Entry point for single-dataset experiments
├── search.py           # Hyperparameter optimization with Optuna
├── search_multi.py     # Multi-dataset hyperparameter search
├── hyperparameter_sensitivity.py  # Parameter sensitivity analysis
├── demo_test.py        # Demo/testing script
├── test.py             # Load saved model and evaluate
└── README.md
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
python main.py --name Quake_Smart-seq2_Trachea --dataid 8
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
python test.py --name Quake_Smart-seq2_Trachea --model_path /path/to/model.pt
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




## Method Overview

ScRGCL consists of three main components:

1. **Momentum Contrastive Encoder**: Two encoders (query and key) with momentum update, where the key encoder is updated via exponential moving average of the query encoder.

2. **Three-Pronged Contrastive Loss**:
   - **Instance Loss**: Contrastive loss at the single-cell level
   - **Cluster Loss**: Contrastive loss at the cluster level using pseudo-labels from k-means
   - **Prototype Loss**: Prototype-based contrastive learning to refine cluster boundaries

3. **High-Confidence Graph Construction**: Uses k-means pseudo-labels to construct a confidence-weighted adjacency matrix, which is combined with k-NN similarity graph for robust clustering.

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
