#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

current_path = os.path.dirname(os.path.abspath(__file__))
# 上级目录
parent_path = os.path.dirname(current_path)
print(f"当前路径: {parent_path}")
sys.path.append(parent_path)
from src import train
from src.utils import get_dataset, set_random_seed, DualLogger
from config.opt import args, reset_args
# import sys
# from pathlib import Path

# current_path = os.path.dirname(os.path.abspath(__file__))
# # 上级目录
# parent_path = os.path.dirname(current_path)
# print(f"当前路径: {parent_path}")
# sys.path.append(parent_path)


def format_dropout_tag(dropout, dropout_weak=None, dropout_strong=None):
    if dropout_weak is None and dropout_strong is None:
        return f"dropout{dropout:g}"

    resolved_dropout_weak = dropout if dropout_weak is None else dropout_weak
    resolved_dropout_strong = dropout if dropout_strong is None else dropout_strong
    return f"dropout_w{resolved_dropout_weak:g}_s{resolved_dropout_strong:g}"


def format_value_tag(value):
    return f"{value:g}"


def run_experiment(gene_exp, cluster_number, dataset, real_label, hyperparameter_name, hyperparameter_value, logger=None,
                   dropout_weak=None, dropout_strong=None):
    """
    运行单次实验，设置指定的超参数值
    
    参数:
    - gene_exp: 基因表达矩阵
    - cluster_number: 聚类数量
    - dataset: 数据集名称
    - real_label: 真实标签
    - hyperparameter_name: 超参数名称
    - hyperparameter_value: 超参数值
    - logger: 日志记录器
    
    返回:
    - 包含ARI和NMI等评估指标的结果字典
    """
    # 设置超参数
    if hasattr(args, hyperparameter_name):
        setattr(args, hyperparameter_name, hyperparameter_value)
    
    # 记录当前超参数设置
    if logger:
        logger.write(f"\n当前实验: {hyperparameter_name} = {hyperparameter_value}")
        logger.write(f"args.temperature = {args.temperature}")
        logger.write(f"args.k = {args.k}")
        logger.write(f"args.n_neighbors = {args.n_neighbors}")
        logger.write(f"args.batch_size = {args.batch_size}")
        logger.write(f"args.dropout = {args.dropout}")
        logger.write(f"args.dropout_weak = {dropout_weak}")
        logger.write(f"args.dropout_strong = {dropout_strong}")
        logger.write(f"args.lambda_c = {args.lambda_c}")
        logger.write(f"args.lambda_p = {args.lambda_p}")
        logger.write(f"args.lr = {args.lr}")
        logger.write(f"args.seed = {args.seed}")
        logger.write(f"args.noise = {args.noise}")
        logger.write(f"args.m = {args.m}")

    # 运行模型训练

    current_date = time.strftime("%Y%m%d%H")
    dropout_tag = format_dropout_tag(args.dropout, dropout_weak=dropout_weak, dropout_strong=dropout_strong)
    log_name = (
        f"{dataset}_{hyperparameter_name}_{format_value_tag(hyperparameter_value)}_{dropout_tag}_{current_date}"
    )
    results = train.train_model(gene_exp=gene_exp, cluster_number=cluster_number, real_label=real_label,
                            epochs=args.epoch, lr=args.lr, temperature=args.temperature,
                            dropout=args.dropout, dropout_weak=dropout_weak,
                            dropout_strong=dropout_strong,
                            layers=[args.enc_1, args.enc_2, args.enc_3, args.mlp_dim], batch_size=args.batch_size,
                            m=args.m, lambda_i=args.lambda_i, lambda_c=args.lambda_c, lambda_p=args.lambda_p,
                            k=args.k, n_neighbors=args.n_neighbors, save_pred=True, noise=args.noise, logger=logger, dataset=dataset,
                            save_fig_flag=False, log_name=log_name)
    
    # 记录结果
    if logger:
        logger.write(f"============= 结果 =============")
        logger.write(f"best   {results['epochq']}")
        logger.write(f"ARI    {results['ariq']*100:.2f}")
        logger.write(f"NMI    {results['nmiq']*100:.2f}")
    
    return results

def sensitivity_analysis(hyperparameter_name, value_range, dataset_name=None, seed=100, dropout_equal=False):
    """
    对指定超参数进行敏感度分析
    
    参数:
    - hyperparameter_name: 超参数名称
    - value_range: 超参数值范围列表
    - dataset_name: 数据集名称，默认为None（使用args中的设置）
    - seed: 随机种子
    - dropout_equal: 当超参数为 dropout 时，将 dropout_weak 和 dropout_strong 设置为相同值
    """
    # 重置参数
    reset_args(args)
    
    # 设置数据集
    if dataset_name:
        args.sensitivity_dataset = dataset_name
    
    # 设置随机种子
    set_random_seed(seed)
    args.seed = seed
    
    # 获取数据集
    gene_exp, real_label = get_dataset(args.sensitivity_dataset)
    cluster_number = np.unique(real_label).shape[0]
    
    # 创建日志记录器
    current_date = time.strftime("%Y%m%d%H%M")
    if hyperparameter_name == 'dropout':
        if dropout_equal:
            dropout_tag = f"dropout_w{format_value_tag(value_range[0])}-{format_value_tag(value_range[-1])}"
        else:
            dropout_tag = f"dropout_w0_s{format_value_tag(value_range[0])}-{format_value_tag(value_range[-1])}"
    else:
        dropout_tag = format_dropout_tag(args.dropout, args.dropout_weak, args.dropout_strong)
    sweep_tag = f"{hyperparameter_name}_{format_value_tag(value_range[0])}-{format_value_tag(value_range[-1])}"
    log_name = f"{args.sensitivity_dataset}_{sweep_tag}_{dropout_tag}_{current_date}.txt"
    logger = DualLogger(root=f"./sensitivity_logs/{args.sensitivity_dataset}", filename=log_name, show_time=True)
    
    logger.write(f"============= 超参数敏感度分析: {hyperparameter_name} =============")
    logger.write(f"数据集: {args.sensitivity_dataset}")
    logger.write(f"基因表达矩阵形状: {gene_exp.shape}")
    logger.write(f"聚类数量: {cluster_number}")
    logger.write(f"超参数值范围: {value_range}")
    
    # 存储结果
    ari_scores = []
    nmi_scores = []
    
    # 对每个超参数值运行实验
    for value in value_range:
        logger.write(f"\n============= 测试 {hyperparameter_name} = {value} =============")
        
        if hyperparameter_name == 'dropout':
            if dropout_equal:
                run_dropout_weak = value
                run_dropout_strong = value
            else:
                run_dropout_weak = 0.0
                run_dropout_strong = value
        else:
            run_dropout_weak = args.dropout_weak
            run_dropout_strong = args.dropout_strong

        # 运行实验
        results = run_experiment(
            gene_exp=gene_exp,
            cluster_number=cluster_number,
            dataset=args.sensitivity_dataset,
            real_label=real_label,
            hyperparameter_name=hyperparameter_name,
            hyperparameter_value=value,
            logger=logger,
            dropout_weak=run_dropout_weak,
            dropout_strong=run_dropout_strong,
        )
        
        # 收集结果
        ari_scores.append(results['ariq'] * 100)
        nmi_scores.append(results['nmiq'] * 100)
        
        logger.write(f"ARI: {results['ariq']*100:.2f}, NMI: {results['nmiq']*100:.2f}")
    
    # 绘制折线图
    plt.figure(figsize=(10, 5))
    
    # ARI折线图
    plt.subplot(1, 2, 1)
    plt.plot(value_range, ari_scores, 'o-', linewidth=2, markersize=8)
    plt.title(f'ARI Score vs param:{hyperparameter_name}')
    plt.xlabel(hyperparameter_name)
    plt.ylabel('ARI Score (%)')
    plt.grid(True)
    
    # NMI折线图
    plt.subplot(1, 2, 2)
    plt.plot(value_range, nmi_scores, 'o-', linewidth=2, markersize=8, color='orange')
    plt.title(f'NMI Score vs param:{hyperparameter_name}')
    plt.xlabel(hyperparameter_name)
    plt.ylabel('NMI Score (%)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    save_dir = f"./sensitivity_logs/{args.sensitivity_dataset}"
    os.makedirs(save_dir, exist_ok=True)
    figure_name = f"{args.sensitivity_dataset}_{sweep_tag}_{dropout_tag}_{current_date}.png"
    plt.savefig(f"{save_dir}/{figure_name}", dpi=300)
    logger.write(f"\n图表已保存至: {save_dir}/{figure_name}")
    
    # 打印最佳结果
    best_ari_idx = np.argmax(ari_scores)
    best_nmi_idx = np.argmax(nmi_scores)
    
    logger.write(f"\n============= 最佳结果 =============")
    logger.write(f"最佳ARI: {ari_scores[best_ari_idx]:.2f}% 在 {hyperparameter_name} = {value_range[best_ari_idx]}")
    logger.write(f"最佳NMI: {nmi_scores[best_nmi_idx]:.2f}% 在 {hyperparameter_name} = {value_range[best_nmi_idx]}")
    
    # 返回结果数据
    return {
        'parameter_name': hyperparameter_name,
        'parameter_values': value_range,
        'ari_scores': ari_scores,
        'nmi_scores': nmi_scores
    }

# 默认超参数范围
DEFAULT_PARAM_RANGES = {
    'temperature': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # 'dropout': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.99],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'k': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'n_neighbors': [2] + list(range(5, 51, 5))
}

def main():
    """主函数，使用默认超参数范围运行敏感度分析"""
    import sys
    
    # 默认参数值
    param = 'dropout'
    # dataset = 'Pollen'
    # dataset = 'Chung'
    dataset = 'Quake_10x_Spleen'
    seed = 100
    
    # 命令行参数（仅用于设置数据集、种子和是否让 dropuot_weak/dropout_strong 相同），否则使用默认值
    dropout_equal = False
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--sensitivity_dataset' and i + 1 < len(sys.argv):
            dataset = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--seed' and i + 1 < len(sys.argv):
            try:
                seed = int(sys.argv[i + 1])
                i += 2
            except ValueError:
                print(f"错误：seed必须是整数，而不是 {sys.argv[i + 1]}")
                return
        elif sys.argv[i] == '--param' and i + 1 < len(sys.argv):
            param_input = sys.argv[i + 1]
            if param_input in DEFAULT_PARAM_RANGES:
                param = param_input
            else:
                print(f"错误：不支持的超参数 '{param_input}'。支持的超参数有: {list(DEFAULT_PARAM_RANGES.keys())}")
                return
            i += 2
        elif sys.argv[i] == '--dropout_equal':
            dropout_equal = True
            i += 1
        else:
            i += 1
    
    # 获取超参数范围
    if param not in DEFAULT_PARAM_RANGES:
        print(f"错误：不支持的超参数 '{param}'。支持的超参数有: {list(DEFAULT_PARAM_RANGES.keys())}")
        return
    
    value_range = DEFAULT_PARAM_RANGES[param]
    
    print(f"开始分析超参数: {param}")
    print(f"超参数值范围: {value_range}")
    if dataset:
        print(f"数据集: {dataset}")
    print(f"随机种子: {seed}")
    
    # 运行敏感度分析
    sensitivity_analysis(
        hyperparameter_name=param,
        value_range=value_range,
        dataset_name=dataset,
        seed=seed,
        dropout_equal=dropout_equal
    )

if __name__ == "__main__":
    main()