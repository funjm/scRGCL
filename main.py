import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch

from opt import args, reset_args, test_ablation
import numpy as np
import train
from utils import get_dataset, preprocess, preprocess_h5ad, get_logger, DualLogger,set_random_seed, show_heat_map
import warnings
import sys
warnings.simplefilter("ignore")

Tensor = torch.cuda.FloatTensor


if __name__ == "__main__":
    data_dict = {0: 'Quake_10x_Bladder', 1: 'Quake_10x_Limb_Muscle', 2: 'Quake_10x_Spleen',
                 3: 'Quake_Smart-seq2_Diaphragm', 4: 'Quake_Smart-seq2_Limb_Muscle',
                 5: 'Romanov', 6: 'Muraro', 7: 'Klein', 8: 'Quake_Smart-seq2_Trachea',
                 9:'Pollen', 10:'Chung', 11:'Baron1', 12:'Baron2', 13:'Baron3', 14:'Baron4'
                 }
    
    # 判断当前运行环境是否为 IDE（如 PyCharm、VSCode）或命令行终端  
    if "pydevd" in sys.modules or "debugpy" in sys.modules:
        # 在 IDE 调试器中运行
        args.dataid = 9
        # args.epoch = 200
    else:
        # 在命令行终端运行，保持原有 argparse 传入的参数
        pass

    args.name = data_dict[args.dataid]

    reset_args(args)
    # args.epoch = 2000
    # test_ablation(args)

    gene_exp = []
    real_label = []
    dataset = args.name

    # 获取当前日期，格式化为YYYYMMDD
    current_date = time.strftime("%Y%m%d%H")
    # 生成包含实时日期的日志文件名
    log_name = f"{dataset}_main_{current_date}"
    logger = DualLogger(root=f"./training_logs/{dataset}", filename=f"{log_name}.txt", show_time=True)

    for arg, value in vars(args).items():
        logger.write(f"{arg:>15}: {value}")

    logger.write(f"============= {dataset} =============")
    logger.write(f"\nargs.temperature = {args.temperature}\nargs.k = {args.k}\nargs.n_neighbors = {args.n_neighbors}\nargs.batch_size = {args.batch_size}\n\nargs.dropout = {args.dropout}\nargs.lambda_c = {args.lambda_c}\nargs.lambda_p = {args.lambda_p}\nargs.lr = {args.lr}\nargs.seed = {args.seed}\nargs.noise = {args.noise}\nargs.m = {args.m}\n")
    gene_exp, real_label = get_dataset(args.name)


    print(f"The gene expression matrix shape is {gene_exp.shape}")
    logger.write(f"Gene Expression is {gene_exp.shape}")
    cluster_number = np.unique(real_label).shape[0]
    print(f"The real clustering num is {cluster_number} ")
    logger.write(f"Clustering Num is {cluster_number}")

    set_random_seed(args.seed)
    results = train.train_model(gene_exp=gene_exp, cluster_number=cluster_number, real_label=real_label,
                            epochs=args.epoch, lr=args.lr, temperature=args.temperature,
                            dropout=args.dropout, layers=[args.enc_1, args.enc_2, args.enc_3, args.mlp_dim], batch_size=args.batch_size,
                            m=args.m, lambda_i=args.lambda_i, lambda_c=args.lambda_c, lambda_p=args.lambda_p,
                            k=args.k, n_neighbors=args.n_neighbors, save_pred=True, noise=args.noise, logger=logger, dataset=args.name,
                            save_fig_flag=True, log_name=log_name)


    logger.write(f"============= {dataset}: RESULT =============")
    logger.write(f"\nargs.temperature = {args.temperature}\nargs.k = {args.k}\nargs.n_neighbors = {args.n_neighbors}\nargs.batch_size = {args.batch_size}\n\nargs.dropout = {args.dropout}\nargs.lambda_c = {args.lambda_c}\nargs.lambda_p = {args.lambda_p}\nargs.lr = {args.lr}\nargs.seed = {args.seed}\nargs.noise = {args.noise}\nargs.m = {args.m}\n")

    logger.write(f"\nbest   {results['epochq']}" +
                f"\nARI    {results['ariq']*100:.2f}" +
                f"\nNMI    {results['nmiq']*100:.2f}" +
                f"\nACC    {results['accq']*100:.2f}" +
                 f"\nF1 {results['f1q']*100:.2f}"
                 )
