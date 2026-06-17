import optuna
import sys
import os

# 1. 获取当前 main.py 所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取上一级目录（也就是项目根目录）的路径
parent_dir = os.path.dirname(current_dir)

# 3. 将根目录加入到系统路径中
sys.path.append(parent_dir)
from src import train
from config.opt import parser
from src.utils import get_dataset
import numpy as np
import time
from src.utils import preprocess, preprocess_h5ad, get_logger, DualLogger, set_random_seed, show_heat_map, select_best_gpu
import pandas as pd 

gpu_id = select_best_gpu()
print(f"选择的 GPU 是: {gpu_id}")
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)



args = parser.parse_args()      # 真正解析命令行


h5_datasets = []
h5ad_datasets = ['Romanov', 'Pancreas_mouse', '10X_PBMC', 'Quake_10x_Bladder', 'Quake_10x_Spleen',
                 "Quake_Smart-seq2_Lung", "Quake_Smart-seq2_Diaphragm", "Quake_Smart-seq2_Trachea", 'Quake_Smart-seq2_Limb_Muscle']
ziscDesk_datasets = ['Muraro', 'Pollen', 'Adam', 'Baron1', 'Baron2', 'Baron3', 'Baron4', 'Baron_mouse1', 'Chung']
merged_dataset = ["human_brain", "Klein", "filtered_mPCTC"]
ca_datasets = ['small_size_4000_Data_Dong2020_Prostate', 'Data_He2021_Prostate', 'small_size_4000_Data_Song2022_Prostate', 'small_size_4000_Data_Chen2021_Prostate']
# 获取当前日期，格式化为YYYYMMDD
current_date = time.strftime("%Y%m%d%H")
# 生成包含实时日期的日志文件名（用于日志文件与训练输出标识）
log_name_txt = f"{args.name}_search_{current_date}_log.txt"
log_name = f"{args.name}_search_{current_date}"
logger = DualLogger(root=f"./training_logs/{args.name}", filename=log_name_txt, show_time=True)



gene_exp, real_label = get_dataset(args.name)
cluster_number = np.unique(real_label).shape[0]


def objective(trial):
    print("searching dataset: ", args.name)
    if args.name in 'merged_annotated_cells':
        args.eval_on_cpu = True
    # 1. 建议参数
    # args.lr         = trial.suggest_categorical('lr', [0.003, 0.008])
    args.temperature= trial.suggest_categorical('temperature', [0.2,0.3, 0.4,0.5,0.6,0.7, 0.8, 0.9])
    args.k          = trial.suggest_categorical('k', [0.4,0.5, 0.6,0.7,0.8,0.9])
    # args.lambda_c   = trial.suggest_categorical('lambda_c', [0.01,1,10,100])
    # args.lambda_p   = trial.suggest_categorical('lambda_p', [0.01,0.1,1,10,100])
    args.n_neighbors= trial.suggest_int('n_neighbors', 4, 9, step=1)
    # args.batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    args.batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    # args.noise      = trial.suggest_categorical('noise', [0, 0.1])

    # logger.write(f"\nargs.temperature = {args.temperature}\nargs.k = {args.k}\nargs.n_neighbors = {args.n_neighbors}\nargs.batch_size = {args.batch_size}\n\nargs.dropout = {args.dropout}\nargs.lambda_c = {args.lambda_c}\nargs.lambda_p = {args.lambda_p}\nargs.lr = {args.lr}\nargs.seed = {args.seed}\nargs.noise = {args.noise}\nargs.m = {args.m}\n")
    set_random_seed(args.seed)   
    # 2. 训练 & 返回验证指标（越大越好）
    results = train.train_model(gene_exp=gene_exp, cluster_number=cluster_number, real_label=real_label,
                                epochs=args.epoch, lr=args.lr, temperature=args.temperature,
                                dropout=args.dropout, dropout_weak=args.dropout_weak,
                                dropout_strong=args.dropout_strong,
                                layers=[args.enc_1, args.enc_2, args.enc_3, args.mlp_dim], batch_size=args.batch_size,
                                m=args.m, lambda_i=args.lambda_i, lambda_c=args.lambda_c, lambda_p=args.lambda_p,
                                k=args.k, n_neighbors=args.n_neighbors, save_pred=True, noise=args.noise,
                                use_cpu=not args.cuda, evaluate_training=True, eval_on_cpu=args.eval_on_cpu,
                                logger=logger, dataset=args.name, save_fig_flag=False, log_name=log_name)
    
    logger.write(f"============= {args.name}: RESULT =============")
    logger.write(f"\nargs.temperature = {args.temperature}\nargs.k = {args.k}\nargs.n_neighbors = {args.n_neighbors}\nargs.batch_size = {args.batch_size}\n\nargs.dropout = {args.dropout}\nargs.dropout_weak = {args.dropout_weak}\nargs.dropout_strong = {args.dropout_strong}\nargs.lambda_c = {args.lambda_c}\nargs.lambda_p = {args.lambda_p}\nargs.lr = {args.lr}\nargs.seed = {args.seed}\nargs.noise = {args.noise}\nargs.m = {args.m}\n")

    logger.write(f"\nbest   {results['epochq']}" +
                f"\nARI    {results['ariq']*100:.2f}" +
                f"\nNMI    {results['nmiq']*100:.2f}" +
                f"\nACC    {results['accq']*100:.2f}" +
                 f"\nF1 {results['f1q']*100:.2f}"
                 )
    return results['ariq']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=80, timeout=86400)   # 跑 200 组或 24 h

print('最佳参数：', study.best_params)
logger.write(f"最佳参数：{study.best_params}")