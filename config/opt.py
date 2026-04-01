import argparse

parser = argparse.ArgumentParser(description='ScRGCL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', type=str, default='Quake_Smart-seq2_Trachea')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--epoch', type=int, default=400)  # 200
parser.add_argument('--select_gene', type=int, default=2000)  # 2000
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--dropout', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.003)  # 0.2
parser.add_argument('--m', type=float, default=0.5)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--lambda_i', type=float, default=1.0)
parser.add_argument('--lambda_c', type=float, default=1.0)
parser.add_argument('--lambda_p', type=float, default=1.0)
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--enc_1', type=int, default=200)
parser.add_argument('--enc_2', type=int, default=40)
parser.add_argument('--enc_3', type=int, default=60)
parser.add_argument('--mlp_dim', type=int, default=40)
parser.add_argument('--cluster_methods', type=str, default="KMeans")
parser.add_argument('--dataid', type=int, default=0)
parser.add_argument('--n_neighbors', type=int, default=3)
parser.add_argument('--k', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--test', type=int, default=0)
# 灵敏度
parser.add_argument('--param', type=str, default='n_neighbors')
parser.add_argument('--sensitivity_dataset', type=str, default='None')


args = parser.parse_args()

def reset_args(args):
    print("before reset_args:\n", args)

    if args.name in ['Quake_10x_Bladder']:
        args.temperature = 0.2
        args.k = 0.6
        args.n_neighbors = 6
        args.batch_size = 1024
        
        args.epoch =260
    elif args.name in ['Quake_10x_Limb_Muscle']:
        args.temperature = 0.6
        args.k = 0.7
        args.n_neighbors = 5
        args.batch_size = 256
        
    elif args.name in ['Quake_10x_Spleen']:
        args.temperature = 0.7
        args.k = 0.9
        args.n_neighbors = 6
        args.batch_size = 1024
        
        args.epoch = 100
    elif args.name in ['Quake_Smart-seq2_Diaphragm']:
        args.temperature = 0.2
        args.k = 0.9
        args.n_neighbors = 7
        args.batch_size = 256
        
        args.epoch = 200
    elif args.name in ['Quake_Smart-seq2_Limb_Muscle']:
        # args.batch_size = 1500
        # args.n_neighbors = 3
        # args.lambda_c = 0.01
        # args.lambda_p = 0.01
        # args.temperature = 0.3
        # args.k = 0.7
        args.temperature = 0.9
        args.k = 0.5
        args.n_neighbors = 8
        args.batch_size = 1024
        
        args.epoch = 200
        
    elif args.name in ['Romanov']:
        args.temperature = 0.2
        args.k = 0.8
        args.n_neighbors = 5
        args.batch_size = 1500
    elif args.name in ['Klein']:
        args.temperature = 0.2
        args.k = 0.7
        args.n_neighbors = 4
        args.batch_size = 256
        
#         args.temperature = 0.7
# args.k = 0.5
# args.n_neighbors = 4
# args.batch_size = 1024
        
        args.epoch =400
    elif args.name in ['Quake_Smart-seq2_Trachea']:               
        args.temperature = 0.6
        args.k = 0.6
        args.n_neighbors = 6
        args.batch_size = 400
        #  {'n_neighbors': 6, 'k': 0.7, 'batch_size': 256, 'temperature': 0.3}
    elif args.name in ['Pollen']:        
        args.temperature = 0.3
        args.k = 0.4
        args.n_neighbors = 4
        args.batch_size = 256
    elif args.name in ['Chung']:
        # 59.66	43.84	63.74	67.17	70.53
        args.temperature = 0.6
        args.k = 0.7
        args.n_neighbors = 6
        args.batch_size = 512
        
        args.temperature = 0.7
        args.k = 0.4
        args.n_neighbors = 8
        args.batch_size = 512
        # args.seed = 0
        args.epoch = 200
    elif args.name in ['Baron1']:
        args.temperature = 0.6
        args.k = 0.7
        args.n_neighbors = 9
        args.batch_size = 1024
    elif args.name in ['Baron2']:
        args.temperature = 0.7
        args.k = 0.9
        args.n_neighbors = 5
        args.batch_size = 400
    elif args.name in ['Baron3']:
        # args.temperature = 0.7
        # args.k = 0.8
        # args.n_neighbors = 7
        # args.batch_size = 512
        
        args.temperature = 0.7
        args.k = 0.8
        args.n_neighbors = 6
        args.batch_size = 512
        
        args.epoch = 200
    elif args.name in ['Baron4']:
        args.temperature = 0.7
        args.k = 0.6
        args.n_neighbors = 8
        args.batch_size = 256
    elif args.name in ['Muraro']:
        args.temperature = 0.3
        args.k = 0.7
        args.n_neighbors = 8
        args.batch_size = 1024    
    
        args.epoch = 100
    print("after reset_args:\n", args)


def test_ablation(args):
    print("before test_ablation:\n", args)
    if args.test == 1:
        args.lambda_i = 0 # mask loss_instance
    elif args.test == 2:
        args.lambda_c = 0 # mask loss_cluster
    elif args.test == 3:
        args.lambda_p = 0 # mask loss_prototype
    elif args.test == 4:
        args.lambda_c = 0 # mask loss_cluster
        args.lambda_p = 0 # mask loss_prototype
    print("after test_ablation:\n", args)


