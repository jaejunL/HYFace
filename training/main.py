import os
import json
import argparse
import datetime

from utils import utils
from train import train

if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    warnings.simplefilter(action='ignore', category=UserWarning) 
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser()
    # set parameters
    parser.add_argument('-c', '--config', type=str, default="./configs/test.json", help='JSON file for configuration')
    parser.add_argument('--init', default='1', type=utils.str2bool, help='1 for initial training, 0 for continuing')
    parser.add_argument('--group_name', default='test', type=str)
    parser.add_argument('--exp_name', default=nowDatetime, type=str)
    parser.add_argument('--arg_save', default='True', type=utils.str2bool, help='argument save or not')
    parser.add_argument('--test', default='false', type=utils.str2bool, help='whether test or not')
    parser.add_argument('--resume', default='false', type=utils.str2bool, help='whether resume or not')
    parser.add_argument('--log_all', default=1, type=int, help='whether wandb log all gpu or only 1')
    parser.add_argument('--hubert', default=0, type=utils.str2bool, help='whether use hubert emb or not')
    parser.add_argument('--pretrain', nargs='+', default=['.'], help='which pretrained model to use')
    parser.add_argument('--fixtrain', nargs='+', default=['.'], help='which model to fix')
    # gpu parameters
    parser.add_argument('--gpus', nargs='+', default=None, help='gpus')
    parser.add_argument('--port', default='6056', type=str, help='port')
    parser.add_argument('--n_nodes', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int) # n개의 gpu가 한 node: n개의 gpu마다 main_worker를 실행시킨다.
    parser.add_argument('--rank', default=0, type=int, help='ranking within the nodes')
    base_args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_num) for gpu_num in base_args.gpus])
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = base_args.port
    if base_args.test == True:
        os.environ['WANDB_MODE'] = "dryrun"
    os.environ['WANDB_RUN_ID'] = base_args.group_name + '_' + base_args.exp_name
    if base_args.init == '0':
        os.environ["WANDB_RESUME"] = "must"

    if 'vc' in base_args.config:
        base_args.base_dir = os.path.join('/disk3/jaejun/vc', base_args.group_name)
    else:
        base_args.base_dir = os.path.join('/disk3/jaejun/gaddy', base_args.group_name)
    # Make directories to save results, i.e., codes, checkpoints, analysis
    if base_args.arg_save:
        os.makedirs(os.path.join(base_args.base_dir, 'codes'), exist_ok=True)
        os.makedirs(os.path.join(base_args.base_dir, 'logs/samples'), exist_ok=True)
        os.makedirs(os.path.join(base_args.base_dir, 'checkpoints'), exist_ok=True)
        include_dir = []
        exclude_dir = ['__pycache__', '.ipynb_checkpoints', 'filelists', 'configs', 'wandb', 'preprocess', 'text', 'jupyter', 'text_alignments']
        include_ext = ['.py']
        utils.copy_DirStructure_and_Files(os.getcwd(), include_dir, exclude_dir, include_ext, base_args.base_dir)

    config_path = base_args.config
    config_save_path = os.path.join(base_args.base_dir, "logs", "config.json")
    if base_args.init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)
    args = utils.HParams(**config)
    args.base_args = base_args

    train(args)


# python main.py --c=configs/config.json --group_name=base --exp_name=train --pretrain=v --fixtrain=v --gpus=7
# python main.py --c=configs/config16.json --group_name=base16 --exp_name=train --pretrain=v --fixtrain=v --gpus=8 --port=6067
# python main.py --c=configs/config16.json --group_name=sr16 --exp_name=train2 --pretrain=v --fixtrain=v --gpus=1 --port=6061

# python main.py --c=configs/config16.json --group_name=new_base --exp_name=train --pretrain=v --fixtrain=v --gpus=1 --port=6061
# python main.py --c=configs/config16.json --group_name=hubert --exp_name=train --pretrain=v --fixtrain=v --gpus=2 --port=6062 --hubert=1

# python main.py --c=configs/config_vc.json --group_name=vctk_mask0 --exp_name=train2 --pretrain=v --fixtrain=v --gpus=3 --port=6063 --hubert=1
# python main.py --c=configs/config_vc.json --group_name=vctk_mask1 --exp_name=train2 --pretrain=v --fixtrain=v --gpus=4 --port=6064 --hubert=1
# python main.py --c=configs/config_vc.json --group_name=vctk_mask2 --exp_name=train2 --pretrain=v --fixtrain=v --gpus=5 --port=6065 --hubert=1
