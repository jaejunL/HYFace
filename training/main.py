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
    parser.add_argument('--meta_root', type=str, default="/home/jaejun/hyface/training", help='Code running directory')
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json", help='JSON file for configuration')
    parser.add_argument('--init', default='1', type=utils.str2bool, help='1 for initial training, 0 for continuing')
    parser.add_argument('--group_name', default='test', type=str)
    parser.add_argument('--exp_name', default=nowDatetime, type=str)
    parser.add_argument('--arg_save', default='True', type=utils.str2bool, help='argument save or not')
    parser.add_argument('--test', default='false', type=utils.str2bool, help='whether test or not')
    parser.add_argument('--resume', default='false', type=utils.str2bool, help='whether resume or not')
    parser.add_argument('--log_all', default=1, type=int, help='whether wandb log all gpu or only 1')
    # gpu parameters
    parser.add_argument('--gpus', nargs='+', default=None, help='gpus')
    parser.add_argument('--port', default='6000', type=str, help='port')
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

    base_args.base_dir = os.path.join('/disk3/jaejun/hyface', base_args.group_name)

    # Make directories to save results, i.e., codes, checkpoints, analysis
    if base_args.arg_save:
        os.makedirs(os.path.join(base_args.base_dir, 'codes'), exist_ok=True)
        os.makedirs(os.path.join(base_args.base_dir, 'logs/samples'), exist_ok=True)
        os.makedirs(os.path.join(base_args.base_dir, 'checkpoints'), exist_ok=True)
        include_dir = []
        exclude_dir = ['__pycache__', '.ipynb_checkpoints', 'filelists', 'configs', 'wandb', 'preprocess', 'test', 'jupyter', 'praat']
        include_ext = ['.py']
        utils.copy_DirStructure_and_Files(os.getcwd(), include_dir, exclude_dir, include_ext, base_args.base_dir)

    config_path = os.path.join(base_args.meta_root, base_args.config)
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

## This is for hyface
# python main.py -c='configs/base.json' --group_name=230810 --exp_name=train --gpus=1,2,3,4 --workers=8 --port=0104
# python main.py -c='configs/bshall_nimbre.json' --group_name=bshall --exp_name=train2 --gpus=3,4 --workers=8 --port=0304 --test=1
# python main.py -c='configs/bshall_disc.json' --group_name=bshall_disc --exp_name=train2 --gpus=1,2,3,4 --workers=8 --port=0104 --test=1
# python main.py -c='configs/bshall_ecapa.json' --group_name=bshall_ecapa --exp_name=train --gpus=1,2,3,4 --workers=8 --port=0104 --test=1
# python main.py -c='configs/bshall_pretrained_ecapa.json' --group_name=bshall_pretrained_ecapa --exp_name=train --gpus=1,2,3,4 --workers=8 --port=0104 --test=1

# 23.10.19
# python main.py -c='configs/nimbre_large.json' --group_name=nimbre_large --exp_name=train --gpus=3,4,5,6 --workers=8 --port=0304 --test=1
# below is for pos_random
# python main.py -c='configs/nimbre_large.json' --group_name=nimbre_large_pos --exp_name=train --gpus=8,9,10,11 --workers=8 --port=0811 --test=1
 
 

# ## This is for f2v
# 23.09.08
# python main.py -c='configs/f2v_ecapa.json' --group_name=f2v_ecapa --exp_name=train --gpus=3,4,5,6 --workers=8 --port=0306
# python main.py -c='configs/f2v_ecapa_vitpretrained.json' --group_name=f2v_ecapa_vitpretrain --exp_name=train --gpus=8,9,10,11 --workers=8 --port=0811 
# 23.10.21
# python main.py -c='configs/bshall_fimbre.json' --group_name=temp --exp_name=train --gpus=3,4,5,6 --workers=8 --port=0306 --test=1

