import os
import json
import argparse
import datetime
from train import train
import utils

if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    warnings.simplefilter(action='ignore', category=UserWarning) 
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser()
    # set parameters
    parser.add_argument('--write_root', type=str, default="/disk3/jaejun/HYFace", help='HYFace model saving directory')
    parser.add_argument('--model', default="main", type=str, help="'main' for VC network, 'sub' for pitch estimation network")
    parser.add_argument('--test', default='false', type=utils.str2bool, help='whether test or not')
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
    # if base_args.test == True:
        # os.environ['WANDB_MODE'] = "dryrun"
    # os.environ['WANDB_RUN_ID'] = f'HYFace_{base_args.model}'

    base_args.base_dir = os.path.join(base_args.write_root, base_args.model)
    os.makedirs(os.path.join(base_args.base_dir, 'checkpoints'), exist_ok=True)
    configs_dir = f'configs/{base_args.model}.json'
    with open(configs_dir, "r") as f:
        data = f.read()
    config = json.loads(data)
    
    args = utils.HParams(**config)
    args.base_args = base_args

    train(args)

# python main.py --write_root=/disk3/jaejun/HYFace --model=main --gpus=1,2,3,4 --port=0104 --test=1
# python main.py --write_root=/disk3/jaejun/HYFace --model=sub --gpus=10,11 --port=1011 --test=1
