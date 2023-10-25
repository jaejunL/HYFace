import os
import json
import time
import wave
import random
import argparse
import numpy as np
from typing import Dict, Tuple

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.loader import VoxCeleb2
from hyface import Nansy, BShall_Ecapa, BShall_Nimbre
from utils import utils, data_utils, audio_utils


if __name__ == '__main__':
    from hyface import Nansy, BShall_Nimbre, BShall_Ecapa
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', nargs='+', default=None, help='gpus')
    parser.add_argument('--port', default='6000', type=str, help='port')
    parser.add_argument('--n_nodes', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int) # n개의 gpu가 한 node: n개의 gpu마다 main_worker를 실행시킨다.
    parser.add_argument('--rank', default=0, type=int, help='ranking within the nodes')
    base_args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(base_args.gpu)
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = base_args.port
    
    config_path = '/disk3/jaejun/hyface/bshall_ecapa/logs/config.json'
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    args = utils.HParams(**config)
    args.base_args = base_args

    if args.model.generator == "nansy":
        hyface = Nansy(args)
    elif args.model.generator == "bshall" and args.model.timbre.type == "nansy":
        hyface = BShall_Nimbre(args)
    elif args.model.generator == "bshall" and args.model.timbre.type == "ecapa":
        hyface = BShall_Ecapa(args)
        
    # args.meta_root = '/home/jaejun/hyface/training'
    # args.base_dir = '/home/jaejun/temp_jaejun/hyface/bshall_ecapa/checkpoints'

    index = 2000
    checkpoint_path = f'/disk3/jaejun/hyface/bshall_ecapa/checkpoints/G_{index}.pth'
    hyface, _, _, _ = utils.load_checkpoint(checkpoint_path, hyface, None)
    
    
    # valid_filelist = args.data.validation_files
    dataset = VoxCeleb2(args)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8,
                    collate_fn=None, pin_memory=True, drop_last=False,
                    worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    if args.model.generator == "nansy":
        hyface = Nansy(args)
    elif args.model.generator == "bshall" and args.model.timbre.type == "nansy":
        hyface = BShall_Nimbre(args)
    elif args.model.generator == "bshall" and args.model.timbre.type == "ecapa":
        hyface = BShall_Ecapa(args)
    hyface = hyface.to('cuda:0')
    
    hyface.eval()
    for batch_idx, bunch in enumerate(loader):
        audio, paths = bunch
        audio_tensor = torch.tensor(audio, device=torch.device(f'cuda:0'))
        timbres = hyface.analyze_timbre(audio_tensor).detach().cpu().numpy()
        
        for j, timbre in enumerate(timbres):
            write_path = paths[j].replace('original', 'modified/ecapa_16000').replace('.wav','.emb')
            os.makedirs(os.path.dirname(write_path), exist_ok=True)
            np.save(write_path, timbre)
            print(f'percent:{np.round(batch_idx/len(loader)*100,2)}, path:{write_path}', end='\r')