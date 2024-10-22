import os
import sys
import json
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
import utils
from models import F2F0
from data_utils import Dataset_Sub

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split)[0] for line in f]
    return filepaths_and_text


if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    warnings.simplefilter(action='ignore', category=UserWarning) 
    
    parser = argparse.ArgumentParser()
    # set parameters
    parser.add_argument('--model_root', type=str, default="/disk3/jaejun/HYFace", help='your-HYFace-model-root')
    parser.add_argument('--aud_root', type=str, default='/disk2/LRS3/modified/auds', help='your-LRS3-aud-root')
    parser.add_argument('--img_root', type=str, default='/disk2/LRS3/modified_original/imgs', help='your-LRS3-img-root')
    parser.add_argument('--target_gender', type=str, default='male', help='target gender')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size of test set Dataloader')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    base_args = parser.parse_args()

    # Setting
    model_name = 'sub'

    # Data
    genders = ['male', 'female']
    speakers = {}
    for gender in genders:
        speakers[gender] = load_filepaths_and_text(f'evaluation/{gender}_speakers.txt')

    testset = Dataset_Sub(base_args.aud_root, base_args.img_root, typ="test")
    testloader = DataLoader(testset, shuffle=False, batch_size=1, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    net_sub = F2F0(imgsize=112)

    epochs = [50, 100, 200, 400]
    for epoch in epochs:
        check_path = os.path.join(base_args.model_root, model_name, f'checkpoints/G_{epoch}.pth')
        net_sub, _, _, _ = utils.load_checkpoint(check_path, net_sub, None)
        net_sub = net_sub.to('cuda:0')
        net_sub.eval()    
        for target_gender in genders:
            deviations = []
            for i, (avgf0, face, img_dir) in enumerate(testset):
                speaker = os.path.basename(os.path.dirname(os.path.dirname(img_dir)))
                if speaker not in speakers[target_gender]:
                    continue
                predf0_out, _ = net_sub.infer(face.unsqueeze(0).to('cuda:0'))
                deviation = torch.abs(avgf0 - torch.mean(predf0_out.detach().cpu()))
                deviations.append(deviation.numpy())
                print(f'Model epoch:{epoch}, gender:{target_gender}, iter:{i}, deviation:{deviation}', end='\r')
            print('\n')
            print(f'Model epoch:{epoch}, Gender:{target_gender}, Average deviation: {np.round(np.mean(deviations),2)}')
        
# CUDA_VISIBLE_DEVICES=0 python evaluation/sub_eval.py