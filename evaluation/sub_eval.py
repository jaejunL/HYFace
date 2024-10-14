import os
import sys
import json
import time
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

# Setting
aud_dir = '/disk2/LRS3/modified/auds'
img_dir = '/disk2/LRS3/modified_original/imgs'
model_root = '/disk3/jaejun/HYFace'
model_name = 'sub'

# Data
gender = 'female' # male or female
speaker_list = f'evaluation/{gender}_speakers.txt'
speakers = load_filepaths_and_text(speaker_list)

testset = Dataset_Sub(aud_dir, img_dir, typ="test")
testloader = DataLoader(testset, shuffle=False, batch_size=1, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

net_sub = F2F0(imgsize=112)

epochs = [20, 50, 70, 100, 150, 200, 250, 300, 350, 400]
for epoch in epochs:
    check_path = os.path.join(model_root, model_name, f'checkpoints/G_{epoch}.pth')
    net_sub, _, _, _ = utils.load_checkpoint(check_path, net_sub, None)
    net_sub = net_sub.to('cuda:0')
    net_sub.eval()    
    
    # dev_dict = {}
    deviations = []
    for i, (avgf0, face, img_dir) in enumerate(testset):
        speaker = os.path.basename(os.path.dirname(os.path.dirname(img_dir)))
        if speaker not in speakers:
            continue
        predf0_out, _ = net_sub.infer(face.unsqueeze(0).to('cuda:0'))
        deviation = torch.abs(avgf0 - torch.mean(predf0_out.detach().cpu()))
        deviations.append(deviation.numpy())
        print(f'epoch {i}, deviation:{deviation}', end='\r')
    print('\n')
    print(f'Model epoch:{epoch}, Average deviation: {np.mean(deviations)}')
    
# CUDA_VISIBLE_DEVICES=10 python evaluation/sub_eval.py