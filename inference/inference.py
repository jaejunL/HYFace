import os
import sys
import json
import random
import argparse
import numpy as np

import librosa
import PIL
from PIL import Image
import torch
import torchaudio
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from transformers import HubertConfig, HubertModel

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn, F2F0)
import utils
from utils import load_wav_to_torch

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

trans = transforms.Compose([transforms.Resize((112,112), interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(112), transforms.ToTensor()])  

def load_config(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    args = utils.HParams(**config)
    return args

def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split)[0] for line in f]
  return filepaths_and_text

def load_img(img_dir):
    img = Image.open(img_dir)
    img_tensor = trans(img)
    return img_tensor.unsqueeze(0)

def load_wav(wav_dir):
    audio, sampling_rate = load_wav_to_torch(wav_dir)
    if sampling_rate != 16000:
        raise ValueError("Sample Rate not match")
    audio_norm = audio / 32768.0
    return audio_norm.unsqueeze(0)
    
class HYFace_Conversion(object):
    def __init__(self, main_config, sub_config, main_ckpt_path, sub_ckpt_path):
        self.build_hyface(main_config, sub_config, main_ckpt_path, sub_ckpt_path)

    def build_hyface(self, main_config, sub_config, main_ckpt_path, sub_ckpt_path):
        self.hyface_netc = HubertModelWithFinalProj.from_pretrained("lengyue233/content-vec-best")
        self.f = self.hyface_netc.to('cuda:0')
        self.hyface_netc.eval()
                
        self.hyface_netg = SynthesizerTrn(main_config.data.filter_length // 2 + 1,
                main_config.train.segment_size // main_config.data.hop_length,
                **main_config.model)
        self.hyface_netg, _, _, _ = utils.load_checkpoint(main_ckpt_path, self.hyface_netg, None)
        self.hyface_netg = self.hyface_netg.to('cuda:0')
        self.hyface_netg.eval()
        
        self.hyface_netf = F2F0(imgsize=112)
        self.hyface_netf, _, _, _ = utils.load_checkpoint(sub_ckpt_path, self.hyface_netf, None)
        self.hyface_netf = self.hyface_netf.to('cuda:0')
        self.hyface_netf.eval()

    def synth_hyface(self, source_c, target_f):
        source_c = self.hyface_netc(source_c)["last_hidden_state"]
        source_c = F.interpolate(source_c.transpose(-1,-2), source_c.shape[1]*2, mode="nearest")
        target_f0, _ = self.hyface_netf.infer(target_f)
        synth, _ = self.hyface_netg.infer(source_c, None, None, 0.35, avgf0=target_f0, face=target_f)
        return synth.detach().cpu().squeeze(0)
    
if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    warnings.simplefilter(action='ignore', category=UserWarning) 
    
    parser = argparse.ArgumentParser()
    # set parameters
    parser.add_argument('--main_model', type=str, default="pretrain/main.pth", help='your-HYFace-main-model-root')
    parser.add_argument('--sub_model', type=str, default="pretrain/sub.pth", help='your-HYFace-sub-model-root')
    parser.add_argument('--source', type=str, default='inference/source.wav', help='your-source-audio-root')
    parser.add_argument('--target', type=str, default='inference/target.jpg', help='your-target-img-root')
    base_args = parser.parse_args()

    # Setting
    main_config = load_config('configs/main.json')
    sub_config = load_config('configs/sub.json')
    hyface_conversion = HYFace_Conversion(main_config, sub_config, base_args.main_model, base_args.sub_model)

    auds = load_wav(base_args.source)
    imgs = load_img(base_args.target)
    synth = hyface_conversion.synth_hyface(auds.to('cuda:0'), imgs.to('cuda:0'))
    torchaudio.save(f'inference/synth.wav', synth, 16000)
    print(f"Successfully synthesized")
    
# CUDA_VISIBLE_DEVICES=0 python inference/inference.py