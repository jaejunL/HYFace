import os
import glob
import json
import random
import argparse
import numpy as np

import torch
import torchaudio
import torch.nn as nn
from torchvision import transforms
from transformers import HubertConfig, HubertModel

from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn, F2F0)
from modules.mel_processing import spectrogram_torch
from utils import load_wav_to_torch

def load_config(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    args = utils.HParams(**config)
    return args

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)
        
class HYFace_Conversion(object):
    def __init__(self, hyface_netg_config, hyface_netg_weight_path, hyface_netf_config, hyface_netf_weight_path):
        self.build_hyface(hyface_netg_config, hyface_netg_weight_path, hyface_netf_config, hyface_netf_weight_path)
        self.trans = transforms.Compose([transforms.Resize((112,112), interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(112), transforms.ToTensor()])    
 
    def load_img(self, img_dir):
        img = Image.open(img_dir)
        img_tensor = self.trans(img)
        return img_tensor
    
    def extract_c(self, wav_dir):
        audio, sampling_rate = load_wav_to_torch(wav_dir)
        if sampling_rate != 16000:
            raise ValueError("Sample Rate not match")
        audio_norm = audio / 32768.0
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, 1024, 16000, 160, 1024, center=False)
        c = self.hyface_netc(torchaudio.load(wav_dir)[0].to('cuda:0'))["last_hidden_state"].detach().squeeze()
        c = utils.repeat_expand_2d(c.transpose(-1,-2), spec.shape[-1], mode="nearest")
        return c.unsqueeze(0)           

    def build_hyface(self, netg_args, netg_path, netf_args, netf_path):
        self.hyface_netc = HubertModelWithFinalProj.from_pretrained("lengyue233/content-vec-best")
        self.hyface_netc = self.hyface_netc.to('cuda:0')
        self.hyface_netc.eval()
        
        self.hyface_netg = SynthesizerTrn(netg_args.data.filter_length // 2 + 1,
                netg_args.train.segment_size // netg_args.data.hop_length,
                **netg_args.model)
        self.hyface_netg, _, _, _ = utils.load_checkpoint(netg_path, self.hyface_netg, None)
        self.hyface_netg = self.hyface_netg.to('cuda:0')
        self.hyface_netg.eval()
        
        self.hyface_netf = F2F0(imgsize=112)
        self.hyface_netf, _, _, _ = utils.load_checkpoint(netf_path, self.hyface_netf, None)
        self.hyface_netf = self.hyface_netf.to('cuda:0')
        self.hyface_netf.eval()

    def synth_hyface(self, wavdir_s, imgdir_t):
        source_c = self.extract_c(wavdir_s).to('cuda:0')
        target_f = self.load_img(imgdir_t).to('cuda:0')
        target_f0, _ = self.hyface_netf.infer(target_f.unsqueeze(0))
        synth, _ = self.hyface_netg.infer(source_c, None, None, 0, avgf0=target_f0, face=target_f.unsqueeze(0))
        return synth.detach().cpu()

        
if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    warnings.simplefilter(action='ignore', category=UserWarning) 
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser()
    # set parameters
    parser.add_argument('--model_root', type=str, default="/disk3/jaejun/HYFace", help='your-HYFace-model-root')
    parser.add_argument('--sample_write_root', type=str, default='.')
    base_args = parser.parse_args()
    
    
    
    