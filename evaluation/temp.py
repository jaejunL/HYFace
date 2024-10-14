import os
import sys
import glob
import json
import time
import shutil
import numpy as np
import random
import torch

from PIL import Image
import PIL
import torch.nn as nn
import torchaudio
from torchvision import transforms
from transformers import HubertConfig, HubertModel

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn, F2F0)
from modules.mel_processing import spectrogram_torch
from modules.F0Predictor.FCPEF0Predictor import FCPEF0Predictor
import utils
from utils import load_wav_to_torch

def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text

def random_hetero_speaker(speaker, speaker_bunch):
    random_speaker = random.choice(speaker_bunch) 
    while speaker == random_speaker:
        random_speaker = random.choice(speaker_bunch) 
    return random_speaker

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
        
class Demo_Conversion(object):
    def __init__(self, hyface_netg_config, hyface_netg_weight_path, hyface_netf_config, hyface_netf_weight_path):
        self.build_hyface(hyface_netg_config, hyface_netg_weight_path, hyface_netf_config, hyface_netf_weight_path)
        self.build_fvmvc()
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

    def extract_c2(self, wav_dir, f0_dir):
        audio, sampling_rate = load_wav_to_torch(wav_dir)
        if sampling_rate != 16000:
            raise ValueError("Sample Rate not match")
        audio_norm = audio / 32768.0
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, 1024, 16000, 160, 1024, center=False)

        _, uv = np.load(f0_dir,allow_pickle=True)
        uv = torch.FloatTensor(np.array(uv,dtype=float))
        uv = utils.repeat_expand_2d(uv[None,:], spec.shape[-1], mode="nearest").squeeze()

        c = self.hyface_netc(torchaudio.load(wav_dir)[0].to('cuda:0'))["last_hidden_state"].detach().squeeze()
        c = utils.repeat_expand_2d(c.transpose(-1,-2), spec.shape[-1], mode="nearest")
        return c.unsqueeze(0), uv.unsqueeze(0)
                
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
        
        # self.hyface_netf = F2F0(netf_args.model.timbre.type)
        self.hyface_netf = F2F0(112)
        self.hyface_netf, _, _, _ = utils.load_checkpoint(netf_path, self.hyface_netf, None)
        self.hyface_netf = self.hyface_netf.to('cuda:0')
        self.hyface_netf.eval()

    def build_fvmvc(self):
        pass

    def synth_hyface(self, wavdir_s, imgdir_t):
        source_c = self.extract_c(wavdir_s).to('cuda:0')
        target_f = self.load_img(imgdir_t).to('cuda:0')
        # target_f0 = self.hyface_netf.infer(target_f.unsqueeze(0))
        target_f0, _ = self.hyface_netf.infer(target_f.unsqueeze(0))
        synth, _ = self.hyface_netg.infer(source_c, None, None, 0.35, avgf0=target_f0, face=target_f.unsqueeze(0))
        # synth, _ = self.hyface_netg.infer(source_c, None, None, 0, predict_f0=True, vol=None, avgf0=target_f0, face=target_f.unsqueeze(0))
        return synth.detach().cpu()
        
    def demo2_save(self, wavdir, imgdir_t, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        target_speaker = imgdir_t.split('/')[-3]
        shutil.copy2(wavdir, os.path.join(save_folder, os.path.basename(wavdir)))
        synth = self.synth_hyface(wavdir, imgdir_t).squeeze(0)
        synth_name = f"{os.path.basename(wavdir)[:-4]}_{imgdir_t.split('/')[-2]}_{imgdir_t.split('/')[-1][:-4]}.wav"
        torchaudio.save(os.path.join(save_folder, synth_name), synth, 16000)
        shutil.copy2(imgdir_t, os.path.join(save_folder, target_speaker+'_'+os.path.basename(imgdir_t)))
        return 1
    

img_root = f'/disk2/LRS3/modified_original/imgs_frontal/test'
wav_root = f'/disk2/LRS3/modified_original/wav16_split/test'
spk_root = f'/home/jaejun/sovits/training/filelists/lrs3gender'
speaker_ms = load_filepaths_and_text(os.path.join(spk_root, f'lrs3male_eval.txt'))
speaker_fs = load_filepaths_and_text(os.path.join(spk_root, f'lrs3female_eval.txt'))

netg_index, netf_index = 400, 400
hyface_netg_config_path = '/disk3/jaejun/sovits/avgf0ce3/logs/config.json'
hyface_netg_config = load_config(hyface_netg_config_path)
hyface_netg_weight_path = f'/disk3/jaejun/sovits/avgf0ce3/checkpoints/G_{netg_index}.pth'
# hyface_netf_config_path = '/disk3/jaejun/f0ce/base/logs/config.json'
hyface_netf_config_path = '/home/jaejun/HYFace/configs/sub.json'
# hyface_netf_weight_path = f'/disk3/jaejun/f0ce/base/checkpoints/G_{netf_index}.pth'
hyface_netf_weight_path = f'/disk3/jaejun/HYFace/sub/checkpoints/G_{netf_index}.pth'
hyface_netf_config = load_config(hyface_netf_config_path)

conversion = Demo_Conversion(hyface_netg_config, hyface_netg_weight_path,
                        hyface_netf_config, hyface_netf_weight_path
                        )

img_dirs = []

# for demo 2
trial = 1
save_folder = f'/home/jaejun/sovits/testing/eval/demo/demo_hy2/{trial}'
wav_dir = '/home/jaejun/sovits/testing/eval/homogeneity/source_female_target_male/0/2UStOghblfE/3/00002.wav'

random.shuffle(speaker_ms)
random.shuffle(speaker_fs)
for i in range(2):
    speaker_m = speaker_ms[i]
    img_dirs = glob.glob(os.path.join(img_root, speaker_m[0], '*/*.jpg'))
    img_dir = random.choice(img_dirs)
    conversion.demo2_save(wav_dir, img_dir, save_folder)

    speaker_f = speaker_fs[i]
    img_dirs = glob.glob(os.path.join(img_root, speaker_f[0], '*/*.jpg'))
    img_dir = random.choice(img_dirs)
    conversion.demo2_save(wav_dir, img_dir, save_folder)    