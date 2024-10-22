import os
import sys
import glob
import json
import shutil
import pickle
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
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import HubertConfig, HubertModel
from resemblyzer import preprocess_wav, normalize_volume, trim_long_silences

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn, F2F0)
from modules.mel_processing import spectrogram_torch
from modules.F0Predictor.FCPEF0Predictor import FCPEF0Predictor
import utils
from utils import load_wav_to_torch

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

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

class Evalset(torch.utils.data.Dataset):
    def __init__(self, aud_root, img_root, speakers, source_gender, target_gender, seed=1234):
        
        self.img_dirs = []
        self.aud_dirs = []
        self.target_speakers = []
        random.seed(seed)
        for i, source_speaker in enumerate(speakers[source_gender]):
            wav_dirs = glob.glob(os.path.join(aud_root, 'test', source_speaker, '*.wav'))
            random.shuffle(wav_dirs)
            for j in range(50):
                wav_dir = wav_dirs[j%len(wav_dirs)]
                self.aud_dirs.append(wav_dir)
        for i, target_speaker in enumerate(speakers[target_gender]):
            img_dirs = glob.glob(os.path.join(img_root, 'test', target_speaker, '*/*.jpg'))
            random.shuffle(img_dirs)
            for j in range(50):
                img_dir = img_dirs[j%len(img_dirs)]
                self.img_dirs.append(img_dir)
                self.target_speakers.append(target_speaker)
        assert len(self.img_dirs) == len(self.aud_dirs) == 2500

        self.f0_predictor = utils.get_f0_predictor("fcpe",512,44100,device='cpu',threshold=0.05)
        self.trans = transforms.Compose([transforms.Resize((112,112), interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(112), transforms.ToTensor()])    
    
    def load_wav(self, wav_dir):
        audio, sampling_rate = load_wav_to_torch(wav_dir)
        if sampling_rate != 16000:
            raise ValueError("Sample Rate not match")
        audio_norm = audio / 32768.0
        audio_norm = audio_norm.unsqueeze(0)
        return audio_norm

    def load_img(self, img_dir):
        img = Image.open(img_dir)
        img_tensor = self.trans(img)
        return img_tensor
                            
    def __getitem__(self, index):
        q_index, r_index = index // 50, index % 50
        aud_dir = self.aud_dirs[index]
        img_dir = self.img_dirs[int(50*r_index+q_index)]
        aud = self.load_wav(aud_dir)
        img = self.load_img(img_dir)
        target_speaker = self.target_speakers[int(50*r_index+q_index)]
        return aud, img, aud_dir, img_dir, target_speaker
            
    def __len__(self):
        return 50*50

class Evalset_Collate:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        max_wav_len = max([x[0].shape[-1] for x in batch])
        
        wav_padded = torch.FloatTensor(len(batch), max_wav_len)
        face_padded = torch.FloatTensor(len(batch), 3, 112, 112)
        wav_padded.zero_()
        face_padded.zero_()
        
        aud_dirs = []
        img_dirs = []
        gtwav_dirs = []
        target_speakers = []
        for i in range(len(batch)):
            wav = batch[i][0]
            wav_padded[i, :wav.size(1)] = wav
            face = batch[i][1]
            face_padded[i] = face
            aud_dirs.append(batch[i][2])
            img_dirs.append(batch[i][3])
            target_speakers.append(batch[i][4])
        return wav_padded, face_padded, aud_dirs, img_dirs, target_speakers
    
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
        synth, _ = self.hyface_netg.infer(source_c, None, None, 1, avgf0=target_f0, face=target_f)
        return synth.detach().cpu()

class Evaluation(object):
    def __init__(self, aud_root, speakers, avgf0_path):
        self.aud_root = aud_root
        self.speakers = speakers
        self.avgf0_path = avgf0_path
        
    def load_fcpe(self):
        with open(self.avgf0_path, 'rb') as fr:
            self.avgf0 = pickle.load(fr)        
        self.f0_predictor = utils.get_f0_predictor("fcpe",512,44100,device=None,threshold=0.05)
        
    def load_resemble(self):
        from resemblyzer import VoiceEncoder
        self.resemble_encoder = VoiceEncoder()

    def extract_spkemb(self, wav_dir):
        wav = preprocess_wav(wav_dir)
        embed = self.resemble_encoder.embed_utterance(wav)
        return embed
    
    def extract_spkemb_from_wav(self, wav):
        audio_norm_target_dBFS = -30
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
        wav = trim_long_silences(wav)
        embed = self.resemble_encoder.embed_utterance(wav)
        return embed

    def extract_f0(self, wav):
        wav, _ = librosa.effects.trim(wav, top_db=40)
        # normalize peak
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak
        wav44 = librosa.resample(wav, orig_sr=16000, target_sr=44100)
        f0_441, uv = self.f0_predictor.compute_f0_uv(wav44)
        if np.sum(uv) != 0:
            pred_f0 = np.sum(f0_441 * uv) / np.sum(uv)
        else:
            pred_f0 = np.mean(f0_441)
        return pred_f0
                    
    def consistency(self, synth, gt_speaker):
        pred = self.extract_spkemb_from_wav(synth.numpy().squeeze(0))
        consistency = []
        wav_dirs = glob.glob(os.path.join(self.aud_root, 'test', gt_speaker, '*.wav'))
        random.shuffle(wav_dirs)
        for j in range(10):
            gt_dir = wav_dirs[j%len(wav_dirs)]
            gt = self.extract_spkemb(gt_dir)
            consistency.append(pred @ gt)
        return np.mean(consistency)
    
    def rnd_consistency(self, synth, gt_speaker, gt_gender):
        pred = self.extract_spkemb_from_wav(synth.numpy().squeeze(0))
        rnd_consistency = []
        cnt = 0
        random.shuffle(speakers[gt_gender])
        while len(rnd_consistency) < 10:
            target_speaker = self.speakers[gt_gender][cnt]
            cnt += 1
            if gt_speaker == target_speaker:
                continue
            wav_dirs = glob.glob(os.path.join(self.aud_root, 'test', target_speaker, '*.wav'))
            random.shuffle(wav_dirs)
            gt_dir = wav_dirs[0]
            gt = self.extract_spkemb(gt_dir)
            rnd_consistency.append(pred @ gt)
        return np.mean(rnd_consistency)
            
    def f0deviation(self, synth, gt_speaker):        
        pred_f0 = self.extract_f0(synth.numpy().squeeze(0))
        gt_f0 = self.avgf0[gt_speaker]
        return np.abs(pred_f0-gt_f0)

    
if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    warnings.simplefilter(action='ignore', category=UserWarning) 
    
    parser = argparse.ArgumentParser()
    # set parameters
    parser.add_argument('--model_root', type=str, default="/disk3/jaejun/HYFace", help='your-HYFace-model-root')
    parser.add_argument('--aud_root', type=str, default='/disk2/LRS3/modified', help='your-LRS3-aud-root')
    parser.add_argument('--img_root', type=str, default='/disk2/LRS3/modified', help='your-LRS3-img-root')
    parser.add_argument('--main_epoch', type=str, default=300, help='your-main-model-epoch')
    parser.add_argument('--sub_epoch', type=str, default=200, help='your-sub-model-epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of test set Dataloader')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--save_samples', type=int, default=0)
    base_args = parser.parse_args()


    # Data
    genders = ['male', 'female']
    speakers = {}
    for gender in genders:
        speakers[gender] = load_filepaths_and_text(f'evaluation/{gender}_speakers.txt')
            
    # Setting
    avgf0_path = os.path.join(base_args.aud_root, 'avg_mu_test.pickle')
    main_config = load_config('configs/main.json')
    sub_config = load_config('configs/sub.json')

    evaluation = Evaluation(base_args.aud_root, speakers, avgf0_path)
    evaluation.load_resemble()
    evaluation.load_fcpe()
        
    # main_epoch = base_args.main_epoch
    # sub_epoch = base_args.sub_epoch
    main_epochs = [300]
    sub_epochs = [200]
    for main_epoch in main_epochs:
        for sub_epoch in sub_epochs:
            main_ckpt_path = os.path.join(base_args.model_root, 'main', f'checkpoints/G_{main_epoch}.pth')
            sub_ckpt_path = os.path.join(base_args.model_root, 'sub', f'checkpoints/G_{sub_epoch}.pth')
            if base_args.save_samples:
                sample_root = os.path.join(base_args.model_root, 'samples', f'{str(main_epoch)}_{str(sub_epoch)}')
                os.makedirs(sample_root, exist_ok=True)
            hyface_conversion = HYFace_Conversion(main_config, sub_config, main_ckpt_path, sub_ckpt_path)

            random.seed(base_args.seed)
            for target_gender in genders:
                for source_gender in genders:
                    metric = {'consistency':[], 'rnd_consistency':[], 'f0deviation':[]}
                    evalset = Evalset(base_args.aud_root, base_args.img_root, speakers, source_gender, target_gender, seed=base_args.seed)
                    eval_loader = DataLoader(evalset, num_workers=8, shuffle=False, pin_memory=True, batch_size=base_args.batch_size,
                                            collate_fn=Evalset_Collate(), worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
                    for i, (auds, imgs, aud_dirs, img_dirs, target_speakers) in enumerate(eval_loader):
                        print(f'Index:{int(i*base_args.batch_size)}/{len(evalset)}', end='\r')
                        synths = hyface_conversion.synth_hyface(auds.to('cuda:0'), imgs.to('cuda:0'))
                        for j, synth in enumerate(synths):
                            consistency = evaluation.consistency(synth, target_speakers[j])
                            rnd_consistency = evaluation.rnd_consistency(synth, target_speakers[j], target_gender)
                            f0deviation = evaluation.f0deviation(synth, target_speakers[j])
                            metric['consistency'].append(consistency)
                            metric['rnd_consistency'].append(rnd_consistency)
                            metric['f0deviation'].append(f0deviation)
                            if base_args.save_samples:
                                source_speaker = os.path.basename(os.path.dirname(aud_dirs[j]))
                                sample_write_dir = os.path.join(sample_root, target_speakers[j])
                                os.makedirs(sample_write_dir, exist_ok=True)
                                shutil.copy2(aud_dirs[j], os.path.join(sample_write_dir, f'{source_speaker}.wav'))
                                shutil.copy2(img_dirs[j], os.path.join(sample_write_dir, f'{source_speaker}.jpg'))
                                torchaudio.save(os.path.join(sample_write_dir, f'{source_speaker}_synth.wav'), synth, 16000)
                    print(f"Epoch-M:{main_epoch},S:{sub_epoch}, Source Gender:{source_gender}, Target Gender:{target_gender}, Consistency:{np.round(np.mean(metric['consistency']), 5)}, Rnd_consistency:{np.round(np.mean(metric['rnd_consistency']), 5)} F0_deviation:{np.round(np.mean(metric['f0deviation']), 2)}")
            
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_eval.py --batch_size=1
# CUDA_VISIBLE_DEVICES=5 python evaluation/main_eval.py --batch_size=1