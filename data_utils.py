import os
import glob
import random
import pickle
import numpy as np

import torch
import torch.utils.data
import torchaudio.transforms as T

from PIL import Image
import PIL
from torchvision import transforms

import utils
from modules.mel_processing import spectrogram_torch
from utils import load_filepaths_and_text, load_wav_to_torch

class Dataset_Main(torch.utils.data.Dataset):

    def __init__(self, args, aud_dir, img_dir, typ):
        self.args = args
        self.typ = typ
        self.audio_files = glob.glob(os.path.join(aud_dir, typ, '*', '*.wav'))
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.img_dir = os.path.join(img_dir, typ)
        # self.img_dir = os.path.join(img_dir.replace("modified", "modified_original"))
        
        with open(os.path.join(aud_dir, f'avg_mu_{typ}.pickle'), 'rb') as fr:
            self.avgf0s = pickle.load(fr)
            
        self.max_wav_value = args.data.max_wav_value
        self.sampling_rate = args.data.sampling_rate
        self.filter_length = args.data.filter_length
        self.hop_length = args.data.hop_length
        self.win_length = args.data.win_length
        self.unit_interpolate_mode = args.data.unit_interpolate_mode
        self.sampling_rate = args.data.sampling_rate
        self.resampler = T.Resample(16000, self.sampling_rate)
        self.slice_len = 400
        
        self.trans = transforms.Compose([transforms.Resize((112,112), interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(112),
                transforms.ToTensor(),
                ])
                    
    def get_audio(self, filename):
        # filename = filename.replace("modified", "modified_original")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            audio = self.resampler(audio)
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)

        spec = spectrogram_torch(audio_norm, self.filter_length,
                                    self.sampling_rate, self.hop_length, self.win_length,
                                    center=False)
        spec = torch.squeeze(spec, 0)

        speaker = os.path.basename(os.path.dirname(filename))
        clip_name = os.path.basename(filename).replace('.wav','').split('_')[0]
        
        ## f0
        avgf0 = self.avgf0s[speaker]
        avgf0 = torch.FloatTensor(np.array(avgf0,dtype=float)).unsqueeze(0)
        
        f0, uv = np.load(filename.replace('.wav','.f0.npy'),allow_pickle=True)
        f0 = torch.FloatTensor(np.array(f0,dtype=float))
        f0 = utils.repeat_expand_2d(f0[None,:], spec.shape[-1], mode=self.unit_interpolate_mode).squeeze()
        uv = torch.FloatTensor(np.array(uv,dtype=float)) if self.args.data.uv_embedding else None
        uv = utils.repeat_expand_2d(uv[None,:], spec.shape[-1], mode=self.unit_interpolate_mode).squeeze() if self.args.data.uv_embedding else None
        
        ## face
        speaker_dir = os.path.join(self.img_dir, speaker)        
        if len(os.listdir(os.path.join(speaker_dir, clip_name))) < 1:
            print(f'[Warning] There is no face image in speaker-{speaker} & clip-{clip_name}')
        img_name = random.choice(os.listdir(os.path.join(speaker_dir, clip_name)))
        img_dir = os.path.join(self.img_dir, speaker, clip_name, img_name)        
        img = Image.open(img_dir)
        img_tensor = self.trans(img)
        
        ## content
        c = torch.load(filename.replace('.wav','.emb'))
        c = utils.repeat_expand_2d(c.transpose(-1,-2), spec.shape[-1], mode=self.unit_interpolate_mode)

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (c.size(-1), spec.size(-1), f0.shape, filename)
        assert abs(audio_norm.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        spec, c, f0 = spec[:, :lmin], c[:, :lmin], f0[:lmin]
        uv = uv[:lmin] if uv is not None else None
        audio_norm = audio_norm[:, :lmin * self.hop_length]
        return c, f0, spec, audio_norm, uv, avgf0, img_tensor, img_dir

    def random_slice(self, c, f0, spec, audio_norm, uv, avgf0, img_tensor, img_dir):
        if spec.shape[1] > self.slice_len:
            start = random.randint(0, spec.shape[1]-self.slice_len)
            end = start + self.slice_len - 10
            spec, c, f0 = spec[:, start:end], c[:, start:end], f0[start:end]
            uv = uv[start:end] if uv is not None else None
            audio_norm = audio_norm[:, start * self.hop_length : end * self.hop_length]
        return c, f0, spec, audio_norm, uv, avgf0, img_tensor, img_dir

    def __getitem__(self, index):
        return self.random_slice(*self.get_audio(self.audio_files[index]))

    def __len__(self):
        return len(self.audio_files)


class Dataset_Sub(torch.utils.data.Dataset):

    def __init__(self, aud_dir, img_dir, typ):
        self.typ = typ
        # img_dir = os.path.join(img_dir.replace("modified", "modified_original"))
        
        self.clip_dirs = glob.glob(os.path.join(img_dir, typ, '*', '*'))
        random.seed(1234)
        random.shuffle(self.clip_dirs)
        self.img_dir = os.path.join(img_dir, typ)
        
        with open(os.path.join(aud_dir, f'avg_mu_{typ}.pickle'), 'rb') as fr:
            self.avgf0s = pickle.load(fr)
            
        self.trans = transforms.Compose([transforms.Resize((112,112), interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(112),
                transforms.ToTensor(),
                ])
                    
    def get_audio(self, clip_dir):
        speaker = os.path.basename(os.path.dirname(clip_dir))

        ## f0
        avgf0 = self.avgf0s[speaker]
        avgf0 = torch.FloatTensor(np.array(avgf0,dtype=float)).unsqueeze(0)
        
        ## face
        speaker_dir = os.path.join(self.img_dir, speaker)        
        if len(os.listdir(clip_dir)) < 1:
            print(f'[Warning] There is no face image in speaker-{speaker} & clip-{os.path.basename(clip_dir)}')
        img_name = random.choice(os.listdir(clip_dir))
        img_dir = os.path.join(clip_dir, img_name)        
        img = Image.open(img_dir)
        img_tensor = self.trans(img)
        return avgf0, img_tensor, img_dir

    def __getitem__(self, index):
        return self.get_audio(self.clip_dirs[index])

    def __len__(self):
        return len(self.clip_dirs)
    
    
class Collate:

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]),
            dim=0, descending=True)

        max_c_len = max([x[0].size(1) for x in batch])
        max_f0_len = max_c_len
        max_wav_len = max([x[3].size(1) for x in batch])
        max_avgf0_len = 1
        
        lengths = torch.LongTensor(len(batch))

        c_padded = torch.FloatTensor(len(batch), batch[0][0].shape[0], max_c_len)
        f0_padded = torch.FloatTensor(len(batch), max_f0_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_c_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        uv_padded = torch.FloatTensor(len(batch), max_c_len)
        avgf0_padded = torch.FloatTensor(len(batch), max_avgf0_len)
        face_padded = torch.FloatTensor(len(batch), 3, 112, 112)
        
        c_padded.zero_()
        spec_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        uv_padded.zero_()
        avgf0_padded.zero_()
        face_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]
            c_padded[i, :, :c.size(1)] = c
            lengths[i] = c.size(1)

            f0 = row[1]
            f0_padded[i, :f0.size(0)] = f0

            spec = row[2]
            spec_padded[i, :, :spec.size(1)] = spec

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav

            uv = row[4]
            if uv is not None:
                uv_padded[i, :uv.size(0)] = uv
            else:
                uv_padded = None

            avgf0 = row[5]
            avgf0_padded[i, :avgf0.size(0)] = avgf0
        
            face_tensor = row[6]
            face_padded[i] = face_tensor
        return c_padded, f0_padded, spec_padded, wav_padded, uv_padded, avgf0_padded, face_padded, lengths

    
if __name__ == "__main__":
    import json

    config_path = 'configs/base.json'
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    args = utils.HParams(**config)
    trainset = Dataset(args, args.data.aud_dir, args.data.img_dir, typ="pretrain")
    validset = Dataset(args, args.data.aud_dir, args.data.img_dir, typ="trainval")
    print(f'Lengths of Train set:{len(trainset)}, Valid set:{len(validset)}')