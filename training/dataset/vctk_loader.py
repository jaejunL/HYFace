import re
import os
import sys
import numpy as np
import random
import scipy
import json
import sys
import pickle
from copy import copy
import string

import torch
import torchaudio
import torchaudio.transforms as transforms
import torch.nn.functional as F

sys.path.append('../')
from utils.data_utils import load_audio

# hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")

class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, win_length, n_mels, center):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=n_mels,
            mel_scale="slaney",
        )

    def forward(self, wav):
        wav = F.pad(wav, ((self.n_fft - self.hop_length) // 2, (self.n_fft - self.hop_length) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel.squeeze(0)


"""Multi speaker version"""
class VCTK_dataset(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, args, filelist):
        self.args = args
        self.aud_root_dir = args.data.audio_root_dir
        self.speakers = self.load_text(filelist)
        self.sample_rate = args.data.sample_rate
        self.get_audio_dir()
        random.seed(1234)

        self.logmel = LogMelSpectrogram(sample_rate=args.data.sample_rate, n_fft=args.data.filter_length,
                hop_length=args.data.hop_length, win_length=args.data.win_length, n_mels=args.data.n_mel_channels, center=False)

    def load_text(self, path):
        with open(path, encoding='utf-8') as f:
            text = [line.strip() for line in f]
        return text

    def get_audio_dir(self):
        self.audio_dir = []
        for speaker in self.speakers:
            aud_files = os.listdir(os.path.join(self.aud_root_dir, speaker))
            for aud_file in aud_files:
                self.audio_dir.append(os.path.join(self.aud_root_dir, speaker, aud_file))

    ## Modified version by Jaejun
    def load_utterance(self, index):
        index = int(index)
        audio_path = self.audio_dir[index]
        hubert_path = audio_path.replace('wav16_cleaned', 'hubert_soft').replace('.wav', '.emb')
        ecapa_path = audio_path.replace('wav16_cleaned', 'ecapa').replace('.wav', '.emb')

        wav, _ = torchaudio.load(filepath=audio_path)
        mfccs = self.logmel(wav).squeeze(0).transpose(0,1)

        # mfccs = load_audio(self.args, audio_path, sample_rate=self.args.data.sample_rate)
        mfccs = mfccs[:int(np.floor(mfccs.shape[0]/2)*2),:]
        hubert_emb = torch.load(hubert_path)[:int(mfccs.shape[0]/2),:]
        ecapa_emb = torch.load(ecapa_path)

        return mfccs, hubert_emb, ecapa_emb

    def __getitem__(self, index):
        mfccs, hubert_emb, ecapa_emb = self.load_utterance(index)
        result = {'audio_features':mfccs, 'hubert':hubert_emb, 'ecapa':ecapa_emb}
        return result
    
    def __len__(self):
        return len(self.audio_dir)

    # @staticmethod
    def collate_raw(self, batch):
        batch_size = len(batch)
        audio_features = []
        lengths = []
        hubert_embs = []
        ecapa_embs = []

        audio_features = [ex['audio_features'] for ex in batch]
        hubert_embs = [ex['hubert'] for ex in batch]
        ecapa_embs = [ex['ecapa'] for ex in batch]
        lengths = [int(ex['hubert'].shape[0]*2) for ex in batch]

        result = {'audio_features':audio_features,
                  'hubert':hubert_embs,
                  'ecapa':ecapa_embs,
                  'lengths':lengths}
        return result   