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
    
    def load_audio(self, args, audio_path, sample_rate):
        info = torchaudio.info(audio_path)
        if info.sample_rate != sample_rate:
            raise ValueError(f'Sample rate wrong')
        frame_diff = info.num_frames - args.data.segment_length
        frame_offset = random.randint(0, max(frame_diff, 0))

        wav, _ = torchaudio.load(filepath=audio_path, frame_offset=frame_offset, num_frames=args.data.segment_length)
        if wav.size(-1) < args.data.segment_length:
            wav = F.pad(wav, (0, args.data.segment_length - wav.size(-1)))

        gain = random.random() * (0.99 - 0.4) + 0.4
        flip = -1 if random.random() > 0.5 else 1
        wav = flip * gain * wav / wav.abs().max()
        return wav

    ## Modified version by Jaejun
    def load_utterance(self, index):
        index = int(index)
        audio_path = self.audio_dir[index]
        # mfccs, audio = load_audio(self.args, audio_path, sample_rate=self.args.data.sample_rate)
        wav = self.load_audio(self.args, audio_path, sample_rate=self.args.data.sample_rate)
        tgt_logmel = self.logmel(wav.unsqueeze(0)).squeeze(0)

        return wav, tgt_logmel

    def __getitem__(self, index):
        wav, tgt_logmel = self.load_utterance(index)
        src_logmel = tgt_logmel.clone()
        return wav, src_logmel, tgt_logmel
    
    def __len__(self):
        return len(self.audio_dir)
