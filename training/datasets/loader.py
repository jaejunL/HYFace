import os
import time
import wave
import random
import numpy as np
from typing import Dict, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from utils import data_utils, audio_utils

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                args,
                meta_root = 'filelists',
                mode='train',
                datasets=['vctk'],
                sample_rate = 16000, 
                ):
        self.args = args
        self.mode = mode
        self.datasets = datasets
        self.sample_rate = sample_rate
        self.max_sec = 4
        self.max_len = sample_rate * self.max_sec
        self.data_files = []
        for dset in datasets:
            meta_file_path = os.path.join(meta_root, '{}_{}.txt').format(dset, mode)
            files = data_utils.load_text(meta_file_path)
            self.data_files += files

    def __getitem__(self, index):
        # audio load
        audio_path = self.data_files[index]
        audio = audio_utils.load_wav(path=audio_path, max_len=self.max_len)
        # hubert load
        hubert_path = audio_path.replace('wav16_cleaned', 'hubert_soft').replace('.wav', '.emb')
        hubert_emb = torch.load(hubert_path).squeeze(0)[:self.max_sec*50]
        return audio, hubert_emb

    def collate(self, bunch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Collate bunch of datum to the batch data.
        Args:
            bunch: B x [np.float32; [T]], speech signal.
        Returns:
            batch data.
                speeches: [np.float32; [B, T]], speech signal.
                lengths: [np.long; [B]], speech lengths.
        """
        # [B]
        frame_lengths = np.array([hubert_emb.shape[0]*2 for _, hubert_emb in bunch])
        # []
        max_framelen = frame_lengths.max()
        # [B, T]
        audio_pad = np.stack([
            np.pad(audio[:frame_lengths[i]*160], [0, max_framelen*160 - frame_lengths[i]*160]) for i, (audio, _) in enumerate(bunch)])
        hubert_pad = pad_sequence([hubert_emb for _, hubert_emb in bunch]).transpose(0,1)
        data = {'audio':audio_pad, 'hubert':hubert_pad.transpose(-1,-2), 'frame_lengths':frame_lengths}
        return data

    def __len__(self):
        return len(self.data_files)
