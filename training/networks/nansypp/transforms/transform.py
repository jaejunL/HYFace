from typing import Optional

import os
import json
import numpy as np
import librosa

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

from .cqt import CQT2010v2

class ConstantQTransform(nn.Module):
    """Constant Q-Transform.
    """
    def __init__(self,
                 hop_length: int,
                 fmin: float,
                 bins: int,
                 bins_per_octave: int,
                 sr: int = 16000):
        """Initializer.
        Args:
            strides: the number of the samples between adjacent frame.
            fmin: frequency min.
            bins: the number of the output bins.
            bins_per_octave: the number of the frequency bins per octave.
            sr: sampling rate.
        """
        super().__init__()
        # unknown `strides`
        # , since linguistic information is 50fps, strides could be 441
        # fmin=32.7(C0)
        # bins=191, bins_per_octave=24
        # , fmax = 2 ** (bins / bins_per_octave) * fmin
        #        = 2 ** (191 / 24) * 32.7
        #        = 8132.89
        self.resampler = T.Resample(sr, sr*2)
        self.cqt = CQT2010v2(
            sr=sr*2,
            hop_length=hop_length*2,
            fmin=fmin,
            n_bins=bins,
            bins_per_octave=bins_per_octave,
            trainable=False,
            pad_mode="constant",
            earlydownsample=False,
            output_format='Magnitude',
            verbose=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply CQT on inputs.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
        Returns:
            [torch.float32; [B, bins, T / strides]], CQT magnitudes.
        """
        resampled_inputs = self.resampler(inputs)
        return self.cqt(resampled_inputs)

class LogMelSpectrogram(nn.Module):
    """log-Mel scale spectrogram.
    """
    def __init__(self,
                 strides: int,
                 windows: int,
                 mel: int,
                 fmin: int = 0,
                 fmax: Optional[int] = 8000,
                 sr: int = 16000):
        """Initializer.
        Args:
            strides: hop length, the number of the frames between adjacent windows.
            windows: length of the windows.
            mel: size of the mel filterbanks.
            fmin, fmax: minimum, maximum frequency,
                if fmax is None, use half of the sample rate as default.
            sr: sample rate.
        """
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=windows,
            hop_length=strides, f_min=fmin, f_max=fmax, n_mels=mel)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Generate the log-mel scale spectrogram.
        Args:
            audio: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
        Returns:
            [torch.float32; [B, mel, T / strides]], log-mel spectrogram
        """
        # [B, mel, T / strides]
        return torch.log(self.melspec(audio) + 1e-7)