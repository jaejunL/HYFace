from typing import Any, Dict, Optional, Tuple

import os
import sys
import json
import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

from nansy_networks.wav2vec2 import Wav2Vec2Wrapper
from nansy_networks.linguistic import LinguisticEncoder
from nansy_networks.pitch import PitchEncoder
from nansy_networks.timbre import TimbreEncoder
from nansy_networks.synthesizer import FrameLevelSynthesizer, SampleLevelSynthesizer

from transforms.transform import ConstantQTransform, LogMelSpectrogram
from utils import utils, audio_utils, data_utils


class Nansypp(nn.Module):
    """NANSY++: Unified Voice Synthesis with Neural Analysis and Synthesis.
    """
    def __init__(self, args):
        """Initializer.
        Args:
            config: NANSY++ configurations.
        """
        super().__init__()
        self.args = args
        # assume the output channels of wav2vec2.forward is `config.w2v2_channels`
        self.wav2vec2 = Wav2Vec2Wrapper(sr=args.data.sample_rate, linguistic=15, cache=args.pretrain.wav2vec2_cache)
        self.linguistic = LinguisticEncoder(self.wav2vec2.channels,
                    args.model.linguistic.hiddens, args.model.linguistic.preconv, args.model.linguistic.kernels, args.model.leak, args.model.dropout)
        self.cqt = ConstantQTransform(sr=args.data.sample_rate, hop_length=args.features.cqt.hop_length, fmin=args.features.cqt.fmin,
                                      bins=args.features.cqt.n_bins, bins_per_octave=args.features.cqt.bins_per_octave)
        self.cqt_center = (args.features.cqt.n_bins - args.model.pitch.bins) // 2
        self.pitch = PitchEncoder(
                    args.model.pitch.bins, args.model.pitch.prekernels, args.model.pitch.kernels, args.model.pitch.channels, args.model.pitch.blocks, args.model.pitch.gru, args.model.pitch.hiddens, args.model.pitch.f0_bins)
        self.register_buffer('pitch_bins', 
                    # linear space in log-scale
                    torch.linspace(np.log(args.model.pitch.start), np.log(args.model.pitch.end), args.model.pitch.f0_bins).exp())
        self.logmel = LogMelSpectrogram(args.features.mel.hop, args.features.mel.win, args.features.mel.bin,
                    args.features.mel.fmin, args.features.mel.fmax, args.data.sample_rate)
        self.timbre = TimbreEncoder(args.features.mel.bin,
                    args.model.timbre.global_, args.model.timbre.channels, args.model.timbre.prekernels, args.model.timbre.scale, args.model.timbre.kernels,
                    args.model.timbre.dilations, args.model.timbre.bottleneck, args.model.timbre.hiddens, args.model.timbre.latent, args.model.timbre.timbre,
                    args.model.timbre.tokens, args.model.timbre.heads, args.model.linguistic.hiddens + args.model.timbre.global_ + 3, args.model.timbre.slerp)
        self.frame_synth = FrameLevelSynthesizer(args.model.linguistic.hiddens, args.model.timbre.global_,
                    args.model.synthesizer.f_kernels, args.model.synthesizer.f_dilations, args.model.synthesizer.f_blocks,
                    args.model.leak, args.model.dropout)
        self.sample_synth = SampleLevelSynthesizer(
                    args.features.cqt.hop_length * args.model.synthesizer.sample_rate / args.data.sample_rate, args.model.synthesizer.sample_rate,
                    args.model.synthesizer.s_channels, args.model.linguistic.hiddens, args.model.synthesizer.s_kernels,
                    args.model.synthesizer.s_dilation_rate, args.model.synthesizer.s_layers, args.model.synthesizer.s_cycles)

    def analyze_pitch(self, inputs: torch.Tensor, index: Optional[int] = None)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                
        cqt = self.cqt(inputs)
        freq = self.args.model.pitch.bins
        if index is None:
            index = self.cqt_center
        # [B, N, f0_bins], [B, N], [B, N]
        pitch_bins, p_amp, ap_amp = self.pitch.forward(cqt[:, index:index + freq])
        # [B, N]
        pitch = (pitch_bins * self.pitch_bins).sum(dim=-1)
        # [B, cqt_bins, N], [B, N]
        return cqt, pitch, p_amp, ap_amp

    def analyze_linguistic(self, inputs: torch.Tensor) -> torch.Tensor:
        """Analyze the linguistic informations from inputs.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
        Returns:
            [torch.float32; [B, ling_hiddens, S]], linguistic informations.
        """
        # [B, S, w2v2_channels]
        w2v2 = self.wav2vec2.forward(inputs)
        # [B, ling_hiddens, S]
        return self.linguistic.forward(w2v2.transpose(1, 2))    

    def analyze_timbre(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Analyze the timber informations from inputs.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
        Returns:
            [torch.float32; [B, timb_global]], global timber emebddings.
            [torch.float32; [B, timb_timber, timb_tokens]], timber token bank.
        """
        # [B, mel, T / mel_hop]
        mel = self.logmel.forward(inputs)
        # [B, timb_global], [B, timb_timber, timb_tokens]
        return self.timbre.forward(mel)

    def analyze(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze the input signal.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
        Returns;
            analyzed featuers,
                cqt: [torch.float32; []], CQT features.
                pitch, p_amp, ap_amp: [torch.float2; [B, N]],
                    frame-level pitch and amplitude sequence.
                ling: [torch.float32; [B, ling_hiddens, S]], linguistic informations.
                timber_global: [torch.float32; [B, timb_global]], global timber emebddings.
                timber_bank: [torch.float32; [B, timb_timber, timb_tokens]], timber token bank.
        """
        # [], [B, N]
        cqt, pitch, p_amp, ap_amp = self.analyze_pitch(inputs)
        # [B, ling_hiddens, S]
        ling = self.analyze_linguistic(inputs)
        # [B, timb_global], [B, timb_timber, timb_tokens]
        timber_global, timber_bank = self.analyze_timber(inputs)
        return {
            'cqt': cqt,
            'pitch': pitch,
            'p_amp': p_amp,
            'ap_amp': ap_amp,
            'ling': ling,
            'timber_global': timber_global,
            'timber_bank': timber_bank}

    def synthesize(self,
                   pitch: torch.Tensor,
                   p_amp: torch.Tensor,
                   ap_amp: torch.Tensor,
                   ling: torch.Tensor,
                   timber_global: torch.Tensor,
                   timber_bank: torch.Tensor,
                   noise: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesize the signal.
        Args:
            pitch, p_amp, ap_amp: [torch.float32; [B, N]], frame-level pitch, amplitude sequence.
            ling: [torch.float32; [B, ling_hiddens, S]], linguistic features.
            timber_global: [torch.float32; [B, timb_global]], global timber.
            timber_bank: [torch.float32; [B, timb_timber, timb_tokens]], timber token bank.
            noise: [torch.float32; [B, T]], predefined noise for excitation signal, if provided.
        Returns:
            [torch.float32; [B, T]], excitation and synthesized speech signal.
        """
        # S
        ling_len = ling.shape[-1]
        # [B, 3, S]
        pitch_rel = F.interpolate(torch.stack([pitch, p_amp, ap_amp], dim=1), size=ling_len)
        # [B, 3 + ling_hiddens + timb_global, S]
        contents = torch.cat([
            pitch_rel, ling, timber_global[..., None].repeat(1, 1, ling_len)], dim=1)
        # [B, timber_global, S]
        timber_sampled = self.timbre.sample_timber(contents, timber_global, timber_bank)
        # [B, ling_hiddens, S]
        frame = self.frame_synth.forward(ling, timber_sampled)
        # [B, T], [B, T]
        return self.sample_synth.forward(pitch, p_amp, ap_amp, frame, noise)

    def forward(self, inputs: torch.Tensor, noise: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Reconstruct input audio.
        Args:
            inputs: [torch.float32; [B, T]], input signal.
            noise: [torch.float32; [B, T]], predefined noise for excitation, if provided.
        Returns:
            [torch.float32; [B, T]], reconstructed.
            auxiliary outputs, reference `Nansypp.analyze`.
        """
        features = self.analyze(inputs)
        # [B, T]
        excitation, synth = self.synthesize(
            features['pitch'],
            features['p_amp'],
            features['ap_amp'],
            features['ling'],
            features['timber_global'],
            features['timber_bank'],
            noise=noise)
        # update
        features['excitation'] = excitation
        return synth, features
