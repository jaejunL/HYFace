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
import torchaudio.transforms as transforms

from speechbrain.pretrained import EncoderClassifier

from networks.timbre import TimbreEncoder
from networks.synthesizer import FrameLevelSynthesizer
from networks.bshall import AcousticModel
from utils import utils, audio_utils, data_utils


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


class Nansy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.logmel = LogMelSpectrogram(sample_rate=args.data.sample_rate, n_fft=args.features.mel.win,
                hop_length=args.features.mel.hop, win_length=args.features.mel.win, n_mels=args.features.mel.bin, center=False)
        self.timbre = TimbreEncoder(args.features.mel.bin,
                    args.model.timbre.global_, args.model.timbre.channels, args.model.timbre.prekernels, args.model.timbre.scale, args.model.timbre.kernels,
                    args.model.timbre.dilations, args.model.timbre.bottleneck, args.model.timbre.hiddens, args.model.timbre.latent, args.model.timbre.timbre,
                    args.model.timbre.tokens, args.model.timbre.heads, args.model.linguistic.hiddens + args.model.timbre.global_, args.model.timbre.slerp)
        self.frame_synth = FrameLevelSynthesizer(args.model.linguistic.hiddens, args.model.timbre.global_,
                    args.model.synthesizer.f_kernels, args.model.synthesizer.f_dilations, args.model.synthesizer.f_blocks,
                    args.model.leak, args.model.dropout, args.features.mel.bin)

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
        timber_global, timber_bank = self.timbre.forward(mel)
        return timber_global, timber_bank

    def synthesize(self,
                   ling: torch.Tensor,
                   timber_global: torch.Tensor,
                   timber_bank: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        # S
        ling_len = ling.shape[-1]
        # [B, ling_hiddens + timb_global, S]
        contents = torch.cat([
            ling, timber_global[..., None].repeat(1, 1, ling_len)], dim=1)
        # [B, timber_global, S]
        timber_sampled = self.timbre.sample_timber(contents, timber_global, timber_bank)
        # [B, ling_hiddens, S]
        frame = self.frame_synth.forward(ling, timber_sampled)
        # [B, T], [B, T]
        return frame

    def forward(self, inputs: torch.Tensor, ling: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        timber_global, timber_bank = self.analyze_timbre(inputs)
        # [B, T]
        synth = self.synthesize(ling, timber_global, timber_bank)
        return synth, timber_global, timber_bank


class BShall_Ecapa(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.logmel = LogMelSpectrogram(sample_rate=args.data.sample_rate, n_fft=args.features.mel.win,
                hop_length=args.features.mel.hop, win_length=args.features.mel.win, n_mels=args.features.mel.bin, center=False)
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
        self.frame_synth = AcousticModel(args.model.linguistic.hiddens, args.model.linguistic.output_dim,
                    args.model.timbre.global_, args.model.timbre.output_dim,
                    args.model.synthesizer.decoder_dim, args.features.mel.bin, True)
        if "bshall" in args.model.pretrained:
            pretrained_ac = torch.hub.load("bshall/acoustic-model:main", "hubert_soft")
            self.load_pretrained_model(pretrained_ac)

    def load_pretrained_model(self, pretrained):
        # Content encoder
        saved_state_dict = pretrained.encoder.state_dict()
        if hasattr(self.frame_synth.content_encoder, 'module'):
            state_dict = self.frame_synth.content_encoder.module.state_dict()
        else:
            state_dict = self.frame_synth.content_encoder.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            try:
                new_state_dict[k] = saved_state_dict[k]
            except:
                print("Param {} is not in the checkpoint".format(k))
                new_state_dict[k] = v
        if hasattr(self.frame_synth.content_encoder, 'module'):
            self.frame_synth.content_encoder.module.load_state_dict(new_state_dict)
        else:
            self.frame_synth.content_encoder.load_state_dict(new_state_dict)

        # Decoder
        pretrained.decoder.prenet = nn.Identity()
        pretrained.decoder.proj = nn.Identity()
        saved_state_dict = pretrained.decoder.state_dict()
        if hasattr(self.frame_synth.decoder, 'module'):
            state_dict = self.frame_synth.decoder.module.state_dict()
        else:
            state_dict = self.frame_synth.decoder.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            try:
                new_state_dict[k] = saved_state_dict[k]
            except:
                print("Param {} is not in the checkpoint".format(k))
                new_state_dict[k] = v
        if hasattr(self.frame_synth.decoder, 'module'):
            self.frame_synth.decoder.module.load_state_dict(new_state_dict)
        else:
            self.frame_synth.decoder.load_state_dict(new_state_dict)

    @torch.no_grad()
    def analyze_timbre(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Analyze the timber informations from inputs.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
        Returns:
            [torch.float32; [B, timb_global]], global timber emebddings.
            [torch.float32; [B, timb_timber, timb_tokens]], timber token bank.
        """
        # [B, timb_global], [B, timb_timber, timb_tokens]
        embeddings = self.classifier.encode_batch(inputs)
        return embeddings

    def synthesize(self,
                   ling: torch.Tensor,
                   timber: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        # S
        ling_len = ling.shape[-1]
        # [B, 1, S]
        timber = timber.repeat(1,ling_len,1)
        # [B, ling_hiddens, S]
        frame = self.frame_synth.forward(ling.transpose(1, 2), timber)
        # [B, T], [B, T]
        return frame.transpose(1, 2)

    def forward(self, inputs: torch.Tensor, ling: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        timbre_global = self.analyze_timbre(inputs)
        # [B, T]
        synth = self.synthesize(ling, timber_global, None)
        return synth, timber_global, timber_bank


class BShall_Nimbre(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.logmel = LogMelSpectrogram(sample_rate=args.data.sample_rate, n_fft=args.features.mel.win,
                hop_length=args.features.mel.hop, win_length=args.features.mel.win, n_mels=args.features.mel.bin, center=False)
        self.timbre = TimbreEncoder(args.features.mel.bin,
                    args.model.timbre.global_, args.model.timbre.channels, args.model.timbre.prekernels, args.model.timbre.scale, args.model.timbre.kernels,
                    args.model.timbre.dilations, args.model.timbre.bottleneck, args.model.timbre.hiddens, args.model.timbre.latent, args.model.timbre.timbre,
                    args.model.timbre.tokens, args.model.timbre.heads, args.model.linguistic.hiddens + args.model.timbre.global_, args.model.timbre.slerp)
        self.frame_synth = AcousticModel(args.model.linguistic.hiddens, args.model.timbre.global_,
                    args.model.synthesizer.decoder_dim, args.features.mel.bin, True)

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
        timber_global, timber_bank = self.timbre.forward(mel)
        return timber_global, timber_bank

    def synthesize(self,
                   ling: torch.Tensor,
                   timber: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        # S
        ling_len = ling.shape[-1]
        timber_global, timber_bank = timber
        # [B, ling_hiddens + timb_global, S]
        contents = torch.cat([
            ling, timber_global[..., None].repeat(1, 1, ling_len)], dim=1)
        # [B, timber_global, S]
        timber_sampled = self.timbre.sample_timber(contents, timber_global, timber_bank)
        # [B, ling_hiddens, S]
        frame = self.frame_synth.forward(ling.transpose(1, 2), timber_sampled.transpose(1, 2))
        # [B, T], [B, T]
        return frame.transpose(1, 2)

    def forward(self, inputs: torch.Tensor, ling: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        timber_global, timber_bank = self.analyze_timbre(inputs)
        # [B, T]
        synth = self.synthesize(ling, timber_global, timber_bank)
        return synth, timber_global, timber_bank