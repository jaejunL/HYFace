from typing import List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavenet import WaveNet


## Frame-level Synthesizer

class ConvGLU(nn.Module):
    """Dropout - Conv1d - GLU and conditional layer normalization.
    """
    def __init__(self,
                 channels: int,
                 kernels: int,
                 dilations: int,
                 dropout: float,
                 cond: Optional[int] = None):
        """Initializer.
        Args:
            channels: size of the input channels.
            kernels: size of the convolutional kernels.
            dilations: dilation rate of the convolution.
            dropout: dropout rate.
            cond: size of the condition channels, if provided.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels * 2, kernels, dilation=dilations,
                      padding=(kernels - 1) * dilations // 2),
            nn.GLU(dim=1))
        
        self.cond = cond
        if cond is not None:
            self.cond = nn.Conv1d(cond, channels * 2, 1)

    def forward(self, inputs: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """Transform the inputs with given conditions.
        Args:
            inputs: [torch.float32; [B, channels, T]], input channels.
            cond: [torch.float32; [B, cond, T]], if provided.
        Returns:
            [torch.float32; [B, channels, T]], transformed.
        """
        # [B, channels, T]
        x = inputs + self.conv(inputs)
        if cond is not None:
            assert self.cond is not None, 'condition module does not exists'
            # [B, channels, T]
            x = F.instance_norm(x, use_input_stats=True)
            # [B, channels, T]
            weight, bias = self.cond(cond).chunk(2, dim=1)
            # [B, channels, T]
            x = x * weight + bias
        return x


class CondSequential(nn.Module):
    """Sequential pass with conditional inputs.
    """
    def __init__(self, modules: List[nn.Module]):
        """Initializer.
        Args:
            modules: list of torch modules.
        """
        super().__init__()
        self.lists = nn.ModuleList(modules)
    
    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Pass the inputs to modules.
        Args:
            inputs: arbitary input tensors.
            args, kwargs: positional, keyword arguments.
        Returns:
            output tensor.
        """
        x = inputs
        for module in self.lists:
            x = module.forward(x, *args, **kwargs)
        return x


class FrameLevelSynthesizer(nn.Module):
    """Frame-level synthesizer.
    """
    def __init__(self,
                 channels: int,
                 embed: int,
                 kernels: int,
                 dilations: List[int],
                 blocks: int,
                 leak: float,
                 dropout: float):
        """Initializer.
        Args:
            channels: size of the input channels.
            embed: size of the time-varying timber embeddings.
            kernels: size of the convolutional kernels.
            dilations: dilation rates.
            blocks: the number of the 1x1 ConvGLU blocks after dilated ConvGLU.
            leak: negative slope of the leaky relu.
            dropout: dropout rates.
        """
        super().__init__()
        # channels=1024
        # unknown `leak`, `dropout`
        self.preconv = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.LeakyReLU(leak),
            nn.Dropout(dropout))
        # kernels=3, dilations=[1, 3, 9, 27, 1, 3, 9, 27], blocks=2
        self.convglu = CondSequential(
            [
                ConvGLU(channels, kernels, dilation, dropout, cond=embed)
                for dilation in dilations]
            + [
                ConvGLU(channels, 1, 1, dropout, cond=embed)
                for _ in range(blocks)])

        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, inputs: torch.Tensor, timber: torch.Tensor) -> torch.Tensor:
        """Synthesize in frame-level.
        Args:
            inputs: [torch.float32; [B, channels, T]], input features.
            timber: [torch.float32; [B, embed, T]], time-varying timber embeddings.
        Returns;
            [torch.float32; [B, channels, T]], outputs.
        """
        # [B, channels, T]
        x = self.preconv(inputs)
        # [B, channels, T] 
        x = self.convglu(x, timber)
        # [B, channels, T]
        return self.proj(x)


## Signal Generator
class SignalGenerator(nn.Module):
    """Additive sinusoidal, subtractive filtered noise signal generator.
    """
    def __init__(self, scale: int, sr: int):
        """Initializer.
        Args:
            scale: upscaling factor.
            sr: sampling rate.
        """
        super().__init__()
        self.sr = sr
        self.upsampler = nn.Upsample(scale_factor=scale, mode='linear')

    def forward(self,
                pitch: torch.Tensor,
                p_amp: torch.Tensor,
                ap_amp: torch.Tensor,
                noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate the signal.
        Args:
            pitch: [torch.float32; [B, N]], frame-level pitch sequence.
            p_amp, ap_amp: [torch.float32; [B, N]], periodical, aperiodical amplitude.
            noise: [torch.float32; [B, T]], predefined noise, if provided.
        Returns:
            [torch.float32; [B, T(=N x scale)]], base signal.
        """
        # [B, T]
        pitch = self.upsampler(pitch[:, None]).squeeze(dim=1)
        p_amp = self.upsampler(p_amp[:, None]).squeeze(dim=1)
        # [B, T]
        phase = torch.cumsum(2 * np.pi * pitch / self.sr, dim=-1)
        # [B, T]
        x = p_amp * torch.sin(phase)
        # [B, T]
        ap_amp = self.upsampler(ap_amp[:, None]).squeeze(dim=1)
        if noise is None:
            # [B, T], U[-1, 1] sampled
            noise = torch.rand_like(x) * 2. - 1.
        # [B, T]
        y = ap_amp * noise
        return x + y


## Sample-level Synthesizer
class SampleLevelSynthesizer(nn.Module):
    """Signal-level synthesizer.
    """
    def __init__(self,
                 scale: int,
                 sr: int,
                 channels: int,
                 aux: int,
                 kernels: int,
                 dilation_rate: int,
                 layers: int,
                 cycles: int):
        """Initializer.
        Args:
            scale: upscaling factor.
            sr: sampling rate.
            channels: size of the hidden channels.
            aux: size of the auxiliary input channels.
            kernels: size of the convolutional kernels.
            dilation_rate: dilaion rate.
            layers: the number of the wavenet blocks in single cycle.
            cycles: the number of the cycles.
        """
        super().__init__()
        self.excitation = SignalGenerator(scale, sr)

        self.wavenet = WaveNet(
            channels,
            aux,
            kernels,
            dilation_rate,
            layers,
            cycles)

    def forward(self,
                pitch: torch.Tensor,
                p_amp: torch.Tensor,
                ap_amp: torch.Tensor,
                frame: torch.Tensor,
                noise: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the signal.
        Args:
            pitch: [torch.float32; [B, N]], frame-level pitch sequence.
            p_amp, ap_amp: [torch.float32; [B, N]], periodical, aperiodical amplitude.
            frame: [torch.float32; [B, aux, N']], frame-level feature map.
            noise: [torch.float32; [B, T]], predefined noise for excitation signal, if provided.
        Returns:
            [torch.float32; [B, T]], excitation signal and generated signal.
        """
        # [B, T]
        excitation = self.excitation.forward(pitch, p_amp, ap_amp, noise=noise)
        # [B, aux, T]
        interp = F.interpolate(frame, size=excitation.shape[-1], mode='linear')
        # [B, T]
        signal = self.wavenet.forward(excitation, interp)
        return excitation, signal