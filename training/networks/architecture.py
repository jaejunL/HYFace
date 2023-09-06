import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from models.transformer import TransformerEncoderLayer
from models import modules
from commons import init_weights, get_padding


class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)

class Model(nn.Module):
    def __init__(self, args, num_outs, num_aux_outs=None):
        super().__init__()

        if args.base_args.hubert:
            self.conv_blocks = nn.Sequential(
                ResBlock(8, args.model.model_size, 2),
                ResBlock(args.model.model_size, args.model.model_size, 2),
                ResBlock(args.model.model_size, args.model.model_size, 2),
                ResBlock(args.model.model_size, args.model.model_size, 2),
            )        
        else:
            self.conv_blocks = nn.Sequential(
                ResBlock(8, args.model.model_size, 2),
                ResBlock(args.model.model_size, args.model.model_size, 2),
                ResBlock(args.model.model_size, args.model.model_size, 2),
            )
        self.w_raw_in = nn.Linear(args.model.model_size, args.model.model_size)

        encoder_layer = TransformerEncoderLayer(d_model=args.model.model_size, nhead=8, relative_positional=True, relative_positional_distance=100, dim_feedforward=3072, dropout=args.model.dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, args.model.num_layers)
        self.w_out = nn.Linear(args.model.model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(args.model.model_size, num_aux_outs)

    def forward(self, x_raw):
        # x shape is (batch, time, electrode)

        if self.training: # shift augmentation (maybe for training robustness of dtw loss?)
            r = random.randrange(8)
            if r > 0:
                x_raw[:,:-r,:] = x_raw[:,r:,:] # shift left r
                x_raw[:,-r:,:] = 0

        x_raw = x_raw.transpose(1,2) # put channel before time for conv
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1,2)
        x_raw = self.w_raw_in(x_raw)

        x = x_raw

        x = x.transpose(0,1) # put time first
        x = self.transformer(x)
        x = x.transpose(0,1)

        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)
        else:
            return self.w_out(x)


class Vocoder(torch.nn.Module):
    """
      HifiGAN official github
      https://github.com/jik876/hifi-gan/blob/master/models.py
    """    
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Vocoder, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

