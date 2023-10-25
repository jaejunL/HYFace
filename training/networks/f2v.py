import os
import sys
import json
import numpy as np
import librosa
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# import timm

from utils import utils, audio_utils, data_utils


class F2V_Ecapa(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=args.model.pretrained)
        self.vit.head = nn.Linear(in_features=args.model.vit.hidden, out_features=args.model.timbre.global_, bias=True)
        
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        pred = self.vit(imgs)
        return pred

def nce_loss(img, aud, tau):
    # calculate nce loss for mean-visual representation and mean-audio representation
    img_norm = torch.nn.functional.normalize(img, dim=-1)
    aud_norm = torch.nn.functional.normalize(aud, dim=-1)
    total = torch.mm(img_norm, aud_norm.transpose(0, 1)) / tau
    nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
    c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=aud_norm.device))) / total.shape[0]
    return nce, c_acc

def mae_loss(pred, target):
    loss = (pred - target) ** 2
    return loss.mean(dim=-1).sum()

