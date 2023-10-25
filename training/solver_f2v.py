import os
import json
import wandb
import itertools
import numpy as np
from time import gmtime, strftime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from datasets.f2v_loader import FaceNAudio_Spkwise, FaceNEcapaAVg_Filewise
from networks.f2v import F2V_Ecapa, nce_loss, mae_loss

from utils import audio_utils

class Solver(object):
    def __init__(self, args):
        self.args = args
        self.wandb = wandb.init(entity='jjlee0721', project='F2V', group=args.base_args.group_name, config=args)
        self.global_step = 0
        wandb.run.name = args.base_args.exp_name
        # self.device = torch.device(f'cuda:{self.args.base_args.gpu}')
        
        self.cmap = np.array(plt.get_cmap('viridis').colors)

    def build_dataset(self, args):
        self.mp_context = torch.multiprocessing.get_context('fork')

        args.train.batch_size = int(args.train.batch_size / args.base_args.ngpus_per_node)
        self.trainset = FaceNEcapaAVg_Filewise(args, meta_root=os.path.join(args.base_args.meta_root, 'filelists/VGG_Face'),
                        mode='train', img_datasets=args.data.img_datasets, sample_rate=args.data.sample_rate)
        self.validset = FaceNEcapaAVg_Filewise(args, meta_root=os.path.join(args.base_args.meta_root, 'filelists/VGG_Face'),
                        mode='valid', img_datasets=args.data.img_datasets, sample_rate=args.data.sample_rate)
        self.train_sampler = DistributedSampler(self.trainset, shuffle=True, rank=self.args.base_args.gpu)
        self.train_loader = DataLoader(self.trainset, batch_size=args.train.batch_size,
                                shuffle=False, num_workers=args.base_args.workers,
                                multiprocessing_context=self.mp_context,
                                # collate_fn=self.trainset.collate,
                                pin_memory=True, sampler=self.train_sampler, drop_last=True,
                                worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
        self.valid_sampler = DistributedSampler(self.validset, shuffle=True, rank=self.args.base_args.gpu)
        self.valid_loader = DataLoader(self.validset, batch_size=args.train.batch_size,
                                shuffle=False, num_workers=args.base_args.workers,
                                multiprocessing_context=self.mp_context,
                                # collate_fn=self.validset.collate,
                                pin_memory=True, sampler=self.valid_sampler, drop_last=True,
                                worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
        self.max_iter = len(self.train_loader)

    def build_models(self, args):
        ####################### Distribute Models to Machines #######################
        torch.cuda.set_device(self.args.base_args.gpu)
        if args.model.generator == "vit" and args.model.timbre.type == "ecapa":
            f2v = F2V_Ecapa(args)
        f2v = f2v.to('cuda:{}'.format(self.args.base_args.gpu))
        f2v = DistributedDataParallel(f2v, device_ids=[self.args.base_args.gpu], output_device=self.args.base_args.gpu, find_unused_parameters=True)
        self.net = {'f2v':f2v}

    def build_losses(self, args):
        pass

    def build_optimizers(self, args):
        optim_g = torch.optim.Adam(self.net['f2v'].parameters(), args.train.learning_rate_g,
                                   weight_decay=args.train.weight_decay, betas=(args.train.beta1, args.train.beta2))
        self.optim = {'g':optim_g}

    def loss_generator(self, img: torch.Tensor, aud: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Data to Cuda tensor
        img = img.to(device=torch.device(f'cuda:{self.args.base_args.gpu}'))
        aud = aud.to(device=torch.device(f'cuda:{self.args.base_args.gpu}'))

        pred = self.net['f2v'](img)
        loss_nce, c_acc = nce_loss(pred, aud.squeeze(), self.args.train.tau)
        loss_mae = mae_loss(pred, aud.squeeze())
        
        loss = loss_nce + self.args.train.mae_weight * loss_mae
        
        losses = {
            'gen/loss': loss.item(),
            'gen/contrastive': loss_nce.item(),
            'gen/mae': loss_mae.item()
            }
        return loss, losses

    def wandb_log(self, loss_dict, epoch, phase="train"):
        wandb_dict = {}
        wandb_dict.update(loss_dict)
        wandb_dict.update({"epoch":epoch})
        if phase == "train":
            wandb_dict.update({"global_step": self.global_step})
            with torch.no_grad():
                grad_norm = np.mean([
                    torch.norm(p.grad).item() for p in self.net['f2v'].parameters() if p.grad is not None])
                param_norm = np.mean([
                    torch.norm(p).item() for p in self.net['f2v'].parameters() if p.dtype == torch.float32])
            wandb_dict.update({ "common/grad-norm":grad_norm, "common/param-norm":param_norm})
            wandb_dict.update({ "common/learning-rate-g":self.optim['g'].param_groups[0]['lr']})
        elif phase == "valid":
            wandb_dict = dict(('valid/'+ key, np.mean(value)) for (key, value) in wandb_dict.items())
        self.wandb.log(wandb_dict)

    def train(self, args, epoch):
        self.net['f2v'].train()
        for batch_idx, bunch in enumerate(self.train_loader):
            img, aud = bunch
            loss_g, losses_g = self.loss_generator(img, aud)
            
            # update
            self.optim['g'].zero_grad()
            loss_g.backward()
            self.optim['g'].step()
            
            # train log
            if args.base_args.rank % args.base_args.ngpus_per_node == 0:
                if self.global_step % args.train.log_interval == 0:
                    print("\r[Epoch:{:3d}, {:.0f}%, Step:{}] [Loss C:{:.5f}, Loss MAE:{:.5f}] [{}]"
                        .format(epoch, 100.*batch_idx/self.max_iter, self.global_step,
                                losses_g['gen/contrastive'], losses_g['gen/mae'],
                                strftime('%Y-%m-%d %H:%M:%S', gmtime())))
                    if args.base_args.test != 1:
                        self.wandb_log(losses_g, epoch, "train")
            if args.base_args.test:
                if batch_idx > 10:
                    break
            self.global_step += 1
        return (losses_g).keys()

    def validation(self, args, epoch, losses_keys):
        losses = {key: [] for key in losses_keys}
        with torch.no_grad():
            self.net['f2v'].eval()
            for batch_idx, bunch in enumerate(self.valid_loader):
                img, aud = bunch
                _, losses_g = self.loss_generator(img, aud)
                for key, val in (losses_g).items():
                    losses[key].append(val)
                if args.base_args.test:
                    if batch_idx > 10:
                        break            
            # validation log
            if args.base_args.rank % args.base_args.ngpus_per_node == 0:
                print("\r[Validation Epoch:{:3d}] [Loss G:{:.5f}]"
                    .format(epoch, np.mean(losses['gen/loss'])))
                if args.base_args.test != 1:
                    self.wandb_log(losses, epoch, "valid")