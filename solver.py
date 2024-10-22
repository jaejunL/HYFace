import os
import time
# import wandb

import numpy as np
from time import gmtime, strftime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import utils
from data_utils import Dataset_Main, Dataset_Sub, Collate
from models import SynthesizerTrn, MultiPeriodDiscriminator, F2F0
import modules.commons as commons
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

class Solver_Main(object):
    def __init__(self, args):
        self.args = args
        # self.wandb = wandb.init(entity='Your-WanDB-ID', project='Your-WanDB-ProjectName', group=args.base_args.model, config=args)
        self.global_step = 0
    
    def build_dataset(self, args):
        mp_context = torch.multiprocessing.get_context('fork')
        args.train.batch_size = int(args.train.batch_size / args.base_args.ngpus_per_node)
        self.trainset = Dataset_Main(args, args.data.aud_dir, args.data.img_dir, typ="pretrain")
        self.validset = Dataset_Main(args, args.data.aud_dir, args.data.img_dir, typ="trainval")
        self.train_sampler = DistributedSampler(self.trainset, shuffle=True, rank=self.args.base_args.gpu)
        self.train_loader = DataLoader(self.trainset, num_workers=args.base_args.workers, shuffle=False, pin_memory=True,
                                batch_size=args.train.batch_size, collate_fn=Collate(),
                                multiprocessing_context=mp_context, sampler=self.train_sampler, drop_last=True,
                                worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
        self.valid_sampler = DistributedSampler(self.validset, shuffle=True, rank=self.args.base_args.gpu)
        self.valid_loader = DataLoader(self.validset, num_workers=args.base_args.workers, shuffle=False, pin_memory=True,
                                batch_size=args.train.batch_size, collate_fn=Collate(),
                                multiprocessing_context=mp_context, sampler=self.valid_sampler, drop_last=True,
                                worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))        
        self.max_iter = len(self.train_loader)
        
    def build_models(self, args):
        torch.cuda.set_device(self.args.base_args.gpu)
        net_g = SynthesizerTrn(
            args.data.filter_length // 2 + 1,
            args.train.segment_size // args.data.hop_length,
            **args.model)
        net_d = MultiPeriodDiscriminator(args.model.use_spectral_norm).to('cuda:{}'.format(self.args.base_args.gpu))
        net_g = net_g.to('cuda:{}'.format(self.args.base_args.gpu))
        net_d = net_d.to('cuda:{}'.format(self.args.base_args.gpu))
        net_g = DDP(net_g, device_ids=[self.args.base_args.gpu], output_device=self.args.base_args.gpu, find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[self.args.base_args.gpu], output_device=self.args.base_args.gpu, find_unused_parameters=True)
        self.net = {'g':net_g, 'd':net_d}
    
    def build_optimizers(self, args):
        optim_g = torch.optim.AdamW(self.net['g'].parameters(), args.train.learning_rate,
                                    betas=args.train.betas, eps=args.train.eps)
        optim_d = torch.optim.AdamW(self.net['d'].parameters(), args.train.learning_rate,
                                    betas=args.train.betas, eps=args.train.eps)
        self.optim = {'g':optim_g, 'd':optim_d}
        
        self.warmup_epoch = args.train.warmup_epochs
        self.half_type = torch.bfloat16 if args.train.half_type=="bf16" else torch.float16
        self.scaler = GradScaler(enabled=args.train.fp16_run)
        
    def build_scheduler(self, args, epoch_str):
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim['g'], gamma=args.train.lr_decay, last_epoch=epoch_str - 2)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim['d'], gamma=args.train.lr_decay, last_epoch=epoch_str - 2)
        self.scheduler = {'g':scheduler_g, 'd':scheduler_d}
        
    def train(self, args, epoch):
        self.net['g'].train()
        self.net['d'].train()
        if epoch <= self.warmup_epoch:
            for param_group in self.optim['g'].param_groups:
                param_group['lr'] = args.train.learning_rate / self.warmup_epoch * epoch
            for param_group in self.optim['d'].param_groups:
                param_group['lr'] = args.train.learning_rate / self.warmup_epoch * epoch

        for batch_idx, items in enumerate(self.train_loader):
            losses, auxes = self.loss_generator(args, items, phase="train")
            
            # train log
            if args.base_args.rank % args.base_args.ngpus_per_node == 0:
                if self.global_step % args.train.log_interval == 0:
                    print("\r[Epoch:{:3d}, {:.0f}%, Step:{}] [Loss G:{:.5f}] [{}]"
                        .format(epoch, 100.*batch_idx/self.max_iter, self.global_step, losses['gen/total'], strftime('%Y-%m-%d %H:%M:%S', gmtime())))
                    # if args.base_args.test != 1:
                        # self.wandb_log(losses, epoch, "train")
            if args.base_args.test:
                if batch_idx > 100:
                    break
            self.global_step += 1
            
    def validation(self, args, epoch):
        with torch.no_grad():
            self.net['g'].eval()
            self.net['d'].eval()
            for batch_idx, items in enumerate(self.valid_loader):
                losses, auxes = self.loss_generator(args, items, phase="valid")
                if args.base_args.test:
                    if batch_idx > 10:
                        break
                # validation log
                if args.base_args.rank % args.base_args.ngpus_per_node == 0:
                    print("\r[Validation Epoch:{:3d}] [Loss G:{:.5f}]".format(epoch, losses['gen/total']))
                    # if args.base_args.test != 1:
                        # self.wandb_log(losses, epoch, "valid")
                        
    def loss_generator(self, args, items, phase="train") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        c, f0, spec, y, uv, avgf0, face, lengths = items
        spec, y = spec.cuda(args.base_args.rank, non_blocking=True), y.cuda(args.base_args.rank, non_blocking=True)
        c = c.cuda(args.base_args.rank, non_blocking=True)
        f0 = f0.cuda(args.base_args.rank, non_blocking=True)
        uv = uv.cuda(args.base_args.rank, non_blocking=True) if uv is not None else None
        avgf0 = avgf0.cuda(args.base_args.rank, non_blocking=True) if avgf0 is not None else None
        face = face.cuda(args.base_args.rank, non_blocking=True) if face is not None else None
        lengths = lengths.cuda(args.base_args.rank, non_blocking=True)
        
        mel = spec_to_mel_torch(
            spec,
            args.data.filter_length,
            args.data.n_mel_channels,
            args.data.sampling_rate,
            args.data.mel_fmin,
            args.data.mel_fmax)        

        with autocast(enabled=args.train.fp16_run, dtype=self.half_type):
            y_hat, ids_slice, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), lf0, pred_lf0, avglf0, fredavglf0 = self.net['g'](c, f0, uv, spec, mel, c_lengths=lengths,
                                                                                spec_lengths=lengths, avgf0=avgf0, face=face)
            y_mel = commons.slice_segments(mel, ids_slice, args.train.segment_size // args.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                args.data.filter_length,
                args.data.n_mel_channels,
                args.data.sampling_rate,
                args.data.hop_length,
                args.data.win_length,
                args.data.mel_fmin,
                args.data.mel_fmax
            )
            
            y = commons.slice_segments(y, ids_slice * args.data.hop_length, args.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = self.net['d'](y, y_hat.detach())

            with autocast(enabled=False, dtype=self.half_type):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        
        if phase == "train":
            self.optim['d'].zero_grad()
            self.scaler.scale(loss_disc_all).backward()
            self.scaler.unscale_(self.optim['d'])
            grad_norm_d = commons.clip_grad_value_(self.net['d'].parameters(), None)
            self.scaler.step(self.optim['d'])

        with autocast(enabled=args.train.fp16_run, dtype=self.half_type):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net['d'](y, y_hat)
            with autocast(enabled=False, dtype=self.half_type):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * args.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * args.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_lf0 = F.mse_loss(pred_lf0, lf0)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
                if fredavglf0 is not None:
                    loss_ff0 = F.mse_loss(fredavglf0, avglf0) if self.net['g'].module.use_automatic_f0_prediction else 0
                    loss_gen_all += loss_ff0

        if phase == "train":                    
            self.optim['g'].zero_grad()
            self.scaler.scale(loss_gen_all).backward()
            self.scaler.unscale_(self.optim['g'])
            grad_norm_g = commons.clip_grad_value_(self.net['g'].parameters(), None)
            self.scaler.step(self.optim['g'])
            self.scaler.update()
        
        losses = {
            'gen/total': loss_gen_all.item(),
            'gen/recon': loss_gen.item(),
            'gen/fmap': loss_fm.item(),
            'gen/mel': loss_mel.item(),
            'gen/kl': loss_kl.item(),
            'gen/lf0': loss_lf0.item(),
            'disc/total': loss_disc_all.item(),
            'disc/loss': loss_disc.item()}
        if fredavglf0 is not None:
            losses['gen/ff0'] = loss_ff0.item()
        media = {
            "mel_gt": y_mel[0].data.cpu().detach().numpy(),
            "mel_synth": y_hat_mel[0].data.cpu().detach().numpy()}                   
                     
        return losses, media
                
    # def wandb_log(self, loss_dict, epoch, phase="train"):
    #     wandb_dict = {}
    #     wandb_dict.update(loss_dict)
    #     wandb_dict.update({"epoch":epoch})
    #     if phase == "train":
    #         wandb_dict.update({"global_step": self.global_step})
    #         with torch.no_grad():
    #             grad_norm = np.mean([
    #                 torch.norm(p.grad).item() for p in self.net['g'].parameters() if p.grad is not None])
    #             param_norm = np.mean([
    #                 torch.norm(p).item() for p in self.net['g'].parameters() if p.dtype == torch.float32])
    #         wandb_dict.update({ "common/grad-norm":grad_norm, "common/param-norm":param_norm})
    #         wandb_dict.update({ "common/learning-rate-g":self.optim['g'].param_groups[0]['lr']})
    #     elif phase == "valid":
    #         wandb_dict = dict(('valid/'+ key, np.mean(value)) for (key, value) in wandb_dict.items())
    #     self.wandb.log(wandb_dict)
        
class Solver_Sub(object):
    def __init__(self, args):
        self.args = args
        # self.wandb = wandb.init(entity='Your-WanDB-ID', project='Your-WanDB-ProjectName', group=args.base_args.model, config=args)
        self.global_step = 0
    
    def build_dataset(self, args):
        mp_context = torch.multiprocessing.get_context('fork')
        args.train.batch_size = int(args.train.batch_size / args.base_args.ngpus_per_node)
        self.trainset = Dataset_Sub(args.data.aud_dir, args.data.img_dir, typ="pretrain")
        self.validset = Dataset_Sub(args.data.aud_dir, args.data.img_dir, typ="trainval")
        self.train_sampler = DistributedSampler(self.trainset, shuffle=True, rank=self.args.base_args.gpu)
        self.train_loader = DataLoader(self.trainset, num_workers=args.base_args.workers, shuffle=False, pin_memory=True,
                                batch_size=args.train.batch_size,
                                multiprocessing_context=mp_context, sampler=self.train_sampler, drop_last=True,
                                worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
        self.valid_sampler = DistributedSampler(self.validset, shuffle=True, rank=self.args.base_args.gpu)
        self.valid_loader = DataLoader(self.validset, num_workers=args.base_args.workers, shuffle=False, pin_memory=True,
                                batch_size=args.train.batch_size,
                                multiprocessing_context=mp_context, sampler=self.valid_sampler, drop_last=True,
                                worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))        
        self.max_iter = len(self.train_loader)
        
    def build_models(self, args):
        torch.cuda.set_device(self.args.base_args.gpu)
        net_g = F2F0(imgsize=112)
        net_g = net_g.to('cuda:{}'.format(self.args.base_args.gpu))
        net_g = DDP(net_g, device_ids=[self.args.base_args.gpu], output_device=self.args.base_args.gpu, find_unused_parameters=True)
        self.net = {'g':net_g}
    
    def build_optimizers(self, args):
        optim_g = torch.optim.AdamW(self.net['g'].parameters(), args.train.learning_rate,
                                    betas=args.train.betas, eps=args.train.eps)
        self.optim = {'g':optim_g}
        self.warmup_epoch = args.train.warmup_epochs
        
    def build_scheduler(self, args, epoch_str):
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim['g'], gamma=args.train.lr_decay, last_epoch=epoch_str - 2)
        self.scheduler = {'g':scheduler_g}
        
    def train(self, args, epoch):
        self.net['g'].train()
        if epoch <= self.warmup_epoch:
            for param_group in self.optim['g'].param_groups:
                param_group['lr'] = args.train.learning_rate / self.warmup_epoch * epoch

        for batch_idx, items in enumerate(self.train_loader):
            losses = self.loss_generator(args, items, phase="train")
            
            # train log
            if args.base_args.rank % args.base_args.ngpus_per_node == 0:
                if self.global_step % args.train.log_interval == 0:
                    print("\r[Epoch:{:3d}, {:.0f}%, Step:{}] [Loss G:{:.5f}] [{}]"
                        .format(epoch, 100.*batch_idx/self.max_iter, self.global_step, losses['gen/total'], strftime('%Y-%m-%d %H:%M:%S', gmtime())))
                    # if args.base_args.test != 1:
                        # self.wandb_log(losses, epoch, "train")
            if args.base_args.test:
                if batch_idx > 100:
                    break
            self.global_step += 1
            
    def validation(self, args, epoch):
        with torch.no_grad():
            self.net['g'].eval()
            for batch_idx, items in enumerate(self.valid_loader):
                losses = self.loss_generator(args, items, phase="valid")
                if args.base_args.test:
                    if batch_idx > 10:
                        break
                # validation log
                if args.base_args.rank % args.base_args.ngpus_per_node == 0:
                    print("\r[Validation Epoch:{:3d}] [Loss G:{:.5f}]".format(epoch, losses['gen/total']))
                    # if args.base_args.test != 1:
                        # self.wandb_log(losses, epoch, "valid")
                        
    def loss_generator(self, args, items, phase="train") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        avgf0, face, _ = items
        avgf0 = avgf0.cuda(args.base_args.rank, non_blocking=True) if avgf0 is not None else None
        avglf0 = 2595. * torch.log10(1. + avgf0 / 700.) / 500
        face = face.cuda(args.base_args.rank, non_blocking=True) if face is not None else None        
        
        fredavglf0, face_emb = self.net['g'](face=face)
        loss_lf0 = F.mse_loss(fredavglf0, avglf0)        
        total_loss = loss_lf0

        if phase == "train":                    
            self.optim['g'].zero_grad()
            total_loss.backward()
            grad_norm_g = commons.clip_grad_value_(self.net['g'].parameters(), None)
            self.optim['g'].step()
        losses = {'gen/loss_lf0': loss_lf0.item(),
                  'gen/total': total_loss.item()}                
        return losses
                
    # def wandb_log(self, loss_dict, epoch, phase="train"):
    #     wandb_dict = {}
    #     wandb_dict.update(loss_dict)
    #     wandb_dict.update({"epoch":epoch})
    #     if phase == "train":
    #         wandb_dict.update({"global_step": self.global_step})
    #         with torch.no_grad():
    #             grad_norm = np.mean([
    #                 torch.norm(p.grad).item() for p in self.net['g'].parameters() if p.grad is not None])
    #             param_norm = np.mean([
    #                 torch.norm(p).item() for p in self.net['g'].parameters() if p.dtype == torch.float32])
    #         wandb_dict.update({ "common/grad-norm":grad_norm, "common/param-norm":param_norm})
    #         wandb_dict.update({ "common/learning-rate-g":self.optim['g'].param_groups[0]['lr']})
    #     elif phase == "valid":
    #         wandb_dict = dict(('valid/'+ key, np.mean(value)) for (key, value) in wandb_dict.items())
    #     self.wandb.log(wandb_dict)

