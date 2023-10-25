import os
import json
import wandb
import itertools
import numpy as np
from time import gmtime, strftime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torchaudio

from datasets.f2v_loader import Faubert_Dataset
# from networks.discriminator import Discriminator
from hyface import Nansy, BShall_Ecapa, BShall_Nimbre, BShall_Fimbre
from hifigan.generator import HifiganGenerator
from hifigan.discriminator import (
    HifiganDiscriminator,
    feature_loss,
    discriminator_loss,
    generator_loss,
)

from utils import audio_utils, utils, data_utils

class Solver(object):
    def __init__(self, args):
        self.args = args
        self.wandb = wandb.init(entity='jjlee0721', project='f2v', group=args.base_args.group_name, config=args)
        self.global_step = 0
        wandb.run.name = args.base_args.exp_name
        # self.device = torch.device(f'cuda:{self.args.base_args.gpu}')
        
        self.cmap = np.array(plt.get_cmap('viridis').colors)

    def build_dataset(self, args):
        self.mp_context = torch.multiprocessing.get_context('fork')

        args.train.batch_size = int(args.train.batch_size / args.base_args.ngpus_per_node)
        self.trainset = Faubert_Dataset(args, meta_root=os.path.join(args.base_args.meta_root, 'filelists/VGG_Face'),
                        mode='train', datasets=args.data.img_datasets, sample_rate=args.data.sample_rate)
        self.validset = Faubert_Dataset(args, meta_root=os.path.join(args.base_args.meta_root, 'filelists/VGG_Face'),
                        mode='valid', datasets=args.data.img_datasets, sample_rate=args.data.sample_rate)
        self.train_sampler = DistributedSampler(self.trainset, shuffle=True, rank=self.args.base_args.gpu)
        self.train_loader = DataLoader(self.trainset, batch_size=args.train.batch_size,
                                shuffle=False, num_workers=args.base_args.workers,
                                multiprocessing_context=self.mp_context, collate_fn=self.trainset.collate,
                                pin_memory=True, sampler=self.train_sampler, drop_last=True,
                                worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
        self.valid_sampler = DistributedSampler(self.validset, shuffle=True, rank=self.args.base_args.gpu)
        self.valid_loader = DataLoader(self.validset, batch_size=args.train.batch_size,
                                shuffle=False, num_workers=args.base_args.workers,
                                multiprocessing_context=self.mp_context, collate_fn=self.validset.collate,
                                pin_memory=True, sampler=self.valid_sampler, drop_last=True,
                                worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
        self.max_iter = len(self.train_loader)

    def build_models(self, args):
        ####################### Distribute Models to Machines #######################
        torch.cuda.set_device(self.args.base_args.gpu)
        if args.model.generator == "vits_face" and args.model.timbre.type == "nimbre":
            index = 3000
            checkpoint_path = f'/disk3/jaejun/hyface/bshall_nimbre/checkpoints/G_{index}.pth'
            nimbre_short = BShall_Nimbre(args)
            nimbre_short.frame_synth = nn.Identity()
            nimbre_short, _, _, _ = utils.load_checkpoint(checkpoint_path, nimbre_short, None)
            nimbre_short = nimbre_short.to('cuda:{}'.format(self.args.base_args.gpu))
            nimbre_short = DistributedDataParallel(nimbre_short, device_ids=[self.args.base_args.gpu], output_device=self.args.base_args.gpu, find_unused_parameters=True)
            
            self.fimber_query = torch.load('/disk3/jaejun/hyface/bshall_nimbre/timbre_query.pth').to(device=torch.device(f'cuda:{self.args.base_args.gpu}'))
            
            vit_checkpoint_path = '/disk3/jaejun/hyface/bshall_fimbre/ViT-P12S8.pth'
            f2v = BShall_Fimbre(args)
            f2v.fimbre.vit_face = utils.load_vitface_checkpoint(vit_checkpoint_path, f2v.fimbre.vit_face)
        f2v = f2v.to('cuda:{}'.format(self.args.base_args.gpu))
        f2v = DistributedDataParallel(f2v, device_ids=[self.args.base_args.gpu], output_device=self.args.base_args.gpu, find_unused_parameters=True)
        self.net = {'f2v':f2v, 'timbre':nimbre_short}

    def build_losses(self, args):
        pass

    def build_optimizers(self, args):
        optim_g = torch.optim.Adam(self.net['f2v'].parameters(), args.train.learning_rate_g,
                                    (args.train.beta1, args.train.beta2))
        self.optim = {'g':optim_g}

    def loss_generator(self, image: torch.Tensor, audio: np.ndarray, ling: torch.Tensor, lengths: np.ndarray) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Data to Cuda tensor
        image = image.to(device=torch.device(f'cuda:{self.args.base_args.gpu}'))
        audio = torch.tensor(audio, device=torch.device(f'cuda:{self.args.base_args.gpu}'))
        ling = ling.to(device=torch.device(f'cuda:{self.args.base_args.gpu}'))
        length = torch.as_tensor(lengths, device=torch.device(f'cuda:{self.args.base_args.gpu}'))

        # for fimbre
        fimbre_global, fimbre_local = self.net['f2v'].module.analyze_fimbre(image, self.fimber_query)
        mel_synth = self.net['f2v'].module.synthesize(ling, (fimbre_global, fimbre_local))
        
        # for nimbre
        mel_gt = self.net['timbre'].module.logmel.forward(audio)
        timbre_global, timbre_local = self.net['timbre'].module.analyze_timbre(audio)

        # reconstruction loss
        mel_loss = F.l1_loss(mel_synth, mel_gt, reduction="none")
        mel_loss = torch.sum(mel_loss, dim=(1,2)) / mel_synth.size(-1)*length
        mel_loss = torch.mean(mel_loss)
        
        global_loss = F.l1_loss(fimbre_global, timbre_global, reduction="none")
        global_loss = torch.sum(global_loss, dim=-1)
        global_loss = torch.mean(global_loss)
        
        local_loss = F.l1_loss(fimbre_local, timbre_local, reduction="none")
        local_loss = torch.sum(local_loss, dim=(1,2))
        local_loss = torch.mean(local_loss)

        rctor_loss = mel_loss*0.0001 + global_loss + local_loss*0.000001
        loss = rctor_loss
        losses = {
            'gen/loss': loss.item(),
            'gen/rctor': rctor_loss.item(),
            'gen/mel_loss': mel_loss.item(),
            'gen/global_loss': global_loss.item(),
            'gen/local_loss': local_loss.item()}
        media =  {
            'mel_synth': mel_synth.cpu().detach().numpy(),
            'mel_gt': mel_gt.cpu().detach().numpy()}
        return loss, losses, media


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


    def wandb_media(self, aux_g, gt_audio, phase="train"):
        # image sample
        mel_gt = wandb.Image(audio_utils.mel_img(aux_g['mel_gt'][0], self.cmap))
        mel_synth = wandb.Image(audio_utils.mel_img(aux_g['mel_synth'][0], self.cmap))
        wandb.log({"mel-gt/"+ phase : mel_gt, "mel-synth/"+ phase : mel_synth})
        # audio sample
        audio_gt = wandb.Audio(gt_audio[0], sample_rate=self.args.data.sample_rate)
        wandb.log({"speech/"+ phase : audio_gt})

    def train(self, args, epoch):
        self.net['f2v'].train()
        self.net['timbre'].eval()
        for batch_idx, bunch in enumerate(self.train_loader):
            image, audio, ling, lengths = bunch['image'], bunch['audio'], bunch['hubert'], bunch['frame_lengths']
            loss_g, losses_g, aux_g = self.loss_generator(image, audio, ling, lengths)
            # update
            self.optim['g'].zero_grad()
            loss_g.backward()
            self.optim['g'].step()
            
            # train log
            if args.base_args.rank % args.base_args.ngpus_per_node == 0:
                if self.global_step % args.train.log_interval == 0:
                    print("\r[Epoch:{:3d}, {:.0f}%, Step:{}] [Loss G:{:.5f}] [{}]"
                        .format(epoch, 100.*batch_idx/self.max_iter, self.global_step, loss_g, strftime('%Y-%m-%d %H:%M:%S', gmtime())))
                    if args.base_args.test != 1:
                        self.wandb_log(losses_g, epoch, "train")
                if self.global_step % args.train.sample_interval == 0:
                    if args.base_args.test != 1:
                        self.wandb_media(aux_g, audio)
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
                image, audio, ling, lengths = bunch['image'], bunch['audio'], bunch['hubert'], bunch['frame_lengths']
                _, losses_g, aux_g = self.loss_generator(image, audio, ling, lengths)
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
                if args.base_args.test != 1:
                    self.wandb_media(aux_g, audio, "valid")