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

from datasets.loader import Dataset
# from networks.discriminator import Discriminator
from hyface import Nansy, BShall_Ecapa, BShall_Nimbre
from hifigan.generator import HifiganGenerator
from hifigan.discriminator import (
    HifiganDiscriminator,
    feature_loss,
    discriminator_loss,
    generator_loss,
)

from utils import audio_utils

class Solver(object):
    def __init__(self, args):
        self.args = args
        self.wandb = wandb.init(entity='jjlee0721', project='hyface', group=args.base_args.group_name, config=args)
        self.global_step = 0
        wandb.run.name = args.base_args.exp_name
        # self.device = torch.device(f'cuda:{self.args.base_args.gpu}')
        
        self.cmap = np.array(plt.get_cmap('viridis').colors)

    def build_dataset(self, args):
        self.mp_context = torch.multiprocessing.get_context('fork')

        args.train.batch_size = int(args.train.batch_size / args.base_args.ngpus_per_node)
        self.trainset = Dataset(args, meta_root=os.path.join(args.base_args.meta_root, 'filelists'),
                        mode='train', datasets=['vctk'], sample_rate=args.data.sample_rate)
        self.validset = Dataset(args, meta_root=os.path.join(args.base_args.meta_root, 'filelists'),
                        mode='valid', datasets=['vctk'], sample_rate=args.data.sample_rate)
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
        if args.model.generator == "nansy":
            hyface = Nansy(args)
        elif args.model.generator == "bshall" and args.model.timbre.type == "nansy":
            hyface = BShall_Nimbre(args)
        elif args.model.generator == "bshall" and args.model.timbre.type == "ecapa":
            hyface = BShall_Ecapa(args)

        hyface = hyface.to('cuda:{}'.format(self.args.base_args.gpu))
        hyface = DistributedDataParallel(hyface, device_ids=[self.args.base_args.gpu], output_device=self.args.base_args.gpu, find_unused_parameters=True)
        if args.train.discriminator == True:
            gen = HifiganGenerator()
            gen = gen.to('cuda:{}'.format(self.args.base_args.gpu))
            gen = DistributedDataParallel(gen, device_ids=[self.args.base_args.gpu], output_device=self.args.base_args.gpu, find_unused_parameters=True)
            disc = HifiganDiscriminator()
            disc = disc.to('cuda:{}'.format(self.args.base_args.gpu))
            disc = DistributedDataParallel(disc, device_ids=[self.args.base_args.gpu], output_device=self.args.base_args.gpu, find_unused_parameters=True)
            self.net = {'hyface':hyface, 'gen':gen, 'disc':disc}
        else:
            self.net = {'hyface':hyface}

    def build_losses(self, args):
        pass

    def build_optimizers(self, args):
        optim_g = torch.optim.Adam(self.net['hyface'].parameters(), args.train.learning_rate_g,
                                    (args.train.beta1, args.train.beta2))
        if args.train.discriminator == True:
            optim_d = torch.optim.Adam(self.net['disc'].parameters(), args.train.learning_rate_d,
                                        (args.train.beta1, args.train.beta2))
            self.optim = {'g':optim_g, 'd':optim_d}
        else:
            self.optim = {'g':optim_g}

    def loss_generator(self, audio: np.ndarray, ling: torch.Tensor, lengths: np.ndarray) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Data to Cuda tensor
        audio = torch.tensor(audio, device=torch.device(f'cuda:{self.args.base_args.gpu}'))
        ling = torch.tensor(ling, device=torch.device(f'cuda:{self.args.base_args.gpu}'))
        length = torch.as_tensor(lengths, device=torch.device(f'cuda:{self.args.base_args.gpu}'))

        # B, T
        bsize, timestemps = audio.shape
        # [B, timb_global(192)], [B, timb_timbre(128), timb_tokens(50)]
        timbre = self.net['hyface'].module.analyze_timbre(audio)
        # [B, T], [B, T]
        mel_synth = self.net['hyface'].module.synthesize(ling, timbre)
        # truncating
        # synth = synth[..., :timestemps]
        
        # reconstruction loss
        mel_gt = self.net['hyface'].module.logmel.forward(audio)
        
        # mel_loss = (mel_synth-mel_gt).abs().mean()
        mel_loss = F.l1_loss(mel_synth, mel_gt, reduction="none")
        mel_loss = torch.sum(mel_loss, dim=(1,2)) / (mel_synth.size(-1)*length)
        mel_loss = torch.mean(mel_loss)
        rctor_loss = mel_loss
        
        if self.args.train.discriminator == True:
            audio_synth = self.net['gen'](mel_synth)
            # discriminative
            _, fmaps_r = self.net['disc'](audio.unsqueeze(1))
            logits_f, fmaps_f = self.net['disc'](audio_synth)

            fmap_loss = feature_loss(fmaps_r, fmaps_f)
            d_fake, _ = generator_loss(logits_f)
            # reweighting
            weight = (rctor_loss / fmap_loss).detach()
            # loss = weight * fmap_loss + rctor_loss
            loss = 10*rctor_loss + 0.5 * fmap_loss + d_fake
            losses = {
                'gen/loss': loss.item(),
                'gen/d-fake': d_fake.item(),
                'gen/fmap': fmap_loss.item(),
                'gen/rctor': rctor_loss.item()}
            media =  {
                'mel_synth': mel_synth.cpu().detach().numpy(),
                'mel_gt': mel_gt.cpu().detach().numpy(),
                'audio_synth': audio_synth.squeeze(1).cpu().detach().numpy(),
                'audio_gt': audio.cpu().detach().numpy()}
        else:
            loss = rctor_loss
            losses = {
                'gen/loss': loss.item(),
                'gen/rctor': rctor_loss.item()}
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
                    torch.norm(p.grad).item() for p in self.net['hyface'].parameters() if p.grad is not None])
                param_norm = np.mean([
                    torch.norm(p).item() for p in self.net['hyface'].parameters() if p.dtype == torch.float32])
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
        if self.args.train.discriminator == True:
            # audio sample
            audio_gt = wandb.Audio(gt_audio[0], sample_rate=self.args.data.sample_rate)
            audio_synth = wandb.Audio(aux_g['audio_synth'][0], sample_rate=self.args.data.sample_rate)
            wandb.log({"speech/"+ phase : audio_gt, "speech-synth/"+ phase : audio_synth})
        else:
            # audio sample
            audio_gt = wandb.Audio(gt_audio[0], sample_rate=self.args.data.sample_rate)
            wandb.log({"speech/"+ phase : audio_gt})

    def train(self, args, epoch):
        self.net['hyface'].train()
        for batch_idx, bunch in enumerate(self.train_loader):
            audio, ling, lengths = bunch['audio'], bunch['hubert'], bunch['frame_lengths']
            loss_g, losses_g, aux_g = self.loss_generator(audio, ling, lengths)
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
            self.net['hyface'].eval()
            for batch_idx, bunch in enumerate(self.valid_loader):
                audio, ling, lengths = bunch['audio'], bunch['hubert'], bunch['frame_lengths']
                _, losses_g, aux_g = self.loss_generator(audio, ling, lengths)
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