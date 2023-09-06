import os
import time
from time import gmtime, strftime
import wandb

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from solver import Solver
from solver_vc import Solver_VC
from utils import utils

def train(args, run=None):
    wandb.require(experiment="service")
    wandb.setup()
    
    if 'vc' in args.base_args.config:
        solver = Solver_VC(args)
    else:
        solver = Solver(args)
    
    ngpus_per_node = int(torch.cuda.device_count()/args.base_args.n_nodes)
    print("use {} gpu machine".format(ngpus_per_node))
    args.base_args.world_size = ngpus_per_node * args.base_args.n_nodes
    mp.spawn(worker, nprocs=ngpus_per_node, args=(solver, ngpus_per_node, args))

def worker(gpu, solver, ngpus_per_node, args):
    args.base_args.rank = args.base_args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl', 
                            world_size=args.base_args.world_size,
                            init_method='env://',
                            rank=args.base_args.rank)
    args.base_args.gpu = gpu
    args.base_args.ngpus_per_node = ngpus_per_node

    solver.build_dataset(args)
    solver.build_models(args)
    solver.build_losses(args)
    solver.build_optimizers(args)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(os.path.join(args.base_args.base_dir, 'checkpoints'), "G_*.pth"), solver.net['g'], solver.optim['g'])
    except:
        epoch_str = 1

    try:
        _, _, _, _ = utils.load_checkpoint(utils.latest_checkpoint_path(os.path.join(args.base_args.base_dir, 'checkpoints'), "V_*.pth"), solver.net['v'], solver.optim['v'])
    except:
        if 'v' in args.base_args.pretrain[0] and args.data.sample_rate == 22050:
            state_dict = utils.load_checkpoint_hifigan(args.pretrain.hifigan_v, solver.net['v'], token='generator')
            if 'v' in args.base_args.fixtrain[0]:
                solver.net['v'].eval()
            # solver.net['v'].remove_weight_norm()
            if args.base_args.rank % ngpus_per_node == 0:
                print("Pretrained vocoder model is loaded")

    solver.build_scheduler(args, epoch_str)

    # if args.resume:
    # solver.validate(args, int(start_epoch/args.save_model_interval))

    if args.base_args.rank % ngpus_per_node == 0:
        print("start from epoch {}".format(epoch_str))

    for epoch in range(epoch_str, args.train.total_epochs + 1):
        start_time = time.time()
        solver.train_sampler.set_epoch(epoch)

        train_loss = solver.train(args, epoch)
        val_loss = solver.test(args, epoch)

        solver.scheduler['g'].step(val_loss)
        if 'v' not in args.base_args.fixtrain[0]:
            solver.scheduler['v'].step(val_loss)

        # save checkpoint
        if args.base_args.rank % ngpus_per_node == 0:
            if epoch % args.train.save_model_interval == 0:
                checkpoint_dir = os.path.join(args.base_args.base_dir, 'checkpoints')
                utils.save_checkpoint(solver.net['g'], solver.optim['g'], args.train.learning_rate, epoch,
                            os.path.join(checkpoint_dir, "G_{}.pth".format(epoch)))
                if 'v' not in args.base_args.fixtrain[0]:
                    utils.save_checkpoint(solver.net['v'], solver.optim['v'], args.train.learning_rate, epoch,
                            os.path.join(checkpoint_dir, "V_{}.pth".format(epoch)))      
            end_time = time.time()
            # solver.save_audio(args, epoch, solver.validset[0])
            print(f'Training time:{end_time-start_time:.1f} sec')
        time.sleep(1)


if __name__ == "__main__":
    print("This is 'train.py' code")