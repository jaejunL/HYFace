import os
import time
import wandb

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from utils import utils

def train(args, run=None):
    wandb.require(experiment="service")
    wandb.setup()
    
    if args.model.name == "hyface":
        from solver import Solver
    elif args.model.name == "f2v":
        if args.model.timbre.type == 'ecapa':
            from solver_f2v import Solver
        elif args.model.timbre.type == 'nimbre':
            from solver_fimber import Solver
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

    # Loading
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(os.path.join(args.base_args.base_dir, 'checkpoints'), "G_*.pth"), solver.net['hyface'], solver.optim['g'])
    except:
        epoch_str = 1

    if args.train.discriminator == True:
        try:
            _, _, _, _ = utils.load_checkpoint(utils.latest_checkpoint_path(os.path.join(args.model.hifigan.generator_path, 'checkpoints'), "G_*.pth"), solver.net['gen'], None)
        except:
            pass
        try:
            _, _, _, _ = utils.load_checkpoint(utils.latest_checkpoint_path(os.path.join(args.model.hifigan.discriminator_path, 'checkpoints'), "D_*.pth"), solver.net['disc'], None)
        except:
            pass

    if args.base_args.rank % ngpus_per_node == 0:
        print("start from epoch {}".format(epoch_str))

    for epoch in range(epoch_str, args.train.total_epochs + 1):
        start_time = time.time()
        solver.train_sampler.set_epoch(epoch)

        losses_keys = solver.train(args, epoch)
        val_loss = solver.validation(args, epoch, losses_keys)

        # save checkpoint
        if args.base_args.rank % ngpus_per_node == 0:
            if epoch % args.train.save_model_interval == 0:
                checkpoint_dir = os.path.join(args.base_args.base_dir, 'checkpoints')
                utils.save_checkpoint(solver.net[args.model.name], solver.optim['g'], None, epoch,
                            os.path.join(checkpoint_dir, "G_{}.pth".format(epoch)))
            end_time = time.time()
            # solver.save_audio(args, epoch, solver.validset[0])
            print(f'Training time:{end_time-start_time:.1f} sec')
        time.sleep(1)


if __name__ == "__main__":
    print("This is 'train.py' code")