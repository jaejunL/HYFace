import os
import time
import wave
import random
import numpy as np
from typing import Dict, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from utils import data_utils, audio_utils

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                args,
                meta_root = 'filelists',
                mode='train',
                datasets=['vctk'],
                sample_rate = 16000, 
                ):
        self.args = args
        self.mode = mode
        self.datasets = datasets
        self.sample_rate = sample_rate
        self.max_sec = 4
        self.max_len = sample_rate * self.max_sec
        self.data_files = []
        for dset in datasets:
            meta_file_path = os.path.join(meta_root, '{}_{}.txt').format(dset, mode)
            files = data_utils.load_text(meta_file_path)
            self.data_files += files

    # def __getitem__(self, index):
    #     # audio load
    #     audio_path = self.data_files[index]
    #     audio = audio_utils.load_wav(path=audio_path, max_len=self.max_len)
    #     # hubert load
    #     if 'vctk' in audio_path:
    #         hubert_path = audio_path.replace('wav16_cleaned', 'hubert_soft').replace('.wav', '.emb')
    #     elif 'VoxCeleb2' in audio_path:
    #         hubert_path = audio_path.replace('original', 'modified/hubert_soft').replace('.wav', '.emb')
    #     hubert_emb = torch.load(hubert_path).squeeze(0)[:self.max_sec*50]
    #     return audio, hubert_emb

    def __getitem__(self, index):
        # audio load
        audio_path = self.data_files[index]
        audio, pos = audio_utils.load_wav(path=audio_path, max_len=self.max_len, pos='random')
        # hubert load
        hubert_pos = int(pos / 320)
        if 'vctk' in audio_path:
            hubert_path = audio_path.replace('wav16_cleaned', 'hubert_soft').replace('.wav', '.emb')
        elif 'VoxCeleb2' in audio_path:
            hubert_path = audio_path.replace('original', 'modified/hubert_soft').replace('.wav', '.emb')
        hubert_emb = torch.load(hubert_path).squeeze(0)[hubert_pos:hubert_pos+self.max_sec*50]
        return audio, hubert_emb

    def collate(self, bunch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Collate bunch of datum to the batch data.
        Args:
            bunch: B x [np.float32; [T]], speech signal.
        Returns:
            batch data.
                speeches: [np.float32; [B, T]], speech signal.
                lengths: [np.long; [B]], speech lengths.
        """
        # [B]
        frame_lengths = np.array([hubert_emb.shape[0]*2 for _, hubert_emb in bunch])
        #print(frame_lengths)
        #for audio, _ in bunch:
        #    print(audio.shape)
        #stop
        # []
        max_framelen = frame_lengths.max()
        # [B, T]
        audio_pad = np.stack([
            np.pad(audio[:frame_lengths[i]*160], [0, max_framelen*160 - frame_lengths[i]*160]) for i, (audio, _) in enumerate(bunch)])
        hubert_pad = pad_sequence([hubert_emb for _, hubert_emb in bunch]).transpose(0,1)
        data = {'audio':audio_pad, 'hubert':hubert_pad.transpose(-1,-2), 'frame_lengths':frame_lengths}
        return data

#        # [B]
#        frame_lengths = np.array([audio.shape[0] for audio, _ in bunch])
#        # []
#        max_framelen = frame_lengths.max()
#        # [B, T]
#        audio_pad = np.stack([
#            np.pad(audio, [0, max_framelen - frame_lengths[i]]) for i, (audio, _) in enumerate(bunch)])
#        hubert_pad = pad_sequence([hubert_emb for _, hubert_emb in bunch]).transpose(0,1)
#        data = {'audio':audio_pad[:,:hubert_pad.shape[-1]*320], 'hubert':hubert_pad.transpose(-1,-2), 'frame_lengths':frame_lengths}
#        return data
    
    def __len__(self):
        return len(self.data_files)





if __name__ == '__main__':
    import os
    import json
    import argparse
    import numpy as np
    import sys
    sys.path.append('../../training')
    sys.path.append('../networks/nansypp')

    from torch.utils.data import DataLoader

    from networks.discriminator import Discriminator
    from nansypp import Nansypp
    from hyface import Nansy, BShall_Nimbre, BShall_Ecapa
    from utils import audio_utils
    from utils.data_utils import phoneme_inventory, decollate_tensor, combine_fixed_length
    from utils import utils
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', default=None, help='gpus')
    parser.add_argument('--port', default='6000', type=str, help='port')
    parser.add_argument('--n_nodes', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int) # n개의 gpu가 한 node: n개의 gpu마다 main_worker를 실행시킨다.
    parser.add_argument('--rank', default=0, type=int, help='ranking within the nodes')
    base_args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_num) for gpu_num in base_args.gpus])
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = base_args.port
    
    config_path = '/home/jaejun/temp_jaejun/hyface/bshall_ecapa/logs/config.json'
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    args = utils.HParams(**config)

    args = utils.HParams(**config)
    args.base_args = base_args

    if args.model.generator == "nansy":
        hyface = Nansy(args)
    elif args.model.generator == "bshall" and args.model.timbre.type == "nansy":
        hyface = BShall_Nimbre(args)
    elif args.model.generator == "bshall" and args.model.timbre.type == "ecapa":
        hyface = BShall_Ecapa(args)
        
    args.meta_root = '/home/jaejun/hyface/training'
    args.base_dir = '/home/jaejun/temp_jaejun/hyface/bshall_ecapa/checkpoints'

    index = 2000
    checkpoint_path = f'/home/jaejun/temp_jaejun/hyface/bshall_ecapa/checkpoints/G_{index}.pth'
    hyface, _, _, _ = utils.load_checkpoint(checkpoint_path, hyface, None)
    
    
    # valid_filelist = args.data.validation_files
    # dataset = VoxCeleb2(typ='test', filelist=valid_filelist, args=args)
    # valid_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8,
    #                 collate_fn=None, pin_memory=True, drop_last=False,
    #                 worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    # for i, (speaker, filename, audio) in enumerate(valid_loader, 1):
    #     print('i:{}, audio shape{}'.format(speaker, audio.shape))
    #     if i > 2000:
    #         break

    # validset2 = ImgSpkLoader(typ='test', filelist=valid_filelist, args=args)
    # valid_loader2 = DataLoader(validset2, batch_size=1, shuffle=False, num_workers=8,
    #                 collate_fn=None, pin_memory=True, drop_last=False,
    #                 worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    # for i, (speaker, filename, img) in enumerate(valid_loader2, 1):
    #     print('i:{}, img shape{}'.format(speaker, img.shape))
    #     if i > 2000:
    #         break