import os
import glob 
import argparse
import torch
import torchaudio
from torch import nn
from transformers import HubertConfig, HubertModel

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        # The final projection layer is only used for backward compatibility.
        # Following https://github.com/auspicious3000/contentvec/issues/6
        # Remove this layer is necessary to achieve the desired outcome.
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)
        
parser = argparse.ArgumentParser()
parser.add_argument('--lrs3_root', type=str, default='/disk2/LRS3/original', help='original LRS3 dataset root')
parser.add_argument('--types', nargs='+', default='pretrain', help='pretrain / trainval / test')
args = parser.parse_args()

types = args.types # 'pretrain', 'trainval', 'test'
lrs3_root = args.lrs3_root # put your `original` directory here
auds_dir = lrs3_root.replace('original','modified')

model = HubertModelWithFinalProj.from_pretrained("lengyue233/content-vec-best")
model = model.to('cuda:0')

for typ in types:
    speakers = os.listdir(os.path.join(lrs3_root, typ))
    speakers.sort()
    print(f'Type:{typ}, # of {len(speakers)} speakers')
    for i, speaker in enumerate(speakers):
        # if i > 2:
            # break
        print(f'Types:{typ}, Speaker index:{i}/{len(speakers)}', end='\r')
        wav_dirs = glob.glob(os.path.join(auds_dir, typ, speaker, '*.wav'))
        for j, wav_dir in enumerate(wav_dirs):
            y, sr = torchaudio.load(wav_dir)
            output = model(y.to(device=torch.device('cuda:0')))
            torch.save(output["last_hidden_state"].detach().cpu().squeeze(), wav_dir.replace('.wav', '.emb'))
print('\n')

# CUDA_VISIBLE_DEVICES=0 python preprocessing/contentvec_save.py --lrs3_root '/disk2/LRS3/original' --types test trainval pretrain
