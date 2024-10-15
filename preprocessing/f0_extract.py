import os
import sys
import glob 
import pickle
import argparse
import librosa
import numpy as np

import torch
import torchaudio
from torch import nn

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from modules.F0Predictor.FCPEF0Predictor import FCPEF0Predictor
import utils

def trim_wav(wav, top_db=40):
    return librosa.effects.trim(wav, top_db=top_db)

def normalize_peak(wav, threshold=1.0):
    peak = np.abs(wav).max()
    if peak > threshold:
        wav = 0.98 * wav / peak
    return wav

parser = argparse.ArgumentParser()
parser.add_argument('--lrs3_root', type=str, default='/disk2/LRS3/original', help='original LRS3 dataset root')
parser.add_argument('--types', nargs='+', default='pretrain', help='pretrain / trainval / test')
args = parser.parse_args()

types = args.types # 'pretrain', 'trainval', 'test'
lrs3_root = args.lrs3_root # put your `original` directory here
auds_dir = lrs3_root.replace('original','modified')

f0p = "fcpe"
f0_predictor = utils.get_f0_predictor(f0p,512,44100,device=None,threshold=0.05)

for typ in types:
    avg_f0, mu_f0, var_f0 = {}, {}, {}
    speakers = os.listdir(os.path.join(lrs3_root, typ))
    speakers.sort()
    print(f'Type:{typ}, # of {len(speakers)} speakers')
    for i, speaker in enumerate(speakers):
        avg_f0[speaker] = []
        # if i > 2:
            # break
        print(f'Types:{typ}, Speaker index:{i}/{len(speakers)}', end='\r')
        wav_dirs = glob.glob(os.path.join(auds_dir, typ, speaker, '*.wav'))
        for j, wav_dir in enumerate(wav_dirs):
            wav16, sr = librosa.load(wav_dir, sr=16000)
            wav16, _ = trim_wav(wav16)
            wav16 = normalize_peak(wav16)
            wav44 = librosa.resample(wav16, orig_sr=16000, target_sr=44100)
            f0_441, uv = f0_predictor.compute_f0_uv(wav44)
            np.save(wav_dir.replace('.wav', '.f0.npy'), np.asanyarray((f0_441, uv), dtype=object))
            avg_f0[speaker] += list(f0_441*uv)
        temp = avg_f0[speaker]
        while 0.0 in temp:
            temp.remove(0.0)
        mu_f0[speaker] = np.mean(temp)
        var_f0[speaker] = np.std(temp)
    with open(os.path.join(auds_dir, f'avg_mu_{typ}.pickle'), 'wb') as fw:
            pickle.dump(mu_f0, fw)
    with open(os.path.join(auds_dir, f'avg_stdv_{typ}.pickle'), 'wb') as fw:
            pickle.dump(var_f0, fw)
print('\n')

# python preprocessing/f0_extract.py --lrs3_root '/disk2/LRS3/original' --types test trainval pretrain
# python preprocessing/f0_extract.py --lrs3_root '/disk2/LRS3/original' --types test
