import os
import glob 
import shutil
import argparse
import torchaudio

parser = argparse.ArgumentParser()
parser.add_argument('--lrs3_root', type=str, default='/disk2/LRS3/original', help='original LRS3 dataset root')
parser.add_argument('--types', nargs='+', default='pretrain', help='pretrain / trainval / test')
args = parser.parse_args()

types = args.types # 'pretrain', 'trainval', 'test'
lrs3_root = args.lrs3_root # put your `original` directory here
temp_root = lrs3_root.replace('original','temporary')
imgs_dir = lrs3_root.replace('original','modified/imgs')
auds_dir = lrs3_root.replace('original','modified/auds')

max_lengths = 16000 * 10
for typ in types:
    speakers = os.listdir(os.path.join(lrs3_root, typ))
    speakers.sort()
    print(f'Type:{typ}, # of {len(speakers)} speakers')
    for i, speaker in enumerate(speakers):
        # if i > 2:
            # break
        print(f'Types:{typ}, Speaker index:{i}/{len(speakers)}', end='\r')
        os.makedirs(os.path.join(auds_dir, typ, speaker), exist_ok=True)
        wav_dirs = glob.glob(os.path.join(temp_root, typ, speaker, '*.wav'))
        for j, wav_dir in enumerate(wav_dirs):
            y, sr = torchaudio.load(wav_dir)
            if y.shape[-1] > max_lengths:
                quot = int(y.shape[-1]/max_lengths)
                cnt = 0
                for k in range(quot):
                    if k == 0:
                        continue
                    split_y = y[:,(k-1)*max_lengths:k*max_lengths]
                    write_dir = wav_dir.replace('temporary','modified/auds').replace('.wav',f'_{str(cnt)}.wav')
                    torchaudio.save(write_dir, split_y, sample_rate=16000, encoding="PCM_S", bits_per_sample=16)
                    cnt += 1
                write_dir = wav_dir.replace('temporary','modified/auds').replace('.wav',f'_{str(cnt)}.wav')
                split_y = y[:,k*max_lengths:]
                torchaudio.save(write_dir, split_y, sample_rate=16000, encoding="PCM_S", bits_per_sample=16)
            else:
                write_dir = wav_dir.replace('temporary','modified/auds')
                shutil.copy(wav_dir, write_dir)
print('\n')

# python preprocess/wav_split.py --lrs3_root '/disk2/LRS3/original' --types test trainval pretrain


