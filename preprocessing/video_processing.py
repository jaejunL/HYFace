import os
import sys
import glob
import argparse

import cv2
import torchaudio

parser = argparse.ArgumentParser()
parser.add_argument('--lrs3_root', type=str, default='/disk2/LRS3/original', help='original LRS3 dataset root')
parser.add_argument('--types', nargs='+', default=['pretrain'], help='pretrain / trainval / test')
args = parser.parse_args()

types = args.types # 'pretrain', 'trainval', 'test'
lrs3_root = args.lrs3_root # put your `original` directory here
temp_root = lrs3_root.replace('original','modified')

max_wav_lengths = 16000 * 10
for typ in types:
    speakers = os.listdir(os.path.join(lrs3_root, typ))
    speakers.sort()
    print(f'\nType:{typ}, # of {len(speakers)} speakers')
    for i, speaker in enumerate(speakers):
        print(f'Types:{typ}, Speaker index:{i}/{len(speakers)}', end='\r')
        os.makedirs(os.path.join(temp_root, typ, speaker), exist_ok=True)
        video_paths = glob.glob(os.path.join(lrs3_root, typ, speaker, '*.mp4'))

        for video_path in video_paths:
            video_name = os.path.basename(video_path)
            wav_path = f"{temp_root}/{typ}/{speaker}/{video_name[:-4]}.wav"
            
            # For 16k audio extracting
            os.system(f'ffmpeg -y -i {video_path} -filter:v fps=25 -ac 1 -ar 16000 {temp_root}/{typ}/{speaker}/{video_name} >/dev/null 2>&1')
            os.system(f'ffmpeg -y -i {temp_root}/{typ}/{speaker}/{video_name} {wav_path} >/dev/null 2>&1')
            os.system(f'rm -r {temp_root}/{typ}/{speaker}/{video_name} >/dev/null 2>&1')

            # For audio
            y, sr = torchaudio.load(wav_path)
            if y.shape[-1] > max_wav_lengths:
                quot = int(y.shape[-1]/max_wav_lengths)
                cnt = 0
                for k in range(quot):
                    if k == 0:
                        continue
                    split_y = y[:,(k-1)*max_wav_lengths:k*max_wav_lengths]
                    write_dir = wav_path.replace('.wav',f'_{str(cnt)}.wav')
                    torchaudio.save(write_dir, split_y, sample_rate=16000, encoding="PCM_S", bits_per_sample=16)
                    cnt += 1
                write_dir = wav_path.replace('.wav',f'_{str(cnt)}.wav')
                split_y = y[:,k*max_wav_lengths:]
                torchaudio.save(write_dir, split_y, sample_rate=16000, encoding="PCM_S", bits_per_sample=16)
                os.system(f'rm -r {wav_path}')
            else:
                pass            
            
            # For image
            vc = cv2.VideoCapture(video_path)
            rval = vc.isOpened()
            output_path = video_path.replace('original','modified')[:-4]
            c = 0
            if not rval:
                f = open(os.path.join(temp_root, 'failed.txt'), 'a+')
                f.write(video_path + '\n')
                f.close()
            while rval:
                c = c + 1
                rval, frame = vc.read()
                if c == 1:
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    else:
                        if (vc.get(cv2.CAP_PROP_FRAME_COUNT)==len(glob.glob(output_path+'/*.jpg'))):
                            break
                if rval:
                    cv2.imwrite(output_path + '/' + str(c).zfill(4) + '.jpg', frame) 
                else:
                    break
            vc.release()

# python preprocessing/video_processing.py --lrs3_root '/disk2/LRS3/original' --types test trainval pretrain
# python preprocessing/video_processing.py --lrs3_root '/disk2/LRS3/original' --types trainval
