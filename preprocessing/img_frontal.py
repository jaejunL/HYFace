import os
import glob
import shutil
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--lrs3_root', type=str, default='/disk2/LRS3/original', help='original LRS3 dataset root')
parser.add_argument('--types', nargs='+', default='pretrain', help='pretrain / trainval / test')
args = parser.parse_args()

types = args.types # 'pretrain', 'trainval', 'test'
lrs3_root = args.lrs3_root # put your `original` directory here
temp_root = lrs3_root.replace('original','modified')
imgs_dir = lrs3_root.replace('original','modified/imgs_frontal')

xml_dir = os.path.join(os.path.dirname(lrs3_root), 'haarcascade_frontalface_default.xml')
face_classifier_frontal = cv2.CascadeClassifier(xml_dir)

for typ in types:
    speakers = os.listdir(os.path.join(lrs3_root, typ))
    speakers.sort()
    print(f'Type:{typ}, # of {len(speakers)} speakers')
    for i, speaker in enumerate(speakers):
        print(f'Types:{typ}, Speaker index:{i}/{len(speakers)}', end='\r')
        os.makedirs(os.path.join(temp_root, typ, speaker), exist_ok=True)
        image_folders = [dat for dat in glob.glob(os.path.join(temp_root, typ, speaker, '*')) if '.wav' not in os.path.basename(dat)]
        
        # if i > 2:
            # break
        for image_folder in image_folders:
            save_dir = os.path.join(imgs_dir, typ, speaker, os.path.basename(image_folder))
            os.makedirs(save_dir, exist_ok=True)
            image_dirs = glob.glob(os.path.join(image_folder, '*.jpg'))
            for image_dir in image_dirs:
                image = cv2.imread(image_dir)
                faces_frontal = face_classifier_frontal.detectMultiScale(image)
                if len(faces_frontal) > 0:
                    if faces_frontal[0][2] > 80:
                        shutil.copy(image_dir, os.path.join(save_dir, os.path.basename(image_dir)))
print('\n')
                        
# python preprocess/img_frontal.py --lrs3_root '/disk2/LRS3/original' --types test trainval pretrain
# python preprocess/img_frontal.py --lrs3_root '/disk2/LRS3/original' --types pretrain
                        