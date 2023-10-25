import os
import sys
import json
import time
import wave
import random
import numpy as np
from typing import Dict, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
import PIL
from torchvision import transforms

from utils import data_utils, utils, audio_utils

class FaceNAudio_Spkwise(torch.utils.data.Dataset):
    def __init__(self,
                args,
                meta_root = 'filelists/VGG_Face',
                mode='train',
                img_datasets=['VGG_Face_Spk'],
                sample_rate = 16000, 
                ):
        self.args = args
        self.mode = mode
        self.img_datasets = img_datasets
        self.sample_rate = sample_rate
        self.max_sec = 4
        self.max_len = sample_rate * self.max_sec
        self.data_files = []
        for dset in img_datasets:
            meta_file_path = os.path.join(meta_root, '{}_{}.txt').format(dset, mode)
            files = data_utils.load_text(meta_file_path)
            self.data_files += files
        self.data_files_len = len(self.data_files)
        self.trans = transforms.Compose([transforms.Resize((args.features.image.size,args.features.image.size), interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(args.features.image.size), transforms.ToTensor(),
                # transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])])
                transforms.Normalize(mean=[0.4829, 0.4049, 0.3712], std=[0.2643, 0.2398, 0.2335])])

    def __getitem__(self, index):
        spkr = self.data_files[index % self.data_files_len]
        
        # image load
        img_names = os.listdir(os.path.join(self.args.data.img_datadir, spkr))
        random_img_name = random.choice(img_names)
        img_path = os.path.join(self.args.data.img_datadir, spkr, random_img_name)
        img = Image.open(img_path)
        img_tensor = self.trans(img)

        # audio load
        aud_names = os.listdir(os.path.join(self.args.data.aud_datadir, spkr))
        random_aud_name = random.choice(aud_names)
        aud_path = os.path.join(self.args.data.aud_datadir, spkr, random_aud_name)
        aud = audio_utils.load_wav(path=aud_path, sr=self.args.data.sample_rate,
                                   max_len=self.max_len, pos='random')
        aud_pad = np.pad(aud[:self.max_len], [0, self.max_len - len(aud)])
        return img_tensor, aud_pad

    def __len__(self):
        return len(self.data_files)*self.args.data.img_nums

class FaceNEcapaAVg_Filewise(torch.utils.data.Dataset):
    def __init__(self,
                args,
                meta_root = 'filelists/VGG_Face',
                mode='train',
                img_datasets=['VGG_Face'],
                sample_rate = 16000,
                ):
        self.args = args
        self.mode = mode
        self.img_datasets = img_datasets
        self.data_files = []
        for dset in img_datasets:
            meta_file_path = os.path.join(meta_root, '{}_{}.txt').format(dset, mode)
            files = data_utils.load_text(meta_file_path)
            self.data_files += files
        self.data_files_len = len(self.data_files)
        self.trans = transforms.Compose([transforms.Resize((args.features.image.size,args.features.image.size), interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(args.features.image.size), transforms.ToTensor(),
                # transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])])
                transforms.Normalize(mean=[0.4829, 0.4049, 0.3712], std=[0.2643, 0.2398, 0.2335])])

    def path_to_spk(self, path):
        return os.path.basename(os.path.dirname(path))
                                
    def __getitem__(self, index):
        # image load
        img_path = self.data_files[index]
        img = Image.open(img_path)
        img_tensor = self.trans(img)

        # audio load
        spk = self.path_to_spk(img_path)
        ecapa_avg = np.load(os.path.join(self.args.data.aud_datadir, spk + '.npy'))
        
        return img_tensor, torch.tensor(ecapa_avg)

    def __len__(self):
        return len(self.data_files)


class Faubert_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                args,
                meta_root = 'filelists',
                mode='train',
                datasets=['VGG_Face'],
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
        self.trans = transforms.Compose([transforms.Resize((args.features.image.size,args.features.image.size), interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(args.features.image.size), transforms.ToTensor(),
                transforms.Normalize(mean=[0.4829, 0.4049, 0.3712], std=[0.2643, 0.2398, 0.2335])])

    def __getitem__(self, index):
        # image load
        img_path = self.data_files[index]
        img = Image.open(img_path)
        img_tensor = self.trans(img)
        
        # audio load
        aud_folder = os.path.dirname(img_path).replace('VGG_Face2/data','VoxCeleb2/VoxCeleb2')
        aud_names = os.listdir(aud_folder)
        random_aud_name = random.choice(aud_names)
        aud_path = os.path.join(aud_folder, random_aud_name)
        aud, pos = audio_utils.load_wav(path=aud_path, max_len=self.max_len, pos='random')
        
        # hubert load
        hubert_pos = int(pos / 320)
        hubert_path = aud_path.replace('original', 'modified/hubert_soft').replace('.wav', '.emb')
        hubert_emb = torch.load(hubert_path).squeeze(0)[hubert_pos:hubert_pos+self.max_sec*50]
        return img_tensor, aud, hubert_emb

    def collate(self, bunch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Collate bunch of datum to the batch data.
        Args:
            bunch: B x [np.float32; [T]], speech signal.
        Returns:
            batch data.
                speeches: [np.float32; [B, T]], speech signal.
                lengths: [np.long; [B]], speech lengths.
        """
        img_pad = torch.stack([img_tensor for img_tensor, _, _ in bunch])
        # [B]
        frame_lengths = np.array([hubert_emb.shape[0]*2 for _, _, hubert_emb in bunch])
        # []
        max_framelen = frame_lengths.max()
        # [B, T]
        aud_pad = np.stack([
            np.pad(aud[:frame_lengths[i]*160], [0, max_framelen*160 - frame_lengths[i]*160]) for i, (img, aud, _) in enumerate(bunch)])
        hubert_pad = pad_sequence([hubert_emb for _, _, hubert_emb in bunch]).transpose(0,1)
        data = {'image':img_pad, 'audio':aud_pad, 'hubert':hubert_pad.transpose(-1,-2), 'frame_lengths':frame_lengths}
        return data

    def __len__(self):
        return len(self.data_files)


class TMP_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                meta_root = 'filelists',
                mode='train',
                img_datasets=['VGG_Face'],
                sample_rate = 16000, 
                ):
        self.mode = mode
        self.img_datasets = img_datasets
        self.sample_rate = sample_rate
        self.max_sec = 4
        self.max_len = sample_rate * self.max_sec
        self.data_files = []
        for dset in img_datasets:
            meta_file_path = os.path.join(meta_root, '{}_{}.txt').format(dset, mode)
            files = data_utils.load_text(meta_file_path)
            self.data_files += files
        self.data_files_len = len(self.data_files)
        self.trans = transforms.Compose([transforms.Resize((224,224), interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(224), transforms.ToTensor()])
        
    def get_image(self, index):
        img = Image.open()
        
    def __getitem__(self, index):
        img_path = self.data_files[index]
        img = Image.open(img_path)
        img_tensor = self.trans(img)
        return img_tensor

    def __len__(self):
        return len(self.data_files)
    
# class AudImgLoader(torch.utils.data.Dataset):
#     def __init__(self,
#                 args,
#                 meta_root = 'filelists',
#                 mode='train',
#                 datasets=['VGG_Face2'],
#                 sample_rate = 16000, 
#                 ):
#         self.args = args
#         self.mode = mode
#         self.datasets = datasets
#         self.sample_rate = sample_rate
#         self.max_sec = 4
#         self.max_len = sample_rate * self.max_sec        
#         self.data_files = []
#         for dset in datasets:
#             meta_file_path = os.path.join(meta_root, '{}_{}.txt').format(dset, mode)
#             files = data_utils.load_text(meta_file_path)
#             self.data_files += files
            
            
                    
#         self.speakers = load_filepaths_and_text(filelist)
#         self.aud_root_dir = args.data.audio_root_dir
#         self.img_root_dir = args.data.image_root_dir
        
#         self.sr = args.data.sampling_rate
#         self.target_sec = args.data.target_sec

#         self.trans = transforms.Compose([transforms.Resize((args.data.image_size,args.data.image_size)), 
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
                            
#         self.batch_size = args.train.batch_size

#         if typ in ['train', 'dev']:
#             self.typ = 'dev'
#         else:
#             self.typ = typ

#     def get_speaker(self, index):
#         pos_speaker = self.speakers[index]
#         return pos_speaker

#     def get_audio(self, pos_speaker):
#         aud_full_path = os.path.join(self.aud_root_dir, self.typ, pos_speaker[0])
#         aud_files = os.listdir(aud_full_path)
#         random.shuffle(aud_files)
#         audio1 = self.load_wav(os.path.join(aud_full_path, aud_files[0]))
#         audio2 = self.load_wav(os.path.join(aud_full_path, aud_files[1]))
#         return audio1, audio2

#     def get_image(self, pos_speaker):
#         # Below code is need because of difference configuration of train/test between VoxCeleb2 & VGGFace2
#         if os.path.isdir(os.path.join(self.img_root_dir, 'train', pos_speaker[0])):
#             img_full_path = os.path.join(os.path.join(self.img_root_dir, 'train', pos_speaker[0]))
#         else:
#             img_full_path = os.path.join(os.path.join(self.img_root_dir, 'test', pos_speaker[0]))
#         img_files = os.listdir(img_full_path)
#         random.shuffle(img_files)
#         image1 = Image.open(os.path.join(img_full_path, img_files[0]))
#         image2 = Image.open(os.path.join(img_full_path, img_files[1]))
#         return image1, image2

#     def load_wav(self, path):
#         audio = wave.open(path, 'r')
#         audio_len = audio.getnframes()
#         target_len = self.sr * self.target_sec

#         """ 
#         crop or repeat audio to be the target length
#         """			
#         if audio_len < target_len:
#             # repeat
#             audio.setpos(0)
#             audio = audio.readframes(audio_len) # load all frames since it is shorter than target_len
#             audio = np.frombuffer(audio, dtype=np.int16)
#             audio = np.float32(audio / 2**15) # to float 32
#             audio = repeat_aud(audio, org_len=audio_len, desired_len=target_len)
#         else:
#             # crop
#             end = random.randint(target_len, audio_len)
#             start = int(end-target_len)
#             audio.setpos(start)
#             audio = audio.readframes(target_len)
#             audio = np.frombuffer(audio, dtype=np.int16)
#             audio = np.float32(audio / 2**15)
#         return audio

#     def normalize_audio(self, audio):
#         return rms_normalize(audio, 0.01)

#     def normalize_image(self, image):
#         return self.trans(image)

#     def __getitem__(self, index):
#         pos_speaker = self.get_speaker(index)
#         audio1, audio2 = self.get_audio(pos_speaker)
#         audio1 = self.normalize_audio(audio1)
#         audio2 = self.normalize_audio(audio2)

#         image1, image2 = self.get_image(pos_speaker)
#         image1 = self.normalize_image(image1)
#         image2 = self.normalize_image(image2)
#         return audio1, audio2, image1, image2
    
#     def __len__(self):
#         return len(self.speakers)


# class AudSpkLoader(torch.utils.data.Dataset):
#     def __init__(self, typ, filelist, args):
#         self.speakers = load_filepaths_and_text(filelist)
#         self.aud_root_dir = args.data.audio_root_dir
#         self.sr = args.data.sampling_rate
#         self.target_sec = args.data.target_sec
#         self.batch_size = args.train.batch_size
#         if typ in ['train', 'dev']:
#             self.typ = 'dev'
#         else:
#             self.typ = typ
#         self.aud_file_full_paths = []
#         self.get_valid_list()

#     def get_valid_list(self):
#         for speaker in self.speakers:
#             aud_full_path = os.path.join(self.aud_root_dir, self.typ, speaker[0])
#             aud_files = os.listdir(aud_full_path)
#             for aud_file in aud_files:
#                 aud_file_full_path = os.path.join(aud_full_path, aud_file)
#                 self.aud_file_full_paths.append(aud_file_full_path)

#     def load_wav(self, path):
#         audio = wave.open(path, 'r')
#         audio_len = audio.getnframes()
#         target_len = self.sr * self.target_sec

#         """ 
#         crop or repeat audio to be the target length
#         """			
#         if audio_len < target_len:
#             # repeat
#             audio.setpos(0)
#             audio = audio.readframes(audio_len) # load all frames since it is shorter than target_len
#             audio = np.frombuffer(audio, dtype=np.int16)
#             audio = np.float32(audio / 2**15) # to float 32
#             audio = repeat_aud(audio, org_len=audio_len, desired_len=target_len)
#         else:
#             # crop
#             audio.setpos(0)
#             audio = audio.readframes(target_len)
#             audio = np.frombuffer(audio, dtype=np.int16)
#             audio = np.float32(audio / 2**15)
#         return audio

#     def normalize_audio(self, audio):
#         return rms_normalize(audio, 0.01)

#     def __getitem__(self, index):
#         audio = self.load_wav(self.aud_file_full_paths[index])
#         audio = self.normalize_audio(audio)
#         filename = self.aud_file_full_paths[index].split('/')[-1]
#         speaker = self.aud_file_full_paths[index].split('/')[-2]
#         return speaker, filename, audio
    
#     def __len__(self):
#         return len(self.aud_file_full_paths)


# class ImgSpkLoader(torch.utils.data.Dataset):
#     def __init__(self, typ, filelist, args):
#         self.speakers = load_filepaths_and_text(filelist)
#         self.img_root_dir = args.data.image_root_dir
#         self.trans = transforms.Compose([transforms.Resize((args.data.image_size,args.data.image_size)), 
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
#         self.batch_size = args.train.batch_size
#         if typ in ['train', 'dev']:
#             self.typ = 'dev'
#         else:
#             self.typ = typ
#         self.img_file_full_paths = []
#         self.get_valid_list()

#     def get_valid_list(self):
#         for speaker in self.speakers:
#             if os.path.isdir(os.path.join(self.img_root_dir, 'train', speaker[0])):
#                 img_full_path = os.path.join(os.path.join(self.img_root_dir, 'train', speaker[0]))
#             else:
#                 img_full_path = os.path.join(os.path.join(self.img_root_dir, 'test', speaker[0]))
#             img_files = os.listdir(img_full_path)
#             for img_file in img_files:
#                 img_file_full_path = os.path.join(img_full_path, img_file)
#                 self.img_file_full_paths.append(img_file_full_path)

#     def normalize_image(self, image):
#         return self.trans(image)

#     def __getitem__(self, index):
#         image = Image.open(self.img_file_full_paths[index])
#         image = self.normalize_image(image)
#         filename = self.img_file_full_paths[index].split('/')[-1]
#         speaker = self.img_file_full_paths[index].split('/')[-2]
#         return speaker, filename, image
    
#     def __len__(self):
#         return len(self.img_file_full_paths)

# class Evalset(torch.utils.data.Dataset):
#     def __init__(self, embeddings, labels, les, speakers):
#         self.speakers = speakers
#         lbs = self.make_embs_lbls(embeddings, labels)
#         self.lbls_to_clss(lbs, les)
#         assert len(self.embs) == len(self.clss['gender'])
#         assert len(self.embs) == len(self.clss['age'])
#         assert len(self.embs) == len(self.clss['ethnicity'])
#         assert len(self.embs) == len(self.clss['pitch'])
#         assert len(self.embs) == len(self.clss['ecapa'])

#     def make_embs_lbls(self, embeddings, labels):
#         self.embs = []
#         lbls = {}
#         gender_lb = []
#         age_lb = []
#         ethnicity_lb = []
#         pitch_lb = []
#         ecapa_lb = []
#         for speaker in self.speakers:
#             self.embs += embeddings[speaker]
#             gender_lb += [labels['gender'][speaker]]*len(embeddings[speaker])
#             age_lb += [np.round(float(labels['age'][speaker])/80,4)]*len(embeddings[speaker])
#             ethnicity_lb += [labels['ethnicity'][speaker]]*len(embeddings[speaker])
#             pitch_lb += [np.round(float(labels['pitch'][speaker])/250,4)]*len(embeddings[speaker])
#             ecapa_lb += [torch.load(os.path.join('/home/jaejun/f2v/gmc/gmc_jj/filelists/voxceleb_test_ecapa',speaker + '.ecapa'))]*len(embeddings[speaker])
#         self.embs = torch.stack(self.embs, axis=0)
#         lbs = {'gender':gender_lb, 'age':age_lb, 'ethnicity':ethnicity_lb, 'pitch':pitch_lb, 'ecapa':ecapa_lb}
#         return lbs

#     def lbls_to_clss(self, lbs, les):
#         gender_cls = les['gender'].transform(lbs['gender'])
#         age_cls = lbs['age']
#         ethnicity_cls = les['ethnicity'].transform(lbs['ethnicity'])
#         pitch_cls = lbs['pitch']
#         ecapa_cls = lbs['ecapa']
#         self.clss = {'gender':gender_cls, 'age':age_cls, 'ethnicity':ethnicity_cls, 'pitch':pitch_cls, 'ecapa':ecapa_cls}
        
#     def __getitem__(self, index):
#         gender_cls = torch.tensor(self.clss['gender'][index]).unsqueeze(0).float()
#         age_cls = torch.tensor(self.clss['age'][index]).unsqueeze(0).float()
#         ethnicity_cls = F.one_hot(torch.tensor(self.clss['ethnicity'][index]), num_classes=4).float()
#         pitch_cls = torch.tensor(self.clss['pitch'][index]).unsqueeze(0).float()
#         ecapa_cls = self.clss['ecapa'][index].float()
#         clss = {'gender':gender_cls, 'age':age_cls, 'ethnicity':ethnicity_cls, 'pitch':pitch_cls, 'ecapa':ecapa_cls}
#         return self.embs[index], clss
    
#     def __len__(self):
#         return len(self.embs)


if __name__ == '__main__':
    import json
    import numpy as np
    from torch.utils.data import DataLoader

    json_string = '''{
        "train": {
            "batch_size": 16
            },
        "data": {
            "training_files":"filelists/voxceleb_train_speakerlist.txt",
            "validation_files":"filelists/voxceleb_test_speakerlist.txt",
            "audio_root_dir":"/data2/VoxCeleb2",
            "image_root_dir":"/data1/VGG_Face2/data",
            "sampling_rate": 16000,
            "target_sec": 6,
            "image_size": 128
            }
    }'''

    config = json.loads(json_string)
    args = HParams(**config)

    # filelist = args.data.training_files
    # trainset = AudImgLoader(typ='train', filelist=filelist, args=args)
    # train_loader = DataLoader(trainset, batch_size=args.train.batch_size,
    #                     shuffle=False, num_workers=8,
    #                     collate_fn=None, pin_memory=True, drop_last=True,
    #                     worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    # for i, (wav, image) in enumerate(train_loader, 1):
    #     print('i:{}, audio shape{}, image shape{}'.format(i, wav.shape, image.shape))
    #     if i > 10:
    #         break


    valid_filelist = args.data.validation_files
    validset = AudSpkLoader(typ='test', filelist=valid_filelist, args=args)
    valid_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=8,
                    collate_fn=None, pin_memory=True, drop_last=False,
                    worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    for i, (speaker, filename, audio) in enumerate(valid_loader, 1):
        print('i:{}, audio shape{}'.format(speaker, audio.shape))
        if i > 2000:
            break

    validset2 = ImgSpkLoader(typ='test', filelist=valid_filelist, args=args)
    valid_loader2 = DataLoader(validset2, batch_size=1, shuffle=False, num_workers=8,
                    collate_fn=None, pin_memory=True, drop_last=False,
                    worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    for i, (speaker, filename, img) in enumerate(valid_loader2, 1):
        print('i:{}, img shape{}'.format(speaker, img.shape))
        if i > 2000:
            break