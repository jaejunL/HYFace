import re
import os
import sys
import numpy as np
import random
import scipy
import json
import sys
import pickle
from copy import copy
import string

import torch
import torchaudio
import torch.nn.functional as F

sys.path.append('../')
from utils.data_utils import load_audio, get_emg_features, FeatureNormalizer, phoneme_inventory, read_phonemes, TextTransform

hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,8):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)

def optimize_size(size1, size2, factor):
    max_size = max(size1, size2)
    while max_size % factor != 0:
        max_size += 1
    return max_size

def batch_zeropad(batch, size):
    return F.pad(batch, (0, 0, 0, size-batch.shape[0]), 'constant', 0)

def phoneme_silpad(phoneme, size):
    return F.pad(phoneme, (0, size-phoneme.shape[0]), 'constant', phoneme_inventory.index('sil'))

## modified 
def load_utterance(args, base_dir, index, limit_length=False, debug=False, text_align_directory=None):
    index = int(index)
    raw_emg = np.load(os.path.join(base_dir, f'{index}_emg.npy'))
    before = os.path.join(base_dir, f'{index-1}_emg.npy')
    after = os.path.join(base_dir, f'{index+1}_emg.npy')
    if os.path.exists(before):
        raw_emg_before = np.load(before)
    else:
        raw_emg_before = np.zeros([0,raw_emg.shape[1]])
    if os.path.exists(after):
        raw_emg_after = np.load(after)
    else:
        raw_emg_after = np.zeros([0,raw_emg.shape[1]])

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0]:x.shape[0]-raw_emg_after.shape[0],:]
    if args.data.sample_rate == 22050:
        emg_orig = apply_to_all(subsample, x, 689.06, 1000)
        x = apply_to_all(subsample, x, 516.79, 1000)
    elif args.data.sample_rate == 16000:
        emg_orig = apply_to_all(subsample, x, 500.0, 1000)
        x = apply_to_all(subsample, x, 375.0, 1000)
    emg = x

    for c in args.data.remove_channels:
        emg[:,int(c)] = 0
        emg_orig[:,int(c)] = 0

    emg_features = get_emg_features(emg)

    mfccs = load_audio(args, os.path.join(base_dir, f'{index}_audio_clean.flac'),
            max_frames=min(emg_features.shape[0], 800 if limit_length else float('inf')), sample_rate=args.data.sample_rate)

    if emg_features.shape[0] > mfccs.shape[0]:
        emg_features = emg_features[:mfccs.shape[0],:]
    assert emg_features.shape[0] == mfccs.shape[0]
    emg = emg[6:6+6*emg_features.shape[0],:]
    emg_orig = emg_orig[8:8+8*emg_features.shape[0],:]
    assert emg.shape[0] == emg_features.shape[0]*6

    with open(os.path.join(base_dir, f'{index}_info.json')) as f:
        info = json.load(f)

    sess = os.path.basename(base_dir)
    tg_fname = f'{text_align_directory}/{sess}/{sess}_{index}_audio.TextGrid'
    if os.path.exists(tg_fname):
        phonemes = read_phonemes(tg_fname, mfccs.shape[0])
    else:
        phonemes = np.zeros(mfccs.shape[0], dtype=np.int64)+phoneme_inventory.index('sil')

    return mfccs, emg_features, info['text'], (info['book'],info['sentence_index']), phonemes, emg_orig.astype(np.float32), None


## Modified version by Jaejun
def load_utterance2(args, base_dir, index, limit_length=False, debug=False, text_align_directory=None):
    index = int(index)

    if args.base_args.hubert:
        emg_sr = 800
    elif args.data.sample_rate == 22050:
        emg_sr = 689.06
    elif args.data.sample_rate == 16000:
        emg_sr = 500
    
    raw_emg = np.load(os.path.join(base_dir, f'{index}_emg.npy'))
    before = os.path.join(base_dir, f'{index-1}_emg.npy')
    after = os.path.join(base_dir, f'{index+1}_emg.npy')
    if os.path.exists(before):
        raw_emg_before = np.load(before)
    else:
        raw_emg_before = np.zeros([0,raw_emg.shape[1]])
    if os.path.exists(after):
        raw_emg_after = np.load(after)
    else:
        raw_emg_after = np.zeros([0,raw_emg.shape[1]])

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0]:x.shape[0]-raw_emg_after.shape[0],:]
    emg_orig = apply_to_all(subsample, x, emg_sr, 1000)

    for c in args.data.remove_channels:
        emg_orig[:,int(c)] = 0

    # emg_features = get_emg_features(emg)

    mfccs = load_audio(args, os.path.join(base_dir, f'{index}_audio_clean.flac'), sample_rate=args.data.sample_rate)

    hubert_path = os.path.join(base_dir).replace('emg_data', 'modified/hubert_soft')

    if os.path.exists(os.path.join(hubert_path, f'{index}_hubert_soft.emb')):
        hubert_emb = torch.load(os.path.join(hubert_path, f'{index}_hubert_soft.emb'))[0]
    else:
        os.makedirs(hubert_path, exist_ok=True)
        wav, sr = torchaudio.load(os.path.join(base_dir, f'{index}_audio_clean.flac'))
        hubert_emb = hubert.units(wav.unsqueeze(0))
        torch.save(hubert_emb, os.path.join(hubert_path, f'{index}_hubert_soft.emb'))
        hubert_emb = hubert_emb[0]

    if args.base_args.hubert:
        max_len = hubert_emb.shape[0]
        scale = 50 # hubert soft_vc sample rate (50hz)
    else:
        max_len = mfccs.shape[0]
        scale = args.data.sample_rate / 256

    with open(os.path.join(base_dir, f'{index}_info.json')) as f:
        info = json.load(f)

    sess = os.path.basename(base_dir)
    tg_fname = f'{text_align_directory}/{sess}/{sess}_{index}_audio.TextGrid'
    if os.path.exists(tg_fname):
        phonemes = read_phonemes(tg_fname, max_len, scale)
    else:
        phonemes = np.zeros(mfccs.shape[0], dtype=np.int64)+phoneme_inventory.index('sil')

    return mfccs, None, info['text'], (info['book'], info['sentence_index']), phonemes, emg_orig.astype(np.float32), hubert_emb


class EMGDirectory(object):
    def __init__(self, session_index, directory, silent, exclude_from_testset=False):
        self.session_index = session_index
        self.directory = directory
        self.silent = silent
        self.exclude_from_testset = exclude_from_testset

    def __lt__(self, other):
        return self.session_index < other.session_index

    def __repr__(self):
        return self.directory


class EMGDataset(torch.utils.data.Dataset):
    def __init__(self, args, base_dir=None, limit_length=False, dev=False, test=False, no_testset=False, no_normalizers=False):

        self.text_align_directory = "/disk2/silent_speech/text_alignments"
        self.args = args

        if no_testset:
            devset = []
            testset = []
        else:
            with open(self.args.data.testset_file) as f:
                testset_json = json.load(f)
                devset = testset_json['dev']
                testset = testset_json['test']

        directories = []
        if base_dir is not None:
            directories.append(EMGDirectory(0, base_dir, False))
        else:
            for sd in self.args.data.silent_data_directories:
                for session_dir in sorted(os.listdir(sd)):
                    directories.append(EMGDirectory(len(directories), os.path.join(sd, session_dir), True))

            has_silent = len(self.args.data.silent_data_directories) > 0
            for vd in self.args.data.voiced_data_directories:
                for session_dir in sorted(os.listdir(vd)):
                    directories.append(EMGDirectory(len(directories), os.path.join(vd, session_dir), False, exclude_from_testset=has_silent))

        self.example_indices = []
        self.voiced_data_locations = {} # map from book/sentence_index to directory_info/index
        for directory_info in directories:
            for fname in os.listdir(directory_info.directory):
                m = re.match(r'(\d+)_info.json', fname) # m : file index of info file
                if m is not None:
                    idx_str = m.group(1)
                    with open(os.path.join(directory_info.directory, fname)) as f:
                        info = json.load(f)
                        if info['sentence_index'] >= 0: # boundary clips of silence are marked -1
                            location_in_testset = [info['book'], info['sentence_index']] in testset
                            location_in_devset = [info['book'], info['sentence_index']] in devset
                            if (test and location_in_testset and not directory_info.exclude_from_testset) \
                                    or (dev and location_in_devset and not directory_info.exclude_from_testset) \
                                    or (not test and not dev and not location_in_testset and not location_in_devset):
                                self.example_indices.append((directory_info,int(idx_str)))

                            if not directory_info.silent:
                                location = (info['book'], info['sentence_index'])
                                self.voiced_data_locations[location] = (directory_info,int(idx_str))

        self.example_indices.sort()
        random.seed(0)
        random.shuffle(self.example_indices)

        self.no_normalizers = no_normalizers
        if not self.no_normalizers:
            self.mfcc_norm, self.emg_norm = pickle.load(open(self.args.data.normalizers_file,'rb'))

        sample_mfccs, _, _, _, _, _, _ = load_utterance2(self.args, self.example_indices[0][0].directory, self.example_indices[0][1])
        self.num_speech_features = sample_mfccs.shape[1]
        # self.num_features = sample_emg.shape[1]
        self.limit_length = limit_length
        self.num_sessions = len(directories)

        self.text_transform = TextTransform()

    def silent_subset(self):
        result = copy(self)
        silent_indices = []
        for example in self.example_indices:
            if example[0].silent:
                silent_indices.append(example)
        result.example_indices = silent_indices
        return result

    def subset(self, fraction):
        result = copy(self)
        result.example_indices = self.example_indices[:int(fraction*len(self.example_indices))]
        return result

    def resize(self, result):
        if self.args.base_args.hubert:
            if result['silent']:
                optim_size = optimize_size(result['parallel_voiced_hubert'].shape[0]*16, result['raw_emg'].shape[0], 16)
                result['parallel_voiced_hubert'] = batch_zeropad(result['parallel_voiced_hubert'], optim_size//16)
                result['raw_emg'] = batch_zeropad(result['raw_emg'], optim_size)
            else:
                optim_size = optimize_size(result['hubert'].shape[0]*16, result['raw_emg'].shape[0], 16)
                result['hubert'] = batch_zeropad(result['hubert'], optim_size//16)
                result['raw_emg'] = batch_zeropad(result['raw_emg'], optim_size)
            result['phonemes'] = phoneme_silpad(result['phonemes'], optim_size//16)
        else:
            if result['silent']:
                optim_size = optimize_size(result['parallel_voiced_audio_features'].shape[0]*8, result['raw_emg'].shape[0], 8)
                result['parallel_voiced_audio_features'] = batch_zeropad(result['parallel_voiced_audio_features'], optim_size//8)
                result['raw_emg'] = batch_zeropad(result['raw_emg'], optim_size)
            else:
                optim_size = optimize_size(result['audio_features'].shape[0]*8, result['raw_emg'].shape[0], 8)
                result['audio_features'] = batch_zeropad(result['audio_features'], optim_size//8)
                result['raw_emg'] = batch_zeropad(result['raw_emg'], optim_size)
            result['phonemes'] = phoneme_silpad(result['phonemes'], optim_size//8)
        return result            

    def __len__(self):
        return len(self.example_indices)

    # # @lru_cache(maxsize=None)
    # def __getitem__(self, i):
    #     directory_info, idx = self.example_indices[i]
    #     mfccs, emg, text, book_location, phonemes, raw_emg = load_utterance(self.args, directory_info.directory, idx, self.limit_length, text_align_directory=self.text_align_directory)
    #     raw_emg = raw_emg / 20
    #     raw_emg = 50*np.tanh(raw_emg/50.)

    #     if not self.no_normalizers:
    #         mfccs = self.mfcc_norm.normalize(mfccs)
    #         emg = self.emg_norm.normalize(emg)
    #         emg = 8*np.tanh(emg/8.)

    #     session_ids = np.full(emg.shape[0], directory_info.session_index, dtype=np.int64)
    #     audio_file = f'{directory_info.directory}/{idx}_audio_clean.flac'

    #     text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)

    #     result = {'audio_features':torch.from_numpy(mfccs).pin_memory(), 'emg':torch.from_numpy(emg).pin_memory(), 'text':text, 'text_int': torch.from_numpy(text_int).pin_memory(), 'file_label':idx, 'session_ids':torch.from_numpy(session_ids).pin_memory(), 'book_location':book_location, 'silent':directory_info.silent, 'raw_emg':torch.from_numpy(raw_emg).pin_memory()}

    #     if directory_info.silent:
    #         voiced_directory, voiced_idx = self.voiced_data_locations[book_location]
    #         voiced_mfccs, voiced_emg, _, _, phonemes, _ = load_utterance(self.args, voiced_directory.directory, voiced_idx, False, text_align_directory=self.text_align_directory)

    #         if not self.no_normalizers:
    #             voiced_mfccs = self.mfcc_norm.normalize(voiced_mfccs)
    #             voiced_emg = self.emg_norm.normalize(voiced_emg)
    #             voiced_emg = 8*np.tanh(voiced_emg/8.)

    #         result['parallel_voiced_audio_features'] = torch.from_numpy(voiced_mfccs).pin_memory()
    #         result['parallel_voiced_emg'] = torch.from_numpy(voiced_emg).pin_memory()

    #         audio_file = f'{voiced_directory.directory}/{voiced_idx}_audio_clean.flac'

    #     result['phonemes'] = torch.from_numpy(phonemes).pin_memory() # either from this example if vocalized or aligned example if silent
    #     result['audio_file'] = audio_file
    #     return result

    def __getitem__(self, i):
        directory_info, idx = self.example_indices[i]
        mfccs, emg, text, book_location, phonemes, raw_emg, hubert_emb = load_utterance2(self.args, directory_info.directory, idx, self.limit_length, text_align_directory=self.text_align_directory)
        # mfccs, emg, text, book_location, phonemes, raw_emg, hubert_emb = load_utterance(self.args, directory_info.directory, idx, self.limit_length, text_align_directory=self.text_align_directory)

        raw_emg = raw_emg / 20
        raw_emg = 50*np.tanh(raw_emg/50.)

        if not self.no_normalizers:
            mfccs = self.mfcc_norm.normalize(mfccs)
            # emg = self.emg_norm.normalize(emg)
            # emg = 8*np.tanh(emg/8.)

        # session_ids = np.full(emg.shape[0], directory_info.session_index, dtype=np.int64)
        session_ids = np.full(mfccs.shape[0], directory_info.session_index, dtype=np.int64)
        audio_file = f'{directory_info.directory}/{idx}_audio_clean.flac'

        text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)

        # result = {'audio_features':torch.from_numpy(mfccs), 'emg':torch.from_numpy(emg), 'text':text, 'text_int': torch.from_numpy(text_int), 'file_label':idx, 'session_ids':torch.from_numpy(session_ids), 'book_location':book_location, 'silent':directory_info.silent, 'raw_emg':torch.from_numpy(raw_emg), 'hubert':hubert_emb}
        result = {'audio_features':torch.from_numpy(mfccs), 'emg':None, 'text':text, 'text_int': torch.from_numpy(text_int), 'file_label':idx, 'session_ids':torch.from_numpy(session_ids), 'book_location':book_location, 'silent':directory_info.silent, 'raw_emg':torch.from_numpy(raw_emg), 'hubert':hubert_emb}

        if directory_info.silent:
            voiced_directory, voiced_idx = self.voiced_data_locations[book_location]
            voiced_mfccs, _, _, _, phonemes, voiced_emg, voiced_hubert_emb = load_utterance2(self.args, voiced_directory.directory, voiced_idx, False, text_align_directory=self.text_align_directory)
            # voiced_mfccs, _, _, _, phonemes, voiced_emg, voiced_hubert_emb = load_utterance(self.args, voiced_directory.directory, voiced_idx, False, text_align_directory=self.text_align_directory)

            if not self.no_normalizers:
                voiced_mfccs = self.mfcc_norm.normalize(voiced_mfccs)
                # voiced_emg = self.emg_norm.normalize(voiced_emg)
                # voiced_emg = 8*np.tanh(voiced_emg/8.)

            result['parallel_voiced_audio_features'] = torch.from_numpy(voiced_mfccs)
            result['parallel_voiced_emg'] = torch.from_numpy(voiced_emg)
            result['parallel_voiced_hubert'] = voiced_hubert_emb
            audio_file = f'{voiced_directory.directory}/{voiced_idx}_audio_clean.flac'
        result['phonemes'] = torch.from_numpy(phonemes) # either from this example if vocalized or aligned example if silent
        result['audio_file'] = audio_file

        result = self.resize(result)

        return result

    # @staticmethod
    def collate_raw(self, batch):
        batch_size = len(batch)
        audio_features = []
        audio_feature_lengths = []
        parallel_emg = []
        hubert_embs = []
        for ex in batch:
            if ex['silent']:
                audio_features.append(ex['parallel_voiced_audio_features'])
                audio_feature_lengths.append(ex['parallel_voiced_audio_features'].shape[0])
                parallel_emg.append(ex['parallel_voiced_emg'])
                hubert_embs.append([ex['parallel_voiced_hubert']])
            else:
                audio_features.append(ex['audio_features'])
                audio_feature_lengths.append(ex['audio_features'].shape[0])
                parallel_emg.append(np.zeros(1))
                hubert_embs.append([ex['hubert']])

        phonemes = [ex['phonemes'] for ex in batch]
        emg = [ex['emg'] for ex in batch]
        raw_emg = [ex['raw_emg'] for ex in batch]
        session_ids = [ex['session_ids'] for ex in batch]
        # if self.args.base_args.hubert:
            # lengths = [ex.shape[0] for ex in hubert_embs]
        # else:
            # lengths = [ex.shape[0] for ex in audio_features]
        lengths = [ex['phonemes'].shape[0] for ex in batch]
        silent = [ex['silent'] for ex in batch]
        text_ints = [ex['text_int'] for ex in batch]
        text_lengths = [ex['text_int'].shape[0] for ex in batch]

        result = {'audio_features':audio_features,
                  'audio_feature_lengths':audio_feature_lengths,
                  'emg':emg,
                  'raw_emg':raw_emg,
                  'parallel_voiced_emg':parallel_emg,
                  'phonemes':phonemes,
                  'session_ids':session_ids,
                  'lengths':lengths,
                  'silent':silent,
                  'text_int':text_ints,
                  'text_int_lengths':text_lengths,
                  'hubert':hubert_embs}
        return result


def make_normalizers(args):
    dataset = EMGDataset(args, no_normalizers=True)
    mfcc_samples = []
    emg_samples = []
    for d in dataset:
        mfcc_samples.append(d['audio_features'])
        emg_samples.append(d['emg'])
        if len(emg_samples) > 50:
            break
    mfcc_norm = FeatureNormalizer(mfcc_samples, share_scale=True)
    emg_norm = FeatureNormalizer(emg_samples, share_scale=False)
    pickle.dump((mfcc_norm, emg_norm), open(args.data.normalizers_file, 'wb'))


if __name__ == '__main__':
    import argparse
    from utils import utils
    import datetime
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) 

    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser()
    # set parameters
    parser.add_argument('-c', '--config', type=str, default="./configs/config.json", help='JSON file for configuration')
    parser.add_argument('--init', default='1', type=utils.str2bool, help='1 for initial training, 0 for continuing')
    parser.add_argument('--group_name', default='test', type=str)
    parser.add_argument('--exp_name', default=nowDatetime, type=str)
    parser.add_argument('--arg_save', default='True', type=utils.str2bool, help='argument save or not')
    parser.add_argument('--test', default='false', type=utils.str2bool, help='whether test or not')
    parser.add_argument('--resume', default='false', type=utils.str2bool, help='whether resume or not')
    parser.add_argument('--log_all', default=1, type=int, help='whether wandb log all gpu or only 1')
    parser.add_argument('--pretrain', nargs='+', default=['.'], help='which pretrained model to use')
    parser.add_argument('--fixtrain', nargs='+', default=['.'], help='which model to fix')
    # gpu parameters
    parser.add_argument('--gpus', nargs='+', default=None, help='gpus')
    parser.add_argument('--port', default='6056', type=str, help='port')
    parser.add_argument('--n_nodes', default=1, type=int)
    parser.add_argument('--workers', default=4, type=int) # n개의 gpu가 한 node: n개의 gpu마다 main_worker를 실행시킨다.
    parser.add_argument('--rank', default=0, type=int, help='ranking within the nodes')
    base_args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_num) for gpu_num in base_args.gpus])
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = base_args.port
    if base_args.test == True:
        os.environ['WANDB_MODE'] = "dryrun"
    os.environ['WANDB_RUN_ID'] = base_args.exp_name
    if base_args.init == '0':
        os.environ["WANDB_RESUME"] = "must"

    base_args.base_dir = os.path.join('/disk3/jaejun/gaddy', base_args.group_name)
    # Make directories to save results, i.e., codes, checkpoints, analysis
    if base_args.arg_save:
        os.makedirs(os.path.join(base_args.base_dir, 'codes'), exist_ok=True)
        os.makedirs(os.path.join(base_args.base_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(base_args.base_dir, 'checkpoints'), exist_ok=True)
        include_dir = []
        exclude_dir = ['__pycache__', '.ipynb_checkpoints', 'filelists', 'configs', 'wandb', 'preprocess', 'text', 'jupyter', 'text_alignments']
        include_ext = ['.py']
        utils.copy_DirStructure_and_Files(os.getcwd(), include_dir, exclude_dir, include_ext, base_args.base_dir)

    config_path = base_args.config
    config_save_path = os.path.join(base_args.base_dir, "logs", "config.json")
    if base_args.init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)
    args = utils.HParams(**config)

    d = EMGDataset(args)
    for i in range(len(d)):
        d[i]
        print(f'{i/len(d)*100}%',end='\r')