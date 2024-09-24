# Hear Your Face: Face-based voice conversion with F0 estimation
This repository contains the official implementation of our paper
([link](https://www.isca-archive.org/interspeech_2024/lee24d_interspeech.html),
[arxiv](https://www.arxiv.org/abs/2408.09802))
, _Hear Your Face: Face-based voice conversion with F0 estimation_, published at Interspeech 2024.

> **Note**: The implementation is currently a work in progress and is expected to be completed by the end of October.

And also, don't miss our [demo](https://jaejunl.github.io/HYFace_Demo/).

# Intro
This implementation is built upon a __so-vits-svc__ ([link](https://github.com/svc-develop-team/so-vits-svc)), a dedicated project for singing voice conversion. We highly recommend exploring their page for further explanations of the module we used, except for those related to face image processing.

# Dataset
We used LRS3 dataset ([arxiv](https://arxiv.org/abs/1809.00496), [website](https://mmai.io/datasets/lip_reading/)), consists of 5,502 videos from TED and TEDx.

### Configuration
Simply place the dataset in the `original` directory with the following file structure:
```
original
├───pretrain
│   ├───Speaker 1
│   │   ├───00001.mp4, 00001.txt
│   │   ├───...
│   │   └───xxxxx.mp4, xxxxx.txt
│   ├───...
│   └───Speaker n
└───trainval
└───test
```

# Preprocessing
After running our all preprocessing modules, the preprocessed dataset in the `modified` directory will have the following structure:
```
modified
├───imgs
│   ├───pretrain
│   │   ├───Speaker 1
│   │   │   ├───00001
│   │   │   │   ├───xxxx.jpg
│   │   │   │   ├───...
│   │   │   ├───...
│   │   │   └───xxxxx
│   │   ├───...
│   │   └───Speaker n
│   └───trainval
│   └───test
└───auds
    ├───pretrain
    │   ├───Speaker 1
    │   │   ├───00001.wav, 00001.emb, 00001.f0.npy
    │   │   ├───...
    │   │   └───xxxxx.wav, xxxxx.emb, xxxxx.f0.npy
    │   ├───...
    │   └───Speaker n
    └───trainval
    └───test
    └───avg_mu_pretrain.pickle, avg_mu_trainval.pickle, avg_mu_test.pickle
```
We recommend using multi-processing, as all the provided codes below are single-process based and can be quite slow.

### Video split
Running `preprocessing/video_processing.py` will split the original videos into 25fps images and 16kHz audio files.
```
python preprocessing/video_processing.py --lrs3_root 'your-LRS3-original-root' --types pretrain trainval test
```

### Frontal image sifting
Download the frontal face detector and place it in the same root as `original`.\
We used OpenCV Haarcascades model `haarcascade_frontalface_default.xml` ([link](https://github.com/kipr/opencv/tree/master/data/haarcascades)), but you can use your own.\
Then run `preprocessing/img_frontal.py`, it will copy only centured face images into the `modified/imgs` directory.
```
python preprocessing/img_frontal.py --lrs3_root 'your-LRS3-original-root' --types pretrain trainval test
```

### Wav split
We split any wav files longer than 10 seconds into multiple shorter sub-files.
```
python preprocessing/wav_split.py --lrs3_root 'your-LRS3-original-root' --types pretrain trainval test
```

### ContentVec save
Extract the ContentVec embeddings from the split wav files. We used the Huggin Face version by _lengyue233_ ([Link](https://huggingface.co/lengyue233/content-vec-best)).\
The code below will automatically download and save the Hugging Face model.
```
CUDA_VISIBLE_DEVICES=0 python preprocessing/contentvec_save.py --lrs3_root 'your-LRS3-original-root' --types pretrain trainval test
```

### F0 extracting
For extracting F0 information, we use [FCPE(Fast Context-base Pitch Estimator)](https://github.com/CNChTu/FCPE), download the pre-trained model ([fcpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt)) and place in under the `pretrain` directory. Then run the code below.\
Not only does it save the F0 information in the shape `(2, n)` (representing frame-wise F0 values and VAD (voice activicy detection) values), but it also automatically saves speaker-wise average F0 values in a pickle file, such as `modified/auds/avg_mu_pretrain.pickle`.
```
python preprocessing/f0_extract.py --lrs3_root 'your-LRS3-original-root' --types pretrain trainval test
```
