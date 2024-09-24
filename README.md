# Hear Your Face: Face-based voice conversion with F0 estimation
This repository contains the official implementation of our paper
([link](https://www.isca-archive.org/interspeech_2024/lee24d_interspeech.html),
[arxiv](https://www.arxiv.org/abs/2408.09802))
, _Hear Your Face: Face-based voice conversion with F0 estimation_, published at Interspeech 2024.

> **Note**: The implementation is currently a work in progress and is expected to be completed by the end of October.

And also, don't miss our [demo](https://jaejunl.github.io/HYFace_Demo/).

## Dataset
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

## Preprocessing
We highly recommend using multi-processing, as all the provided codes below are single-process based and can be quite slow.
### Video split
Running `preprocessing/video_processing.py` will split the original videos into 25fps images and 16kHz audio files.
```
python preprocessing/video_processing.py --lrs3_root 'your-LRS3-original-root' --types pretrain trainval test
```
### Frontal image sifting
Download the frontal face detector and place it in the same root as `original`.

We used OpenCV Haarcascades model `haarcascade_frontalface_default.xml` ([link](https://github.com/kipr/opencv/tree/master/data/haarcascades)), but you can use your own.

Then run `preprocessing/img_frontal.py`, it will copy only centured face images into the `modified/imgs` directory.
```
python preprocessing/img_frontal.py --lrs3_root 'your-LRS3-original-root' --types pretrain trainval test
```

### Wav split


After running our preprocessing module, the preprocessed dataset in the `modified` directory will have the following structure:
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
├───audios
│   ├───pretrain
│   │   ├───Speaker 1
│   │   │   ├───00001.wav, 00001.emb
│   │   │   ├───...
│   │   │   └───xxxxx.wav, xxxxx.emb
│   │   ├───...
│   │   └───Speaker n
│   └───trainval
│   └───test
```


## Preprocessing


