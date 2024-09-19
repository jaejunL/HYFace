# Hear Your Face: Face-based voice conversion with F0 estimation
This repository contains the implementation of our paper
([link](https://www.isca-archive.org/interspeech_2024/lee24d_interspeech.html),
[arxiv](https://www.arxiv.org/abs/2408.09802))
, _Hear Your Face: Face-based voice conversion with F0 estimation_, published at Interspeech 2024.

> **Note**: The implementation is currently a work in progress and is expected to be completed by the end of October.

And also, don't miss our [demo](https://jaejunl.github.io/HYFace_Demo/).

## Dataset
We used LRS3 dataset ([arxiv](https://arxiv.org/abs/1809.00496), [website](https://mmai.io/datasets/lip_reading/)), consists of 5,502 videos from TED and TEDx.

### Dataset Configuration
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


