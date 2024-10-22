# Hear Your Face: Face-based voice conversion with F0 estimation
This repository contains the official implementation of our paper
([link](https://www.isca-archive.org/interspeech_2024/lee24d_interspeech.html),
[arxiv](https://www.arxiv.org/abs/2408.09802)), _Hear Your Face: Face-based voice conversion with F0 estimation_, published at Interspeech 2024.

> **Note**: The code was released now on 24.10.22 (v1.0)! If you have any questions or encounter any issues while running the code, feel free to contact us.

And also, don't miss our [demo](https://jaejunl.github.io/HYFace_Demo/).

# Intro
This implementation is built upon a __so-vits-svc__ ([link](https://github.com/svc-develop-team/so-vits-svc)), a dedicated project for singing voice conversion. We highly recommend exploring their page for further explanations of the module we used, except for those related to face image processing.\
For face image processing, we use a Vision Transformer (ViT) similar to the implementation of Face-Transformer ([link](https://github.com/zhongyy/Face-Transformer/tree/main/copy-to-vit_pytorch-path)).

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
<!--
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
```-->
```
modified
├───pretrain
│   ├───Speaker 1
│   │   ├───00001, 00001.wav, 00001.emb, 00001.f0.npy
│   │   │   ├───xxxx.jpg
│   │   │   ├───...
│   │   ├───...
│   │   └───xxxxx, xxxxx.wav, xxxxx.emb, xxxxx.f0.npy
│   ├───...
│   └───Speaker n
└───trainval
└───test
└───avg_mu_pretrain.pickle, avg_mu_trainval.pickle, avg_mu_test.pickle

```
We recommend using multi-processing, as all the provided codes below are single-process based and can be quite slow, especially for 'video_processing.py'.

### Video split
Running `preprocessing/video_processing.py` will split the original videos into 25fps images and 16kHz audio files.\
We split any wav files longer than 10 seconds into multiple shorter sub-files.
```
python preprocessing/video_processing.py --lrs3_root 'your-LRS3-original-root' --types pretrain trainval test
```

<!--
### Wav split
We split any wav files longer than 10 seconds into multiple shorter sub-files.
```
python preprocessing/wav_split.py --lrs3_root 'your-LRS3-original-root' --types pretrain trainval test
```
-->

### ContentVec save
Extract the ContentVec embeddings ('****.emb') from the split wav files. We used the Huggin Face version by _lengyue233_ ([Link](https://huggingface.co/lengyue233/content-vec-best)).\
The code below will automatically download and save the Hugging Face model.
```
CUDA_VISIBLE_DEVICES=0 python preprocessing/contentvec_save.py --lrs3_root 'your-LRS3-original-root' --types pretrain trainval test
```

### F0 extracting
For extracting F0 information ('****.f0.npy'), we use [FCPE(Fast Context-base Pitch Estimator)](https://github.com/CNChTu/FCPE), download the pre-trained model ([fcpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt)) and place in under the `pretrain` directory. Then run the code below.\
Not only does it save the F0 information in the shape `(2, n)` (representing frame-wise F0 values and VAD (voice activicy detection) values), but it also automatically saves speaker-wise average F0 values in a pickle file, such as `modified/avg_mu_pretrain.pickle`.
```
python preprocessing/f0_extract.py --lrs3_root 'your-LRS3-original-root' --types pretrain trainval test
```

### (Optional) Frontal image sifting
> This step can take a significant amount of time. Although we used only frontal images for training in our paper, we found that skipping this preprocessing step does not lead to significant performance degradation. Therefore, we recommend skipping this step, or alternatively, using your own method for selecting frontal images.

Download the frontal face detector and place it in the same root as `original`.\
We used OpenCV Haarcascades model `haarcascade_frontalface_default.xml` ([link](https://github.com/kipr/opencv/tree/master/data/haarcascades)), but you can use your own.\
Then run `preprocessing/img_frontal.py`, it will copy only centured face images into the `modified/imgs_frontal` directory.
```
python preprocessing/img_frontal.py --lrs3_root 'your-LRS3-original-root' --types pretrain trainval test
```

# Train
For training, both the 'main' model (our primary voice conversion model) and the 'sub' model (Average F0 estimation network) need to be trained. For more details, please refer to our paper.\
Make sure to update the data paths ('data.aud_dir' and 'data.img_dir') in the configuration files ('configs/main.json' and 'configs/sub.json').\
If you want to verify code execution or GPU usage before starting a serious training process, use the `--test=1` argument in the parser. For WANDB users, please refer to the commented sections in 'main.py', 'train.py', and 'solver.py'.

### Training the main model
```
python main.py --write_root='your-model-save-root' --model=main --gpus=1,2,3,4
```
### Training the sub model
```
python main.py --write_root='your-model-save-root' --model=sub --gpus=3,4
```

## Evaluation
To evaluate your own model, we provide 'evaluation/sub_eval.py' for assessing the deviation of average F0 for the sub model, and 'evaluation/main_eval.py' for evaluating both objective consistency and average F0 for the main model. We found that using 300 epochs for the main model and 200 epochs for the sub models, the main evaluation results will be approximately as follows.
> Source-Male, Target-Male: Consistency-0.5785, F0_dev-23.67\
Source-Female, Target-Male: Consistency-0.5731, F0_dev-23.06\
Source-Male, Target-Female: Consistency-0.5601, F0_dev-32.00\
Source-Female, Target-Female: Consistency-0.5645, F0_dev-31.68

Make sure to update the dataset paths and specify the model weights paths. To save your synthesized speech, use the argument `--save_samples=1`.
```
CUDA_VISIBLE_DEVICES=0 python evaluation/main_eval.py --save_samples=1
CUDA_VISIBLE_DEVICES=0 python evaluation/sub_eval.py
```


# Inference
For inference, first save the main model weights ('main.pth') and sub model weights ('sub.pth') in the 'pretrain' folder.\
Next, save your source audio file ('source.wav') and target face image file ('target.jpg') in the 'inference' folder. Altenatively, you can use your own file paths by referring to the parser inside 'inference.py' file. Then, run the code below.
```
CUDA_VISIBLE_DEVICES=0 python inference/inference.py
```
### Model weight
For the pretrained model weights, please contact us bia email (jjlee0721@snu.ac.kr), including your affiliation and the purpose for using the model weights. Alternatively, you can train your own model using the training code provided above (the same code used to generate the pretrained model weights).\
Our pretrained model was trained for 300 epochs for the main model and 200 epochs for the sub model.


