<h1 align="center">
  Flexible Biometrics Recognition: Bridging the Multimodality Gap through Attention, Alignment and Prompt Tuning
</h1>
<p align="center">
  Leslie Ching Ow Tiong<sup>*,1</sup>,  Dick Sigmund<sup>*,2</sup>, Chen-Hui Chan<sup>3</sup>, Andrew Beng Jin Teoh<sup>&dagger;,4</sup>
  <br/>
  <sup>1</sup>Samsung Electronics,  <sup>2</sup>AIDOT Inc., <sup>3</sup>Korea Institute of Science and Technology, <sup>4</sup>Yonsei University
  <br/>
  <sup>*</sup>These authors contributed equally
  <br/>
  <sup>&dagger;</sup>Corresponding author
  <br/><br/>
  [YouTube](https://img.shields.io/badge/video-youtube-red.svg)]
</p>
<br/>


## Introduction
This repository contains the source code for the contributions in this article _Flexible Biometrics Recognition: Bridging the Multimodality Gap through Attention, Alignment and Prompt Tuning_, which is accepted by CVPR 2024.

Version 1.0 (13.05.2024)
<br/> <br/>

## Dataset
We utilize the VGGFace2 and MAAD datasets to train this model, which are available as follows:
- [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- [MAAD](https://github.com/pterhoer/MAAD-Face)
<br> <br>

## Requirements
  1) [Anaconda3](https://www.anaconda.com/download)
  2) [PyTorch](https://pytorch.org/get-started/locally)
  3) [RTDL](https://pypi.org/project/rtdl)
  4) [Natsort](https://pypi.org/project/natsort)
<br/> <br/>

## Coding Usage
- For training, please run `main.py` with the given configurations in config.py
```shell
$ python main.py --training_mode --dataset_name "VGGFace2"
```

- For evaluation, please run `main.py` with the given configurations in config.py
```shell
$ python main.py --dataset_name "other"
```

_For better understanding, please refer to this_ [code usage](code%20usage) _directory for further details. In this directory, we provide two examples:_ `code usage(Dataset).py` _and_ `code usage(MFA-ViT).py`. 
<br/> <br/>

## Compatibility
We tested the codes with:
  1) PyTorch 1.13.1 with and without GPU, under Ubuntu 18.04/20.04 and Anaconda3 (Python 3.8 and above)
  2) PyTorch 1.12.0 with and without GPU, under Windows 10 and Anaconda3 (Python 3.8 and above)
<br> <br>

## Pretrained Download
Our pretrained model can be accessed via [here](https://drive.google.com/drive/folders/1aX_UBGZN_as84h5Ioq42JC_zAjwa4uDw?usp=sharing). Upon downloading, it should be decompressed and configured within a directory named _pretrained_ to ensure proper setup.
<br/> <br/>

## License
This work is an open-source under MIT license.
<br/> <br/>

## Citing
```
@InProceedings{FBR_2024_CVPR,
    author    = {Tiong, Leslie Ching Ow and Sigmund, Dick and Chan, Chen-Hui and Teoh, Andrew Beng Jin},
    title     = {Flexible Biometrics Recognition: Bridging the Multimodality Gap through Attention, Alignment and Prompt Tuning},
    booktitle = {Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024},
}
```
