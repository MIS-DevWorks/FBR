# Flexible Biometrics Recognition: Bridging the Multimodality Gap through Attention, Alignment and Prompt Tuning


## Introduction
This repository contains the source code for the contributions in this article _Flexible Biometrics Recognition: Bridging the Multimodality Gap through Attention, Alignment and Prompt Tuning_, which is accepted by CVPR 2024.

Version 1.0 (19.10.2023)
<br> <br>

## Dataset
We utilize the VGGFace2 and MAAD datasets to train this model, which are available as follows:
- [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- [MAAD](https://github.com/pterhoer/MAAD-Face)
<br> <br>

## Coding Usage
- For training, please run `main.py` with the given configurations in config.py
```shell
$ python main.py --training_mode --dataset_name "VGGFace2"
```

- For evaluation, please run `main.py` with the given configurations in config.py
```shell
$ python main.py --dataset_name "other"
```

_For better understanding, please refer to this_ [code usage](code%20usage) _directory for further details. In this directory, we provide two examples:_ `code usage(Dataset).py` _and_ `code usage(MFA-ViT).py` 
<br> <br>

## Compatibility
We tested the codes with:
  1) PyTorch 1.13.1 with and without GPU, under Ubuntu 18.04/20.04 and Anaconda3 (Python 3.8 and above)
  2) PyTorch 1.12.0 with and without GPU, under Windows 10 and Anaconda3 (Python 3.8 and above)

_Please ensure that the PyTorch version used is up to 1.13.1, as this is required by the Python Library Package_ [rtdl](https://github.com/Yura52/rtdl). 
<br> <br>

## Pretrained Download
Our pretrained model can be found [here](https://drive.google.com/drive/folders/14ZKsEBJ9jweiU8obViKHzHEyiLts8UWK?usp=sharing).
<br> <br>

## License
This work is an open-source under MIT license.
<br> <br>

## Citing
```
@InProceedings{FBR_2024_CVPR,
    author    = {Tiong, Leslie Ching Ow and Sigmund, Dick and Chan, Chen-Hui and Teoh, Andrew Beng Jin},
    title     = {Flexible Biometrics Recognition: Bridging the Multimodality Gap through Attention, Alignment and Prompt Tuning},
    booktitle = {Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024},
}
```
