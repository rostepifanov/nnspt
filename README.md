# Neural Network Signal Processing on Torch

![Python version support](https://img.shields.io/pypi/pyversions/nnspt)
[![PyPI version](https://badge.fury.io/py/nnspt.svg)](https://badge.fury.io/py/nnspt)
[![Downloads](https://pepy.tech/badge/nnspt/month)](https://pepy.tech/project/nnspt?versions=0.0.*)

NNSPT is a Python library for neural network signal processing on PyTorch.

## Table of contents
- [Authors](#authors)
- [Installation](#installation)
- [A simple example](#a-simple-example)
- [Available components](#available-components)
- [Citing](#citing)

## Authors
[**Rostislav Epifanov** — Researcher in Novosibirsk]()

## Installation
Installation from PyPI:

```
pip install nnspt
```

Installation from GitHub:

```
pip install git+https://github.com/rostepifanov/nnspt
```

## A simple example
```python
from nnspt.segmentation.unet import Unet

model = Unet(encoder='tv-resnet34')
```

## Available components
#### Encoders

  * <details> <summary>ResNet</summary>

    | Name        | Weights | Params |
    | ---:        | :---:   | :---:  |
    |tv-resnet18  | -       | 3.8M   |
    |tv-resnet34  | -       | 7.2M   |
    |tv-resnet50  | -       | 15.9M  |
    |tv-resnet101 | -       | 28.2M  |
    |tv-resnet152 | -       | 38.4M  |
  </details>

  * <details> <summary>ResNeXt</summary>

    | Name                 | Weights | Params |
    | ---:                 | :---:   | :---:  |
    | tv-resnext50_32x4d   | -       | 22M    |
    | tv-resnext101_32x4d  | -       | 40.3M  |
    | tv-resnext101_32x8d  | -       | 79.6M  |
    | tv-resnext101_32x16d | -       | 163.5M |
    | tv-resnext101_32x32d | -       | 352.6M |
    | tv-resnext101_32x48d | -       | 570M   |
  </details>

  * <details> <summary>DenseNet</summary>

    | Name           | Weights | Params |
    | ---:           | :---:   | :---:  |
    | tv-densenet121 | -       | 5.5M   |
    | tv-densenet169 | -       | 10.4M  |
    | tv-densenet201 | -       | 15.6M  |
    | tv-densenet161 | -       | 22.1M  |
  </details>

  * <details> <summary>EfficientNetV1</summary>

    | Name                 | Weights | Params |
    | ---:                 | :---:   | :---:  |
    | timm-efficientnet-b0 | -       | 3.4M   |
    | timm-efficientnet-b1 | -       | 5.9M   |
    | timm-efficientnet-b2 | -       | 6.9M   |
    | timm-efficientnet-b3 | -       | 9.8M   |
    | timm-efficientnet-b4 | -       | 16.3M  |
    | timm-efficientnet-b5 | -       | 26.7M  |
    | timm-efficientnet-b6 | -       | 38.6M  |
    | timm-efficientnet-b7 | -       | 61.1M  |
  </details>

  * <details> <summary>EfficientNetLite</summary>

    | Name                    | Weights | Params |
    | ---:                    | :---:   | :---:  |
    | timm-efficientnet-lite0 | -       | 2.8M   |
    | timm-efficientnet-lite1 | -       | 3.5M   |
    | timm-efficientnet-lite2 | -       | 4.1M   |
    | timm-efficientnet-lite3 | -       | 6.1M   |
    | timm-efficientnet-lite4 | -       | 10.7M  |
  </details>

#### Pretraining

  * Autoencoder

#### Segmentation

  * Unet [[paper](https://arxiv.org/abs/1505.04597)]

## Citing

If you find this library useful for your research, please consider citing:

```
@misc{epifanov2023ecgmentations,
  Author = {Rostislav Epifanov},
  Title = {NNSTP},
  Year = {2023},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/rostepifanov/nnspt}}
}
```