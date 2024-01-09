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

      - tv-resnet18
      - tv-resnet34
      - tv-resnet50
      - tv-resnet101
      - tv-resnet152
  </details>

  * <details> <summary>ResNeXt</summary>

      - tv-resnext50_32x4d
      - tv-resnext101_32x4d
      - tv-resnext101_32x8d
      - tv-resnext101_32x16d
      - tv-resnext101_32x32d
      - tv-resnext101_32x48d
  </details>

  * <details> <summary>DenseNet</summary>

      - tv-densenet121
      - tv-densenet169
      - tv-densenet201
      - tv-densenet161

  </details>

  * <details> <summary>EfficientNetV1</summary>

      - timm-efficientnet-b0
      - timm-efficientnet-b1
      - timm-efficientnet-b2
      - timm-efficientnet-b3
      - timm-efficientnet-b4
      - timm-efficientnet-b5
      - timm-efficientnet-b6
      - timm-efficientnet-b7

  </details>

  * <details> <summary>EfficientNetLite</summary>

      - timm-efficientnet-lite0
      - timm-efficientnet-lite1
      - timm-efficientnet-lite2
      - timm-efficientnet-lite3
      - timm-efficientnet-lite4

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