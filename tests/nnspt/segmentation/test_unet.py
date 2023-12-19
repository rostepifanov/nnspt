import pytest

import torch

from nnspt.segmentation.unet import Unet

ENCODERS = [
    'tv-resnet18',
    'tv-resnet34',
    'tv-resnet50',
    'tv-resnet101',
    'tv-resnet152',
    'tv-resnext50_32x4d',
    'tv-resnext101_32x4d',
    'tv-resnext101_32x8d',
    'tv-resnext101_32x16d',
    'tv-resnext101_32x32d',
    'tv-resnext101_32x48d',
]

@pytest.mark.unet
@pytest.mark.segmentation
@pytest.mark.parametrize('encoder', ENCODERS)
def test_Unet_CASE_creation(encoder):
    IN_CHANNELS = 3
    OUT_CHANNELS = 2
    DEPTH = 5

    model = Unet(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS, 
        depth=DEPTH,
        encoder=encoder
    )

    model.eval()

    x = torch.randn(1, IN_CHANNELS, 64)
    y = model(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == OUT_CHANNELS
    assert y.shape[2] == x.shape[2]
