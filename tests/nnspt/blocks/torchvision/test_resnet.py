import pytest

import torch

from nnspt.blocks.encoders import Encoder

NAMES = [
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

@pytest.mark.resnet
@pytest.mark.encoders
@pytest.mark.parametrize('name', NAMES)
def test_ResNetEncoder_CASE_creation(name):
    IN_CHANNELS = 3
    DEPTH = 5

    encoder = Encoder(
        in_channels=IN_CHANNELS,
        depth=DEPTH,
        name=name
    )

    assert len(encoder.out_channels) == DEPTH + 1

    encoder.eval()

    x = torch.randn(1, IN_CHANNELS, 64)
    y = encoder(x)

    for nchannels, yi in zip(encoder.out_channels, y):
        assert yi.shape[0] == x.shape[0]
        assert yi.shape[1] == nchannels
