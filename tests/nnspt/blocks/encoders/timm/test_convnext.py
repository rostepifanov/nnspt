import pytest

import torch

from nnspt.blocks.encoders import Encoder
from nnspt.blocks.encoders.timm import convnext_encoders

ENCODERS = convnext_encoders.keys()

@pytest.mark.convnext
@pytest.mark.encoders
@pytest.mark.parametrize('name', ENCODERS)
def test_ConvNeXtEncoder_CASE_creation(name):
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
