import pytest

import torch

from nnspt.blocks.encoders import Encoder
from nnspt.segmentation.unetpp import Unetpp
from nnspt.blocks.encoders.builder import nnspt_encoders

ENCODERS = nnspt_encoders.keys()

@pytest.mark.unetpp
@pytest.mark.segmentation
def test_Unetpp_CASE_creation_AND_defaul_parameters():
    model = Unetpp()

@pytest.mark.unetpp
@pytest.mark.segmentation
def test_Unetpp_CASE_wrong_encoder_name():
    with pytest.raises(KeyError) as e:
        model = Unetpp(encoder='wrong_name')

@pytest.mark.unetpp
@pytest.mark.segmentation
@pytest.mark.parametrize('name', ENCODERS)
def test_Unetpp_CASE_creation(name):
    IN_CHANNELS = 12
    OUT_CHANNELS = 2
    DEPTH = 4

    model = Unetpp(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        depth=DEPTH,
        encoder=name,
    )

    model.eval()

    x = torch.randn(1, IN_CHANNELS, 64)
    y = model(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == OUT_CHANNELS
    assert y.shape[2] == x.shape[2]

@pytest.mark.unet
@pytest.mark.segmentation
@pytest.mark.parametrize('name', ENCODERS)
def test_Unet_CASE_load_encoder(name):
    IN_CHANNELS = 3
    OUT_CHANNELS = 2
    DEPTH = 5

    encoder = Encoder(
        in_channels=IN_CHANNELS,
        depth=DEPTH,
        name=name,
    )

    model = Unetpp(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        depth=DEPTH,
        encoder=name,
    )

    state = encoder.state_dict()
    model.encoder.load_state_dict(state)
