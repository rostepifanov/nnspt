import torch, torch.nn as nn

from nnspt.blocks.encoders.torchvision import *
from nnspt.blocks.encoders.timm import *
from nnspt.blocks.encoders.misc import __classinit

nnspt_encoders = {}
nnspt_encoders.update(resnet_encoders)
nnspt_encoders.update(densenet_encoders)
nnspt_encoders.update(efficientnet_encoders)
nnspt_encoders.update(convnext_encoders)

@__classinit
class Encoder(object):
    """Fake class for creation of nnspt encoders by name
    """
    @classmethod
    def _init__class(cls):
        return cls()

    @staticmethod
    def _patch(encoder, in_channels, default_in_channels=3):
        for node in encoder.modules():
            if isinstance(node, nn.Conv1d) and node.in_channels == default_in_channels:
                break

        encoder.out_channels_ = (in_channels, *encoder.out_channels_[1:])

        weight = node.weight.detach()
        node.in_channels = in_channels

        nweight = torch.Tensor(
            node.out_channels,
            in_channels // node.groups,
            *node.kernel_size
        )

        for i in range(in_channels):
            nweight[:, i] = weight[:, i % default_in_channels]

        node.weight = nn.parameter.Parameter(nweight)

    def __call__(self, in_channels=3, depth=5, name='tv-resnet34'):
        """
            :args:
                in_channels (int): number of channels of input tensor
                depth (int): depth of encoder
                name (str): name of encoder to create

            :return:
                created encoder
        """
        try:
            type = nnspt_encoders[name]['encoder']
        except:
            raise KeyError('Wrong encoder name `{}`, supported encoders: {}'.format(name, list(nnspt_encoders.keys())))

        params = nnspt_encoders[name]['params']
        params.update(depth=depth)

        encoder = type(**params)
        self._patch(encoder, in_channels)

        return encoder
