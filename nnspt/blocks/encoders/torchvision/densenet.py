import torch.nn as nn

from torchvision.models.densenet import DenseNet

from nnspt.blocks.encoders.base import EncoderBase
from nnspt.blocks.encoders.converters import Converter1d

class _DenseNetTransition(nn.Module):
    def __init__(self, transition_):
        super().__init__()

        self.add_module('norm', transition_.norm)
        self.add_module('relu', transition_.relu)
        self.add_module('conv', transition_.conv)
        self.add_module('pool', transition_.pool)

    def forward(self, x):
        x = self.norm(x)
        s = self.relu(x)
        x = self.conv(s)
        x = self.pool(x)

        return x, s

class DenseNetEncoder(DenseNet, EncoderBase):
    """Builder for encoder from DenseNet family
    """
    def __init__(self, out_channels, depth=5, **kwargs):
        """
            :NOTE:

            :args:
                out_channels (list of int): channel number of output tensors, including intermediate ones
                depth (int): depth of encoder
        """
        super().__init__(**kwargs)

        self.in_channels = 3
        self.out_channels_ = out_channels

        self.depth = depth

        del self.classifier

        Converter1d.convert(self)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(
                self.features.conv0,
                self.features.norm0,
                self.features.relu0
            ),
            nn.Sequential(
                self.features.pool0,
                self.features.denseblock1,
                _DenseNetTransition(
                    self.features.transition1
                ),
            ),
            nn.Sequential(
                self.features.denseblock2,
                _DenseNetTransition(
                    self.features.transition2
                )
            ),
            nn.Sequential(
                self.features.denseblock3,
                 _DenseNetTransition(
                    self.features.transition3
                )
            ),
            nn.Sequential(
                self.features.denseblock4,
                self.features.norm5
            ),
        ]

    def forward(self, x):
        """
            :args:
                x (torch.tensor[batch_size, in_channels, length]): input tensor

            :return:
                list of torch.tensor[batch_size, schannels, slength]
        """
        stages = self.get_stages()
        features = []

        for i in range(self.depth + 1):
            x = stages[i](x)

            if isinstance(x, tuple):
                x, s = x
                features.append(s)
            else:
                features.append(x)

        return features

densenet_encoders = {
    'tv-densenet121': {
        'encoder': DenseNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 1024),
            'num_init_features': 64,
            'growth_rate': 32,
            'block_config': (6, 12, 24, 16),
        },
    },
    'tv-densenet169': {
        'encoder': DenseNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1280, 1664),
            'num_init_features': 64,
            'growth_rate': 32,
            'block_config': (6, 12, 32, 32),
        },
    },
    'tv-densenet201': {
        'encoder': DenseNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1792, 1920),
            'num_init_features': 64,
            'growth_rate': 32,
            'block_config': (6, 12, 48, 32),
        },
    },
    'tv-densenet161': {
        'encoder': DenseNetEncoder,
        'params': {
            'out_channels': (3, 96, 384, 768, 2112, 2208),
            'num_init_features': 96,
            'growth_rate': 48,
            'block_config': (6, 12, 36, 24),
        },
    },
}
