import torch.nn as nn

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from nnspt.blocks.encoders.base import EncoderBase
from nnspt.blocks.encoders.converters import Converter1d

class ResNetEncoder(ResNet, EncoderBase):
    """Builder for encoder from ResNet family such as ResNet, ResNeXt, and etc
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

        del self.fc
        del self.avgpool

        Converter1d.convert(self)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
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
            features.append(x)

        return features

resnet_encoders = {
    'tv-resnet18': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 64, 128, 256, 512),
            'block': BasicBlock,
            'layers': [2, 2, 2, 2],
        },
    },
    'tv-resnet34': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 64, 128, 256, 512),
            'block': BasicBlock,
            'layers': [3, 4, 6, 3],
        },
    },
    'tv-resnet50': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
        },
    },
    'tv-resnet101': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
        },
    },
    'tv-resnet152': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 8, 36, 3],
        },
    },
    'tv-resnext50_32x4d': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
            'groups': 32,
            'width_per_group': 4,
        },
    },
    'tv-resnext101_32x4d': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 4,
        },
    },
    'tv-resnext101_32x8d': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 8,
        },
    },
    'tv-resnext101_32x16d': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 16,
        },
    },
    'tv-resnext101_32x32d': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 32,
        },
    },
    'tv-resnext101_32x48d': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 48,
        },
    },
}
