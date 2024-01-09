import torch.nn as nn

from collections import OrderedDict

from nnspt.blocks.encoders.misc import __classinit
from nnspt.blocks.encoders.converters.base import Converter

@__classinit
class Converter1d(Converter.__class__):
    """Class to provide layer converters from 2D to 1D
    """
    @classmethod
    def _init__class(cls):
        cls._registry = {
            nn.Conv2d: getattr(cls, '_func_Conv2d'),
            nn.MaxPool2d: getattr(cls, '_func_MaxPool2d'),
            nn.AvgPool2d: getattr(cls, '_func_AvgPool2d'),
            nn.BatchNorm2d: getattr(cls, '_func_BatchNorm2d'),
        }

        return cls()

    @classmethod
    def _func_Conv2d(cls, layer2d):
        kwargs = {
            'in_channels': layer2d.in_channels,
            'out_channels': layer2d.out_channels,
            'kernel_size': cls.__squeze_tuple(layer2d.kernel_size),
            'stride': cls.__squeze_tuple(layer2d.stride),
            'padding': cls.__squeze_tuple(layer2d.padding),
            'dilation': cls.__squeze_tuple(layer2d.dilation),
            'groups': layer2d.groups,
            'bias': 'bias' in layer2d.state_dict(),
            'padding_mode': layer2d.padding_mode,
        }

        state2d = layer2d.state_dict()

        state1d = OrderedDict()

        state1d['weight'] = state2d['weight'][:, :, :, 0]

        if 'bias' in state2d:
            state1d['bias'] = state2d['bias']

        layer1d = nn.Conv1d(**kwargs)
        layer1d.load_state_dict(state1d)

        return layer1d

    @staticmethod
    def __squeze_tuple(param):
        assert param[0] == param[1]

        return (param[0], )

    @classmethod
    def _func_MaxPool2d(cls, layer2d):
        kwargs = {
            'kernel_size': layer2d.kernel_size,
            'stride': layer2d.stride,
            'padding': layer2d.padding,
            'dilation': layer2d.dilation,
            'return_indices': layer2d.return_indices,
            'ceil_mode': layer2d.ceil_mode,
        }

        layer1d = nn.MaxPool1d(**kwargs)

        return layer1d

    @classmethod
    def _func_AvgPool2d(cls, layer2d):
        kwargs = {
            'kernel_size': layer2d.kernel_size,
            'stride': layer2d.stride,
            'padding': layer2d.padding,
            'ceil_mode': layer2d.ceil_mode,
            'count_include_pad': layer2d.count_include_pad,
        }

        layer1d = nn.AvgPool1d(**kwargs)

        return layer1d

    @classmethod
    def _func_BatchNorm2d(cls, layer2d):
        kwargs = {
            'num_features': layer2d.num_features,
            'eps': layer2d.eps,
            'momentum': layer2d.momentum,
            'affine': layer2d.affine,
            'track_running_stats': layer2d.track_running_stats,
        }

        layer1d = nn.BatchNorm1d(**kwargs)

        return layer1d
