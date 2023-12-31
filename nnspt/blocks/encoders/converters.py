import timm
import types
import torch.nn as nn

from collections import OrderedDict

from nnspt.blocks.encoders.misc import __classinit

@__classinit
class Convertor(object):
    """Class to provide layer convertors for adaptation of CNN
    """
    @classmethod
    def _init__class(cls):
        """
            :return:
                instance of class, similar to singleton pattern
        """
        cls._registry = { }

        return cls()

    def convert(self, model):
        """
            :NOTE:
                conversion takes inplace

            :args:
                model (torch.nn.Module): PyTorch model
        """
        def __is_generator_empty(generator):
            try:
                next(generator)
                return False
            except StopIteration:
                return True

        stack = [model]

        while stack:
            node = stack[-1]

            stack.pop()

            for name, child in node.named_children():
                if not __is_generator_empty(child.children()):
                    stack.append(child)

                setattr(node, name, self(child))

    def __call__(self, layer):
        """
            :args:
                layer (torch.nn.Module): PyTorch layer to convert

            :return:
                converted layer
        """
        if type(layer) in self._registry:
            return self._registry[type(layer)](layer)
        else:
            return self._func_None(layer)

    @classmethod
    def _func_None(cls, layer):
        """
            :NOTE:
                identity convertation
        """
        return layer

@__classinit
class Convertor1d(Convertor.__class__):
    """Class to provide layer convertors from 2D to 1D
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

@__classinit
class ConvertorTimm(Convertor.__class__):
    """Class to provide layer convertors to timm custom classes
    """
    @classmethod
    def _init__class(cls):
        cls._registry = {
            timm.layers.norm_act.BatchNormAct2d: getattr(cls, '_func_timm_BatchNormAct2d'),
            timm.models._efficientnet_blocks.SqueezeExcite: getattr(cls, '_func_timm_SqueezeExcite'),
        }

        return cls()

    @staticmethod
    def _timm_batchnormact2d_forward(self, x):
        """
            :NOTE:
                it is a copy of timm function without shape assert
        """
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = nn.functional.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

        x = self.drop(x)
        x = self.act(x)

        return x

    @classmethod
    def _func_timm_BatchNormAct2d(cls, layer):
        layer.forward = types.MethodType(cls._timm_batchnormact2d_forward, layer)

        return layer

    @staticmethod
    def _timm_squeezeexcite_forward(self, x):
        """
            :NOTE:
                it is a copy of timm function with correct operations under dims
        """
        x_se = x.mean((2, ), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

    @classmethod
    def _func_timm_SqueezeExcite(cls, layer):
        layer.forward = types.MethodType(cls._timm_squeezeexcite_forward, layer)

        return layer
