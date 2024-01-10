import timm
import types
import torch.nn as nn

from nnspt.blocks.encoders.misc import __classinit
from nnspt.blocks.encoders.converters.base import Converter

@__classinit
class ConverterTimm(Converter.__class__):
    """Class to provide layer converters to timm custom classes
    """
    @classmethod
    def _init__class(cls):
        cls._registry = {
            timm.layers.grn.GlobalResponseNorm: getattr(cls, '_func_timm_GlobalResponseNorm'),
            timm.layers.norm.LayerNorm2d: getattr(cls, '_func_timm_LayerNorm2d'),
            timm.layers.norm_act.BatchNormAct2d: getattr(cls, '_func_timm_BatchNormAct2d'),
            timm.models._efficientnet_blocks.SqueezeExcite: getattr(cls, '_func_timm_SqueezeExcite'),
            timm.models.convnext.ConvNeXtBlock: getattr(cls, '_func_timm_ConvNeXtBlock'),
        }

        return cls()

    @classmethod
    def _func_timm_GlobalResponseNorm(cls, layer):
        if layer.channel_dim == -1:
            layer.spatial_dim = (1, )
            layer.wb_shape = (1, 1, -1)
        else:
            layer.spatial_dim = (2, )
            layer.wb_shape = (1, -1, 1)

        return layer

    @staticmethod
    def _timm_layernorm2dforward(self, x):
        """
            :NOTE:
                it is a copy of timm.layers.norm.LayerNorm2d function with correct operations under dims
        """
        x = x.permute(0, 2, 1)
        if self._fast_norm:
            x = timm.layers.fast_norm.fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 2, 1)

        return x

    @classmethod
    def _func_timm_LayerNorm2d(cls, layer):
        layer.forward = types.MethodType(cls._timm_layernorm2dforward, layer)

        return layer

    @staticmethod
    def _timm_batchnormact2d_forward(self, x):
        """
            :NOTE:
                it is a copy of timm.layers.norm_act.BatchNormAct2d function without shape assert
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
                it is a copy of timm.layers.squeeze_excite.SEModule function with correct operations under dims
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

    @staticmethod
    def _func_timm_convnextblockforward(self, x):
        """
            :NOTE:
                it is a copy of timm.models.convnext.ConvNeXtBlock function with correct operations under dims
        """
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 2, 1)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)
        return x

    @classmethod
    def _func_timm_ConvNeXtBlock(cls, layer):
        layer.forward = types.MethodType(cls._func_timm_convnextblockforward, layer)

        return layer
