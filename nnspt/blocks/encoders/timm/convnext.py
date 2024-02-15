import torch.nn as nn

from timm.models.convnext import ConvNeXt

from nnspt.blocks.encoders.base import EncoderBase
from nnspt.blocks.encoders.converters import Converter1d, ConverterTimm

class ConvNeXtEncoder(ConvNeXt, EncoderBase):
    """Builder for encoder from ConvNeXtEncoder family
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

        del self.head

        ConverterTimm.convert(self)
        Converter1d.convert(self)

    def get_stages(self):
        return [
            nn.Identity(),
            self.stem,
            self.stages[0],
            self.stages[1],
            self.stages[2],
            self.stages[3],
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

convnext_encoders = {
    'timm-convnext-atto': {
        'encoder': ConvNeXtEncoder,
        'params': {
            'out_channels': (3, 40, 40, 80, 160, 320),
            'depths': (2, 2, 6, 2),
            'dims': (40, 80, 160, 320),
            'conv_mlp': True,
        },
    },
   'timm-convnext-femto': {
        'encoder': ConvNeXtEncoder,
        'params': {
            'out_channels': (3, 48, 48, 96, 192, 384),
            'depths': (2, 2, 6, 2),
            'dims': (48, 96, 192, 384),
            'conv_mlp': True,
        },
    },
    'timm-convnext-pico': {
        'encoder': ConvNeXtEncoder,
        'params': {
            'out_channels': (3, 64, 64, 128, 256, 512),
            'depths': (2, 2, 6, 2),
            'dims': (64, 128, 256, 512),
            'conv_mlp': True,
        },
    },
    'timm-convnext-nano': {
        'encoder': ConvNeXtEncoder,
        'params': {
            'out_channels': (3, 80, 80, 160, 320, 640),
            'depths': (2, 2, 8, 2),
            'dims': (80, 160, 320, 640),
            'conv_mlp': True,
        },
    },
    'timm-convnext-tiny': {
        'encoder': ConvNeXtEncoder,
        'params': {
            'out_channels': (3, 96, 96, 192, 384, 768),
            'depths': (3, 3, 9, 3),
            'dims': (96, 192, 384, 768),
            'conv_mlp': True,
        },
    },
    'timm-convnext-small': {
        'encoder': ConvNeXtEncoder,
        'params': {
            'out_channels': (3, 96, 96, 192, 384, 768),
            'depths': (3, 3, 27, 3),
            'dims': (96, 192, 384, 768),
            'conv_mlp': True,
        },
    },
    'timm-convnext-base': {
        'encoder': ConvNeXtEncoder,
        'params': {
            'out_channels': (3, 128, 128, 256, 512, 1024),
            'depths': (3, 3, 27, 3),
            'dims': (128, 256, 512, 1024),
            'conv_mlp': True,
        },
    },
    'timm-convnext-large': {
        'encoder': ConvNeXtEncoder,
        'params': {
            'out_channels': (3, 192, 192, 384, 768, 1536),
            'depths': (3, 3, 27, 3),
            'dims': (192, 384, 768, 1536),
            'conv_mlp': True,
        },
    },
    'timm-convnext-xlarge': {
        'encoder': ConvNeXtEncoder,
        'params': {
            'out_channels': (3, 256, 256, 512, 1024, 2048),
            'depths': (3, 3, 27, 3),
            'dims': (256, 512, 1024, 2048),
            'conv_mlp': True,
        },
    },
    'timm-convnext-xxlarge': {
        'encoder': ConvNeXtEncoder,
        'params': {
            'out_channels': (3, 384, 384, 768, 1536, 3072),
            'depths': (3, 4, 30, 3),
            'dims': (384, 768, 1536, 3072),
            'conv_mlp': True,
        },
    },
    'timm-convnextv2-atto': {
        'encoder': ConvNeXtEncoder,
        'params': {
            'out_channels': (3, 40, 40, 80, 160, 320),
            'depths': (2, 2, 6, 2),
            'dims': (40, 80, 160, 320),
            'use_grn': True,
            'ls_init_value': None,
        },
    },
    'timm-convnextv2-femto': {
        'encoder': ConvNeXtEncoder,
        'params': {
            'out_channels': (3, 48, 48, 96, 192, 384),
            'depths': (2, 2, 6, 2),
            'dims': (48, 96, 192, 384),
            'use_grn': True,
            'ls_init_value': None
        },
    },
    'timm-convnextv2-pico': {
        'encoder': ConvNeXtEncoder,
        'params': {
            'out_channels': (3, 64, 64, 128, 256, 512),
            'depths': (2, 2, 6, 2),
            'dims': (64, 128, 256, 512),
            'use_grn': True,
            'ls_init_value': None
        },
    },
}
