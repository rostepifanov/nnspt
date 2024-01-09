import torch.nn as nn

from functools import partial
from timm.layers.activations import Swish
from timm.models.efficientnet import EfficientNet, decode_arch_def, round_channels, default_cfgs

from nnspt.blocks.encoders.base import EncoderBase
from nnspt.blocks.encoders.converters import Converter1d, ConverterTimm

def get_efficientnet_kwargs(channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
    """
        :NOTE:
            it is a modified copy of _gen_efficientnet from timm package
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]

    kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=Swish,
        drop_rate=drop_rate,
        drop_path_rate=0.2,
    )

    return kwargs

def gen_efficientnet_lite_kwargs(channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
    """
        :NOTE:
            it is a modified copy of _gen_efficientnet_lite from timm package
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r2_k5_s2_e6_c40'],
        ['ir_r3_k3_s2_e6_c80'],
        ['ir_r3_k5_s1_e6_c112'],
        ['ir_r4_k5_s2_e6_c192'],
        ['ir_r1_k3_s1_e6_c320'],
    ]

    kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
        num_features=1280,
        stem_size=32,
        fix_stem=True,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=nn.ReLU6,
        drop_rate=drop_rate,
        drop_path_rate=0.2,
    )

    return kwargs

class EfficientNetEncoder(EfficientNet, EncoderBase):
    """Builder for encoder from EfficientNet family
    """
    def __init__(self, stage_idxs, out_channels, depth=5, **kwargs):
        """
            :NOTE:

            :args:
                stage_idxs (list of int): nested parameters for timm efficientnet
                out_channels (list of int): channel number of output tensors, including intermediate ones
                depth (int): depth of encoder
        """
        super().__init__(**kwargs)

        self.in_channels = 3
        self.out_channels_ = out_channels

        self.depth = depth
        self._stage_idxs = stage_idxs

        del self.classifier

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv_stem, self.bn1),
            self.blocks[:self._stage_idxs[0]],
            self.blocks[self._stage_idxs[0]:self._stage_idxs[1]],
            self.blocks[self._stage_idxs[1]:self._stage_idxs[2]],
            self.blocks[self._stage_idxs[2]:],
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

class EfficientNetV1Encoder(EfficientNetEncoder):
    """Builder for EfficientNetV1 encoders
    """
    def __init__(
        self,
        stage_idxs,
        out_channels,
        depth=5,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        drop_rate=0.2,
    ):
        kwargs = get_efficientnet_kwargs(channel_multiplier, depth_multiplier, drop_rate)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)

        ConverterTimm.convert(self)
        Converter1d.convert(self)

class EfficientNetLiteEncoder(EfficientNetEncoder):
    """Builder for EfficientNetLite encoders
    """
    def __init__(
        self,
        stage_idxs,
        out_channels,
        depth=5,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        drop_rate=0.2,
    ):
        kwargs = gen_efficientnet_lite_kwargs(channel_multiplier, depth_multiplier, drop_rate)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)

        ConverterTimm.convert(self)
        Converter1d.convert(self)

efficientnet_encoders = {
    'timm-efficientnet-b0': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 32, 24, 40, 112, 320),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.0,
            'drop_rate': 0.2,
        },
    },
    'timm-efficientnet-b1': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 32, 24, 40, 112, 320),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.1,
            'drop_rate': 0.2,
        },
    },
    'timm-efficientnet-b2': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 32, 24, 48, 120, 352),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.1,
            'depth_multiplier': 1.2,
            'drop_rate': 0.3,
        },
    },
    'timm-efficientnet-b3': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 40, 32, 48, 136, 384),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.2,
            'depth_multiplier': 1.4,
            'drop_rate': 0.3,
        },
    },
    'timm-efficientnet-b4': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 48, 32, 56, 160, 448),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.4,
            'depth_multiplier': 1.8,
            'drop_rate': 0.4,
        },
    },
    'timm-efficientnet-b5': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 48, 40, 64, 176, 512),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.6,
            'depth_multiplier': 2.2,
            'drop_rate': 0.4,
        },
    },
    'timm-efficientnet-b6': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 56, 40, 72, 200, 576),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.8,
            'depth_multiplier': 2.6,
            'drop_rate': 0.5,
        },
    },
    'timm-efficientnet-b7': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 64, 48, 80, 224, 640),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 2.0,
            'depth_multiplier': 3.1,
            'drop_rate': 0.5,
        },
    },
    'timm-efficientnet-b8': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 72, 56, 88, 248, 704),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 2.2,
            'depth_multiplier': 3.6,
            'drop_rate': 0.5,
        },
    },
    'timm-efficientnet-l2': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 136, 104, 176, 480, 1376),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 4.3,
            'depth_multiplier': 5.3,
            'drop_rate': 0.5,
        },
    },
    'timm-efficientnet-lite0': {
        'encoder': EfficientNetLiteEncoder,
        'params': {
            'out_channels': (3, 32, 24, 40, 112, 320),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.0,
            'drop_rate': 0.2,
        },
    },
    'timm-efficientnet-lite1': {
        'encoder': EfficientNetLiteEncoder,
        'params': {
            'out_channels': (3, 32, 24, 40, 112, 320),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.1,
            'drop_rate': 0.2,
        },
    },
    'timm-efficientnet-lite2': {
        'encoder': EfficientNetLiteEncoder,
        'params': {
            'out_channels': (3, 32, 24, 48, 120, 352),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.1,
            'depth_multiplier': 1.2,
            'drop_rate': 0.3,
        },
    },
    'timm-efficientnet-lite3': {
        'encoder': EfficientNetLiteEncoder,
        'params': {
            'out_channels': (3, 32, 32, 48, 136, 384),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.2,
            'depth_multiplier': 1.4,
            'drop_rate': 0.3,
        },
    },
    'timm-efficientnet-lite4': {
        'encoder': EfficientNetLiteEncoder,
        'params': {
            'out_channels': (3, 32, 32, 56, 160, 448),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.4,
            'depth_multiplier': 1.8,
            'drop_rate': 0.4,
        },
    },
}
