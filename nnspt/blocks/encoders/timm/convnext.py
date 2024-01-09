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
}
