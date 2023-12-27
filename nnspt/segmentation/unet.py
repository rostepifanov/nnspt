import torch, torch.nn as nn

from nnspt.blocks import Encoder, SegmentationHead
from nnspt.segmentation.base import SegmentationSingleHeadModel

class DecoderBlock(nn.Module):
    """Unet decoder block
    """
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
        ):
        """
            :args:
                in_channels (int): number of input channel
                skip_channels (int): number of skip channel
                out_channels (int): number of output channel
        """
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels + skip_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x, skip=None, shape=None):
        if shape is not None:
            scale_factor = []

            getdim = lambda vector, axis : vector.shape[axis]

            naxis = len(x.shape)
            for axis in range(2, naxis):
                scale_factor.append(shape[axis]/getdim(x, axis))

            scale_factor = tuple(scale_factor)
        else:
            scale_factor = 2

        x = nn.functional.interpolate(x, scale_factor=scale_factor, mode='nearest')

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x

class Decoder(nn.Module):
    """Unet decoder
    """
    def __init__(
            self,
            nblocks,
            echannels,
            dchannels
        ):
        """
            :args:
                nblocks (int): depth of decoder
                echannels (list of int): number of channels in encoder
                dchannels (list of int): number of channels in decoder
        """
        super().__init__()

        in_channels = (echannels[0], *dchannels[:-1])
        skip_channels = (*echannels[1:], 0)
        out_channels = dchannels[-nblocks:]

        blocks = []

        for in_, skip_, out_ in zip(in_channels, skip_channels, out_channels):
            block = DecoderBlock(in_, skip_, out_)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, *feats):
        """
            :args:
                feats (list of torch.tensor): list of latent features
        """
        feats = feats[::-1]

        x = feats[0]
        skips = feats[1:]

        for i, block in enumerate(self.blocks):
            if i < len(skips) - 1:
                skip = skips[i]
                shape = skips[i].shape
            else:
                skip = None
                shape = skips[i].shape

            x = block(x, skip, shape)

        return x

class Unet(SegmentationSingleHeadModel):
    """Unet is a fully convolution neural network for semantic segmentation.
        See details in https://arxiv.org/abs/1505.04597
    """
    def __init__( 
            self,
            in_channels=12,
            out_channels=2,
            depth=5,
            encoder='tv-resnet34',
        ):
        """
            :args:
                in_channels (int): number of channels of input tensor
                out_channels (int): number of channels of output tensor
                depth (int): depth of model
                name (str): architecture of encoder in model
        """
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            depth=depth,
            name=encoder,
        )

        self.decoder = Decoder(
            nblocks=depth,
            echannels=self.encoder.out_channels[:0:-1],
            dchannels=self.encoder.out_channels[:0:-1],
        )

        self.head = SegmentationHead(
            in_channels=self.encoder.out_channels[1],
            out_channels=out_channels,
            kernel_size=3,
        )

        self.name = 'u-{}'.format(encoder)
        self.initialize()
