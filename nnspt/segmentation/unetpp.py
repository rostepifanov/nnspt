import torch, torch.nn as nn

from nnspt.blocks import Encoder, SegmentationHead
from nnspt.segmentation.unet import DecoderBlock
from nnspt.segmentation.base import SegmentationSingleHeadModel

class Decoder(nn.Module):
    """Unetpp decoder

        :NOTE:
            blocks have the following notation:

            B (0, 0) -- B (0, 1) -- B (0, 2) (depth, layer)
                  B (1, 1) -- B (1, 2)
                        B (2, 2)
    """
    def __init__(
            self,
            nblocks,
            channels,
        ):
        """
            :args:
                nblocks (int): depth of decoder
                channels (list of int): number of channels
        """
        super().__init__()

        channels = channels[-nblocks:]

        in_channels = channels
        skip_channels = (*channels[1:], 0)
        out_channels = channels

        blocks = {}
        skips = {}

        for idx in range(nblocks):
            skips[f's_{idx+1}_{idx}'] = channels[-idx-1]

        for idx in range(nblocks):
            for jdx in range(nblocks - idx):
                depth = jdx
                layer = idx+jdx

                in_ = in_channels[-depth-1]
                skip_ = skip_channels[-depth-1]
                out_ = out_channels[-depth-1]

                if depth > 0:
                    for sdx in range(layer-depth):
                        skip_ += skips[f's_{depth}_{layer-sdx-2}']

                skips[f's_{depth}_{layer}'] = out_

                block = DecoderBlock(in_, skip_, out_)
                blocks[f'b_{depth}_{layer}'] = block

            if idx == 0:
                in_channels = (0, *in_channels[:-1])
                skip_channels = (0, *skip_channels[:-2], 0)

        self.blocks = nn.ModuleDict(blocks)
        self.nblocks = nblocks

    def forward(self, *feats):
        """
            :args:
                feats (list of torch.tensor): list of latent features
        """
        xs = dict()

        for idx, x in enumerate(feats):
            xs[f'x_{idx}_{idx-1}'] = x

        for idx in range(self.nblocks):
            for jdx in range(self.nblocks - idx):
                depth = jdx
                layer = idx+jdx

                block = self.blocks[f'b_{depth}_{layer}']

                if depth == 0:
                    skip = None
                    shape = xs[f'x_{0}_{-1}'].shape
                else:
                    skip = torch.concat([ xs[f'x_{depth}_{layer-sdx-1}'] for sdx in range(layer-depth+1) ], axis=1)
                    shape = xs[f'x_{depth}_{layer-1}'].shape

                x = xs[f'x_{depth+1}_{layer}']
                x = block(x, skip, shape)
                xs[f'x_{depth}_{layer}'] = x

        return xs[f'x_{0}_{self.nblocks-1}']

class Unetpp(SegmentationSingleHeadModel):
    """Unetpp is a fully convolution neural network for semantic segmentation.
       See details in https://arxiv.org/abs/1807.10165
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
            channels=self.encoder.out_channels[:0:-1],
        )

        self.head = SegmentationHead(
            in_channels=self.encoder.out_channels[1],
            out_channels=out_channels,
            kernel_size=3,
        )

        self.name = 'unetpp-{}'.format(encoder)
        self.initialize()
