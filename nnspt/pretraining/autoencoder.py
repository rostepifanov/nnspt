import torch, torch.nn as nn

from nnspt.blocks import Encoder

class DecoderBlock(nn.Module):
    """Autoencoder decoder block
    """
    def __init__(
            self,
            in_channels,
            out_channels,
        ):
        """
            :args:
                in_channels (int): number of input channel
                out_channels (int): number of output channel
        """
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
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

    def forward(self, x, shape=None):
        if shape is not None:
            scale_factor = []

            getdim = lambda vector, axis : vector.shape[axis]

            naxis = len(x.shape)
            for axis in range(2, naxis):
                scale_factor.append(shape[axis]/getdim(x, axis))

            scale_factor = tuple(scale_factor)
        else:
            scale_factor = 2

        x = nn.functional.interpolate(x, scale_factor=scale_factor, mode='linear')

        x = self.conv1(x)
        x = self.conv2(x)

        return x

class Decoder(nn.Module):
    """Autoencoder decoder
    """
    def __init__(
            self,
            nblocks,
            channels
        ):
        """
            :args:
                nblocks (int): depth of decoder
                channels (list of int): number of channels in decoder
        """
        super().__init__()

        in_channels = channels[-nblocks-1:]
        out_channels = channels[-nblocks:]

        blocks = []

        for in_, out_ in zip(in_channels, out_channels):
            block = DecoderBlock(in_, out_)
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
            shape = skips[i].shape
            x = block(x, shape)

        return x

class Autoencoder(nn.Module):
    """Autoencoder with simple encoder-decoder structure
    """
    def __init__(
            self,
            nchannels=12,
            depth=5,
            encoder='tv-resnet34',
        ):
        """
            :args:
                nchannels (int): number of channels of input tensor
                depth (int): depth of model
                name (str): architecture of encoder in model
        """
        super().__init__()

        self.encoder = Encoder(
            in_channels=nchannels,
            depth=depth,
            name=encoder,
        )

        self.decoder = Decoder(
            nblocks=depth,
            channels=self.encoder.out_channels[::-1],
        )

        self.name = 'ae-{}'.format(encoder)
        self.initialize()

    def initialize(self):
        """
            :NOTE:
                function to init weights of torch layers
        """

        for node in self.modules():
            if isinstance(node, nn.Conv1d):
                nn.init.kaiming_uniform_(node.weight, mode='fan_in', nonlinearity='relu')
                if node.bias is not None: nn.init.constant_(node.bias, 0)

            elif isinstance(node, (nn.BatchNorm1d , nn.LayerNorm)):
                nn.init.constant_(node.weight, 1)
                nn.init.constant_(node.bias, 0)

    def forward(self, x):
        """
            :args:
                x (torch.tensor[batch_size, in_channels, length]): input tensor

            :return:
                torch.tensor[batch_size, out_channels, length]
        """

        f = self.encoder(x)
        x = self.decoder(*f)

        return x
