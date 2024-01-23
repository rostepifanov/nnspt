import torch, torch.nn as nn

class SpatialChannelSqueezeExcitationLayer(nn.Module):
    """Spatial and Channel 'Squeeze & Excitation' layer
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, (in_channels + reduction - 1) // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d((in_channels + reduction - 1) // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        self.sSE = nn.Sequential(
            nn.Conv1d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
