import torch.nn as nn

class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, dropout_rate=0.):
        super().__init__(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_channels, out_channels),
        )

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
        )
