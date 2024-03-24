import torch.nn as nn

class ClassificationHead(nn.Sequential):
    """Head for classification tasks
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.):
        """
            :args:
                in_channels: int
                    number of input channel
                out_channels: int
                    number of output channel
                dropout_rate: float, optional
                    rate of dropout
        """
        super().__init__(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_channels, out_channels),
        )

class SegmentationHead(nn.Sequential):
    """Head for segmentation tasks
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
            :args:
                in_channels: int
                    number of input channel
                out_channels: int
                    number of output channel
                kernel_size: int, optional
                    size of convolution kernel
        """
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
        )
