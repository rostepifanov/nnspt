import torch.nn as nn

class SegmentationSingleHeadModel(nn.Module):
    """Base class for segmentation model with single head
    """
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
                x: [batch_size, in_channels, length] torch.tensor
                    input tensor

            :return:
                output: [batch_size, out_channels, length] torch.tensor
                    logits of class probabilities
        """

        f = self.encoder(x)
        x = self.decoder(*f)
        x = self.head(x)

        return x
