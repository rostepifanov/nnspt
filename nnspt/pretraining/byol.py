import torch, torch.nn as nn
import torch.optim.swa_utils as tsu

from nnspt.blocks.encoders import Encoder

def get_ema_avg_fn(alpha=0.99):
    """Funtion to average weights by EMA

        :NOTE:
            Will be deprecated after moving to torch 2.0.0
    """
    @torch.no_grad()
    def ema_update(ema_param, current_param, num_averaged):
        return alpha * ema_param + (1 - alpha) * current_param

    return ema_update

class BYOLProjector(nn.Sequential):
    def __init__(self, nfeautures, reduction=8):
        """
            :args:
                nfeautures: int
                    size of encoder latent features
                reduction: int, optional
                    reduction ratio of encoder latent feature
        """
        nembedding = (nfeautures + reduction - 1) // reduction

        super().__init__(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(nfeautures, nembedding)
        )

class BYOLPredictor(nn.Sequential):
    def __init__(self, nfeautures, reduction=8):
        """
            :args:
                nfeautures: int
                    size of encoder latent features
                reduction: int, optional
                    reduction ratio of encoder latent feature
        """
        nembedding = (nfeautures + reduction - 1) // reduction

        super().__init__(
            nn.GELU(),
            nn.BatchNorm1d(nembedding),
            nn.Linear(nembedding, nembedding)
        )

class BYOL(torch.nn.Module):
    """BYOL model. See details in https://arxiv.org/abs/2006.07733
    """
    def __init__(
        self,
        nchannels=12,
        depth=5,
        encoder='tv-resnet34',
        reduction=8
    ):
        """
            :args:
                nchannels: int
                    number of channels of input tensor
                depth: int, optional
                    depth of model
                encoder: str, optional
                    architecture of encoder in model
                reduction: int, optional
                    reduction ratio of latent feature of encoder
        """
        super().__init__()

        encoder = Encoder(in_channels=nchannels, depth=depth, name=encoder)
        projector = BYOLProjector(nfeautures=encoder.out_channels[-1], reduction=reduction)
        predictor =  BYOLPredictor(nfeautures=encoder.out_channels[-1], reduction=reduction)

        self.online = torch.nn.ModuleDict({
            'encoder': encoder,
            'projector': projector,
            'predictor': predictor,
        })

        self.target = torch.nn.ModuleDict({
            'encoder': tsu.AveragedModel(encoder, avg_fn=get_ema_avg_fn()),
            'projector': tsu.AveragedModel(projector, avg_fn=get_ema_avg_fn()),
        })

    @property
    def encoder(self):
        """
            :return:
                output: Encoder
                    pretrained encoder
        """
        return self.target.encoder.module

    def forward(self, xo, xt):
        """
            :args:
                xo: [batch_size, nchannels, length] torch.tensor
                    input tensor for online path
                xt: [batch_size, nchannels, length] torch.tensor
                    input tensor for target path

            :return:
                output: tuple of [batch_size, nfeatures] torch.tensor
                    qo is prediction of xo, zt is projection of xt
        """
        fo = self.online.encoder(xo)
        zo = self.online.projector(fo[-1])
        qo = self.online.predictor(zo)

        qo = qo / torch.norm(qo, dim=-1, keepdim=True)

        with torch.no_grad():
            ft = self.target.encoder(xt)
            zt = self.target.projector(ft[-1])

            zt = zt / torch.norm(zt, dim=-1, keepdim=True)

        self.target.encoder.update_parameters(self.online.encoder)
        self.target.projector.update_parameters(self.online.projector)

        return qo, zt
