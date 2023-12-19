from nnspt.blocks.encoders.torchvision import *
from nnspt.blocks.encoders.mics import __classinit

nnspt_encoders = {}
nnspt_encoders.update(resnet_encoders)

@__classinit
class Encoder(object):
    """Fake class for creation of nnspt encoders by name
    """
    @classmethod
    def _init__class(cls):
        return cls()

    def __call__(self, name, in_channels=3, depth=5):
        """
            :args:
                name (str): name of encoder to create
                in_channels (int): number of channels of input tensor
                depth (int): depth of encoder

            :return:
                created encoder
        """
        try:
            type = nnspt_encoders[name]['encoder']
        except:
            raise KeyError('Wrong encoder name `{}`, supported encoders: {}'.format(name, list(nnspt_encoders.keys())))

        params = nnspt_encoders[name]['params']
        params.update(depth=depth)

        encoder = type(**params)

        return encoder
