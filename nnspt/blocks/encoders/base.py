class EncoderBase(object):
    """Base class of all encoders in nnspt
    """
    @property
    def out_channels(self):
        """
            :return:
                output: list of int
                    number of channels in forward path in encoder
        """
        return self.out_channels_[:self.depth + 1]
