from nnspt.blocks.encoders.misc import __classinit

@__classinit
class Converter(object):
    """Class to provide layer converters for adaptation of CNN
    """
    @classmethod
    def _init__class(cls):
        """
            :return:
                instance of class, similar to singleton pattern
        """
        cls._registry = { }

        return cls()

    def convert(self, model):
        """
            :NOTE:
                conversion takes inplace

            :args:
                model (torch.nn.Module): PyTorch model
        """
        def __is_generator_empty(generator):
            try:
                next(generator)
                return False
            except StopIteration:
                return True

        stack = [model]

        while stack:
            node = stack[-1]

            stack.pop()

            for name, child in node.named_children():
                if not __is_generator_empty(child.children()):
                    stack.append(child)

                setattr(node, name, self(child))

    def __call__(self, layer):
        """
            :args:
                layer (torch.nn.Module): PyTorch layer to convert

            :return:
                converted layer
        """
        if type(layer) in self._registry:
            return self._registry[type(layer)](layer)
        else:
            return self._func_None(layer)

    @classmethod
    def _func_None(cls, layer):
        """
            :NOTE:
                identity convertation
        """
        return layer
