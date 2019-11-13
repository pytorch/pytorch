from . import Identity, ModuleDict, Sequential


class ResidualBlock(Sequential):
    r"""A residual block with an identity shortcut connection.

    Modules will be added to it in the order they are passed in the constructor.
    Each module's output will be fed to the next module as input. Alternately,
    an OrderedDict can be passed in. The final module's output will be added to
    the original input and returned. The input and output must be the same
    shape.

    See: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual
    Learning for Image Recognition" (https://arxiv.org/abs/1512.03385), and
    "Identity Mappings in Deep Residual Networks"
    (https://arxiv.org/abs/1603.05027).
    """

    def forward(self, input):
        output = input.clone()
        for module in self:
            output = module(output)
        return input + output


class ResidualBlockWithShortcut(ModuleDict):
    r"""A residual block with a non-identity shortcut connection.

    Modules will be added to the 'main' branch in the order they are passed in
    the constructor. Each module's output will be fed to the next module as
    input. Alternately, an OrderedDict can be passed in. The final module's
    output will be added to the output of the 'shortcut' branch. The two
    branches' outputs must be the same shape.

    See: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual
    Learning for Image Recognition" (https://arxiv.org/abs/1512.03385), and
    "Identity Mappings in Deep Residual Networks"
    (https://arxiv.org/abs/1603.05027).
    """

    def __init__(self, *args, shortcut=Identity()):
        super(ResidualBlockWithShortcut, self).__init__()
        self['main'] = Sequential(*args)
        self['shortcut'] = shortcut

    def forward(self, input):
        output_main = self['main'](input.clone())
        output_shortcut = self['shortcut'](input)
        return output_main + output_shortcut
