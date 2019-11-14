from . import Identity, ModuleDict, Sequential


class ResidualBlock(Sequential):
    r"""A residual block with an identity shortcut connection.

    As in :class:`~torch.nn.Sequential`, modules will be added to it in the
    order they are passed in the constructor, and an :class:`OrderedDict` can be
    passed instead. The final module's output will be added to the original
    input and returned. The input and output must be :ref:`broadcastable
    <broadcasting-semantics>`.

    Here is an example MNIST classifier::

        model = nn.Sequential(
            nn.Conv2d(1, 10, 1),
            nn.ResidualBlock(
                nn.ReLU(),
                nn.Conv2d(10, 10, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(10, 10, 3, padding=1),
            ),
            nn.MaxPool2d(2),
            nn.ResidualBlock(
                nn.ReLU(),
                nn.Conv2d(10, 10, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(10, 10, 3, padding=1),
            ),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(7*7*10, 10),
            nn.LogSoftmax(dim=-1),
        )

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

    As in :class:`~torch.nn.Sequential`, modules will be added to the 'main'
    branch in the order they are passed in the constructor, and an
    :class:`OrderedDict` can be passed instead. The :attr:`shortcut` keyword
    argument specifies a module that performs the mapping for the shortcut
    connection.  The output of the 'main' branch will be added to the output of
    the 'shortcut' branch and returned. They must be :ref:`broadcastable
    <broadcasting-semantics>`.

    This module is useful where the 'main' branch has an output shape that is
    different from its input shape, so :class:`~torch.nn.ResidualBlock` cannot
    be used. The shortcut mapping may be used to adjust the shape of the input
    to match. Here is an example MNIST classifier::

        model = nn.Sequential(
            nn.ResidualBlockWithShortcut(
                nn.Conv2d(1, 10, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(10, 10, 3, stride=2, padding=1),
                shortcut=nn.Conv2d(1, 10, 1, stride=2, bias=False),
            ),
            nn.ResidualBlockWithShortcut(
                nn.ReLU(),
                nn.Conv2d(10, 20, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(20, 20, 3, stride=2, padding=1),
                shortcut=nn.Conv2d(10, 20, 1, stride=2, bias=False),
            ),
            nn.Flatten(),
            nn.Linear(7*7*20, 10),
            nn.LogSoftmax(dim=-1),
        )

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
