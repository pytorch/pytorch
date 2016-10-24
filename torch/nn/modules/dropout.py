from .module import Module

class Dropout(Module):
    """Randomly zeroes some of the elements of the input tensor.
    The elements to zero are randomized on every forward call.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to True, will do this operation in-place. Default: false
    Input Shape: Any : Input can be of any shape
    Output Shape:Same  : Output is of the same shape as input
    Examples:
        >>> m = nn.Dropout(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16))
        >>> output = m(input)
    """
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return self._backend.Dropout(self.p, self.train, self.inplace)(input)


class Dropout2d(Module):
    """Randomly zeroes whole channels of the input tensor.
    The input is 4D (batch x channels, height, width) and each channel
    is of size (1, height, width).
    The channels to zero are randomized on every forward call.
    Usually the input comes from Conv2d modules.

    As described in the paper &quot;Efficient Object Localization Using Convolutional
    Networks&quot; (http:arxiv.org/abs/1411.4280), if adjacent pixels within
    feature maps are strongly correlated (as is normally the case in early
    convolution layers) then iid dropout will not regularize the activations
    and will otherwise just result in an effective learning rate decrease.
    In this case, nn.Dropout2d will help promote independence between
    feature maps and should be used instead.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to True, will do this operation in-place. Default: false
    Input Shape: [*, *, *, *] : Input can be of any sizes of 4D shape
    Output Shape:Same  : Output is of the same shape as input
    Examples:
        >>> m = nn.Dropout2d(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16, 32, 32))
        >>> output = m(input)
    """
    def __init__(self, p=0.5, inplace=False):
        super(Dropout2d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return self._backend.Dropout2d(self.p, self.train, self.inplace)(input)


class Dropout3d(Module):
    """Randomly zeroes whole channels of the input tensor.
    The input is 5D (batch x channels, depth, height, width) and each channel
    is of size (1, depth, height, width).
    The channels to zero are randomized on every forward call.
    Usually the input comes from Conv3d modules.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to True, will do this operation in-place. Default: false
    Input Shape: [*, *, *, *, *] : Input can be of any sizes of 5D shape
    Output Shape:Same  : Output is of the same shape as input
    Examples:
        >>> m = nn.Dropout3d(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16, 4, 32, 32))
        >>> output = m(input)
    """
    def __init__(self, p=0.5, inplace=False):
        super(Dropout3d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return self._backend.Dropout3d(self.p, self.train, self.inplace)(input)

