import torch
from .module import Module
from .. import functional as F


class Dropout(Module):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability *p* using samples from a bernoulli distribution.
    The elements to zero are randomized on every forward call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of *1/(1-p)* during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: `Any`. Input can be of any shape
        - Output: `Same`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16))
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) \
            + inplace_str + ')'


class Dropout2d(Module):
    r"""Randomly zeroes whole channels of the input tensor.
    The channels to zero-out are randomized on every forward call.

    *Usually the input comes from Conv2d modules.*

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then iid dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout2d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zeroed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> m = nn.Dropout2d(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16, 32, 32))
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       http://arxiv.org/abs/1411.4280
    """

    def __init__(self, p=0.5, inplace=False):
        super(Dropout2d, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout2d(input, self.p, self.training, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) \
            + inplace_str + ')'


class Dropout3d(Module):
    r"""Randomly zeroes whole channels of the input tensor.
    The channels to zero are randomized on every forward call.

    *Usually the input comes from Conv3d modules.*

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then iid dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout3d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zeroed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::

        >>> m = nn.Dropout3d(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16, 4, 32, 32))
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       http://arxiv.org/abs/1411.4280
    """

    def __init__(self, p=0.5, inplace=False):
        super(Dropout3d, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout3d(input, self.p, self.training, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) \
            + inplace_str + ')'


class AlphaDropout(Module):
    r"""Applies Alpha Dropout over the input.

    Alpha Dropout is a type of Dropout that maintains the self-normalizing
    property.
    For an input with zero mean and unit standard deviation, the output of
    Alpha Dropout maintains the original mean and standard deviation of the
    input.
    Alpha Dropout goes hand-in-hand with SELU activation function, which ensures
    that the outputs have zero mean and unit standard deviation.

    During training, it randomly masks some of the elements of the input
    tensor with probability *p* using samples from a bernoulli distribution.
    The elements to masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit standard deviation.

    During evaluation the module simply computes an identity function.

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .

    Args:
        p (float): probability of an element to be dropped. Default: 0.5

    Shape:
        - Input: `Any`. Input can be of any shape
        - Output: `Same`. Output is of the same shape as input

    Examples::

        >>> m = nn.AlphaDropout(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16))
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """

    def __init__(self, p=0.5):
        super(AlphaDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, input):
        return F.alpha_dropout(input, self.p, self.training)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'


class Zoneout(Module):
    r"""During training of an RNN, randomly swaps some of the elements of the
    input tensor with its values from a previous time-step with probability *p*
    using samples from a Bernoulli distribution. The elements to be swapped are
    randomized on every time-step by default, but a shared mask can be
    provided.

    Zoneout is a variant of dropout designed specifically for regularizing
    recurrent connections of LSTMs or GRUs. While dropout applies a zero mask
    to its inputs, zoneout applies an identity mask when incrementing a
    time-step.

    It has proven to be an effective technique for regularization of LSTMs
    and GRUs as, contrary to dropout, gradient information and state
    information are more readily propagated through time. For further
    information, consult the paper
    `Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activation`_ .

    Similarly to dropout, during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: None.
        inplace: If set to ``True``, will do this operation in-place.
        Default: ``False``
        mask: `ByteTensor`. A mask used to select elements to be swapped.
        The intended use case for this argument is sharing a zoneout mask
        across several time-steps.

    Shape:
        - Input: `Any`. A pair of tensors of the same shape
        - Output: `Same`. Output is of the same shape as input

    Examples::

        >>> zoneout = nn.Zoneout(p=0.15)
        >>> current_hidden_state = Variable(torch.Tensor([1, 2, 3]))
        >>> previous_hidden_state = Variable(torch.Tensor([4, 5, 6]))
        >>> output = zoneout(current_hidden_state, previous_hidden_state)

    Using a shared mask:
        >>> mask = torch.ByteTensor(1, 3).bernoulli()
        >>> zoneout = nn.Zoneout(mask=mask)
        >>> current_hidden_state = Variable(torch.Tensor([1, 2, 3]))
        >>> previous_hidden_state = Variable(torch.Tensor([4, 5, 6]))
        >>> output = zoneout(current_hidden_state, previous_hidden_state)

    Wrapping around a `GRUCell`:
        >>> rnn = nn.GRUCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> h = Variable(torch.randn(3, 20))
        >>> h_prev = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     h = zoneout(h, h_prev)
        ...     h, h_prev = rnn(input[i], h_prev), h
        ...     output.append(h)

    .. _Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activation:
    https://arxiv.org/abs/1606.01305
    """

    def __init__(self, p=None, inplace=False, mask=None):
        super(Zoneout, self).__init__()
        if p is None and mask is None:
            raise ValueError("Either p or mask must be provided")
        if p is not None and mask is not None:
            raise ValueError("Only one of p and mask can be provided")
        if p is not None and (p < 0 or p > 1):
            raise ValueError("zoneout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        if mask is not None and \
                not isinstance(mask, torch.ByteTensor) and \
                not isinstance(mask, torch.cuda.ByteTensor):
            raise ValueError("mask must be a ByteTensor")
        self.p = p
        self.inplace = inplace
        self.mask = mask

    def forward(self, previous_input, current_input):
        return F.zoneout(previous_input, current_input, self.p, self.mask,
                         self.training, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        if self.mask is not None:
            mask_str = 'mask=ByteTensor of size ' + \
                       'x'.join(str(size) for size in self.mask.size())
        else:
            mask_str = 'p=' + str(self.p)
        return self.__class__.__name__ + '(' + mask_str + inplace_str + ')'
