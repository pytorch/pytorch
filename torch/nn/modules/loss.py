from torch.autograd import Variable
import torch
from .module import Module
from .container import Sequential
from .activation import LogSoftmax
from .. import functional as F


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute value of the
    element-wise difference between input `x` and target `y`:

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|`

    `x` and `y` arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the constructor argument
    `size_average=False`.

    Args:
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to False, the losses are instead summed for
           each minibatch. Default: True

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = nn.L1Loss()
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> target = autograd.Variable(torch.randn(3, 5))
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def forward(self, input, target):
        _assert_no_grad(target)
        return F.l1_loss(input, target, size_average=self.size_average)


class NLLLoss(_WeightedLoss):
    r"""The negative log likelihood loss. It is useful to train a classification
    problem with n classes

    If provided, the optional argument `weights` should be a 1D Tensor assigning
    weight to each of the classes.

    This is particularly useful when you have an unbalanced training set.

    The input given through a forward call is expected to contain
    log-probabilities of each class: input has to be a 2D Tensor of size
    `(minibatch, n)`

    Obtaining log-probabilities in a neural network is easily achieved by
    adding a  `LogSoftmax`  layer in the last layer of your network.

    You may use `CrossEntropyLoss`  instead, if you prefer not to add an extra
    layer.

    The target that this loss expects is a class index
    `(0 to N-1, where N = number of classes)`

    The loss can be described as::

        loss(x, class) = -x[class]

    or in the case of the weights argument it is specified as follows::

        loss(x, class) = -weights[class] * x[class]

    or in the case of ignore_index::

        loss(x, class) = class != ignoreIndex ? -weights[class] * x[class] : 0

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
           class. If given, has to be a Tensor of size "nclasses"
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to False, the losses are instead summed for
           each minibatch. Default: True
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average
            is True, the loss is averaged over non-ignored targets.

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`
        - Target: :math:`(N)` where each value is `0 <= targets[i] <= C-1`

    Examples::

        >>> m = nn.LogSoftmax()
        >>> loss = nn.NLLLoss()
        >>> # input is of size nBatch x nClasses = 3 x 5
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> # each element in target has to have 0 <= value < nclasses
        >>> target = autograd.Variable(torch.LongTensor([1, 0, 4]))
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(NLLLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.nll_loss(input, target, self.weight, self.size_average,
                          self.ignore_index)


class NLLLoss2d(NLLLoss):
    r"""This is negative log likehood loss, but for image inputs. It computes
    NLL loss per-pixel.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a 1D Tensor having as many elements,
            as there are classes.
        size_average: By default, the losses are averaged over observations
            for each minibatch. However, if the field size_average is set to
            False, the losses are instead summed for each minibatch.
            Default: True

    Shape:
        - Input: :math:`(N, C, H, W)` where `C = number of classes`
        - Target: :math:`(N, H, W)` where each value is `0 <= targets[i] <= C-1`

    Examples::

        >>> m = nn.Conv2d(16, 32, (3, 3)).float()
        >>> loss = nn.NLLLoss2d()
        >>> # input is of size nBatch x nClasses x height x width
        >>> input = autograd.Variable(torch.randn(3, 16, 10, 10))
        >>> # each element in target has to have 0 <= value < nclasses
        >>> target = autograd.Variable(torch.LongTensor(3, 8, 8).random_(0, 4))
        >>> output = loss(m(input), target)
        >>> output.backward()
    """
    pass


class PoissonNLLLoss(_Loss):
    r"""Negative log likelihood loss with Poisson distribution of target.

    The loss can be described as::

        target ~ Pois(input)
        loss(input, target) = input - target * log(input) + log(target!)

    The last term can be omitted or approximised with Stirling formula. The
    approximation is used for target values more than 1. For targets less or
    equal to 1 zeros are added to the loss.

    Args:
        log_input (bool, optional): if True the loss is computed as
            `exp(input) - target * input`, if False the loss is
            `input - target * log(input)`.
        full (bool, optional): whether to compute full loss, i. e. to add the
            Stirling approximation term
            `target * log(target) - target + 0.5 * log(2 * pi * target)`.
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field size_average
            is set to False, the losses are instead summed for each minibatch.

    Examples::

        >>> loss = nn.PoissonNLLLoss()
        >>> log_input = autograd.Variable(torch.randn(5, 2), requires_grad=True)
        >>> target = autograd.Variable(torch.randn(5, 2))
        >>> output = loss(log_input, target)
        >>> output.backward()
    """
    def __init__(self, log_input=True, full=False, size_average=True):
        super(PoissonNLLLoss, self).__init__()
        self.log_input = log_input
        self.full = full
        self.size_average = size_average

    def forward(self, log_input, target):
        _assert_no_grad(target)
        return F.poisson_nll_loss(log_input, target, self.log_input, self.full, self.size_average)


class KLDivLoss(_Loss):
    r"""The `Kullback-Leibler divergence`_ Loss

    KL divergence is a useful distance measure for continuous distributions
    and is often useful when performing direct regression over the space of
    (discretely sampled) continuous output distributions.

    As with `NLLLoss`, the `input` given is expected to contain
    *log-probabilities*, however unlike `ClassNLLLoss`, `input` is not
    restricted to a 2D Tensor, because the criterion is applied element-wise.

    This criterion expects a `target` `Tensor` of the same size as the
    `input` `Tensor`.

    The loss can be described as:

    .. math:: loss(x, target) = 1/n \sum(target_i * (log(target_i) - x_i))

    By default, the losses are averaged for each minibatch over observations
    **as well as** over dimensions. However, if the field
    `size_average` is set to `False`, the losses are instead summed.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
    """
    def forward(self, input, target):
        _assert_no_grad(target)
        return F.kl_div(input, target, size_average=self.size_average)


class MSELoss(_Loss):
    r"""Creates a criterion that measures the mean squared error between
    `n` elements in the input `x` and target `y`:

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|^2`

    `x` and `y` arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the internal variable
    `size_average` to `False`.

    Args:
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to False, the losses are instead summed for
           each minibatch. Default: True

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = nn.MSELoss()
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> target = autograd.Variable(torch.randn(3, 5))
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def forward(self, input, target):
        _assert_no_grad(target)
        return F.mse_loss(input, target, size_average=self.size_average)


class BCELoss(_WeightedLoss):
    r"""Creates a criterion that measures the Binary Cross Entropy
    between the target and the output:

    .. math:: loss(o, t) = - 1/n \sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

    or in the case of the weights argument being specified:

    .. math:: loss(o, t) = - 1/n \sum_i weights[i] * (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.

    By default, the losses are averaged for each minibatch over observations
    *as well as* over dimensions. However, if the field `size_average` is set
    to `False`, the losses are instead summed.

    """
    def forward(self, input, target):
        _assert_no_grad(target)
        return F.binary_cross_entropy(input, target, weight=self.weight,
                                      size_average=self.size_average)


class BCEWithLogitsLoss(Module):
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    This Binary Cross Entropy between the target and the output logits
    (no sigmoid applied) is:

    .. math:: loss(o, t) = - 1/n \sum_i (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))

    or in the case of the weights argument being specified:

    .. math:: loss(o, t) = - 1/n \sum_i weights[i] * (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.

    By default, the losses are averaged for each minibatch over observations
    *as well as* over dimensions. However, if the field `size_average` is set
    to `False`, the losses are instead summed.

    """
    def __init__(self, weight=None, size_average=True):
        super(BCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        if self.weight is not None:
            return F.binary_cross_entropy_with_logits(input, target, Variable(self.weight), self.size_average)
        else:
            return F.binary_cross_entropy_with_logits(input, target, size_average=self.size_average)


class HingeEmbeddingLoss(_Loss):
    r"""Measures the loss given an input `x` which is a 2D mini-batch tensor
    and a labels `y`, a 1D tensor containg values (`1` or `-1`).
    This is usually used for measuring whether two inputs are similar or
    dissimilar, e.g. using the L1 pairwise distance, and is typically used
    for learning nonlinear embeddings or semi-supervised learning::

                         { x_i,                  if y_i ==  1
        loss(x, y) = 1/n {
                         { max(0, margin - x_i), if y_i == -1

    `x` and `y` arbitrary shapes with a total of `n` elements each
    the sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the internal
    variable `size_average=False`.

    The `margin` has a default value of `1`, or can be set in the constructor.
    """

    def __init__(self, margin=1.0, size_average=True):
        super(HingeEmbeddingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input, target):
        return F.hinge_embedding_loss(input, target, self.margin, self.size_average)


class MultiLabelMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a multi-class multi-classification
    hinge loss (margin-based loss) between input `x`  (a 2D mini-batch `Tensor`)
    and output `y` (which is a 2D `Tensor` of target class indices).
    For each sample in the mini-batch::

        loss(x, y) = sum_ij(max(0, 1 - (x[y[j]] - x[i]))) / x.size(0)

    where `i == 0` to `x.size(0)`, `j == 0` to `y.size(0)`,
    `y[j] != 0`, and `i != y[j]` for all `i` and `j`.

    `y` and `x` must have the same size.

    The criterion only considers the first non zero `y[j]` targets.

    This allows for different samples to have variable amounts of target classes
    """
    def forward(self, input, target):
        _assert_no_grad(target)
        return F.multilabel_margin_loss(input, target, size_average=self.size_average)


class SmoothL1Loss(_Loss):
    r"""Creates a criterion that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.
    It is less sensitive to outliers than the `MSELoss` and in some cases
    prevents exploding gradients (e.g. see "Fast R-CNN" paper by Ross Girshick).
    Also known as the Huber loss::

                              { 0.5 * (x_i - y_i)^2, if |x_i - y_i| < 1
        loss(x, y) = 1/n \sum {
                              { |x_i - y_i| - 0.5,   otherwise

    `x` and `y` arbitrary shapes with a total of `n` elements each
    the sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the internal variable
    `size_average` to `False`
    """
    def forward(self, input, target):
        _assert_no_grad(target)
        return F.smooth_l1_loss(input, target, size_average=self.size_average)


class SoftMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a two-class classification
    logistic loss between input `x` (a 2D mini-batch Tensor) and
    target `y` (which is a tensor containing either `1` or `-1`).

    ::

        loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x.nelement()

    The normalization by the number of elements in the input can be disabled by
    setting `self.size_average` to `False`.
    """
    def forward(self, input, target):
        _assert_no_grad(target)
        return F.soft_margin_loss(input, target, size_average=self.size_average)


class CrossEntropyLoss(_WeightedLoss):
    r"""This criterion combines `LogSoftMax` and `NLLLoss` in one single class.

    It is useful when training a classification problem with `n` classes.
    If provided, the optional argument `weights` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain scores for each class.

    `input` has to be a 2D `Tensor` of size `(minibatch,n)`.

    This criterion expects a class index (0 to nClasses-1) as the
    `target` for each value of a 1D tensor of size `minibatch`

    The loss can be described as::

        loss(x, class) = -log(exp(x[class]) / (\sum_j exp(x[j])))
                       = -x[class] + log(\sum_j exp(x[j]))

    or in the case of the `weights` argument being specified::

        loss(x, class) = weights[class] * (-x[class] + log(\sum_j exp(x[j])))

    The losses are averaged across observations for each minibatch.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
           If given, has to be a Tensor of size "nclasses"
        size_average (bool, optional): By default, the losses are averaged over observations for each minibatch.
           However, if the field size_average is set to False, the losses are
           instead summed for each minibatch.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average is
            True, the loss is averaged over non-ignored targets.

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`
        - Target: :math:`(N)` where each value is `0 <= targets[i] <= C-1`

    Examples::

        >>> loss = nn.CrossEntropyLoss()
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> target = autograd.Variable(torch.LongTensor(3).random_(5))
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.cross_entropy(input, target, self.weight, self.size_average,
                               self.ignore_index)


class MultiLabelSoftMarginLoss(_WeightedLoss):
    r"""Creates a criterion that optimizes a multi-label one-versus-all
    loss based on max-entropy, between input `x`  (a 2D mini-batch `Tensor`) and
    target `y` (a binary 2D `Tensor`). For each sample in the minibatch::

       loss(x, y) = - sum_i (y[i] * log( 1 / (1 + exp(-x[i])) )
                         + ( (1-y[i]) * log(exp(-x[i]) / (1 + exp(-x[i])) ) )

    where `i == 0` to `x.nElement()-1`, `y[i]  in {0,1}`.
    `y` and `x` must have the same size.
    """

    def forward(self, input, target):
        return F.multilabel_soft_margin_loss(input, target, self.weight, self.size_average)


class CosineEmbeddingLoss(Module):
    r"""Creates a criterion that measures the loss given  an input tensors
    x1, x2 and a `Tensor` label `y` with values 1 or -1.
    This is used for measuring whether two inputs are similar or dissimilar,
    using the cosine distance, and is typically used for learning nonlinear
    embeddings or semi-supervised learning.

    `margin` should be a number from `-1` to `1`, `0` to `0.5` is suggested.
    If `margin` is missing, the default value is `0`.

    The loss function for each sample is::

                     { 1 - cos(x1, x2),              if y ==  1
        loss(x, y) = {
                     { max(0, cos(x1, x2) - margin), if y == -1

    If the internal variable `size_average` is equal to `True`,
    the loss function averages the loss over the batch samples;
    if `size_average` is `False`, then the loss function sums over the
    batch samples. By default, `size_average = True`.
    """

    def __init__(self, margin=0, size_average=True):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, target):
        return F.cosine_embedding_loss(input1, input2, target, self.margin, self.size_average)


class MarginRankingLoss(Module):
    r"""Creates a criterion that measures the loss given
    inputs `x1`, `x2`, two 1D mini-batch `Tensor`s,
    and a label 1D mini-batch tensor `y` with values (`1` or `-1`).

    If `y == 1` then it assumed the first input should be ranked higher
    (have a larger value) than the second input, and vice-versa for `y == -1`.

    The loss function for each sample in the mini-batch is::

        loss(x, y) = max(0, -y * (x1 - x2) + margin)

    if the internal variable `size_average = True`,
    the loss function averages the loss over the batch samples;
    if `size_average = False`, then the loss function sums over the batch
    samples.
    By default, `size_average` equals to `True`.
    """

    def __init__(self, margin=0, size_average=True):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, target):
        return F.margin_ranking_loss(input1, input2, target, self.margin, self.size_average)


class MultiMarginLoss(Module):
    r"""Creates a criterion that optimizes a multi-class classification hinge
    loss (margin-based loss) between input `x` (a 2D mini-batch `Tensor`) and
    output `y` (which is a 1D tensor of target class indices,
    `0` <= `y` <= `x.size(1)`):

    For each mini-batch sample::

        loss(x, y) = sum_i(max(0, (margin - x[y] + x[i]))^p) / x.size(0)
                     where `i == 0` to `x.size(0)` and `i != y`.

    Optionally, you can give non-equal weighting on the classes by passing
    a 1D `weights` tensor into the constructor.

    The loss function then becomes:

        loss(x, y) = sum_i(max(0, w[y] * (margin - x[y] - x[i]))^p) / x.size(0)

    By default, the losses are averaged over observations for each minibatch.
    However, if the field `size_average` is set to `False`,
    the losses are instead summed.
    """

    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(MultiMarginLoss, self).__init__()
        if p != 1 and p != 2:
            raise ValueError("only p == 1 and p == 2 supported")
        assert weight is None or weight.dim() == 1
        self.p = p
        self.margin = margin
        self.size_average = size_average
        self.weight = weight

    def forward(self, input, target):
        return F.multi_margin_loss(input, target, self.p, self.margin,
                                   self.weight, self.size_average)


class TripletMarginLoss(Module):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3 and a margin with a value greater than 0.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n`: anchor, positive examples and negative
    example respectively. The shape of all input variables should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    .. math::
        L(a, p, n) = \frac{1}{N} \left( \sum_{i=1}^N \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\} \right)

    where :math:`d(x_i, y_i) = \| {\bf x}_i - {\bf y}_i \|_2^2`.

    Args:
        anchor: anchor input tensor
        positive: positive input tensor
        negative: negative input tensor
        p: the norm degree. Default: 2

    Shape:
        - Input: :math:`(N, D)` where `D = vector dimension`
        - Output: :math:`(N, 1)`

    >>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    >>> input1 = autograd.Variable(torch.randn(100, 128))
    >>> input2 = autograd.Variable(torch.randn(100, 128))
    >>> input3 = autograd.Variable(torch.randn(100, 128))
    >>> output = triplet_loss(input1, input2, input3)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """

    def __init__(self, margin=1.0, p=2, eps=1e-6, swap=False):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, self.margin,
                                     self.p, self.eps, self.swap)

# TODO: L1HingeEmbeddingCriterion
# TODO: MSECriterion weight
# TODO: ClassSimplexCriterion
