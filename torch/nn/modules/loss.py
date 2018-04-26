import warnings

import torch
from .module import Module
from .container import Sequential
from .activation import LogSoftmax
from .. import functional as F


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class _Loss(Module):
    def __init__(self, size_average=True, reduce=True):
        super(_Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True, reduce=True):
        super(_WeightedLoss, self).__init__(size_average, reduce)
        self.register_buffer('weight', weight)


class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute value of the
    element-wise difference between input `x` and target `y`:

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|,

    where :math:`N` is the batch size. If reduce is ``True``, then:

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}

    `x` and `y` arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the constructor argument
    `size_average=False`.

    Args:
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to ``False``, the losses are instead summed for
           each minibatch. Ignored when reduce is ``False``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed
           for each minibatch. When reduce is ``False``, the loss function returns
           a loss per input/target element instead and ignores size_average.
           Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If reduce is ``False``, then
          :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = nn.L1Loss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, size_average=True, reduce=True):
        super(L1Loss, self).__init__(size_average, reduce)

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.l1_loss(input, target, size_average=self.size_average,
                         reduce=self.reduce)


class NLLLoss(_WeightedLoss):
    r"""The negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    If provided, the optional argument `weight` should be a 1D Tensor assigning
    weight to each of the classes. This is particularly useful when you have an
    unbalanced training set.

    The input given through a forward call is expected to contain
    log-probabilities of each class. `input` has to be a Tensor of size either
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 2` for the `K`-dimensional case (described later).

    Obtaining log-probabilities in a neural network is easily achieved by
    adding a  `LogSoftmax`  layer in the last layer of your network.
    You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
    layer.

    The target that this loss expects is a class index
    `(0 to C-1, where C = number of classes)`

    If :attr:`reduce` is ``False``, the loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore_index}\},

    where :math:`N` is the batch size. If :attr:`reduce` is ``True`` (default),
    then

    .. math::
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, & \text{if}\;
            \text{size_average} = \text{True},\\
            \sum_{n=1}^N l_n,  & \text{if}\;
            \text{size_average} = \text{False}.
        \end{cases}

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 2`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below). In the case of images, it computes NLL loss per-pixel.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
           class. If given, it has to be a Tensor of size `C`. Otherwise, it is
           treated as if having all ones.
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch with weights set by
           :attr:`weight`. However, if the field :attr:`size_average` is set to
           ``False``, the losses are instead summed for each minibatch. Ignored
           when :attr:`reduce` is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When
            :attr:`size_average` is ``True``, the loss is averaged over
            non-ignored targets.
        reduce (bool, optional): By default, the losses are averaged or summed
            for each minibatch. When :attr:`reduce` is ``False``, the loss
            function returns a loss per batch instead and
            ignores :attr:`size_average`. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
            :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 2`
            in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
            :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 2` in the case of
            K-dimensional loss.
        - Output: scalar. If reduce is ``False``, then the same size
            as the target: :math:`(N)`, or
            :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 2` in the case
            of K-dimensional loss.

    Examples::

        >>> m = nn.LogSoftmax()
        >>> loss = nn.NLLLoss()
        >>> # input is of size N x C = 3 x 5
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> # each element in target has to have 0 <= value < C
        >>> target = torch.tensor([1, 0, 4])
        >>> output = loss(m(input), target)
        >>> output.backward()
        >>>
        >>>
        >>> # 2D loss example (used, for example, with image inputs)
        >>> N, C = 5, 4
        >>> loss = nn.NLLLoss()
        >>> # input is of size N x C x height x width
        >>> data = torch.randn(N, 16, 10, 10)
        >>> m = nn.Conv2d(16, C, (3, 3))
        >>> # each element in target has to have 0 <= value < C
        >>> target = torch.tensor(N, 8, 8).random_(0, C)
        >>> output = loss(m(data), target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(NLLLoss, self).__init__(weight, size_average, reduce)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.nll_loss(input, target, self.weight, self.size_average,
                          self.ignore_index, self.reduce)


class NLLLoss2d(NLLLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        warnings.warn("NLLLoss2d has been deprecated. "
                      "Please use NLLLoss instead as a drop-in replacement and see "
                      "http://pytorch.org/docs/master/nn.html#torch.nn.NLLLoss for more details.")
        super(NLLLoss2d, self).__init__(weight, size_average, ignore_index, reduce)


class PoissonNLLLoss(_Loss):
    r"""Negative log likelihood loss with Poisson distribution of target.

    The loss can be described as:

    .. math::
        \text{target} \sim \mathrm{Poisson}(\text{input})

        \text{loss}(\text{input}, \text{target}) = \text{input} - \text{target} * \log(\text{input})
                                    + \log(\text{target!})

    The last term can be omitted or approximated with Stirling formula. The
    approximation is used for target values more than 1. For targets less or
    equal to 1 zeros are added to the loss.

    Args:
        log_input (bool, optional): if ``True`` the loss is computed as
            :math:`\exp(\text{input}) - \text{target}*\text{input}`, if ``False`` the loss is
            :math:`\text{input} - \text{target}*\log(\text{input}+\text{eps})`.
        full (bool, optional): whether to compute full loss, i. e. to add the
            Stirling approximation term

            .. math::
                \text{target}*\log(\text{target}) - \text{target} + 0.5 * \log(2\pi\text{target}).
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field `size_average`
            is set to ``False``, the losses are instead summed for each minibatch.
        eps (float, optional): Small value to avoid evaluation of :math:`\log(0)` when
            :attr:`log_input == False`. Default: 1e-8
        reduce (bool, optional): By default, the losses are averaged
            over observations for each minibatch, or summed, depending on
            size_average. When reduce is ``False``, returns a loss per input/target
            element instead and ignores `size_average`. Default: ``True``

    Examples::

        >>> loss = nn.PoissonNLLLoss()
        >>> log_input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> output = loss(log_input, target)
        >>> output.backward()
    """
    def __init__(self, log_input=True, full=False, size_average=True, eps=1e-8, reduce=True):
        super(PoissonNLLLoss, self).__init__(size_average, reduce)
        self.log_input = log_input
        self.full = full
        self.eps = eps

    def forward(self, log_input, target):
        _assert_no_grad(target)
        return F.poisson_nll_loss(log_input, target, self.log_input, self.full,
                                  self.size_average, self.eps, self.reduce)


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

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = y_n \odot \left( \log y_n - x_n \right),

    where :math:`N` is the batch size. If reduce is ``True``, then:

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}

    By default, the losses are averaged for each minibatch over observations
    **as well as** over dimensions. However, if the field
    `size_average` is set to ``False``, the losses are instead summed.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

    Args:
        size_average (bool, optional: By default, the losses are averaged
            for each minibatch over observations **as well as** over
            dimensions. However, if ``False`` the losses are instead summed.
        reduce (bool, optional): By default, the losses are averaged
            over observations for each minibatch, or summed, depending on
            size_average. When reduce is ``False``, returns a loss per input/target
            element instead and ignores size_average. Default: ``True``

    Shape:
        - input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - target: :math:`(N, *)`, same shape as the input
        - output: scalar. If `reduce` is ``True``, then :math:`(N, *)`,
            same shape as the input

    """
    def __init__(self, size_average=True, reduce=True):
        super(KLDivLoss, self).__init__(size_average, reduce)

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.kl_div(input, target, size_average=self.size_average, reduce=self.reduce)


class MSELoss(_Loss):
    r"""Creates a criterion that measures the mean squared error between
    `n` elements in the input `x` and target `y`.

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If reduce is ``True``, then:

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets :attr:`size_average` to ``False``.

    To get a batch of losses, a loss per batch element, set `reduce` to
    ``False``. These losses are not averaged and are not affected by
    `size_average`.

    Args:
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to ``False``, the losses are instead summed for
           each minibatch. Only applies when reduce is ``True``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged
           over observations for each minibatch, or summed, depending on
           size_average. When reduce is ``False``, returns a loss per input/target
           element instead and ignores size_average. Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = nn.MSELoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, size_average=True, reduce=True):
        super(MSELoss, self).__init__(size_average, reduce)

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.mse_loss(input, target, size_average=self.size_average, reduce=self.reduce)


class BCELoss(_WeightedLoss):
    r"""Creates a criterion that measures the Binary Cross Entropy
    between the target and the output:

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    where :math:`N` is the batch size. If reduce is ``True``, then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `y` should be numbers
    between 0 and 1.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        size_average (bool, optional): By default, the losses are averaged
            over observations for each minibatch. However, if the field
            size_average is set to ``False``, the losses are instead summed for
            each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on size_average. When reduce
            is False, returns a loss per input/target element instead and ignores
            size_average. Default: True

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If `reduce` is False, then `(N, *)`, same shape as
          input.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """
    def __init__(self, weight=None, size_average=True, reduce=True):
        super(BCELoss, self).__init__(weight, size_average, reduce)

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.binary_cross_entropy(input, target, weight=self.weight,
                                      size_average=self.size_average,
                                      reduce=self.reduce)


class BCEWithLogitsLoss(_Loss):
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ t_n \cdot \log \sigma(x_n)
        + (1 - t_n) \cdot \log (1 - \sigma(x_n)) \right],

    where :math:`N` is the batch size. If reduce is ``True``, then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        size_average (bool, optional): By default, the losses are averaged
            over observations for each minibatch. However, if the field
            size_average is set to ``False``, the losses are instead summed for
            each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on size_average. When reduce
            is False, returns a loss per input/target element instead and ignores
            size_average. Default: True

     Shape:
         - Input: :math:`(N, *)` where `*` means, any number of additional
           dimensions
         - Target: :math:`(N, *)`, same shape as the input

     Examples::

        >>> loss = nn.BCEWithLogitsLoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, weight=None, size_average=True, reduce=True):
        super(BCEWithLogitsLoss, self).__init__(size_average, reduce)
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        if self.weight is not None:
            return F.binary_cross_entropy_with_logits(input, target,
                                                      self.weight,
                                                      self.size_average,
                                                      reduce=self.reduce)
        else:
            return F.binary_cross_entropy_with_logits(input, target,
                                                      size_average=self.size_average,
                                                      reduce=self.reduce)


class HingeEmbeddingLoss(_Loss):
    r"""Measures the loss given an input tensor `x` and a labels tensor `y`
    containing values (`1` or `-1`).
    This is usually used for measuring whether two inputs are similar or
    dissimilar, e.g. using the L1 pairwise distance as `x`, and is typically
    used for learning nonlinear embeddings or semi-supervised learning::

    The loss function for :math:`n`-th sample in the mini-batch is:

    .. math::
        l_n = \begin{cases}
            x_n, & \text{if}\; y_n = 1,\\
            \max \{0, \Delta - x_n\}, & \text{if}\; y_n = -1,
        \end{cases}

    and the total loss functions is

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}

    where :math:`L = \{l_1,\dots,l_N\}^\top`.

    Args:
        margin (float, optional): Has a default value of `1`.
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch.
            Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on :attr:`size_average`. When
            :attr:`reduce` is ``False``, returns a loss per batch element instead and
            ignores :attr:`size_average`. Default: ``True``

    Shape:
        - Input: Tensor of arbitrary shape. The sum operation operates over all the elements.
        - Target: Same shape as input.
        - Output: scalar. If reduce is ``False``, then same shape as the input
    """

    def __init__(self, margin=1.0, size_average=True, reduce=True):
        super(HingeEmbeddingLoss, self).__init__(size_average, reduce)
        self.margin = margin

    def forward(self, input, target):
        return F.hinge_embedding_loss(input, target, self.margin, self.size_average,
                                      self.reduce)


class MultiLabelMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a multi-class multi-classification
    hinge loss (margin-based loss) between input `x`  (a 2D mini-batch `Tensor`)
    and output `y` (which is a 2D `Tensor` of target class indices).
    For each sample in the mini-batch:

    .. math::
        \text{loss}(x, y) = \sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\text{x.size}(0)}

    where `i == 0` to `x.size(0)`, `j == 0` to `y.size(0)`,
    :math:`y[j] \geq 0`, and :math:`i \neq y[j]` for all `i` and `j`.

    `y` and `x` must have the same size.

    The criterion only considers a contiguous block of non-negative targets that
    starts at the front.

    This allows for different samples to have variable amounts of target classes

    Args:
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch.
            Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on :attr:`size_average`. When
            :attr:`reduce` is ``False``, returns a loss per batch element instead and
            ignores :attr:`size_average`. Default: ``True``

    Shape:
        - Input: :math:`(C)` or :math:`(N, C)` where `N` is the batch size and `C`
          is the number of classes.
        - Target: :math:`(C)` or :math:`(N, C)`, same shape as the input.
        - Output: scalar. If `reduce` is False, then `(N)`.
    """
    def __init__(self, size_average=True, reduce=True):
        super(MultiLabelMarginLoss, self).__init__(size_average, reduce)

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.multilabel_margin_loss(input, target, size_average=self.size_average,
                                        reduce=self.reduce)


class SmoothL1Loss(_Loss):
    r"""Creates a criterion that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.
    It is less sensitive to outliers than the `MSELoss` and in some cases
    prevents exploding gradients (e.g. see "Fast R-CNN" paper by Ross Girshick).
    Also known as the Huber loss:

    .. math::
        \text{loss}(x, y) = \frac{1}{n} \sum_{i} z_{i}

    where :math:`z_{i}` is given by:

    .. math::
        z_{i} =
        \begin{cases}
        0.5 (x_i - y_i)^2, & \text{if } |x_i - y_i| < 1 \\
        |x_i - y_i| - 0.5, & \text{otherwise }
        \end{cases}

    `x` and `y` arbitrary shapes with a total of `n` elements each
    the sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets :attr:`size_average` to ``False``

    Args:
        size_average (bool, optional): By default, the losses are averaged
           over all elements. However, if the field size_average is set to ``False``,
           the losses are instead summed. Ignored when reduce is ``False``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed
           over elements. When reduce is ``False``, the loss function returns
           a loss per input/target element instead and ignores size_average.
           Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If reduce is ``False``, then
          :math:`(N, *)`, same shape as the input

    """
    def __init__(self, size_average=True, reduce=True):
        super(SmoothL1Loss, self).__init__(size_average, reduce)

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.smooth_l1_loss(input, target, size_average=self.size_average,
                                reduce=self.reduce)


class SoftMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a two-class classification
    logistic loss between input tensor `x` and target tensor `y` (containing 1 or
    -1).

    .. math::
        \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}

    Args:
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch.
            Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on :attr:`size_average`. When
            :attr:`reduce` is ``False``, returns a loss per batch element instead and
            ignores :attr:`size_average`. Default: ``True``

    Shape:
        - Input: Tensor of arbitrary shape.
        - Target: Same shape as input.
        - Output: scalar. If reduce is ``False``, then same shape as the input

    """
    def __init__(self, size_average=True, reduce=True):
        super(SoftMarginLoss, self).__init__(size_average, reduce)

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.soft_margin_loss(input, target, size_average=self.size_average,
                                  reduce=self.reduce)


class CrossEntropyLoss(_WeightedLoss):
    r"""This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 2` for the `K`-dimensional case (described later).

    This criterion expects a class index (0 to `C-1`) as the
    `target` for each value of a 1D tensor of size `minibatch`

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    or in the case of the `weight` argument being specified:

    .. math::
        \text{loss}(x, class) = weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)

    The losses are averaged across observations for each minibatch.

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 2`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).


    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
           If given, has to be a Tensor of size `C`
        size_average (bool, optional): By default, the losses are averaged over observations for each minibatch.
           However, if the field `size_average` is set to ``False``, the losses are
           instead summed for each minibatch. Ignored if reduce is ``False``.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When `size_average` is
            ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on `size_average`. When reduce
            is ``False``, returns a loss per batch instead and ignores
            size_average. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
            :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 2`
            in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
            :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 2` in the case of
            K-dimensional loss.
        - Output: scalar. If reduce is ``False``, then the same size
            as the target: :math:`(N)`, or
            :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 2` in the case
            of K-dimensional loss.

    Examples::

        >>> loss = nn.CrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.cross_entropy(input, target, self.weight, self.size_average,
                               self.ignore_index, self.reduce)


class MultiLabelSoftMarginLoss(_WeightedLoss):
    r"""Creates a criterion that optimizes a multi-label one-versus-all
    loss based on max-entropy, between input `x` and target `y` of size `(N, C)`.
    For each sample in the minibatch:

    .. math::
        loss(x, y) = - \sum_i y[i] * \log((1 + \exp(-x[i]))^{-1})
                         + (1-y[i]) * \log\left(\frac{\exp(-x[i])}{(1 + \exp(-x[i]))}\right)

    where `i == 0` to `x.nElement()-1`, `y[i]  in {0,1}`.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
           class. If given, it has to be a Tensor of size `C`. Otherwise, it is
           treated as if having all ones.
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch.
            Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on :attr:`size_average`. When
            :attr:`reduce` is ``False``, returns a loss per batch element instead and
            ignores :attr:`size_average`. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` where `N` is the batch size and `C` is the number of classes.
        - Target: :math:`(N, C)`, same shape as the input.
        - Output: scalar. If `reduce` is False, then `(N)`.
    """

    def __init__(self, weight=None, size_average=True, reduce=True):
        super(MultiLabelSoftMarginLoss, self).__init__(weight, size_average, reduce)

    def forward(self, input, target):
        return F.multilabel_soft_margin_loss(input, target, self.weight, self.size_average,
                                             self.reduce)


class CosineEmbeddingLoss(_Loss):
    r"""Creates a criterion that measures the loss given input tensors
    :math:`x_1`, :math:`x_2` and a `Tensor` label `y` with values 1 or -1.
    This is used for measuring whether two inputs are similar or dissimilar,
    using the cosine distance, and is typically used for learning nonlinear
    embeddings or semi-supervised learning.

    The loss function for each sample is:

    .. math::
        \text{loss}(x, y) =
        \begin{cases}
        1 - \cos(x_1, x_2), & \text{if } y == 1 \\
        \max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y == -1
        \end{cases}

    Args:
        margin (float, optional): Should be a number from `-1` to `1`, `0` to `0.5`
            is suggested. If `margin` is missing, the default value is `0`.
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch.
            Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on :attr:`size_average`. When
            :attr:`reduce` is ``False``, returns a loss per batch element instead and
            ignores :attr:`size_average`. Default: ``True``
    """

    def __init__(self, margin=0, size_average=True, reduce=True):
        super(CosineEmbeddingLoss, self).__init__(size_average, reduce)
        self.margin = margin

    def forward(self, input1, input2, target):
        return F.cosine_embedding_loss(input1, input2, target, self.margin, self.size_average,
                                       self.reduce)


class MarginRankingLoss(_Loss):
    r"""Creates a criterion that measures the loss given
    inputs `x1`, `x2`, two 1D mini-batch `Tensor`s,
    and a label 1D mini-batch tensor `y` with values (`1` or `-1`).

    If `y == 1` then it assumed the first input should be ranked higher
    (have a larger value) than the second input, and vice-versa for `y == -1`.

    The loss function for each sample in the mini-batch is:

    .. math::
        \text{loss}(x, y) = \max(0, -y * (x1 - x2) + \text{margin})

    Args:
        margin (float, optional): Has a default value of `0`.
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch.
            Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on :attr:`size_average`. When
            :attr:`reduce` is ``False``, returns a loss per batch element instead and
            ignores :attr:`size_average`. Default: ``True``

    Shape:
        - Input: :math:`(N, D)` where `N` is the batch size and `D` is the size of a sample.
        - Target: :math:`(N)`
        - Output: scalar. If `reduce` is False, then `(N)`.
    """

    def __init__(self, margin=0, size_average=True, reduce=True):
        super(MarginRankingLoss, self).__init__(size_average, reduce)
        self.margin = margin

    def forward(self, input1, input2, target):
        return F.margin_ranking_loss(input1, input2, target, self.margin, self.size_average,
                                     self.reduce)


class MultiMarginLoss(_WeightedLoss):
    r"""Creates a criterion that optimizes a multi-class classification hinge
    loss (margin-based loss) between input `x` (a 2D mini-batch `Tensor`) and
    output `y` (which is a 1D tensor of target class indices,
    :math:`0 \leq y \leq \text{x.size}(1)`):

    For each mini-batch sample, the loss in terms of the 1D input `x` and scalar
    output `y` is:

    .. math::
        \text{loss}(x, y) = \frac{\sum_i \max(0, \text{margin} - x[y] + x[i]))^p}{\text{x.size}(0)}

    where `i == 0` to `x.size(0)` and :math:`i \neq y`.

    Optionally, you can give non-equal weighting on the classes by passing
    a 1D `weight` tensor into the constructor.

    The loss function then becomes:

    .. math::
        \text{loss}(x, y) = \frac{\sum_i \max(0, w[y] * (\text{margin} - x[y] - x[i]))^p)}{\text{x.size}(0)}

    Args:
        p (int, optional): Has a default value of `1`. `1` and `2` are the only
            supported values
        margin (float, optional): Has a default value of `1`.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch.
            Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on :attr:`size_average`. When
            :attr:`reduce` is ``False``, returns a loss per batch element instead and
            ignores :attr:`size_average`. Default: ``True``

    """

    def __init__(self, p=1, margin=1, weight=None, size_average=True, reduce=True):
        super(MultiMarginLoss, self).__init__(weight, size_average, reduce)
        if p != 1 and p != 2:
            raise ValueError("only p == 1 and p == 2 supported")
        assert weight is None or weight.dim() == 1
        self.p = p
        self.margin = margin

    def forward(self, input, target):
        return F.multi_margin_loss(input, target, self.p, self.margin, self.weight,
                                   self.size_average, self.reduce)


class TripletMarginLoss(_Loss):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3 and a margin with a value greater than 0.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n`: anchor, positive examples and negative
    example respectively. The shapes of all input tensors should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}

    where :math:`d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p`.

    Args:
        margin (float, optional): Default: `1`.
        p (int, optional): The norm degree for pairwise distance. Default: `2`.
        swap (float, optional): The distance swap is described in detail in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. Default: ``False``.
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch.
            Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on :attr:`size_average`. When
            :attr:`reduce` is ``False``, returns a loss per batch element instead and
            ignores :attr:`size_average`. Default: ``True``

    Shape:
        - Input: :math:`(N, D)` where `D` is the vector dimension.
        - Output: scalar. If `reduce` is False, then `(N)`.

    >>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    >>> input1 = torch.randn(100, 128, requires_grad=True)
    >>> input2 = torch.randn(100, 128, requires_grad=True)
    >>> input3 = torch.randn(100, 128, requires_grad=True)
    >>> output = triplet_loss(input1, input2, input3)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """

    def __init__(self, margin=1.0, p=2, eps=1e-6, swap=False, size_average=True, reduce=True):
        super(TripletMarginLoss, self).__init__(size_average, reduce)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, self.margin, self.p,
                                     self.eps, self.swap, self.size_average, self.reduce)

# TODO: L1HingeEmbeddingCriterion
# TODO: MSECriterion weight
# TODO: ClassSimplexCriterion
