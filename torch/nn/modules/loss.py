import warnings

import torch
from .module import Module
from .container import Sequential
from .activation import LogSoftmax
from .. import functional as F
from .. import _reduction as _Reduction
from ..._jit_internal import weak_module, weak_script_method


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)


@weak_module
class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute error (MAE) between each element in
    the input `x` and target `y`.

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|,

    where :math:`N` is the batch size. If reduce is ``True``, then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if size\_average} = \text{True;}\\
            \operatorname{sum}(L),  & \text{if size\_average} = \text{False.}
        \end{cases}

    `x` and `y` are tensors of arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the constructor argument
    `size_average=False`.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

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
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(L1Loss, self).__init__(size_average, reduce, reduction)

    @weak_script_method
    def forward(self, input, target):
        return F.l1_loss(input, target, reduction=self.reduction)


@weak_module
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
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore\_index}\},

    where :math:`N` is the batch size. If :attr:`reduce` is ``True`` (default),
    then

    .. math::
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, & \text{if}\;
            \text{size\_average} = \text{True},\\
            \sum_{n=1}^N l_n,  & \text{if}\;
            \text{size\_average} = \text{False}.
        \end{cases}

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 2`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below). In the case of images, it computes NLL loss per-pixel.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When
            :attr:`size_average` is ``True``, the loss is averaged over
            non-ignored targets.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

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
        >>> conv = nn.Conv2d(16, C, (3, 3))
        >>> m = nn.LogSoftmax()
        >>> # each element in target has to have 0 <= value < C
        >>> target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        >>> output = loss(m(conv(data)), target)
        >>> output.backward()
    """
    __constants__ = ['ignore_index', 'weight', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(NLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    @weak_script_method
    def forward(self, input, target):
        return F.nll_loss(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


@weak_module
class NLLLoss2d(NLLLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        warnings.warn("NLLLoss2d has been deprecated. "
                      "Please use NLLLoss instead as a drop-in replacement and see "
                      "https://pytorch.org/docs/master/nn.html#torch.nn.NLLLoss for more details.")
        super(NLLLoss2d, self).__init__(weight, size_average, ignore_index, reduce, reduction)


@weak_module
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
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        eps (float, optional): Small value to avoid evaluation of :math:`\log(0)` when
            :attr:`log_input == False`. Default: 1e-8
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Examples::

        >>> loss = nn.PoissonNLLLoss()
        >>> log_input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> output = loss(log_input, target)
        >>> output.backward()
    """
    __constants__ = ['log_input', 'full', 'eps', 'reduction']

    def __init__(self, log_input=True, full=False, size_average=None,
                 eps=1e-8, reduce=None, reduction='mean'):
        super(PoissonNLLLoss, self).__init__(size_average, reduce, reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps

    @weak_script_method
    def forward(self, log_input, target):
        return F.poisson_nll_loss(log_input, target, log_input=self.log_input, full=self.full,
                                  eps=self.eps, reduction=self.reduction)


@weak_module
class KLDivLoss(_Loss):
    r"""The `Kullback-Leibler divergence`_ Loss

    KL divergence is a useful distance measure for continuous distributions
    and is often useful when performing direct regression over the space of
    (discretely sampled) continuous output distributions.

    As with :class:`~torch.nn.NLLLoss`, the `input` given is expected to contain
    *log-probabilities*. However, unlike :class:`~torch.nn.NLLLoss`, `input` is not
    restricted to a 2D Tensor.
    The targets are given as *probabilities* (i.e. without taking the logarithm).

    This criterion expects a `target` `Tensor` of the same size as the
    `input` `Tensor`.

    The unreduced (i.e. with :attr:`reduce` set to ``False``) loss can be described as:

    .. math::
        l(x,y) = L := \{ l_1,\dots,l_N \}, \quad
        l_n = y_n \cdot \left( \log y_n - x_n \right)

    where the index :math:`N` spans all dimensions of ``input`` and :math:`L` has the same
    shape as ``input``. If :attr:`reduce` is ``True`` (the default), then:

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size\_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size\_average} = \text{False}.
        \end{cases}

    In default reduction mode 'mean', the losses are averaged for each minibatch over observations
    **as well as** over dimensions. 'batchmean' mode gives the correct KL divergence where losses
    are averaged over batch dimension only. 'mean' mode's behavior will be changed to the same as
    'batchmean' in the next major release.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'batchmean' | 'sum' | 'mean'.
            'none': no reduction will be applied.
            'batchmean': the sum of the output will be divided by batchsize.
            'sum': the output will be summed.
            'mean': the output will be divided by the number of elements in the output.
            Default: 'mean'
        .. note:: :attr:`size_average` and :attr:`reduce` are in the process of being deprecated,
            and in the meantime, specifying either of those two args will override :attr:`reduction`.
        .. note:: `reduction='mean'` doesn't return the true kl divergence value, please use
            `reduction='batchmean'` which aligns with KL math definition.
            In the next major release, 'mean' will be changed to be the same as 'batchmean'.


    Shape:
        - input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - target: :math:`(N, *)`, same shape as the input
        - output: scalar by default. If `reduce` is ``False``, then :math:`(N, *)`,
            the same shape as the input

    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(KLDivLoss, self).__init__(size_average, reduce, reduction)

    @weak_script_method
    def forward(self, input, target):
        return F.kl_div(input, target, reduction=self.reduction)


@weak_module
class MSELoss(_Loss):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input `x` and target `y`.

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If reduce is ``True``, then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size\_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size\_average} = \text{False}.
        \end{cases}

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets :attr:`size_average` to ``False``.

    To get a batch of losses, a loss per batch element, set `reduce` to
    ``False``. These losses are not averaged and are not affected by
    `size_average`.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

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
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    @weak_script_method
    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction)


@weak_module
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
            \operatorname{mean}(L), & \text{if}\; \text{size\_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size\_average} = \text{False}.
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `y` should be numbers
    between 0 and 1.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

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
    __constants__ = ['reduction', 'weight']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)

    @weak_script_method
    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


@weak_module
class BCEWithLogitsLoss(_Loss):
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log \sigma(x_n)
        + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right],

    where :math:`N` is the batch size. If reduce is ``True``, then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if size\_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if size\_average} = \text{False}.
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.

    It's possible to trade off recall and precision by adding weights to positive examples.
    In this case the loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ p_n y_n \cdot \log \sigma(x_n)
        + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right],

    where :math:`p_n` is the positive weight of class :math:`n`.
    :math:`p_n > 1` increases the recall, :math:`p_n < 1` increases the precision.

    For example, if a dataset contains 100 positive and 300 negative examples of a single class,
    then `pos_weight` for the class should be equal to :math:`\frac{300}{100}=3`.
    The loss would act as if the dataset contains :math:`3\times 100=300` positive examples.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.

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
    __constants__ = ['weight', 'pos_weight', 'reduction']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    @weak_script_method
    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)


@weak_module
class HingeEmbeddingLoss(_Loss):
    r"""Measures the loss given an input tensor `x` and a labels tensor `y`
    containing values (`1` or `-1`).
    This is usually used for measuring whether two inputs are similar or
    dissimilar, e.g. using the L1 pairwise distance as `x`, and is typically
    used for learning nonlinear embeddings or semi-supervised learning.

    The loss function for :math:`n`-th sample in the mini-batch is

    .. math::
        l_n = \begin{cases}
            x_n, & \text{if}\; y_n = 1,\\
            \max \{0, \Delta - x_n\}, & \text{if}\; y_n = -1,
        \end{cases}

    and the total loss functions is

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if size\_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if size\_average} = \text{False}.
        \end{cases}

    where :math:`L = \{l_1,\dots,l_N\}^\top`.

    Args:
        margin (float, optional): Has a default value of `1`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Shape:
        - Input: Tensor of arbitrary shape. The sum operation operates over all the elements.
        - Target: Same shape as input.
        - Output: scalar. If reduce is ``False``, then same shape as the input
    """
    __constants__ = ['margin', 'reduction']

    def __init__(self, margin=1.0, size_average=None, reduce=None, reduction='mean'):
        super(HingeEmbeddingLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    @weak_script_method
    def forward(self, input, target):
        return F.hinge_embedding_loss(input, target, margin=self.margin, reduction=self.reduction)


@weak_module
class MultiLabelMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a multi-class multi-classification
    hinge loss (margin-based loss) between input `x`  (a 2D mini-batch `Tensor`)
    and output `y` (which is a 2D `Tensor` of target class indices).
    For each sample in the mini-batch:

    .. math::
        \text{loss}(x, y) = \sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\text{x.size}(0)}

    where :math:`i == 0` to :math:`x.size(0)`, \
    :math:`j == 0` to :math:`y.size(0)`, \
    :math:`y[j] \geq 0`, \
    and :math:`i \neq y[j]` for all :math:`i` and :math:`j`.

    `y` and `x` must have the same size.

    The criterion only considers a contiguous block of non-negative targets that
    starts at the front.

    This allows for different samples to have variable amounts of target classes

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Shape:
        - Input: :math:`(C)` or :math:`(N, C)` where `N` is the batch size and `C`
          is the number of classes.
        - Target: :math:`(C)` or :math:`(N, C)`, same shape as the input.
        - Output: scalar. If `reduce` is False, then `(N)`.
    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MultiLabelMarginLoss, self).__init__(size_average, reduce, reduction)

    @weak_script_method
    def forward(self, input, target):
        return F.multilabel_margin_loss(input, target, reduction=self.reduction)


@weak_module
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
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If reduce is ``False``, then
          :math:`(N, *)`, same shape as the input

    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(SmoothL1Loss, self).__init__(size_average, reduce, reduction)

    @weak_script_method
    def forward(self, input, target):
        return F.smooth_l1_loss(input, target, reduction=self.reduction)


@weak_module
class SoftMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a two-class classification
    logistic loss between input tensor `x` and target tensor `y` (containing 1 or
    -1).

    .. math::
        \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Shape:
        - Input: Tensor of arbitrary shape.
        - Target: Same shape as input.
        - Output: scalar. If reduce is ``False``, then same shape as the input

    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(SoftMarginLoss, self).__init__(size_average, reduce, reduction)

    @weak_script_method
    def forward(self, input, target):
        return F.soft_margin_loss(input, target, reduction=self.reduction)


@weak_module
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
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When `size_average` is
            ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

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
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    @weak_script_method
    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


@weak_module
class MultiLabelSoftMarginLoss(_WeightedLoss):
    r"""Creates a criterion that optimizes a multi-label one-versus-all
    loss based on max-entropy, between input `x` and target `y` of size `(N, C)`.
    For each sample in the minibatch:

    .. math::
        loss(x, y) = - \frac{1}{C} * \sum_i y[i] * \log((1 + \exp(-x[i]))^{-1})
                         + (1-y[i]) * \log\left(\frac{\exp(-x[i])}{(1 + \exp(-x[i]))}\right)

    where `i == 0` to `x.nElement()-1`, `y[i]  in {0,1}`.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Shape:
        - Input: :math:`(N, C)` where `N` is the batch size and `C` is the number of classes.
        - Target: :math:`(N, C)`, same shape as the input.
        - Output: scalar. If `reduce` is False, then `(N)`.
    """
    __constants__ = ['weight', 'reduction']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(MultiLabelSoftMarginLoss, self).__init__(weight, size_average, reduce, reduction)

    @weak_script_method
    def forward(self, input, target):
        return F.multilabel_soft_margin_loss(input, target, weight=self.weight, reduction=self.reduction)


@weak_module
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
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'
    """
    __constants__ = ['margin', 'reduction']

    def __init__(self, margin=0., size_average=None, reduce=None, reduction='mean'):
        super(CosineEmbeddingLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    @weak_script_method
    def forward(self, input1, input2, target):
        return F.cosine_embedding_loss(input1, input2, target, margin=self.margin, reduction=self.reduction)


@weak_module
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
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Shape:
        - Input: :math:`(N, D)` where `N` is the batch size and `D` is the size of a sample.
        - Target: :math:`(N)`
        - Output: scalar. If `reduce` is False, then `(N)`.
    """
    __constants__ = ['margin', 'reduction']

    def __init__(self, margin=0., size_average=None, reduce=None, reduction='mean'):
        super(MarginRankingLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    @weak_script_method
    def forward(self, input1, input2, target):
        return F.margin_ranking_loss(input1, input2, target, margin=self.margin, reduction=self.reduction)


@weak_module
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
        \text{loss}(x, y) = \frac{\sum_i \max(0, w[y] * (\text{margin} - x[y] + x[i]))^p)}{\text{x.size}(0)}

    Args:
        p (int, optional): Has a default value of `1`. `1` and `2` are the only
            supported values
        margin (float, optional): Has a default value of `1`.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'
    """
    __constants__ = ['p', 'margin', 'weight', 'reduction']

    def __init__(self, p=1, margin=1., weight=None, size_average=None,
                 reduce=None, reduction='mean'):
        super(MultiMarginLoss, self).__init__(weight, size_average, reduce, reduction)
        if p != 1 and p != 2:
            raise ValueError("only p == 1 and p == 2 supported")
        assert weight is None or weight.dim() == 1
        self.p = p
        self.margin = margin

    @weak_script_method
    def forward(self, input, target):
        return F.multi_margin_loss(input, target, p=self.p, margin=self.margin,
                                   weight=self.weight, reduction=self.reduction)


@weak_module
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


    where

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    Args:
        margin (float, optional): Default: `1`.
        p (int, optional): The norm degree for pairwise distance. Default: `2`.
        swap (float, optional): The distance swap is described in detail in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. Default: ``False``.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

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
    __constants__ = ['margin', 'p', 'eps', 'swap', 'reduction']

    def __init__(self, margin=1.0, p=2., eps=1e-6, swap=False, size_average=None,
                 reduce=None, reduction='mean'):
        super(TripletMarginLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    @weak_script_method
    def forward(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, margin=self.margin, p=self.p,
                                     eps=self.eps, swap=self.swap, reduction=self.reduction)


@weak_module
class CTCLoss(_Loss):
    r"""The Connectionist Temporal Classification loss.

    Args:
        blank (int, optional): blank label. Default :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default: 'mean'

    Inputs:
        log_probs: Tensor of size :math:`(T, N, C)` where `C = number of characters in alphabet including blank`,
            `T = input length`, and `N = batch size`.
            The logarithmized probabilities of the outputs
            (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
        targets: Tensor of size :math:`(N, S)` or `(sum(target_lengths))`.
            Targets (cannot be blank). In the second form, the targets are assumed to be concatenated.
        input_lengths: Tuple or tensor of size :math:`(N)`.
            Lengths of the inputs (must each be :math:`\leq T`)
        target_lengths: Tuple or tensor of size  :math:`(N)`.
            Lengths of the targets


    Example::

        >>> ctc_loss = nn.CTCLoss()
        >>> log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
        >>> targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
        >>> input_lengths = torch.full((16,), 50, dtype=torch.long)
        >>> target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
        >>> loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        >>> loss.backward()

    Reference:
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf

    .. Note::
        In order to use CuDNN, the following must be satisfied: :attr:`targets` must be
        in concatenated format, all :attr:`input_lengths` must be `T`.  :math:`blank=0`,
        :attr:`target_lengths` :math:`\leq 256`, the integer arguments must be of
        dtype :attr:`torch.int32`.

        The regular implementation uses the (more common in PyTorch) `torch.long` dtype.


    .. include:: cudnn_deterministic.rst


    """
    __constants__ = ['blank', 'reduction']

    def __init__(self, blank=0, reduction='mean'):
        super(CTCLoss, self).__init__(reduction=reduction)
        self.blank = blank

    @weak_script_method
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, self.blank, self.reduction)

# TODO: L1HingeEmbeddingCriterion
# TODO: MSECriterion weight
# TODO: ClassSimplexCriterion
