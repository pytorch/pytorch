# mypy: allow-untyped-defs
import numbers
from typing import Optional, Union

import torch
from torch import Size, Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter

from ._functions import CrossMapLRN2d as _cross_map_lrn2d
from .module import Module


__all__ = ["LocalResponseNorm", "CrossMapLRN2d", "LayerNorm", "GroupNorm", "RMSNorm"]


class LocalResponseNorm(Module):
    r"""Applies local response normalization over an input signal.

    The input signal is composed of several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    .. math::
        b_{c} = a_{c}\left(k + \frac{\alpha}{n}
        \sum_{c'=\max(0, c-n/2)}^{\min(N-1,c+n/2)}a_{c'}^2\right)^{-\beta}

    Args:
        size: amount of neighbouring channels used for normalization
        alpha: multiplicative factor. Default: 0.0001
        beta: exponent. Default: 0.75
        k: additive factor. Default: 1

    Shape:
        - Input: :math:`(N, C, *)`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> lrn = nn.LocalResponseNorm(2)
        >>> signal_2d = torch.randn(32, 5, 24, 24)
        >>> signal_4d = torch.randn(16, 5, 7, 7, 7, 7)
        >>> output_2d = lrn(signal_2d)
        >>> output_4d = lrn(signal_4d)

    """

    __constants__ = ["size", "alpha", "beta", "k"]
    size: int
    alpha: float
    beta: float
    k: float

    def __init__(
        self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
    ) -> None:
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.local_response_norm(input, self.size, self.alpha, self.beta, self.k)

    def extra_repr(self):
        """
        Return the extra representation of the module.
        """
        return "{size}, alpha={alpha}, beta={beta}, k={k}".format(**self.__dict__)


class CrossMapLRN2d(Module):
    size: int
    alpha: float
    beta: float
    k: float

    def __init__(
        self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1
    ) -> None:
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return _cross_map_lrn2d.apply(input, self.size, self.alpha, self.beta, self.k)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return "{size}, alpha={alpha}, beta={beta}, k={k}".format(**self.__dict__)


_shape_t = Union[int, list[int], Size]


class LayerNorm(Module):
    r"""Applies Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The variance is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`elementwise_affine` is ``True``). Default: ``True``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = nn.LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = torch.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = nn.LayerNorm([C, H, W])
        >>> output = layer_norm(input)

    .. image:: ../_static/img/nn/layer_norm.jpg
        :scale: 50 %

    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class GroupNorm(Module):
    r"""Applies Group Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The variance is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    """

    __constants__ = ["num_groups", "num_channels", "eps", "affine"]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, affine={affine}".format(
            **self.__dict__
        )


class RMSNorm(Module):
    r"""Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/pdf/1910.07467.pdf>`__

    .. math::
        y_i = \frac{x_i}{\mathrm{RMS}(x)} * \gamma_i, \quad
        \text{where} \quad \text{RMS}(x) = \sqrt{\epsilon + \frac{1}{n} \sum_{i=1}^{n} x_i^2}

    The RMS is taken over the last ``D`` dimensions, where ``D``
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the RMS is computed over
    the last 2 dimensions of the input.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: ``torch.finfo(x.dtype).eps``
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> rms_norm = nn.RMSNorm([2, 3])
        >>> input = torch.randn(2, 2, 3)
        >>> rms_norm(input)

    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: Optional[float]
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass.
        """
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


# TODO: ContrastiveNorm2d
# TODO: DivisiveNorm2d
# TODO: SubtractiveNorm2d
