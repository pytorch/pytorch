
import warnings
from torch import Tensor

from .batchnorm import _LazyNormBase, _NormBase
from .. import functional as F

__all__ = ['InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d', 'LazyInstanceNorm1d',
           'LazyInstanceNorm2d', 'LazyInstanceNorm3d']

class _InstanceNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _get_no_batch_dim(self):
        raise NotImplementedError

    def _handle_no_batch_input(self, input):
        return self._apply_instance_norm(input.unsqueeze(0)).squeeze(0)

    def _apply_instance_norm(self, input):
        return F.instance_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum if self.momentum is not None else 0.0,
            self.eps,
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        # at version 1: removed running_mean and running_var when
        # track_running_stats=False (default)
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            for name in ('running_mean', 'running_var'):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    'Unexpected running stats buffer(s) {names} for {klass} '
                    'with track_running_stats=False. If state_dict is a '
                    'checkpoint saved before 0.4.0, this may be expected '
                    'because {klass} does not track running stats by default '
                    'since 0.4.0. Please remove these keys from state_dict. If '
                    'the running stats are actually needed, instead set '
                    'track_running_stats=True in {klass} to enable them. See '
                    'the documentation of {klass} for details.'
                    .format(names=" and ".join(f'"{k}"' for k in running_stats_keys),
                            klass=self.__class__.__name__))
                for key in running_stats_keys:
                    state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        feature_dim = input.dim() - self._get_no_batch_dim()
        if input.size(feature_dim) != self.num_features:
            if self.affine:
                raise ValueError(
                    f"expected input's size at dim={feature_dim} to match num_features"
                    f" ({self.num_features}), but got: {input.size(feature_dim)}.")
            else:
                warnings.warn(f"input's size at dim={feature_dim} does not match num_features. "
                              "You can silence this warning by not passing in num_features, "
                              "which is not used because affine=False")

        if input.dim() == self._get_no_batch_dim():
            return self._handle_no_batch_input(input)

        return self._apply_instance_norm(input)


class InstanceNorm1d(_InstanceNorm):
    r"""Applies Instance Normalization.

    This operation applies Instance Normalization
    over a 2D (unbatched) or 3D (batched) input as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input) if :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    .. note::
        :class:`InstanceNorm1d` and :class:`LayerNorm` are very similar, but
        have some subtle differences. :class:`InstanceNorm1d` is applied
        on each channel of channeled data like multidimensional time series, but
        :class:`LayerNorm` is usually applied on entire sample and often in NLP
        tasks. Additionally, :class:`LayerNorm` applies elementwise affine
        transform, while :class:`InstanceNorm1d` usually don't apply affine
        transform.

    Args:
        num_features: number of features or channels :math:`C` of the input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, L)` or :math:`(C, L)`
        - Output: :math:`(N, C, L)` or :math:`(C, L)` (same shape as input)

    Examples::

        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm1d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm1d(100, affine=True)
        >>> input = torch.randn(20, 100, 40)
        >>> output = m(input)
    """

    def _get_no_batch_dim(self):
        return 2

    def _check_input_dim(self, input):
        if input.dim() not in (2, 3):
            raise ValueError(f'expected 2D or 3D input (got {input.dim()}D input)')


class LazyInstanceNorm1d(_LazyNormBase, _InstanceNorm):
    r"""A :class:`torch.nn.InstanceNorm1d` module with lazy initialization of the ``num_features`` argument.

    The ``num_features`` argument of the :class:`InstanceNorm1d` is inferred from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`, `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`(C, L)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, L)` or :math:`(C, L)`
        - Output: :math:`(N, C, L)` or :math:`(C, L)` (same shape as input)
    """

    cls_to_become = InstanceNorm1d  # type: ignore[assignment]

    def _get_no_batch_dim(self):
        return 2

    def _check_input_dim(self, input):
        if input.dim() not in (2, 3):
            raise ValueError(f'expected 2D or 3D input (got {input.dim()}D input)')


class InstanceNorm2d(_InstanceNorm):
    r"""Applies Instance Normalization.

    This operation applies Instance Normalization
    over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size) if :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    .. note::
        :class:`InstanceNorm2d` and :class:`LayerNorm` are very similar, but
        have some subtle differences. :class:`InstanceNorm2d` is applied
        on each channel of channeled data like RGB images, but
        :class:`LayerNorm` is usually applied on entire sample and often in NLP
        tasks. Additionally, :class:`LayerNorm` applies elementwise affine
        transform, while :class:`InstanceNorm2d` usually don't apply affine
        transform.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)` or :math:`(C, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input)

    Examples::

        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm2d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm2d(100, affine=True)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """

    def _get_no_batch_dim(self):
        return 3

    def _check_input_dim(self, input):
        if input.dim() not in (3, 4):
            raise ValueError(f'expected 3D or 4D input (got {input.dim()}D input)')


class LazyInstanceNorm2d(_LazyNormBase, _InstanceNorm):
    r"""A :class:`torch.nn.InstanceNorm2d` module with lazy initialization of the ``num_features`` argument.

    The ``num_features`` argument of the :class:`InstanceNorm2d` is inferred from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)` or :math:`(C, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input)
    """

    cls_to_become = InstanceNorm2d  # type: ignore[assignment]

    def _get_no_batch_dim(self):
        return 3

    def _check_input_dim(self, input):
        if input.dim() not in (3, 4):
            raise ValueError(f'expected 3D or 4D input (got {input.dim()}D input)')


class InstanceNorm3d(_InstanceNorm):
    r"""Applies Instance Normalization.

    This operation applies Instance Normalization
    over a 5D input (a mini-batch of 3D inputs with additional channel dimension) as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size C (where C is the input size) if :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    .. note::
        :class:`InstanceNorm3d` and :class:`LayerNorm` are very similar, but
        have some subtle differences. :class:`InstanceNorm3d` is applied
        on each channel of channeled data like 3D models with RGB color, but
        :class:`LayerNorm` is usually applied on entire sample and often in NLP
        tasks. Additionally, :class:`LayerNorm` applies elementwise affine
        transform, while :class:`InstanceNorm3d` usually don't apply affine
        transform.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input)

    Examples::

        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm3d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm3d(100, affine=True)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)
    """

    def _get_no_batch_dim(self):
        return 4

    def _check_input_dim(self, input):
        if input.dim() not in (4, 5):
            raise ValueError(f'expected 4D or 5D input (got {input.dim()}D input)')


class LazyInstanceNorm3d(_LazyNormBase, _InstanceNorm):
    r"""A :class:`torch.nn.InstanceNorm3d` module with lazy initialization of the ``num_features`` argument.

    The ``num_features`` argument of the :class:`InstanceNorm3d` is inferred from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input)
    """

    cls_to_become = InstanceNorm3d  # type: ignore[assignment]

    def _get_no_batch_dim(self):
        return 4

    def _check_input_dim(self, input):
        if input.dim() not in (4, 5):
            raise ValueError(f'expected 4D or 5D input (got {input.dim()}D input)')
