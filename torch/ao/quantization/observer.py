# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# temporarily skip RUF for this file for now, we can re-enable
# after move the affine quantization related things to torchao
# noqa: RUF
"""
This module implements observers which are used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

import operator
import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
    calculate_qmin_qmax,
    check_min_max_valid,
    is_per_channel,
    is_per_tensor,
    validate_qmin_qmax,
)
from torch.fx import Node


__all__ = [
    "default_affine_fixed_qparams_observer",
    "default_debug_observer",
    "default_dynamic_quant_observer",
    "default_fixed_qparams_range_0to1_observer",
    "default_fixed_qparams_range_neg1to1_observer",
    "default_float_qparams_observer",
    "default_float_qparams_observer_4bit",
    "default_histogram_observer",
    "default_observer",
    "default_per_channel_weight_observer",
    "default_placeholder_observer",
    "default_reuse_input_observer",
    "default_symmetric_fixed_qparams_observer",
    "default_weight_observer",
    "get_observer_state_dict",
    "load_observer_state_dict",
    "per_channel_weight_observer_range_neg_127_to_127",
    "weight_observer_range_neg_127_to_127",
    "FixedQParamsObserver",
    "HistogramObserver",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "NoopObserver",
    "ObserverBase",
    "PerChannelMinMaxObserver",
    "PlaceholderObserver",
    "RecordingObserver",
    "ReuseInputObserver",
    "UniformQuantizationObserverBase",
    "AffineQuantizedObserverBase",
    "Granularity",
    "MappingType",
    "PerAxis",
    "PerBlock",
    "PerGroup",
    "PerRow",
    "PerTensor",
    "PerToken",
    "TorchAODType",
    "ZeroPointDomain",
    "get_block_size",
]


class _PartialWrapper:
    def __init__(self, p):
        self.p = p
        self.callable_args = {}

    def __call__(self, *args, **keywords):
        # call each arg in callable_args and add them partial, then run with keywords
        # skip if arg_name in keywords so its possible to overwrite
        for arg_name in self.callable_args:
            if arg_name not in keywords:
                keywords = {**keywords, arg_name: self.callable_args[arg_name]()}
        return self.p(*args, **keywords)

    def __repr__(self):
        return self.p.__repr__() + self.callable_args.__repr__()

    def with_args(self, **kwargs):
        return _with_args(self, **kwargs)

    def with_callable_args(self, **kwargs):
        result = _PartialWrapper(p=self.p)
        result.callable_args = {**self.callable_args, **kwargs}
        return result


def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances. Can be used in conjunction with
    _callable_args

    Example::

        >>> # xdoctest: +SKIP("Undefined vars")
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


def _with_callable_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories args that need to be
    called at construction time.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances and those arguments should only
    be calculated at construction time. Can be used in conjunction with _with_args

    Example::

        >>> # xdoctest: +SKIP("Undefined vars")
        >>> Foo.with_callable_args = classmethod(_with_callable_args)
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_callable_args(cur_time=get_time_func).with_args(name="dan")
        >>> foo_instance1 = foo_builder()
        >>> # wait 50
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1.creation_time) == id(foo_instance2.creation_time)
        False
    """
    r = _PartialWrapper(partial(cls_or_self))
    return r.with_callable_args(**kwargs)


ABC: Any = ABCMeta("ABC", (object,), {})  # compatible with Python 2 *and* 3:


class ObserverBase(ABC, nn.Module):
    r"""Base observer Module.
    Any observer implementation should derive from this class.

    Concrete observers should follow the same API. In forward, they will update
    the statistics of the observed Tensor. And they should provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        is_dynamic: indicator for whether the observer is a placeholder for dynamic quantization
        or static quantization
    """

    def __init__(self, dtype, is_dynamic: bool = False):
        super().__init__()
        self.dtype = dtype
        self.is_dynamic = is_dynamic

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    with_args = classmethod(_with_args)
    with_callable_args = classmethod(_with_callable_args)


class UniformQuantizationObserverBase(ObserverBase):
    r"""Common base for all observers using uniform quantization to calculate
    scale and zero_point.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used.
        reduce_range: Reduces the range of the quantized data type by 1 bit.
                      This is sometimes required to avoid instruction overflow.
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    .. warning::

        :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.
               or `torch.int8` or `torch.uint8`

    .. warning::

        :attr:`qscheme` can only take one of the following options:

        - ``torch.per_tensor_affine``
        - ``torch.per_tensor_symmetric``
        - ``torch.per_channel_affine``
        - ``torch.per_channel_symmetric``
    """

    # Note: the version is shared by all observer types
    #
    # Version 1/None
    #   self
    #
    # Version 2 (base class only, does not include child class buffers)
    #   self
    #   |--- eps : Tensor
    #
    # Version 3
    #   for HistogramObserver only, changed the shape of uninitialized
    #   min_val and max_val buffers from torch.Size([0]) to torch.Size([])
    #   for PerChannelObservers, changed the name of the buffers from min_vals
    #   to min_val and from max_vals to max_val.
    _version = 3

    eps: torch.Tensor

    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        super().__init__(dtype=dtype, is_dynamic=is_dynamic, **kwargs)
        self.qscheme = qscheme
        if reduce_range:
            warnings.warn(
                "Please use quant_min and quant_max to specify the range for observers. \
                    reduce_range will be deprecated in a future release of PyTorch."
            )
        self.reduce_range = reduce_range
        self.register_buffer("eps", torch.tensor([eps], **factory_kwargs))
        assert self.qscheme in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
            torch.per_channel_affine,
            torch.per_channel_symmetric,
            torch.per_channel_affine_float_qparams,
        ), (
            "Default Observer only works for per_tensor_affine, \
                per_tensor_symmetric, per_channel_affine, \
                per_channel_symmetric and per_channel_float_qparams quantization scheme"
        )

        _ALLOWED_DTYPES = (
            torch.qint8,
            torch.quint8,
            torch.quint4x2,
            torch.qint32,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.uint16,
        )

        assert self.dtype in _ALLOWED_DTYPES, (
            f"Default Observer only works for {_ALLOWED_DTYPES} data type"
        )
        self.has_customized_qrange = (quant_min is not None) and (quant_max is not None)
        if self.has_customized_qrange:
            validate_qmin_qmax(quant_min, quant_max)
        self.quant_min, self.quant_max = calculate_qmin_qmax(
            quant_min,
            quant_max,
            self.has_customized_qrange,
            self.dtype,
            self.reduce_range,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version == 1:
            # eps was moved to a buffer in version 2
            eps = torch.tensor([torch.finfo(torch.float32).eps])
            state_dict[prefix + "eps"] = eps

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def _validate_qmin_qmax(self, quant_min: int, quant_max: int) -> None:
        r"""Validates that the user-specified quantization range is properly initialized
        and within the given bound supported by the observer dtype.

        To accommodate lower-bit quantization with respect to the existing torch.qint8 and
        torch.quint8 datatypes, the user can choose to use dynamic quantization range by passing
        in a tuple of initial qmin and qmax values. One use case is these customized qmin and qmax
        values are used to calculate static estimates of the scale and zero point for aggressive lower-bit
        fake quantization. These estimates are compared against parameters learned through backpropagation.
        The related literatures for scale and zero point via backpropagation are as follows:

        Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS
        Trained Quantization Thresholds: https://arxiv.org/pdf/1903.08066.pdf
        """
        # The variable names are prefixed with "initial" because their values (qmin and qmax) might be adjusted
        # based on whether quantization range is reduced and the datatype (signed/unsigned) used by the observer.
        assert quant_min <= 0 <= quant_max, (
            "Used-specified quantization range must include 0."
        )
        assert quant_min < quant_max, (
            "qmin must be strictly less than qmax for user-specified quantization range."
        )

    @torch.jit.export
    def _calculate_qparams(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases

        Args:
            min_val: Minimum values per channel
            max_val: Maximum values per channel

        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """
        # Functionally equivalent to 'determine_qparams' in utils.py. Observers must be torchscriptable however and qscheme
        # as far as I can tell is not allowed to passed as a parameter in torchscript functions. This makes refactoring observer
        # to use this utility a massive pain and very gross. For now Im opting just to duplicate as this code
        # seems unlikely to change (last update over 1 year ago) and when torchscript is fully deprecated we can refactor.
        # TODO(jakeszwe, jerryzh168)
        if not check_min_max_valid(min_val, max_val):
            return torch.tensor([1.0], device=min_val.device.type), torch.tensor(
                [0], device=min_val.device.type
            )

        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

        if (
            self.qscheme == torch.per_tensor_symmetric
            or self.qscheme == torch.per_channel_symmetric
        ):
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
            if self.dtype in [torch.quint8, torch.uint8]:
                if self.has_customized_qrange:
                    # When customized quantization range is used, down-rounded midpoint of the range is chosen.
                    zero_point = zero_point.new_full(
                        zero_point.size(), (quant_min + quant_max) // 2
                    )
                else:
                    zero_point = zero_point.new_full(zero_point.size(), 128)
            elif self.dtype in [torch.uint16]:
                zero_point = zero_point.new_full(zero_point.size(), 2**15)
        elif self.qscheme == torch.per_channel_affine_float_qparams:
            scale = (max_val - min_val) / float(quant_max - quant_min)
            scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
            # We use the quantize function
            # xq = Round(Xf * inv_scale + zero_point),
            # setting zero_point to (-1 * min *inv_scale) we get
            # Xq = Round((Xf - min) * inv_scale)
            zero_point = -1 * min_val / scale
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)

        # For scalar values, cast them to Tensors of size 1 to keep the shape
        # consistent with default values in FakeQuantize.
        if len(scale.shape) == 0:
            # TODO: switch to scale.item() after adding JIT support
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            # TODO: switch to zero_point.item() after adding JIT support
            zero_point = torch.tensor(
                [int(zero_point)], dtype=zero_point.dtype, device=device
            )
            if self.qscheme == torch.per_channel_affine_float_qparams:
                zero_point = torch.tensor(
                    [float(zero_point)], dtype=zero_point.dtype, device=device
                )

        return scale, zero_point

    @torch.jit.export
    def reset_min_max_vals(self):
        raise NotImplementedError("Cannot reset min/max values in the given observer.")


# Originally, this class was called `_ObserverBase`.  Keeping the old name around
# for backwards compatibility.
# TODO(after v1.13): delete this
_ObserverBase = UniformQuantizationObserverBase


class MinMaxObserver(UniformQuantizationObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running min and max values.

    This observer uses the tensor min/max statistics to compute the quantization
    parameters. The module records the running minimum and maximum of incoming
    tensors, and uses this statistic to compute the quantization parameters.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    Given running min/max as :math:`x_\text{min}` and :math:`x_\text{max}`,
    scale :math:`s` and zero point :math:`z` are computed as:

    The running minimum/maximum :math:`x_\text{min/max}` is computed as:

    .. math::

        \begin{array}{ll}
        x_\text{min} &= \begin{cases}
            \min(X) & \text{if~}x_\text{min} = \text{None} \\
            \min\left(x_\text{min}, \min(X)\right) & \text{otherwise}
        \end{cases}\\
        x_\text{max} &= \begin{cases}
            \max(X) & \text{if~}x_\text{max} = \text{None} \\
            \max\left(x_\text{max}, \max(X)\right) & \text{otherwise}
        \end{cases}\\
        \end{array}

    where :math:`X` is the observed tensor.

    The scale :math:`s` and zero point :math:`z` are then computed as:

    .. math::

        \begin{aligned}
            \text{if Symmetric:}&\\
            &s = 2 \max(|x_\text{min}|, x_\text{max}) /
                \left( Q_\text{max} - Q_\text{min} \right) \\
            &z = \begin{cases}
                0 & \text{if dtype is qint8} \\
                128 & \text{otherwise}
            \end{cases}\\
            \text{Otherwise:}&\\
                &s = \left( x_\text{max} - x_\text{min}  \right ) /
                    \left( Q_\text{max} - Q_\text{min} \right ) \\
                &z = Q_\text{min} - \text{round}(x_\text{min} / s)
        \end{aligned}

    where :math:`Q_\text{min}` and :math:`Q_\text{max}` are the minimum and
    maximum of the quantized data type.

    .. warning:: :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "MinMaxObserver's qscheme only support torch.per_tensor_symmetric \
                    and torch.per_tensor_affine."
            )
        # TODO: MinMaxObserver by itself doesn't support dynamic quantization, but
        # if it's inherited by MovingAverageObserver, and averaging_constant is 1, it
        # supports dynamic quantization, we may need to better error checking here

        # For x86 quantized kernels, we need to ensure that the vpmaddubsw
        # instruction does not overflow. We allow for a reduce_range argument to
        # observers that reduces the quantized range to (0,127) or (-64, 63).
        # For more details see aten/src/ATen/native/quantized/cpu/qconv.cpp
        # This is not an optimal choice for non x86 backends as it loses a bit
        # of precision for activations.
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        if (
            self.qscheme == torch.per_tensor_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric \
                                       quantization for quint8"
            )

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        r"""Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))


class MovingAverageMinMaxObserver(MinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    moving average of the min and max values.

    This observer computes the quantization parameters based on the moving
    averages of minimums and maximums of the incoming tensors. The module
    records the average minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The moving average min/max is computed as follows

    .. math::

        \begin{array}{ll}
                x_\text{min} = \begin{cases}
                    \min(X) & \text{if~}x_\text{min} = \text{None} \\
                    (1 - c) x_\text{min} + c \min(X) & \text{otherwise}
                \end{cases}\\
                x_\text{max} = \begin{cases}
                    \max(X) & \text{if~}x_\text{max} = \text{None} \\
                    (1 - c) x_\text{max} + c \max(X) & \text{otherwise}
                \end{cases}\\
        \end{array}

    where :math:`x_\text{min/max}` is the running average min/max, :math:`X` is
    is the incoming tensor, and :math:`c` is the ``averaging_constant``.

    The scale and zero point are then computed as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`.

    .. note:: Only works with ``torch.per_tensor_affine`` quantization scheme.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    def __init__(
        self,
        averaging_constant=0.01,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                f"MovingAverageMinMaxObserver's qscheme only support \
                torch.per_tensor_symmetric and torch.per_tensor_affine. \
                but got: {qscheme}"
            )
        self.averaging_constant = averaging_constant
        if is_dynamic and self.averaging_constant != 1:
            raise NotImplementedError(
                "MovingAverageMinMaxObserver doesn't support dynamic quantization for "
                f"averaging constant of {self.averaging_constant}"
            )
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if min_val == float("inf") and max_val == float("-inf"):
            min_val, max_val = torch.aminmax(x)
        else:
            min_val_cur, max_val_cur = torch.aminmax(x)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig


class PerChannelMinMaxObserver(UniformQuantizationObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        ch_axis: Channel axis
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`, with the difference
    that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "PerChannelMinMaxObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine and torch.per_channel_affine_float_qparams."
            )
        if is_dynamic:
            raise NotImplementedError(
                "PerChannelMinMaxObserver doesn't support dynamic quantization"
            )
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.ch_axis = ch_axis
        self.register_buffer("min_val", torch.tensor([], **factory_kwargs))
        self.register_buffer("max_val", torch.tensor([], **factory_kwargs))
        if (
            self.qscheme == torch.per_channel_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric quantization for quint8"
            )

    def forward(self, x_orig):
        return self._forward(x_orig)

    def _forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        return self._calculate_qparams(self.min_val, self.max_val)

    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, torch.Tensor],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ):
        version = local_metadata.get("version", None)
        if version is not None and version < 3:
            local_state = ["min_vals", "max_vals"]
            expected_min_name = "min_vals"
            expected_max_name = "max_vals"
        else:
            local_state = ["min_val", "max_val"]
            expected_min_name = "min_val"
            expected_max_name = "max_val"
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading min_val or max_val
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == expected_min_name:
                    self.min_val.resize_(val.shape)
                elif name == expected_max_name:
                    self.max_val.resize_(val.shape)
                else:
                    warnings.warn(
                        f"Observer load_from_state_dict got unexpected name {name}"
                    )
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == expected_min_name:
                        self.min_val.copy_(val)
                    elif name == expected_max_name:
                        self.max_val.copy_(val)
                    else:
                        warnings.warn(
                            f"Observer load_from_state_dict got unexpected name {name}"
                        )
            elif strict:
                missing_keys.append(key)

        if not torch.jit.is_scripting():
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                False,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

    def _load_from_state_dict_script(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, torch.Tensor],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ):
        self._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        # This used to be torch.ones but that does not work because
        # JIT compiler can optimize it via common subexpression elimination
        # in which case both min_val and max_val point to the same tensor.
        self.min_val = torch.rand(
            0,
        )
        self.max_val = torch.rand(
            0,
        )


class MovingAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MovingAverageMinMaxObserver`, with the
    difference that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(
        self,
        averaging_constant=0.01,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "MovingAveragePerChannelMinMaxObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine and torch.per_channel_affine_float_qparams."
            )
        if is_dynamic:
            raise NotImplementedError(
                "MovingAveragePerChannelMinMaxObserver doesn't support dynamic quantization"
            )
        super().__init__(
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        self.averaging_constant = averaging_constant

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig


class HistogramObserver(UniformQuantizationObserverBase):
    r"""
    The module records the running histogram of tensor values along with
    min/max values. ``calculate_qparams`` will calculate scale and zero_point.

    Args:
        bins: Number of bins to use for the histogram
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The scale and zero point are computed as follows:

    1. Create the histogram of the incoming inputs.
        The histogram is computed continuously, and the ranges per bin change
        with every new tensor observed.
    2. Search the distribution in the histogram for optimal min/max values.
        The search for the min/max values ensures the minimization of the
        quantization error with respect to the floating point model.
    3. Compute the scale and zero point the same way as in the
        :class:`~torch.ao.quantization.MinMaxObserver`
    """

    histogram: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        bins: int = 2048,
        dtype: torch.dtype = torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "HistogramObserver's qscheme only support torch.per_tensor_symmetric \
                    and torch.per_tensor_affine."
            )
        if is_dynamic:
            raise NotImplementedError(
                "HistogramObserver doesn't support dynamic quantization"
            )
        # bins: The number of bins used for histogram calculation.
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.bins = bins
        self.register_buffer("histogram", torch.zeros(self.bins, **factory_kwargs))
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        self.dst_nbins = 2 ** torch.iinfo(self.dtype).bits
        self.upsample_rate = (
            16  # used to reduce quantization errors when upscaling histogram
        )

    def _get_norm(
        self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin
        ) / 3
        return density * norm

    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
        r"""
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.bins, device=self.histogram.device)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = torch.clamp(
            torch.div(src_bin_begin, dst_bin_width, rounding_mode="floor"),
            0,
            self.dst_nbins - 1,
        )
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            torch.div(src_bin_end, dst_bin_width, rounding_mode="floor"),
            0,
            self.dst_nbins - 1,
        )
        density = self.histogram / bin_width

        norm = torch.zeros(self.bins, device=self.histogram.device)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(
            delta_begin,
            torch.ones(self.bins, device=self.histogram.device) * delta_end,
            density,
        )

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)

        return norm.sum().item()

    def _non_linear_param_search(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        assert self.histogram.size()[0] == self.bins, "bins mismatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = torch.sum(self.histogram).item()
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self._compute_quantization_error(next_start_bin, next_end_bin)

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    def _upscale_histogram(
        self,
        histogram: torch.Tensor,
        orig_min: torch.Tensor,
        orig_max: torch.Tensor,
        update_min: torch.Tensor,
        update_max: torch.Tensor,
    ):
        # this turns the histogram into a more fine-coarsed histogram to reduce
        # bin quantization errors
        histogram = histogram.repeat_interleave(self.upsample_rate) / self.upsample_rate
        bin_size = (orig_max - orig_min) / (self.bins * self.upsample_rate)
        mid_points_histogram = (
            torch.linspace(
                orig_min,
                orig_max,
                self.bins * self.upsample_rate + 1,
                device=orig_min.device,
            )[:-1].to(histogram.device)
            + 0.5 * bin_size
        )
        boundaries_new_histogram = torch.linspace(
            update_min, update_max, self.bins + 1, device=update_min.device
        ).to(histogram.device)
        # this maps the mid-poits of the histogram to the new histogram's space
        bucket_assignments = (
            torch.bucketize(mid_points_histogram, boundaries_new_histogram, right=True)
            - 1
        )
        # this then maps the histogram mid-points in the new space, weighted by the original histogram's values
        # this is just the old histogram in the new histogram's space

        # In case due to numerical issues the values land higher/lower than the maximum/minimum
        bucket_assignments[bucket_assignments >= self.bins] = self.bins - 1
        bucket_assignments[bucket_assignments < 0] = 0

        update_histogram = torch.bincount(
            bucket_assignments, weights=histogram, minlength=self.bins
        )
        return update_histogram

    def _combine_histograms(
        self,
        orig_hist: torch.Tensor,
        orig_min: torch.Tensor,
        orig_max: torch.Tensor,
        update_hist: torch.Tensor,
        update_min: torch.Tensor,
        update_max: torch.Tensor,
    ) -> torch.Tensor:
        # If the new min and max are the same as the current min and max,
        # we can just add the new histogram to the original histogram
        if update_min == orig_min and update_max == orig_max:
            return orig_hist + update_hist

        # If the orig hist only has one value (i.e., the min and max are the same)
        # we can just add it into new histogram
        if orig_min == orig_max:
            bin_value = torch.sum(orig_hist)
            transformed_orig_hist = (
                torch.histc(orig_min, bins=self.bins, min=update_min, max=update_max)  # type: ignore[arg-type]
                * bin_value
            )
            return transformed_orig_hist + update_hist

        # We assume the update_hist is already in the target range, we will map the orig_max to it
        assert update_min <= orig_min
        assert update_max >= orig_max

        # Now we need to turn the old_histogram, into the range of the new histogram
        transformed_orig_hist = self._upscale_histogram(
            orig_hist,
            orig_min,
            orig_max,
            update_min,
            update_max,
        )

        return update_hist + transformed_orig_hist

    def reset_histogram(
        self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> None:
        self.min_val.resize_(min_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.resize_(max_val.shape)
        self.max_val.copy_(max_val)
        assert min_val.numel() == 1 and max_val.numel() == 1, (
            "histogram min/max values must be scalar."
        )
        new_histogram = torch.histc(x, self.bins, min=min_val, max=max_val)  # type: ignore[arg-type]
        self.histogram.detach_().resize_(new_histogram.shape)
        self.histogram.copy_(new_histogram)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:  # pyre-ignore[14]
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        x_min, x_max = torch.aminmax(x)
        # want to ignore torch.inf since we don't actually
        # want to make our quantization range infinite
        # and in practice those values will be clamped
        if x_min == -torch.inf or x_max == torch.inf:
            warnings.warn("torch.inf detected in input tensor, ignoring input")
            x = x[x.abs() != torch.inf]
            if x.numel() == 0:
                return x_orig
            x_min, x_max = torch.aminmax(x)

        current_min = self.min_val
        current_max = self.max_val

        is_uninitialized = self.min_val == float("inf") or self.max_val == float("-inf")
        if is_uninitialized:
            self.reset_histogram(x, x_min, x_max)
        else:
            update_min, update_max = x_min, x_max
            new_min = torch.min(current_min, update_min)
            new_max = torch.max(current_max, update_max)

            # TODO: For some reason, this is required for it to pass torchscript test
            # new_min and new_max should already have requires_grad set to False
            new_min, new_max = new_min.detach(), new_max.detach()
            update_histogram = torch.histc(
                x,
                self.bins,
                min=new_min,  # type: ignore[arg-type]
                max=new_max,  # type: ignore[arg-type]
            ).to(self.histogram.device)
            if new_min == current_min and new_max == current_max:
                combined_histogram = self.histogram + update_histogram
                self.histogram.detach_().resize_(combined_histogram.shape)
                self.histogram.copy_(combined_histogram)
            else:
                combined_histogram = self._combine_histograms(
                    self.histogram,
                    current_min,
                    current_max,
                    update_histogram,
                    new_min,
                    new_max,
                )
                self.histogram.detach_().resize_(combined_histogram.shape)
                self.histogram.copy_(combined_histogram)
                self.min_val.detach_().resize_(new_min.shape)
                self.min_val.copy_(new_min)
                self.max_val.detach_().resize_(new_max.shape)
                self.max_val.copy_(new_max)

        return x_orig

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        is_uninitialized = self.min_val == float("inf") and self.max_val == float(
            "-inf"
        )
        if is_uninitialized:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0], device=self.min_val.device.type), torch.tensor(
                [0], device=self.min_val.device.type
            )
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        new_min, new_max = self._non_linear_param_search()

        return self._calculate_qparams(new_min, new_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "min_val"] = self.min_val
        destination[prefix + "max_val"] = self.max_val

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 3:
            # if min_val and max_val are not initialized, update their shape
            # to account for the differences between v2 and v3
            min_val_name, max_val_name = prefix + "min_val", prefix + "max_val"
            if min_val_name in state_dict:
                if state_dict[min_val_name].shape == torch.Size([0]):
                    state_dict[min_val_name] = torch.tensor(float("inf"))
            if max_val_name in state_dict:
                if state_dict[max_val_name].shape == torch.Size([0]):
                    state_dict[max_val_name] = torch.tensor(float("-inf"))

        local_state = ["min_val", "max_val"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"


class FixedQParamsObserver(ObserverBase):
    r"""
    Observer that simulates quantize and dequantize with fixed
    quantization parameters in training time. Only per tensor
    quantization is supported.

    Args:
        `scale` (float): fixed scale for the observer
        `zero_point` (int): fixed zero point for the observer
        `dtype`, `qscheme`, `quant_min`, `quant_max`
    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        scale,
        zero_point,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0,
        quant_max=255,
        is_dynamic=False,
        **kwargs,
    ):
        if is_dynamic:
            raise NotImplementedError(
                "FixedQParamsObserver doesn't support dynamic quantization"
            )
        super().__init__(dtype=dtype, is_dynamic=is_dynamic, **kwargs)
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.register_buffer("scale", torch.tensor([scale], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([zero_point], dtype=torch.int))
        self.dtype = dtype
        self.qscheme = qscheme

    def forward(self, X):
        return X

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        return self.scale, self.zero_point


class PlaceholderObserver(ObserverBase):
    r"""
    Observer that doesn't do anything and just passes its configuration to the
    quantized module's ``.from_float()``.

    Can be used for quantization to float16 which doesn't require determining
    ranges.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        quant_min: minimum value in quantized domain (TODO: align behavior with other observers)
        quant_max: maximum value in quantized domain
        custom_op_name: (temporary) specify this observer for an operator that doesn't require any observation
                        (Can be used in Graph Mode Passes for special case ops).
        compute_dtype (deprecated): if set, marks the future quantize function to use
                       dynamic quantization instead of static quantization.
                       This field is deprecated, use `is_dynamic=True` instead.
        is_dynamic: if True, the `quantize` function in the reference model
                    representation taking stats from this observer instance will
                    use dynamic quantization.
    """

    def __init__(
        self,
        dtype=torch.float32,
        custom_op_name="",
        compute_dtype=None,
        quant_min=None,
        quant_max=None,
        qscheme=None,
        eps=None,
        is_dynamic=False,
    ) -> None:
        super().__init__(dtype=dtype, is_dynamic=is_dynamic)
        if qscheme is None:
            qscheme = torch.per_tensor_affine
        if eps is None:
            eps = torch.finfo(torch.float32).eps

        # dtype of input of the target operator, e.g. for dynamic quantization
        # ops, the dtype will be float32
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps
        self.custom_op = custom_op_name
        # used for configuration of computation type for dynamic quantization
        if compute_dtype:
            is_dynamic = True
            warnings.warn(
                "Please use `is_dynamic` instead of `compute_dtype`. \
                    `compute_dtype` will be deprecated in a future release \
                    of PyTorch."
            )

    def forward(self, x):
        return x

    @torch.jit.export
    def extra_repr(self):
        return f"dtype={self.dtype}, is_dynamic={self.is_dynamic}"

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        raise Exception(  # noqa: TRY002
            "calculate_qparams should not be called for PlaceholderObserver"
        )


class RecordingObserver(ObserverBase):
    r"""
    The module is mainly for debug and records the tensor values during runtime.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
    """

    __annotations__ = {"tensor_val": list[Optional[torch.Tensor]]}

    def __init__(self, dtype=torch.quint8):
        super().__init__(dtype=dtype, is_dynamic=False)
        self.tensor_val = []

    def forward(self, x):
        self.tensor_val.append(x.clone())
        return x

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        raise Exception(  # noqa: TRY002
            "calculate_qparams should not be called for RecordingObserver"
        )

    @torch.jit.export
    def get_tensor_value(self):
        return self.tensor_val


class NoopObserver(ObserverBase):
    r"""
    Observer that doesn't do anything and just passes its configuration to the
    quantized module's ``.from_float()``.

    Primarily used for quantization to float16 which doesn't require determining
    ranges.

    Args:
        dtype: Quantized data type
        custom_op_name: (temporary) specify this observer for an operator that doesn't require any observation
                        (Can be used in Graph Mode Passes for special case ops).
    """

    def __init__(self, dtype=torch.float16, custom_op_name="") -> None:
        super().__init__(dtype=dtype, is_dynamic=False)
        self.dtype = dtype
        self.custom_op = custom_op_name

    def forward(self, x):
        return x

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        raise Exception(  # noqa: TRY002
            "calculate_qparams should not be called for NoopObserver"
        )


class ReuseInputObserver(ObserverBase):
    r"""This observer is used when we want to reuse the observer from the operator
    that produces the input Tensor, typically used for operators like reshape, e.g.
    ```
    x0 = ...
    x1 = x0.reshape()
    ```
    if we configure x0 to be observed by some observer, let's say MinMaxObserver,
    and reshape is configured with ReuseInputObserver, we'll reuse the observer instance
    for x0 for x1 (output of reshape). If x0 is not observed, we also won't observe x1.

    Note: this is only enabled in FX Graph Mode Quantization
    """

    def __init__(self) -> None:
        super().__init__(torch.quint8, is_dynamic=False)

    def forward(self, x):
        return x

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        raise Exception(  # noqa: TRY002
            "calculate_qparams should not be called for ReuseInputObserver"
        )


"""
# Experimental Affine Quantization Feature START
We plan to merge the following with torchao repo after we move pt2e flow to torchao
copied from https://github.com/pytorch/ao/blob/main/torchao/quantization/observer.py
"""
from dataclasses import dataclass
from enum import auto, Enum


class MappingType(Enum):
    """How floating point number is mapped to integer number

    symmetric mapping means floating point range is symmetrically mapped to integer range
    let's say we have floating point range (-3.5, 10.2) and integer range (-8, 7) (int4)
    we'll use (-10.2, 10.2) as the range for floating point and map that to (-8, 7)
    e.g. scale = (10.2 - (-10.2)) / (7 - (-8))

    SYMMETRIC_NO_CLIPPING_ERR is a variant of symmetric mapping, where the scale is the max of smin
    and smax, where smin = min_val_neg / quant_min, and smax = max_val_pos / quant_max. By calculating
    smin and smax individually, there can be less round error on negative values, and no out-of-range
    of all floating point values.

    asymmetric mapping means we just directly map the floating point range to integer range,
    for the above example, we will map (-3.5, 10.2) to (-8, 7) and calculate quantization parameter
    based on this mapping
    e.g. scale = (10.2 - (-3.5)) / (7 - (-8))
    """

    SYMMETRIC = auto()
    SYMMETRIC_NO_CLIPPING_ERR = auto()
    ASYMMETRIC = auto()


class ZeroPointDomain(Enum):
    """Enum that indicate whether zero_point is in integer domain or floating point domain

    integer domain: quantized_val = (float_val / scale) (integer) + zero_point (integer)
    float domain: quantized_val = (float_val - (zero_point (float) - scale * mid_point)) / scale
    none domain: quantized_val = (float_val / scale)
    """

    INT = auto()
    FLOAT = auto()
    NONE = auto()


class TorchAODType(Enum):
    """
    Placeholder for dtypes that do not exist in PyTorch core yet.
    """

    # torch.int1 to torch.int7 will be added to PyTorch 2.6
    # These will remain here for BC with older PyTorch versions
    INT1 = auto()
    INT2 = auto()
    INT3 = auto()
    INT4 = auto()
    INT5 = auto()
    INT6 = auto()
    INT7 = auto()


@dataclass(frozen=True)
class Granularity:
    """
    Base class for representing the granularity of quantization.

    This class serves as a parent for specific granularity types used in
    quantization operations, such as per-tensor or per-axis quantization.
    """


@dataclass(frozen=True)
class PerBlock(Granularity):
    """
    Represents per-block granularity in quantization. See
    :func:`~torchao.quantization.quant_primitives.quantize_affine` for docs for
    `block_size`

    Attributes:
        block_size (Tuple[int, ...]): The size of each quantization group
    """

    block_size: tuple[int, ...]


@dataclass(frozen=True)
class PerTensor(Granularity):
    """
    Represents per-tensor granularity in quantization.

    This granularity type calculates the quantization parameters
    based off the entire tensor.

    """


@dataclass(frozen=True)
class PerAxis(Granularity):
    """
    Represents per-axis granularity in quantization.

    This granularity type calculates different quantization parameters
    along a specified axis of the tensor.

    For example if the input tensor is shape [8, 16] and axis=0, then
    the quantization parameters are calculated for each row of the tensor.
    Giving a total of 8 quantization parameters.

    Attributes:
        axis (int): The axis along which reduction is performed.
    """

    axis: int


@dataclass(frozen=True)
class PerGroup(Granularity):
    """
    Represents per-channel group granularity in quantization.

    This granularity type calculates different quantization parameters
    for each group of <group_size> elements.

    For example if the input tensor is shape [8, 16], and the group size is 4, then
    the input tensor is reshaped to [64, 4]
    quantization parameters are calculated for each group of 4 elements,
    giving a total of 64 quantization parameters.

    Attributes:
        group_size (int): The size of each quantization group

    """

    group_size: int


class PerRow(Granularity):
    """
    Represents row-wise granularity in quantization.

    This is a special case of per-axis quantization and is unique to Float8 matmuls
    where the input is quantized with a block_size of (1, ..., input.shape[-1]). And the weight
    is quantized with a block_size of (1, weight.shape[1]).
    """


class PerToken(Granularity):
    """
    Represents per-token granularity in quantization.

    This granularity type calculates a different set of quantization parameters
    for each token, which is represented as the last dimension of the tensor.

    For example, if the input tensor has shape [2, 3, 4], then there are 6 tokens
    with 4 elements each, and we will calculate 6 sets of quantization parameters,
    one for each token.

    If the input tensor has only two dimensions, e.g. [8, 16], then this is
    equivalent to `PerAxis(axis=0)`, which yields 8 sets of quantization parameters.
    """


def get_block_size(
    input_shape: tuple[int, ...], granularity: Granularity
) -> tuple[int, ...]:
    """Get the block size based on the input shape and granularity type.

    Args:
        input_shape: The input tensor shape possibly more than 2 dimensions
        granularity: The granularity type of the quantization
    """
    assert isinstance(granularity, Granularity), (
        "Please provide an instance of Granularity, not subclass of it"
    )
    if isinstance(granularity, PerTensor):
        return input_shape
    elif isinstance(granularity, PerAxis):
        block_size = list(input_shape)
        block_size[granularity.axis] = 1
        return tuple(block_size)
    elif isinstance(granularity, PerRow):
        return (1,) * (len(input_shape) - 1) + (input_shape[-1],)
    elif isinstance(granularity, PerGroup):
        assert len(input_shape) == 2, (
            f"Expecting input shape dim to be 2 for per group quantization, gotinput shape: {input_shape}"
        )
        return (1, granularity.group_size)
    elif isinstance(granularity, PerToken):
        block_size = [1] * len(input_shape)
        block_size[-1] = input_shape[-1]
        return tuple(block_size)
    raise ValueError(f"Unsupported Granularity: {granularity}")


class AffineQuantizedObserverBase(ABC, torch.nn.Module):
    """Observer module for affine quantization (https://github.com/pytorch/ao/tree/main/torchao/quantization#affine-quantization)

    Args:
      `granularity` and `block_size`: The granularity of the quantization,
        must specify at least one, if both are specified `block_size` takes precedence
        Current supported granularity type are `PerTensor` and `PerAxis`
      other args: please see `:class:torchao.dtypes.AffineQuantizedTensor`
    """

    with_args = classmethod(_with_args)

    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: Optional[ZeroPointDomain] = ZeroPointDomain.INT,
        # there could be some extra args that's ignored
        **kwargs,
    ):
        super().__init__()
        assert granularity is not None, "granularity is None"

        self.mapping_type = mapping_type
        self.target_dtype = target_dtype
        self.granularity = granularity
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps
        self.scale_dtype = scale_dtype
        self.zero_point_dtype = zero_point_dtype
        self.preserve_zero = preserve_zero
        self.zero_point_domain = zero_point_domain
        # populatd during forward
        self.block_size = None
        self.original_dtype = None

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """forward function should take the input tensor
        and updates internal stats and return the original input Tensor
        """

    @abstractmethod
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate quantization parameter based on the stats attached to the observer module
        and returns a tuple of scale and zero_point Tensor
        """

    def convert(self, model: torch.fx.GraphModule, observer_node: Node):
        """
        Converts the observer node in the graph into its quantized representation

        Args:
            model: graph module to convert the observer node in
            observer_node: the observer node to convert
        """
        from torch.ao.quantization.fx.utils import create_getattr_from_value

        with model.graph.inserting_before(observer_node):
            assert self.block_size is not None, "Expecting block_size to be populated"
            assert self.original_dtype is not None, (
                "Expecting original_dtype to be populated"
            )
            if hasattr(self, "is_dynamic") and self.is_dynamic:
                choose_qparams_affine = model.graph.call_function(
                    torch.ops.pt2e_quant.choose_qparams_affine,
                    (
                        observer_node.args[0],
                        self.mapping_type.name,
                        self.block_size,
                        self.target_dtype,
                        self.quant_min,
                        self.quant_max,
                        self.eps,
                        self.scale_dtype,
                        self.zero_point_dtype,
                        self.preserve_zero,
                        self.zero_point_domain.name,
                    ),
                )
                scale_node = model.graph.call_function(
                    operator.getitem, (choose_qparams_affine, 0)
                )
                zero_point_node = model.graph.call_function(
                    operator.getitem, (choose_qparams_affine, 1)
                )
            else:
                scale, zero_point = self.calculate_qparams()
                scale_node = create_getattr_from_value(
                    model, model.graph, "_scale", scale
                )
                zero_point_node = create_getattr_from_value(
                    model, model.graph, "_zero_point", zero_point
                )

            q_node = model.graph.call_function(
                torch.ops.pt2e_quant.quantize_affine,
                (
                    observer_node.args[0],
                    self.block_size,
                    scale_node,
                    zero_point_node,
                    self.target_dtype,
                    self.quant_min,
                    self.quant_max,
                    self.zero_point_domain.name,
                ),
                {},
            )
            dq_node = model.graph.call_function(
                torch.ops.pt2e_quant.dequantize_affine,
                (
                    q_node,
                    self.block_size,
                    scale_node,
                    zero_point_node,
                    self.target_dtype,
                    self.quant_min,
                    self.quant_max,
                    self.zero_point_domain.name,
                ),
                {"output_dtype": self.original_dtype},
            )
            observer_node.replace_all_uses_with(dq_node)
            model.graph.erase_node(observer_node)


def _is_observer_script_module(mod, obs_type_name):
    """Returns true if given mod is an instance of Observer script module."""
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # qualified name looks like '__torch__.torch.ao.quantization.observer.___torch_mangle_2.MinMaxObserver'
        suffix = mod._c.qualified_name.split(".", 1)[1]
        name = re.sub(r"\.___torch_mangle_\d+", "", suffix)
        return obs_type_name in name
    return False


# Experimental Affine Quantization Feature END


def _is_activation_post_process(module):
    return isinstance(
        module,
        (
            torch.ao.quantization.ObserverBase,
            torch.ao.quantization.FakeQuantizeBase,
            AffineQuantizedObserverBase,
        ),
    ) or _is_observer_script_module(module, "quantization.observer")


def _is_per_channel_script_obs_instance(module):
    if isinstance(module, torch.jit.RecursiveScriptModule):
        return _is_observer_script_module(
            module, "quantization.observer.PerChannelMinMaxObserver"
        ) or _is_observer_script_module(
            module, "quantization.observer.MovingAveragePerChannelMinMaxObserver"
        )
    return False


def get_observer_state_dict(mod):
    r"""
    Returns the state dict corresponding to the observer stats.
    Traverse the model state_dict and extract out the stats.
    """
    od = OrderedDict()
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        for k, v in mod.state_dict().items():
            if "observer" in k:
                od[k] = v
    else:
        # path for GraphModule and nn.Module (eager mode)
        for k, v in mod.state_dict().items():
            if "activation_post_process" in k:
                od[k] = v
    od._metadata = mod.state_dict()._metadata  # type: ignore[attr-defined]
    return od


def load_observer_state_dict(mod, obs_dict):
    r"""
    Given input model and a state_dict containing model observer stats,
    load the stats back into the model. The observer state_dict can be saved
    using torch.ao.quantization.get_observer_state_dict
    """
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    for name, module in mod.named_modules():
        prefix = name + "."
        if _is_activation_post_process(module):
            if _is_per_channel_script_obs_instance(module):
                # For per-channel observers we need to call a custom load_from_state_dict to resize the tensor.
                # However this is not called when the module is scripted and we end up calling the default one in module.py
                module._load_from_state_dict_script(
                    obs_dict, prefix, {}, True, missing_keys, unexpected_keys, []
                )
            else:
                module._load_from_state_dict(
                    obs_dict, prefix, {}, False, missing_keys, unexpected_keys, []
                )
    for k in missing_keys:
        if "observer" in k or "activation_post_process" in k:
            raise Exception(  # noqa: TRY002
                f"Missing keys for observer {k} in state_dict"
            )
    for k in unexpected_keys:
        if "observer" in k or "activation_post_process" in k:
            raise Exception(  # noqa: TRY002
                f"Unexpected keys for observer {k} in state_dict"
            )


# Restrict activations to be in the range (0,127)
default_observer = MinMaxObserver.with_args(quant_min=0, quant_max=127)
"""
Default observer for static quantization, usually used for debugging.
"""

default_placeholder_observer = PlaceholderObserver
"""
Default placeholder observer, usually used for quantization to torch.float16.
"""

default_debug_observer = RecordingObserver
"""
Default debug-only observer.
"""

default_weight_observer = MinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
)
"""
Default weight observer.
"""

weight_observer_range_neg_127_to_127 = MinMaxObserver.with_args(
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
    quant_min=-127,
    quant_max=127,
    eps=2**-12,
)
"""
Symmetric weight observer with the 8-bit values restricted to [-127, +127], excluding -128.
"""

default_histogram_observer = HistogramObserver.with_args(quant_min=0, quant_max=127)
"""
Default histogram observer, usually used for PTQ.
"""

default_per_channel_weight_observer = PerChannelMinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric
)
"""
Default per-channel weight observer, usually used on backends where per-channel
weight quantization is supported, such as `fbgemm`.
"""

per_channel_weight_observer_range_neg_127_to_127 = PerChannelMinMaxObserver.with_args(
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
    quant_min=-127,
    quant_max=127,
    eps=2**-12,
)
"""
Per-channel, symmetric weight observer with the 8-bit values restricted to [-127, +127], excluding -128.
"""

default_dynamic_quant_observer = PlaceholderObserver.with_args(
    dtype=torch.quint8,
    quant_min=0,
    quant_max=255,
    is_dynamic=True,
)
"""
Default observer for dynamic quantization.
"""

default_float_qparams_observer = PerChannelMinMaxObserver.with_args(
    dtype=torch.quint8, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0
)
"""
Default observer for a floating point zero-point.
"""

default_float_qparams_observer_4bit = PerChannelMinMaxObserver.with_args(
    dtype=torch.quint4x2, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0
)
"""
Default observer for a floating point zero-point and 4 bit activations.
"""

# TODO(future PR): remove these defaults and enforce activation functions
# to explicitly specify their output range
default_fixed_qparams_range_neg1to1_observer = FixedQParamsObserver.with_args(
    scale=2.0 / 256.0, zero_point=128, dtype=torch.quint8, quant_min=0, quant_max=255
)
default_fixed_qparams_range_0to1_observer = FixedQParamsObserver.with_args(
    scale=1.0 / 256.0, zero_point=0, dtype=torch.quint8, quant_min=0, quant_max=255
)
# TODO: the following 2 variables are kept for backwards compatibility; remove after a few releases
default_symmetric_fixed_qparams_observer = default_fixed_qparams_range_neg1to1_observer
default_affine_fixed_qparams_observer = default_fixed_qparams_range_0to1_observer

"""
Default observers for fixed qparams operations.
"""

default_reuse_input_observer = ReuseInputObserver
"""
Default observer for operators like reshape that reuses the observer of input to
the operator
"""
