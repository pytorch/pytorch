from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.

    Example::

        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    class _PartialWrapper(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, *args, **keywords):
            return self.p(*args, **keywords)

        def __repr__(self):
            return self.p.__repr__()

        with_args = _with_args
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


ABC = ABCMeta(str("ABC"), (object,), {})  # compatible with Python 2 *and* 3:


class ObserverBase(ABC, nn.Module):
    r"""Base observer Module.
    Any observer implementation should derive from this class.

    Concrete observers should follow the same API. In forward, they will update
    the statistics of the observed Tensor. And they should provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    Args:
        dtype: Quantized data type
    """
    def __init__(self, dtype):
        super(ObserverBase, self).__init__()
        self.dtype = dtype

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    with_args = classmethod(_with_args)


class _ObserverBase(ObserverBase):
    r"""Internal common base for all qint/quint8 observers.

    This base is for commonly used parameters used internally.
    Users should use `~torch.quantization.observer.ObserverBase` as a base class
    for custom observers.

    Args:
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        reduce_range: Reduces the range of the quantized data type by 1 bit.
                      This is sometimes required to avoid instruction overflow.
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.

    .. warning::

        :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.

    .. warning::

        :attr:`qscheme` can only take one of the following options:

        - ``torch.per_tensor_affine``
        - ``torch.per_tensor_symmetric``
        - ``torch.per_channel_affine``
        - ``torch.per_channel_symmetric``
    """

    # Version 1/None
    #   self
    #
    # Version 2
    #   self
    #   |--- eps : Tensor
    _version = 2

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None):
        super(_ObserverBase, self).__init__(dtype=dtype)
        self.qscheme = qscheme
        if reduce_range:
            warnings.warn(
                "Please use quant_min and quant_max to specify the range for observers. \
                    reduce_range will be deprecated in a future release of PyTorch."
            )
        self.reduce_range = reduce_range
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        assert self.qscheme in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
            torch.per_channel_affine,
            torch.per_channel_symmetric,
            torch.per_channel_affine_float_qparams,
        ), "Default Observer only works for per_tensor_affine, \
                per_tensor_symmetric, per_channel_affine, \
                per_channel_symmetric and per_channel_float_qparams quantization scheme"
        assert self.dtype in (
            torch.qint8,
            torch.quint8,
        ), "Default Observer only works for qint8 and quint8 data type"
        self.has_customized_qrange = (quant_min is not None) and (quant_max is not None)
        if self.has_customized_qrange:
            self._validate_qmin_qmax(quant_min, quant_max)
        self.quant_min = quant_min
        self.quant_max = quant_max

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        version = local_metadata.get('version', None)

        if version is None or version == 1:
            # eps was moved to a buffer in version 2
            eps = torch.tensor([torch.finfo(torch.float32).eps])
            state_dict[prefix + 'eps'] = eps

        super(ObserverBase, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                        missing_keys, unexpected_keys, error_msgs)

    @torch.jit.export
    def _validate_qmin_qmax(self, quant_min, quant_max):
        # type: (int, int) -> None
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
        assert quant_min <= 0 <= quant_max, "Used-specified quantization range must include 0."
        assert quant_min < quant_max, "qmin must be strictly less than qmax for user-specified quantization range."

    @torch.jit.export
    def _calculate_qmin_qmax(self):
        # type: () -> Tuple[int, int]
        r"""Calculates actual qmin and qmax based on the quantization range,
        observer datatype and if range is reduced.
        """
        if self.has_customized_qrange:
            # This initialization here is to be resolve TorchScript compilation issues and allow
            # using of refinement to decouple initial_qmin and initial_qmax from quantization range.
            # The actual values of initial_qmin and initial_qmax will be reset below.
            initial_quant_min, initial_quant_max = 0, 255
            # The following assignment of self.qmin and self.qmax to the local variables and the if check refine the
            # attribute from Optional valid integers for use, based on TorchScript's requirements.
            custom_quant_min, custom_quant_max = self.quant_min, self.quant_max
            if custom_quant_min is not None and custom_quant_max is not None:
                initial_quant_min, initial_quant_max = custom_quant_min, custom_quant_max

            qrange_len = initial_quant_max - initial_quant_min + 1
            assert 0 < qrange_len <= 256, \
                "quantization range should be positive and not exceed the maximum bit range (=256)."
            if self.dtype == torch.qint8:
                quant_min, quant_max = -qrange_len // 2, qrange_len // 2 - 1
            else:
                quant_min, quant_max = 0, qrange_len - 1
            if self.reduce_range:
                quant_min, quant_max = quant_min // 2, quant_max // 2
        else:
            # Fallback onto default 8-bit qmin and qmax calculation if dynamic range is not used.
            if self.dtype == torch.qint8:
                if self.reduce_range:
                    quant_min, quant_max = -64, 63
                else:
                    quant_min, quant_max = -128, 127
            else:
                if self.reduce_range:
                    quant_min, quant_max = 0, 127
                else:
                    quant_min, quant_max = 0, 255
        return quant_min, quant_max

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        r"""Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases

        Args:
            min_val: Minimum values per channel
            max_val: Maximum values per channel

        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """
        if min_val.numel() == 0 or max_val.numel() == 0:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0]), torch.tensor([0])

        if min_val.dim() == 0 or max_val.dim() == 0:
            if min_val == float('inf') and max_val == float('-inf'):
                warnings.warn(
                    "must run observer before calling calculate_qparams.\
                                        Returning default scale and zero point "
                )
                return torch.tensor([1.0]), torch.tensor([0])

            assert min_val <= max_val, "min {} should be less than max {}".format(
                min_val, max_val
            )
        else:
            assert torch.all(min_val <= max_val), "min {} should be less than max {}".format(
                min_val, max_val
            )

        quant_min, quant_max = self._calculate_qmin_qmax()
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = torch.ones(min_val_neg.size(), dtype=torch.float32)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64)
        device = 'cuda' if min_val_neg.is_cuda else 'cpu'

        if self.qscheme == torch.per_tensor_symmetric or self.qscheme == torch.per_channel_symmetric:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
            if self.dtype == torch.quint8:
                if self.has_customized_qrange:
                    # When customized quantization range is used, down-rounded midpoint of the range is chosen.
                    zero_point = zero_point.new_full(zero_point.size(), (quant_min + quant_max) // 2)
                else:
                    zero_point = zero_point.new_full(zero_point.size(), 128)
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
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)

        # For scalar values, cast them to Tensors of size 1 to keep the shape
        # consistent with default values in FakeQuantize.
        if len(scale.shape) == 0:
            # TODO: switch to scale.item() after adding JIT support
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            # TODO: switch to zero_point.item() after adding JIT support
            zero_point = torch.tensor([int(zero_point)], dtype=zero_point.dtype, device=device)
            if self.qscheme == torch.per_channel_affine_float_qparams:
                zero_point = torch.tensor([float(zero_point)], dtype=zero_point.dtype, device=device)


        return scale, zero_point


class MinMaxObserver(_ObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running min and max values.

    This observer uses the tensor min/max statistics to compute the quantization
    parameters. The module records the running minimum and maximum of incoming
    tensors, and uses this statistic to compute the quantization parameters.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.

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

    .. warning:: Only works with ``torch.per_tensor_symmetric`` quantization scheme

    .. warning:: :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None):
        # For x86 quantized kernels, we need to ensure that the vpmaddubsw
        # instruction does not overflow. We allow for a reduce_range argument to
        # observers that reduces the quantized range to (0,127) or (-64, 63).
        # For more details see aten/src/ATen/native/quantized/cpu/qconv.cpp
        # This is not an optimal choice for non x86 backends as it loses a bit
        # of precision for activations.

        super(MinMaxObserver, self).__init__(dtype=dtype,
                                             qscheme=qscheme,
                                             reduce_range=reduce_range,
                                             quant_min=quant_min,
                                             quant_max=quant_max)
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        if self.qscheme == torch.per_tensor_symmetric and \
           self.reduce_range and \
           self.dtype == torch.quint8:
            raise NotImplementedError("Cannot reduce range for symmetric \
                                       quantization for quint8")

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch._aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)


class MovingAverageMinMaxObserver(MinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    moving average of the min and max values.

    This observer computes the quantization parameters based on the moving
    averages of minimums and maximums of the incoming tensors. The module
    records the average minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.

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
    :class:`~torch.quantization.observer.MinMaxObserver`.

    .. note:: Only works with ``torch.per_tensor_affine`` quantization scheme.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """
    def __init__(self, averaging_constant=0.01, dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None):
        self.averaging_constant = averaging_constant
        super(MovingAverageMinMaxObserver, self).__init__(dtype=dtype,
                                                          qscheme=qscheme,
                                                          reduce_range=reduce_range,
                                                          quant_min=quant_min,
                                                          quant_max=quant_max)

    def forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if min_val == float('inf') and max_val == float('-inf'):
            min_val, max_val = torch._aminmax(x)
        else:
            min_val_cur, max_val_cur = torch._aminmax(x)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig


class MinMaxDynamicQuantObserver(MinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    tensor min and max values in dynamic quantization.

    This observer will mimic the quantization steps followed in the operator
    to compute the activation tensor quantization parameters at run-time.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit

    .. warning:: Only works with ``torch.per_tensor_symmetric`` quantization scheme

    .. warning:: :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 0.1 and 0.
    """

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""

        if self.max_val == float('-inf') and self.min_val == float('inf'):
            return torch.tensor([1.0]), torch.tensor([0])

        assert self.min_val <= self.max_val, "min {} should be less than max {}".format(
            self.min_val, self.max_val
        )

        if self.dtype == torch.qint8:
            if self.reduce_range:
                qmin, qmax = -64, 63
            else:
                qmin, qmax = -128, 127
        else:  # dtype == torch.quint8
            if self.reduce_range:
                qmin, qmax = 0, 127
            else:
                qmin, qmax = 0, 255

        max_val, min_val = self.max_val.to(dtype=torch.float), self.min_val.to(dtype=torch.float)

        # Extend the min_val and max_val to ensure that it contains 0.
        min_val = torch.min(min_val, torch.tensor(0.).to(dtype=torch.float))
        max_val = torch.max(max_val, torch.tensor(0.).to(dtype=torch.float))

        scale = (max_val.to(dtype=torch.double) - min_val) / float(qmax - qmin)

        if scale == 0.0 or torch.isinf(1.0 / scale):
            scale = torch.tensor(0.1).to(dtype=torch.float)
            zero_point = 0

        zero_point_from_min = qmin - min_val / scale.to(dtype=torch.double)
        zero_point_from_max = qmax - max_val / scale.to(dtype=torch.double)
        zero_point_from_min_error = abs(qmin) - abs(min_val / scale.to(dtype=torch.double))
        zero_point_from_max_error = abs(qmax) - abs(max_val / scale.to(dtype=torch.double))

        if zero_point_from_min_error < zero_point_from_max_error:
            initial_zero_point = zero_point_from_min
        else:
            initial_zero_point = zero_point_from_max

        nudged_zero_point = 0

        if initial_zero_point < qmin:
            nudged_zero_point = qmin
        elif initial_zero_point > qmax:
            nudged_zero_point = qmax
        else:
            nudged_zero_point = int(initial_zero_point.round())

        return scale.to(dtype=torch.float), torch.tensor([nudged_zero_point])

class PerChannelMinMaxObserver(_ObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        ch_axis: Channel axis
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.

    The quantization parameters are computed the same way as in
    :class:`~torch.quantization.observer.MinMaxObserver`, with the difference
    that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(self, ch_axis=0, dtype=torch.quint8,
                 qscheme=torch.per_channel_affine, reduce_range=False,
                 quant_min=None, quant_max=None):
        super(PerChannelMinMaxObserver, self).__init__(dtype=dtype,
                                                       qscheme=qscheme,
                                                       reduce_range=reduce_range,
                                                       quant_min=quant_min,
                                                       quant_max=quant_max)
        self.ch_axis = ch_axis
        self.register_buffer('min_vals', torch.tensor([]))
        self.register_buffer('max_vals', torch.tensor([]))
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

    @torch.jit.ignore
    def _forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        min_vals = self.min_vals
        max_vals = self.max_vals
        x_dim = x.size()

        new_axis_list = list(range(len(x_dim)))
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(tuple(new_axis_list))
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_vals.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_vals.numel() == 0 or max_vals.numel() == 0:
            min_vals, max_vals = torch._aminmax(y, 1)
        else:
            min_vals_cur, max_vals_cur = torch._aminmax(y, 1)
            min_vals = torch.min(min_vals_cur, min_vals)
            max_vals = torch.max(max_vals_cur, max_vals)
        self.min_vals.resize_(min_vals.shape)
        self.max_vals.resize_(max_vals.shape)
        self.min_vals.copy_(min_vals)
        self.max_vals.copy_(max_vals)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        return self._calculate_qparams(self.min_vals, self.max_vals)

    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_vals, self.max_vals)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        local_state = ['min_vals', 'max_vals']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading min_vals or max_vals
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == 'min_vals':
                    self.min_vals.resize_(val.shape)
                else:
                    self.max_vals.resize_(val.shape)
            elif strict:
                missing_keys.append(key)
        super(PerChannelMinMaxObserver, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                                    missing_keys, unexpected_keys, error_msgs)

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

    The quantization parameters are computed the same way as in
    :class:`~torch.quantization.observer.MovingAverageMinMaxObserver`, with the
    difference that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(self, averaging_constant=0.01, ch_axis=0, dtype=torch.quint8,
                 qscheme=torch.per_channel_affine, reduce_range=False,
                 quant_min=None, quant_max=None):
        super(MovingAveragePerChannelMinMaxObserver, self).__init__(
            ch_axis=ch_axis, dtype=dtype, qscheme=qscheme,
            reduce_range=reduce_range, quant_min=quant_min, quant_max=quant_max)
        self.averaging_constant = averaging_constant

    def forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_vals.dtype)
        min_vals = self.min_vals
        max_vals = self.max_vals
        x_dim = x.size()

        new_axis_list = list(range(len(x_dim)))
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(tuple(new_axis_list))
        y = torch.flatten(y, start_dim=1)
        if min_vals.numel() == 0 or max_vals.numel() == 0:
            min_vals, max_vals = torch._aminmax(y, 1)
        else:
            min_vals_cur, max_vals_cur = torch._aminmax(y, 1)
            min_vals = min_vals + self.averaging_constant * (min_vals_cur - min_vals)
            max_vals = max_vals + self.averaging_constant * (max_vals_cur - max_vals)
        self.min_vals.resize_(min_vals.shape)
        self.max_vals.resize_(max_vals.shape)
        self.min_vals.copy_(min_vals)
        self.max_vals.copy_(max_vals)
        return x_orig

class HistogramObserver(_ObserverBase):
    r"""
    The module records the running histogram of tensor values along with
    min/max values. ``calculate_qparams`` will calculate scale and zero_point.

    Args:
        bins: Number of bins to use for the histogram
        upsample_rate: Factor by which the histograms are upsampled, this is
                       used to interpolate histograms with varying ranges across observations
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit

    The scale and zero point are computed as follows:

    1. Create the histogram of the incoming inputs.
        The histogram is computed continuously, and the ranges per bin change
        with every new tensor observed.
    2. Search the distribution in the histogram for optimal min/max values.
        The search for the min/max values ensures the minimization of the
        quantization error with respect to the floating point model.
    3. Compute the scale and zero point the same way as in the
        :class:`~torch.quantization.MinMaxObserver`
    """

    def __init__(self, bins=2048, upsample_rate=128, dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine, reduce_range=False):
        # bins: The number of bins used for histogram calculation.
        super(HistogramObserver, self).__init__(dtype=dtype,
                                                qscheme=qscheme,
                                                reduce_range=reduce_range)
        self.bins = bins
        self.register_buffer('histogram', torch.zeros(self.bins))
        self.register_buffer('min_val', torch.tensor([]))
        self.register_buffer('max_val', torch.tensor([]))
        self.dst_nbins = 2 ** torch.iinfo(self.dtype).bits
        self.upsample_rate = upsample_rate

    @torch.jit.ignore
    def _non_linear_param_search(self):
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        def _get_norm(delta_begin, delta_end, density, norm_type):
            r"""
            Compute the norm of the values uniformaly distributed between
            delta_begin and delta_end.

            norm = density * (integral_{begin, end} x^2)
                 = density * (end^3 - begin^3) / 3
            """
            assert norm_type == "L2", "Only L2 norms are currently supported"
            norm = 0.0
            if norm_type == "L2":
                norm = (
                    delta_end * delta_end * delta_end
                    - delta_begin * delta_begin * delta_begin
                ) / 3
            return density * norm

        def _compute_quantization_error(next_start_bin, next_end_bin, norm_type):
            r"""
            Compute the quantization error if we use start_bin to end_bin as the
            min and max to do the quantization.
            """
            bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

            dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
            if dst_bin_width == 0.0:
                return 0.0

            src_bin = torch.arange(self.bins)
            # distances from the beginning of first dst_bin to the beginning and
            # end of src_bin
            src_bin_begin = (src_bin - next_start_bin) * bin_width
            src_bin_end = src_bin_begin + bin_width

            # which dst_bins the beginning and end of src_bin belong to?
            dst_bin_of_begin = torch.clamp(src_bin_begin // dst_bin_width, 0, self.dst_nbins - 1)
            dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

            dst_bin_of_end = torch.clamp(src_bin_end // dst_bin_width, 0, self.dst_nbins - 1)
            dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width

            density = self.histogram / bin_width

            norm = torch.zeros(self.bins)

            delta_begin = src_bin_begin - dst_bin_of_begin_center
            delta_end = dst_bin_width / 2
            norm += _get_norm(delta_begin, delta_end, density, norm_type)

            norm += (dst_bin_of_end - dst_bin_of_begin - 1) * _get_norm(
                -dst_bin_width / 2, dst_bin_width / 2, density, norm_type
            )

            dst_bin_of_end_center = (
                dst_bin_of_end * dst_bin_width + dst_bin_width / 2
            )

            delta_begin = -dst_bin_width / 2
            delta_end = src_bin_end - dst_bin_of_end_center
            norm += _get_norm(delta_begin, delta_end, density, norm_type)

            return norm.sum()

        assert self.histogram.size()[0] == self.bins, "bins mistmatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = sum(self.histogram)
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
            norm = _compute_quantization_error(next_start_bin, next_end_bin, "L2")

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    @torch.jit.ignore
    def _adjust_min_max(self, combined_min, combined_max, upsample_rate):
        # type: (Tensor, Tensor, int) -> Tuple[Tensor, Tensor, int, int]
        # We ensure that:
        # (combined_max - combined_min)/(downsample_rate*Nbins) = (max - min)/(upsample_rate*Nbins)
        # This allows us to have a common grid of resolution s, where we can align
        # the input histogram
        # start_idx maps min_val to the histogram bin index.

        hist_bin_width = (self.max_val - self.min_val) / (self.bins * upsample_rate)
        downsample_rate = torch.ceil((combined_max - combined_min) / (self.bins * hist_bin_width)).to(torch.int).item()
        e = downsample_rate * (self.bins * hist_bin_width) - (combined_max - combined_min)
        # Relax only the max, not the min, so that for one sided distributions, min stays at zero
        combined_max = combined_max + e
        combined_min = combined_min
        start_idx = torch.round((self.min_val - combined_min) / hist_bin_width).to(torch.int).item()
        return combined_min, combined_max, downsample_rate, start_idx

    @torch.jit.ignore
    def _combine_histograms(self, orig_hist, new_hist, upsample_rate, downsample_rate, start_idx, Nbins):
        # type: (Tensor, Tensor, int, int, int, int) -> Tensor
        # First up-sample the histogram with new data by a factor of L
        # This creates an approximate probability density thats piecwise constant
        upsampled_histogram = new_hist.repeat_interleave(upsample_rate)
        # Now insert the upsampled histogram into the output
        # histogram, which is initialized with zeros.
        # The offset at which the histogram is introduced is determined
        # by the start index as the output histogram can cover a wider range
        histogram_with_output_range = torch.zeros((Nbins * downsample_rate), device=orig_hist.device)
        histogram_with_output_range[start_idx:Nbins * upsample_rate + start_idx] = upsampled_histogram
        # Compute integral histogram, double precision is needed to ensure
        # that there are no overflows
        integral_histogram = torch.cumsum(histogram_with_output_range, 0,
                                          dtype=torch.double)[downsample_rate - 1 :: downsample_rate]
        # Finally perform interpolation
        shifted_integral_histogram = torch.zeros((Nbins), device=orig_hist.device)
        shifted_integral_histogram[1:Nbins] = integral_histogram[0:-1]
        interpolated_histogram = (integral_histogram - shifted_integral_histogram) / upsample_rate
        orig_hist = orig_hist + interpolated_histogram.to(torch.float)
        return orig_hist

    def forward(self, x_orig):
        # type: (Tensor) -> Tensor
        x = x_orig.detach()
        min_val = self.min_val
        max_val = self.max_val
        same_values = False
        if min_val.numel() > 0 and max_val.numel() > 0:
            same_values = min_val.item() == max_val.item()
        if min_val.numel() == 0 or max_val.numel() == 0 or same_values:
            min_val, max_val = torch._aminmax(x)
            self.min_val.resize_(min_val.shape)
            self.min_val.copy_(min_val)
            self.max_val.resize_(max_val.shape)
            self.max_val.copy_(max_val)
            torch.histc(x, self.bins, min=min_val, max=max_val, out=self.histogram)
        else:
            new_min, new_max = torch._aminmax(x)
            combined_min = torch.min(new_min, min_val)
            combined_max = torch.max(new_max, max_val)
            # combine the existing histogram and new histogram into 1 histogram
            # We do this by first upsampling the histogram to a dense grid
            # and then downsampling the histogram efficiently
            combined_min, combined_max, downsample_rate, start_idx = \
                self._adjust_min_max(combined_min, combined_max, self.upsample_rate)
            combined_histogram = torch.histc(x, self.bins, min=combined_min, max=combined_max)
            if combined_min == min_val and combined_max == max_val:
                combined_histogram += self.histogram
            else:
                combined_histogram = self._combine_histograms(
                    combined_histogram,
                    self.histogram,
                    self.upsample_rate,
                    downsample_rate,
                    start_idx,
                    self.bins)

            self.histogram.resize_(combined_histogram.shape)
            self.histogram.copy_(combined_histogram)
            self.min_val.resize_(combined_min.shape)
            self.min_val.copy_(combined_min)
            self.max_val.resize_(combined_max.shape)
            self.max_val.copy_(combined_max)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0]), torch.tensor([0])
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        new_min, new_max = self._non_linear_param_search()

        return self._calculate_qparams(new_min, new_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(HistogramObserver, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'min_val'] = self.min_val
        destination[prefix + 'max_val'] = self.max_val

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        local_state = ['min_val', 'max_val']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(HistogramObserver, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                             missing_keys, unexpected_keys, error_msgs)

class PlaceholderObserver(ObserverBase):
    r"""
    Observer that doesn't do anything and just passes its configuration to the
    quantized module's ``.from_float()``.

    Can be used for quantization to float16 which doesn't require determining
    ranges.

    Args:
        dtype: Quantized data type
        custom_op_name: (temporary) specify this observer for an operator that doesn't require any observation
                        (Can be used in Graph Mode Passes for special case ops).
    """
    def __init__(self, dtype=torch.float16, custom_op_name=""):
        super(PlaceholderObserver, self).__init__(dtype=dtype)
        self.dtype = dtype
        self.custom_op = custom_op_name

    def forward(self, x):
        return x

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception("calculate_qparams should not be called for PlaceholderObserver")



class RecordingObserver(_ObserverBase):
    r"""
    The module is mainly for debug and records the tensor values during runtime.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
    """
    __annotations__ = {"tensor_val": List[Optional[torch.Tensor]]}

    def __init__(self, **kwargs):
        super(RecordingObserver, self).__init__(**kwargs)
        self.tensor_val = []

    def forward(self, x):
        self.tensor_val.append(x.clone())
        return x

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception("calculate_qparams should not be called for RecordingObserver")

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
    def __init__(self, dtype=torch.float16, custom_op_name=""):
        super(NoopObserver, self).__init__(dtype=dtype)
        self.dtype = dtype
        self.custom_op = custom_op_name

    def forward(self, x):
        return x

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception("calculate_qparams should not be called for NoopObserver")


# Restrict activations to be in the range (0,127)
default_observer = MinMaxObserver.with_args(reduce_range=True)
default_debug_observer = RecordingObserver
default_weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
default_histogram_observer = HistogramObserver.with_args(reduce_range=True)
default_per_channel_weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
default_dynamic_quant_observer = MinMaxDynamicQuantObserver
default_float_qparams_observer = PerChannelMinMaxObserver.with_args(dtype=torch.quint8,
                                                                    qscheme=torch.per_channel_affine_float_qparams,
                                                                    ch_axis=0)
