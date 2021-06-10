import torch
import torch.nn as nn

from ..observer import (
    PerChannelMinMaxObserver, _with_args
)

from collections import namedtuple
import warnings


class _InputEqualizationObserver(nn.Module):
    r"""Observer for tracking the running min/max values of input columns, and
    computing the quantization parameters for the overall min/max input values.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme
        quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.

    The running minimum/maximum :math:`x_\text{min/max}` are computed in the
    same way as :class:`~torch.quantization.observer.PerChannelMinMaxObserver`,
    with the difference that the running min/max values are stored per column.

    The qparams are calculated by multiplying the min/max input column values
    with the equalization scale, reducing to find the global min/max input
    values, and then calculating in the same way as in
    :class:`~torch.quantization.observer.MinMaxObserver`

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 quant_min=None, quant_max=None, factory_kwargs=None) -> None:
        super(_InputEqualizationObserver, self).__init__()

        if qscheme not in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            raise TypeError("Input qscheme must be per-tensor")

        self.dtype = dtype

        self.input_obs = PerChannelMinMaxObserver(ch_axis=1, dtype=dtype,
                                                  qscheme=qscheme,
                                                  quant_min=quant_min,
                                                  quant_max=quant_max,
                                                  factory_kwargs=factory_kwargs)

        self.equalization_scale = torch.empty(0)

    def forward(self, x_orig):
        # TODO: Allow for convoluational layers
        if not (x_orig.ndim == 2):
            raise ValueError("InputEqualizationObserver only supports Linear layers")

        return self.input_obs(x_orig)

    def get_input_minmax(self):
        return (self.input_obs.min_vals, self.input_obs.max_vals)

    def set_equalization_scale(self, equalization_scale):
        self.equalization_scale = equalization_scale

    def calculate_qparams(self):
        r"""
        Returns the scale/zero_point for the input and weight rows
        """

        if self.equalization_scale.nelement() == 0:
            warnings.warn(
                "Must call calculate_scale before calling calculate_qparams.\
                Returning default scale and zero point. "
            )
            return torch.tensor([1.0]), torch.tensor([0]), torch.tensor([1.0]), torch.tensor([0])

        # Calculate qparams for the scaled min/max inputs
        # Scale the input by the equalization scale located at the same column
        # index
        (min_inputs, max_inputs) = self.get_input_minmax()
        min_input_scaled = torch.min(torch.mul(min_inputs, self.equalization_scale))
        max_input_scaled = torch.max(torch.mul(max_inputs, self.equalization_scale))
        (scale_input, zero_point_input) = self.input_obs._calculate_qparams(min_input_scaled, max_input_scaled)

        return scale_input, zero_point_input

    with_args = classmethod(_with_args)


class _WeightEqualizationObserver(nn.Module):
    r"""Observer for tracking the running min/max values of weight columns and
    rows, and computing the quantization parameters for the weight rows.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme
        quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.

    This observer is made up of 2 PerChannelMinMaxObservers
        - weight_col_obs: Used to record the running minimum and maximum of
        columns of incoming weight tensors
        - weight_row_obs: Used to record the running minimum and maximum of
        rows of incoming weight tensors

    The running minimum/maximum :math:`w_\text{min/max}` are computed in the
    same way as :class:`~torch.quantization.observer.PerChannelMinMaxObserver`.

    The qparams are calculated by multiplying the min/max weight row values
    with the inverse of the equalization scale, and then calculating in the same
    way as in :class:`~torch.quantization.observer.PerChannelMinMaxObserver`

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine, quant_min=None,
                 quant_max=None, factory_kwargs=None) -> None:
        super(_WeightEqualizationObserver, self).__init__()

        self.dtype = dtype

        self.weight_col_obs = PerChannelMinMaxObserver(ch_axis=1, dtype=dtype,
                                                       qscheme=qscheme,
                                                       quant_min=quant_min,
                                                       quant_max=quant_max,
                                                       factory_kwargs=factory_kwargs)

        self.weight_row_obs = PerChannelMinMaxObserver(ch_axis=0, dtype=dtype,
                                                       qscheme=qscheme,
                                                       quant_min=quant_min,
                                                       quant_max=quant_max,
                                                       factory_kwargs=factory_kwargs)

        self.equalization_scale = torch.empty(0)

    def forward(self, w_orig):
        # TODO: Allow for convoluational layers
        if not (w_orig.ndim == 2):
            raise ValueError("WeightEqualizationObserver only supports Linear layers")

        return self._forward(w_orig)

    def _forward(self, w_orig):
        r"""
        Calculates the min/max values of each weight column and weight row.
        """

        w_orig = self.weight_col_obs(w_orig)
        w_orig = self.weight_row_obs(w_orig)

        # Calculate the column indices of the min/max weight in each row
        num_row, _ = w_orig.shape
        min_weights_ind = []
        max_weights_ind = []
        for i in range(num_row):
            min_weights_ind.append(torch.nonzero(w_orig[i] == self.weight_row_obs.min_vals[i])[0][0])
            max_weights_ind.append(torch.nonzero(w_orig[i] == self.weight_row_obs.max_vals[i])[0][0])
        self.min_weights_ind = torch.tensor(min_weights_ind)
        self.max_weights_ind = torch.tensor(max_weights_ind)

        return w_orig

    def get_weight_col_minmax(self):
        return (self.weight_col_obs.min_vals, self.weight_col_obs.max_vals)

    def get_weight_row_minmax(self):
        return (self.weight_row_obs.min_vals, self.weight_row_obs.max_vals)

    def set_equalization_scale(self, equalization_scale):
        self.equalization_scale = equalization_scale

    def calculate_qparams(self):
        r"""
        Returns the scale/zero_point for the input and weight rows
        """

        if self.equalization_scale.nelement() == 0:
            warnings.warn(
                "Must call calculate_scale before calling calculate_qparams.\
                Returning default scale and zero point. "
            )
            return torch.tensor([1.0]), torch.tensor([0]), torch.tensor([1.0]), torch.tensor([0])

        if self.min_weights_ind is None or self.max_weights_ind is None:
            warnings.warn(
                "Must find the column indicies of the minimum of each row in the \
                weights in order to calculate the qparams calculate the \
                qparams. Returning default scale and zero point. "
            )
            return torch.tensor([1.0]), torch.tensor([0]), torch.tensor([1.0]), torch.tensor([0])

        # Calculate the qparams for weights by using the rows
        # Scale the weight rows by the reciprocal of the equalization scale
        # located at the same column index
        (min_weights, max_weights) = self.get_weight_row_minmax()
        min_weights_scaled = torch.mul(min_weights, torch.reciprocal(self.equalization_scale[self.min_weights_ind]))
        max_weights_scaled = torch.mul(max_weights, torch.reciprocal(self.equalization_scale[self.max_weights_ind]))
        (scale_weight, zero_point_weight) = self.weight_row_obs._calculate_qparams(min_weights_scaled, max_weights_scaled)

        return scale_weight, zero_point_weight

    with_args = classmethod(_with_args)


def calculate_equalization_scale(input_obs: _InputEqualizationObserver,
                                 weight_obs: _WeightEqualizationObserver) -> torch.Tensor:
    r""" Calculates the equalization scale and sets the equalization_scale value
    in the observers.

    Args:
        input_obs: Observer that tracks the ranges for the input columns
        weight_obs: Observer that tracks the ranges for the weight columns
    """

    (min_inputs, max_inputs) = input_obs.get_input_minmax()
    (min_weights, max_weights) = weight_obs.get_weight_col_minmax()

    if not (min_inputs.shape == min_weights.shape):
        raise ValueError(
            "Input and Weight must have the same column dimension. " +
            f"Found {min_inputs.shape} and {max_inputs.shape} instead."
        )

    equalization_scale = torch.sqrt((max_weights - min_weights) / (max_inputs - min_inputs))

    return equalization_scale


class EqualizationQConfig(namedtuple('EqualizationQConfig', ['input_activation', 'weight'])):
    """
    Describes how to quantize a layer or a part of the network specifically for
    input-weight equalization by providing settings (observer classes) for
    inputs, outputs, and weights.

    Note that EqualizationQConfig needs to contain observer **classes** (like
    MinMaxObserver) or a callable that returns instances on invocation, not the
    concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of
    the layers.

    Observer classes have usually reasonable default arguments, but they can be
    overwritten with `with_args` method (that behaves like functools.partial):

    my_qconfig = EqualizationQConfig(input_activation=_InputEqualizationObserver.with_args(dtype=torch.qint8),
                                    weight=_WeightEqualizationObserver.with_args(dtype=torch.qint8))
    """
    def __new__(cls, input_activation=torch.nn.Identity, weight=torch.nn.Identity):
        if isinstance(input_activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError("EqualizationQConfig received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        self = super(EqualizationQConfig, cls).__new__(cls, input_activation, weight)
        return self


input_equalization_observer = _InputEqualizationObserver.with_args(
    dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_equalization_observer = _WeightEqualizationObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
default_equalization_qconfig = EqualizationQConfig(input_activation=input_equalization_observer,
                                                   weight=weight_equalization_observer)
