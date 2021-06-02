import torch
import torch.nn as nn
from torch.quantization.observer import PerChannelMinMaxObserver

import warnings


class _InputWeightObserver(nn.Module):
    r"""Observer for computing the scale factor needed for input-weight
    equalization based on the running min and max values of the input and weight
    columns, and for computing the quantization parameters based on the running
    min and max values of weight rows.

    Args:
        input_dtype: Quantized data type for the input
        input_qscheme: Quantization scheme to be used for the input. This should
            be either per_tensor_affine or per_tensor_symmetric.
        input_quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        input_quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.
        weight_dtype: Quantized data type for the weight
        weight_qscheme: Quantization scheme to be used for the weight. This should
            be either per_tensor_affine or per_tensor_symmetric.
        weight_quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        weight_quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.

    This observer is made up of 3 PerChannelMinMaxObservers
        - input_obs: Used to record the running minimum and maximum of the
        columns of incoming input tensors
        - weight_col_obs: Used to record the running minimum and maximum of
        columns of incoming weight tensors
        - weight_row_obs: Used to record the running minimum and maximum of
        rows of incoming weight tensors

    Given running min/max of the input columns as :math:`x_\text{min}` and :math:`x_\text{max}`,
    and the running min/max of the weight columns as :math:`w_\text{min}` and :math:`w_\text{max}`,
    scale :math:`S` is computed as:

    The running minimum/maximum :math:`x_\text{min/max}` and
    :math:`w_\text{min/max}` are computed in the same way as
    :class:`~torch.quantization.observer.MinMaxObserver`, with the difference
    that the running min/max values are stored per column.

    The scale factor :math:`S` is then computed as:

    .. math::
        S = \sqrt{\frac{x_{max} - x_{min}}{w_{max} - w{min}}}

    where :math:`X` is the observed tensor.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(self, input_dtype=torch.quint8, input_qscheme=torch.per_tensor_affine,
                 input_quant_min=None, input_quant_max=None, weight_dtype=torch.quint8,
                 weight_qscheme=torch.per_tensor_affine, weight_quant_min=None,
                 weight_quant_max=None, factory_kwargs=(None, None)) -> None:
        super(_InputWeightObserver, self).__init__()

        if input_qscheme not in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            raise TypeError("Input qscheme must be per-tensor")

        self.input_obs = PerChannelMinMaxObserver(ch_axis=1, dtype=input_dtype,
                                                  qscheme=input_qscheme,
                                                  quant_min=input_quant_min,
                                                  quant_max=input_quant_max,
                                                  factory_kwargs=factory_kwargs[0])

        self.weight_col_obs = PerChannelMinMaxObserver(ch_axis=1, dtype=weight_dtype,
                                                       qscheme=weight_qscheme,
                                                       quant_min=weight_quant_min,
                                                       quant_max=weight_quant_max,
                                                       factory_kwargs=factory_kwargs[1])

        self.weight_row_obs = PerChannelMinMaxObserver(ch_axis=0, dtype=weight_dtype,
                                                       qscheme=weight_qscheme,
                                                       quant_min=weight_quant_min,
                                                       quant_max=weight_quant_max,
                                                       factory_kwargs=factory_kwargs[1])

        self.equalization_scale = torch.empty(0)

    def forward(self, x_orig, w_orig):
        # TODO: Allow for convoluational layers
        if not (x_orig.ndim == 2 and w_orig.ndim == 2 and x_orig.shape[1] == w_orig.shape[1]):
            raise ValueError(
                "Input and Weight must have the same column dimension. " +
                f"Found {x_orig.shape} and {w_orig.shape} instead."
            )

        return self._forward(x_orig, w_orig)

    def _forward(self, x_orig, w_orig):
        r"""
        Calculates the min/max values of each input column, weight column, and weight row.
        """

        x_orig = self.input_obs(x_orig)
        w_orig = self.weight_col_obs(w_orig)
        w_orig = self.weight_row_obs(w_orig)

        # Calculate the column indices of the min/max weight in each row
        num_row, _ = w_orig.shape
        min_weights_ind = []
        max_weights_ind = []
        for i in range(num_row):
            min_weights_ind.append(torch.where(w_orig[i] == self.weight_row_obs.min_vals[i])[0][0])
            max_weights_ind.append(torch.where(w_orig[i] == self.weight_row_obs.max_vals[i])[0][0])
        self.min_weights_ind = torch.tensor(min_weights_ind)
        self.max_weights_ind = torch.tensor(max_weights_ind)

        return x_orig, w_orig

    def get_input_minmax(self):
        return (self.input_obs.min_vals, self.input_obs.max_vals)

    def get_weight_col_minmax(self):
        return (self.weight_col_obs.min_vals, self.weight_col_obs.max_vals)

    def get_weight_row_minmax(self):
        return (self.weight_row_obs.min_vals, self.weight_row_obs.max_vals)

    def calculate_equalization_scale(self) -> torch.Tensor:
        (min_inputs, max_inputs) = self.get_input_minmax()
        (min_weights, max_weights) = self.get_weight_col_minmax()

        self.equalization_scale = torch.sqrt((max_weights - min_weights) / (max_inputs - min_inputs))
        return self.equalization_scale

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

        (min_inputs, max_inputs) = self.get_input_minmax()
        min_input_ind = torch.argmin(min_inputs)
        max_input_ind = torch.argmax(max_inputs)

        # Calculate qparams for the scaled min/max inputs
        # Scale the input by the equalization scale located at the same column
        # index
        min_input_scaled = torch.mul(min_inputs[min_input_ind], self.equalization_scale[min_input_ind])
        max_input_scaled = torch.mul(max_inputs[max_input_ind], self.equalization_scale[max_input_ind])
        (scale_input, zero_point_input) = self.input_obs._calculate_qparams(min_input_scaled, max_input_scaled)

        # Calculate the qparams for weights by using the rows
        # Scale the weight rows by the reciprocal of the equalization scale
        # located at the same column index
        (min_weights, max_weights) = self.get_weight_row_minmax()
        min_weights_scaled = torch.mul(min_weights, torch.reciprocal(self.equalization_scale[self.min_weights_ind]))
        max_weights_scaled = torch.mul(max_weights, torch.reciprocal(self.equalization_scale[self.max_weights_ind]))
        (scale_weight, zero_point_weight) = self.weight_row_obs._calculate_qparams(min_weights_scaled, max_weights_scaled)

        return (scale_input, zero_point_input, scale_weight, zero_point_weight)
