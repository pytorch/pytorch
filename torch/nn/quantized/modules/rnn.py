# coding=utf-8
r"""Quantized RNN modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numbers

import torch
from torch import Tensor  # noqa: F401

import torch.nn as nn
from torch.nn import _VF
from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import PackedSequence

from torch._jit_internal import Tuple, Optional, List  # noqa: F401


def _prepack(weight, bias=None):
    return torch.ops.quantized.linear_prepack(weight, bias)

def _reshape_list(list, shape):
    if len(shape) == 1:
        assert(len(list) == shape[0])
        return list
    new_list = []
    step = 1
    for d in shape[1:]:
        step *= d
    for d in range(shape[0]):
        start = d * step
        stop = start + step
        new_list.append(_reshape_list(list[start:stop], shape[1:]))
    return new_list


class PackedParameter(torch.nn.Module):
    def __init__(self, param):
        super(PackedParameter, self).__init__()
        self.param = param

    @torch.jit.export
    def __getstate__(self):
        return (self._unpack(), self.training)

    @torch.jit.export
    def __setstate__(self, state):
        self.param = torch.ops.quantized.linear_prepack(*state[0])
        self.training = state[1]

    def _unpack(self):
        return torch.ops.quantized.linear_unpack(self.param)

    # This only exists because there's a bug in recursive scripting
    # that arises only in Python 2 where a recursively scripted
    # module does not have a forward(). We can delete this once we
    # drop python 2 support
    def forward(self):
        raise RuntimeError('PackedParameter cannot be called')


class RNNBase(torch.nn.Module):
    _FLOAT_MODULE = nn.RNNBase

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0., bidirectional=False,
                 # Weight parameters
                 flat_weights_names=None,  # torch.nn.RNNBase._flat_weight_names
                 flat_weights=None,        # torch.nn.RNNBase._flat_weight
                 weights_scale=None,        # Scale for the weights
                 weights_zero_point=None):  # ZP for the weights
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        # Assume the weights are coming from the torch.nn.RNNBase.
        # The naming for variables there is `weight_` and `bias_`.
        assert(flat_weights_names is not None), "I need `flat_weights_names`"
        assert(flat_weights is not None), "I need `flat_weights`"

        step = int(self.bias) + 1
        self._flat_weights_names = flat_weights_names
        self._all_weights = _reshape_list(self._flat_weights_names,
                                          (num_layers, 2 * step))
        self._flat_weights = []
        for idx in range(0, len(self._flat_weights_names), step * 2):
            weight_ih = flat_weights[idx].clone().detach().requires_grad_(False)
            weight_hh = flat_weights[idx + 1].clone().detach().requires_grad_(False)
            bias_ih = None
            bias_hh = None
            if self.bias:
                bias_ih = flat_weights[idx + 2].clone().detach().requires_grad_(False)
                bias_hh = flat_weights[idx + 3].clone().detach().requires_grad_(False)

            if weights_scale is None:
                # Compute the weights here
                min = weight_ih.min().item()
                max = weight_ih.max().item()
                weights_scale = (max - min) / 256
                weights_zero_point = 0
            weight_ih = torch.quantize_per_tensor(weight_ih, scale=weights_scale,
                                                  zero_point=weights_zero_point,
                                                  dtype=torch.qint8)
            if weights_scale is None:
                # Compute the weights here
                min = weight_hh.min().item()
                max = weight_hh.max().item()
                weights_scale = (max - min) / 256
                weights_zero_point = 0
            weight_hh = torch.quantize_per_tensor(weight_hh, scale=weights_scale,
                                                  zero_point=weights_zero_point,
                                                  dtype=torch.qint8)
            wb_ih = _prepack(weight_ih, bias_ih)
            wb_hh = _prepack(weight_hh, bias_hh)
            self._flat_weights.append(PackedParameter(wb_ih))
            self._flat_weights.append(PackedParameter(wb_hh))

    def check_input(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> None
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def get_expected_hidden_size(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> Tuple[int, int, int]
        if batch_sizes is not None:
            mini_batch = batch_sizes[0]
            mini_batch = int(mini_batch)
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(self, hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
        # type: (Tensor, Tuple[int, int, int], str) -> None
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

    def check_forward_args(self, input, hidden, batch_sizes):
        # type: (Tensor, Tensor, Optional[Tensor]) -> None
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx, permutation):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)

    @classmethod
    def from_float(cls, mod):
        raise NotImplementedError("This is still WIP...")


class LSTM(RNNBase):
    _FLOAT_MODULE = nn.LSTM

    __overloads__ = {'forward': ['forward_packed', 'forward_tensor']}

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

    def _get_name(self):
        return 'StaticQuantizedLSTM'

    def forward_impl(self, input, hx, batch_sizes, max_batch_size, sorted_indices):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor], int, Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size)
            h = torch.quantize_per_tensor(zeros, scale=0.0625, zero_point=0,
                                          dtype=torch.quint8)
            x = torch.quantize_per_tensor(zeros, scale=5.95e-8, zero_point=0,
                                          dtype=torch.qint32)
            hx = (h, x)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)

        weight_values = []
        for mod in self._flat_weights:
            weight_values.append(mod.param)

        if batch_sizes is None:
            result = _VF.quantized_lstm(input, hx, weight_values, self.bias, self.num_layers,
                                        float(self.dropout), self.training, self.bidirectional,
                                        self.batch_first, dtype=torch.quint8, use_dynamic=False)
        else:
            result = _VF.quantized_lstm(input, batch_sizes, hx, weight_values, self.bias,
                                        self.num_layers, float(self.dropout), self.training,
                                        self.bidirectional, dtype=torch.quint8, use_dynamic=False)
        output = result[0]
        hidden = result[1:]

        return output, hidden

    @torch.jit.export
    def forward_tensor(self, input, hx=None):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        batch_sizes = None
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None

        output, hidden = self.forward_impl(
            input, hx, batch_sizes, max_batch_size, sorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    @torch.jit.export
    def forward_packed(self, input, hx=None):
        # type: (PackedSequence, Optional[Tuple[Tensor, Tensor]]) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]  # noqa
        input, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = batch_sizes[0]
        max_batch_size = int(max_batch_size)

        output, hidden = self.forward_impl(
            input, hx, batch_sizes, max_batch_size, sorted_indices)

        output = PackedSequence(output, batch_sizes,
                                sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    def permute_hidden(self, hx, permutation):
        # type: (Tuple[Tensor, Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        if permutation is None:
            return hx
        return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)

    def check_forward_args(self, input, hidden, batch_sizes):
        # type: (Tensor, Tuple[Tensor, Tensor], Optional[Tensor])->None
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden[0], expected_hidden_size,
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], expected_hidden_size,
                               'Expected hidden[1] size {}, got {}')

    @torch.jit.ignore
    def forward(self, input, hx=None):
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, hx)
        else:
            return self.forward_tensor(input, hx)

    @classmethod
    def from_float(cls, mod):
        raise NotImplementedError("[WIP] Still Working on It!")
