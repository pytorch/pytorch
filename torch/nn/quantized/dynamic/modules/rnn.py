from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict
import numbers

import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch import _VF
from torch._jit_internal import Tuple, Optional, List  # noqa: F401
from torch.nn.utils.rnn import PackedSequence


def apply_permutation(tensor, permutation, dim=1):
    # type: (Tensor, Tensor, int) -> Tensor
    return tensor.index_select(dim, permutation)

class PackedParameter(torch.nn.Module):
    def __init__(self, param):
        super(PackedParameter, self).__init__()
        self.param = param

    @torch.jit.export
    def __getstate__(self):
        return (torch.ops.quantized.linear_unpack(self.param), self.training)

    @torch.jit.export
    def __setstate__(self, state):
        self.param = torch.ops.quantized.linear_prepack(*state[0])
        self.training = state[1]

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(PackedParameter, self)._save_to_state_dict(destination, prefix,
                                                         keep_vars)
        (w, b) = self.unpack()

        destination[prefix + 'weight'] = w
        destination[prefix + 'bias'] = b

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight = state_dict[prefix + 'weight']
        bias = state_dict[prefix + 'bias']
        self.param = torch.ops.quantized.linear_prepack(weight, bias)
        state_dict.pop(prefix + 'weight')
        state_dict.pop(prefix + 'bias')

        super(PackedParameter, self)._load_from_state_dict(state_dict, prefix,
                                                           local_metadata,
                                                           False, missing_keys,
                                                           unexpected_keys,
                                                           error_msgs)

    def __repr__(self):
        return repr(self.unpack())

    def unpack(self):
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
                 dropout=0., bidirectional=False, dtype=torch.qint8):
        super(RNNBase, self).__init__()

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.dtype = dtype
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

        self._all_weight_names = []
        _all_weight_values = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                def process_weights(ihhh, layer, suffix, qweight, bias, dtype):
                    if dtype == torch.qint8:
                        # for each layer, for each direction we need to quantize and pack
                        # weights and pack parameters in this order:
                        #
                        #   w_ih, w_hh
                        packed_weight = \
                            torch.ops.quantized.linear_prepack(qweight, bias)

                        params = [packed_weight]
                        pos_names = ['w']
                        ret_name = ['{}_{}_l{}{}'.format(
                            name, ihhh, layer, suffix) for name in pos_names]
                        return params, ret_name
                    else:
                        # for each layer, for each direction we need to quantize and pack
                        # weights and pack parameters in this order:
                        #
                        #   packed_ih, packed_hh, b_ih, b_hh
                        packed_weight = torch.ops.quantized.linear_prepack_fp16(
                            qweight)

                        params = [packed_weight, bias]
                        pos_names = ['packed', 'b']
                        ret_name = ['{}_{}_l{}{}'.format(name, ihhh, layer, suffix) for name in pos_names]
                        return params, ret_name

                if dtype == torch.qint8:
                    w_ih = torch._empty_affine_quantized(
                        [gate_size, layer_input_size], scale=1, zero_point=0, dtype=torch.qint8)
                    w_hh = torch._empty_affine_quantized(
                        [gate_size, hidden_size], scale=1, zero_point=0, dtype=torch.qint8)
                    b_ih = torch.empty([gate_size], dtype=torch.float)
                    # Second bias vector included for CuDNN compatibility. Only one
                    # bias vector is needed in standard definition.
                    b_hh = torch.empty([gate_size], dtype=torch.float)

                else:
                    w_ih = torch.Tensor(gate_size, layer_input_size).float()
                    w_hh = torch.Tensor(gate_size, hidden_size).float()
                    b_ih = torch.Tensor(gate_size).float()
                    # Second bias vector included for CuDNN compatibility. Only one
                    # bias vector is needed in standard definition.
                    b_hh = torch.Tensor(gate_size).float()

                suffix = '_reverse' if direction == 1 else ''
                ih_params, ih_param_names = process_weights(
                    'ih', layer, suffix, w_ih, b_ih, dtype)
                hh_params, hh_param_names = process_weights(
                    'hh', layer, suffix, w_hh, b_hh, dtype)

                for (ih, ih_name), (hh, hh_name) in zip(zip(ih_params, ih_param_names), zip(hh_params, hh_param_names)):
                    self._all_weight_names.extend([ih_name, hh_name])
                    _all_weight_values.extend([PackedParameter(p) for p in [ih, hh]])

            self._all_weight_values = torch.nn.ModuleList(_all_weight_values)

    def _get_name(self):
        return 'DynamicQuantizedRNN'

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __repr__(self):
        # We don't want to show `ModuleList` children, hence custom
        # `__repr__`. This is the same as nn.Module.__repr__, except the check
        # for the `PackedParameter` and `nn.ModuleList`.
        # You should still override `extra_repr` to add more info.
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            if isinstance(module, (PackedParameter, nn.ModuleList)):
                continue
            mod_str = repr(module)
            mod_str = nn.modules.module._addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

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
            raise RuntimeError(msg.format(
                expected_hidden_size, tuple(hx.size())))

    def check_forward_args(self, input, hidden, batch_sizes):
        # type: (Tensor, Tensor, Optional[Tensor]) -> None
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)
        self.check_hidden_size(hidden, expected_hidden_size,
                               msg='Expected hidden size {}, got {}')

    def permute_hidden(self, hx, permutation):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)

    @property
    def all_weights(self):
        result = OrderedDict()
        for idx, name in enumerate(self._all_weight_names):
            result[name] = self._all_weight_values[idx].unpack()
        return result

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == torch.nn.LSTM, 'nn.quantized.dynamic.RNNBase.from_float only works for nn.LSTM'
        assert hasattr(
            mod, 'qconfig'), 'Input float module must have qconfig defined'

        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer = mod.qconfig.weight()
        else:
            # We have the circular import issues if we import the qconfig in the beginning of this file:
            # https://github.com/pytorch/pytorch/pull/24231. The current workaround is to postpone the
            # import until we need it.
            from torch.quantization.qconfig import default_dynamic_qconfig
            weight_observer = default_dynamic_qconfig.weight()

        dtype = weight_observer.dtype
        supported_scalar_types = [torch.qint8, torch.float16]
        if dtype not in supported_scalar_types:
            raise RuntimeError('Unsupported dtype for dynamic RNN quantization: {}'.format(dtype))

        if mod.mode == 'LSTM':
            qRNNBase = LSTM(mod.input_size, mod.hidden_size, mod.num_layers,
                            mod.bias, mod.batch_first, mod.dropout, mod.bidirectional, dtype)
        else:
            raise NotImplementedError('Only LSTM is supported for QuantizedRNN for now')

        num_directions = 2 if mod.bidirectional else 1

        assert mod.bias

        qRNNBase._all_weight_names = []
        _all_weight_values = []
        for layer in range(qRNNBase.num_layers):
            for direction in range(num_directions):
                layer_input_size = qRNNBase.input_size if layer == 0 else qRNNBase.hidden_size * num_directions

                def process_weights(ihhh, layer, suffix, dtype):
                    weight_name = 'weight_{}_l{}{}'.format(ihhh, layer, suffix)
                    bias_name = 'bias_{}_l{}{}'.format(ihhh, layer, suffix)

                    weight = getattr(mod, weight_name)
                    bias = getattr(mod, bias_name)

                    if dtype == torch.qint8:
                        # for each layer, for each direction we need to quantize and pack
                        # weights and pack parameters in this order:
                        #
                        #   w_ih, w_hh
                        weight_observer(weight)
                        wt_scale, wt_zp = weight_observer.calculate_qparams()
                        qweight = torch.quantize_per_tensor(
                            weight.float(), float(wt_scale), int(wt_zp), torch.qint8)
                        packed_weight = \
                            torch.ops.quantized.linear_prepack(qweight, bias)

                        params = [packed_weight]
                        pos_names = ['w']
                        ret_name = ['{}_{}_l{}{}'.format(
                            name, ihhh, layer, suffix) for name in pos_names]
                        return params, ret_name
                    else:
                        # for each layer, for each direction we need to quantize and pack
                        # weights and pack parameters in this order:
                        #
                        #   packed_ih, packed_hh, b_ih, b_hh
                        packed_weight = torch.fbgemm_pack_gemm_matrix_fp16(
                            weight.float())

                        params = [packed_weight, bias]
                        pos_names = ['packed', 'b']
                        ret_name = ['{}_{}_l{}{}'.format(name, ihhh, layer, suffix) for name in pos_names]
                        return params, ret_name

                suffix = '_reverse' if direction == 1 else ''
                ih_params, ih_param_names = process_weights('ih', layer, suffix, dtype)
                hh_params, hh_param_names = process_weights('hh', layer, suffix, dtype)

                for (ih, ih_name), (hh, hh_name) in zip(zip(ih_params, ih_param_names), zip(hh_params, hh_param_names)):
                    qRNNBase._all_weight_names.extend([ih_name, hh_name])
                    _all_weight_values.extend([PackedParameter(p) for p in [ih, hh]])
        qRNNBase._all_weight_values = torch.nn.ModuleList(_all_weight_values)

        return qRNNBase


class LSTM(RNNBase):

    _FLOAT_MODULE = nn.LSTM

    __overloads__ = {'forward': ['forward_packed', 'forward_tensor']}

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

    def _get_name(self):
        return 'DynamicQuantizedLSTM'

    def forward_impl(self, input, hx, batch_sizes, max_batch_size, sorted_indices):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor], int, Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)

        weight_values = []
        for mod in self._all_weight_values:
            weight_values.append(mod.param)

        if batch_sizes is None:
            result = _VF.quantized_lstm(input, hx, weight_values, self.bias, self.num_layers,
                                        float(self.dropout), self.training, self.bidirectional,
                                        self.batch_first, dtype=self.dtype, use_dynamic=True)
        else:
            result = _VF.quantized_lstm(input, batch_sizes, hx, weight_values, self.bias,
                                        self.num_layers, float(self.dropout), self.training,
                                        self.bidirectional, dtype=self.dtype, use_dynamic=True)
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
        return super(LSTM, cls).from_float(mod)
