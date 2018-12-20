import torch
import copy
import numbers
from typing import Tuple

from torch.nn.utils.rnn import PackedSequence


class QuantizedLinear(torch.jit.ScriptModule):
    __constants__ = ['scale', 'zero_point']

    def __init__(self, other):
        super(QuantizedLinear, self).__init__()
        self.in_features = other.in_features
        self.out_features = other.out_features
        # Quantize weight and discard the original
        self.weight, self.col_offsets, self.scale, self.zero_point = torch.fbgemm_linear_quantize_weight(
            other.weight.clone().float())
        self.weight = torch.nn.Parameter(self.weight, requires_grad=False)
        self.col_offsets = torch.nn.Parameter(self.col_offsets, requires_grad=False)
        assert other.bias is not None, 'QuantizedLinear requires a bias'
        self.bias = torch.nn.Parameter(other.bias.clone().float())

        self.register_buffer(
            'packed_tensor_ptr',
            torch.fbgemm_pack_quantized_matrix(self.weight.clone(), self.weight.size(1), self.weight.size(0)))

    @torch.jit.script_method
    def _unpack(self):
        self.packed_tensor_ptr.set_(
            torch.fbgemm_pack_quantized_matrix(
                self.weight, self.weight.size(1), self.weight.size(0)))

    @torch.jit.script_method
    def _pack(self):
        self.packed_tensor_ptr.set_(
            torch.zeros(torch.jit.annotate(List[int], []), dtype=torch.uint8).detach())

    @torch.jit.script_method
    def forward(self, input):
        out = torch.fbgemm_linear_int8_weight(
            input.float(), self.weight, self.packed_tensor_ptr, self.col_offsets,
            self.scale, self.zero_point, self.bias)
        return out.type_as(input)

    def extra_repr(self):
        repr = 'in_features={in_features}, out_features={out_features}, ' \
               'scale={scale}, zero_point={zero_point}'.format(**self.__dict__)
        return repr


class QuantizedLSTM(torch.jit.ScriptModule):
    def __init__(self, other):
        super(QuantizedLSTM, self).__init__()
        self.mode = other.mode
        self.input_size = other.input_size
        self.hidden_size = other.hidden_size
        self.num_layers = other.num_layers
        if not other.bias:
            raise ValueError("QuantizedLSTM requires bias terms")
        self.batch_first = other.batch_first
        self.dropout = other.dropout
        self.bidirectional = other.bidirectional
        num_directions = 2 if self.bidirectional else 1

        if not isinstance(self.dropout, numbers.Number) or not 0 <= self.dropout <= 1 or \
                isinstance(self.dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if self.dropout > 0:
            raise ValueError("QuantizedLSTM does not support dropout")

        if self.mode != 'LSTM':
            raise ValueError("QuantizedLSTM found mode is not LSTM!")

        self._all_weights = []
        other_param_itr = iter(other._all_weights)
        for layer in range(self.num_layers):
            for direction in range(num_directions):
                other_layer_names = next(other_param_itr)
                w_ih, w_hh, b_ih, b_hh = [getattr(other, name).clone() for name in other_layer_names]

                b_ih, b_hh = b_ih.float(), b_hh.float()

                w_ih, col_offsets_ih, scale_ih, zero_point_ih = \
                    torch.fbgemm_linear_quantize_weight(w_ih.clone().float())
                w_hh, col_offsets_hh, scale_hh, zero_point_hh = \
                    torch.fbgemm_linear_quantize_weight(w_hh.clone().float())

                packed_ih = torch.fbgemm_pack_quantized_matrix(w_ih, w_ih.size(1), w_ih.size(0))
                packed_hh = torch.fbgemm_pack_quantized_matrix(w_hh, w_hh.size(1), w_hh.size(0))

                layer_params = [w_ih, w_hh, b_ih, b_hh, packed_ih,
                                packed_hh, col_offsets_ih, col_offsets_hh,
                                scale_ih, scale_hh, zero_point_ih, zero_point_hh]
                param_names = ['w_ih', 'w_hh', 'b_ih', 'b_hh', 'packed_ih',
                               'packed_hh', 'col_offsets_ih', 'col_offsets_hh',
                               'scale_ih', 'scale_hh', 'zero_point_ih', 'zero_point_hh']

                suffix = '_reverse' if direction == 1 else ''
                for i in range(len(param_names)):
                    param_names[i] = (param_names[i] + "{}{}").format(layer, suffix)

                for name, param in zip(param_names, layer_params):
                    _param = param
                    if isinstance(_param, int):
                        _param = torch.tensor([_param], dtype=torch.long)
                        print('int', _param)
                    if isinstance(_param, float):
                        _param = torch.tensor([_param], dtype=torch.float)
                        print('float', _param)
                    setattr(self, name, torch.nn.Parameter(_param, requires_grad=False))
                self._all_weights.append(param_names)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions,
                                 max_batch_size, self.hidden_size,
                                 requires_grad=False)
            hx = (hx, hx)

        self.check_forward_args(input, hx, batch_sizes)
        _impl = torch._C._VariableFunctions.quantized_lstm
        if batch_sizes is None:
            result = _impl(input, hx, self._flat_weights, True, self.num_layers,
                           self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _impl(input, batch_sizes, hx, self._flat_weights, True,
                           self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:] if self.mode == 'LSTM' else result[1]

        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    @property
    def _flat_weights(self):
        return list(self._parameters.values())

def quantize_linear_modules(module):
    for name, mod in module.named_modules():
        if mod is module:
            continue
        if isinstance(mod, torch.nn.Linear):
            setattr(module, name, QuantizedLinear(mod))
        quantize_linear_modules(mod)


def quantize_lstm_modules(module):
    for name, mod in module.named_modules():
        if mod is module:
            continue
        if isinstance(mod, torch.nn.LSTM):
            setattr(module, name, QuantizedLSTM(mod))
        quantize_lstm_modules(mod)
