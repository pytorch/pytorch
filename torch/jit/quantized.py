from torch import Tensor, _VF  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
import torch

import warnings

from typing import List, Optional, Tuple


class QuantizedLinear(torch.jit.ScriptModule):
    __constants__ = ['scale', 'zero_point']

    def __init__(self, other):
        super(QuantizedLinear, self).__init__()
        warnings.warn(
            "torch.jit.QuantizedLinear is deprecated and will be removed in an upcoming "
            "PyTorch release. Please use the torch.ao.nn.quantized.dynamic.Linear instead.")

        self.in_features = other.in_features
        self.out_features = other.out_features
        # Quantize weight and discard the original
        self.weight, self.col_offsets, self.scale, self.zero_point = torch.fbgemm_linear_quantize_weight(
            other.weight.clone(memory_format=torch.contiguous_format).float())
        self.weight = torch.nn.Parameter(self.weight, requires_grad=False)
        self.col_offsets = torch.nn.Parameter(self.col_offsets, requires_grad=False)
        assert other.bias is not None, 'QuantizedLinear requires a bias'
        self.bias = torch.nn.Parameter(other.bias.clone(memory_format=torch.contiguous_format).float(), requires_grad=False)

        self.register_buffer(
            'packed_tensor_ptr',
            torch.fbgemm_pack_quantized_matrix(self.weight.clone(memory_format=torch.contiguous_format)))

    @torch.jit.script_method
    def _unpack(self):
        self.packed_tensor_ptr.set_(
            torch.fbgemm_pack_quantized_matrix(self.weight))

    @torch.jit.script_method
    def _pack(self):
        self.packed_tensor_ptr.set_(
            torch.zeros(torch.jit.annotate(List[int], []), dtype=torch.uint8).detach())

    @torch.jit.script_method
    def forward(self, input):
        out = torch.fbgemm_linear_int8_weight_fp32_activation(
            input.float(), self.weight, self.packed_tensor_ptr, self.col_offsets,
            self.scale, self.zero_point, self.bias)
        return out.to(input.dtype)

    def extra_repr(self):
        repr = 'in_features={in_features}, out_features={out_features}, ' \
               'scale={scale}, zero_point={zero_point}'.format(**self.__dict__)
        return repr

# FP16 weights
class QuantizedLinearFP16(torch.jit.ScriptModule):

    def __init__(self, other):
        super(QuantizedLinearFP16, self).__init__()
        warnings.warn(
            "torch.jit.QuantizedLinearFP16 is deprecated and will be removed in an upcoming "
            "PyTorch release. Please use the torch.ao.nn.quantized.dynamic.Linear instead.")
        self.in_features = other.in_features
        self.out_features = other.out_features
        self.original_weight = other.weight
        self.weight = torch.fbgemm_pack_gemm_matrix_fp16(
            other.weight.clone(memory_format=torch.contiguous_format).float())
        assert other.bias is not None, 'QuantizedLinearFP16 requires a bias'
        self.bias = torch.nn.Parameter(other.bias.clone(memory_format=torch.contiguous_format).float(), requires_grad=False)
        self.register_buffer('packed_weight', self.weight)

    @torch.jit.script_method
    def _unpack(self):
        self.packed_weight.set_(
            torch.fbgemm_pack_gemm_matrix_fp16(
                self.original_weight))

    @torch.jit.script_method
    def _pack(self):
        self.packed_weight.set_(
            torch.zeros(torch.jit.annotate(List[int], []), dtype=torch.uint8).detach())

    @torch.jit.script_method
    def forward(self, input):
        out = torch.fbgemm_linear_fp16_weight_fp32_activation(
            input.float(), self.packed_weight, self.bias)
        return out

    def extra_repr(self):
        repr = 'in_features={in_features}, out_features={out_features}, '.format(**self.__dict__)
        return repr

# Quantized RNN cell implementations
class QuantizedRNNCellBase(torch.jit.ScriptModule):
    __constants__ = ['input_size', 'hidden_size', 'bias', 'scale_hh', 'scale_ih',
                     'zero_point_ih', 'zero_point_hh']

    def __init__(self, other):
        super(QuantizedRNNCellBase, self).__init__()
        warnings.warn(
            "torch.jit.QuantizedRNNCellBase is deprecated and will be removed in an upcoming "
            "PyTorch release. Please use the torch.ao.nn.quantized.dynamic.RNNCell instead.")

        self.input_size = other.input_size
        self.hidden_size = other.hidden_size
        self.bias = other.bias
        if not self.bias:
            raise ValueError("Quantized RNN cells require bias terms")

        weight_ih, col_offsets_ih, self.scale_ih, self.zero_point_ih = \
            torch.fbgemm_linear_quantize_weight(other.weight_ih.clone(memory_format=torch.contiguous_format).float())
        self.register_buffer('weight_ih', weight_ih)
        self.register_buffer('col_offsets_ih', col_offsets_ih)
        weight_hh, col_offsets_hh, self.scale_hh, self.zero_point_hh = \
            torch.fbgemm_linear_quantize_weight(other.weight_hh.clone(memory_format=torch.contiguous_format).float())
        self.register_buffer('weight_hh', weight_hh)
        self.register_buffer('col_offsets_hh', col_offsets_hh)

        packed_ih = torch.fbgemm_pack_quantized_matrix(self.weight_ih)
        self.register_buffer('packed_ih', packed_ih)
        packed_hh = torch.fbgemm_pack_quantized_matrix(self.weight_hh)
        self.register_buffer('packed_hh', packed_hh)

        self.bias_ih = torch.nn.Parameter(other.bias_ih.clone(memory_format=torch.contiguous_format).float(), requires_grad=False)
        self.bias_hh = torch.nn.Parameter(other.bias_hh.clone(memory_format=torch.contiguous_format).float(), requires_grad=False)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    @torch.jit.script_method
    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    @torch.jit.script_method
    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    # TODO: for some reason weak_script_method causes a destruction of the
    # module to occur, which in turn frees the packed_ih object via its DataPtr
    # deleter. This is bizarre and should probably get fixed.
    # @torch._jit_internal.weak_script_method
    @torch.jit.script_method
    def _unpack(self):
        self.packed_ih.set_(torch.fbgemm_pack_quantized_matrix(self.weight_ih))
        self.packed_hh.set_(torch.fbgemm_pack_quantized_matrix(self.weight_hh))

    # @torch._jit_internal.weak_script_method
    @torch.jit.script_method
    def _pack(self):
        self.packed_ih.set_(
            torch.zeros(torch.jit.annotate(List[int], []), dtype=torch.uint8).detach())
        self.packed_hh.set_(
            torch.zeros(torch.jit.annotate(List[int], []), dtype=torch.uint8).detach())


class QuantizedRNNCell(QuantizedRNNCellBase):
    __constants__ = ['input_size', 'hidden_size', 'bias', 'scale_hh', 'scale_ih',
                     'zero_point_ih', 'zero_point_hh', 'nonlinearity']

    def __init__(self, other):
        super(QuantizedRNNCell, self).__init__(other)
        warnings.warn(
            "torch.jit.QuantizedRNNCell is deprecated and will be removed in an upcoming "
            "PyTorch release. Please use the torch.ao.nn.quantized.dynamic.RNNCell instead.")
        self.nonlinearity = other.nonlinearity

    @torch.jit.script_method
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        if self.nonlinearity == "tanh":
            ret = _VF.quantized_rnn_tanh_cell(
                input, hx, self.weight_ih, self.weight_hh, self.bias_ih,
                self.bias_hh, self.packed_ih, self.packed_hh, self.col_offsets_ih,
                self.col_offsets_hh, self.scale_ih, self.scale_hh, self.zero_point_ih,
                self.zero_point_hh
            )
        elif self.nonlinearity == "relu":
            ret = _VF.quantized_rnn_relu_cell(
                input, hx, self.weight_ih, self.weight_hh, self.bias_ih,
                self.bias_hh, self.packed_ih, self.packed_hh, self.col_offsets_ih,
                self.col_offsets_hh, self.scale_ih, self.scale_hh, self.zero_point_ih,
                self.zero_point_hh
            )
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        return ret


class QuantizedLSTMCell(QuantizedRNNCellBase):
    def __init__(self, other):
        super(QuantizedLSTMCell, self).__init__(other)
        warnings.warn(
            "torch.jit.QuantizedLSTMCell is deprecated and will be removed in an upcoming "
            "PyTorch release. Please use the torch.ao.nn.quantized.dynamic.LSTMCell instead.")

    @torch.jit.script_method
    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        self.check_forward_input(input)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        return _VF.quantized_lstm_cell(
            input, hx, self.weight_ih, self.weight_hh, self.bias_ih,
            self.bias_hh, self.packed_ih, self.packed_hh, self.col_offsets_ih,
            self.col_offsets_hh, self.scale_ih, self.scale_hh, self.zero_point_ih,
            self.zero_point_hh
        )


class QuantizedGRUCell(QuantizedRNNCellBase):
    def __init__(self, other):
        super(QuantizedGRUCell, self).__init__(other)
        warnings.warn(
            "torch.jit.QuantizedGRUCell is deprecated and will be removed in an upcoming "
            "PyTorch release. Please use the torch.ao.nn.quantized.dynamic.GRUCell instead.")

    @torch.jit.script_method
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        return _VF.quantized_gru_cell(
            input, hx, self.weight_ih, self.weight_hh, self.bias_ih,
            self.bias_hh, self.packed_ih, self.packed_hh, self.col_offsets_ih,
            self.col_offsets_hh, self.scale_ih, self.scale_hh, self.zero_point_ih,
            self.zero_point_hh
        )


def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)


class QuantizedRNNBase(torch.jit.ScriptModule):
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional', 'dtype']

    def __init__(self, other, dtype=torch.int8):
        super(QuantizedRNNBase, self).__init__()
        warnings.warn(
            "torch.jit.QuantizedRNNBase is deprecated and will be removed in an upcoming "
            "PyTorch release. Please use the torch.ao.nn.quantized.dynamic instead.")
        self.mode = other.mode
        self.input_size = other.input_size
        self.hidden_size = other.hidden_size
        self.num_layers = other.num_layers
        self.bias = other.bias
        self.batch_first = other.batch_first
        if self.mode != 'GRU':
            assert not self.batch_first
        self.dropout = other.dropout
        self.bidirectional = other.bidirectional
        num_directions = 2 if self.bidirectional else 1
        self.dtype = dtype

        assert self.bias

        # TODO: support more than just LSTM
        if self.mode != 'LSTM' and self.mode != 'GRU':
            raise RuntimeError('Only LSTM or GRU is supported for QuantizedRNN')

        if dtype != torch.int8 and dtype != torch.float16:
            raise RuntimeError('Unsupported dtype: {}'.format(dtype))

        self.all_weights = []
        for layer in range(self.num_layers):
            for direction in range(num_directions):
                layer_input_size = self.input_size if layer == 0 else self.hidden_size * num_directions

                suffix = '_reverse' if direction == 1 else ''

                def get_weight_bias(ihhh):
                    weight_name = 'weight_{}_l{}{}'.format(ihhh, layer, suffix)
                    bias_name = 'bias_{}_l{}{}'.format(ihhh, layer, suffix)

                    weight = getattr(other, weight_name)
                    bias = getattr(other, bias_name)
                    return weight, bias

                weight_ih, bias_ih = get_weight_bias('ih')
                weight_hh, bias_hh = get_weight_bias('hh')

                if dtype == torch.int8:
                    cell_params = torch.ops.quantized.make_quantized_cell_params(
                        weight_ih, weight_hh, bias_ih, bias_hh)
                else:
                    packed_ih = torch.ops.quantized.linear_prepack_fp16(
                        weight_ih.float(), bias_ih)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(
                        weight_hh.float(), bias_hh)

                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(
                        packed_ih, packed_hh)

                setattr(self, 'cell_params_{}_{}'.format(layer, suffix), cell_params)
                self.all_weights.append(cell_params)

    @torch.jit.script_method
    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    @torch.jit.script_method
    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    @torch.jit.script_method
    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
                          msg: str = 'Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    @torch.jit.script_method
    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]) -> None:
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)
        self.check_hidden_size(hidden, expected_hidden_size, msg='Expected hidden size {}, got {}')

    @torch.jit.script_method
    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]) -> Tensor:
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)


class QuantizedLSTM(QuantizedRNNBase):
    __overloads__ = {'forward': ['forward_packed', 'forward_tensor']}

    def __init__(self, other, dtype):
        super(QuantizedLSTM, self).__init__(other, dtype)
        warnings.warn(
            "torch.jit.QuantizedLSTM is deprecated and will be removed in an upcoming "
            "PyTorch release. Please use the torch.ao.nn.quantized.dynamic.LSTM instead.")

    @torch.jit.script_method
    def forward_impl(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]], batch_sizes: Optional[Tensor],
                     max_batch_size: int, sorted_indices: Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
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
        assert batch_sizes is None
        result = torch.quantized_lstm(input, hx, self.all_weights, self.bias, self.num_layers,
                                      float(self.dropout), self.training, self.bidirectional,
                                      self.batch_first, dtype=self.dtype, use_dynamic=False)
        output = result[0]
        hidden = result[1:]

        return output, hidden

    @torch.jit.script_method
    def forward_tensor(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        batch_sizes = None
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None

        output, hidden = self.forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)

        return output, self.permute_hidden(hidden, unsorted_indices)

    @torch.jit.script_method
    def forward_packed(self, input: PackedSequence, hx: Optional[Tuple[Tensor, Tensor]] = None
                       ) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]:
        input, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = batch_sizes[0]
        max_batch_size = int(max_batch_size)

        output, hidden = self.forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)

        output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)


    @torch.jit.script_method
    def permute_hidden(self, hx: Tuple[Tensor, Tensor], permutation: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if permutation is None:
            return hx
        return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)

    @torch.jit.script_method
    def check_forward_args(self, input: Tensor, hidden: Tuple[Tensor, Tensor], batch_sizes: Optional[Tensor]) -> None:
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden[0], expected_hidden_size,
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], expected_hidden_size,
                               'Expected hidden[1] size {}, got {}')

    def forward(self, input, hx=None):
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, hx)
        else:
            return self.forward_tensor(input, hx)


class QuantizedGRU(QuantizedRNNBase):
    __overloads__ = {'forward': ['forward_packed', 'forward_tensor']}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "torch.jit.QuantizedGRU is deprecated and will be removed in an upcoming "
            "PyTorch release. Please use the torch.ao.nn.quantized.dynamic.GRU instead.")


    @torch.jit.script_method
    def forward_impl(self, input: Tensor, hx: Optional[Tensor], batch_sizes: Optional[Tensor], max_batch_size: int,
                     sorted_indices: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = torch.quantized_gru(input, hx, self.all_weights, self.bias, self.num_layers,
                                         float(self.dropout), self.training, self.bidirectional,
                                         self.batch_first)
        else:
            result = torch.quantized_gru(input, batch_sizes, hx, self.all_weights, self.bias, self.num_layers,
                                         float(self.dropout), self.training, self.bidirectional)

        output = result[0]
        hidden = result[1]

        return output, hidden

    @torch.jit.script_method
    def forward_tensor(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_sizes = None
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None

        output, hidden = self.forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    @torch.jit.script_method
    def forward_packed(self, input: PackedSequence, hx: Optional[Tensor] = None) -> Tuple[PackedSequence, Tensor]:
        input, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = batch_sizes[0]
        max_batch_size = int(max_batch_size)

        output, hidden = self.forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)

        output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    def forward(self, input, hx=None):
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, hx)
        else:
            return self.forward_tensor(input, hx)


def quantize_rnn_cell_modules(module):
    warnings.warn("quantize_rnn_cell_modules function has been deprecated. "
                  "Please use torch.ao.quantization.quantize_dynamic API instead.")
    reassign = {}
    for name, mod in module.named_modules():
        if mod is module:
            continue
        new_mod = quantize_rnn_cell_modules(mod)
        if new_mod is not mod:
            reassign[name] = new_mod
    for name, mod in reassign.items():
        setattr(module, name, mod)
    if isinstance(module, torch.nn.LSTMCell):
        return QuantizedLSTMCell(module)
    if isinstance(module, torch.nn.GRUCell):
        return QuantizedGRUCell(module)
    if isinstance(module, torch.nn.RNNCell):
        return QuantizedRNNCell(module)
    return module


def quantize_linear_modules(module, dtype=torch.int8):
    warnings.warn("quantize_linear_modules function has been deprecated. "
                  "Please use torch.ao.quantization.quantize_dynamic API instead.")

    reassign = {}
    for name, mod in module.named_modules():
        if mod is module:
            continue
        new_mod = quantize_linear_modules(mod, dtype)
        if new_mod is not mod:
            reassign[name] = new_mod

    for name, mod in reassign.items():
        setattr(module, name, mod)
    if isinstance(module, torch.nn.Linear):
        if dtype == torch.int8:
            return QuantizedLinear(module)
        elif dtype == torch.float16:
            return QuantizedLinearFP16(module)
        else:
            raise RuntimeError(
                "Unsupported dtype: {}".format(dtype))
    return module


def quantize_rnn_modules(module, dtype=torch.int8):
    warnings.warn("quantize_rnn_modules function has been deprecated. "
                  "Please use torch.ao.quantization.quantize_dynamic API instead.")
    reassign = {}
    for name, mod in module.named_modules():
        if mod is module:
            continue
        new_mod = quantize_rnn_modules(mod, dtype)
        if new_mod is not mod:
            reassign[name] = new_mod

    for name, mod in reassign.items():
        setattr(module, name, mod)
    if isinstance(module, torch.nn.LSTM):
        if dtype != torch.int8 and dtype != torch.float16:
            raise RuntimeError("Unsupported dtype: {}".format(dtype))
        return QuantizedLSTM(module, dtype)
    if isinstance(module, torch.nn.GRU):
        return QuantizedGRU(module)
    return module
