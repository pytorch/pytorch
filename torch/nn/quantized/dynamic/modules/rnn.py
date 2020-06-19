from __future__ import absolute_import, division, print_function, unicode_literals

import numbers
import warnings

import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.nn.quantized.modules.utils import _quantize_weight

def apply_permutation(tensor, permutation, dim=1):
    # type: (Tensor, Tensor, int) -> Tensor
    return tensor.index_select(dim, permutation)

class PackedParameter(torch.nn.Module):
    def __init__(self, param):
        super(PackedParameter, self).__init__()
        self.param = param

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(PackedParameter, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'param'] = self.param

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.param = state_dict[prefix + 'param']
        super(PackedParameter, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                           missing_keys, unexpected_keys, error_msgs)

class RNNBase(torch.nn.Module):

    _FLOAT_MODULE = nn.RNNBase

    _version = 2

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
        self.version = 2
        self.training = False
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

        _all_weight_values = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = torch.randn(gate_size, layer_input_size).to(torch.float)
                w_hh = torch.randn(gate_size, hidden_size).to(torch.float)
                b_ih = torch.randn(gate_size).to(torch.float)
                b_hh = torch.randn(gate_size).to(torch.float)
                if dtype == torch.qint8:
                    w_ih = torch.quantize_per_tensor(w_ih, scale=0.1, zero_point=0, dtype=torch.qint8)
                    w_hh = torch.quantize_per_tensor(w_hh, scale=0.1, zero_point=0, dtype=torch.qint8)
                    packed_ih = \
                        torch.ops.quantized.linear_prepack(w_ih, b_ih)
                    packed_hh = \
                        torch.ops.quantized.linear_prepack(w_hh, b_hh)
                    if self.version is None or self.version < 2:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(
                            packed_ih, packed_hh, b_ih, b_hh)
                    else:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(
                            packed_ih, packed_hh, b_ih, b_hh, True)

                else:

                    packed_ih = torch.ops.quantized.linear_prepack_fp16(w_ih, b_ih)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(w_hh, b_hh)
                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(
                        packed_ih, packed_hh)

                _all_weight_values.append(PackedParameter(cell_params))
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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        self.version = version
        super(RNNBase, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                   missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_float(cls, mod):
        assert type(mod) in set(
            [torch.nn.LSTM,
             torch.nn.GRU]
        ), 'nn.quantized.dynamic.RNNBase.from_float only works for nn.LSTM and nn.GRU'
        assert hasattr(
            mod,
            'qconfig'
        ), 'Input float module must have qconfig defined'

        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer_method = mod.qconfig.weight
        else:
            # We have the circular import issues if we import the qconfig in the beginning of this file:
            # https://github.com/pytorch/pytorch/pull/24231. The current workaround is to postpone the
            # import until we need it.
            from torch.quantization.qconfig import default_dynamic_qconfig
            weight_observer_method = default_dynamic_qconfig.weight

        dtype = weight_observer_method().dtype
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

        _all_weight_values = []
        for layer in range(qRNNBase.num_layers):
            for direction in range(num_directions):
                layer_input_size = qRNNBase.input_size if layer == 0 else qRNNBase.hidden_size * num_directions

                suffix = '_reverse' if direction == 1 else ''

                def retrieve_weight_bias(ihhh):
                    weight_name = 'weight_{}_l{}{}'.format(ihhh, layer, suffix)
                    bias_name = 'bias_{}_l{}{}'.format(ihhh, layer, suffix)
                    weight = getattr(mod, weight_name)
                    bias = getattr(mod, bias_name)
                    return weight, bias

                weight_ih, bias_ih = retrieve_weight_bias('ih')
                weight_hh, bias_hh = retrieve_weight_bias('hh')

                if dtype == torch.qint8:
                    def quantize_and_pack(w, b):
                        weight_observer = weight_observer_method()
                        weight_observer(w)
                        qweight = _quantize_weight(w.float(), weight_observer)
                        packed_weight = \
                            torch.ops.quantized.linear_prepack(qweight, b)
                        return packed_weight
                    packed_ih = quantize_and_pack(weight_ih, bias_ih)
                    packed_hh = quantize_and_pack(weight_hh, bias_hh)
                    if qRNNBase.version is None or qRNNBase.version < 2:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(
                            packed_ih, packed_hh, bias_ih, bias_hh)
                    else:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(
                            packed_ih, packed_hh, bias_ih, bias_hh, True)

                elif dtype == torch.float16:
                    packed_ih = torch.ops.quantized.linear_prepack_fp16(
                        weight_ih.float(), bias_ih)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(
                        weight_hh.float(), bias_hh)

                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(
                        packed_ih, packed_hh)
                else:
                    raise RuntimeError('Unsupported dtype specified for dynamic quantized LSTM!')

                _all_weight_values.append(PackedParameter(cell_params))
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

        _all_params = ([m.param for m in self._all_weight_values])
        if batch_sizes is None:
            result = torch.quantized_lstm(input, hx, _all_params, self.bias, self.num_layers,
                                          float(self.dropout), self.training, self.bidirectional,
                                          self.batch_first, dtype=self.dtype, use_dynamic=True)
        else:
            result = torch.quantized_lstm(input, batch_sizes, hx, _all_params, self.bias,
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



class RNNCellBase(torch.nn.Module):
    # _FLOAT_MODULE = nn.CellRNNBase
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias=True, num_chunks=4, dtype=torch.qint8):
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if bias:
            self.bias_ih = torch.randn(num_chunks * hidden_size).to(dtype=torch.float)
            self.bias_hh = torch.randn(num_chunks * hidden_size).to(dtype=torch.float)
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        weight_ih = torch.randn(num_chunks * hidden_size, input_size).to(torch.float)
        weight_hh = torch.randn(num_chunks * hidden_size, hidden_size).to(torch.float)
        if dtype == torch.qint8:
            weight_ih = torch.quantize_per_tensor(weight_ih, scale=1, zero_point=0, dtype=torch.qint8)
            weight_hh = torch.quantize_per_tensor(weight_hh, scale=1, zero_point=0, dtype=torch.qint8)

        if dtype == torch.qint8:
            # for each layer, for each direction we need to quantize and pack
            # weights and pack parameters in this order:
            #
            #   w_ih, w_hh
            packed_weight_ih = \
                torch.ops.quantized.linear_prepack(weight_ih, self.bias_ih)
            packed_weight_hh = \
                torch.ops.quantized.linear_prepack(weight_hh, self.bias_hh)
        else:
            # for each layer, for each direction we need to quantize and pack
            # weights and pack parameters in this order:
            #
            #   packed_ih, packed_hh, b_ih, b_hh
            packed_weight_ih = torch.ops.quantized.linear_prepack_fp16(
                weight_ih, self.bias_ih)
            packed_weight_hh = torch.ops.quantized.linear_prepack_fp16(
                weight_hh, self.bias_hh)

        self._packed_weight_ih = packed_weight_ih
        self._packed_weight_hh = packed_weight_hh

    def _get_name(self):
        return 'DynamicQuantizedRNNBase'

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        # type: (Tensor, Tensor, str) -> None
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    @classmethod
    def from_float(cls, mod):
        assert type(mod) in set([torch.nn.LSTMCell,
                                 torch.nn.GRUCell,
                                 torch.nn.RNNCell]), 'nn.quantized.dynamic.RNNCellBase.from_float \
                                 only works for nn.LSTMCell, nn.GRUCell and nn.RNNCell'
        assert hasattr(
            mod, 'qconfig'), 'Input float module must have qconfig defined'

        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer_method = mod.qconfig.weight
        else:
            # We have the circular import issues if we import the qconfig in the beginning of this file:
            # https://github.com/pytorch/pytorch/pull/24231. The current workaround is to postpone the
            # import until we need it.
            from torch.quantization.qconfig import default_dynamic_qconfig
            weight_observer_method = default_dynamic_qconfig.weight

        dtype = weight_observer_method().dtype
        supported_scalar_types = [torch.qint8, torch.float16]
        if dtype not in supported_scalar_types:
            raise RuntimeError('Unsupported dtype for dynamic RNN quantization: {}'.format(dtype))

        if type(mod) == torch.nn.LSTMCell:
            qRNNCellBase = LSTMCell(mod.input_size, mod.hidden_size, bias=mod.bias, dtype=dtype)
        elif type(mod) == torch.nn.GRUCell:
            qRNNCellBase = GRUCell(mod.input_size, mod.hidden_size, bias=mod.bias, dtype=dtype)
        elif type(mod) == torch.nn.RNNCell:
            qRNNCellBase = RNNCell(mod.input_size, mod.hidden_size, bias=mod.bias, nonlinearity=mod.nonlinearity, dtype=dtype)
        else:
            raise NotImplementedError('Only LSTMCell, GRUCell and RNNCell \
            are supported for QuantizedRNN for now')


        assert mod.bias

        def process_weights(weight, bias, dtype):

            if dtype == torch.qint8:
                # for each layer, for each direction we need to quantize and pack
                # weights and pack parameters in this order:
                #
                #   w_ih, w_hh
                weight_observer = weight_observer_method()
                weight_observer(weight)
                qweight = _quantize_weight(weight.float(), weight_observer)
                packed_weight = \
                    torch.ops.quantized.linear_prepack(qweight, bias)

                return packed_weight
            else:
                # for each layer, for each direction we need to quantize and pack
                # weights and pack parameters in this order:
                #
                #   packed_ih, packed_hh, b_ih, b_hh
                packed_weight = torch.ops.quantized.linear_prepack_fp16(
                    weight.float(), bias)

                return packed_weight

        qRNNCellBase._packed_weight_ih = process_weights(mod.weight_ih, mod.bias_ih, dtype)
        qRNNCellBase._packed_weight_hh = process_weights(mod.weight_hh, mod.bias_hh, dtype)
        return qRNNCellBase


class RNNCell(RNNCellBase):
    r"""An Elman RNN cell with tanh or ReLU non-linearity.

    .. math::

        h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})

    If :attr:`nonlinearity` is `'relu'`, then ReLU is used in place of tanh.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``

    Inputs: input, hidden
        - **input** of shape `(batch, input_size)`: tensor containing input features
        - **hidden** of shape `(batch, hidden_size)`: tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch

    Shape:
        - Input1: :math:`(N, H_{in})` tensor containing input features where
          :math:`H_{in}` = `input_size`
        - Input2: :math:`(N, H_{out})` tensor containing the initial hidden
          state for each element in the batch where :math:`H_{out}` = `hidden_size`
          Defaults to zero if not provided.
        - Output: :math:`(N, H_{out})` tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    Examples::

        >>> rnn = nn.RNNCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx = rnn(input[i], hx)
                output.append(hx)
    """
    __constants__ = ['input_size', 'hidden_size', 'bias', 'nonlinearity']

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh", dtype=torch.qint8):
        super(RNNCell, self).__init__(input_size, hidden_size, bias, num_chunks=1, dtype=dtype)
        self.nonlinearity = nonlinearity

    def _get_name(self):
        return 'DynamicQuantizedRNNCell'

    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        if self.nonlinearity == "tanh":
            ret = torch.ops.quantized.quantized_rnn_tanh_cell_dynamic(
                input, hx,
                self._packed_weight_ih, self._packed_weight_hh,
                self.bias_ih, self.bias_hh)
        elif self.nonlinearity == "relu":
            ret = torch.ops.quantized.quantized_rnn_relu_cell_dynamic(
                input, hx,
                self._packed_weight_ih, self._packed_weight_hh,
                self.bias_ih, self.bias_hh)
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        return ret

    @classmethod
    def from_float(cls, mod):
        return super(RNNCell, cls).from_float(mod)


class LSTMCell(RNNCellBase):
    r"""A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_size)`: tensor containing input features
        - **h_0** of shape `(batch, hidden_size)`: tensor containing the initial hidden
          state for each element in the batch.
        - **c_0** of shape `(batch, hidden_size)`: tensor containing the initial cell state
          for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

    Outputs: (h_1, c_1)
        - **h_1** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch
        - **c_1** of shape `(batch, hidden_size)`: tensor containing the next cell state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    Examples::

        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx, cx = rnn(input[i], (hx, cx))
                output.append(hx)
    """

    def __init__(self, *args, **kwargs):
        super(LSTMCell, self).__init__(*args, num_chunks=4, **kwargs)

    def _get_name(self):
        return 'DynamicQuantizedLSTMCell'

    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]
        self.check_forward_input(input)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        return torch.ops.quantized.quantized_lstm_cell_dynamic(
            input, hx,
            self._packed_weight_ih, self._packed_weight_hh,
            self.bias_ih, self.bias_hh)

    @classmethod
    def from_float(cls, mod):
        return super(LSTMCell, cls).from_float(mod)

class GRUCell(RNNCellBase):
    r"""A gated recurrent unit (GRU) cell

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, hidden
        - **input** of shape `(batch, input_size)`: tensor containing input features
        - **hidden** of shape `(batch, hidden_size)`: tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch

    Shape:
        - Input1: :math:`(N, H_{in})` tensor containing input features where
          :math:`H_{in}` = `input_size`
        - Input2: :math:`(N, H_{out})` tensor containing the initial hidden
          state for each element in the batch where :math:`H_{out}` = `hidden_size`
          Defaults to zero if not provided.
        - Output: :math:`(N, H_{out})` tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    Examples::

        >>> rnn = nn.GRUCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx = rnn(input[i], hx)
                output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True, dtype=torch.qint8):
        super(GRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3, dtype=dtype)

    def _get_name(self):
        return 'DynamicQuantizedGRUCell'

    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        return torch.ops.quantized.quantized_gru_cell_dynamic(
            input, hx,
            self._packed_weight_ih, self._packed_weight_hh,
            self.bias_ih, self.bias_hh,
        )

    @classmethod
    def from_float(cls, mod):
        return super(GRUCell, cls).from_float(mod)
