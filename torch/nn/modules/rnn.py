import math
import torch
from torch.autograd import Variable

from .module import Module
from .utils import _pair, _triple


class RNNBase(Module):
    # FIXME: docstring

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False, dropout=0):
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        self.all_weights = []
        super_weights = {}
        for layer in range(num_layers):
            # FIXME: sizes are different for LSTM/GRU
            layer_input_size = input_size if layer == 0 else hidden_size
            if mode == 'LSTM':
                gate_size = 4 * hidden_size
            elif mode == 'GRU':
                gate_size = 3 * hidden_size
            else:
                gate_size = hidden_size

            w_ih = Variable(torch.Tensor(gate_size, layer_input_size), requires_grad=True)
            w_hh = Variable(torch.Tensor(gate_size, hidden_size), requires_grad=True)
            b_ih = Variable(torch.Tensor(gate_size), requires_grad=True)
            b_hh = Variable(torch.Tensor(gate_size), requires_grad=True)

            super_weights['weight_ih_l{}'.format(layer)] = w_ih
            super_weights['weight_hh_l{}'.format(layer)] = w_hh
            if bias:
                super_weights['bias_ih_l{}'.format(layer)] = b_ih
                super_weights['bias_hh_l{}'.format(layer)] = b_hh
                self.all_weights += [(w_ih, w_hh, b_ih, b_hh)]
            else:
                self.all_weights += [(w_ih, w_hh)]

        super(RNNBase, self).__init__(
            **super_weights
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, input, hx):
        func = self._backend.RNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.batch_first,
            self.dropout,
            train=True,  #FIXME
            bidirectional=False,  #FIXME
        )

        return func(input, self.all_weights, hx)


class RNN(RNNBase):
    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__('RNN_TANH', *args, **kwargs)

class RNNReLU(RNNBase):
    def __init__(self, *args, **kwargs):
        super(RNNReLU, self).__init__('RNN_RELU', *args, **kwargs)

class LSTM(RNNBase):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

class GRU(RNNBase):
    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__('GRU', *args, **kwargs)
