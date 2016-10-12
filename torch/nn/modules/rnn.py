import math
import torch
from torch.autograd import Variable

from .module import Module
from .utils import _pair, _triple


class RNN(Module):
    # FIXME: docstring

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        self.all_weights = []
        super_weights = {}
        for layer in range(num_layers):
            # FIXME: sizes are different for LSTM/GRU
            layer_input_size = input_size if layer == 0 else hidden_size
            w_ih = Variable(torch.Tensor(layer_input_size, hidden_size), requires_grad=True)
            w_hh = Variable(torch.Tensor(hidden_size, hidden_size), requires_grad=True)
            b_ih = Variable(torch.Tensor(hidden_size), requires_grad=True)
            b_hh = Variable(torch.Tensor(hidden_size), requires_grad=True)
            self.all_weights += [(w_ih, w_hh, b_ih, b_hh)]

            super_weights['l{}_w_ih'.format(layer)] = w_ih
            super_weights['l{}_w_hh'.format(layer)] = w_hh
            super_weights['l{}_b_ih'.format(layer)] = b_ih
            super_weights['l{}_b_hh'.format(layer)] = b_hh

        super(RNN, self).__init__(
            **super_weights
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for layer_weights in self.all_weights:
            for weight in layer_weights:
                weight.data.uniform_(-stdv, stdv)


    def forward(self, input, hx):
        func = self._backend.RNN(
            'RNN_RELU',
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.batch_first,
            self.dropout,
            train=True,  #FIXME
            bidirectional=False,  #FIXME
        )

        return func(input, self.all_weights, hx)
