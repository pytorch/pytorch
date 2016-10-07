import math
import torch
from torch.autograd import Variable

from .module import Module
from .utils import _pair, _triple


class RNN(Module):
    # FIXME: docstring

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0):
        num_weights_hh = num_layers * hidden_size**2
        num_weights_ih = input_size * hidden_size + (num_layers - 1) * hidden_size**2
        num_biases = num_layers * hidden_size * 2  # 2 for ih and hh (?)
        num_weights = num_weights_hh + num_weights_ih + num_biases

        super(RNN, self).__init__(
            weight=torch.cuda.FloatTensor(num_weights)
        )
        self.hidden_size = hidden_size
        self.func = self._backend.RNN(
            'RNN_RELU', input_size, hidden_size, num_layers, batch_first, dropout,
            True,  #FIXME train,
            False,  #FIXME bidirectional
        )
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.weight.data.uniform_(-stdv, stdv)


    def forward(self, input, hx):
        return self.func(input, self.weight, hx)
