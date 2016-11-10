from torch.autograd import Function, NestedIOFunction, Variable
from torch._thnn import type2backend
import torch.backends.cudnn as cudnn
try:
    import torch.backends.cudnn.rnn
except ImportError:
    pass


# FIXME: write a proper function library
from .thnn import Tanh, Sigmoid, Threshold
from .linear import Linear
from .dropout import Dropout


def _wrap(fn, *args):
    def inner(*inner_args):
        return fn(*args)(*inner_args)
    return inner
tanh = _wrap(Tanh)
sigmoid = _wrap(Sigmoid)
ReLU = _wrap(Threshold, 0, 0, False)


# get around autograd's lack of None-handling
def linear(input, w, b):
    if b is not None:
        return Linear()(input, w, b)
    else:
        return Linear()(input, w)


def RNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hy = ReLU(linear(input, w_ih, b_ih) + linear(hidden, w_hh, b_hh))
        return hy


def RNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hy = tanh(linear(input, w_ih, b_ih) + linear(hidden, w_hh, b_hh))
        return hy


def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hx, cx = hidden
        gates = linear(input, w_ih, b_ih) + linear(hx, w_hh, b_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = sigmoid(ingate)
        forgetgate = sigmoid(forgetgate)
        cellgate = tanh(cellgate)
        outgate = sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * tanh(cy)

        return hy, cy


def GRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        gi = linear(input, w_ih, b_ih)
        gh = linear(hidden, w_hh, b_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = sigmoid(i_r + h_r)
        inputgate = sigmoid(i_i + h_i)
        newgate = tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        return hy


def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):

    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight):
        assert(len(weight) == total_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j

                hy, output = inner(input, hidden[l], weight[l])
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, 2)

            if dropout != 0 and i < num_layers - 1:
                input = Dropout(p=dropout, train=train, inplace=False)(input)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward

def Recurrent(inner, reverse=False):
    def forward(input, hidden, weight):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            # hack to handle LSTM
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def AutogradRNN(mode, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0, train=True, bidirectional=False):

    if mode == 'RNN_RELU':
        cell = RNNReLUCell
    elif mode == 'RNN_TANH':
        cell = RNNTanhCell
    elif mode == 'LSTM':
        cell = LSTMCell
    elif mode == 'GRU':
        cell = GRUCell
    else:
        raise Exception('Unknown mode: {}'.format(mode))

    if bidirectional:
        layer = (Recurrent(cell), Recurrent(cell, reverse=True))
    else:
        layer = (Recurrent(cell),)

    func = StackedRNN(layer,
                      num_layers,
                      (mode == 'LSTM'),
                      dropout=dropout,
                      train=train)

    def forward(input, weight, hidden):
        if batch_first:
            input.transpose(0, 1)

        nexth, output = func(input, hidden, weight)

        if batch_first:
            output.transpose(0, 1)

        return output, nexth

    return forward


class CudnnRNN(NestedIOFunction):
    def __init__(self, mode, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0, train=True, bidirectional=False):
        super(CudnnRNN, self).__init__()
        self.mode = cudnn.rnn.get_cudnn_mode(mode)
        self.input_mode = cudnn.CUDNN_LINEAR_INPUT
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.train = train
        self.bidirectional = 1 if bidirectional else 0
        self.num_directions = 2 if bidirectional else 1
        self.seed = torch.IntTensor(1).random_()[0]

    def forward_extended(self, input, weight, hx):

        assert(cudnn.is_acceptable(input))

        output = input.new()

        if torch.is_tensor(hx):
            hy = hx.new()
        else:
            hy = tuple(h.new() for h in hx)

        cudnn.rnn.forward(self, input, hx, weight, output, hy)

        self.save_for_backward(input, hx, weight, output)
        return output, hy


    def backward_extended(self, grad_output, grad_hy):
        input, hx, weight, output = self.saved_tensors

        grad_input, grad_weight, grad_hx = None, None, None

        assert(cudnn.is_acceptable(input))

        grad_input = input.new()
        grad_weight = input.new()
        grad_hx = input.new()
        if torch.is_tensor(hx):
            grad_hx = input.new()
        else:
            grad_hx = tuple(h.new() for h in hx)

        cudnn.rnn.backward_grad(
            self,
            input,
            hx,
            weight,
            output,
            grad_output,
            grad_hy,
            grad_input,
            grad_hx)

        if self.needs_input_grad[1]:
            grad_weight = [tuple(w.new().resize_as_(w).zero_() for w in layer_weight) for layer_weight in weight]
            cudnn.rnn.backward_weight(
                self,
                input,
                hx,
                output,
                weight,
                grad_weight)

        return grad_input, grad_weight, grad_hx


def RNN(*args, **kwargs):
    def forward(input, *fargs, **fkwargs):
        if cudnn.is_acceptable(input.data):
            func = CudnnRNN(*args, **kwargs)
        else:
            func = AutogradRNN(*args, **kwargs)
        return func(input, *fargs, **fkwargs)

    return forward
