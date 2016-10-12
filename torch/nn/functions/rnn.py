from torch.autograd import Function
from torch._thnn import type2backend
import torch.backends.cudnn as cudnn
try:
    import torch.backends.cudnn.rnn
except ImportError:
    print "Couldn't import cudnn.rnn"
    pass
import torch.backends.cudnn.rnn




def _getCudnnMode(mode):
    if mode == 'RNN_RELU':
        return cudnn.CUDNN_RNN_RELU
    elif mode == 'RNN_TANH':
        return cudnn.CUDNN_RNN_TANH
    elif mode == 'LSTM':
        return cudnn.CUDNN_LSTM
    elif mode == 'GRU':
        return cudnn.CUDNN_GRU
    else:
        raise Exception("Unknown mode: {}".format(mode))

import thnn
import linear

ReLU = thnn.Threshold(0, 0)
tanh = thnn.Tanh
linear = linear.Linear
sigmoid = thnn.Sigmoid

def RNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hy = ReLU(linear(input, w_ih, b_ih) +
                  linear(hidden, w_hh, b_hh))
        return hy, hy

def RNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hy = tanh(linear(input, w_ih, b_ih) +
                  linear(hidden, w_hh, b_hh))
        return hy, hy

def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hx, cx = hidden
        hsz = hx.size(1)
        gates = linear(input, w_ih, b_ih) + linear(hx, w_hh, b_hh)
        # FIXME: chunk
        ingate     = sigmoid(gates[:,0*hsz:1*hsz])
        forgetgate = sigmoid(gates[:,1*hsz:2*hsz])
        cellgate   = tanh(   gates[:,2*hsz:3*hsz])
        outgate    = sigmoid(gates[:,3*hsz:4*hsz])
        nextc = (forgetgate * c) + (ingate * cellgate)
        nexth = outgate * tanh(nextc)

        return [nexth, nextc], nexth

def GRU(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hsz = hx.size(1)
        gi = linear(input, w_ih, b_ih)
        gh = linear(hidden, w_hh, b_hh)
        # FIXME: chunk
        resetgate = sigmoid(gi[:,0*hsz:1*hsz] + gh[:,0*hsz:1*hsz])
        inputgate = sigmoid(gi[:,0*hsz:1*hsz] + gh[:,0*hsz:1*hsz])
        output    = tanh(gi[2] + resetgate * gh[2])
        nexth     = output + inputgate * (hidden - output)

        return nexth, output  # FIXME: nexth, nexth ???

def StackedRNN(cell, num_layers):
    def forward(input, hx, weight):
        assert(len(weight) == num_layers)
        next_hidden = []
        for i in range(num_layers):
            hy, input = cell(input, hidden[i], weight[i])
            next_hidden.append(hy)

        return next_hidden, input

    return forward

def Recurrent(rnn):
    def forward(input, hidden, weight):
        output = None
        for i in range(input.size(0)):
            hidden, y = rnn(input[i], hidden, weight)
            if not output:
                output = input.new(input.size(0), *y.size())
            output[i] = y

        return hidden, output

    return forward

def THNN_RNN(self, mode, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0, train=True, bidirectional=False):
    if num_layers != 1:
        raise NotImplementedError()
    if bidirectional:
        raise NotImplementedError()
    if mode != 'RNN_RELU':
        raise NotImplementedError()
    if dropout != 0:
        raise NotImplementedError()

    if mode == 'RNN_RELU':
        cell = RNNReLUCell
    elif mode == 'RNN_TANH':
        cell = RNNTanhCell
    elif mode == 'RNN_LSTM':
        cell = LSTMCell
    elif mode == 'RNN_GRU':
        cell = GRUCell
    else:
        raise Exception('Unknown mode: {}'.format(mode))

    func = Recurrent(StackedRNN(cell, num_layers))

    def forward(self, input, weight, hidden):
        if batch_first:
            input.transpose(0, 1)

        nexth, output = func(input, hx, hidden)

        if batch_first:
            output.transpose(0, 1)

        return output, nexth

    return forward


class CudnnRNN(Function):
    def __init__(self, mode, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0, train=True, bidirectional=False):
        super(CudnnRNN, self).__init__()
        self.mode = _getCudnnMode(mode)
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

    def forward(self, input, weight, hx, cx=None):
        assert(cudnn.is_acceptable(input))

        output = input.new()

        hy = hx.new()
        cy = cx.new() if cx else None

        cudnn.rnn.forward(self, input, hx, cx, weight, output, hy, cy)

        if cx is not None:
            self.save_for_backward(input, hx, cx, weight, output)
            return output, hy, cy
        else:
            self.save_for_backward(input, hx, weight, output)
            return output, hy


    def backward(self, grad_output, grad_hy, grad_cy=None):
        tensors = self.saved_tensors
        if len(tensors) == 5:
            input, hx, cx, weight, output = tensors

        else:
            input, hx, weight, output = tensors
            cx = None

        grad_input, grad_weight, grad_hx, grad_cx = None, None, None, None

        assert(cudnn.is_acceptable(input))

        grad_input = weight.new()
        grad_weight = weight.new()
        grad_hx = weight.new()
        if cx:
            grad_cx = weight.new()

        cudnn.rnn.backward_grad(
            self,
            input,
            hx, cx,
            weight,
            output,
            grad_output,
            grad_hy, grad_cy,
            grad_input,
            grad_hx, grad_cx)

        if self.needs_input_grad[1]:
            grad_weight = weight.new()
            cudnn.rnn.backward_weight(
                self,
                input,
                hx,
                output,
                weight,
                grad_weight)

        # FIXME: zero out grad_bias if necessary :)

        if cx is not None:
            return grad_input, grad_weight, grad_hx, grad_cx
        else:
            return grad_input, grad_weight, grad_hx


def RNN(*args, **kwargs):
    def forward(input, *fargs, **fkwargs):
        if cudnn.is_acceptable(input.data):
            func = CudnnRNN(*args, **kwargs)
        else:
            func = THNN_RNN(*args, **kwargs)
        return func(input, *fargs, **fkwargs)

    return forward
