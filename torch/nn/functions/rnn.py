from torch.autograd import Function, NestedIOFunction, Variable
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



# FIXME: write a proper function library!
import thnn
import linear
import activation
def _wrap(fn, *args):
    def inner(*inner_args):
        return fn(*args)(*inner_args)
    return inner
tanh = _wrap(thnn.Tanh)
linear = _wrap(linear.Linear)
sigmoid = _wrap(thnn.Sigmoid)
ReLU = _wrap(thnn.Threshold, 0, 0, False)

def RNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hy = ReLU(linear(input, w_ih, b_ih) + linear(hidden, w_hh, b_hh))
        return hy, hy

def RNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hy = tanh(linear(input, w_ih, b_ih) + linear(hidden, w_hh, b_hh))
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
        nextc = (forgetgate * cx) + (ingate * cellgate)
        nexth = outgate * tanh(nextc)

        return (nexth, nextc), nexth

def GRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hsz = hidden.size(1)
        gi = linear(input, w_ih, b_ih)
        gh = linear(hidden, w_hh, b_hh)
        # FIXME: chunk

        # this is a bit weird, it doesn't match the order of parameters
        # implied by the cudnn docs, and it also uses nexth for output...
        resetgate = sigmoid(gi[:,0*hsz:1*hsz] + gh[:,0*hsz:1*hsz])
        updategate = sigmoid(gi[:,1*hsz:2*hsz] + gh[:,1*hsz:2*hsz])
        output    = tanh(gi[:,2*hsz:3*hsz] + resetgate * gh[:,2*hsz:3*hsz])
        nexth     = output + updategate * (hidden - output)

        return nexth, nexth  # FIXME: nexth, nexth ???

def StackedRNN(cell, num_layers, lstm=False):
    def forward(input, hidden, weight):
        assert(len(weight) == num_layers)
        next_hidden = []

        if lstm:
            hidden = zip(*hidden)

        for i in range(num_layers):
            hy, input = cell(input, hidden[i], *weight[i])
            next_hidden.append(hy)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                input.cat(next_h, 0).view(num_layers, *next_h[0].size()),
                input.cat(next_c, 0).view(num_layers, *next_c[0].size())
            )
        else:
            next_hidden = input.cat(next_hidden, 0).view(
                num_layers, *next_hidden[0].size()) # FIXME: why input.cat???

        return next_hidden, input

    return forward

def Recurrent(rnn):
    def forward(input, hidden, weight):
        output = []
        for i in range(input.size(0)):
            hidden, y = rnn(input[i], hidden, weight)
            output.append(y)

        output = input.cat(output, 0).view(input.size(0), *output[0].size())  # yikes!
        return hidden, output

    return forward

def THNN_RNN(mode, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0, train=True, bidirectional=False):
    if bidirectional:
        raise NotImplementedError()
    if dropout != 0:
        raise NotImplementedError()

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

    func = Recurrent(StackedRNN(cell, num_layers, (mode == 'LSTM')))

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

        # FIXME: zero out grad_bias if necessary :)

        return grad_input, grad_weight, grad_hx


def RNN(*args, **kwargs):
    def forward(input, *fargs, **fkwargs):
        if cudnn.is_acceptable(input.data):
            func = CudnnRNN(*args, **kwargs)
        else:
            func = THNN_RNN(*args, **kwargs)
        return func(input, *fargs, **fkwargs)

    return forward
