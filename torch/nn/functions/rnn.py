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

def THNN_RNN(self, mode, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0, train=True, bidirectional=False):
    if num_layers != 1:
        raise NotImplementedError()
    if bidirectional:
        raise NotImplementedError()
    if mode != 'RNN_RELU':
        raise NotImplementedError()
    if dropout != 0:
        raise NotImplementedError()

    def forward(self, input, weight, hx, cx=None):
        if batch_first:
            input.transpose(0, 1)
        seq_len = input.size(0)
        for i in range(seq_len):
            if mode == 'RNN_RELU':
                hx = nn.ReLU(XXX)
            else:
                raise Exception('Unknown mode: {}'.format(mode))
        if batch_first:
            output.transpose(0, 1)
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
        # ???
        # we can't use the same seed at every iteration
        self.seed = torch.IntTensor(1).random_()[0]

    def forward(self, input, weight, hx, cx=None):
        assert(cudnn.is_acceptable(input))

        output = input.new()

        hy = hx.new()
        cy = cx.new() if cx else None

        cudnn.rnn.forward(self, input, hx, cx, weight, output, hy, cy)

        if cx is not None:
            self.save_for_backward(input, hx, cx, weight, output, hy, cy)
            return output, hy, cy
        else:
            self.save_for_backward(input, hx, weight, output, hy)
            return output, hy


    def backward(self, grad_output, grad_hy, grad_cy=None):
        tensors = self.saved_tensors
        if len(tensors) == 5:
            input, hx, weight, output, hy = tensors
            cx, cy = None, None
        else:
            input, hx, cx, weight, output, hy, cy = tensors

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
            return CudnnRNN(*args, **kwargs)(input, *fargs, **fkwargs)
        else:
            return THNN_RNN(*args, **kwargs)(input, *fargs, **fkwargs)
    return forward
