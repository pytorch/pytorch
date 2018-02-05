import torch.cuda
import torch.backends.cudnn as cudnn
from torch.backends.cudnn import check_error
import ctypes
from torch.autograd import Variable


def get_cudnn_mode(mode):
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


class Unserializable(object):

    def __init__(self, inner):
        self.inner = inner

    def get(self):
        return self.inner

    def __getstate__(self):
        # Note: can't return {}, because python2 won't call __setstate__
        # if the value evaluates to False
        return "<unserializable>"

    def __setstate__(self, state):
        self.inner = None


def init_dropout_descriptor(handle, dropout, train, dropout_seed, dropout_state):
    dropout_desc_name = 'desc_' + str(torch.cuda.current_device())
    dropout_p = dropout if train else 0
    if (dropout_desc_name not in dropout_state) or (dropout_state[dropout_desc_name].get() is None):
        dropout_state[dropout_desc_name] = Unserializable(
            cudnn.DropoutDescriptor(handle, dropout_p, dropout_seed)
        )
    dropout_desc = dropout_state[dropout_desc_name].get()
    dropout_desc.set_dropout(dropout_p, dropout_seed)
    return dropout_desc


def get_dropout_state(fn, handle):
    dropout_desc_name = 'desc_' + str(torch.cuda.current_device())
    dropout_p = fn.dropout if fn.train else 0
    dropout_desc = fn.dropout_state[dropout_desc_name].get()
    return dropout_desc.state
