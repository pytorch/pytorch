import torch.cuda
import torch.backends.cudnn as cudnn


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


# NB: We don't actually need this class anymore (in fact, we could serialize the
# dropout state for even better reproducibility), but it is kept for backwards
# compatibility for old models.
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


def init_dropout_state(ty, device, dropout, train, dropout_seed, dropout_state):
    dropout_desc_name = 'desc_' + str(torch.cuda.current_device())
    dropout_p = dropout if train else 0
    if (dropout_desc_name not in dropout_state) or (dropout_state[dropout_desc_name].get() is None):
        dropout_state[dropout_desc_name] = Unserializable(
            torch._cudnn_init_dropout_state(dropout_p, train, dropout_seed, ty=ty, device=device)
            if dropout_p != 0 else None
        )
    dropout_ts = dropout_state[dropout_desc_name].get()
    return dropout_ts
