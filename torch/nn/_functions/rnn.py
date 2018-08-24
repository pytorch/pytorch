import warnings
from torch.autograd import NestedIOFunction
import torch.backends.cudnn as cudnn
from .. import functional as F
import torch
import itertools
from functools import partial

try:
    import torch.backends.cudnn.rnn
except ImportError:
    pass


def _select_rnn_impl(mode, input_size, hidden_size, num_layers=1, batch_first=False,
                     dropout=0, train=True, bidirectional=False, dropout_state=None):
    hidden_is_tensor = True
    if mode == 'RNN_RELU':
        impl = torch._C._VariableFunctions.rnn_relu
    elif mode == 'RNN_TANH':
        impl = torch._C._VariableFunctions.rnn_tanh
    elif mode == 'LSTM':
        hidden_is_tensor = False
        impl = torch._C._VariableFunctions.lstm
    elif mode == 'GRU':
        impl = torch._C._VariableFunctions.gru
    else:
        raise Exception('Unknown mode: {}'.format(mode))

    def forward(input, weight, hidden, batch_sizes):
        has_biases = len(weight[0]) == 4
        weight = sum(weight, type(weight[0])())

        if batch_sizes is None:
            result = impl(input, hidden, weight, has_biases, num_layers, dropout, train, bidirectional, batch_first)
        else:
            result = impl(input, batch_sizes, hidden, weight, has_biases, num_layers, dropout, train, bidirectional)
        return result[0], (result[1] if hidden_is_tensor else result[1:])

    return forward


def get_rnn_impl(*args, **kwargs):

    def forward(input, *fargs, **fkwargs):
        func = _select_rnn_impl(*args, **kwargs)

        # Hack for the tracer that allows us to represent RNNs as single
        # nodes and export them to ONNX in this form
        # Check the first argument explicitly to reduce the overhead of creating
        # the lambda. We need special handling here because the forward()
        # function gets reconstructed each and every time when RNN() is invoked
        # and we don't want to pay the cost of decorator invocation
        if torch._C._get_tracing_state():
            from torch.onnx import symbolic
            sym = symbolic.RNN_symbolic_builder(*args, **kwargs)
            cell_type = args[0]

            bound_symbolic = partial(symbolic.rnn_trace_override_symbolic,
                                     cell_type, func, sym)

            decorator = torch.onnx.symbolic_override(bound_symbolic)
            func = decorator(func)

        return func(input, *fargs, **fkwargs)

    return forward
