import math
import torch
from torch.autograd import Variable

from .module import Module
from .utils import _pair, _triple


class RNNBase(Module):

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
    """Applies a multi-layer RNN with tanh non-linearity to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:
    ```
    h_t = tanh(w_ih * x_t + b_ih  +  w_hh * h_(t-1) + b_hh)
    ```
    where `h_t` is the hidden state at time t, and `x_t` is the hidden
    state of the previous layer at time t or `input_t` for the first layer.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: the size of the convolving kernel.
        bias: If False, then the layer does not use bias weights b_ih and b_hh (default=True).
        batch_first: If True, then the input tensor is provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer
    Input: input, h_0
        input: A (seq_len x batch x input_size) tensor containing the features of the input sequence.
        h_0: A (num_layers x batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
    Output: output, h_n
        output: A (seq_len x batch x hidden_size) tensor containing the output features (h_k) from the last layer of the RNN, for each k
        h_n: A (num_layers x batch x hidden_size) tensor containing the hidden state for k=seq_len
    Members:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer, of shape (input_size x hidden_size)
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer, of shape (hidden_size x hidden_size)
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer, of shape (hidden_size)
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer, of shape (hidden_size)
    Examples:
        >>> rnn = nn.RNN(10, 20, 2)
        >>> input = Variable(torch.randn(5, 3, 10))
        >>> h0 = Variable(torch.randn(2, 3, 20))
        >>> output, hn = rnn(input, h0)
    """

    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__('RNN_TANH', *args, **kwargs)

class RNNReLU(RNNBase):
    """Applies a multi-layer RNN with ReLU non-linearity to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:
    ```
    h_t = ReLU(w_ih x_t + b_ih + w_hh h_(t-1) + b_hh)
    ```
    where `h_t` is the hidden state at time t, and `x_t` is the hidden
    state of the previous layer at time t or `input_t` for the first layer.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: the size of the convolving kernel.
        bias: If False, then the layer does not use bias weights b_ih and b_hh (default=True).
        batch_first: If True, then the input tensor is provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer
    Input: input, h_0
        input: A (seq_len x batch x input_size) tensor containing the features of the input sequence.
        h_0: A (num_layers x batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
    Output: output, h_n
        output: A (seq_len x batch x hidden_size) tensor containing the output features (h_k) from the last layer of the RNN, for each k
        h_n: A (num_layers x batch x hidden_size) tensor containing the hidden state for k=seq_len
    Members:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer, of shape (input_size x hidden_size)
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer, of shape (hidden_size x hidden_size)
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer, of shape (hidden_size)
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer, of shape (hidden_size)
    Examples:
        >>> rnn = nn.RNNReLU(10, 20, 2)
        >>> input = Variable(torch.randn(5, 3, 10))
        >>> h0 = Variable(torch.randn(2, 3, 20))
        >>> output, hn = rnn(input, h0)
    """

    def __init__(self, *args, **kwargs):
        super(RNNReLU, self).__init__('RNN_RELU', *args, **kwargs)

class LSTM(RNNBase):
    """Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:
    ```
            i_t = sigmoid(W_ii x_t + b_ii + W_hi h_(t-1) + b_hi)
            f_t = sigmoid(W_if x_t + b_if + W_hf h_(t-1) + b_hf)
            g_t = tanh(W_ig x_t + b_ig + W_hc h_(t-1) + b_hg)
            o_t = sigmoid(W_io x_t + b_io + W_ho h_(t-1) + b_ho)
            c_t = f_t * c_(t-1) + i_t * c_t
            h_t = o_t * tanh(c_t)
    ```
    where `h_t` is the hidden state at time t, `c_t` is the cell state at time t,
    `x_t` is the hidden state of the previous layer at time t or input_t for the first layer,
    and `i_t`, `f_t`, `g_t`, `o_t` are the input, forget, cell, and out gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: the size of the convolving kernel.
        bias: If False, then the layer does not use bias weights b_ih and b_hh (default=True).
        batch_first: If True, then the input tensor is provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer
    Input: input, (h_0, c_0)
        input: A (seq_len x batch x input_size) tensor containing the features of the input sequence.
        h_0: A (num_layers x batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
        c_0: A (num_layers x batch x hidden_size) tensor containing the initial cell state for each element in the batch.
    Output: output, (h_n, c_n)
        output: A (seq_len x batch x hidden_size) tensor containing the output features (h_t) from the last layer of the RNN, for each t
        h_n: A (num_layers x batch x hidden_size) tensor containing the hidden state for t=seq_len
        c_n: A (num_layers x batch x hidden_size) tensor containing the cell state for t=seq_len
    Members:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer (W_ir|W_ii|W_in), of shape (input_size x 3*hidden_size)
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer (W_hr|W_hi|W_hn), of shape (hidden_size x 3*hidden_size)
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer (b_ir|b_ii|b_in), of shape (3*hidden_size)
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer (W_hr|W_hi|W_hn), of shape (3*hidden_size)
    Examples:
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = Variable(torch.randn(5, 3, 10))
        >>> h0 = Variable(torch.randn(2, 3, 20))
        >>> c0 = Variable(torch.randn(2, 3, 20))
        >>> output, hn = rnn(input, (h0, c0))
    """
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

class GRU(RNNBase):
    """Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:
    ```
            r_t = sigmoid(W_ir x_t + b_ir + W_hr h_(t-1) + b_hr)
            i_t = sigmoid(W_ii x_t + b_ii + W_hi h_(t-1) + b_hi)
            n_t = tanh(W_in x_t + resetgate * W_hn h_(t-1))
            h_t = (1 - i_t) * n_t + i_t * h_(t-1)
    ```
    where `h_t` is the hidden state at time t, `x_t` is the hidden
    state of the previous layer at time t or input_t for the first layer,
    and `r_t`, `i_t`, `n_t` are the reset, input, and new gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: the size of the convolving kernel.
        bias: If False, then the layer does not use bias weights b_ih and b_hh (default=True).
        batch_first: If True, then the input tensor is provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer
    Input: input, h_0
        input: A (seq_len x batch x input_size) tensor containing the features of the input sequence.
        h_0: A (num_layers x batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
    Output: output, h_n
        output: A (seq_len x batch x hidden_size) tensor containing the output features (h_t) from the last layer of the RNN, for each t
        h_n: A (num_layers x batch x hidden_size) tensor containing the hidden state for t=seq_len
    Members:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer (W_ir|W_ii|W_in), of shape (input_size x 3*hidden_size)
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer (W_hr|W_hi|W_hn), of shape (hidden_size x 3*hidden_size)
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer (b_ir|b_ii|b_in), of shape (3*hidden_size)
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer (W_hr|W_hi|W_hn), of shape (3*hidden_size)
    Examples:
        >>> rnn = nn.GRU(10, 20, 2)
        >>> input = Variable(torch.randn(5, 3, 10))
        >>> h0 = Variable(torch.randn(2, 3, 20))
        >>> output, hn = rnn(input, h0)
    """

    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__('GRU', *args, **kwargs)


# FIXME: add module wrappers around XXXCell, and maybe StackedRNN and Recurrent
