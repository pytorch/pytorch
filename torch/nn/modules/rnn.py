import math
import torch

from .module import Module
from ..parameter import Parameter


class RNNBase(Module):

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.all_weights = []
        super_weights = {}
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                if mode == 'LSTM':
                    gate_size = 4 * hidden_size
                elif mode == 'GRU':
                    gate_size = 3 * hidden_size
                else:
                    gate_size = hidden_size

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))

                suffix = '_reverse' if direction == 1 else ''
                super_weights['weight_ih_l{}{}'.format(layer, suffix)] = w_ih
                super_weights['weight_hh_l{}{}'.format(layer, suffix)] = w_hh
                if bias:
                    super_weights['bias_ih_l{}{}'.format(layer, suffix)] = b_ih
                    super_weights['bias_hh_l{}{}'.format(layer, suffix)] = b_hh
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
            self.training,
            self.bidirectional
        )

        return func(input, self.all_weights, hx)


class RNN(RNNBase):
    """Applies a multi-layer Elman RNN with tanh or ReLU non-linearity to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:
    ```
    h_t = tanh(w_ih * x_t + b_ih  +  w_hh * h_(t-1) + b_hh)
    ```
    where `h_t` is the hidden state at time t, and `x_t` is the hidden
    state of the previous layer at time t or `input_t` for the first layer.
    If nonlinearity='relu', then ReLU is used instead of tanh.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: the size of the convolving kernel.
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
        bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        batch_first: If True, then the input tensor is provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer
    Inputs: input, h_0
        input: A (seq_len x batch x input_size) tensor containing the features of the input sequence.
        h_0: A (num_layers x batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
    Outputs: output, h_n
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
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise Exception("Unknown nonlinearity: {}".format(
                    kwargs['nonlinearity']
                ))
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'

        super(RNN, self).__init__(mode, *args, **kwargs)


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
        bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        batch_first: If True, then the input tensor is provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer
    Inputs: input, (h_0, c_0)
        input: A (seq_len x batch x input_size) tensor containing the features of the input sequence.
        h_0: A (num_layers x batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
        c_0: A (num_layers x batch x hidden_size) tensor containing the initial cell state for each element in the batch.
    Outputs: output, (h_n, c_n)
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
        bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        batch_first: If True, then the input tensor is provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer
    Inputs: input, h_0
        input: A (seq_len x batch x input_size) tensor containing the features of the input sequence.
        h_0: A (num_layers x batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
    Outputs: output, h_n
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


class RNNCell(Module):
    """An Elman RNN cell with tanh or ReLU non-linearity.
    ```
    h' = tanh(w_ih * x + b_ih  +  w_hh * h + b_hh)
    ```
    If nonlinearity='relu', then ReLU is used in place of tanh.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
    Inputs: input, hidden
        input: A (batch x input_size) tensor containing input features
        hidden: A (batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
    Outputs: h'
        h': A (batch x hidden_size) tensor containing the next hidden state for each element in the batch
    Members:
        weight_ih: the learnable input-hidden weights, of shape (input_size x hidden_size)
        weight_hh: the learnable hidden-hidden weights, of shape (hidden_size x hidden_size)
        bias_ih: the learnable input-hidden bias, of shape (hidden_size)
        bias_hh: the learnable hidden-hidden bias, of shape (hidden_size)
    Examples:
        >>> rnn = nn.RNNCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input, hx)
        ...     output[i] = hx
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        super(RNNCell, self).__init__(
            weight_ih = torch.Tensor(hidden_size, input_size),
            weight_hh = torch.Tensor(hidden_size, hidden_size),
            bias_ih = torch.Tensor(hidden_size) if bias else None,
            bias_hh = torch.Tensor(hidden_size) if bias else None,
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, input, hx):
        if self.nonlinearity == "tanh":
            func = self._backend.RNNTanhCell
        elif self.nonlinearity == "relu":
            func = self._backend.RNNReLUCell
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        return func(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )


class LSTMCell(Module):
    """A long short-term memory (LSTM) cell.
    ```
    i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)
    f = sigmoid(W_if x + b_if + W_hf h + b_hf)
    g = tanh(W_ig x + b_ig + W_hc h + b_hg)
    o = sigmoid(W_io x + b_io + W_ho h + b_ho)
    c' = f * c + i * c
    h' = o * tanh(c_t)
    ```

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    Inputs: input, hidden
        input: A (batch x input_size) tensor containing input features
        hidden: A (batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
    Outputs: h', c'
        h': A (batch x hidden_size) tensor containing the next hidden state for each element in the batch
        c': A (batch x hidden_size) tensor containing the next cell state for each element in the batch
    Members:
        weight_ih: the learnable input-hidden weights, of shape (input_size x hidden_size)
        weight_hh: the learnable hidden-hidden weights, of shape (hidden_size x hidden_size)
        bias_ih: the learnable input-hidden bias, of shape (hidden_size)
        bias_hh: the learnable hidden-hidden bias, of shape (hidden_size)
    Examples:
        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> cx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input, (hx, cx))
        ...     output[i] = hx
    """

    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        super(LSTMCell, self).__init__(
            weight_ih = torch.Tensor(4*hidden_size, input_size),
            weight_hh = torch.Tensor(4*hidden_size, hidden_size),
            bias_ih = torch.Tensor(4*hidden_size) if bias else None,
            bias_hh = torch.Tensor(4*hidden_size) if bias else None,
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, input, hx):
        return self._backend.LSTMCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )


class GRUCell(Module):
    """A gated recurrent unit (GRU) cell
    ```
    r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
    i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)
    n = tanh(W_in x + resetgate * W_hn h)
    h' = (1 - i) * n + i * h
    ```

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    Inputs: input, hidden
        input: A (batch x input_size) tensor containing input features
        hidden: A (batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
    Outputs: h'
        h': A (batch x hidden_size) tensor containing the next hidden state for each element in the batch
    Members:
        weight_ih: the learnable input-hidden weights, of shape (input_size x hidden_size)
        weight_hh: the learnable hidden-hidden weights, of shape (hidden_size x hidden_size)
        bias_ih: the learnable input-hidden bias, of shape (hidden_size)
        bias_hh: the learnable hidden-hidden bias, of shape (hidden_size)
    Examples:
        >>> rnn = nn.RNNCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input, hx)
        ...     output[i] = hx
    """

    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        super(GRUCell, self).__init__(
            weight_ih = torch.Tensor(3*hidden_size, input_size),
            weight_hh = torch.Tensor(3*hidden_size, hidden_size),
            bias_ih = torch.Tensor(3*hidden_size) if bias else None,
            bias_hh = torch.Tensor(3*hidden_size) if bias else None,
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, input, hx):
        return self._backend.GRUCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
