import math
import torch
from torch.autograd import Variable

from ..module import Module

class RNN(Module):
    """An Elman RNN cell with tanh linearity.
    ```
    h' = tanh(w_ih * x + b_ih  +  w_hh * h + b_hh)
    ```

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights b_ih and b_hh (default=True).
    Input: input, h
        input: A (batch x input_size) tensor containing input features
        hidden: A (batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
    Output: h'
        h': A (batch x hidden_size) tensor containing the next hidden state for each element in the batch
    Members:
        weight_ih: the learnable input-hidden weights, of shape (input_size x hidden_size)
        weight_hh: the learnable hidden-hidden weights, of shape (hidden_size x hidden_size)
        bias_ih: the learnable input-hidden bias, of shape (hidden_size)
        bias_hh: the learnable hidden-hidden bias, of shape (hidden_size)
    Examples:
        >>> rnn = nn.rnn.cell.RNN(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     output[i] = hx = rnn(input, hx)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        super(RNN, self).__init__(
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
        return self._backend.RNNTanhCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )

class RNNReLU(Module):
    """An Elman RNN cell with ReLU linearity.
    ```
    h' = ReLU(w_ih * x + b_ih  +  w_hh * h + b_hh)
    ```

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights b_ih and b_hh (default=True).
    Input: input, h
        input: A (batch x input_size) tensor containing input features
        hidden: A (batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
    Output: h'
        h': A (batch x hidden_size) tensor containing the next hidden state for each element in the batch
    Members:
        weight_ih: the learnable input-hidden weights, of shape (input_size x hidden_size)
        weight_hh: the learnable hidden-hidden weights, of shape (hidden_size x hidden_size)
        bias_ih: the learnable input-hidden bias, of shape (hidden_size)
        bias_hh: the learnable hidden-hidden bias, of shape (hidden_size)
    Examples:
        >>> rnn = nn.rnn.cell.RNNReLU(10, 20)
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

        super(RNNReLU, self).__init__(
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
        return self._backend.RNNReLUCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )

class LSTM(Module):
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
        bias: If False, then the layer does not use bias weights b_ih and b_hh (default=True).
    Input: input, h
        input: A (batch x input_size) tensor containing input features
        hidden: A (batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
    Output: h', c'
        h': A (batch x hidden_size) tensor containing the next hidden state for each element in the batch
        h': A (batch x hidden_size) tensor containing the next cell state for each element in the batch
    Members:
        weight_ih: the learnable input-hidden weights, of shape (input_size x hidden_size)
        weight_hh: the learnable hidden-hidden weights, of shape (hidden_size x hidden_size)
        bias_ih: the learnable input-hidden bias, of shape (hidden_size)
        bias_hh: the learnable hidden-hidden bias, of shape (hidden_size)
    Examples:
        >>> rnn = nn.rnn.cell.LSTM(10, 20)
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

        super(LSTM, self).__init__(
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

class GRU(Module):
    """A gated recurrent unit (GRU) cell with ReLU linearity.
    ```
    r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
    i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)
    n = tanh(W_in x + resetgate * W_hn h)
    h' = (1 - i) * n + i * h
    ```

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights b_ih and b_hh (default=True).
    Input: input, h
        input: A (batch x input_size) tensor containing input features
        hidden: A (batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
    Output: h'
        h': A (batch x hidden_size) tensor containing the next hidden state for each element in the batch
    Members:
        weight_ih: the learnable input-hidden weights, of shape (input_size x hidden_size)
        weight_hh: the learnable hidden-hidden weights, of shape (hidden_size x hidden_size)
        bias_ih: the learnable input-hidden bias, of shape (hidden_size)
        bias_hh: the learnable hidden-hidden bias, of shape (hidden_size)
    Examples:
        >>> rnn = nn.rnn.cell.RNNReLU(10, 20)
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

        super(GRU, self).__init__(
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
