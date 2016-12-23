## Recurrent layers
### RNN

Applies a multi-layer Elman RNN with tanh or ReLU non-linearity to an input sequence.

```python
h_t = tanh(w_ih * x_t + b_ih  +  w_hh * h_(t-1) + b_hh)
```

```python
rnn = nn.RNN(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
output, hn = rnn(input, h0)
```



For each element in the input sequence, each layer computes the following
function:
where `h_t` is the hidden state at time t, and `x_t` is the hidden
state of the previous layer at time t or `input_t` for the first layer.
If nonlinearity='relu', then ReLU is used instead of tanh.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
input_size |  | The number of expected features in the input x
hidden_size |  | The number of features in the hidden state h
num_layers |  | the size of the convolving kernel.
nonlinearity | 'tanh' | The non-linearity to use ['tanh'|'relu'].
bias | True | If False, then the layer does not use bias weights b_ih and b_hh.
batch_first |  | If True, then the input tensor is provided as (batch, seq, feature)
dropout |  | If non-zero, introduces a dropout layer on the outputs of each RNN layer
bidirectional | False | If True, becomes a bidirectional RNN.

#### Inputs

Parameter | Default | Description
--------- | ------- | -----------
input |  | A (seq_len x batch x input_size) tensor containing the features of the input sequence.
h_0 |  | A (num_layers x batch x hidden_size) tensor containing the initial hidden state for each element in the batch.

#### Outputs

Parameter |  Description
--------- |  -----------
output | A (seq_len x batch x hidden_size) tensor containing the output features (h_k) from the last layer of the RNN, for each k
h_n | A (num_layers x batch x hidden_size) tensor containing the hidden state for k=seq_len

#### Members

Parameter | Description
--------- | -----------
weight_ih_l[k] | the learnable input-hidden weights of the k-th layer, of shape (input_size x hidden_size)
weight_hh_l[k] | the learnable hidden-hidden weights of the k-th layer, of shape (hidden_size x hidden_size)
bias_ih_l[k] | the learnable input-hidden bias of the k-th layer, of shape (hidden_size)
bias_hh_l[k] | the learnable hidden-hidden bias of the k-th layer, of shape (hidden_size)
### LSTM

Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

```python
i_t = sigmoid(W_ii x_t + b_ii + W_hi h_(t-1) + b_hi)
f_t = sigmoid(W_if x_t + b_if + W_hf h_(t-1) + b_hf)
g_t = tanh(W_ig x_t + b_ig + W_hc h_(t-1) + b_hg)
o_t = sigmoid(W_io x_t + b_io + W_ho h_(t-1) + b_ho)
c_t = f_t * c_(t-1) + i_t * c_t
h_t = o_t * tanh(c_t)
```

```python
rnn = nn.LSTM(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
c0 = Variable(torch.randn(2, 3, 20))
output, hn = rnn(input, (h0, c0))
```



For each element in the input sequence, each layer computes the following
function:
where `h_t` is the hidden state at time t, `c_t` is the cell state at time t,
`x_t` is the hidden state of the previous layer at time t or input_t for the first layer,
and `i_t`, `f_t`, `g_t`, `o_t` are the input, forget, cell, and out gates, respectively.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
input_size |  | The number of expected features in the input x
hidden_size |  | The number of features in the hidden state h
num_layers |  | the size of the convolving kernel.
bias | True | If False, then the layer does not use bias weights b_ih and b_hh.
batch_first |  | If True, then the input tensor is provided as (batch, seq, feature)
dropout |  | If non-zero, introduces a dropout layer on the outputs of each RNN layer
bidirectional | False | If True, becomes a bidirectional RNN.

#### Inputs

Parameter | Default | Description
--------- | ------- | -----------
input |  | A (seq_len x batch x input_size) tensor containing the features of the input sequence.
h_0 |  | A (num_layers x batch x hidden_size) tensor containing the initial hidden state for each element in the batch.
c_0 |  | A (num_layers x batch x hidden_size) tensor containing the initial cell state for each element in the batch.

#### Outputs

Parameter |  Description
--------- |  -----------
output | A (seq_len x batch x hidden_size) tensor containing the output features (h_t) from the last layer of the RNN, for each t
h_n | A (num_layers x batch x hidden_size) tensor containing the hidden state for t=seq_len
c_n | A (num_layers x batch x hidden_size) tensor containing the cell state for t=seq_len

#### Members

Parameter | Description
--------- | -----------
weight_ih_l[k] | the learnable input-hidden weights of the k-th layer (W_ir|W_ii|W_in), of shape (input_size x 3*hidden_size)
weight_hh_l[k] | the learnable hidden-hidden weights of the k-th layer (W_hr|W_hi|W_hn), of shape (hidden_size x 3*hidden_size)
bias_ih_l[k] | the learnable input-hidden bias of the k-th layer (b_ir|b_ii|b_in), of shape (3*hidden_size)
bias_hh_l[k] | the learnable hidden-hidden bias of the k-th layer (W_hr|W_hi|W_hn), of shape (3*hidden_size)
### GRU

Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

```python
r_t = sigmoid(W_ir x_t + b_ir + W_hr h_(t-1) + b_hr)
i_t = sigmoid(W_ii x_t + b_ii + W_hi h_(t-1) + b_hi)
n_t = tanh(W_in x_t + resetgate * W_hn h_(t-1))
h_t = (1 - i_t) * n_t + i_t * h_(t-1)
```

```python
rnn = nn.GRU(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
output, hn = rnn(input, h0)
```



For each element in the input sequence, each layer computes the following
function:
where `h_t` is the hidden state at time t, `x_t` is the hidden
state of the previous layer at time t or input_t for the first layer,
and `r_t`, `i_t`, `n_t` are the reset, input, and new gates, respectively.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
input_size |  | The number of expected features in the input x
hidden_size |  | The number of features in the hidden state h
num_layers |  | the size of the convolving kernel.
bias | True | If False, then the layer does not use bias weights b_ih and b_hh.
batch_first |  | If True, then the input tensor is provided as (batch, seq, feature)
dropout |  | If non-zero, introduces a dropout layer on the outputs of each RNN layer
bidirectional | False | If True, becomes a bidirectional RNN.

#### Inputs

Parameter | Default | Description
--------- | ------- | -----------
input |  | A (seq_len x batch x input_size) tensor containing the features of the input sequence.
h_0 |  | A (num_layers x batch x hidden_size) tensor containing the initial hidden state for each element in the batch.

#### Outputs

Parameter |  Description
--------- |  -----------
output | A (seq_len x batch x hidden_size) tensor containing the output features (h_t) from the last layer of the RNN, for each t
h_n | A (num_layers x batch x hidden_size) tensor containing the hidden state for t=seq_len

#### Members

Parameter | Description
--------- | -----------
weight_ih_l[k] | the learnable input-hidden weights of the k-th layer (W_ir|W_ii|W_in), of shape (input_size x 3*hidden_size)
weight_hh_l[k] | the learnable hidden-hidden weights of the k-th layer (W_hr|W_hi|W_hn), of shape (hidden_size x 3*hidden_size)
bias_ih_l[k] | the learnable input-hidden bias of the k-th layer (b_ir|b_ii|b_in), of shape (3*hidden_size)
bias_hh_l[k] | the learnable hidden-hidden bias of the k-th layer (W_hr|W_hi|W_hn), of shape (3*hidden_size)
### RNNCell

An Elman RNN cell with tanh or ReLU non-linearity.

```python
h' = tanh(w_ih * x + b_ih  +  w_hh * h + b_hh)
```

```python
rnn = nn.RNNCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
    hx = rnn(input, hx)
    output[i] = hx
```

If nonlinearity='relu', then ReLU is used in place of tanh.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
input_size |  | The number of expected features in the input x
hidden_size |  | The number of features in the hidden state h
bias | True | If False, then the layer does not use bias weights b_ih and b_hh.
nonlinearity | 'tanh' | The non-linearity to use ['tanh'|'relu'].

#### Inputs

Parameter | Default | Description
--------- | ------- | -----------
input |  | A (batch x input_size) tensor containing input features
hidden |  | A (batch x hidden_size) tensor containing the initial hidden state for each element in the batch.

#### Outputs

Parameter |  Description
--------- |  -----------
h' | A (batch x hidden_size) tensor containing the next hidden state for each element in the batch

#### Members

Parameter | Description
--------- | -----------
weight_ih | the learnable input-hidden weights, of shape (input_size x hidden_size)
weight_hh | the learnable hidden-hidden weights, of shape (hidden_size x hidden_size)
bias_ih | the learnable input-hidden bias, of shape (hidden_size)
bias_hh | the learnable hidden-hidden bias, of shape (hidden_size)
### LSTMCell

A long short-term memory (LSTM) cell.

```python
i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)
f = sigmoid(W_if x + b_if + W_hf h + b_hf)
g = tanh(W_ig x + b_ig + W_hc h + b_hg)
o = sigmoid(W_io x + b_io + W_ho h + b_ho)
c' = f * c + i * c
h' = o * tanh(c_t)
```

```python
rnn = nn.LSTMCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
cx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
    hx, cx = rnn(input, (hx, cx))
    output[i] = hx
```



#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
input_size |  | The number of expected features in the input x
hidden_size |  | The number of features in the hidden state h
bias | True | If False, then the layer does not use bias weights b_ih and b_hh.

#### Inputs

Parameter | Default | Description
--------- | ------- | -----------
input |  | A (batch x input_size) tensor containing input features
hidden |  | A (batch x hidden_size) tensor containing the initial hidden state for each element in the batch.

#### Outputs

Parameter |  Description
--------- |  -----------
h' | A (batch x hidden_size) tensor containing the next hidden state for each element in the batch
c' | A (batch x hidden_size) tensor containing the next cell state for each element in the batch

#### Members

Parameter | Description
--------- | -----------
weight_ih | the learnable input-hidden weights, of shape (input_size x hidden_size)
weight_hh | the learnable hidden-hidden weights, of shape (hidden_size x hidden_size)
bias_ih | the learnable input-hidden bias, of shape (hidden_size)
bias_hh | the learnable hidden-hidden bias, of shape (hidden_size)
### GRUCell

A gated recurrent unit (GRU) cell

```python
r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)
n = tanh(W_in x + resetgate * W_hn h)
h' = (1 - i) * n + i * h
```

```python
rnn = nn.RNNCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
    hx = rnn(input, hx)
    output[i] = hx
```



#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
input_size |  | The number of expected features in the input x
hidden_size |  | The number of features in the hidden state h
bias | True | If False, then the layer does not use bias weights b_ih and b_hh.

#### Inputs

Parameter | Default | Description
--------- | ------- | -----------
input |  | A (batch x input_size) tensor containing input features
hidden |  | A (batch x hidden_size) tensor containing the initial hidden state for each element in the batch.

#### Outputs

Parameter |  Description
--------- |  -----------
h' | A (batch x hidden_size) tensor containing the next hidden state for each element in the batch

#### Members

Parameter | Description
--------- | -----------
weight_ih | the learnable input-hidden weights, of shape (input_size x hidden_size)
weight_hh | the learnable hidden-hidden weights, of shape (hidden_size x hidden_size)
bias_ih | the learnable input-hidden bias, of shape (hidden_size)
bias_hh | the learnable hidden-hidden bias, of shape (hidden_size)
