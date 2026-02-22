Recurrent Layers
================

Recurrent layers process sequential data by maintaining hidden state across time steps.
They are essential for tasks involving sequences: language modeling, speech recognition,
time series prediction, and more.

- **RNN**: Basic recurrent layer (simple but prone to vanishing gradients)
- **LSTM**: Long Short-Term Memory (gated architecture, handles long-range dependencies)
- **GRU**: Gated Recurrent Unit (simpler than LSTM, often similar performance)
- **Cell variants**: Single-step versions for custom loop implementations

**Key parameters:**

- ``input_size``: Number of features in input
- ``hidden_size``: Number of features in hidden state
- ``num_layers``: Number of stacked recurrent layers
- ``batch_first``: If true, input shape is ``[batch, seq, features]``
- ``bidirectional``: Process sequence in both directions

RNN
---

.. doxygenclass:: torch::nn::RNN
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::RNNImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto rnn = torch::nn::RNN(
       torch::nn::RNNOptions(128, 256)  // input_size, hidden_size
           .num_layers(2)
           .batch_first(true)
           .bidirectional(false));

   auto input = torch::randn({32, 10, 128});  // [batch, seq_len, input_size]
   auto [output, hidden] = rnn->forward(input);

LSTM
----

.. doxygenclass:: torch::nn::LSTM
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::LSTMImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto lstm = torch::nn::LSTM(
       torch::nn::LSTMOptions(128, 256)
           .num_layers(2)
           .batch_first(true)
           .dropout(0.1)
           .bidirectional(true));

   auto input = torch::randn({32, 10, 128});
   auto [output, state] = lstm->forward(input);
   auto [h_n, c_n] = state;  // hidden state, cell state

GRU
---

.. doxygenclass:: torch::nn::GRU
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::GRUImpl
   :members:
   :undoc-members:

RNNCell
-------

.. doxygenclass:: torch::nn::RNNCell
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::RNNCellImpl
   :members:
   :undoc-members:

LSTMCell
--------

.. doxygenclass:: torch::nn::LSTMCell
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::LSTMCellImpl
   :members:
   :undoc-members:

GRUCell
-------

.. doxygenclass:: torch::nn::GRUCell
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::GRUCellImpl
   :members:
   :undoc-members:
