__all__ = [
    "LSTM",
    "LSTMCell",
    "MultiheadAttention",
]

from torch.ao.nn.quantizable.modules.activation import MultiheadAttention
from torch.ao.nn.quantizable.modules.rnn import LSTM, LSTMCell
