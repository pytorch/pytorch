from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence
from typing import Optional, Tuple

class LstmFlatteningResult(nn.LSTM):
    def forward(self, input, *fargs, **fkwargs):
        output, (hidden, cell) = nn.LSTM.forward(self, input, *fargs, **fkwargs)
        return output, hidden, cell


class LstmFlatteningResultWithSeqLength(nn.Module):
    def __init__(self, input_size, hidden_size, layers, bidirect, dropout, batch_first):
        super().__init__()

        self.batch_first = batch_first
        self.inner_model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(self, input: PackedSequence, hx:Optional[Tuple[Tensor, Tensor]]=None):
        output, (hidden, cell) = self.inner_model.forward(input, hx)
        return output, hidden, cell


class LstmFlatteningResultWithoutSeqLength(nn.Module):
    def __init__(self, input_size, hidden_size, layers, bidirect, dropout, batch_first):
        super().__init__()

        self.batch_first = batch_first
        self.inner_model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(self, input, hx:Optional[Tuple[Tensor, Tensor]]=None):
        output, (hidden, cell) = self.inner_model.forward(input, hx)
        return output, hidden, cell
