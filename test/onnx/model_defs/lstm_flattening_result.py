from torch import nn


class LstmFlatteningResult(nn.LSTM):
    def forward(self, input, *fargs, **fkwargs):
        output, (hidden, cell) = nn.LSTM.forward(self, input, *fargs, **fkwargs)
        return output, hidden, cell
