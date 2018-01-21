import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.autograd import Variable

import math


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.nvol_grad = None

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def save_grad(self):
        def hook(grad):
            self.nvol_grad = grad
        return hook

    def custom(self, start, end):
        def custom_forward(*inputs):
            output, hidden = self.rnn(
                inputs[0][start:(end+1)], (inputs[1], inputs[2])
            )
            return output, hidden[0], hidden[1]
        return custom_forward

    def forward(self, input, hidden, targets, chunks=4):
        total_modules = input.shape[0]
        chunk_size = int(math.floor(float(total_modules) / chunks))
        start, end = 0, -1
        emb = self.drop(self.encoder(input))

        output = []
        for j in range(chunks):
            start = end + 1
            end = start + chunk_size - 1
            if j == (chunks - 1):
                end = total_modules - 1
            out = checkpoint.checkpoint(self.custom(start, end), emb, hidden[0], hidden[1])
            output.append(out[0])
            hidden = (out[1], out[2])
        output = torch.cat(output, 0)
        hidden = (out[1], out[2])

        output = self.drop(output).view(output.size(0) * output.size(1), output.size(2))
        out = Variable(output.data, requires_grad=True)
        out.register_hook(self.save_grad())
        total_modules = out.shape[0]
        chunks = 10
        chunk_size = int(math.floor(float(total_modules) / chunks))
        start, end = 0, -1
        for i in range(chunks):
            start = end + 1
            end = start + chunk_size - 1
            if i == (chunks - 1):
                end = total_modules - 1
            decoded = self.decoder(out[start:(end+1)])
            loss = self.criterion(decoded, targets[start:(end+1)])
            loss.backward()
        return output, hidden

    def backward(self, output):
        output.backward(self.nvol_grad.data)
