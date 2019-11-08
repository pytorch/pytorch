import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc


class Encoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout):
        super(Encoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return self.drop(self.encoder(input))


class Decoder(nn.Module):
    def __init__(self, ntoken, nhid, dropout):
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, output):
        return self.decoder(self.drop(output))


class RNNModel(nn.Module):
    def __init__(self, ps, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.encoder_rref = rpc.remote(ps, Encoder, args=(ntoken, ninp, dropout))

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
