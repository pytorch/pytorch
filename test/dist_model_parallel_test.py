import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.optim as optim
import torch.distributed.rpc as rpc
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef

from dist_utils import INIT_METHOD_TEMPLATE, dist_init, TEST_CONFIG

import unittest


def _call_method(method, obj_rref, *args, **kwargs):
    return method(obj_rref.local_value(), *args, **kwargs)


def _remote_method(method, obj_rref, *args, **kwargs):
    return rpc.rpc_sync(
        obj_rref.owner(),
        _call_method,
        args=[method, obj_rref] + list(args),
        kwargs=kwargs
    )


def _parameter_rrefs(module):
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs


class EmbeddingTable(nn.Module):
    def __init__(self, ntoken, ninp, dropout):
        super(EmbeddingTable, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return self.drop(self.encoder(input))


class RNN(nn.Module):
    def __init__(self, ninp, nhid, nlayers, dropout):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)

    def forward(self, emb, hidden):
        return self.lstm(emb, hidden)


class Decoder(nn.Module):
    def __init__(self, ntoken, nhid, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, output):
        return self.decoder(self.drop(output))


class RNNModel(nn.Module):
    def __init__(self, ps, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()

        self.emb_table_rref = rpc.remote(ps, EmbeddingTable, args=(ntoken, ninp, dropout))
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder_rref = rpc.remote(ps, Decoder, args=(ntoken, nhid, dropout))

    def forward(self, input, hidden):
        emb = _remote_method(EmbeddingTable.forward, self.emb_table_rref, input)
        output, hidden = self.rnn(emb, hidden)
        decoded = _remote_method(Decoder.forward, self.decoder_rref, output)
        return decoded, hidden

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(_remote_method(_parameter_rrefs, self.emb_table_rref))
        for p in self.rnn.parameters():
            remote_params.append(RRef(p))
        remote_params.extend(_remote_method(_parameter_rrefs, self.decoder_rref))
        return remote_params


@unittest.skipIf(
    not torch._six.PY3, "Pytorch distributed optim does not support python2"
)
class DistModelParallelTest(object):
    @property
    def world_size(self):
        return 4

    @property
    def init_method(self):
        return INIT_METHOD_TEMPLATE.format(
            file_name=self.file_name, rank=self.rank, world_size=self.world_size
        )

    @dist_init()
    def test_rnn(self):
        ps = 'worker%d' % ((self.rank + 1) % self.world_size)
        batch = 5
        nindices = 3
        ntoken = 10
        ninp = 2
        nhid = 3
        nlayers = 4
        hidden = (
            torch.randn(nlayers, nindices, nhid),
            torch.randn(nlayers, nindices, nhid)
        )

        rnn = RNNModel(ps, ntoken, ninp, nhid, nlayers)
        opt = DistributedOptimizer(
            optim.SGD,
            rnn.parameter_rrefs(),
            lr=0.05,
        )
        for _ in range(2):
            with dist_autograd.context() as ctx_id:
                inp = torch.LongTensor(batch, nindices) % ntoken
                hidden[0].detach_()
                hidden[1].detach_()
                output, hidden = rnn(inp, hidden)
                dist_autograd.backward([output.sum()])
                opt.step()
