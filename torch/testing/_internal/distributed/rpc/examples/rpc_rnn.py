import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.rpc import RRef
from torch.distributed.optim import DistributedOptimizer

from torch.testing._internal.dist_utils import dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture


def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    return rpc.rpc_sync(
        rref.owner(),
        _call_method,
        args=[method, rref] + list(args),
        kwargs=kwargs
    )


def _parameter_rrefs(module):
    r"""
    Create one RRef for each parameter in the given local module, and return a
    list of RRefs.
    """
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs


class EmbeddingTable(nn.Module):
    r"""
    Encoding layers of the RNNModel
    """
    def __init__(self, ntoken, ninp, dropout):
        super(EmbeddingTable, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)

    def forward(self, input):
        if torch.cuda.is_available():
            input = input.cuda()
        return self.drop(self.encoder(input)).cpu()


class Decoder(nn.Module):
    r"""
    Decoding layers of the RNNModel
    """
    def __init__(self, ntoken, nhid, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)

    def forward(self, output):
        return self.decoder(self.drop(output))


class RNNModel(nn.Module):
    r"""
    A distributed RNN model which puts embedding table and decoder parameters on
    a remote parameter server, and locally holds parameters for the LSTM module.
    The structure of the RNN model is borrowed from the word language model
    example. See https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """
    def __init__(self, ps, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()

        # setup embedding table remotely
        self.emb_table_rref = rpc.remote(ps, EmbeddingTable, args=(ntoken, ninp, dropout))
        # setup LSTM locally
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        # setup decoder remotely
        self.decoder_rref = rpc.remote(ps, Decoder, args=(ntoken, nhid, dropout))

    def forward(self, input, hidden):
        # pass input to the remote embedding table and fetch emb tensor back
        emb = _remote_method(EmbeddingTable.forward, self.emb_table_rref, input)
        output, hidden = self.rnn(emb, hidden)
        # pass output to the remote decoder and get the decoded output back
        decoded = _remote_method(Decoder.forward, self.decoder_rref, output)
        return decoded, hidden

    def parameter_rrefs(self):
        remote_params = []
        # get RRefs of embedding table
        remote_params.extend(_remote_method(_parameter_rrefs, self.emb_table_rref))
        # create RRefs for local parameters
        remote_params.extend(_parameter_rrefs(self.rnn))
        # get RRefs of decoder
        remote_params.extend(_remote_method(_parameter_rrefs, self.decoder_rref))
        return remote_params


def _run_trainer():
    r"""
    The trainer creates a distributed RNNModel and a DistributedOptimizer. Then,
    it performs training using random input data.
    """
    batch = 5
    ntoken = 7
    ninp = 2

    nhid = 3
    nindices = 6
    nlayers = 4
    hidden = (
        torch.randn(nlayers, nindices, nhid),
        torch.randn(nlayers, nindices, nhid)
    )

    model = RNNModel('ps', ntoken, ninp, nhid, nlayers)

    # setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch():
        for _ in range(5):
            data = torch.LongTensor(batch, nindices) % ntoken
            target = torch.LongTensor(batch, ntoken) % nindices
            yield data, target

    # train for 10 iterations
    for epoch in range(10):
        # create distributed autograd context
        for data, target in get_next_batch():
            with dist_autograd.context() as context_id:
                hidden[0].detach_()
                hidden[1].detach_()
                output, hidden = model(data, hidden)
                loss = criterion(output, target)
                # run distributed backward pass
                dist_autograd.backward(context_id, [loss])
                # run distributed optimizer
                opt.step(context_id)
                # not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
        print("Training epoch {}".format(epoch))

class RPCRNNTest(RpcAgentTestFixture):
    @property
    def world_size(self):
        return 2

    @property
    def name(self):
        return "ps" if self.rank == 0 else "trainer"

    @dist_init
    def test_rpc_rnn(self):
        if self.rank == 1:
            _run_trainer()
        else:
            # parameter server does nothing
            pass
