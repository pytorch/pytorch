import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import rpc

from torch.nn.parallel import DistributedDataParallel as DDP


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


class FakeEmbeddingTable(nn.Module):
    def __init__(self):
        super(FakeEmbeddingTable, self).__init__()
        # 1 token only
        self.em = nn.Embedding(1, 8)

    def forward(self, input):
        return self.em(input)


class DdpModelWithRpc(nn.Module):
    def __init__(self, ps):
        super(MultiMachineModel, self).__init__()
        self.em_rref = rpc.remote(ps, FakeEmbeddingTable)
        self.net = DDP(torch.nn.Linear(ninp, 10))

    def forward(self, x):
        emb = _remote_method(EmbeddingTable.forward, self.em_rref, x)
        return self.net(emb)
