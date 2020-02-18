import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import rpc

from torch.nn.parallel import DistributedDataParallel as DDP


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


class SimpleNet(nn.Module):
    def __init__(self, d_in, d_out):
        super(SimpleNet, self).__init__()
        self.net = nn.Linear(d_in, d_out)

    def forward(self, input):
        return nn.ReLU(self.net(input))


class DdpModelWithRpc(nn.Module):
    def __init__(self, remote_server):
        super(DdpModelWithRpc, self).__init__()
        self.net1 = DDP(SimpleNet(5, 8))
        self.rref = rpc.remote(remote_server, 8, 5)
        self.net2 = DDP(SimpleNet(5, 3))

    def forward(self, x):
        x = self.net1(x)
        x = _remote_method(SimpleNet.forward, self.rref, x)
        return self.net2(x)
