from torch.distributed import rpc
from torch.distributed.optim import DistributedOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.distributed as dist
import torch.nn as nn

from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT


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

class TestDdpWithRpc(TestCase):
    TRAINER_NAME_TEMPLATE = 'trainer.{:02d}'
    REMOTE_WORKER_NAME = 'remote_worker'
    TRAINER_GROUP = 'trainer_group'
    NUM_TRAINERS = 4

    def setUp(self):
        super(TestDdpWithRpc, self).setUp()
        self.model = DdpModelWithRpc()
        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)
        # os.environ["WORLD_SIZE"] = str(self.NUM_TRAINERS + 1)


    def run_trainer(self, rank):
        trainer_name = self.TRAINER_NAME_TEMPLATE.format(rank)
        #  For DDP communications
        dist.init_process_group("gloo", group_name=self.TRAINER_GROUP, rank=rank, world_size=self.NUM_TRAINERS)
        # This group includes the remote worker
        rpc.init_rpc(
            name=trainer_name,
            rank=rank,
            world_size=self.NUM_TRAINERS + 1,
        )

    def run_remote_worker(self):
        # This group includes the remote worker
        rpc.init_rpc(
            name=self.REMOTE_WORKER_NAME,
            rank=self.NUM_TRAINERS,
            world_size=self.NUM_TRAINERS + 1,
        )


if __name__ == "__main__":
    print("Running as main")
