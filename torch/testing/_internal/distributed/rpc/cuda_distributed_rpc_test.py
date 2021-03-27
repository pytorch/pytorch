import os
import time
import torch
import torch.distributed.autograd as autograd
import torch.distributed.rpc as rpc
import torch.nn as nn

from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)
from torch.testing._internal.dist_utils import (
    dist_init,
)


class MyModule(nn.Module):
    def __init__(self, device, comm_mode):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(1000, 1000).to(device)
        self.comm_mode = comm_mode

    def forward(self, x):
        # x.to() is a no-op if x is already on self.device
        y = self.linear(x.to(self.device))
        return y.cpu() if self.comm_mode == "cpu" else y

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]


def measure(comm_mode):
    # local module on "worker0/cuda:0"
    lm = MyModule("cuda:0", comm_mode)
    # remote module on "worker1/cuda:1"
    rm = rpc.remote("worker1", MyModule, args=("cuda:1", comm_mode))
    # prepare random inputs
    x = torch.randn(1000, 1000).cuda(0)

    tik = time.time()
    for _ in range(10):
        with autograd.context() as ctx:
            y = rm.rpc_sync().forward(lm(x))
            autograd.backward(ctx, [y.sum()])
    # synchronize on "cuda:0" to make sure that all pending CUDA ops are
    # included in the measurements
    torch.cuda.current_stream("cuda:0").synchronize()
    tok = time.time()
    print(f"{comm_mode} RPC total execution time: {tok - tik}")


class CudaDistributedRPCTest(RpcAgentTestFixture):

    @property
    def world_size(self):
        return 2

    @dist_init(setup_rpc=False)
    def test_cuda_distributed_rpc(self):

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

        if self.rank == 0:
            options.set_device_map("worker1", {0: 1})
            rpc.init_rpc(
                f"worker{self.rank}",
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=options
            )
            measure(comm_mode="cpu")
            measure(comm_mode="cuda")
        else:
            rpc.init_rpc(
                f"worker{self.rank}",
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=options
            )

        # block until all rpcs finish
        rpc.shutdown()
