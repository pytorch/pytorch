from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch import nn
import torch
from torch.testing._internal.dist_utils import INIT_METHOD_TEMPLATE, dist_init
from torch.testing._internal.common_distributed import (
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
    skip_if_rocm,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)
from torch.distributed._pipeline.sync import Pipe

class PipeWithDDPTest(RpcAgentTestFixture):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    @dist_init
    @skip_if_rocm
    def test_basic_nccl(self):
        self._run_basic_test("nccl")

    @skip_if_lt_x_gpu(4)
    @requires_gloo()
    @dist_init
    @skip_if_rocm
    def test_basic_gloo(self):
        self._run_basic_test("gloo")

    def _run_basic_test(self, backend):
        dist.init_process_group(
            backend="nccl",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),
            world_size=self.world_size,
            rank=self.rank,
        )
        pg = dist.new_group([0, 2])
        # Use 4 GPUs, two replicas of a pipe across GPU 0 and 1 and another
        # pipe between GPU 2 and 3. Both replicas are replicated via DDP.
        if self.rank == 0 or self.rank == 2:

            fc1 = nn.Linear(16, 8).cuda(self.rank)
            fc2 = nn.Linear(8, 4).cuda(self.rank + 1)
            model = nn.Sequential(
                fc1,
                fc2
            )
            # TODO: Doesn't work with other checkpointing modes.
            model = Pipe(model, chunks=2, checkpoint="never")
            model = DistributedDataParallel(model, process_group=pg)
            out = model(torch.rand(16, 16).cuda(self.rank)).local_value()
            out.sum().backward()

            # Check grads
            output = [torch.empty_like(fc1.weight.grad), torch.empty_like(fc1.weight.grad)]
            pg.allgather(output, fc1.weight.grad)
            torch.cuda.synchronize(self.rank)
            self.assertEqual(output[0], output[1])

            output = [torch.empty_like(fc2.weight.grad), torch.empty_like(fc2.weight.grad)]
            pg.allgather(output, fc2.weight.grad)
            torch.cuda.synchronize(self.rank + 1)
            self.assertEqual(output[0], output[1])
