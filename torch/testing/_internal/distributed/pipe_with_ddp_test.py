import torch
import torch.distributed as dist
import unittest

from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.dist_utils import INIT_METHOD_TEMPLATE, dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)
from torch.testing._internal.common_distributed import (
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
    skip_if_rocm,
)
from torch.distributed._pipeline.sync import Pipe

class PipeWithDDPTest(RpcAgentTestFixture):
    @property
    def world_size(self) -> int:
        return 2

    def cuda_visible_devices(self) -> str:
        return "{},{}".format(2 * self.rank, 2 * self.rank + 1)

    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    @dist_init
    @skip_if_rocm
    def test_basic_nccl_ckpt_never(self):
        self._run_basic_test("nccl", "never")

    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    @dist_init
    @skip_if_rocm
    @unittest.skip("DDP doesn't work with checkpointing")
    def test_basic_nccl_ckpt_always(self):
        self._run_basic_test("nccl", "always")

    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    @dist_init
    @skip_if_rocm
    @unittest.skip("DDP doesn't work with checkpointing")
    def test_basic_nccl_ckpt_except_last(self):
        self._run_basic_test("nccl", "except_last")

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @dist_init
    @skip_if_rocm
    def test_basic_gloo_ckpt_never(self):
        self._run_basic_test("gloo", "never")

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @dist_init
    @skip_if_rocm
    @unittest.skip("DDP doesn't work with checkpointing")
    def test_basic_gloo_ckpt_always(self):
        self._run_basic_test("gloo", "always")

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    @dist_init
    @skip_if_rocm
    @unittest.skip("DDP doesn't work with checkpointing")
    def test_basic_gloo_ckpt_except_last(self):
        self._run_basic_test("gloo", "except_last")

    def _run_basic_test(self, backend, checkpoint):
        dist.init_process_group(
            backend="nccl",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),
            world_size=self.world_size,
            rank=self.rank,
        )

        # Use 4 GPUs, two replicas of a pipe across GPU 0 and 1 and another
        # pipe between GPU 2 and 3. Both replicas are replicated via DDP.
        # We use CUDA_VISIBLE_DEVICES to ensure each process only can see two devices.
        # HACK: Since barrier() creates a tensor on the GPU index matching the rank,
        # the barrier() call ends up happening between GPU0 on the first process and
        # GPU3 (which represents GPU index 1 for rank 1 when CUDA_VISIBLE_DEVICES is
        # set to 2,3 for process 2) on the second process. As a result, the
        # communicator is established between GPU0 and GPU3 and hence we invert the
        # pipe on the second process to ensure all the communicators match. This is
        # an ugly hack that needs to be resolved in ProcessGroupNCCL properly.
        fc1 = nn.Linear(16, 8).cuda(self.rank)
        fc2 = nn.Linear(8, 4).cuda((self.rank + 1) % self.world_size)
        model = nn.Sequential(
            fc1,
            fc2
        )
        model = Pipe(model, chunks=2, checkpoint=checkpoint)
        model = DistributedDataParallel(model)
        out = model(torch.rand(16, 16).cuda(self.rank)).local_value()
        out.sum().backward()

        # Check grads
        output = [torch.empty_like(fc1.weight.grad), torch.empty_like(fc1.weight.grad)]
        dist.all_gather(output, fc1.weight.grad)
        self.assertEqual(output[0], output[1])

        output = [torch.empty_like(fc2.weight.grad), torch.empty_like(fc2.weight.grad)]
        dist.all_gather(output, fc2.weight.grad)
        self.assertEqual(output[0], output[1])
