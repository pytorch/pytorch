# Owner(s): ["oncall: distributed"]
#
# Tests specific to the in-tree torchcomms NCCL backends.

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


class ProcessGroupNCCLLazyTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl-lazy"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_lazy_pair_channels(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        before_collective = backend._num_active_channels()
        t = torch.full((4,), 1.0, device=self.device)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        self.assertEqual(backend._num_active_channels(), before_collective)

        send_t = torch.full((4,), float(self.rank), device=self.device)
        recv_t = torch.empty((4,), device=self.device)
        nxt = (self.rank + 1) % self.world_size
        prev = (self.rank - 1) % self.world_size
        if self.rank % 2 == 0:
            dist.send(send_t, nxt)
            dist.recv(recv_t, prev)
        else:
            dist.recv(recv_t, prev)
            dist.send(send_t, nxt)
        torch.cuda.synchronize()
        self.assertEqual(recv_t, torch.full((4,), float(prev), device=self.device))

        expected = 1 if nxt == prev else 2
        self.assertGreaterEqual(backend._num_active_channels(), expected)


if __name__ == "__main__":
    if TEST_CUDA:
        run_tests()
