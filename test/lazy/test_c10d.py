# Owner(s): ["oncall: jit"]

import copy
import os
import sys
import torch
import torch._lazy
import torch._lazy.ts_backend
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from datetime import timedelta
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN, NO_MULTIPROCESSING_SPAWN
from torch.testing._internal.common_distributed import MultiProcessTestCase

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if not torch.cuda.is_available() or not dist.is_nccl_available():
    print("CUDA/NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print("Skip dev-asan as torch + multiprocessing spawn have known issues", file=sys.stderr)
    sys.exit(0)

if NO_MULTIPROCESSING_SPAWN:
    print("Spawn not available, skipping tests.", file=sys.stderr)
    sys.exit(0)

torch._lazy.ts_backend.init()
os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())

class TestDistributedLazy(MultiProcessTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)
        os.environ["LTC_TS_CUDA"] = "1"

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        # initialize the process group
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

        self.run_test(test_name, pipe)
        dist.destroy_process_group()
        sys.exit(0)

    # Needed since MultiProcessTestCase assumes a world_size of 4, but we
    # run these tests under other various world_sizes.
    @property
    def world_size(self):
        return int(os.environ["WORLD_SIZE"])

    def _all_close(self, parameters_a, parameters_b):
        for param_a, param_b in zip(parameters_a, parameters_b):
            self.assertEqual(param_a.cpu(), param_b.cpu())

    def _broadcast(self, tensor, source):
        if self.rank == 0:
            tensor = source.to(tensor.device)
        dist.broadcast(tensor, 0)
        return tensor

    def test_BroadcastLazy(self):
        # disable all JIT optimizations and fusions.
        torch._C._jit_set_bailout_depth(0)

        source = torch.randn(2, 3)
        device_lazy = torch.device("lazy", self.rank)
        result_lazy = self._broadcast(torch.zeros(2, 3).to(device_lazy), source)
        torch._lazy.mark_step(str(device_lazy))

        device_cuda = torch.device("cuda", self.rank)
        result_cuda = self._broadcast(torch.zeros(2, 3).to(device_cuda), source)

        self._all_close(result_lazy, result_cuda)

    def _all_reduce(self, tensor):
        dist.all_reduce(tensor)
        return tensor

    def test_AllReduceLazy(self):
        # disable all JIT optimizations and fusions.
        torch._C._jit_set_bailout_depth(0)

        source = torch.full((2, 3), self.rank + 1)
        device_lazy = torch.device("lazy", self.rank)
        result_lazy = self._all_reduce(source.to(device_lazy))
        torch._lazy.mark_step(str(device_lazy))

        device_cuda = torch.device("cuda", self.rank)
        result_cuda = self._all_reduce(source.to(device_cuda))

        self._all_close(result_lazy, result_cuda)


if __name__ == "__main__":
    run_tests()
