# Owner(s): ["oncall: distributed"]

import sys
import torch
import torch.distributed as dist
import unittest
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn as nn

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

HAS_CUDA = torch.cuda.is_available()

class Test(TestCase):
    @unittest.skipIf(not HAS_CUDA, 'No CUDA')
    def test_construct_fsdp(self):
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=2, store=store
        )
        sharded_module = FSDP(nn.Linear(2, 3, device='cuda'))

if __name__ == "__main__":
    run_tests()
