# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import fully_shard
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_dist_composable import (
    CompositeModel,
    UnitModule,
)
from torch.distributed.fsdp._common_utils import (
    _get_fully_sharded_submodule_names,
    _get_managed_param_names,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)



class TestUtils(FSDPTest):
    @property
    def world_size(self):
        return 2

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    def test_get_fully_sharded_submodule_names(self):
        model = CompositeModel(torch.device("cuda"))
        fully_shard(
            model,
            policy=ModuleWrapPolicy({UnitModule}),
        )
        sharded_submodule_names = _get_fully_sharded_submodule_names(model)
        self.assertEqual(sharded_submodule_names, ['UnitModule', 'UnitModule', 'CompositeModel'])

    @skip_if_lt_x_gpu(2)
    def test_get_managed_param_names(self):
        model = CompositeModel(torch.device("cuda"))
        fully_shard(
            model,
            policy=ModuleWrapPolicy({UnitModule}),
        )
        managed_param_names = _get_managed_param_names(model)
        self.assertEqual(managed_param_names, [
            ['l1.weight', 'l1.bias', 'l2.weight', 'l2.bias'], 
            ['u1.l1.weight', 'u1.l1.bias', 'u1.seq.1.weight', 'u1.seq.1.bias', 'u1.l2.weight', 'u1.l2.bias'], 
            ['u2.l1.weight', 'u2.l1.bias', 'u2.seq.1.weight', 'u2.seq.1.bias', 'u2.l2.weight', 'u2.l2.bias']
        ])

if __name__ == "__main__":
    run_tests()