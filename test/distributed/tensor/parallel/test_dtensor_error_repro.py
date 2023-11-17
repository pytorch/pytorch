# Owner(s): ["oncall: distributed"]

import copy
import io
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

import torch.nn.functional as F
from torch.distributed._tensor import DTensor, init_device_mesh, Replicate, Shard
from torch.distributed._device_mesh import init_device_mesh
from torch.distributed._tensor import distribute_tensor, DTensor, Shard
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import (
    _get_module_fsdp_state,
    clean_tensor_name,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PairwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.fsdp import DTensorExtensions
from torch.distributed.tensor.parallel.input_reshard import input_reshard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class TestDTensorErrorRepro(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_repro_full_tensor_hang_error(self):
        mesh = init_device_mesh("cuda", (2, 2))
        global_tensor = torch.rand((10, 5), device="cuda")
        dt = distribute_tensor(global_tensor, mesh, [Shard(0), Shard(0)])
        # Sometimes it hang on all ranks. Sometimes it hang on part of the ranks,
        # as I can see the print on some of the ranks but not all of them.
        print(dt.full_tensor())

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_repro_size_mismatch_error(self):
        mesh = init_device_mesh("cuda", (2, 2))
        global_tensor = torch.rand((10, 5), device="cuda")
        dt = distribute_tensor(global_tensor, mesh, [Shard(0), Shard(0)])
        dt_replicate = dt.redistribute(placements=(Replicate(), Replicate()))
        print(f"{self.rank=}, {global_tensor.size()}, {dt_replicate.size()}")
        # assert would fail here since the size is different
        """
        An error example:
        [rank1]:[2023-11-17 15:31:22,854] torch.testing._internal.common_distributed:
        [ERROR] AssertionError: The values for attribute 'shape' do not match:
        torch.Size([10, 5]) != torch.Size([8, 5]).
        """
        self.assertEqual(global_tensor, dt_replicate.to_local())

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_repro_hang_on_print_after_redistribute(self):
        mesh = init_device_mesh("cuda", (2, 2))
        global_tensor = torch.rand((10, 5), device="cuda")
        dt = distribute_tensor(global_tensor, mesh, [Shard(0), Shard(0)])
        dt_replicate = dt.redistribute(placements=(Replicate(), Replicate()))
        print(f"{self.rank=}, finish redistribute.")
        # Although it would finish redistribute, it would hang when I try to print
        # the value of the redistributed dtensor.
        # Same as the full_tensor error, sometimes it display the values from 1-2 ranks.
        print(f"{self.rank=}, {dt_replicate=}")

if __name__ == "__main__":
    run_tests()
