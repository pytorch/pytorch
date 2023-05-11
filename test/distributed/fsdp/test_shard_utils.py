# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor, _gather_state_dict, _create_chunk_dtensor
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests

from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.placement_types import Shard
import torch.distributed as dist
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

class TestShardUtilsDistributed(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _create_tensor(self, *size):
        # Keep everything deterministic.
        torch.manual_seed(0)
        return torch.rand(*size).cuda()

    @skip_if_lt_x_gpu(2)
    def test_create_chunk_sharded_tensor(self):
        for size in ((1,), (1, 6), (12,), (12, 6), (25,), (25, 6)):
            tensor = self._create_tensor(*size)

            sharded_tensor = _create_chunk_sharded_tensor(
                tensor,
                self.rank,
                self.world_size,
                torch.cuda.device_count(),
                _get_default_group(),
            )
            output = torch.empty(*size).cuda() if self.rank == 0 else None
            sharded_tensor.gather(0, output)
            if self.rank == 0:
                self.assertEqual(tensor, output)

    @skip_if_lt_x_gpu(2)
    def test_create_chunk_dtensor(self):
        for size in ((1,), (1, 6)):
            tensor = self._create_tensor(*size)
            dtensor = _create_chunk_dtensor(
                tensor,
                self.rank,
                self.world_size,
                torch.cuda.device_count(),
                _get_default_group(),
            )
            print(f"rank:{dist.get_rank()}, dtensor:{dtensor}")
            output = torch.empty(*size).cuda() if self.rank == 0 else None
            # sharded_tensor.gather(0, output)
            # if self.rank == 0:
            #     self.assertEqual(tensor, output)
    
    # @skip_if_lt_x_gpu(2)
    @with_comms
    def test_gather_state_dict(self):
        mesh_tensor = torch.arange(self.world_size).reshape(2, 1)
        print("test_gather_state_dict")
        # construct a cuda device mesh
        mesh = DeviceMesh('cpu', mesh_tensor)

        # construct a dist tensor on 2d device mesh and test if works
        shard_spec = [Shard(0), Shard(1)]
        torch.random.manual_seed(dist.get_rank())
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        state_dict = {'dtensor': dist_tensor}

        print(f"rank:{dist.get_rank()}, state_dict: {state_dict}")
        gathered_state_dict = _gather_state_dict(state_dict)
        print(f"gathered_state_dict: {gathered_state_dict}")


if __name__ == "__main__":
    run_tests()
