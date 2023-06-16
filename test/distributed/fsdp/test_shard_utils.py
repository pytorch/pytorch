# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist

from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import Shard
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import (
    _create_chunk_dtensor,
    _create_chunk_sharded_tensor,
    _gather_state_dict,
)
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
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


class TestShardUtilsDistributedDTensor(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _create_tensor(self, *size):
        # Keep everything deterministic.
        torch.manual_seed(0)
        return torch.rand(*size).cuda()

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_create_chunk_dtensor(self):
        device_mesh = self.build_device_mesh()

        for size in ((1,), (1, 6), (12,), (12, 6), (25,), (25, 6)):
            tensor = self._create_tensor(*size)
            tensor_chunks = torch.chunk(tensor, self.world_size, dim=0)

            dtensor = _create_chunk_dtensor(tensor, self.rank, device_mesh)
            local_tensor = dtensor.to_local()

            if local_tensor.numel() != 0:
                self.assertEqual(local_tensor, tensor_chunks[self.rank])
            else:
                self.assertEqual(self.rank >= len(tensor_chunks), True)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_gather_state_dict_dtensor(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        torch.random.manual_seed(dist.get_rank())
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        state_dict = {"dtensor": dist_tensor}

        gathered_state_dict = _gather_state_dict(state_dict)
        expected_gathered_dtensor = device_mesh.all_gather(
            dist_tensor.to_local(), mesh_dim=0, gather_dim=0
        )
        self.assertEqual(expected_gathered_dtensor, gathered_state_dict["dtensor"])


if __name__ == "__main__":
    run_tests()
