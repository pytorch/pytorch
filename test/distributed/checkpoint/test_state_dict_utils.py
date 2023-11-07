# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import Shard
from torch.distributed.checkpoint._state_dict_utils import _gather_state_dict
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)


class TestStateDictUtils(DTensorTestBase):
    @property
    def world_size(self):
        return 2

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
        expected_gathered_dtensor = funcol.all_gather_tensor(
            dist_tensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )
        self.assertEqual(expected_gathered_dtensor, gathered_state_dict["dtensor"])


if __name__ == "__main__":
    run_tests()
