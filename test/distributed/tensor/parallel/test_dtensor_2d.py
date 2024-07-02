import torch
from torch.distributed._tensor import distribute_tensor, init_device_mesh, Shard
from torch.distributed._tensor._utils import compute_local_shape_and_global_offset
from torch.testing._internal.common_utils import run_tests

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

class TestDTensor2D(DTensorTestBase):
    @property
    def world_size(self):
        return 6

    @with_comms
    def test_dtensor_2d_uneven(self):
        global_tensor = torch.arange(8)
        mesh = init_device_mesh(self.device_type, (3, 2), mesh_dim_names=("dp", "tp"))
        dtensor = distribute_tensor(global_tensor, mesh, (Shard(0), Shard(0)))
        # local shards seems wrong
        print(f"{self.rank=}, {dtensor.to_local().shape=}, {dtensor.to_local()=}")

        shape, offset = compute_local_shape_and_global_offset(global_tensor.shape, mesh, (Shard(0), Shard(0)))
        # local shape and offsets also seem wrong
        print(f"compute_local_shape_and_global_offset -- {self.rank=}, {shape=}, {offset=}")


if __name__ == "__main__":
    run_tests()
