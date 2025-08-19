import numpy as np
import torch
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.testing import assert_close
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class DTensorTest(DTensorTestBase):
    @property
    def backend(self):
        return "gloo"

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def world_size(self):
        return np.prod(list(self.mesh_dim_sizes.values())).item()

    @property
    def mesh_dim_sizes(self) -> dict[str, int]:
        """Mapping from mesh dimension names to sizes."""
        # The following mesh shapes cause the following errors:
        #
        # error out on self.destroy_pg()
        return {"data": 2, "fsdp": 3, "tensor": 5}
        # return {"data": 16}

        # return {"data": 8, "fsdp": 1, "tensor": 1}
        #
        # SIGSEGV:
        # return {"data": 4, "fsdp": 2, "tensor": 1}
        # return {"data": 8, "fsdp": 1, "tensor": 1}

        # The following mesh shapes work:
        # return {"data": 8}
        # return {"data": 6}

    def build_device_mesh(self) -> DeviceMesh:
        print("self.device_type: ", self.device_type)
        return init_device_mesh(
            self.device_type,
            mesh_shape=tuple(self.mesh_dim_sizes.values()),
            mesh_dim_names=tuple(self.mesh_dim_sizes.keys()),
        )

    def assert_dtensor_close(self, x: DTensor | Tensor, y: DTensor | Tensor, **kwargs):
        x = x.full_tensor() if isinstance(x, DTensor) else x
        y = y.full_tensor() if isinstance(y, DTensor) else y
        assert_close(x, y.to(x.device), **kwargs)

    @with_comms
    def test_dtensor_constructor(self):
        torch.manual_seed(1234)
        device_mesh = self.build_device_mesh()
        print(device_mesh)
        placements = tuple([Shard(0)] + [Replicate()] * (len(self.mesh_dim_sizes) - 1))
        print(placements)
        local_tensor = torch.randn(3, 3, requires_grad=True)
        num_data_replicas = self.mesh_dim_sizes["data"]

        spec = DTensorSpec(
            device_mesh,
            placements,
            tensor_meta=TensorMeta(
                torch.Size([num_data_replicas * 3, 3]),
                local_tensor.stride(),
                local_tensor.dtype,
            ),
        )

        dist_tensor = DTensor(
            local_tensor,
            spec,
            requires_grad=True,
        )
        print(dist_tensor.shape)
        self.assertEqual(dist_tensor.size(), torch.Size((num_data_replicas * 3, 3)))
        sums_from_dist = dist_tensor.view(num_data_replicas, *local_tensor.shape).sum(
            dim=(1, 2)
        )
        sums_from_local = DTensor.from_local(
            local_tensor.sum().view([1]),
            device_mesh=device_mesh,
            placements=placements,
        )
        print(sums_from_dist, "vs", sums_from_local)
        self.assert_dtensor_close(sums_from_dist, sums_from_local)


if __name__ == "__main__":
    run_tests()
