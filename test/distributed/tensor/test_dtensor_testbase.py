# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import numpy as np

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class DTensorTestBaseUtilCPUTest(DTensorTestBase):
    """
    This class tests if the basic functionalities of DTensorTestBase are
    working as expected on CPU, regardless of the presence of CUDA devices.
    """

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
        return {"data": 2, "fsdp": 3, "tensor": 5}

    def build_device_mesh(self) -> DeviceMesh:
        return init_device_mesh(
            self.device_type,
            mesh_shape=tuple(self.mesh_dim_sizes.values()),
            mesh_dim_names=tuple(self.mesh_dim_sizes.keys()),
        )

    @with_comms
    def test_dtensor_testbase_destroy_pg(self):
        # This tests destroy_pg() correctly finishes.
        device_mesh = self.build_device_mesh()  # noqa: F841


if __name__ == "__main__":
    run_tests()
