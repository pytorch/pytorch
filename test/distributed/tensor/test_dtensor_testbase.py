# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import numpy as np

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification, run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class DTensorTestBaseUtilCPUTest(DTensorTestBase):
    """
    This class tests if the basic functionalities of DTensorTestBase are
    working as expected on CPU, regardless of the presence of CUDA devices.
    """

    hw_classification = HardwareClassification.CPU

    @property
    def backend(self):
        return "gloo"

    @property
    def world_size(self):
        return np.prod(list(self.mesh_dim_sizes.values())).item()

    @property
    def mesh_dim_sizes(self) -> dict[str, int]:
        """Mapping from mesh dimension names to sizes."""
        return {"data": 2, "fsdp": 3, "tensor": 5}

    def _build_device_mesh(self, device_type: str) -> DeviceMesh:
        return init_device_mesh(
            device_type,
            mesh_shape=tuple(self.mesh_dim_sizes.values()),
            mesh_dim_names=tuple(self.mesh_dim_sizes.keys()),
        )

    @with_comms
    def test_dtensor_testbase_destroy_pg(self, device):
        # This tests destroy_pg() correctly finishes.
        device_type = torch.device(device).type
        device_mesh = self._build_device_mesh(device_type)  # noqa: F841


instantiate_device_type_tests(
    DTensorTestBaseUtilCPUTest,
    globals(),
    only_for=["cpu"],
)


if __name__ == "__main__":
    run_tests()
