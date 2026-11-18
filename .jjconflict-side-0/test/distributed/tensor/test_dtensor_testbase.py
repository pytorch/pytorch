# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from unittest.mock import patch

import numpy as np

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor import common_dtensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorContinuousTestBase,
    DTensorTestBase,
    with_comms,
)


class DTensorContinuousTestBaseTest(TestCase):
    def test_device_selection_matches_instance_property(self):
        class Fixture(DTensorContinuousTestBase):
            world_size = 4

        with (
            patch.object(common_dtensor, "TEST_CUDA", True),
            patch.object(common_dtensor, "DEVICE_TYPE", "cuda"),
        ):
            for count in (0, 1, 4, 8):
                with (
                    self.subTest(device_count=count),
                    patch.object(common_dtensor, "DEVICE_COUNT", count, create=True),
                ):
                    expected = "cpu" if count < Fixture.world_size else "cuda"
                    self.assertEqual(Fixture().device_type, expected)
                    self.assertEqual(
                        Fixture.backend_str(), "gloo" if expected == "cpu" else "nccl"
                    )

    def test_cpu_fallback_backend(self):
        class FallbackTest(DTensorContinuousTestBase):
            world_size = 4

        with patch.object(common_dtensor, "DEVICE_COUNT", 1, create=True):
            self.assertEqual(FallbackTest.backend_str(), "gloo")

    def test_cpu_fallback_does_not_bind_accelerator(self):
        class FallbackTest(DTensorContinuousTestBase):
            world_size = 4

        with (
            patch.object(common_dtensor, "DEVICE_COUNT", 1, create=True),
            patch.object(torch.accelerator, "is_available", return_value=True),
            patch.object(torch.accelerator, "device_count", return_value=1),
            patch.object(torch.accelerator, "set_device_index") as set_device,
            patch.object(MultiProcContinuousTest, "_init_pg") as init_pg,
        ):
            FallbackTest._init_pg(3, 4, "unused")
            init_pg.assert_called_once_with(3, 4, "unused")
            set_device.assert_not_called()


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
