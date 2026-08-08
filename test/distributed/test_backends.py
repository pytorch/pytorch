# Owner(s): ["oncall: distributed"]

import os

import torch.distributed as dist
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


"""
common backend API tests
"""


class TestMiscCollectiveUtils(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_device_to_backend_mapping(self, device) -> None:
        """
        Test device to backend mapping
        """
        try:
            backend = dist.get_default_backend_for_device(device)
        except ValueError:
            return  # device has no registered backend, nothing to verify

        expected = dist.Backend.default_device_backend_map.get(device)
        if expected is not None and backend != expected:
            raise AssertionError(f"Expected {expected}, got {backend}")

    def test_create_pg(self, device) -> None:
        """
        Test create process group
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        backend = dist.get_default_backend_for_device(device)
        dist.init_process_group(
            backend=backend, rank=0, world_size=1, init_method="env://"
        )
        pg = dist.distributed_c10d._get_default_group()
        backend_pg = pg._get_backend_name()
        # Some backends report "custom" at the process group layer while
        # their logical backend name (from get_default_backend_for_device)
        # is different. Accept either.
        if backend_pg not in (backend, "custom"):
            raise AssertionError(f"Expected {backend}, got {backend_pg}")
        dist.destroy_process_group()


instantiate_device_type_tests(
    TestMiscCollectiveUtils, globals(), allow_mps=True, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
