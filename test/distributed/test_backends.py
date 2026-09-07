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
        if "cuda" in device:
            if dist.get_default_backend_for_device(device) != "nccl":
                raise AssertionError(
                    f"Expected nccl, got {dist.get_default_backend_for_device(device)}"
                )
        elif "cpu" in device:
            if dist.get_default_backend_for_device(device) != "gloo":
                raise AssertionError(
                    f"Expected gloo, got {dist.get_default_backend_for_device(device)}"
                )
        elif "mps" in device:
            if dist.get_default_backend_for_device(device) != "gloo":
                raise AssertionError(
                    f"Expected gloo, got {dist.get_default_backend_for_device(device)}"
                )
        elif "xpu" in device:
            if dist.get_default_backend_for_device(device) != "xccl":
                raise AssertionError(
                    f"Expected xccl, got {dist.get_default_backend_for_device(device)}"
                )
        else:
            with self.assertRaises(ValueError):
                dist.get_default_backend_for_device(device)

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
        # Derive the expected name from the backend type instead of accepting
        # any "custom" process group, which would hide a wrong-backend PG.
        expected_type = dist.Backend.backend_type_map[backend]
        expected_name = (
            "custom"
            if expected_type == dist.ProcessGroup.BackendType.CUSTOM
            else backend
        )
        self.assertEqual(backend_pg, expected_name)
        dist.destroy_process_group()


instantiate_device_type_tests(
    TestMiscCollectiveUtils, globals(), allow_mps=True, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
