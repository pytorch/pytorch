# Owner(s): ["oncall: distributed"]

import os

import torch.distributed as dist
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


"""
common backend API tests
"""


class TestMiscCollectiveUtils(TestCase):
    def test_device_to_backend_mapping(self, device) -> None:
        """
        Test device to backend mapping
        """
        if "cuda" in device:
            assert dist.get_default_backend_for_device(device) == "nccl"
        elif "cpu" in device:
            assert dist.get_default_backend_for_device(device) == "gloo"
        elif "hpu" in device:
            assert dist.get_default_backend_for_device(device) == "hccl"
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
        assert backend_pg == backend
        dist.destroy_process_group()


devices = ["cpu", "cuda", "hpu"]
instantiate_device_type_tests(TestMiscCollectiveUtils, globals(), only_for=devices)

if __name__ == "__main__":
    run_tests()
