# Owner(s): ["oncall: distributed"]

import unittest

import torch

from torch.distributed._composable.fsdp._fsdp_init import _normalize_device
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_fsdp import FSDPTestMultiThread
from torch.testing._internal.common_utils import run_tests


class TestFullyShardInitDevice(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 1

    def test_normalize_device_cpu(self):
        for device in ("cpu", torch.device("cpu")):
            self.assertEqual(_normalize_device(device), torch.device("cpu"))

    def test_normalize_device_meta(self):
        for device in ("meta", torch.device("meta")):
            self.assertEqual(_normalize_device(device), torch.device("meta"))

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_normalize_device_cuda(self):
        for device in (
            "cuda",
            0,
            "cuda:0",
            torch.device("cuda"),
            torch.device("cuda", 0),
        ):
            self.assertEqual(_normalize_device(device), torch.device("cuda", 0))
        if torch.cuda.device_count() > 1:
            with torch.cuda.device(1):
                for device in ("cuda", torch.device("cuda")):
                    self.assertEqual(_normalize_device(device), torch.device("cuda", 1))


if __name__ == "__main__":
    run_tests()
