# Owner(s): ["oncall: distributed"]

# Description:
# this file tests the testing utils defined in torch/testing/_internal/common_distributed.py
# on test classes that are defined to test PyTorch Distributed components such as
# DTensor and FSDP.

import torch

from torch.testing._internal.common_distributed import MultiProcessTestCase, MultiThreadedTestCase, skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms

# file: torch/testing/_internal/common_fsdp.py
class UtilsTestOnFSDPTest(FSDPTest):
    @property
    def world_size(self):
        # this is the largest number of GPUs on any test instance
        return 8

    @skip_if_lt_x_gpu(8)
    def test_skip_if_lt_x_gpu(self):
        if torch.cuda.device_count() < 8:
            raise RuntimeError("This test is supposed to be skipped. Examine why `skip_if_lt_x_gpu` is not working.")


# file: torch/testing/_internal/common_distributed.py
class UtilsTestOnMultiProcessTestCase(MultiProcessTestCase):
    @property
    def world_size(self):
        # this is the largest number of GPUs on any test instance
        return 8

    @skip_if_lt_x_gpu(8)
    def test_skip_if_lt_x_gpu(self):
        if torch.cuda.device_count() < 8:
            raise RuntimeError("This test is supposed to be skipped. Examine why `skip_if_lt_x_gpu` is not working.")


class UtilsTestOnMultiThreadedTestCase(MultiThreadedTestCase):
    @property
    def world_size(self):
        # this is the largest number of GPUs on any test instance
        return 8

    @skip_if_lt_x_gpu(8)
    def test_skip_if_lt_x_gpu(self):
        if torch.cuda.device_count() < 8:
            raise RuntimeError("This test is supposed to be skipped. Examine why `skip_if_lt_x_gpu` is not working.")


# file: torch/testing/_internal/distributed/_tensor/common_dtensor.py
class UtilsTestOnDTensorTestBase(DTensorTestBase):
    @property
    def world_size(self):
        # this is the largest number of GPUs on any test instance
        return 8

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_skip_if_lt_x_gpu(self):
        if torch.cuda.device_count() < 8:
            raise RuntimeError("This test is supposed to be skipped. Examine why `skip_if_lt_x_gpu` is not working.")
