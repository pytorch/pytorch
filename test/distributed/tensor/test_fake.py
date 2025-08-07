# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


class TestFakeDTensor(TestCase):
    def test_fake_dtensor_operations(self):
        # Use FakeTensorMode to handle CUDA tensors without actual CUDA
        fake_mode = FakeTensorMode()
        world_size = 4

        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=world_size
        )
        device_mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            (2, world_size // 2),
        )

        # Create fake CUDA tensor using FakeTensorMode
        with fake_mode:
            x = torch.randn(1, 1, device="cuda")
            x = DTensor.from_local(x, device_mesh, [Shard(0), Shard(1)])

            # Test basic DTensor operations
            self.assertIsInstance(x, DTensor)

            # Test sum operation
            r = x.sum(1)
            self.assertIsInstance(r, DTensor)


if __name__ == "__main__":
    run_tests()
