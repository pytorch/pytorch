# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch.distributed.tensor import distribute_tensor, Shard
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    with_comms,
)


if TEST_WITH_DEV_DBG_ASAN:
    print("Skip dev-asan", file=sys.stderr)
    sys.exit(0)


class TestDTensorFFTOps(DTensorTestBase):
    @with_comms
    def test_fft2_preserves_batch_shard(self):
        mesh = self.build_device_mesh()

        x = torch.randn(8, 4, 4, device=self.device_type, dtype=torch.cfloat)
        dt = distribute_tensor(x, mesh, [Shard(0)])

        out = torch.fft.fft2(dt, dim=(-2, -1))

        self.assertEqual(out.placements, (Shard(0),))
        self.assertEqual(out.full_tensor(), torch.fft.fft2(x, dim=(-2, -1)))


instantiate_device_type_tests = create_local_tensor_test_class(TestDTensorFFTOps)

if __name__ == "__main__":
    run_tests()
