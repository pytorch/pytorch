# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch

from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed._tensor.random import _set_offset, get_rng_state, manual_seed

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_unless_torch_gpu,
    with_comms,
)


class DistTensorRandomOpTest(DTensorTestBase):
    @with_comms
    @skip_unless_torch_gpu
    def test_manual_seed(self):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        manual_seed(1234, device_mesh)
        with self.assertRaisesRegex(RuntimeError, "different seed values"):
            manual_seed(self.rank, device_mesh)

    @with_comms
    @skip_unless_torch_gpu
    def test_deterministic_uniform_1d(self):
        def check_rng_state(seed: int, offset: int, device_mesh: DeviceMesh) -> None:
            state = get_rng_state(device_mesh)
            seed_int64 = state[-16:-8].view(torch.int64)
            offset_int64 = state[-8:].view(torch.int64)
            self.assertEqual(seed_int64, torch.tensor([seed]))
            self.assertEqual(offset_int64, torch.tensor([offset]))

        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        size = [4, 1]

        # initialize rng state
        manual_seed(1234, device_mesh)
        check_rng_state(1234, 0, device_mesh)
        _tensor = torch.empty(*size, device="cuda")
        dtensor = DTensor.from_local(_tensor, device_mesh, [Shard(1)])

        # preprocess rng offset
        global_size = dtensor.numel()
        local_size = dtensor.to_local().numel()
        state = get_rng_state(device_mesh)
        offset = state[-8:].view(torch.int64)[0].item()
        _set_offset(offset + self.rank * local_size, device_mesh)
        check_rng_state(1234, self.rank * local_size, device_mesh)

        # random op call
        dtensor.uniform_(0, 1)

        # postprocess rng offset
        _set_offset(offset + global_size, device_mesh)
        check_rng_state(1234, global_size, device_mesh)

        dtensor = dtensor.redistribute(device_mesh, [Replicate()])
        local_tensor = dtensor.to_local()

        for shard_num in range(self.world_size):
            if self.rank == shard_num:
                self.assertEqual(local_tensor[:, shard_num], local_tensor[:, self.rank])
            else:
                self.assertNotEqual(
                    local_tensor[:, shard_num], local_tensor[:, self.rank]
                )

        # TODO: support dropout
        with self.assertRaisesRegex(RuntimeError, "supported"):
            dropout = torch.nn.Dropout(p=0.2)
            dtensor = dropout(dtensor)


if __name__ == "__main__":
    run_tests()
