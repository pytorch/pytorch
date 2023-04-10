# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch

from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed._tensor.random import _set_offset, get_rng_state, manual_seed

from torch.distributed.distributed_c10d import broadcast_object_list

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_unless_torch_gpu,
    with_comms,
)


class DistTensorRandomOpTest(DTensorTestBase):
    @with_comms
    def test_device_mesh_init(self):
        # device mesh init should sync seed and store it as an attribute
        # However, we have not figured out how we want to use the seed, so
        # temporarily device_mesh._seed will just equal to each rank's
        # `initial_seed()`.
        torch.cuda.manual_seed(self.rank)
        object_list = [torch.cuda.initial_seed()]
        broadcast_object_list(object_list)
        seed_from_rank_0 = int(object_list[0])

        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # TODO: sync seed once we figure out how we want to use the seed
        if self.rank != 0:
            with self.assertRaises(AssertionError):
                self.assertEqual(seed_from_rank_0, device_mesh._seed)

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

        # allgather the local tensors
        dtensor = dtensor.redistribute(device_mesh, [Replicate()])
        local_tensor = dtensor.to_local()

        # compare with local tensors from other ranks
        for other_rank in range(self.world_size):
            if self.rank != other_rank:
                # other rank should have a different local tensor
                self.assertNotEqual(
                    local_tensor[:, self.rank], local_tensor[:, other_rank]
                )

    @with_comms
    @skip_unless_torch_gpu
    def test_deterministic_dropout_1d(self):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        size = [4, 1]

        # initialize rng state
        manual_seed(1234, device_mesh)
        _tensor = torch.empty(*size, device="cuda")
        dtensor = DTensor.from_local(_tensor, device_mesh, [Shard(1)])

        # a random op call shifts the offset
        dtensor.uniform_(0, 1)

        # the dtensor is now replicate on all ranks
        dtensor = dtensor.redistribute(device_mesh, [Replicate()])

        # TODO: support dropout
        with self.assertRaisesRegex(RuntimeError, "supported"):
            dropout = torch.nn.Dropout(p=0.2)
            dtensor = dropout(dtensor)

            # allgather the local tensors
            dtensor = DTensor.from_local(dtensor.to_local(), device_mesh, [Shard(0)])
            dtensor = dtensor.redistribute(device_mesh, [Replicate()])
            local_tensor = dtensor.to_local()

            # compare with local tensors from other ranks
            self_slice = slice(4 * self.rank, 4 * self.rank + 4)
            for other_rank in range(self.world_size):
                if self.rank != other_rank:
                    # other rank should have an identical local tensor
                    other_slice = slice(4 * other_rank, 4 * other_rank + 4)
                    self.assertEqual(
                        local_tensor[self_slice, :], local_tensor[other_slice, :]
                    )


if __name__ == "__main__":
    run_tests()
