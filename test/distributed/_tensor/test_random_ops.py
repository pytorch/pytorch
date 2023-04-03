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
    def check_rng_state(self, seed: int, offset: int, device_mesh: DeviceMesh) -> None:
        state = get_rng_state(device_mesh)
        seed_int64 = state[-16:-8].view(torch.int64)
        offset_int64 = state[-8:].view(torch.int64)
        self.assertEqual(seed_int64, torch.tensor([seed]))
        self.assertEqual(offset_int64, torch.tensor([offset]))

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
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        size = [4, 1]

        # initialize rng state
        manual_seed(1234, device_mesh)
        self.check_rng_state(1234, 0, device_mesh)
        _tensor = torch.empty(*size, device="cuda")
        dtensor = DTensor.from_local(_tensor, device_mesh, [Shard(1)])

        # get rng offset for checking correctness
        global_size = dtensor.numel()
        state = get_rng_state(device_mesh)
        offset = state[-8:].view(torch.int64)[0].item()
        offset_after_op = offset + global_size

        # random op call
        dtensor.uniform_(0, 1)

        # check rng offset is correctly synchroized after perform op 
        self.check_rng_state(1234, offset_after_op, device_mesh)

        dtensor = dtensor.redistribute(device_mesh, [Replicate()])
        local_tensor = dtensor.to_local()

        for shard_num in range(self.world_size):
            if self.rank == shard_num:
                self.assertEqual(local_tensor[:,shard_num], local_tensor[:,self.rank])
            else:
                self.assertNotEqual(local_tensor[:,shard_num], local_tensor[:,self.rank])

        dtensor.uniform_(0, 1)
        local_tensor = dtensor.to_local()
        tensor_list = [torch.empty_like(local_tensor) for i in range(self.world_size)]
        device_mesh.all_gather(tensor_list, local_tensor)
        # check if every rank generate the same random numbers
        for t in tensor_list:
            self.assertEqual(local_tensor, t)

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

    @with_comms
    @skip_unless_torch_gpu
    def test_deterministic_uniform_nd(self):
        mesh = torch.arange(self.world_size).reshape(2, 2, -1)
        device_mesh = DeviceMesh(self.device_type, mesh)
        dtensor_size = [4 for l in mesh.size()]  # DTensor shape replace with self.world_size
        # initialize rng state
        manual_seed(1234, device_mesh)
        self.check_rng_state(1234, 0, device_mesh)

        placements_list = [  # this list of placements should be enough to cover
            [Shard(0), Shard(1), Shard(2)],
            [Shard(2), Shard(1), Shard(0)],
            [Shard(1), Replicate(), Shard(0)],
            [Replicate(), Replicate(), Replicate()],
        ]

        dim_map_list = [
            [0, 1, 2],
            [2, 1, 0],
            [2, 0, -1],
            [-1, -1, -1],
        ]

        coord = device_mesh.get_coordinate()
        assert coord is not None

        for (placements, dim_map) in zip(placements_list, dim_map_list):
            # shard shape:
            shard_shape = [
                mesh.size()[dim] if dim >= 0 else 1
                for dim in dim_map
            ]
            # shard coord:
            shard_coord = [
                coord[dim] if dim >= 0 else 0
                for dim in dim_map
            ]
            strides = [1]
            for l in shard_shape[:0:-1]:
                strides.append(strides[-1] * l)
            strides = strides[::-1]
            # shard idx:
            shard_idx = sum([x * y for x, y in zip(shard_coord, strides)])
            # compute local size
            local_tensor_size = [size // n_shards for size, n_shards in zip(dtensor_size, shard_shape)]
            _tensor = torch.empty(*local_tensor_size, device='cuda')
            dtensor = DTensor.from_local(_tensor, device_mesh, placements)
            self.assertEqual(dtensor._spec.dim_map, dim_map)

            # get rng offset for checking correctness
            global_size = dtensor.numel()
            state = get_rng_state(device_mesh)
            offset = state[-8:].view(torch.int64)[0].item()
            offset_after_op = offset + global_size

            # random op call
            dtensor.uniform_(0, 1)

            # check rng offset is correctly synchroized after perform op 
            self.check_rng_state(1234, offset_after_op, device_mesh)

            local_tensor = dtensor.to_local()
            dtensor = dtensor.redistribute(device_mesh, [Replicate(), Replicate(), Replicate()])
            local_tensor_gathered = dtensor.to_local()
            # generate shard's range on each dim
            shard_range_on_dim = [list(range(0, l+1, l // n)) for l, n in zip(dtensor_size, shard_shape)]
            shard_range_on_dim = [
                [
                    (dim_range[i],dim_range[i+1])
                    for i in range(len(dim_range)-1)
                ]
                for dim_range in shard_range_on_dim
            ]
            from itertools import product
            shard_range_comb = list(product(*shard_range_on_dim))
            shard_range_comb = [
                [
                    slice(*t) for t in shard_range
                ]
                for shard_range in shard_range_comb
            ]

            for idx in range(len(shard_range_comb)):
                slice_idx = shard_range_comb[idx]
                if idx == shard_idx:
                    self.assertEqual(local_tensor_gathered[slice_idx], local_tensor)
                else:
                    self.assertNotEqual(local_tensor_gathered[slice_idx], local_tensor)


if __name__ == "__main__":
    run_tests()
