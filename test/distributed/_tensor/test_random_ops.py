# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools

import torch

from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed._tensor.random import (
    _calc_shard_linear_idx,
    _get_rng_offset,
    _get_shard_coord,
    _get_shard_size,
    get_rng_state,
    manual_seed,
)

from torch.distributed.distributed_c10d import broadcast_object_list

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
    def test_shard_info(self):
        # test internal help functions:
        #   _get_shard_coord, _get_shard_size, _calc_shard_linear_idx
        mesh = torch.arange(self.world_size).reshape(2, 2, -1)
        device_mesh = DeviceMesh(self.device_type, mesh)
        local_tensor = torch.empty(*[self.world_size for _ in mesh.size()])

        placements_list = [  # this list of placements should be enough to cover
            [Shard(0), Shard(1), Shard(2)],
            [Shard(2), Shard(1), Shard(0)],
            [Shard(1), Replicate(), Shard(0)],
            [Replicate(), Replicate(), Replicate()],
        ]
        exp_shard_coord = [
            {0: [0, 0, 0], 1: [0, 1, 0], 2: [1, 0, 0], 3: [1, 1, 0]},
            {0: [0, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1], 3: [0, 1, 1]},
            {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 1, 0], 3: [0, 1, 0]},
            {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]},
        ]
        exp_shard_size = [
            [2, 2, 1],
            [1, 2, 2],
            [1, 2, 1],
            [1, 1, 1],
        ]
        exp_shard_linear_idx = [
            {0: 0, 1: 1, 2: 2, 3: 3},
            {0: 0, 1: 2, 2: 1, 3: 3},
            {0: 0, 1: 0, 2: 1, 3: 1},
            {0: 0, 1: 0, 2: 0, 3: 0},
        ]

        for idx, placements in enumerate(placements_list):
            dtensor = DTensor.from_local(local_tensor, device_mesh, placements)
            spec = dtensor._spec
            shard_coord = _get_shard_coord(spec)
            shard_size = _get_shard_size(spec)
            shard_linear_idx = _calc_shard_linear_idx(shard_coord, shard_size)
            self.assertEqual(shard_coord, exp_shard_coord[idx][self.rank])
            self.assertEqual(shard_size, exp_shard_size[idx])
            self.assertEqual(shard_linear_idx, exp_shard_linear_idx[idx][self.rank])

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

    @with_comms
    @skip_unless_torch_gpu
    def test_deterministic_uniform_nd(self):
        mesh = torch.arange(self.world_size).reshape(2, 2, -1)
        device_mesh = DeviceMesh(self.device_type, mesh)
        _local_tensor = torch.empty(
            *[self.world_size for _ in mesh.size()], device="cuda"
        )
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

        for placements, dim_map in zip(placements_list, dim_map_list):
            dtensor = DTensor.from_local(_local_tensor, device_mesh, placements)
            spec = dtensor._spec
            self.assertEqual(spec.dim_map, dim_map)

            # get shard information
            shard_coord = _get_shard_coord(spec)
            shard_size = _get_shard_size(spec)
            shard_linear_idx = _calc_shard_linear_idx(shard_coord, shard_size)

            # compute local size
            local_tensor_size = list(_local_tensor.size())

            # get rng offset for checking correctness
            global_size = dtensor.numel()
            old_offset = _get_rng_offset(device_mesh)
            post_op_offset = old_offset + global_size

            # random op call
            dtensor.uniform_(0, 1)

            # check rng offset is correctly synchroized after performing the op
            self.check_rng_state(1234, post_op_offset, device_mesh)

            # the local shard
            local_tensor = dtensor.to_local()
            dtensor = dtensor.redistribute(
                device_mesh, [Replicate(), Replicate(), Replicate()]
            )
            # the allgather-ed tensor
            local_tensor_gathered = dtensor.to_local()
            # generate shard's range on each dim
            shard_range_on_dim = [
                list(range(0, l_dist + 1, l_local))
                for l_dist, l_local in zip(dtensor.size(), local_tensor_size)
            ]
            shard_range_on_dim = [
                [(dim_range[i], dim_range[i + 1]) for i in range(len(dim_range) - 1)]
                for dim_range in shard_range_on_dim
            ]

            shard_range_comb = list(itertools.product(*shard_range_on_dim))
            shard_range_comb = [
                [slice(*t) for t in shard_range] for shard_range in shard_range_comb
            ]

            for idx in range(len(shard_range_comb)):
                slice_idx = shard_range_comb[idx]
                if idx == shard_linear_idx:
                    self.assertEqual(local_tensor_gathered[slice_idx], local_tensor)
                else:
                    self.assertNotEqual(local_tensor_gathered[slice_idx], local_tensor)


if __name__ == "__main__":
    run_tests()
