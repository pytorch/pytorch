# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools

import torch

from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor._utils import compute_local_offset
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed._tensor.random import (
    _calc_shard_linear_idx,
    DTensorEquivalentRandomOps,
    get_rng_state,
    is_rng_supported_mesh,
    manual_seed,
)

from torch.distributed.distributed_c10d import broadcast_object_list

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    skip_unless_torch_gpu,
    with_comms,
)


class DistTensorRandomInitTest(DTensorTestBase):
    def _run_init_op(self, init_op, *args, **kwargs):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        input_size = (8, 4)

        # NOTE: currently random initialization on cuda device has different
        # behavior from other devices. Unify the test once the behavior is unified.
        if not is_rng_supported_mesh(device_mesh):
            input_tensor = torch.randn(*input_size, device=self.device_type)
            dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
            local_tensor_clone = torch.clone(input_tensor)
            torch.manual_seed(self.rank)
            local_tensor_clone = init_op(local_tensor_clone, *args, **kwargs)
            torch.manual_seed(self.rank)
            dtensor = init_op(dtensor, *args, **kwargs)
            self.assertEqual(local_tensor_clone, dtensor.to_local())
        else:
            # create DTensor from Tensor
            _tensor = torch.empty(*input_size, device="cuda")
            dtensor = DTensor.from_local(_tensor, device_mesh, [Shard(1)])

            # DTensor random init
            dtensor = init_op(dtensor, *args, **kwargs)
            local_tensor = dtensor.to_local()

            # allgather the local tensors
            dtensor = dtensor.redistribute(device_mesh, [Replicate()])
            local_tensor_gathered = dtensor.to_local()

            # compare with local tensors from other ranks
            for other_rank in range(self.world_size):
                if self.rank != other_rank:
                    slice_idx = [
                        slice(input_size[0]),
                        slice(
                            other_rank * input_size[1], (other_rank + 1) * input_size[1]
                        ),
                    ]
                    # other rank should have a different local tensor
                    self.assertNotEqual(local_tensor_gathered[slice_idx], local_tensor)

    @with_comms
    def test_init_ops(self):
        self._run_init_op(
            torch.nn.init.kaiming_uniform_,
            a=0,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        self._run_init_op(torch.nn.init.normal_, mean=1.5, std=0.8)
        self._run_init_op(torch.nn.init.uniform_, a=0, b=1.2)


class DistTensorRandomOpTest(DTensorTestBase):
    @with_comms
    @skip_unless_torch_gpu
    def test_device_mesh_init(self):
        # TODO: make this check a wrap around each test case
        # use a random tensor to simply verify the default RNG state is unchanged.
        size = [4, 4]
        torch.cuda.manual_seed(0)
        local_tensor_1 = torch.randn(size, device="cuda")
        torch.cuda.manual_seed(0)

        # device mesh init should sync seed and store it as an attribute
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        seed, offset = get_rng_state(device_mesh)

        # verify the default RNG state is unchanged.
        local_tensor_2 = torch.randn(size, device="cuda")
        self.assertEqual(local_tensor_1, local_tensor_2)

        # verify the device mesh attributes
        object_list = [seed]
        broadcast_object_list(object_list)
        seed_from_rank_0 = int(object_list[0])
        self.assertEqual(seed_from_rank_0, seed)
        self.assertEqual(0, offset)

    @with_comms
    @skip_unless_torch_gpu
    def test_manual_seed(self):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # use a random tensor to simply verify the default RNG state is unchanged.
        size = [4, 4]
        torch.cuda.manual_seed(0)
        local_tensor_1 = torch.randn(size, device="cuda")
        torch.cuda.manual_seed(0)

        manual_seed(1234, device_mesh)
        seed, offset = get_rng_state(device_mesh)

        # verify the default RNG state is unchanged.
        local_tensor_2 = torch.randn(size, device="cuda")
        self.assertEqual(local_tensor_1, local_tensor_2)

        # verify the device mesh attributes
        self.assertEqual(1234, seed)
        self.assertEqual(0, offset)

        # manual_seed requires identical seed argument on all calling ranks
        with self.assertRaisesRegex(RuntimeError, "different seed values"):
            manual_seed(self.rank, device_mesh)

    @with_comms
    @skip_unless_torch_gpu
    def test_deterministic_dropout_1d(self):
        # use a random tensor to simply verify the default RNG state is unchanged.
        torch.cuda.manual_seed(0)
        local_tensor_1 = torch.randn([4, 4], device="cuda")
        torch.cuda.manual_seed(0)

        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        size = [4, 1]

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

        # verify the default RNG state is unchanged.
        local_tensor_2 = torch.randn([4, 4], device="cuda")
        self.assertEqual(local_tensor_1, local_tensor_2)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_deterministic_uniform_2d(self):
        mesh = torch.arange(self.world_size).reshape(2, 2)
        device_mesh = DeviceMesh(self.device_type, mesh)
        _local_tensor = torch.empty(
            *[self.world_size for _ in mesh.size()], device="cuda"
        )

        placements_list = [  # this list of placements should be enough to cover
            [Shard(0), Shard(1)],
            [Shard(1), Shard(0)],
            [Shard(0), Replicate()],
            [Replicate(), Shard(0)],
            [Shard(1), Replicate()],
            [Replicate(), Shard(1)],
            [Replicate(), Replicate()],
        ]

        shard_index_list = [
            {0: 0, 1: 1, 2: 2, 3: 3},
            {0: 0, 1: 2, 2: 1, 3: 3},
            {0: 0, 1: 0, 2: 1, 3: 1},
            {0: 0, 1: 1, 2: 0, 3: 1},
            {0: 0, 1: 0, 2: 1, 3: 1},
            {0: 0, 1: 1, 2: 0, 3: 1},
            {0: 0, 1: 0, 2: 0, 3: 0},
        ]

        coordinate = device_mesh.get_coordinate()
        assert coordinate is not None

        for placements, shard_index in zip(placements_list, shard_index_list):
            dtensor = DTensor.from_local(_local_tensor, device_mesh, placements)

            # check shard information is correct
            shard_coord = [
                coordinate[mesh_dim] if mesh_dim >= 0 else 0
                for mesh_dim in dtensor._spec.dim_map
            ]

            shard_size = [
                device_mesh.size(mesh_dim) if mesh_dim >= 0 else 1
                for mesh_dim in dtensor._spec.dim_map
            ]

            shard_linear_idx = _calc_shard_linear_idx(shard_coord, shard_size)
            self.assertEqual(shard_linear_idx, shard_index[self.rank])

            # compute local size and offset
            local_shard_offset = compute_local_offset(
                dtensor.shape, device_mesh, placements
            )

            # get the local shard size and local shard offset for each shard
            # local_shard_list_on_dim[i] has the list of all shards on that dim
            # as a tuple (local_shard_offset, local_shard_size)
            dtensor_shape = dtensor.shape
            local_shard_list_on_dim = [[(0, l)] for l in dtensor_shape]
            for idx, placement in enumerate(placements):
                if isinstance(placement, Shard):
                    mesh_dim_size = device_mesh.size(idx)
                    shard_dim = placement.dim
                    local_shard_list_on_dim[shard_dim] = []
                    for shard_idx_on_dim in range(mesh_dim_size):
                        shard_size, shard_offset = placement._local_shard_size_on_dim(
                            dtensor_shape[shard_dim],
                            mesh_dim_size,
                            shard_idx_on_dim,
                            return_offset=True,
                        )
                        local_shard_list_on_dim[shard_dim].append(
                            (shard_offset, shard_size)
                        )

            local_shard_comb = itertools.product(*local_shard_list_on_dim)

            # random op call
            dtensor.uniform_(0, 1)

            # the local shard
            local_tensor = dtensor.to_local()
            dtensor = dtensor.redistribute(
                device_mesh, [Replicate(), Replicate(), Replicate()]
            )
            # the allgather-ed tensor
            local_tensor_gathered = dtensor.to_local()

            # compare local tensor with each other shard
            for other_local_shard in local_shard_comb:
                other_local_shard_offset, _ = zip(*other_local_shard)
                slice_idx = [
                    slice(offset, offset + size) for offset, size in other_local_shard
                ]
                if local_shard_offset == other_local_shard_offset:
                    self.assertEqual(local_tensor_gathered[slice_idx], local_tensor)
                else:
                    self.assertNotEqual(local_tensor_gathered[slice_idx], local_tensor)

    @with_comms
    @skip_unless_torch_gpu
    def test_strict_mode(self):
        local_size = [4, 4]
        d_size = [4, 1]
        placements = [Shard(1)]

        with (
            DeviceMesh(self.device_type, torch.arange(self.world_size)) as device_mesh,
            DTensorEquivalentRandomOps(),
        ):
            device = device_mesh.device_type

            # local tensor
            torch.cuda.manual_seed(1234)
            local_tensor = torch.empty(*local_size, device=device)
            local_tensor.uniform_()

            # dtensor
            manual_seed(1234, device_mesh)
            dtensor = DTensor.from_local(
                torch.empty(*d_size, device=device),
                device_mesh,
                placements,
            )
            dtensor.uniform_()
            dtensor = dtensor.redistribute(device_mesh, [Replicate()])

            self.assertEqual(local_tensor, dtensor.to_local())


if __name__ == "__main__":
    run_tests()
