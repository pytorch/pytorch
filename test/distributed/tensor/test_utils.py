# Owner(s): ["oncall: distributed"]


import itertools
import math
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any
from unittest import expectedFailure

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import (
    local_tensor_mode,
    LocalTensor,
    LocalTensorMode,
    maybe_run_for_local_tensor,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._ops.utils import (
    is_tensor_evenly_shardable,
    is_tensor_shardable,
)
from torch.distributed.tensor._utils import (
    _compute_local_shape_and_global_offset,
    compute_global_tensor_info,
    compute_global_tensor_shape,
    compute_local_shape_and_global_offset,
    compute_local_tensor_info,
    ExplicitRedistributionContext,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    generate_shard_orders,
    LocalDTensorTestBase,
    patched_distribute_tensor as _distribute_tensor,
    shard_order_to_placement,
    with_comms,
)


c10d_functional = torch.ops.c10d_functional


def SS(td, sf):
    return _StridedShard(td, split_factor=sf)


S = Shard
R = Replicate()


class LocalTest(TestCase):
    def test_compute_local_shape_and_global_offset_uneven(self):
        # This case is not only 'uneven' bug also has an empty shard
        # (e.g. most DP ranks have local shape 18,4096, one has 8,4096, one has 0,4096
        global_shape = (4096, 4096)
        DP = 30
        TP = 8
        mesh_shape = (DP, TP)
        placements = [_StridedShard(0, split_factor=8), Shard(0)]
        TP_shard_size = global_shape[0] / TP
        for my_coordinate in itertools.product(range(DP), range(TP)):
            local_shape, global_offset = _compute_local_shape_and_global_offset(
                global_shape, mesh_shape, lambda idx: my_coordinate[idx], placements
            )
            dp_rank, tp_rank = my_coordinate
            expected_shard_size = 18
            expected_shard_offset = tp_rank * TP_shard_size + 18 * dp_rank
            if dp_rank == 28:
                expected_shard_size = 8
            elif dp_rank == 29:
                expected_shard_size = 0
                # we define the offset value of a zero-sized shard as the dim size
                # this actually matters, because DCP uses offset to deduplicate shards when saving
                expected_shard_offset = 4096
            self.assertEqual(local_shape, (expected_shard_size, 4096))
            self.assertEqual(global_offset, (expected_shard_offset, 0))

        # S, S uneven without empty
        global_shape = (18, 2)
        DP = 4
        TP = 2
        mesh_shape = (DP, TP)
        placements = [Shard(0), Shard(0)]
        for my_coordinate in itertools.product(range(DP), range(TP)):
            dp_rank, tp_rank = my_coordinate
            local_shape, global_offset = _compute_local_shape_and_global_offset(
                global_shape, mesh_shape, lambda idx: my_coordinate[idx], placements
            )

            dp012_shard_size = 5
            if dp_rank in (0, 1, 2):
                tp0_shard_size = 3
                if tp_rank == 0:
                    expected_shard_offset = dp012_shard_size * dp_rank
                    expected_shard_size = 3
                else:
                    assert tp_rank == 1
                    expected_shard_offset = dp012_shard_size * dp_rank + tp0_shard_size
                    expected_shard_size = 2
            else:
                assert dp_rank == 3
                tp0_shard_size = 2
                if tp_rank == 0:
                    expected_shard_offset = dp012_shard_size * dp_rank
                    expected_shard_size = 2
                else:
                    assert tp_rank == 1
                    expected_shard_offset = dp012_shard_size * dp_rank + tp0_shard_size
                    expected_shard_size = 1
            self.assertEqual(local_shape, (expected_shard_size, 2))
            self.assertEqual(global_offset, (expected_shard_offset, 0))

        # S, S uneven with empty
        global_shape = (13, 2)
        DP = 4
        TP = 2
        mesh_shape = (DP, TP)
        placements = [Shard(0), Shard(0)]
        for my_coordinate in itertools.product(range(DP), range(TP)):
            dp_rank, tp_rank = my_coordinate
            local_shape, global_offset = _compute_local_shape_and_global_offset(
                global_shape, mesh_shape, lambda idx: my_coordinate[idx], placements
            )

            dp012_shard_size = 4
            if dp_rank in (0, 1, 2):
                tp0_shard_size = 2
                if tp_rank == 0:
                    expected_shard_offset = dp012_shard_size * dp_rank
                    expected_shard_size = 2
                else:
                    assert tp_rank == 1
                    expected_shard_offset = dp012_shard_size * dp_rank + tp0_shard_size
                    expected_shard_size = 2
            else:
                assert dp_rank == 3
                tp0_shard_size = 1
                if tp_rank == 0:
                    expected_shard_offset = dp012_shard_size * dp_rank
                    expected_shard_size = 1
                else:
                    assert tp_rank == 1
                    expected_shard_offset = global_shape[0]
                    expected_shard_size = 0
            self.assertEqual(local_shape, (expected_shard_size, 2))
            self.assertEqual(global_offset, (expected_shard_offset, 0))

        # SS, Shard
        global_shape = (18, 2)
        DP = 4
        TP = 2
        mesh_shape = (DP, TP)
        placements = [_StridedShard(0, split_factor=TP), Shard(0)]
        TP_shard_size = int(global_shape[0] / TP)
        for my_coordinate in itertools.product(range(DP), range(TP)):
            dp_rank, tp_rank = my_coordinate
            local_shape, global_offset = _compute_local_shape_and_global_offset(
                global_shape, mesh_shape, lambda idx: my_coordinate[idx], placements
            )
            expected_shard_size = 3
            expected_shard_offset = (
                tp_rank * TP_shard_size + expected_shard_size * dp_rank
            )
            if dp_rank == 3:
                expected_shard_size = 0
                expected_shard_offset = 18
            self.assertEqual(local_shape, (expected_shard_size, 2))
            self.assertEqual(global_offset, (expected_shard_offset, 0))

        # SS, SS
        global_shape = (39, 2)
        DP = 4
        TP = 2
        mesh_shape = (DP, TP)
        placements = [
            _StridedShard(0, split_factor=3),
            _StridedShard(0, split_factor=4),
        ]
        for my_coordinate in itertools.product(range(DP), range(TP)):
            dp_rank, tp_rank = my_coordinate
            local_shape, global_offset = _compute_local_shape_and_global_offset(
                global_shape, mesh_shape, lambda idx: my_coordinate[idx], placements
            )
            if dp_rank in (0, 1, 2):
                tp0_shard_size = 8
                if tp_rank == 0:
                    expected_shard_offset = 4 * dp_rank
                    expected_shard_size = tp0_shard_size
                else:
                    assert tp_rank == 1
                    expected_shard_offset = 4 * dp_rank + 2
                    expected_shard_size = 4
            else:
                assert dp_rank == 3
                tp0_shard_size = 3
                if tp_rank == 0:
                    expected_shard_offset = 4 * dp_rank
                    expected_shard_size = 3
                else:
                    assert tp_rank == 1
                    expected_shard_offset = global_shape[0]
                    expected_shard_size = 0
            self.assertEqual(local_shape, (expected_shard_size, 2))
            self.assertEqual(global_offset, (expected_shard_offset, 0))

        # (Shard, SS)
        global_shape = (18, 2)
        DP = 4
        TP = 2
        mesh_shape = (DP, TP)
        placements = [Shard(0), _StridedShard(0, split_factor=2)]
        for my_coordinate in itertools.product(range(DP), range(TP)):
            dp_rank, tp_rank = my_coordinate
            local_shape, global_offset = _compute_local_shape_and_global_offset(
                global_shape, mesh_shape, lambda idx: my_coordinate[idx], placements
            )
            if dp_rank in (0, 1, 2):
                tp0_shard_size = 3
                if tp_rank == 0:
                    expected_shard_offset = 5 * dp_rank
                    expected_shard_size = tp0_shard_size
                else:
                    assert tp_rank == 1
                    expected_shard_offset = 5 * dp_rank + 2
                    expected_shard_size = 2
            else:
                assert dp_rank == 3
                if tp_rank == 0:
                    expected_shard_offset = 5 * dp_rank
                    expected_shard_size = 2
                else:
                    assert tp_rank == 1
                    expected_shard_offset = 5 * dp_rank + 1
                    expected_shard_size = 1
            self.assertEqual(local_shape, (expected_shard_size, 2))
            self.assertEqual(global_offset, (expected_shard_offset, 0))

        # (Shard, SS, Shard)
        global_shape = (39, 2)
        mesh0, mesh1, mesh2 = 4, 2, 3
        mesh_shape = (mesh0, mesh1, mesh2)
        placements = [Shard(0), _StridedShard(0, split_factor=2), Shard(0)]
        for my_coordinate in itertools.product(
            range(mesh0), range(mesh1), range(mesh2)
        ):
            mesh0_rank, mesh1_rank, mesh2_rank = my_coordinate
            local_shape, global_offset = _compute_local_shape_and_global_offset(
                global_shape, mesh_shape, lambda idx: my_coordinate[idx], placements
            )
            if mesh0_rank in (0, 1, 2):
                if mesh1_rank == 0:
                    if mesh2_rank == 0:
                        expected_shard_offset = 10 * mesh0_rank
                        expected_shard_size = 2
                    elif mesh2_rank == 1:
                        expected_shard_offset = 10 * mesh0_rank + 2
                        expected_shard_size = 2
                    else:
                        expected_shard_offset = 10 * mesh0_rank + 6
                        expected_shard_size = 2
                else:
                    assert mesh1_rank == 1
                    if mesh2_rank == 0:
                        expected_shard_offset = 10 * mesh0_rank + 3
                        expected_shard_size = 2
                    elif mesh2_rank == 1:
                        expected_shard_offset = 10 * mesh0_rank + 8
                        expected_shard_size = 2
                    else:
                        assert mesh2_rank == 2
                        expected_shard_size = 0
                        expected_shard_offset = global_shape[0]
            else:
                assert mesh0_rank == 3
                if mesh1_rank == 0:
                    if mesh2_rank in (0, 1):
                        expected_shard_offset = 10 * mesh0_rank + 2 * mesh2_rank
                        expected_shard_size = 2
                    else:
                        assert mesh2_rank == 2
                        expected_shard_offset = 10 * mesh0_rank + 6
                        expected_shard_size = 1
                else:
                    assert mesh1_rank == 1
                    if mesh2_rank == 0:
                        expected_shard_offset = 10 * mesh0_rank + 3
                        expected_shard_size = 2
                    elif mesh2_rank == 1:
                        expected_shard_offset = 10 * mesh0_rank + 7
                        expected_shard_size = 2
                    else:
                        expected_shard_offset = global_shape[0]
                        expected_shard_size = 0
            self.assertEqual(local_shape, (expected_shard_size, 2))
            self.assertEqual(global_offset, (expected_shard_offset, 0))


class UtilTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    def _compute_start_end_offsets(self, global_offset, local_size, n_dim):
        offset = []
        for i in range(n_dim):
            offset.append(((global_offset[i]), (global_offset[i] + local_size[i])))
        return offset

    @with_comms
    def test_compute_global_tensor_shape_1D(self):
        one_d_placements = [[Shard(1)], [Shard(0)], [Replicate()]]
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        for placements in one_d_placements:
            if isinstance(placements[0], Shard):
                uneven_dim = list(range(self.world_size))

                @maybe_run_for_local_tensor
                def get_local_shape(rank):
                    local_shape = (
                        torch.Size([5, uneven_dim[rank]])
                        if placements[0].dim == 1
                        else torch.Size([uneven_dim[rank], 5])
                    )
                    return local_shape

                local_shape = get_local_shape(self.rank)
                expected_global_shape = (
                    torch.Size([5, sum(uneven_dim)])
                    if placements[0].dim == 1
                    else torch.Size([sum(uneven_dim), 5])
                )
            else:
                expected_global_shape = torch.Size([5, 5])
                local_shape = torch.Size([5, 5])

            global_shape = compute_global_tensor_shape(
                local_shape, device_mesh, placements
            )
            self.assertEqual(global_shape, expected_global_shape)

    @with_comms
    def test_compute_global_tensor_shape_1D_invalid_shape(self):
        one_d_placement = [Shard(1)]
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        uneven_dim = list(range(self.world_size))

        @maybe_run_for_local_tensor
        def get_local_shape(rank):
            local_shape = (
                torch.Size([5, uneven_dim[rank]])
                if rank % 2 == 0
                else torch.Size([6, uneven_dim[rank]])
            )
            return local_shape

        local_shape = get_local_shape(self.rank)

        with self.assertRaisesRegex(
            RuntimeError,
            "Non-sharded dimensions should have identical size across ranks.",
        ):
            _ = compute_global_tensor_shape(
                local_shape,
                device_mesh,
                one_d_placement,
            )

    @with_comms
    def test_compute_global_tensor_shape_failure_2D(self):
        placement_2D = [Shard(0), Shard(1)]
        device_mesh_2D = init_device_mesh(self.device_type, (2, 2))
        with self.assertRaisesRegex(
            NotImplementedError,
            "compute_global_tensor_shape only supports 1 placement for now.",
        ):
            _ = compute_global_tensor_shape(
                torch.Size([2, 2]),
                device_mesh_2D,
                placement_2D,
            )
        placement_1D = [Shard(0)]
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected one placement per mesh dim",
        ):
            _ = compute_global_tensor_shape(
                torch.Size([2, 2]),
                device_mesh_2D,
                placement_1D,
            )

    @with_comms
    def test_compute_local_shape_and_global_offset_1D(self):
        one_d_placements = [[Shard(0)], [Replicate()]]

        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        for placements in one_d_placements:
            # When the placements is [Shard(0)], we test for three different scenarios:
            # 1) sharding resulting in empty shards on all or some of the ranks
            # 2) sharding resulting in shards of different size across different ranks
            # 3) sharding resulting in non-empty shards of same size across all ranks
            for size in range(self.world_size * 2 + 1):
                global_tensor = torch.arange(size)
                global_shape = global_tensor.size()

                dtensor = distribute_tensor(global_tensor, device_mesh, placements)
                local_size, global_offset = compute_local_shape_and_global_offset(
                    global_shape, device_mesh, placements
                )
                dim = self._compute_start_end_offsets(global_offset, local_size, 1)
                dim0_start, dim0_end = dim[0][0], dim[0][1]

                # Check the local tensor of dtensor is exactly the same
                # if we slice the global_tensor with local_size and global_offset
                self.assertEqual(
                    dtensor.to_local(),
                    global_tensor[dim0_start:dim0_end],
                )

    @with_comms
    def test_compute_local_shape_and_global_offset_2D(self):
        two_d_placements_options = [Shard(0), Shard(1), Replicate()]
        # Generating 6 two-d placements combinations
        two_d_placements = list(
            itertools.combinations_with_replacement(two_d_placements_options, 2)
        )

        # mesh: 2 * 4
        device_mesh = init_device_mesh(self.device_type, (2, 4))
        for placements in two_d_placements:
            for dim_0_size in range(1, 9):
                nelem = 64 // dim_0_size * dim_0_size
                global_tensor = torch.arange(nelem).view(dim_0_size, -1)
                global_shape = global_tensor.size()

                dtensor = distribute_tensor(global_tensor, device_mesh, placements)
                local_size, global_offset = compute_local_shape_and_global_offset(
                    global_shape, device_mesh, placements
                )

                dim = self._compute_start_end_offsets(global_offset, local_size, 2)
                dim0_start, dim0_end = dim[0][0], dim[0][1]
                dim1_start, dim1_end = dim[1][0], dim[1][1]

                @maybe_run_for_local_tensor
                def maybe_compute_rankwise(
                    dim0_start,
                    dim0_end,
                    dim1_start,
                    dim1_end,
                    local_tensor,
                    global_tensor,
                ):
                    # Check the local tensor of dtensor is exactly the same
                    # if we slice the global_tensor with local_size and global_offset
                    self.assertEqual(
                        local_tensor,
                        global_tensor[dim0_start:dim0_end, dim1_start:dim1_end],
                    )

                maybe_compute_rankwise(
                    dim0_start,
                    dim0_end,
                    dim1_start,
                    dim1_end,
                    dtensor.to_local(),
                    global_tensor,
                )

    @with_comms
    def test_compute_local_shape_and_global_offset_3D(self):
        global_tensor_shape = torch.Size([2 * self.world_size, 2 * self.world_size])
        mesh_size_0 = 2
        mesh_size_1 = 2
        mesh_size_2 = self.world_size // (mesh_size_0 * mesh_size_1)
        global_mesh = init_device_mesh(
            self.device_type,
            (mesh_size_0, mesh_size_1, mesh_size_2),
            mesh_dim_names=("mesh-0", "mesh-1", "mesh-2"),
        )
        placements = [
            _StridedShard(0, split_factor=mesh_size_1),
            Shard(0),
            Shard(0),
        ]
        local_shape, global_offset = compute_local_shape_and_global_offset(
            global_tensor_shape, global_mesh, placements
        )
        mesh0_rank, mesh1_rank, mesh2_rank = global_mesh.get_coordinate()
        self.assertEqual(local_shape, [2, 2 * self.world_size])
        self.assertEqual(
            global_offset, (4 * mesh0_rank + 8 * mesh1_rank + 2 * mesh2_rank, 0)
        )

    @with_comms
    def test_compute_local_shape_and_global_offset_4D(self):
        global_tensor_shape = torch.Size([2 * self.world_size, 2 * self.world_size])
        mesh_size_0 = 1
        mesh_size_1 = 2
        mesh_size_2 = 2
        mesh_size_3 = self.world_size // (mesh_size_0 * mesh_size_1 * mesh_size_2)
        global_mesh = init_device_mesh(
            self.device_type,
            (mesh_size_0, mesh_size_1, mesh_size_2, mesh_size_3),
            mesh_dim_names=("mesh-0", "mesh-1", "mesh-2", "mesh-3"),
        )
        placements = [
            _StridedShard(0, split_factor=mesh_size_1),
            _StridedShard(1, split_factor=mesh_size_3),
            Shard(0),
            Shard(1),
        ]
        local_shape, global_offset = compute_local_shape_and_global_offset(
            global_tensor_shape, global_mesh, placements
        )
        mesh0_rank, mesh1_rank, mesh2_rank, mesh3_rank = global_mesh.get_coordinate()
        self.assertEqual(
            local_shape, (2 * mesh_size_1 * mesh_size_3, 2 * mesh_size_0 * mesh_size_2)
        )
        self.assertEqual(
            global_offset,
            (8 * mesh2_rank + 4 * mesh0_rank, 8 * mesh3_rank + 4 * mesh1_rank),
        )
        placements = [
            _StridedShard(0, split_factor=mesh_size_1),
            _StridedShard(1, split_factor=mesh_size_3),
            Shard(0),
            Shard(0),
        ]
        local_shape, global_offset = compute_local_shape_and_global_offset(
            global_tensor_shape, global_mesh, placements
        )
        mesh0_rank, mesh1_rank, mesh2_rank, mesh3_rank = global_mesh.get_coordinate()
        self.assertEqual(
            local_shape, (2 * mesh_size_1, 2 * mesh_size_2 * mesh_size_3 * mesh_size_0)
        )
        self.assertEqual(
            global_offset,
            (8 * mesh2_rank + 0 * mesh0_rank + 4 * mesh3_rank, 4 * mesh1_rank),
        )

    @with_comms
    def test_fsdp_tp_meta_compute(self):
        # FSDP + TP sharding
        tp_size = 2
        dp_size = self.world_size // tp_size
        global_mesh = init_device_mesh(
            self.device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp")
        )
        # local shard shape is [2, 2]
        global_tensor_shape = torch.Size([2 * self.world_size, 2])
        placements = [_StridedShard(0, split_factor=tp_size), Shard(0)]

        local_shape, global_offset = compute_local_shape_and_global_offset(
            global_tensor_shape, global_mesh, placements
        )
        assert global_mesh.get_coordinate is not None
        dp_rank = global_mesh.get_local_rank("dp")
        tp_rank = global_mesh.get_local_rank("tp")
        shard_idx_on_dim_0 = tp_rank * dp_size + dp_rank
        expected_local_shape = (2, 2)
        expected_global_offset = (shard_idx_on_dim_0 * 2, 0)
        self.assertEqual(local_shape, expected_local_shape)
        self.assertEqual(global_offset, expected_global_offset)

    @with_comms
    def test_uneven_fsdp_tp_meta_compute(self):
        # FSDP + TP uneven sharding
        tp_size = 2
        dp_size = self.world_size // tp_size
        global_mesh = init_device_mesh(
            self.device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp")
        )
        global_tensor_shape = torch.Size([15, 5])
        placements = [_StridedShard(0, split_factor=tp_size), Shard(0)]
        local_shape, global_offset = compute_local_shape_and_global_offset(
            global_tensor_shape, global_mesh, placements
        )
        rank = global_mesh.get_rank()
        expected_shapes = [2, 2, 2, 2, 2, 2, 2, 1]
        expected_offsets = [0, 8, 2, 10, 4, 12, 6, 14]

        @maybe_run_for_local_tensor
        def maybe_compute_rankwise(rank, local_shape, global_offset):
            self.assertEqual(local_shape[0], expected_shapes[rank])
            self.assertEqual(global_offset[0], expected_offsets[rank])

        maybe_compute_rankwise(rank, local_shape, global_offset)

    @with_comms
    def test_hsdp_tp_meta_compute(self):
        # HSDP + TP sharding
        tp_size = 2
        dp_shard_size = 2
        dp_replic_size = self.world_size // (dp_shard_size * tp_size)
        global_mesh = init_device_mesh(
            self.device_type,
            (dp_replic_size, dp_shard_size, tp_size),
            mesh_dim_names=("dp_replic", "dp_shard", "tp"),
        )
        # local shard shape is [2, 2]
        global_tensor_shape = torch.Size([2 * dp_shard_size * tp_size, 2])
        placements = [Replicate(), _StridedShard(0, split_factor=tp_size), Shard(0)]

        local_shape, global_offset = compute_local_shape_and_global_offset(
            global_tensor_shape, global_mesh, placements
        )
        assert global_mesh.get_coordinate is not None
        dp_shard_rank = global_mesh.get_local_rank("dp_shard")
        tp_rank = global_mesh.get_local_rank("tp")
        shard_idx_on_dim_0 = tp_rank * dp_shard_size + dp_shard_rank
        expected_local_shape = (2, 2)
        expected_global_offset = (shard_idx_on_dim_0 * 2, 0)
        self.assertEqual(local_shape, expected_local_shape)
        self.assertEqual(global_offset, expected_global_offset)


class UtilSingleDeviceTest(TestCase):
    def test_compute_global_tensor_info_unsupported_placement(self):
        class MockDeviceMesh:
            def size(self, x):
                return x

        class FakePlacement(Placement):
            pass

        device_mesh: Any = MockDeviceMesh()
        local_tensor = torch.tensor([1])
        with self.assertRaises(RuntimeError):
            compute_global_tensor_info(local_tensor, device_mesh, [FakePlacement()])

    def test_compute_global_tensor_info_non_shard_placements(self):
        class MockDeviceMesh:
            def size(self, x):
                return x

        device_mesh: Any = MockDeviceMesh()
        local_tensor = torch.tensor([[1], [2]])
        global_size, global_stride = compute_global_tensor_info(
            local_tensor, device_mesh, [Replicate(), Partial()]
        )
        self.assertEqual(global_size, local_tensor.size())
        self.assertEqual(global_stride, local_tensor.stride())

    def test_compute_global_tensor_info_shard_placement(self):
        class MockDeviceMesh:
            def size(self, dim):
                return dim + 2

        device_mesh: Any = MockDeviceMesh()
        local_tensor = torch.tensor([[[1], [2], [3]], [[4], [5], [6]]])
        global_size, global_stride = compute_global_tensor_info(
            local_tensor, device_mesh, [Shard(0), Shard(1), Shard(2)]
        )
        self.assertEqual(
            global_size, [(i + 2) * x for (i, x) in enumerate(local_tensor.size())]
        )
        self.assertEqual(global_stride[0], local_tensor.stride()[0] * 3 * 4)
        self.assertEqual(global_stride[1], local_tensor.stride()[1])
        self.assertEqual(global_stride[2], local_tensor.stride()[2] * 3)

    def test_compute_tensor_info(self):
        from torch.testing._internal.distributed.fake_pg import FakeStore

        world_size = 256
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=world_size
        )
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cpu",
            (8, 8, 4),
            mesh_dim_names=(
                "dp",
                "tp",
                "cp",
            ),
        )
        assert world_size == mesh.shape[0] * mesh.shape[1] * mesh.shape[2]

        # Add Partial() when we are allowed to redistribute to it
        options = [Shard(0), Shard(1), Shard(2), Replicate()]
        all_placements = [tuple(p) for p in itertools.product(options, repeat=3)]
        for placements in all_placements:
            local_tensor = torch.empty_strided(
                (4, 4, 4),
                (16, 4, 1),
            )
            local_dt = DTensor.from_local(local_tensor, mesh, placements)

            global_shape, global_stride = compute_global_tensor_info(
                local_tensor, mesh, placements
            )
            global_dt = local_dt.redistribute(mesh, [Replicate()] * mesh.ndim)
            self.assertEqual(global_shape, global_dt.size())
            self.assertEqual(global_stride, global_dt.stride())

            global_tensor = torch.empty_strided(
                global_shape,
                global_stride,
            )
            new_local_shape, new_local_stride = compute_local_tensor_info(
                global_tensor,
                mesh,
                placements,
            )
            self.assertEqual(new_local_shape, local_tensor.size())
            self.assertEqual(new_local_stride, local_tensor.stride())

            new_local_dt = global_dt.redistribute(mesh, placements)
            self.assertEqual(new_local_shape, new_local_dt.to_local().size())
            self.assertEqual(new_local_stride, new_local_dt.to_local().stride())

        torch.distributed.destroy_process_group()


class TestStridedSharding(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_1d_mesh_strided_sharding(self):
        mesh_1d = init_device_mesh(self.device_type, (self.world_size,))
        # Test 1: 1-d tensor over 1-d mesh
        x = torch.arange(2 * self.world_size, device=self.device_type)
        """
        contiguous sharding: [0, 1 | 2, 3 | 4, 5 | 6, 7]
        """
        shard_placement = _StridedShard(0, split_factor=1)  # same as Shard(0)
        tensor_list, _ = shard_placement._split_tensor(x, self.world_size)

        @maybe_run_for_local_tensor
        def maybe_compute_rankwise(rank, tensor_list, x):
            shard_x = tensor_list[rank]
            self.assertEqual(shard_x, x.view(self.world_size, -1)[rank])
            return shard_x

        shard_x = maybe_compute_rankwise(self.rank, tensor_list, x)

        # shard_to_replicate
        full_tensor = shard_placement._to_replicate_tensor(
            shard_x,
            mesh_1d,
            mesh_dim=0,
            current_logical_shape=list(x.shape),
        )
        self.assertEqual(full_tensor, x)

        """
        strided sharding: [0, 4 | 1, 5 | 2, 6 | 3, 7]
        """
        shard_placement = _StridedShard(0, split_factor=2)
        tensor_list, _ = shard_placement._split_tensor(x, self.world_size)

        @maybe_run_for_local_tensor
        def maybe_compute_rankwise(rank, tensor_list, x):
            shard_x = tensor_list[rank]
            self.assertEqual(shard_x, x.view(-1, self.world_size).swapdims(-1, 0)[rank])
            return shard_x

        shard_x = maybe_compute_rankwise(self.rank, tensor_list, x)

        # shard_to_replicate
        full_tensor = shard_placement._to_replicate_tensor(
            shard_x,
            mesh_1d,
            mesh_dim=0,
            current_logical_shape=list(x.shape),
        )
        self.assertEqual(full_tensor, x)

    @with_comms
    def test_2d_mesh_strided_sharding(self):
        # Test 2: 1-d tensor over 2-d mesh
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dim0", "dim1")
        )
        mesh_dim0_size = mesh_2d["dim0"].size()
        mesh_dim1_size = mesh_2d["dim1"].size()
        mesh_dim0_local_rank = mesh_2d["dim0"].get_local_rank(mesh_dim=0)
        mesh_dim1_local_rank = mesh_2d["dim1"].get_local_rank(mesh_dim=0)
        x = torch.arange(2 * self.world_size, device=self.device_type)
        """
        contiguous sharding: [
            [ 0, 1 | 2, 3 ],
            [ 4, 5 | 6, 7 ],
        ]
        """
        # shard on mesh dim-0
        shard_placement_dim0 = _StridedShard(0, split_factor=1)  # same as Shard(0)
        tensor_list, _ = shard_placement_dim0._split_tensor(x, mesh_dim0_size)
        shard_placement_dim1 = _StridedShard(0, split_factor=1)  # same as Shard(0)

        @maybe_run_for_local_tensor
        def maybe_compute_rankwise_strided(mesh_dim0_local_rank):
            expected_shard_dim0 = x.view(mesh_dim0_size, -1)[mesh_dim0_local_rank]
            shard_x = tensor_list[mesh_dim0_local_rank]
            self.assertEqual(shard_x, expected_shard_dim0)
            return shard_x, expected_shard_dim0

        shard_x, expected_shard_dim0 = maybe_compute_rankwise_strided(
            mesh_dim0_local_rank
        )
        tensor_list, _ = shard_placement_dim1._split_tensor(shard_x, mesh_dim1_size)

        @maybe_run_for_local_tensor
        def maybe_compute_rankwise_strided(mesh_dim1_local_rank):
            expected_shard_dim1 = shard_x.view(mesh_dim1_size, -1)[mesh_dim1_local_rank]
            shard_x2 = tensor_list[mesh_dim1_local_rank]
            self.assertEqual(shard_x2, expected_shard_dim1)

            return shard_x2, expected_shard_dim0

        shard_x, expected_shard_dim0 = maybe_compute_rankwise_strided(
            mesh_dim1_local_rank
        )

        # shard_to_replicate on mesh dim-1
        full_tensor = shard_placement_dim1._to_replicate_tensor(
            shard_x,
            mesh_2d,
            mesh_dim=1,
            current_logical_shape=list(expected_shard_dim0.shape),
        )
        self.assertEqual(full_tensor, expected_shard_dim0)

        # shard_to_replicate on mesh dim-0
        full_tensor = shard_placement_dim0._to_replicate_tensor(
            full_tensor.reconcile() if self.is_local_tensor_enabled else full_tensor,
            mesh_2d,
            mesh_dim=0,
            current_logical_shape=list(x.shape),
        )

        self.assertEqual(full_tensor, x)

        """
        strided sharding: [
            [ 0, 1 | 4, 5 ],
            [ 2, 3 | 6, 7 ],
        ]
        """
        split_factor = 2
        # shard on mesh dim-0
        shard_placement_dim0 = _StridedShard(0, split_factor=split_factor)
        tensor_list, _ = shard_placement_dim0._split_tensor(x, mesh_dim0_size)
        shard_placement_dim1 = _StridedShard(0, split_factor=1)  # same as Shard(0)

        @maybe_run_for_local_tensor
        def maybe_compute_rankwise_strided(mesh_dim0_local_rank):
            shard_x = tensor_list[mesh_dim0_local_rank]
            expected_shard_dim0 = (
                torch.tensor([0, 1, 4, 5], device=self.device_type)
                if mesh_dim0_local_rank == 0
                else torch.tensor([2, 3, 6, 7], device=self.device_type)
            )
            self.assertEqual(shard_x, expected_shard_dim0)
            return shard_x, expected_shard_dim0

        shard_x, expected_shard_dim0 = maybe_compute_rankwise_strided(
            mesh_dim0_local_rank
        )
        tensor_list, _ = shard_placement_dim1._split_tensor(shard_x, mesh_dim1_size)

        @maybe_run_for_local_tensor
        def maybe_compute_rankwise_strided(mesh_dim1_local_rank):
            shard_x2 = tensor_list[mesh_dim1_local_rank]
            expected_shard_dim1 = expected_shard_dim0.view(mesh_dim1_size, -1)[
                mesh_dim1_local_rank
            ]
            self.assertEqual(shard_x2, expected_shard_dim1)
            return shard_x2, expected_shard_dim0

        shard_x, expected_shard_dim0 = maybe_compute_rankwise_strided(
            mesh_dim1_local_rank
        )

        # shard_to_replicate on mesh dim-1
        full_tensor = shard_placement_dim1._to_replicate_tensor(
            shard_x,
            mesh_2d,
            mesh_dim=1,
            current_logical_shape=list(expected_shard_dim0.shape),
        )
        self.assertEqual(full_tensor, expected_shard_dim0)

        # shard_to_replicate on mesh dim-0
        full_tensor = shard_placement_dim0._to_replicate_tensor(
            full_tensor.reconcile() if self.is_local_tensor_enabled else full_tensor,
            mesh_2d,
            mesh_dim=0,
            current_logical_shape=list(x.shape),
        )
        self.assertEqual(full_tensor, x)

    @with_comms
    def test_2d_mesh_2d_tensor_strided_sharding(self):
        # Test 2: 1-d tensor over 2-d mesh
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dim0", "dim1")
        )
        mesh_dim0_size = mesh_2d["dim0"].size()
        mesh_dim1_size = mesh_2d["dim1"].size()
        mesh_dim0_local_rank = mesh_2d["dim0"].get_local_rank(mesh_dim=0)
        mesh_dim1_local_rank = mesh_2d["dim1"].get_local_rank(mesh_dim=0)
        x = torch.arange(2 * self.world_size, device=self.device_type).reshape(2, -1)
        """
        strided sharding:
            rank 0: [[0], [4]]
            rank 1: [[2], [6]]
            rank 2: [[1], [5]]
            rank 3: [[3], [7]]
        """
        split_factor = 2
        # shard on mesh dim-0
        shard_placement_dim0 = _StridedShard(1, split_factor=split_factor)
        tensor_list, _ = shard_placement_dim0._split_tensor(x, mesh_dim0_size)

        @maybe_run_for_local_tensor
        def maybe_compute_rankwise_strided(mesh_dim0_local_rank, tensor_list):
            shard_x2 = tensor_list[mesh_dim0_local_rank]
            expected_shard_dim0 = (
                torch.tensor([[0, 2], [4, 6]], device=self.device_type)
                if mesh_dim0_local_rank == 0
                else torch.tensor([[1, 3], [5, 7]], device=self.device_type)
            )
            self.assertEqual(shard_x2, expected_shard_dim0)
            return shard_x2, expected_shard_dim0

        shard_x, expected_shard_dim0 = maybe_compute_rankwise_strided(
            mesh_dim0_local_rank, tensor_list
        )

        # shard on mesh dim-1
        shard_placement_dim1 = _StridedShard(1, split_factor=1)  # same as Shard(1)
        tensor_list, _ = shard_placement_dim1._split_tensor(shard_x, mesh_dim1_size)

        @maybe_run_for_local_tensor
        def maybe_compute_rankwise_strided(mesh_dim1_local_rank, rank, tensor_list):
            shard_x = tensor_list[mesh_dim1_local_rank]
            expected_shard_dim1 = [
                torch.tensor(value, device=self.device_type)
                for value in [[[0], [4]], [[2], [6]], [[1], [5]], [[3], [7]]]
            ][rank]
            self.assertEqual(shard_x, expected_shard_dim1)

            return shard_x, expected_shard_dim0

        shard_x, expected_shard_dim0 = maybe_compute_rankwise_strided(
            mesh_dim1_local_rank, self.rank, tensor_list
        )

        # shard_to_replicate on mesh dim-1
        full_tensor = shard_placement_dim1._to_replicate_tensor(
            shard_x,
            mesh_2d,
            mesh_dim=1,
            current_logical_shape=list(expected_shard_dim0.shape),
        )

        self.assertEqual(
            full_tensor,
            expected_shard_dim0.reconcile()
            if self.is_local_tensor_enabled
            else expected_shard_dim0,
        )

        # shard_to_replicate on mesh dim-0
        full_tensor = shard_placement_dim0._to_replicate_tensor(
            full_tensor,
            mesh_2d,
            mesh_dim=0,
            current_logical_shape=list(x.shape),
        )
        self.assertEqual(full_tensor, x)

    @with_comms
    def test_2d_mesh_uneven_strided_shard(self):
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size // 2, 2),
            mesh_dim_names=("fsdp", "tp"),
        )

        for size in (2, 3, 5, 11):
            tensor = torch.arange(size, device=self.device_type).view(1, -1)
            dtensor = distribute_tensor(
                tensor,
                device_mesh=mesh,
                placements=(Replicate(), Replicate()),
            ).redistribute(
                mesh, placements=(_StridedShard(dim=1, split_factor=2), Shard(1))
            )
            self.assertEqual(dtensor.full_tensor(), tensor)


class Test_StridedShard_Propagation(LocalDTensorTestBase):
    @property
    def world_size(self) -> int:
        return 16

    @with_comms
    def test_einsum_propagation(self):
        with LocalTensorMode(ranks=self.world_size):
            mesh = init_device_mesh("cpu", (4, 4))
            input_tensor = torch.arange(16 * 16).float().view(16, 16)
            A = distribute_tensor(input_tensor, mesh, [Shard(1), Shard(1)])
            B1 = distribute_tensor(input_tensor, mesh, [Shard(0), Shard(0)])
            B2 = distribute_tensor(
                input_tensor,
                mesh,
                [_StridedShard(0, split_factor=mesh.size(1)), Shard(0)],
            )
            with CommDebugMode() as comm_mode:
                # res1 will be (Partial, Partial), no redistribution needed
                res1 = A @ B1
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 0
            )

            with CommDebugMode() as comm_mode:
                # `A @ B2` will trigger redistribution on both inputs as below:
                # A: S(1)[0]S(1)[1]
                # B2: _S(0, 4)S(0)[0]->RS(0)->RR->S(0)R->S(0)[0]S(0)[1]
                # The final output res2's placements will be PP.
                res2 = A @ B2
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 2
            )
            assert isinstance(res1, DTensor)
            assert isinstance(res2, DTensor)
            self.assertEqual(res1.full_tensor(), res2.full_tensor())

    @with_comms
    def test_pointwise_propagation(self):
        with LocalTensorMode(ranks=self.world_size):
            mesh = init_device_mesh("cpu", (2, 2, 2, 2))
            input_tensor = torch.arange(32).float().view(2, 16)
            A = distribute_tensor(
                input_tensor,
                mesh,
                [Shard(1), _StridedShard(1, split_factor=2), Shard(1), Shard(0)],
            )
            with CommDebugMode() as comm_mode:
                res1 = torch.sum(A, dim=0)
            self.assertEqual(comm_mode.get_total_counts(), 0)
            assert isinstance(res1, DTensor)
            self.assertEqual(res1.full_tensor(), torch.sum(input_tensor, dim=0))
            with CommDebugMode() as comm_mode:
                res2 = torch.sum(A, dim=1)
            self.assertEqual(comm_mode.get_total_counts(), 0)
            assert isinstance(res2, DTensor)
            self.assertEqual(res2.full_tensor(), torch.sum(input_tensor, dim=1))

    @with_comms
    def test_inplace_op_with_strided_shard(self):
        """Test that inplace ops work correctly with strided shard placements.

        This verifies that the inplace_op flag is correctly passed during strategy
        expansion, ensuring that incompatible Partial strategies are filtered out
        for inplace ops like mul_.
        """
        with LocalTensorMode(ranks=self.world_size):
            mesh = init_device_mesh("cpu", (4, 4))
            input_tensor = torch.arange(32).float().view(2, 16)
            # Create a strided shard DTensor (similar to FSDP+TP pattern)
            A = distribute_tensor(
                input_tensor,
                mesh,
                [_StridedShard(1, split_factor=mesh.size(1)), Shard(1)],
            )
            original_placements = A.placements

            # Test inplace mul_ with scalar - this should preserve the strided shard placement
            with CommDebugMode() as comm_mode:
                A.mul_(0.9)
            # Inplace op should not require redistribution
            self.assertEqual(comm_mode.get_total_counts(), 0)
            # Placements should be preserved
            self.assertEqual(A.placements, original_placements)
            # Verify correctness
            expected = input_tensor * 0.9
            self.assertEqual(A.full_tensor(), expected)

            # Test inplace add_ with scalar
            B = distribute_tensor(
                input_tensor,
                mesh,
                [_StridedShard(1, split_factor=mesh.size(1)), Shard(1)],
            )
            with CommDebugMode() as comm_mode:
                B.add_(1.0)
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(B.placements, original_placements)
            self.assertEqual(B.full_tensor(), input_tensor + 1.0)

    def run_view_propagation(
        self,
        mesh,
        original_full_tensor,
        original_placements,
        view_into_shape: list[int],
        expected_placements_after_view,
    ):
        assert math.prod(original_full_tensor.shape) == math.prod(view_into_shape)
        # verify user specified `expected_placements_after_view `is correct
        A = distribute_tensor(original_full_tensor, mesh, original_placements)
        B = original_full_tensor.view(*view_into_shape)
        C = distribute_tensor(B, mesh, expected_placements_after_view)
        self.assertEqual(
            A.to_local().view(-1),
            C.to_local().view(-1),
            msg="Defined `expected_placements_after_view` is incorrect",
        )
        # verify the propagated sharding spec after view is correct
        viewed_A = A.view(*view_into_shape)
        assert isinstance(viewed_A, DTensor)
        assert viewed_A.placements == tuple(expected_placements_after_view)
        self.assertEqual(A.to_local().view(-1), viewed_A.to_local().view(-1))

    @with_comms
    def test_view_propagation(self):
        with LocalTensorMode(ranks=12):
            mesh = init_device_mesh("cpu", (3, 2, 2))

            input_tensor = torch.randn(24, 12, 24)
            self.run_view_propagation(
                mesh,
                input_tensor,
                [SS(0, 2), SS(1, 2), S(0)],
                [12, 2, 12, 24],
                [SS(0, 2), SS(2, 2), S(0)],
            )
            self.run_view_propagation(
                mesh, input_tensor, [SS(0, 2), R, R], [12, 2, 12, 24], [SS(0, 2), R, R]
            )

            mesh = init_device_mesh("cpu", (3, 4))
            input_tensor = torch.randn(48, 35, 26)
            self.run_view_propagation(
                mesh, input_tensor, [SS(0, 2), S(0)], [24, 4, 35, 13], [SS(0, 2), S(0)]
            )
            self.run_view_propagation(
                mesh, input_tensor, [S(0), SS(0, 2)], [24, 4, 35, 13], [S(0), SS(0, 2)]
            )

            input_tensor = torch.randn(2, 48, 2)
            self.run_view_propagation(
                mesh, input_tensor, [SS(1, 2), S(1)], [2, 12, 4, 2], [SS(1, 2), S(1)]
            )
            self.run_view_propagation(
                mesh, input_tensor, [SS(1, 2), R], [2, 12, 4, 2], [SS(1, 2), R]
            )

    @expectedFailure
    @with_comms
    def test_view_propagation_not_supported_yet(self):
        # TODO: need to extend the StridedShard support for view propagation
        with LocalTensorMode(ranks=16):
            mesh = init_device_mesh("cpu", (2, 2, 2, 2))
            input_tensor = torch.randn(4, 4, 8)
            # case: StridedShard dim shift
            self.run_view_propagation(
                mesh, input_tensor, [SS(2, 2), R, R, R], [4, 4, 2, 4], [S(3), R, R, R]
            )
            # case: StridedShard split factor updated
            self.run_view_propagation(
                mesh,
                input_tensor,
                [SS(2, 4), S(2), R, R],
                [4, 4, 2, 4],
                [SS(3, 2), S(2), R, R],
            )


class Test_StridedShard_Optimizer(DTensorTestBase):
    """Test optimizer updates with _StridedShard placement using FSDP+TP.

    This test uses FSDP+TP to create parameters with placement
    (_StridedShard(0, split_factor=tp_size), Shard(0)) and verifies
    that various optimizers can correctly update these parameters.

    The pattern follows _TestClipGradNormBase from test_fully_shard_clip_grad_norm_.py
    """

    @property
    def world_size(self) -> int:
        return 4

    def _test_optimizer_with_fsdp_tp(
        self,
        optimizer_cls: type,
        optimizer_kwargs: dict | None = None,
    ):
        """Test an optimizer with FSDP+TP parallelized model.

        Creates a 2D mesh (dp_size x tp_size), applies TP then FSDP,
        verifies parameters have _StridedShard placement, and runs optimizer steps.
        """
        from copy import deepcopy

        import torch.nn as nn
        from torch.distributed._composable import replicate
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            parallelize_module,
            RowwiseParallel,
        )

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        dp_size = 2
        tp_size = self.world_size // dp_size
        global_mesh = init_device_mesh(
            self.device_type,
            (dp_size, tp_size),
            mesh_dim_names=("dp", "tp"),
        )
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]

        # Create simple model with linear layers
        torch.manual_seed(42)

        class SimpleMLP(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.in_proj = nn.Linear(dim, dim * 4, bias=False)
                self.out_proj = nn.Linear(dim * 4, dim, bias=False)

            def forward(self, x):
                return self.out_proj(torch.relu(self.in_proj(x)))

        dim = 16
        model = SimpleMLP(dim)
        ref_model = replicate(
            deepcopy(model).to(self.device_type),
            process_group=dp_mesh.get_group(),
        )
        ref_optim = optimizer_cls(ref_model.parameters(), lr=1e-2, **optimizer_kwargs)

        # Apply TP first (ColwiseParallel on in_proj, RowwiseParallel on out_proj)
        parallelize_module(
            model,
            tp_mesh,
            {
                "in_proj": ColwiseParallel(),
                "out_proj": RowwiseParallel(),
            },
        )
        # Apply FSDP on top - this creates _StridedShard placement
        fully_shard(model, mesh=dp_mesh)
        optim = optimizer_cls(model.parameters(), lr=1e-2, **optimizer_kwargs)

        # Verify parameters have correct placement with _StridedShard
        # pyrefly: ignore [bad-assignment]
        for name, param in model.named_parameters():
            self.assertIsInstance(param, DTensor)
            # FSDP+TP creates _StridedShard for the FSDP dimension
            # The placement should be (_StridedShard(dim, split_factor=tp_size), <TP placement>)
            self.assertEqual(len(param.placements), 2)
            fsdp_placement = param.placements[0]
            # Verify FSDP creates _StridedShard when combined with TP
            self.assertIsInstance(
                fsdp_placement,
                (_StridedShard, Shard),
                f"Parameter {name} has unexpected FSDP placement: {fsdp_placement}",
            )

        # Run training loop
        torch.manual_seed(42 + dp_mesh.get_local_rank() + 1)
        inp = torch.randn((4, dim), device=self.device_type)

        for iter_idx in range(5):
            ref_optim.zero_grad()
            ref_model(inp).sum().backward()

            optim.zero_grad()
            model(inp).sum().backward()

            # Verify gradients match
            for ref_param, param in zip(ref_model.parameters(), model.parameters()):
                self.assertEqual(ref_param.grad, param.grad.full_tensor())

            ref_optim.step()
            optim.step()

            # Verify parameters still have correct placement after optimizer step
            for name, param in model.named_parameters():
                self.assertIsInstance(param, DTensor)
                self.assertEqual(len(param.placements), 2)

            # Verify parameter values match after update
            for ref_param, param in zip(ref_model.parameters(), model.parameters()):
                self.assertEqual(
                    ref_param,
                    param.full_tensor(),
                    msg=f"Parameter mismatch at iteration {iter_idx}",
                )

    @with_comms
    def test_sgd_optimizer(self):
        self._test_optimizer_with_fsdp_tp(torch.optim.SGD)

    @with_comms
    def test_sgd_optimizer_with_momentum(self):
        self._test_optimizer_with_fsdp_tp(
            torch.optim.SGD, optimizer_kwargs={"momentum": 0.9}
        )

    @with_comms
    def test_adam_optimizer(self):
        self._test_optimizer_with_fsdp_tp(torch.optim.Adam)

    @with_comms
    def test_adamw_optimizer(self):
        self._test_optimizer_with_fsdp_tp(torch.optim.AdamW)

    @with_comms
    def test_rmsprop_optimizer(self):
        self._test_optimizer_with_fsdp_tp(torch.optim.RMSprop)

    @with_comms
    def test_adagrad_optimizer(self):
        self._test_optimizer_with_fsdp_tp(torch.optim.Adagrad)

    @with_comms
    def test_adadelta_optimizer(self):
        self._test_optimizer_with_fsdp_tp(torch.optim.Adadelta)


class Test_StridedShard_with_shard_order(LocalDTensorTestBase):
    @property
    def world_size(self) -> int:
        return 32

    @with_comms
    def test_StridedShard_to_shard_order(self):
        with LocalTensorMode(ranks=self.world_size):
            mesh = DeviceMesh("cpu", torch.arange(self.world_size).view(2, 2, 2, 2, 2))
            shard_iter = generate_shard_orders(mesh, 3)
            # It takes ~4.8h to complete total 2520 shard order combinations here
            # using LocalTensor. So we only randomly pick 25 shard orders to test.
            all_shard_order = list(shard_iter)
            import random

            random.seed(42)
            shard_order_choices = random.sample(
                all_shard_order, min(25, len(all_shard_order))
            )

            x = torch.randn(32, 32, 32)
            for shard_order in shard_order_choices:
                a = _distribute_tensor(x, mesh, None, shard_order)

                placement_without_stridedshard = shard_order_to_placement(
                    shard_order, mesh
                )
                placements_with_stridedshard = (
                    DTensorSpec._convert_shard_order_to_StridedShard(
                        shard_order, placement_without_stridedshard, mesh
                    )
                )
                b = distribute_tensor(x, mesh, placements_with_stridedshard)
                shard_order_from_stridedshard = (
                    DTensorSpec._maybe_convert_StridedShard_to_shard_order(
                        placements_with_stridedshard, mesh
                    )
                )
                self.assertEqual(shard_order, shard_order_from_stridedshard)
                self.assertEqual(a.to_local(), b.to_local())

    @with_comms
    def test_StridedShard_not_convertible_to_shard_order(self):
        with LocalTensorMode(ranks=self.world_size):
            mesh = DeviceMesh("cpu", torch.arange(self.world_size).view(4, 8))
            unconvertible_placements_list = [
                [_StridedShard(0, split_factor=2), _StridedShard(1, split_factor=2)],
                [_StridedShard(0, split_factor=2), Shard(1)],
                [_StridedShard(1, split_factor=16), Shard(1)],
            ]
            for placements in unconvertible_placements_list:
                shard_order = DTensorSpec._maybe_convert_StridedShard_to_shard_order(
                    tuple(placements), mesh
                )
                self.assertIsNone(shard_order)


class Test2DStridedLocalShard(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_fsdp1_tp_2d_dtensor_local_shards_and_offsets(self):
        # We are mimicking the behavior of FSDP1 + TP.
        # Currently, the 2D DTensor's local shard is correct, since from_local + redistribute incurs a all_gather behind the scene.
        # When we have a global_tensor of [0, 1, 2, 3, 4, 5, 6, 7], the local shard of 2D DTensor would be:
        # rank0: [0, 1], rank1: [2, 3], rank2: [4, 5], rank3: [6, 7]
        with CommDebugMode() as comm_mode:
            global_tensor = torch.arange(8).view(4, 2)
            mesh_2d = init_device_mesh(
                self.device_type, (2, 2), mesh_dim_names=("DP", "TP")
            )
            tp_mesh = mesh_2d["TP"]
            dtensor_tp = distribute_tensor(
                global_tensor, tp_mesh, placements=[Shard(0)]
            )
            dtensor_2d = DTensor.from_local(
                dtensor_tp.to_local(), mesh_2d, [Replicate(), Shard(0)], run_check=False
            ).redistribute(mesh_2d, [Shard(0), Shard(0)])
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 1
            )

        self.assertEqual(
            dtensor_2d.to_local(), global_tensor[self.rank : self.rank + 1]
        )
        # compute_local_shape_and_global_offset currently does take into consideration of strided sharding,
        # which should after strided sharding is added.
        local_size, global_offset = compute_local_shape_and_global_offset(
            global_tensor.shape, mesh_2d, [Shard(0), Shard(0)]
        )
        self.assertEqual(local_size, torch.Size([1, 2]))
        self.assertEqual(global_offset, torch.Size([self.rank, 0]))

    @with_comms
    def test_fsdp2_tp_2d_dtensor_local_shards_and_offsets(self):
        # We are mimicking the behavior of FSDP2 + TP.
        # Currently, the 2D DTensor's local shard is incorrect for resharding, since we want to avoid extra communication.
        # It's incorrect for resharding, since `compute_local_shape_and_global_offset`
        # doesn't know the correct offsets for resharding.
        # When we have a global_tensor of [0, 1, 2, 3, 4, 5, 6, 7], the local shard of 2D DTensor would be:
        # local tensor -- rank0: [0, 1], rank1: [4, 5], rank2: [2, 3], rank3: [6, 7]
        # current offsets -- rank0: [0, 0], rank1: [1, 0], rank2: [2, 0], rank3: [3, 0]
        # Ideally, with strided sharding, the offsets should be  rank0: [0, 0], rank1: [2, 0], rank2: [1, 0], rank3: [3, 0]
        # TODO: to make the local shard of FSDP2 + TP correct for resharding, it would require strided_sharding
        # as well as let compute_local_shape_and_global_offset takes into consideration of strided_sharding.
        global_tensor = torch.arange(8).view(4, 2)
        with CommDebugMode() as comm_mode:
            mesh_2d = init_device_mesh(
                self.device_type, (2, 2), mesh_dim_names=("DP", "TP")
            )
            tp_mesh = mesh_2d["TP"]
            dtensor_tp = distribute_tensor(
                global_tensor, tp_mesh, placements=[Shard(0)]
            )
            chunks = list(torch.chunk(dtensor_tp.to_local(), 2, dim=0))

            @maybe_run_for_local_tensor
            def get_sharded_param(rank, chunks):
                shard_rank = 0 if rank // 2 == 0 else 1
                return chunks[shard_rank]

            sharded_param = get_sharded_param(self.rank, chunks)
            spec_2d = DTensorSpec(
                mesh=mesh_2d,
                placements=(_StridedShard(0, split_factor=2), Shard(0)),
                tensor_meta=TensorMeta(
                    global_tensor.size(),
                    global_tensor.stride(),
                    global_tensor.dtype,
                ),
            )

            dtensor_2d = DTensor(
                sharded_param,
                spec_2d,
                requires_grad=False,
            )

        self.assertEqual(
            comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 0
        )

        self.assertEqual(global_tensor, dtensor_2d.full_tensor())


class LocalTensorTestBase(TestCase):
    def assertEqual(self, lhs, rhs, **kwargs):
        mode = local_tensor_mode()
        with nullcontext() if mode is None else mode.disable():
            if isinstance(lhs, LocalTensor) and isinstance(rhs, LocalTensor):
                assert isinstance(lhs, LocalTensor) and isinstance(rhs, LocalTensor)
                super().assertEqual(lhs._ranks, rhs._ranks)
                for r in lhs._ranks:
                    super().assertEqual(
                        lhs._local_tensors[r],
                        rhs._local_tensors[r],
                        lambda m: f"rank {r}: {m}",
                    )
            elif isinstance(lhs, LocalTensor) or isinstance(rhs, LocalTensor):
                lhs, rhs = (lhs, rhs) if isinstance(lhs, LocalTensor) else (rhs, lhs)
                for r in lhs._ranks:
                    super().assertEqual(
                        lhs._local_tensors[r], rhs, lambda m: f"rank {r}: {m}"
                    )
            else:
                return super().assertEqual(lhs, rhs, **kwargs)

    @property
    def world_size(self):
        raise NotImplementedError("override world-size in your subclass")

    def build_device_mesh(self) -> DeviceMesh:
        return init_device_mesh("cpu", (self.world_size,))

    def setUp(self):
        super().setUp()
        torch.distributed.init_process_group(
            # TODO: test other ranks too
            "fake",
            rank=0,
            world_size=self.world_size,
        )

    def tearDown(self):
        super().tearDown()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass


class TestStridedShardCollectiveOpUtils:
    from collections import namedtuple

    ShardConfig = namedtuple("ShardConfig", ["mesh_dim", "split_factor"], defaults=(1,))

    def _convert_default_order_placements_to_ShardConfig(
        self, placements: Sequence[Placement]
    ) -> dict[int, list["TestStridedShardCollectiveOpUtils.ShardConfig"]]:
        """
        Convert placements to a shard_map for use with _get_logical_shape.

        Given placements like [Shard(0), _StridedShard(0, split_factor=2), Shard(1)],
        creates a mapping from tensor dimension to list of ShardConfigs:
        {
            0: [ShardConfig(mesh_dim=0, split_factor=1), ShardConfig(mesh_dim=1, split_factor=2)],
            1: [ShardConfig(mesh_dim=2, split_factor=1)]
        }

        Each mesh_dim corresponds to the index of the placement in the placements list.
        """
        shard_map: dict[int, list[TestStridedShardCollectiveOpUtils.ShardConfig]] = {}

        for mesh_dim, placement in enumerate(placements):
            if isinstance(placement, _StridedShard):
                tensor_dim = placement.dim
                split_factor = placement.split_factor
            elif isinstance(placement, Shard):
                tensor_dim = placement.dim
                split_factor = 1
            else:
                continue

            if tensor_dim not in shard_map:
                shard_map[tensor_dim] = []
            shard_map[tensor_dim].append(
                self.ShardConfig(mesh_dim=mesh_dim, split_factor=split_factor)
            )

        return shard_map

    def _get_logical_shape(
        self,
        shard_map: dict[int, list["TestStridedShardCollectiveOpUtils.ShardConfig"]],
        mesh: DeviceMesh,
        operate_mesh_dim: int,
        full_tensor_shape: tuple[int, ...],
    ) -> list[int]:
        """
        Compute the logical shape after applying sharding, excluding `operate_mesh_dim`.

        Args:
            shard_map: Maps tensor dim to list of ShardConfigs describing how it's sharded
            mesh: The device mesh
            operate_mesh_dim: The mesh dimension to exclude from shape computation
            full_tensor_shape: The original full tensor shape

        Returns:
            The logical shape after applying all sharding except on operate_mesh_dim
        """
        new_logical_shape = list(full_tensor_shape)
        coordinate = mesh.get_coordinate()
        assert coordinate is not None

        for tensor_dim, shard_configs in shard_map.items():
            for config in shard_configs:
                if operate_mesh_dim == config.mesh_dim:
                    continue
                if config.split_factor == 1:
                    placement = Shard(tensor_dim)
                else:
                    placement = _StridedShard(
                        tensor_dim, split_factor=config.split_factor
                    )
                new_size, _ = placement._local_shard_size_and_offset(
                    curr_local_size=new_logical_shape[tensor_dim],
                    num_chunks=mesh.size(mesh_dim=config.mesh_dim),
                    rank=coordinate[config.mesh_dim],
                )
                new_logical_shape[tensor_dim] = new_size

        return new_logical_shape


class TestStridedShardReplicate(TestStridedShardCollectiveOpUtils, DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_StridedShard_to_replicate(self):
        mesh = self.build_device_mesh()
        for split_factor in range(2, 17):
            for tensor_size in range(1, 200):
                a = torch.arange(tensor_size)
                src_p = (_StridedShard(0, split_factor=split_factor),)
                a_dt = distribute_tensor(a, mesh, src_p, src_data_rank=None)
                logical_shape = self._get_logical_shape(
                    self._convert_default_order_placements_to_ShardConfig(src_p),
                    mesh,
                    0,
                    a.shape,
                )
                p = _StridedShard(0, split_factor=split_factor)
                a_dt_after_to_replicate = p._to_replicate_tensor(
                    a_dt.to_local(), mesh, 0, logical_shape
                )
                b_dt = distribute_tensor(a, mesh, [Replicate()], src_data_rank=None)
                self.assertEqual(
                    a_dt_after_to_replicate,
                    b_dt.to_local(),
                    f"{tensor_size=}, placements={src_p}",
                )

    @with_comms
    def test_replicate_to_StridedShard(self):
        mesh = self.build_device_mesh()
        coordinate = mesh.get_coordinate()
        for split_factor in range(2, 17):
            for tensor_size in range(1, 200):
                a = torch.arange(tensor_size)
                a_dt_replicate = distribute_tensor(
                    a, mesh, [Replicate()], src_data_rank=None
                )
                p = _StridedShard(0, split_factor=split_factor)
                a_dt_strided = p._replicate_to_strided_shard(
                    a_dt_replicate.to_local(), mesh, 0, coordinate[0]
                )
                b_dt = distribute_tensor(a, mesh, (p,), src_data_rank=None)
                self.assertEqual(
                    a_dt_strided,
                    b_dt.to_local(),
                )


class TestExplicitRedistribute(LocalTensorTestBase):
    @property
    def world_size(self):
        return 4

    def test_message_fn_not_called_in_fastpath(self):
        """Test that message_fn is not called when no ExplicitRedistributionContext is active.

        This ensures that string formatting overhead is avoided in the common case.
        """
        from unittest.mock import patch

        from torch.distributed.tensor._op_schema import OpSchema

        with LocalTensorMode(self.world_size):
            device_mesh = self.build_device_mesh()
            dim = 128
            x = torch.randn(8, dim, requires_grad=True)
            A = torch.randn(dim, dim, requires_grad=True)

            # Prepare DTensors that will trigger redistribution
            dx = distribute_tensor(x, device_mesh, [Shard(0)])
            dA = distribute_tensor(A, device_mesh, [Shard(0)])

            # Without ExplicitRedistributionContext, OpSchema.__str__ should NOT be called
            with patch.object(OpSchema, "__str__", autospec=True) as mock_str:
                torch.matmul(dx, dA)
                mock_str.assert_not_called()

    def test_explicit_matmul(self):
        with LocalTensorMode(self.world_size):
            device_mesh = self.build_device_mesh()
            dim = 128
            x = torch.randn(8, dim, requires_grad=True)
            A = torch.randn(dim, dim, requires_grad=True)

            # Prepare DTensors
            dx = distribute_tensor(x, device_mesh, [Shard(0)])
            dA = distribute_tensor(A, device_mesh, [Shard(0)])

            # implicit redistribute works as usual by default
            with CommDebugMode() as comm_mode:
                torch.matmul(dx, dA)
            self.assertEqual(comm_mode.get_total_counts(), 1)

            # explicit redistribute works too
            with ExplicitRedistributionContext():
                with self.assertRaisesRegex(RuntimeError, "Implicit redistribution"):
                    torch.matmul(dx, dA)
            with ExplicitRedistributionContext(mode="warn"):
                with self.assertLogs(
                    torch.distributed.tensor._utils.logger, level="WARN"
                ) as captured:
                    torch.matmul(dx, dA)
                    self.assertEqual(len(captured.output), 1)
                    self.assertRegex(
                        captured.output[0],
                        r"WARNING:.*Implicit redistribution occurred",
                    )
                    # TODO enable this once fixing the issue that op_info.schema is None in some calls to
                    # redistribute_local_tensor
                    self.assertRegex(
                        captured.output[0],
                        r".*aten\.mm\.default.*",
                    )

            # explicit redistribute allows manual redistribute
            with ExplicitRedistributionContext():
                dA_repl = dA.redistribute(device_mesh, [Replicate()])
                torch.matmul(dx, dA_repl)

            dx = distribute_tensor(x, device_mesh, [Shard(0)])
            dA = distribute_tensor(A, device_mesh, [Replicate()])
            with ExplicitRedistributionContext(strict=True):
                dY = torch.matmul(dx, dA_repl)
                loss = dY.sum()

                # we now see the error during backwards
                with self.assertRaisesRegex(RuntimeError, "Implicit redistribution"):
                    loss.backward(retain_graph=True)

                with ExplicitRedistributionContext(strict=False):
                    # but since it's a 'free' redistribute, we can still do it under non-strict mode
                    loss.backward(retain_graph=True)

                with ExplicitRedistributionContext(enable=False):
                    # and we can disable
                    loss.backward(retain_graph=True)

                # and re-enable
                with self.assertRaisesRegex(RuntimeError, "Implicit redistribution"):
                    loss.backward(retain_graph=True)


class TestIsTensorShardable(LocalTensorTestBase):
    @property
    def world_size(self):
        return 8

    def _create_spec(
        self, mesh_shape: tuple[int, ...], placements: list[Placement]
    ) -> DTensorSpec:
        mesh = init_device_mesh("cpu", mesh_shape)
        return DTensorSpec(mesh=mesh, placements=tuple(placements))

    def test_is_tensor_shardable(self):
        spec = self._create_spec((4,), [Shard(0)])
        self.assertTrue(is_tensor_shardable([8], spec))
        self.assertTrue(is_tensor_shardable([10], spec))
        self.assertFalse(is_tensor_shardable([2], spec))

        spec = self._create_spec((4, 2), [Shard(0), Shard(0)])
        self.assertTrue(is_tensor_shardable([8, 8], spec))

        spec = self._create_spec((4, 2), [Shard(0), _StridedShard(0, split_factor=2)])
        # not shardable now because of the split_factor
        self.assertFalse(is_tensor_shardable([8, 8], spec))

    def test_is_tensor_evenly_shardable(self):
        spec = self._create_spec((4,), [Shard(0)])
        self.assertTrue(is_tensor_evenly_shardable([8], spec))
        self.assertFalse(is_tensor_evenly_shardable([10], spec))

        spec = self._create_spec((4, 2), [Shard(0), Shard(0)])
        self.assertTrue(is_tensor_evenly_shardable([16, 8], spec))

        spec = self._create_spec((4, 2), [_StridedShard(0, split_factor=3), Shard(0)])
        # not evenly shardable now because of the split_factor
        self.assertFalse(is_tensor_evenly_shardable([16, 8], spec))


UtilTestWithLocalTensor = create_local_tensor_test_class(UtilTest)
TestStridedShardingWithLocalTensor = create_local_tensor_test_class(TestStridedSharding)
Test2DStridedLocalShardWithLocalTensor = create_local_tensor_test_class(
    Test2DStridedLocalShard
)

if __name__ == "__main__":
    run_tests()
