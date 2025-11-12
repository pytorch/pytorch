# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import contextlib
import itertools
import unittest

import torch
from torch.distributed._local_tensor import (
    maybe_disable_local_tensor_mode,
    maybe_run_for_local_tensor,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._collective_utils import shard_dim_alltoall
from torch.distributed.tensor._dtensor_spec import ShardOrderEntry
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import _StridedShard, MaskPartial
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUDA,
    TEST_HPU,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    generate_shard_orders,
    make_full_tensor,
    map_local_tensor_for_rank,
    patched_distribute_tensor as _distribute_tensor,
    redistribute,
    with_comms,
)
from torch.utils._debug_mode import DebugMode


funcol = torch.ops.c10d_functional


class RedistributeTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    @parametrize("dtype", [torch.float32, torch.cfloat])
    def test_shard_to_replicate_forward_backward(self, dtype):
        # 1) test shard -> replicate forward
        device_mesh = self.build_device_mesh()
        replica_spec = [Replicate()]

        input_sizes_and_shard_dim = [
            ((self.world_size * 3, 3), 0),
            ((self.world_size * 3 + 1, 3), 0),
            ((self.world_size * 3 + 2, 3), 0),
            ((3, self.world_size * 3), 1),
            ((3, self.world_size * 3 + 1), 1),
            ((3, self.world_size * 3 + 2), 1),
        ]

        comm_mode = CommDebugMode()
        for input_size, shard_dim in input_sizes_and_shard_dim:
            shard_spec = [Shard(shard_dim)]
            expected_tensor = torch.randn(
                input_size, device=self.device_type, requires_grad=True, dtype=dtype
            )
            dtensor = distribute_tensor(expected_tensor, device_mesh, shard_spec)
            with comm_mode:
                reshard_dtensor = dtensor.redistribute(device_mesh, replica_spec)
            self.assertEqual(reshard_dtensor.size(), torch.Size(input_size))
            self.assertEqual(expected_tensor, reshard_dtensor.to_local())
            self.assertEqual(
                comm_mode.get_comm_counts()[funcol.all_gather_into_tensor], 1
            )

            # 2) test shard -> replicate backward:
            # should give gradient as shard
            grad_output = torch.ones_like(reshard_dtensor)
            with comm_mode:
                reshard_dtensor.backward(grad_output)
            grad_input = dtensor.grad
            self.assertEqual(grad_input.placements, shard_spec)
            self.assertEqual(
                grad_input.to_local(),
                torch.ones(dtensor.to_local().size(), dtype=dtype),
            )
            self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_replicate_to_replicate_forward_backward(self):
        device_mesh = self.build_device_mesh()
        replica_spec = [Replicate()]
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)

        comm_mode = CommDebugMode()

        # 1) test replicate -> replicate forward
        replica_tensor = distribute_tensor(local_tensor, device_mesh, replica_spec)
        with comm_mode:
            reshard_replica_tensor = replica_tensor.redistribute(
                device_mesh, replica_spec
            )
        self.assertEqual(replica_tensor.size(), local_tensor.size())
        self.assertEqual(replica_tensor, reshard_replica_tensor)
        self.assertEqual(comm_mode.get_total_counts(), 0)

        # 2) test replicate -> replicate backward:
        # should give gradient as replicate
        grad_output = torch.ones_like(reshard_replica_tensor)
        with comm_mode:
            reshard_replica_tensor.backward(grad_output)
        grad_input = replica_tensor.grad
        self.assertEqual(grad_input.placements, replica_spec)
        self.assertEqual(grad_input.to_local(), torch.ones(12, 3))
        self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    @parametrize("dtype", [torch.float32, torch.cfloat])
    def test_replicate_to_local_partial_grad(self, dtype):
        device_mesh = self.build_device_mesh()
        replica_spec = [Replicate()]
        local_tensor = torch.randn(
            12, 3, device=self.device_type, requires_grad=True, dtype=dtype
        )

        replica_tensor = distribute_tensor(local_tensor, device_mesh, replica_spec)

        comm_mode = CommDebugMode()

        with comm_mode:
            out = replica_tensor.redistribute(placements=[Replicate()]).to_local(
                grad_placements=[Partial()]
            )
            out.backward(torch.ones_like(out))

        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertEqual(comm_mode.get_comm_counts()[funcol.all_reduce], 1)

    @with_comms
    def test_replicate_to_shard_forward_backward(self):
        device_mesh = self.build_device_mesh()
        replica_spec = [Replicate()]

        input_sizes_and_shard_dim = [
            ((self.world_size * 3, 3), 0),
            ((self.world_size * 3 + 1, 3), 0),
            ((self.world_size * 3 + 2, 3), 0),
            ((3, self.world_size * 3), 1),
            ((3, self.world_size * 3 + 1), 1),
            ((3, self.world_size * 3 + 2), 1),
        ]

        comm_mode = CommDebugMode()
        for input_size, shard_dim in input_sizes_and_shard_dim:
            shard_spec = [Shard(shard_dim)]
            # 1) test replicate -> shard forward
            local_replica = torch.randn(
                input_size, device=self.device_type, requires_grad=True
            )
            splitted_list = list(
                torch.chunk(local_replica, self.world_size, dim=shard_dim)
            )

            # make local tensor as the element of the corresponding chunked list
            local_tensor = map_local_tensor_for_rank(
                splitted_list, self.rank, lambda tl, r: tl[r]
            )
            replica_tensor = distribute_tensor(local_replica, device_mesh, replica_spec)
            with comm_mode:
                reshard_tensor = replica_tensor.redistribute(device_mesh, shard_spec)
            self.assertEqual(reshard_tensor.size(), replica_tensor.size())
            self.assertEqual(reshard_tensor.placements, shard_spec)
            self.assertEqual(reshard_tensor.to_local(), local_tensor)
            self.assertEqual(comm_mode.get_total_counts(), 0)

            # 2) test replicate -> shard backward:
            # should give gradient as replicate
            grad_output = torch.ones_like(reshard_tensor)
            with comm_mode:
                reshard_tensor.backward(grad_output)
            grad_input = replica_tensor.grad
            self.assertEqual(grad_input.placements, replica_spec)
            self.assertEqual(grad_input.to_local(), torch.ones(input_size))
            self.assertEqual(comm_mode.get_total_counts(), 1)
            self.assertEqual(
                comm_mode.get_comm_counts()[funcol.all_gather_into_tensor], 1
            )

    @with_comms
    @parametrize("dtype", [torch.float32, torch.cfloat])
    def test_partial_to_replicate_forward_backward(self, dtype):
        # Although we don't allow user to reshard to produce a partial
        # placement (i.e. user can't reshard to partial), we do allow
        # replicate to partial internally, and also partial to replicate
        # backward should work as expected
        device_mesh = self.build_device_mesh()
        partial_local = torch.ones(
            12, 3, device=self.device_type, requires_grad=True, dtype=dtype
        )
        partial_spec = [Partial()]
        replica_spec = [Replicate()]

        comm_mode = CommDebugMode()
        # test partial -> replicate, which trigger all_reduce
        partial_tensor = DTensor.from_local(partial_local, device_mesh, partial_spec)
        with comm_mode:
            global_partial_tensor = partial_tensor.redistribute(
                device_mesh, replica_spec
            )

        self.assertEqual(partial_tensor.size(), partial_local.size())
        self.assertEqual(
            partial_local * self.world_size, global_partial_tensor.to_local()
        )
        self.assertEqual(comm_mode.get_comm_counts()[funcol.all_reduce], 1)

        # test backward to have replicate grad on partial
        # for from_local backward, we want the replicate() -> partial() to be
        # pass through.
        with comm_mode:
            global_partial_tensor.backward(torch.ones_like(global_partial_tensor))
        self.assertIsNotNone(partial_local.grad)
        self.assertEqual(partial_local.grad.size(), partial_local.size())
        self.assertEqual(
            partial_local.grad, torch.ones_like(partial_local, dtype=dtype)
        )
        self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_replicate_to_replicate_forward_backward_datatype_conversion(self):
        device_mesh = self.build_device_mesh()
        replica_spec = [Replicate()]

        forward_datatypes = [
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            torch.float32,
            torch.bfloat16,
            torch.float32,
            None,
            None,
        ]
        backward_datatypes = [
            torch.bfloat16,
            torch.float32,
            torch.bfloat16,
            torch.float32,
            None,
            None,
            torch.bfloat16,
            torch.float32,
        ]

        comm_mode = CommDebugMode()

        for forward_dtype, backward_dtype in zip(forward_datatypes, backward_datatypes):
            local_tensor = torch.randn(
                12, 3, device=self.device_type, requires_grad=True
            )
            # 1) test replicate -> replicate forward
            #    forward datatype cast to self.forward_dtype and backward datatype cast to self.backward_dtype
            replica_tensor = distribute_tensor(local_tensor, device_mesh, replica_spec)

            with comm_mode:
                reshard_replica_tensor = replica_tensor.redistribute(
                    device_mesh,
                    replica_spec,
                    forward_dtype=forward_dtype,
                    backward_dtype=backward_dtype,
                )
            self.assertEqual(replica_tensor.size(), local_tensor.size())
            self.assertEqual(replica_tensor.to(forward_dtype), reshard_replica_tensor)
            self.assertEqual(comm_mode.get_total_counts(), 0)

            # 2) test replicate -> replicate backward:
            # should give gradient as replicate
            grad_output = torch.ones_like(reshard_replica_tensor)
            with comm_mode:
                reshard_replica_tensor.backward(grad_output)
            grad_input = replica_tensor.grad
            self.assertEqual(grad_input.placements, replica_spec)
            self.assertEqual(grad_input.to_local(), torch.ones(12, 3))
            self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_shard_to_replicate_forward_backward_datatype_conversion(self):
        device_mesh = self.build_device_mesh()
        replica_spec = [Replicate()]

        shard_dim_and_input_sizes = [
            (0, (self.world_size * 3, 3)),
            (0, (self.world_size * 3 + 1, 3)),
            (0, (self.world_size * 3 + 2, 3)),
            (1, (3, self.world_size * 3)),
            (1, (3, self.world_size * 3 + 1)),
            (1, (3, self.world_size * 3 + 2)),
        ]

        forward_datatypes = [
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            torch.float32,
            torch.bfloat16,
            torch.float32,
            None,
            None,
        ]
        backward_datatypes = [
            torch.bfloat16,
            torch.float32,
            torch.bfloat16,
            torch.float32,
            None,
            None,
            torch.bfloat16,
            torch.float32,
        ]

        comm_mode = CommDebugMode()

        for forward_dtype, backward_dtype in zip(forward_datatypes, backward_datatypes):
            for shard_dim, input_size in shard_dim_and_input_sizes:
                # 1) test shard -> replicate forward
                shard_spec = [Shard(shard_dim)]
                expected_tensor = torch.randn(
                    input_size, device=self.device_type, requires_grad=True
                )
                dtensor = distribute_tensor(expected_tensor, device_mesh, shard_spec)
                with comm_mode:
                    reshard_dtensor = dtensor.redistribute(
                        device_mesh,
                        replica_spec,
                        forward_dtype=forward_dtype,
                        backward_dtype=backward_dtype,
                    )
                self.assertEqual(reshard_dtensor.size(), torch.Size(input_size))
                self.assertEqual(
                    expected_tensor.to(forward_dtype), reshard_dtensor.to_local()
                )
                self.assertEqual(
                    comm_mode.get_comm_counts()[funcol.all_gather_into_tensor], 1
                )

                # 2) test shard -> replicate backward:
                # should give gradient as shard
                grad_output = torch.ones_like(reshard_dtensor)
                with comm_mode:
                    reshard_dtensor.backward(grad_output)
                grad_input = dtensor.grad
                self.assertEqual(grad_input.placements, shard_spec)
                self.assertEqual(
                    grad_input.to_local(), torch.ones(dtensor.to_local().size())
                )
                self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_replicate_to_partial(self):
        device_mesh = self.build_device_mesh()
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)
        partial_spec = Partial()
        replica_spec = Replicate()
        # 1) test replicate -> partial forward
        replica_tensor = distribute_tensor(local_tensor, device_mesh, [replica_spec])
        with self.assertRaisesRegex(RuntimeError, "Can not redistribute"):
            partial_tensor = replica_tensor.redistribute(device_mesh, [partial_spec])

        from torch.distributed.tensor._redistribute import Redistribute

        comm_mode = CommDebugMode()

        with comm_mode:
            partial_tensor = Redistribute.apply(
                replica_tensor, device_mesh, [partial_spec]
            )
        self.assertEqual(partial_tensor.size(), local_tensor.size())
        # test it successfully zero out the contents on other ranks
        self.assertEqual(
            replica_tensor.to_local() / self.world_size, partial_tensor.to_local()
        )
        self.assertEqual(comm_mode.get_total_counts(), 0)

        # replicate to partial on sub groups
        local_tensor = torch.randn(12, 3, device=self.device_type)
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size).reshape(self.world_size // 2, 2),
        )
        # 1) test replicate -> partial on 2d-mesh subgroups
        replica_tensor = distribute_tensor(
            local_tensor, device_mesh, [replica_spec, replica_spec]
        )
        with comm_mode:
            partial_tensor = Redistribute.apply(
                replica_tensor, device_mesh, [partial_spec, partial_spec]
            )
        self.assertEqual(partial_tensor.size(), local_tensor.size())

        self.assertEqual(
            replica_tensor.to_local() / self.world_size,
            partial_tensor.to_local(),
        )
        self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    @parametrize("dtype", [torch.float32, torch.cfloat])
    def test_partial_to_shard(self, dtype):
        device_mesh = self.build_device_mesh()
        partial_spec = [Partial()]
        my_rank = self.rank

        input_sizes_and_shard_dim = [
            ((self.world_size * 3, 3), 0),
            ((self.world_size * 3 + 1, 3), 0),
            ((self.world_size * 3 + 2, 3), 0),
            ((3, self.world_size * 3), 1),
            ((3, self.world_size * 3 + 1), 1),
            ((3, self.world_size * 3 + 2), 1),
        ]

        comm_mode = CommDebugMode()

        for input_size, shard_dim in input_sizes_and_shard_dim:
            shard_spec = [Shard(shard_dim)]

            partial_local = torch.ones(input_size, device=self.device_type, dtype=dtype)
            partial_tensor = DTensor.from_local(
                partial_local, device_mesh, partial_spec, run_check=False
            )

            full_chunk_size = (
                input_size[shard_dim] + self.world_size - 1
            ) // self.world_size
            chunk_sizes = [
                max(
                    min(input_size[shard_dim], full_chunk_size * (idx + 1))
                    - full_chunk_size * idx,
                    0,
                )
                for idx in range(self.world_size)
            ]

            @maybe_run_for_local_tensor
            def _compute_local_shape(rank) -> list[int]:
                local_shape = list(input_size)
                local_shape[shard_dim] = chunk_sizes[rank]
                return local_shape

            local_shape = _compute_local_shape(my_rank)

            # test partial to shard, trigger reduce_scatter
            with comm_mode:
                scatter_shard_tensor = partial_tensor.redistribute(
                    device_mesh, shard_spec
                )
            self.assertEqual(scatter_shard_tensor.size(), partial_tensor.size())
            self.assertEqual(scatter_shard_tensor.placements, shard_spec)
            self.assertEqual(
                scatter_shard_tensor.to_local(),
                torch.ones(local_shape, dtype=dtype) * self.world_size,
            )
            self.assertEqual(
                comm_mode.get_comm_counts()[funcol.reduce_scatter_tensor], 1
            )

    @with_comms
    def test_redistribute_negative_shard_dim(self):
        device_mesh = self.build_device_mesh()
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)
        shard_spec = [Shard(1)]
        shard_minus_spec = [Shard(-1)]

        shard_tensor = distribute_tensor(local_tensor, device_mesh, shard_spec)
        self.assertEqual(shard_tensor.placements[0].dim, 1)
        reshard_tensor = shard_tensor.redistribute(device_mesh, shard_minus_spec)
        self.assertEqual(reshard_tensor.placements[0].dim, 1)

    @with_comms
    def test_redistribute_uneven_sharding(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))
        data_to_test = [
            # uneven on last mesh dim
            torch.randn((10, 5), device=self.device_type),
            # uneven on both mesh dims
            torch.randn((9, 5), device=self.device_type),
            # smaller than mesh dim shape
            torch.randn((3, 5), device=self.device_type),
            torch.randn((1, 3), device=self.device_type),
        ]

        sharding_to_tests = [
            [Shard(0), Shard(0)],
            [Shard(0), Shard(1)],
        ]

        for input_tensor in data_to_test:
            for placements in sharding_to_tests:
                dt = distribute_tensor(input_tensor, mesh, placements)
                dt_full_tensor = dt.full_tensor()
                self.assertEqual(dt_full_tensor, input_tensor)

    @with_comms
    @parametrize("dtype", [torch.float32, torch.cfloat])
    def test_redistribute_shard_dim_change(self, dtype):
        # test 1d device mesh
        mesh_1d = self.build_device_mesh()
        data_to_test = [
            # evenly sharded case
            torch.randn((8, 8), device=self.device_type, dtype=dtype),
            # 3d or more dims
            torch.randn((8, 8, 8), device=self.device_type, dtype=dtype),
            # uneven case 1
            torch.randn((8, 5), device=self.device_type, dtype=dtype),
            # uneven case 2
            torch.randn((5, 8), device=self.device_type, dtype=dtype),
            # uneven case 3
            torch.randn((5, 5), device=self.device_type, dtype=dtype),
        ]

        sharding_src_dst_pairs = [([Shard(0)], [Shard(1)]), ([Shard(1)], [Shard(0)])]

        comm_mode = CommDebugMode()

        for input_data in data_to_test:
            for src, dst in sharding_src_dst_pairs:
                expected_dt = distribute_tensor(input_data.clone(), mesh_1d, dst)
                sharded_dt = distribute_tensor(input_data, mesh_1d, src)
                with comm_mode:
                    out_dt = sharded_dt.redistribute(mesh_1d, dst)
                self.assertEqual(out_dt.placements, expected_dt.placements)
                local_out_dt = out_dt.to_local()
                local_expected_dt = expected_dt.to_local()
                self.assertEqual(out_dt.to_local(), expected_dt.to_local())
                if TEST_HPU or TEST_CUDA:
                    self.assertEqual(
                        comm_mode.get_comm_counts()[
                            torch.ops._dtensor.shard_dim_alltoall
                        ],
                        1,
                    )
                else:
                    # TODO: Integrate local tensor with CommDebugMode
                    if not self.is_local_tensor_enabled:
                        self.assertEqual(
                            comm_mode.get_comm_counts()[funcol.all_gather_into_tensor],
                            1,
                        )

        # test 2d device mesh
        mesh_2d = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )
        data_to_test_2d = [
            # evenly sharded case
            torch.randn((8, 8), device=self.device_type, dtype=dtype),
            # 3d or more dims
            torch.randn((8, 8, 8), device=self.device_type, dtype=dtype),
            # uneven case 1
            torch.randn((8, 5), device=self.device_type, dtype=dtype),
            # uneven case 2
            torch.randn((5, 8), device=self.device_type, dtype=dtype),
            # uneven case 3
            torch.randn((5, 5), device=self.device_type, dtype=dtype),
        ]
        sharding_src_dst_pairs_2d = [
            ([Shard(0), Shard(1)], [Shard(0), Shard(0)]),
            ([Shard(0), Shard(1)], [Shard(1), Shard(0)]),
            ([Shard(0), Shard(0)], [Shard(1), Shard(1)]),
        ]
        comm_counts_2d = [
            1,  # 1: S1 -> S0
            2,  # 1: S1 -> R, 0: S0 -> S1, 1: R -> S0
            2,  # 1: S0 -> R, 0: S0 -> S1, 1: R -> S1
        ]

        for input_data in data_to_test_2d:
            if input_data.ndim > 2:
                sharding_spec_combs = sharding_src_dst_pairs_2d + [
                    ([Shard(0), Shard(2)], [Shard(1), Shard(0)]),
                    ([Shard(1), Shard(1)], [Shard(1), Shard(2)]),
                ]
                comm_counts_2d = comm_counts_2d + [
                    2,  # 1. S2 -> R, 0: S0 -> S1, 1: R -> S0
                    1,  # 1: S1 -> S2
                ]
            else:
                sharding_spec_combs = sharding_src_dst_pairs_2d

            for idx, (src, dst) in enumerate(sharding_spec_combs):
                expected_dt = distribute_tensor(input_data.clone(), mesh_2d, dst)
                sharded_dt = distribute_tensor(input_data, mesh_2d, src)
                with comm_mode:
                    out_dt = sharded_dt.redistribute(mesh_2d, dst)

                self.assertEqual(out_dt.placements, expected_dt.placements)
                if not self.is_local_tensor_enabled:
                    self.assertEqual(comm_mode.get_total_counts(), comm_counts_2d[idx])

                local_out_dt = out_dt.to_local()
                local_expected_dt = expected_dt.to_local()
                self.assertEqual(local_out_dt, local_expected_dt)

    @with_comms
    @parametrize("dtype", [torch.float32, torch.cfloat])
    def test_shard_dim_alltoall(self, dtype):
        # init 2d mesh here so we can test when group_rank != global_rank
        mesh = init_device_mesh(self.device_type, (2, 2))
        tensor = torch.randn(12, self.world_size, device=self.device_type, dtype=dtype)
        new_tensor = shard_dim_alltoall(tensor, 0, 1, mesh, 0)

        meta_tensor = torch.randn(12, self.world_size, device="meta")
        new_meta_tensor = shard_dim_alltoall(meta_tensor, 0, 1, mesh, 0)

        self.assertEqual(new_tensor.shape, new_meta_tensor.shape)
        self.assertEqual(new_tensor.stride(), new_meta_tensor.stride())

    @with_comms
    def test_one_chunk_mesh(self):
        # mesh size is 1 on second dim
        mesh = init_device_mesh(self.device_type, (4, 1))

        srcs = [Shard(1), Replicate(), Partial()]
        dsts = [Shard(0), Shard(1), Replicate()]

        comm_mode = CommDebugMode()

        for src, dst in itertools.product(srcs, dsts):
            tensor = torch.randn(16, 8, device=self.device_type)
            dt = DTensor.from_local(tensor, mesh, [Shard(0), src])

            with comm_mode:
                out = dt.redistribute(mesh, [Shard(0), dst])

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(out.placements, [Shard(0), dst])

    @with_comms
    def test_redistribute_to_partial(self):
        mesh = init_device_mesh(self.device_type, (2, 2))

        tensor = torch.randn(12, 8, device=self.device_type)

        test_cases = [
            # Partial to Partial is allowed
            ([Partial(), Shard(0)], [Partial(), Shard(0)], True),
            ([Partial(), Shard(0)], [Partial(), Shard(1)], True),
            ([Shard(0), Partial()], [Replicate(), Partial()], True),
            ([Shard(0), Partial("prod")], [Replicate(), Partial("prod")], True),
            # Non-Partial to Partial is NOT allowed
            ([Shard(0), Replicate()], [Shard(0), Partial()], False),
            ([Shard(0), Replicate()], [Replicate(), Partial()], False),
            ([Shard(0), Shard(1)], [Replicate(), Partial()], False),
            # Partial to partial is allowed, if only the reduction ops is the same
            ([Shard(0), Partial("prod")], [Replicate(), Partial("sum")], False),
        ]

        for src, dst, allow in test_cases:
            dt = DTensor.from_local(tensor, mesh, src)
            raise_context = (
                self.assertRaisesRegex(RuntimeError, "Can not redistribute")
                if not allow
                else contextlib.nullcontext()
            )

            with raise_context:
                out = dt.redistribute(mesh, dst)
                self.assertEqual(out.placements, dst)


instantiate_parametrized_tests(RedistributeTest)


class MultiDimRedistributeTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 8

    @with_comms
    def test_multi_dim_mesh(self):
        devices = torch.arange(self.world_size)
        for mesh_shape in [devices, devices.view(4, 2), devices.view(2, 2, 2)]:
            mesh_shape = torch.arange(self.world_size).view(-1, 2)
            device_mesh = DeviceMesh(self.device_type, mesh_shape)
            tensor_shape = (16, 24)

            if torch.distributed.get_rank() == 0:
                full_tensor = torch.randn(*tensor_shape)
            else:
                # these should be entirely ignored
                # because distribute_tensor is expected to override shards in ranks != 0
                full_tensor = torch.ones(*tensor_shape)

            possibilities = [Replicate()] + [Shard(i) for i in range(full_tensor.ndim)]
            all_outputs = list(itertools.product(*(mesh_shape.ndim * [possibilities])))
            all_inputs = list(
                itertools.product(*(mesh_shape.ndim * [possibilities + [Partial()]]))
            )

            for inputs in all_inputs:
                # if partial, temporarily make it Replicated, then replace replicated with partial afterwards
                repl_inputs = [Replicate() if s.is_partial() else s for s in inputs]
                dt = distribute_tensor(full_tensor, device_mesh, repl_inputs)

                if repl_inputs != inputs:
                    # create a new DTensor reinterpreting some of the replicated entries as "Partial"
                    dt = DTensor.from_local(
                        dt.to_local(), device_mesh, inputs, run_check=False
                    )

                for outputs in all_outputs:
                    # redistribute on target outputs
                    dt2 = dt.redistribute(device_mesh, outputs)

                    # replicate and then get first shard
                    local_full = dt2.full_tensor()

                    if torch.distributed.get_rank() == 0:
                        self.assertEqual(local_full.shape, full_tensor.shape)

                        num_sums = 1
                        for idx, input in enumerate(inputs):
                            if input.is_partial():
                                num_sums *= mesh_shape.size(idx)
                        expected = num_sums * full_tensor
                        self.assertEqual(local_full, expected)

    @with_comms
    def test_redistribute_shard_dim_multi_dim_mesh(self):
        mesh = init_device_mesh(self.device_type, (2, 2, 2))
        input_data = torch.randn((8, 8, 8), device=self.device_type)

        sharding_src_dst_pairs_3d = [
            ([Shard(0), Shard(0), Shard(0)], [Shard(1), Shard(1), Shard(1)]),
            ([Shard(0), Shard(1), Shard(0)], [Shard(1), Shard(0), Shard(0)]),
            ([Shard(0), Shard(1), Shard(2)], [Shard(2), Shard(1), Shard(0)]),
            ([Shard(1), Shard(0), Shard(0)], [Replicate(), Shard(0), Shard(0)]),
            ([Shard(1), Replicate(), Shard(0)], [Replicate(), Shard(0), Shard(0)]),
            ([Shard(0), Shard(0), Shard(1)], [Shard(0), Shard(1), Shard(2)]),
        ]
        comm_counts_3d = [
            3,  # 2: S0 - R, 1: S1 -> R, 0: S0 -> S1
            3,  # 2: S0 -> R, 1: S1 -> R, 0: S0 -> S1, 1: R -> S0, 2: R -> S0
            2,  # 2: S2 -> R, 0: S1 -> S2
            1,  # 0: S1 -> R
            2,  # 2: S0 -> R, 1: R -> S0, 2: R -> S0, 0: S1 -> R
            2,  # 2: S1 -> S2, 1: S0 -> S1
        ]

        comm_mode = CommDebugMode()
        for idx, (src_placement, dst_placement) in enumerate(sharding_src_dst_pairs_3d):
            expected_dt = distribute_tensor(input_data.clone(), mesh, dst_placement)
            sharded_dt = distribute_tensor(input_data, mesh, src_placement)

            with comm_mode:
                out_dt = sharded_dt.redistribute(mesh, dst_placement)

            self.assertEqual(out_dt.placements, expected_dt.placements)
            self.assertEqual(comm_mode.get_total_counts(), comm_counts_3d[idx])

            local_out_dt = out_dt.to_local()
            local_expected_dt = expected_dt.to_local()
            self.assertEqual(local_out_dt, local_expected_dt)


class DistributeWithDeviceOrderTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 8

    def _extract_redistribute_trace_from_debug_mode(self, s: str) -> str:
        import re

        match = re.search(r"trace:\s*(.*)\)", s)
        if match:
            trace_str = match.group(1)
            return trace_str
        else:
            return ""

    @with_comms
    def test_ordered_redistribute(self):
        """Test ordered redistribution with various sharding syntaxes"""
        torch.manual_seed(21)
        mesh = init_device_mesh(self.device_type, (2, 2, 2))
        input_data = torch.randn((8, 8, 8), device=self.device_type)
        sharding_src_dst_pairs_with_expected_trace = [
            (
                (
                    [Shard(0), Shard(0), Shard(0)],
                    (ShardOrderEntry(tensor_dim=0, mesh_dims=(0, 1, 2)),),
                ),
                (
                    [Replicate(), Shard(0), Shard(0)],
                    (ShardOrderEntry(tensor_dim=0, mesh_dims=(1, 2)),),
                ),
            ),
            (
                (
                    [Shard(0), Shard(0), Shard(0)],
                    (ShardOrderEntry(tensor_dim=0, mesh_dims=(1, 0, 2)),),
                ),
                (
                    [Replicate(), Shard(0), Shard(0)],
                    (ShardOrderEntry(tensor_dim=0, mesh_dims=(1, 2)),),
                ),
            ),
            (
                (
                    [Shard(0), Shard(0), Shard(0)],
                    (ShardOrderEntry(tensor_dim=0, mesh_dims=(1, 0, 2)),),
                ),
                (
                    [Shard(0), Shard(0), Replicate()],
                    (ShardOrderEntry(tensor_dim=0, mesh_dims=(0, 1)),),
                ),
            ),
            # If we use the graph search solution, the redistribution path will
            # be S(0)[0, 1] -> S(0)[0]S(1)[1] -> S(1)[1] -> S(0)[2]S(1)[1],
            # which takes only 1 comm count. However, this placement follows the
            # default device order and the greedy solution will be triggered,
            # which results in path: S(0)[0, 1] -> S(0)[0]S(1)[1] -> S(1)[1] ->
            # S(0)[2]S(1)[1] with 2 comm count
            (
                (
                    [Shard(0), Shard(0), Replicate()],
                    (ShardOrderEntry(tensor_dim=0, mesh_dims=(0, 1)),),
                ),
                (
                    [Replicate(), Shard(1), Shard(0)],
                    (
                        ShardOrderEntry(tensor_dim=0, mesh_dims=(2,)),
                        ShardOrderEntry(tensor_dim=1, mesh_dims=(1,)),
                    ),
                ),
            ),
        ]
        for idx, ((src_placement, src_order), (dst_placement, dst_order)) in enumerate(
            sharding_src_dst_pairs_with_expected_trace
        ):
            sharded_dt = _distribute_tensor(
                input_data.clone(), mesh, src_placement, shard_order=src_order
            )
            with DebugMode(record_torchfunction=False) as debug_mode:
                sharded_dt = redistribute(sharded_dt, mesh, dst_placement, dst_order)
            trace_str = self._extract_redistribute_trace_from_debug_mode(
                debug_mode.debug_string()
            )
            if idx == 0:
                self.assertExpectedInline(
                    trace_str,
                    """S(0)[0]S(0)[1]S(0)[2]->S(0)[0]S(0)[1]S(1)->S(0)S(1)[1]S(1)[0]->RS(1)[1]S(1)[0]->RS(0)S(1)->RS(0)[0]S(0)[1]""",
                )
            elif idx == 1:
                self.assertExpectedInline(
                    trace_str,
                    """S(0)[1]S(0)[0]S(0)[2]->S(0)[1]S(0)[0]S(1)->RS(0)S(1)->RS(0)[0]S(0)[1]""",
                )
            elif idx == 2:
                self.assertExpectedInline(
                    trace_str,
                    """S(0)[1]S(0)[0]S(0)[2]->S(0)[1]S(0)[0]R->S(1)S(0)R->S(1)S(2)R->S(0)S(2)R->S(0)[0]S(0)[1]R""",
                )
            elif idx == 3:
                self.assertExpectedInline(
                    trace_str,
                    """S(0)[0]S(0)[1]R->S(0)S(1)R->RS(1)R->RS(1)S(0)""",
                )
            expected_dt = _distribute_tensor(
                input_data.clone(), mesh, dst_placement, shard_order=dst_order
            )
            self.assertEqual(sharded_dt.to_local(), expected_dt.to_local())

    @with_comms
    def test_generate_shard_orders(self):
        """Check if `generate_shard_orders` generates unique sharding combinations"""
        import math

        test_inputs = [
            {"mesh": init_device_mesh(self.device_type, (2, 2, 2)), "tensor_rank": 2},
            {"mesh": init_device_mesh(self.device_type, (2, 2, 2)), "tensor_rank": 3},
            {"mesh": init_device_mesh(self.device_type, (2, 2, 2)), "tensor_rank": 4},
        ]
        for test_input in test_inputs:
            all_combinations = []
            for shard_order in generate_shard_orders(
                test_input["mesh"], test_input["tensor_rank"]
            ):
                all_combinations.append(shard_order)  # noqa: PERF402
            for i in range(len(all_combinations)):
                for j in range(i + 1, len(all_combinations)):
                    assert all_combinations[i] != all_combinations[j], (
                        f"Duplicate elements found in all_combinations {all_combinations[i]}, {all_combinations[j]}"
                    )
            expected_total_combination = 0
            N = test_input["mesh"].ndim
            M = test_input["tensor_rank"]
            for i in range(1, N + 1):
                # assign total i split of device to tensor dims
                if M < i:
                    continue
                device_combination_count = math.comb(
                    N - 1, i - 1
                )  # choose i-1 non-empty segments from a list of size N
                tensor_dim_order_permutation = math.comb(M, i)  # choose i tensor dims
                expected_total_combination += (
                    device_combination_count * tensor_dim_order_permutation
                )
            # multiply by total possible permutation of device order
            expected_total_combination *= math.factorial(N)
            self.assertEqual(len(all_combinations), expected_total_combination)

    @with_comms
    def test_ordered_distribute_all_combination(self):
        """Exhaustively test all possible sharding combinations and verify correctness"""
        torch.manual_seed(21)

        with maybe_disable_local_tensor_mode():
            mesh = init_device_mesh(self.device_type, (2, 2, 2))
            input_tensor_shape = [
                # even sharding
                (16, 8),
                (8, 16, 32),
                (8, 32, 16, 16),
                # uneven sharding with padding
                (17, 5),
                (13, 2, 13),
                (33, 16, 8, 1),
            ]

        # 1. Verify correctness of distribute_tensor from Tensor to DTensor.
        for tensor_shape in input_tensor_shape:
            input_data = torch.randn(tensor_shape, device=self.device_type)
            tensor_rank = input_data.ndim
            with maybe_disable_local_tensor_mode():
                shard_orders = generate_shard_orders(mesh, tensor_rank)
            for shard_order in shard_orders:
                sharded_dt = _distribute_tensor(
                    input_data.clone(), mesh, placements=None, shard_order=shard_order
                )
                self.assertEqual(make_full_tensor(sharded_dt), input_data)

        # 2. Verify the correctness of redistribution from DTensor to DTensor.
        # This test repeatedly redistributes a DTensor to various ordered
        # placements and checks that the resulting tensor matches the original
        # full tensor.
        for tensor_shape in input_tensor_shape:
            input_data = torch.randn(tensor_shape, device=self.device_type)
            tensor_rank = input_data.ndim
            prev_sharded_dt = None
            with maybe_disable_local_tensor_mode():
                shard_orders = generate_shard_orders(mesh, tensor_rank)
            for shard_order in shard_orders:
                if prev_sharded_dt is None:
                    prev_sharded_dt = _distribute_tensor(
                        input_data.clone(),
                        mesh,
                        placements=None,
                        shard_order=shard_order,
                    )
                else:
                    sharded_dt = redistribute(
                        prev_sharded_dt, mesh, placements=None, shard_order=shard_order
                    )
                    self.assertEqual(make_full_tensor(sharded_dt), input_data)
                    prev_sharded_dt = sharded_dt

    @with_comms
    def test_ordered_redistribute_with_partial(self):
        """Test mixing Partial in the original placements and do redistribute."""
        # This test takes 226s to complete on 8XA100...
        torch.manual_seed(21)
        with maybe_disable_local_tensor_mode():
            mesh = init_device_mesh(self.device_type, (2, 2, 2))
            input_tensor_shape = [
                # even sharding
                (16, 8),
                (8, 16, 32),
                # uneven sharding with padding
                (17, 5),
                (13, 2, 13),
                (33, 16, 8, 1),
            ]
            placement_choice = [
                Shard(0),
                Shard(1),
                Shard(2),
                Partial("sum"),
                Partial("min"),
                Replicate(),
            ]
            # pick 3 for the 3D mesh
            partial_placement_comb = list(itertools.combinations(placement_choice, 3))

        def _is_valid_placement(placements, tensor_rank):
            # Check if placements is valid for tensor with rank `tensor_rank`
            for placement in placements:
                if isinstance(placement, Shard):
                    if placement.dim >= tensor_rank:
                        return False
            return True

        for shape in input_tensor_shape:
            for placements in partial_placement_comb:
                if not _is_valid_placement(placements, len(shape)):
                    continue
                local_tensor = torch.randn(shape, device=self.device_type)
                full_tensor = DTensor.from_local(local_tensor, mesh, placements)
                with maybe_disable_local_tensor_mode():
                    shard_orders = generate_shard_orders(mesh, len(shape))
                for shard_order in shard_orders:
                    sharded_dt = redistribute(
                        full_tensor, mesh, placements=None, shard_order=shard_order
                    )
                    self.assertEqual(
                        make_full_tensor(sharded_dt), make_full_tensor(full_tensor)
                    )

    @unittest.skip(
        "Temporarily skipping until we support special placement types in "
        "graph based redistribution"
    )
    @with_comms
    def test_ordered_redistribute_for_special_placement(self):
        """Test ordered redistribution with special placement"""
        torch.manual_seed(21)
        mesh = init_device_mesh(self.device_type, (8,))
        input_data = torch.randn((8, 8), device=self.device_type)
        src_placement = [Shard(1)]
        tgt_placement = [
            (MaskPartial(offset_shape=torch.Size([10, 20]), offset_dim=0),)
        ]
        sharded_dt = _distribute_tensor(
            input_data.clone(),
            mesh,
            src_placement,
            shard_order=(ShardOrderEntry(tensor_dim=1, mesh_dims=(0,)),),
        )
        sharded_dt = redistribute(sharded_dt, mesh, tgt_placement, shard_order=None)

    @with_comms
    def test_shard_order_same_data_as_strided_shard(self):
        device_mesh = init_device_mesh(self.device_type, (4, 2))
        x = torch.randn(8, 4, device=self.device_type)
        # specify right-to-left order use _StridedShard
        strided_placement = [_StridedShard(-2, split_factor=2), Shard(-2)]
        x_strided_dt = distribute_tensor(x, device_mesh, strided_placement)
        # specify right-to-left order use ordered shard
        x_ordered_dt = _distribute_tensor(
            x,
            device_mesh,
            placements=[Shard(0), Shard(0)],
            shard_order=(ShardOrderEntry(tensor_dim=0, mesh_dims=(1, 0)),),
        )
        self.assertEqual(x_ordered_dt.to_local(), x_strided_dt.to_local())


RedistributeTestWithLocalTensor = create_local_tensor_test_class(
    RedistributeTest,
)

MultiDimRedistributeTestWithLocalTensor = create_local_tensor_test_class(
    MultiDimRedistributeTest,
    skipped_tests=["test_multi_dim_mesh"],
)

DistributeWithDeviceOrderTestWithLocalTensor = create_local_tensor_test_class(
    DistributeWithDeviceOrderTest,
)

if __name__ == "__main__":
    run_tests()
