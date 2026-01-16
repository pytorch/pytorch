# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import contextlib
import itertools
import logging
import random
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
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
from torch.distributed.tensor._collective_utils import (
    redistribute_cost,
    shard_dim_alltoall,
)
from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
    ShardOrderEntry,
    TensorMeta,
)
from torch.distributed.tensor._redistribute import (
    _FlattenedTransformInfo,
    _gen_transform_infos,
    _optimize_transform_infos,
    _TransformInfo,
    redistribute_local_tensor,
    use_min_cost_redistribution_plan,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import _MaskPartial, _StridedShard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
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
    def test_partial_to_partial_forward_backward(self, dtype):
        device_mesh = self.build_device_mesh()
        partial_spec = [Partial()]

        # Create a partial tensor - each rank has the same local tensor
        # When reduced, it would be multiplied by world_size
        partial_local = torch.ones(
            12, 3, device=self.device_type, requires_grad=True, dtype=dtype
        )

        comm_mode = CommDebugMode()

        # 1) test partial -> partial forward: should be no-op
        partial_tensor = distribute_tensor(partial_local, device_mesh, partial_spec)

        with comm_mode:
            reshard_partial_tensor = partial_tensor.redistribute(
                device_mesh, partial_spec
            )
        self.assertEqual(partial_tensor.size(), partial_local.size())
        self.assertEqual(partial_tensor, reshard_partial_tensor)
        self.assertEqual(partial_tensor.to_local(), reshard_partial_tensor.to_local())
        # Verify no communication in forward
        self.assertEqual(comm_mode.get_total_counts(), 0)

        # 2) test partial -> partial backward: should be no-op
        grad_output = DTensor.from_local(
            torch.ones(12, 3, device=self.device_type, dtype=dtype),
            device_mesh,
            partial_spec,  # ← Use Partial placement!
        )
        with comm_mode:
            reshard_partial_tensor.backward(grad_output)
        grad_input = partial_tensor.grad
        self.assertEqual(grad_input.placements, partial_spec)
        self.assertEqual(grad_input.to_local(), torch.ones(12, 3, dtype=dtype))
        # Verify no communication in backward
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

    def _test_all_gather_optimization(
        self,
        global_shape: tuple[int, ...],
        placements_src: list,
        placements_dst: list,
        should_use_view: bool,
    ):
        """
        Helper method to test all_gather optimization behavior.

        Args:
            global_shape: Shape of the global tensor
            placements_src: Source placements for distribution
            placements_dst: Destination placements for redistribution
            should_use_view: Whether the optimization should use view (True) or split+cat (False)
        """
        device_mesh = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )

        global_tensor = torch.randn(*global_shape, device=self.device_type)
        dt = distribute_tensor(global_tensor, device_mesh, placements_src)

        # Capture the operations
        with DebugMode(record_torchfunction=False) as debug_mode:
            result = dt.redistribute(device_mesh, placements_dst)

        debug_str = debug_mode.debug_string()

        # Verify optimization behavior
        if should_use_view:
            # Should see 'view' but not 'split' or 'cat'
            self.assertIn("aten::view", debug_str)
            self.assertNotIn("aten::split", debug_str)
            self.assertNotIn("aten::cat", debug_str)
        else:
            # Should see 'split' and 'cat'
            self.assertIn("aten::split", debug_str)
            self.assertIn("aten::cat", debug_str)

        # Verify correctness: result should have the right placements
        self.assertEqual(result.placements, placements_dst)

        # Verify correctness: local tensor contents should match expected
        expected_dt = distribute_tensor(global_tensor, device_mesh, placements_dst)
        self.assertEqual(result.to_local(), expected_dt.to_local())

    @with_comms
    def test_all_gather_view_optimization_batch1(self):
        """
        Test that all_gather with gather_dim=1 and batch=1 uses a pure view
        instead of split+cat. This optimization avoids unnecessary copies.
        """
        # Test case: Shard on dim 1 on both mesh dimensions, then redistribute
        # to unshard on the last mesh dim. With batch=1 on the first dimension,
        # the all_gather should use a pure view operation.
        self._test_all_gather_optimization(
            global_shape=(1, 8192, 6144),
            placements_src=[Shard(1), Shard(1)],
            placements_dst=[Shard(1), Replicate()],
            should_use_view=True,
        )

    @with_comms
    def test_all_gather_no_optimization_gather_dim_not_1(self):
        """
        Test that all_gather with batch=1 but gather_dim != 1 still falls back
        to split+cat, since the view optimization only applies to gather_dim=1.
        """
        # Test case: Shard on dim 2, then redistribute to unshard on last mesh dim.
        # Even though batch=1 on first dimension, gather_dim will be 2 (not 1),
        # so we can't use the view optimization.
        self._test_all_gather_optimization(
            global_shape=(1, 8192, 6144),
            placements_src=[Shard(2), Shard(2)],
            placements_dst=[Shard(2), Replicate()],
            should_use_view=False,
        )

    @with_comms
    def test_all_gather_split_cat_fallback_batch_gt_1(self):
        """
        Test that all_gather with gather_dim=1 and batch>1 falls back to
        split+cat (not optimized to view since that would require a copy).
        """
        # Test case: Similar to batch=1 test, but with batch=4 on first dimension.
        # This should fall back to split+cat since the view optimization
        # doesn't apply when batch > 1.
        self._test_all_gather_optimization(
            global_shape=(4, 8192, 6144),
            placements_src=[Shard(1), Shard(1)],
            placements_dst=[Shard(1), Replicate()],
            should_use_view=False,
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

    @skip_if_lt_x_gpu(4)
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
                if torch.accelerator.is_available():
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
                    """S(0)[0]S(0)[1]S(0)[2]->S(0)[0]S(0)[1]R->S(0)RR->RRR->RS(0)R->RS(0)[0]S(0)[1]""",
                )
            elif idx == 1:
                self.assertExpectedInline(
                    trace_str,
                    """S(0)[1]S(0)[0]S(0)[2]->S(0)[1]S(0)[0]R->RS(0)R->RS(0)[0]S(0)[1]""",
                )
            elif idx == 2:
                self.assertExpectedInline(
                    trace_str,
                    """S(0)[1]S(0)[0]S(0)[2]->S(0)[1]S(0)[0]R->RS(0)R->RRR->S(0)RR->S(0)[0]S(0)[1]R""",
                )
            elif idx == 3:
                self.assertExpectedInline(
                    trace_str,
                    """S(0)[0]S(0)[1]R->S(0)RR->S(0)S(1)R->RS(1)R->RS(1)S(0)""",
                )
            expected_dt = _distribute_tensor(
                input_data.clone(), mesh, dst_placement, shard_order=dst_order
            )
            self.assertEqual(sharded_dt.to_local(), expected_dt.to_local())

    @with_comms
    def test_force_min_cost_redistribution_plan(self):
        """
        Test that the disable_graph_based_transform context manager correctly controls
        the redistribution algorithm selection (graph-based vs greedy).
        """
        # Set deterministic seed for reproducible tensor generation
        torch.manual_seed(21)
        mesh = init_device_mesh(self.device_type, (2, 2, 2))
        input_data = torch.randn((8, 8, 8), device=self.device_type)

        # the redistribution path differs if we use graph-based or greedy search solution
        src_placement, src_order = (
            [Shard(0), Shard(0), Shard(0)],  # All mesh dims shard tensor dim 0
            (
                ShardOrderEntry(tensor_dim=0, mesh_dims=(0, 1, 2)),
            ),  # Device order: 0→1→2
        )
        dst_placement, dst_order = (
            [Shard(1), Shard(1), Shard(1)],  # All mesh dims shard tensor dim 1
            (
                ShardOrderEntry(tensor_dim=1, mesh_dims=(0, 1, 2)),
            ),  # Device order: 0→1→2
        )

        # Test both graph-based (enable_graph=True) and greedy (enable_graph=False) algorithms
        for idx, enable_graph in enumerate([True, False]):
            sharded_dt = _distribute_tensor(
                input_data.clone(), mesh, src_placement, shard_order=src_order
            )

            with (
                use_min_cost_redistribution_plan(enabled=enable_graph),
                DebugMode(record_torchfunction=False) as debug_mode,
            ):
                sharded_dt = redistribute(sharded_dt, mesh, dst_placement, dst_order)
            trace_str = self._extract_redistribute_trace_from_debug_mode(
                debug_mode.debug_string()
            )

            # Validate graph-based algorithm trace (idx=0, disable_graph=False)
            # Graph-based uses optimal path search (Dijkstra's algorithm)
            # Expected path has 6 transformations with strategic intermediate states
            # Path: S(0)[0,1,2] → S(0)[0,1]S(2) → S(0)S(2)[1,0] →
            #       S(1)S(2)[1,0] → S(1)[0,1]S(2) → S(1)[0,1,2]
            if idx == 0:
                self.assertExpectedInline(
                    trace_str,
                    """S(0)[0]S(0)[1]S(0)[2]->S(0)[0]S(0)[1]R->S(0)RR->RRR->S(1)RR->S(1)[0]S(1)[1]R->S(1)[0]S(1)[1]S(1)[2]""",
                )
            # Validate greedy algorithm trace (idx=1, disable_graph=True)
            # Greedy uses simple heuristic approach (processes mesh dims sequentially)
            # Expected path has 6 transformations but with different intermediate states
            # Path: S(0)[0,1,2] → S(0)[0,1]R → S(0)RR →
            #       S(1)RR → S(1)[0,1]R → S(1)[0,1,2]
            elif idx == 1:
                self.assertExpectedInline(
                    trace_str,
                    """S(0)[0]S(0)[1]S(0)[2]->S(0)[0]S(0)[1]R->S(0)RR->S(1)RR->S(1)[0]S(1)[1]R->S(1)[0]S(1)[1]S(1)[2]""",
                )
            expected_dt = _distribute_tensor(
                input_data.clone(), mesh, dst_placement, shard_order=dst_order
            )
            self.assertEqual(sharded_dt.to_local(), expected_dt.to_local())

            # Clear the transformation cache between iterations. Without this,
            # the second iteration would use cached paths from the first
            _gen_transform_infos.cache_clear()

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
                # uneven sharding with padding
                (17, 5),
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
    def test_graph_based_redistribute_cost(self):
        """
        This test verifies the correctness of
            1. redistribute_cost, and
            2. min-cost redistribution algorithm

        Give src placements `SRC` and target placements `DST`, below formula
        should always hold based on the min cost graph algorithm:
        redistribute_cost(SRC, DST) <= redistribute_cost(SRC, INT) + redistribute_cost(INT, DST) for all INT
        """
        torch.manual_seed(21)

        with maybe_disable_local_tensor_mode():
            mesh = init_device_mesh(self.device_type, (2, 2, 2))
            input_tensor_shape = [
                # even sharding
                (16, 8),
                # uneven sharding with padding
                (13, 2, 13),
            ]

        for tensor_shape in input_tensor_shape:
            input_data = torch.randn(tensor_shape, device=self.device_type)
            tensor_rank = input_data.ndim
            with maybe_disable_local_tensor_mode():
                shard_orders = generate_shard_orders(mesh, tensor_rank)

            shard_orders = list(shard_orders)
            rng = random.Random(42)
            rng.shuffle(shard_orders)
            with use_min_cost_redistribution_plan(enabled=True):
                for i in range(0, len(shard_orders), 2):
                    src_order, dst_order = shard_orders[i : i + 2]
                    # prepare SRC DTensorSpec
                    src_dtensor = _distribute_tensor(
                        input_data.clone(),
                        mesh,
                        placements=None,
                        shard_order=src_order,
                    )
                    # prepare DST DTensorSpec
                    dst_dtensor = _distribute_tensor(
                        input_data.clone(),
                        mesh,
                        placements=None,
                        shard_order=dst_order,
                    )
                    src_to_dst_cost = redistribute_cost(
                        src_dtensor._spec, dst_dtensor._spec
                    )
                    # chose every two to reduce the number of tests
                    for intermediate_order in shard_orders[::2]:
                        # prepare INT DTensorSpec
                        intermediate_dtensor = _distribute_tensor(
                            input_data.clone(),
                            mesh,
                            placements=None,
                            shard_order=intermediate_order,
                        )
                        src_to_int_cost = redistribute_cost(
                            src_dtensor._spec, intermediate_dtensor._spec
                        )
                        int_to_dst_cost = redistribute_cost(
                            intermediate_dtensor._spec, dst_dtensor._spec
                        )
                        self.assertTrue(
                            src_to_dst_cost <= src_to_int_cost + int_to_dst_cost,
                            f"{tensor_shape=}, {src_order=}, {dst_order=}, {intermediate_order=}",
                        )

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

    @with_comms
    def test_redistribute_partial_to_different_partial_not_supported(self):
        # Test that redistributing from one Partial type to another raises an error
        device_mesh = self.build_device_mesh()
        local_tensor = torch.randn(4, 4, device=self.device_type)

        src_spec = DTensorSpec(
            device_mesh,
            (Partial("sum"),),
            tensor_meta=TensorMeta(
                local_tensor.size(),
                local_tensor.stride,
                local_tensor.dtype,
            ),
        )
        tgt_spec = DTensorSpec(
            device_mesh,
            (Partial("avg"),),
            tensor_meta=TensorMeta(
                local_tensor.size(),
                local_tensor.stride,
                local_tensor.dtype,
            ),
        )

        # Attempting to redistribute to Partial("max") should raise an error
        with self.assertRaisesRegex(
            AssertionError,
            r"Redistribution from one partial type.*to another.*is unsupported",
        ):
            redistribute_local_tensor(
                local_tensor,
                src_spec,
                tgt_spec,
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
            (_MaskPartial(offset_shape=torch.Size([10, 20]), offset_dim=0),)
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


class TransformInfoTest(TestCase):
    """Tests for _TransformInfo._comm_type_key method."""

    def test_comm_type_key(self):
        """Test _comm_type_key returns correct keys for different placement transforms."""
        # Comm ops: return string key
        test_cases = [
            ((Partial("sum"), Replicate()), "allreduce_sum"),
            ((Partial("max"), Replicate()), "allreduce_max"),
            ((Partial("sum"), Shard(0)), "reduce_scatter_sum"),
            ((Shard(0), Replicate()), "all_gather"),
            ((Shard(0), Shard(1)), "all_to_all"),
        ]
        for placements, expected_key in test_cases:
            info = _TransformInfo(0, placements, [8, 8])
            self.assertEqual(info._comm_type_key(), expected_key)

        # Local ops: return None
        local_cases = [
            (Replicate(), Shard(0)),  # local split
            (Replicate(), Partial("sum")),  # local partition
            (Replicate(), Replicate()),  # noop
        ]
        for placements in local_cases:
            info = _TransformInfo(0, placements, [8, 8])
            self.assertIsNone(info._comm_type_key())


class OptimizeFlattenedReductionsTest(TestCase):
    """Tests for _optimize_transform_infos helper.

    Uses fake process group since these tests don't perform actual communications.
    """

    def setUp(self):
        super().setUp()
        store = dist.HashStore()
        dist.init_process_group(backend="fake", rank=0, world_size=8, store=store)

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def test_no_flattened_mesh_returns_original(self):
        """When no flattened mesh exists, original transforms are returned with a warning."""
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("A", "B", "C"))

        transform_infos = [
            _TransformInfo(
                mesh_dim=0,
                src_dst_placements=(Partial("sum"), Replicate()),
                logical_shape=[8, 8],
            ),
            _TransformInfo(
                mesh_dim=2,
                src_dst_placements=(Partial("sum"), Replicate()),
                logical_shape=[8, 8],
            ),
        ]

        # Use a fresh warning cache to avoid global side effects
        with patch(
            "torch.distributed.tensor._redistribute._warned_missing_flattened_meshes",
            new=set(),
        ):
            with self.assertLogs(
                "torch.distributed.tensor._redistribute", level=logging.WARNING
            ) as log_context:
                src_placements = (Partial("sum"), Replicate(), Partial("sum"))
                dst_placements = (Replicate(), Replicate(), Replicate())
                result = _optimize_transform_infos(
                    transform_infos, mesh, src_placements, dst_placements
                )

            self.assertEqual(len(result), 2)
            self.assertTrue(all(isinstance(r, _TransformInfo) for r in result))

            # Verify warning message content
            self.assertEqual(len(log_context.output), 1)
            warning_msg = log_context.output[0]
            self.assertIn("While redistributing from", warning_msg)
            self.assertIn("Partial(sum)", warning_msg)  # src placement
            self.assertIn("Replicate()", warning_msg)  # dst placement
            self.assertIn("2 sequential", warning_msg)  # number of ops
            self.assertIn("allreduce_sum", warning_msg)  # comm type
            self.assertIn('"A"', warning_msg)  # mesh dim name
            self.assertIn('"C"', warning_msg)  # mesh dim name
            self.assertIn("suboptimal", warning_msg)

            # Verify warning is only issued once for same mesh dims
            with self.assertNoLogs(
                "torch.distributed.tensor._redistribute", level=logging.WARNING
            ):
                _optimize_transform_infos(
                    transform_infos, mesh, src_placements, dst_placements
                )

    def test_consecutive_reductions_flattened(self):
        """Consecutive same-type reductions on consecutive mesh dims are flattened."""
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("A", "B", "C"))
        mesh["A", "B"]._flatten("A_B")

        # Dims 0 and 1 are consecutive
        transform_infos = [
            _TransformInfo(
                mesh_dim=0,
                src_dst_placements=(Partial("sum"), Replicate()),
                logical_shape=[8, 8],
            ),
            _TransformInfo(
                mesh_dim=1,
                src_dst_placements=(Partial("sum"), Replicate()),
                logical_shape=[8, 8],
            ),
        ]

        result = _optimize_transform_infos(
            transform_infos,
            mesh,
            (Partial("sum"), Partial("sum"), Replicate()),
            (Replicate(), Replicate(), Replicate()),
        )

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], _FlattenedTransformInfo)
        self.assertEqual(result[0].original_mesh_dims, (0, 1))

    def test_non_consecutive_dims_not_merged(self):
        """Reductions on dims 0 and 2 with different op on dim 1 are NOT merged.

        Only consecutive same-type operations are merged to preserve correctness.
        Reordering operations can produce wrong results.
        """
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("A", "B", "C"))
        mesh["A", "C"]._flatten("A_C")

        # Shard on dim 1 is being transformed to Replicate
        transform_infos = [
            _TransformInfo(
                mesh_dim=0,
                src_dst_placements=(Partial("sum"), Replicate()),
                logical_shape=[8, 8],
            ),
            _TransformInfo(
                mesh_dim=1,
                src_dst_placements=(Shard(0), Replicate()),
                logical_shape=[8, 8],
            ),
            _TransformInfo(
                mesh_dim=2,
                src_dst_placements=(Partial("sum"), Replicate()),
                logical_shape=[8, 8],
            ),
        ]

        result = _optimize_transform_infos(
            transform_infos,
            mesh,
            (Partial("sum"), Shard(0), Partial("sum")),
            (Replicate(), Replicate(), Replicate()),
        )

        # Non-consecutive allreduces should NOT be merged (would change order)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(r, _TransformInfo) for r in result))

    def test_all_dims_flattened(self):
        """All dims with same reduce_op are flattened when full mesh exists."""
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("A", "B", "C"))
        mesh["A", "B", "C"]._flatten("A_B_C")

        transform_infos = [
            _TransformInfo(
                mesh_dim=0,
                src_dst_placements=(Partial("sum"), Replicate()),
                logical_shape=[8, 8],
            ),
            _TransformInfo(
                mesh_dim=1,
                src_dst_placements=(Partial("sum"), Replicate()),
                logical_shape=[8, 8],
            ),
            _TransformInfo(
                mesh_dim=2,
                src_dst_placements=(Partial("sum"), Replicate()),
                logical_shape=[8, 8],
            ),
        ]

        result = _optimize_transform_infos(
            transform_infos,
            mesh,
            (Partial("sum"), Partial("sum"), Partial("sum")),
            (Replicate(), Replicate(), Replicate()),
        )

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], _FlattenedTransformInfo)
        self.assertEqual(result[0].original_mesh_dims, (0, 1, 2))

    def test_single_reduction_not_flattened(self):
        """A single reduction is not flattened even if mesh exists."""
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("A", "B", "C"))
        mesh["A", "C"]._flatten("A_C")

        transform_infos = [
            _TransformInfo(
                mesh_dim=0,
                src_dst_placements=(Partial("sum"), Replicate()),
                logical_shape=[8, 8],
            ),
        ]

        result = _optimize_transform_infos(
            transform_infos,
            mesh,
            (Partial("sum"), Replicate(), Replicate()),
            (Replicate(), Replicate(), Replicate()),
        )

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], _TransformInfo)

    def test_empty_input(self):
        """Empty input returns empty output."""
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("A", "B", "C"))

        result = _optimize_transform_infos(
            [],
            mesh,
            (Replicate(), Replicate(), Replicate()),
            (Replicate(), Replicate(), Replicate()),
        )

        self.assertEqual(result, [])

    def test_all_gathers_grouped(self):
        """Multiple all-gather operations are grouped and flattened."""
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("A", "B", "C"))
        mesh["A", "B"]._flatten("A_B")

        transform_infos = [
            _TransformInfo(
                mesh_dim=0,
                src_dst_placements=(Shard(0), Replicate()),
                logical_shape=[8, 8],
            ),
            _TransformInfo(
                mesh_dim=1,
                src_dst_placements=(Shard(0), Replicate()),
                logical_shape=[8, 8],
            ),
        ]

        result = _optimize_transform_infos(
            transform_infos,
            mesh,
            (Shard(0), Shard(0), Replicate()),
            (Replicate(), Replicate(), Replicate()),
        )

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], _FlattenedTransformInfo)
        self.assertEqual(result[0].original_mesh_dims, (0, 1))

    def test_mixed_collective_types_not_grouped(self):
        """Different collective types (allreduce vs all-gather) are not grouped."""
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("A", "B", "C"))
        mesh["A", "B"]._flatten("A_B")

        transform_infos = [
            _TransformInfo(
                mesh_dim=0,
                src_dst_placements=(Partial("sum"), Replicate()),  # allreduce
                logical_shape=[8, 8],
            ),
            _TransformInfo(
                mesh_dim=1,
                src_dst_placements=(Shard(0), Replicate()),  # all-gather
                logical_shape=[8, 8],
            ),
        ]

        result = _optimize_transform_infos(
            transform_infos,
            mesh,
            (Partial("sum"), Shard(0), Replicate()),
            (Replicate(), Replicate(), Replicate()),
        )

        # Should remain as 2 separate transforms since collective types differ
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(r, _TransformInfo) for r in result))


class MultiDimRedistributeOptimizationTest(DTensorTestBase):
    """Integration tests for multi-dimensional redistribute correctness and optimization.

    These tests verify numerical correctness of various redistribute patterns
    on multi-dimensional meshes, including the flattened mesh optimization.
    """

    @property
    def world_size(self) -> int:
        return 8

    @with_comms
    def test_multi_dim_redistribute(self):
        """Test various redistribute patterns for correctness and comm count optimization.

        Each test case is: (mesh_shape, flattened_dims, src_placements, dst_placements, expected_comm_counts)
        - mesh_shape: tuple of mesh dimensions
        - flattened_dims: list of tuples of dims to flatten, e.g., [("A", "B")] or None
        - src_placements: source placements
        - dst_placements: target placements
        - expected_comm_counts: dict of {collective_op: count} for verification
        """
        # Define test cases
        test_cases = [
            # ===== (Partial, Partial) -> (Replicate, Replicate) with flattened mesh =====
            # Should merge 2 allreduces into 1
            (
                (4, 2),
                [("A", "B")],
                (Partial("sum"), Partial("sum")),
                (Replicate(), Replicate()),
                {funcol.all_reduce: 1},
                "2 allreduces merged to 1",
            ),
            # ===== (Partial, Partial) -> (Shard(0), Shard(0)) nested sharding =====
            # Two reduce_scatters to same tensor dim (nested) - merged
            (
                (4, 2),
                [("A", "B")],
                (Partial("sum"), Partial("sum")),
                (Shard(0), Shard(0)),
                {funcol.reduce_scatter_tensor: 1},
                "nested reduce_scatter",
            ),
            # ===== (Partial, Partial) -> (Shard(0), Shard(1)) different tensor dims =====
            # Two reduce_scatters to different tensor dims - not merged
            (
                (4, 2),
                [("A", "B")],
                (Partial("sum"), Partial("sum")),
                (Shard(0), Shard(1)),
                {funcol.reduce_scatter_tensor: 2},
                "reduce_scatter to different dims",
            ),
            # ===== 3D mesh: (Partial, Partial, Shard) -> (Shard, Shard, Shard) =====
            # Complex case: reduce_scatter + allreduce + allgather
            (
                (2, 2, 2),
                [("A", "B")],
                (Partial("sum"), Partial("sum"), Shard(0)),
                (Shard(0), Shard(0), Shard(0)),
                {
                    funcol.reduce_scatter_tensor: 1,
                    funcol.all_reduce: 1,
                    funcol.all_gather_into_tensor: 1,
                },
                "Partial,Partial,Shard -> Shard,Shard,Shard",
            ),
            # ===== 3D mesh: (Partial, Partial, Partial) -> (Replicate, Replicate, Replicate) =====
            # Three allreduces - can merge all with full flattened mesh
            (
                (2, 2, 2),
                [("A", "B", "C")],
                (Partial("sum"), Partial("sum"), Partial("sum")),
                (Replicate(), Replicate(), Replicate()),
                {funcol.all_reduce: 1},
                "3 allreduces merged to 1",
            ),
            # ===== 3D mesh: (Partial, Shard, Partial) -> (Replicate, Replicate, Replicate) =====
            # Non-consecutive allreduces with allgather in between - cannot merge allreduces
            (
                (2, 2, 2),
                [("A", "C")],
                (Partial("sum"), Shard(0), Partial("sum")),
                (Replicate(), Replicate(), Replicate()),
                {funcol.all_reduce: 2, funcol.all_gather_into_tensor: 1},
                "non-consecutive allreduces with shard in between",
            ),
            # ===== 3D mesh: (Partial, Replicate, Partial) -> (Replicate, Replicate, Replicate) =====
            # Non-consecutive allreduces with no op in between - merged
            (
                (2, 2, 2),
                [("A", "C")],
                (Partial("sum"), Replicate(), Partial("sum")),
                (Replicate(), Replicate(), Replicate()),
                {funcol.all_reduce: 1},
                "non-consecutive allreduces with replica in between",
            ),
            # ===== 2D mesh: (Shard, Shard) -> (Replicate, Replicate) =====
            # Two allgathers on same tensor dim (nested) - merged
            (
                (4, 2),
                [("A", "B")],
                (Shard(0), Shard(0)),
                (Replicate(), Replicate()),
                {funcol.all_gather_into_tensor: 1},
                "2 allgathers nested",
            ),
            # ===== 2D mesh: (Shard0, Shard1) -> (Replicate, Replicate) =====
            # Two allgathers on different tensor dim not merged
            (
                (4, 2),
                [("A", "B")],
                (Shard(0), Shard(1)),
                (Replicate(), Replicate()),
                {funcol.all_gather_into_tensor: 2},
                "2 separate allgathers",
            ),
            # ===== 3D mesh: (Partial, Partial, Replicate) -> (Shard(0), Shard(1), Replicate) =====
            # Two reduce_scatters to different tensor dims - not merged
            (
                (2, 2, 2),
                [("A", "B")],
                (Partial("sum"), Partial("sum"), Replicate()),
                (Shard(0), Shard(1), Replicate()),
                {funcol.reduce_scatter_tensor: 2},
                "reduce_scatter with Replicate unchanged",
            ),
        ]

        for (
            mesh_shape,
            flattened_dims,
            src_placements,
            dst_placements,
            expected_counts,
            desc,
        ) in test_cases:
            with self.subTest(desc=desc):
                # Create mesh with dim names
                dim_names = tuple(chr(ord("A") + i) for i in range(len(mesh_shape)))
                mesh = init_device_mesh(
                    self.device_type, mesh_shape, mesh_dim_names=dim_names
                )

                # Flatten specified dimensions
                if flattened_dims:
                    for dims_to_flatten in flattened_dims:
                        flat_name = "_".join(dims_to_flatten)
                        mesh[dims_to_flatten]._flatten(flat_name)

                # Create global tensor - size must be divisible by mesh for sharding
                global_shape = tuple(16 for _ in range(max(3, len(mesh_shape))))
                global_tensor = torch.randn(global_shape, device=self.device_type)

                # Create source DTensor
                # For Partial dims, use Replicate then reinterpret as Partial
                src_for_distribute = [
                    Replicate() if p.is_partial() else p for p in src_placements
                ]
                base_dt = distribute_tensor(
                    global_tensor, mesh, list(src_for_distribute)
                )
                local = base_dt.to_local()
                dt = DTensor.from_local(local, mesh, src_placements, run_check=False)

                # Redistribute with comm tracking
                comm_mode = CommDebugMode()
                with comm_mode:
                    result = dt.redistribute(mesh, list(dst_placements))

                # Check comm counts
                for op, expected_count in expected_counts.items():
                    actual_count = comm_mode.get_comm_counts().get(op, 0)
                    self.assertEqual(
                        actual_count,
                        expected_count,
                        f"{desc}: expected {expected_count} {op}, got {actual_count}",
                    )

                # Verify placements
                self.assertEqual(tuple(result.placements), tuple(dst_placements))

                # Verify numerical correctness via full_tensor
                # The full tensor should equal global_tensor * (product of partial mesh dim sizes)
                num_partial_sums = 1
                for i, p in enumerate(src_placements):
                    if p.is_partial():
                        num_partial_sums *= mesh.size(i)

                result_full = result.full_tensor()
                expected_full = global_tensor * num_partial_sums
                self.assertEqual(result_full, expected_full)


class FlattenedReductionIntegrationTest(DTensorTestBase):
    """Integration tests for flattened reduction optimization.

    These tests verify that redistribute actually performs fewer communications
    when flattened meshes are available.
    """

    @property
    def world_size(self):
        return 8

    @with_comms
    def test_merging_reductions(self):
        """Tests various combinations in the same test function to avoid setup time"""
        mesh = init_device_mesh(
            self.device_type, (2, 2, 2), mesh_dim_names=("A", "B", "C")
        )

        # Test: All three consecutive dims with same reduce op CAN be merged
        mesh["A", "B", "C"]._flatten("A_B_C")

        local_tensor = torch.ones(8, 8, device=self.device_type)
        dt = DTensor.from_local(
            local_tensor,
            mesh,
            (Partial("sum"), Partial("sum"), Partial("sum")),
            run_check=False,
        )

        comm_mode = CommDebugMode()
        with comm_mode:
            result = dt.redistribute(mesh, (Replicate(), Replicate(), Replicate()))

        # With full flattened mesh, should have 1 allreduce instead of 3
        self.assertEqual(comm_mode.get_total_counts(), 1)

        # Verify correctness: result should be 8x (reduced across all 8 ranks)
        expected = local_tensor * 8
        self.assertEqual(result.to_local(), expected)

        # Test: Two consecutive dims (A,B) with same reduce op CAN be merged
        mesh["A", "B"]._flatten("A_B")

        local_tensor = torch.ones(8, 8, device=self.device_type)
        dt = DTensor.from_local(
            local_tensor,
            mesh,
            (Partial("sum"), Partial("sum"), Replicate()),
            run_check=False,
        )

        comm_mode = CommDebugMode()
        with comm_mode:
            result = dt.redistribute(mesh, (Replicate(), Replicate(), Replicate()))

        # Dims 0 and 1 are consecutive, should merge to 1 allreduce
        self.assertEqual(comm_mode.get_total_counts(), 1)

        # Verify correctness: sum across A (2) and B (2) = 4x
        expected = local_tensor * 4
        self.assertEqual(result.to_local(), expected)

        # Test: Non-consecutive dims (A,C) with Replicate on B - CAN merge now
        mesh["A", "C"]._flatten("A_C")

        local_tensor = torch.ones(8, 8, device=self.device_type)
        dt = DTensor.from_local(
            local_tensor,
            mesh,
            (Partial("sum"), Replicate(), Partial("sum")),
            run_check=False,
        )

        comm_mode = CommDebugMode()
        with comm_mode:
            result = dt.redistribute(mesh, (Replicate(), Replicate(), Replicate()))

        # Replicate on dim B (not Partial) - CAN merge dims 0 and 2
        self.assertEqual(comm_mode.get_total_counts(), 1)

        # Verify correctness: sum across A (2) and C (2) = 4x
        expected = local_tensor * 4
        self.assertEqual(result.to_local(), expected)

        # Test: Non-consecutive dims (A,C) with Shard on B - CAN merge
        local_tensor = torch.ones(8, 8, device=self.device_type)
        dt = DTensor.from_local(
            local_tensor,
            mesh,
            (Partial("sum"), Shard(0), Partial("sum")),
            run_check=False,
        )

        comm_mode = CommDebugMode()
        with comm_mode:
            result = dt.redistribute(mesh, (Replicate(), Replicate(), Replicate()))

        # Shard on dim B breaks the consecutive allreduce sequence
        # So we have 3 ops: allreduce on A, allgather on B, allreduce on C
        # (non-consecutive allreduces are not merged to preserve correctness)
        self.assertEqual(comm_mode.get_total_counts(), 3)

        # Verify correctness: sum across A (2) and C (2) = 4x, gather on B
        expected = torch.ones(8 * 2, 8, device=self.device_type) * 4
        self.assertEqual(result.to_local(), expected)


RedistributeTestWithLocalTensor = create_local_tensor_test_class(
    RedistributeTest,
)

MultiDimRedistributeTestWithLocalTensor = create_local_tensor_test_class(
    MultiDimRedistributeTest,
    skipped_tests=["test_multi_dim_mesh"],
)

MultiDimRedistributeOptimizationTestWithLocalTensor = create_local_tensor_test_class(
    MultiDimRedistributeOptimizationTest,
)

DistributeWithDeviceOrderTestWithLocalTensor = create_local_tensor_test_class(
    DistributeWithDeviceOrderTest,
)

if __name__ == "__main__":
    run_tests()
