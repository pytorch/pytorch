# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import contextlib
import itertools
import math
from typing import cast

import torch
import torch.distributed as dist
from torch import rand, randn, Tensor
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._ops._view_ops import (
    _ViewShardingPropagator,
    Broadcast,
    dim_maps,
    Flatten,
    InputDim,
    propagate_shape_and_sharding,
    Repeat,
    Singleton,
    Split,
    view_groups,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import _StridedShard, Placement
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorContinuousTestBase,
    LocalDTensorContinuousTestBase,
)
from torch.utils import _pytree as pytree


class TestViewOps(DTensorContinuousTestBase):
    world_size = 6

    @staticmethod
    def _get_all_factorizations(n):
        """Return all ways to factor n into a sorted tuple of integers > 1.

        Each factorization has length >= 2 (non-trivial unflatten).
        Only sorted (non-decreasing) factorizations are returned since permutations
        exercise the same split_factor logic with different constants.

        Examples:
            12 -> [(2, 6), (3, 4), (2, 2, 3)]
            36 -> [(2, 18), (3, 12), (4, 9), (6, 6), (2, 2, 9), (2, 3, 6), (3, 3, 4), (2, 2, 3, 3)]
        """

        def get_sorted_factorizations(remaining, min_factor=2):
            if remaining == 1:
                return [()]
            result = []
            for f in range(min_factor, remaining + 1):
                if remaining % f == 0:
                    for sub in get_sorted_factorizations(remaining // f, f):
                        result.append((f,) + sub)
            return result

        return [f for f in get_sorted_factorizations(n) if len(f) >= 2]

    def test_view_groups(self):
        self.assertEqual(
            view_groups([2, 3], [3, 2]),
            (
                Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 0),
                Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 1),
            ),
        )
        self.assertEqual(
            view_groups([3, 4, 5], [12, 5]),
            (Flatten((InputDim(0), InputDim(1))), InputDim(2)),
        )
        self.assertEqual(
            view_groups([2, 3, 4, 5, 7], [12, 70]),
            (
                Split(
                    Flatten(
                        (
                            InputDim(0),
                            InputDim(1),
                            InputDim(2),
                            InputDim(3),
                            InputDim(4),
                        )
                    ),
                    (12, 70),
                    0,
                ),
                Split(
                    Flatten(
                        (
                            InputDim(0),
                            InputDim(1),
                            InputDim(2),
                            InputDim(3),
                            InputDim(4),
                        )
                    ),
                    (12, 70),
                    1,
                ),
            ),
        )
        self.assertEqual(
            view_groups([2, 3, 4, 5, 7], [3, 8, 7, 5]),
            (
                Split(Flatten((InputDim(0), InputDim(1), InputDim(2))), (3, 8), 0),
                Split(Flatten((InputDim(0), InputDim(1), InputDim(2))), (3, 8), 1),
                Split(Flatten((InputDim(3), InputDim(4))), (7, 5), 0),
                Split(Flatten((InputDim(3), InputDim(4))), (7, 5), 1),
            ),
        )
        self.assertEqual(
            view_groups([3, 4, 8, 3], [12, 4, 2, 3]),
            (
                Flatten((InputDim(0), InputDim(1))),
                Split(InputDim(2), (4, 2), 0),
                Split(InputDim(2), (4, 2), 1),
                InputDim(3),
            ),
        )
        self.assertEqual(
            view_groups([3, 24], [1, 3, 2, 4, 1, 3, 1]),
            (
                Singleton(),
                InputDim(0),
                Split(InputDim(1), (2, 4, 3), 0),
                Split(InputDim(1), (2, 4, 3), 1),
                Singleton(),
                Split(InputDim(1), (2, 4, 3), 2),
                Singleton(),
            ),
        )
        self.assertEqual(
            view_groups([1, 1, 3, 2, 1, 1], [6, 1, 1, 1]),
            (
                Flatten((InputDim(2), InputDim(3))),
                InputDim(4),
                InputDim(5),
                Singleton(),
            ),
        )
        self.assertEqual(
            view_groups([1, 1, 12, 1, 1, 1, 2, 5, 1], [3, 4, 1, 10]),
            (
                Split(InputDim(2), (3, 4), 0),
                Split(InputDim(2), (3, 4), 1),
                InputDim(3),
                Flatten((InputDim(6), InputDim(7))),
            ),
        )
        self.assertEqual(
            view_groups([2, 3, 4], [2, -1, 4]),
            (InputDim(0), InputDim(1), InputDim(2)),
        )

    def call_dt_test(self, op, args, kwargs, device_mesh: DeviceMesh):
        dim_map = dim_maps[op]
        rules = dim_map(*args, **kwargs)
        outputs = op(*args, **kwargs)
        flat_args = pytree.arg_tree_leaves(*args)
        in_shape = flat_args[0].shape

        no_shard_dims = set()
        for rule in rules:
            if isinstance(rule, Repeat):
                if isinstance(rule.input_dim, InputDim):
                    no_shard_dims.add(rule.input_dim.input_dim)
            elif isinstance(rule, Flatten):
                for dim in rule.input_dims[1:]:
                    if isinstance(dim, InputDim):
                        no_shard_dims.add(dim.input_dim)
            elif isinstance(rule, Split):
                if isinstance(rule.input_dim, Flatten):
                    for dim in rule.input_dim.input_dims[1:]:
                        if isinstance(dim, InputDim):
                            no_shard_dims.add(dim.input_dim)

        if op == torch.unbind:
            no_shard_dims.add(kwargs.get("dim", 0))

        sharding_choices = cast(list[Placement], [Replicate()]) + [
            Shard(i) for i, s in enumerate(in_shape) if s > 1 and i not in no_shard_dims
        ]

        all_sharding_choices = itertools.product(
            *(device_mesh.ndim * [sharding_choices])
        )

        outer_mesh = device_mesh["outer"]
        inner_mesh = device_mesh["inner"]
        inner_mesh_size = inner_mesh.size()
        strided_sharding_choices = [
            (_StridedShard(i, split_factor=inner_mesh_size), Shard(i))
            for i, s in enumerate(in_shape)
            if s > 1 and i not in no_shard_dims
        ]

        for in_shard in itertools.chain(all_sharding_choices, strided_sharding_choices):
            if isinstance(in_shard[0], _StridedShard):
                if op != Tensor.view:
                    continue
                # cannot produce DTensor using ``distribute_tensor()``
                # with ``_StridedShard``. Need to distribute the input
                # over inner mesh dim first, then distribute the
                # _local_tensor over the outer mesh dim.
                in_dt = distribute_tensor(args[0], inner_mesh, (in_shard[1],))
                in_dt = distribute_tensor(
                    in_dt._local_tensor, outer_mesh, (Shard(in_shard[0].dim),)
                )
                in_dt = DTensor.from_local(
                    in_dt._local_tensor,
                    device_mesh,
                    in_shard,
                )
            else:
                in_dt = distribute_tensor(args[0], device_mesh, in_shard)

            comm_mode = CommDebugMode()
            with comm_mode:
                out_dt = op(in_dt, *args[1:], **kwargs)

            self.assertEqual(
                comm_mode.get_total_counts(), 0, "Expected no redistribution."
            )

            full_out = out_dt.full_tensor()

            if dist.get_rank() == 0:
                self.assertEqual(outputs, full_out)

    def dimmap_test(self, op, args, expected_rule_output):
        rules = dim_maps[op](*args)
        self.assertEqual(rules, expected_rule_output)
        self.call_dt_test(op, args, {}, self.device_mesh)

    def test_illegal_views(self):
        device_mesh = self.build_device_mesh()
        # 1D mesh [6] (see above)
        tensor = torch.randn((6, 256))
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        shard = dtensor.redistribute(device_mesh=device_mesh, placements=[Shard(dim=0)])
        # view should be legal, since sharding is even and flatten includes only one sharded dim
        shard.view(-1)

        shard = dtensor.redistribute(device_mesh=device_mesh, placements=[Shard(dim=1)])
        # Shard(1) on the last flattened dim with uneven sharding is now allowed
        # (produces _StridedShard output)
        shard.view(-1)

        # 8 is the uneven case since mesh dim is 6
        tensor = torch.randn((8, 256))
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        shard = dtensor.redistribute(device_mesh=device_mesh, placements=[Shard(dim=0)])
        with self.assertRaisesRegex(RuntimeError, "Sharding propagation failed"):
            shard.view(-1)

        # assuming world size is 4+, tensor is shardable on dim 1 with size 256
        # but not viewable when the resulting dim 1 has size 2
        tensor = torch.randn((8, 256))
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        shard = dtensor.redistribute(device_mesh=device_mesh, placements=[Shard(dim=1)])
        with self.assertRaisesRegex(RuntimeError, "Sharding propagation failed"):
            shard.view(8, 2, -1)

    def test_double_shard_split_validation(self):
        """[Shard(0), Shard(0)] through Split correctly validates divisibility."""
        # Compatible: reshape (24,)→(6,4) with [Shard(0), Shard(0)] on mesh (2,3)
        # submesh_size = 2*3 = 6, split_id=0 out_size=6, 6%6==0 → passes
        result = propagate_shape_and_sharding(
            [Shard(0), Shard(0)],
            (24,),
            dim_maps[torch.Tensor.view](torch.empty(24), [6, 4]),
            (2, 3),
        )
        self.assertEqual(len(result), 2)
        input_tgt, output = result
        self.assertEqual(list(input_tgt), [Shard(0), Shard(0)])
        self.assertEqual(list(output), [Shard(0), Shard(0)])

        # Incompatible: reshape (12,)→(3,4) with [Shard(0), Shard(0)] on mesh (2,3)
        # out_size=3 is not divisible by mesh dim 0 size=2 → error
        with self.assertRaisesRegex(
            RuntimeError, "not evenly divisible by mesh dimension"
        ):
            propagate_shape_and_sharding(
                [Shard(0), Shard(0)],
                (12,),
                dim_maps[torch.Tensor.view](torch.empty(12), [3, 4]),
                (2, 3),
                strict_view=True,
            )

    def test_view_ops(self):
        mesh_shape = (dist.get_world_size() // 2, 2)
        self.device_mesh = init_device_mesh(
            self.device_type, mesh_shape=mesh_shape, mesh_dim_names=("outer", "inner")
        )
        self.dimmap_test(torch.atleast_1d, (randn(()),), (Singleton(),))
        self.dimmap_test(torch.atleast_1d, (randn(24),), (InputDim(0),))
        self.dimmap_test(torch.atleast_1d, (randn(24, 36),), (InputDim(0), InputDim(1)))

        self.dimmap_test(torch.atleast_2d, (randn(()),), (Singleton(), Singleton()))
        self.dimmap_test(torch.atleast_2d, (randn(24),), (Singleton(), InputDim(0)))
        self.dimmap_test(torch.atleast_2d, (randn(24, 36),), (InputDim(0), InputDim(1)))
        self.dimmap_test(
            torch.atleast_2d,
            (randn(24, 36, 48),),
            (InputDim(0), InputDim(1), InputDim(2)),
        )

        self.dimmap_test(
            torch.atleast_3d,
            (randn(()),),
            (Singleton(), Singleton(), Singleton()),
        )
        self.dimmap_test(
            torch.atleast_3d,
            (randn(24),),
            (Singleton(), InputDim(0), Singleton()),
        )
        self.dimmap_test(
            torch.atleast_3d,
            (randn(24, 36),),
            (InputDim(0), InputDim(1), Singleton()),
        )
        self.dimmap_test(
            torch.atleast_3d,
            (randn(24, 36, 42),),
            (InputDim(0), InputDim(1), InputDim(2)),
        )
        self.dimmap_test(
            torch.atleast_3d,
            (randn(24, 36, 42, 24),),
            (InputDim(0), InputDim(1), InputDim(2), InputDim(3)),
        )

        with self.assertRaises(AssertionError):
            dim_maps[torch.broadcast_to](randn(24, 36), (1, 2, 4))

        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 36), (1, 24, 36)),
            (Singleton(), InputDim(0), InputDim(1)),
        )
        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 36), (42, 24, 36)),
            (Broadcast(Singleton(), 42), InputDim(0), InputDim(1)),
        )
        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 1, 36), (12, 24, 24, 36)),
            (
                Broadcast(Singleton(), 12),
                InputDim(0),
                Broadcast(InputDim(1), 24),
                InputDim(2),
            ),
        )
        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 36), (-1, 36)),
            (InputDim(0), InputDim(1)),
        )
        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 1, 36), (-1, 1, 36)),
            (InputDim(0), InputDim(1), InputDim(2)),
        )

        self.dimmap_test(
            torch.broadcast_to,
            (randn(36, 1, 24), (12, 36, 42, 24)),
            (
                Broadcast(Singleton(), 12),
                InputDim(0),
                Broadcast(InputDim(1), 42),
                InputDim(2),
            ),
        )

        self.dimmap_test(
            Tensor.expand,
            (randn(24, 1, 36, 1), 36, 24, 42, -1, 24),
            (
                Broadcast(Singleton(), 36),
                InputDim(0),
                Broadcast(InputDim(1), 42),
                InputDim(2),
                Broadcast(InputDim(3), 24),
            ),
        )

        self.dimmap_test(
            Tensor.expand,
            (randn(24, 1, 36, 1), (36, 24, 42, -1, 24)),
            (
                Broadcast(Singleton(), 36),
                InputDim(0),
                Broadcast(InputDim(1), 42),
                InputDim(2),
                Broadcast(InputDim(3), 24),
            ),
        )

        self.dimmap_test(
            torch.flatten,
            (randn(24, 36),),
            (Flatten((InputDim(0), InputDim(1))),),
        )
        self.dimmap_test(torch.flatten, (randn(42),), (InputDim(0),))
        self.dimmap_test(torch.flatten, (randn(()),), (Singleton(),))

        self.dimmap_test(
            torch.movedim,
            (randn(12, 24, 48, 96), 1, 2),
            (InputDim(0), InputDim(2), InputDim(1), InputDim(3)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(6, 12, 24), 1, 0),
            (InputDim(1), InputDim(0), InputDim(2)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(24, 12, 6), (1, 2), (0, 1)),
            (InputDim(1), InputDim(2), InputDim(0)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(24, 6, 12), (0, 2, 1), (2, 1, 0)),
            (InputDim(1), InputDim(2), InputDim(0)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(24, 12), (1, 0), (0, 1)),
            (InputDim(1), InputDim(0)),
        )

        self.dimmap_test(
            torch.movedim,
            (randn(36, 24, 12), (1, 2), (0, 1)),
            (InputDim(1), InputDim(2), InputDim(0)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(36, 24, 12), (1, 2), (-3, -2)),
            (InputDim(1), InputDim(2), InputDim(0)),
        )

        self.dimmap_test(
            torch.permute,
            (randn(24, 36, 42), (2, 0, 1)),
            (InputDim(2), InputDim(0), InputDim(1)),
        )
        self.dimmap_test(
            torch.permute,
            (randn(24, 36, 42), (-1, -3, -2)),
            (InputDim(2), InputDim(0), InputDim(1)),
        )

        self.dimmap_test(
            torch.ravel,
            (randn(24, 36),),
            (Flatten((InputDim(0), InputDim(1))),),
        )
        self.dimmap_test(torch.ravel, (randn(42),), (InputDim(0),))
        self.dimmap_test(torch.ravel, (randn(()),), (Singleton(),))

        self.dimmap_test(
            Tensor.repeat,
            (randn(24, 36), 1, 2, 1, 1, 2),
            (
                Singleton(),
                Broadcast(Singleton(), 2),
                Singleton(),
                InputDim(0),
                Repeat(InputDim(1), 2),
            ),
        )

        self.dimmap_test(
            torch.reshape,
            (randn(6, 12, 24), (72, 24)),
            (Flatten((InputDim(0), InputDim(1))), InputDim(2)),
        )

        self.dimmap_test(
            torch.tile,
            (randn(24, 36), (1, 2, 1, 1, 2)),
            (
                Singleton(),
                Broadcast(Singleton(), 2),
                Singleton(),
                InputDim(0),
                Repeat(InputDim(1), 2),
            ),
        )
        self.dimmap_test(
            torch.tile,
            (randn(42, 24, 36), (1, 3)),
            (InputDim(0), InputDim(1), Repeat(InputDim(2), 3)),
        )

        self.dimmap_test(
            torch.transpose,
            (randn(24, 60, 42, 60), 2, 0),
            (InputDim(2), InputDim(1), InputDim(0), InputDim(3)),
        )
        self.dimmap_test(
            torch.transpose,
            (randn(24, 60, 42, 60), -1, 0),
            (InputDim(3), InputDim(1), InputDim(2), InputDim(0)),
        )

        self.dimmap_test(
            torch.unsqueeze,
            (randn(42, 24, 36), 1),
            (InputDim(0), Singleton(), InputDim(1), InputDim(2)),
        )
        self.dimmap_test(
            Tensor.view,
            (randn(6, 12, 24), 72, 24),
            (Flatten((InputDim(0), InputDim(1))), InputDim(2)),
        )

        self.dimmap_test(Tensor.view, (randn(1, 1, 12), -1), (InputDim(2),))

        self.dimmap_test(
            Tensor.view,
            (randn(1, 1, 42, 24), -1),
            (Flatten((InputDim(2), InputDim(3))),),
        )

        self.dimmap_test(
            Tensor.view,
            (randn(1, 1, 42, 1, 24, 1), -1),
            (Flatten((InputDim(2), InputDim(input_dim=3), InputDim(4))),),
        )

        self.dimmap_test(
            Tensor.view,
            (randn(48, 35, 26), (24, 4, 35, 13)),
            (
                Split(
                    Flatten(input_dims=(InputDim(0), InputDim(1), InputDim(2))),
                    group_shape=(24, 4, 35, 13),
                    split_id=0,
                ),
                Split(
                    Flatten(input_dims=(InputDim(0), InputDim(1), InputDim(2))),
                    group_shape=(24, 4, 35, 13),
                    split_id=1,
                ),
                Split(
                    Flatten(input_dims=(InputDim(0), InputDim(1), InputDim(2))),
                    group_shape=(24, 4, 35, 13),
                    split_id=2,
                ),
                Split(
                    Flatten(input_dims=(InputDim(0), InputDim(1), InputDim(2))),
                    group_shape=(24, 4, 35, 13),
                    split_id=3,
                ),
            ),
        )

    # TODO: Currently functional collectives on complex numbers are not fully supported,
    # so we are having a standalone test for view_as_complex and view_as_real combined.
    # Once complex numbers are supported, we can add the following to the dim_map test.
    #
    # self.dimmap_test(
    #     torch.view_as_complex,
    #     (randn(24, 13, 2),),
    #     (
    #         InputDim(0),
    #         Flatten((InputDim(1), InputDim(2))),
    #     ),
    # )
    # self.dimmap_test(
    #     torch.view_as_real,
    #     (torch.randn(24, 13, dtype=torch.cfloat),),
    #     (
    #         InputDim(0),
    #         Split(InputDim(1), (13, 2), 0),
    #         Split(InputDim(1), (13, 2), 1),
    #     ),
    # )

    def test_complex_view_ops(self):
        self.device_mesh = DeviceMesh(
            self.device_type, torch.arange(dist.get_world_size()).view(-1, 2)
        )
        inp = randn(24, 13, 2)
        intermediate = torch.view_as_complex(inp)
        out = torch.view_as_real(intermediate)

        # test dim_map correctness
        expected_view_as_complex_rule = (
            InputDim(0),
            Flatten((InputDim(1), InputDim(2))),
        )
        view_as_complex_rule = dim_maps[torch.view_as_complex](inp)
        self.assertEqual(view_as_complex_rule, expected_view_as_complex_rule)
        expected_view_as_real_rule = (
            InputDim(0),
            Split(InputDim(1), (13, 2), 0),
            Split(InputDim(1), (13, 2), 1),
        )
        view_as_real_rule = dim_maps[torch.view_as_real](intermediate)
        self.assertEqual(view_as_real_rule, expected_view_as_real_rule)

        # test sharded computation correctness
        # NOTE: For the input to torch.view_as_complex, sharding
        #       on the last two dimensions is not supported.
        sharding_choices: list[Placement] = [Replicate(), Shard(0)]
        all_sharding_choices = itertools.product(
            *(self.device_mesh.ndim * [sharding_choices])
        )

        for inp_shard in all_sharding_choices:
            inp_dt = distribute_tensor(inp, self.device_mesh, inp_shard)

            comm_mode = CommDebugMode()
            with comm_mode:
                intermediate_dt = torch.view_as_complex(inp_dt)
                out_dt = torch.view_as_real(intermediate_dt)

            self.assertEqual(
                comm_mode.get_total_counts(), 0, "Expected no redistribution."
            )
            self.assertEqual(out, out_dt.full_tensor())

    def test_view_as_complex_no_pmax_pmin(self):
        """
        view_as_complex converts real tensors to complex. Complex numbers don't
        have a total ordering, so P(max) and P(min) placements are invalid.
        This test verifies that these placements are not propagated.
        """
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        inp = torch.randn(8, 2)

        # Create input with P(sum) - this should work
        inp_dt_psum = DTensor.from_local(
            inp.clone(), device_mesh, [Partial("sum")], run_check=False
        )
        result_psum = torch.view_as_complex(inp_dt_psum)
        # P(sum) should propagate since sum is valid for complex
        self.assertIsInstance(result_psum.placements[0], Partial)
        self.assertEqual(result_psum.placements[0].reduce_op, "sum")

        # Create input with P(max) - should redistribute to Replicate
        inp_dt_pmax = DTensor.from_local(
            inp.clone(), device_mesh, [Partial("max")], run_check=False
        )
        result_pmax = torch.view_as_complex(inp_dt_pmax)
        # P(max) should NOT propagate - output should be Replicate
        self.assertIsInstance(result_pmax.placements[0], Replicate)

    def test_dtensor_view_op_uneven(self):
        """
        When the sharded dimension is unchanged, the view op should not trigger any communication.
        And the behavior should be the same as operating under single-device.

        Test two uneven cases for view op:
            1) the sharded tensor dim is 1 so that only the first rank has an non-empty shard.
            2) the sharded tensor dim is uneven such that some ranks have full shards,
                smaller non-empty shards, and empty shards.
        """
        dim0_sizes = [1, self.world_size + 1]
        for dim0_size in dim0_sizes:
            p = torch.randn(dim0_size, 2, 2, 2)
            mesh = init_device_mesh(self.device_type, (self.world_size,))
            dtensor = distribute_tensor(p, mesh, [Shard(0)])

            with CommDebugMode() as comm_mode:
                view = dtensor.view(dim0_size, 2, 4)
                self.assertEqual(len(comm_mode.get_comm_counts()), 0)
                # when no communication happens, the data pointer should be the same.
                self.assertEqual(
                    view.to_local().data_ptr(), dtensor.to_local().data_ptr()
                )

                view = dtensor.view(dim0_size, 4, 2)
                self.assertEqual(
                    view.to_local().data_ptr(), dtensor.to_local().data_ptr()
                )
                self.assertEqual(len(comm_mode.get_comm_counts()), 0)

                view = dtensor.view(dim0_size, 8)
                self.assertEqual(
                    view.to_local().data_ptr(), dtensor.to_local().data_ptr()
                )
                self.assertEqual(len(comm_mode.get_comm_counts()), 0)

                view = dtensor.view(dtensor.shape)
                self.assertEqual(
                    view.to_local().data_ptr(), dtensor.to_local().data_ptr()
                )
                self.assertEqual(len(comm_mode.get_comm_counts()), 0)

    def _get_viewed_tensor_dims(self, tensor_dims, flatten_start, flatten_end):
        if isinstance(tensor_dims, tuple):
            tensor_dims = list(tensor_dims)
        flatten_dims = tensor_dims[flatten_start:flatten_end]
        if len(flatten_dims) > 0:
            flatten_dim = math.prod(flatten_dims)
        else:
            flatten_dim = None
        leading_dims = tensor_dims[:flatten_start]
        trailing_dims = tensor_dims[flatten_end:]
        view_shapes = []
        if len(leading_dims) > 0:
            view_shapes.extend(leading_dims)
        if flatten_dim is not None:
            view_shapes.append(flatten_dim)
        if len(trailing_dims) > 0:
            view_shapes.extend(trailing_dims)
        return tuple(view_shapes)

    def test_dtensor_flatten_multi_mesh(self):
        """Test flatten operations across 1D and 2D meshes with all placement patterns.

        Iterates over 1D and 2D mesh configurations to test single-shard (S, SR, RS),
        multi-shard (SS), and replicate (R, RR) patterns with even/uneven tensor dim sizes.
        """
        cases = [
            ((6,), [("S",), ("R",)]),
            ((3, 2), [("S", "R"), ("R", "S"), ("S", "S"), ("R", "R")]),
        ]
        for mesh_shape, patterns in cases:
            if self.world_size < math.prod(mesh_shape):
                continue
            mesh = init_device_mesh(self.device_type, mesh_shape)
            mesh_ndim = len(mesh_shape)
            for pattern in patterns:
                shard_mesh_dims = [i for i, p in enumerate(pattern) if p == "S"]
                num_shard = len(shard_mesh_dims)
                for tensor_ndim in [2, 3, 4]:
                    for flatten_start in range(tensor_ndim):
                        for flatten_end in range(flatten_start + 2, tensor_ndim + 1):
                            if num_shard == 1:
                                self._run_flatten_single_shard(
                                    mesh,
                                    mesh_ndim,
                                    shard_mesh_dims[0],
                                    tensor_ndim,
                                    flatten_start,
                                    flatten_end,
                                )
                            elif num_shard == 2:
                                self._run_flatten_ss(
                                    mesh,
                                    tensor_ndim,
                                    flatten_start,
                                    flatten_end,
                                )
                            else:
                                even = 2 * mesh.size(0)
                                dim_vals = [even - 1, even, even + 1]
                                all_dims = list(
                                    itertools.product(dim_vals, repeat=tensor_ndim)
                                )
                                rep_placements = tuple([Replicate()] * mesh_ndim)
                                for tensor_dims in all_dims:
                                    with self.subTest(
                                        dims=tensor_dims,
                                        flat=(flatten_start, flatten_end),
                                        mesh=mesh_shape,
                                    ):
                                        self._test_dtensor_flatten_replicate(
                                            tensor_dims,
                                            flatten_start,
                                            flatten_end,
                                            mesh,
                                            rep_placements,
                                        )

    def _run_flatten_single_shard(
        self,
        mesh,
        mesh_ndim,
        shard_mesh_dim,
        tensor_ndim,
        flatten_start,
        flatten_end,
    ):
        even_val = 2 * mesh.size(shard_mesh_dim)
        dim_vals = [even_val - 1, even_val, even_val + 1]
        all_dims = list(itertools.product(dim_vals, repeat=tensor_ndim))
        for shard_dim in range(flatten_start, flatten_end):
            for tensor_dims in all_dims:
                ctx = contextlib.nullcontext()
                if tensor_dims[shard_dim] % mesh.size(
                    shard_mesh_dim
                ) != 0 and shard_dim != (flatten_end - 1):
                    ctx = self.assertRaisesRegex(
                        RuntimeError,
                        "is not evenly divisible by mesh dimension",
                    )
                with (
                    self.subTest(
                        dims=tensor_dims,
                        shard=shard_dim,
                        mesh_dim=shard_mesh_dim,
                        flat=(flatten_start, flatten_end),
                    ),
                    ctx,
                ):
                    self._test_dtensor_flatten_single_shard(
                        tensor_dims,
                        flatten_start,
                        flatten_end,
                        mesh,
                        shard_dim,
                        shard_mesh_dim,
                    )

    def _run_flatten_ss(self, mesh, tensor_ndim, flatten_start, flatten_end):
        for shard_dim0 in range(flatten_start, flatten_end):
            for shard_dim1 in range(shard_dim0, flatten_end):
                dim0_values = [
                    2 * mesh.size(0) - 1,
                    2 * mesh.size(0),
                    2 * mesh.size(0) + 1,
                ]
                dim1_values = [
                    2 * mesh.size(1) - 1,
                    2 * mesh.size(1),
                    2 * mesh.size(1) + 1,
                ]
                other_dim_value = 2 * mesh.size(0) * mesh.size(1)
                for dim0_val in dim0_values:
                    for dim1_val in dim1_values:
                        tensor_dims = [other_dim_value] * tensor_ndim
                        tensor_dims[shard_dim0] = dim0_val
                        if shard_dim0 != shard_dim1:
                            tensor_dims[shard_dim1] = dim1_val
                        tensor_dims = tuple(tensor_dims)
                        local_tensor_dims = list(tensor_dims)
                        placements = (Shard(shard_dim0), Shard(shard_dim1))
                        ctx = contextlib.nullcontext()
                        if local_tensor_dims[shard_dim0] % mesh.size(0) != 0:
                            ctx = self.assertRaisesRegex(
                                RuntimeError,
                                "is not evenly divisible by mesh dimension",
                            )
                        local_tensor_dims[shard_dim0] = local_tensor_dims[
                            shard_dim0
                        ] // mesh.size(0)
                        if local_tensor_dims[shard_dim1] % mesh.size(
                            1
                        ) != 0 and shard_dim1 != (flatten_end - 1):
                            ctx = self.assertRaisesRegex(
                                RuntimeError,
                                "is not evenly divisible by mesh dimension",
                            )
                        with (
                            self.subTest(
                                dims=tensor_dims,
                                shard0=shard_dim0,
                                shard1=shard_dim1,
                            ),
                            ctx,
                        ):
                            self._test_dtensor_flatten_2d_ss(
                                tensor_dims,
                                flatten_start,
                                flatten_end,
                                mesh,
                                placements,
                            )

    def _test_dtensor_flatten_single_shard(
        self,
        tensor_dims,
        flatten_start,
        flatten_end,
        mesh,
        shard_dim,
        shard_mesh_dim,
    ):
        mesh_ndim = mesh.ndim
        placements = tuple(
            Shard(shard_dim) if i == shard_mesh_dim else Replicate()
            for i in range(mesh_ndim)
        )
        nelem = math.prod(tensor_dims)
        global_inps: Tensor = torch.arange(nelem).view(tensor_dims)
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)
        viewed_tensor_dims = self._get_viewed_tensor_dims(
            tensor_dims, flatten_start, flatten_end
        )
        comm_mode = CommDebugMode()
        with comm_mode:
            inps_viewed = inps.view(viewed_tensor_dims)
        if shard_dim == flatten_start:
            expected_placement = Shard(flatten_start)
        else:
            split_factor = math.prod(tensor_dims[flatten_start:shard_dim])
            self.assertGreater(split_factor, 1)
            expected_placement = _StridedShard(
                dim=flatten_start, split_factor=split_factor
            )
        expected_placements = tuple(
            expected_placement if i == shard_mesh_dim else Replicate()
            for i in range(mesh_ndim)
        )
        expected_local_tensor = distribute_tensor(
            global_inps.view(viewed_tensor_dims),
            mesh,
            expected_placements,
            src_data_rank=None,
        )._local_tensor
        self.assertEqual(inps_viewed.placements, expected_placements)
        self.assertEqual(inps_viewed._local_tensor, expected_local_tensor)
        self.assertEqual(comm_mode.get_total_counts(), 0)

    def _test_dtensor_flatten_replicate(
        self, tensor_dims, flatten_start, flatten_end, mesh, placements
    ):
        mesh_ndim = len(placements)
        nelem = math.prod(tensor_dims)
        global_inps: Tensor = torch.arange(nelem).view(tensor_dims)
        global_inps_replicate: DTensor = distribute_tensor(
            global_inps, mesh, tuple([Replicate()] * mesh_ndim)
        )
        inps = global_inps_replicate.redistribute(mesh, placements)
        viewed_tensor_dims = self._get_viewed_tensor_dims(
            tensor_dims, flatten_start, flatten_end
        )
        comm_mode = CommDebugMode()
        with comm_mode:
            inps_viewed = inps.view(viewed_tensor_dims)
        expected_placements = tuple([Replicate()] * mesh_ndim)
        expected_local_tensor = (
            distribute_tensor(
                global_inps.view(viewed_tensor_dims),
                mesh,
                tuple([Replicate()] * mesh_ndim),
            )
            .redistribute(mesh, expected_placements)
            ._local_tensor
        )
        self.assertEqual(inps_viewed.placements, expected_placements)
        self.assertEqual(inps_viewed._local_tensor, expected_local_tensor)
        self.assertEqual(comm_mode.get_total_counts(), 0)

    def _get_expected_placements_ss(
        self,
        tensor_dims,
        flatten_start,
        flatten_end,
        mesh,
        placements,
    ):
        local_tensor_dims = list(tensor_dims)
        expected_placements = []
        for idx, placement in enumerate(placements):
            self.assertIsInstance(placement, Shard)
            shard_dim = placement.dim
            if shard_dim == flatten_start:
                # S(flatten_start), S(flatten_start) qualifies
                expected_placement = placement
            else:
                split_factor = math.prod(local_tensor_dims[flatten_start:shard_dim])
                self.assertGreater(split_factor, 1)
                expected_placement = _StridedShard(
                    dim=flatten_start, split_factor=split_factor
                )
            if local_tensor_dims[shard_dim] % mesh.size(idx) != 0:
                # uneven shard on last flattened dim is supported
                self.assertTrue(
                    _ViewShardingPropagator._is_last_shard_in_flatten_range(
                        idx, placements, flatten_start, flatten_end
                    )
                )
                local_tensor_dims[shard_dim] = math.ceil(
                    local_tensor_dims[shard_dim] * 1.0 / mesh.size(idx)
                )
            else:
                local_tensor_dims[shard_dim] = local_tensor_dims[
                    shard_dim
                ] // mesh.size(idx)
            expected_placements.append(expected_placement)
        return tuple(expected_placements)

    def _test_dtensor_flatten_2d_ss(
        self,
        tensor_dims,
        flatten_start,
        flatten_end,
        mesh,
        placements,
    ):
        nelem = math.prod(tensor_dims)
        global_inps: Tensor = torch.arange(nelem).view(tensor_dims)
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)
        viewed_tensor_dims = self._get_viewed_tensor_dims(
            tensor_dims, flatten_start, flatten_end
        )

        comm_mode = CommDebugMode()
        with comm_mode:
            inps_viewed = inps.view(viewed_tensor_dims)

        expected_placements = self._get_expected_placements_ss(
            tensor_dims,
            flatten_start,
            flatten_end,
            mesh,
            placements,
        )

        expected_local_tensor = distribute_tensor(
            global_inps.view(viewed_tensor_dims),
            mesh,
            expected_placements,
            src_data_rank=None,
        )._local_tensor
        self.assertEqual(inps_viewed.placements, expected_placements)
        self.assertEqual(inps_viewed._local_tensor, expected_local_tensor)
        self.assertEqual(comm_mode.get_total_counts(), 0)

    def test_dtensor_flatten_shard_outside_range(self):
        """Test that Shard on a dim outside the flatten range passes through correctly.

        When flattening dims [flatten_start, flatten_end), a Shard on a dim outside
        this range should be preserved with adjusted dim index:
        - Shard before range: dim unchanged
        - Shard after range: dim shifts by -(flatten_end - flatten_start - 1)
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        # Use sizes divisible by mesh size to avoid uneven-shard complications
        dim_size = mesh.size(0) * 2
        test_cases = [
            # (tensor_dims, flatten_start, flatten_end, shard_dim, expected_shard_dim)
            # Shard before range
            ([dim_size] * 3, 1, 3, 0, 0),
            ([dim_size] * 4, 1, 3, 0, 0),
            ([dim_size] * 4, 2, 4, 0, 0),
            ([dim_size] * 4, 2, 4, 1, 1),
            # Shard after range
            ([dim_size] * 4, 0, 2, 2, 1),
            ([dim_size] * 4, 0, 2, 3, 2),
            ([dim_size] * 4, 0, 3, 3, 1),
            ([dim_size] * 4, 1, 3, 3, 2),
        ]

        for (
            tensor_dims,
            flatten_start,
            flatten_end,
            shard_dim,
            expected_shard_dim,
        ) in test_cases:
            with self.subTest(
                dims=tensor_dims, shard=shard_dim, flat=(flatten_start, flatten_end)
            ):
                placements = (Shard(shard_dim),)
                nelem = math.prod(tensor_dims)
                global_inps = torch.arange(nelem).view(tensor_dims)
                dt = distribute_tensor(
                    global_inps, mesh, placements, src_data_rank=None
                )

                flat_dims = self._get_viewed_tensor_dims(
                    tensor_dims, flatten_start, flatten_end
                )
                comm_mode = CommDebugMode()
                with comm_mode:
                    dt_flat = dt.view(flat_dims)

                expected_placements = (Shard(expected_shard_dim),)
                self.assertEqual(dt_flat.placements, expected_placements)
                expected_local = distribute_tensor(
                    global_inps.view(flat_dims),
                    mesh,
                    expected_placements,
                    src_data_rank=None,
                )._local_tensor
                self.assertEqual(dt_flat._local_tensor, expected_local)
                self.assertEqual(comm_mode.get_total_counts(), 0)

        # 2D mesh: test shard outside range with (Shard, Replicate) and (Replicate, Shard)
        mesh_2d = init_device_mesh(self.device_type, (3, self.world_size // 3))
        dim_size_2d = mesh_2d.size(0) * mesh_2d.size(1) * 2
        test_cases_2d = [
            # (tensor_dims, flatten_start, flatten_end, shard_dim)
            ([dim_size_2d] * 4, 0, 2, 2),  # shard after range
            ([dim_size_2d] * 4, 0, 2, 3),  # shard after range
            ([dim_size_2d] * 4, 1, 3, 0),  # shard before range
            ([dim_size_2d] * 4, 2, 4, 0),  # shard before range
            ([dim_size_2d] * 4, 2, 4, 1),  # shard right before range
        ]
        for tensor_dims, flatten_start, flatten_end, shard_dim in test_cases_2d:
            num_merged = flatten_end - flatten_start - 1
            expected_shard_dim = (
                shard_dim if shard_dim < flatten_start else shard_dim - num_merged
            )
            for mesh_dim_idx in range(2):
                with self.subTest(
                    dims=tensor_dims, shard=shard_dim, mesh_dim=mesh_dim_idx
                ):
                    placements = tuple(
                        Shard(shard_dim) if i == mesh_dim_idx else Replicate()
                        for i in range(2)
                    )
                    nelem = math.prod(tensor_dims)
                    global_inps = torch.arange(nelem).view(tensor_dims)
                    dt = distribute_tensor(
                        global_inps, mesh_2d, placements, src_data_rank=None
                    )

                    flat_dims = self._get_viewed_tensor_dims(
                        tensor_dims, flatten_start, flatten_end
                    )
                    comm_mode = CommDebugMode()
                    with comm_mode:
                        dt_flat = dt.view(flat_dims)

                    expected_placements = tuple(
                        Shard(expected_shard_dim) if i == mesh_dim_idx else Replicate()
                        for i in range(2)
                    )
                    self.assertEqual(dt_flat.placements, expected_placements)
                    expected_local = distribute_tensor(
                        global_inps.view(flat_dims),
                        mesh_2d,
                        expected_placements,
                        src_data_rank=None,
                    )._local_tensor
                    self.assertEqual(dt_flat._local_tensor, expected_local)
                    self.assertEqual(comm_mode.get_total_counts(), 0)

    def test_dtensor_flatten_unflatten_roundtrip(self):
        """Flatten then unflatten should recover the original placements and data.

        Tests the full round-trip: DTensor -> flatten -> unflatten -> compare with original.
        Covers sharding on dims before, within, and after the flatten range.
        Also tests unflatten -> flatten direction.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        dim_size = mesh.size(0) * 2  # evenly divisible

        # flatten -> unflatten round-trip
        test_cases = [
            # (tensor_dims, flatten_start, flatten_end, shard_dim)
            # Shard on first of flattened dims
            ([dim_size, dim_size], 0, 2, 0),
            # Shard on second of flattened dims
            ([dim_size, dim_size], 0, 2, 1),
            # 3D: shard in range (leading)
            ([dim_size, dim_size, dim_size], 0, 2, 0),
            # 3D: shard in range (second)
            ([dim_size, dim_size, dim_size], 0, 2, 1),
            # 3D: shard outside range (after)
            ([dim_size, dim_size, dim_size], 0, 2, 2),
            # 3D: shard before range
            ([dim_size, dim_size, dim_size], 1, 3, 0),
            # 3D: shard in range (first of second pair)
            ([dim_size, dim_size, dim_size], 1, 3, 1),
            # 3D: shard in range (second of second pair)
            ([dim_size, dim_size, dim_size], 1, 3, 2),
            # 4D: shard first of 3 flattened
            ([dim_size] * 4, 0, 3, 0),
            # 4D: shard last of 3 flattened
            ([dim_size] * 4, 0, 3, 2),
            # 4D: shard outside range (after)
            ([dim_size] * 4, 0, 3, 3),
            # 4D: shard before range
            ([dim_size] * 4, 1, 3, 0),
            # 4D: shard after range
            ([dim_size] * 4, 1, 3, 3),
        ]

        for tensor_dims, flatten_start, flatten_end, shard_dim in test_cases:
            with self.subTest(dims=tensor_dims, shard=shard_dim, direction="flatten"):
                placements = (Shard(shard_dim),)
                nelem = math.prod(tensor_dims)
                global_inps = torch.arange(nelem).view(tensor_dims)
                dt = distribute_tensor(
                    global_inps, mesh, placements, src_data_rank=None
                )

                # Flatten
                flat_dims = self._get_viewed_tensor_dims(
                    tensor_dims, flatten_start, flatten_end
                )
                dt_flat = dt.view(flat_dims)

                # Unflatten back
                dt_roundtrip = dt_flat.view(tensor_dims)

                self.assertEqual(dt_roundtrip.placements, placements)
                self.assertEqual(dt_roundtrip._local_tensor, dt._local_tensor)

        # unflatten -> flatten round-trip
        # Start with a flattened tensor, unflatten it, then flatten back
        for tensor_dims, flatten_start, flatten_end, shard_dim in test_cases:
            with self.subTest(dims=tensor_dims, shard=shard_dim, direction="unflatten"):
                flat_dims = self._get_viewed_tensor_dims(
                    tensor_dims, flatten_start, flatten_end
                )
                # Compute the shard dim in the flattened view
                if shard_dim < flatten_start:
                    flat_shard_dim = shard_dim
                elif shard_dim < flatten_end:
                    flat_shard_dim = (
                        flatten_start  # all flattened dims merge to flatten_start
                    )
                else:
                    flat_shard_dim = shard_dim - (flatten_end - flatten_start - 1)

                placements = (Shard(flat_shard_dim),)
                nelem = math.prod(flat_dims)
                global_inps = torch.arange(nelem).view(flat_dims)
                dt = distribute_tensor(
                    global_inps, mesh, placements, src_data_rank=None
                )

                # Unflatten
                dt_unflat = dt.view(tensor_dims)

                # Flatten back
                dt_roundtrip = dt_unflat.view(flat_dims)

                self.assertEqual(dt_roundtrip.placements, placements)
                self.assertEqual(dt_roundtrip._local_tensor, dt._local_tensor)

    def generate_tensor_dims_1d(
        self, tensor_ndim, flatten_start, flatten_end, shard_dim, mesh
    ):
        tensor_dims_unflatten = [2 * mesh.size(0) + 1] * tensor_ndim
        for tensor_dim in [
            2 * mesh.size(0) - 1,
            2 * mesh.size(0),
            2 * mesh.size(0) + 1,
        ]:
            tensor_dims_unflatten[shard_dim] = tensor_dim
            local_tensor_dims_unflatten = list(tensor_dims_unflatten)
            local_tensor_dims_unflatten[shard_dim] = math.ceil(
                tensor_dim * 1.0 / mesh.size(0)
            )
            nelem_flatten = math.prod(tensor_dims_unflatten[flatten_start:flatten_end])
            tensor_dims_flatten = (
                tensor_dims_unflatten[0:flatten_start]
                + [nelem_flatten]
                + tensor_dims_unflatten[flatten_end:]
            )
            yield (
                tensor_dims_unflatten,
                local_tensor_dims_unflatten,
                tensor_dims_flatten,
            )

    def generate_tensor_dims_1d_after_flatten(
        self, tensor_ndim, unflatten_dim, shard_dim, mesh
    ):
        tensor_dims = [mesh.size(0) * mesh.size(0)] * tensor_ndim
        for unflatten_dim_value in [
            mesh.size(0) * mesh.size(0) - 1,
            mesh.size(0) * mesh.size(0),
            mesh.size(0) * mesh.size(0) + 1,
        ]:
            for shard_dim_value in [
                mesh.size(0) * mesh.size(0) - 1,
                mesh.size(0) * mesh.size(0),
                mesh.size(0) * mesh.size(0) + 1,
            ]:
                tensor_dims[unflatten_dim] = unflatten_dim_value
                tensor_dims[shard_dim] = shard_dim_value
                yield list(tensor_dims)

    def test_dtensor_unflatten_1d(self):
        mesh: DeviceMesh = init_device_mesh(self.device_type, (self.world_size,))

        # flatten -> view -> unflatten
        for tensor_ndim in [2, 3, 4]:
            for flatten_start in range(tensor_ndim):
                for flatten_end in range(flatten_start + 2, tensor_ndim + 1):
                    for shard_dim in range(flatten_start, flatten_end):
                        expected_placements = (Shard(shard_dim),)
                        for (
                            tensor_dims_unflatten,
                            local_tensor_dims_unflatten,
                            tensor_dims_flatten,
                        ) in self.generate_tensor_dims_1d(
                            tensor_ndim, flatten_start, flatten_end, shard_dim, mesh
                        ):
                            ctx = contextlib.nullcontext()
                            is_uneven = (
                                tensor_dims_unflatten[shard_dim] % mesh.size(0) != 0
                            )
                            is_last_dim = shard_dim == flatten_end - 1
                            if is_uneven and not is_last_dim:
                                ctx = self.assertRaisesRegex(
                                    RuntimeError,
                                    "is not evenly divisible by mesh dimension",
                                )
                            with (
                                self.subTest(
                                    dims=tensor_dims_unflatten,
                                    shard=shard_dim,
                                    flat=(flatten_start, flatten_end),
                                ),
                                ctx,
                            ):
                                self._test_dtensor_unflatten_1d_shard(
                                    tensor_dims_unflatten,
                                    local_tensor_dims_unflatten,
                                    tensor_dims_flatten,
                                    flatten_start,
                                    expected_placements,
                                    mesh,
                                )

        # any factoring on unflatten_dim
        for tensor_ndim in [1, 2, 3, 4]:
            for unflatten_dim in range(tensor_ndim):
                for shard_dim in range(tensor_ndim):
                    for tensor_dims in self.generate_tensor_dims_1d_after_flatten(
                        tensor_ndim, unflatten_dim, shard_dim, mesh
                    ):
                        with self.subTest(
                            dims=tensor_dims, shard=shard_dim, unflatten=unflatten_dim
                        ):
                            placements = (Shard(shard_dim),)
                            self._test_dtensor_unflatten_1d_shard_arbitrary(
                                tensor_dims,
                                unflatten_dim,
                                placements,
                                mesh,
                            )

        # Replicate: unflatten should preserve Replicate placement
        for tensor_ndim in [1, 2, 3, 4]:
            for unflatten_dim in range(tensor_ndim):
                tensor_dims = [6] * tensor_ndim
                tensor_dims[unflatten_dim] = 12  # will unflatten to (3, 4)
                global_tensor = torch.arange(math.prod(tensor_dims)).view(tensor_dims)
                dt = distribute_tensor(
                    global_tensor, mesh, (Replicate(),), src_data_rank=None
                )
                # Unflatten dimension to (3, 4)
                unflatten_shape = list(tensor_dims)
                unflatten_shape[unflatten_dim : unflatten_dim + 1] = [3, 4]
                dt_unflattened = dt.view(unflatten_shape)
                self.assertEqual(dt_unflattened.placements, (Replicate(),))
                self.assertEqual(dt_unflattened.shape, torch.Size(unflatten_shape))

    def _test_dtensor_unflatten_1d_shard(
        self,
        tensor_dims_unflatten,
        local_tensor_dims_unflatten,
        tensor_dims_flatten,
        flatten_start,
        expected_placements,
        mesh,
    ):
        shard_dim = expected_placements[0].dim
        split_factor = math.prod(local_tensor_dims_unflatten[flatten_start:shard_dim])
        if split_factor == 1:
            self.assertEqual(shard_dim, flatten_start)
            placements = (Shard(flatten_start),)
        else:
            placements = (_StridedShard(flatten_start, split_factor=split_factor),)
        nelem = math.prod(tensor_dims_flatten)
        global_inps = torch.arange(nelem).view(tensor_dims_flatten)
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)
        inps_viewed = inps.view(tensor_dims_unflatten)
        self.assertEqual(inps_viewed.placements, expected_placements)
        expected_local_tensor = distribute_tensor(
            torch.arange(nelem).view(tensor_dims_unflatten),
            mesh,
            expected_placements,
            src_data_rank=None,
        )._local_tensor
        self.assertEqual(inps_viewed._local_tensor, expected_local_tensor)

    def _test_dtensor_unflatten_1d_shard_arbitrary(
        self, tensor_dims, unflatten_dim, placements, mesh
    ):
        shard_dim = placements[0].dim
        self.assertIsInstance(placements[0], Shard)
        self.assertNotIsInstance(placements[0], _StridedShard)

        tensor_dims = list(tensor_dims)  # Make a mutable copy

        # Get all non-trivial factorizations of unflatten_dim size
        unflatten_size = tensor_dims[unflatten_dim]
        factorizations = self._get_all_factorizations(unflatten_size)

        # Skip if no non-trivial factorization exists (prime number)
        if not factorizations:
            return

        # Create the global tensor once (reused for all factorizations)
        nelem = math.prod(tensor_dims)
        global_inps = torch.arange(nelem).view(tensor_dims)

        # Distribute the tensor
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)

        # Test each factorization (can be any length >= 2)
        for factors in factorizations:
            tensor_dims_unflatten = (
                tensor_dims[:unflatten_dim]
                + list(factors)
                + tensor_dims[unflatten_dim + 1 :]
            )

            # Determine if we expect an error
            first_factor = factors[0]

            ctx = contextlib.nullcontext()
            expect_error = False

            if shard_dim == unflatten_dim:
                # When unflattening the sharded dimension, check factor alignment
                uneven_shard = tensor_dims[shard_dim] % mesh.size(0) != 0
                first_factor_aligned = first_factor % mesh.size(0) == 0

                expect_error = not first_factor_aligned or uneven_shard

            if expect_error:
                ctx = self.assertRaisesRegex(
                    RuntimeError, "is not evenly divisible by mesh dimension"
                )

            with ctx:
                comm_mode = CommDebugMode()
                with comm_mode:
                    inps_viewed = inps.view(tensor_dims_unflatten)

                # Number of new dimensions added by unflatten
                num_new_dims = len(factors) - 1

                # Compute expected placements after unflatten
                if shard_dim < unflatten_dim:
                    expected_placements = (Shard(shard_dim),)
                elif shard_dim == unflatten_dim:
                    expected_placements = (Shard(unflatten_dim),)
                else:
                    expected_placements = (Shard(shard_dim + num_new_dims),)

                self.assertEqual(inps_viewed.placements, expected_placements)
                self.assertEqual(comm_mode.get_total_counts(), 0)

                expected_local = distribute_tensor(
                    global_inps.view(tensor_dims_unflatten),
                    mesh,
                    expected_placements,
                    src_data_rank=None,
                )._local_tensor
                self.assertEqual(inps_viewed._local_tensor, expected_local)

    def test_dtensor_unflatten_multi_mesh(self):
        """Test unflatten across 2D and 3D meshes with all placement patterns.

        Iterates over 2D and 3D mesh configurations to test multi-shard (SS, SSS),
        mixed (R+SS, RR+SS), and replicate (RR, RRR) patterns.
        """
        # (mesh_shape, num_replicate_dims)
        cases = [
            ((3, 2), 0),  # (SS, SS)
            ((3, 2), 1),  # (R, SS)
            ((3, 2), 2),  # (R, R)
            ((2, 2, 2), 0),  # (SS, SS, SS)
            ((2, 2, 2), 1),  # (R, SS, SS)
            ((2, 2, 2), 2),  # (R, R, SS)
            ((2, 2, 2), 3),  # (R, R, R)
        ]
        for mesh_shape, num_rep in cases:
            if self.world_size < math.prod(mesh_shape):
                continue
            mesh = init_device_mesh(self.device_type, mesh_shape)
            mesh_ndim = len(mesh_shape)
            num_shard = mesh_ndim - num_rep
            flattened_size = math.prod(mesh_shape) * 12
            factorizations = self._get_all_factorizations(flattened_size)

            if num_shard == 0:
                # All-replicate: test that Replicate is preserved
                for factors in factorizations:
                    for prefix_size in [0, 1, 2]:
                        with self.subTest(
                            factors=factors,
                            prefix=prefix_size,
                            mesh=mesh_shape,
                        ):
                            tensor_dims_unflatten = (
                                [6] * prefix_size + list(factors) + [3]
                            )
                            nelem_flatten = math.prod(factors)
                            tensor_dims_flatten = (
                                [6] * prefix_size + [nelem_flatten] + [3]
                            )
                            nelem = math.prod(tensor_dims_flatten)
                            global_tensor = torch.arange(nelem).view(
                                tensor_dims_flatten
                            )
                            rep_placements = tuple([Replicate()] * mesh_ndim)
                            dt = distribute_tensor(
                                global_tensor,
                                mesh,
                                rep_placements,
                                src_data_rank=None,
                            )
                            dt_unflattened = dt.view(tensor_dims_unflatten)
                            self.assertEqual(
                                dt_unflattened.placements,
                                rep_placements,
                            )
                            self.assertEqual(
                                dt_unflattened.shape,
                                torch.Size(tensor_dims_unflatten),
                            )
            else:
                # Sharded cases: pick shard indices from factorizations
                min_factors = num_shard + 1  # need prefix + num_shard shard dims
                valid = [f for f in factorizations if len(f) >= min_factors]
                for factors in valid:
                    n = len(factors)
                    for shard_indices in itertools.combinations(
                        range(1, n),
                        num_shard,
                    ):
                        # Error detection: uneven on non-(last-shard AND last-factor)
                        expect_error = False
                        for rank, si in enumerate(shard_indices):
                            mesh_dim = num_rep + rank
                            is_last_shard = rank == num_shard - 1
                            is_last_factor = si == n - 1
                            if factors[si] % mesh.size(mesh_dim) != 0 and not (
                                is_last_shard and is_last_factor
                            ):
                                expect_error = True
                                break
                        with self.subTest(
                            factors=factors,
                            shard=shard_indices,
                            mesh=mesh_shape,
                        ):
                            if expect_error:
                                with self.assertRaisesRegex(
                                    RuntimeError,
                                    "is not evenly divisible by mesh dimension",
                                ):
                                    self._test_dtensor_unflatten_factors(
                                        factors,
                                        shard_indices,
                                        num_rep,
                                        mesh,
                                    )
                            else:
                                self._test_dtensor_unflatten_factors(
                                    factors,
                                    shard_indices,
                                    num_rep,
                                    mesh,
                                )

    def _test_dtensor_unflatten_factors(
        self,
        factors,
        shard_indices,
        num_rep,
        mesh,
    ):
        """Test unflatten with _StridedShard placements for any mesh dimensionality.

        Args:
            factors: Tuple of factors representing the unflatten shape
            shard_indices: Tuple of indices within factors for each sharded mesh dim
            num_rep: Number of leading Replicate mesh dims
            mesh: DeviceMesh
        """
        num_shard = len(shard_indices)
        # SS cases need a suffix dim outside the flatten range
        suffix = [3] if num_shard >= 2 else []
        tensor_dims_unflatten = list(factors) + suffix
        nelem_flatten = math.prod(factors)
        tensor_dims_flatten = [nelem_flatten] + suffix

        # Build input placements and expected output placements
        placements = []
        expected_placements = []
        for i in range(mesh.ndim):
            if i < num_rep:
                placements.append(Replicate())
                expected_placements.append(Replicate())
            else:
                rank = i - num_rep
                si = shard_indices[rank]
                # split_factor = prod(factors[:si]) / prod(earlier mesh dims)
                sf = math.prod(factors[:si])
                for j in range(rank):
                    sf //= mesh.size(num_rep + j)
                placements.append(_StridedShard(0, split_factor=sf))
                expected_placements.append(Shard(si))

        placements = tuple(placements)
        expected_placements = tuple(expected_placements)

        nelem = math.prod(tensor_dims_flatten)
        global_inps = torch.arange(nelem).view(tensor_dims_flatten)
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)

        comm_mode = CommDebugMode()
        with comm_mode:
            inps_viewed = inps.view(tensor_dims_unflatten)

        self.assertEqual(inps_viewed.placements, expected_placements)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        expected_local_tensor = distribute_tensor(
            torch.arange(nelem).view(tensor_dims_unflatten),
            mesh,
            expected_placements,
            src_data_rank=None,
        )._local_tensor
        self.assertEqual(inps_viewed._local_tensor, expected_local_tensor)

    def test_dtensor_flatten_unflatten_2d_reversed_mesh(self):
        """Test flatten/unflatten with reversed mesh shape (2, 3) to catch ordering bugs."""
        self.assertEqual(self.world_size, 6)
        mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        dim_size = mesh.size(0) * mesh.size(1) * 2  # divisible by both mesh dims

        # Flatten with (S, S) pattern
        tensor_ndim = 3
        for flatten_start in range(tensor_ndim):
            for flatten_end in range(flatten_start + 2, tensor_ndim + 1):
                for shard_dim0 in range(flatten_start, flatten_end):
                    for shard_dim1 in range(shard_dim0, flatten_end):
                        with self.subTest(
                            shard0=shard_dim0,
                            shard1=shard_dim1,
                            flat=(flatten_start, flatten_end),
                        ):
                            tensor_dims = tuple([dim_size] * tensor_ndim)
                            placements = (Shard(shard_dim0), Shard(shard_dim1))
                            self._test_dtensor_flatten_2d_ss(
                                tensor_dims,
                                flatten_start,
                                flatten_end,
                                mesh,
                                placements,
                            )

        # Unflatten with (SS, SS) pattern — representative factorizations
        factors_list = [(6, 4, 3), (4, 6, 2), (3, 2, 6), (2, 6, 4)]
        for factors in factors_list:
            for shard_idx0 in range(1, len(factors) - 1):
                for shard_idx1 in range(shard_idx0 + 1, len(factors)):
                    with self.subTest(factors=factors, s0=shard_idx0, s1=shard_idx1):
                        factor0 = factors[shard_idx0]
                        factor1 = factors[shard_idx1]
                        uneven0 = factor0 % mesh.size(0) != 0
                        is_last1 = shard_idx1 == len(factors) - 1
                        uneven1 = factor1 % mesh.size(1) != 0 and not is_last1
                        if uneven0 or uneven1:
                            with self.assertRaisesRegex(
                                RuntimeError, "is not evenly divisible"
                            ):
                                self._test_dtensor_unflatten_factors(
                                    factors,
                                    (shard_idx0, shard_idx1),
                                    0,
                                    mesh,
                                )
                        else:
                            self._test_dtensor_unflatten_factors(
                                factors,
                                (shard_idx0, shard_idx1),
                                0,
                                mesh,
                            )

    def test_dtensor_unflatten_ss_and_s_same_dim(self):
        """Test unflatten when _StridedShard and Shard both shard the same tensor dim."""
        mesh = init_device_mesh(self.device_type, (3, self.world_size // 3))

        global_tensor = torch.arange(4 * 6 * 3).view(4, 6, 3)
        inps = distribute_tensor(global_tensor, mesh, [Shard(1), Shard(0)])

        # Flatten dims 0,1 -> [24, 3] with [_StridedShard(0, 4), Shard(0)]
        comm_mode = CommDebugMode()
        with comm_mode:
            flattened = inps.flatten(0, 1)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(flattened.shape, (24, 3))
        self.assertIsInstance(flattened.placements[0], _StridedShard)
        self.assertEqual(flattened.placements[0].split_factor, 4)
        self.assertIsInstance(flattened.placements[1], Shard)

        # Unflatten dim 0 (24) -> (12, 2): both mesh dims stay on output dim 0
        expected_placements = (_StridedShard(0, split_factor=4), Shard(0))
        comm_mode = CommDebugMode()
        with comm_mode:
            unflattened = flattened.view(12, 2, 3)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(unflattened.placements, expected_placements)
        expected_local = distribute_tensor(
            torch.arange(4 * 6 * 3).view(12, 2, 3),
            mesh,
            expected_placements,
            src_data_rank=None,
        )._local_tensor
        self.assertEqual(unflattened._local_tensor, expected_local)

    def test_view_redistribution(self):
        """
        This test is added to demonstrate "incorrect" view ops behavior if redistribution happens.
        """

        x = torch.randn(4, 4)
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        dtensor_x = distribute_tensor(x, mesh, (Shard(0),))

        with self.assertRaisesRegex(RuntimeError, "Sharding propagation failed"):
            dtensor_x.view(-1, 8)

    def test_squeeze_(self):
        mesh_2d = init_device_mesh(self.device_type, (3, 2), mesh_dim_names=("a", "b"))
        self.init_manual_seed_for_rank()
        x = torch.randn((1, 4), device=self.device_type)
        dist_x = DTensor.from_local(x, mesh_2d, [Partial(), Shard(1)])
        self._test_op_on_dtensor(
            torch.ops.aten.squeeze_.dim,
            dist_x,
            0,
        )
        # check DTensor subclass metadata as well as placements
        self.assertEqual(dist_x.shape, torch.Size([8]))
        self.assertEqual(
            dist_x.stride(),
            (1,),
        )
        self.assertEqual(dist_x.placements, [Partial(), Shard(0)])

        # squeeze_ should not trigger any communication
        y = torch.randn((1, 4), device=self.device_type)
        dist_y = DTensor.from_local(y, mesh_2d, [Partial(), Shard(1)])
        with CommDebugMode() as comm_mode:
            torch.ops.aten.squeeze_.dim(dist_y, 0)
        self.assertEqual(comm_mode.get_total_counts(), 0)

    def test_squeeze_variants(self):
        """Test squeeze.default, squeeze.dim, and squeeze.dims with DTensor."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # squeeze.dims on sharded tensor - squeeze non-sharded dims
        with self.subTest("dims_sharded"):
            x = torch.randn(self.world_size, 1, 1, 8, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Shard(0)])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze((1, 2))
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([self.world_size, 8]))
            self.assertEqual(result.placements, (Shard(0),))
            self.assertEqual(result.to_local().shape, torch.Size([1, 8]))

        # squeeze.dim on sharded tensor - squeeze non-sharded dim
        with self.subTest("dim_sharded"):
            x = torch.randn(self.world_size, 1, 8, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Shard(0)])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze(1)
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([self.world_size, 8]))
            self.assertEqual(result.placements, (Shard(0),))
            self.assertEqual(result.to_local().shape, torch.Size([1, 8]))

        # squeeze.default on replicated tensor
        with self.subTest("default_replicated"):
            x = torch.randn(4, 1, 1, 8, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Replicate()])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze()
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([4, 8]))
            self.assertEqual(result.placements, (Replicate(),))

        # squeeze.dims on replicated tensor
        with self.subTest("dims_replicated"):
            x = torch.randn(2, 1, 3, 1, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Replicate()])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze((1, 3))
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([2, 3]))
            self.assertEqual(result.placements, (Replicate(),))

        # squeeze non-singleton dim is a no-op
        with self.subTest("non_singleton_noop"):
            x = torch.randn(self.world_size, 4, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Shard(0)])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze(1)
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([self.world_size, 4]))
            self.assertEqual(result.placements, (Shard(0),))

        # Partial passes through squeeze unchanged
        with self.subTest("partial_max_passthrough"):
            x = torch.randn(1, 4, device=self.device_type)
            dt = DTensor.from_local(x, mesh, [Partial("max")])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze(0)
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([4]))
            self.assertEqual(result.placements, (Partial("max"),))

        # Partial("sum") also passes through
        with self.subTest("partial_sum_passthrough"):
            x = torch.randn(1, device=self.device_type)
            dt = DTensor.from_local(x, mesh, [Partial("sum")])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze()
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([]))
            self.assertEqual(result.placements, (Partial("sum"),))

        # squeeze must not remove sharded dim (local size 1, global size > 1)
        with self.subTest("preserve_sharded_dim_default"):
            x = (
                torch.arange(self.world_size * 8, device=self.device_type)
                .reshape(self.world_size, 8)
                .float()
            )
            dt = distribute_tensor(x, mesh, [Shard(0)])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze()
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([self.world_size, 8]))
            self.assertEqual(result._local_tensor.shape, torch.Size([1, 8]))
            self.assertEqual(result.placements, (Shard(0),))
            self.assertEqual(result.full_tensor(), x)

        # same as above but via squeeze.dim (single int arg)
        with self.subTest("preserve_sharded_dim_explicit"):
            x = (
                torch.arange(self.world_size * 8, device=self.device_type)
                .reshape(self.world_size, 8)
                .float()
            )
            dt = distribute_tensor(x, mesh, [Shard(0)])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze(0)
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([self.world_size, 8]))
            self.assertEqual(result._local_tensor.shape, torch.Size([1, 8]))
            self.assertEqual(result.placements, (Shard(0),))
            self.assertEqual(result.full_tensor(), x)

        # squeeze.dims with mixed singleton/non-singleton dims
        with self.subTest("mixed_dims"):
            x = torch.randn(1, 4, 1, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Replicate()])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze((0, 1, 2))
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([4]))
            self.assertEqual(result.placements, (Replicate(),))
            self.assertEqual(result.full_tensor(), x.squeeze())

        # sharded non-singleton dim preserved, shard index shifts
        with self.subTest("shard_index_shift"):
            x = torch.randn(1, self.world_size, 1, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Shard(1)])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze((0, 1, 2))
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([self.world_size]))
            self.assertEqual(result._local_tensor.shape, torch.Size([1]))
            self.assertEqual(result.placements, (Shard(0),))
            self.assertEqual(result.full_tensor(), x.squeeze((0, 2)))

        # S(0) on globally-singleton dim becomes R after squeeze (#174136)
        with self.subTest("singleton_shard_becomes_replicate"):
            x = torch.randn(1, 4, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Shard(0)])
            result = dt.squeeze()
            self.assertEqual(result.shape, torch.Size([4]))
            self.assertEqual(result.placements, (Replicate(),))
            self.assertEqual(result.full_tensor(), x.squeeze())

    def test_storage_offset_slice(self):
        """
        Test that storage_offset is properly tracked on DTensor when slicing
        a replicated tensor.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create a replicated DTensor
        tensor = torch.randn(10, device=self.device_type)
        dtensor = distribute_tensor(tensor, mesh, [Replicate()])

        # Perform a slice operation [1:]
        with CommDebugMode() as comm_mode:
            sliced_dtensor = dtensor[1:]
            # Slicing should not trigger any communication
            self.assertEqual(comm_mode.get_total_counts(), 0)

        # Verify that the DTensor's storage_offset matches the expected value
        self.assertEqual(sliced_dtensor.storage_offset(), 1)

        # Verify that the local tensor also has the correct storage_offset
        self.assertEqual(sliced_dtensor.to_local().storage_offset(), 1)

        # Verify the shape is correct
        self.assertEqual(sliced_dtensor.shape, torch.Size([9]))

        # Verify the values are correct
        expected = tensor[1:]
        self.assertEqual(sliced_dtensor.full_tensor(), expected)

    def test_storage_offset_shard_dim0_slice_dim1(self):
        """
        Test that storage_offset is properly tracked when tensor is sharded on dim 0
        and sliced on dim 1.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create a 2D tensor and shard on dim 0
        tensor = torch.randn(12, 8, device=self.device_type)
        dtensor = distribute_tensor(tensor, mesh, [Shard(0)])

        # Perform a slice operation [:, 2:]
        with CommDebugMode() as comm_mode:
            sliced_dtensor = dtensor[:, 2:]
            # Slicing should not trigger any communication
            self.assertEqual(comm_mode.get_total_counts(), 0)

        # The storage_offset should be 2 (skipping 2 elements in each row)
        self.assertEqual(sliced_dtensor.storage_offset(), 2)

        # Verify that the local tensor also has the correct storage_offset
        self.assertEqual(sliced_dtensor.to_local().storage_offset(), 2)

        # Verify the shape is correct
        expected_shape = torch.Size([12, 6])
        self.assertEqual(sliced_dtensor.shape, expected_shape)

        # Verify the values are correct
        expected = tensor[:, 2:]
        self.assertEqual(sliced_dtensor.full_tensor(), expected)

    def test_storage_offset_shard_dim1_slice_dim0(self):
        """
        Test that storage_offset is properly tracked when tensor is sharded on dim 1
        and sliced on dim 0.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create a 2D tensor and shard on dim 1
        tensor = torch.randn(10, 12, device=self.device_type)
        dtensor = distribute_tensor(tensor, mesh, [Shard(1)])

        # Perform a slice operation [2:, :]
        with CommDebugMode() as comm_mode:
            sliced_dtensor = dtensor[2:, :]
            # Slicing should not trigger any communication
            self.assertEqual(comm_mode.get_total_counts(), 0)

        local_dim1_size = 12 // self.world_size
        expected_offset = 2 * local_dim1_size
        self.assertEqual(sliced_dtensor.storage_offset(), expected_offset)

        self.assertEqual(sliced_dtensor.to_local().storage_offset(), expected_offset)

        # Verify the shape is correct
        expected_shape = torch.Size([8, 12])
        self.assertEqual(sliced_dtensor.shape, expected_shape)

        # Verify the values are correct
        expected = tensor[2:, :]
        self.assertEqual(sliced_dtensor.full_tensor(), expected)

    def test_view_groups_unbacked_symint(self):
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()

        def fresh_sym():
            return shape_env.create_unbacked_symint()

        # Same symbol forwards correctly
        u0 = fresh_sym()
        self.assertEqual(
            view_groups([4, u0, 3], [4, u0, 3]),
            (InputDim(0), InputDim(1), InputDim(2)),
        )

        # Same symbol with concrete split
        u1 = fresh_sym()
        self.assertEqual(
            view_groups([u1, 12], [u1, 3, 4]),
            (
                InputDim(0),
                Split(InputDim(1), (3, 4), 0),
                Split(InputDim(1), (3, 4), 1),
            ),
        )

        # Same symbol with concrete flatten
        u2 = fresh_sym()
        self.assertEqual(
            view_groups([u2, 3, 4], [u2, 12]),
            (InputDim(0), Flatten((InputDim(1), InputDim(2)))),
        )

        # Concrete split with trailing symbol
        u3 = fresh_sym()
        self.assertEqual(
            view_groups([6, u3], [2, 3, u3]),
            (
                Split(InputDim(0), (2, 3), 0),
                Split(InputDim(0), (2, 3), 1),
                InputDim(1),
            ),
        )

        # Singletons with symbolic dims
        u4 = fresh_sym()
        self.assertEqual(
            view_groups([1, 1, u4, 3], [u4, 3]),
            (InputDim(2), InputDim(3)),
        )

        # Flatten symbolic dims using same product expression
        u5 = fresh_sym()
        u6 = fresh_sym()
        self.assertEqual(
            view_groups([u5, u6], [u5 * u6]),
            (Flatten((InputDim(0), InputDim(1))),),
        )

        # Partial fallback: concrete prefix resolves, symbolic suffix falls back
        u7 = fresh_sym()
        u8 = fresh_sym()
        result = view_groups([4, u7, u8], [4, u7 * u8])
        # dim 0 resolves as InputDim(0), dims 1-2 flatten
        self.assertEqual(
            result,
            (InputDim(0), Flatten((InputDim(1), InputDim(2)))),
        )

        # Different unbacked symbols (1-to-1): fallback simplifies to InputDim
        u9 = fresh_sym()
        u10 = fresh_sym()
        result = view_groups([4, u9], [4, u10])
        self.assertEqual(result, (InputDim(0), InputDim(1)))

    def test_view_groups_unbacked_sharding_propagation(self):
        """Test that sharding is correctly propagated through view_groups with symbolic shapes."""
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()

        def fresh_sym():
            return shape_env.create_unbacked_symint()

        mesh_sizes = (2, 3)

        # InputDim forwarding: Shard(1) on symbolic dim should propagate
        u0 = fresh_sym()
        from_shape = (4, u0, 6)
        to_shape = (4, u0, 6)
        rule = view_groups(from_shape, to_shape)
        input_placements = [Shard(1), Replicate()]
        inp_tgt, out_plc = propagate_shape_and_sharding(
            input_placements, from_shape, rule, mesh_sizes
        )
        self.assertEqual(out_plc, [Shard(1), Replicate()])

        # Same symbol with concrete split: Shard(0) on symbolic dim should forward
        u1 = fresh_sym()
        from_shape = (u1, 12)
        to_shape = (u1, 3, 4)
        rule = view_groups(from_shape, to_shape)
        input_placements = [Shard(0), Replicate()]
        inp_tgt, out_plc = propagate_shape_and_sharding(
            input_placements, from_shape, rule, mesh_sizes
        )
        self.assertEqual(out_plc, [Shard(0), Replicate()])

        # Shard on concrete dim that gets split: should propagate to split_id=0
        u2 = fresh_sym()
        from_shape = (6, u2)
        to_shape = (2, 3, u2)
        rule = view_groups(from_shape, to_shape)
        input_placements = [Shard(0), Replicate()]
        inp_tgt, out_plc = propagate_shape_and_sharding(
            input_placements, from_shape, rule, mesh_sizes
        )
        self.assertEqual(out_plc, [Shard(0), Replicate()])

        # Flatten [u, 16] -> [u*16] with Shard(0) on unbacked dim (leftmost)
        u3 = fresh_sym()
        for ms in mesh_sizes:  # needed for sharding
            torch._check(u3 % ms == 0)
        from_shape = (u3, 16)
        to_shape = (u3 * 16,)
        rule = view_groups(from_shape, to_shape)
        input_placements = [Shard(0), Replicate()]
        inp_tgt, out_plc = propagate_shape_and_sharding(
            input_placements, from_shape, rule, mesh_sizes
        )
        self.assertEqual(out_plc, [Shard(0), Replicate()])

        # Flatten [16, u] -> [16*u] with Shard(1) on unbacked dim (non-leftmost):
        # should force replicate since non-leftmost dims can't propagate through flatten
        u4 = fresh_sym()
        from_shape = (16, u4)
        to_shape = (16 * u4,)
        rule = view_groups(from_shape, to_shape)
        input_placements = [Shard(1), Replicate()]
        inp_tgt, out_plc = propagate_shape_and_sharding(
            input_placements, from_shape, rule, mesh_sizes
        )
        self.assertEqual(out_plc, [Replicate(), Replicate()])

    def test_input_dim_rejects_int_comparison(self):
        """InputDim.__eq__ should raise TypeError when compared with int.

        Regression guard: shard.dim == in_dim (int == InputDim) silently
        returned False for over 3 years, making downstream validation dead code.
        """
        dim = InputDim(0)
        # InputDim == InputDim is fine
        self.assertEqual(dim, InputDim(0))
        self.assertNotEqual(dim, InputDim(1))
        # hash contract: equal InputDims must have equal hashes
        self.assertEqual(hash(dim), hash(InputDim(0)))
        # hash is salted so it won't collide with raw int in dicts/sets
        self.assertNotEqual(hash(dim), hash(0))
        # InputDim == int raises (catches `dim == some_int`)
        with self.assertRaisesRegex(TypeError, "Did you mean to use .input_dim"):
            _ = dim == 0
        # int == InputDim also raises (catches the original bug: `shard.dim == in_dim`)
        with self.assertRaisesRegex(TypeError, "Did you mean to use .input_dim"):
            _ = 0 == dim


TestViewOpsWithLocalTensor = create_local_tensor_test_class(
    TestViewOps,
    skipped_tests=[
        # Comparing data pointers is not supported for local tensor
        "test_dtensor_view_op_uneven",
        # These tests use ShapeEnv directly, not local tensor tests
        "test_view_groups_unbacked_symint",
        "test_view_groups_unbacked_sharding_propagation",
        # Too many test cases for LocalTensorMode dispatch overhead
        "test_dtensor_flatten_1d",
        "test_dtensor_flatten_2d",
        "test_dtensor_flatten_multi_mesh",
    ],
    base_class=LocalDTensorContinuousTestBase,
)


if __name__ == "__main__":
    run_tests()
