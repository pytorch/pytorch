# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools
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
    Broadcast,
    dim_maps,
    Flatten,
    InputDim,
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
    DTensorTestBase,
    with_comms,
)
from torch.utils import _pytree as pytree


class TestViewOps(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 6

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

    @with_comms
    def test_illegal_views(self):
        device_mesh = self.build_device_mesh()
        # 1D mesh [6] (see above)
        tensor = torch.randn((6, 256))
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        shard = dtensor.redistribute(device_mesh=device_mesh, placements=[Shard(dim=0)])
        # view should be legal, since sharding is even and flatten includes only one sharded dim
        shard.view(-1)

        shard = dtensor.redistribute(device_mesh=device_mesh, placements=[Shard(dim=1)])
        with self.assertRaisesRegex(RuntimeError, "Sharding propagation failed"):
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

    @with_comms
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
    @with_comms
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

    @with_comms
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

    @with_comms
    def test_view_redistribution(self):
        """
        This test is added to demonstrate "incorrect" view ops behavior if redistribution happens.
        """

        x = torch.randn(4, 4)
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        dtensor_x = distribute_tensor(x, mesh, (Shard(0),))

        with self.assertRaisesRegex(RuntimeError, "Sharding propagation failed"):
            dtensor_x.view(-1, 8)

    @with_comms
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

    @with_comms
    def test_squeeze_variants(self):
        """Test all squeeze variants with DTensor.

        Tests squeeze.default, squeeze.dim, and squeeze.dims.
        For sharded tensors, we only squeeze dims that are NOT sharded to avoid
        the FIXME bug where sharded dims with local size 1 get incorrectly squeezed.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Test 1: squeeze.dims on sharded tensor - squeeze non-sharded dims
        x1 = torch.randn(self.world_size, 1, 1, 8, device=self.device_type)
        dt1 = distribute_tensor(x1, mesh, [Shard(0)])
        result1 = dt1.squeeze((1, 2))
        self.assertEqual(result1.shape, torch.Size([self.world_size, 8]))
        self.assertEqual(result1.placements, (Shard(0),))
        self.assertEqual(result1.to_local().shape, torch.Size([1, 8]))

        # Test 2: squeeze.dim on sharded tensor - squeeze non-sharded dim
        x2 = torch.randn(self.world_size, 1, 8, device=self.device_type)
        dt2 = distribute_tensor(x2, mesh, [Shard(0)])
        result2 = dt2.squeeze(1)
        self.assertEqual(result2.shape, torch.Size([self.world_size, 8]))
        self.assertEqual(result2.placements, (Shard(0),))
        self.assertEqual(result2.to_local().shape, torch.Size([1, 8]))

        # Test 3: squeeze.default on replicated tensor (avoids FIXME bug)
        x3 = torch.randn(4, 1, 1, 8, device=self.device_type)
        dt3 = distribute_tensor(x3, mesh, [Replicate()])
        result3 = dt3.squeeze()
        self.assertEqual(result3.shape, torch.Size([4, 8]))
        self.assertEqual(result3.placements, (Replicate(),))

        # Test 4: squeeze.dims on replicated tensor
        x4 = torch.randn(2, 1, 3, 1, device=self.device_type)
        dt4 = distribute_tensor(x4, mesh, [Replicate()])
        result4 = dt4.squeeze((1, 3))
        self.assertEqual(result4.shape, torch.Size([2, 3]))
        self.assertEqual(result4.placements, (Replicate(),))

        # Test 5: squeeze dim that is NOT size 1 (should be no-op)
        x5 = torch.randn(self.world_size, 4, device=self.device_type)
        dt5 = distribute_tensor(x5, mesh, [Shard(0)])
        result5 = dt5.squeeze(1)  # dim 1 has size 4, not 1
        self.assertEqual(result5.shape, torch.Size([self.world_size, 4]))
        self.assertEqual(result5.placements, (Shard(0),))

        # Test 6: squeeze [1] with Partial("max") -> [] Replicate
        # When output numel=1, max(x)=x so P(max) reduces to R trivially
        x6 = torch.randn(1, device=self.device_type)
        dt6 = DTensor.from_local(x6, mesh, [Partial("max")])
        result6 = dt6.squeeze()
        self.assertEqual(result6.shape, torch.Size([]))
        self.assertEqual(result6.placements, (Replicate(),))
        # Verify redistribution happened and value is correct
        self.assertEqual(result6.to_local(), result6.full_tensor())
        self.assertEqual(result6.full_tensor(), x6.squeeze())

        # Test 7: squeeze [1] with Partial("min") -> [] Replicate
        x7 = torch.randn(1, device=self.device_type)
        dt7 = DTensor.from_local(x7, mesh, [Partial("min")])
        result7 = dt7.squeeze()
        self.assertEqual(result7.shape, torch.Size([]))
        self.assertEqual(result7.placements, (Replicate(),))
        self.assertEqual(result7.to_local(), result7.full_tensor())
        self.assertEqual(result7.full_tensor(), x7.squeeze())

        # Test 8: squeeze [1,1] with Partial("max") -> [] Replicate (numel=1)
        x8 = torch.randn(1, 1, device=self.device_type)
        dt8 = DTensor.from_local(x8, mesh, [Partial("max")])
        result8 = dt8.squeeze()
        self.assertEqual(result8.shape, torch.Size([]))
        self.assertEqual(result8.placements, (Replicate(),))
        self.assertEqual(result8.to_local(), result8.full_tensor())
        self.assertEqual(result8.full_tensor(), x8.squeeze())

        # Test 9: squeeze [1,4] dim=0 with Partial("max") -> [4] stays Partial
        # numel=4 > 1, so max reduction is NOT trivial
        x9 = torch.randn(1, 4, device=self.device_type)
        dt9 = DTensor.from_local(x9, mesh, [Partial("max")])
        result9 = dt9.squeeze(0)
        self.assertEqual(result9.shape, torch.Size([4]))
        self.assertEqual(result9.placements, (Partial("max"),))

        # Test 10: squeeze [1] with Partial("sum") -> [] stays Partial
        # sum is NOT trivial even for numel=1: sum(x,x,x,...)=n*x
        x10 = torch.randn(1, device=self.device_type)
        dt10 = DTensor.from_local(x10, mesh, [Partial("sum")])
        result10 = dt10.squeeze()
        self.assertEqual(result10.shape, torch.Size([]))
        self.assertEqual(result10.placements, (Partial("sum"),))

        # Test 11: view [1] to [1,1] with Partial("max") -> Replicate
        # numel=1, so P(max) reduces to R trivially
        x11 = torch.randn(1, device=self.device_type)
        dt11 = DTensor.from_local(x11, mesh, [Partial("max")])
        result11 = dt11.view(1, 1)
        self.assertEqual(result11.shape, torch.Size([1, 1]))
        self.assertEqual(result11.placements, (Replicate(),))
        self.assertEqual(result11.to_local(), result11.full_tensor())
        self.assertEqual(result11.full_tensor(), x11.view(1, 1))

        # Test 12: view [1] to [1,1] with Partial("min") -> Replicate
        x12 = torch.randn(1, device=self.device_type)
        dt12 = DTensor.from_local(x12, mesh, [Partial("min")])
        result12 = dt12.view(1, 1)
        self.assertEqual(result12.shape, torch.Size([1, 1]))
        self.assertEqual(result12.placements, (Replicate(),))
        self.assertEqual(result12.to_local(), result12.full_tensor())
        self.assertEqual(result12.full_tensor(), x12.view(1, 1))

        # Test 13: view [4] to [2,2] with Partial("max") stays Partial
        # numel=4 > 1, not trivial
        x13 = torch.randn(4, device=self.device_type)
        dt13 = DTensor.from_local(x13, mesh, [Partial("max")])
        result13 = dt13.view(2, 2)
        self.assertEqual(result13.shape, torch.Size([2, 2]))
        self.assertEqual(result13.placements, (Partial("max"),))

        # Test 14: squeeze on sharded dim should NOT remove it
        # Global shape [world_size, 8], local shape [1, 8] - dim 0 is NOT globally singleton
        x14 = (
            torch.arange(self.world_size * 8, device=self.device_type)
            .reshape(self.world_size, 8)
            .float()
        )
        dt14 = distribute_tensor(x14, mesh, [Shard(0)])
        result14 = dt14.squeeze()  # should not squeeze dim 0
        self.assertEqual(result14.shape, torch.Size([self.world_size, 8]))
        self.assertEqual(result14._local_tensor.shape, torch.Size([1, 8]))
        self.assertEqual(result14.placements, (Shard(0),))
        self.assertEqual(result14.full_tensor(), x14)

        # Test 15: squeeze.dims on sharded dim should NOT remove it
        x15 = (
            torch.arange(self.world_size * 8, device=self.device_type)
            .reshape(self.world_size, 8)
            .float()
        )
        dt15 = distribute_tensor(x15, mesh, [Shard(0)])
        result15 = dt15.squeeze((0,))  # explicitly try to squeeze dim 0
        self.assertEqual(result15.shape, torch.Size([self.world_size, 8]))
        self.assertEqual(result15._local_tensor.shape, torch.Size([1, 8]))
        self.assertEqual(result15.placements, (Shard(0),))
        self.assertEqual(result15.full_tensor(), x15)

        # Test 16: squeeze.dims with mixed singleton/non-singleton dims (filtering)
        # Global shape [1, 4, 1] - dims 0 and 2 are singleton, dim 1 is not
        # squeeze.dims([0, 1, 2]) should only squeeze dims 0 and 2
        x16 = torch.randn(1, 4, 1, device=self.device_type)
        dt16 = distribute_tensor(x16, mesh, [Replicate()])
        result16 = dt16.squeeze(
            (0, 1, 2)
        )  # asks for all dims, but only 0,2 are singleton
        self.assertEqual(result16.shape, torch.Size([4]))
        self.assertEqual(result16.placements, (Replicate(),))
        self.assertEqual(result16.full_tensor(), x16.squeeze())

        # Test 17: squeeze.dims filtering with sharded non-singleton dim
        # Global [1, world_size, 1] sharded on dim 1, local [1, 1, 1]
        # squeeze.dims([0, 1, 2]) should only squeeze dims 0 and 2 (globally singleton)
        x17 = torch.randn(1, self.world_size, 1, device=self.device_type)
        dt17 = distribute_tensor(x17, mesh, [Shard(1)])
        result17 = dt17.squeeze((0, 1, 2))
        self.assertEqual(result17.shape, torch.Size([self.world_size]))
        self.assertEqual(result17._local_tensor.shape, torch.Size([1]))
        self.assertEqual(result17.placements, (Shard(0),))  # dim shifted after squeeze
        self.assertEqual(result17.full_tensor(), x17.squeeze((0, 2)))

        # Test 18: squeeze with uneven sharding (tensor_dim < mesh_dim)
        # Global [1, 4] sharded on dim 0 -> rank 0 gets [1, 4], others get [0, 4]
        # Squeeze should remove dim 0 (global singleton) and redistribute to Replicate
        x18 = torch.randn(1, 4, device=self.device_type)
        dt18 = distribute_tensor(x18, mesh, [Shard(0)])
        result18 = dt18.squeeze()
        self.assertEqual(result18.shape, torch.Size([4]))
        self.assertEqual(result18.placements, (Replicate(),))
        self.assertEqual(result18.full_tensor(), x18.squeeze())

    @with_comms
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

    @with_comms
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

    @with_comms
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


TestViewOpsWithLocalTensor = create_local_tensor_test_class(
    TestViewOps,
    skipped_tests=[
        # Comparing data pointers is not supported for local tensor
        "test_dtensor_view_op_uneven",
    ],
)

if __name__ == "__main__":
    run_tests()
