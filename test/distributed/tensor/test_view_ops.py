# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools
from typing import cast
import math

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
        # result = view_groups([3, 4, 1, 5], [12, 5])
        # self.assertEqual(
        #     result,
        #     (Flatten((InputDim(0), InputDim(1), InputDim(2))), InputDim(3))
        # )
        self.assertEqual(
            view_groups([2, 3], [3, 2]),
            (
                Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 0),
                Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 1),
            ),
        )
        view_groups([3, 4, 5], [12, 5])
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
    def test_dtensor_flatten_1d(self):
        mesh: DeviceMesh = init_device_mesh(self.device_type, (self.world_size,))
        # cover uneven sharding with mesh size +/- 1
        for seq_len in [mesh.size(0) - 1, mesh.size(0), mesh.size(0) + 1]:
            self._test_dtensor_flatten_1d(mesh, seq_len)

    def _test_dtensor_flatten_1d(self, mesh, seq_len):
        batch_size, dim = 6, 3
        global_inps: Tensor = torch.arange(batch_size * seq_len * dim).view(batch_size, seq_len, dim)
        global_inps_replicate: DTensor = distribute_tensor(global_inps, mesh, (Replicate(), ))
        inps = global_inps_replicate.redistribute(mesh, (Shard(1),))
        comm_mode = CommDebugMode()
        with comm_mode:
            inps_viewed = inps.view(batch_size * seq_len, dim)
        expected_placements = (_StridedShard(dim=0, split_factor=batch_size),)
        expected_local_tensor = distribute_tensor(global_inps.view(batch_size * seq_len, dim), mesh, (Replicate(), )).redistribute(mesh, expected_placements)._local_tensor
        self.assertEqual(inps_viewed.placements, expected_placements)
        self.assertEqual(inps_viewed._local_tensor, expected_local_tensor)
        self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_dtensor_flatten_2d(self):
        mesh: DeviceMesh = init_device_mesh(self.device_type, (self.world_size // 2, 2))
        # cover uneven sharding with mesh size +/- 1
        # for seq_len in [mesh.size(0) - 1, mesh.size(0), mesh.size(0) + 1]:
        #     for dim1 in [mesh.size(1) - 1, mesh.size(1), mesh.size(1) + 1]:
        
        for seq_len in [mesh.size(0) * 2]:
            for dim1 in [mesh.size(1) * 2 - 1, mesh.size(1) * 2, mesh.size(1) * 2 + 1]:
                self._test_dtensor_flatten_2d(mesh, seq_len, dim1)

        for seq_len in [mesh.size(0) * 2 - 1, mesh.size(0) * 2 + 1]:
            for dim1 in [mesh.size(1) * 2]:
                # expect error
                pass

    def _test_dtensor_flatten_2d(self, mesh, seq_len, dim1):
        # S1, S2
        batch_size, dim2 = 6, 3
        global_inps = torch.arange(batch_size * seq_len * dim1 * dim2).view(
            batch_size, seq_len, dim1, dim2
        )
        inps = distribute_tensor(global_inps, mesh, (Shard(1), Shard(2)))
        print(f"distribute_tensor: {torch.distributed.get_rank()=}", flush=True)
        # comm_mode = CommDebugMode()
        # with comm_mode:
        inps_viewed = inps.view(batch_size * seq_len * dim1, dim2)
        print(f"inps_viewed: {torch.distributed.get_rank()=}", flush=True)
        expected_placements = (
            _StridedShard(dim=0, split_factor=batch_size),
            _StridedShard(dim=0, split_factor=batch_size * math.ceil(seq_len * 1.0 / mesh.size(0))),
        )
        self.assertEqual(inps_viewed.placements, expected_placements)
        # self.assertEqual(comm_mode.get_total_counts(), 0)

        # import fbvscode
        # fbvscode.set_trace()

        # R, S2
        # global_inps = torch.arange(batch_size * seq_len * dim1 * dim2).view(
        #     batch_size, seq_len, dim1, dim2
        # )
        # inps = distribute_tensor(global_inps, mesh, (Replicate(), Shard(2)))
        # comm_mode = CommDebugMode()
        # with comm_mode:
        #     inps_viewed = inps.view(batch_size * seq_len * dim1, dim2)
        # expected_placements = (
        #     Replicate(),
        #     _StridedShard(dim=0, split_factor=36),
        # )
        # self.assertEqual(inps_viewed.placements, expected_placements)
        # self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_dtensor_unflatten(self):
        # 1D case
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        batch_size, seq_len, dim = 6, 6, 3
        global_inps = torch.arange(batch_size * seq_len * dim).view(
            batch_size * seq_len, dim
        )
        inps = distribute_tensor(global_inps, mesh, (_StridedShard(0, split_factor=6),))
        inps_viewed = inps.view(batch_size, seq_len, dim)
        expected_placements = (Shard(1),)
        self.assertEqual(inps_viewed.placements, expected_placements)

        # 2D case: S1, S2
        mesh = init_device_mesh(self.device_type, (self.world_size // 2, 2))
        batch_size, seq_len, dim1, dim2 = 6, 6, 6, 3
        global_inps = torch.arange(batch_size * seq_len * dim1 * dim2).view(
            batch_size * seq_len * dim1, dim2
        )
        inps = distribute_tensor(
            global_inps,
            mesh,
            (
                _StridedShard(dim=0, split_factor=6),
                _StridedShard(dim=0, split_factor=12),
            ),
        )
        inps_viewed = inps.view(batch_size, seq_len, dim1, dim2)
        expected_placements = (Shard(1), Shard(2))
        self.assertEqual(inps_viewed.placements, expected_placements)

        # 2D case: R, S2
        mesh = init_device_mesh(self.device_type, (self.world_size // 2, 2))
        batch_size, seq_len, dim1, dim2 = 6, 6, 6, 3
        global_inps = torch.arange(batch_size * seq_len * dim1 * dim2).view(
            batch_size * seq_len * dim1, dim2
        )
        inps = distribute_tensor(
            global_inps, mesh, (Replicate(), _StridedShard(dim=0, split_factor=36))
        )
        inps_viewed = inps.view(batch_size, seq_len, dim1, dim2)
        expected_placements = (
            Replicate(),
            Shard(2),
        )
        self.assertEqual(inps_viewed.placements, expected_placements)

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
        "test_dtensor_flatten_1d",
        "test_dtensor_flatten_2d",
    ],
)

if __name__ == "__main__":
    run_tests()
