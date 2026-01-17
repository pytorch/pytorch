# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import contextlib
import copy
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
    _is_last_shard_on_tensor_dim_plus,
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

    @with_comms
    def test_dtensor_flatten_1d(self):
        mesh: DeviceMesh = init_device_mesh(self.device_type, (self.world_size,))
        for tensor_ndim in [2, 3, 4]:
            for flatten_start in range(tensor_ndim):
                for flatten_end in range(flatten_start + 2, tensor_ndim + 1):
                    # Shard
                    for shard_dim in range(flatten_start, flatten_end):
                        tensor_dim_values = [
                            2 * mesh.size(0) - 1,
                            2 * mesh.size(0),
                            2 * mesh.size(0) + 1,
                        ]
                        for tensor_dims in list(
                            itertools.product(tensor_dim_values, repeat=tensor_ndim)
                        ):
                            placements = (Shard(shard_dim),)
                            ctx = contextlib.nullcontext()
                            # uneven shard on last dim (flatten_end - 1) is supported
                            if tensor_dims[shard_dim] % mesh.size(
                                0
                            ) != 0 and shard_dim != (flatten_end - 1):
                                ctx = self.assertRaises(RuntimeError)
                            with ctx:
                                self._test_dtensor_flatten_1d_shard(
                                    tensor_dims,
                                    flatten_start,
                                    flatten_end,
                                    mesh,
                                    placements,
                                )

                    # Replicate
                    tensor_dim_values = [
                        2 * mesh.size(0) - 1,
                        2 * mesh.size(0),
                        2 * mesh.size(0) + 1,
                    ]
                    for tensor_dims in list(
                        itertools.product(tensor_dim_values, repeat=tensor_ndim)
                    ):
                        placements = (Replicate(),)
                        self._test_dtensor_flatten_replicate(
                            tensor_dims,
                            flatten_start,
                            flatten_end,
                            mesh,
                            placements,
                        )

    def _test_dtensor_flatten_1d_shard(
        self, tensor_dims, flatten_start, flatten_end, mesh, placements
    ):
        shard_dim = placements[0].dim
        nelem = math.prod(tensor_dims)
        global_inps: Tensor = torch.arange(nelem).view(tensor_dims)
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)
        viewed_tensor_dims = self._get_viewed_tensor_dims(
            tensor_dims, flatten_start, flatten_end
        )
        comm_mode = CommDebugMode()
        with comm_mode:
            inps_viewed = inps.view(viewed_tensor_dims)
        if placements[0] == Shard(flatten_start):
            expected_placements = (Shard(flatten_start),)
        else:
            split_factor = math.prod(tensor_dims[flatten_start:shard_dim])
            assert split_factor > 1
            expected_placements = (
                _StridedShard(dim=flatten_start, split_factor=split_factor),
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

    @with_comms
    def test_dtensor_flatten_2d(self):
        assert self.world_size == 6
        mesh_ndim = 2
        mesh: DeviceMesh = init_device_mesh(self.device_type, (3, self.world_size // 3))

        for tensor_ndim in [2, 3, 4]:
            for flatten_start in range(tensor_ndim):
                for flatten_end in range(flatten_start + 2, tensor_ndim + 1):
                    # # S, R and R, S
                    for shard_dim in range(flatten_start, flatten_end):
                        for shard_placement_idx in range(mesh_ndim):
                            tensor_dim_values = [
                                2 * mesh.size(shard_placement_idx) - 1,
                                2 * mesh.size(shard_placement_idx),
                                2 * mesh.size(shard_placement_idx) + 1,
                            ]
                            for tensor_dims in list(
                                itertools.product(tensor_dim_values, repeat=tensor_ndim)
                            ):
                                placements = tuple(
                                    Shard(shard_dim)
                                    if idx == shard_placement_idx
                                    else Replicate()
                                    for idx in range(mesh_ndim)
                                )
                                ctx = contextlib.nullcontext()
                                # uneven shard on last dim (flatten_end - 1) is supported
                                if tensor_dims[shard_dim] % mesh.size(
                                    shard_placement_idx
                                ) != 0 and shard_dim != (flatten_end - 1):
                                    ctx = self.assertRaises(RuntimeError)
                                with ctx:
                                    self._test_dtensor_flatten_2d_sr_rs(
                                        tensor_dims,
                                        flatten_start,
                                        flatten_end,
                                        mesh,
                                        placements,
                                        shard_placement_idx,
                                    )

                    # S, S
                    for shard_dim0 in range(flatten_start, flatten_end):
                        for shard_dim1 in range(shard_dim0, flatten_end):
                            tensor_dim_values = [
                                2 * self.world_size - 1,
                                2 * self.world_size,
                                2 * self.world_size + 1,
                            ]
                            for tensor_dims in list(
                                itertools.product(tensor_dim_values, repeat=tensor_ndim)
                            ):
                                local_tensor_dims = list(tensor_dims)
                                placements = (Shard(shard_dim0), Shard(shard_dim1))
                                ctx = contextlib.nullcontext()
                                if local_tensor_dims[shard_dim0] % mesh.size(0) != 0:
                                    ctx = self.assertRaises(RuntimeError)
                                local_tensor_dims[shard_dim0] = local_tensor_dims[
                                    shard_dim0
                                ] // mesh.size(0)
                                if local_tensor_dims[shard_dim1] % mesh.size(
                                    1
                                ) != 0 and shard_dim1 != (flatten_end - 1):
                                    ctx = self.assertRaises(RuntimeError)
                                local_tensor_dims[shard_dim1] = math.ceil(
                                    local_tensor_dims[shard_dim1] * 1.0 / mesh.size(1)
                                )
                                with ctx:
                                    self._test_dtensor_flatten_2d_ss(
                                        tensor_dims,
                                        flatten_start,
                                        flatten_end,
                                        mesh,
                                        placements,
                                    )

                    # Replicate
                    tensor_dims = [2 * mesh.size(0) - 1] * tensor_ndim
                    placements = (Replicate(), Replicate())
                    self._test_dtensor_flatten_replicate(
                        tensor_dims,
                        flatten_start,
                        flatten_end,
                        mesh,
                        placements,
                    )

    def _test_dtensor_flatten_2d_sr_rs(
        self,
        tensor_dims,
        flatten_start,
        flatten_end,
        mesh,
        placements,
        shard_placement_idx,
    ):
        shard_placement = placements[shard_placement_idx]
        shard_dim = shard_placement.dim
        nelem = math.prod(tensor_dims)
        global_inps: Tensor = torch.arange(nelem).view(tensor_dims)
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)
        viewed_tensor_dims = self._get_viewed_tensor_dims(
            tensor_dims, flatten_start, flatten_end
        )
        comm_mode = CommDebugMode()
        with comm_mode:
            inps_viewed = inps.view(viewed_tensor_dims)
        if shard_placement == Shard(flatten_start):
            expected_placement = Shard(flatten_start)
        else:
            split_factor = math.prod(tensor_dims[flatten_start:shard_dim])
            assert split_factor > 1
            expected_placement = _StridedShard(
                dim=flatten_start, split_factor=split_factor
            )
        expected_placements = (
            (expected_placement, Replicate())
            if shard_placement_idx == 0
            else (Replicate(), expected_placement)
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
            assert isinstance(placement, Shard)
            shard_dim = placement.dim
            if shard_dim == flatten_start:
                if idx == 0 or all(p.dim == flatten_start for p in placements[:idx]):
                    # S(flatten_start), S(flatten_start) qualifies
                    expected_placement = placement
            else:
                split_factor = math.prod(local_tensor_dims[flatten_start:shard_dim])
                assert split_factor > 1
                expected_placement = _StridedShard(
                    dim=flatten_start, split_factor=split_factor
                )
            if local_tensor_dims[shard_dim] % mesh.size(idx) != 0:
                # uneven shard on last flattened dim is supported
                assert _is_last_shard_on_tensor_dim_plus(idx, placements)
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
            local_tensor_dims_unflatten = copy.deepcopy(tensor_dims_unflatten)
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
            yield tensor_dims

    @with_comms
    def test_dtensor_unflatten_1d(self):
        mesh: DeviceMesh = init_device_mesh(self.device_type, (self.world_size,))

        # self._test_dtensor_unflatten_1d_shard(
        #     (13, 13, 11),
        #     (13, 13, 6),
        #     (13, 143),
        #     1,
        #     (Shard(2), ),
        #     mesh,
        # )
        # return

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
                            if tensor_dims_unflatten[shard_dim] % mesh.size(
                                0
                            ) != 0 and shard_dim != (flatten_end - 1):
                                ctx = self.assertRaises(RuntimeError)
                            with ctx:
                                self._test_dtensor_unflatten_1d_shard(
                                    tensor_dims_unflatten,
                                    local_tensor_dims_unflatten,
                                    tensor_dims_flatten,
                                    flatten_start,
                                    expected_placements,
                                    mesh,
                                )

        # any factoring on unflatten_dim
        for tensor_ndim in [2, 3, 4]:
            for unflatten_dim in range(tensor_ndim):
                for shard_dim in range(tensor_ndim):
                    for tensor_dims in self.generate_tensor_dims_1d_after_flatten(
                        tensor_ndim, unflatten_dim, shard_dim, mesh
                    ):
                        placements = (Shard(shard_dim),)
                        self._test_dtensor_unflatten_1d_shard_arbitrary(
                            tensor_dims,
                            unflatten_dim,
                            placements,
                            mesh,
                        )

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
            assert shard_dim == flatten_start
            placements = (Shard(flatten_start),)
        else:
            placements = (_StridedShard(flatten_start, split_factor=split_factor),)
        nelem = math.prod(tensor_dims_flatten)
        global_inps = torch.arange(nelem).view(tensor_dims_flatten)
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)
        inps_viewed = inps.view(tensor_dims_unflatten)
        self.assertEqual(inps_viewed.placements, expected_placements)

    def _get_all_factorizations(self, n):
        """
        Return all ways to factor n into a tuple of integers > 1.
        Each factorization has length >= 2 (non-trivial unflatten).
        Order matters for view operations, so all permutations are included.

        Examples:
            12 -> [(2, 6), (6, 2), (3, 4), (4, 3), (2, 2, 3), (2, 3, 2), (3, 2, 2)]
            36 -> [(2, 18), (18, 2), (3, 12), ..., (2, 2, 9), ..., (2, 2, 3, 3), ...]
        """
        from itertools import permutations

        def get_sorted_factorizations(remaining, min_factor=2):
            """Get all factorizations where factors are in non-decreasing order."""
            if remaining == 1:
                return [()]

            result = []
            for f in range(min_factor, remaining + 1):
                if remaining % f == 0:
                    for sub in get_sorted_factorizations(remaining // f, f):
                        result.append((f,) + sub)
            return result

        # Get sorted factorizations (to avoid duplicates before permutation)
        sorted_facts = get_sorted_factorizations(n)

        # Filter for length >= 2 (non-trivial) and generate all permutations
        all_factorizations = set()
        for factors in sorted_facts:
            if len(factors) >= 2:
                for perm in permutations(factors):
                    all_factorizations.add(perm)

        return list(all_factorizations)

    def _test_dtensor_unflatten_1d_shard_arbitrary(
        self, tensor_dims, unflatten_dim, placements, mesh
    ):
        shard_dim = placements[0].dim
        assert isinstance(placements[0], Shard) and not isinstance(
            placements[0], _StridedShard
        )

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

            # Determine if we expect an error due to uneven sharding
            # Uneven sharding on shard_dim causes sharding propagation to fail
            uneven_shard = tensor_dims[shard_dim] % mesh.size(0) != 0

            # Also check if the unflatten would cause issues when shard_dim == unflatten_dim
            # and the first factor doesn't align well with mesh.size(0)
            first_factor = factors[0]

            # When shard_dim == unflatten_dim, the sharding propagates to the first new dimension.
            # This works if first_factor % mesh.size(0) == 0 (even sharding on new dim).
            # If first_factor < mesh.size(0) and not divisible, we get strided shard or error.
            # If first_factor > mesh.size(0) and not divisible, we get uneven shard on new dim.
            if shard_dim == unflatten_dim:
                if first_factor % mesh.size(0) == 0:
                    # Even sharding on first new dimension
                    unflatten_shard_issue = False
                elif first_factor < mesh.size(0):
                    # Would require strided shard - expect error
                    unflatten_shard_issue = True
                else:
                    # first_factor > mesh.size(0) but not divisible - uneven shard on new dim
                    unflatten_shard_issue = True
            else:
                unflatten_shard_issue = False

            if uneven_shard or unflatten_shard_issue:
                ctx = self.assertRaisesRegex(
                    RuntimeError, "Sharding propagation failed"
                )
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                comm_mode = CommDebugMode()
                with comm_mode:
                    inps_viewed = inps.view(tensor_dims_unflatten)

                # Only verify results if we didn't expect an error
                if not uneven_shard and not unflatten_shard_issue:
                    # Number of new dimensions added by unflatten
                    num_new_dims = len(factors) - 1

                    # Compute expected placements after unflatten
                    if shard_dim < unflatten_dim:
                        # Shard dim unaffected
                        expected_placements = (Shard(shard_dim),)
                    elif shard_dim == unflatten_dim:
                        # Sharding on the dim being unflattened
                        # The sharding applies to the first of the new dimensions
                        expected_placements = (Shard(unflatten_dim),)
                    else:
                        # shard_dim > unflatten_dim: shifts by number of new dimensions
                        expected_placements = (Shard(shard_dim + num_new_dims),)

                    self.assertEqual(inps_viewed.placements, expected_placements)
                    self.assertEqual(comm_mode.get_total_counts(), 0)

                    # Verify local tensor values
                    expected_local = distribute_tensor(
                        global_inps.view(tensor_dims_unflatten),
                        mesh,
                        expected_placements,
                        src_data_rank=None,
                    )._local_tensor
                    self.assertEqual(inps_viewed._local_tensor, expected_local)

    @with_comms
    def test_dtensor_unflatten_2d(self):
        assert self.world_size == 6
        mesh: DeviceMesh = init_device_mesh(self.device_type, (2, 3))
        batch_size, dim2 = 2, 3
        for seq_len in [2 * mesh.size(0)]:
            for dim1 in [2 * mesh.size(1) - 1, 2 * mesh.size(1), 2 * mesh.size(1) + 1]:
                self._test_dtensor_unflatten_2d(mesh, batch_size, seq_len, dim1, dim2)

        for seq_len in [2 * mesh.size(0) - 1, 2 * mesh.size(0), 2 * mesh.size(0) + 1]:
            for dim1 in [2 * mesh.size(1) - 1, 2 * mesh.size(1), 2 * mesh.size(1) + 1]:
                self._test_dtensor_unflatten_2d_replicate(
                    mesh, batch_size, seq_len, dim1, dim2
                )

        for seq_len in [2 * mesh.size(0) - 1, 2 * mesh.size(0) + 1]:
            for dim1 in [2 * mesh.size(1)]:
                pass
                # raise error
                # self._test_dtensor_unflatten_2d(mesh, batch_size, seq_len, dim1, dim2)

    def _test_dtensor_unflatten_2d(self, mesh, batch_size, seq_len, dim1, dim2):
        # S1, S2
        global_inps = torch.arange(batch_size * seq_len * dim1 * dim2).view(
            batch_size * seq_len * dim1, dim2
        )
        expected_placements = (Shard(1), Shard(2))
        placements = (
            _StridedShard(dim=0, split_factor=batch_size),
            _StridedShard(dim=0, split_factor=batch_size * (seq_len // mesh.size(0))),
        )
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)
        expected_inp_viewed = distribute_tensor(
            global_inps.view(batch_size, seq_len, dim1, dim2), mesh, expected_placements
        )
        inps_viewed = inps.view(batch_size, seq_len, dim1, dim2)
        self.assertEqual(inps_viewed.placements, expected_placements)
        self.assertEqual(inps_viewed._local_tensor, expected_inp_viewed._local_tensor)

    def _test_dtensor_unflatten_2d_replicate(
        self, mesh, batch_size, seq_len, dim1, dim2
    ):
        global_inps = torch.arange(batch_size * seq_len * dim1 * dim2).view(
            batch_size * seq_len * dim1, dim2
        )
        placements = (
            Replicate(),
            _StridedShard(dim=0, split_factor=batch_size * seq_len),
        )
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)
        inps_viewed = inps.view(batch_size, seq_len, dim1, dim2)
        expected_placements = (
            Replicate(),
            Shard(2),
        )
        expected_inp_viewed = distribute_tensor(
            global_inps.view(batch_size, seq_len, dim1, dim2), mesh, expected_placements
        )
        self.assertEqual(inps_viewed.placements, expected_placements)
        self.assertEqual(inps_viewed._local_tensor, expected_inp_viewed._local_tensor)

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


class TestViewOps3D(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 8

    @with_comms
    def test_dtensor_unflatten_3d(self):
        assert self.world_size == 8
        mesh: DeviceMesh = init_device_mesh(self.device_type, (2, 2, 2))
        batch_size, dim3 = 2, 3
        for seq_len in [2 * mesh.size(0)]:
            for dim1 in [2 * mesh.size(1)]:
                for dim2 in [
                    2 * mesh.size(2) - 1,
                    2 * mesh.size(2),
                    2 * mesh.size(2) + 1,
                ]:
                    self._test_dtensor_unflatten_3d(
                        mesh, batch_size, seq_len, dim1, dim2, dim3
                    )

        for seq_len in [2 * mesh.size(0) - 1, 2 * mesh.size(0) + 1]:
            for dim1 in [2 * mesh.size(1)]:
                for dim2 in [2 * mesh.size(2)]:
                    pass
                    # expect error
                    # self._test_dtensor_unflatten_3d(mesh, batch_size, seq_len, dim1, dim2, dim3)

        for seq_len in [2 * mesh.size(0)]:
            for dim1 in [2 * mesh.size(1) - 1, 2 * mesh.size(1) + 1]:
                for dim2 in [2 * mesh.size(2)]:
                    pass
                    # expect error
                    # self._test_dtensor_unflatten_3d(mesh, batch_size, seq_len, dim1, dim2, dim3)

    def _test_dtensor_unflatten_3d(self, mesh, batch_size, seq_len, dim1, dim2, dim3):
        # S1, S2, S3
        global_inps = torch.arange(batch_size * seq_len * dim1 * dim2 * dim3).view(
            batch_size * seq_len * dim1 * dim2, dim3
        )
        expected_placements = (Shard(1), Shard(2), Shard(3))
        inps = distribute_tensor(
            global_inps,
            mesh,
            (Replicate(), Replicate(), Replicate()),
        ).redistribute(
            mesh,
            (
                _StridedShard(dim=0, split_factor=batch_size),
                _StridedShard(
                    dim=0, split_factor=batch_size * (seq_len // mesh.size(0))
                ),
                _StridedShard(
                    dim=0,
                    split_factor=batch_size
                    * (seq_len // mesh.size(0))
                    * (dim1 // mesh.size(1)),
                ),
            ),
        )
        expected_inp_viewed = distribute_tensor(
            global_inps.view(batch_size, seq_len, dim1, dim2, dim3),
            mesh,
            expected_placements,
        )
        inps_viewed = inps.view(batch_size, seq_len, dim1, dim2, dim3)
        self.assertEqual(inps_viewed.placements, expected_placements)
        self.assertEqual(inps_viewed._local_tensor, expected_inp_viewed._local_tensor)


TestViewOpsWithLocalTensor = create_local_tensor_test_class(
    TestViewOps,
    skipped_tests=[
        # Comparing data pointers is not supported for local tensor
        "test_dtensor_view_op_uneven",
        "test_dtensor_flatten_1d",
        "test_dtensor_flatten_2d",
        "test_dtensor_unflatten_1d",
        "test_dtensor_unflatten_2d",
        "test_dtensor_unflatten_2d_special",
    ],
)

TestViewOps3DWithLocalTensor = create_local_tensor_test_class(
    TestViewOps3D,
)

if __name__ == "__main__":
    run_tests()
