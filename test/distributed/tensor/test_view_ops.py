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
        """
        Test view ops that flatten multiple dimensions on a 1D mesh.

        Coverage:
        - Tensor ranks: 2D, 3D, and 4D tensors
        - Flatten ranges: All valid (flatten_start, flatten_end) pairs where
          at least 2 dimensions are flattened
        - Placements:
          - Shard(dim) for each dim within the flattened range
          - Replicate()
        - Tensor dimension sizes: Even (divisible by mesh size), uneven-smaller,
          and uneven-larger relative to mesh size, in all combinations
        - Error cases: Verifies RuntimeError for uneven sharding on non-last
          flattened dimensions

        Expected output placements:
        - Shard on flatten_start -> remains Shard(flatten_start)
        - Shard on other dims in range -> becomes _StridedShard(flatten_start, split_factor)
        - Replicate -> remains Replicate
        """
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
                                ctx = self.assertRaisesRegex(
                                    RuntimeError,
                                    "is not evenly divisible by mesh dimension",
                                )
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
                                    ctx = self.assertRaisesRegex(
                                        RuntimeError,
                                        "is not evenly divisible by mesh dimension",
                                    )
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
                        # Note: only test shard_dim1 >= shard_dim0 as the helper
                        # _get_expected_placements_ss assumes this ordering
                        for shard_dim1 in range(shard_dim0, flatten_end):
                            # Use mesh dimension sizes for proper uneven testing
                            # shard_dim0 is sharded on mesh dim 0, shard_dim1 on mesh dim 1
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
                            # For non-sharded dims, use a value divisible by both mesh sizes
                            other_dim_value = 2 * mesh.size(0) * mesh.size(1)

                            for dim0_val in dim0_values:
                                for dim1_val in dim1_values:
                                    # Build tensor_dims with appropriate values
                                    tensor_dims = [other_dim_value] * tensor_ndim
                                    tensor_dims[shard_dim0] = dim0_val
                                    if shard_dim0 != shard_dim1:
                                        tensor_dims[shard_dim1] = dim1_val
                                    # else: same dim sharded on both mesh dims, use dim0_val

                                    tensor_dims = tuple(tensor_dims)
                                    local_tensor_dims = list(tensor_dims)
                                    placements = (Shard(shard_dim0), Shard(shard_dim1))
                                    ctx = contextlib.nullcontext()
                                    # Error if shard_dim0 size not divisible by mesh dim 0 size
                                    if (
                                        local_tensor_dims[shard_dim0] % mesh.size(0)
                                        != 0
                                    ):
                                        ctx = self.assertRaisesRegex(
                                            RuntimeError,
                                            "is not evenly divisible by mesh dimension",
                                        )
                                    local_tensor_dims[shard_dim0] = local_tensor_dims[
                                        shard_dim0
                                    ] // mesh.size(0)
                                    # Error if shard_dim1 size (after first shard) not divisible
                                    # by mesh dim 1 size, unless it's the last flattened dim
                                    if local_tensor_dims[shard_dim1] % mesh.size(
                                        1
                                    ) != 0 and shard_dim1 != (flatten_end - 1):
                                        ctx = self.assertRaisesRegex(
                                            RuntimeError,
                                            "is not evenly divisible by mesh dimension",
                                        )
                                    local_tensor_dims[shard_dim1] = math.ceil(
                                        local_tensor_dims[shard_dim1]
                                        * 1.0
                                        / mesh.size(1)
                                    )
                                    with ctx:
                                        self._test_dtensor_flatten_2d_ss(
                                            tensor_dims,
                                            flatten_start,
                                            flatten_end,
                                            mesh,
                                            placements,
                                        )

                    # Replicate, Replicate
                    # Test multiple tensor dimension combinations
                    tensor_dim_values = [
                        2 * mesh.size(0) - 1,
                        2 * mesh.size(0),
                        2 * mesh.size(0) + 1,
                    ]
                    for tensor_dims in itertools.product(
                        tensor_dim_values, repeat=tensor_ndim
                    ):
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
                yield list(tensor_dims)

    @with_comms
    def test_dtensor_unflatten_1d(self):
        """
        Test unflatten (view) operations on DTensor with 1D mesh.

        Coverage:
        - Tensor dimensions: 1D, 2D, 3D, 4D tensors
        - Unflatten ranges: All valid (flatten_start, flatten_end) pairs
        - Shard positions: All dimensions within the flattened range

        Placement types tested:
        - Shard: When shard_dim == flatten_start (split_factor=1)
        - _StridedShard: When shard_dim > flatten_start (split_factor>1)
        - Replicate: Preserved through unflatten operations

        Dimension sizes tested:
        - Even: 2 * mesh.size(0) - divisible by mesh size
        - Uneven: 2 * mesh.size(0) ± 1 - not divisible by mesh size

        Error cases verified:
        - Uneven sharding on non-last dimension raises
          "is not evenly divisible by mesh dimension"
        - Uneven sharding on last dimension is allowed (becomes _StridedShard)

        Additional coverage:
        - 1D tensor unflatten: [N] -> [a, b, ...] with Shard(0) input
        - Arbitrary factorizations: Tests all non-trivial factorizations of
          dimension sizes (e.g., 36 -> (6,6), (4,9), (2,3,6), etc.)
        - Cross-dimension sharding: Sharding on dimension != unflatten dimension
          (sharded dimension passes through unchanged)
        """
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
                            # Error expected when:
                            # 1. The shard dimension is not evenly divisible by mesh size
                            # 2. AND it's NOT the last dimension in the flattened range
                            # (last dimension allows uneven sharding via _StridedShard)
                            is_uneven = (
                                tensor_dims_unflatten[shard_dim] % mesh.size(0) != 0
                            )
                            is_last_dim = shard_dim == flatten_end - 1
                            if is_uneven and not is_last_dim:
                                ctx = self.assertRaisesRegex(
                                    RuntimeError,
                                    "is not evenly divisible by mesh dimension",
                                )
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
        for tensor_ndim in [1, 2, 3, 4]:
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

            # Determine if we expect an error
            # When shard_dim == unflatten_dim, the sharding propagates to the first new dimension.
            # This works if first_factor % mesh.size(0) == 0 (even sharding on new dim).
            # If first_factor < mesh.size(0) and not divisible, we get strided shard or error.
            # If first_factor > mesh.size(0) and not divisible, we get uneven shard on new dim.
            first_factor = factors[0]

            ctx = contextlib.nullcontext()
            expect_error = False

            if shard_dim == unflatten_dim:
                # When unflatening the sharded dimension, check factor alignment
                uneven_shard = tensor_dims[shard_dim] % mesh.size(0) != 0
                first_factor_aligned = first_factor % mesh.size(0) == 0

                # Error cases:
                # 1. First factor not aligned with mesh size (too small or not divisible)
                # 2. Input dimension itself is unevenly sharded
                expect_error = not first_factor_aligned or uneven_shard
            # else: shard_dim != unflatten_dim
            # The sharded dimension is just passed through unchanged.
            # This works regardless of whether the sharded dim is even/uneven.

            if expect_error:
                ctx = self.assertRaisesRegex(
                    RuntimeError, "is not evenly divisible by mesh dimension"
                )

            with ctx:
                comm_mode = CommDebugMode()
                with comm_mode:
                    inps_viewed = inps.view(tensor_dims_unflatten)

                if expect_error:
                    continue

                # Verify results for successful cases
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

    def generate_tensor_dims_2d(
        self, tensor_ndim, flatten_start, flatten_end, shard_dim, mesh, mesh_dim_idx
    ):
        """Generate tensor dimensions for 2D mesh unflatten tests.

        Similar to generate_tensor_dims_1d but uses the appropriate mesh dimension size.
        """
        tensor_dims_unflatten = [2 * mesh.size(mesh_dim_idx) + 1] * tensor_ndim
        for tensor_dim in [
            2 * mesh.size(mesh_dim_idx) - 1,
            2 * mesh.size(mesh_dim_idx),
            2 * mesh.size(mesh_dim_idx) + 1,
        ]:
            tensor_dims_unflatten[shard_dim] = tensor_dim
            local_tensor_dims_unflatten = copy.deepcopy(tensor_dims_unflatten)
            local_tensor_dims_unflatten[shard_dim] = math.ceil(
                tensor_dim * 1.0 / mesh.size(mesh_dim_idx)
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

    @with_comms
    def test_dtensor_unflatten_2d(self):
        """
        Test unflatten (view) operations on DTensor with 2D mesh.

        Coverage:
        - Tensor dimensions: 2D, 3D, 4D tensors
        - Unflatten ranges: All valid (flatten_start, flatten_end) pairs
        - Shard positions: All dimensions within the flattened range

        Placement types tested:
        - (_StridedShard, _StridedShard): Both mesh dims shard same flattened dim,
          but map to DIFFERENT output dims after unflatten
        - (Replicate, _StridedShard): Replicate on mesh dim 0, strided shard on mesh dim 1
        - (Replicate, Replicate): Replicate on both mesh dimensions

        Note: (Shard, Shard) where both mesh dims would shard the SAME output dim
        is not supported because the view ops can't disambiguate the target.
        Plain Shard(0) on mesh dim 1 also doesn't propagate correctly through unflatten;
        we need _StridedShard with split_factor to indicate the target output dim.

        For _StridedShard placements:
        - _StridedShard(flatten_start, split_factor) -> Shard(shard_dim) after unflatten
        - split_factor encodes the output dimension to shard

        Dimension sizes tested:
        - Even: 2 * mesh.size(mesh_dim) - divisible by mesh size
        - Uneven: 2 * mesh.size(mesh_dim) ± 1 - not divisible by mesh size

        Error cases verified:
        - Uneven sharding on non-last dimension raises
          "is not evenly divisible by mesh dimension"
        - Uneven sharding on last dimension is allowed (becomes _StridedShard)
        """
        assert self.world_size == 6
        mesh: DeviceMesh = init_device_mesh(self.device_type, (3, self.world_size // 3))

        # Test (_StridedShard, _StridedShard) pattern - both mesh dims shard the same
        # flattened dim but map to DIFFERENT output dims after unflatten.
        # Uses all factorizations to comprehensively test unflatten behavior.
        # For this pattern:
        # - shard_dim0 and shard_dim1 must be adjacent (shard_dim1 = shard_dim0 + 1)
        # - We need at least 2 factors in the flatten range for two shard dims
        # - We need at least 1 factor as prefix (for split_factor > 1)
        # - We need at least 1 dim outside flatten range (view ops fail on 1D)
        #
        # We iterate over factorizations of a fixed flattened size (e.g., 144).
        # For each factorization with len >= 3, we pick adjacent indices for sharding.
        flattened_size = 144  # = 12 * 6 * 2, has many factorizations
        factorizations = self._get_all_factorizations(flattened_size)

        # Filter for factorizations with >= 3 factors (need prefix + 2 shard dims)
        valid_factorizations = [f for f in factorizations if len(f) >= 3]

        for factors in valid_factorizations:
            num_factors = len(factors)
            # shard_dim0 is at index 1..num_factors-2 (need prefix before, shard_idx1 after)
            for shard_idx0 in range(1, num_factors - 1):
                shard_idx1 = shard_idx0 + 1  # Adjacent

                # Check divisibility for error detection
                factor0 = factors[shard_idx0]
                factor1 = factors[shard_idx1]
                mesh_size0 = mesh.size(0)  # = 3
                mesh_size1 = mesh.size(1)  # = 2

                # Error only occurs for non-last split dimensions.
                # shard_idx0 is never the last (since shard_idx1 = shard_idx0 + 1 exists)
                # shard_idx1 may or may not be the last
                is_last_split_idx1 = shard_idx1 == num_factors - 1
                uneven_shard_on_mesh_dim0 = factor0 % mesh_size0 != 0
                uneven_shard_on_mesh_dim1 = (
                    factor1 % mesh_size1 != 0 and not is_last_split_idx1
                )
                expect_error = uneven_shard_on_mesh_dim0 or uneven_shard_on_mesh_dim1

                if expect_error:
                    with self.assertRaisesRegex(
                        RuntimeError, "is not evenly divisible by mesh dimension"
                    ):
                        self._test_dtensor_unflatten_2d_ss_factors(
                            factors,
                            shard_idx0,
                            shard_idx1,
                            mesh,
                        )
                else:
                    self._test_dtensor_unflatten_2d_ss_factors(
                        factors,
                        shard_idx0,
                        shard_idx1,
                        mesh,
                    )

        # Test (Replicate, _StridedShard) pattern - mesh dim 0 replicates, mesh dim 1 shards.
        # Uses all factorizations to comprehensively test unflatten behavior.
        # For this pattern:
        # - shard_idx must be > 0 (need prefix for split_factor > 1)
        # - Factor at shard_idx must be divisible by mesh.size(1) for even sharding
        #   (except for last split dim where uneven sharding is allowed)
        flattened_size_rs = 72  # = 12 * 6, has many factorizations
        factorizations_rs = self._get_all_factorizations(flattened_size_rs)

        # Filter for factorizations with >= 2 factors (need prefix + shard dim)
        valid_factorizations_rs = [f for f in factorizations_rs if len(f) >= 2]

        for factors in valid_factorizations_rs:
            num_factors = len(factors)
            # shard_idx is at index 1..num_factors-1 (need prefix before)
            for shard_idx in range(1, num_factors):
                factor = factors[shard_idx]
                mesh_size1 = mesh.size(1)  # = 2

                # Error only occurs for non-last split dimensions
                is_last_split = shard_idx == num_factors - 1
                expect_error = factor % mesh_size1 != 0 and not is_last_split

                if expect_error:
                    with self.assertRaisesRegex(
                        RuntimeError, "is not evenly divisible by mesh dimension"
                    ):
                        self._test_dtensor_unflatten_2d_rs_factors(
                            factors,
                            shard_idx,
                            mesh,
                        )
                else:
                    self._test_dtensor_unflatten_2d_rs_factors(
                        factors,
                        shard_idx,
                        mesh,
                    )

        # Test (Replicate, Replicate): unflatten should preserve Replicate placement
        # Uses all factorizations to comprehensively test that Replicate is preserved.
        flattened_size_rr = 72  # = 12 * 6, has many factorizations
        factorizations_rr = self._get_all_factorizations(flattened_size_rr)

        for factors in factorizations_rr:
            # Test with the flattened dim at different positions
            for prefix_size in [0, 1, 2]:
                tensor_dims_unflatten = [6] * prefix_size + list(factors) + [3]
                nelem_flatten = math.prod(factors)
                tensor_dims_flatten = [6] * prefix_size + [nelem_flatten] + [3]

                nelem = math.prod(tensor_dims_flatten)
                global_tensor = torch.arange(nelem).view(tensor_dims_flatten)
                dt = distribute_tensor(
                    global_tensor, mesh, (Replicate(), Replicate()), src_data_rank=None
                )

                dt_unflattened = dt.view(tensor_dims_unflatten)
                self.assertEqual(dt_unflattened.placements, (Replicate(), Replicate()))
                self.assertEqual(
                    dt_unflattened.shape, torch.Size(tensor_dims_unflatten)
                )

    def _test_dtensor_unflatten_2d_rs_factors(
        self,
        factors,
        shard_idx,
        mesh,
    ):
        """Test unflatten with (Replicate, _StridedShard) placements.

        Args:
            factors: Tuple of factors representing the unflatten shape
            shard_idx: Index within factors for mesh dim 1's shard (must be > 0)
            mesh: 2D DeviceMesh
        """
        assert shard_idx > 0, "shard_idx must be > 0 for split_factor > 1"

        # Build tensor: factors only (no dim outside flatten range needed for RS)
        tensor_dims_unflatten = list(factors)
        flatten_start = 0
        shard_dim = shard_idx

        # Flatten the range
        nelem_flatten = math.prod(factors)
        tensor_dims_flatten = [nelem_flatten]

        # Compute split_factor: product of dims before shard_dim
        split_factor = math.prod(factors[:shard_idx])
        assert split_factor > 1, f"Expected split_factor > 1, got {split_factor}"

        input_placement = _StridedShard(flatten_start, split_factor=split_factor)
        placements = (Replicate(), input_placement)
        expected_placements = (Replicate(), Shard(shard_dim))

        nelem = math.prod(tensor_dims_flatten)
        global_inps = torch.arange(nelem).view(tensor_dims_flatten)
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)
        inps_viewed = inps.view(tensor_dims_unflatten)
        self.assertEqual(inps_viewed.placements, expected_placements)

    def _test_dtensor_unflatten_2d_ss_factors(
        self,
        factors,
        shard_idx0,
        shard_idx1,
        mesh,
    ):
        """Test unflatten with (_StridedShard, _StridedShard) placements.

        Both mesh dims shard the same flattened dim but map to different output dims.

        Args:
            factors: Tuple of factors representing the unflatten shape within the
                     flatten range (e.g., (12, 6, 2) means unflatten to these dims)
            shard_idx0: Index within factors for mesh dim 0's shard (0-indexed)
            shard_idx1: Index within factors for mesh dim 1's shard (must be shard_idx0 + 1)
            mesh: 2D DeviceMesh
        """
        assert shard_idx1 == shard_idx0 + 1, "Shard indices must be adjacent"

        # Build tensor: factors for flatten range + one dim outside
        # The flatten range is [0, len(factors)), and we add one dim at the end
        tensor_dims_unflatten = list(factors) + [3]  # Add dim outside flatten range
        flatten_start = 0
        shard_dim0 = shard_idx0
        shard_dim1 = shard_idx1

        # Flatten the range
        nelem_flatten = math.prod(factors)
        tensor_dims_flatten = [nelem_flatten, 3]

        # Compute split_factors for both mesh dims
        # For mesh dim 0: product of dims before shard_dim0 (using global dims)
        split_factor0 = math.prod(factors[:shard_idx0])
        if split_factor0 == 0:
            split_factor0 = 1  # empty product

        # For mesh dim 1: product of dims before shard_dim1, with shard_dim0 localized
        dims_for_sf1 = list(factors[:shard_idx1])
        dims_for_sf1[shard_idx0] = dims_for_sf1[shard_idx0] // mesh.size(0)
        split_factor1 = math.prod(dims_for_sf1)
        if split_factor1 == 0:
            split_factor1 = 1  # handle truncation from uneven division

        input_placement0 = _StridedShard(flatten_start, split_factor=split_factor0)
        input_placement1 = _StridedShard(flatten_start, split_factor=split_factor1)
        placements = (input_placement0, input_placement1)
        expected_placements = (Shard(shard_dim0), Shard(shard_dim1))

        nelem = math.prod(tensor_dims_flatten)
        global_inps = torch.arange(nelem).view(tensor_dims_flatten)
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)

        comm_mode = CommDebugMode()
        with comm_mode:
            inps_viewed = inps.view(tensor_dims_unflatten)

        self.assertEqual(inps_viewed.placements, expected_placements)
        self.assertEqual(comm_mode.get_total_counts(), 0)

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

        # Error cases: uneven seq_len or uneven dim1
        error_cases = [
            # (seq_len_values, dim1_values) - uneven seq_len
            ([2 * mesh.size(0) - 1, 2 * mesh.size(0) + 1], [2 * mesh.size(1)]),
            # (seq_len_values, dim1_values) - uneven dim1
            ([2 * mesh.size(0)], [2 * mesh.size(1) - 1, 2 * mesh.size(1) + 1]),
        ]
        for seq_len_values, dim1_values in error_cases:
            for seq_len in seq_len_values:
                for dim1 in dim1_values:
                    for dim2 in [2 * mesh.size(2)]:
                        global_inps = torch.arange(
                            batch_size * seq_len * dim1 * dim2 * dim3
                        ).view(batch_size * seq_len * dim1 * dim2, dim3)
                        placements = (
                            _StridedShard(dim=0, split_factor=batch_size),
                            _StridedShard(
                                dim=0,
                                split_factor=batch_size * (seq_len // mesh.size(0)),
                            ),
                            _StridedShard(
                                dim=0,
                                split_factor=batch_size
                                * (seq_len // mesh.size(0))
                                * (dim1 // mesh.size(1)),
                            ),
                        )
                        inps = distribute_tensor(
                            global_inps, mesh, placements, src_data_rank=None
                        )
                        with self.assertRaisesRegex(
                            RuntimeError, "is not evenly divisible by mesh dimension"
                        ):
                            inps.view(batch_size, seq_len, dim1, dim2, dim3)

    def _test_dtensor_unflatten_3d(self, mesh, batch_size, seq_len, dim1, dim2, dim3):
        # S1, S2, S3
        global_inps = torch.arange(batch_size * seq_len * dim1 * dim2 * dim3).view(
            batch_size * seq_len * dim1 * dim2, dim3
        )
        expected_placements = (Shard(1), Shard(2), Shard(3))
        inps = distribute_tensor(
            global_inps,
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
            src_data_rank=None,
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
        "test_dtensor_unflatten_3d",
    ],
)

TestViewOps3DWithLocalTensor = create_local_tensor_test_class(
    TestViewOps3D,
)

if __name__ == "__main__":
    run_tests()
