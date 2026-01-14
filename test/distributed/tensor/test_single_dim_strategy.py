# Owner(s): ["oncall: distributed"]


from itertools import permutations
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    TupleStrategy,
)
from torch.distributed.tensor._ops._matrix_ops import mm_single_dim_strategy
from torch.distributed.tensor._ops._pointwise_ops import (
    single_mesh_dim_linear_pointwise_strategy,
)
from torch.distributed.tensor._ops._tensor_ops import cat_single_dim_strategy
from torch.distributed.tensor._ops.single_dim_strategy import (
    _expand_single_dim_strategy_to_mesh,
    _fill_single_dim_strategy_placeholders,
    _get_num_tensor_inputs,
    _get_unique_placements,
    _insert_single_dim_replication_strategy,
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._sharding_prop import _select_min_cost_strategy
from torch.distributed.tensor.placement_types import _StridedShard, Placement
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


def _get_mm_metas(M=64, K=32, N=64) -> tuple[TensorMeta, TensorMeta]:
    """
    Helper function to generate tensor metadata for matmul operation (M, K) @ (K, N) -> (M, N).
    """
    left_meta = TensorMeta(
        shape=torch.Size([M, K]),
        stride=(K, 1),
        dtype=torch.float32,
    )
    right_meta = TensorMeta(
        shape=torch.Size([K, N]),
        stride=(N, 1),
        dtype=torch.float32,
    )
    return left_meta, right_meta


def _get_mm_specs(
    mesh: DeviceMesh,
    left_meta: TensorMeta,
    right_meta: TensorMeta,
    left_placements: tuple[Placement, ...],
    right_placements: tuple[Placement, ...],
) -> tuple[DTensorSpec, DTensorSpec]:
    """
    Helper function to generate DTensorSpecs for matmul operation (M, K) @ (K, N) -> (M, N).
    """
    left_spec = DTensorSpec(
        mesh=mesh, placements=left_placements, tensor_meta=left_meta
    )
    right_spec = DTensorSpec(
        mesh=mesh, placements=right_placements, tensor_meta=right_meta
    )
    return left_spec, right_spec


class TestExpandPlaceholder(TestCase):
    def setUp(self):
        super().setUp()
        self.world_size = 8
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def test_foreach_ops_variants(self):
        mesh = DeviceMesh("cpu", mesh=torch.arange(8).reshape(2, 2, 2))

        def _test_op(op, *args, linearity=None):
            # creates specs, computes single-dim strategy, and expands to mesh
            out_spec = None
            specs = []
            for arg in args:
                if isinstance(arg, list):
                    tensor_specs = []
                    for p, t in arg:
                        spec = DTensorSpec(
                            mesh,
                            p,
                            TensorMeta(t.shape, t.stride(), t.dtype),
                        )
                        tensor_specs.append(
                            OpStrategy([OpSpec(spec)]),
                        )
                    list_spec = TupleStrategy(tuple(tensor_specs))
                    if out_spec is None:
                        out_spec = list_spec
                    specs.append(list_spec)
                else:
                    specs.append(arg)

            output_meta = [spec.tensor_meta for spec in out_spec.children]

            op_schema = OpSchema(op=op, args_schema=tuple(specs), kwargs_schema={})
            strategy_fn = single_mesh_dim_linear_pointwise_strategy(
                linearity=linearity or -1
            )
            expanded = _expand_single_dim_strategy_to_mesh(
                mesh, op_schema, strategy_fn, output_meta
            )
            strategy = expanded(op, op_schema.args_meta, op_schema.kwargs_meta)

            # check expanded strategy
            self.assertIsInstance(strategy, TupleStrategy)
            self.assertEqual(
                len(strategy.children), len(args[0])
            )  # no. of list elements
            if linearity == 1:
                self.assertEqual(
                    len(strategy.children[0].strategies), 125
                )  # len([S(0), S(1), S(2), R, P]) ** 3 = 125
            else:
                self.assertGreaterAlmostEqual(
                    len(strategy.children[0].strategies), 64
                )  # len([S(0), S(1), S(2), R]) ** 3 = 64

        t = torch.empty((8, 8, 8))
        shard0 = (Shard(0), Replicate(), Replicate())
        shard1 = (Replicate(), Shard(1), Replicate())

        # unary ops
        for op in [
            torch.ops.aten._foreach_abs.default,
            torch.ops.aten._foreach_sqrt.default,
            torch.ops.aten._foreach_exp.default,
        ]:
            _test_op(op, [(shard0, t), (shard1, t)])

        # .List variants
        for op in [
            torch.ops.aten._foreach_add.List,
            torch.ops.aten._foreach_mul.List,
            torch.ops.aten._foreach_div.List,
        ]:
            _test_op(
                op,
                [(shard0, t), (shard1, t)],
                [(shard1, t), (shard0, t)],
                linearity=(1 if "add" in str(op) else -1),
            )

        # .Scalar variants
        for op in [
            torch.ops.aten._foreach_add.Scalar,
            torch.ops.aten._foreach_mul.Scalar,
        ]:
            _test_op(op, [(shard0, t), (shard1, t)], 2.0)

        # .Tensor variant with single Tensor (list[Tensor], Tensor)
        single_tensor_spec = DTensorSpec(
            mesh,
            shard0,
            TensorMeta(t.shape, t.stride(), t.dtype),
        )
        _test_op(
            torch.ops.aten._foreach_mul.Tensor,
            [(shard0, t), (shard1, t)],
            OpStrategy([OpSpec(single_tensor_spec)]),
        )

        # .Scalar ternary variants
        for op in [
            torch.ops.aten._foreach_addcmul.Scalar,
            torch.ops.aten._foreach_addcdiv_.Scalar,
            torch.ops.aten._foreach_lerp_.Scalar,
        ]:
            _test_op(op, [(shard0, t)], [(shard0, t)], [(shard0, t)], 1.0)

        # Test inplace variant
        _test_op(
            torch.ops.aten._foreach_add_.List, [(shard0, t)], [(shard0, t)], linearity=1
        )

    def test_expand_foreach_add_to_3d_mesh(self):
        mesh = DeviceMesh("cpu", mesh=torch.arange(8).reshape(2, 2, 2))

        def _expand_foreach_add_list(
            inputs_a: list[torch.Tensor],
            inputs_b: list[torch.Tensor],
            placements_a: list[tuple[Placement, ...]],
            placements_b: list[tuple[Placement, ...]],
        ) -> TupleStrategy:
            specs_a = (
                DTensorSpec(mesh, placement, TensorMeta(t.shape, t.stride(), t.dtype))
                for placement, t in zip(placements_a, inputs_a)
            )
            specs_b = (
                DTensorSpec(mesh, placement, TensorMeta(t.shape, t.stride(), t.dtype))
                for placement, t in zip(placements_b, inputs_b)
            )
            op_schema = OpSchema(
                op=torch.ops.aten._foreach_add.List,
                args_schema=(
                    TupleStrategy(
                        tuple(OpStrategy([OpSpec(spec)]) for spec in specs_a)
                    ),
                    TupleStrategy(
                        tuple(OpStrategy([OpSpec(spec)]) for spec in specs_b)
                    ),
                ),
                kwargs_schema={},
            )
            output_tensor_meta = [
                TensorMeta(t.shape, t.stride(), t.dtype) for t in inputs_a
            ]
            expanded_strategy_fn = _expand_single_dim_strategy_to_mesh(
                mesh,
                op_schema,
                single_mesh_dim_linear_pointwise_strategy(linearity=1),
                output_tensor_meta,
            )
            strategy = expanded_strategy_fn(
                torch.ops.aten._foreach_add.List,
                op_schema.args_meta,
                op_schema.kwargs_meta,
            )
            assert isinstance(strategy, TupleStrategy)
            return strategy

        # Note: using sizes that are multiples of mesh sizes so every sharding option is valid,
        # (S0, S1, R, Psum, Pavg) ** 3 = 125
        expected_num_strategies = (125, 8)
        # Test Replicate + Shard gives Shard
        inputs_a = [torch.empty((8, 8, 8))] * 2
        placements_a = [
            (Replicate(), Shard(0), Shard(1)),
            (Partial("sum"), Partial("sum"), Partial("avg")),
        ]
        inputs_b = [torch.empty((8, 8, 8))] * 2
        placements_b = [
            (Shard(0), Replicate(), Replicate()),
            (Replicate(), Partial("sum"), Partial("sum")),
        ]
        expected_output_placements = [
            (Shard(0), Replicate(), Shard(1)),
            # P(avg) -> P(sum) is currently not supported, but could be in principle
            (Partial("sum"), Partial("sum"), Replicate()),
        ]
        tuple_strategy = _expand_foreach_add_list(
            inputs_a, inputs_b, placements_a, placements_b
        )
        self.assertEqual(len(tuple_strategy.children), 2)
        for child_i, child in enumerate(tuple_strategy.children):
            assert isinstance(child, OpStrategy)
            self.assertEqual(len(child.strategies), expected_num_strategies[child_i])

            # _select_min_cost_strategy can have multiple min-cost strategies,
            # so just assert the expected placement has equal cost.
            def sum_cost(x):
                return sum(sum(y) for y in x)

            expected = expected_output_placements[child_i]
            min_cost_strategy = _select_min_cost_strategy(child)
            for strategy in child.strategies:
                if strategy.output_spec.placements == expected:
                    self.assertEqual(
                        sum_cost(strategy.redistribute_cost),
                        sum_cost(min_cost_strategy.redistribute_cost),
                    )

    def test_expand_cat_strategy_to_3d_mesh(self):
        mesh = DeviceMesh("cpu", mesh=torch.arange(8).reshape(2, 2, 2))

        def _expand_cat(
            inputs: list[torch.Tensor], placements: list[tuple[Placement, ...]], dim=0
        ) -> OpStrategy:
            specs = (
                DTensorSpec(mesh, placement, TensorMeta(t.shape, t.stride(), t.dtype))
                for placement, t in zip(placements, inputs)
            )
            op_schema = OpSchema(
                op=torch.ops.aten.mm.default,
                args_schema=(
                    TupleStrategy(tuple(OpStrategy([OpSpec(spec)]) for spec in specs)),
                    dim,
                ),
                kwargs_schema={},
                schema_info=RuntimeSchemaInfo(1, needs_pytree=True),
            )

            # Compute output tensor_meta for cat operation
            # Cat concatenates along the specified dim, so output shape is input shapes
            # with concat dim size summed
            first_input = inputs[0]
            output_shape = list(first_input.shape)
            output_shape[dim] = sum(inp.shape[dim] for inp in inputs)
            output_meta = TensorMeta(
                shape=torch.Size(output_shape),
                stride=first_input.stride(),
                dtype=first_input.dtype,
            )

            expanded_strategy_fn = _expand_single_dim_strategy_to_mesh(
                mesh, op_schema, cat_single_dim_strategy, output_meta
            )
            strategy = expanded_strategy_fn(
                torch.ops.aten.cat.default, op_schema.args_meta, op_schema.kwargs_meta
            )
            assert isinstance(strategy, OpStrategy)
            return strategy

        # Note: using sizes that are multiples of mesh sizes so every sharding option is valid,
        # (S1, S2, Psum, R) ** mesh.ndim
        expected_num_strategies = 4**3
        # Test Replicate + Shard gives Shard
        inputs = [torch.empty((8, 8, 8))] * 2
        placements = [
            (Replicate(), Replicate(), Shard(1)),
            (Shard(1), Shard(1), Shard(1)),
        ]
        strategy = _expand_cat(inputs, placements)
        self.assertEqual(len(strategy.strategies), expected_num_strategies)
        min_cost_strategy = _select_min_cost_strategy(strategy)
        self.assertEqual(
            min_cost_strategy.output_spec.placements, (Shard(1), Shard(1), Shard(1))
        )

        # Test Partial + Partial gives Partial
        inputs = [torch.empty((8, 8, 8))] * 2
        placements = [
            (Partial("sum"), Replicate(), Shard(1)),
            (Partial("sum"), Shard(1), Shard(1)),
        ]
        strategy = _expand_cat(inputs, placements)
        self.assertEqual(len(strategy.strategies), expected_num_strategies)
        min_cost_strategy = _select_min_cost_strategy(strategy)
        self.assertEqual(
            min_cost_strategy.output_spec.placements,
            (Partial("sum"), Shard(1), Shard(1)),
        )

        # Test a large number of inputs with arbitrary placements
        # Note: currently, 10k inputs takes 23 sec. 1k inputs takes 3 sec
        num_input = 100
        inputs = [torch.empty((8, 8, 8))] * num_input
        ndim = 3
        placement_options = [Shard(i) for i in range(ndim)] + [
            Replicate(),
            Partial("sum"),
        ]
        placement_perms = list(permutations(placement_options, ndim))
        placements = [
            placement_perms[i % len(placement_perms)] for i in range(len(inputs))
        ]
        strategy = _expand_cat(inputs, placements)
        self.assertEqual(len(strategy.strategies), expected_num_strategies)

        # Test 'cant shard based on tensor dim % mesh dim' case
        inputs = [torch.empty((8, 4, 8))] * 2
        placements = [
            (Replicate(), Replicate(), Shard(1)),
            (Shard(1), Shard(1), Replicate()),
        ]
        strategy = _expand_cat(inputs, placements)
        # Only strategy filtered out should be S1S1S1.
        self.assertEqual(len(strategy.strategies), expected_num_strategies - 1)
        min_cost_strategy = _select_min_cost_strategy(strategy)
        # We can shard tensor dim 1 at most twice, then we run out of values
        self.assertEqual(
            min_cost_strategy.output_spec.placements, (Shard(1), Shard(1), Replicate())
        )
        # for i, s in enumerate(strategy.strategies):
        #     print(f"{i=}, cost={s.redistribute_cost}, {s}")

    def test_expand_matmul_like_strategy_to_3d_mesh(self):
        """Test expanding matmul-like single-dim strategies to a 3D mesh.

        This test verifies that:
        1. Single-dim matmul strategies (S0,R -> S0 and R,S1->S1) are correctly expanded to 3D
        2. The implicit full-replication rule is included
        3. _ShardingPlaceholder is correctly replaced with actual Shard placements
        4. tensor_meta is properly populated for output and input specs
        """
        mesh = DeviceMesh("cpu", mesh=torch.arange(8).reshape(2, 2, 2))
        M, K, N = 64, 32, 64
        left_meta, right_meta = _get_mm_metas(M, K, N)

        # Compute expected output tensor_meta for matmul: (M, K) @ (K, N) -> (M, N)
        output_meta = TensorMeta(
            shape=torch.Size([M, N]),
            stride=(N, 1),
            dtype=torch.float32,
        )

        # Create DTensorSpec for inputs with Shard placements using helper
        # Left input sharded on dim 0 across first mesh dim, replicated on others
        # Right input replicated across all mesh dims
        left_spec, right_spec = _get_mm_specs(
            mesh,
            left_meta,
            right_meta,
            left_placements=(Shard(0), Replicate(), Replicate()),
            right_placements=(Replicate(), Replicate(), Replicate()),
        )

        # Create OpSchema
        op_schema = OpSchema(
            op=torch.ops.aten.mm.default,
            args_schema=(
                OpStrategy([OpSpec(left_spec)]),
                OpStrategy([OpSpec(right_spec)]),
            ),
            kwargs_schema={},
        )

        # Expand the strategy to the full mesh
        expanded_strategy_fn = _expand_single_dim_strategy_to_mesh(
            mesh, op_schema, mm_single_dim_strategy, output_meta
        )
        strategy = expanded_strategy_fn(
            torch.ops.aten.matmul.default, op_schema.args_meta, op_schema.kwargs_meta
        )
        assert isinstance(strategy, OpStrategy)

        # For a 3D mesh with 4 single-dim strategies (3 explicit + 1 implicit replicate),
        # we should have 4^3 = 64 strategies maximum
        self.assertEqual(len(strategy.strategies), 64)

        all_replicate_found = False
        shard_0_found = False
        for op_spec in strategy.strategies:
            output_spec = op_spec.output_spec
            input_specs = op_spec.input_specs
            assert input_specs is not None

            # Verify tensor_meta is populated for output spec
            self.assertIsNotNone(
                output_spec.tensor_meta, "Output spec should have tensor_meta populated"
            )
            self.assertEqual(output_spec.tensor_meta.shape, torch.Size([M, N]))
            self.assertEqual(output_spec.tensor_meta.dtype, torch.float32)

            # Verify tensor_meta is populated for input specs
            self.assertIsNotNone(
                input_specs[0].tensor_meta,
                "Left input spec should have tensor_meta populated",
            )
            self.assertEqual(input_specs[0].tensor_meta.shape, torch.Size([M, K]))
            self.assertEqual(input_specs[0].tensor_meta.dtype, torch.float32)

            self.assertIsNotNone(
                input_specs[1].tensor_meta,
                "Right input spec should have tensor_meta populated",
            )
            self.assertEqual(input_specs[1].tensor_meta.shape, torch.Size([K, N]))
            self.assertEqual(input_specs[1].tensor_meta.dtype, torch.float32)

            # Check if this is the all-replicate strategy
            if (
                all(isinstance(p, Replicate) for p in output_spec.placements)
                and all(isinstance(p, Replicate) for p in input_specs[0].placements)
                and all(isinstance(p, Replicate) for p in input_specs[1].placements)
            ):
                all_replicate_found = True

            # Placeholders should have been filled
            self.assertFalse(
                any(isinstance(p, _ShardingPlaceholder) for p in output_spec.placements)
            )
            for input_spec in input_specs:
                self.assertFalse(
                    any(
                        isinstance(p, _ShardingPlaceholder)
                        for p in input_spec.placements
                    )
                )
            if any(
                isinstance(p, Shard) and p.dim == 0 for p in input_specs[0].placements
            ):
                shard_0_found = True

        self.assertTrue(
            all_replicate_found,
            "Implicit full-replication rule not found in expanded strategies",
        )
        # Verify at least one strategy has Shard(0) placement for left input
        self.assertTrue(
            shard_0_found,
            "No strategy found with Shard(0) for left input",
        )

    def test_fill_single_dim_strategy_placeholders(self):
        """Test _fill_single_dim_strategy_placeholders with different input sharding types.

        This test validates that:
        1. All-replicate inputs -> no sharding expansion (only implicit replicate rule)
        2. Shard-only inputs -> only Shard expansion
        3. Mixed Shard and _StridedShard inputs -> both types of expansion
        """
        left_meta, right_meta = _get_mm_metas()
        single_dim_strategies = mm_single_dim_strategy(
            torch.ops.aten.matmul.default, (left_meta, right_meta), {}
        )

        # Test Case 1: All-replicate inputs - no sharding expansion
        # Expected: Only implicit all-replicate rule (no sharding builders available)
        expected_replicate = [
            [Replicate(), Replicate(), Replicate()],  # Implicit all-replicate
        ]
        single_dim_strategies = _insert_single_dim_replication_strategy(
            single_dim_strategies, num_input_tensors=2
        )
        expanded_replicate = _fill_single_dim_strategy_placeholders(
            {Replicate()}, single_dim_strategies
        )

        self.assertEqual(expanded_replicate, expected_replicate)

        # Test Case 2: (_Strided)Shard-only inputs - only (_Strided)Shard expansion
        # Expected: 3 strategies with placeholders filled using (_Strided)Shard + implicit replicate
        expected_shard = [
            [Replicate(), Replicate(), Replicate()],
            [Partial(), Shard(1), Shard(0)],
            [Shard(0), Shard(0), Replicate()],
            [Shard(1), Replicate(), Shard(1)],
        ]

        expanded_shard = _fill_single_dim_strategy_placeholders(
            {Replicate(), Shard(0), Shard(1)}, single_dim_strategies
        )
        self.assertEqual(expanded_shard, expected_shard)

        expected_strided_shard = [
            [Replicate(), Replicate(), Replicate()],
            [
                Partial(),
                _StridedShard(1, split_factor=2),
                _StridedShard(0, split_factor=2),
            ],
            [
                Partial(),
                _StridedShard(1, split_factor=4),
                _StridedShard(0, split_factor=4),
            ],
            [
                _StridedShard(dim=0, split_factor=2),
                _StridedShard(dim=0, split_factor=2),
                Replicate(),
            ],
            [
                _StridedShard(dim=0, split_factor=4),
                _StridedShard(dim=0, split_factor=4),
                Replicate(),
            ],
            [
                _StridedShard(dim=1, split_factor=2),
                Replicate(),
                _StridedShard(dim=1, split_factor=2),
            ],
            [
                _StridedShard(dim=1, split_factor=4),
                Replicate(),
                _StridedShard(dim=1, split_factor=4),
            ],
        ]
        expanded_strided_shard = _fill_single_dim_strategy_placeholders(
            {
                _StridedShard(0, split_factor=2),
                _StridedShard(0, split_factor=4),
            },
            single_dim_strategies,
        )
        self.assertEqual(expanded_strided_shard, expected_strided_shard)

        # Test Case 3: Mixed Shard and _StridedShard inputs - both types of expansion
        # Expected: 3 strategies * 2 shard types (Shard and _StridedShard) + implicit replicate
        expected_mixed = [
            [Replicate(), Replicate(), Replicate()],
            [Partial(), Shard(1), Shard(0)],
            [
                Partial(),
                _StridedShard(1, split_factor=2),
                _StridedShard(0, split_factor=2),
            ],
            [Shard(0), Shard(0), Replicate()],
            [
                _StridedShard(0, split_factor=2),
                _StridedShard(0, split_factor=2),
                Replicate(),
            ],
            [Shard(1), Replicate(), Shard(1)],
            [
                _StridedShard(1, split_factor=2),
                Replicate(),
                _StridedShard(1, split_factor=2),
            ],
        ]

        expanded_mixed = _fill_single_dim_strategy_placeholders(
            {
                Replicate(),
                Shard(1),
                Shard(0),
                _StridedShard(1, split_factor=2),
                _StridedShard(0, split_factor=2),
            },
            single_dim_strategies,
        )

        self.assertEqual(expanded_mixed, expected_mixed)

    def test_opschema_hash_includes_placements(self):
        """Test OpSchema hashing includes placements for LRU cache correctness."""
        mesh = DeviceMesh("cpu", mesh=torch.arange(8).reshape(2, 2, 2))
        meta = TensorMeta(torch.Size([8, 8]), (8, 1), torch.float32)

        # Identical placements should hash the same
        spec1 = DTensorSpec(mesh, (Shard(0), Replicate(), Replicate()), meta)
        spec2 = DTensorSpec(mesh, (Shard(0), Replicate(), Replicate()), meta)
        schema1 = OpSchema(
            torch.ops.aten.add.Tensor, (OpStrategy([OpSpec(spec1)]),), {}
        )
        schema2 = OpSchema(
            torch.ops.aten.add.Tensor, (OpStrategy([OpSpec(spec2)]),), {}
        )
        self.assertEqual(hash(schema1), hash(schema2))

        # Different placements should hash differently
        spec3 = DTensorSpec(mesh, (Replicate(), Shard(1), Replicate()), meta)
        schema3 = OpSchema(
            torch.ops.aten.add.Tensor, (OpStrategy([OpSpec(spec3)]),), {}
        )
        self.assertNotEqual(hash(schema1), hash(schema3))

    def test_get_unique_placements_includes_kwargs(self):
        """Test that _get_unique_placements includes placements from kwargs (e.g., out tensor).

        This is a regression test for the fix where out-variant ops like torch.mul(..., out=...)
        were failing because the 'out' kwarg tensor placements weren't being counted.
        """
        mesh = DeviceMesh("cpu", mesh=torch.arange(4))
        meta = TensorMeta(torch.Size([8, 8]), (8, 1), torch.float32)

        # Create specs with different placements for args and kwargs
        arg_spec = DTensorSpec(mesh, (Shard(0),), meta)
        kwarg_spec = DTensorSpec(mesh, (Shard(1),), meta)

        # Create OpSchema with both args and kwargs tensors
        op_schema = OpSchema(
            op=torch.ops.aten.mul.out,
            args_schema=(
                OpStrategy([OpSpec(arg_spec)]),
                OpStrategy([OpSpec(arg_spec)]),
            ),
            kwargs_schema={"out": OpStrategy([OpSpec(kwarg_spec)])},
        )

        # _get_unique_placements should include both arg and kwarg placements
        unique_placements = _get_unique_placements(op_schema)
        self.assertIn(Shard(0), unique_placements)
        self.assertIn(Shard(1), unique_placements)

    def test_get_num_tensor_inputs_includes_kwargs(self):
        """Test that _get_num_tensor_inputs counts tensor kwargs (e.g., out tensor).

        This is a regression test for the fix where out-variant ops like torch.mul(..., out=...)
        were failing with 'input_specs(N) != strategies(M)' because the 'out' kwarg wasn't counted.
        """
        mesh = DeviceMesh("cpu", mesh=torch.arange(4))
        meta = TensorMeta(torch.Size([8, 8]), (8, 1), torch.float32)
        spec = DTensorSpec(mesh, (Shard(0),), meta)

        # Create OpSchema with 2 arg tensors and 1 kwarg tensor
        op_schema = OpSchema(
            op=torch.ops.aten.mul.out,
            args_schema=(
                OpStrategy([OpSpec(spec)]),
                OpStrategy([OpSpec(spec)]),
            ),
            kwargs_schema={"out": OpStrategy([OpSpec(spec)])},
        )

        # _get_num_tensor_inputs should count both args and kwargs tensors
        num_inputs = _get_num_tensor_inputs(op_schema)
        self.assertEqual(num_inputs, 3)  # 2 args + 1 out kwarg


@torch.library.custom_op("mylib::dummy_add", mutates_args=())
def dummy_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


@dummy_add.register_fake
def _dummy_add_fake(x, y):
    return torch.empty_like(x)


class TestSingleDimStrategyRegistration(TestCase):
    def setUp(self):
        super().setUp()
        self.world_size = 4
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )

    def tearDown(self):
        super().tearDown()
        torch.distributed.destroy_process_group()

    @patch(
        "torch.distributed.tensor._api.DTensor._op_dispatcher.sharding_propagator.op_single_dim_strategy_funcs",
        {},
    )
    def test_register_single_dim_strategy(self):
        mesh = DeviceMesh("cpu", torch.arange(self.world_size))
        x = torch.randn(8, 16)
        y = torch.randn(8, 16)

        x_dt = distribute_tensor(x, mesh, [Shard(0)])
        y_dt = distribute_tensor(y, mesh, [Shard(0)])

        with self.assertRaisesRegex(
            NotImplementedError,
            r"Operator.*dummy_add.*does not have a sharding strategy registered",
        ):
            _ = torch.ops.mylib.dummy_add(x_dt, y_dt)

        @register_single_dim_strategy(torch.ops.mylib.dummy_add.default)
        def dummy_add_single_dim_strategy(op, args_schema, kwargs_schema):
            # implicit replication only, is valid
            return []

        # Verify the strategy was registered in the mock
        self.assertIn(
            torch.ops.mylib.dummy_add.default,
            DTensor._op_dispatcher.sharding_propagator.op_single_dim_strategy_funcs,
        )

        # Now the op should run with DTensor
        torch.ops.mylib.dummy_add(x_dt, y_dt)


if __name__ == "__main__":
    run_tests()
