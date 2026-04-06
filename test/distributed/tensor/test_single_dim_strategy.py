# Owner(s): ["oncall: distributed"]


from itertools import chain, permutations, product
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
    _BINARY_ADDITIVE_RULES,
    _common_pointwise_single_dim_strategy,
    _MUL_RULES,
    _UNARY_LINEAR_RULES,
)
from torch.distributed.tensor._ops._tensor_ops import cat_single_dim_strategy
from torch.distributed.tensor._ops.single_dim_strategy import (
    _dijkstra_expand_single_dim_strategy_to_mesh,
    _expand_single_dim_strategy_to_mesh,
    _fill_single_dim_strategy_placeholders,
    _get_num_tensor_inputs,
    _get_unique_placements,
    _insert_single_dim_replication_strategy,
    _ShardingPlaceholder,
    _SingleDimStrategyInfo,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy
from torch.distributed.tensor._redistribute import use_min_cost_redistribution_plan
from torch.distributed.tensor._sharding_prop import _select_min_cost_strategy
from torch.distributed.tensor.placement_types import (
    _MaskPartial,
    _StridedShard,
    Placement,
)
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
        self.world_size = 64
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def test_foreach_ops_variants(self):
        mesh = DeviceMesh("cpu", mesh=torch.arange(8).reshape(2, 2, 2))

        def _test_op(op, *args, linearity=None, inplace=None):
            # Auto-detect inplace ops by checking for "_." in the op name (e.g., _foreach_add_.List)
            if inplace is None:
                inplace = "_." in str(op)

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
            extra_rules = _BINARY_ADDITIVE_RULES if linearity == 1 else None
            strategy_fn = _common_pointwise_single_dim_strategy(
                partial_extra_rules=extra_rules
            )
            expanded = _expand_single_dim_strategy_to_mesh(
                mesh, op_schema, _SingleDimStrategyInfo(strategy_fn), output_meta
            )
            strategy = expanded(op, op_schema.args_meta, op_schema.kwargs_meta)

            # check expanded strategy
            self.assertIsInstance(strategy, TupleStrategy)
            self.assertEqual(
                len(strategy.children), len(args[0])
            )  # no. of list elements
            if inplace:
                # For inplace ops, the self argument cannot be redistributed,
                # so there should be exactly 1 strategy (the input placement)
                self.assertEqual(len(strategy.children[0].strategies), 1)
            elif linearity == 1:
                # See test_expand_foreach_add_to_3d_mesh for derivation of 634.
                self.assertEqual(len(strategy.children[0].strategies), 634)
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

        # Test inplace variant (auto-detected via "_." in op name)
        _test_op(
            torch.ops.aten._foreach_add_.List,
            [(shard0, t)],
            [(shard0, t)],
            linearity=1,
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
                _SingleDimStrategyInfo(
                    _common_pointwise_single_dim_strategy(
                        partial_extra_rules=_BINARY_ADDITIVE_RULES
                    )
                ),
                output_tensor_meta,
            )
            strategy = expanded_strategy_fn(
                torch.ops.aten._foreach_add.List,
                op_schema.args_meta,
                op_schema.kwargs_meta,
            )
            self.assertIsInstance(strategy, TupleStrategy)
            return strategy

        # Sizes are multiples of mesh dims so every shard option is valid.
        # Per mesh dim: 4 non-partial (S0,S1,S2,R) + 6 partial rules
        # (1 Psum, 3 Pavg, 1 Pmax, 1 Pmin). Mixed-partial filter allows
        # Psum+Pavg to coexist but rejects other mixes. By inclusion-exclusion
        # on a 3D mesh (n=4 non-partial, partial rule counts by type):
        #   no partials:    4^3                                    =  64
        #   Psum only:      (4+1)^3 - 4^3                         =  61
        #   Pavg only:      (4+3)^3 - 4^3                         = 279
        #   Pmax only:      (4+1)^3 - 4^3                         =  61
        #   Pmin only:      (4+1)^3 - 4^3                         =  61
        #   Psum+Pavg mix:  (4+1+3)^3 - (4+3)^3 - (4+1)^3 + 4^3  = 108
        #   total: 634
        # For scalars (no Shard), n=1 (R only): same formula gives 139.
        expected_num_strategies = (634, 139)
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
            # P(avg),P(avg),R rule lets dim 2 keep P(avg) from input_a, only
            # redistributing input_b to R (cheaper than both inputs → R)
            (Partial("sum"), Partial("sum"), Partial("avg")),
        ]
        tuple_strategy = _expand_foreach_add_list(
            inputs_a, inputs_b, placements_a, placements_b
        )
        self.assertEqual(len(tuple_strategy.children), 2)
        for child_i, child in enumerate(tuple_strategy.children):
            self.assertIsInstance(child, OpStrategy)
            self.assertEqual(len(child.strategies), expected_num_strategies[child_i])

            def sum_cost(x):
                return sum(sum(y) for y in x)

            # Multiple strategies can produce the same output placement but
            # with different input specs (and costs, including inf). Check
            # that the cheapest strategy with the expected output ties with
            # the overall minimum.
            expected = expected_output_placements[child_i]
            min_cost_strategy = _select_min_cost_strategy(child)
            min_cost = sum_cost(min_cost_strategy.redistribute_cost)
            expected_costs = [
                sum_cost(s.redistribute_cost)
                for s in child.strategies
                if s.output_spec.placements == expected
            ]
            self.assertTrue(len(expected_costs) > 0)
            self.assertAlmostEqual(min(expected_costs), min_cost, places=5)

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
                mesh,
                op_schema,
                _SingleDimStrategyInfo(cat_single_dim_strategy),
                output_meta,
            )
            strategy = expanded_strategy_fn(
                torch.ops.aten.cat.default, op_schema.args_meta, op_schema.kwargs_meta
            )
            self.assertIsInstance(strategy, OpStrategy)
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
            mesh, op_schema, _SingleDimStrategyInfo(mm_single_dim_strategy), output_meta
        )
        strategy = expanded_strategy_fn(
            torch.ops.aten.matmul.default, op_schema.args_meta, op_schema.kwargs_meta
        )
        self.assertIsInstance(strategy, OpStrategy)

        # For a 3D mesh with 8 single-dim strategies per mesh dim
        # (3 sharding + 4 per-input linearity + 1 implicit replicate),
        # we get 8^3 = 512 strategy combinations.
        self.assertEqual(len(strategy.strategies), 512)

        all_replicate_found = False
        shard_0_found = False
        for op_spec in strategy.strategies:
            output_spec = op_spec.output_spec
            input_specs = op_spec.input_specs
            self.assertIsNotNone(input_specs)

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
        # Expected: The implicit all-replicate rule plus the per-input linearity
        # strategies (which have no placeholders and pass through unchanged).
        # Strategies with placeholders are dropped since there are no shard builders.
        expected_replicate = [
            [Replicate(), Replicate(), Replicate()],
            [Partial("sum"), Partial("sum"), Replicate()],
            [Partial("sum"), Replicate(), Partial("sum")],
            [Partial("avg"), Partial("avg"), Replicate()],
            [Partial("avg"), Replicate(), Partial("avg")],
        ]
        single_dim_strategies = _insert_single_dim_replication_strategy(
            single_dim_strategies, num_outputs=1, num_input_tensors=2
        )
        expanded_replicate = _fill_single_dim_strategy_placeholders(
            {Replicate()}, single_dim_strategies
        )

        self.assertEqual(expanded_replicate, expected_replicate)

        # Test Case 2: (_Strided)Shard-only inputs - only (_Strided)Shard expansion
        # Expected: 3 strategies with placeholders filled using (_Strided)Shard,
        # plus the per-input linearity strategies (no placeholders), plus implicit replicate
        expected_shard = [
            [Replicate(), Replicate(), Replicate()],
            [Partial(), Shard(1), Shard(0)],
            [Shard(0), Shard(0), Replicate()],
            [Shard(1), Replicate(), Shard(1)],
            [Partial("sum"), Partial("sum"), Replicate()],
            [Partial("sum"), Replicate(), Partial("sum")],
            [Partial("avg"), Partial("avg"), Replicate()],
            [Partial("avg"), Replicate(), Partial("avg")],
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
            # Per-input linearity strategies (no placeholders, pass through unchanged)
            [Partial("sum"), Partial("sum"), Replicate()],
            [Partial("sum"), Replicate(), Partial("sum")],
            [Partial("avg"), Partial("avg"), Replicate()],
            [Partial("avg"), Replicate(), Partial("avg")],
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
        # Expected: 3 strategies * 2 shard types (Shard and _StridedShard),
        # plus per-input linearity strategies, plus implicit replicate
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
            # Per-input linearity strategies (no placeholders, pass through unchanged)
            [Partial("sum"), Partial("sum"), Replicate()],
            [Partial("sum"), Replicate(), Partial("sum")],
            [Partial("avg"), Partial("avg"), Replicate()],
            [Partial("avg"), Replicate(), Partial("avg")],
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

    def test_expand_strategy_handles_symbolic_shapes(self):
        """Test that _create_expanded_strategy handles symbolic shapes (SymInts).
        When using dynamic shapes with torch.compile, TensorMeta may contain SymInts
        which are not hashable. This test verifies that the caching logic gracefully
        falls back to uncached execution instead of raising TypeError.
        """
        from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymNode

        mesh = DeviceMesh("cpu", mesh=torch.arange(4))

        # Create a ShapeEnv and symbolic size
        shape_env = ShapeEnv()
        from torch._dynamo.source import ConstantSource

        sym_size = 8
        symbol = shape_env.create_symbol(
            sym_size, source=ConstantSource("test_sym_size")
        )
        sym_int = torch.SymInt(SymNode(symbol, shape_env, int, hint=sym_size))

        # Create TensorMeta with symbolic shape - this contains unhashable SymInts
        symbolic_shape = torch.Size([sym_int, 4])
        symbolic_meta = TensorMeta(symbolic_shape, (4, 1), torch.float32)

        # Verify that the symbolic TensorMeta is indeed unhashable
        with self.assertRaises(TypeError):
            hash(symbolic_meta)

        # Create a regular (hashable) TensorMeta and spec
        regular_meta = TensorMeta(torch.Size([8, 4]), (4, 1), torch.float32)
        regular_spec = DTensorSpec(mesh, (Shard(0),), regular_meta)

        # Create OpSchema with regular (hashable) specs but symbolic output_tensor_meta
        op_schema = OpSchema(
            op=torch.ops.aten.mul.Tensor,
            args_schema=(
                OpStrategy([OpSpec(regular_spec)]),
                OpStrategy([OpSpec(regular_spec)]),
            ),
            kwargs_schema={},
        )

        single_mesh_dim_strategies = [
            [Shard(0), Shard(0), Shard(0)],
            [Replicate(), Replicate(), Replicate()],
        ]

        # This should work without raising TypeError because the caching
        # gracefully falls back to uncached execution when output_tensor_meta
        # contains unhashable SymInts
        result = expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            output_tensor_meta=symbolic_meta,
        )

        # Verify result is valid
        self.assertIsInstance(result, OpStrategy)
        self.assertGreater(len(result.strategies), 0)

    def test_strategy_length_validation(self):
        """Test that _PreparedSingleDimStrategy validates strategy length against
        the op schema. Strategies must include placements for all outputs, args,
        and tensor kwargs. A strategy missing the out kwarg placement should fail.
        """
        from torch.distributed.tensor._ops.single_dim_strategy import (
            _PreparedSingleDimStrategy,
        )

        mesh = DeviceMesh("cpu", mesh=torch.arange(4))
        meta = TensorMeta(torch.Size([8, 8]), (8, 1), torch.float32)

        arg_spec = DTensorSpec(mesh, (Shard(0),), meta)
        out_spec = DTensorSpec(mesh, (Shard(0),), meta)

        op_schema = OpSchema(
            op=torch.ops.aten.mul.out,
            args_schema=(
                OpStrategy([OpSpec(arg_spec)]),
                OpStrategy([OpSpec(arg_spec)]),
            ),
            kwargs_schema={"out": OpStrategy([OpSpec(out_spec)])},
        )

        # Strategy missing the out kwarg placement: [output, arg1, arg2] = 3
        # but mul.out has 1 output + 3 inputs (2 args + 1 out kwarg) = 4 expected
        def bad_strategy(op, args, kwargs):
            return [[Shard(0), Shard(0), Shard(0)]]

        with self.assertRaisesRegex(AssertionError, r"Strategy length 3 != expected 4"):
            _PreparedSingleDimStrategy(bad_strategy, op_schema, meta)

        # Strategy with correct length: [output, arg1, arg2, out_kwarg] = 4
        def good_strategy(op, args, kwargs):
            return [[Shard(0), Shard(0), Shard(0), Shard(0)]]

        # Should not raise
        prepared = _PreparedSingleDimStrategy(good_strategy, op_schema, meta)
        self.assertEqual(prepared.num_outputs, 1)
        self.assertEqual(prepared.num_inputs, 3)

    def test_out_variant_partial_propagation(self):
        """Test that partial rules work correctly for .out variant ops.

        For mul.out with rule [P(sum), P(sum), R, P(sum)]
        (output=P(sum), arg1=P(sum), arg2=R, out_kwarg=P(sum)),
        the out kwarg should get P(sum) matching the output.
        """
        mesh = DeviceMesh("cpu", mesh=torch.arange(4))
        meta = TensorMeta(torch.Size([8, 8]), (8, 1), torch.float32)

        partial_spec = DTensorSpec(mesh, (Partial("sum"),), meta)
        replicate_spec = DTensorSpec(mesh, (Replicate(),), meta)

        op_schema = OpSchema(
            op=torch.ops.aten.mul.out,
            args_schema=(
                OpStrategy([OpSpec(partial_spec)]),
                OpStrategy([OpSpec(replicate_spec)]),
            ),
            kwargs_schema={"out": OpStrategy([OpSpec(partial_spec)])},
        )

        # [output, arg1, arg2, out_kwarg] — includes out kwarg placement
        single_mesh_dim_strategies = [
            [Partial("sum"), Partial("sum"), Replicate(), Partial("sum")],
            [Replicate(), Replicate(), Replicate(), Replicate()],
        ]

        result = expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            output_tensor_meta=meta,
            input_index=1,
        )

        self.assertIsInstance(result, OpStrategy)
        # Should have strategies where output=P(sum) and out_kwarg=P(sum)
        found_partial_strategy = False
        for strategy in result.strategies:
            if strategy.output_spec.placements == (Partial("sum"),):
                found_partial_strategy = True
                # The out kwarg (last input) must match the output
                out_kwarg_placement = strategy.input_specs[2].placements
                self.assertEqual(
                    out_kwarg_placement,
                    (Partial("sum"),),
                    "out kwarg should be P(sum) matching the output, not R",
                )
        self.assertTrue(
            found_partial_strategy,
            "Expected a strategy with Partial('sum') output for mul.out",
        )

    def test_expand_multi_output_strategy(self):
        """Test expanding single-dim strategies for multi-output ops.

        Uses aten.topk (2 tensor outputs, 1 tensor input) to verify that
        multi-output strategies are expanded correctly with proper output specs.
        """
        mesh = DeviceMesh("cpu", mesh=torch.arange(4))

        input_meta = TensorMeta(
            shape=torch.Size([8, 4]),
            stride=(4, 1),
            dtype=torch.float32,
        )

        # topk returns (values, indices) — 2 outputs
        output_metas = (
            TensorMeta(torch.Size([8, 2]), (2, 1), torch.float32),
            TensorMeta(torch.Size([8, 2]), (2, 1), torch.int64),
        )

        input_spec = DTensorSpec(
            mesh=mesh,
            placements=(Shard(0),),
            tensor_meta=input_meta,
        )

        op_schema = OpSchema(
            op=torch.ops.aten.topk.default,
            args_schema=(OpStrategy([OpSpec(input_spec)]), 2, -1, True, True),
            kwargs_schema={},
        )

        # 2 outputs + 1 input = 3 placements per strategy
        def mock_multi_output_strategy(op, args_schema, kwargs_schema):
            return [
                [Partial(), Partial(), Shard(0)],
            ]

        expanded_strategy_fn = _expand_single_dim_strategy_to_mesh(
            mesh,
            op_schema,
            _SingleDimStrategyInfo(mock_multi_output_strategy),
            output_metas,
        )
        strategy = expanded_strategy_fn(
            torch.ops.aten.topk.default,
            op_schema.args_meta,
            op_schema.kwargs_meta,
        )

        self.assertIsInstance(strategy, OpStrategy)
        self.assertGreaterEqual(len(strategy.strategies), 1)

        for op_spec in strategy.strategies:
            output_specs = op_spec.output_specs
            self.assertIsInstance(
                output_specs, tuple, "Multi-output op should have tuple output_specs"
            )
            self.assertEqual(
                len(output_specs), 2, "Should have 2 output specs for topk"
            )
            for i, out_spec in enumerate(output_specs):
                self.assertIsNotNone(out_spec, f"Output {i} spec should not be None")
                self.assertIsInstance(out_spec, DTensorSpec)

            self.assertEqual(len(op_spec.input_specs), 1, "Should have 1 input tensor")

    def test_inplace_op_partial_input_raises_clear_error(self):
        """Test that inplace ops with Partial input raise a clear error.

        When an inplace op (like clamp_) is called on a tensor with Partial placement,
        and no valid strategy preserves that placement (because clamp doesn't support
        Partial), we should raise a clear error message instead of a cryptic
        "min() arg is an empty sequence" error.

        This tests the fix in expand_to_full_mesh_op_strategy that detects when all
        strategies are filtered out due to inplace placement mismatch.
        """
        mesh = DeviceMesh("cpu", mesh=torch.arange(4))

        # Create a 0-dimensional (scalar) tensor with Partial placement
        # This is a minimal case where there are no Shard dimensions available
        input_meta = TensorMeta(
            shape=torch.Size([]),  # scalar tensor
            stride=(),
            dtype=torch.float32,
        )

        # Create input spec with Partial placement
        input_spec = DTensorSpec(
            mesh=mesh,
            placements=(Partial(),),
            tensor_meta=input_meta,
        )

        # Create OpSchema for an inplace op (clamp_)
        op_schema = OpSchema(
            op=torch.ops.aten.clamp_.default,
            args_schema=(OpStrategy([OpSpec(input_spec)]),),
            kwargs_schema={},
        )

        # Define a single-dim strategy that only supports Replicate
        # (like clamp which doesn't preserve Partial)
        def mock_pointwise_strategy(op, args_schema, kwargs_schema):
            # For a scalar (0-dim) tensor, there are no Shard strategies
            # Only the implicit Replicate strategy will be added
            return []

        # This should raise a clear error about inplace ops not supporting
        # placement changes, not a cryptic "min() arg is an empty sequence"
        with self.assertRaisesRegex(
            RuntimeError,
            "in-place operations that require placement changes are not supported",
        ):
            expanded_strategy_fn = _expand_single_dim_strategy_to_mesh(
                mesh,
                op_schema,
                _SingleDimStrategyInfo(mock_pointwise_strategy),
                input_meta,
            )
            expanded_strategy_fn(
                torch.ops.aten.clamp_.default,
                op_schema.args_meta,
                op_schema.kwargs_meta,
            )

    def test_expand_filters_mixed_partial_types(self):
        """Test that expand_to_full_mesh_op_strategy filters out mixed partial types.

        When single-dim strategies are expanded to a multi-dimensional mesh, some
        combinations could create specs with mixed Partial reduce types (e.g.,
        Partial("sum") and Partial("max") in the same placement list). These
        combinations should be filtered out since mixed partial types don't commute.

        The exception is sum+avg which DO commute and should be allowed.
        """
        mesh = DeviceMesh("cpu", mesh=torch.arange(4).reshape(2, 2))
        meta = TensorMeta(torch.Size([8, 8]), (8, 1), torch.float32)

        # Create input spec with Replicate placement
        input_spec = DTensorSpec(mesh, (Replicate(), Replicate()), meta)

        # Create OpSchema
        op_schema = OpSchema(
            op=torch.ops.aten.mul.Tensor,
            args_schema=(
                OpStrategy([OpSpec(input_spec)]),
                OpStrategy([OpSpec(input_spec)]),
            ),
            kwargs_schema={},
        )

        # Define strategies that would create mixed partials when expanded:
        # - Strategy 1: Partial("sum") for all tensors
        # - Strategy 2: Partial("max") for all tensors
        # When expanded to 2D mesh, combinations like (P_sum, P_max) should be filtered
        single_mesh_dim_strategies = [
            [Partial("sum"), Partial("sum"), Partial("sum")],
            [Partial("max"), Partial("max"), Partial("max")],
            [Replicate(), Replicate(), Replicate()],
        ]

        result = expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            output_tensor_meta=meta,
        )

        # Verify no strategy has mixed partial types (except sum+avg)
        for strategy in result.strategies:
            output_spec = strategy.output_spec
            partial_reduce_ops = {
                p.reduce_op for p in output_spec.placements if isinstance(p, Partial)
            }
            # Either 0 or 1 partial type, or exactly {"sum", "avg"}
            if len(partial_reduce_ops) > 1:
                self.assertEqual(
                    partial_reduce_ops,
                    {"sum", "avg"},
                    f"Found invalid mixed partials: {partial_reduce_ops}",
                )

        # Verify that homogeneous partial strategies ARE included
        # (P_sum, P_sum) and (P_max, P_max) should be valid
        found_all_sum = False
        found_all_max = False
        for strategy in result.strategies:
            output_spec = strategy.output_spec
            if all(
                isinstance(p, Partial) and p.reduce_op == "sum"
                for p in output_spec.placements
            ):
                found_all_sum = True
            if all(
                isinstance(p, Partial) and p.reduce_op == "max"
                for p in output_spec.placements
            ):
                found_all_max = True

        self.assertTrue(found_all_sum, "Should include (P_sum, P_sum) strategy")
        self.assertTrue(found_all_max, "Should include (P_max, P_max) strategy")

    def test_expand_filters_partial_subclass_with_same_reduce_op(self):
        """Partial subclasses with the same reduce_op should still be treated as mixed."""
        mesh = DeviceMesh("cpu", mesh=torch.arange(4).reshape(2, 2))
        meta = TensorMeta(torch.Size([8, 8]), (8, 1), torch.float32)
        input_spec = DTensorSpec(mesh, (Replicate(), Replicate()), meta)
        op_schema = OpSchema(
            op=torch.ops.aten.mul.Tensor,
            args_schema=(
                OpStrategy([OpSpec(input_spec)]),
                OpStrategy([OpSpec(input_spec)]),
            ),
            kwargs_schema={},
        )

        # _MaskPartial() defaults to reduce_op="sum", same as Partial("sum"),
        # but they have different reduction semantics and should not be mixed.
        single_mesh_dim_strategies = [
            [Partial("sum"), Partial("sum"), Partial("sum")],
            [_MaskPartial(), _MaskPartial(), _MaskPartial()],
            [Replicate(), Replicate(), Replicate()],
        ]

        result = expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            output_tensor_meta=meta,
        )

        for strategy in result.strategies:
            output_spec = strategy.output_spec
            partial_types = {
                type(p) for p in output_spec.placements if isinstance(p, Partial)
            }
            self.assertLessEqual(
                len(partial_types),
                1,
                f"Should not mix Partial subclasses: {output_spec.placements}",
            )

    def test_expand_allows_sum_avg_partial_mix(self):
        """Test that sum+avg partial mix is allowed since they commute."""
        mesh = DeviceMesh("cpu", mesh=torch.arange(4).reshape(2, 2))
        meta = TensorMeta(torch.Size([8, 8]), (8, 1), torch.float32)

        input_spec = DTensorSpec(mesh, (Replicate(), Replicate()), meta)

        op_schema = OpSchema(
            op=torch.ops.aten.mul.Tensor,
            args_schema=(
                OpStrategy([OpSpec(input_spec)]),
                OpStrategy([OpSpec(input_spec)]),
            ),
            kwargs_schema={},
        )

        # Define strategies with sum and avg partials
        single_mesh_dim_strategies = [
            [Partial("sum"), Partial("sum"), Partial("sum")],
            [Partial("avg"), Partial("avg"), Partial("avg")],
            [Replicate(), Replicate(), Replicate()],
        ]

        result = expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            output_tensor_meta=meta,
        )

        # Verify that (P_sum, P_avg) combinations ARE included
        found_sum_avg_mix = False
        for strategy in result.strategies:
            output_spec = strategy.output_spec
            partial_reduce_ops = {
                p.reduce_op for p in output_spec.placements if isinstance(p, Partial)
            }
            if partial_reduce_ops == {"sum", "avg"}:
                found_sum_avg_mix = True
                break

        self.assertTrue(
            found_sum_avg_mix,
            "Should include mixed (P_sum, P_avg) strategies since sum+avg commute",
        )


class TestDijkstraExpandSingleDimStrategy(TestCase):
    def setUp(self):
        super().setUp()
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=64, store=store)

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def _get_mm_output_meta(self, M=64, K=32, N=64):
        return TensorMeta(
            shape=torch.Size([M, N]),
            stride=(N, 1),
            dtype=torch.float32,
        )

    def _compare_pq_vs_full_expansion(self, mesh, left_placements, right_placements):
        """Run both PQ search and full expansion, verify same min cost."""
        M, K, N = 64, 32, 64
        left_meta, right_meta = _get_mm_metas(M, K, N)
        output_meta = self._get_mm_output_meta(M, K, N)
        left_spec, right_spec = _get_mm_specs(
            mesh, left_meta, right_meta, left_placements, right_placements
        )

        op_schema = OpSchema(
            op=torch.ops.aten.mm.default,
            args_schema=(
                OpStrategy([OpSpec(left_spec)]),
                OpStrategy([OpSpec(right_spec)]),
            ),
            kwargs_schema={},
        )

        # PQ search
        pq_strategy = _dijkstra_expand_single_dim_strategy_to_mesh(
            mesh, op_schema, mm_single_dim_strategy, output_tensor_meta=output_meta
        )
        self.assertIsNotNone(pq_strategy)
        self.assertIsInstance(pq_strategy, OpStrategy)
        self.assertEqual(len(pq_strategy.strategies), 1)
        pq_cost = sum(chain.from_iterable(pq_strategy.strategies[0].redistribute_cost))

        # Full expansion reference using graph-based (min-cost) redistribution
        # planning. PQ's Dijkstra search over all per-dim transition orderings
        # finds the globally optimal ordering (accounting for comm_bytes updates),
        # which can be strictly cheaper than even the graph-based planner.
        with use_min_cost_redistribution_plan():
            expanded_strategy_fn = _expand_single_dim_strategy_to_mesh(
                mesh,
                op_schema,
                _SingleDimStrategyInfo(mm_single_dim_strategy),
                output_meta,
            )
            ref_strategy = expanded_strategy_fn(
                torch.ops.aten.mm.default,
                op_schema.args_meta,
                op_schema.kwargs_meta,
            )
        ref_min_cost = min(
            sum(chain.from_iterable(s.redistribute_cost))
            for s in ref_strategy.strategies
        )

        self.assertLessEqual(
            pq_cost,
            ref_min_cost + 1e-9,
            msg=(
                f"PQ cost {pq_cost} > ref min cost {ref_min_cost} for "
                f"left={left_placements}, right={right_placements}"
            ),
        )

    def test_dijkstra_expand_single_dim_strategy_to_mesh_basic(self):
        mesh = DeviceMesh("cpu", mesh=torch.arange(8).reshape(2, 2, 2))
        left_meta, right_meta = _get_mm_metas()
        output_meta = self._get_mm_output_meta()
        left_spec, right_spec = _get_mm_specs(
            mesh,
            left_meta,
            right_meta,
            left_placements=(Shard(0), Replicate(), Replicate()),
            right_placements=(Replicate(), Replicate(), Replicate()),
        )

        op_schema = OpSchema(
            op=torch.ops.aten.mm.default,
            args_schema=(
                OpStrategy([OpSpec(left_spec)]),
                OpStrategy([OpSpec(right_spec)]),
            ),
            kwargs_schema={},
        )

        strategy = _dijkstra_expand_single_dim_strategy_to_mesh(
            mesh, op_schema, mm_single_dim_strategy, output_tensor_meta=output_meta
        )

        self.assertIsInstance(strategy, OpStrategy)
        self.assertEqual(len(strategy.strategies), 1)

        op_spec = strategy.strategies[0]
        output_spec = op_spec.output_spec
        input_specs = op_spec.input_specs

        # Zero-cost match: left stays Shard(0), right stays Replicate
        self.assertEqual(output_spec.placements, (Shard(0), Replicate(), Replicate()))
        self.assertEqual(
            input_specs[0].placements, (Shard(0), Replicate(), Replicate())
        )
        self.assertEqual(
            input_specs[1].placements, (Replicate(), Replicate(), Replicate())
        )
        # Verify output has tensor_meta
        self.assertIsNotNone(output_spec.tensor_meta)
        self.assertEqual(output_spec.tensor_meta.shape, torch.Size([64, 64]))

    def test_dijkstra_expand_single_dim_strategy_to_mesh_hard(self):
        """Verify PQ search matches full expansion min cost on 3D mesh."""
        mesh = DeviceMesh("cpu", mesh=torch.arange(8).reshape(2, 2, 2))
        self._compare_pq_vs_full_expansion(
            mesh,
            left_placements=(Shard(0), Replicate(), Replicate()),
            right_placements=(Shard(1), Shard(0), Replicate()),
        )

    def test_dijkstra_expand_single_dim_strategy_to_mesh_hard_4d(self):
        """Verify PQ search matches full expansion min cost on 4D mesh."""
        mesh = DeviceMesh("cpu", mesh=torch.arange(16).reshape(2, 2, 2, 2))
        self._compare_pq_vs_full_expansion(
            mesh,
            left_placements=(Replicate(), Shard(0), Replicate(), Replicate()),
            right_placements=(Shard(0), Shard(1), Shard(0), Replicate()),
        )

    def test_pq_vs_full_expansion_data_driven(self):
        """Data-driven comparison across mesh shapes, all placement types for mm.

        Enumerates all combos of R, S(0), S(1), P(sum) for each mesh dim on
        both inputs, across 1D/2D/3D meshes.
        """
        placement_options = [Shard(0), Shard(1), Replicate(), Partial("sum")]
        mesh_configs = [
            ("1d", torch.arange(4)),
            ("2d", torch.arange(4).reshape(2, 2)),
            ("3d", torch.arange(8).reshape(2, 2, 2)),
        ]

        for mesh_name, mesh_tensor in mesh_configs:
            mesh = DeviceMesh("cpu", mesh=mesh_tensor)
            ndim = mesh.ndim

            all_placements = list(product(placement_options, repeat=ndim))

            for left_pl in all_placements:
                for right_pl in all_placements:
                    with self.subTest(mesh=mesh_name, left=left_pl, right=right_pl):
                        self._compare_pq_vs_full_expansion(mesh, left_pl, right_pl)

    def test_strided_shard_fallback(self):
        """Verify PQ search returns None for StridedShard inputs."""
        mesh = DeviceMesh("cpu", mesh=torch.arange(4).reshape(2, 2))
        left_meta, right_meta = _get_mm_metas()
        output_meta = self._get_mm_output_meta()
        left_spec = DTensorSpec(
            mesh,
            (_StridedShard(0, split_factor=2), Replicate()),
            tensor_meta=left_meta,
        )
        right_spec = DTensorSpec(
            mesh,
            (Replicate(), Replicate()),
            tensor_meta=right_meta,
        )

        op_schema = OpSchema(
            op=torch.ops.aten.mm.default,
            args_schema=(
                OpStrategy([OpSpec(left_spec)]),
                OpStrategy([OpSpec(right_spec)]),
            ),
            kwargs_schema={},
        )

        result = _dijkstra_expand_single_dim_strategy_to_mesh(
            mesh, op_schema, mm_single_dim_strategy, output_tensor_meta=output_meta
        )
        self.assertIsNone(result)

    def test_if_elif_dead_end_fix(self):
        """Counterexample: state has R dims but chunking is useless, needs allgather.

        On a 2D mesh with strategies [R, S(0) -> S(0)] + [R, R -> R],
        input state left=(S(0), R), right=(R, S(0)).
        Optimal = 1 allgather on right dim1: left=(S(0), R), right=(R, R)
        -> matches [S(0),R->S(0)] on dim0 and [R,R->R] on dim1, cost = 1 allgather.

        Old if/elif code would try to chunk first (since some dims are R),
        miss the direct allgather path, and find a 2-allgather solution.
        """
        mesh = DeviceMesh("cpu", mesh=torch.arange(4).reshape(2, 2))
        M, K, N = 64, 32, 64
        left_meta, right_meta = _get_mm_metas(M, K, N)
        output_meta = self._get_mm_output_meta(M, K, N)

        left_spec = DTensorSpec(
            mesh,
            (Shard(0), Replicate()),
            tensor_meta=left_meta,
        )
        right_spec = DTensorSpec(
            mesh,
            (Replicate(), Shard(0)),
            tensor_meta=right_meta,
        )

        op_schema = OpSchema(
            op=torch.ops.aten.mm.default,
            args_schema=(
                OpStrategy([OpSpec(left_spec)]),
                OpStrategy([OpSpec(right_spec)]),
            ),
            kwargs_schema={},
        )

        strategy = _dijkstra_expand_single_dim_strategy_to_mesh(
            mesh, op_schema, mm_single_dim_strategy, output_tensor_meta=output_meta
        )
        self.assertIsNotNone(strategy)
        self.assertIsInstance(strategy, OpStrategy)
        self.assertEqual(len(strategy.strategies), 1)

        # The optimal strategy should have the same cost as full expansion
        pq_cost = sum(chain.from_iterable(strategy.strategies[0].redistribute_cost))

        # Full expansion reference
        expanded_strategy_fn = _expand_single_dim_strategy_to_mesh(
            mesh,
            op_schema,
            _SingleDimStrategyInfo(mm_single_dim_strategy),
            output_meta,
        )
        ref_strategy = expanded_strategy_fn(
            torch.ops.aten.mm.default,
            op_schema.args_meta,
            op_schema.kwargs_meta,
        )
        ref_min_cost = min(
            sum(chain.from_iterable(s.redistribute_cost))
            for s in ref_strategy.strategies
        )

        self.assertAlmostEqual(pq_cost, ref_min_cost, places=5)

    def test_transitions_tracked(self):
        """Verify that PQ search tracks transitions for TransformInfo."""
        mesh = DeviceMesh("cpu", mesh=torch.arange(4).reshape(2, 2))
        left_meta, right_meta = _get_mm_metas()
        output_meta = self._get_mm_output_meta()

        # Use placements that require redistribution to verify transitions are tracked
        left_spec = DTensorSpec(
            mesh,
            (Shard(0), Shard(0)),
            tensor_meta=left_meta,
        )
        right_spec = DTensorSpec(
            mesh,
            (Replicate(), Replicate()),
            tensor_meta=right_meta,
        )

        op_schema = OpSchema(
            op=torch.ops.aten.mm.default,
            args_schema=(
                OpStrategy([OpSpec(left_spec)]),
                OpStrategy([OpSpec(right_spec)]),
            ),
            kwargs_schema={},
        )

        strategy = _dijkstra_expand_single_dim_strategy_to_mesh(
            mesh, op_schema, mm_single_dim_strategy, output_tensor_meta=output_meta
        )
        self.assertIsNotNone(strategy)

        # The _pq_transitions attribute should exist
        transitions = getattr(strategy, "_pq_transitions", None)
        self.assertIsNotNone(transitions)
        self.assertIsInstance(transitions, list)
        # Each transition should be (input_idx, mesh_dim, src, dst)
        for t in transitions:
            self.assertEqual(len(t), 4)

    def test_pq_reachability_collect_all_2d(self):
        """On a 2D mesh, verify PQ search can reach all valid strategies.

        For every starting placement combo, run _dijkstra_expand_single_dim_strategy_to_mesh with
        _collect_all_matches and union all collected sets. Assert the full
        expansion reference set is a subset of the collected union.
        """
        from itertools import product

        mesh = DeviceMesh("cpu", mesh=torch.arange(4).reshape(2, 2))
        M, K, N = 64, 32, 64
        left_meta, right_meta = _get_mm_metas(M, K, N)
        output_meta = self._get_mm_output_meta(M, K, N)

        placement_options = [Shard(0), Shard(1), Replicate(), Partial("sum")]
        all_placements = list(product(placement_options, repeat=mesh.ndim))

        # Get full expansion reference set
        ref_left_spec, ref_right_spec = _get_mm_specs(
            mesh,
            left_meta,
            right_meta,
            left_placements=(Replicate(), Replicate()),
            right_placements=(Replicate(), Replicate()),
        )
        wrapped_schema = OpSchema(
            op=torch.ops.aten.mm.default,
            args_schema=(
                OpStrategy([OpSpec(ref_left_spec)]),
                OpStrategy([OpSpec(ref_right_spec)]),
            ),
            kwargs_schema={},
        )
        expanded_fn = _expand_single_dim_strategy_to_mesh(
            mesh,
            wrapped_schema,
            _SingleDimStrategyInfo(mm_single_dim_strategy),
            output_meta,
        )
        ref_strategy = expanded_fn(
            torch.ops.aten.mm.default,
            wrapped_schema.args_meta,
            wrapped_schema.kwargs_meta,
        )
        full_expansion_set = {
            tuple(spec.placements for spec in s.input_specs)
            for s in ref_strategy.strategies
        }

        # Collect all reachable matches from every starting placement
        collected_union: set[tuple[tuple[Placement, ...], ...]] = set()
        for left_pl in all_placements:
            for right_pl in all_placements:
                left_spec, right_spec = _get_mm_specs(
                    mesh,
                    left_meta,
                    right_meta,
                    left_pl,
                    right_pl,
                )
                op_schema = OpSchema(
                    op=torch.ops.aten.mm.default,
                    args_schema=(
                        OpStrategy([OpSpec(left_spec)]),
                        OpStrategy([OpSpec(right_spec)]),
                    ),
                    kwargs_schema={},
                )
                matches: set[tuple[tuple[Placement, ...], ...]] = set()
                _dijkstra_expand_single_dim_strategy_to_mesh(
                    mesh,
                    op_schema,
                    mm_single_dim_strategy,
                    output_tensor_meta=output_meta,
                    _collect_all_matches=matches,
                )
                collected_union.update(matches)

        missing = full_expansion_set - collected_union
        self.assertEqual(
            missing,
            set(),
            f"PQ search missed {len(missing)} strategies reachable by full expansion",
        )

    def test_single_dim_transition_reachability(self):
        """Verify single-dim transition rules form a connected graph.

        For mm on a 2D mesh, collect all placements that appear per input
        position, build a directed graph from transition rules, and BFS from
        each placement to assert all others are reachable.
        """
        from collections import deque

        mesh = DeviceMesh("cpu", mesh=torch.arange(4).reshape(2, 2))
        M, K, N = 64, 32, 64
        left_meta, right_meta = _get_mm_metas(M, K, N)
        output_meta = self._get_mm_output_meta(M, K, N)

        ref_left_spec, ref_right_spec = _get_mm_specs(
            mesh,
            left_meta,
            right_meta,
            left_placements=(Replicate(), Replicate()),
            right_placements=(Replicate(), Replicate()),
        )
        wrapped_schema = OpSchema(
            op=torch.ops.aten.mm.default,
            args_schema=(
                OpStrategy([OpSpec(ref_left_spec)]),
                OpStrategy([OpSpec(ref_right_spec)]),
            ),
            kwargs_schema={},
        )
        expanded_fn = _expand_single_dim_strategy_to_mesh(
            mesh,
            wrapped_schema,
            _SingleDimStrategyInfo(mm_single_dim_strategy),
            output_meta,
        )
        ref_strategy = expanded_fn(
            torch.ops.aten.mm.default,
            wrapped_schema.args_meta,
            wrapped_schema.kwargs_meta,
        )

        # Collect all placements per input position per mesh dim
        for input_idx in range(2):
            for mesh_dim in range(mesh.ndim):
                all_placements: set[Placement] = set()
                for s in ref_strategy.strategies:
                    all_placements.add(s.input_specs[input_idx].placements[mesh_dim])

                # Build directed graph from transition rules
                def is_sharding(p: Placement) -> bool:
                    return isinstance(p, Shard)

                edges: dict[Placement, set[Placement]] = {
                    p: set() for p in all_placements
                }
                for src in all_placements:
                    for dst in all_placements:
                        if src == dst:
                            continue
                        # R -> S, R -> P (free)
                        if isinstance(src, Replicate) and (
                            is_sharding(dst) or isinstance(dst, Partial)
                        ):
                            edges[src].add(dst)
                        # S -> R (allgather), S -> S' (all-to-all)
                        if is_sharding(src) and (
                            isinstance(dst, Replicate) or is_sharding(dst)
                        ):
                            edges[src].add(dst)
                        # P -> R (allreduce), P -> S (reduce-scatter)
                        if isinstance(src, Partial) and (
                            isinstance(dst, Replicate) or is_sharding(dst)
                        ):
                            edges[src].add(dst)

                # BFS from each placement, assert all others reachable
                for start in all_placements:
                    visited: set[Placement] = set()
                    q = deque([start])
                    while q:
                        node = q.popleft()
                        if node in visited:
                            continue
                        visited.add(node)
                        q.extend(edges.get(node, set()))
                    self.assertEqual(
                        visited,
                        all_placements,
                        f"input_idx={input_idx}, mesh_dim={mesh_dim}: "
                        f"from {start}, unreachable: {all_placements - visited}",
                    )

    def test_pq_cost_not_underestimated(self):
        """Verify PQ per-input costs >= graph-based redistribute cost.

        The PQ computes costs per-dim independently. This test checks that
        each per-input PQ cost is at least as high as the graph-based
        (min-cost) redistribute planner's cost for the same source→target.

        The graph-based planner is the right baseline because both it and
        the PQ find optimal transition orderings. The greedy planner uses
        a fixed ordering that can be more expensive, so comparing against
        greedy would flag the PQ's ordering optimizations as false positives.
        """
        from torch.distributed.tensor._collective_utils import redistribute_cost
        from torch.distributed.tensor._redistribute import _gen_transform_infos

        # Clear cached transform infos so use_min_cost_redistribution_plan()
        # takes effect (earlier tests may have cached greedy results).
        _gen_transform_infos.cache_clear()

        placement_options = [Shard(0), Shard(1), Replicate(), Partial("sum")]
        mesh = DeviceMesh("cpu", mesh=torch.arange(4).reshape(2, 2))
        M, K, N = 64, 32, 64
        output_meta = self._get_mm_output_meta(M, K, N)
        all_placements = list(product(placement_options, repeat=mesh.ndim))

        underestimated = []
        for left_pl in all_placements:
            for right_pl in all_placements:
                left_meta, right_meta = _get_mm_metas(M, K, N)
                left_spec, right_spec = _get_mm_specs(
                    mesh, left_meta, right_meta, left_pl, right_pl
                )
                op_schema = OpSchema(
                    op=torch.ops.aten.mm.default,
                    args_schema=(
                        OpStrategy([OpSpec(left_spec)]),
                        OpStrategy([OpSpec(right_spec)]),
                    ),
                    kwargs_schema={},
                )
                pq_strategy = _dijkstra_expand_single_dim_strategy_to_mesh(
                    mesh,
                    op_schema,
                    mm_single_dim_strategy,
                    output_tensor_meta=output_meta,
                )
                self.assertIsNotNone(pq_strategy)
                pq_spec = pq_strategy.strategies[0]
                with use_min_cost_redistribution_plan():
                    for input_idx, (init_spec, target_spec) in enumerate(
                        zip([left_spec, right_spec], pq_spec.input_specs)
                    ):
                        pq_input_cost = pq_spec.redistribute_cost[input_idx][0]
                        actual_cost = redistribute_cost(init_spec, target_spec)
                        if pq_input_cost < actual_cost - 1e-9:
                            underestimated.append(
                                f"left={left_pl} right={right_pl} "
                                f"input_idx={input_idx}: "
                                f"pq={pq_input_cost:.4f} < "
                                f"actual={actual_cost:.4f} "
                                f"target={target_spec.placements}"
                            )

        self.assertEqual(
            underestimated,
            [],
            f"PQ underestimated cost for {len(underestimated)} cases:\n"
            + "\n".join(underestimated[:20]),
        )

    def test_pq_selects_optimal_actual_cost_strategy(self):
        """Verify PQ-selected strategy has graph-based cost <= full expansion min.

        Both the PQ and the reference use graph-based (min-cost) redistribution
        planning for a fair comparison. This catches cases where the PQ selects
        a genuinely worse strategy, not just ordering differences.
        """
        from torch.distributed.tensor._collective_utils import redistribute_cost
        from torch.distributed.tensor._redistribute import _gen_transform_infos

        _gen_transform_infos.cache_clear()

        placement_options = [Shard(0), Shard(1), Replicate(), Partial("sum")]
        mesh = DeviceMesh("cpu", mesh=torch.arange(4).reshape(2, 2))
        M, K, N = 64, 32, 64
        output_meta = self._get_mm_output_meta(M, K, N)
        all_placements = list(product(placement_options, repeat=mesh.ndim))

        wrong_strategy = []
        for left_pl in all_placements:
            for right_pl in all_placements:
                left_meta, right_meta = _get_mm_metas(M, K, N)
                left_spec, right_spec = _get_mm_specs(
                    mesh, left_meta, right_meta, left_pl, right_pl
                )
                op_schema = OpSchema(
                    op=torch.ops.aten.mm.default,
                    args_schema=(
                        OpStrategy([OpSpec(left_spec)]),
                        OpStrategy([OpSpec(right_spec)]),
                    ),
                    kwargs_schema={},
                )
                pq_strategy = _dijkstra_expand_single_dim_strategy_to_mesh(
                    mesh,
                    op_schema,
                    mm_single_dim_strategy,
                    output_tensor_meta=output_meta,
                )
                self.assertIsNotNone(pq_strategy)
                pq_spec = pq_strategy.strategies[0]

                with use_min_cost_redistribution_plan():
                    # Actual cost of PQ-selected strategy (graph-based)
                    pq_actual = sum(
                        redistribute_cost(init, tgt)
                        for init, tgt in zip(
                            [left_spec, right_spec], pq_spec.input_specs
                        )
                    )

                    # Reference: full expansion with graph-based planning
                    expanded_fn = _expand_single_dim_strategy_to_mesh(
                        mesh,
                        op_schema,
                        _SingleDimStrategyInfo(mm_single_dim_strategy),
                        output_meta,
                    )
                    ref = expanded_fn(
                        torch.ops.aten.mm.default,
                        op_schema.args_meta,
                        op_schema.kwargs_meta,
                    )
                ref_min = min(
                    sum(chain.from_iterable(s.redistribute_cost))
                    for s in ref.strategies
                )

                if pq_actual > ref_min + 1e-9:
                    wrong_strategy.append(
                        f"left={left_pl} right={right_pl}: "
                        f"pq_actual={pq_actual:.4f} > "
                        f"ref_min={ref_min:.4f} "
                        f"target_left={pq_spec.input_specs[0].placements} "
                        f"target_right={pq_spec.input_specs[1].placements}"
                    )

        self.assertEqual(
            wrong_strategy,
            [],
            f"PQ selected worse strategy in {len(wrong_strategy)} cases:\n"
            + "\n".join(wrong_strategy[:20]),
        )


@torch.library.custom_op("mylib::dummy_add", mutates_args=())
def dummy_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


@dummy_add.register_fake
def _dummy_add_fake(x, y):
    return torch.empty_like(x)


@torch.library.custom_op("mylib::dummy_check", mutates_args=())
def dummy_check(x: torch.Tensor) -> None:
    """A no-output op similar to _linalg_check_errors."""


@dummy_check.register_fake
def _dummy_check_fake(x):
    return None


class TestCommonPointwiseSingleDimStrategy(TestCase):
    """Unit tests for _common_pointwise_single_dim_strategy raw rule generation."""

    def _meta(self, *dims: int) -> TensorMeta:
        shape = torch.Size(dims)
        stride = torch.empty(shape).stride()
        return TensorMeta(shape=shape, stride=stride, dtype=torch.float32)

    @staticmethod
    def _normalize(rules):
        """Convert rules to a comparable form (replace _ShardingPlaceholder with Shard)."""
        out = []
        for rule in rules:
            out.append(
                tuple(
                    Shard(p.dim) if isinstance(p, _ShardingPlaceholder) else p
                    for p in rule
                )
            )
        return out

    def test_unary_2d_shard_rules(self):
        """Unary op with a 2D tensor should produce two Shard rules."""
        fn = _common_pointwise_single_dim_strategy()
        rules = self._normalize(fn(torch.ops.aten.abs.default, (self._meta(4, 8),), {}))
        self.assertEqual(
            rules,
            [(Shard(0), Shard(0)), (Shard(1), Shard(1))],
        )

    def test_unary_with_partial_extra_rules(self):
        """Unary op with _UNARY_LINEAR_RULES should append partial rules."""
        fn = _common_pointwise_single_dim_strategy(
            partial_extra_rules=_UNARY_LINEAR_RULES
        )
        rules = self._normalize(fn(torch.ops.aten.neg.default, (self._meta(4, 8),), {}))
        expected = [
            (Shard(0), Shard(0)),
            (Shard(1), Shard(1)),
            (Partial("sum"), Partial("sum")),
            (Partial("avg"), Partial("avg")),
        ]
        self.assertEqual(rules, expected)

    def test_binary_broadcast_replicate(self):
        """When one input is broadcast, the broadcast dim gets Replicate."""
        fn = _common_pointwise_single_dim_strategy()
        # (4, 8) + (8,) — dim 0 is broadcast for the second arg
        rules = self._normalize(
            fn(torch.ops.aten.add.Tensor, (self._meta(4, 8), self._meta(8)), {})
        )
        self.assertEqual(
            rules,
            [(Shard(0), Shard(0), Replicate()), (Shard(1), Shard(1), Shard(0))],
        )

    def test_partial_extra_rules_filtered_by_arity(self):
        """Binary extra rules are filtered out when only one tensor arg is present."""
        fn = _common_pointwise_single_dim_strategy(
            partial_extra_rules=_MUL_RULES + _UNARY_LINEAR_RULES
        )
        # Scalar promotion: mul.Tensor with one tensor arg
        rules = self._normalize(
            fn(torch.ops.aten.mul.Tensor, (self._meta(4, 8), 2.0), {})
        )
        # Length-3 _MUL_RULES should be filtered out, only length-2 _UNARY_LINEAR_RULES kept
        expected = [
            (Shard(0), Shard(0)),
            (Shard(1), Shard(1)),
            (Partial("sum"), Partial("sum")),
            (Partial("avg"), Partial("avg")),
        ]
        self.assertEqual(rules, expected)

    def test_binary_with_additive_rules(self):
        """Binary op with _BINARY_ADDITIVE_RULES appends the right partial rules."""
        fn = _common_pointwise_single_dim_strategy(
            partial_extra_rules=_BINARY_ADDITIVE_RULES
        )
        rules = self._normalize(
            fn(torch.ops.aten.add.Tensor, (self._meta(4, 8), self._meta(4, 8)), {})
        )
        expected = [
            (Shard(0), Shard(0), Shard(0)),
            (Shard(1), Shard(1), Shard(1)),
            (Partial("sum"), Partial("sum"), Partial("sum")),
            (Partial("avg"), Partial("avg"), Partial("avg")),
            (Partial("avg"), Partial("avg"), Replicate()),
            (Partial("max"), Partial("max"), Replicate()),
            (Partial("min"), Partial("min"), Replicate()),
            (Partial("avg"), Replicate(), Partial("avg")),
        ]
        self.assertEqual(rules, expected)


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

    @patch(
        "torch.distributed.tensor._api.DTensor._op_dispatcher.sharding_propagator.op_single_dim_strategy_funcs",
        {},
    )
    def test_register_single_dim_strategy_no_output(self):
        """Test that single-dim strategy works for ops with no tensor output.

        This tests the fix for operators like _linalg_check_errors that return None.
        Previously, this would fail with:
        "_propagate_tensor_meta_non_cached returned None for ..., but tensor_meta is required"
        """
        mesh = DeviceMesh("cpu", torch.arange(self.world_size))
        x = torch.randn(8, 16)
        x_dt = distribute_tensor(x, mesh, [Shard(0)])

        # Register a single-dim strategy for the no-output op
        @register_single_dim_strategy(torch.ops.mylib.dummy_check.default)
        def dummy_check_single_dim_strategy(op, args_schema, kwargs_schema):
            # For no-output ops, return empty list (replicate-only)
            return []

        # This should work without raising "tensor_meta is required" error
        result = torch.ops.mylib.dummy_check(x_dt)

        # Verify the result is None (no tensor output)
        self.assertIsNone(result, "No-output op should return None")


if __name__ == "__main__":
    run_tests()
