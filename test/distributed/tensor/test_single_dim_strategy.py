# Owner(s): ["oncall: distributed"]


import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, Partial, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    OpSchema,
    OpStrategy,
)
from torch.distributed.tensor._ops.utils import (
    _expand_single_dim_strategy_to_mesh,
    _fill_single_dim_strategy_placeholders,
)
from torch.distributed.tensor.placement_types import (
    _ShardingPlaceholder,
    _StridedShard,
    Placement,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


def _get_mm_like_single_dim_strategies() -> list[
    list[Placement | _ShardingPlaceholder]
]:
    """
    Helper function to generate matmul-like single-dim strategies with placeholders.

    Returns strategies for matmul operation (M, K) @ (K, N) -> (M, N):
    - S0,R -> S0: Output and left input sharded on dim 0, right input replicated
    - R,S1 -> S1: Output and right input sharded on dim 1, left input replicated
    - Contracting dim: Output partial, both inputs sharded on contracting dims
    """
    return [
        # Output sharded on dim 0, left input sharded on dim 0, right input replicated
        [_ShardingPlaceholder(0), _ShardingPlaceholder(0), Replicate()],
        # Output sharded on dim 1, left input replicated, right input sharded on dim 1
        [_ShardingPlaceholder(1), Replicate(), _ShardingPlaceholder(1)],
        # Contracting dim: output partial, both inputs sharded on contracting dim
        [Partial(), _ShardingPlaceholder(1), _ShardingPlaceholder(0)],
    ]


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
    left_placements: tuple[Placement],
    right_placements: tuple[Placement],
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
        # Initialize fake process group for testing
        self.world_size = 8  # 3D mesh size
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def test_expand_matmul_like_strategy_to_3d_mesh(self):
        """Test expanding matmul-like single-dim strategies to a 3D mesh.

        This test verifies that:
        1. Single-dim matmul strategies (S0,R -> S0 and R,S1->S1) are correctly expanded to 3D
        2. The implicit full-replication rule is included
        3. _ShardingPlaceholder is correctly replaced with actual Shard placements
        """
        # Create a fake 3D mesh of size 8 (2x2x2)
        mesh = DeviceMesh("cpu", mesh=torch.arange(8).reshape(2, 2, 2))

        # Use helpers to create tensor metadata for matmul: (M, K) @ (K, N) -> (M, N)
        left_meta, right_meta = _get_mm_metas()

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
            args_schema=(left_spec, right_spec),
            kwargs_schema={},
        )

        # Use the helper function to get matmul-like single-dim strategies
        def matmul_single_dim_strategy(
            args_schema: ArgsType, kwargs_schema: KwargsType
        ) -> list[list[Placement | _ShardingPlaceholder]]:
            return _get_mm_like_single_dim_strategies()

        # Expand the strategy to the full mesh
        expanded_strategy_fn = _expand_single_dim_strategy_to_mesh(
            mesh, op_schema, matmul_single_dim_strategy
        )

        # Call the expanded strategy with TensorMeta (not DTensorSpec)
        strategy = expanded_strategy_fn((left_meta, right_meta), {})

        # Verify the strategy is an OpStrategy
        self.assertIsInstance(strategy, OpStrategy)

        # Verify we have strategies (should be product of single-dim strategies across mesh dims)
        # For a 3D mesh with 4 single-dim strategies (3 explicit + 1 implicit replicate),
        # we should have 4^3 = 64 strategies maximum (some filtered out if not shardable)
        self.assertEqual(len(strategy.strategies), 64)

        # Verify that the implicit full-replication rule is present
        # All-replicate should have output and both inputs all replicated
        all_replicate_found = False
        for op_spec in strategy.strategies:
            output_spec = op_spec.output_spec
            input_specs = op_spec.input_specs

            # Check if this is the all-replicate strategy
            if (
                all(isinstance(p, Replicate) for p in output_spec.placements)
                and all(isinstance(p, Replicate) for p in input_specs[0].placements)
                and all(isinstance(p, Replicate) for p in input_specs[1].placements)
            ):
                all_replicate_found = True
                break

        self.assertTrue(
            all_replicate_found,
            "Implicit full-replication rule not found in expanded strategies",
        )

        # Verify that _ShardingPlaceholder has been replaced with actual Shard placements
        for op_spec in strategy.strategies:
            output_spec = op_spec.output_spec
            input_specs = op_spec.input_specs

            # Check output placements
            for p in output_spec.placements:
                self.assertNotIsInstance(
                    p,
                    _ShardingPlaceholder,
                    "Found _ShardingPlaceholder in output placements",
                )

            # Check input placements
            for input_spec in input_specs:
                for p in input_spec.placements:
                    self.assertNotIsInstance(
                        p,
                        _ShardingPlaceholder,
                        "Found _ShardingPlaceholder in input placements",
                    )

        # Verify at least one strategy has Shard(0) placement for left input
        shard_0_found = False
        for op_spec in strategy.strategies:
            input_specs = op_spec.input_specs
            if any(
                isinstance(p, Shard) and p.dim == 0 for p in input_specs[0].placements
            ):
                shard_0_found = True
                break

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
        # Create a mesh for testing
        mesh = DeviceMesh("cpu", [0, 1, 2, 3])

        # Use helpers to create tensor metadata for matmul: (M, K) @ (K, N) -> (M, N)
        left_meta, right_meta = _get_mm_metas()

        # Use the helper function to get matmul-like single-dim strategies
        single_dim_strategies = _get_mm_like_single_dim_strategies()

        # Test Case 1: All-replicate inputs - no sharding expansion
        left_spec_replicate = DTensorSpec(
            mesh=mesh,
            placements=(Replicate(),),
            tensor_meta=left_meta,
        )
        right_spec_replicate = DTensorSpec(
            mesh=mesh,
            placements=(Replicate(),),
            tensor_meta=right_meta,
        )
        op_schema_replicate = OpSchema(
            op=torch.ops.aten.mm.default,
            args_schema=(left_spec_replicate, right_spec_replicate),
            kwargs_schema={},
        )

        # Expected: Only implicit all-replicate rule (no sharding builders available)
        expected_replicate = [
            [Replicate(), Replicate(), Replicate()],  # Implicit all-replicate
        ]

        expanded_replicate = _fill_single_dim_strategy_placeholders(
            mesh, op_schema_replicate, single_dim_strategies
        )

        self.assertEqual(expanded_replicate, expected_replicate)

        # Test Case 2: Shard-only inputs - only Shard expansion
        left_spec_shard = DTensorSpec(
            mesh=mesh,
            placements=(Shard(0),),
            tensor_meta=left_meta,
        )
        right_spec_shard = DTensorSpec(
            mesh=mesh,
            placements=(Shard(1),),
            tensor_meta=right_meta,
        )
        op_schema_shard = OpSchema(
            op=torch.ops.aten.mm.default,
            args_schema=(left_spec_shard, right_spec_shard),
            kwargs_schema={},
        )

        # Expected: 3 strategies with placeholders filled using Shard + implicit replicate
        expected_shard = [
            [Shard(0), Shard(0), Replicate()],  # Strategy 1 with Shard
            [Shard(1), Replicate(), Shard(1)],  # Strategy 2 with Shard
            [Partial(), Shard(1), Shard(0)],  # Strategy 3 with Shard
            [Replicate(), Replicate(), Replicate()],  # Implicit all-replicate
        ]

        expanded_shard = _fill_single_dim_strategy_placeholders(
            mesh, op_schema_shard, single_dim_strategies
        )

        self.assertEqual(expanded_shard, expected_shard)

        # Test Case 3: Mixed Shard and _StridedShard inputs - both types of expansion
        left_spec_mixed = DTensorSpec(
            mesh=mesh,
            placements=(_StridedShard(0, split_factor=2),),
            tensor_meta=left_meta,
        )
        right_spec_mixed = DTensorSpec(
            mesh=mesh,
            placements=(Shard(1),),
            tensor_meta=right_meta,
        )
        op_schema_mixed = OpSchema(
            op=torch.ops.aten.mm.default,
            args_schema=(left_spec_mixed, right_spec_mixed),
            kwargs_schema={},
        )

        # Expected: 3 strategies * 2 shard types (Shard and _StridedShard) + implicit replicate
        expected_mixed = [
            # Strategy 1 with Shard
            [Shard(0), Shard(0), Replicate()],
            # Strategy 1 with _StridedShard
            [
                _StridedShard(0, split_factor=2),
                _StridedShard(0, split_factor=2),
                Replicate(),
            ],
            # Strategy 2 with Shard
            [Shard(1), Replicate(), Shard(1)],
            # Strategy 2 with _StridedShard
            [
                _StridedShard(1, split_factor=2),
                Replicate(),
                _StridedShard(1, split_factor=2),
            ],
            # Strategy 3 with Shard
            [Partial(), Shard(1), Shard(0)],
            # Strategy 3 with _StridedShard
            [
                Partial(),
                _StridedShard(1, split_factor=2),
                _StridedShard(0, split_factor=2),
            ],
            # Implicit all-replicate
            [Replicate(), Replicate(), Replicate()],
        ]

        expanded_mixed = _fill_single_dim_strategy_placeholders(
            mesh, op_schema_mixed, single_dim_strategies
        )

        self.assertEqual(expanded_mixed, expected_mixed)


if __name__ == "__main__":
    run_tests()
