# Owner(s): ["oncall: distributed"]


import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, Partial, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OpSpec, OpStrategy
from torch.distributed.tensor._ops._matrix_ops import mm_single_dim_strategy
from torch.distributed.tensor._ops.utils import (
    _args_schema_with_tensor_meta,
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
            args_schema=(
                OpStrategy([OpSpec(left_spec)]),
                OpStrategy([OpSpec(right_spec)]),
            ),
            kwargs_schema={},
        )

        # Expand the strategy to the full mesh
        expanded_strategy_fn = _expand_single_dim_strategy_to_mesh(
            mesh, op_schema, mm_single_dim_strategy
        )
        args, kwargs = _args_schema_with_tensor_meta((left_meta, right_meta), {})
        # Call the expanded strategy with TensorMeta (not DTensorSpec)
        strategy = expanded_strategy_fn(torch.ops.aten.matmul.default, args, kwargs)
        assert isinstance(strategy, OpStrategy)

        # Verify we have strategies (should be product of single-dim strategies across mesh dims)
        # For a 3D mesh with 4 single-dim strategies (3 explicit + 1 implicit replicate),
        # we should have 4^3 = 64 strategies maximum (some filtered out if not shardable)
        self.assertEqual(len(strategy.strategies), 64)

        all_replicate_found = False
        shard_0_found = False
        for op_spec in strategy.strategies:
            output_spec = op_spec.output_spec
            input_specs = op_spec.input_specs
            assert input_specs is not None

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
        # Create a mesh for testing
        mesh = DeviceMesh("cpu", [0, 1, 2, 3])

        # Use helpers to create tensor metadata for matmul: (M, K) @ (K, N) -> (M, N)
        left_meta, right_meta = _get_mm_metas()

        # Use the helper function to get matmul-like single-dim strategies
        single_dim_strategies = mm_single_dim_strategy(
            torch.ops.aten.matmul.default, (left_meta, right_meta), {}
        )

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
            args_schema=(
                OpStrategy([OpSpec(left_spec_replicate)]),
                OpStrategy([OpSpec(right_spec_replicate)]),
            ),
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
            args_schema=(
                OpStrategy([OpSpec(left_spec_shard)]),
                OpStrategy([OpSpec(right_spec_shard)]),
            ),
            kwargs_schema={},
        )

        # Expected: 3 strategies with placeholders filled using Shard + implicit replicate
        expected_shard = [
            [Partial(), Shard(1), Shard(0)],
            [Shard(0), Shard(0), Replicate()],
            [Shard(1), Replicate(), Shard(1)],
            [Replicate(), Replicate(), Replicate()],
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
            args_schema=(
                OpStrategy([OpSpec(left_spec_mixed)]),
                OpStrategy([OpSpec(right_spec_mixed)]),
            ),
            kwargs_schema={},
        )

        # Expected: 3 strategies * 2 shard types (Shard and _StridedShard) + implicit replicate
        expected_mixed = [
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
            [Replicate(), Replicate(), Replicate()],
        ]

        expanded_mixed = _fill_single_dim_strategy_placeholders(
            mesh, op_schema_mixed, single_dim_strategies
        )

        self.assertEqual(expanded_mixed, expected_mixed)


if __name__ == "__main__":
    run_tests()
