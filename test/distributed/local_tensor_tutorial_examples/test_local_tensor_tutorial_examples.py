# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Tests for LocalTensor Tutorial Examples
========================================

This test suite directly invokes the core pattern functions from the tutorial
examples to ensure they remain correct as LocalTensor evolves.

Each test imports and calls the same functions that are included in the
tutorial documentation via literalinclude, ensuring the tutorial stays
accurate.

Tests verify results using expected values returned by the core functions
themselves (as tuples), so there are no hardcoded expected values in this file.
"""

# Import core pattern functions from each example module
from example_01_basic_operations import (
    access_individual_shards,
    arithmetic_operations,
    create_local_tensor,
    reconcile_identical_shards,
    use_local_tensor_mode,
)
from example_02_collective_operations import (
    all_gather_tensors,
    all_reduce_sum,
    broadcast_from_rank,
    reduce_scatter_tensors,
)
from example_03_dtensor_integration import (
    distribute_and_verify,
    dtensor_linear_layer,
    dtensor_matmul,
)
from example_04_uneven_sharding import (
    create_uneven_shards,
    dtensor_uneven_sharding,
    local_int_node_arithmetic,
)
from example_05_rank_specific import (
    disable_mode_temporarily,
    use_maybe_disable,
    use_maybe_run_decorator,
    use_rank_map,
    use_tensor_map,
)
from example_06_multidim_mesh import create_2d_mesh, create_3d_mesh, hybrid_parallelism

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import LocalTensor
from torch.testing._internal.common_utils import run_tests, TestCase


class TestExample01BasicOperations(TestCase):
    """Tests for Example 1: Basic LocalTensor Operations."""

    def test_create_local_tensor(self):
        """Test creating a LocalTensor from per-rank tensors."""
        lt, (exp_shape, exp_ranks, exp_rank_0, exp_rank_1) = create_local_tensor()

        self.assertIsInstance(lt, LocalTensor)
        self.assertEqual(lt.shape, exp_shape)
        self.assertEqual(lt._ranks, exp_ranks)
        self.assertTrue(torch.equal(lt._local_tensors[0], exp_rank_0))
        self.assertTrue(torch.equal(lt._local_tensors[1], exp_rank_1))

    def test_arithmetic_operations(self):
        """Test arithmetic operations on LocalTensor."""
        (doubled, added), (exp_d0, exp_d1, exp_a0) = arithmetic_operations()

        self.assertTrue(torch.equal(doubled._local_tensors[0], exp_d0))
        self.assertTrue(torch.equal(doubled._local_tensors[1], exp_d1))
        self.assertTrue(torch.equal(added._local_tensors[0], exp_a0))

    def test_reconcile_identical_shards(self):
        """Test reconcile() extracts tensor when shards are identical."""
        result, expected = reconcile_identical_shards()

        self.assertIsInstance(result, torch.Tensor)
        self.assertNotIsInstance(result, LocalTensor)
        self.assertTrue(torch.equal(result, expected))

    def test_use_local_tensor_mode(self):
        """Test LocalTensorMode auto-creates LocalTensors."""
        (is_local, num_ranks), (exp_local, exp_ranks) = use_local_tensor_mode()

        self.assertEqual(is_local, exp_local)
        self.assertEqual(num_ranks, exp_ranks)

    def test_access_individual_shards(self):
        """Test accessing shards via dict and attribute."""
        (shard_0, shard_1), (exp_0, exp_1) = access_individual_shards()

        self.assertTrue(torch.equal(shard_0, exp_0))
        self.assertTrue(torch.equal(shard_1, exp_1))


class TestExample02CollectiveOperations(TestCase):
    """Tests for Example 2: Collective Operations."""

    @classmethod
    def setUpClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group("fake", rank=0, world_size=3)
        cls.pg = dist.distributed_c10d._get_default_group()

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_all_reduce_sum(self):
        """Test all_reduce with SUM operation."""
        result, expected = all_reduce_sum(self.pg)
        self.assertTrue(torch.equal(result, expected))

    def test_broadcast_from_rank(self):
        """Test broadcast from source rank."""
        result, expected = broadcast_from_rank(self.pg, src_rank=0)
        self.assertTrue(torch.equal(result, expected))

    def test_all_gather_tensors(self):
        """Test all_gather collects tensors from all ranks."""
        results, expected = all_gather_tensors(self.pg)

        self.assertEqual(len(results), len(expected))
        for actual, exp in zip(results, expected):
            self.assertTrue(torch.equal(actual, exp))

    def test_reduce_scatter_tensors(self):
        """Test reduce_scatter distributes reduced results."""
        rank_outputs, expected = reduce_scatter_tensors(self.pg)

        self.assertEqual(len(rank_outputs), len(expected))
        for rank in rank_outputs:
            self.assertTrue(torch.equal(rank_outputs[rank], expected[rank]))


class TestExample03DTensorIntegration(TestCase):
    """Tests for Example 3: DTensor Integration."""

    @classmethod
    def setUpClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group("fake", rank=0, world_size=4)

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_distribute_and_verify(self):
        """Test DTensor distribution and reconstruction."""
        (sharded_actual, replicated_actual), (sharded_exp, replicated_exp) = (
            distribute_and_verify()
        )

        self.assertTrue(torch.equal(sharded_actual, sharded_exp))
        self.assertTrue(torch.equal(replicated_actual, replicated_exp))

    def test_dtensor_matmul(self):
        """Test DTensor matrix multiplication."""
        actual, expected = dtensor_matmul()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_dtensor_linear_layer(self):
        """Test DTensor linear layer forward pass."""
        actual, expected = dtensor_linear_layer()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))


class TestExample04UnevenSharding(TestCase):
    """Tests for Example 4: Uneven Sharding."""

    @classmethod
    def setUpClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group("fake", rank=0, world_size=3)

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_create_uneven_shards(self):
        """Test LocalTensor with uneven shard sizes."""
        (lt, is_symint), exp_shapes = create_uneven_shards()

        self.assertIsInstance(lt, LocalTensor)
        self.assertTrue(is_symint)
        for rank, exp_shape in exp_shapes.items():
            self.assertEqual(lt._local_tensors[rank].shape, exp_shape)

    def test_local_int_node_arithmetic(self):
        """Test LocalIntNode arithmetic operations."""
        (add_result, mul_result), (exp_add, exp_mul) = local_int_node_arithmetic()

        self.assertEqual(add_result, exp_add)
        self.assertEqual(mul_result, exp_mul)

    def test_dtensor_uneven_sharding(self):
        """Test DTensor with unevenly divisible dimensions."""
        (rows_per_rank, matches), exp_total = dtensor_uneven_sharding()

        self.assertTrue(matches)
        self.assertEqual(sum(rows_per_rank.values()), exp_total)


class TestExample05RankSpecific(TestCase):
    """Tests for Example 5: Rank-Specific Computations."""

    def test_use_rank_map(self):
        """Test rank_map creates per-rank values."""
        values, expected = use_rank_map()
        self.assertEqual(values, expected)

    def test_use_tensor_map(self):
        """Test tensor_map transforms shards per-rank."""
        values, expected = use_tensor_map()
        self.assertEqual(values, expected)

    def test_disable_mode_temporarily(self):
        """Test mode.disable() creates regular tensors."""
        (inside, disabled), (exp_inside, exp_disabled) = disable_mode_temporarily()

        self.assertEqual(inside, exp_inside)
        self.assertEqual(disabled, exp_disabled)

    def test_use_maybe_disable(self):
        """Test maybe_disable_local_tensor_mode utility."""
        (outside, inside), (exp_outside, exp_inside) = use_maybe_disable()

        self.assertEqual(outside, exp_outside)
        self.assertEqual(inside, exp_inside)

    def test_use_maybe_run_decorator(self):
        """Test @maybe_run_for_local_tensor decorator."""
        values, expected = use_maybe_run_decorator()
        self.assertEqual(values, expected)


class TestExample06MultidimMesh(TestCase):
    """Tests for Example 6: Multi-Dimensional Meshes."""

    @classmethod
    def setUpClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group("fake", rank=0, world_size=24)

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_create_2d_mesh(self):
        """Test 2D mesh creation."""
        (shape, names, size), (exp_shape, exp_names, exp_size) = create_2d_mesh()

        self.assertEqual(shape, exp_shape)
        self.assertEqual(names, exp_names)
        self.assertEqual(size, exp_size)

    def test_hybrid_parallelism(self):
        """Test hybrid DP+TP parallelism."""
        actual, expected = hybrid_parallelism()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_create_3d_mesh(self):
        """Test 3D mesh for DP+TP+PP."""
        actual, expected = create_3d_mesh()
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    run_tests()
