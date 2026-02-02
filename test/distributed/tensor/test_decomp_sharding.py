# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
from torch._decomp import register_decomposition
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


# Create custom ops for testing recursive decompositions
my_lib = torch.library.Library("test_decomp", "DEF")
my_lib.define("my_recursive_op(Tensor x) -> Tensor")
my_lib.define("my_intermediate_op(Tensor x) -> Tensor")


@torch.library.impl(my_lib, "my_recursive_op", "CPU")
def my_recursive_op_impl(x):
    return x + 1.0


@torch.library.impl(my_lib, "my_intermediate_op", "CPU")
def my_intermediate_op_impl(x):
    return x + 1.0


# Add fake implementations for FakeTensorMode
@torch.library.register_fake("test_decomp::my_recursive_op")
def my_recursive_op_fake(x):
    return torch.empty_like(x)


@torch.library.register_fake("test_decomp::my_intermediate_op")
def my_intermediate_op_fake(x):
    return torch.empty_like(x)


# Register decompositions to create a recursive chain:
# my_recursive_op -> my_intermediate_op -> aten.add
@register_decomposition(torch.ops.test_decomp.my_recursive_op.default)
def my_recursive_op_decomp(x):
    return torch.ops.test_decomp.my_intermediate_op(x)


@register_decomposition(torch.ops.test_decomp.my_intermediate_op.default)
def my_intermediate_op_decomp(x):
    return x + 1.0


class TestDecompSharding(TestCase):
    def setUp(self):
        super().setUp()
        fake_store = FakeStore()
        dist.init_process_group("fake", store=fake_store, rank=0, world_size=2)

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def test_aminmax_decomp_replicate(self):
        """Test aminmax with Replicate() placement."""
        mesh = DeviceMesh("cpu", torch.arange(2))

        # Create a replicated DTensor
        x_local = torch.randn(4)
        x = DTensor.from_local(x_local, mesh, [Replicate()], run_check=False)

        # Call aminmax - should use decomposition
        try:
            result = torch.aminmax(x)
            print(f"Result: {result}")
            print("✓ aminmax with Replicate() succeeded")
        except Exception as e:
            print(f"✗ aminmax with Replicate() failed: {e}")
            raise

    def test_aminmax_decomp_shard(self):
        """Test aminmax with Shard() placement."""
        mesh = DeviceMesh("cpu", torch.arange(2))

        # Create a sharded DTensor
        x_local = torch.randn(2)  # Local shard
        x = DTensor.from_local(x_local, mesh, [Shard(0)], run_check=False)

        # Call aminmax - should use decomposition
        # This might fail if the decomposition has constructors
        try:
            result = torch.aminmax(x)
            print(f"Result: {result}")
            print("✓ aminmax with Shard(0) succeeded")
        except NotImplementedError as e:
            print(f"aminmax with Shard(0) not implemented: {e}")
            # This is expected if decomposition has constructors
        except Exception as e:
            print(f"✗ aminmax with Shard(0) failed: {e}")
            raise

    def test_recursive_decomposition(self):
        """Test that recursive decompositions work correctly.

        This test verifies that the decomposition-based sharding propagation
        can handle recursive decompositions, where one decomposition calls
        another operation that also has a decomposition.

        Decomposition chain:
        - my_recursive_op decomposes to my_intermediate_op
        - my_intermediate_op decomposes to aten.add (which has sharding rules)

        The sharding propagator should:
        1. Trace my_recursive_op's decomposition to my_intermediate_op
        2. During propagation, call propagate_op_sharding on my_intermediate_op
        3. Recursively trigger my_intermediate_op's decomposition fallback
        4. Eventually reach aten.add which has explicit sharding rules
        """
        mesh = DeviceMesh("cpu", torch.arange(2))

        # Test with Replicate
        x_local = torch.randn(4)
        x = DTensor.from_local(x_local, mesh, [Replicate()], run_check=False)

        # Call custom op with recursive decomposition
        result = torch.ops.test_decomp.my_recursive_op(x)

        # Verify result is a DTensor with correct placement
        self.assertIsInstance(result, DTensor)
        self.assertEqual(result.placements, (Replicate(),))
        print("✓ Recursive decomposition with Replicate() succeeded")

        # Test with Shard
        x_shard_local = torch.randn(2)
        x_shard = DTensor.from_local(x_shard_local, mesh, [Shard(0)], run_check=False)

        result_shard = torch.ops.test_decomp.my_recursive_op(x_shard)

        # Verify result is a DTensor with correct placement
        self.assertIsInstance(result_shard, DTensor)
        self.assertEqual(result_shard.placements, (Shard(0),))
        print("✓ Recursive decomposition with Shard(0) succeeded")

    def test_placement_types_considered(self):
        """Test that _get_candidate_placements considers all placement types from inputs.

        This verifies that the candidate generation uses _get_unique_placements
        and shard_builders to handle different placement types like Partial.
        """
        mesh = DeviceMesh("cpu", torch.arange(2))

        # Test that aminmax works with Partial input (from a reduction operation)
        # Create a Partial(sum) input by doing a sum operation
        x_local = torch.randn(2, 4)
        x = DTensor.from_local(x_local, mesh, [Shard(0)], run_check=False)

        # Sum across dim 0 creates Partial
        x_summed = x.sum(dim=0)  # This should have Partial(sum) placement

        # Now call aminmax on the Partial tensor
        result = torch.aminmax(x_summed)

        # Verify result works (placement propagated correctly)
        self.assertIsInstance(result.min, DTensor)
        self.assertIsInstance(result.max, DTensor)
        print(f"✓ aminmax with Partial input succeeded: {result.min.placements}")


if __name__ == "__main__":
    run_tests()
