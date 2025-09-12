#!/usr/bin/env python3


import torch
import torch.distributed as dist
from torch.distributed._local_tensor import LocalTensor, LocalTensorMode
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


class TestLocalTensor(TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mode = None
        self.device = torch.device("cpu")
        self.shape = (2, 3)
        self.dtype = torch.float32

        # Create sample local tensors for different ranks
        self.local_tensors = {
            0: torch.randn(self.shape, dtype=self.dtype, device=self.device),
            1: torch.randn(self.shape, dtype=self.dtype, device=self.device),
            2: torch.randn(self.shape, dtype=self.dtype, device=self.device),
        }

        # Create identical local tensors for consistency tests
        base_tensor = torch.randn(self.shape, dtype=self.dtype, device=self.device)
        self.identical_local_tensors = {
            0: base_tensor.clone(),
            1: base_tensor.clone(),
            2: base_tensor.clone(),
        }

    def tearDown(self):
        self.mode = None
        super().tearDown()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    def assertEqual(self, lhs, rhs, **kwargs):
        if self.mode is not None:
            old = self.mode._disable
            self.mode._disable = True
        try:
            if isinstance(lhs, LocalTensor) and isinstance(rhs, LocalTensor):
                assert isinstance(lhs, LocalTensor) and isinstance(rhs, LocalTensor)
                super().assertEqual(lhs._ranks, rhs._ranks)
                for r in lhs._ranks:
                    super().assertEqual(lhs._local_tensors[r], rhs._local_tensors[r], lambda m: f"rank {r}: {m}")
            elif isinstance(lhs, LocalTensor) or isinstance(rhs, LocalTensor):
                lhs, rhs = (lhs, rhs) if isinstance(lhs, LocalTensor) else (rhs, lhs)
                for r in lhs._ranks:
                    super().assertEqual(lhs._local_tensors[r], rhs, lambda m: f"rank {r}: {m}")
            else:
                return super().assertEqual(lhs, rhs, **kwargs)
        finally:
            if self.mode is not None:
                self.mode._disable = old

    def test_local_tensor_creation(self):
        """Test basic LocalTensor creation."""
        lt = LocalTensor(self.local_tensors)

        self.assertIsInstance(lt, LocalTensor)
        self.assertEqual(lt.shape, self.shape)
        self.assertEqual(lt.dtype, self.dtype)
        self.assertEqual(lt.device, self.device)
        self.assertFalse(lt.requires_grad)
        self.assertEqual(len(lt._local_tensors), 3)

    def test_local_tensor_no_grad(self):
        """Test LocalTensor creation - requires_grad is always False."""
        lt = LocalTensor(self.local_tensors)
        self.assertFalse(lt.requires_grad)

    def test_local_tensor_creation_fails_with_grad_tensors(self):
        """Test that LocalTensor creation fails when local tensors have requires_grad=True."""
        grad_tensors = {
            rank: tensor.requires_grad_(True)
            for rank, tensor in self.local_tensors.items()
        }

        with self.assertRaises(AssertionError):
            LocalTensor(grad_tensors)

    def test_local_tensor_shape_consistency(self):
        """Test that LocalTensor enforces shape consistency."""
        inconsistent_tensors = {
            0: torch.randn((2, 3), dtype=self.dtype, device=self.device),
            1: torch.randn(
                (3, 2), dtype=self.dtype, device=self.device
            ),  # Different shape
        }

        with self.assertRaises(AssertionError):
            LocalTensor(inconsistent_tensors)

    def test_local_tensor_dtype_consistency(self):
        """Test that LocalTensor enforces dtype consistency."""
        inconsistent_tensors = {
            0: torch.randn(self.shape, dtype=torch.float32, device=self.device),
            1: torch.randn(
                self.shape, dtype=torch.float64, device=self.device
            ),  # Different dtype
        }

        with self.assertRaises(AssertionError):
            LocalTensor(inconsistent_tensors)

    def test_tensor_flatten_unflatten(self):
        """Test tensor flatten/unflatten protocol for PT2 tracing."""
        lt = LocalTensor(self.local_tensors)

        # Test flatten
        inner_tensors, flatten_spec = lt.__tensor_flatten__()
        self.assertEqual(inner_tensors, ["_local_tensors"])
        self.assertEqual(flatten_spec, ())

        # Test unflatten
        flattened_tensors = {"_local_tensors": self.local_tensors}
        reconstructed = LocalTensor.__tensor_unflatten__(
            flattened_tensors, flatten_spec, lt.shape, lt.stride()
        )

        self.assertIsInstance(reconstructed, LocalTensor)
        self.assertFalse(reconstructed.requires_grad)
        self.assertEqual(len(reconstructed._local_tensors), 3)

    def test_basic_arithmetic_operations(self):
        """Test basic arithmetic operations on LocalTensors."""
        lt1 = LocalTensor(self.identical_local_tensors)
        lt2 = LocalTensor(self.identical_local_tensors)

        # Test addition
        result_add = lt1 + lt2
        self.assertIsInstance(result_add, LocalTensor)
        self.assertEqual(len(result_add._local_tensors), 3)

        # Verify the operation was applied to each local tensor
        for rank in self.identical_local_tensors.keys():
            expected = (
                self.identical_local_tensors[rank] + self.identical_local_tensors[rank]
            )
            self.assertEqual(result_add._local_tensors[rank], expected)

        # Test multiplication
        result_mul = lt1 * 2.0
        self.assertIsInstance(result_mul, LocalTensor)
        for rank in self.identical_local_tensors.keys():
            expected = self.identical_local_tensors[rank] * 2.0
            self.assertEqual(result_mul._local_tensors[rank], expected)

    def test_tensor_operations(self):
        """Test various tensor operations on LocalTensors."""
        lt = LocalTensor(self.identical_local_tensors)

        # Test reshape
        reshaped = lt.reshape(-1)
        self.assertIsInstance(reshaped, LocalTensor)
        self.assertEqual(reshaped.shape, (6,))

        # Test transpose
        transposed = lt.transpose(0, 1)
        self.assertIsInstance(transposed, LocalTensor)
        self.assertEqual(transposed.shape, (3, 2))

        # Test sum
        summed = lt.sum()
        self.assertIsInstance(summed, LocalTensor)

        # Test mean
        mean_result = lt.mean()
        self.assertIsInstance(mean_result, LocalTensor)

    def test_mixed_operations_with_regular_tensors(self):
        """Test operations between LocalTensors and regular tensors."""
        lt = LocalTensor(self.identical_local_tensors)
        regular_tensor = torch.ones_like(self.identical_local_tensors[0])

        # Test LocalTensor + regular tensor
        result = lt + regular_tensor
        self.assertIsInstance(result, LocalTensor)

        for rank in self.identical_local_tensors.keys():
            expected = self.identical_local_tensors[rank] + regular_tensor
            self.assertEqual(result._local_tensors[rank], expected)

    def test_gradient_propagation(self):
        """Test that gradients work correctly with LocalTensors."""
        # Create LocalTensor
        lt = LocalTensor(self.identical_local_tensors)

        # Simple operation - LocalTensors always have requires_grad=False
        result = lt * 2.0
        self.assertFalse(result.requires_grad)

    def test_local_tensor_mode(self):
        """Test LocalTensorMode functionality."""
        lt = LocalTensor(self.identical_local_tensors)

        with LocalTensorMode(lt._ranks):
            result = lt + 1.0
            self.assertIsInstance(result, LocalTensor)

            regular = torch.ones(2, 2)
            regular_result = regular + 1.0
            self.assertIsInstance(regular, LocalTensor)
            self.assertIsInstance(regular_result, LocalTensor)

    def test_empty_local_tensors(self):
        """Test behavior with empty local tensors dict."""
        with self.assertRaises(StopIteration):  # next() on empty iterator
            LocalTensor({})

    def test_single_rank_tensor(self):
        """Test LocalTensor with only one rank."""
        single_rank_tensors = {0: torch.randn(self.shape)}
        lt = LocalTensor(single_rank_tensors)

        self.assertIsInstance(lt, LocalTensor)
        self.assertEqual(len(lt._local_tensors), 1)

        # Operations should still work
        result = lt * 2.0
        self.assertIsInstance(result, LocalTensor)
        self.assertEqual(len(result._local_tensors), 1)

    def test_complex_operations(self):
        """Test more complex tensor operations."""
        lt = LocalTensor(self.identical_local_tensors)

        # Chain multiple operations
        result = ((lt + 1.0) * 2.0).transpose(0, 1).sum(dim=0)
        self.assertIsInstance(result, LocalTensor)

        # Verify the result makes sense
        for rank in self.identical_local_tensors.keys():
            expected = (
                ((self.identical_local_tensors[rank] + 1.0) * 2.0)
                .transpose(0, 1)
                .sum(dim=0)
            )
            self.assertEqual(result._local_tensors[rank], expected)

    def test_collective_reduction_operations(self):
        """Test different reduction operations for all_reduce."""
        # Create different tensors for each rank with simple values for testing
        test_tensors = {
            0: torch.tensor([[1.0, 4.0], [2.0, 5.0]]),
            1: torch.tensor([[2.0, 1.0], [3.0, 6.0]]),
            2: torch.tensor([[3.0, 2.0], [1.0, 4.0]]),
        }

        # Set up process group once
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=3
        )
        fake_pg = torch.distributed.distributed_c10d._get_default_group()

        # Test SUM reduction
        lt_sum = LocalTensor({k: v.clone() for k, v in test_tensors.items()})
        dist.all_reduce(lt_sum, op=dist.ReduceOp.SUM, group=fake_pg)
        expected_sum = torch.tensor([[6.0, 7.0], [6.0, 15.0]])  # Sum of all tensors
        for rank in test_tensors.keys():
            self.assertEqual(lt_sum._local_tensors[rank], expected_sum)

        # Test MAX reduction
        lt_max = LocalTensor({k: v.clone() for k, v in test_tensors.items()})
        dist.all_reduce(lt_max, op=dist.ReduceOp.MAX, group=fake_pg)
        expected_max = torch.tensor([[3.0, 4.0], [3.0, 6.0]])  # Max across all tensors
        for rank in test_tensors.keys():
            self.assertEqual(lt_max._local_tensors[rank], expected_max)

        # Test MIN reduction
        lt_min = LocalTensor({k: v.clone() for k, v in test_tensors.items()})
        dist.all_reduce(lt_min, op=dist.ReduceOp.MIN, group=fake_pg)
        expected_min = torch.tensor([[1.0, 1.0], [1.0, 4.0]])  # Min across all tensors
        for rank in test_tensors.keys():
            self.assertEqual(lt_min._local_tensors[rank], expected_min)

    def test_collectives_within_local_tensor_mode(self):
        """Test that collective operations work within LocalTensorMode context."""
        test_tensors = {
            0: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            1: torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        }
        lt = LocalTensor(test_tensors)
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=2
        )
        fake_pg = torch.distributed.distributed_c10d._get_default_group()

        with LocalTensorMode(lt._ranks):
            # Test all_reduce within mode
            lt_sum = LocalTensor({k: v.clone() for k, v in test_tensors.items()})
            dist.all_reduce(lt_sum, group=fake_pg)

            expected_sum = torch.tensor([[6.0, 8.0], [10.0, 12.0]])
            for rank in test_tensors.keys():
                self.assertEqual(lt_sum._local_tensors[rank], expected_sum)

            # Test broadcast within mode
            lt_broadcast = LocalTensor({k: v.clone() for k, v in test_tensors.items()})
            dist.broadcast(lt_broadcast, src=0, group=fake_pg)

            for rank in test_tensors.keys():
                self.assertEqual(lt_broadcast._local_tensors[rank], test_tensors[0])

            # Test that regular operations still work
            result = lt + 1.0
            self.assertIsInstance(result, LocalTensor)

    def test_all_reduce_collective(self):
        """Test that all_reduce collective operation works correctly with LocalTensor."""
        # Create different tensors for each rank
        different_tensors = {
            0: torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            1: torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
            2: torch.tensor([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]),
        }

        # Create a fake process group for testing
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=3
        )
        fake_pg = torch.distributed.distributed_c10d._get_default_group()

        # Test all_reduce with SUM (default)
        lt_sum = LocalTensor({k: v.clone() for k, v in different_tensors.items()})
        lt_sum = lt_sum + 1
        dist.all_reduce(lt_sum, group=fake_pg)

        # Verify all ranks have the sum of all tensors (after adding 1 to each)
        expected_sum = torch.tensor([[114.0, 225.0, 336.0], [447.0, 558.0, 669.0]])
        for rank in different_tensors.keys():
            self.assertEqual(lt_sum._local_tensors[rank], expected_sum)

    def test_broadcast_collective(self):
        """Test that broadcast collective operation works correctly with LocalTensor."""
        # Create different tensors for each rank
        different_tensors = {
            0: torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            1: torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
            2: torch.tensor([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]),
        }

        # Create a fake process group for testing
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=3
        )
        fake_pg = torch.distributed.distributed_c10d._get_default_group()

        # Test broadcast from rank 1
        lt_broadcast = LocalTensor({k: v.clone() for k, v in different_tensors.items()})
        dist.broadcast(lt_broadcast, src=1, group=fake_pg)

        # Verify all ranks have rank 1's original tensor
        expected_broadcast = different_tensors[1]
        for rank in different_tensors.keys():
            self.assertEqual(lt_broadcast._local_tensors[rank], expected_broadcast)

    def test_all_gather_collective(self):
        """Test that all_gather collective operation works correctly with LocalTensor."""
        # Create different tensors for each rank
        different_tensors = {
            0: torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            1: torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
            2: torch.tensor([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]),
        }

        # Create a fake process group for testing
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=3
        )
        fake_pg = torch.distributed.distributed_c10d._get_default_group()

        # Test all_gather
        lt_gather = LocalTensor(different_tensors)
        tensor_list = [torch.zeros_like(lt_gather) for _ in range(3)]

        dist.all_gather(tensor_list, lt_gather, group=fake_pg)

        # Verify each position in tensor_list contains the corresponding rank's tensor
        self.assertEqual(tensor_list[0], different_tensors[0])
        self.assertEqual(tensor_list[1], different_tensors[1])
        self.assertEqual(tensor_list[2], different_tensors[2])

    def test_non_collective_operations_work(self):
        """Test that regular operations still work and don't trigger collective detection."""
        lt = LocalTensor(self.identical_local_tensors)

        with LocalTensorMode(lt._ranks):
            # These should work fine
            result1 = lt + 1.0
            result2 = lt.sum()
            result3 = lt.transpose(0, 1)
            result4 = torch.relu(lt)

            self.assertIsInstance(result1, LocalTensor)
            self.assertIsInstance(result2, LocalTensor)
            self.assertIsInstance(result3, LocalTensor)
            self.assertIsInstance(result4, LocalTensor)

    world_size = 2

    def build_device_mesh(self) -> DeviceMesh:
        return init_device_mesh("cpu", (self.world_size,))

    def test_dtensor_addmm(self):
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            # TODO: test other ranks too
            "fake",
            store=fake_store,
            rank=0,
            world_size=self.world_size,
        )
        fake_pg = torch.distributed.distributed_c10d._get_default_group()
        device_mesh = self.build_device_mesh()

        with LocalTensorMode(self.world_size) as mode:
            self.mode = mode

            shard_spec = [Shard(0)]
            replica_spec = [Replicate()]

            tensor_to_shard = torch.randn(12, 8)
            mat1 = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
            tensor_to_replicate = torch.randn(8, 4)
            mat2 = distribute_tensor(tensor_to_replicate, device_mesh, replica_spec)
            input_tensor = torch.randn(4)
            input = distribute_tensor(input_tensor, device_mesh, replica_spec)
            print(tensor_to_shard)
            print(mat1)
            print(tensor_to_replicate)
            print(mat2)
            print(input_tensor)
            print(input)

            dist_res = torch.addmm(input, mat1, mat2)
            local_res = torch.addmm(input_tensor, tensor_to_shard, tensor_to_replicate)
            print(dist_res)
            print(local_res)
            full_tensor = dist_res.full_tensor()
            print(full_tensor)
            self.assertEqual(full_tensor, local_res)


if __name__ == "__main__":
    run_tests()
