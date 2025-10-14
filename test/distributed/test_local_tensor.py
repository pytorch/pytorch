# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import (
    local_tensor_mode,
    LocalTensor,
    LocalTensorMode,
)
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class LocalTensorTestBase(TestCase):
    def assertEqual(self, lhs, rhs, **kwargs):
        mode = local_tensor_mode()
        with nullcontext() if mode is None else mode.disable():
            if isinstance(lhs, LocalTensor) and isinstance(rhs, LocalTensor):
                assert isinstance(lhs, LocalTensor) and isinstance(rhs, LocalTensor)
                super().assertEqual(lhs._ranks, rhs._ranks)
                for r in lhs._ranks:
                    super().assertEqual(
                        lhs._local_tensors[r],
                        rhs._local_tensors[r],
                        lambda m: f"rank {r}: {m}",
                    )
            elif isinstance(lhs, LocalTensor) or isinstance(rhs, LocalTensor):
                lhs, rhs = (lhs, rhs) if isinstance(lhs, LocalTensor) else (rhs, lhs)
                for r in lhs._ranks:
                    super().assertEqual(
                        lhs._local_tensors[r], rhs, lambda m: f"rank {r}: {m}"
                    )
            else:
                return super().assertEqual(lhs, rhs, **kwargs)

    @property
    def world_size(self):
        raise NotImplementedError("override world-size in your subclass")

    def build_device_mesh(self) -> DeviceMesh:
        return init_device_mesh("cpu", (self.world_size,))

    def setUp(self):
        super().setUp()
        torch.distributed.init_process_group(
            # TODO: test other ranks too
            "fake",
            rank=0,
            world_size=self.world_size,
        )

    def tearDown(self):
        super().tearDown()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass


class TestLocalTensorWorld2(LocalTensorTestBase):
    world_size = 2

    def test_local_tensor_dtype_consistency(self):
        """Test that LocalTensor enforces dtype consistency."""
        device = torch.device("cpu")
        shape = (2, 3)

        inconsistent_tensors = {
            0: torch.randn(shape, dtype=torch.float32, device=device),
            1: torch.randn(
                shape, dtype=torch.float64, device=device
            ),  # Different dtype
        }

        with self.assertRaises(AssertionError):
            LocalTensor(inconsistent_tensors)

    def test_local_tensor_creation_fails_with_grad_tensors(self):
        """Test that LocalTensor creation fails when local tensors have requires_grad=True."""
        device = torch.device("cpu")
        shape = (2, 3)
        dtype = torch.float32

        # Create sample local tensors for different ranks
        local_tensors = {
            0: torch.randn(shape, dtype=dtype, device=device, requires_grad=True),
            1: torch.randn(shape, dtype=dtype, device=device, requires_grad=True),
        }

        with self.assertRaises(AssertionError):
            LocalTensor(local_tensors)

        # TODO: test flatten/unflatten

    def test_basic_arithmetic_operations(self):
        """Test basic arithmetic operations on LocalTensors."""
        device = torch.device("cpu")
        shape = (2, 3)
        dtype = torch.float32

        # Create identical local tensors for consistency tests
        base_tensor = torch.randn(shape, dtype=dtype, device=device)
        identical_local_tensors = {
            0: base_tensor.clone(),
            1: base_tensor.clone(),
        }

        lt1 = LocalTensor(identical_local_tensors)
        lt2 = LocalTensor(identical_local_tensors)

        # Test addition
        result_add = lt1 + lt2
        self.assertIsInstance(result_add, LocalTensor)
        self.assertEqual(len(result_add._local_tensors), 2)

        # Verify the operation was applied to each local tensor
        for rank in identical_local_tensors.keys():
            expected = identical_local_tensors[rank] + identical_local_tensors[rank]
            self.assertEqual(result_add._local_tensors[rank], expected)

        # Test multiplication
        result_mul = lt1 * 2.0
        self.assertIsInstance(result_mul, LocalTensor)
        for rank in identical_local_tensors.keys():
            expected = identical_local_tensors[rank] * 2.0
            self.assertEqual(result_mul._local_tensors[rank], expected)

    # TODO: consider an op-info test; we don't actually need to cover all ops
    # but it will help make sure views and more exotic things are done
    # correctly (in standard subclass style)

    def test_mixed_operations_with_regular_tensors(self):
        """Test operations between LocalTensors and regular tensors."""
        device = torch.device("cpu")
        shape = (2, 3)
        dtype = torch.float32

        # Create identical local tensors for consistency tests
        base_tensor = torch.randn(shape, dtype=dtype, device=device)
        identical_local_tensors = {
            0: base_tensor.clone(),
            1: base_tensor.clone(),
        }

        lt = LocalTensor(identical_local_tensors)
        regular_tensor = torch.ones_like(identical_local_tensors[0])

        # Test LocalTensor + regular tensor
        result = lt + regular_tensor
        self.assertIsInstance(result, LocalTensor)

        for rank in identical_local_tensors.keys():
            expected = identical_local_tensors[rank] + regular_tensor
            self.assertEqual(result._local_tensors[rank], expected)

    def test_local_tensor_mode(self):
        """Test LocalTensorMode functionality."""
        device = torch.device("cpu")
        shape = (2, 3)
        dtype = torch.float32

        # Create identical local tensors for consistency tests
        base_tensor = torch.randn(shape, dtype=dtype, device=device)
        identical_local_tensors = {
            0: base_tensor.clone(),
            1: base_tensor.clone(),
        }

        lt = LocalTensor(identical_local_tensors)

        with LocalTensorMode(lt._ranks):
            result = lt + 1.0
            self.assertIsInstance(result, LocalTensor)

            regular = torch.ones(2, 2)
            regular_result = regular + 1.0
            self.assertIsInstance(regular, LocalTensor)
            self.assertIsInstance(regular_result, LocalTensor)

    def test_empty_local_tensors(self):
        """Test behavior with empty local tensors dict."""
        # TODO: raise a better error here
        with self.assertRaises(StopIteration):  # next() on empty iterator
            LocalTensor({})

    def test_collectives_within_local_tensor_mode(self):
        """Test that collective operations work within LocalTensorMode context."""
        test_tensors = {
            0: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            1: torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        }
        lt = LocalTensor(test_tensors)
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

    def test_scalar_mul_reduction_bug(self):
        with LocalTensorMode(self.world_size):
            mesh = self.build_device_mesh()

            tensor = torch.tensor([10, 10]).float()
            dt = distribute_tensor(tensor, device_mesh=mesh, placements=[Shard(0)])
            y = dt.sum() * 1  # noqa: F841

            tensor = torch.arange(10).reshape(10, 1).float().requires_grad_()
            dt = distribute_tensor(tensor, device_mesh=mesh, placements=[Shard(0)])

            print(dt.sum() * 1, dt.sum() * 2, dt.sum() * 3)

    def test_uneven_sharding_mean_bug(self):
        with LocalTensorMode(self.world_size):
            mesh = self.build_device_mesh()
            tensor = torch.arange(12).reshape(-1, 4).float()

            dt = distribute_tensor(tensor, device_mesh=mesh, placements=[Shard(0)])

            mean = dt.mean()
            self.assertEqual(mean.placements, [Replicate()])
            full = mean.full_tensor()
            self.assertEqual(tensor.mean(), full)

    def test_uneven_sharding_prod(self):
        with LocalTensorMode(self.world_size):
            mesh = self.build_device_mesh()
            tensor = (torch.arange(12) + 1).reshape(-1, 4).float()

            dt = distribute_tensor(tensor, device_mesh=mesh, placements=[Shard(0)])

            x = dt.prod()
            full = x.full_tensor()
            self.assertEqual(tensor.prod(), full)

    def test_even_sharding_mean_is_partial(self):
        with LocalTensorMode(self.world_size):
            mesh = self.build_device_mesh()
            tensor = torch.arange(16).reshape(4, 4).float()

            dt = distribute_tensor(tensor, device_mesh=mesh, placements=[Shard(0)])

            mean = dt.mean()
            full = mean.full_tensor()
            self.assertEqual(tensor.mean(), full)
            self.assertEqual(mean.placements, [Partial("avg")])


class TestLocalTensorWorld3(LocalTensorTestBase):
    world_size = 3

    def test_collective_reduction_operations(self):
        """Test different reduction operations for all_reduce."""
        # Create different tensors for each rank with simple values for testing
        test_tensors = {
            0: torch.tensor([[1.0, 4.0], [2.0, 5.0]]),
            1: torch.tensor([[2.0, 1.0], [3.0, 6.0]]),
            2: torch.tensor([[3.0, 2.0], [1.0, 4.0]]),
        }

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

    def test_all_reduce_collective(self):
        """Test that all_reduce collective operation works correctly with LocalTensor."""
        # Create different tensors for each rank
        different_tensors = {
            0: torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            1: torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
            2: torch.tensor([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]),
        }

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

        fake_pg = torch.distributed.distributed_c10d._get_default_group()

        # Test all_gather
        lt_gather = LocalTensor(different_tensors)
        tensor_list = [torch.zeros_like(lt_gather) for _ in range(3)]

        dist.all_gather(tensor_list, lt_gather, group=fake_pg)

        # Verify each position in tensor_list contains the corresponding rank's tensor
        self.assertEqual(tensor_list[0], different_tensors[0])
        self.assertEqual(tensor_list[1], different_tensors[1])
        self.assertEqual(tensor_list[2], different_tensors[2])


class TestLocalTensorWorld4(LocalTensorTestBase):
    world_size = 4

    def test_dtensor_cat(self):
        with LocalTensorMode(self.world_size):
            device_mesh = self.build_device_mesh()

            t1 = torch.arange(16).view(4, 4).float()
            d1 = distribute_tensor(t1, device_mesh, [Replicate()])
            t2 = (torch.arange(16) + 16).view(4, 4).float()
            d2 = distribute_tensor(t2, device_mesh, [Shard(0)])

            local_res = torch.cat([t1, t2], dim=-1)
            dist_res = torch.cat([d1, d2], dim=-1)
            full_tensor = dist_res.full_tensor()
            self.assertEqual(full_tensor, local_res)


class TestLocalTensorWorld8(LocalTensorTestBase):
    world_size = 8

    def test_dtensor_addmm(self):
        with LocalTensorMode(self.world_size):
            device_mesh = self.build_device_mesh()

            shard_spec = [Shard(0)]
            replica_spec = [Replicate()]

            tensor_to_shard = torch.randn(12, 8)
            mat1 = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
            tensor_to_replicate = torch.randn(8, 4)
            mat2 = distribute_tensor(tensor_to_replicate, device_mesh, replica_spec)
            input_tensor = torch.randn(4)
            input = distribute_tensor(input_tensor, device_mesh, replica_spec)

            dist_res = torch.addmm(input, mat1, mat2)
            local_res = torch.addmm(input_tensor, tensor_to_shard, tensor_to_replicate)
            full_tensor = dist_res.full_tensor()
            self.assertEqual(full_tensor, local_res)


if __name__ == "__main__":
    run_tests()
