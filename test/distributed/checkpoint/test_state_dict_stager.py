# Owner(s): ["oncall: distributed"]

import dataclasses
import os
import tempfile
import unittest
from datetime import timedelta

import psutil

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import (
    init_from_local_shards,
    Shard as ShardedTensorShard,
    ShardedTensor,
    ShardMetadata,
)
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.checkpoint._state_dict_stager import StateDictStager
from torch.distributed.checkpoint.staging import _ReplicationStager
from torch.distributed.checkpoint.state_dict_saver import async_save
from torch.distributed.tensor import DeviceMesh, distribute_tensor
from torch.testing._internal.common_distributed import (
    HAS_ACCELERATOR,
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def create_cpu_state_dict(state_dict):
    cpu_state_dict = {}
    for key, value in state_dict.items():
        cpu_state_dict[key] = value.cpu()
    return cpu_state_dict


def compare_state_dicts(gpu_state_dict, cpu_state_dict, rtol=1e-5, atol=1e-8):
    """
    Compare if two state dictionaries (one on GPU, one on CPU) are otherwise the same.

    This function checks if the tensors in both state dictionaries have the same values,
    shapes, dtypes, etc., ignoring the device difference. It also checks if tensors that
    share storage in one state dict also share storage in the other.

    Args:
        gpu_state_dict: The state dictionary with tensors on GPU
        cpu_state_dict: The state dictionary with tensors on CPU
        rtol: Relative tolerance for comparing tensor values
        atol: Absolute tolerance for comparing tensor values

    Returns:
        bool: True if the state dictionaries are equivalent, False otherwise
        str: Error message if the state dictionaries are not equivalent, empty string otherwise
    """
    # Track storage data pointers to check storage sharing
    gpu_storage_ptrs = {}
    cpu_storage_ptrs = {}

    def compare_objects(gpu_obj, cpu_obj, path=""):
        # If objects are tensors, compare them
        if isinstance(gpu_obj, torch.Tensor) and isinstance(cpu_obj, torch.Tensor):
            # Check if devices are as expected
            if gpu_obj.device.type != device_type:
                return (
                    False,
                    f"Expected accelerator tensor, got {gpu_obj.device.type} tensor at {path}",
                )
            if cpu_obj.device.type != "cpu":
                return (
                    False,
                    f"Expected CPU tensor, got {cpu_obj.device.type} tensor at {path}",
                )
            if gpu_obj.storage_offset() != cpu_obj.storage_offset():
                return (
                    False,
                    f"Storage offset mismatch at {path}: {gpu_obj.storage_offset()} vs {cpu_obj.storage_offset()}",
                )

            if not torch.equal(gpu_obj.cpu(), cpu_obj):
                return (
                    False,
                    f"Tensors are not same at {path}",
                )

            # Track storage sharing
            gpu_storage_ptr = gpu_obj.storage().data_ptr()
            cpu_storage_ptr = cpu_obj.storage().data_ptr()

            if gpu_storage_ptr in gpu_storage_ptrs:
                # This GPU tensor shares storage with another tensor
                # Check if the corresponding CPU tensors also share storage
                if cpu_storage_ptr != gpu_storage_ptrs[gpu_storage_ptr]:
                    return (
                        False,
                        f"Storage sharing mismatch: GPU tensors share storage but CPU tensors don't at {path}",
                    )
            else:
                # First time seeing this storage
                gpu_storage_ptrs[gpu_storage_ptr] = cpu_storage_ptr
                cpu_storage_ptrs[cpu_storage_ptr] = gpu_storage_ptr

            return True, ""

        # If objects are dictionaries, compare them recursively
        elif isinstance(gpu_obj, dict) and isinstance(cpu_obj, dict):
            if gpu_obj.keys() != cpu_obj.keys():
                return (
                    False,
                    f"Dictionary keys mismatch at {path}: {gpu_obj.keys()} vs {cpu_obj.keys()}",
                )

            for key in gpu_obj:
                result, error = compare_objects(
                    gpu_obj[key], cpu_obj[key], f"{path}.{key}" if path else key
                )
                if not result:
                    return False, error

            return True, ""

        # If objects are lists, tuples, or sets, compare them recursively
        elif isinstance(gpu_obj, (list, tuple, set)) and isinstance(
            cpu_obj, (list, tuple, set)
        ):
            if len(gpu_obj) != len(cpu_obj):
                return (
                    False,
                    f"Collection length mismatch at {path}: {len(gpu_obj)} vs {len(cpu_obj)}",
                )
            if type(gpu_obj) is not type(cpu_obj):
                return (
                    False,
                    f"Collection type mismatch at {path}: {type(gpu_obj)} vs {type(cpu_obj)}",
                )

            for i, (gpu_item, cpu_item) in enumerate(zip(gpu_obj, cpu_obj)):
                result, error = compare_objects(gpu_item, cpu_item, f"{path}[{i}]")
                if not result:
                    return False, error

            return True, ""

        # If objects are custom classes, compare their attributes
        elif hasattr(gpu_obj, "__dict__") and hasattr(cpu_obj, "__dict__"):
            if type(gpu_obj) is not type(cpu_obj):
                return (
                    False,
                    f"Object type mismatch at {path}: {type(gpu_obj)} vs {type(cpu_obj)}",
                )

            result, error = compare_objects(
                gpu_obj.__dict__, cpu_obj.__dict__, f"{path}.__dict__"
            )
            if not result:
                return False, error

            return True, ""

        # For other types, use direct equality comparison
        else:
            if type(gpu_obj) is not type(cpu_obj):
                return (
                    False,
                    f"Type mismatch at {path}: {type(gpu_obj)} vs {type(cpu_obj)}",
                )
            if gpu_obj != cpu_obj:
                return False, f"Value mismatch at {path}: {gpu_obj} vs {cpu_obj}"

            return True, ""

    # Start the recursive comparison
    result, error = compare_objects(gpu_state_dict, cpu_state_dict)
    return result, error


@dataclasses.dataclass
class TestStruct:
    tensor1: torch.Tensor


@dataclasses.dataclass
class NestedTensorStruct:
    tensor: torch.Tensor
    value: int = 42


@dataclasses.dataclass
class ComplexDataClass:
    tensor: torch.Tensor
    name: str
    values: list[float]
    nested: NestedTensorStruct


@dataclasses.dataclass(frozen=True)
class FrozenDataClass:
    tensor: torch.Tensor
    value: int = 100


class TestStateDictStager(TestCase):
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_views(self):
        test_configs = [
            (False, False),  # pin_memory=False, share_memory=False,
            (True, False),  # pin_memory=True, share_memory=False
            (False, True),  # pin_memory=False, share_memory=True
            (True, True),  # pin_memory=True, share_memory=True
        ]
        for pin_memory, share_memory in test_configs:
            with self.subTest(pin_memory=pin_memory, share_memory=share_memory):
                tensor1 = torch.randn(4, 4).to(device_type)
                tensor2 = tensor1.view(16)
                tensor3 = torch.randn(4, 4).to(device_type)
                state_dict = {
                    "tensor1": tensor1,
                    "tensor2": tensor2,
                    "recursive": {
                        "tensor3": tensor3,
                        "type": TestStruct(tensor1=tensor3.narrow(0, 0, 2)),
                    },
                }
                if (
                    state_dict["tensor1"].storage().data_ptr()
                    != state_dict["tensor2"].storage().data_ptr()
                ):
                    raise AssertionError("tensor1 and tensor2 should share storage")

                stager = StateDictStager(
                    pin_memory=pin_memory, share_memory=share_memory
                )

                cpu_state_dict = stager.stage(state_dict)

                # Calculate stats
                num_storages = len(stager._cached_storage_mapping)
                num_bytes = sum(
                    storage.nbytes()
                    for storage in stager._cached_storage_mapping.values()
                )

                # Validate tensor count and bytes
                expected_storage_cnt = 2
                if num_storages != expected_storage_cnt:
                    raise AssertionError(
                        f"Expected {expected_storage_cnt} storages, got {num_storages}"
                    )

                # Calculate expected bytes
                # Note: Only unique storages are counted in the byte count
                expected_bytes = (
                    tensor1.numel() * tensor1.element_size()
                    + tensor3.numel()  # tensor1 and tensor2 share storage
                    * tensor3.element_size()  # tensor3 and its narrow view share storage
                )
                if num_bytes != expected_bytes:
                    raise AssertionError(
                        f"Expected {expected_bytes} bytes, got {num_bytes}"
                    )
                # Verify that the CPU state dict is equivalent to the original GPU state dict
                result, error = compare_state_dicts(state_dict, cpu_state_dict)
                if not result:
                    raise AssertionError(f"State dicts are not equivalent: {error}")

                # Additional checks for storage sharing
                if cpu_state_dict["tensor1"].device != torch.device("cpu"):
                    raise AssertionError(
                        f"Expected tensor1 on cpu, got {cpu_state_dict['tensor1'].device}"
                    )
                if cpu_state_dict["tensor2"].device != torch.device("cpu"):
                    raise AssertionError(
                        f"Expected tensor2 on cpu, got {cpu_state_dict['tensor2'].device}"
                    )
                if (
                    cpu_state_dict["tensor1"].storage().data_ptr()
                    != cpu_state_dict["tensor2"].storage().data_ptr()
                ):
                    raise AssertionError("cpu tensor1 and tensor2 should share storage")

                recursive = cpu_state_dict["recursive"]
                if recursive["tensor3"].device != torch.device("cpu"):
                    raise AssertionError(
                        f"Expected tensor3 on cpu, got {recursive['tensor3'].device}"
                    )
                if recursive["type"].tensor1.device != torch.device("cpu"):
                    raise AssertionError(
                        f"Expected type.tensor1 on cpu, got {recursive['type'].tensor1.device}"
                    )
                if (
                    recursive["tensor3"].storage().data_ptr()
                    != recursive["type"].tensor1.storage().data_ptr()
                ):
                    raise AssertionError(
                        "tensor3 and type.tensor1 should share storage"
                    )

    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_caching(self):
        """
        Test that the StateDictStager correctly caches and reuses storages.
        """
        test_configs = [
            (False, False),  # pin_memory=False, share_memory=False,
            (True, False),  # pin_memory=True, share_memory=False
            (False, True),  # pin_memory=False, share_memory=True
            (True, True),  # pin_memory=True, share_memory=True
        ]
        for pin_memory, share_memory in test_configs:
            with self.subTest(pin_memory=pin_memory, share_memory=share_memory):
                # Create test tensors and state dict
                tensor1 = torch.randn(4, 4).to(device_type)
                tensor2 = tensor1.view(16)
                tensor3 = torch.randn(4, 4).to(device_type)
                state_dict = {
                    "tensor1": tensor1,
                    "tensor2": tensor2,
                    "recursive": {
                        "tensor3": tensor3,
                        "type": TestStruct(tensor1=tensor3.narrow(0, 0, 2)),
                    },
                }

                # Create a StateDictStager instance
                stager = StateDictStager(
                    pin_memory=pin_memory, share_memory=share_memory
                )

                # First call to stage with staging context
                cpu_state_dict1 = stager.stage(state_dict)

                # Get the number of cached storages after first stage
                num_storages1 = len(stager._cached_storage_mapping)

                # Verify the first result is correct
                result, error = compare_state_dicts(state_dict, cpu_state_dict1)
                if not result:
                    raise AssertionError(
                        f"First state dict is not equivalent to original: {error}"
                    )

                # Modify the original tensors
                tensor1.fill_(0)
                tensor3.fill_(0)

                # Second call to stage with staging context
                cpu_state_dict2 = stager.stage(state_dict)

                # Get the number of cached storages after second stage
                num_storages2 = len(stager._cached_storage_mapping)

                # Verify that the second CPU state dict is equivalent to the modified original state dict
                result, error = compare_state_dicts(state_dict, cpu_state_dict2)
                if not result:
                    raise AssertionError(
                        f"Second state dict is not equivalent to modified original: {error}"
                    )

                # Verify that the number of cached storages hasn't changed
                if num_storages1 != num_storages2:
                    raise AssertionError(
                        f"Storage count changed: {num_storages1} vs {num_storages2}"
                    )

                # Verify that the tensors in the second state dict have the same storage pointers as the first
                if (
                    cpu_state_dict1["tensor1"].storage().data_ptr()
                    != cpu_state_dict2["tensor1"].storage().data_ptr()
                ):
                    raise AssertionError("Storage pointers should match for tensor1")
                if (
                    cpu_state_dict1["tensor2"].storage().data_ptr()
                    != cpu_state_dict2["tensor2"].storage().data_ptr()
                ):
                    raise AssertionError("Storage pointers should match for tensor2")
                if (
                    cpu_state_dict1["recursive"]["tensor3"].storage().data_ptr()
                    != cpu_state_dict2["recursive"]["tensor3"].storage().data_ptr()
                ):
                    raise AssertionError("Storage pointers should match for tensor3")

                # Modify the original tensors again with different values
                tensor1.fill_(42.0)

                # Third call to stage with staging context
                cpu_state_dict3 = stager.stage(state_dict)

                # Verify that the third CPU state dict reflects the updated values
                if not torch.all(cpu_state_dict3["tensor1"] == 42.0):
                    raise AssertionError(
                        "Updated values should be reflected in the cached state dict for tensor1"
                    )
                if not torch.all(cpu_state_dict3["tensor2"] == 42.0):
                    raise AssertionError(
                        "Updated values should be reflected in the cached state dict for tensor2"
                    )

    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_tensor_attrs(self):
        """
        Test that tensor attributes are preserved during stage with StateDictStager.
        """
        tensor1 = torch.randn(4, 4).to(device_type)
        tensor2 = tensor1.view(16)
        tensor3 = torch.randn(4, 4).to(device_type)

        # Add custom attributes to tensors
        tensor1.a = 42
        tensor1.b = 43
        tensor3.c = 44

        state_dict = {
            "tensor1": tensor1,
            "tensor2": tensor2,
            "recursive": {
                "tensor3": tensor3,
                "type": TestStruct(tensor1=tensor3.narrow(0, 0, 2)),
            },
        }

        stager = StateDictStager(pin_memory=True, share_memory=True)
        cpu_state_dict = stager.stage(state_dict)

        # Verify that tensor attributes are preserved
        if not hasattr(cpu_state_dict["tensor1"], "a"):
            raise AssertionError("Tensor attribute 'a' was not preserved")
        if cpu_state_dict["tensor1"].a != 42:
            raise AssertionError(
                f"Tensor attribute 'a' has incorrect value: {cpu_state_dict['tensor1'].a}"
            )
        if not hasattr(cpu_state_dict["tensor1"], "b"):
            raise AssertionError("Tensor attribute 'b' was not preserved")
        if cpu_state_dict["tensor1"].b != 43:
            raise AssertionError(
                f"Tensor attribute 'b' has incorrect value: {cpu_state_dict['tensor1'].b}"
            )
        if not hasattr(cpu_state_dict["recursive"]["tensor3"], "c"):
            raise AssertionError("Tensor attribute 'c' was not preserved")
        if cpu_state_dict["recursive"]["tensor3"].c != 44:
            raise AssertionError(
                f"Tensor attribute 'c' has incorrect value: {cpu_state_dict['recursive']['tensor3'].c}"
            )

    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_different_dtypes(self):
        """
        Test that StateDictStager works correctly with tensors of different data types.
        """
        # Create tensors with different dtypes
        tensors = {
            "float32": torch.randn(4, 4, dtype=torch.float32).to(device_type),
            "float64": torch.randn(4, 4, dtype=torch.float64).to(device_type),
            "int32": torch.randint(-100, 100, (4, 4), dtype=torch.int32).to(
                device_type
            ),
            "int64": torch.randint(-100, 100, (4, 4), dtype=torch.int64).to(
                device_type
            ),
            "bool": torch.randint(0, 2, (4, 4), dtype=torch.bool).to(device_type),
        }

        # Create a state dict with these tensors
        state_dict = tensors.copy()

        stager = StateDictStager()
        cpu_state_dict = stager.stage(state_dict)

        # Verify that all tensors have been correctly copied to CPU with the right dtypes
        for dtype_name, original_tensor in tensors.items():
            cpu_tensor = cpu_state_dict[dtype_name]
            self.assertEqual(
                cpu_tensor.device.type, "cpu", f"Tensor {dtype_name} should be on CPU"
            )
            self.assertEqual(
                cpu_tensor.dtype,
                original_tensor.dtype,
                f"Tensor {dtype_name} has incorrect dtype",
            )
            self.assertTrue(
                torch.allclose(cpu_tensor, original_tensor.cpu()),
                f"Tensor {dtype_name} has incorrect values",
            )

    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_empty_tensors(self):
        """
        Test that StateDictStager works correctly with empty tensors.
        """
        test_configs = [
            (False, False),  # pin_memory=False, share_memory=False,
            (True, False),  # pin_memory=True, share_memory=False
            (False, True),  # pin_memory=False, share_memory=True
            (True, True),  # pin_memory=True, share_memory=True
        ]
        for pin_memory, share_memory in test_configs:
            with self.subTest(pin_memory=pin_memory, share_memory=share_memory):
                # Create empty tensors with different shapes
                tensors = {
                    "empty_0d": torch.tensor([], dtype=torch.float32).to(device_type),
                    "empty_1d": torch.tensor([], dtype=torch.float32)
                    .reshape(0)
                    .to(device_type),
                    "empty_2d": torch.tensor([], dtype=torch.float32)
                    .reshape(0, 0)
                    .to(device_type),
                    "empty_3d": torch.tensor([], dtype=torch.float32)
                    .reshape(0, 0, 0)
                    .to(device_type),
                    "zero_dim": torch.tensor(0.0).to(device_type),  # scalar tensor
                }

                # Create a state dict with these tensors
                state_dict = tensors.copy()

                cpu_state_dict = StateDictStager(pin_memory, share_memory).stage(
                    state_dict
                )

                # Verify that all tensors have been correctly copied to CPU
                for tensor_name, original_tensor in tensors.items():
                    cpu_tensor = cpu_state_dict[tensor_name]

                    self.assertEqual(
                        cpu_tensor.device.type,
                        "cpu",
                        f"Tensor {tensor_name} should be on CPU",
                    )
                    self.assertEqual(
                        cpu_tensor.shape,
                        original_tensor.shape,
                        f"Tensor {tensor_name} has incorrect shape",
                    )
                    self.assertEqual(
                        cpu_tensor.dtype,
                        original_tensor.dtype,
                        f"Tensor {tensor_name} has incorrect dtype",
                    )

    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_complex_storage_sharing(self):
        """
        Test that StateDictStager correctly handles complex storage sharing scenarios.
        """
        # Create a base tensor
        base_tensor = torch.randn(10, 10).to(device_type)

        # Create various views and slices that share storage
        view1 = base_tensor.view(100)
        view2 = base_tensor.view(10, 10)
        slice1 = base_tensor[2:8, 2:8]
        slice2 = base_tensor[:, :5]
        slice3 = view1[10:60]

        # Create a state dict with these tensors
        state_dict = {
            "base": base_tensor,
            "view1": view1,
            "view2": view2,
            "slice1": slice1,
            "slice2": slice2,
            "slice3": slice3,
        }
        cpu_state_dict = StateDictStager().stage(state_dict)

        # Verify that all tensors have been correctly copied to CPU
        result, error = compare_state_dicts(state_dict, cpu_state_dict)
        self.assertTrue(result, f"State dicts are not equivalent: {error}")

        # Verify storage sharing is preserved
        # All these tensors should share the same storage
        storage_ptr = cpu_state_dict["base"].storage().data_ptr()
        self.assertEqual(
            cpu_state_dict["view1"].storage().data_ptr(),
            storage_ptr,
            "view1 should share storage with base",
        )
        self.assertEqual(
            cpu_state_dict["view2"].storage().data_ptr(),
            storage_ptr,
            "view2 should share storage with base",
        )
        self.assertEqual(
            cpu_state_dict["slice1"].storage().data_ptr(),
            storage_ptr,
            "slice1 should share storage with base",
        )
        self.assertEqual(
            cpu_state_dict["slice2"].storage().data_ptr(),
            storage_ptr,
            "slice2 should share storage with base",
        )
        self.assertEqual(
            cpu_state_dict["slice3"].storage().data_ptr(),
            storage_ptr,
            "slice3 should share storage with base",
        )

        # Verify that modifying the base tensor affects all views and slices
        cpu_state_dict["base"].fill_(42.0)
        self.assertTrue(
            torch.all(cpu_state_dict["view1"] == 42.0),
            "view1 should reflect changes to base",
        )
        self.assertTrue(
            torch.all(cpu_state_dict["view2"] == 42.0),
            "view2 should reflect changes to base",
        )
        self.assertTrue(
            torch.all(cpu_state_dict["slice1"] == 42.0),
            "slice1 should reflect changes to base",
        )
        self.assertTrue(
            torch.all(cpu_state_dict["slice2"] == 42.0),
            "slice2 should reflect changes to base",
        )
        self.assertTrue(
            torch.all(cpu_state_dict["slice3"] == 42.0),
            "slice3 should reflect changes to base",
        )

    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_dataclasses(self):
        # Create tensors
        tensor1 = torch.randn(4, 4).to(device_type)
        tensor2 = torch.randn(8, 8).to(device_type)
        tensor3 = torch.randn(2, 6).to(device_type)
        tensor4 = torch.randn(3, 5).to(device_type)

        # Create dataclass instances
        nested = NestedTensorStruct(tensor=tensor3)
        complex_dc = ComplexDataClass(
            tensor=tensor1, name="test", values=[1.0, 2.0, 3.0], nested=nested
        )
        frozen_dc = FrozenDataClass(tensor=tensor4)

        # Create a state dict with these dataclasses
        state_dict = {
            "regular_tensor": tensor2,
            "complex_dataclass": complex_dc,
            "frozen_dataclass": frozen_dc,
        }

        # Stage the state dict
        stager = StateDictStager(pin_memory=False, share_memory=False)
        cpu_state_dict = stager.stage(state_dict)

        # Verify regular tensor
        self.assertEqual(cpu_state_dict["regular_tensor"].device.type, "cpu")
        self.assertTrue(torch.allclose(cpu_state_dict["regular_tensor"], tensor2.cpu()))

        # Verify complex dataclass
        complex_cpu = cpu_state_dict["complex_dataclass"]
        self.assertEqual(complex_cpu.name, "test")
        self.assertEqual(complex_cpu.values, [1.0, 2.0, 3.0])
        self.assertEqual(complex_cpu.tensor.device.type, "cpu")
        self.assertTrue(torch.allclose(complex_cpu.tensor, tensor1.cpu()))

        # Verify nested dataclass inside complex dataclass
        nested_cpu = complex_cpu.nested
        self.assertEqual(nested_cpu.value, 42)
        self.assertEqual(nested_cpu.tensor.device.type, "cpu")
        self.assertTrue(torch.allclose(nested_cpu.tensor, tensor3.cpu()))

        # Verify frozen dataclass
        frozen_cpu = cpu_state_dict["frozen_dataclass"]
        self.assertEqual(frozen_cpu.value, 100)
        self.assertEqual(frozen_cpu.tensor.device.type, "cpu")
        self.assertTrue(torch.allclose(frozen_cpu.tensor, tensor4.cpu()))

        # Verify that modifying the original tensors doesn't affect the staged ones
        tensor1.fill_(99.0)
        tensor3.fill_(88.0)
        tensor4.fill_(77.0)

        self.assertFalse(torch.allclose(complex_cpu.tensor, tensor1.cpu()))
        self.assertFalse(torch.allclose(nested_cpu.tensor, tensor3.cpu()))
        self.assertFalse(torch.allclose(frozen_cpu.tensor, tensor4.cpu()))

    def test_cpu_storage_independence(self):
        """
        Test ensures CPU tensors passed to StateDictStager are actually cloned
        """
        # Create test tensors
        tensor1 = torch.randn(4, 4)
        tensor2 = torch.randn(8, 8)

        # Create a state dict with these tensors
        state_dict = {
            "tensor1": tensor1,
            "tensor2": tensor2,
        }

        cpu_state_dict = StateDictStager().stage(state_dict)
        cpu_tensor1 = cpu_state_dict["tensor1"]
        cpu_tensor2 = cpu_state_dict["tensor2"]

        # Verify that the CPU tensors have different storage pointers than the original tensors
        self.assertNotEqual(
            tensor1.storage().data_ptr(),
            cpu_tensor1.storage().data_ptr(),
            "CPU tensor should have a different storage pointer than the original tensor",
        )
        self.assertNotEqual(
            tensor2.storage().data_ptr(),
            cpu_tensor2.storage().data_ptr(),
            "CPU tensor should have a different storage pointer than the original tensor",
        )

        self.assertTrue(
            torch.allclose(tensor1, cpu_tensor1),
            "CPU tensor should have the same values as the original tensor",
        )
        self.assertTrue(
            torch.allclose(tensor2, cpu_tensor2),
            "CPU tensor should have the same values as the original tensor",
        )

        # Modify the original CPU tensors and validate staged tensors are not modified
        cloned_orginial1 = tensor1.clone()
        cloned_orginia2 = tensor2.clone()
        tensor1.fill_(99.0)
        tensor2.fill_(88.0)

        self.assertFalse(torch.allclose(cloned_orginial1, tensor1))
        self.assertTrue(
            torch.allclose(cloned_orginial1, cpu_tensor1),
            "CPU tensor should have the same values as the original tensor",
        )
        self.assertTrue(
            torch.allclose(cloned_orginia2, cpu_tensor2),
            "CPU tensor should have the same values as the original tensor",
        )

    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_tensor_pinned_and_shared(self):
        """
        Test that verifies tensors are actually pinned and shared using tensor.is_pinned() and tensor.is_shared() methods.
        """
        # Create test tensors
        tensor1 = torch.randn(4, 4).to(device_type)
        tensor2 = torch.randn(8, 8).to(device_type)

        # Create a state dict with these tensors
        state_dict = {
            "tensor1": tensor1,
            "tensor2": tensor2,
        }

        # Test all combinations of pin_memory and share_memory
        test_configs = [
            (False, False),  # pin_memory=False, share_memory=False
            (True, False),  # pin_memory=True, share_memory=False
            (False, True),  # pin_memory=False, share_memory=True
            (True, True),  # pin_memory=True, share_memory=True
        ]

        for pin_memory, share_memory in test_configs:
            with self.subTest(pin_memory=pin_memory, share_memory=share_memory):
                # Create stager with specific configuration
                stager = StateDictStager(
                    pin_memory=pin_memory, share_memory=share_memory
                )
                cpu_state_dict = stager.stage(state_dict)

                # Get the staged tensors
                cpu_tensor1 = cpu_state_dict["tensor1"]
                cpu_tensor2 = cpu_state_dict["tensor2"]

                # Verify tensor device
                self.assertEqual(
                    cpu_tensor1.device.type, "cpu", "Staged tensor should be on CPU"
                )
                self.assertEqual(
                    cpu_tensor2.device.type, "cpu", "Staged tensor should be on CPU"
                )

                # Verify tensor values
                self.assertTrue(
                    torch.allclose(cpu_tensor1, tensor1.cpu()),
                    "CPU tensor should have the same values as the original tensor",
                )
                self.assertTrue(
                    torch.allclose(cpu_tensor2, tensor2.cpu()),
                    "CPU tensor should have the same values as the original tensor",
                )

                # Verify pinned memory status
                self.assertEqual(
                    cpu_tensor1.is_pinned(),
                    pin_memory,
                    f"Tensor pinned status should be {pin_memory}",
                )
                self.assertEqual(
                    cpu_tensor2.is_pinned(),
                    pin_memory,
                    f"Tensor pinned status should be {pin_memory}",
                )

                # Verify shared memory status
                self.assertEqual(
                    cpu_tensor1.is_shared(),
                    share_memory,
                    f"Tensor shared status should be {share_memory}",
                )
                self.assertEqual(
                    cpu_tensor2.is_shared(),
                    share_memory,
                    f"Tensor shared status should be {share_memory}",
                )

                # Verify storage sharing is consistent with tensor sharing
                if share_memory:
                    # When share_memory is True, the storage should also be shared
                    self.assertTrue(
                        cpu_tensor1.storage().is_shared(),
                        "When share_memory=True, tensor storage should be shared",
                    )
                    self.assertTrue(
                        cpu_tensor2.storage().is_shared(),
                        "When share_memory=True, tensor storage should be shared",
                    )
                else:
                    # When share_memory is False, the storage should not be shared
                    self.assertFalse(
                        cpu_tensor1.storage().is_shared(),
                        "When share_memory=False, tensor storage should not be shared",
                    )
                    self.assertFalse(
                        cpu_tensor2.storage().is_shared(),
                        "When share_memory=False, tensor storage should not be shared",
                    )


class TestDTensorStateDictStager(DTensorTestBase):
    @with_comms
    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_dtensor(self):
        """
        Test that StateDictStager works correctly with DTensors.
        """
        # Create a DTensor
        device_mesh = dist.DeviceMesh(
            self.device_type, list(range(dist.get_world_size()))
        )
        tensor = torch.randn(3, 3, device=self.device_type)
        dtensor = DTensor.from_local(tensor, device_mesh, [Shard(0)])

        dtensor = dtensor + 1
        dtensor = dtensor * 2

        state_dict = {
            "dtensor": dtensor,
        }

        stager = StateDictStager(pin_memory=True, share_memory=True)
        cpu_state_dict = stager.stage(state_dict)

        # Verify the original DTensor has the expected values
        self.assertTrue(torch.allclose(dtensor.to_local(), (tensor + 1) * 2))
        self.assertTrue(
            torch.allclose(
                cpu_state_dict["dtensor"].to_local(), dtensor.to_local().cpu()
            )
        )
        self.assertEqual(cpu_state_dict["dtensor"]._spec, dtensor._spec)
        self.assertEqual(cpu_state_dict["dtensor"].size(), dtensor.size())

    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_async_save_no_memory_leak(self):
        # repeatedly calling async_save should not cause memory to grow by
        # the checkpoint size each iteration
        model_size_mb = 128
        num_elements = (model_size_mb * 1024 * 1024) // 4
        tensor = torch.randn(num_elements, device=device_type).to(torch.float32)
        state_dict = {"weights": tensor}

        with tempfile.TemporaryDirectory() as temp_dir:
            # warmup - first save may allocate some buffers
            f = async_save(
                state_dict, checkpoint_id=os.path.join(temp_dir, "warmup"), no_dist=True
            )
            f.result()

            baseline = psutil.Process().memory_info().rss / (1024 * 1024)

            num_saves = 5
            for i in range(num_saves):
                f = async_save(
                    state_dict,
                    checkpoint_id=os.path.join(temp_dir, f"step_{i}"),
                    no_dist=True,
                )
                f.result()

            final = psutil.Process().memory_info().rss / (1024 * 1024)

            growth = final - baseline
            max_allowed = model_size_mb * 1.2

            self.assertLess(
                growth,
                max_allowed,
                f"Memory grew {growth:.0f}MB over {num_saves} saves (baseline={baseline:.0f}MB). "
                f"This indicates a memory leak. Max allowed: {max_allowed:.0f}MB",
            )


class TestReplicationStager(DTensorTestBase):
    """
    Test suite for _ReplicationStager functionality.
    Tests replication of state_dict across training ranks using CPU tensors only.
    """

    @property
    def backend(self) -> str:
        return "cpu:gloo,cuda:nccl"

    def _create_simple_state_dict(self, rank: int) -> dict:
        """
        Create a simple state_dict with CPU tensors, deterministically unique per rank.

        Args:
            rank: The rank number to create unique tensors for

        Returns:
            dict: A state dictionary with CPU tensors
        """
        # Create unique tensors for each rank
        torch.manual_seed(42 + rank)  # Different seed per rank

        return {
            "layer1.weight": torch.randn(64, 128, device="cpu"),
            "layer1.bias": torch.randn(64, device="cpu"),
            "layer2.weight": torch.randn(32, 64, device="cpu"),
            "layer2.bias": torch.randn(32, device="cpu"),
            "nested": {
                "param": torch.randn(16, 16, device="cpu"),
                "buffer": torch.randn(8, device="cpu"),
            },
            "scalar": torch.tensor(float(rank), device="cpu"),
        }

    def _verify_simple_state_dict_replication(
        self, replicated_dict: dict, rank: int, partner_rank: int
    ):
        """
        Verify that replication worked correctly.

        Args:
            replicated_dict: The replicated state_dict received from partner
            rank: Current rank
            partner_rank: Partner rank we should have received from
        """
        # Create expected state_dict (what partner rank would have created)
        expected_dict = self._create_simple_state_dict(partner_rank)

        def compare_tensors(actual, expected, path=""):
            if isinstance(actual, dict) and isinstance(expected, dict):
                self.assertEqual(
                    actual.keys(), expected.keys(), f"Keys mismatch at {path}"
                )
                for key in actual:
                    compare_tensors(
                        actual[key], expected[key], f"{path}.{key}" if path else key
                    )
            elif isinstance(actual, torch.Tensor) and isinstance(
                expected, torch.Tensor
            ):
                self.assertEqual(
                    actual.device.type, "cpu", f"Tensor at {path} should be on CPU"
                )
                self.assertEqual(
                    actual.shape, expected.shape, f"Shape mismatch at {path}"
                )
                self.assertEqual(
                    actual.dtype, expected.dtype, f"Dtype mismatch at {path}"
                )
                self.assertTrue(
                    torch.equal(actual, expected), f"Values mismatch at {path}"
                )
            else:
                self.assertEqual(actual, expected, f"Value mismatch at {path}")

        compare_tensors(replicated_dict, expected_dict)

    def _create_dtensor_state_dict(self, rank: int, device_mesh: DeviceMesh) -> dict:
        """
        Create state_dict with DTensor and regular tensors for deterministic testing
        due to DTensor Shard, Replicate placements.

        Args:
            rank: Current rank
            device_mesh: DeviceMesh for DTensor creation

        Returns:
            dict: State dictionary with DTensors
        """
        # Create a large global tensor with deterministic values
        # Each position contains a unique value that encodes both position and rank info
        global_size = 128
        global_tensor = torch.arange(0, global_size * 16, dtype=torch.float32).reshape(
            global_size, 16
        )

        # Create DTensor with Shard(0) - each rank gets different rows
        sharded_dtensor = distribute_tensor(global_tensor, device_mesh, [Shard(0)])

        # Create DTensor with Replicate() - all ranks have the same data
        replicated_global = torch.full(
            (8, 8), float(global_size * 100), dtype=torch.float32, device="cpu"
        )
        replicated_dtensor = distribute_tensor(
            replicated_global, device_mesh, [Replicate()]
        )

        return {
            "sharded_param": sharded_dtensor,
            "replicated_param": replicated_dtensor,
            "rank_scalar": torch.tensor(float(rank), device="cpu"),
        }

    def _verify_dtensor_replication(
        self, replicated_dict: dict, rank: int, partner_rank: int
    ):
        """
        Verify DTensor replication accuracy by checking local shards and global reconstruction.

        Args:
            replicated_dict: Replicated state_dict received from partner
            rank: Current rank
            partner_rank: Partner rank we should have received from
        """
        # Verify sharded DTensor
        if "sharded_param" in replicated_dict:
            replicated_sharded = replicated_dict["sharded_param"]
            self.assertIsInstance(replicated_sharded, DTensor, "Should receive DTensor")

            # Get local shard from replicated DTensor
            replicated_local = replicated_sharded.to_local()

            # Create expected local shard (what partner rank would have)
            expected_global = torch.arange(0, 128 * 16, dtype=torch.float32).reshape(
                128, 16
            )

            # Calculate expected shard for this rank's position
            world_size = dist.get_world_size()
            shard_size = 128 // world_size
            start_idx = partner_rank * shard_size
            end_idx = (partner_rank + 1) * shard_size
            expected_local = expected_global[start_idx:end_idx]

            self.assertTrue(
                torch.equal(replicated_local, expected_local),
                "Sharded DTensor value mismatch",
            )

            # Verify DTensor metadata is preserved
            self.assertEqual(
                replicated_sharded._spec.placements[0].__class__.__name__,
                "Shard",
                "DTensor should maintain Shard placement",
            )

        # Verify replicated DTensor
        if "replicated_param" in replicated_dict:
            replicated_replicated = replicated_dict["replicated_param"]
            self.assertIsInstance(
                replicated_replicated, DTensor, "Should receive DTensor"
            )

            # Get local data from replicated DTensor
            replicated_local = replicated_replicated.to_local()

            # Expected value should be global_size * 100
            expected_value = float(128 * 100)
            expected_tensor = torch.full(
                (8, 8), expected_value, dtype=torch.float32, device="cpu"
            )

            self.assertTrue(
                torch.equal(replicated_local, expected_tensor),
                "Replicated DTensor value mismatch",
            )

            # Verify DTensor metadata is preserved
            self.assertEqual(
                replicated_replicated._spec.placements[0].__class__.__name__,
                "Replicate",
                "DTensor should maintain Replicate placement",
            )

        # Verify regular tensors
        if "rank_scalar" in replicated_dict:
            self.assertEqual(
                replicated_dict["rank_scalar"].item(),
                float(partner_rank),
                f"Rank scalar should be {partner_rank}, got {replicated_dict['rank_scalar'].item()}",
            )

    def _create_sharded_tensor_state_dict(self, rank: int, world_size: int) -> dict:
        """
        Create state_dict with ShardedTensor for deterministic testing.

        Args:
            rank: Current rank
            world_size: Total world size

        Returns:
            dict: State dictionary with ShardedTensor
        """
        # Create deterministic local shard for this rank
        global_size = 64
        shard_size = global_size // world_size
        start_idx = rank * shard_size
        end_idx = (rank + 1) * shard_size

        # Create local tensor with deterministic values
        local_tensor = torch.arange(
            start_idx * 8, end_idx * 8, dtype=torch.float32, device="cpu"
        ).reshape(shard_size, 8)

        # Create ShardedTensor using init_from_local_shards
        sharded_tensor = init_from_local_shards(
            [
                ShardedTensorShard(
                    tensor=local_tensor,
                    metadata=ShardMetadata(
                        shard_offsets=[start_idx, 0],
                        shard_sizes=[shard_size, 8],
                        placement=f"rank:{rank}/cpu",
                    ),
                )
            ],
            global_size,
            8,
        )

        return {
            "sharded_tensor": sharded_tensor,
            "rank_scalar": torch.tensor(float(rank), device="cpu"),
        }

    def _verify_sharded_tensor_replication(
        self, replicated_dict: dict, rank: int, partner_rank: int
    ):
        """
        Verify ShardedTensor replication accuracy by checking local shards and metadata.

        Args:
            replicated_dict: Replicated state_dict received from partner
            rank: Current rank
            partner_rank: Partner rank we should have received from
        """
        # Verify sharded tensor
        if "sharded_tensor" in replicated_dict:
            replicated_sharded = replicated_dict["sharded_tensor"]
            self.assertIsInstance(
                replicated_sharded, ShardedTensor, "Should receive ShardedTensor"
            )

            # Get local shard from replicated ShardedTensor
            local_shards = replicated_sharded.local_shards()
            self.assertEqual(
                len(local_shards), 1, "Should have exactly one local shard"
            )

            local_shard = local_shards[0]
            replicated_local = local_shard.tensor

            # Create expected local shard (what partner rank would have)
            world_size = dist.get_world_size()
            global_size = 64
            shard_size = global_size // world_size
            start_idx = partner_rank * shard_size
            end_idx = (partner_rank + 1) * shard_size

            expected_local = torch.arange(
                start_idx * 8, end_idx * 8, dtype=torch.float32, device="cpu"
            ).reshape(shard_size, 8)

            self.assertTrue(
                torch.equal(replicated_local, expected_local),
                "Sharded tensor value mismatch",
            )

            # Verify shard metadata is preserved
            expected_metadata = ShardMetadata(
                shard_offsets=[start_idx, 0],
                shard_sizes=[shard_size, 8],
                placement=f"rank:{partner_rank}/cpu",
            )
            self.assertEqual(
                local_shard.metadata.shard_offsets,
                expected_metadata.shard_offsets,
                "Shard offsets should match",
            )
            self.assertEqual(
                local_shard.metadata.shard_sizes,
                expected_metadata.shard_sizes,
                "Shard sizes should match",
            )

        # Verify regular tensors
        if "rank_scalar" in replicated_dict:
            self.assertEqual(
                replicated_dict["rank_scalar"].item(),
                float(partner_rank),
                f"Rank scalar should be {partner_rank}, got {replicated_dict['rank_scalar'].item()}",
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_replication_basic(self):
        """Test basic replication functionality with world_size=16"""
        world_size = dist.get_world_size()

        current_rank = dist.get_rank()

        # Create unique DTensor state_dict for this rank
        state_dict = self._create_simple_state_dict(current_rank)

        # Initialize replication stager
        stager = _ReplicationStager(
            pg=dist.new_group(backend=dist.Backend.GLOO),
            timeout=timedelta(seconds=30),
            device=torch.device("cpu"),
        )

        # Perform replication
        replicated_dict = stager.stage(state_dict)

        # Calculate expected partner rank
        partner_rank = (current_rank + world_size // 2) % world_size

        # Verify DTensor replication
        self._verify_simple_state_dict_replication(
            replicated_dict, current_rank, partner_rank
        )

        # Clean up
        stager.close()

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_replication_dtensors(self):
        """Test replication with DTensor and mixed tensor types"""
        world_size = dist.get_world_size()

        current_rank = dist.get_rank()

        # Create CPU-based DeviceMesh for DTensor
        device_mesh = DeviceMesh("cpu", list(range(world_size)))

        # Create DTensor state_dict which includes different tensor types
        state_dict = self._create_dtensor_state_dict(current_rank, device_mesh)

        # Initialize replication stager
        stager = _ReplicationStager(
            pg=dist.group.WORLD,
            timeout=timedelta(seconds=30),
            device=torch.device("cpu"),
        )

        # Perform replication
        result = stager.stage(state_dict)

        # Wait for completion
        from concurrent.futures import Future

        if isinstance(result, Future):
            replicated_dict = result.result()
        else:
            replicated_dict = result

        # Calculate expected partner
        partner_rank = (current_rank + world_size // 2) % world_size

        # Verify all DTensor types are correctly replicated
        self._verify_dtensor_replication(replicated_dict, current_rank, partner_rank)

        # Clean up
        stager.close()

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_replication_sharded_tensors(self):
        """Test replication with ShardedTensor and mixed tensor types"""
        world_size = dist.get_world_size()

        current_rank = dist.get_rank()

        # Create ShardedTensor state_dict for this rank
        state_dict = self._create_sharded_tensor_state_dict(current_rank, world_size)

        # Initialize replication stager
        stager = _ReplicationStager(
            pg=dist.group.WORLD,
            timeout=timedelta(seconds=30),
            device=torch.device("cpu"),
        )

        # Perform replication
        result = stager.stage(state_dict)

        # Wait for completion
        from concurrent.futures import Future

        if isinstance(result, Future):
            replicated_dict = result.result()
        else:
            replicated_dict = result

        # Calculate expected partner
        partner_rank = (current_rank + world_size // 2) % world_size

        # Verify all ShardedTensor types are correctly replicated
        self._verify_sharded_tensor_replication(
            replicated_dict, current_rank, partner_rank
        )

        # Clean up
        stager.close()

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_replication_persistence(self):
        """Test persistence functionality in _ReplicationStager"""
        world_size = dist.get_world_size()

        current_rank = dist.get_rank()

        # Test 1: Default storage directory (auto-generated tempdir)
        with tempfile.TemporaryDirectory() as _:
            # Create state_dict for this rank
            state_dict = self._create_simple_state_dict(current_rank)

            # Initialize stager with default storage_dir (None)
            stager = _ReplicationStager(
                pg=dist.group.WORLD,
                timeout=timedelta(seconds=30),
                device=torch.device("cpu"),
                storage_dir=None,  # Let it create its own tempdir
            )

            # Perform replication to trigger persistence
            stager.stage(state_dict)

            # Calculate expected partner rank
            partner_rank = (current_rank + world_size // 2) % world_size

            # Verify file was created with correct naming convention
            expected_path = stager._get_persisted_path(current_rank, partner_rank)

            self.assertTrue(
                os.path.exists(expected_path),
                f"Persisted file should exist at {expected_path}",
            )

            # Verify the storage directory was created
            self.assertTrue(
                os.path.isdir(stager._storage_dir), "Storage directory should exist"
            )
            self.assertTrue(
                stager._storage_dir.startswith(tempfile.gettempdir()),
                "Default storage directory should be in system temp directory",
            )

            # Load and verify the persisted state_dict matches the received one
            loaded_state_dict = torch.load(expected_path)
            self._verify_simple_state_dict_replication(
                loaded_state_dict, current_rank, partner_rank
            )

            # Clean up
            stager.close()

        # Test 2: Custom storage directory
        with tempfile.TemporaryDirectory() as custom_storage_dir:
            # Create custom subdirectory
            custom_subdir = os.path.join(custom_storage_dir, "custom_replication_test")

            # Create state_dict for this rank
            state_dict = self._create_simple_state_dict(current_rank)

            # Initialize stager with custom storage_dir
            stager = _ReplicationStager(
                pg=dist.group.WORLD,
                timeout=timedelta(seconds=30),
                device=torch.device("cpu"),
                storage_dir=custom_subdir,
            )

            # Perform replication to trigger persistence
            stager.stage(state_dict)

            # Verify custom storage directory was created and used
            self.assertEqual(
                stager._storage_dir,
                custom_subdir,
                "Should use custom storage directory",
            )
            self.assertTrue(
                os.path.isdir(custom_subdir),
                "Custom storage directory should be created",
            )

            # Verify file was created in custom directory
            expected_path = stager._get_persisted_path(current_rank, partner_rank)

            self.assertTrue(
                os.path.exists(expected_path),
                f"Persisted file should exist in custom directory at {expected_path}",
            )

            # Load and verify the persisted state_dict
            loaded_state_dict = torch.load(expected_path)
            self._verify_simple_state_dict_replication(
                loaded_state_dict, current_rank, partner_rank
            )

            # Clean up
            stager.close()


if __name__ == "__main__":
    run_tests()
