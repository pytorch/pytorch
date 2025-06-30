# Owner(s): ["oncall: distributed"]

import dataclasses

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import Shard
from torch.distributed.checkpoint._state_dict_stager import StateDictStager
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import requires_cuda, run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


def create_cpu_state_dict(state_dict):
    cpu_state_dict = {}
    for key, value in state_dict.items():
        cpu_state_dict[key] = value.cpu()
    return cpu_state_dict


def compare_state_dicts(cuda_state_dict, cpu_state_dict, rtol=1e-5, atol=1e-8):
    """
    Compare if two state dictionaries (one on CUDA, one on CPU) are otherwise the same.

    This function checks if the tensors in both state dictionaries have the same values,
    shapes, dtypes, etc., ignoring the device difference. It also checks if tensors that
    share storage in one state dict also share storage in the other.

    Args:
        cuda_state_dict: The state dictionary with tensors on CUDA
        cpu_state_dict: The state dictionary with tensors on CPU
        rtol: Relative tolerance for comparing tensor values
        atol: Absolute tolerance for comparing tensor values

    Returns:
        bool: True if the state dictionaries are equivalent, False otherwise
        str: Error message if the state dictionaries are not equivalent, empty string otherwise
    """
    # Track storage data pointers to check storage sharing
    cuda_storage_ptrs = {}
    cpu_storage_ptrs = {}

    def compare_objects(cuda_obj, cpu_obj, path=""):
        # If objects are tensors, compare them
        if isinstance(cuda_obj, torch.Tensor) and isinstance(cpu_obj, torch.Tensor):
            # Check if devices are as expected
            if cuda_obj.device.type != "cuda":
                return (
                    False,
                    f"Expected CUDA tensor, got {cuda_obj.device.type} tensor at {path}",
                )
            if cpu_obj.device.type != "cpu":
                return (
                    False,
                    f"Expected CPU tensor, got {cpu_obj.device.type} tensor at {path}",
                )
            if cuda_obj.storage_offset() != cpu_obj.storage_offset():
                return (
                    False,
                    f"Storage offset mismatch at {path}: {cuda_obj.storage_offset()} vs {cpu_obj.storage_offset()}",
                )

            if not torch.equal(cuda_obj.cpu(), cpu_obj):
                return (
                    False,
                    f"Tensors are not same at {path}",
                )

            # Track storage sharing
            cuda_storage_ptr = cuda_obj.storage().data_ptr()
            cpu_storage_ptr = cpu_obj.storage().data_ptr()

            if cuda_storage_ptr in cuda_storage_ptrs:
                # This CUDA tensor shares storage with another tensor
                # Check if the corresponding CPU tensors also share storage
                if cpu_storage_ptr != cuda_storage_ptrs[cuda_storage_ptr]:
                    return (
                        False,
                        f"Storage sharing mismatch: CUDA tensors share storage but CPU tensors don't at {path}",
                    )
            else:
                # First time seeing this storage
                cuda_storage_ptrs[cuda_storage_ptr] = cpu_storage_ptr
                cpu_storage_ptrs[cpu_storage_ptr] = cuda_storage_ptr

            return True, ""

        # If objects are dictionaries, compare them recursively
        elif isinstance(cuda_obj, dict) and isinstance(cpu_obj, dict):
            if cuda_obj.keys() != cpu_obj.keys():
                return (
                    False,
                    f"Dictionary keys mismatch at {path}: {cuda_obj.keys()} vs {cpu_obj.keys()}",
                )

            for key in cuda_obj:
                result, error = compare_objects(
                    cuda_obj[key], cpu_obj[key], f"{path}.{key}" if path else key
                )
                if not result:
                    return False, error

            return True, ""

        # If objects are lists, tuples, or sets, compare them recursively
        elif isinstance(cuda_obj, (list, tuple, set)) and isinstance(
            cpu_obj, (list, tuple, set)
        ):
            if len(cuda_obj) != len(cpu_obj):
                return (
                    False,
                    f"Collection length mismatch at {path}: {len(cuda_obj)} vs {len(cpu_obj)}",
                )
            if type(cuda_obj) != type(cpu_obj):
                return (
                    False,
                    f"Collection type mismatch at {path}: {type(cuda_obj)} vs {type(cpu_obj)}",
                )

            for i, (cuda_item, cpu_item) in enumerate(zip(cuda_obj, cpu_obj)):
                result, error = compare_objects(cuda_item, cpu_item, f"{path}[{i}]")
                if not result:
                    return False, error

            return True, ""

        # If objects are custom classes, compare their attributes
        elif hasattr(cuda_obj, "__dict__") and hasattr(cpu_obj, "__dict__"):
            if type(cuda_obj) != type(cpu_obj):
                return (
                    False,
                    f"Object type mismatch at {path}: {type(cuda_obj)} vs {type(cpu_obj)}",
                )

            result, error = compare_objects(
                cuda_obj.__dict__, cpu_obj.__dict__, f"{path}.__dict__"
            )
            if not result:
                return False, error

            return True, ""

        # For other types, use direct equality comparison
        else:
            if type(cuda_obj) != type(cpu_obj):
                return (
                    False,
                    f"Type mismatch at {path}: {type(cuda_obj)} vs {type(cpu_obj)}",
                )
            if cuda_obj != cpu_obj:
                return False, f"Value mismatch at {path}: {cuda_obj} vs {cpu_obj}"

            return True, ""

    # Start the recursive comparison
    result, error = compare_objects(cuda_state_dict, cpu_state_dict)
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
    @requires_cuda
    def test_views(self):
        test_configs = [
            (False, False),  # pin_memory=False, share_memory=False,
            (True, False),  # pin_memory=True, share_memory=False
            (False, True),  # pin_memory=False, share_memory=True
            (True, True),  # pin_memory=True, share_memory=True
        ]
        for pin_memory, share_memory in test_configs:
            with self.subTest(pin_memory=pin_memory, share_memory=share_memory):
                tensor1 = torch.randn(4, 4).cuda()
                tensor2 = tensor1.view(16)
                tensor3 = torch.randn(4, 4).cuda()
                state_dict = {
                    "tensor1": tensor1,
                    "tensor2": tensor2,
                    "recursive": {
                        "tensor3": tensor3,
                        "type": TestStruct(tensor1=tensor3.narrow(0, 0, 2)),
                    },
                }
                assert (
                    state_dict["tensor1"].storage().data_ptr()
                    == state_dict["tensor2"].storage().data_ptr()
                )

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
                assert num_storages == expected_storage_cnt, (
                    f"Expected {expected_storage_cnt} storages, got {num_storages}"
                )

                # Calculate expected bytes
                # Note: Only unique storages are counted in the byte count
                expected_bytes = (
                    tensor1.numel() * tensor1.element_size()
                    + tensor3.numel()  # tensor1 and tensor2 share storage
                    * tensor3.element_size()  # tensor3 and its narrow view share storage
                )
                assert num_bytes == expected_bytes, (
                    f"Expected {expected_bytes} bytes, got {num_bytes}"
                )
                # Verify that the CPU state dict is equivalent to the original CUDA state dict
                result, error = compare_state_dicts(state_dict, cpu_state_dict)
                assert result, f"State dicts are not equivalent: {error}"

                # Additional checks for storage sharing
                assert cpu_state_dict["tensor1"].device == torch.device("cpu")
                assert cpu_state_dict["tensor2"].device == torch.device("cpu")
                assert (
                    cpu_state_dict["tensor1"].storage().data_ptr()
                    == cpu_state_dict["tensor2"].storage().data_ptr()
                )

                recursive = cpu_state_dict["recursive"]
                assert recursive["tensor3"].device == torch.device("cpu")
                assert recursive["type"].tensor1.device == torch.device("cpu")
                assert (
                    recursive["tensor3"].storage().data_ptr()
                    == recursive["type"].tensor1.storage().data_ptr()
                )

    @requires_cuda
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
                tensor1 = torch.randn(4, 4).cuda()
                tensor2 = tensor1.view(16)
                tensor3 = torch.randn(4, 4).cuda()
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
                assert result, (
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
                assert result, (
                    f"Second state dict is not equivalent to modified original: {error}"
                )

                # Verify that the number of cached storages hasn't changed
                assert num_storages1 == num_storages2, (
                    f"Storage count changed: {num_storages1} vs {num_storages2}"
                )

                # Verify that the tensors in the second state dict have the same storage pointers as the first
                assert (
                    cpu_state_dict1["tensor1"].storage().data_ptr()
                    == cpu_state_dict2["tensor1"].storage().data_ptr()
                ), "Storage pointers should match for tensor1"
                assert (
                    cpu_state_dict1["tensor2"].storage().data_ptr()
                    == cpu_state_dict2["tensor2"].storage().data_ptr()
                ), "Storage pointers should match for tensor2"
                assert (
                    cpu_state_dict1["recursive"]["tensor3"].storage().data_ptr()
                    == cpu_state_dict2["recursive"]["tensor3"].storage().data_ptr()
                ), "Storage pointers should match for tensor3"

                # Modify the original tensors again with different values
                tensor1.fill_(42.0)

                # Third call to stage with staging context
                cpu_state_dict3 = stager.stage(state_dict)

                # Verify that the third CPU state dict reflects the updated values
                assert torch.all(cpu_state_dict3["tensor1"] == 42.0), (
                    "Updated values should be reflected in the cached state dict"
                )
                assert torch.all(cpu_state_dict3["tensor2"] == 42.0), (
                    "Updated values should be reflected in the cached state dict"
                )

    @requires_cuda
    def test_tensor_attrs(self):
        """
        Test that tensor attributes are preserved during stage with StateDictStager.
        """
        tensor1 = torch.randn(4, 4).cuda()
        tensor2 = tensor1.view(16)
        tensor3 = torch.randn(4, 4).cuda()

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
        assert hasattr(cpu_state_dict["tensor1"], "a"), (
            "Tensor attribute 'a' was not preserved"
        )
        assert cpu_state_dict["tensor1"].a == 42, (
            "Tensor attribute 'a' has incorrect value"
        )
        assert hasattr(cpu_state_dict["tensor1"], "b"), (
            "Tensor attribute 'b' was not preserved"
        )
        assert cpu_state_dict["tensor1"].b == 43, (
            "Tensor attribute 'b' has incorrect value"
        )
        assert hasattr(cpu_state_dict["recursive"]["tensor3"], "c"), (
            "Tensor attribute 'c' was not preserved"
        )
        assert cpu_state_dict["recursive"]["tensor3"].c == 44, (
            "Tensor attribute 'c' has incorrect value"
        )

    @requires_cuda
    def test_different_dtypes(self):
        """
        Test that StateDictStager works correctly with tensors of different data types.
        """
        # Create tensors with different dtypes
        tensors = {
            "float32": torch.randn(4, 4, dtype=torch.float32).cuda(),
            "float64": torch.randn(4, 4, dtype=torch.float64).cuda(),
            "int32": torch.randint(-100, 100, (4, 4), dtype=torch.int32).cuda(),
            "int64": torch.randint(-100, 100, (4, 4), dtype=torch.int64).cuda(),
            "bool": torch.randint(0, 2, (4, 4), dtype=torch.bool).cuda(),
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

    @requires_cuda
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
                    "empty_0d": torch.tensor([], dtype=torch.float32).cuda(),
                    "empty_1d": torch.tensor([], dtype=torch.float32).reshape(0).cuda(),
                    "empty_2d": torch.tensor([], dtype=torch.float32)
                    .reshape(0, 0)
                    .cuda(),
                    "empty_3d": torch.tensor([], dtype=torch.float32)
                    .reshape(0, 0, 0)
                    .cuda(),
                    "zero_dim": torch.tensor(0.0).cuda(),  # scalar tensor
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

    @requires_cuda
    def test_complex_storage_sharing(self):
        """
        Test that StateDictStager correctly handles complex storage sharing scenarios.
        """
        # Create a base tensor
        base_tensor = torch.randn(10, 10).cuda()

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

    @requires_cuda
    def test_dataclasses(self):
        # Create tensors
        tensor1 = torch.randn(4, 4).cuda()
        tensor2 = torch.randn(8, 8).cuda()
        tensor3 = torch.randn(2, 6).cuda()
        tensor4 = torch.randn(3, 5).cuda()

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

    @requires_cuda
    def test_tensor_pinned_and_shared(self):
        """
        Test that verifies tensors are actually pinned and shared using tensor.is_pinned() and tensor.is_shared() methods.
        """
        # Create test tensors
        tensor1 = torch.randn(4, 4).cuda()
        tensor2 = torch.randn(8, 8).cuda()

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
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_dtensor(self):
        """
        Test that StateDictStager works correctly with DTensors.
        """
        # Create a DTensor
        device_mesh = dist.DeviceMesh("cuda", list(range(dist.get_world_size())))
        tensor = torch.randn(3, 3, device="cuda")
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


if __name__ == "__main__":
    run_tests()
