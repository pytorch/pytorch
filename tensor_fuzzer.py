import torch
import random
from typing import List, Tuple, Optional, Union, NamedTuple


# Global configuration for tensor fuzzing
class FuzzerConfig:
    """Global configuration for tensor fuzzing behavior."""
    use_real_values: bool = False  # If False, use zeros; if True, use random values


class TensorSpec(NamedTuple):
    """Specification for a tensor argument."""
    size: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype


class ScalarSpec(NamedTuple):
    """Specification for a scalar argument."""
    dtype: torch.dtype


# Union type for specs
Spec = Union[TensorSpec, ScalarSpec]
from torch._prims_common import is_non_overlapping_and_dense


def fuzz_torch_tensor_type() -> torch.dtype:
    """
    Fuzzes PyTorch tensor data types by randomly selecting and returning different dtypes.
    
    Returns:
        torch.dtype: A randomly selected PyTorch tensor data type
    """
    
    # Available PyTorch tensor data types (excluding unsigned types to avoid compatibility issues)
    tensor_dtypes: List[torch.dtype] = [
        torch.float32,
        torch.float64, 
        torch.float16,
        torch.bfloat16,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.complex64,
        torch.complex128,
    ]
    
    # Randomly select and return a data type
    return random.choice(tensor_dtypes)


def fuzz_tensor_size(max_dims: int = 6, max_size_per_dim: int = 30) -> Tuple[int, ...]:
    """
    Fuzzes PyTorch tensor sizes by generating random tensor shapes.

    Args:
        max_dims: Maximum number of dimensions (default: 6)
        max_size_per_dim: Maximum size for each dimension (default: 100)
    
    Returns:
        Tuple[int, ...]: A tuple representing tensor shape/size
    """
    
    # Randomly choose number of dimensions (0 to max_dims)
    # 0 dimensions = scalar tensor
    num_dims: int = random.randint(0, max_dims)
    
    if num_dims == 0:
        # Scalar tensor (0-dimensional)
        return ()
    
    # Generate random sizes for each dimension
    sizes: List[int] = []
    for _ in range(num_dims):
        # Include edge cases:
        # - 5% chance of size 0 (empty tensor in that dimension)
        # - 10% chance of size 1 (singleton dimension)
        # - 80% chance of normal size (2 to max_size_per_dim)
        
        rand_val: float = random.random()
        if rand_val < 0.05:
            # Empty dimension
            size: int = 0
        elif rand_val < 0.2:
            # Singleton dimension
            size = 1
        else:
            # Normal size
            size = random.randint(2, max_size_per_dim)
        
        sizes.append(size)
    
    return tuple(sizes)


def fuzz_valid_stride(size: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Fuzzes PyTorch tensor strides by generating valid stride patterns for a given size.
    
    Args:
        size: Tensor shape/size as a tuple of integers
    
    Returns:
        Tuple[int, ...]: A tuple representing valid tensor strides
    """
    
    if len(size) == 0:
        # Scalar tensor has no strides
        return ()
    
    # Choose stride pattern type
    stride_types = [
        "contiguous",                        # Normal contiguous memory layout
        "transposed",                        # Transposed dimensions  
        "custom_gaps",                       # Custom strides with gaps (non-dense)
        "minimal",                           # Minimal valid strides (all ones)
        "nonoverlapping_and_dense",          # Non-overlapping and dense (contiguous)
        "nonoverlapping_and_dense_non_contig", # Non-overlapping and dense but not contiguous
        "overlapping",                       # Overlapping memory access (zero strides)
        "sparse_gaps"                        # Large gaps (definitely non-dense)
    ]
    
    stride_type: str = random.choice(stride_types)
    
    if stride_type in ["contiguous", "nonoverlapping_and_dense"]:
        # Standard contiguous strides: stride[i] = product of sizes[i+1:]
        return tuple(_compute_contiguous_strides(size))
    
    elif stride_type == "transposed":
        # Create transposed version - swap some dimensions' strides
        base_strides = list(_compute_contiguous_strides(size))
        
        if len(base_strides) >= 2:
            # Randomly swap strides of two dimensions
            i, j = random.sample(range(len(base_strides)), 2)
            base_strides[i], base_strides[j] = base_strides[j], base_strides[i]
        
        return tuple(base_strides)
    
    elif stride_type == "custom_gaps":
        # Create strides with custom gaps/spacing
        base_strides = list(_compute_contiguous_strides(size))
        
        # Add random gaps to some strides
        for i in range(len(base_strides)):
            if size[i] != 0 and random.random() < 0.3:  # 30% chance to add gap
                gap_multiplier: int = random.randint(2, 5)
                base_strides[i] *= gap_multiplier
        
        return tuple(base_strides)
    
    elif stride_type == "minimal":
        # Minimal valid strides (all ones)
        return tuple([1] * len(size))
    
    elif stride_type == "nonoverlapping_and_dense_non_contig":
        # Non-overlapping and dense but not contiguous (e.g., column-major)
        return tuple(_compute_non_contiguous_dense_strides(size))
    
    elif stride_type == "overlapping":
        # Create overlapping strides (zero strides for some dimensions)
        base_strides = list(_compute_contiguous_strides(size))
        
        # Randomly set some strides to 0 to cause overlapping
        for i in range(len(base_strides)):
            if size[i] > 1 and random.random() < 0.4:  # 40% chance to make overlapping
                base_strides[i] = 0
        
        return tuple(base_strides)
    
    elif stride_type == "sparse_gaps":
        # Create strides with very large gaps (definitely non-dense)
        base_strides = list(_compute_contiguous_strides(size))
        
        # Add very large gaps to create sparse layout
        for i in range(len(base_strides)):
            if size[i] > 1:
                gap_multiplier: int = random.randint(10, 100)  # Much larger gaps
                base_strides[i] *= gap_multiplier
        
        return tuple(base_strides)
    
    # Fallback to contiguous
    return tuple(_compute_contiguous_strides(size))


def _compute_contiguous_strides(size: Tuple[int, ...]) -> List[int]:
    """
    Helper function to compute standard contiguous strides for a given size.
    
    Args:
        size: Tensor shape/size as a tuple of integers
    
    Returns:
        List[int]: List of contiguous strides
    """
    strides: List[int] = []
    current_stride: int = 1
    
    # Calculate strides from right to left
    for i in range(len(size) - 1, -1, -1):
        strides.insert(0, current_stride)
        # For dimensions with size 0, keep stride as is
        if size[i] != 0:
            current_stride *= size[i]
    
    return strides


def _compute_non_contiguous_dense_strides(size: Tuple[int, ...]) -> List[int]:
    """
    Helper function to compute non-contiguous but dense strides (e.g., column-major order).
    
    Args:
        size: Tensor shape/size as a tuple of integers
    
    Returns:
        List[int]: List of non-contiguous dense strides
    """
    if len(size) <= 1:
        # For 0D or 1D tensors, return same as contiguous
        return _compute_contiguous_strides(size)
    
    # Generate different dense patterns
    patterns = [
        "column_major",     # Reverse order (left to right instead of right to left)
        "random_permute",   # Random permutation of dimensions
        "middle_out",       # Start from middle dimension
    ]
    
    pattern: str = random.choice(patterns)
    
    if pattern == "column_major":
        # Column-major order: calculate strides from left to right
        strides: List[int] = [0] * len(size)
        current_stride: int = 1
        
        # Calculate strides from left to right (opposite of contiguous)
        for i in range(len(size)):
            strides[i] = current_stride
            # For dimensions with size 0, keep stride as is
            if size[i] != 0:
                current_stride *= size[i]
        
        return strides
    
    elif pattern == "random_permute":
        # Create a valid permutation that's still dense
        # Create dimension permutation
        indices = list(range(len(size)))
        random.shuffle(indices)
        
        # Apply permutation to get new dense layout
        new_strides = [0] * len(size)
        current_stride = 1
        
        # Sort indices by their corresponding size to maintain density
        sorted_indices = sorted(indices, key=lambda i: size[i] if size[i] != 0 else float('inf'))
        
        for idx in sorted_indices:
            new_strides[idx] = current_stride
            if size[idx] != 0:
                current_stride *= size[idx]
        
        return new_strides
    
    elif pattern == "middle_out":
        # Start from middle dimension and work outward
        strides = [0] * len(size)
        current_stride = 1
        
        # Start from middle
        middle = len(size) // 2
        processed = [False] * len(size)
        
        # Process middle first
        strides[middle] = current_stride
        if size[middle] != 0:
            current_stride *= size[middle]
        processed[middle] = True
        
        # Process alternating left and right
        for offset in range(1, len(size)):
            for direction in [-1, 1]:
                idx = middle + direction * offset
                if 0 <= idx < len(size) and not processed[idx]:
                    strides[idx] = current_stride
                    if size[idx] != 0:
                        current_stride *= size[idx]
                    processed[idx] = True
                    break
        
        return strides
    
    # Fallback to contiguous
    return _compute_contiguous_strides(size)


def _compute_storage_size_needed(size: Tuple[int, ...], strides: Tuple[int, ...]) -> int:
    """Compute minimum storage size needed for given shape and strides."""
    if not size:
        return 1
    
    # Find maximum offset
    max_offset = 0
    for dim_size, stride in zip(size, strides):
        if dim_size > 1:
            max_offset += (dim_size - 1) * abs(stride)
    
    return max_offset + 1


def fuzz_tensor(size: Optional[Tuple[int, ...]] = None, 
                stride: Optional[Tuple[int, ...]] = None, 
                dtype: Optional[torch.dtype] = None,
                seed: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """
    Create a tensor with fuzzed size, stride, and dtype.
    
    Args:
        size: Tensor shape. If None, will be randomly generated.
        stride: Tensor stride. If None, will be randomly generated based on size.
        dtype: Tensor data type. If None, will be randomly generated.
        seed: Random seed for reproducibility. If None, will be randomly generated.
    
    Returns:
        Tuple[torch.Tensor, int]: A tuple of (tensor, seed_used) where tensor has 
        the specified or randomly generated properties, and seed_used is the seed 
        that was used for generation (for reproducibility).
    """
    # Generate or use provided seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Generate random values if not provided
    if size is None:
        size = fuzz_tensor_size()
    
    if dtype is None:
        dtype = fuzz_torch_tensor_type()
    
    if stride is None:
        stride = fuzz_valid_stride(size)
    
    # Handle empty tensor case
    if len(size) == 0:
        return torch.zeros((), dtype=dtype), seed
    
    # Calculate required storage size for the custom stride
    required_storage = _compute_storage_size_needed(size, stride)
    
    # Create base tensor with sufficient storage
    if FuzzerConfig.use_real_values:
        # Use random values based on dtype
        if dtype.is_floating_point:
            base_tensor = torch.randn(required_storage, dtype=dtype)
        elif dtype in [torch.complex64, torch.complex128]:
            # Create complex tensor with random real and imaginary parts
            real_part = torch.randn(required_storage, dtype=torch.float32 if dtype == torch.complex64 else torch.float64)
            imag_part = torch.randn(required_storage, dtype=torch.float32 if dtype == torch.complex64 else torch.float64)
            base_tensor = torch.complex(real_part, imag_part).to(dtype)
        elif dtype == torch.bool:
            base_tensor = torch.randint(0, 2, (required_storage,), dtype=torch.bool)
        else:  # integer types
            base_tensor = torch.randint(-100, 100, (required_storage,), dtype=dtype)
    else:
        # Use zeros (default behavior)
        base_tensor = torch.zeros(required_storage, dtype=dtype)
    
    # Create strided tensor view
    strided_tensor = torch.as_strided(base_tensor, size, stride)
    
    return strided_tensor, seed


def fuzz_tensor_simple(size: Optional[Tuple[int, ...]] = None, 
                       stride: Optional[Tuple[int, ...]] = None, 
                       dtype: Optional[torch.dtype] = None,
                       seed: Optional[int] = None) -> torch.Tensor:
    """
    Convenience function that returns just the tensor without the seed.
    
    Args:
        size: Tensor shape. If None, will be randomly generated.
        stride: Tensor stride. If None, will be randomly generated based on size.
        dtype: Tensor data type. If None, will be randomly generated.
        seed: Random seed for reproducibility. If None, uses current random state.
    
    Returns:
        torch.Tensor: A tensor with the specified or randomly generated properties.
    """
    tensor, _ = fuzz_tensor(size, stride, dtype, seed)
    return tensor


def fuzz_non_contiguous_dense_tensor(
    size: Optional[Tuple[int, ...]] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Specifically generates tensors that are non-contiguous but dense and non-overlapping.
    
    Args:
        size: Tensor shape/size. If None, auto-generated.
        dtype: PyTorch tensor data type. If None, auto-generated.
        
    Returns:
        torch.Tensor: A non-contiguous but dense tensor
    """
    if dtype is None:
        dtype = fuzz_torch_tensor_type()
    
    if size is None:
        size = fuzz_tensor_size()
    
    # Force non-contiguous but dense stride patterns
    if len(size) <= 1:
        # For 0D or 1D tensors, return contiguous (they're trivially dense)
        tensor, _ = fuzz_tensor(size, None, dtype)
        return tensor
    
    # Choose from patterns that guarantee non-contiguous but dense
    patterns = [
        "column_major",
        "transposed", 
        "permuted_dense"
    ]
    
    pattern = random.choice(patterns)
    
    if pattern == "column_major":
        # Column-major order (non-contiguous but dense)
        stride = tuple(_compute_non_contiguous_dense_strides(size))
    elif pattern == "transposed":
        # Simple transpose of last two dimensions
        base_strides = _compute_contiguous_strides(size)
        if len(base_strides) >= 2:
            # Swap last two dimensions' strides
            base_strides[-1], base_strides[-2] = base_strides[-2], base_strides[-1]
        stride = tuple(base_strides)
    else:  # permuted_dense
        # Random permutation that maintains density
        stride = tuple(_compute_non_contiguous_dense_strides(size))
    
    tensor, _ = fuzz_tensor(size, stride, dtype)
    return tensor


def test_fuzzing_tensors() -> None:
    """
    Test function that demonstrates all fuzzing capabilities.
    Shows different ways to use the fuzz_tensor function and individual fuzzers.
    """
    
    # Track statistics
    total_tests = 20
    contiguous_count = 0
    non_contiguous_dense_count = 0
    non_contiguous_non_dense_count = 0
    overlapping_count = 0
    
    # Test individual fuzzers with statistics
    for i in range(total_tests):
        try:
            # Create tensor using fuzz_tensor
            tensor, seed = fuzz_tensor()
   
            is_contig = tensor.is_contiguous()
            is_dense = is_non_overlapping_and_dense(tensor)
            
            # Check for overlapping by looking for zero strides with size > 1
            has_overlapping = any(stride == 0 and size > 1 
                                for stride, size in zip(tensor.stride(), tensor.shape))

            if is_contig:
                contiguous_count += 1
            elif has_overlapping:
                overlapping_count += 1
            elif is_dense:
                non_contiguous_dense_count += 1         
            else:
                non_contiguous_non_dense_count += 1
            print("    ==============")
            print(f"    Shape: {tensor.shape}, Stride: {tensor.stride()}")
            print(f"    is_contiguous: {is_contig}, is_non_overlapping_and_dense: {is_dense}")
    
        except Exception as e:
            print(f"  Error in test {i+1}: {e}")
    
    print(f"\n=== Statistics from {total_tests} random tensors ===")
    print(f"Contiguous: {contiguous_count} ({contiguous_count/total_tests*100:.1f}%)")
    print(f"Non-contiguous but dense: {non_contiguous_dense_count} ({non_contiguous_dense_count/total_tests*100:.1f}%)")
    print(f"Overlapping: {overlapping_count} ({overlapping_count/total_tests*100:.1f}%)")
    print(f"Non-contiguous and non-dense: {non_contiguous_non_dense_count} ({non_contiguous_non_dense_count/total_tests*100:.1f}%)")


def test_fuzz_spec():
    """Test the fuzz_spec function to see random spec generation."""
    from ops_fuzzer import fuzz_spec, TensorSpec, ScalarSpec
    
    print("Testing fuzz_spec with random generation:")
    
    for i in range(5):
        spec = fuzz_spec()
        print(f"\n{i+1}. Generated spec: {spec}")
        if isinstance(spec, TensorSpec):
            print(f"   Type: TensorSpec, Size: {spec.size}, Stride: {spec.stride}, Dtype: {spec.dtype}")
        elif isinstance(spec, ScalarSpec):
            print(f"   Type: ScalarSpec, Dtype: {spec.dtype}")


def test_fuzz_op():
    """Test the fuzz_op function with different target layouts."""
    from ops_fuzzer import fuzz_op, TensorSpec, ScalarSpec
    
    print("Testing fuzz_op with different layouts:")
    
    # Test scalar tensor
    print("\n1. Scalar tensor:")
    scalar_spec = ScalarSpec(dtype=torch.float32)
    op_name, arg_specs = fuzz_op(scalar_spec)
    print(f"   Operation: {op_name}")
    print(f"   Argument specs: {arg_specs}")
    
    # Test 1D tensor
    print("\n2. 1D tensor:")
    tensor_spec_1d = TensorSpec(size=(5,), stride=(1,), dtype=torch.float32)
    op_name, arg_specs = fuzz_op(tensor_spec_1d)
    print(f"   Operation: {op_name}")
    print(f"   Argument specs: {arg_specs}")
    
    # Test 2D tensor
    print("\n3. 2D tensor:")
    tensor_spec_2d = TensorSpec(size=(3, 4), stride=(4, 1), dtype=torch.float32)
    op_name, arg_specs = fuzz_op(tensor_spec_2d)
    print(f"   Operation: {op_name}")
    print(f"   Argument specs: {arg_specs}")
    
    # Test with specific stride
    print("\n4. 2D tensor with specific stride:")
    tensor_spec_custom = TensorSpec(size=(2, 3), stride=(3, 1), dtype=torch.int32)
    op_name, arg_specs = fuzz_op(tensor_spec_custom)
    print(f"   Operation: {op_name}")
    print(f"   Argument specs: {arg_specs}")


def test_fuzz_spec_with_fuzz_op():
    """Test using fuzz_spec to generate random specs and then use them with fuzz_op."""
    from ops_fuzzer import fuzz_spec, fuzz_op
    
    print("\nTesting fuzz_spec combined with fuzz_op:")
    
    for i in range(3):
        # Generate random spec
        random_spec = fuzz_spec()
        print(f"\n{i+1}. Random spec: {random_spec}")
        
        # Use it with fuzz_op
        op_name, arg_specs = fuzz_op(random_spec)
        print(f"   Operation: {op_name}")
        print(f"   Required arguments: {len(arg_specs)} tensors/scalars")
        for j, arg_spec in enumerate(arg_specs):
            print(f"   Arg {j+1}: {arg_spec}")


def fuzz_scalar(spec, seed: Optional[int] = None) -> Union[float, int, bool, complex]:
    """
    Create a Python scalar value from a ScalarSpec.
    
    Args:
        spec: ScalarSpec containing the desired dtype
        seed: Random seed for reproducibility. If None, uses current random state.
        
    Returns:
        Python scalar (float, int, bool, complex) matching the dtype
    """
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
    
    # Create a scalar value based on dtype
    if spec.dtype.is_floating_point:
        return random.uniform(-10.0, 10.0)
    elif spec.dtype in [torch.complex64, torch.complex128]:
        return complex(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0))
    else:  # integer or bool
        if spec.dtype == torch.bool:
            return random.choice([True, False])
        else:
            return random.randint(-10, 10)
