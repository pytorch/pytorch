import torch
import random
from typing import Tuple, List
from tensor_fuzzer import fuzz_torch_tensor_type, fuzz_tensor_size, fuzz_valid_stride, TensorSpec, ScalarSpec, Spec


def fuzz_spec() -> Spec:
    """
    Generate a random Spec (either TensorSpec or ScalarSpec) using tensor fuzzing functions.
    
    Utilizes:
    - fuzz_torch_tensor_type() for random dtype
    - fuzz_tensor_size() for random tensor size  
    - fuzz_valid_stride() for random valid strides
    
    Returns:
        Spec: Either a TensorSpec (80% probability) or ScalarSpec (20% probability) with random properties
    """
    # Get random dtype
    dtype = fuzz_torch_tensor_type()
    
    # 20% probability of returning ScalarSpec
    if random.random() < 0.2:
        return ScalarSpec(dtype=dtype)
    
    # 80% probability of returning TensorSpec
    # Get random size and corresponding stride
    size = fuzz_tensor_size()
    stride = fuzz_valid_stride(size)
    return TensorSpec(size=size, stride=stride, dtype=dtype)


def fuzz_op(target_spec: Spec, depth, stack_size) -> Tuple[str, List[Spec]]:
    """
    Given an output specification, returns an operation that can 
    produce a tensor with that layout.
    
    Supports:
    - For scalars: scalar_add, scalar_multiply, item, constant, arg
    - For tensors: aten.add, aten.mul, constant, arg
    
    Args:
        target_spec: Desired output specification (TensorSpec or ScalarSpec)
        depth: Maximum depth for operation generation. At depth 0, only leaf operations 
               (constant, arg) are allowed. Higher depths allow more complex operations.
        stack_size: Current stack size. When < 10, reduces probability of leaf operations.
    
    Returns:
        Tuple of (operation_name, list_of_argument_specs) where each argument spec
        describes the layout requirements for the operation's inputs
    """
    if isinstance(target_spec, ScalarSpec):
        if depth == 0:
            # At depth 0, only allow leaf operations
            ops = ["constant", "arg"]
            chosen_op = random.choice(ops)
        else:
            # At higher depths, allow all scalar operations
            non_leaf_ops = ["scalar_add", "scalar_multiply", "torch.ops.aten.item"]
            leaf_ops = ["constant", "arg"]
            
            # Reduce probability of leaf operations when stack_size < 10
            if stack_size < 10:
                # 80% chance of non-leaf, 20% chance of leaf
                if random.random() < 0.8:
                    chosen_op = random.choice(non_leaf_ops)
                else:
                    chosen_op = random.choice(leaf_ops)
            else:
                # Normal probability distribution
                all_ops = non_leaf_ops + leaf_ops
                chosen_op = random.choice(all_ops)
        
        if chosen_op == "scalar_add":
            return _get_scalar_add_args_specs(target_spec)
        elif chosen_op == "scalar_multiply":
            return _get_scalar_multiply_args_specs(target_spec)
        elif chosen_op == "torch.ops.aten.item":
            return _get_item_args_specs(target_spec)
        elif chosen_op == "constant":
            return _get_constant_args_specs(target_spec)
        else:  # arg
            return _get_arg_args_specs(target_spec)
    
    elif isinstance(target_spec, TensorSpec):
        if depth == 0:
            # At depth 0, only allow leaf operations
            ops = ["constant", "arg"]
            chosen_op = random.choice(ops)
        else:
            # At higher depths, allow all tensor operations
            non_leaf_ops = ["torch.ops.aten.add", "torch.ops.aten.mul"]
            leaf_ops = ["constant", "arg"]
            
            # Reduce probability of leaf operations when stack_size < 10
            if stack_size < 10:
                # 80% chance of non-leaf, 20% chance of leaf
                if random.random() < 0.8:
                    chosen_op = random.choice(non_leaf_ops)
                else:
                    chosen_op = random.choice(leaf_ops)
            else:
                # Normal probability distribution
                all_ops = non_leaf_ops + leaf_ops
                chosen_op = random.choice(all_ops)
        
        if chosen_op == "torch.ops.aten.add":
            return _get_aten_add_args_specs(target_spec)
        elif chosen_op == "torch.ops.aten.mul":
            return _get_aten_mul_args_specs(target_spec)
        elif chosen_op == "constant":
            return _get_constant_args_specs(target_spec)
        else:  # arg
            return _get_arg_args_specs(target_spec)


def _get_scalar_add_args_specs(target_spec: ScalarSpec) -> Tuple[str, List[Spec]]:
    """Get argument specifications for scalar_add operation using type promotion rules."""
    # Use PyTorch's implicit type promotion rules to generate diverse input types
    arg_specs = _get_promoted_scalar_args(target_spec.dtype)
    return "scalar_add", arg_specs


def _get_scalar_multiply_args_specs(target_spec: ScalarSpec) -> Tuple[str, List[Spec]]:
    """Get argument specifications for scalar_multiply operation using type promotion rules."""
    # Use PyTorch's implicit type promotion rules to generate diverse input types
    arg_specs = _get_promoted_scalar_args(target_spec.dtype)
    return "scalar_multiply", arg_specs


# Define promotion chains - types that can promote to the target
# PyTorch promotion hierarchy (simplified):
# - bool < int8 < int16 < int32 < int64 < float16 < float32 < float64 < complex64 < complex128
# - uint types have limited promotion support
_PROMOTION_CHAINS = {
    torch.bool: [torch.bool],
    torch.int8: [torch.bool, torch.int8],
    torch.int16: [torch.bool, torch.int8, torch.int16],
    torch.int32: [torch.bool, torch.int8, torch.int16, torch.int32],
    torch.int64: [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64],
    torch.float16: [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16],
    torch.float32: [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32],
    torch.float64: [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64],
    torch.complex64: [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.complex64],
    torch.complex128: [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128],
}


def _get_promoted_dtypes(target_dtype: torch.dtype) -> List[torch.dtype]:
    """
    Generate two dtypes that will promote to target_dtype via PyTorch's type promotion rules.
    """
    
    # Get compatible input types for the target dtype
    compatible_types = _PROMOTION_CHAINS.get(target_dtype, [target_dtype])
    
    # Strategy: Choose between same type or mixed promotion
    strategies = ["same_type", "mixed_promotion"]
    strategy = random.choice(strategies)
    
    if strategy == "same_type":
        # Both args same type as target
        return [target_dtype, target_dtype]
    
    else:  # mixed_promotion
        # Mixed types where the result will promote to target_dtype
        lower_types = compatible_types[:-1]  # All except the last (target_dtype)
        
        if lower_types:
            # One arg is target_dtype, one is lower (will promote to target)
            lower_dtype = random.choice(lower_types)
            if random.random() < 0.5:
                return [target_dtype, lower_dtype]
            else:
                return [lower_dtype, target_dtype]
        else:
            # Fallback to same type if no lower types available
            return [target_dtype, target_dtype]


def _get_promoted_scalar_args(target_dtype: torch.dtype) -> List[Spec]:
    """
    Generate two argument specs that will promote to target_dtype via PyTorch's type promotion rules.
    """
    arg_dtypes = _get_promoted_dtypes(target_dtype)
    
    # For ScalarSpec output, both inputs must be ScalarSpec
    # (mixing with 0-D TensorSpec would produce 0-D TensorSpec output)
    return [ScalarSpec(arg_dtypes[0]), ScalarSpec(arg_dtypes[1])]


def _get_item_args_specs(target_spec: ScalarSpec) -> Tuple[str, List[Spec]]:
    """Get argument specifications for torch.ops.aten.item operation."""
    # torch.ops.aten.item: tensor -> scalar (extract single element)
    # Create a tensor spec that can produce a scalar via .item()
    tensor_spec = TensorSpec(size=(1,), stride=(1,), dtype=target_spec.dtype)  # 1-D tensor with 1 element
    arg_specs: List[Spec] = [tensor_spec]
    return "torch.ops.aten.item", arg_specs


def _get_aten_add_args_specs(target_spec: TensorSpec) -> Tuple[str, List[Spec]]:
    """Get argument specifications for torch.ops.aten.add operation using type promotion rules."""
    # Use promotion rules to generate diverse tensor input types
    arg_dtypes = _get_promoted_dtypes(target_spec.dtype)
    
    arg_specs: List[Spec] = [
        TensorSpec(target_spec.size, target_spec.stride, arg_dtypes[0]),
        TensorSpec(target_spec.size, target_spec.stride, arg_dtypes[1])
    ]
    return "torch.ops.aten.add", arg_specs


def _get_aten_mul_args_specs(target_spec: TensorSpec) -> Tuple[str, List[Spec]]:
    """Get argument specifications for torch.ops.aten.mul operation using type promotion rules."""
    # Use promotion rules to generate diverse tensor input types
    arg_dtypes = _get_promoted_dtypes(target_spec.dtype)
    
    arg_specs: List[Spec] = [
        TensorSpec(target_spec.size, target_spec.stride, arg_dtypes[0]),
        TensorSpec(target_spec.size, target_spec.stride, arg_dtypes[1])
    ]
    return "torch.ops.aten.mul", arg_specs


def _get_constant_args_specs(target_spec: Spec) -> Tuple[str, List[Spec]]:
    """Get argument specifications for constant operation."""
    # Constant operation takes no arguments - generates a fixed constant value/tensor
    return "constant", []


def _get_arg_args_specs(target_spec: Spec) -> Tuple[str, List[Spec]]:
    """Get argument specifications for arg operation."""
    # Arg operation takes no arguments - adds a new input argument to the fuzzed program
    return "arg", []
