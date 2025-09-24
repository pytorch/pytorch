# mypy: ignore-errors
"""Utility functions for generating tensor descriptors in code comments."""

from typing import Optional

from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec

def format_tensor_descriptor(spec: Spec, var_name: Optional[str] = None) -> str:
    """
    Format a tensor or scalar spec as a descriptor comment.

    Args:
        spec: TensorSpec or ScalarSpec to format
        var_name: Optional variable name to include in comment

    Returns:
        Formatted descriptor string like "size=(64, 176, 96), stride=(16896, 96, 1), dtype=bfloat16, device=cuda"
    """
    if isinstance(spec, ScalarSpec):
        # For scalars, just show the dtype
        dtype_str = str(spec.dtype).replace("torch.", "")
        return f"dtype={dtype_str}"
    elif isinstance(spec, TensorSpec):
        # For tensors, show size, stride, dtype, and device (assuming cuda for now)
        size_str = str(tuple(spec.size))
        stride_str = str(tuple(spec.stride))
        dtype_str = str(spec.dtype).replace("torch.", "")
        device_str = "cuda"  # Most fuzzing is done on GPU

        return f"size={size_str}, stride={stride_str}, dtype={dtype_str}, device={device_str}"
    else:
        return "unknown_spec"


def generate_tensor_descriptor_comment(output_var: str, output_spec: Spec) -> str:
    """
    Generate a full comment line with tensor descriptor for the output variable.

    Args:
        output_var: Name of the output variable
        output_spec: Specification of the output tensor/scalar

    Returns:
        Comment string like "# size=(64, 176, 96), stride=(16896, 96, 1), dtype=bfloat16, device=cuda"
    """
    descriptor = format_tensor_descriptor(output_spec, output_var)
    return f"# {descriptor}"
