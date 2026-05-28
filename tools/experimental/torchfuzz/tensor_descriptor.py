# mypy: ignore-errors
"""Utility functions for generating tensor descriptors in code comments."""

from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec


def format_tensor_descriptor(spec: Spec) -> str:
    """
    Format a tensor or scalar spec as a descriptor comment.

    Args:
        spec: TensorSpec or ScalarSpec to format

    Returns:
        Formatted descriptor string like "size=(64, 176, 96), stride=(16896, 96, 1), dtype=bfloat16, device=cuda"
    """
    if isinstance(spec, ScalarSpec):
        # For scalars, just show the dtype
        dtype_str = str(spec.dtype).replace("torch.", "")
        return f"dtype={dtype_str}"
    elif isinstance(spec, TensorSpec):
        # For tensors, show size, stride, dtype, and device (queried from active plugin).
        # Imported lazily to avoid an import cycle with codegen.py.
        from torchfuzz.codegen import get_device_info

        size_str = str(tuple(spec.size))
        stride_str = str(tuple(spec.stride))
        dtype_str = str(spec.dtype).replace("torch.", "")
        device_str = get_device_info().device_name

        return f"size={size_str}, stride={stride_str}, dtype={dtype_str}, device={device_str}"
    else:
        return "unknown_spec"
