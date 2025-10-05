"""Type promotion utilities for PyTorch operations.

This module provides utilities to handle PyTorch's type promotion rules correctly,
ensuring that the fuzzer's type system matches PyTorch's actual behavior.
"""

def get_promoted_dtype(input_dtype: str, operation: str) -> str:
    """
    Get the output dtype after applying a PyTorch operation to a tensor.

    Args:
        input_dtype: The input tensor's dtype as a string
        operation: The operation name (e.g., 'exp', 'sqrt', 'div')

    Returns:
        The resulting dtype after type promotion
    """

    # Operations that promote integer types to float
    FLOAT_PROMOTING_OPS = {
        'exp', 'sqrt', 'log', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
        'asin', 'acos', 'atan', 'sigmoid', 'softmax', 'gelu', 'relu',
        'div', 'pow', 'reciprocal'
    }

    # Integer dtypes
    INTEGER_DTYPES = {"int8", "int16", "int32", "int64", "uint8", "bool"}

    if operation in FLOAT_PROMOTING_OPS and input_dtype in INTEGER_DTYPES:
        # PyTorch promotes integer types to float32 for these operations
        return "float32"

    # For operations that don't promote types, return the input dtype
    return input_dtype


def can_produce_integer_tensor(operation: str) -> bool:
    """
    Check if an operation can produce integer tensors.

    Args:
        operation: The operation name

    Returns:
        True if the operation can produce integer tensors, False otherwise
    """

    # Operations that always promote to float
    FLOAT_ONLY_OPS = {
        'exp', 'sqrt', 'log', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
        'asin', 'acos', 'atan', 'sigmoid', 'softmax', 'gelu', 'relu',
        'div', 'pow', 'reciprocal'
    }

    return operation not in FLOAT_ONLY_OPS


def is_integer_dtype(dtype: str) -> bool:
    """Check if a dtype is an integer type."""
    return dtype in {"int8", "int16", "int32", "int64", "uint8", "bool"}


def is_float_dtype(dtype: str) -> bool:
    """Check if a dtype is a float type."""
    return dtype in {"float16", "float32", "float64", "bfloat16"}
