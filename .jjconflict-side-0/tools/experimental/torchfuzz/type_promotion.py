"""Type promotion utilities for torchfuzz operators."""

import random

import torch


# Define promotion chains - types that can promote to the target
# PyTorch promotion hierarchy (simplified):
# - bool < int8 < int16 < int32 < int64 < float16 < float32 < float64 < complex64 < complex128
# - uint types have limited promotion support
PROMOTION_CHAINS = {
    torch.bool: [torch.bool],
    torch.int8: [torch.bool, torch.int8],
    torch.int16: [torch.bool, torch.int8, torch.int16],
    torch.int32: [torch.bool, torch.int8, torch.int16, torch.int32],
    torch.int64: [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64],
    torch.float16: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
    ],
    torch.float32: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
    ],
    torch.float64: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    ],
    torch.complex64: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.complex64,
    ],
    torch.complex128: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
}


def get_promoted_dtypes(target_dtype: torch.dtype) -> list[torch.dtype]:
    """
    Generate two dtypes that will promote to target_dtype via PyTorch's type promotion rules.
    """
    # Get compatible input types for the target dtype
    compatible_types = PROMOTION_CHAINS.get(target_dtype, [target_dtype])

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


def get_dtype_name(dtype: torch.dtype) -> str:
    """Get string name for a torch dtype."""
    return str(dtype).split(".")[-1]


def get_promotion_table_for_strings() -> dict:
    """
    Get promotion table using string dtype names for backward compatibility.
    Returns dictionary mapping output dtype string to possible input dtype string pairs.
    """
    return {
        "float32": [
            ("float32", "float32"),
            ("bfloat16", "float32"),
            ("float32", "bfloat16"),
            ("float16", "float32"),
            ("float32", "float16"),
        ],
        "bfloat16": [
            ("bfloat16", "bfloat16"),
            ("float32", "bfloat16"),
            ("bfloat16", "float32"),
        ],
        "float16": [
            ("float16", "float16"),
            ("float32", "float16"),
            ("float16", "float32"),
        ],
        "int32": [
            ("int32", "int32"),
            ("int64", "int32"),
            ("int32", "int64"),
        ],
        "int64": [
            ("int64", "int64"),
            ("int32", "int64"),
            ("int64", "int32"),
        ],
        "bool": [
            ("bool", "bool"),
        ],
    }


def get_dtype_map() -> dict:
    """Get mapping from string names to torch dtypes."""
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
        "int8": torch.int8,
        "int16": torch.int16,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }


def get_scalar_promotion_pairs(
    target_dtype: torch.dtype,
) -> list[tuple[torch.dtype, torch.dtype]]:
    """
    Get promotion pairs for scalar operations.
    Returns list of (dtype1, dtype2) tuples that promote to target_dtype.
    """
    return (
        [
            (torch.float32, torch.float32),
            (torch.float16, torch.float32),
            (torch.float32, torch.float16),
            (torch.int32, torch.float32),
            (torch.float32, torch.int32),
        ]
        if target_dtype == torch.float32
        else [
            (torch.float64, torch.float64),
            (torch.float32, torch.float64),
            (torch.float64, torch.float32),
        ]
        if target_dtype == torch.float64
        else [
            (torch.int32, torch.int32),
            (torch.int64, torch.int32),
            (torch.int32, torch.int64),
        ]
        if target_dtype == torch.int32
        else [
            (torch.int64, torch.int64),
            (torch.int32, torch.int64),
            (torch.int64, torch.int32),
        ]
        if target_dtype == torch.int64
        else [(target_dtype, target_dtype)]
    )
