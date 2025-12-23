# mypy: allow-untyped-defs
"""
Custom extern kernel codegen registry.

This module provides a registry that maps operators to their custom codegen
implementations. This allows us to keep all custom implementations in one place
and easily extend support for both Python and C++ wrappers.

To add a new custom implementation:
1. Create a codegen function with signature:
   def my_codegen(node: ir.FallbackKernel, writeline: Callable[[str], None]) -> None:
2. Register it in CUSTOM_EXTERN_KERNEL_CODEGEN with the operator name as key

Example:
    CUSTOM_EXTERN_KERNEL_CODEGEN = {
        "torch.ops.higher_order.print": {
            "python": generate_print_python,
            "cpp": generate_print_cpp,  # Optional, falls back to python if not provided
        },
    }
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from .. import ir


def generate_print_python(
    node: ir.FallbackKernel,
    writeline: Callable[[str], None],
) -> None:
    """
    Generate a builtin print call for the print HOP fallback (Python wrapper).

    This function generates Python code that calls the builtin print function
    with format string interpolation.

    Args:
        node: The FallbackKernel IR node representing the print HOP call.
        writeline: A function that writes a line of code to the output buffer.

    Example generated code:
        print('x = {x}, y = {y}'.format(x=buf0, y=buf1))
    """
    codegen_args = node.codegen_args()
    codegen_kwargs = node.codegen_kwargs()

    # First arg is the format string
    format_str = codegen_args[0] if codegen_args else "''"

    if codegen_kwargs:
        kwargs_str = ", ".join(codegen_kwargs)
        writeline(f"print({format_str}.format({kwargs_str}))")
    else:
        writeline(f"print({format_str})")


# Registry mapping operator names to their custom codegen implementations
# Each entry can have:
#   - "python": codegen function for Python wrapper (required)
#   - "cpp": codegen function for C++ wrapper (optional, falls back to default if not provided)
CUSTOM_EXTERN_KERNEL_CODEGEN: dict[
    str,
    dict[str, Callable[[ir.FallbackKernel, Callable[[str], None]], None]],
] = {
    "torch.ops.higher_order.print": {
        "python": generate_print_python,
        # "cpp": generate_print_cpp,  # Add C++ implementation when needed
    },
}


def get_custom_codegen(
    op_name: str | None,
    wrapper_type: str = "python",
) -> Callable[[ir.FallbackKernel, Callable[[str], None]], None] | None:
    """
    Get the custom codegen function for an operator.

    Args:
        op_name: The operator name (e.g., "torch.ops.higher_order.print"), or None
        wrapper_type: Either "python" or "cpp"

    Returns:
        The codegen function if found, None otherwise.
        For cpp wrapper, falls back to python implementation if cpp not provided.
    """
    if op_name is None or op_name not in CUSTOM_EXTERN_KERNEL_CODEGEN:
        return None

    codegen_map = CUSTOM_EXTERN_KERNEL_CODEGEN[op_name]
    # Try to get the specific wrapper type without fallback between cpp and python
    return codegen_map.get(wrapper_type)


def has_custom_codegen(op_name: str) -> bool:
    """Check if an operator has a custom codegen implementation."""
    return op_name in CUSTOM_EXTERN_KERNEL_CODEGEN
