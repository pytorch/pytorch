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
        "torch.ops.higher_order.print": CustomCodegen(
            python=generate_print_python,
            cpp=generate_print_cpp,  # Optional
        ),
    }

Usage:
    codegen = CUSTOM_EXTERN_KERNEL_CODEGEN[op]
    codegen.python(node, writeline)  # Direct attribute access
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from .. import ir

    # Type alias for codegen function signature (only used for type checking)
    CodegenFunc = Callable[[ir.FallbackKernel, Callable[[str], None]], None]


@dataclass
class CustomCodegen:
    """
    Container for custom codegen implementations.

    Attributes:
        python: Codegen function for Python wrapper (optional)
        cpp: Codegen function for C++ wrapper (optional)
    """

    python: Any = None
    cpp: Any = None


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
        print('x = {}, y = {}'.format(buf0, buf1))
        print('x = {x}, y = {y}'.format(x=buf0, y=buf1))
        print('x = {}, y = {y}'.format(buf0, y=buf1))
    """
    codegen_args: list[str] = node.codegen_args()
    codegen_kwargs: list[str] = node.codegen_kwargs()

    # First arg is the format string
    if not codegen_args:
        raise ValueError(
            "generate_print_python requires a format string as the first positional argument"
        )
    format_str: str = codegen_args[0]

    # Remaining args are positional arguments for .format()
    positional_args = codegen_args[1:]

    args_str = ", ".join(positional_args + codegen_kwargs)
    writeline(
        f"print({format_str}.format({args_str}))"
        if args_str
        else f"print({format_str})"
    )


# Registry mapping operator names to their custom codegen implementations
# Usage: CUSTOM_EXTERN_KERNEL_CODEGEN[op_name].python(node, writeline)
CUSTOM_EXTERN_KERNEL_CODEGEN: dict[str, CustomCodegen] = {
    "torch.ops.higher_order.print": CustomCodegen(
        python=generate_print_python,
    ),
}
