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


def generate_print_cpp(
    node: ir.FallbackKernel,
    writeline: Callable[[str], None],
) -> None:
    """Generate std::cout call with format string interpolation."""
    import re

    args = node.codegen_args()
    if not args:
        raise ValueError("generate_print_cpp requires a format string")

    format_str = args[0].strip('"\'')

    # Get arg names from schema (first arg is format_str, rest are values)
    # Schema: print(str format_str, int arg0, *, int x, int y) -> ()
    schema = node.op_overload._schema if hasattr(node.op_overload, "_schema") else None
    arg_names = []
    if schema:
        # Skip first arg (format_str), get names of remaining args
        arg_names = [arg.name for arg in schema.arguments[1:]]

    # Map arg names to their values (args[1:] are the values)
    arg_values = args[1:]
    name_to_value = dict(zip(arg_names, arg_values)) if arg_names else {}

    # Also handle any explicit kwargs from codegen_kwargs
    for kv in node.codegen_kwargs():
        if "=" in kv:
            k, v = kv.split("=", 1)
            name_to_value[k.strip()] = v.strip()

    parts, last_end, pos_idx = [], 0, 0
    for m in re.finditer(r"\{(\w*)\}", format_str):
        if m.start() > last_end:
            parts.append(f'"{format_str[last_end:m.start()]}"')
        name = m.group(1)
        if name == "":
            # Positional placeholder {} - use argN naming convention
            pos_name = f"arg{pos_idx}"
            if pos_name in name_to_value:
                parts.append(name_to_value[pos_name])
            elif pos_idx < len(arg_values):
                parts.append(arg_values[pos_idx])
            else:
                parts.append('""')
            pos_idx += 1
        elif name in name_to_value:
            parts.append(name_to_value[name])
        else:
            parts.append('""')
        last_end = m.end()

    if last_end < len(format_str):
        parts.append(f'"{format_str[last_end:]}"')

    writeline(f"std::cout << {' << '.join(parts or [args[0]])} << std::endl;")


# Registry mapping operator names to their custom codegen implementations
# Usage: CUSTOM_EXTERN_KERNEL_CODEGEN[op_name].python(node, writeline)
CUSTOM_EXTERN_KERNEL_CODEGEN: dict[str, CustomCodegen] = {
    "torch.ops.higher_order.print": CustomCodegen(
        python=generate_print_python,
        cpp=generate_print_cpp,
    ),
}
