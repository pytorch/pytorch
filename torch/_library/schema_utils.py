"""
Schema-based detection for functional → out variant conversion.

This module enables Inductor to automatically find out variants for functional
custom ops, eliminating the need for manual registration. Custom ops that follow
the `.out` overload naming convention will be auto-detected.

Example:
    # User registers ops using .out convention
    @torch.library.custom_op("mylib::my_op", mutates_args=())
    def my_op(x: Tensor) -> Tensor: ...

    @torch.library.custom_op("mylib::my_op.out", mutates_args=("out",))
    def my_op_out(out: Tensor, x: Tensor) -> None: ...

    # Inductor automatically finds the out variant
    out_op = find_out_variant(torch.ops.mylib.my_op.default)
    # Returns: torch.ops.mylib.my_op.out
"""

import dataclasses
import logging
from typing import Optional

import torch
from torch._ops import OpOverload


log = logging.getLogger(__name__)

# Cache parsed schemas to avoid repeated parsing
_schema_cache: dict[OpOverload, Optional["FunctionSchema"]] = {}


def _parse_schema(op: OpOverload):
    """Parse op schema to torchgen FunctionSchema with caching."""
    if op in _schema_cache:
        return _schema_cache[op]

    try:
        from torchgen.model import FunctionSchema

        result = FunctionSchema.parse(str(op._schema))
    except Exception as e:
        log.debug("Failed to parse schema for %s: %s", op, e)
        result = None

    _schema_cache[op] = result
    return result


def find_out_variant(functional_op: OpOverload) -> Optional[OpOverload]:
    """
    Find the out variant of a functional op using schema matching.

    This works for ops registered with the `.out` overload convention:
    - mylib::my_op.default (functional) → mylib::my_op.out

    Args:
        functional_op: A functional custom op

    Returns:
        The matching out variant OpOverload, or None if not found
    """
    try:
        from torchgen.model import SchemaKind
    except ImportError:
        log.debug("torchgen not available")
        return None

    # Parse and verify it's functional
    schema = _parse_schema(functional_op)
    if schema is None or schema.kind() != SchemaKind.functional:
        return None

    # Get normalized signature for comparison (without returns)
    signature = dataclasses.replace(schema.signature(), returns=())

    # Get all overloads for this op
    qualified_name = str(functional_op._schema.name)
    try:
        _, overload_names = torch._C._jit_get_operation(qualified_name)
    except Exception:
        return None

    # Get the op packet
    try:
        ns = functional_op.namespace
        op_name = qualified_name.split("::")[1]
        packet = getattr(getattr(torch.ops, ns), op_name)
    except AttributeError:
        return None

    # Find matching out variant
    for name in overload_names:
        if name == functional_op._overloadname:
            continue

        try:
            candidate = getattr(packet, name)
        except AttributeError:
            continue

        candidate_schema = _parse_schema(candidate)
        if candidate_schema is None:
            continue

        if candidate_schema.kind() == SchemaKind.out:
            candidate_sig = dataclasses.replace(
                candidate_schema.signature(), returns=()
            )
            if candidate_sig == signature:
                return candidate

    return None


def get_out_arg_count(out_op: OpOverload) -> int:
    """Get the number of out arguments for an out variant op."""
    schema = _parse_schema(out_op)
    if schema is None:
        return 0
    return len(schema.arguments.out)


def clear_cache() -> None:
    """Clear the schema cache. For testing only."""
    _schema_cache.clear()
