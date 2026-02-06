from __future__ import annotations

import dataclasses
import logging

import torch
from torchgen.model import FunctionSchema, SchemaKind

log = logging.getLogger(__name__)


def get_out_arg_count(out_op: torch._ops.OpOverload) -> int:
    """Get the number of out arguments for an out variant op.

    Out arguments are mutable tensor arguments that appear at the end of
    the argument list in the out variant's schema.
    """
    native_schema = _pybind_schema_to_native_schema(out_op._schema)
    if native_schema is None:
        return 0

    if native_schema.kind() != SchemaKind.out:
        return 0

    return len(native_schema.arguments.out)


def get_out_arg_names(out_op: torch._ops.OpOverload) -> list[str]:
    """Get the names of out arguments for an out variant op.

    Out arguments are keyword-only mutable tensor arguments in the schema.
    Returns an empty list if the op is not an out variant.
    """
    native_schema = _pybind_schema_to_native_schema(out_op._schema)
    if native_schema is None:
        return []

    if native_schema.kind() != SchemaKind.out:
        return []

    return [arg.name for arg in native_schema.arguments.out]


def to_out_variant(op: torch._ops.OpOverload) -> torch._ops.OpOverload | None:
    """
    Given a functional operator overload, return its corresponding out variant.

    Uses signature matching to find the correct out variant among all overloads.
    """
    native_schema = _pybind_schema_to_native_schema(op._schema)
    if native_schema is None:
        return None

    # Only convert functional ops
    if native_schema.kind() != SchemaKind.functional:
        return None

    # Get the normalized signature for matching
    signature = dataclasses.replace(native_schema.signature(), returns=())

    # Get the op packet to access all overloads
    namespace = op.namespace
    op_name = op._schema.name.split("::")[1]
    torch_packet = getattr(getattr(torch.ops, namespace), op_name)

    # Search through all overloads for matching out variant
    overload_names = torch._C._jit_get_operation(op._schema.name)[1]
    for overload_name in overload_names:
        candidate = getattr(torch_packet, overload_name)
        candidate_native_schema = _pybind_schema_to_native_schema(candidate._schema)
        if candidate_native_schema is None:
            continue

        if candidate_native_schema.kind() != SchemaKind.out:
            continue

        candidate_signature = dataclasses.replace(
            candidate_native_schema.signature(), returns=()
        )
        if candidate_signature == signature:
            return candidate

    return None


def _pybind_schema_to_native_schema(
    schema: torch._C.FunctionSchema,
) -> FunctionSchema | None:
    """Convert a pybind FunctionSchema to a torchgen FunctionSchema."""
    try:
        return FunctionSchema.parse(str(schema))
    except Exception:
        log.debug("Failed to parse schema: %s", schema)
        return None


def check_out_variant(functional_op, expected_out_op):
    """
    Verify that to_out_variant returns the expected out variant for a functional op.

    Returns True if successful, raises AssertionError with detailed message otherwise.
    """
    out_op = to_out_variant(functional_op)
    if out_op is None:
        raise AssertionError(
            f"to_out_variant({functional_op}) returned None. "
            f"Expected to find out variant {expected_out_op}. "
            f"Check that the out variant overload name follows the correct naming convention "
            f"(e.g., 'op.out' for default overload or 'op.overload_out' for named overloads)."
        )
    if out_op != expected_out_op:
        raise AssertionError(
            f"to_out_variant({functional_op}) returned {out_op}, "
            f"but expected {expected_out_op}. "
            f"The out variant name does not match the functional op."
        )
    return True
