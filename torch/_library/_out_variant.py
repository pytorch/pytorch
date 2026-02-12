"""
Utilities for finding out-variant overloads of functional custom ops.

Given a functional op (e.g., mylib.foo.default), finds its corresponding
out-variant overload (e.g., mylib.foo.out) by matching schemas.

Based on angelayi's to_out_variant() API (#174473).
"""

from __future__ import annotations

import logging

import torch


log = logging.getLogger(__name__)


def _is_functional(schema: torch._C.FunctionSchema) -> bool:
    """
    A schema is functional if no argument is written to and the name doesn't
    end with '_'.
    """
    op_name = schema.name.split("::")[-1]
    if op_name.endswith("_"):
        return False
    return not any(arg.is_write for arg in schema.arguments)


def _signatures_match(
    schema_a: torch._C.FunctionSchema,
    schema_b: torch._C.FunctionSchema,
) -> bool:
    """Compare two schemas by their non-out arguments (name, type, default value)."""
    non_out_args_a = [arg for arg in schema_a.arguments if not arg.is_out]
    non_out_args_b = [arg for arg in schema_b.arguments if not arg.is_out]
    if len(non_out_args_a) != len(non_out_args_b):
        return False
    for a, b in zip(non_out_args_a, non_out_args_b):
        if a.name != b.name:
            return False
        if a.type != b.type:
            return False
        if a.default_value != b.default_value:
            return False
    return True


def get_out_arg_count(out_op: torch._ops.OpOverload) -> int:
    """Get the number of out arguments for an out variant op."""
    schema = out_op._schema
    return sum(1 for arg in schema.arguments if arg.is_out)


def get_out_arg_names(out_op: torch._ops.OpOverload) -> list[str]:
    """Get the names of out arguments for an out variant op."""
    schema = out_op._schema
    return [arg.name for arg in schema.arguments if arg.is_out]


def _is_tensor_list_return(schema: torch._C.FunctionSchema) -> bool:
    """
    Check if the schema returns Tensor[] (a single list-of-tensor return).

    Tensor[] has len(schema.returns) == 1 with a ListType element type of
    TensorType, but the .out variant may have N out args for the N tensors
    in the list. This helper lets to_out_variant() skip the strict count
    check for list returns.
    """
    if len(schema.returns) != 1:
        return False
    ret_type = schema.returns[0].type
    return isinstance(ret_type, torch.ListType) and ret_type.getElementType().isSubtypeOf(
        torch.TensorType.get()
    )


def to_out_variant(op: torch._ops.OpOverload) -> torch._ops.OpOverload | None:
    """
    Given a functional operator overload, return its corresponding out variant.

    Uses signature matching to find the correct out variant among all overloads.
    Returns None if no matching out variant is found or if the op is not functional.
    """
    schema = op._schema

    if not _is_functional(schema):
        return None

    # Get the op packet to access all overloads
    namespace = op.namespace
    op_name = schema.name.split("::")[1]
    torch_packet = getattr(getattr(torch.ops, namespace), op_name)

    # Search through all overloads for matching out variant
    for overload_name in torch_packet.overloads():
        candidate = getattr(torch_packet, overload_name)
        candidate_schema = candidate._schema

        if not any(arg.is_out for arg in candidate_schema.arguments):
            continue

        if not _signatures_match(schema, candidate_schema):
            continue

        out_args = [arg for arg in candidate_schema.arguments if arg.is_out]
        # For Tensor[] return type, schema.returns has length 1 (a single
        # list-of-tensor return) but the .out variant has N out args for
        # the N tensors in the list. Skip the count check in that case.
        if not _is_tensor_list_return(schema) and len(out_args) != len(
            schema.returns
        ):
            continue

        return candidate

    return None
