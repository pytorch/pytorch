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
        if len(out_args) != len(schema.returns):
            continue

        return candidate

    return None


def check_out_variant(
    functional_op: torch._ops.OpOverload, expected_out_op: torch._ops.OpOverload
) -> None:
    """
    Checks that to_out_variant returns the expected out variant for a functional op.
    Raises AssertionError if the out variant is not valid.
    """
    out_op = to_out_variant(functional_op)
    if out_op is None:
        out_variants_info = _get_out_variants_info(functional_op)
        raise AssertionError(
            f"We did not find an out variant for {functional_op}. Some common mistakes include:\n"
            "  1. The out variant is not an overload of the original op (e.g., 'op.out' or 'op.overload_out') \n"
            "  2. The out variant's input arguments does not match the functional op's signature (excluding the out kwargs).\n"
            "  3. The original operator is not functional.\n"
            f"Found overloads for {functional_op}:\n"
            f"{out_variants_info}"
        )
    if out_op != expected_out_op:
        raise AssertionError(
            f"to_out_variant({functional_op}) returned {out_op}, "
            f"but expected {expected_out_op}. "
            f"The out variant name does not match the functional op."
        )


def _get_out_variants_info(functional_op: torch._ops.OpOverload) -> str:
    """Collect information about all overloads for debugging."""
    namespace = functional_op.namespace
    op_name = functional_op._schema.name.split("::")[1]
    torch_packet = getattr(getattr(torch.ops, namespace), op_name)

    overloads_info: list[str] = []
    for overload_name in torch_packet.overloads():
        candidate = getattr(torch_packet, overload_name)
        overloads_info.append(f"  - {overload_name}: {candidate._schema}")

    return "\n".join(overloads_info)
