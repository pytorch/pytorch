from __future__ import annotations

import dataclasses
import logging

import torch
from torchgen.model import FunctionSchema, SchemaKind


log = logging.getLogger(__name__)


def to_out_variant(op: torch._ops.OpOverload) -> torch._ops.OpOverload | None:
    """
    Given a functional operator overload, return its corresponding out variant.

    Uses signature matching to find the correct out variant among all overloads.
    """
    native_schema = _parse_schema(op._schema)

    # Only convert functional ops
    if native_schema.kind() != SchemaKind.functional:
        raise RuntimeError(
            f"Failed to find out variant for op '{op}' as its schema is not functional. \n"
            f"  {op._schema}"
        )
        return None

    # Get the normalized signature for matching
    signature = dataclasses.replace(native_schema.signature(), returns=())

    # Get the op packet to access all overloads
    namespace = op.namespace
    op_name = op._schema.name.split("::")[1]
    torch_packet = getattr(getattr(torch.ops, namespace), op_name)

    # Search through all overloads for matching out variant
    for overload_name in torch_packet.overloads():
        candidate = getattr(torch_packet, overload_name)
        candidate_native_schema = _parse_schema(candidate._schema)

        if candidate_native_schema.kind() != SchemaKind.out:
            continue

        candidate_signature = dataclasses.replace(
            candidate_native_schema.signature(), returns=()
        )
        if candidate_signature != signature:
            continue

        if len(candidate_native_schema.arguments.out) != len(native_schema.returns):
            continue

        return candidate

    return None


def _parse_schema(
    schema: torch._C.FunctionSchema,
) -> FunctionSchema:
    """Convert a pybind FunctionSchema to a torchgen FunctionSchema."""
    try:
        return FunctionSchema.parse(str(schema))
    except Exception as e:
        raise ValueError(
            f"Failed to parse schema '{schema}'. This means we will "
            "not be able to find the corresponding op or out variant."
        ) from e


def check_out_variant(
    functional_op: torch._ops.OpOverload, expected_out_op: torch._ops.OpOverload
) -> None:
    """
    Checks that to_out_variant returns the expected out variant for a functional op.
    Raises AssertionError if the out variant is not valid.
    """
    out_op = to_out_variant(functional_op)
    if out_op is None:
        # Collect information about all out variants for debugging
        out_variants_info = _get_out_variants_info(functional_op)
        raise AssertionError(
            f"We did not find an out variant for {functional_op}. Some common mistakes include:\n"
            "  1. The out variant is not an overload of the original op (e.g., 'op.out' or 'op.overload_out') \n"
            "  2. The out variant's input arguments does not match the functional op's signature.\n"
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


def _get_out_variants_info(functional_op) -> str:
    """Collect information about all overloads for debugging."""
    namespace = functional_op.namespace
    op_name = functional_op._schema.name.split("::")[1]
    torch_packet = getattr(getattr(torch.ops, namespace), op_name)

    overloads_info: list[str] = []
    for overload_name in torch_packet.overloads():
        candidate = getattr(torch_packet, overload_name)
        overloads_info.append(f"  - {overload_name}: {candidate._schema}")

    return "\n".join(overloads_info)
