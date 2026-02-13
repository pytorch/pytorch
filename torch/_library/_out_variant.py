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


def _is_mutable_arg(arg: torch._C.Argument) -> bool:
    return arg.alias_info is not None and arg.alias_info.is_write


def _signatures_match(
    schema_a: torch._C.FunctionSchema,
    schema_b: torch._C.FunctionSchema,
) -> bool:
    """Compare two schemas by their non-mutable arguments (name, type, default value)."""
    non_mutable_args_a = [arg for arg in schema_a.arguments if not _is_mutable_arg(arg)]
    non_mutable_args_b = [arg for arg in schema_b.arguments if not _is_mutable_arg(arg)]
    if len(non_mutable_args_a) != len(non_mutable_args_b):
        return False
    for a, b in zip(non_mutable_args_a, non_mutable_args_b):
        if a.name != b.name:
            return False
        if str(a.type) != str(b.type):
            return False
        if a.default_value != b.default_value:
            return False
    return True


def _has_valid_out_variant_returns(
    schema: torch._C.FunctionSchema,
    mutable_args: list[torch._C.Argument],
) -> bool:
    """Out variant must return either nothing or the mutable args themselves."""
    if len(schema.returns) == 0:
        return True

    if len(schema.returns) != len(mutable_args):
        return False

    # Each return must alias exactly one mutable arg, in order
    for ret, arg in zip(schema.returns, mutable_args):
        if ret.alias_info is None or arg.alias_info is None:
            return False
        if ret.alias_info.before_set != arg.alias_info.before_set:
            return False
    return True


def to_out_variant(op: torch._ops.OpOverload) -> torch._ops.OpOverload | None:
    """
    Given a functional operator overload, return its corresponding out variant.
    """
    schema = op._schema

    if not _is_functional(schema):
        raise RuntimeError(
            f"Failed to find out variant for op '{op}' as its schema is not functional. \n"
            f"  {schema}"
        )

    # Get the op packet to access all overloads
    namespace = op.namespace
    op_name = schema.name.split("::")[1]
    torch_packet = getattr(getattr(torch.ops, namespace), op_name)

    # Search through all overloads for matching out variant
    for overload_name in torch_packet.overloads():
        candidate = getattr(torch_packet, overload_name)

        if torch.Tag.out_variant not in candidate.tags:
            continue

        candidate_schema = candidate._schema

        if not _signatures_match(schema, candidate_schema):
            continue

        # We assume that all mutable args are used for out
        mutable_args = [
            arg for arg in candidate_schema.arguments if _is_mutable_arg(arg)
        ]
        if len(mutable_args) != len(schema.returns):
            continue

        if not _has_valid_out_variant_returns(candidate_schema, mutable_args):
            raise RuntimeError(
                f"Out variant {candidate} has invalid returns. "
                f"Expected either no returns or returns that alias the mutable args, "
                f"got: {candidate_schema}"
            )

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
        tagged_info = _get_out_variants_info(functional_op)
        raise AssertionError(
            f"We did not find an out variant for {functional_op}. Some common mistakes include:\n"
            "  1. The out variant is missing the torch.Tag.out_variant tag.\n"
            "  2. The out variant is not an overload of the original op (e.g., 'op.out' or 'op.overload_out') \n"
            "  3. The out variant's input arguments does not match the functional op's signature (excluding the mutable args).\n"
            "  4. The original operator is not functional.\n"
            f"Overloads tagged with out_variant:\n"
            f"{tagged_info or '  (none)'}"
        )
    if out_op != expected_out_op:
        raise AssertionError(
            f"to_out_variant({functional_op}) returned {out_op}, "
            f"but expected {expected_out_op}. "
            f"The out variant name does not match the functional op."
        )


def _get_out_variants_info(functional_op) -> str:
    """Collect information about overloads tagged with out_variant for debugging."""
    namespace = functional_op.namespace
    op_name = functional_op._schema.name.split("::")[1]
    torch_packet = getattr(getattr(torch.ops, namespace), op_name)

    overloads_info: list[str] = []
    for overload_name in torch_packet.overloads():
        candidate = getattr(torch_packet, overload_name)
        if torch.Tag.out_variant in candidate.tags:
            overloads_info.append(f"  - {overload_name}: {candidate._schema}")

    return "\n".join(overloads_info)
