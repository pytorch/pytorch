from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def get_out_arg_count(out_op: torch._ops.OpOverload) -> int:
    """Get the number of out arguments for an out variant op.

    Out arguments are mutable tensor arguments (with is_write=True).
    Supports both:
    - Standard format: Tensor(a!) out (kwarg-only, with alias)
    - vLLM format: Tensor! result (positional, without alias)
    """
    return len(_get_mutable_tensor_args(out_op._schema))


def get_out_arg_names(out_op: torch._ops.OpOverload) -> list[str]:
    """Get the names of out arguments for an out variant op.

    Out arguments are mutable tensor arguments (with is_write=True).
    Supports both:
    - Standard format: Tensor(a!) out (kwarg-only, with alias)
    - vLLM format: Tensor! result (positional, without alias)
    """
    return [arg.name for arg in _get_mutable_tensor_args(out_op._schema)]


def to_out_variant(op: torch._ops.OpOverload) -> torch._ops.OpOverload | None:
    """
    Given a functional operator overload, return its corresponding out variant.

    Uses signature matching to find the correct out variant among all overloads.
    The out variant must have the same non-mutable arguments as the functional op.
    """
    schema = op._schema

    # Only convert functional ops (no mutable args)
    if _get_mutable_tensor_args(schema):
        return None

    # Get the op packet to access all overloads
    namespace = op.namespace
    op_name = schema.name.split("::")[1]
    torch_packet = getattr(getattr(torch.ops, namespace), op_name)

    # Get non-mutable args signature for matching
    func_args = _get_non_mutable_args_signature(schema)

    # Search through all overloads for matching out variant
    overload_names = torch._C._jit_get_operation(schema.name)[1]
    for overload_name in overload_names:
        candidate = getattr(torch_packet, overload_name)
        candidate_schema = candidate._schema

        # Must have mutable args to be an out variant
        if not _get_mutable_tensor_args(candidate_schema):
            continue

        # Non-mutable args must match
        candidate_args = _get_non_mutable_args_signature(candidate_schema)
        if candidate_args == func_args:
            return candidate

    return None


def _get_mutable_tensor_args(
    schema: torch._C.FunctionSchema,
) -> list[torch._C.Argument]:
    """Get all mutable tensor arguments from a schema.

    Uses pybind API directly to support both:
    - Tensor(a!) format (with alias annotation)
    - Tensor! format (without alias annotation, used by vLLM)

    Both formats have arg.alias_info.is_write = True
    """
    mutable_args = []
    for arg in schema.arguments:
        if arg.alias_info is not None and arg.alias_info.is_write:
            # Check if it's a tensor type
            type_str = str(arg.type)
            if "Tensor" in type_str:
                mutable_args.append(arg)
    return mutable_args


def _get_non_mutable_args_signature(
    schema: torch._C.FunctionSchema,
) -> list[tuple[str, str]]:
    """Get a signature of non-mutable arguments for matching.

    Returns a list of (name, type_str) tuples for non-mutable args.
    """
    result = []
    for arg in schema.arguments:
        is_mutable = arg.alias_info is not None and arg.alias_info.is_write
        if not is_mutable:
            result.append((arg.name, str(arg.type)))
    return result


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
