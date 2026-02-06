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


def generate_out_variant(
    op: torch._ops.OpOverload,
    out_names: list[str],
    lib: torch.library.Library,
) -> None:
    """
    Generate and register an out variant for a functional operator.

    Args:
        op: The functional operator overload.
        out_names: List of names for the out parameters (e.g., ["out", "out_scale"]).
        lib: The Library object to register with.
    """

    namespace, op_name = op.name().split("::")
    op_name = op_name.split(".")[0]
    schema = op._schema

    overload_name = op._overloadname
    if overload_name == "default":
        out_overload_name = "out"
    else:
        out_overload_name = f"{overload_name}_out"

    num_tensor_outputs = _count_tensor_outputs(schema)
    if len(out_names) != num_tensor_outputs:
        raise ValueError(
            f"out_names has {len(out_names)} names but op returns {num_tensor_outputs} tensors"
        )

    op_packet = getattr(getattr(torch.ops, namespace), op_name)
    if hasattr(op_packet, out_overload_name):
        raise ValueError(
            f"Cannot generate out variant: {namespace}::{op_name}.{out_overload_name} already exists."
        )

    out_schema_str = _functional_to_out_schema(
        schema, out_names, namespace, op_name, out_overload_name
    )

    def out_impl(*args, **kwargs):
        out_tensors = [kwargs.pop(name) for name in out_names]
        result = op(*args, **kwargs)

        if isinstance(result, tuple):
            for out_tensor, res in zip(out_tensors, result):
                out_tensor.resize_as_(res)
                out_tensor.copy_(res)
            return tuple(out_tensors)
        else:
            out_tensors[0].resize_as_(result)
            out_tensors[0].copy_(result)
            return out_tensors[0]

    def out_fake(*args, **kwargs):
        out_tensors = [kwargs.pop(name) for name in out_names]
        if len(out_tensors) == 1:
            return out_tensors[0]
        return tuple(out_tensors)

    lib.define(out_schema_str)

    lib_overload_name = f"{op_name}.{out_overload_name}"
    lib.impl(lib_overload_name, out_impl, "CompositeExplicitAutograd")
    lib.impl(lib_overload_name, out_fake, "Meta")

    torch._ops._refresh_packet(op_packet)


def _count_tensor_outputs(schema: torch._C.FunctionSchema) -> int:
    returns = schema.returns
    count = 0
    for ret in returns:
        ret_type = ret.type
        if isinstance(ret_type, torch._C.TensorType):
            count += 1
        else:
            raise ValueError(
                f"autogen_out only supports operators with all-tensor outputs. "
                f"Found non-tensor return type: {ret_type}"
            )
    return count


def _functional_to_out_schema(
    schema: torch._C.FunctionSchema,
    out_names: list[str],
    namespace: str,
    op_name: str,
    out_overload_name: str,
) -> str:
    """
    Convert functional schema to out variant schema string.

    Input:  "mylib::foo(Tensor x, float scale) -> Tensor", out_names=["result"], out_overload_name="out"
    Output: "mylib::foo.out(Tensor x, float scale, *, Tensor(a!) result) -> Tensor(a!)"
    """
    # Add the out parameters as keyword-only args
    # Use alias annotations a, b, c, ...
    alias_chars = "abcdefghijklmnopqrstuvwxyz"
    if len(out_names) > len(alias_chars):
        raise ValueError(
            f"autogen_out supports at most {len(alias_chars)} outputs, got {len(out_names)}"
        )

    args = list(schema.arguments)

    # Add out args as keyword-only args with alias annotations
    for i, name in enumerate(out_names):
        alias_char = alias_chars[i]
        alias_set = {f"alias::{alias_char}"}
        alias_info = torch._C._AliasInfo(  # pyrefly: ignore[missing-attribute]
            True, alias_set, alias_set
        )
        out_arg = torch._C.Argument(
            name,
            torch._C.TensorType.get(),
            None,  # N
            None,  # default_value
            True,  # kwarg_only
            alias_info,
        )
        args.append(out_arg)

    # Create return arguments with matching alias annotations
    returns: list[torch._C.Argument] = []
    for i in range(len(out_names)):
        alias_char = alias_chars[i]
        alias_set = {f"alias::{alias_char}"}
        alias_info = torch._C._AliasInfo(  # pyrefly: ignore[missing-attribute]
            True, alias_set, alias_set
        )
        ret = torch._C.Argument(
            "",  # return args have no name
            torch._C.TensorType.get(),
            None,
            None,
            False,
            alias_info,
        )
        returns.append(ret)

    out_schema = torch._C.FunctionSchema(
        op_name,
        out_overload_name,
        args,
        returns,
        False,  # is_vararg
        False,  # is_varret
    )

    return str(out_schema)
