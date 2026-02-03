from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import torch
from torch import _C
from torchgen.model import FunctionSchema, SchemaKind


if TYPE_CHECKING:
    from .custom_ops import CustomOpDef

log = logging.getLogger(__name__)


def generate_out_variant(custom_op: CustomOpDef, out_names: list[str]) -> None:
    """
    Given a functional custom op, generate and register its out variant.

    Args:
        custom_op: The CustomOpDef for the functional op
        out_names: List of names for the out parameters (e.g., ["output", "output_scale"])
    """
    op_overload = custom_op._opoverload

    namespace, op_name = op_overload.name().split("::")
    op_name = op_name.split(".")[0]
    schema = op_overload._schema

    overload_name = op_overload._overloadname
    if overload_name == "default":
        out_overload_name = "out"
    else:
        out_overload_name = f"{overload_name}_out"

    num_tensor_outputs = _count_tensor_outputs(schema)
    if len(out_names) != num_tensor_outputs:
        raise ValueError(
            f"autogen_out has {len(out_names)} names but op returns {num_tensor_outputs} tensors"
        )

    op_packet = getattr(getattr(torch.ops, namespace), op_name)
    if hasattr(op_packet, out_overload_name):
        raise ValueError(
            f"Cannot autogen out variant: {namespace}::{op_name}.{out_overload_name} already exists. "
            "Remove autogen_out or remove the manual out variant definition."
        )

    out_schema_str = _functional_to_out_schema(
        schema, out_names, namespace, op_name, out_overload_name
    )

    def out_impl(*args, **kwargs):
        # Extract out tensors from kwargs by name
        out_tensors = [kwargs.pop(name) for name in out_names]

        # Call the functional variant
        result = op_overload(*args, **kwargs)

        # Copy results to out tensors
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

    # Register the out variant
    custom_op._lib.define(out_schema_str)

    lib_overload_name = f"{op_name}.{out_overload_name}"

    # Register only for the original op's dispatch keys
    for device_type in custom_op._backend_fns:
        if device_type is None:
            dispatch_key = "CompositeExplicitAutograd"
        else:
            dispatch_key = _C._dispatch_key_for_device(device_type)
        custom_op._lib.impl(lib_overload_name, out_impl, dispatch_key)

    custom_op._lib.impl(lib_overload_name, out_fake, "Meta")

    # Refresh the op packet so overloads() includes the new out variant
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
        f"{namespace}::{op_name}",
        out_overload_name,
        args,
        returns,
        False,  # is_vararg
        False,  # is_varret
    )

    return str(out_schema)


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
