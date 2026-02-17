"""
Inductor lowering of functional custom ops to their out-variant overload.

Instead of routing through FallbackKernel (ExternKernelAlloc, should_allocate=False),
this module lowers to ExternKernelOut (should_allocate=True) so that output buffers
participate in Inductor's AllocateLine.plan() buffer reuse.

Supports:
    - Single-output ops: functional(x) -> Tensor  (PR A2)
    - Multi-output ops: functional(x) -> (Tensor, ...)  (PR A3)

Gated behind ``config.lower_custom_ops_to_out_variant`` (default False).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
from torch._ops import OpOverload

from . import config, ir


if TYPE_CHECKING:
    from collections.abc import Sequence


log = logging.getLogger(__name__)


def try_lower_to_out_variant(
    kernel: OpOverload,
    example_output: Any,
    tensor_args: Sequence[ir.IRNode],
    non_tensor_args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Optional[Union[ir.IRNode, list[ir.IRNode]]]:
    """
    Attempt to lower a functional custom op to its out-variant via ExternKernelOut.

    Returns the IR node(s) if lowering succeeds, or None to fall through to
    FallbackKernel.  For single-output ops returns one node; for multi-output
    ops returns a list of nodes.

    Called from the FallbackKernel.create() hook in ir.py when
    ``config.lower_custom_ops_to_out_variant`` is True.
    """
    if not config.lower_custom_ops_to_out_variant:
        return None

    if not isinstance(kernel, OpOverload):
        return None

    from torch._library._out_variant import (
        _is_functional,
        get_out_arg_names,
        to_out_variant,
    )

    if not _is_functional(kernel._schema):
        return None

    out_op = to_out_variant(kernel)
    if out_op is None:
        return None

    out_arg_names = get_out_arg_names(out_op)

    if isinstance(example_output, torch.Tensor):
        if len(out_arg_names) != 1:
            log.debug(
                "Skipping %s: single output but %d out args",
                kernel,
                len(out_arg_names),
            )
            return None
        return _lower_single_output(
            kernel, out_op, out_arg_names,
            example_output, tensor_args, non_tensor_args, kwargs,
        )
    elif isinstance(example_output, (tuple, list)):
        tensors = [t for t in example_output if isinstance(t, torch.Tensor)]
        if len(tensors) == len(example_output) and len(tensors) == len(out_arg_names):
            return _lower_multi_output(
                kernel, out_op, out_arg_names,
                example_output, tensor_args, non_tensor_args, kwargs,
            )
        log.debug(
            "Skipping %s: %d tensor outputs vs %d out args",
            kernel, len(tensors), len(out_arg_names),
        )
        return None
    else:
        return None


def _lower_single_output(
    kernel: OpOverload,
    out_op: OpOverload,
    out_arg_names: list[str],
    example_output: torch.Tensor,
    tensor_args: Sequence[ir.IRNode],
    non_tensor_args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Optional[ir.IRNode]:
    """Lower a single-output functional op to ExternKernelOut.

    Implemented in a follow-up commit.
    """
    return None


def _lower_multi_output(
    kernel: OpOverload,
    out_op: OpOverload,
    out_arg_names: list[str],
    example_output: Union[tuple, list],
    tensor_args: Sequence[ir.IRNode],
    non_tensor_args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Optional[list[ir.IRNode]]:
    """Lower a multi-output functional op to ExternKernelOut.

    Implemented in a follow-up commit.
    """
    return None


def _make_python_kernel_name(out_op: OpOverload) -> str:
    """Build the fully-qualified Python kernel name for an out-variant op.

    Example: torch.ops.mylib.add_one.out
    """
    ns = out_op.namespace
    op_name = out_op._schema.name.split("::")[1]
    overload = out_op._overloadname
    return f"torch.ops.{ns}.{op_name}.{overload}"
