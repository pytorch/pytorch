"""
Lower functional custom ops to out-variant ExternKernelOut for buffer reuse.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

import torch
from torch._ops import OpOverload

from . import ir


if TYPE_CHECKING:
    from collections.abc import Sequence


log = logging.getLogger(__name__)


def try_lower_to_out_variant(
    kernel: OpOverload,
    example_output: Any,
    tensor_args: Sequence[ir.IRNode],
    non_tensor_args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Optional[ir.IRNode]:
    """Try lowering a functional custom op to ExternKernelOut.

    On success, returns IR node(s) that replace the FallbackKernel path.
    On failure (no out-variant, non-functional, unsupported output type),
    returns None to fall through to FallbackKernel.
    """
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
            kernel,
            out_op,
            example_output,
            tensor_args,
            non_tensor_args,
            kwargs,
        )

    # TODO: multi-output ops (tuple/list returns) will be handled in a
    # follow-up via CustomOpMultiOutputNode + AllocatingMultiOutput.
    return None


def _lower_single_output(
    kernel: OpOverload,
    out_op: OpOverload,
    example_output: torch.Tensor,
    tensor_args: Sequence[ir.IRNode],
    non_tensor_args: Sequence[Any],
    kwargs: dict[str, Any],
) -> ir.ExternKernelOut:
    """Lower a single-output functional op to ExternKernelOut."""
    layout = ir.FixedLayout(
        device=example_output.device,
        dtype=example_output.dtype,
        size=[*example_output.shape],
        stride=[*example_output.stride()],
    )

    python_kernel_name = _make_python_kernel_name(out_op)

    node = ir.ExternKernelOut(
        layout=layout,
        inputs=list(tensor_args),
        constant_args=list(non_tensor_args),
        kwargs=kwargs,
        python_kernel_name=python_kernel_name,
        op_overload=out_op,
    )

    log.debug("Lowered %s -> %s via ExternKernelOut", kernel, out_op)
    return node


def _make_python_kernel_name(out_op: OpOverload) -> str:
    """Build fully-qualified kernel name, e.g. 'torch.ops.mylib.add_one.out'."""
    ns = out_op.namespace
    op_name = out_op._schema.name.split("::")[1]
    overload = out_op._overloadname
    return f"torch.ops.{ns}.{op_name}.{overload}"
