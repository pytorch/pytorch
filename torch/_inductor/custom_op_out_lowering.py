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
from .virtualized import V


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
    """Lower a single-output functional op to ExternKernelOut."""
    layout = ir.FixedLayout(
        device=example_output.device,
        dtype=example_output.dtype,
        size=[*example_output.shape],
        stride=[*example_output.stride()],
    )

    node = CustomOpExternKernelOut(
        layout=layout,
        inputs=list(tensor_args),
        constant_args=list(non_tensor_args),
        out_op=out_op,
        out_arg_names=out_arg_names,
        op_overload=out_op,
        kwargs=kwargs,
    )

    log.debug("Lowered %s -> %s (out args: %s)", kernel, out_op, out_arg_names)
    return node


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


def _codegen_input_args(node: ir.ExternKernel) -> list[str]:
    """Codegen positional + constant args as a list of strings."""
    assert ir.is_node_sequence(node.inputs)
    args = [x.codegen_reference() for x in node.inputs]  # type: ignore[union-attr]
    for const in node.constant_args:
        args.append(V.graph.wrapper_code.val_to_arg_str(const))
    return args


def _codegen_kwargs(node: ir.ExternKernel, skip_names: set[str]) -> list[str]:
    """Codegen keyword args, skipping any names in ``skip_names``."""
    result = []
    for k, v in node.kwargs.items():
        if k not in skip_names:
            result.append(f"{k}={V.graph.wrapper_code.val_to_arg_str(v)}")
    return result


class CustomOpExternKernelOut(ir.ExternKernelOut):
    """
    ExternKernelOut for single-output custom ops with flexible out-arg naming.

    Unlike the base ExternKernelOut which hardcodes ``out=`` in codegen,
    this uses the actual out-arg name from the op's schema (e.g., "result",
    "output").

    Inherits ``should_allocate() = True`` from ExternKernelOut, enabling
    Inductor's AllocateLine.plan() buffer reuse.
    """

    out_op: OpOverload
    out_arg_names: list[str]

    def __init__(
        self,
        layout: ir.Layout,
        inputs: Sequence[ir.IRNode],
        constant_args: Sequence[Any] = (),
        out_op: Optional[OpOverload] = None,
        out_arg_names: Optional[list[str]] = None,
        op_overload: Optional[OpOverload] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        assert out_op is not None, "out_op is required"
        python_kernel_name = _make_python_kernel_name(out_op)
        super().__init__(
            layout=layout,
            inputs=inputs,
            constant_args=constant_args,
            kwargs=kwargs,
            python_kernel_name=python_kernel_name,
            op_overload=op_overload,
        )
        self.out_op = out_op
        self.out_arg_names = out_arg_names or ["out"]

    def codegen(self, wrapper: Any) -> None:
        self.codegen_comment(wrapper)
        args = _codegen_input_args(self)
        kwargs_list = _codegen_kwargs(self, skip_names=set(self.out_arg_names))
        kernel_name = self.get_kernel_name()
        out_ref = self.codegen_reference()

        all_args = [*args, *kwargs_list]
        all_args.append(f"{self.out_arg_names[0]}={out_ref}")
        wrapper.writeline(f"{kernel_name}({', '.join(all_args)})")
