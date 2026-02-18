"""
Lower functional custom ops to out-variant ExternKernelOut for buffer reuse.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
from torch._ops import OpOverload

from . import ir
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
    """Try lowering a functional custom op to ExternKernelOut.

    Returns IR node(s) on success, or None to fall through to FallbackKernel.
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
            out_arg_names,
            example_output,
            tensor_args,
            non_tensor_args,
            kwargs,
        )
    elif isinstance(example_output, (tuple, list)):
        tensors = [t for t in example_output if isinstance(t, torch.Tensor)]
        if len(tensors) == len(example_output) and len(tensors) == len(out_arg_names):
            return _lower_multi_output(
                kernel,
                out_op,
                out_arg_names,
                example_output,
                tensor_args,
                non_tensor_args,
                kwargs,
            )
        log.debug(
            "Skipping %s: %d tensor outputs vs %d out args",
            kernel,
            len(tensors),
            len(out_arg_names),
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

    Creates a packed CustomOpMultiOutputNode that emits the .out() call,
    with AllocatingMultiOutput children (should_allocate=True) as output buffers.
    """
    # Validate all outputs are tensors on the same device before creating IR nodes.
    # This avoids leaking a registered packed node if we bail out midway.
    device: Optional[torch.device] = None
    for i, tensor_out in enumerate(example_output):
        if not isinstance(tensor_out, torch.Tensor):
            log.debug("Skipping %s: non-tensor in multi-output at index %d", kernel, i)
            return None
        if device is None:
            device = tensor_out.device
        elif tensor_out.device != device:
            log.debug(
                "Skipping %s: mixed devices at index 0=%s vs index %d=%s",
                kernel,
                device,
                i,
                tensor_out.device,
            )
            return None

    assert device is not None, "empty multi-output should have been caught earlier"
    packed = CustomOpMultiOutputNode(
        layout=ir.MultiOutputLayout(device=device),
        inputs=list(tensor_args),
        constant_args=list(non_tensor_args),
        out_op=out_op,
        out_arg_names=out_arg_names,
        op_overload=out_op,
        kwargs=kwargs,
    )

    outputs = []
    for i, tensor_out in enumerate(example_output):
        layout = ir.FixedLayout(
            device=tensor_out.device,
            dtype=tensor_out.dtype,
            size=[*tensor_out.shape],
            stride=[*tensor_out.stride()],
        )
        multi_out = AllocatingMultiOutput(
            layout=layout,
            input=packed,
            indices=[(type(example_output), i)],
        )
        outputs.append(multi_out)

    packed.output_nodes = outputs

    log.debug(
        "Lowered %s -> %s (multi-output, %d outputs, out args: %s)",
        kernel,
        out_op,
        len(outputs),
        out_arg_names,
    )
    container = tuple if isinstance(example_output, tuple) else list
    return container(outputs)  # type: ignore[return-value]


def _make_python_kernel_name(out_op: OpOverload) -> str:
    """Build fully-qualified kernel name, e.g. 'torch.ops.mylib.add_one.out'."""
    ns = out_op.namespace
    op_name = out_op._schema.name.split("::")[1]
    overload = out_op._overloadname
    return f"torch.ops.{ns}.{op_name}.{overload}"


def _codegen_input_args(node: ir.ExternKernel) -> list[str]:
    """Codegen positional + constant args as strings."""
    assert ir.is_node_sequence(node.inputs)
    args = [x.codegen_reference() for x in node.inputs]  # type: ignore[union-attr]
    for const in node.constant_args:
        args.append(V.graph.wrapper_code.val_to_arg_str(const))
    return args


def _codegen_kwargs(node: ir.ExternKernel, skip_names: set[str]) -> list[str]:
    """Codegen keyword args, skipping names in ``skip_names``."""
    result = []
    for k, v in node.kwargs.items():
        if k not in skip_names:
            result.append(f"{k}={V.graph.wrapper_code.val_to_arg_str(v)}")
    return result


class CustomOpExternKernelOut(ir.ExternKernelOut):
    """ExternKernelOut with flexible out-arg naming from schema.

    Uses the schema's actual out-arg name (e.g. "result") instead of
    hardcoded "out=". Inherits should_allocate()=True for buffer reuse.
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


class AllocatingMultiOutput(ir.MultiOutput):
    """MultiOutput child with should_allocate()=True.

    Pre-allocated by the parent's codegen. Skips base MultiOutput codegen
    (no tuple-indexing needed since buffers are allocated via AllocateLine).
    """

    def should_allocate(self) -> bool:
        return True

    def codegen(self, wrapper: Any) -> None:
        if not self.skip_size_stride_alignment_checks:
            self.codegen_size_asserts(wrapper)
            self.codegen_alignment_asserts(wrapper)


class CustomOpMultiOutputNode(ir.ExternKernel):
    """Packed node for multi-output .out() calls.

    Codegen emits .out(..., out1=buf1, out2=buf2) referencing pre-allocated
    AllocatingMultiOutput child buffers.
    """

    out_op: OpOverload
    out_arg_names: list[str]
    output_nodes: list[AllocatingMultiOutput]

    def __init__(
        self,
        layout: ir.MultiOutputLayout,
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
            None,
            layout,
            self.unwrap_storage(inputs),
            constant_args,
            kwargs or {},
            None,
            python_kernel_name,
            None,
            (),
            op_overload,
        )
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        self.out_op = out_op
        self.out_arg_names = out_arg_names or []
        self.output_nodes = []

    def should_allocate(self) -> bool:
        # The packed node itself doesn't allocate;
        # its AllocatingMultiOutput children do.
        return False

    def codegen(self, wrapper: Any) -> None:
        self.codegen_comment(wrapper)
        kernel_name = self.get_kernel_name()

        # NOTE: Allocate child output buffers BEFORE emitting the .out() call.
        # In the normal multi-output flow (FallbackKernel), the packed node
        # stores the result and children extract via indexing (buf1 = buf0[0]).
        # In our flow, the .out() call needs pre-allocated buffers as arguments,
        # so we explicitly trigger allocation here.
        for out_node in self.output_nodes:
            wrapper.codegen_allocation(out_node)

        args = _codegen_input_args(self)
        kwargs_list = _codegen_kwargs(self, skip_names=set(self.out_arg_names))

        all_args = [*args, *kwargs_list]
        for out_name, out_node in zip(self.out_arg_names, self.output_nodes):
            all_args.append(f"{out_name}={out_node.get_name()}")

        # Assign to packed node name so "del buf0" doesn't raise UnboundLocalError
        wrapper.writeline(f"{self.get_name()} = {kernel_name}({', '.join(all_args)})")
