"""
Inductor lowering support for custom op out-variant decomposition.

This module redirects functional custom ops from FallbackKernel (ExternKernelAlloc,
should_allocate=False) to ExternKernelOut (should_allocate=True) via the out-variant overload.

Key benefit: ExternKernelOut participates in Inductor's buffer reuse via
AllocateLine.plan() in memory_planning.py, so dead buffers with matching
(size, dtype, device) are automatically reused for the output allocation.
ExternKernelAlloc does NOT participate in this for now(see TODO at memory_planning.py:694).

Supports:
    - Single-output ops: functional(x) -> Tensor TO out_variant(x, *, out=buf) -> Tensor
    - Multi-output ops: functional(x) -> (Tensor, Tensor) TO out_variant(x, *, out1=buf1, out2=buf2) -> (Tensor, Tensor)
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

    Returns the IR node(s) if successful, None if not applicable.
    For single-output: returns a single CustomOpExternKernelOut node.
    For multi-output: returns a list of AllocatingMultiOutput nodes.

    The caller (fallback_handler in lowering.py) wraps the result with
    TensorBox.create() via pytree.tree_map(wrap_tensors, ...).
    """
    if not config.lower_custom_ops_to_out_variant:
        return None

    if not isinstance(kernel, OpOverload):
        return None

    from torch._library._out_variant import _is_functional, to_out_variant

    if not _is_functional(kernel._schema):
        return None

    out_op = to_out_variant(kernel)
    if out_op is None:
        return None

    out_schema = out_op._schema
    out_arg_names = [arg.name for arg in out_schema.arguments if arg.is_out]

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

    log.debug(
        "Lowered %s -> %s (out args: %s)",
        kernel,
        out_op,
        out_arg_names,
    )
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
    """Lower a multi-output functional op to a packed node + AllocatingMultiOutput children.

    Architecture:
        1. Create a CustomOpMultiOutputNode (packed) with MultiOutputLayout.
           This node's codegen emits the .out(..., out1=buf1, out2=buf2) call.
        2. Create AllocatingMultiOutput children (one per tensor output) with
           FixedLayout and should_allocate()=True — this makes each output
           buffer participate in AllocateLine.plan() buffer reuse.
        3. Return the list of AllocatingMultiOutput nodes. The caller wraps
           each in TensorBox.create().
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
                "Skipping %s: mixed devices in multi-output (index 0: %s, index %d: %s)",
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
    return type(example_output)(outputs)  # type: ignore[return-value]


def _make_python_kernel_name(out_op: OpOverload) -> str:
    """Build the fully-qualified python kernel name for an out-variant op."""
    ns = out_op.namespace
    op_name = out_op._schema.name.split("::")[1]
    overload = out_op._overloadname
    return f"torch.ops.{ns}.{op_name}.{overload}"


class CustomOpExternKernelOut(ir.ExternKernelOut):
    """
    An ExternKernelOut specialized for single-output custom ops with flexible
    out-arg naming.

    Unlike the base ExternKernelOut which hardcodes `out=` in codegen,
    this uses the actual out-arg name(s) from the op's schema (e.g., "result",
    "output", etc.).

    Inherits should_allocate() = True from ExternKernelOut, which enables
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
        assert out_op is not None
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
        args = self._codegen_input_args()
        kwargs_list = self._codegen_kwargs(skip_out=True)
        kernel_name = self.get_kernel_name()
        out_ref = self.codegen_reference()

        all_args = [*args, *kwargs_list]
        all_args.append(f"{self.out_arg_names[0]}={out_ref}")
        wrapper.writeline(f"{kernel_name}({', '.join(all_args)})")

    def _codegen_input_args(self) -> list[str]:
        assert ir.is_node_sequence(self.inputs)
        args = [x.codegen_reference() for x in self.inputs]  # type: ignore[union-attr]
        for const in self.constant_args:
            args.append(V.graph.wrapper_code.val_to_arg_str(const))
        return args

    def _codegen_kwargs(self, skip_out: bool = False) -> list[str]:
        result = []
        for k, v in self.kwargs.items():
            if skip_out and k in self.out_arg_names:
                continue
            result.append(f"{k}={V.graph.wrapper_code.val_to_arg_str(v)}")
        return result


class AllocatingMultiOutput(ir.MultiOutput):
    """
    MultiOutput with should_allocate()=True for buffer reuse.

    The parent CustomOpMultiOutputNode emits the .out() call; these children
    are pre-allocated destinations. We skip base codegen which would emit
    redundant "buf1 = buf0[0]" tuple-indexing (see MultiOutput.codegen).
    """

    def should_allocate(self) -> bool:
        return True

    def codegen(self, wrapper: Any) -> None:
        # Skip the base MultiOutput codegen which would emit "buf = packed[i]".
        # The parent CustomOpMultiOutputNode emits the .out() call instead.
        # The buffer was already allocated via AllocateLine (should_allocate=True).
        if not self.skip_size_stride_alignment_checks:
            # Still run size/stride/alignment asserts for safety.
            self.codegen_size_asserts(wrapper)
            self.codegen_alignment_asserts(wrapper)


class CustomOpMultiOutputNode(ir.ExternKernel):
    """
    Packed node for multi-output custom op out-variant calls.

    Has MultiOutputLayout. Its codegen emits the .out(..., out1=buf1, out2=buf2)
    call, referencing the pre-allocated buffers of its AllocatingMultiOutput children.

    The children (AllocatingMultiOutput) have should_allocate()=True, so their
    buffers participate in AllocateLine.plan() buffer reuse.
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
        assert out_op is not None
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
        # In the normal multi-output flow (FallbackKernel), the packed node codegen
        # stores the result and children extract via indexing (buf1 = buf0[0]).
        # In our flow, the .out() call needs pre-allocated buffers as arguments.
        # The scheduler processes children AFTER the packed parent, but we need
        # their allocations now. So we explicitly trigger allocation here.
        for out_node in self.output_nodes:
            wrapper.codegen_allocation(out_node)

        # Input args (tensor + constant)
        assert ir.is_node_sequence(self.inputs)
        args = [x.codegen_reference() for x in self.inputs]  # type: ignore[union-attr]
        for const in self.constant_args:
            args.append(V.graph.wrapper_code.val_to_arg_str(const))

        # Kwargs (excluding out args)
        for k, v in self.kwargs.items():
            if k not in self.out_arg_names:
                args.append(f"{k}={V.graph.wrapper_code.val_to_arg_str(v)}")

        # Out args: reference the pre-allocated buffers of children.
        for out_name, out_node in zip(self.out_arg_names, self.output_nodes):
            args.append(f"{out_name}={out_node.get_name()}")

        # Assign the result to this packed node's name (buf0 = op.out(...)).
        # The scheduler registers this node as a buffer and will emit "del buf0"
        # when it's no longer needed. Without this assignment, "del buf0" would
        # fail with an UnboundLocalError.
        wrapper.writeline(f"{self.get_name()} = {kernel_name}({', '.join(args)})")
