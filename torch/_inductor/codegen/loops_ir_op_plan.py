"""Compile-time conversion of rendered op options to the existing wrapper ABI.

No selection, tuning, or runtime IR interpretation happens here. The caller
supplies an already validated/rendered option and its physical IR bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torch._inductor.codegen.loops_ir_op_options import (
    ArgRole,
    GeneratedKernel,
    NativeCall,
)
from torch._inductor.loops_ir_backend import (
    _ExternInput,
    _ExternResult,
    _ExternStep,
    _KernelStep,
    _TensorKey,
)


@dataclass(frozen=True)
class FrozenOptionStep:
    kind: str
    step: _KernelStep | _ExternStep
    kernel_source: tuple[str, str] | None
    native_source: tuple[str, str] | None
    generated_span: tuple[str, int] | None
    result_layout: tuple[str, tuple[int, ...], tuple[int, ...]] | None


def freeze_option_step(
    rendered: GeneratedKernel | NativeCall, variant: Any
) -> FrozenOptionStep:
    """Freeze pointer-relative bindings; retain no native IR objects.

    Regular operands point at their bound view. Fused extra reads and stores
    already carry root-relative offsets in the rendered source, so those
    pointers must point at the root, not at the bound view a second time.
    """
    generated = isinstance(rendered, GeneratedKernel)
    if not generated and not isinstance(rendered, NativeCall):
        raise TypeError("expected a rendered op option")
    output = variant.output_binding
    if output is None or int(output.root_offset) != 0:
        raise ValueError("option output must have an offset-zero root binding")
    root = str(output.root)
    elements = int(output.root_elements)
    if not root or elements < 0:
        raise ValueError("invalid option output binding")

    args = rendered.launch.args if generated else rendered.args
    views: list[_ExternInput] = []
    output_count = 0
    for position, arg in enumerate(args):
        if arg.dtype != torch.float32:
            raise ValueError("op options currently require f32 arguments")
        if type(arg.slot) is not int or arg.slot < 0:
            raise ValueError("invalid option argument slot")
        if type(arg.required_elements) is not int or arg.required_elements < 0:
            raise ValueError("invalid option required span")
        if arg.role == ArgRole.INPUT:
            binding = variant.input_bindings[arg.slot]
            if binding is None:
                raise ValueError("rendered input has no physical binding")
            rel = int(binding.root_offset)
        elif arg.role == ArgRole.EXTRA_INPUT:
            binding = variant.extra_bindings[arg.slot]
            rel = 0
        elif arg.role == ArgRole.OUTPUT:
            if not generated:
                raise ValueError("native option must return its output")
            binding = output
            rel = 0
            output_count += 1
        else:
            raise ValueError("unknown option argument role")
        sizes, strides = tuple(arg.sizes), tuple(arg.strides)
        span = int(binding.root_elements)
        if (
            len(sizes) != len(strides)
            or any(type(n) is not int or n < 0 for n in (*sizes, *strides))
            or rel < 0
            or span < 0
        ):
            raise ValueError("invalid option argument geometry")
        extent = (
            0 if 0 in sizes else 1 + sum((n - 1) * s for n, s in zip(sizes, strides))
        )
        if not str(binding.root):
            raise ValueError("option argument has no physical root")
        if rel + max(extent, arg.required_elements) > span:
            raise ValueError("option argument exceeds its physical root")
        views.append(
            _ExternInput(
                f"option_arg_{position}",
                str(binding.root),
                rel,
                sizes,
                strides,
                span,
            )
        )

    if generated:
        if output_count != 1 or args[-1].role != ArgRole.OUTPUT:
            raise ValueError("generated option requires one final output argument")
        launch = rendered.launch
        grid = tuple(launch.grid)
        if len(grid) != 3 or any(type(n) is not int or n < 0 for n in grid):
            raise ValueError("invalid option grid")
        kernel_step = _KernelStep(
            rendered.kernel_name,
            tuple(v.base for v in views),
            grid,
            launch.num_warps,
            launch.enable_fp_fusion,
            not launch.disable_ftz,
            any(n == 0 for n in grid),
            None,
            tuple(views),
        )
        return FrozenOptionStep(
            "kernel",
            kernel_step,
            (rendered.kernel_name, rendered.source),
            None,
            (root, elements),
            None,
        )

    layout = variant.signature.output.layout
    sizes, strides = tuple(layout.sizes), tuple(layout.strides)
    if len(sizes) != len(strides) or any(
        type(n) is not int or n < 0 for n in (*sizes, *strides)
    ):
        raise ValueError("invalid native option output geometry")
    extent = 0 if 0 in sizes else 1 + sum((n - 1) * s for n, s in zip(sizes, strides))
    if extent > elements:
        raise ValueError("native option output exceeds its physical root")
    epilogue = variant.fused_loops.epilogue
    if epilogue is not None and int(epilogue.store_offset) != 0:
        raise ValueError("native option cannot return a shifted output root")
    native_step = _ExternStep(
        rendered.function_name,
        None,
        rendered.function_name,
        tuple(views),
        tuple(_TensorKey(v.key) for v in views),
        (),
        (_ExternResult(root, torch.float32, sizes, strides),),
        "tensor",
        True,
    )
    return FrozenOptionStep(
        "extern",
        native_step,
        None,
        (rendered.function_name, rendered.source),
        None,
        (root, sizes, strides),
    )
