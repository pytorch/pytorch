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
    _FrozenPlan,
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
    if output is None or int(output.root_offset) < 0:
        raise ValueError("option output requires a valid physical root binding")
    if not generated and int(output.root_offset) != 0:
        raise ValueError("native option output requires an offset-zero root binding")
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
    if extent != elements:
        raise ValueError("native option output extent must match its physical root")
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


def replace_option_step(
    plan: _FrozenPlan,
    option: FrozenOptionStep,
    *,
    consumed_buffers: frozenset[str],
    step_outputs: tuple[tuple[str, ...], ...],
    step_positions: tuple[int, ...],
    placement: int,
) -> tuple[_FrozenPlan, tuple[tuple[str, ...], ...], tuple[int, ...]]:
    """Apply a proven IR claim only when it covers whole frozen DAG steps.

    ``step_outputs`` and ``step_positions`` are snapshotted from the verified
    KernelIR and its LoopsIR producer order. The option registry owns fusion;
    this conversion checks that prior pointwise fusion did not absorb an
    unclaimed operation or eliminate an input frontier the option still needs.
    Invalid physical claims raise, leaving the caller's baseline unchanged.
    """
    if len(plan.steps) != len(step_outputs) or len(plan.steps) != len(step_positions):
        raise ValueError("option step metadata does not match the frozen plan")
    if type(placement) is not int or placement < 0:
        raise ValueError("invalid option placement")
    if tuple(sorted(step_positions)) != step_positions:
        raise ValueError("option step positions must be topological")
    if option.generated_span is not None:
        produced = option.generated_span[0]
    elif option.result_layout is not None:
        produced = option.result_layout[0]
    else:
        raise ValueError("option has no produced root")
    if produced not in consumed_buffers:
        raise ValueError("option output is not part of its IR claim")
    removed: set[int] = set()
    for position, outputs in enumerate(step_outputs):
        output_set = set(outputs)
        if output_set & consumed_buffers:
            if not output_set <= consumed_buffers:
                raise ValueError("option only partially consumes a frozen step")
            removed.add(position)
    if not removed:
        raise ValueError("option has no executable baseline steps")

    def reads(kind: str, step: Any, outputs: tuple[str, ...]) -> set[str]:
        if kind == "extern":
            return {v.base for v in step.inputs}
        if kind == "kernel":
            return set(step.param_bases) - set(outputs)
        raise ValueError("unknown frozen step kind")

    available = set(plan.input_bases) | {b.name for b in plan.constant_bindings}
    if consumed_buffers & available:
        raise ValueError("option claim consumes a graph input")
    if any(out.base in consumed_buffers - {produced} for out in plan.out_plans):
        raise ValueError("option claim eliminates a graph output")
    for index, ((kind, step), outputs, position) in enumerate(
        zip(plan.steps, step_outputs, step_positions, strict=True)
    ):
        if index not in removed and position < placement:
            available.update(outputs)
        if index not in removed:
            needed = reads(kind, step, outputs)
            if needed & (consumed_buffers - {produced}):
                raise ValueError("a remaining step reads an eliminated option buffer")
            if produced in needed and position < placement:
                raise ValueError("option output is needed before its placement")
    if not reads(option.kind, option.step, (produced,)) <= available:
        raise ValueError("option frontier is unavailable at its placement")

    combined = [
        (position, kind, step, outputs)
        for index, ((kind, step), outputs, position) in enumerate(
            zip(plan.steps, step_outputs, step_positions, strict=True)
        )
        if index not in removed
    ]
    combined.append((placement, option.kind, option.step, (produced,)))
    # Insert before the next logical stage at the same boundary.
    combined.sort(key=lambda item: (item[0], item[2] is not option.step))
    steps = [(kind, step) for _, kind, step, _ in combined]
    names = [step.name for kind, step in steps if kind == "kernel"]
    sources = {
        name: source for name, source in plan.kernel_sources.items() if name in names
    }
    params = {
        name: value for name, value in plan.kernel_params.items() if name in names
    }
    grids = {name: value for name, value in plan.kernel_grids.items() if name in names}
    if option.kernel_source:
        if not isinstance(option.step, _KernelStep):
            raise ValueError("kernel source is attached to a non-kernel option")
        name, source = option.kernel_source
        if name in sources and (
            sources[name] != source or tuple(params[name]) != option.step.param_bases
        ):
            raise ValueError("selected kernel name collides with a different binding")
        sources[name] = source
        params[name] = list(option.step.param_bases)
        grids[name] = option.step.grid
    local_names = {
        step.target_dotted
        for kind, step in steps
        if kind == "extern" and step.target_is_local
    }
    native_sources = {
        name: source for name, source in plan.native_sources if name in local_names
    }
    if option.native_source:
        name, source = option.native_source
        if name in native_sources and native_sources[name] != source:
            raise ValueError("selected native name collides with a different source")
        native_sources[name] = source
    spans = {
        name: elements
        for name, elements in plan.gen_spans
        if name not in consumed_buffers
    }
    coverage = dict(
        zip((name for name, _ in plan.gen_spans), plan.span_full_coverage, strict=True)
    )
    if option.generated_span:
        name, elements = option.generated_span
        spans[name] = elements
        coverage[name] = False
    layouts = {
        name: (sizes, strides)
        for name, sizes, strides in plan.result_layouts
        if name not in consumed_buffers
    }
    if option.result_layout:
        name, sizes, strides = option.result_layout
        layouts[name] = (sizes, strides)
    updated = plan._replace(
        steps=steps,
        gen_spans=tuple(spans.items()),
        span_full_coverage=tuple(coverage[name] for name in spans),
        kernel_sources=sources,
        kernel_names=names,
        kernel_params=params,
        kernel_grids=grids,
        native_sources=tuple(native_sources.items()),
        result_layouts=tuple(
            (name, sizes, strides) for name, (sizes, strides) in layouts.items()
        ),
        num_generated_kernels=sum(kind == "kernel" for kind, _ in steps),
        num_extern_calls=sum(kind == "extern" for kind, _ in steps),
    )
    return (
        updated,
        tuple(outputs for _, _, _, outputs in combined),
        tuple(position for position, _, _, _ in combined),
    )
