"""Format and emit opt-in FlexGEMM compilation diagnostics.

The ``flex_gemm`` logger follows lowering from the captured FX body through
semantic analysis, buffer planning, generated CuTeDSL, and kernel selection.
Each phase is also available as a structured trace artifact.
"""

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.kernel.gemm_epilogue import (
    NormalizedGetItem,
    NormalizedPrepareSoftmax,
    NormalizedReduction,
    NormalizedSelect,
    NormalizedSplit,
    NormalizedSqueeze,
    NormalizedUnsupportedReduction,
    NormalizedView,
)
from torch._logging import LazyString, trace_structured


if TYPE_CHECKING:
    from torch._inductor import ir
    from torch._inductor.heuristics.template.flex_gemm import GemmConfigKey
    from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
        FlexGemmEpilogueAnalysis,
    )


flex_gemm_log = logging.getLogger(__name__)


def log_flex_gemm_artifact(
    name: str,
    payload_fn: Callable[[], str],
    *,
    lowering_name: str | None = None,
    verbose: bool = False,
) -> None:
    """Emit one lazily rendered local and structured FlexGEMM phase."""
    heading = "FLEXGEMM LOWERING"
    if lowering_name is not None:
        heading += f" [{lowering_name}]"
    flex_gemm_log.log(
        logging.DEBUG if verbose else logging.INFO,
        "%s\n ===== %s =====\n%s",
        heading,
        name.replace("_", " ").upper(),
        LazyString(payload_fn),
    )
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": f"flex_gemm_{name}",
            "encoding": "string",
            "verbosity": "verbose" if verbose else "concise",
            **({} if lowering_name is None else {"lowering_name": lowering_name}),
        },
        payload_fn=payload_fn,
    )


def _append_items(lines: list[str], label: str, items: Iterable[str]) -> None:
    """Append an indented report section, including an explicit empty marker."""
    values = tuple(items)
    lines.append(f"{label}:")
    lines.extend(f"  {value}" for value in values or ("(none)",))


def _format_ir_tensor(name: str, node: "ir.IRNode") -> str:
    """Format the tensor contract visible to Inductor lowering."""
    stride = node.maybe_get_stride()
    return (
        f"{name}: shape={tuple(node.get_size())}, "
        f"stride={'unrealized' if stride is None else tuple(stride)}, "
        f"dtype={node.get_dtype()}, device={node.get_device_or_error()}"
    )


def format_flex_gemm_problem(
    graph_module: torch.fx.GraphModule,
    gemm_op: torch._ops.OpOverload,
    gemm_inputs: Sequence[tuple[str, "ir.IRNode"]],
    captures: Sequence[tuple[str, "ir.IRNode"]],
    *,
    alpha: float,
    beta: float,
    tuned: bool,
    fast_math: bool,
    explicit_config: dict[str, Any] | None,
) -> str:
    """Render the inputs and captured body entering FlexGEMM analysis."""
    lines = [
        f"gemm_op: {gemm_op}",
        "kernel_options:",
        "  backend: QUACK",
        f"  tuned: {tuned}",
        f"  fast_math: {fast_math}",
        f"  config: {explicit_config!r}",
        f"alpha: {alpha}",
        f"beta: {beta}",
    ]
    _append_items(
        lines,
        "gemm_inputs",
        (_format_ir_tensor(name, node) for name, node in gemm_inputs),
    )
    _append_items(
        lines,
        "captures",
        (_format_ir_tensor(name, node) for name, node in captures),
    )
    lines.extend(
        (
            "body:",
            graph_module.print_readable(
                print_output=False,
                include_stride=True,
                include_device=True,
            ).strip(),
        )
    )
    return "\n".join(lines)


def _format_fx_tensor(node: torch.fx.Node) -> str:
    """Format an FX node using its captured fake-tensor metadata."""
    meta = node.meta.get("val")
    if not isinstance(meta, torch.Tensor):
        return node.name
    return (
        f"{node.name}: shape={tuple(meta.shape)}, stride={tuple(meta.stride())}, "
        f"dtype={meta.dtype}"
    )


def _format_geometry(geometry: Any) -> str:
    """Format grouped GEMM geometry in logical M/N terms."""
    axis = "M" if geometry.axis == 0 else "N"
    return f"axis={axis}, group={geometry.group}"


def _format_output_contraction(contraction: Any | None) -> str:
    """Format the logical contraction applied to the main output."""
    if contraction is None:
        return "none"
    layout = "chunked" if contraction.chunked else "interleaved"
    return f"N-axis, group={contraction.group}, layout={layout}"


def _format_normalized_dataflow(node: torch.fx.Node, normalized: Any) -> str:
    """Render one normalized FX operation as compact dataflow."""
    match normalized:
        case NormalizedView(shape=shape):
            operation = f"view(shape={shape})"
        case NormalizedReduction(
            dim=dim,
            keepdim=keepdim,
            reduction_type=reduction_type,
        ):
            operation = f"{reduction_type}(dim={dim}, keepdim={keepdim})"
        case NormalizedPrepareSoftmax(dim=dim):
            operation = f"prepare_softmax(dim={dim})"
        case NormalizedSqueeze():
            operation = "squeeze"
        case NormalizedGetItem(index=index):
            operation = f"getitem(index={index})"
        case NormalizedSplit(split_size=split_size, dim=dim):
            operation = f"split(size={split_size}, dim={dim})"
        case NormalizedSelect(dim=dim, index=index):
            operation = f"select(dim={dim}, index={index})"
        case NormalizedUnsupportedReduction():
            operation = f"unsupported_reduction({node.target})"
        case _:
            return repr(normalized)
    return f"{normalized.source.name} -> {operation}"


def format_flex_gemm_analysis(analysis: "FlexGemmEpilogueAnalysis") -> str:
    """Render the semantic decisions a FlexGEMM developer acts on first."""
    outputs = analysis.outputs
    lines = [
        "outputs:",
        f"  main: {_format_fx_tensor(outputs.output)}",
        f"  output_contraction: {_format_output_contraction(outputs.output_contraction)}",
    ]
    if outputs.aux_outputs:
        lines.append("  auxiliary:")
        lines.extend(
            f"    {_format_fx_tensor(output)}" for output in outputs.aux_outputs
        )
    else:
        lines.append("  auxiliary: (none)")

    lines.append("")
    if outputs.local_reduce is None:
        lines.append("local_reduction: none")
    else:
        local_reduce = outputs.local_reduce
        store = local_reduce.store
        consumers = []
        if local_reduce.feeds_main:
            consumers.append("main")
        if store is not None:
            consumers.append("returned")
        normalized = analysis.local_reduce.graph.normalized_nodes.get(
            local_reduce.match.value_node
        )
        dataflow = (
            str(local_reduce.match.value_node.target)
            if normalized is None
            else _format_normalized_dataflow(local_reduce.match.value_node, normalized)
        )
        lines.extend(
            (
                "local_reduction:",
                f"  value: {local_reduce.match.value_node.name}",
                f"  dataflow: {dataflow}",
                f"  geometry: {_format_geometry(local_reduce.match.geometry)}",
                f"  consumers: {' + '.join(consumers)}",
            )
        )
        if store is not None:
            lines.extend(
                (
                    f"  returned_as: {store.node.name}",
                    "  output_layout: dense",
                )
            )

    lines.append("")
    _append_items(
        lines,
        "config_constraints",
        map(_format_geometry, analysis.required_geometries),
    )
    return "\n".join(lines)


def format_flex_gemm_analysis_details(
    analysis: "FlexGemmEpilogueAnalysis",
) -> str:
    """Render normalized nodes and recognizer records for deep debugging."""
    lines: list[str] = []
    for label, values in (
        ("normalized_nodes", analysis.local_reduce.graph.normalized_nodes),
        ("grouped_layouts", analysis.local_reduce.grouped_tensors),
        ("local_reduce_matches", analysis.local_reduce.matches),
        (
            "output_contraction_select_indices",
            analysis.output_contraction_select_indices,
        ),
    ):
        _append_items(
            lines,
            label,
            (f"{node.name}: {value!r}" for node, value in values.items()),
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_tensor_meta(meta: torch.Tensor) -> str:
    """Format output metadata used for allocation and ABI planning."""
    return (
        f"shape={tuple(meta.shape)}, stride={tuple(meta.stride())}, dtype={meta.dtype}"
    )


def format_flex_gemm_lowering_plan(
    logical_output_size: Sequence[Any],
    physical_output_size: Sequence[Any],
    output_dtype: torch.dtype,
    capture_kinds: Sequence[tuple[str, str]],
    aux_metas: Sequence[torch.Tensor],
    local_reduce_metas: Sequence[torch.Tensor],
) -> str:
    """Render buffer allocation and runtime-ABI decisions."""
    lines = [
        "output_storage:",
        f"  logical: shape={tuple(logical_output_size)}, dtype={output_dtype}",
        f"  physical: shape={tuple(physical_output_size)}",
        "",
    ]
    _append_items(
        lines,
        "captures",
        (f"{name} -> {kind}" for name, kind in capture_kinds),
    )
    lines.append("")
    _append_items(lines, "auxiliary_storage", map(_format_tensor_meta, aux_metas))
    lines.append("")
    if local_reduce_metas:
        _append_items(
            lines,
            "local_reduction_storage",
            (
                f"output {index}: {_format_tensor_meta(meta)}"
                for index, meta in enumerate(local_reduce_metas)
            ),
        )
        lines.append("  layout: dense")
    else:
        lines.append("local_reduction_storage: (none)")
    return "\n".join(lines)


def format_flex_gemm_config_key(config_key: "GemmConfigKey") -> str:
    """Render every config field so new GemmConfig fields remain visible."""
    return "\n".join(
        f"{name}: {'auto' if value is None else repr(value)}"
        for name, value in config_key
    )


def format_flex_gemm_config_candidates(
    config_keys: Sequence["GemmConfigKey"],
) -> str:
    """Render every lowering-approved config for verbose diagnostics."""
    lines: list[str] = []
    for index, config_key in enumerate(config_keys):
        _append_items(
            lines,
            f"candidate {index}",
            format_flex_gemm_config_key(config_key).splitlines(),
        )
        lines.append("")
    return "\n".join(lines).rstrip() if lines else "(none)"


def format_flex_gemm_selection(
    choice: "ir.ChoiceCaller | None",
    config_key: "GemmConfigKey | None",
    *,
    candidate_count: int,
    tuned: bool,
) -> str:
    """Render the search summary and selected FlexGEMM template."""
    lines = [
        "search:",
        f"  mode: {'autotuned' if tuned else 'fixed'}",
        f"  lowering_approved_candidates: {candidate_count}",
        "",
        "selected:",
    ]
    if choice is None:
        lines.append("  deferred to a multi-template buffer")
    else:
        lines.append(f"  template: {choice.name}")
        lines.append("  config:")
        lines.extend(
            "    " + line
            for line in (
                ("(unavailable)",)
                if config_key is None
                else format_flex_gemm_config_key(config_key).splitlines()
            )
        )
    lines.extend(
        (
            "",
            "more_detail_commands:",
            '  analysis/codegen: TORCH_LOGS="+flex_gemm"',
            '  autotune timings: TORCH_LOGS="flex_gemm,autotuning"',
            '  generated kernel: TORCH_LOGS="flex_gemm,kernel_code"',
            '  final wrapper: TORCH_LOGS="flex_gemm,output_code"',
            '  candidate failures: TORCH_LOGS="+inductor,flex_gemm"',
        )
    )
    return "\n".join(lines)
