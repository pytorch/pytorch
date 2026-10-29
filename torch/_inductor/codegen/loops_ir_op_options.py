"""Per-operation implementation-option codegen for the C++ LoopsIR path.

The typed registry, configuration schemas, eligibility, candidate manifests
and fusion contracts live in C++
(``torch/csrc/inductor/experimental/loops_ir/op_options``). This module only
loads that isolated extension, freezes the numerical/device context once,
and maps each option's ``CodegenStrategy`` to a codegen function that
consumes ``(config, signature, fused_loops, context)``. Every codegen call
re-validates through the registry before emitting; a rejected selection
raises ``OptionRejected`` and never falls back to another implementation.
"""

from __future__ import annotations

import ctypes
import dataclasses
import enum
import functools
import os
import struct
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Sequence


__all__ = [
    "ArgRole",
    "CompiledKernel",
    "CompiledMetadata",
    "FrozenCodegenContext",
    "GeneratedKernel",
    "KernelArg",
    "LaunchSchema",
    "NativeCall",
    "OP_OPTIONS_NATIVE_SOURCES",
    "codegen",
    "compile_generated_kernel",
    "decode_op_signatures",
    "enumerate_op_candidates",
    "form_fused_op_regions",
    "freeze_codegen_context",
    "load_native_call",
    "load_op_options_native",
    "tensor_layout",
    "tensor_operand",
]

_LOOPS_IR_DIR = (
    Path(__file__).resolve().parents[2] / "csrc/inductor/experimental/loops_ir"
)

# Exact native source list (core ir.cpp + the op_options units). No globs.
OP_OPTIONS_NATIVE_SOURCES = (
    "ir.cpp",
    "op_options/op_types.cpp",
    "op_options/op_config.cpp",
    "op_options/fused_region.cpp",
    "op_options/option_registry.cpp",
    "op_options/gemm_options.cpp",
    "op_options/reduction_options.cpp",
    "op_options/program_facts.cpp",
    "op_options/op_decode.cpp",
    "op_options/op_fusion_regions.cpp",
    "op_options/op_candidates.cpp",
    "op_options/op_options_bindings.cpp",
)


@functools.lru_cache(None)
def load_op_options_native() -> Any:
    from torch.utils.cpp_extension import load

    missing = [
        n for n in OP_OPTIONS_NATIVE_SOURCES if not (_LOOPS_IR_DIR / n).is_file()
    ]
    if missing:
        raise RuntimeError("op_options native sources missing: " + ", ".join(missing))
    return load(
        name="inductor_loops_ir_op_options",
        sources=[str(_LOOPS_IR_DIR / n) for n in OP_OPTIONS_NATIVE_SOURCES],
        extra_cflags=["-O1", "-g0", "-std=c++17"],
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Frozen context and typed ABI helpers.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FrozenCodegenContext:
    """C++ ``CodegenContext`` plus the device properties embedded in
    generated ``triton_meta``. Built once; codegen never rereads globals."""

    native: Any
    device_index: int
    device_properties: Any

    def __post_init__(self) -> None:
        if int(self.native.device_index) != int(self.device_index):
            raise ValueError(
                "FrozenCodegenContext.device_index differs from the native context (cache key)"
            )


def _addmm_lt_policy(ext: Any) -> Any:
    # Mirrors isGloballyDisabledAddmmCudaLt (Blas.cpp). Eager caches the env
    # var on first addmm, so it must not change after freezing. ROCm dispatch
    # is not modelled.
    if torch.version.hip is not None:
        return ext.AddmmLtPolicy.UNKNOWN
    if os.environ.get("DISABLE_ADDMM_CUDA_LT") == "1":
        return ext.AddmmLtPolicy.CUDA_DISABLED
    return ext.AddmmLtPolicy.CUDA_DEFAULT


def freeze_codegen_context(device: torch.device, numerics: str) -> FrozenCodegenContext:
    """Snapshot the stock effective numerical settings for ``numerics``
    ('strict_eager' or 'inductor'). Strict pins fp-fusion off / FTZ off."""
    from torch._inductor import config as inductor_config
    from torch._inductor.runtime.hints import DeviceProperties

    ext = load_op_options_native()
    if numerics not in ("strict_eager", "inductor"):
        raise ValueError(f"unknown numerics policy {numerics!r}")
    if device.type != "cuda" or device.index is None:
        raise ValueError("freeze_codegen_context requires an indexed CUDA device")
    strict = numerics == "strict_eager"
    props = torch.cuda.get_device_properties(device)
    precision = (
        ext.MatmulPrecision.TF32
        if torch.backends.cuda.matmul.fp32_precision == "tf32"
        else ext.MatmulPrecision.IEEE
    )
    native = ext.CodegenContext(
        numerics=ext.NumericsPolicy.STRICT_EAGER
        if strict
        else ext.NumericsPolicy.INDUCTOR,
        device=ext.DeviceKind.CUDA,
        compute_capability=props.major * 10 + props.minor,
        max_shared_memory_bytes=int(props.shared_memory_per_block_optin),
        stock_fp32_matmul_precision=precision,
        enable_fp_fusion=False
        if strict
        else not inductor_config.emulate_precision_casts,
        disable_ftz=True
        if strict
        else bool(inductor_config.eager_numerics.disable_ftz),
        division_rounding=bool(inductor_config.eager_numerics.division_rounding),
        device_index=device.index,
        addmm_lt_policy=_addmm_lt_policy(ext),
    )
    native.validate()
    return FrozenCodegenContext(native, device.index, DeviceProperties.create(device))


_TORCH_TO_STORAGE = {
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.bool: "BOOL",
}


def tensor_layout(t: torch.Tensor) -> Any:
    ext = load_op_options_native()
    dtype = getattr(ext.StorageDtype, _TORCH_TO_STORAGE[t.dtype])
    return ext.TensorLayout(dtype, list(t.shape), list(t.stride()))


def tensor_operand(name: str, t: torch.Tensor) -> Any:
    return load_op_options_native().TensorOperand(name, tensor_layout(t))


class ArgRole(enum.Enum):
    INPUT = "input"
    EXTRA_INPUT = "extra_input"
    OUTPUT = "output"


@dataclasses.dataclass(frozen=True)
class KernelArg:
    """data_ptr-relative tensor ABI. ``required_elements``: elements the
    kernel may touch from the data pointer (output: epilogue store offset +
    layout extent); launch enforces it against the tensor's storage."""

    role: ArgRole
    name: str
    slot: int
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    required_elements: int = 0


@dataclasses.dataclass(frozen=True)
class LaunchSchema:
    grid: tuple[int, int, int]
    num_warps: int
    num_stages: int
    enable_fp_fusion: bool
    disable_ftz: bool
    args: tuple[KernelArg, ...]


@dataclasses.dataclass(frozen=True)
class GeneratedKernel:
    option_id: str
    option_version: int
    cache_key: str
    config: str
    kernel_name: str
    source: str
    launch: LaunchSchema


@dataclasses.dataclass(frozen=True)
class NativeCall:
    option_id: str
    option_version: int
    cache_key: str
    target: str
    function_name: str
    source: str
    args: tuple[KernelArg, ...]


class _AccessMapLike(Protocol):
    offset: int
    coeffs: Sequence[int]


class _RegionOpLike(Protocol):
    kind: Any
    id: int
    lhs: int
    rhs: int
    extra: int
    access: _AccessMapLike
    value: float


class _LoweredRegionLike(Protocol):
    domain: Sequence[int]
    ops: Sequence[_RegionOpLike]
    store_value: int


def _validated(
    option: Any,
    config: Any,
    signature: Any,
    fused_loops: Any,
    context: FrozenCodegenContext,
) -> Any:
    sel = (
        load_op_options_native()
        .default_registry()
        .validate(option.id, signature, config, fused_loops, context.native)
    )
    if sel.strategy != option.strategy:
        raise RuntimeError("validated strategy does not match the option")
    return sel


def _reads_input(sel: Any, slot: int) -> bool:
    region = sel.fusion.prologue_for(slot)
    return region is None or bool(region.reads_self)


def _kernel_args(sel: Any) -> list[KernelArg]:
    """Launch-order ABI: read input slots, extra inputs, then the output.
    An input slot fully computed by a prologue is not an argument."""
    args = [
        KernelArg(
            ArgRole.INPUT, op.name, i, tuple(op.layout.sizes), tuple(op.layout.strides)
        )
        for i, op in enumerate(sel.signature.inputs)
        if _reads_input(sel, i)
    ]
    args += [
        KernelArg(
            ArgRole.EXTRA_INPUT,
            e.name,
            i,
            tuple(e.layout.sizes),
            tuple(e.layout.strides),
        )
        for i, e in enumerate(sel.fusion.extra_inputs)
    ]
    out = sel.signature.output
    required = (
        _store_offset(sel) + int(out.layout.extent()) if int(out.layout.numel()) else 0
    )
    args.append(
        KernelArg(
            ArgRole.OUTPUT,
            out.name,
            0,
            tuple(out.layout.sizes),
            tuple(out.layout.strides),
            required_elements=required,
        )
    )
    return args


def _ptr_name(arg: KernelArg) -> str:
    prefix = {ArgRole.INPUT: "in", ArgRole.EXTRA_INPUT: "x", ArgRole.OUTPUT: "out"}[
        arg.role
    ]
    return f"{prefix}{arg.slot}_ptr"


def _input_ptr(slot: int) -> str:
    return f"in{slot}_ptr"


def _store_offset(sel: Any) -> int:
    epi = sel.fusion.epilogue
    return 0 if epi is None else int(epi.store_offset)


def _kernel_name(sel: Any) -> str:
    return f"{sel.option_id}_{sel.cache_key}"


# ---------------------------------------------------------------------------
# Triton rendering helpers.
# ---------------------------------------------------------------------------


def _f32_scalar(value: float) -> str:
    from torch._inductor.codegen.loops_ir_triton import _f32_scalar_expr

    return _f32_scalar_expr(ctypes.c_float(value).value)


def _affine(offset: int, coeffs: Sequence[int], coords: Sequence[str]) -> str:
    # Zero-coefficient terms are kept so offsets always span the full tile
    # (Triton stores do not broadcast the pointer to the value shape).
    terms = [f"{int(c)} * {coords[d]}" for d, c in enumerate(coeffs)]
    if offset:
        terms.insert(0, str(int(offset)))
    return "(" + " + ".join(terms) + ")" if terms else "0"


def _self_affine(
    sizes: Sequence[int], strides: Sequence[int], coords: Sequence[str]
) -> str:
    coeffs = [s if n > 1 else 0 for n, s in zip(sizes, strides)]
    return _affine(0, coeffs, coords)


def _render_region(
    region: _LoweredRegionLike,
    *,
    self_value: str,
    coords: Sequence[str],
    extra_ptrs: Sequence[str],
    mask: str,
    prefix: str,
    indent: str,
    lines: list[str],
) -> str:
    """Emit the validated region ops; returns the stored value name."""
    kinds = load_op_options_native().RegionOpKind
    values: dict[int, str] = {}
    for op in region.ops:
        name = f"{prefix}{op.id}"
        if op.kind == kinds.LOAD_SELF:
            expr = self_value
        elif op.kind == kinds.LOAD_EXTRA:
            off = _affine(op.access.offset, op.access.coeffs, coords)
            expr = f"tl.load({extra_ptrs[op.extra]} + {off}, mask={mask}, other=0.0)"
        elif op.kind == kinds.CONSTANT:
            expr = _f32_scalar(op.value)
        elif op.kind == kinds.ADD:
            expr = f"{values[op.lhs]} + {values[op.rhs]}"
        elif op.kind == kinds.MUL:
            expr = f"{values[op.lhs]} * {values[op.rhs]}"
        elif op.kind == kinds.RELU:
            expr = f"tl.maximum(tl.full([], 0, tl.int32), {values[op.lhs]}, tl.PropagateNan.ALL)"
        else:
            raise ValueError(f"unhandled region op {op.kind}")
        lines.append(f"{indent}{name} = {expr}")
        values[op.id] = name
    return values[region.store_value]


def _kernel_header(
    sel: Any,
    args: list[KernelArg],
    warps: int,
    stages: int,
    context: FrozenCodegenContext,
) -> list[str]:
    from torch._inductor.runtime.triton_heuristics import FixedGrid

    name = _kernel_name(sel)
    ptrs = [_ptr_name(a) for a in args]
    triton_meta = {
        "signature": dict.fromkeys(ptrs, "*fp32"),
        "device": context.device_properties,
        "constants": {},
        "configs": [{}],
        "enable_fp_fusion": bool(context.native.enable_fp_fusion),
        "disable_ftz": bool(context.native.disable_ftz),
    }
    inductor_meta = {
        **FixedGrid.setup_grid_as_args(),
        "kernel_name": name,
        "mutated_arg_names": [
            p for a, p in zip(args, ptrs) if a.role == ArgRole.OUTPUT
        ],
    }
    return [
        "import triton",
        "import triton.language as tl",
        "from torch._inductor.runtime import triton_heuristics",
        "from torch._inductor.runtime.hints import DeviceProperties",
        "",
        "@triton_heuristics.fixed_config(",
        f"    {{'num_warps': {warps}, 'num_stages': {stages}}}, filename=__file__,",
        f"    triton_meta={triton_meta!r},",
        f"    inductor_meta={inductor_meta!r})",
        "@triton.jit",
        f"def {name}({', '.join(ptrs)}):",
        f'    """{sel.option_id} v{sel.option_version} {sel.config!r} key={sel.cache_key}"""',
    ]


def _launch(
    sel: Any, args: list[KernelArg], grid: tuple[int, int, int], warps: int, stages: int
) -> LaunchSchema:
    return LaunchSchema(
        grid=grid,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=bool(sel.context.enable_fp_fusion),
        disable_ftz=bool(sel.context.disable_ftz),
        args=tuple(args),
    )


def _finish(sel: Any, lines: list[str], launch: LaunchSchema) -> GeneratedKernel:
    return GeneratedKernel(
        option_id=sel.option_id,
        option_version=int(sel.option_version),
        cache_key=sel.cache_key,
        config=sel.config.canonical(),
        kernel_name=_kernel_name(sel),
        source="\n".join(lines) + "\n",
        launch=launch,
    )


# ---------------------------------------------------------------------------
# Generated tiled GEMM (mm / addmm / bmm).
# ---------------------------------------------------------------------------


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _gemm_program_ids(
    cfg: Any, grid_m: int, grid_n: int, batched: bool, lines: list[str]
) -> None:
    g = int(cfg.group_m)
    lines += [
        "    pid = tl.program_id(0)",
        f"    width = {g * grid_n}",
        "    group_id = pid // width",
        f"    group_size = tl.minimum({grid_m} - group_id * {g}, {g})",
        f"    pid_m = group_id * {g} + (pid % group_size)",
        "    pid_n = (pid % width) // group_size",
        f"    rm = pid_m.to(tl.int64) * {cfg.block_m} + tl.arange(0, {cfg.block_m}).to(tl.int64)",
        f"    rn = pid_n.to(tl.int64) * {cfg.block_n} + tl.arange(0, {cfg.block_n}).to(tl.int64)",
        f"    rk = tl.arange(0, {cfg.block_k}).to(tl.int64)",
    ]
    if batched:
        lines.append("    pid_b = tl.program_id(1).to(tl.int64)")


def _gemm_operand(
    sel: Any,
    slot: int,
    coords: list[str],
    mask: str,
    ptr: str,
    extra_ptrs: list[str],
    var: str,
    lines: list[str],
) -> None:
    layout = sel.signature.inputs[slot].layout
    region = sel.fusion.prologue_for(slot)
    if region is None or region.reads_self:
        off = _self_affine(layout.sizes, layout.strides, coords)
        lines.append(f"        {var} = tl.load({ptr} + {off}, mask={mask}, other=0.0)")
    if region is None:
        return
    value = _render_region(
        region,
        self_value=var,
        coords=coords,
        extra_ptrs=extra_ptrs,
        mask=mask,
        prefix=f"{var}_p",
        indent="        ",
        lines=lines,
    )
    # Padding lanes must stay zero inside the dot after the prologue.
    lines.append(f"        {var} = tl.where({mask}, {value}, 0.0)")


def _addmm_bias(sel: Any, ptr: str, mask: str, lines: list[str]) -> None:
    """Render the C++-derived eager addmm formula (``sel.addmm``)."""
    plan, attrs = sel.addmm, sel.signature.matmul
    if plan.scale_product:
        lines.append(f"    acc = acc * {_f32_scalar(attrs.alpha)}")
    if not plan.load_bias:
        return
    bias = sel.signature.inputs[0].layout
    coeffs = [0, 0]
    for d, (n, s) in enumerate(zip(bias.sizes, bias.strides)):
        coeffs[2 - len(bias.sizes) + d] = s if n > 1 else 0
    off = _affine(0, coeffs, ["rm[:, None]", "rn[None, :]"])
    lines.append(f"    bias = tl.load({ptr} + {off}, mask={mask}, other=0.0)")
    if plan.scale_bias:
        lines.append(f"    bias = bias * {_f32_scalar(attrs.beta)}")
    if plan.read_operands:
        lines.append("    acc = acc + bias")
    elif plan.add_positive_zero:
        lines.append(f"    acc = bias + {_f32_scalar(0.0)}")
    else:
        lines.append("    acc = bias")


def _gemm_codegen(
    option: Any,
    config: Any,
    signature: Any,
    fused_loops: Any,
    context: FrozenCodegenContext,
) -> GeneratedKernel:
    ext = load_op_options_native()
    sel = _validated(option, config, signature, fused_loops, context)
    cfg, sig = sel.config, sel.signature
    dims = sig.gemm_dims
    batched = sig.kind == ext.OpKind.BMM
    if dims.batch > 65535:
        raise ValueError("bmm batch exceeds the CUDA grid y limit")
    grid_m, grid_n = _cdiv(dims.m, cfg.block_m), _cdiv(dims.n, cfg.block_n)
    args = _kernel_args(sel)
    ptrs = [_ptr_name(a) for a in args]
    extra_ptrs = [p for a, p in zip(args, ptrs) if a.role == ArgRole.EXTRA_INPUT]
    ia, ib = sig.gemm_operand_a, sig.gemm_operand_b
    lead = ["pid_b"] if batched else []
    precision = "tf32" if cfg.matrix_precision == ext.MatmulPrecision.TF32 else "ieee"

    lines = _kernel_header(sel, args, cfg.num_warps, cfg.num_stages, context)
    _gemm_program_ids(cfg, grid_m, grid_n, batched, lines)
    lines += [
        f"    mask_m = rm < {dims.m}",
        f"    mask_n = rn < {dims.n}",
        f"    acc = tl.zeros(({cfg.block_m}, {cfg.block_n}), dtype=tl.float32)",
    ]
    # Eager BLAS never reads A/B on the alpha == 0 gemm path: no loads,
    # prologues or dot (NaN/Inf in A/B must not propagate).
    if sig.kind != ext.OpKind.ADDMM or sel.addmm.read_operands:
        lines += [
            f"    for k0 in range(0, {dims.k}, {cfg.block_k}):",
            "        kk = k0 + rk",
            f"        mask_k = kk < {dims.k}",
        ]
        _gemm_operand(
            sel,
            ia,
            lead + ["rm[:, None]", "kk[None, :]"],
            "mask_m[:, None] & mask_k[None, :]",
            _input_ptr(ia),
            extra_ptrs,
            "a",
            lines,
        )
        _gemm_operand(
            sel,
            ib,
            lead + ["kk[:, None]", "rn[None, :]"],
            "mask_k[:, None] & mask_n[None, :]",
            _input_ptr(ib),
            extra_ptrs,
            "b",
            lines,
        )
        lines.append(
            f"        acc = tl.dot(a, b, acc, input_precision={precision!r}, out_dtype=tl.float32)"
        )
    out_mask = "mask_m[:, None] & mask_n[None, :]"
    if sig.kind == ext.OpKind.ADDMM:
        _addmm_bias(sel, _input_ptr(0), out_mask, lines)
    out_coords = lead + ["rm[:, None]", "rn[None, :]"]
    value = "acc"
    if sel.fusion.epilogue is not None:
        value = _render_region(
            sel.fusion.epilogue,
            self_value="acc",
            coords=out_coords,
            extra_ptrs=extra_ptrs,
            mask=out_mask,
            prefix="e",
            indent="    ",
            lines=lines,
        )
    out = sig.output.layout
    out_off = (
        f"{_store_offset(sel)} + {_self_affine(out.sizes, out.strides, out_coords)}"
    )
    lines.append(f"    tl.store({ptrs[-1]} + {out_off}, {value}, mask={out_mask})")
    grid = (grid_m * grid_n, dims.batch, 1)
    return _finish(sel, lines, _launch(sel, args, grid, cfg.num_warps, cfg.num_stages))


# ---------------------------------------------------------------------------
# Generated single-axis row reduction (sum / mean).
# ---------------------------------------------------------------------------


def _row_coords(
    sizes: Sequence[int], keep: Sequence[int], lines: list[str]
) -> dict[int, str]:
    """Decompose the flat row index into coordinates of the kept dims."""
    coords: dict[int, str] = {}
    inner = 1
    for d in reversed(keep):
        coords[d] = f"c{d}"
        lines.append(f"    c{d} = (xindex // {inner}) % {max(sizes[d], 1)}")
        inner *= max(sizes[d], 1)
    return coords


def _reduction_body(
    sel: Any,
    in_coords: list[str],
    mask: str,
    ptr: str,
    extra_ptrs: list[str],
    indent: str,
    lines: list[str],
) -> None:
    layout = sel.signature.inputs[0].layout
    region = sel.fusion.prologue_for(0)
    if region is None or region.reads_self:
        off = _self_affine(layout.sizes, layout.strides, in_coords)
        lines.append(f"{indent}v = tl.load({ptr} + {off}, mask={mask}, other=0.0)")
    if region is not None:
        value = _render_region(
            region,
            self_value="v",
            coords=in_coords,
            extra_ptrs=extra_ptrs,
            mask=mask,
            prefix="p",
            indent=indent,
            lines=lines,
        )
        lines.append(f"{indent}v = {value}")


def _reduction_codegen(
    option: Any,
    config: Any,
    signature: Any,
    fused_loops: Any,
    context: FrozenCodegenContext,
) -> GeneratedKernel:
    ext = load_op_options_native()
    sel = _validated(option, config, signature, fused_loops, context)
    cfg, sig = sel.config, sel.signature
    x = sig.inputs[0].layout
    dim = int(sig.reduction_attrs.dims[0])
    keep = [d for d in range(len(x.sizes)) if d != dim]
    r_numel = int(x.sizes[dim])
    rows = int(sig.output.layout.numel())
    bx, br = int(cfg.block_x), int(cfg.block_r)
    args = _kernel_args(sel)
    ptrs = [_ptr_name(a) for a in args]
    extra_ptrs = [p for a, p in zip(args, ptrs) if a.role == ArgRole.EXTRA_INPUT]

    lines = _kernel_header(sel, args, cfg.num_warps, cfg.num_stages, context)
    lines += [
        f"    xindex = (tl.program_id(0).to(tl.int64) * {bx} + tl.arange(0, {bx}).to(tl.int64))[:, None]",
        f"    xmask = xindex < {rows}",
    ]
    coords = _row_coords(x.sizes, keep, lines)
    in_coords = [coords.get(d, "r") for d in range(len(x.sizes))]
    lanes = f"tl.arange(0, {br}).to(tl.int64)[None, :]"
    if br >= r_numel:
        lines += [f"    r = {lanes}", f"    m = xmask & (r < {r_numel})"]
        _reduction_body(sel, in_coords, "m", _input_ptr(0), extra_ptrs, "    ", lines)
        lines += ["    v = tl.where(m, v, 0.0)", "    acc = tl.sum(v, 1)[:, None]"]
    else:
        lines += [
            f"    racc = tl.zeros([{bx}, {br}], tl.float32)",
            f"    for r0 in range(0, {r_numel}, {br}):",
            f"        r = r0 + {lanes}",
            f"        m = xmask & (r < {r_numel})",
        ]
        _reduction_body(
            sel, in_coords, "m", _input_ptr(0), extra_ptrs, "        ", lines
        )
        lines += [
            "        racc = tl.where(m, racc + v, racc)",
            "    acc = tl.sum(racc, 1)[:, None]",
        ]
    if sig.kind == ext.OpKind.MEAN_DIM:
        denom = _f32_scalar(float(r_numel))
        div = "tl.div_rn(acc, {d})" if sel.context.division_rounding else "acc / {d}"
        lines.append("    acc = " + div.format(d=denom))
    out = sig.output.layout
    out_coords = [coords[d] for d in keep]
    if sig.reduction_attrs.keepdim:
        out_coords.insert(dim, "(0 * xindex)")
    value = "acc"
    if sel.fusion.epilogue is not None:
        value = _render_region(
            sel.fusion.epilogue,
            self_value="acc",
            coords=out_coords,
            extra_ptrs=extra_ptrs,
            mask="xmask",
            prefix="e",
            indent="    ",
            lines=lines,
        )
    store_off = f"{_store_offset(sel)} + {_self_affine(out.sizes, out.strides, out_coords)} + 0 * xindex"
    lines.append(f"    tl.store({ptrs[-1]} + {store_off}, {value}, mask=xmask)")
    grid = (_cdiv(rows, bx), 1, 1)
    return _finish(sel, lines, _launch(sel, args, grid, cfg.num_warps, cfg.num_stages))


# ---------------------------------------------------------------------------
# Native eager delegation.
# ---------------------------------------------------------------------------


def _f64_literal(value: float) -> str:
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    return f"_f64(0x{bits:016X})"


def _native_call_expr(sel: Any, names: list[str], extra_names: list[str]) -> str:
    ext = load_op_options_native()
    plan = sel.native
    t = plan.target
    operands = [names[s] for s in plan.operand_slots]
    if t == ext.NativeTarget.ATEN_MM:
        return f"torch.ops.aten.mm.default({operands[0]}, {operands[1]})"
    if t == ext.NativeTarget.ATEN_BMM:
        return f"torch.ops.aten.bmm.default({operands[0]}, {operands[1]})"
    if t == ext.NativeTarget.ATEN_ADDMM:
        if plan.bias_view is not None:
            v = plan.bias_view
            src = extra_names[v.extra]
            bias = (
                src
                if v.direct
                else (
                    f"{src}.as_strided({list(v.sizes)}, {list(v.strides)}, {src}.storage_offset() + {v.offset})"
                )
            )
            operands = [bias] + operands
        return (
            f"torch.ops.aten.addmm.default({operands[0]}, {operands[1]}, {operands[2]}, "
            f"beta={_f64_literal(plan.beta)}, alpha={_f64_literal(plan.alpha)})"
        )
    op = "sum.dim_IntList" if t == ext.NativeTarget.ATEN_SUM_DIM_INTLIST else "mean.dim"
    return (
        f"torch.ops.aten.{op}({operands[0]}, {list(plan.dims)}, {bool(plan.keepdim)})"
    )


def _native_codegen(
    option: Any,
    config: Any,
    signature: Any,
    fused_loops: Any,
    context: FrozenCodegenContext,
) -> NativeCall:
    ext = load_op_options_native()
    sel = _validated(option, config, signature, fused_loops, context)
    args = _kernel_args(sel)
    names = [f"in{a.slot}" for a in args if a.role == ArgRole.INPUT]
    extra_names = [f"x{a.slot}" for a in args if a.role == ArgRole.EXTRA_INPUT]
    out = sel.signature.output.layout
    sizes = list(out.sizes)
    checked = [(d, s) for d, (n, s) in enumerate(zip(out.sizes, out.strides)) if n > 1]
    fn = f"call_{sel.option_id}_{sel.cache_key}"
    lines = [
        "import struct",
        "import torch",
        "",
        "def _f64(bits):",
        "    return struct.unpack('<d', struct.pack('<Q', bits))[0]",
        "",
        f"def {fn}({', '.join(names + extra_names)}):",
        f'    """{sel.option_id} v{sel.option_version} key={sel.cache_key}"""',
        f"    out = {_native_call_expr(sel, names, extra_names)}",
        f"    if list(out.shape) != {sizes} or any(out.stride(d) != s for d, s in {checked}):",
        "        raise RuntimeError('native result layout differs from the declared signature output')",
        "    return out",
    ]
    return NativeCall(
        option_id=sel.option_id,
        option_version=int(sel.option_version),
        cache_key=sel.cache_key,
        target=ext.native_target_name(sel.native.target),
        function_name=fn,
        source="\n".join(lines) + "\n",
        args=tuple(a for a in args if a.role != ArgRole.OUTPUT),
    )


# ---------------------------------------------------------------------------
# Strategy dispatch.
# ---------------------------------------------------------------------------


def _strategies() -> dict[Any, Callable[..., Any]]:
    s = load_op_options_native().CodegenStrategy
    return {
        s.NATIVE_ATEN: _native_codegen,
        s.TRITON_GEMM: _gemm_codegen,
        s.TRITON_REDUCTION: _reduction_codegen,
    }


def codegen(
    option_id: str,
    config: Any,
    signature: Any,
    fused_loops: Any,
    context: FrozenCodegenContext,
) -> GeneratedKernel | NativeCall:
    """Render the selected option. Raises ``OptionRejected`` on any invalid
    config/fusion/numerics combination; never substitutes another option."""
    option = load_op_options_native().default_registry().get(option_id)
    return _strategies()[option.strategy](
        option, config, signature, fused_loops, context
    )


# ---------------------------------------------------------------------------
# Plan passes (C++). The core extension owns KernelProgram/ExternCall/Compute
# registrations and is loaded first; this module never re-registers them.
# ---------------------------------------------------------------------------


def _with_core() -> Any:
    from torch._inductor.loops_ir import load_cpp_ir

    load_cpp_ir()
    return load_op_options_native()


def decode_op_signatures(program: Any) -> Any:
    """``decode_op_signatures`` over a core ``KernelProgram`` (CUDA lane)."""
    ext = _with_core()
    return ext.decode_op_signatures(program, ext.DeviceKind.CUDA)


def form_fused_op_regions(program: Any, decoded: Any) -> Any:
    ext = _with_core()
    return ext.form_fused_op_regions(program, decoded, ext.default_registry())


def enumerate_op_candidates(
    regions: Any, decoded: Any, context: FrozenCodegenContext
) -> Any:
    ext = _with_core()
    return ext.enumerate_op_candidates(
        regions, decoded, ext.default_registry(), context.native
    )


# ---------------------------------------------------------------------------
# Compile / launch (compile-time only machinery: PyCodeCache + AsyncCompile).
# ---------------------------------------------------------------------------


def _load_module(source: str) -> types.ModuleType:
    from torch._inductor.codecache import PyCodeCache

    key, path = PyCodeCache.write(source)
    return PyCodeCache.load_by_key_path(key, path)


def _check_tensor(arg: KernelArg, t: torch.Tensor, device: torch.device | None) -> None:
    if t.dtype != arg.dtype or tuple(t.shape) != arg.sizes:
        raise ValueError(
            f"argument {arg.name!r}: dtype/shape differ from the launch schema"
        )
    if any(
        n > 1 and t.stride(d) != s
        for d, (n, s) in enumerate(zip(arg.sizes, arg.strides))
    ):
        raise ValueError(
            f"argument {arg.name!r}: strides differ from the launch schema"
        )
    if device is not None and t.device != device:
        raise ValueError(f"argument {arg.name!r}: wrong device")
    available = t.untyped_storage().nbytes() // t.element_size() - t.storage_offset()
    if arg.required_elements > available:
        raise ValueError(
            f"argument {arg.name!r}: storage holds {available} elements from the data pointer, "
            f"kernel touches {arg.required_elements}"
        )


def _byte_span(t: torch.Tensor, elements: int) -> tuple[int, int, int]:
    start = t.data_ptr()
    return t.untyped_storage().data_ptr(), start, start + elements * t.element_size()


def _check_output_disjoint(
    args: tuple[KernelArg, ...], tensors: Sequence[torch.Tensor]
) -> None:
    """The generated kernel writes its output in place while reading the
    others: reject any shared storage whose touched byte ranges intersect."""
    out_arg, out = args[-1], tensors[-1]
    if out_arg.required_elements == 0:
        return
    o_store, o_lo, o_hi = _byte_span(out, out_arg.required_elements)
    for arg, t in zip(args[:-1], tensors[:-1]):
        extent = (
            1 + sum((n - 1) * s for n, s in zip(arg.sizes, arg.strides))
            if t.numel()
            else 0
        )
        store, lo, hi = _byte_span(t, extent)
        if store == o_store and lo < o_hi and o_lo < hi:
            raise ValueError(f"output {out_arg.name!r} overlaps argument {arg.name!r}")


@dataclasses.dataclass(frozen=True)
class CompiledMetadata:
    """Facts read back from the compiled binary's launcher."""

    num_warps: int
    num_stages: int
    shared_bytes: int
    registers: int
    spills: int


@dataclasses.dataclass(frozen=True)
class CompiledKernel:
    kernel: GeneratedKernel
    tuner: Any
    device: torch.device

    def launch(self, *tensors: torch.Tensor) -> None:
        args = self.kernel.launch.args
        if len(tensors) != len(args):
            raise ValueError("argument count differs from the launch schema")
        for arg, t in zip(args, tensors):
            _check_tensor(arg, t, self.device)
        _check_output_disjoint(args, tensors)
        gx, gy, gz = self.kernel.launch.grid
        if gx == 0 or gy == 0 or gz == 0:
            return
        stream = torch._C._cuda_getCurrentRawStream(self.device.index)
        self.tuner.run(*tensors, gx, gy, gz, stream=stream)

    def metadata(self) -> CompiledMetadata:
        (launcher,) = self.tuner.launchers
        return CompiledMetadata(
            num_warps=int(launcher.config.num_warps),
            num_stages=int(launcher.config.num_stages),
            shared_bytes=int(launcher.shared),
            registers=int(launcher.n_regs),
            spills=int(launcher.n_spills),
        )

    def ptx(self) -> str | None:
        """PTX from compile-only tuner state when available (None otherwise)."""
        for launcher in getattr(self.tuner, "launchers", None) or []:
            text = (getattr(getattr(launcher, "bin", None), "asm", None) or {}).get(
                "ptx"
            )
            if text:
                return text
        for result in getattr(self.tuner, "compile_results", None) or []:
            text = (getattr(getattr(result, "kernel", None), "asm", None) or {}).get(
                "ptx"
            )
            if text:
                return text
        return None


def compile_generated_kernel(
    kernel: GeneratedKernel, context: FrozenCodegenContext
) -> CompiledKernel:
    body = "\n" + kernel.source.strip() + "\n"
    module_source = "\n".join(
        [
            "from torch._inductor.async_compile import AsyncCompile",
            "async_compile = AsyncCompile()",
            f"{kernel.kernel_name} = async_compile.triton({kernel.kernel_name!r}, '''{body}''', device_str='cuda')",
            "async_compile.wait(globals())",
            "del async_compile",
            "",
        ]
    )
    mod = _load_module(module_source)
    tuner = getattr(mod, kernel.kernel_name)
    tuner.precompile()
    return CompiledKernel(kernel, tuner, torch.device("cuda", context.device_index))


def load_native_call(call: NativeCall) -> Callable[..., torch.Tensor]:
    fn = getattr(_load_module(call.source), call.function_name)

    def run(*tensors: torch.Tensor) -> torch.Tensor:
        if len(tensors) != len(call.args):
            raise ValueError("argument count differs from the native call schema")
        for arg, t in zip(call.args, tensors):
            _check_tensor(arg, t, None)
        return fn(*tensors)

    return run
