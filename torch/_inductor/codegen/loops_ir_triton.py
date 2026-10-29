"""Triton emission for TiledIR (``torch._inductor.codegen.loops_ir_triton``).

Local emitter only: reads an already-tiled ``TiledFunction`` (SSA values,
basic blocks, explicit types/shapes/masks/grid/offsets, numerics policy) and
returns file-backed-ready Python source defining a ``@triton.jit`` kernel
named exactly ``function.name``. No Scheduler/GraphLowering/LoopBody/global-V
dependencies; nothing here inspects LoopIR, KernelIR, or any layout/strides.

Public API
----------
``emit_triton(function) -> str``

Source is produced from Tiled ops (``op``/``id``/``operands``/``buffer``/
``int_value``/``float_value``/``dtype``/``shape``); the only string-op
dispatch in this file is the table below, which mirrors ``tiled_ir.h``::

    program_id  pid scalar                      -> tl.program_id(0).to(tl.int64)
    arange      lane ids                        -> tl.arange(0, BLOCK).to(tl.int64)
    const_int   int_value()                     -> tl.full([], v, tl.int64)
    const_float float_value()                   -> bitcast f32 scalar (exact bits)
    broadcast   operands {scalar}               -> tl.broadcast_to(v, (BLOCK,))
    add/mul/floordiv/mod (i64, same shape)      -> + * // %
    cmp_lt      (i64 -> bool, same shape)       -> <
    select      (cond, on_true, on_false)       -> tl.where(c, t, f)
    fadd/fmul   (f32, order preserved)          -> policy arithmetic (see below)
    relu        (f32)                           -> policy ReLU (see below)
    load        buffer(), {offset, mask}        -> tl.load(pK + off, mask, other=0.0)
    store       buffer(), {value, offset, mask} -> tl.store(pK + off, val, mask)
    cast        precision-preserving no-op      -> alias assignment

Numerics policy (``function.numerics``; validated, never defaulted here):
``strict_eager`` preserves eager bits; ``inductor`` follows stock Inductor's
effective choices. Anything else is rejected.

Strict (``strict_eager``) FP32 arithmetic uses explicit PTX
(``tl.inline_asm_elementwise`` with ``add.rn.f32``/``mul.rn.f32``,
``=f,f,f`` constraints, ``is_pure=True``, ``pack=1``) rather than TL
``+``/``*``: LLVM folds neutral element-wise ops (``x*1``, ``x+(-0)``) and
canonicalizes NaN payloads even with ``enable_fp_fusion=False``, which breaks
eager bitwise parity (e.g. a signaling-NaN input ``0x7FC00001`` must produce
eager's canonical result, not the folded payload). Explicit PTX pins
round-to-nearest and blocks the identity fold/contraction; ``is_pure=True``
still lets the backend CSE identical elementwise ops safely, with no
artificial global serialization. ``enable_fp_fusion`` must be False and
``disable_ftz`` must be True, matching the function flags. Integer index
arithmetic stays ordinary i64 TL ops. CUDA only.

Inductor policy (``inductor``) emits regular typed Triton arithmetic matching
the stock convention (``+``/``*`` on f32 values) and the stock ReLU form
(``tl.maximum`` with an int32 zero against the f32 value, propagating NaN as
stock does: ``tl.maximum(zero_i32, x, tl.PropagateNan.ALL)``). It never forces
fast math: the function-carried ``enable_fp_fusion``/``disable_ftz`` flags are
passed through ``triton_meta`` unchanged, so the backend worker's frozen
effective settings (stock ``enable_fp_fusion=not emulate_precision_casts``,
``disable_ftz=eager_numerics.disable_ftz``) decide fusion/FTZ. No
``use_fast_math``/approx math is emitted (no ``tl.exp``/libdevice or other
approximate ops); f32 generated code only.

Launch contract (implemented by the runtime worker, documented here): the
kernel signature is ``(p0, p1, ...)`` in ``function.parameters`` launch order
(INCLUDING output pointers); the runtime calls
``mod.<name>[(function.grid_size,)](*buffers, num_warps=function.num_warps,
enable_fp_fusion=function.enable_fp_fusion,
enable_reflect_ftz=not function.disable_ftz)``
via file-backed ``PyCodeCache.load`` and skips the launch entirely when
``grid_size == 0``. The fixed-config device path records the same effective
flags in ``triton_meta`` (``enable_fp_fusion``/``disable_ftz`` verbatim from
the function; never forced). Note the FTZ spelling: ``disable_ftz`` is a
TiledFunction attribute (required API), but raw Triton has no ``disable_ftz``
launch kwarg (passing it raises ``KeyError`` -- probed); the equivalent
raw-Triton kwarg is ``enable_reflect_ftz=False``, matching inductor's own
``compile_meta["disable_ftz"]`` -> ``options["enable_reflect_ftz"] = False``
mapping in ``torch/_inductor/runtime/triton_heuristics.py``.

Strictness notes (bitwise parity with eager, same device/settings, strict
policy only):

- No FMA/FTZ/approx: f32 add/mul emit one pinned-RN PTX instruction each
  (``enable_fp_fusion=False`` at launch as well), ``tl.where`` for
  ReLU/select, no ``tl.maximum`` (proven bit-divergent on NaN
  payloads/signed zero on H100), no ``tl.exp``/ libdevice or other
  approximate math. Integer index arithmetic stays ordinary i64 TL ops.
- ReLU is ``tl.where(x < 0.0, +0.0, x)``: NaN (payload intact) and ``-0.0``
  take the false lane (``x < 0.0`` is False for both), negatives become
  ``+0.0``. This matches eager CUDA ``clamp_min`` (explicit ``_isnan``
  propagation in ``UnaryOpsKernel.cu``); verified bit-exact on H100 over
  signed zeros, a NaN payload, infinities, and a subnormal.
- FP32 constants use the exact-bits idiom
  ``tl.full(shape, 0xBITS, tl.uint32).to(tl.float32, bitcast=True)`` (same as
  inductor's ``_f32_const_from_bits``): bit-exact for normals, subnormals,
  and ``-0.0`` (a ``-0.0`` Python literal would collapse to ``+0.0`` through
  Triton's ``value == 0`` null path, so it is never emitted as a literal).
  No data folding or algebraic rewrites are performed here.
- Integer arithmetic is int64 throughout: ``tl.program_id(0)`` (int32) is
  widened with ``.to(tl.int64)``, lanes come from
  ``tl.arange(0, BLOCK).to(tl.int64)``, and int scalars are explicit
  ``tl.full([], v, tl.int64)`` (never bare literals, whose Triton type would
  depend on magnitude). Triton ``//``/``%`` are trunc-div; the tiling pass
  only emits ``floordiv``/``mod`` over provably non-negative dividends with
  positive divisors (flat lane ids, tile strides/extents), where trunc and
  floor coincide -- documented precondition, also asserted by LoopIR's own
  positive-divisor rules.
- ``tl.arange``/block tensors require power-of-2 shapes: non-power-of-2
  ``block_size`` raises a clear error (the caller owns tile sizing; the
  emitter performs no hidden padding decisions).
- Pointer arithmetic is explicit ``pK + off`` on int64 offset vectors; masked
  lanes use ``other=0.0`` on loads (defined dead-lane values) and the store
  mask (dead lanes never escape). Strides/layout are never consulted: offsets
  are already explicit flat trees.
"""

from __future__ import annotations

import re
import struct
from typing import Any

__all__ = ["emit_triton", "TILED_OP_NAMES"]

# Exact op strings from tiled_ir.h. The emitter switches on these (strings are
# more stable than an enum across the C++/Python boundary).
TILED_OP_NAMES = (
    "program_id",
    "arange",
    "const_int",
    "const_float",
    "broadcast",
    "add",
    "mul",
    "floordiv",
    "mod",
    "cmp_lt",
    "select",
    "fadd",
    "fmul",
    "relu",
    "load",
    "store",
    "cast",
)

_INT_OPS = {"add": "+", "mul": "*", "floordiv": "//", "mod": "%"}
# FP32 Tiled ops -> explicit PTX mnemonic (round-to-nearest, CUDA only).
_FLOAT_PT_OPS = {"fadd": "add.rn.f32", "fmul": "mul.rn.f32"}


def _f32_ptx_expr(mnemonic: str, lhs: str, rhs: str) -> str:
    """Explicit-PTX f32 binary: pins RN rounding, blocks LLVM identity
    folds/NaN canonicalization that TL ``+``/``*`` permit even with
    ``enable_fp_fusion=False``. ``is_pure=True`` allows CSE of identical
    elementwise ops; ``pack=1`` keeps lane semantics one-to-one."""
    return (
        "tl.inline_asm_elementwise("
        f"'{mnemonic} $0, $1, $2;', '=f,f,f', "
        f"[{lhs}, {rhs}], dtype=tl.float32, is_pure=True, pack=1)"
    )


def _sanitize_identifier(name: str) -> str:
    """Map to a valid Python identifier (buffer IDs are mapped positionally)."""
    cleaned = re.sub(r"\W", "_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def _f32_bits(value: float) -> int:
    """Exact IEEE-754 binary32 bits of an f32-representable double."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _f32_scalar_expr(value: float) -> str:
    """Bit-exact f32 scalar expression (never a bare literal)."""
    return (
        f"tl.full([], 0x{_f32_bits(value):08X}, tl.uint32).to(tl.float32, bitcast=True)"
    )


def _f32_vector_expr(value: float, block: int) -> str:
    """Bit-exact f32 vector expression (never a bare literal)."""
    return (
        f"tl.full([{block}], 0x{_f32_bits(value):08X}, tl.uint32)"
        ".to(tl.float32, bitcast=True)"
    )


def _require_shape(ins: Any, block: int, ctx: str) -> bool:
    """Validate dtype/shape wire format; return True for vector results."""
    shape = list(ins.shape)
    if shape != [] and shape != [block]:
        raise ValueError(
            f"{ctx}: shape {shape} must be [] or [{block}] "
            "(emitter reads TiledIR, which fixes lanes to block_size)"
        )
    return shape != []


def emit_triton(function: Any, *, device: Any | None = None) -> str:
    """Emit file-backed-ready ``@triton.jit`` source for a TiledFunction.

    Reads only local TiledIR properties: ``name``, ``parameters`` (launch
    order, INCLUDING output pointers), ``blocks`` (``name``/``instructions``/
    ``terminator``), ``grid_size`` (recorded in the header comment; the
    runtime owns grid/launch), ``block_size``, ``num_warps``,
    ``enable_fp_fusion``/``disable_ftz``/``numerics`` (recorded verbatim in
    the header and in ``triton_meta``; the runtime passes num_warps/
    enable_fp_fusion as launch kwargs and maps disable_ftz to raw-Triton
    ``enable_reflect_ftz``). Per instruction: ``op`` (string), ``id``,
    ``operands``, ``buffer``, ``int_value``, ``float_value``, ``dtype``,
    ``shape``. No Python callbacks, no reinterpretation of opaque dumps.

    ``numerics`` must be ``'strict_eager'`` (bit-exact PTX arithmetic and
    signed-zero/NaN-sensitive ReLU) or ``'inductor'`` (regular typed Triton
    arithmetic/ReLU matching the stock convention, with the function-carried
    flags passed through ``triton_meta`` and never forced). Unknown policies
    are rejected.
    """
    name = _sanitize_identifier(function.name)
    if name != function.name:
        raise ValueError(
            f"tiled function name {function.name!r} is not a valid Python "
            "identifier (refusing to guess a renamed kernel; rename upstream)"
        )
    block = int(function.block_size)
    if block <= 0 or (block & (block - 1)) != 0:
        raise ValueError(
            f"tiled function {function.name!r}: block_size {block} is not a "
            "power of 2 (tl.arange/block tensors require it; the caller owns "
            "tile sizing, the emitter will not pad silently)"
        )
    numerics = function.numerics
    if numerics not in ("strict_eager", "inductor"):
        raise ValueError(
            f"unknown numerics policy {numerics!r} "
            "(expected 'strict_eager' or 'inductor')"
        )
    strict = numerics == "strict_eager"
    if strict:
        if function.enable_fp_fusion:
            raise ValueError("strict eager requires enable_fp_fusion=False")
        if not function.disable_ftz:
            raise ValueError("strict eager requires disable_ftz=True")
    # Inductor policy: consume the already-selected effective flags carried
    # by the function (frozen by the backend worker from stock
    # enable_fp_fusion=not emulate_precision_casts /
    # disable_ftz=eager_numerics.disable_ftz). Never forced here.

    # Buffer IDs -> p0, p1, ... in EXACT launch (parameters) order, so the
    # generated signature matches `jit[(grid,)](*buffers, ...)`. Sanitized
    # mapping: no source injection from buffer names.
    params = list(function.parameters)
    param_names: dict[str, str] = {}
    for i, buf in enumerate(params):
        if buf.name in param_names:
            raise ValueError(f"duplicate parameter {buf.name!r}")
        param_names[buf.name] = f"p{i}"
    sig = ", ".join(param_names[buf.name] for buf in params)

    lines: list[str] = []
    lines.append("import triton")
    lines.append("import triton.language as tl")
    if device is not None:
        from torch._inductor.runtime.hints import DeviceProperties
        from torch._inductor.runtime.triton_heuristics import FixedGrid

        if any(buf.dtype != "f32" for buf in params):
            raise ValueError("generated kernels require FP32 pointers")
        triton_meta = {
            "signature": {param_names[buf.name]: "*fp32" for buf in params},
            "device": DeviceProperties.create(device),
            "constants": {},
            "configs": [{}],
            "enable_fp_fusion": bool(function.enable_fp_fusion),
            "disable_ftz": bool(function.disable_ftz),
        }
        stored_buffers = {
            ins.buffer
            for blk in function.blocks
            for ins in blk.instructions
            if ins.op == "store"
        }
        inductor_meta = {
            **FixedGrid.setup_grid_as_args(),
            "kernel_name": name,
            "mutated_arg_names": [
                param_names[buf.name] for buf in params if buf.name in stored_buffers
            ],
        }
        lines.append("from torch._inductor.runtime import triton_heuristics")
        lines.append("from torch._inductor.runtime.hints import DeviceProperties")
        lines.append("")
        lines.append("@triton_heuristics.fixed_config(")
        lines.append(
            f"    {{'num_warps': {int(function.num_warps)}, 'num_stages': 1}}, "
            f"filename=__file__, triton_meta={triton_meta!r}, "
            f"inductor_meta={inductor_meta!r})"
        )
    else:
        lines.append("")
    lines.append("@triton.jit")
    lines.append(f"def {name}({sig}):")
    lines.append(f'    """Tiled kernel {function.name} [{numerics}].')
    lines.append("")
    lines.append(
        f"    grid_size={int(function.grid_size)} block_size={block} "
        f"num_warps={int(function.num_warps)} "
        f"enable_fp_fusion={bool(function.enable_fp_fusion)} "
        f"disable_ftz={bool(function.disable_ftz)} "
        f"numerics={function.numerics!r}."
    )
    lines.append(
        "    Launch: kernel[(grid_size,)](*buffers, "
        f"num_warps={int(function.num_warps)}, "
        f"enable_fp_fusion={bool(function.enable_fp_fusion)}, "
        f"enable_reflect_ftz={not bool(function.disable_ftz)}); "
        "skip when grid_size == 0."
    )
    lines.append('    """')

    defined: set[int] = set()

    def var(i: int) -> str:
        return f"v{i}"

    def use(i: int, ctx: str) -> str:
        if i not in defined:
            raise ValueError(
                f"{ctx}: use of undefined SSA v{i} (emitter requires "
                "dominance-ordered straight-line blocks)"
            )
        return var(i)

    def define(i: int, ctx: str) -> str:
        if i < 0:
            raise ValueError(f"{ctx}: negative SSA id")
        if i in defined:
            raise ValueError(f"{ctx}: duplicate SSA v{i}")
        defined.add(i)
        return var(i)

    blocks = list(function.blocks)
    if not blocks:
        raise ValueError("tiled function has no blocks (expected entry)")
    if blocks[0].name != "entry":
        raise ValueError("first block must be 'entry'")
    for blk in blocks:
        if blk.terminator != "return":
            raise ValueError(
                f"block {blk.name!r}: terminator must be 'return' "
                f"(got {blk.terminator!r})"
            )
        for ins in blk.instructions:
            op = ins.op
            if op not in TILED_OP_NAMES:
                raise ValueError(f"unknown tiled op {op!r}")
            is_vec = _require_shape(ins, block, f"{op} v{ins.id}")
            operands = list(ins.operands)

            if op == "program_id":
                if operands or is_vec or ins.dtype != "i64":
                    raise ValueError("program_id must be an i64 scalar")
                # tl.program_id returns int32; widen explicitly to int64.
                lines.append(
                    f"    {define(ins.id, op)} = tl.program_id(0).to(tl.int64)"
                )
            elif op == "arange":
                if operands or not is_vec or ins.dtype != "i64":
                    raise ValueError("arange must be an i64 vector")
                lines.append(
                    f"    {define(ins.id, op)} = tl.arange(0, {block}).to(tl.int64)"
                )
            elif op == "const_int":
                if operands or is_vec or ins.dtype != "i64":
                    raise ValueError("const_int must be an i64 scalar")
                lines.append(
                    f"    {define(ins.id, op)} = tl.full([], {int(ins.int_value)}, tl.int64)"
                )
            elif op == "const_float":
                if operands or is_vec or ins.dtype != "f32":
                    raise ValueError("const_float must be an f32 scalar")
                lines.append(
                    f"    {define(ins.id, op)} = {_f32_scalar_expr(float(ins.float_value))}"
                )
            elif op == "broadcast":
                if len(operands) != 1 or not is_vec:
                    raise ValueError("broadcast needs 1 scalar operand")
                lines.append(
                    f"    {define(ins.id, op)} = tl.broadcast_to("
                    f"{use(operands[0], op)}, ({block},))"
                )
            elif op in _INT_OPS:
                if len(operands) != 2 or ins.dtype != "i64":
                    raise ValueError(f"{op} needs 2 i64 operands")
                sym = _INT_OPS[op]
                lines.append(
                    f"    {define(ins.id, op)} = {use(operands[0], op)} "
                    f"{sym} {use(operands[1], op)}"
                )
            elif op == "cmp_lt":
                if len(operands) != 2 or ins.dtype != "bool":
                    raise ValueError("cmp_lt needs 2 i64 operands")
                lines.append(
                    f"    {define(ins.id, op)} = {use(operands[0], op)} "
                    f"< {use(operands[1], op)}"
                )
            elif op == "select":
                if len(operands) != 3:
                    raise ValueError("select needs {cond, on_true, on_false}")
                lines.append(
                    f"    {define(ins.id, op)} = tl.where("
                    f"{use(operands[0], op)}, {use(operands[1], op)}, "
                    f"{use(operands[2], op)})"
                )
            elif op in _FLOAT_PT_OPS:
                if len(operands) != 2 or ins.dtype != "f32":
                    raise ValueError(f"{op} needs 2 f32 operands")
                if strict:
                    # Explicit PTX in source order (never TL +/*: LLVM folds
                    # x*1 / x+(-0) and canonicalizes NaNs despite
                    # enable_fp_fusion=False). Contraction is impossible here:
                    # each op is one pinned RN PTX instruction.
                    lines.append(
                        f"    {define(ins.id, op)} = "
                        f"{_f32_ptx_expr(_FLOAT_PT_OPS[op], use(operands[0], op), use(operands[1], op))}"
                    )
                else:
                    # Stock convention: regular typed Triton arithmetic
                    # (fusion/FTZ governed by the function-carried flags in
                    # triton_meta, never forced here; no fast math emitted).
                    sym = {"fadd": "+", "fmul": "*"}[op]
                    lines.append(
                        f"    {define(ins.id, op)} = {use(operands[0], op)} "
                        f"{sym} {use(operands[1], op)}"
                    )
            elif op == "relu":
                if len(operands) != 1 or ins.dtype != "f32":
                    raise ValueError("relu needs 1 f32 operand")
                x = use(operands[0], op)
                if strict:
                    # NaN-payload/signed-zero preserving: x < 0.0 is False for
                    # NaN and -0.0, so both flow to the false lane unchanged;
                    # negatives become +0.0. Never tl.maximum (bit-divergent).
                    lines.append(
                        f"    {define(ins.id, op)} = tl.where("
                        f"{x} < 0.0, {_f32_scalar_expr(0.0)}, {x})"
                    )
                else:
                    # Stock convention (matches TritonOverrides.relu emission):
                    # tl.maximum against an int32 zero with NaN propagation,
                    # exactly as stock emits for fused pointwise ReLU.
                    zero = "tl.full([], 0, tl.int32)"
                    lines.append(
                        f"    {define(ins.id, op)} = tl.maximum("
                        f"{zero}, {x}, tl.PropagateNan.ALL)"
                    )
            elif op == "load":
                if len(operands) != 2 or not is_vec or ins.dtype != "f32":
                    raise ValueError("load needs {offset, mask}, f32 vector")
                if ins.buffer not in param_names:
                    raise ValueError(
                        f"load of unknown parameter {ins.buffer!r} "
                        "(loads must reference launch-order parameters)"
                    )
                ptr = param_names[ins.buffer]
                lines.append(
                    f"    {define(ins.id, op)} = tl.load("
                    f"{ptr} + {use(operands[0], op)}, "
                    f"mask={use(operands[1], op)}, other=0.0)"
                )
            elif op == "store":
                if len(operands) != 3 or ins.id != -1:
                    raise ValueError("store needs {value, offset, mask}")
                if ins.buffer not in param_names:
                    raise ValueError(
                        f"store to unknown parameter {ins.buffer!r} "
                        "(stores must reference launch-order parameters)"
                    )
                ptr = param_names[ins.buffer]
                lines.append(
                    f"    tl.store({ptr} + {use(operands[1], op)}, "
                    f"{use(operands[0], op)}, mask={use(operands[2], op)})"
                )
            elif op == "cast":
                if len(operands) != 1:
                    raise ValueError("cast needs 1 operand")
                # Precision-preserving no-op only (verifier enforces identical
                # dtype/shape); opaque conversions live outside generated code.
                # No semantic cast is emitted -- the value flows through.
                lines.append(f"    {define(ins.id, op)} = {use(operands[0], op)}")
            else:  # pragma: no cover - guarded by TILED_OP_NAMES above
                raise ValueError(f"unhandled tiled op {op!r}")

    lines.append("")
    return "\n".join(lines)
