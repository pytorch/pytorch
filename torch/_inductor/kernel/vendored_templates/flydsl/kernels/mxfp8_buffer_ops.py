# SPDX-License-Identifier: MIT
# Vendored from the MXFP8 grouped GEMM
# package in ROCm/AMD-TorchTitan-Ops (`amd_titan/_ops/fwdgemm/_buffer_ops.py`),
# whose own upstream is the `flydsl-groupped-gemm` development repo. Kept as a
# separate file from `gemm_gfx950.py` so it can be re-synced wholesale. The one
# change is `_aux_is_positional`, which handles the 0.3.1 buffer-op builders.
"""Buffer-descriptor loads and stores, on ops that exist in flydsl 0.2.4 AND 0.3.0.

WHY THIS FILE EXISTS. These kernels were written against flydsl 0.2.4 and called
``flydsl.expr.buffer_ops`` and ``flydsl.expr.vector``. 0.3.0 DELETED both
modules, so the kernels only ran there via a compat shim that grafted the two
0.2.4 modules back under their old names -- which meant carrying 763 lines of
someone else's deleted code, on a path outside this repo.

The shim was never the only option: ``rocdl.MakeBufferRsrcOp``,
``RawPtrBufferLoadOp`` and ``RawPtrBufferStoreOp`` are present in BOTH versions.
0.2.4's ``buffer_ops`` was a convenience layer over exactly those, and this is
that layer, re-implemented against the ops directly and kept to what these
kernels actually use. No shim, no version branch, no ``flydsl.expr.buffer_ops``.

What is deliberately NOT here, because no caller needs it: ``mask=``,
``cache_modifier``, ``stride``, ``base_byte_offset``, ``offset_is_bytes``, the
``from_addr`` descriptor constructor, and the memref-derived size fallback (every
caller passes ``num_records_bytes`` explicitly). Adding one back means porting
its logic from 0.2.4, not guessing.

THE BOUNDS ARE LOAD-BEARING, not an optimisation. ``num_records`` is what makes
an out-of-range read return 0 instead of faulting, and this kernel relies on that:
it over-provisions row-tile slots and lets a surplus block's scale read go past
the plane, where 0 as an e8m0 is 2**-127 and underflows exactly as the epilogue
mask intends. Dropping the bound gives ``hipErrorIllegalAddress`` on real
DSV3-16B shapes -- see the descriptor comments in `_fwd_kernel.py`.
"""

from __future__ import annotations

import inspect
from typing import Optional, Union

from flydsl._mlir import ir
from flydsl._mlir.dialects import (
    arith as std_arith,
    fly as _fly,
    llvm,
    rocdl,
    vector as _vector,
)
from flydsl._mlir.extras import types as T
from flydsl.expr import arith as _arith_ext


# CDNA (gfx9xx) buffer resource flags word, bits 127:96 of the V#:
#   bits [14:12] DATA_FORMAT = 7 (float), bits [18:15] NUM_FORMAT = 4 (32-bit).
# Mirrors LLVM's AMDGPUToROCDL makeBufferRsrc(). gfx950 is CDNA, so the RDNA
# variant (bit 24, OOB_SELECT) is not needed and is not offered -- a kernel that
# ran on RDNA with this constant would silently lose bounds checking.
_CDNA_BUFFER_FLAGS = (7 << 12) | (4 << 15)  # 0x20070
_UNBOUNDED = 0xFFFFFFFF


def _aux_is_positional(op) -> bool:
    """Whether this FlyDSL build still takes ``aux`` as a positional operand.

    flydsl 0.2.4/0.3.0 declare ``RawPtrBuffer{Load,Store}Op(..., soffset, aux)``
    with ``aux`` a required positional ir.Value; 0.3.1 made it a keyword-only
    *attribute* defaulting to None. Passing the 0.2.4 form to 0.3.1 raises
    ``TypeError: __init__() takes 5 positional arguments but 6 were given`` at
    trace time, so the arity is read off the builder rather than assumed.
    """
    parameter = inspect.signature(op.__init__).parameters.get("aux")
    return (
        parameter is not None
        and parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )


_LOAD_AUX_IS_POSITIONAL = _aux_is_positional(rocdl.RawPtrBufferLoadOp)
_STORE_AUX_IS_POSITIONAL = _aux_is_positional(rocdl.RawPtrBufferStoreOp)


def _unwrap(v):
    """DSL Numeric / OpView / ir.Value -> ir.Value."""
    if hasattr(v, "ir_value"):
        v = v.ir_value()
    if hasattr(v, "result"):
        v = v.result
    return v


def _const_i32(v: int):
    return std_arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), int(v))).result


def _const_i16(v: int):
    return std_arith.ConstantOp(T.i16(), ir.IntegerAttr.get(T.i16(), int(v))).result


def _const_i64(v: int):
    return std_arith.ConstantOp(T.i64(), ir.IntegerAttr.get(T.i64(), int(v))).result


def _as_i32(v):
    """Coerce an unwrapped value to i32, index-casting or sign-extending as needed."""
    v = _unwrap(v)
    if isinstance(v.type, ir.IntegerType) and v.type.width == 32:
        return v
    if isinstance(v.type, ir.IndexType):
        return std_arith.IndexCastOp(T.i32(), v).result
    return std_arith.ExtSIOp(T.i32(), v).result


def _as_i64(v):
    v = _unwrap(v)
    if isinstance(v.type, ir.IntegerType) and v.type.width == 64:
        return v
    if isinstance(v.type, ir.IndexType):
        return std_arith.IndexCastOp(T.i64(), v).result
    return std_arith.ExtSIOp(T.i64(), v).result


def create_buffer_resource(
    memref_val,
    max_size: bool = True,
    *,
    num_records_bytes: Optional[Union[int, ir.Value]] = None,
):
    """A buffer resource descriptor (V#, `!llvm.ptr<8>`) over a fly.memref.

    `num_records_bytes` sets the hardware bound and takes precedence over
    `max_size`. Pass it: an unbounded descriptor turns an out-of-range read from
    a returned 0 into a fault (see the module docstring).
    """
    base_ptr = _fly.extract_aligned_pointer_as_index(
        ir.Type.parse("!llvm.ptr"), _unwrap(memref_val)
    )
    if num_records_bytes is None:
        num_records = _const_i64(_UNBOUNDED if max_size else 0)
    elif isinstance(num_records_bytes, int):
        num_records = _const_i64(max(0, min(int(num_records_bytes), _UNBOUNDED)))
    else:
        num_records = _as_i64(num_records_bytes)
    return rocdl.MakeBufferRsrcOp(
        ir.Type.parse("!llvm.ptr<8>"),
        base_ptr,
        _const_i16(0),
        num_records,
        _const_i32(_CDNA_BUFFER_FLAGS),
    ).result


def buffer_load(
    rsrc,
    offset,
    vec_width: int = 4,
    dtype=None,
    soffset_bytes: Optional[Union[int, ir.Value]] = None,
    is_scalar: bool = False,
):
    """Load `vec_width` x `dtype` from `rsrc` at an ELEMENT offset.

    `offset` is in elements and is scaled to bytes here -- the hardware wants
    bytes. `soffset_bytes` rides the instruction's scalar offset field, so a
    small wave-uniform delta costs no VGPR arithmetic.

    `is_scalar` emits `s.buffer.load` instead: a uniform SMEM load landing in
    SGPRs, for wave-uniform addresses only. It forces i32 (the intrinsic returns
    raw dwords) and takes no soffset, matching 0.2.4.
    """
    if is_scalar:
        if vec_width not in (1, 4):
            raise ValueError(f"buffer_load(is_scalar=True): vec_width={vec_width}")
        if soffset_bytes is not None:
            raise ValueError("buffer_load(is_scalar=True) takes no soffset_bytes")
        dtype = T.i32()
    elif dtype is None:
        dtype = T.f32()
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type

    offset = _const_i32(offset) if isinstance(offset, int) else _as_i32(offset)
    # element offset -> byte offset
    offset = std_arith.MulIOp(offset, _const_i32(dtype.width // 8)).result
    result_type = dtype if vec_width == 1 else ir.VectorType.get([vec_width], dtype)

    if is_scalar:
        # The s.buffer.load intrinsic wants the descriptor as v4i32, not the
        # opaque ptr<8> the vector path uses. It takes TWO steps: `llvm.bitcast`
        # only casts pointer-to-pointer, so go through an i128 first. Casting
        # ptr<8> straight to v4i32 fails to compile with
        # "'llvm.bitcast' op can only cast pointers from and to pointers".
        rsrc_v4 = llvm.bitcast(
            ir.VectorType.get([4], T.i32()),
            llvm.ptrtoint(ir.IntegerType.get_signless(128), _unwrap(rsrc)),
        )
        return llvm.call_intrinsic(
            result_type,
            "llvm.amdgcn.s.buffer.load." + ("i32" if vec_width == 1 else "v4i32"),
            [rsrc_v4, offset, _const_i32(0)],
            [],
            [],
        )

    soffset = (
        _const_i32(0)
        if soffset_bytes is None
        else _const_i32(soffset_bytes)
        if isinstance(soffset_bytes, int)
        else _as_i32(soffset_bytes)
    )
    load_args = [result_type, _unwrap(rsrc), offset, soffset]
    if _LOAD_AUX_IS_POSITIONAL:
        load_args.append(_const_i32(0))
    return rocdl.RawPtrBufferLoadOp(*load_args).result


def buffer_store(data, rsrc, offset):
    """Store `data` to `rsrc` at an ELEMENT offset (scaled to bytes here)."""
    data = _unwrap(data)
    offset = _const_i32(offset) if isinstance(offset, int) else _as_i32(offset)
    elem_t = data.type.element_type if hasattr(data.type, "element_type") else data.type
    offset = std_arith.MulIOp(offset, _const_i32(elem_t.width // 8)).result
    store_args = [data, _unwrap(rsrc), offset, _const_i32(0)]
    if _STORE_AUX_IS_POSITIONAL:
        store_args.append(_const_i32(0))
    rocdl.RawPtrBufferStoreOp(*store_args)


def vec_extract(vector, static_position=None, dynamic_position=None):
    """`vector.extract`, replacing 0.3.0's deleted `flydsl.expr.vector.extract`.

    A dynamic index needs a kDynamic sentinel in the static attribute for the
    ODS builder to pair the two, so fill those in when only dynamic indices are
    given.
    """
    static_position = list(static_position or [])
    dynamic_position = [
        _arith_ext.unwrap(i, index=True) for i in (dynamic_position or [])
    ]
    if dynamic_position and len(static_position) < len(dynamic_position):
        static_position += [ir.ShapedType.get_dynamic_size()] * (
            len(dynamic_position) - len(static_position)
        )
    return _vector.ExtractOp(
        _arith_ext.unwrap(vector),
        static_position=static_position,
        dynamic_position=dynamic_position,
    ).result
