import math
from functools import partial
from typing import Any, NamedTuple, TYPE_CHECKING

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

import torch

from ...cutedsl import launch as _L
from ...cutedsl.compat import SmemAllocator
from ...cutedsl.config import (
    TRANSPOSE_TMA_MIN_ELEMS as _TMA_MIN_ELEMS,
    TRANSPOSE_VECTOR_MIN_BYTES as _VECTOR_MIN_BYTES,
)
from ...cutedsl.plan_cache import cached_plan
from ...cutedsl.types import Index, NumericType
from ...utils import capability as cap
from ...utils.device import check_current
from ...utils.tensor import const_data_ptr, row_alignment
from ._packing import (
    flat_copy,
    FLAT_ELEMENTS_PER_THREAD as _PACKED_ELEMENTS_PER_THREAD,
    FLAT_THREADS as _PACKED_THREADS,
    transpose_words,
)
from ._transpose import (
    fake_tiled,
    matrix_view,
    predicate,
    source_row_fast,
    source_row_pow2,
)
from ._types import CacheKey, TransposeSpec


if TYPE_CHECKING:
    from tvm_ffi import Function  # pyrefly: ignore[missing-import]

    from .transpose_tma import _TmaTranspose


_PAYLOAD_THREADS = 128
_PAYLOAD_VECTORS_PER_THREAD = 2
_PAYLOAD_CHUNK = _PAYLOAD_THREADS * _PAYLOAD_VECTORS_PER_THREAD
_PLANS: dict[CacheKey, "Function"] = {}

_TORCH_WORD: dict[int, torch.dtype] = {
    1: torch.uint8,
    2: torch.uint16,
    4: torch.uint32,
    8: torch.uint64,
}
_CUTE_WORD: dict[int, NumericType] = {
    1: cutlass.Uint8,
    2: cutlass.Uint16,
    4: cutlass.Uint32,
    8: cutlass.Uint64,
}
_PACKED: dict[torch.dtype, tuple[int, int]] = {
    torch.quint4x2: (4, 2),
    torch.quint2x4: (2, 4),
}


class _LayoutTranspose:
    def __init__(
        self,
        dtype: NumericType,
        lanes: int,
        tile: int,
        full: bool,
        normalize: bool,
        packed: bool = False,
        two_d: bool = False,
    ) -> None:
        self.dtype: NumericType = dtype
        self.lanes: int = lanes
        self.tile: int = tile
        self.full: bool = full
        self.normalize: bool = normalize
        self.packed: bool = packed
        self.vector: int = 4 if packed else 128 // dtype.width if full else 1
        self.fast: int = min(8, tile // self.vector) if full or packed else tile
        self.threads: int = 256
        self.two_d: bool = two_d

    @cute.jit
    def _transfer(
        self,
        atom: cute.TiledCopy,
        source: cute.Tensor,
        target: cute.Tensor,
        pred: cute.Tensor,
    ) -> None:
        if cutlass.const_expr(not self.full):
            cute.copy(atom, source, target, pred=pred)
        elif cutlass.const_expr(self.packed):
            cute.copy(atom, source, target)
        else:
            cute.autovec_copy(source, target)

    @cute.jit
    def __call__(
        self, src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream
    ) -> None:
        src, dst = matrix_view(src), matrix_view(dst)
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=128 if self.packed else self.dtype.width,
        )
        slow = self.threads // self.fast
        read = cute.make_tiled_copy_tv(
            atom,
            cute.make_layout((slow, self.fast), stride=(self.fast, 1)),
            cute.make_layout((1, self.vector)),
        )
        write = cute.make_tiled_copy_tv(
            atom,
            cute.make_layout((self.fast, slow), stride=(1, self.fast)),
            cute.make_layout((self.vector, 1)),
        )
        strides = (
            (self.tile + 1, 1, self.tile * (self.tile + 1))
            if self.packed
            else ((self.tile + 1) * self.lanes, self.lanes, 1)
        )
        layout = cute.make_layout((self.tile, self.tile, self.lanes), stride=strides)
        middle_div = (
            None
            if cutlass.const_expr(self.two_d)
            else cute.FastDivmodDivisorV2(cutlass.Int32(src.shape[4]))
        )
        self.kernel(src, dst, read, write, layout, middle_div).launch(
            grid=(
                cute.ceil_div(src.shape[0], self.tile),
                cute.ceil_div(src.shape[1], self.tile),
                src.shape[3] * src.shape[4],
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
        read: cute.TiledCopy,
        write: cute.TiledCopy,
        layout: cute.Layout,
        middle_div: Any,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        row, col, batch = cute.arch.block_idx()
        if cutlass.const_expr(self.two_d):
            outer, middle = 0, 0
        else:
            outer, middle = divmod(batch, middle_div)
        tile = (self.tile, self.tile, self.lanes)
        g_src = cute.local_tile(
            src[None, None, None, outer, middle], tile, (row, col, 0)
        )
        g_dst = cute.local_tile(
            dst[None, None, None, outer, middle], tile, (row, col, 0)
        )
        shared = SmemAllocator().allocate_tensor(self.dtype, layout, byte_alignment=128)
        r, w = read.get_slice(tid), write.get_slice(tid)
        r_src, r_shared = r.partition_S(g_src), r.partition_D(shared)
        w_shared, w_dst = w.partition_S(shared), w.partition_D(g_dst)
        r_src = cute.group_modes(r_src, 1, 3)
        r_shared = cute.group_modes(r_shared, 1, 3)
        w_shared = cute.group_modes(w_shared, 1, 3)
        w_dst = cute.group_modes(w_dst, 1, 3)
        coords = cute.make_identity_tensor((src.shape[0], src.shape[1], self.lanes))
        coords = cute.local_tile(coords, tile, (row, col, 0))
        r_pred = predicate(r.partition_S(coords), src.shape[0], src.shape[1])
        r_pred = cute.group_modes(r_pred, 1, 3)
        w_pred = predicate(w.partition_D(coords), src.shape[0], src.shape[1])
        w_pred = cute.group_modes(w_pred, 1, 3)
        for part in cutlass.range_constexpr(cute.size(r_src.shape[1])):
            source, target = r_src[None, part, None], r_shared[None, part, None]
            regs = cute.make_rmem_tensor_like(source)
            if cutlass.const_expr(not self.full):
                regs.fill(0)
            self._transfer(read, source, regs, r_pred[None, part, None])
            if cutlass.const_expr(self.normalize):
                for i in cutlass.range_constexpr(cute.size(regs)):
                    regs[i] = regs[i] != 0
            if cutlass.const_expr(self.packed and self.lanes > 1):
                values = cute.group_modes(regs, 0, 1)
                target = cute.group_modes(target, 0, 1)
                for i in cutlass.range_constexpr(cute.size(values.shape[0])):
                    words = transpose_words(
                        tuple(values[i, p] for p in range(self.lanes)), self.lanes
                    )
                    for p in cutlass.range_constexpr(self.lanes):
                        target[i, p] = words[p]
            else:
                cute.autovec_copy(regs, target)
        cute.arch.barrier()
        for part in cutlass.range_constexpr(cute.size(w_dst.shape[1])):
            source, target = w_shared[None, part, None], w_dst[None, part, None]
            regs = cute.make_rmem_tensor_like(target)
            cute.autovec_copy(source, regs)
            self._transfer(write, regs, target, w_pred[None, part, None])


class _PayloadTranspose:
    def __init__(
        self,
        word_type: NumericType,
        vector_words: int,
        normalize: bool,
        large: bool,
        middle_one: bool = False,
        matrix_pow2: bool = False,
    ) -> None:
        self.word_type: NumericType = word_type
        self.vector_words: int = vector_words
        self.normalize: bool = normalize
        self.large: bool = large
        self.middle_one: bool = middle_one
        self.matrix_pow2: bool = matrix_pow2

    @cute.jit
    def __call__(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
        geometry: tuple[Index, Index, Index, Index, Index, Index, Index],
        stream: cuda.CUstream,
    ) -> None:
        b, middle, a, inner, rows, a_shift, b_shift = geometry
        rows_per_block = cutlass.max(1, _PAYLOAD_CHUNK // inner)
        chunks = cutlass.Int32(cute.ceil_div(cutlass.Int64(inner), _PAYLOAD_CHUNK))
        if cutlass.const_expr(self.matrix_pow2):
            divisors = (
                a_shift,
                middle,
                b_shift,
                cute.FastDivmodDivisorV2(inner),
                cute.FastDivmodDivisorV2(chunks),
            )
        else:
            divisors = tuple(
                cute.FastDivmodDivisorV2(v) for v in (a, middle, b, inner, chunks)
            )
        grid = (
            rows * chunks
            if cutlass.const_expr(self.large)
            else cute.ceil_div(cutlass.Int64(rows), rows_per_block)
        )
        self.kernel(src, dst, divisors, inner, rows, rows_per_block).launch(
            grid=[grid, 1, 1],
            block=[_PAYLOAD_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
        divisors: tuple[Any, ...],
        inner: cutlass.Int32,
        rows: cutlass.Int32,
        rows_per_block: cutlass.Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        for i in cutlass.range_constexpr(_PAYLOAD_VECTORS_PER_THREAD):
            local = tidx + i * _PAYLOAD_THREADS
            if cutlass.const_expr(not self.large):
                row, q = divmod(local, divisors[3])
                dst_row = cutlass.Int64(bidx) * rows_per_block + row
                active = local < rows_per_block * inner
            else:
                dst_row, chunk = divmod(bidx, divisors[4])
                q = chunk * _PAYLOAD_CHUNK + local
                active = q < inner
            if dst_row < rows and active:
                if cutlass.const_expr(self.matrix_pow2):
                    src_row = source_row_pow2(dst_row, divisors[0], divisors[2])
                else:
                    src_row = source_row_fast(dst_row, divisors[:3], self.middle_one)
                src_idx = cutlass.Int64(src_row) * inner + q
                dst_idx = cutlass.Int64(dst_row) * inner + q
                if cutlass.const_expr(self.normalize):
                    value = src[src_idx, 0]
                    value = value != self.word_type(0)
                    dst[dst_idx, 0] = value
                else:
                    cute.autovec_copy(src[src_idx, None], dst[dst_idx, None])


class _PackedTranspose:
    def __init__(self, bits: int, pack: int) -> None:
        self.bits: int = bits
        self.pack: int = pack

    @cute.jit
    def __call__(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
        geometry: tuple[Index, Index, Index, Index, Index],
        stream: cuda.CUstream,
    ) -> None:
        b, middle, a, inner, _ = geometry
        divisors = tuple(cute.FastDivmodDivisorV2(v) for v in (a, middle, b, inner))
        flat_copy(src, dst, geometry, self.bits, self.pack, False, divisors).launch(
            grid=[
                cute.ceil_div(
                    cute.size(dst), _PACKED_THREADS * _PACKED_ELEMENTS_PER_THREAD
                ),
                1,
                1,
            ],
            block=[_PACKED_THREADS, 1, 1],
            stream=stream,
        )


def _word_config(itemsize: int) -> tuple[int, int]:
    word_size = min(itemsize, 8)
    return word_size, itemsize // word_size


def _word_view(t: torch.Tensor, word_size: int, pack: int = 1) -> torch.Tensor:
    if t.is_quantized:
        divisor = word_size * pack
        offset = t.storage_offset() * t.element_size() // divisor
        words = (t.numel() * t.element_size() + divisor - 1) // divisor
        view = torch.empty(0, device=t.device, dtype=_TORCH_WORD[word_size])
        return view.set_(t.untyped_storage(), offset, (words,), (1,))
    return t.as_strided((t.numel(),), (1,)).view(_TORCH_WORD[word_size])


def _tiled_view(
    t: torch.Tensor,
    spec: TransposeSpec,
    word_size: int,
    *,
    source: bool,
    packed: bool = False,
) -> torch.Tensor:
    outer, b, middle, a, _, itemsize = spec
    if packed:
        word_size = 4
        lanes = 4 // itemsize
        a, b = a // lanes, b // lanes
        src_strides = (a * lanes * middle * b, 1, b, lanes * middle * b, middle * b)
        dst_strides = (b * lanes * middle * a, lanes * middle * a, a, 1, middle * a)
    else:
        lanes = itemsize // word_size
        src_strides = (a * middle * b * lanes, lanes, b * lanes, middle * b * lanes, 1)
        dst_strides = (b * middle * a * lanes, middle * a * lanes, a * lanes, lanes, 1)
    return _word_view(t, word_size).as_strided(
        (outer, b, middle, a, lanes),
        src_strides if source else dst_strides,
    )


def _layout_tile(word_size: int, packed: bool) -> int:
    return 32 if (packed and word_size < 4) or word_size >= 8 else 64


def tiled_kernel(
    word_size: int,
    lanes: int,
    normalize: bool,
    full: bool,
    packed: bool = False,
    *,
    tma: bool = False,
    two_d: bool = False,
    shape: tuple[int | cute.SymInt, ...] | None = None,
) -> tuple["_LayoutTranspose | _TmaTranspose", tuple[cute.Tensor, cute.Tensor]]:
    pack = 4 // word_size if packed else lanes
    dtype = cutlass.Uint32 if packed else _CUTE_WORD[word_size]
    if tma:
        from .transpose_tma import _TmaTranspose

        op = _TmaTranspose(dtype)
    else:
        op = _LayoutTranspose(
            dtype, pack, _layout_tile(word_size, packed), full, normalize, packed, two_d
        )
    tensors = fake_tiled(
        dtype,
        pack,
        packed=packed,
        align=16 if full or packed or tma else word_size,
        shape=shape,
    )
    return op, tensors


def _compile_layout(
    word_size: int, lanes: int, normalize: bool, full: bool, packed: bool, two_d: bool
) -> "Function":
    pack, divisor = (4 // word_size, 4) if packed else (lanes, 1)
    shape = (
        1 if two_d else cute.sym_int64(),
        cute.sym_int64(divisibility=divisor),
        1 if two_d else cute.sym_int64(),
        cute.sym_int64(divisibility=divisor),
        pack,
    )
    op, (src, dst) = tiled_kernel(
        word_size, lanes, normalize, full, packed, two_d=two_d, shape=shape
    )
    return _L.compile_kernel(op, src, dst, _L.stream())


def _compile_payload(
    word_size: int,
    vector_words: int,
    normalize: bool,
    large: bool,
    middle_one: bool,
    matrix_pow2: bool,
) -> "Function":
    word_type = _CUTE_WORD[word_size]
    op = _PayloadTranspose(
        word_type, vector_words, normalize, large, middle_one, matrix_pow2
    )
    size = cute.sym_int64()
    align = vector_words * word_size if vector_words > 1 else None
    src = _L.fake_compact(
        word_type, (size, vector_words), stride_order=(1, 0), align=align
    )
    dst = _L.fake_compact(
        word_type, (size, vector_words), stride_order=(1, 0), align=align
    )
    geometry = tuple(cutlass.Int32(1) for _ in range(7))
    return _L.compile_kernel(op, src, dst, geometry, _L.stream())


def _matrix_options(
    src: torch.Tensor,
    dst: torch.Tensor | None,
    spec: TransposeSpec,
    word_size: int,
    normalize: bool,
) -> tuple[bool, bool]:
    outer, b, middle, a, _, itemsize = spec
    vector = 16 // word_size
    if (
        word_size not in _VECTOR_MIN_BYTES
        or normalize
        or a % vector
        or b % vector
        or const_data_ptr(src) % 16
        or (const_data_ptr(dst) if dst is not None else 16) % 16
    ):
        return False, False
    vectorized = a * b * itemsize >= _VECTOR_MIN_BYTES[word_size]
    tma = (
        outer * middle > 1
        and outer * middle * a * b >= _TMA_MIN_ELEMS[word_size]
        and cap.device_ok(src, (10,))
    )
    return tma, vectorized


def _compile_packed(bits: int, pack: int) -> "Function":
    op = _PackedTranspose(bits, pack)
    size = cute.sym_int64()
    src = _L.fake_compact(cutlass.Uint8, (size,))
    dst = _L.fake_compact(cutlass.Uint8, (size,))
    geometry = tuple(cutlass.Int32(1) for _ in range(5))
    return _L.compile_kernel(op, src, dst, geometry, _L.stream())


class _Plan(NamedTuple):
    path: str
    word_size: int
    lanes: int = 1
    vector_words: int = 1
    normalize: bool = False
    full: bool = False
    vectorized: bool = False
    two_d: bool = False
    large: bool = False
    bits: int = 0
    pack: int = 1
    middle_one: bool = False
    matrix_pow2: bool = False


def _plan(src: torch.Tensor, dst: torch.Tensor | None, spec: TransposeSpec) -> _Plan:
    packed = _PACKED.get(src.dtype)
    if packed is not None:
        bits, pack = packed
        return _Plan("packed", 1, bits=bits, pack=pack)
    outer, b, middle, a, inner, itemsize = spec
    word_size, lanes = _word_config(itemsize)
    normalize = src.dtype is torch.bool
    target_alignment = const_data_ptr(dst) if dst is not None else 16
    if inner == 1:
        tma, vectorized = _matrix_options(src, dst, spec, word_size, normalize)
        if tma:
            return _Plan("tma", word_size)
        pack = 4 // word_size if vectorized else 1
        tile = _layout_tile(word_size, vectorized)
        full = a % (tile * pack) == 0 and b % (tile * pack) == 0
        full = full and const_data_ptr(src) % 16 == 0 and target_alignment % 16 == 0
        return _Plan(
            "layout",
            word_size,
            lanes,
            normalize=normalize,
            full=full,
            vectorized=vectorized,
            two_d=outer == 1 and middle == 1,
        )
    vector_bytes = (
        1
        if normalize
        else math.gcd(
            row_alignment(inner, itemsize), const_data_ptr(src), target_alignment
        )
    )
    vector_words = 1
    if vector_bytes > 1:
        word_size = min(vector_bytes, 4)
        vector_words = vector_bytes // word_size
    inner_vectors = inner * itemsize // (word_size * vector_words)
    return _Plan(
        "payload",
        word_size,
        lanes,
        vector_words,
        normalize,
        large=inner_vectors > _PAYLOAD_CHUNK,
        middle_one=middle == 1,
        matrix_pow2=middle == 1 and a & (a - 1) == 0 and b & (b - 1) == 0,
    )


def _indexing_supported(src: torch.Tensor, spec: TransposeSpec, plan: _Plan) -> bool:
    outer, b, middle, a, inner, itemsize = spec
    count = outer * b * middle * a * inner
    if not 0 < count <= 2**31 - 1 or count != src.numel():
        return False
    if plan.path == "packed":
        words = (count * itemsize + plan.pack - 1) // plan.pack
        grid = (words + _PACKED_THREADS * _PACKED_ELEMENTS_PER_THREAD - 1) // (
            _PACKED_THREADS * _PACKED_ELEMENTS_PER_THREAD
        )
        return grid <= 2**31 - 1
    if plan.path == "payload":
        inner_vectors = inner * itemsize // (plan.word_size * plan.vector_words)
        rows = outer * b * middle * a
        if inner_vectors > 2**31 - 1:
            return False
        chunks = (inner_vectors + _PAYLOAD_CHUNK - 1) // _PAYLOAD_CHUNK
        rows_per_block = max(1, _PAYLOAD_CHUNK // inner_vectors)
        grid = (
            rows * chunks
            if plan.large
            else (rows + rows_per_block - 1) // rows_per_block
        )
        return grid <= 2**31 - 1
    tile = 64 if plan.path == "tma" else _layout_tile(plan.word_size, plan.vectorized)
    pack = 4 // plan.word_size if plan.vectorized else 1
    x, y = (b, a) if plan.path == "tma" else (a // pack, b // pack)
    return (
        (x + tile - 1) // tile <= 2**31 - 1
        and (y + tile - 1) // tile <= 65535
        and outer * middle <= 65535
    )


def indexing_supported(
    src: torch.Tensor, dst: torch.Tensor | None, spec: TransposeSpec
) -> bool:
    """Check the selected copy path's logical count and CUDA launch bounds."""
    return _indexing_supported(src, spec, _plan(src, dst, spec))


def run(src: torch.Tensor, dst: torch.Tensor, spec: TransposeSpec) -> torch.Tensor:
    device = src.get_device()
    check_current(device)
    plan = _plan(src, dst, spec)
    if not _indexing_supported(src, spec, plan):
        raise ValueError(
            "transpose copy exceeds supported indexing or CUDA grid limits"
        )
    geometry = None
    outer, b, middle, a, inner, itemsize = spec
    if plan.path == "packed":
        key = (device, "packed", plan.bits, plan.pack)
        build = partial(_compile_packed, plan.bits, plan.pack)
        src_arg = _word_view(src, 1, plan.pack)
        dst_arg = _word_view(dst, 1, plan.pack)
        geometry = (b, middle, a, inner, src.numel())
    elif plan.path == "payload":
        key = (
            device,
            "payload",
            plan.word_size,
            plan.vector_words,
            plan.normalize,
            plan.large,
            plan.middle_one,
            plan.matrix_pow2,
        )
        build = partial(_compile_payload, *key[2:])
        src_arg = _word_view(src, plan.word_size).view(-1, plan.vector_words)
        dst_arg = _word_view(dst, plan.word_size).view(-1, plan.vector_words)
        divisor = plan.word_size * plan.vector_words
        geometry = (
            b,
            middle,
            a,
            inner * itemsize // divisor,
            outer * b * middle * a,
            a.bit_length() - 1,
            b.bit_length() - 1,
        )
    else:
        if plan.path == "tma":
            from .transpose_tma import compile_tma

            key = (device, "tma", plan.word_size)
            build = partial(compile_tma, plan.word_size)
        else:
            key = (
                device,
                "layout",
                plan.word_size,
                plan.lanes,
                plan.normalize,
                plan.full,
                plan.vectorized,
                plan.two_d,
            )
            build = partial(_compile_layout, *key[2:])
        src_arg = _tiled_view(
            src, spec, plan.word_size, source=True, packed=plan.vectorized
        )
        dst_arg = _tiled_view(
            dst, spec, plan.word_size, source=False, packed=plan.vectorized
        )
    fn = cached_plan(_PLANS, key, build, op="aten::copy_")
    args = (_L.read_only(src_arg), dst_arg)
    if geometry is not None:
        args += (tuple(cutlass.Int32(v) for v in geometry),)
    fn(*args, _L.stream())
    return dst
