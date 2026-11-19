import math
from functools import partial
from typing import TYPE_CHECKING

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.memory import SmemAllocator

import torch

from ...cutedsl import launch as _L
from ...cutedsl.config import (
    TRANSPOSE_TMA_MIN_ELEMS as _TMA_MIN_ELEMS,
    TRANSPOSE_VECTOR_MIN_BYTES as _VECTOR_MIN_BYTES,
)
from ...cutedsl.plan_cache import cached_plan
from ...cutedsl.types import NumericType
from ...utils import capability as cap
from ...utils.device import check_current
from ...utils.tensor import const_data_ptr, row_alignment
from ._packing import (
    flat_copy,
    FLAT_ELEMENTS_PER_THREAD as _PACKED_ELEMENTS_PER_THREAD,
    FLAT_THREADS as _PACKED_THREADS,
    transpose_words,
)
from ._transpose import fake_tiled, matrix_view, predicate, source_offset
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
        self.kernel(src, dst, read, write, layout).launch(
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
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        row, col, batch = cute.arch.block_idx()
        outer, middle = batch // src.shape[4], batch % src.shape[4]
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
        b: int,
        middle: int,
        a: int,
        inner_words: int,
        vector_words: int,
        normalize: bool,
    ) -> None:
        self.word_type: NumericType = word_type
        self.b: int = b
        self.middle: int = middle
        self.a: int = a
        self.vector_words: int = vector_words
        self.inner: int = inner_words // vector_words
        self.normalize: bool = normalize
        self.rows_per_block: int = max(1, _PAYLOAD_CHUNK // self.inner)
        self.chunks: int = (self.inner + _PAYLOAD_CHUNK - 1) // _PAYLOAD_CHUNK

    @cute.jit
    def __call__(
        self, src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream
    ) -> None:
        rows = cute.size(dst) // (self.inner * self.vector_words)
        if cutlass.const_expr(self.inner > _PAYLOAD_CHUNK):
            grid = rows * self.chunks
        else:
            grid = cute.ceil_div(rows, self.rows_per_block)
        self.kernel(src, dst).launch(
            grid=[grid, 1, 1],
            block=[_PAYLOAD_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, src: cute.Tensor, dst: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        rows = cute.size(dst) // (self.inner * self.vector_words)
        for i in cutlass.range_constexpr(_PAYLOAD_VECTORS_PER_THREAD):
            local = tidx + i * _PAYLOAD_THREADS
            if cutlass.const_expr(self.inner <= _PAYLOAD_CHUNK):
                dst_row = bidx * self.rows_per_block + local // self.inner
                q = local % self.inner
                active = local < self.rows_per_block * self.inner
            else:
                dst_row = bidx // self.chunks
                q = (bidx % self.chunks) * _PAYLOAD_CHUNK + local
                active = q < self.inner
            if dst_row < rows and active:
                src_idx = source_offset(
                    dst_row, q, self.b, self.middle, self.a, self.inner
                )
                dst_idx = dst_row * self.inner + q
                if cutlass.const_expr(self.normalize):
                    value = src[src_idx, 0]
                    value = value != self.word_type(0)
                    dst[dst_idx, 0] = value
                else:
                    cute.autovec_copy(src[src_idx, None], dst[dst_idx, None])


class _PackedTranspose:
    def __init__(self, spec: TransposeSpec, bits: int, pack: int) -> None:
        outer, self.b, self.middle, self.a, self.inner, _ = spec
        self.numel: int = outer * self.b * self.middle * self.a * self.inner
        self.bits: int = bits
        self.pack: int = pack

    @cute.jit
    def __call__(
        self, src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream
    ) -> None:
        geometry = (self.b, self.middle, self.a, self.inner, self.numel)
        flat_copy(src, dst, geometry, self.bits, self.pack, False).launch(
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
    shape: tuple[int | cute.SymInt, ...] | None = None,
) -> tuple["_LayoutTranspose | _TmaTranspose", tuple[cute.Tensor, cute.Tensor]]:
    pack = 4 // word_size if packed else lanes
    dtype = cutlass.Uint32 if packed else _CUTE_WORD[word_size]
    if tma:
        from .transpose_tma import _TmaTranspose

        op = _TmaTranspose(dtype)
    else:
        op = _LayoutTranspose(
            dtype, pack, _layout_tile(word_size, packed), full, normalize, packed
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
    shape = (1, _L.sym(divisor), 1, _L.sym(divisor), pack) if two_d else None
    op, (src, dst) = tiled_kernel(
        word_size, lanes, normalize, full, packed, shape=shape
    )
    return _L.compile_kernel(op, src, dst, _L.stream())


def _compile_payload(
    word_size: int,
    b: int,
    middle: int,
    a: int,
    inner_words: int,
    vector_words: int,
    normalize: bool,
) -> "Function":
    word_type = _CUTE_WORD[word_size]
    op = _PayloadTranspose(
        word_type, b, middle, a, inner_words, vector_words, normalize
    )
    size = _L.sym()
    vector_bytes = vector_words * word_size
    align = vector_bytes if vector_words > 1 else None
    src = _L.fake_compact(
        word_type, (size, vector_words), stride_order=(1, 0), align=align
    )
    dst = _L.fake_compact(
        word_type, (size, vector_words), stride_order=(1, 0), align=align
    )
    return _L.compile_kernel(op, src, dst, _L.stream())


def _matrix_options(
    src: torch.Tensor,
    dst: torch.Tensor,
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
        or const_data_ptr(dst) % 16
    ):
        return False, False
    vectorized = a * b * itemsize >= _VECTOR_MIN_BYTES[word_size]
    tma = (
        outer * middle > 1
        and outer * middle * a * b >= _TMA_MIN_ELEMS[word_size]
        and cap.device_ok(src, (10,))
    )
    return tma, vectorized


def _compile_packed(spec: TransposeSpec, bits: int, pack: int) -> "Function":
    op = _PackedTranspose(spec, bits, pack)
    size = _L.sym()
    src = _L.fake_compact(cutlass.Uint8, (size,))
    dst = _L.fake_compact(cutlass.Uint8, (size,))
    return _L.compile_kernel(op, src, dst, _L.stream())


def run(src: torch.Tensor, dst: torch.Tensor, spec: TransposeSpec) -> torch.Tensor:
    device = src.get_device()
    check_current(device)
    packed = _PACKED.get(src.dtype)
    if packed is not None:
        bits, pack = packed
        key = (device, "packed", *spec[:-1], bits, pack)
        build = partial(_compile_packed, spec, bits, pack)
        src_arg = _word_view(src, 1, pack)
        dst_arg = _word_view(dst, 1, pack)
    else:
        outer, b, middle, a, inner, itemsize = spec
        word_size, lanes = _word_config(itemsize)
        normalize = src.dtype is torch.bool
        if inner == 1:
            tma, vectorized = _matrix_options(src, dst, spec, word_size, normalize)
            if tma:
                from .transpose_tma import compile_tma

                key = ("tma", device, word_size, outer, middle, a, b)
                build = partial(compile_tma, word_size, outer, middle, a, b)
            else:
                pack = 4 // word_size if vectorized else 1
                tile = _layout_tile(word_size, vectorized)
                full = a % (tile * pack) == 0 and b % (tile * pack) == 0
                full = (
                    full
                    and const_data_ptr(src) % 16 == 0
                    and const_data_ptr(dst) % 16 == 0
                )
                two_d = outer == 1 and middle == 1
                key = (
                    device,
                    "layout",
                    word_size,
                    lanes,
                    normalize,
                    full,
                    vectorized,
                    two_d,
                )
                build = partial(
                    _compile_layout,
                    word_size,
                    lanes,
                    normalize,
                    full,
                    vectorized,
                    two_d,
                )
            src_arg = _tiled_view(
                src, spec, word_size, source=True, packed=vectorized and not tma
            )
            dst_arg = _tiled_view(
                dst, spec, word_size, source=False, packed=vectorized and not tma
            )
        else:
            inner_words = inner * lanes
            vector_bytes = (
                1
                if normalize
                else math.gcd(
                    row_alignment(inner, itemsize),
                    const_data_ptr(src),
                    const_data_ptr(dst),
                )
            )
            if vector_bytes > 1:
                word_size = min(vector_bytes, 4)
                inner_words = inner * itemsize // word_size
                vector_words = vector_bytes // word_size
            else:
                vector_words = 1
            key = (
                device,
                "payload",
                word_size,
                b,
                middle,
                a,
                inner_words,
                vector_words,
                normalize,
            )
            build = partial(
                _compile_payload,
                word_size,
                b,
                middle,
                a,
                inner_words,
                vector_words,
                normalize,
            )
            src_arg = _word_view(src, word_size).view(-1, vector_words)
            dst_arg = _word_view(dst, word_size).view(-1, vector_words)
    fn = cached_plan(_PLANS, key, build, op="aten::copy_")
    fn(_L.read_only(src_arg), dst_arg, _L.stream())
    return dst
