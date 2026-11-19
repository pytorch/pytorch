from typing import Any

import cutlass
import cutlass.cute as cute

from ...cutedsl.types import Index
from ._transpose import source_offset


FLAT_THREADS: int = 256
FLAT_ELEMENTS_PER_THREAD: int = 4


@cute.kernel
def flat_copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    geometry: tuple[Index, Index, Index, Index, Index],
    bits: cutlass.Constexpr,
    pack: cutlass.Constexpr,
    normalize: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    base = bid * (FLAT_THREADS * FLAT_ELEMENTS_PER_THREAD) + tid
    transpose_flat(src, dst, base, geometry, bits, pack, normalize)


@cute.jit
def transpose_words(
    words: tuple[cutlass.Uint32, ...], pack: cutlass.Constexpr
) -> tuple[cutlass.Uint32, ...]:
    if cutlass.const_expr(pack == 2):
        a, b = words
        return (
            cutlass.Uint32(cute.arch.prmt(a, b, 0x5410)),
            cutlass.Uint32(cute.arch.prmt(a, b, 0x7632)),
        )
    a = cute.arch.prmt(words[0], words[1], 0x5140)
    b = cute.arch.prmt(words[0], words[1], 0x7362)
    c = cute.arch.prmt(words[2], words[3], 0x5140)
    d = cute.arch.prmt(words[2], words[3], 0x7362)
    return (
        cutlass.Uint32(cute.arch.prmt(a, c, 0x5410)),
        cutlass.Uint32(cute.arch.prmt(a, c, 0x7632)),
        cutlass.Uint32(cute.arch.prmt(b, d, 0x5410)),
        cutlass.Uint32(cute.arch.prmt(b, d, 0x7632)),
    )


@cute.jit
def packed_byte(
    src: cute.Tensor,
    dst: cute.Tensor,
    byte_idx: Index,
    b: Any,
    middle: Any,
    a: Any,
    inner: Any,
    numel: Any,
    bits: cutlass.Constexpr,
    pack: cutlass.Constexpr,
) -> cutlass.Uint8:
    result = cutlass.Int32(0)
    for lane in cutlass.range_constexpr(pack):
        index = byte_idx * pack + lane
        if index < numel:
            source = source_offset(index // inner, index % inner, b, middle, a, inner)
            value = (cutlass.Int32(src[source // pack]) >> ((source % pack) * bits)) & (
                (1 << bits) - 1
            )
            result |= value << (lane * bits)
    if numel % pack != 0:
        if byte_idx == numel // pack:
            mask = (1 << ((numel % pack) * bits)) - 1
            result |= cutlass.Int32(dst[byte_idx]) & (255 ^ mask)
    return cutlass.Uint8(result)


@cute.jit
def transpose_flat(
    src: cute.Tensor,
    dst: cute.Tensor,
    base: Index,
    geometry: tuple[Index, Index, Index, Index, Index],
    bits: cutlass.Constexpr,
    pack: cutlass.Constexpr,
    normalize: cutlass.Constexpr,
) -> None:
    b, middle, a, inner, _ = geometry
    for k in cutlass.range_constexpr(FLAT_ELEMENTS_PER_THREAD):
        index = base + k * FLAT_THREADS
        if index < cute.size(dst):
            if cutlass.const_expr(pack > 1):
                dst[index] = packed_byte(src, dst, index, *geometry, bits, pack)
            else:
                at = source_offset(index // inner, index % inner, b, middle, a, inner)
                value = src[at]
                if cutlass.const_expr(normalize):
                    value = value != 0
                dst[index] = dst.element_type(value)
