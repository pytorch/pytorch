"""Shared pieces of the CuTeDSL sort kernels: dtype tables, the radix ordinal
map and compile-time operand descriptors.

Keys are handled as raw bits of their width. ``radix_ordinal`` maps them to an
unsigned ordinal whose ascending order is the order of the stock CUDA sort
(cub radix sort): ``-0.0 == +0.0``, negative NaN first and positive NaN last;
``descending`` reverses it, and stable kernels keep ties in input order in
both directions.
"""

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int16, Int32, Int64, Int8, Uint16, Uint32, Uint64, Uint8

import torch
from torch._native.cutedsl.launch import fake_compact, sym


RADIX_BITS = 8
RADIX_BINS = 1 << RADIX_BITS

# Ordinal kind and width in bits for each supported key dtype.
KEY_KINDS = {
    torch.bool: ("unsigned", 8),
    torch.uint8: ("unsigned", 8),
    torch.int8: ("signed", 8),
    torch.int16: ("signed", 16),
    torch.float16: ("float", 16),
    torch.bfloat16: ("float", 16),
    torch.int32: ("signed", 32),
    torch.float32: ("float", 32),
    torch.int64: ("signed", 64),
    torch.float64: ("float", 64),
}
RAW_TORCH = {8: torch.uint8, 16: torch.uint16, 32: torch.uint32, 64: torch.uint64}
RAW_CUTE = {8: Uint8, 16: Uint16, 32: Uint32, 64: Uint64}
INDEX_DTYPES = {
    torch.uint8: Uint8,
    torch.int8: Int8,
    torch.uint16: Uint16,
    torch.int16: Int16,
    torch.uint32: Uint32,
    torch.int32: Int32,
    torch.int64: Int64,
}


def raw_view(t: torch.Tensor) -> torch.Tensor:
    """View a key tensor as unsigned integers of the same width."""
    return t.view(RAW_TORCH[KEY_KINDS[t.dtype][1]])


def index_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def fake_rows(dtype: type[cutlass.Numeric], rows=None, n=None) -> cute.Tensor:
    """Row-major ``(rows, n)`` compile-time operand with dynamic extents."""
    return fake_compact(dtype, (rows or sym(), n or sym()), stride_order=(1, 0))


@cute.jit
def radix_ordinal(
    bits: Int32,
    kind: cutlass.Constexpr,
    key_bits: cutlass.Constexpr,
    descending: cutlass.Constexpr,
    keep_negative_zero: cutlass.Constexpr = False,
) -> Int32:
    """Map raw key bits to an unsigned ordinal in the low ``key_bits`` bits.

    ``bits`` and the result are Int32 for keys of up to 32 bits and Int64 for
    64-bit keys. ``keep_negative_zero`` gives cub's invertible ``to_bit_ordered``,
    whose digit extractor ties -0.0 with +0.0 instead."""
    word = Int64 if const_expr(key_bits == 64) else Int32
    full = const_expr(key_bits in (32, 64))
    mask = word(-1) if full else word((1 << key_bits) - 1)
    sign = word(-(1 << (key_bits - 1))) if full else word(1 << (key_bits - 1))
    ord_ = bits
    if const_expr(kind == "signed"):
        ord_ = bits ^ sign
    elif const_expr(kind == "float"):
        if const_expr(not keep_negative_zero):
            if (bits & (mask ^ sign)) == word(0):
                bits = word(0)  # -0.0 ties with +0.0, as in the stock leaf
        if (bits & sign) != word(0):
            ord_ = ~bits & mask
        else:
            ord_ = bits | sign
    if const_expr(descending):
        ord_ = ~ord_ & mask
    return ord_


@cute.jit
def radix_ordinal_inverse(
    ord_: Int32,
    kind: cutlass.Constexpr,
    key_bits: cutlass.Constexpr,
    descending: cutlass.Constexpr,
) -> Int32:
    """Raw key bits of a ``radix_ordinal(..., keep_negative_zero=True)``."""
    word = Int64 if const_expr(key_bits == 64) else Int32
    full = const_expr(key_bits in (32, 64))
    mask = word(-1) if full else word((1 << key_bits) - 1)
    sign = word(-(1 << (key_bits - 1))) if full else word(1 << (key_bits - 1))
    if const_expr(descending):
        ord_ = ~ord_ & mask
    bits = ord_
    if const_expr(kind == "signed"):
        bits = ord_ ^ sign
    elif const_expr(kind == "float"):
        if (ord_ & sign) != word(0):
            bits = ord_ ^ sign
        else:
            bits = ~ord_ & mask
    return bits


@cute.jit
def match_any(label: Int32, label_bits: cutlass.Constexpr) -> Int32:
    """Lanes of a full warp whose low ``label_bits`` of ``label`` equal this
    lane's: cub ``MatchAny``, one ballot per bit, since ``match.any.sync`` is
    slower when matching a few bits."""
    peers = Int32(-1)
    for b in cutlass.range_constexpr(label_bits):
        bit = (label >> Int32(b)) & Int32(1)
        # Lanes that share this bit: the ballot if it is set, its complement if not.
        peers = peers & (
            Int32(cute.arch.vote_ballot_sync(bit != Int32(0))) ^ (bit - Int32(1))
        )
    return peers
