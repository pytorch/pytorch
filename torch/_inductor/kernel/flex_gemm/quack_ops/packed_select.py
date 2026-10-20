# mypy: allow-untyped-defs
"""Select between BF16 D values and a scalar on packed 32-bit words.

For ``bf16(where(p, x, c))`` the compiler emits packed BF16 conversion followed
by halfword unpack, select and repack before ptxas. Round-to-nearest,
non-saturating conversion commutes with the selection, ``bf16(p ? x : c) ==
(p ? bf16(x) : bf16(c))``, so selecting whole converted words is bit-exact and
keeps the pairs packed.
"""

import functools
import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, vector
from cutlass.cutlass_dsl import dsl_user_op, T


@functools.cache
def packed_select_cache_key() -> str:
    """Include this module's source in generated kernel cache keys."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dsl_user_op
def packed_bf16_select(pred, value, fill, *, fill_on_true=False, loc=None, ip=None):
    """Compute ``where(pred, value, fill)`` for a BF16 D store.

    ``fill_on_true`` swaps the arms. A Float32 fragment with an even flat size
    returns BF16 words whose masks keep the predicates of both lanes, so
    neighboring elements need not share one. Anything else returns the ordinary
    ``cute.where`` in its original dtype for the D store to convert.
    """
    # Captured bool fragments can reach here as i8 vectors.
    pred = pred != cute.full_like(pred, 0, loc=loc, ip=ip)
    fill_value = cute.full_like(value, fill, loc=loc, ip=ip)
    count = ir.VectorType(value.ir_value().type).shape[0]
    if (
        value.element_type is not cutlass.Float32
        or pred.shape != value.shape
        or count % 2
    ):
        arms = (fill_value, value) if fill_on_true else (value, fill_value)
        return cute.where(pred, *arms, loc=loc, ip=ip)
    converted = value.to(cutlass.BFloat16, loc=loc, ip=ip).ir_value()
    word_type = ir.VectorType.get([count // 2], T.i32())
    words = vector.bitcast(word_type, converted, loc=loc, ip=ip)
    fill_bits = fill_value.to(cutlass.BFloat16, loc=loc, ip=ip).ir_value()
    fill_word = vector.extract(
        vector.bitcast(word_type, fill_bits, loc=loc, ip=ip), [], [0], loc=loc, ip=ip
    )
    zero = arith.constant(T.i32(), 0, loc=loc, ip=ip)
    # Little-endian bitcast: flat element 2k is the low half of word k.
    halves = (
        arith.constant(T.i32(), 0x0000FFFF, loc=loc, ip=ip),
        arith.constant(T.i32(), -0x00010000, loc=loc, ip=ip),
    )
    selected = []
    for index in range(count // 2):
        lane_masks = []
        for lane, half in enumerate(halves):
            lane_pred = vector.extract(
                pred.ir_value(), [], [2 * index + lane], loc=loc, ip=ip
            )
            arms = (zero, half) if fill_on_true else (half, zero)
            lane_masks.append(arith.select(lane_pred, *arms, loc=loc, ip=ip))
        mask = arith.ori(*lane_masks, loc=loc, ip=ip)
        word = vector.extract(words, [], [index], loc=loc, ip=ip)
        diff = arith.xori(word, fill_word, loc=loc, ip=ip)
        diff = arith.andi(diff, mask, loc=loc, ip=ip)
        selected.append(arith.xori(fill_word, diff, loc=loc, ip=ip))
    result = vector.bitcast(
        converted.type,
        vector.from_elements(word_type, selected, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(result, value.shape, cutlass.BFloat16)
