# Copyright (c) 2026, Han Guo, Tri Dao.
"""Value vocabulary for epilogue fns: paired/packed lane types (:class:`Pair`,
:class:`F2`, :class:`F16Lanes`) and tuple-polymorphic transcendentals
(:func:`pexp`, :func:`pexp2`) — raw ``cute.math`` fns are not F2-aware, so
mods needing them use these wrappers. Scalar-or-f32x2 contract as
``quack.activation``, whose functions compose directly with these types."""

from typing import NamedTuple

import cutlass.cute as cute
from cutlass import Float32

__all__ = ["F2", "F16Lanes", "Pair", "pack", "pexp", "pexp2", "unpack"]


class Pair(NamedTuple):
    """A two-lanes-per-logical-element epilogue value.

    Pairing is declared with ``mode=`` — the fn body calls ``unpack``/``pack``
    where it uses the lanes:

    * aux output buffer at half of GEMM-N — the accumulator pairs over
      adjacent N columns (gated): ``gate, up = unpack(acc)``; aux values are
      per-pair, and returning ``"D": pack(g, u)`` writes both lanes back.
    * 16-bit C at twice GEMM-N — C and D pack two lanes per 32-bit element
      (dgated): ``x, y = unpack(c)``, return ``"D": pack(dx, dy)``; pass C/D
      as their natural 16-bit tensors.

    As a value it is a plain tuple of the two lanes with lane-wise ``+ - *``
    (scalars broadcast), so ``acc * rstd + bias`` works before unpacking."""

    a: object
    b: object

    @staticmethod
    def _lift(v):
        return v if isinstance(v, tuple) else (v, v)

    def __add__(self, other):
        o = Pair._lift(other)
        return Pair(self.a + o[0], self.b + o[1])

    __radd__ = __add__

    def __mul__(self, other):
        o = Pair._lift(other)
        return Pair(self.a * o[0], self.b * o[1])

    __rmul__ = __mul__

    def __sub__(self, other):
        o = Pair._lift(other)
        return Pair(self.a - o[0], self.b - o[1])

    def __rsub__(self, other):
        o = Pair._lift(other)
        return Pair(o[0] - self.a, o[1] - self.b)

    def __neg__(self):
        return Pair(-self.a, -self.b)


def unpack(value):
    """Split a paired epilogue value into its two lanes: ``x, y = unpack(c)``.
    Fails loudly at trace time if the tensors didn't imply pairing."""
    assert isinstance(value, Pair), (
        "unpack() got a non-paired value. Declare mode='acc_pair' to pair adjacent "
        "accumulator lanes or mode='packed_cd_b16x2' to unpack 16-bit C/D lanes."
    )
    return value.a, value.b


pack = Pair  # returning {"D": pack(dx, dy)} packs both lanes back


class F2(NamedTuple):
    """A packed f32x2 lane pair. IS a tuple, so ``quack.activation`` functions
    take it on their packed path; arithmetic lowers to packed intrinsics.
    Scalar operands broadcast: ``x * alpha`` and ``alpha * x`` both work."""

    lo: object
    hi: object

    @staticmethod
    def _pair(v):
        return v if isinstance(v, tuple) else (v, v)

    def __add__(self, other):
        if isinstance(other, F16Lanes):
            return other.__radd__(self)
        return F2(*cute.arch.add_packed_f32x2(self, F2._pair(other)))

    __radd__ = __add__

    def __mul__(self, other):
        return F2(*cute.arch.mul_packed_f32x2(self, F2._pair(other)))

    __rmul__ = __mul__

    def __sub__(self, other):
        return F2(*cute.arch.sub_packed_f32x2(self, F2._pair(other)))

    def __rsub__(self, other):
        return F2(*cute.arch.sub_packed_f32x2(F2._pair(other), self))

    def __neg__(self):
        return F2(-self.lo, -self.hi)

    def fma(self, mul, add):
        """self * mul + add as one packed FMA."""
        return F2(*cute.arch.fma_packed_f32x2(self, F2._pair(mul), F2._pair(add)))


class F16Lanes(F2):
    """An F2 whose lanes were promoted from a 16-bit float C fragment (fp16 OR
    bf16 — "f16" as in floating-point compute; the PTX forms below take both
    .atypes), remembering the raw 16-bit lanes. Semantically it IS the promoted F2 (activation fns,
    muls, packed intrinsics — every existing use behaves identically), but the
    operations with a mixed-precision ISA form pick the scalar lowering where
    the promote folds into the op, exactly:

    * ``x + c`` / ``c + x`` -> PTX ``add.rn.f32.{f16,bf16}`` -> SASS FHADD
    * ``c - x``             -> PTX ``sub.rn.f32.{f16,bf16}`` -> FHADD w/ neg
      (``x - c`` has no mixed form; it materializes like everything else)

    When only these consume the value, the eager promotes emitted here are
    dead code and NVVM removes them. Not yet exploited: ``fma.rn.f32.abtype``
    (BOTH multiplicands 16-bit -> FHFMA, always bitwise-safe because a 16-bit
    x 16-bit product is exact in f32) — needs a lazy-product value type and a
    consumer; no current epilogue fn multiplies two raw 16-bit operands."""

    def __new__(cls, a16, b16):
        self = super().__new__(cls, a16.to(Float32), b16.to(Float32))
        self._a16 = a16
        self._b16 = b16
        return self

    def __add__(self, other):
        if isinstance(other, F16Lanes):
            # both sides 16-bit: promote one side, mixed-add the other
            other = other._f2()
        if isinstance(other, F2):
            return F2(other.lo + self._a16.to(Float32), other.hi + self._b16.to(Float32))
        return self._f2() + other

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, F16Lanes):
            other = other._f2()
        if isinstance(other, F2):
            return F2(self._a16.to(Float32) - other.lo, self._b16.to(Float32) - other.hi)
        return self._f2() - other

    def _f2(self):
        return F2(self.lo, self.hi)


def pexp(v):
    """Tuple-polymorphic exp: packed f32x2 on pairs, fastmath scalar otherwise."""
    if isinstance(v, tuple):
        return F2(*cute.arch.exp_packed_f32x2(v))
    return cute.math.exp(v, fastmath=True)


def pexp2(v):
    """Tuple-polymorphic exp2 (no packed exp2 intrinsic: lane-wise fastmath)."""
    if isinstance(v, tuple):
        return F2(pexp2(v[0]), pexp2(v[1]))
    return cute.math.exp2(v, fastmath=True)
