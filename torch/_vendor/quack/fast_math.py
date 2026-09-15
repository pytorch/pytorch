# Copyright (c) 2025, Tri Dao.

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Integer


def ceil_log2(x: Integer) -> Int32:
    """ceil(log2(x)) for 1 <= x < 2^31, as 32 - ctlz(x - 1). The llvm.intr.ctlz
    lowers to lzcnt on the host (launch-prep path) and clz on device;
    is_zero_poison=False makes ctlz(0) == 32, so x == 1 correctly yields 0."""
    xm1 = (Int32(x) - 1).ir_value()
    return Int32(32) - Int32(llvm.intr_ctlz(xm1, False))


class FastDivmod:
    """Magic-number unsigned divmod: q = umulhi(n, magic) >> shift, r = n - q * divisor.

    The multiplier and shift are precomputed on the host (kernel params), so the
    device-side divmod is 3 uniform-datapath ops (IMAD.WIDE.U32 + USHF + IMAD) plus
    a select handling divisor == 1 (magic == 0 sentinel, like nvjet). This is the
    lean form without the add-back correction the stock cute FastDivmodDivisor
    emits: with shift = max(ceil_log2(d) - 1, 0) the multiplier ceil(2^(32+s)/d)
    fits 32 bits (worst case d = 2^(c-1)+1 gives m <= 2^32 - 1) and the result is
    exact for all dividends < 2^31, i.e. any non-negative Int32. Same contract and
    algorithm as C++ cutlass::FastDivmod. Negative dividends are OUT of contract
    (they reinterpret through Uint32 to values >= 2^31 and silently divide wrong);
    use cute.FastDivmodDivisorV2 if signed or full-u32 dividends are ever needed.

    Why we keep this instead of cute.FastDivmodDivisorV2 (DSL >= 4.6.1): V2 must be
    exact for the full u32 dividend range, so it lowers to the two-shift add-back
    form q = (hi + ((n - hi) >> s1)) >> s2 -- measured on sm_90a with DSL 4.6.1 that
    is 7 SASS ops per divmod with a 5-op dependency chain to the quotient
    (IMAD.HI -> IADD3 -> SHF -> IADD3 -> SHF). Ours is 6 ops with a 3-op chain
    (IMAD.HI -> SHF -> SEL; the ISETP feeding the SEL is off the critical path and
    amortizes per divisor). Tile schedulers chain divmods back-to-back on the
    next-work-tile latency path, so we keep the < 2^31 contract and the shorter form.
    """

    def __init__(self, divisor: Integer):
        if isinstance(divisor, int):
            assert 0 < divisor < 1 << 31
            divisor = Int32(divisor)  # constants fold through the arithmetic below
        self.divisor = divisor
        # Runs on the CPU at launch prep (host-side jit); the udiv is once per launch.
        s = cutlass.max(ceil_log2(divisor) - 1, Int32(0))
        pow_s = Uint64(Uint32(1) << Uint32(s))
        numer = Uint64(0x100000000) * pow_s
        magic = (numer + Uint64(Uint32(divisor)) - 1) // Uint64(Uint32(divisor))
        self.magic = Uint32(magic & 0xFFFFFFFF)  # 0 when divisor == 1
        self.shift = Uint32(s)

    def __rdivmod__(self, dividend: Integer):
        q = Uint32(cute.arch.mul_hi(Uint32(dividend), self.magic)) >> self.shift
        # divisor == 1 sentinel: magic wrapped to 0, so q is 0; select the dividend
        # instead. The remainder then self-corrects: r = n - n * 1 = 0.
        q = Int32(cutlass.select_(self.magic == Uint32(0), Int32(dividend), Int32(q)))
        r = Int32(dividend) - q * Int32(self.divisor)
        return q, r

    def __rfloordiv__(self, dividend: Integer) -> Int32:
        q, _ = self.__rdivmod__(dividend)
        return q

    def __rmod__(self, dividend: Integer) -> Int32:
        _, r = self.__rdivmod__(dividend)
        return r

    def __extract_mlir_values__(self):
        values = []
        self._values_pos = []
        for obj in [self.magic, self.shift, self.divisor]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        new_obj = object.__new__(FastDivmod)
        for name, n_items in zip(["magic", "shift", "divisor"], self._values_pos):
            setattr(
                new_obj, name, cutlass.new_from_mlir_values(getattr(self, name), values[:n_items])
            )
            values = values[n_items:]
        new_obj._values_pos = self._values_pos
        return new_obj
