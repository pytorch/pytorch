# Copyright (c) 2025, Tri Dao.

import cutlass
import cutlass.cute as cute
from cutlass import Int32


@cute.jit
def clz(x: Int32) -> Int32:
    # for i in cutlass.range_constexpr(32):
    #     if (1 << (31 - i)) & x:
    #         return Int32(i)
    # return Int32(32)
    # Early exit is not supported yet
    res = Int32(32)
    done = False
    for i in cutlass.range(32):
        if ((1 << (31 - i)) & x) and not done:
            res = Int32(i)
            done = True
    return res
