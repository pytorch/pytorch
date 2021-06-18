# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._C._te as te
import torch.fx as fx
import torch.utils._pytree as pytree
from torch.fx import map_arg
from torch.fx.passes.shape_prop import ShapeProp
import operator
import functools
import numpy as np
import timeit
import ctypes
import array
import struct
scope = te.KernelScope()


def make_nnc(dN):
    A = te.Placeholder([dN], torch.float32)
    B = te.Placeholder([dN])

    def compute(i):
        return A.load([i]) + B.load([i])

    C = te.Compute('C', [dN], compute)

    loopnest = te.LoopNest([C])
    loopnest.prepare_for_codegen()
    stmt = te.simplify(loopnest.root_stmt())

    return te.construct_codegen(
        'llvm',
        stmt,
        [A, B, C])


def main(n=16):
    # cc = te.CompileCache(lambda: make_nnc(te.VarHandle("n", te.Dtype.Int)))
    cc = te.CompileCache(lambda: make_nnc(te.ExprHandle.int(n)))

    tA = torch.randn(n)
    tB = torch.randn(n)
    result1 = torch.randn(n)
    result2 = torch.randn(n)

    def nnc_fn():
        cc(tA, tB, result1)

    def aten_fn():
        torch.add(tA, tB, out=result2)


    if False:
        nnc_fn()
        aten_fn()
        torch.testing.assert_allclose(result1, result2)
        print("ok")
        return

    nnc = np.median(timeit.repeat(nnc_fn, number=1000, repeat=100))
    aten = np.median(timeit.repeat(aten_fn, number=1000, repeat=100))

    torch.testing.assert_allclose(result1, result2)
    print(f"n={n:4} nnc={nnc:.5f} aten={aten:.5f} aten/nnc={aten / nnc:.2f}x")


if __name__ == '__main__':
    main(1)
    main(64)
    main(4096)
