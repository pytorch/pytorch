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


def make_nnc(var, dN=None):
    dN = dN or var

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
        [A, B, C, var])


def main(n=16):
    cg = make_nnc(te.VarHandle("n", te.Dtype.Int))

    tA = torch.randn(n, requires_grad=False)
    tB = torch.randn(n, requires_grad=False)
    result1 = None
    result2 = None

    def nnc_fn():
        nonlocal result1
        result1 = cg.call_jansel(tA, tB)

    def aten_fn():
        nonlocal result2
        result2 = torch.add(tA, tB)

    nnc = np.median(timeit.repeat(nnc_fn, number=1000, repeat=100))
    aten = np.median(timeit.repeat(aten_fn, number=1000, repeat=100))

    torch.testing.assert_allclose(result1, result2)
    print(f"n={n:4} nnc={nnc:.5f} aten={aten:.5f} aten/nnc={aten / nnc:.2f}x")


if __name__ == '__main__':
    main(1)
    main(64)
    main(4096)
