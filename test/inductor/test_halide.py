# Owner(s): ["oncall: pt2"]
import functools
import textwrap
import unittest

import torch
from torch._inductor.codecache import HalideCodeCache
from torch._inductor.runtime.hints import HalideInputSpec, HalideMeta
from torch._inductor.test_case import run_tests, TestCase

from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.inductor_utils import HAS_CPU


try:
    import halide

    HAS_HALIDE = halide is not None
except ImportError:
    HAS_HALIDE = False

requires_halide = functools.partial(unittest.skipUnless, HAS_HALIDE, "requires halide")


class HalideTests(TestCase):
    @requires_halide()
    def test_codecache(self):
        fn = HalideCodeCache.generate_halide(
            HalideMeta(
                argtypes=[
                    HalideInputSpec(
                        ctype="float*", name="in_ptr0", numel="static_cast<long>(1024L)"
                    ),
                    HalideInputSpec(
                        ctype="float*", name="in_ptr1", numel="static_cast<long>(1024L)"
                    ),
                    HalideInputSpec(
                        ctype="float*",
                        name="out_ptr0",
                        numel="static_cast<long>(1024L)",
                    ),
                ],
                scheduler="Mullapudi2016",
            ),
            textwrap.dedent(
                """
            @hl.generator(name="kernel")
            class Kernel:
                in_ptr0 = hl.InputBuffer(hl.Float(32), 1)
                in_ptr1 = hl.InputBuffer(hl.Float(32), 1)
                out_ptr0 = hl.OutputBuffer(hl.Float(32), 1)

                def generate(g):
                    in_ptr0 = g.in_ptr0
                    in_ptr1 = g.in_ptr1
                    out_ptr0 = g.out_ptr0
                    xindex = hl.Var('xindex')
                    xindex_dom = hl.RDom([hl.Range(0, 1024)], 'xindex').x
                    x0 = xindex
                    x0_dom = xindex_dom
                    tmp0 = hl.Func()
                    tmp0[xindex] = in_ptr0[x0]
                    tmp1 = hl.Func()
                    tmp1[xindex] = in_ptr1[x0]
                    tmp2 = hl.Func()
                    tmp2[xindex] = tmp0[xindex] + tmp1[xindex]
                    out_ptr0[hl.Var()] = hl.undef(out_ptr0.type())
                    out_ptr0[x0_dom] = tmp2[xindex_dom]

                    assert g.using_autoscheduler()
                    in_ptr0.set_estimates([hl.Range(1024, 1024)])
                    in_ptr1.set_estimates([hl.Range(1024, 1024)])
                    out_ptr0.set_estimates([hl.Range(1024, 1024)])
        """
            ),
        )
        a = torch.randn(1024)
        b = torch.randn(1024)
        c = torch.randn(1024)
        fn(a, b, c)
        self.assertEqual(c, a + b)


if __name__ == "__main__":
    if HAS_CPU and not IS_MACOS and HAS_HALIDE:
        run_tests(needs="filelock")
