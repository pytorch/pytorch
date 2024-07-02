# Owner(s): ["oncall: pt2"]
import os
import sys
import textwrap
import unittest

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._inductor import config
from torch._inductor.codecache import HalideCodeCache
from torch._inductor.runtime.hints import HalideInputSpec, HalideMeta
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import parallel_num_threads

from torch.testing._internal.common_utils import IS_CI, IS_MACOS, IS_WINDOWS
from torch.testing._internal.inductor_utils import HAS_CPU


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor_dynamic_shapes yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    import halide

    HAS_HALIDE = halide is not None
except ImportError:
    HAS_HALIDE = False


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor


make_halide = config.patch(
    {
        "cpu_backend": "halide",
        "cuda_backend": "halide",
        "fallback_random": True,  # TODO(jansel): support random
    }
)


@unittest.skipUnless(HAS_HALIDE, "requires halide")
class HalideTests(TestCase):
    def test_codecache(self):
        fn = HalideCodeCache.generate_halide(
            HalideMeta(
                argtypes=[
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr1",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    HalideInputSpec(
                        ctype="float*",
                        name="out_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                ],
                target="host-no_runtime",
                scheduler="Mullapudi2016",
                scheduler_flags={
                    "parallelism": parallel_num_threads(),
                },
            ),
            textwrap.dedent(
                """
                import halide as hl

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
                        x0 = xindex
                        tmp0 = hl.Func()
                        tmp0[xindex] = in_ptr0[x0]
                        tmp1 = hl.Func()
                        tmp1[xindex] = in_ptr1[x0]
                        tmp2 = hl.Func()
                        tmp2[xindex] = tmp0[xindex] + tmp1[xindex]
                        out_ptr0[x0] = tmp2[xindex]

                        assert g.using_autoscheduler()
                        in_ptr0.set_estimates([hl.Range(1024, 1024)])
                        in_ptr1.set_estimates([hl.Range(1024, 1024)])
                        out_ptr0.set_estimates([hl.Range(1024, 1024)])

                __name__ == '__main__' and hl.main()
                """
            ),
        )
        a = torch.randn(1024)
        b = torch.randn(1024)
        c = torch.randn(1024)
        fn(a, b, c)
        self.assertEqual(c, a + b)


if test_torchinductor.HAS_CPU and HAS_HALIDE:
    SweepInputsCpuHalideTest = make_halide(test_torchinductor.SweepInputsCpuTest)
    CpuHalideTests = make_halide(test_torchinductor.CpuTests)

if (
    test_torchinductor.HAS_GPU
    and HAS_HALIDE
    and os.environ.get("TEST_HALIDE_GPU") == "1"
):
    SweepInputsGPUHalideTest = make_halide(test_torchinductor.SweepInputsGPUTest)
    GPUHalideTests = make_halide(test_torchinductor.GPUTests)

if __name__ == "__main__":
    if HAS_CPU and not IS_MACOS and HAS_HALIDE:
        run_tests(needs="filelock")
