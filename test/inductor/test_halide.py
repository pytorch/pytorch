# Owner(s): ["oncall: pt2"]
import functools
import itertools
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
from torch.utils._triton import has_triton


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor_dynamic_shapes yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    import halide  # @manual

    HAS_HALIDE = halide is not None
except ImportError:
    HAS_HALIDE = False


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library


make_halide = config.patch(
    {
        "halide.scan_kernels": True,
        "cpu_backend": "halide",
        "cuda_backend": "halide",
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

    def test_manual_schedule(self):
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
                scheduler=None,
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

                        assert not g.using_autoscheduler()
                        i = hl.Var()
                        j = hl.Var()
                        out_ptr0.compute_root()
                        out_ptr0.split(xindex, i, j, 32)
                        out_ptr0.parallel(i)
                        out_ptr0.vectorize(j)
                        tmp2.compute_at(out_ptr0, i)
                        tmp2.store_at(out_ptr0, i)
                        tmp1.compute_inline()

                __name__ == '__main__' and hl.main()
                """
            ),
        )
        a = torch.randn(1024)
        b = torch.randn(1024)
        c = torch.randn(1024)
        fn(a, b, c)
        self.assertEqual(c, a + b)

    @unittest.skipUnless(has_triton(), "requires triton")
    def test_random_consistency(self):
        seed = 1234
        shape = (3, 3)
        dtype = torch.float32

        for (rand_fn,) in itertools.product(
            (
                functools.partial(torch.rand, shape, dtype=dtype, device="cuda"),
                functools.partial(torch.randn, shape, dtype=dtype, device="cuda"),
                functools.partial(
                    torch.randint,
                    -1000,
                    1000,
                    size=shape,
                    dtype=torch.int64,
                    device="cuda",
                ),
            )
        ):

            @torch.compile(backend="inductor", options={"cuda_backend": "halide"})
            def get_rand_halide():
                return rand_fn()

            @torch.compile(backend="inductor", options={"cuda_backend": "triton"})
            def get_rand_triton():
                return rand_fn()

            torch.manual_seed(seed)
            halide_output = get_rand_halide()
            torch.manual_seed(seed)
            triton_output = get_rand_triton()

        self.assertEqual(halide_output, triton_output)


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
