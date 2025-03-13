# Owner(s): ["module: inductor"]
import os
import tempfile
from typing import Any, Callable

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.static_cuda_launcher import StaticallyLaunchedCudaKernel
from torch._inductor.runtime.triton_compat import tl, triton
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.test_case import TestCase
from torch.testing._internal.triton_utils import requires_cuda


@requires_cuda
class TestStaticCudaLauncher(TestCase):
    def setUp(self):
        # Create a temporary file to store the cubin.
        # We set delete=False so that the file persists after closing.
        self.tmp_file = tempfile.NamedTemporaryFile(mode="wb")
        self.tmp_file.close()  # Close now; we'll open it for writing later.
        super().setUp()

    def tearDown(self):
        super().tearDown()
        # Delete the temporary cubin file.
        try:
            os.remove(self.tmp_file.name)
        except FileNotFoundError:
            pass

    def _make_launcher(
        self,
        kernel: Callable,
        args: tuple[Any, ...],
        grid: tuple[Any, ...] = (1,),
    ) -> StaticallyLaunchedCudaKernel:
        """
        Compiles a Triton kernel with the provided *args,
        writes its cubin to the temporary file, and returns the file path.
        """
        fn = triton.jit(kernel)
        # Launch the kernel to trigger compilation.
        compiled_kernel = fn[grid](*args)
        result = StaticallyLaunchedCudaKernel(compiled_kernel)
        result.write_cubin_to_file(self.tmp_file.name)
        result.load_kernel()
        return result

    def test_basic(self):
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            y = arg1
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        arg1 = 5
        args = (arg0, arg1)

        launcher = self._make_launcher(simple_kernel, args, (1,))
        self.assertEqual(arg0, torch.tensor([5], dtype=torch.int32, device="cuda"))
        self.assertEqual(launcher.arg_tys, "Oi")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.run((1,), stream, new_arg0, arg1)
        self.assertEqual(new_arg0, arg0)

    # I wish I could macro all int types this into a single unit test on a loop, but
    # 1. variables aren't allowed as type annotations in python
    # 2. triton relies on inspect.get_source to get the type annotations
    # so I can't even use exec() to generate the test cases.
    # So we'll just make a few kernels by hand
    def test_unsigned_integers(self):
        def unsigned_integers(
            arg0, arg1: tl.uint8, arg2: tl.uint16, arg3: tl.uint32, arg4: tl.uint64
        ):
            x = tl.load(arg0)
            y = arg1 + arg2 + arg3 + arg4
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.uint64, device="cuda")
        # Using small numbers creates a Literal type which triton treats as a constant
        args = (arg0, 50, 50, 50, 50)

        launcher = self._make_launcher(unsigned_integers, args, (1,))
        self.assertEqual(arg0, torch.tensor([200], dtype=torch.uint64, device="cuda"))
        self.assertEqual(launcher.arg_tys, "OBHIK")
        new_arg0 = torch.zeros(1, dtype=torch.uint64, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run((1,), stream, new_arg0, 50, 50, 50, 50)
        self.assertEqual(new_arg0, arg0)

    def test_signed_integers(self):
        def signed_integers(
            arg0, arg1: tl.int8, arg2: tl.int16, arg3: tl.int32, arg4: tl.int64
        ):
            x = tl.load(arg0)
            y = arg1 + arg2 + arg3 + arg4
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int64, device="cuda")
        # Using small numbers creates a Literal type which triton treats as a constant
        args = (arg0, 50, 50, 50, 50)

        launcher = self._make_launcher(signed_integers, args, (1,))
        self.assertEqual(arg0, torch.tensor([200], dtype=torch.int64, device="cuda"))
        self.assertEqual(launcher.arg_tys, "Obhil")
        new_arg0 = torch.zeros(1, dtype=torch.int64, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run((1,), stream, new_arg0, 50, 50, 50, 50)
        self.assertEqual(new_arg0, arg0)

    # TODO: floats don't work properly, triton seems to think they're all tl.float32
    # despite type annotations.
    # There's also not really a good way for me to make a float16 in python...
    def test_floats(self):
        def floats(arg0, arg1: tl.float16, arg2: tl.float32, arg3: tl.float64):
            x = tl.load(arg0)
            y = arg1 + arg2 + arg3
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.float64, device="cuda")

        args = (arg0, 1.0, 1.0, 1.0)

        launcher = self._make_launcher(floats, args, (1,))
        # TODO: in Pytorch's pinned version of triton, arg3 is typed as regular float
        # but in triton 3.3.0, this is fixed and it's 0ffd. We'll need to update later.
        self.assertEqual(launcher.arg_tys, "Offf")
        self.assertEqual(arg0, torch.tensor([3.0], dtype=torch.float64, device="cuda"))
        new_arg0 = torch.zeros(1, dtype=torch.float64, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run((1,), stream, new_arg0, 1.0, 1.0, 1.0)
        self.assertEqual(new_arg0, arg0)

    def test_basic_1arg(self):
        def simple_kernel_1_arg(arg0):
            x = tl.load(arg0)
            tl.store(arg0, x + 1)

        arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        launcher = self._make_launcher(simple_kernel_1_arg, (arg0,), (1,))
        self.assertEqual(arg0, torch.tensor([1], dtype=torch.int32, device="cuda"))
        self.assertEqual(launcher.arg_tys, "O")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.run(
            (1,),
            stream,
            new_arg0,
        )
        self.assertEqual(new_arg0, arg0)

    def test_constexpr(self):
        # Constexprs are compiled directly into the cubin file,
        # so we never need to pass it to StaticCudaLauncher.

        @triton.jit
        def kernel_constexpr(arg0, CONSTANT: tl.constexpr):
            x = tl.load(arg0)
            tl.store(arg0, x + CONSTANT)

        # Can't use make_launcher because constexpr needs to be constant
        arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        compiled_kernel = kernel_constexpr[(1,)](arg0, CONSTANT=5)
        launcher = StaticallyLaunchedCudaKernel(compiled_kernel)
        launcher.write_cubin_to_file(self.tmp_file.name)
        launcher.load_kernel()

        self.assertEqual(arg0, torch.tensor([5], dtype=torch.int32, device="cuda"))
        self.assertEqual(launcher.arg_tys, "O")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(
            (1,),
            stream,
            new_arg0,
        )
        self.assertEqual(new_arg0, arg0)

    def test_implied_constant(self):
        """xnumel is unused in this kernel, but isn't explicitly marked as a constexpr"""

        # This kernel was generated by inductor so it has a bunch of unused arguments. We don't change it
        @triton.jit
        def triton_red_fused_any_isinf_0(
            in_ptr0,
            out_ptr0,
            xnumel,  # noqa: F841
            r0_numel,
            XBLOCK: tl.constexpr,
            R0_BLOCK: tl.constexpr,
        ):
            xnumel = 1  # noqa: F841
            rnumel = r0_numel  # noqa: F841
            RBLOCK: tl.constexpr = R0_BLOCK  # noqa: F841
            xoffset = tl.program_id(0) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # noqa: F841
            xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)  # noqa: F841
            r0_base = tl.arange(0, R0_BLOCK)[None, :]
            rbase = r0_base  # noqa: F841
            _tmp3 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
            for r0_offset in range(0, r0_numel, R0_BLOCK):
                r0_index = r0_offset + r0_base
                r0_mask = r0_index < r0_numel
                roffset = r0_offset  # noqa: F841
                rindex = r0_index  # noqa: F841
                r0_0 = r0_index
                tmp0 = tl.load(
                    in_ptr0 + (r0_0), r0_mask, eviction_policy="evict_first", other=0.0
                )
                tmp1 = libdevice.isinf(tmp0).to(tl.int1)
                tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
                tmp4 = _tmp3 | tmp2
                _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
            tmp3 = triton_helpers.any(_tmp3.to(tl.int8), 1)[:, None].to(tl.int1)
            tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)

        arg0 = torch.tensor([0.0, 0.5, float("inf"), 5], device="cuda")
        arg1 = torch.tensor([False], device="cuda")
        arg2 = torch.tensor([False], device="cuda")
        compiled_kernel = triton_red_fused_any_isinf_0[1,](
            arg0, arg1, 1, 128, XBLOCK=1, R0_BLOCK=1
        )

        launcher = StaticallyLaunchedCudaKernel(compiled_kernel)
        launcher.write_cubin_to_file(self.tmp_file.name)
        launcher.load_kernel()

        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run((1,), stream, arg0, arg2, 1, 128)
        self.assertEqual(arg1, arg2)

    def test_too_many_args(self):
        def kernel_too_many_args(
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            arg10,
            arg11,
            arg12,
            arg13,
            arg14,
            arg15,
            arg16,
            arg17,
            arg18,
            arg19,
            arg20,
            arg21,
            arg22,
            arg23,
            arg24,
            arg25,
        ):
            x = tl.load(arg0)
            y = (
                arg1
                + arg2
                + arg3
                + arg4
                + arg5
                + arg6
                + arg7
                + arg8
                + arg9
                + arg10
                + arg11
                + arg12
                + arg13
                + arg14
                + arg15
                + arg16
                + arg17
                + arg18
                + arg19
                + arg20
                + arg21
                + arg22
                + arg23
                + arg24
                + arg25
            )
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        scalar_args = (50,) * 25
        self.assertRaisesRegex(
            NotImplementedError,
            "No static cuda launcher available",
            lambda: self._make_launcher(kernel_too_many_args, (arg0, *scalar_args)),
        )

    def test_kernel_empty_tensor(self):
        # Triton kernel generated by torch.compile of the following:
        # @torch.compile()
        # def foo(x, y):
        #   return torch.cat(((x * 4), y + 10))

        # Running with example input:
        # torch._dynamo.decorators.mark_unbacked(t, 0)
        # x = torch.rand(0, device="cuda")
        # y = torch.rand(20, device="cuda")

        @triton.jit
        def triton_poi_fused_cat_0(
            in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK: tl.constexpr
        ):
            xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:].to(tl.int64)
            xmask = xindex < xnumel
            x0 = xindex
            tmp0 = x0
            tmp3 = ks0
            tmp4 = tmp0 < tmp3
            tmp5 = tl.load(
                in_ptr0 + (x0), xmask & tmp4, eviction_policy="evict_last", other=0.0
            )
            tmp6 = 4.0
            tmp7 = tmp5 * tmp6
            tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
            tmp9 = tl.where(tmp4, tmp7, tmp8)
            tmp10 = tmp0 >= tmp3
            tmp13 = tl.load(
                in_ptr1 + (x0 + ((-1) * ks0)),
                xmask & tmp10,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp14 = 10.0
            tmp15 = tmp13 + tmp14
            tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
            tmp17 = tl.where(tmp10, tmp15, tmp16)
            tmp18 = tl.where(tmp4, tmp9, tmp17)
            tl.store(out_ptr0 + (x0), tmp18, xmask)

        arg0 = 0
        arg1 = torch.randn(0, device="cuda")
        arg2 = torch.randn(20, device="cuda")
        buf0 = torch.empty(20, device="cuda")
        buf1 = torch.empty(20, device="cuda")
        xnumel = 20 + arg0
        compiled_kernel = triton_poi_fused_cat_0[(1,)](
            arg1, arg2, buf0, arg0, xnumel, XBLOCK=32
        )
        launcher = StaticallyLaunchedCudaKernel(compiled_kernel)

        launcher.write_cubin_to_file(self.tmp_file.name)
        launcher.load_kernel()
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.run((1, 1, 1), stream, arg1, arg2, buf1, arg0, xnumel)
        self.assertEqual(buf0, buf1)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
