# Owner(s): ["module: inductor"]
import os
import random
import tempfile
from unittest import mock

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.static_triton_launcher import (
    statically_launched_kernel_by_device,
    StaticallyLaunchedCudaKernel,
    StaticallyLaunchedXpuKernel,
)
from torch._inductor.runtime.triton_compat import (
    CompiledKernel,
    JITFunction,
    tl,
    triton,
)
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_WINDOWS, skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_XPU_AND_TRITON
from torch.testing._internal.triton_utils import requires_gpu_and_triton


if HAS_XPU_AND_TRITON:
    _orig_getitem = JITFunction.__getitem__


def _patched_getitem(self, grid):
    orig_launcher = _orig_getitem(self, grid)

    def launcher_with_native_code(*args, **kwargs):
        kwargs.setdefault("generate_native_code", True)
        return orig_launcher(*args, **kwargs)

    return launcher_with_native_code


@requires_gpu_and_triton
class TestStaticTritonLauncher(TestCase):
    def setUp(self):
        super().setUp()
        self.tmp_files = []
        if HAS_XPU_AND_TRITON:
            JITFunction.__getitem__ = _patched_getitem

    def tearDown(self):
        super().tearDown()
        for tmp_file in self.tmp_files:
            try:
                os.remove(tmp_file.name)
            except OSError:
                pass

        if HAS_XPU_AND_TRITON:
            JITFunction.__getitem__ = _orig_getitem

    def write_cubin_to_tmp(self, kernel: CompiledKernel) -> str:
        """
        Only used for tests where we don't have a cubin path.
        """
        if hasattr(kernel, "_cubin_path"):
            return
        # Just used by tests for now.
        # TODO: derive cubin_path from wherever triton stores the cubin file on disk.
        binary_key = ""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            if GPU_TYPE == "xpu":
                binary_key = "zebin"
            else:
                binary_key = "hsaco" if torch.version.hip else "cubin"

            tmp_file.write(kernel.asm[binary_key])
            self.tmp_files.append(tmp_file)
            return tmp_file.name

    def _make_launcher(
        self,
        compiled_kernel: CompiledKernel,
    ) -> StaticallyLaunchedCudaKernel | StaticallyLaunchedXpuKernel:
        """
        Compiles a Triton kernel with the provided *args,
        writes its cubin to the temporary file, and returns the file path.
        """
        cubin_file = self.write_cubin_to_tmp(compiled_kernel)
        compiled_kernel._cubin_path = cubin_file
        result = statically_launched_kernel_by_device(compiled_kernel, GPU_TYPE)
        # Test reload cubin from raw here
        old_cubin_path = result.cubin_path
        if old_cubin_path is None:
            raise AssertionError
        result.cubin_path = None
        result.reload_cubin_from_raw(old_cubin_path)
        device_interface = get_interface_for_device(GPU_TYPE)
        result.load_kernel(device_interface.current_device())
        return result

    def test_basic(self):
        """Verify _FastCudaLauncher correctly launches a kernel with tensor + scalar args.

        Compiles a simple kernel, wraps it in _FastCudaLauncher, then invokes
        via vectorcall and checks the output tensor has the expected value.
        """

        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            y = arg1
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        arg1 = 5
        args = (arg0, arg1)
        compiled_kernel = simple_kernel[(1,)](*args)
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([5], dtype=torch.int32, device=GPU_TYPE))
        self.assertEqual(launcher.arg_tys, "Oi")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.run(1, 1, 1, stream, new_arg0, arg1)
        self.assertEqual(new_arg0, arg0)

    # I wish I could macro all int types this into a single unit test on a loop, but
    # 1. variables aren't allowed as type annotations in python
    # 2. triton relies on inspect.get_source to get the type annotations
    # so I can't even use exec() to generate the test cases.
    # So we'll just make a few kernels by hand
    def test_unsigned_integers(self):
        @triton.jit
        def unsigned_integers(
            arg0, arg1: tl.uint8, arg2: tl.uint16, arg3: tl.uint32, arg4: tl.uint64
        ):
            x = tl.load(arg0)
            y = arg1 + arg2 + arg3 + arg4
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.uint64, device=GPU_TYPE)
        # Using small numbers creates a Literal type which triton treats as a constant
        args = (arg0, 50, 50, 50, 50)

        compiled_kernel = unsigned_integers[1,](*args)
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([200], dtype=torch.uint64, device=GPU_TYPE))
        self.assertEqual(launcher.arg_tys, "OBHIK")
        new_arg0 = torch.zeros(1, dtype=torch.uint64, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(1, 1, 1, stream, new_arg0, 50, 50, 50, 50)
        self.assertEqual(new_arg0, arg0)

    def test_signed_integers(self):
        @triton.jit
        def signed_integers(
            arg0, arg1: tl.int8, arg2: tl.int16, arg3: tl.int32, arg4: tl.int64
        ):
            x = tl.load(arg0)
            y = arg1 + arg2 + arg3 + arg4
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int64, device=GPU_TYPE)
        # Using small numbers creates a Literal type which triton treats as a constant
        args = (arg0, 50, 50, 50, 50)

        compiled_kernel = signed_integers[1,](*args)
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([200], dtype=torch.int64, device=GPU_TYPE))
        self.assertEqual(launcher.arg_tys, "Obhil")
        new_arg0 = torch.zeros(1, dtype=torch.int64, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(1, 1, 1, stream, new_arg0, 50, 50, 50, 50)
        self.assertEqual(new_arg0, arg0)

    def test_basic_1arg(self):
        @triton.jit
        def simple_kernel_1_arg(arg0):
            x = tl.load(arg0)
            tl.store(arg0, x + 1)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        compiled_kernel = simple_kernel_1_arg[1,](arg0)
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([1], dtype=torch.int32, device=GPU_TYPE))
        self.assertEqual(launcher.arg_tys, "O")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.run(
            1,
            1,
            1,
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
        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        compiled_kernel = kernel_constexpr[(1,)](arg0, CONSTANT=5)
        launcher = self._make_launcher(compiled_kernel)

        self.assertEqual(arg0, torch.tensor([5], dtype=torch.int32, device=GPU_TYPE))
        self.assertEqual(launcher.arg_tys, "O")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(
            1,
            1,
            1,
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
            xnumel,
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

        arg0 = torch.tensor([0.0, 0.5, float("inf"), 5], device=GPU_TYPE)
        arg1 = torch.tensor([False], device=GPU_TYPE)
        arg2 = torch.tensor([False], device=GPU_TYPE)
        compiled_kernel = triton_red_fused_any_isinf_0[1,](
            arg0, arg1, 1, 128, XBLOCK=1, R0_BLOCK=1
        )
        launcher = self._make_launcher(compiled_kernel)

        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        # Don't pass in xnumel, as it is a constant
        launcher.run(1, 1, 1, stream, arg0, arg2, 128)
        self.assertEqual(arg1, arg2)

    def test_kernel_no_args(self):
        # Just an easy way to test incompatible number of arguments
        @triton.jit
        def kernel_no_op():
            pass

        compiled_kernel = kernel_no_op[(1,)]()
        launcher = self._make_launcher(compiled_kernel)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(1, 1, 1, stream)

    def test_high_shared_mem(self):
        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            y = arg1
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        arg1 = 5
        args = (arg0, arg1)
        compiled_kernel = simple_kernel[(1,)](*args)
        # Allocate 50 KB of memory
        compiled_kernel.shared = 50000
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([5], dtype=torch.int32, device=GPU_TYPE))
        self.assertEqual(launcher.arg_tys, "Oi")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.slow_launch_kernel = True
        launcher.run(1, 1, 1, stream, new_arg0, arg1)
        self.assertEqual(new_arg0, arg0)

    @skipIfXpu(msg="Only testing CUDA OOM behavior")
    def test_too_high_shared_mem(self):
        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            y = arg1
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        arg1 = 5
        args = (arg0, arg1)
        compiled_kernel = simple_kernel[(1,)](*args)
        # Allocate too much shared memory
        compiled_kernel.shared = 99999999
        self.assertRaisesRegex(
            RuntimeError,
            "out of resource: simple_kernel",
            lambda: self._make_launcher(compiled_kernel),
        )

    def test_kernel_empty_tensor(self):
        # Triton kernel generated by torch.compile of the following:
        # @torch.compile()
        # def foo(x, y):
        #   return torch.cat(((x * 4), y + 10))

        # Running with example input:
        # torch._dynamo.decorators.mark_unbacked(t, 0)
        # x = torch.rand(0, device=GPU_TYPE)
        # y = torch.rand(20, device=GPU_TYPE)

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
        arg1 = torch.randn(0, device=GPU_TYPE)
        arg2 = torch.randn(20, device=GPU_TYPE)
        buf0 = torch.empty(20, device=GPU_TYPE)
        buf1 = torch.empty(20, device=GPU_TYPE)
        xnumel = 20 + arg0
        compiled_kernel = triton_poi_fused_cat_0[(1,)](
            arg1, arg2, buf0, arg0, xnumel, XBLOCK=32
        )
        launcher = self._make_launcher(compiled_kernel)

        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.run(1, 1, 1, stream, arg1, arg2, buf1, arg0, xnumel)
        self.assertEqual(buf0, buf1)

    def test_kernel_many_args(self):
        N = 200
        # Make 200 arguments
        args = [f"arg_{i}" for i in range(N)]
        decl = ", ".join(args)
        sums = [f"    total += arg_{i}" for i in range(N)]
        sums_str = "\n".join(sums)

        template = f"""
from torch._inductor.runtime.triton_compat import tl, triton
@triton.jit
def kernel_many_args(out_tensor, {decl}):
    out = tl.load(out_tensor)
    total = out
{sums_str}
    tl.store(out_tensor, total)
        """

        result = PyCodeCache.load(template.lstrip())

        kernel_args = tuple(random.random() for _ in range(N))
        buf0 = torch.zeros(1, device=GPU_TYPE)
        compiled_kernel = result.kernel_many_args[1,](buf0, *kernel_args)
        launcher = self._make_launcher(compiled_kernel)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        buf1 = torch.zeros(1, device=GPU_TYPE)
        launcher.run(1, 1, 1, stream, buf1, *kernel_args)
        self.assertEqual(buf0, buf1)


@requires_gpu_and_triton
@torch._inductor.config.patch(
    {"use_static_triton_launcher": True, "strict_static_triton_launcher": True}
)
class TestStaticTritonCompileResult(TestCase):
    """
    Tests static cuda launcher with torch.compile()
    """

    def test_basic_compile(self):
        @torch.compile
        def foo(x, y):
            return x + y

        x = torch.randn(10, device=GPU_TYPE)
        y = torch.randn(10, device=GPU_TYPE)
        self.assertEqual(foo(x, y), x + y)

    # The error gets raised on a worker, so we want to not use a separate process
    @torch._inductor.config.patch("compile_threads", 1)
    def test_incompatible_code(self):
        # User defined triton kernel
        @triton.jit
        def custom_kernel(arg_0, arg_1):
            x = tl.load(arg_0)
            y = arg_1
            tl.store(arg_0, x + y)

        @torch.compile
        def foo(x):
            custom_kernel[1,](x, 5)
            return x

        x = torch.randn(1, device=GPU_TYPE)
        self.assertRaisesRegex(
            torch._inductor.exc.InductorError,
            "CannotStaticallyLaunchKernel: User defined triton kernel",
            lambda: foo(x),
        )

    # The error gets raised on a worker, so we want to not use a separate process
    @torch._inductor.config.patch(
        {"compile_threads": 1, "static_launch_user_defined_triton_kernels": True}
    )
    def test_static_launch_user_defined_triton_kernels(self):
        # User defined triton kernel
        @triton.jit
        def custom_kernel(arg_0, arg_1):
            x = tl.load(arg_0)
            y = arg_1
            tl.store(arg_0, x + y)

        @torch.compile
        def foo(x):
            custom_kernel[1,](x, 5)
            return x

        x = torch.randn(1, device=GPU_TYPE)
        x2 = x.clone().detach_()
        self.assertEqual(foo(x), x2 + 5)

    def test_empty_tensor(self):
        @torch.compile()
        def foo(x, y):
            return torch.cat(((x * 4), y + 10))

        x = torch.rand(0, device=GPU_TYPE)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        y = torch.rand(20, device=GPU_TYPE)
        result = foo(x, y)
        self.assertEqual(result, torch.cat(((x * 4), y + 10)))

    def test_any(self):
        def fn(x):
            return (
                x.any(-1),
                x.isinf().any(),
                torch.all(x.isinf(), dim=0),
                torch.all(torch.logical_not(x.isinf())),
            )

        compiled_fn = torch.compile(fn)
        arg = -torch.rand(64, device=GPU_TYPE, dtype=torch.float64)
        eager_result = fn(arg)
        compiled_result = compiled_fn(arg)
        self.assertEqual(eager_result, compiled_result)
        arg[1] = float("inf")
        eager_result = fn(arg)
        compiled_result = compiled_fn(arg)
        self.assertEqual(eager_result, compiled_result)

    def test_disable_static_triton_launcher(self):
        @torch.compile
        def fn(x, y):
            return torch.cat(((x * 4), y + 10))

        # Test that static cuda launcher is in fact disabled
        with torch._inductor.config.patch("use_static_triton_launcher", False):
            x = torch.rand(20, device=GPU_TYPE)
            y = torch.rand(20, device=GPU_TYPE)
            with mock.patch(
                "torch._inductor.runtime.triton_heuristics.StaticTritonCompileResult.make_launcher"
            ) as mocked:
                result = fn(x, y)
                mocked.assert_not_called()

            self.assertEqual(result, torch.cat(((x * 4), y + 10)))


@requires_gpu_and_triton
@skipIfXpu
class TestFastCudaLauncher(TestCase):
    """Tests for _FastCudaLauncher vectorcall C extension."""

    def setUp(self):
        super().setUp()
        self.tmp_files = []

    def tearDown(self):
        super().tearDown()
        for tmp_file in self.tmp_files:
            try:
                os.remove(tmp_file.name)
            except OSError:
                pass

    def write_cubin_to_tmp(self, kernel: CompiledKernel) -> str:
        if hasattr(kernel, "_cubin_path"):
            return
        binary_key = "hsaco" if torch.version.hip else "cubin"
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(kernel.asm[binary_key])
            self.tmp_files.append(tmp_file)
            return tmp_file.name

    def _make_launcher(
        self,
        compiled_kernel: CompiledKernel,
    ) -> StaticallyLaunchedCudaKernel:
        cubin_file = self.write_cubin_to_tmp(compiled_kernel)
        compiled_kernel._cubin_path = cubin_file
        result = statically_launched_kernel_by_device(compiled_kernel, GPU_TYPE)
        old_cubin_path = result.cubin_path
        if old_cubin_path is None:
            raise AssertionError
        result.cubin_path = None
        result.reload_cubin_from_raw(old_cubin_path)
        device_interface = get_interface_for_device(GPU_TYPE)
        result.load_kernel(device_interface.current_device())
        return result

    def _make_fast_launcher(self, kernel):
        from torch._C import _FastCudaLauncher

        n_scratch = 0
        if getattr(kernel, "has_global_scratch", False):
            n_scratch += 1
        if getattr(kernel, "has_profile_scratch", False):
            n_scratch += 1
        return _FastCudaLauncher(
            kernel.function, kernel.num_warps, kernel.shared, kernel.arg_tys, n_scratch
        )

    def test_basic(self):
        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            y = arg1
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        compiled_kernel = simple_kernel[(1,)](arg0, 5)
        launcher = self._make_launcher(compiled_kernel)
        fast = self._make_fast_launcher(launcher)

        new_arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        fast(1, 1, 1, stream, new_arg0, 5)
        self.assertEqual(
            new_arg0, torch.tensor([5], dtype=torch.int32, device=GPU_TYPE)
        )

    def test_multiple_tensor_args(self):
        """Verify _FastCudaLauncher handles multiple tensor pointer args correctly."""

        @triton.jit
        def add_kernel(a, b, out):
            x = tl.load(a)
            y = tl.load(b)
            tl.store(out, x + y)

        a = torch.tensor([3], dtype=torch.int32, device=GPU_TYPE)
        b = torch.tensor([7], dtype=torch.int32, device=GPU_TYPE)
        out = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        compiled_kernel = add_kernel[(1,)](a, b, out)
        launcher = self._make_launcher(compiled_kernel)
        fast = self._make_fast_launcher(launcher)

        out2 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        fast(1, 1, 1, stream, a, b, out2)
        self.assertEqual(out2, torch.tensor([10], dtype=torch.int32, device=GPU_TYPE))

    def test_zero_grid(self):
        """Verify zero-grid launch is a no-op (kernel does not execute)."""

        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            tl.store(arg0, x + arg1)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        compiled_kernel = simple_kernel[(1,)](arg0, 5)
        launcher = self._make_launcher(compiled_kernel)
        fast = self._make_fast_launcher(launcher)

        target = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        fast(0, 1, 1, stream, target, 99)
        self.assertEqual(target, torch.tensor([0], dtype=torch.int32, device=GPU_TYPE))

    def test_wrong_arg_count(self):
        """Verify _FastCudaLauncher raises RuntimeError on argument count mismatch."""

        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            tl.store(arg0, x + arg1)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        compiled_kernel = simple_kernel[(1,)](arg0, 5)
        launcher = self._make_launcher(compiled_kernel)
        fast = self._make_fast_launcher(launcher)

        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        with self.assertRaises(RuntimeError):
            fast(1, 1, 1, stream, arg0)  # missing arg1


@requires_gpu_and_triton
@torch._inductor.config.patch(
    {
        "use_static_triton_launcher": True,
        "strict_static_triton_launcher": True,
        "use_fast_triton_launcher": True,
    }
)
class TestFastCudaLauncherCompileResult(TestCase):
    """E2E tests verifying _FastCudaLauncher is actually used by torch.compile.

    These tests assert both correctness (output matches eager) and that the
    _FastCudaLauncher C extension was constructed, not silently skipped.
    """

    def _patch_build_fast_launcher(self):
        """Context manager that tracks _build_fast_launcher calls.

        Returns a list that accumulates True/False for each call,
        indicating whether a _FastCudaLauncher was successfully built.
        """
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        results = []
        original = CachingAutotuner._build_fast_launcher

        def tracking_build(autotuner_self, launcher):
            result = original(autotuner_self, launcher)
            results.append(result is not None)
            return result

        return mock.patch.object(
            CachingAutotuner, "_build_fast_launcher", tracking_build
        ), results

    def test_basic_compile(self):
        """Verify torch.compile uses _FastCudaLauncher and produces correct output."""
        patcher, results = self._patch_build_fast_launcher()
        with patcher:

            @torch.compile
            def foo(x, y):
                return x + y

            x = torch.randn(10, device=GPU_TYPE)
            y = torch.randn(10, device=GPU_TYPE)
            self.assertEqual(foo(x, y), x + y)
            self.assertTrue(
                any(results), "_FastCudaLauncher was not built by any CachingAutotuner"
            )

    def test_disable_fast_launcher(self):
        """Verify disabling the config falls back to the regular launcher."""
        patcher, results = self._patch_build_fast_launcher()
        with patcher, torch._inductor.config.patch("use_fast_triton_launcher", False):

            @torch.compile
            def foo(x, y):
                return x + y

            x = torch.randn(10, device=GPU_TYPE)
            y = torch.randn(10, device=GPU_TYPE)
            result = foo(x, y)
            self.assertEqual(result, x + y)
            self.assertFalse(
                any(results),
                "_FastCudaLauncher should not be built when config is disabled",
            )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # TODO: Enable test on XPU windows once supported.
    if not (HAS_XPU_AND_TRITON and IS_WINDOWS):
        run_tests()
