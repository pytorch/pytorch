# Owner(s): ["module: inductor"]
"""Tests for the prototype torch.library.register_aot_impl / TVMFFIKernel path.

The end-to-end tests need CUDA, g++, CuTeDSL (nvidia-cutlass-dsl) and TVM-FFI
(apache-tvm-ffi); they are skipped otherwise. The codegen/registry tests run
anywhere.
"""

import contextlib
import ctypes
import os
import shutil
import unittest

import torch
import torch._inductor
from torch._inductor.aoti_impl import generate_tvm_ffi_shim
from torch._inductor.test_case import TestCase
from torch._library import aoti_impl as aoti_impl_registry
from torch._library.aoti_impl import canonical_op_key
from torch.export import Dim
from torch.library import register_aot_impl, TVMFFIKernel
from torch.testing._internal.common_utils import IS_FBCODE, run_tests


try:
    import cutlass
    import cutlass.cute as cute
    import tvm_ffi  # noqa: F401
    from cutlass.cute.runtime import (
        find_runtime_libraries,
        make_fake_compact_tensor,
        make_fake_stream,
    )

    HAS_CUTEDSL = True
except ImportError:
    HAS_CUTEDSL = False

HAS_E2E_DEPS = (
    HAS_CUTEDSL
    and torch.cuda.is_available()
    and shutil.which("g++") is not None
    and not IS_FBCODE
)


@torch.library.custom_op("aot_impl_test::scale", mutates_args=(), device_types="cuda")
def scale(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return x * alpha


@scale.register_fake
def _(x, alpha=2.0):
    return torch.empty_like(x)


@torch.library.custom_op("aot_impl_other::scale", mutates_args=(), device_types="cuda")
def other_scale(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return x * alpha


@other_scale.register_fake
def _(x, alpha=2.0):
    return torch.empty_like(x)


if HAS_CUTEDSL:

    @cute.kernel
    def _scale_kernel(x: cute.Tensor, out: cute.Tensor, alpha: cutlass.Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 256 + tidx
        if i < cute.size(x):
            out[i] = x[i] * alpha

    @cute.jit
    def _cute_scale(x: cute.Tensor, out: cute.Tensor, alpha: cutlass.Float32, stream):
        n = cute.size(x)
        _scale_kernel(x, out, alpha).launch(
            grid=((n + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    def _scale_builder(x, alpha, *, target):
        syms = {}

        def fake(t):
            shape = []
            for s in t.size():
                if isinstance(s, torch.SymInt):
                    shape.append(syms.setdefault(str(s.node.expr), cute.sym_int32()))
                else:
                    shape.append(int(s))
            return make_fake_compact_tensor(
                cutlass.Float32, tuple(shape), assumed_align=16
            )

        fn = cute.compile(
            _cute_scale,
            fake(x),
            fake(x),
            cutlass.Float32(1.0),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options=f"--enable-tvm-ffi --gpu-arch {target.arch}",
        )
        obj = target.workdir / "scale.o"
        fn.export_to_c(str(obj), function_name="scale")
        return TVMFFIKernel(
            objects=[obj],
            symbol="__tvm_ffi_scale",
            args=["x", "out0", "alpha"],
            link_libs=find_runtime_libraries(enable_tvm_ffi=True),
        )


@contextlib.contextmanager
def _registered(op, builder):
    register_aot_impl(op)(builder)
    try:
        yield
    finally:
        aoti_impl_registry._AOT_IMPLS.pop(canonical_op_key(op), None)


def _preload_companion_libs():
    # The model .so has the companion lib as DT_NEEDED; preloading it by path
    # (it carries a SONAME) satisfies that without LD_LIBRARY_PATH.
    for d in os.environ.get("LIBRARY_PATH", "").split(os.pathsep):
        if os.path.basename(d).startswith("aoti_impl_") and os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith("_aot_impl.so"):
                    ctypes.CDLL(os.path.join(d, f), mode=ctypes.RTLD_GLOBAL)


class TestAOTImplRegistry(TestCase):
    def test_canonical_op_key(self):
        op = torch.ops.aot_impl_test.scale.default
        self.assertEqual(canonical_op_key(op), "aot_impl_test::scale")
        self.assertEqual(canonical_op_key(scale), "aot_impl_test::scale")
        self.assertEqual(
            canonical_op_key("aot_impl_test::scale"), "aot_impl_test::scale"
        )
        self.assertEqual(
            canonical_op_key("aot_impl_test::scale.default"), "aot_impl_test::scale"
        )
        self.assertEqual(
            canonical_op_key("aot_impl_test::scale.out"), "aot_impl_test::scale.out"
        )

    def test_duplicate_registration_rejected(self):
        with _registered("aot_impl_test::scale", lambda *a, **k: None):
            with self.assertRaisesRegex(RuntimeError, "already has an AOT impl"):
                register_aot_impl("aot_impl_test::scale.default")(lambda *a, **k: None)

    def test_tvm_ffi_kernel_validation(self):
        with self.assertRaisesRegex(ValueError, "at least one object"):
            TVMFFIKernel(objects=[], symbol="__tvm_ffi_scale")
        with self.assertRaisesRegex(ValueError, "valid C identifier"):
            TVMFFIKernel(objects=["a.o"], symbol="bad symbol;")


class TestAOTImplShimCodegen(TestCase):
    def test_generated_shim(self):
        op = torch.ops.aot_impl_test.scale.default
        x = torch.empty(16, device="meta")
        kernel = TVMFFIKernel(
            objects=["scale.o"], symbol="__tvm_ffi_scale", args=["x", "out0", "alpha"]
        )
        src, decl = generate_tvm_ffi_shim(
            op, kernel, [x, 2.0], [torch.empty_like(x)], device="cuda"
        )
        self.assertEqual(
            decl,
            "AOTITorchError aoti_torch_cuda_scale(AtenTensorHandle x, double alpha, "
            "AtenTensorHandle* ret0)",
        )
        self.assertIn(
            "const TVMFFIAny args[3] = {tensorArg(&dl_x), tensorArg(&dl_out0), "
            "floatArg(alpha)};",
            src,
        )
        self.assertIn("int64_t sizes[] = {16}; int64_t strides[] = {1};", src)
        self.assertIn("TVMFFIEnvSetStream(kDLCUDA, device, stream, &prev);", src)

    def test_default_arg_order(self):
        # args=None: tensor inputs, then outputs, then scalars.
        op = torch.ops.aot_impl_test.scale.default
        x = torch.empty(4, 8, device="meta")
        kernel = TVMFFIKernel(objects=["scale.o"], symbol="__tvm_ffi_scale")
        src, _ = generate_tvm_ffi_shim(
            op, kernel, [x, 2.0], [torch.empty_like(x)], device="cuda"
        )
        self.assertIn("{tensorArg(&dl_x), tensorArg(&dl_out0), floatArg(alpha)}", src)
        self.assertIn("int64_t sizes[] = {4, 8}; int64_t strides[] = {8, 1};", src)

    def test_unknown_arg_rejected(self):
        op = torch.ops.aot_impl_test.scale.default
        x = torch.empty(4, device="meta")
        kernel = TVMFFIKernel(
            objects=["scale.o"], symbol="__tvm_ffi_scale", args=["x", "out0", "beta"]
        )
        with self.assertRaisesRegex(ValueError, "unknown argument 'beta'"):
            generate_tvm_ffi_shim(
                op, kernel, [x, 2.0], [torch.empty_like(x)], device="cuda"
            )


@unittest.skipIf(not HAS_E2E_DEPS, "needs CUDA, g++, CuTeDSL and TVM-FFI")
class TestAOTImplEndToEnd(TestCase):
    def _compile_and_load(self, model, args, dynamic_shapes=None):
        ep = torch.export.export(model, args, dynamic_shapes=dynamic_shapes)
        pt2 = torch._inductor.aoti_compile_and_package(ep)
        _preload_companion_libs()
        return torch._inductor.aoti_load_package(pt2)

    def test_cutedsl_scale_dynamic(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.relu(torch.ops.aot_impl_test.scale(x, 2.0)) + 1

        with _registered("aot_impl_test::scale", _scale_builder):
            x = torch.randn(4096, device="cuda")
            loaded = self._compile_and_load(M(), (x,), {"x": {0: Dim("n")}})
            for n in (4096, 1000, 7):
                x = torch.randn(n, device="cuda")
                self.assertEqual(loaded(x), torch.relu(x * 2.0) + 1)

    def test_kwargs_and_defaults(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aot_impl_test.scale(
                    x, alpha=3.0
                ) + torch.ops.aot_impl_test.scale(x)

        with _registered("aot_impl_test::scale", _scale_builder):
            x = torch.randn(256, device="cuda")
            ep = torch.export.export(M(), (x,))
            # Two call sites with different scalar values but the same
            # signature share one shim; scalars are runtime arguments.
            pt2 = torch._inductor.aoti_compile_and_package(ep)
            _preload_companion_libs()
            loaded = torch._inductor.aoti_load_package(pt2)
            self.assertEqual(loaded(x), x * 3.0 + x * 2.0)

    def test_mismatched_call_sites_rejected(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return (
                    torch.ops.aot_impl_test.scale(a, 2.0).sum()
                    + torch.ops.aot_impl_test.scale(b, 2.0).sum()
                )

        with _registered("aot_impl_test::scale", _scale_builder):
            a = torch.randn(64, device="cuda")
            b = torch.randn(64, device="cuda", dtype=torch.float64)
            ep = torch.export.export(M(), (a, b))
            with self.assertRaisesRegex(RuntimeError, "different call-site signatures"):
                torch._inductor.aoti_compile_and_package(ep)

    def test_namespace_collision_rejected(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aot_impl_test.scale(
                    x, 2.0
                ) + torch.ops.aot_impl_other.scale(x, 2.0)

        with (
            _registered("aot_impl_test::scale", _scale_builder),
            _registered("aot_impl_other::scale", _scale_builder),
        ):
            x = torch.randn(64, device="cuda")
            ep = torch.export.export(M(), (x,))
            with self.assertRaisesRegex(RuntimeError, "same shim symbol"):
                torch._inductor.aoti_compile_and_package(ep)


if __name__ == "__main__":
    run_tests()
