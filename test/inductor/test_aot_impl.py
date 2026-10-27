# Owner(s): ["module: inductor"]
"""Tests for the prototype torch.library.register_aot_impl / TVMFFIKernel path.

The end-to-end tests need CUDA, g++, CuTeDSL (nvidia-cutlass-dsl) and TVM-FFI
(apache-tvm-ffi); the shim-compile test needs g++ and TVM-FFI. Everything else
runs anywhere.
"""

import contextlib
import ctypes
import os
import shutil
import subprocess
import tempfile
import unittest
from typing import NamedTuple, Optional

import torch
import torch._inductor
from torch._inductor.aoti_impl import generate_tvm_ffi_shim, render_variant
from torch._inductor.test_case import TestCase
from torch._library import aoti_impl as aoti_impl_registry
from torch._library.aoti_impl import canonical_op_key
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import Dim
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.library import register_aot_impl, TVMFFIKernel
from torch.testing._internal.common_utils import IS_FBCODE, run_tests


try:
    import tvm_ffi

    HAS_TVM_FFI = True
except ImportError:
    HAS_TVM_FFI = False

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import (
        find_runtime_libraries,
        make_fake_compact_tensor,
        make_fake_stream,
    )

    HAS_CUTEDSL = True
except ImportError:
    HAS_CUTEDSL = False

HAS_GXX = shutil.which("g++") is not None
HAS_E2E_DEPS = (
    HAS_CUTEDSL
    and HAS_TVM_FFI
    and HAS_GXX
    and torch.cuda.is_available()
    and not IS_FBCODE
)


@torch.library.custom_op("aot_impl_test::scale", mutates_args=(), device_types="cuda")
def scale(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return x * alpha


@scale.register_fake
def _(x, alpha=2.0):
    return torch.empty_like(x)


@torch.library.custom_op(
    "aot_impl_test::scale_bias", mutates_args=(), device_types="cuda"
)
def scale_bias(
    x: torch.Tensor, alpha: float, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return x * alpha if bias is None else x * alpha + bias


@scale_bias.register_fake
def _(x, alpha, bias=None):
    return torch.empty_like(x)


@torch.library.custom_op(
    "aot_impl_test::add_pair", mutates_args=(), device_types="cuda"
)
def add_pair(x: torch.Tensor, pair: list[torch.Tensor]) -> torch.Tensor:
    return x + pair[0] + pair[1]


@add_pair.register_fake
def _(x, pair):
    return torch.empty_like(x)


@torch.library.custom_op("aot_impl_other::scale", mutates_args=(), device_types="cuda")
def other_scale(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return x * alpha


@other_scale.register_fake
def _(x, alpha=2.0):
    return torch.empty_like(x)


# Schema-only ops for the codegen tests.
torch.library.define(
    "aot_impl_test::mixed",
    "(Tensor x, Tensor[] ys, Tensor? b, float? w, int n, bool flag) -> (Tensor, Tensor)",
)
torch.library.define("aot_impl_test::sink", "(Tensor x) -> ()")
torch.library.define("aot_impl_test::bad_out", "(Tensor x, Tensor out0) -> Tensor")


if HAS_CUTEDSL:

    def _grid(n):
        return ((n + 255) // 256, 1, 1)

    @cute.kernel
    def _scale_kernel(x: cute.Tensor, out: cute.Tensor, alpha: cutlass.Float32):
        i = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
        if i < cute.size(x):
            out[i] = x[i] * alpha

    @cute.kernel
    def _scale_bias_kernel(
        x: cute.Tensor, out: cute.Tensor, bias: cute.Tensor, alpha: cutlass.Float32
    ):
        i = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
        if i < cute.size(x):
            out[i] = x[i] * alpha + bias[i]

    @cute.kernel
    def _add3_kernel(x: cute.Tensor, out: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
        i = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
        if i < cute.size(x):
            out[i] = x[i] + a[i] + b[i]

    @cute.jit
    def _cute_scale(x: cute.Tensor, out: cute.Tensor, alpha: cutlass.Float32, stream):
        _scale_kernel(x, out, alpha).launch(
            grid=_grid(cute.size(x)), block=(256, 1, 1), stream=stream
        )

    @cute.jit
    def _cute_scale_const(
        x: cute.Tensor, out: cute.Tensor, alpha: cutlass.Constexpr, stream
    ):
        _scale_kernel(x, out, cutlass.Float32(alpha)).launch(
            grid=_grid(cute.size(x)), block=(256, 1, 1), stream=stream
        )

    @cute.jit
    def _cute_scale_bias(x, out, bias, alpha: cutlass.Float32, stream):
        if cutlass.const_expr(bias is None):
            _scale_kernel(x, out, alpha).launch(
                grid=_grid(cute.size(x)), block=(256, 1, 1), stream=stream
            )
        else:
            _scale_bias_kernel(x, out, bias, alpha).launch(
                grid=_grid(cute.size(x)), block=(256, 1, 1), stream=stream
            )

    class Pair(NamedTuple):
        a: cute.Tensor
        b: cute.Tensor

    @cute.jit
    def _cute_add_pair(x: cute.Tensor, out: cute.Tensor, pair: Pair, stream):
        _add3_kernel(x, out, pair.a, pair.b).launch(
            grid=_grid(cute.size(x)), block=(256, 1, 1), stream=stream
        )

    # The flat wrapper is what crosses TVM-FFI: one parameter per tuple
    # element, rebuilt into the NamedTuple inside the trace (no per-call cost).
    @cute.jit
    def _cute_add_pair_flat(
        x: cute.Tensor, out: cute.Tensor, a: cute.Tensor, b: cute.Tensor, stream
    ):
        _cute_add_pair(x, out, Pair(a, b), stream)

    def _cute_fakes(*tensors):
        syms = {}

        def fake(t):
            shape = tuple(
                syms.setdefault(str(s.node.expr), cute.sym_int32())
                if isinstance(s, torch.SymInt)
                else int(s)
                for s in t.size()
            )
            return make_fake_compact_tensor(cutlass.Float32, shape, assumed_align=16)

        return [None if t is None else fake(t) for t in tensors]

    def _export(fn, target, *args, packed_args):
        compiled = cute.compile(
            fn,
            *args,
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options=f"--enable-tvm-ffi --gpu-arch {target.arch}",
        )
        obj = target.workdir / "kernel.o"
        compiled.export_to_c(str(obj), function_name=target.name)
        return TVMFFIKernel(
            objects=[obj],
            symbol=f"__tvm_ffi_{target.name}",
            args=packed_args,
            link_libs=find_runtime_libraries(enable_tvm_ffi=True),
        )

    def _scale_builder(x, alpha, *, target):
        cx, cout = _cute_fakes(x, x)
        return _export(
            _cute_scale,
            target,
            cx,
            cout,
            cutlass.Float32(1.0),
            packed_args=["x", "out0", "alpha"],
        )

    def _scale_const_builder(x, alpha, *, target):
        # alpha is left out of packed_args: it is baked into the kernel.
        cx, cout = _cute_fakes(x, x)
        return _export(
            _cute_scale_const, target, cx, cout, alpha, packed_args=["x", "out0"]
        )

    def _scale_bias_builder(x, alpha, bias, *, target):
        cx, cout, cbias = _cute_fakes(x, x, bias)
        return _export(
            _cute_scale_bias,
            target,
            cx,
            cout,
            cbias,
            cutlass.Float32(1.0),
            packed_args=["x", "out0", "bias", "alpha"],
        )

    def _add_pair_builder(x, pair, *, target):
        cx, cout, ca, cb = _cute_fakes(x, x, *pair)
        return _export(
            _cute_add_pair_flat,
            target,
            cx,
            cout,
            ca,
            cb,
            packed_args=["x", "out0", "pair"],
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
    # (it carries a unique SONAME) satisfies that without LD_LIBRARY_PATH.
    # RTLD_LOCAL keeps one package's shim from interposing on another's.
    for d in os.environ.get("LIBRARY_PATH", "").split(os.pathsep):
        if os.path.basename(d).startswith("aoti_impl_") and os.path.isdir(d):
            for f in os.listdir(d):
                if "_aot_impl_" in f and f.endswith(".so"):
                    ctypes.CDLL(os.path.join(d, f))


def _kernel(args=None):
    return TVMFFIKernel(objects=["k.o"], symbol="__tvm_ffi_k", args=args)


def _dynamic(*sizes):
    mode = FakeTensorMode(shape_env=ShapeEnv())
    ctx = StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC] * len(sizes))
    return mode, mode.from_tensor(torch.empty(*sizes), symbolic_context=ctx)


@contextlib.contextmanager
def _captured_stderr():
    # The shim reports through C stdio, so capture file descriptor 2.
    captured = []
    with tempfile.TemporaryFile(mode="w+") as f:
        saved = os.dup(2)
        os.dup2(f.fileno(), 2)
        try:
            yield captured
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            f.seek(0)
            captured.append(f.read())


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
        with self.assertRaisesRegex(TypeError, "args entries"):
            TVMFFIKernel(objects=["a.o"], symbol="k", args=["x", object()])


class TestAOTImplShimCodegen(TestCase):
    def test_generated_shim(self):
        op = torch.ops.aot_impl_test.scale.default
        x = torch.empty(16, device="meta")
        variant = render_variant(
            op, _kernel(["alpha", "x", "out0"]), [x, 2.0], [torch.empty_like(x)]
        )
        src, decl = generate_tvm_ffi_shim(op, [variant])
        # The declaration is what the AOTI wrapper calls; the packed array is
        # the TVM-FFI calling convention in TVMFFIKernel.args order.
        self.assertEqual(
            decl,
            "AOTITorchError aoti_torch_cuda_scale(AtenTensorHandle arg_x, "
            "double arg_alpha, AtenTensorHandle* ret0)",
        )
        self.assertEqual(
            variant.packed,
            ("floatArg(arg_alpha)", "tensorArg(&in0_dl)", "tensorArg(&out0_dl)"),
        )
        self.assertIn("__tvm_ffi_k(void*, const TVMFFIAny*", src)

    def test_default_arg_order(self):
        # args=None: tensor inputs, then outputs, then scalars.
        op = torch.ops.aot_impl_test.scale.default
        x = torch.empty(4, 8, device="meta")
        variant = render_variant(op, _kernel(), [x, 2.0], [torch.empty_like(x)])
        self.assertEqual(
            variant.packed,
            ("tensorArg(&in0_dl)", "tensorArg(&out0_dl)", "floatArg(arg_alpha)"),
        )

    def test_unknown_arg_rejected(self):
        op = torch.ops.aot_impl_test.scale.default
        x = torch.empty(4, device="meta")
        with self.assertRaisesRegex(ValueError, r"unknown argument\(s\) \['beta'\]"):
            render_variant(
                op, _kernel(["x", "out0", "beta"]), [x, 2.0], [torch.empty_like(x)]
            )

    def test_optional_list_and_literal_packing(self):
        op = torch.ops.aot_impl_test.mixed.default
        x = torch.empty(4, device="meta")
        outs = [torch.empty_like(x), torch.empty_like(x)]
        fake_args = [x, [x, x], None, 0.5, 3, True]
        _, decl = generate_tvm_ffi_shim(
            op, [render_variant(op, _kernel(), fake_args, outs)]
        )
        self.assertEqual(
            decl,
            "AOTITorchError aoti_torch_cuda_mixed(AtenTensorHandle arg_x, "
            "const AtenTensorHandle* arg_ys, int64_t arg_ys_len_, "
            "AtenTensorHandle* arg_b, double* arg_w, int64_t arg_n, "
            "int32_t arg_flag, AtenTensorHandle* ret0, AtenTensorHandle* ret1)",
        )
        # Lists flatten, an absent Optional packs as None.
        self.assertEqual(
            render_variant(op, _kernel(), fake_args, outs).packed,
            (
                "tensorArg(&in0_dl)",
                "tensorArg(&in1_dl[0])",
                "tensorArg(&in1_dl[1])",
                "noneArg()",
                "tensorArg(&out0_dl)",
                "tensorArg(&out1_dl)",
                "floatArg((*arg_w))",
                "intArg(arg_n)",
                "boolArg(arg_flag)",
            ),
        )
        # Literal entries fill kernel slots the op does not expose.
        literal = render_variant(
            op,
            _kernel(["x", None, 0.25, "out0", "out1", "w", "n", "flag"]),
            fake_args,
            outs,
        )
        self.assertEqual(
            literal.packed[:3], ("tensorArg(&in0_dl)", "noneArg()", "floatArg(0.25)")
        )

    def test_specialization_guards(self):
        # Call sites share a kernel only if their guards match: the guard
        # captures everything the compiled kernel assumes about its inputs.
        op = torch.ops.aot_impl_test.scale.default

        def guard(x, alpha=2.0, args=None):
            out = torch.empty(x.shape, device="meta")
            return render_variant(op, _kernel(args), [x, alpha], [out]).guard

        meta = lambda *s, **kw: torch.empty(*s, device="meta", **kw)  # noqa: E731
        self.assertNotEqual(guard(meta(4)), guard(meta(8)))  # static size
        self.assertNotEqual(guard(meta(4)), guard(meta(4, dtype=torch.float16)))
        self.assertNotEqual(guard(meta(4)), guard(meta(2, 2)))  # rank
        self.assertNotEqual(guard(meta(4)), guard(meta(8)[::2]))  # stride
        self.assertNotEqual(guard(meta(16)), guard(meta(17)[1:]))  # alignment
        self.assertEqual(guard(meta(4), 2.0), guard(meta(4), 3.0))  # runtime float
        baked = ["x", "out0"]
        self.assertNotEqual(guard(meta(4), 2.0, baked), guard(meta(4), 3.0, baked))
        # A size-1 dim's stride never addresses memory, so it is not guarded.
        self.assertFalse(any("strides[0]" in g for g in guard(meta(1, 8))))

        # Static ints can set output sizes, so they are guarded even at runtime.
        op = torch.ops.aot_impl_test.mixed.default
        x = meta(4)
        self.assertNotEqual(
            *(
                render_variant(
                    op, _kernel(), [x, [], None, None, n, True], [x, x]
                ).guard
                for n in (3, 4)
            )
        )

        op = torch.ops.aot_impl_test.scale_bias.default
        x = meta(4)
        present = render_variant(op, _kernel(), [x, 2.0, x], [x]).guard
        absent = render_variant(op, _kernel(), [x, 2.0, None], [x]).guard
        self.assertIn("arg_bias != nullptr", present)
        self.assertIn("arg_bias == nullptr", absent)

    def test_dynamic_dims(self):
        op = torch.ops.aot_impl_test.scale.default
        mode = FakeTensorMode(shape_env=ShapeEnv())
        ctx = StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC] * 2)
        x = mode.from_tensor(torch.empty(8, 8), symbolic_context=ctx)
        with mode:
            out = torch.empty_like(x)
        variant = render_variant(op, _kernel(), [x, 2.0], [out])
        # Output sizes are read from the input at runtime, not baked.
        self.assertEqual(variant.outputs[0].sizes, "in0_dl.shape[0], in0_dl.shape[1]")
        self.assertNotIn("in0_dl.shape[0] == 8L", variant.guard)

    def test_composite_input_size(self):
        # An input whose size is an expression (here s0 - 1) still defines it.
        op = torch.ops.aot_impl_test.scale.default
        mode, x = _dynamic(8)
        with mode:
            y = x[1:]
            out = torch.empty_like(y)
        variant = render_variant(op, _kernel(), [y, 2.0], [out])
        self.assertEqual(variant.outputs[0].sizes, "in0_dl.shape[0]")

    def test_unsupported_rejected(self):
        x = torch.empty(4, device="meta")
        with self.assertRaisesRegex(NotImplementedError, "reserves for outputs"):
            render_variant(
                torch.ops.aot_impl_test.bad_out.default, _kernel(), [x, x], [x]
            )
        mode, d = _dynamic(4)
        op = torch.ops.aot_impl_test.mixed.default
        baked_n = _kernel(["x", "ys", "b", "w", "flag", "out0", "out1"])
        with self.assertRaisesRegex(NotImplementedError, "dynamic scalar"):
            render_variant(op, baked_n, [x, [], None, None, d.shape[0], True], [x, x])
        with self.assertRaisesRegex(NotImplementedError, "does not fit in int64"):
            render_variant(
                torch.ops.aot_impl_test.scale.default,
                _kernel(["x", "out0", 2**64]),
                [x, 2.0],
                [x],
            )

    @unittest.skipIf(not (HAS_GXX and HAS_TVM_FFI), "needs g++ and TVM-FFI")
    def test_generated_shim_compiles(self):
        # Several specializations, Optional/list/scalar args, a 0-dim output
        # and a 3-way symbolic max must all produce valid C++.
        op = torch.ops.aot_impl_test.mixed.default
        mode = FakeTensorMode(shape_env=ShapeEnv())
        ctx = StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC])
        x, y, z = (
            mode.from_tensor(torch.empty(n), symbolic_context=ctx) for n in (8, 9, 10)
        )
        with mode:
            n = torch.sym_max(torch.sym_max(x.shape[0], y.shape[0]), z.shape[0])
            outs = [x.new_empty(n), x.new_empty(())]
        variants = [
            render_variant(op, _kernel(), [x, [y, z], b, w, 3, True], outs)
            for b, w in ((None, 0.5), (x, None))
        ]
        sink = torch.ops.aot_impl_test.sink.default
        shims = [
            generate_tvm_ffi_shim(op, variants)[0],
            # No outputs and an empty packed call.
            generate_tvm_ffi_shim(sink, [render_variant(sink, _kernel([]), [x], [])])[
                0
            ],
        ]
        include = os.path.join(os.path.dirname(torch.__file__), "include")
        tvm_include = os.path.join(os.path.dirname(tvm_ffi.__file__), "include")
        for src in shims:
            with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
                f.write(src)
                f.flush()
                result = subprocess.run(
                    ["g++", "-fsyntax-only", "-std=c++17", "-DUSE_CUDA"]
                    + [f"-I{include}", f"-I{tvm_include}", f.name],
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipIf(not HAS_E2E_DEPS, "needs CUDA, g++, CuTeDSL and TVM-FFI")
class TestAOTImplEndToEnd(TestCase):
    def setUp(self):
        super().setUp()
        self._library_path = os.environ.get("LIBRARY_PATH")

    def tearDown(self):
        # resolve_aot_impls prepends each companion lib's dir.
        if self._library_path is None:
            os.environ.pop("LIBRARY_PATH", None)
        else:
            os.environ["LIBRARY_PATH"] = self._library_path
        super().tearDown()

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
            # Compiled for a compact layout: a strided input matches no
            # specialization and fails loudly instead of miscomputing.
            with _captured_stderr() as err, self.assertRaises(RuntimeError):
                loaded(torch.randn(8192, device="cuda")[::2])
            self.assertIn("no compiled specialization matches", err[0])

    def test_kwargs_and_defaults(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aot_impl_test.scale(
                    x, alpha=3.0
                ) + torch.ops.aot_impl_test.scale(x)

        with _registered("aot_impl_test::scale", _scale_builder):
            x = torch.randn(256, device="cuda")
            loaded = self._compile_and_load(M(), (x,))
            self.assertEqual(loaded(x), x * 3.0 + x * 2.0)

    def test_static_size_specializations(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return torch.ops.aot_impl_test.scale(
                    a, 2.0
                ), torch.ops.aot_impl_test.scale(b, 3.0)

        with _registered("aot_impl_test::scale", _scale_builder):
            a, b = torch.randn(64, device="cuda"), torch.randn(32, device="cuda")
            out_a, out_b = self._compile_and_load(M(), (a, b))(a, b)
            self.assertEqual(out_a, a * 2.0)
            self.assertEqual(out_b, b * 3.0)

    def test_baked_scalar_specializations(self):
        # The builder bakes alpha, so each value is its own specialization
        # and the shim dispatches on alpha at runtime.
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aot_impl_test.scale(
                    x, 2.0
                ) + 10 * torch.ops.aot_impl_test.scale(x, 3.0)

        with _registered("aot_impl_test::scale", _scale_const_builder):
            x = torch.randn(256, device="cuda")
            self.assertEqual(
                self._compile_and_load(M(), (x,))(x), x * 2.0 + 10 * x * 3.0
            )

    def test_optional_tensor_specializations(self):
        class M(torch.nn.Module):
            def forward(self, x, bias):
                return torch.ops.aot_impl_test.scale_bias(
                    x, 2.0, bias
                ) + torch.ops.aot_impl_test.scale_bias(x, 3.0)

        with _registered("aot_impl_test::scale_bias", _scale_bias_builder):
            x, bias = torch.randn(256, device="cuda"), torch.randn(256, device="cuda")
            out = self._compile_and_load(M(), (x, bias))(x, bias)
            self.assertEqual(out, (x * 2.0 + bias) + x * 3.0)

    def test_tuple_arg(self):
        class M(torch.nn.Module):
            def forward(self, x, a, b):
                return torch.ops.aot_impl_test.add_pair(x, [a, b])

        with _registered("aot_impl_test::add_pair", _add_pair_builder):
            x, a, b = (torch.randn(256, device="cuda") for _ in range(3))
            self.assertEqual(self._compile_and_load(M(), (x, a, b))(x, a, b), x + a + b)

    def test_two_packages_same_op(self):
        # Each package must bind to its own shim even when both are loaded
        # in one process (same op, different baked alpha).
        class M(torch.nn.Module):
            def __init__(self, alpha):
                super().__init__()
                self.alpha = alpha

            def forward(self, x):
                return torch.ops.aot_impl_test.scale(x, self.alpha)

        with _registered("aot_impl_test::scale", _scale_const_builder):
            x = torch.randn(256, device="cuda")
            first = self._compile_and_load(M(2.0), (x,))
            second = self._compile_and_load(M(3.0), (x,))
            for _ in range(2):
                self.assertEqual(first(x), x * 2.0)
                self.assertEqual(second(x), x * 3.0)

    def test_builder_symbolic_specialization_rejected(self):
        def builder(x, alpha, *, target):
            if x.shape[0] % 16 == 0:  # records a guard on the symbolic size
                pass
            return _scale_builder(x, alpha, target=target)

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aot_impl_test.scale(x, 2.0)

        with _registered("aot_impl_test::scale", builder):
            x = torch.randn(256, device="cuda")
            ep = torch.export.export(M(), (x,), dynamic_shapes={"x": {0: Dim("n")}})
            with self.assertRaisesRegex(RuntimeError, "specialized on symbolic sizes"):
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
