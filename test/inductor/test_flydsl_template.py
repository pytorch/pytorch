# Owner(s): ["module: inductor"]
import ctypes
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock

import torch

from torch._inductor.test_case import TestCase


try:
    import flydsl  # noqa: F401

    HAS_FLYDSL = True
except ImportError:
    HAS_FLYDSL = False

if HAS_FLYDSL:
    from torch._inductor.codegen.flydsl import flydsl_utils
    from torch._inductor.codegen.flydsl.flydsl_kernel import FlyDSLTemplateKernel


class TestFlyDSLTemplate(TestCase):
    def test_inductor_launcher_specializes_packed_abi(self):
        from torch._inductor.runtime.flydsl_cache import (
            make_flydsl_inductor_launcher,
        )

        observed = []
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

        @callback_type
        def callback(packed):
            slots = ctypes.cast(packed, ctypes.POINTER(ctypes.c_void_p))
            observed.append(
                (
                    ctypes.c_void_p.from_address(slots[0]).value,
                    ctypes.c_void_p.from_address(slots[1]).value,
                    ctypes.c_void_p.from_address(slots[2]).value,
                    ctypes.c_int32.from_address(slots[3]).value,
                    ctypes.c_int32.from_address(slots[4]).value,
                    ctypes.c_int32.from_address(slots[5]).value,
                    ctypes.c_void_p.from_address(slots[6]).value,
                )
            )

        def fill_value(value, storage):
            storage.value = value

        tensors = [torch.empty(1) for _ in range(3)]
        state = SimpleNamespace(
            _spec=[
                (0, ctypes.c_void_p, fill_value),
                (1, ctypes.c_void_p, fill_value),
                (2, ctypes.c_void_p, fill_value),
                (3, ctypes.c_int32, fill_value),
                (4, ctypes.c_int32, fill_value),
                (5, ctypes.c_int32, fill_value),
                (7, ctypes.c_void_p, fill_value),
            ],
            _func_exe=callback,
        )
        executor = SimpleNamespace(_call_state=state)
        with mock.patch.object(
            torch._C, "_FlyDSLMMFp16Bf16CWrapper", None, create=True
        ):
            launcher = make_flydsl_inductor_launcher(
                executor,
                *tensors,
                m=8,
                n=4096,
                k=4096,
                param=object(),
            )
        stream = 0x12345678
        launcher(*tensors, stream)

        self.assertTrue(hasattr(launcher, "_flydsl_keepalive"))
        self.assertEqual(
            observed,
            [
                tuple(tensor.data_ptr() for tensor in tensors)
                + (8, 4096, 4096, stream)
            ],
        )

    def test_inductor_launcher_prefers_native_c_wrapper(self):
        from torch._inductor.runtime.flydsl_cache import (
            make_flydsl_inductor_launcher,
        )

        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        callback = callback_type(lambda packed: None)

        def fill_value(value, storage):
            storage.value = value

        state = SimpleNamespace(
            _spec=[
                (0, ctypes.c_void_p, fill_value),
                (1, ctypes.c_void_p, fill_value),
                (2, ctypes.c_void_p, fill_value),
                (3, ctypes.c_int32, fill_value),
                (4, ctypes.c_int32, fill_value),
                (5, ctypes.c_int32, fill_value),
                (7, ctypes.c_void_p, fill_value),
            ],
            _func_exe=callback,
        )
        executor = SimpleNamespace(_call_state=state)
        tensors = [torch.empty(1) for _ in range(3)]
        native_launcher = object()
        with mock.patch.object(
            torch._C,
            "_FlyDSLMMFp16Bf16CWrapper",
            return_value=native_launcher,
            create=True,
        ) as c_wrapper:
            result = make_flydsl_inductor_launcher(
                executor,
                *tensors,
                m=8,
                n=4096,
                k=4096,
                param=object(),
            )

        func_ptr = ctypes.cast(callback, ctypes.c_void_p).value
        self.assertIs(result, native_launcher)
        c_wrapper.assert_called_once_with(func_ptr, 8, 4096, 4096, executor)

    @unittest.skipUnless(
        hasattr(torch._C, "_FlyDSLMMFp16Bf16CWrapper"),
        "requires _FlyDSLMMFp16Bf16CWrapper",
    )
    def test_native_c_wrapper_packs_flydsl_abi(self):
        observed = []
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

        @callback_type
        def callback(packed):
            slots = ctypes.cast(packed, ctypes.POINTER(ctypes.c_void_p))
            observed.append(
                (
                    ctypes.c_void_p.from_address(slots[0]).value,
                    ctypes.c_void_p.from_address(slots[1]).value,
                    ctypes.c_void_p.from_address(slots[2]).value,
                    ctypes.c_int32.from_address(slots[3]).value,
                    ctypes.c_int32.from_address(slots[4]).value,
                    ctypes.c_int32.from_address(slots[5]).value,
                    ctypes.c_void_p.from_address(slots[6]).value,
                )
            )

        tensors = [torch.empty(1) for _ in range(3)]
        stream = 0x12345678
        func_ptr = ctypes.cast(callback, ctypes.c_void_p).value
        launcher = torch._C._FlyDSLMMFp16Bf16CWrapper(
            func_ptr, 8, 4096, 4096, callback
        )
        launcher(*tensors, stream)

        self.assertEqual(
            observed,
            [
                tuple(tensor.data_ptr() for tensor in tensors)
                + (8, 4096, 4096, stream)
            ],
        )

    def test_compiled_cache_keys_only_on_param(self):
        from torch._inductor.runtime.flydsl_cache import run_cached_flydsl

        jit_func = SimpleNamespace()
        compile_args = (object(),)
        dispatch_args = (object(),)
        compiled = mock.Mock()
        compiler = mock.Mock(return_value=compiled)

        class Param:
            def __cache_signature__(self):
                return ("param",)

        first = run_cached_flydsl(
            jit_func,
            *compile_args,
            constexpr_param=Param(),
            compiler=compiler,
            dispatch_args=dispatch_args,
        )
        second = run_cached_flydsl(
            jit_func,
            object(),
            constexpr_param=Param(),
            compiler=compiler,
            dispatch_args=dispatch_args,
        )

        self.assertIs(first, compiled)
        self.assertIs(second, compiled)
        compiler.assert_called_once_with(jit_func, *compile_args)
        compiled.assert_called_once_with(*dispatch_args)

    def test_compiled_cache_serializes_same_param(self):
        from torch._inductor.runtime.flydsl_cache import run_cached_flydsl

        jit_func = SimpleNamespace()
        compile_started = threading.Event()
        allow_compile = threading.Event()
        compiled = mock.Mock()
        compile_calls = 0

        class Param:
            def __cache_signature__(self):
                return ("param",)

        def compiler(*args):
            nonlocal compile_calls
            compile_calls += 1
            compile_started.set()
            self.assertTrue(allow_compile.wait(5))
            return compiled

        def invoke(value):
            return run_cached_flydsl(
                jit_func,
                object(),
                constexpr_param=Param(),
                compiler=compiler,
                dispatch_args=(value,),
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(invoke, "first")
            self.assertTrue(compile_started.wait(5))
            second = pool.submit(invoke, "second")
            allow_compile.set()
            self.assertIs(first.result(), compiled)
            self.assertIs(second.result(), compiled)

        self.assertEqual(compile_calls, 1)
        compiled.assert_called_once_with("second")

    def test_compiled_cache_retries_after_failure(self):
        from torch._inductor.runtime.flydsl_cache import run_cached_flydsl

        jit_func = SimpleNamespace()
        compiled = mock.Mock()
        compiler = mock.Mock(side_effect=[RuntimeError("compile failed"), compiled])

        class Param:
            def __cache_signature__(self):
                return ("param",)

        kwargs = {
            "constexpr_param": Param(),
            "compiler": compiler,
            "dispatch_args": (),
        }

        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            run_cached_flydsl(jit_func, **kwargs)
        result = run_cached_flydsl(jit_func, **kwargs)

        self.assertIs(result, compiled)
        self.assertEqual(compiler.call_count, 2)
        compiled.assert_not_called()

    def test_inductor_launcher_falls_back_for_unknown_abi(self):
        from torch._inductor.runtime.flydsl_cache import (
            make_flydsl_inductor_launcher,
        )

        class Executor:
            def __init__(self):
                self._call_state = SimpleNamespace(
                    _spec=[(0, ctypes.c_int32, lambda value, storage: None)],
                    _func_exe=None,
                )
                self.calls = []

            def __call__(self, *args):
                self.calls.append(args)

        tensors = [torch.empty(1) for _ in range(3)]
        executor = Executor()
        param = object()
        launcher = make_flydsl_inductor_launcher(
            executor,
            *tensors,
            m=8,
            n=4096,
            k=4096,
            param=param,
        )
        launcher(*tensors, 123)

        self.assertEqual(
            executor.calls,
            [(tensors[0], tensors[1], tensors[2], 8, 4096, 4096, param, 123)],
        )

    def test_gen_imports(self):
        if not HAS_FLYDSL:
            self.skipTest("requires flydsl")

        kernel = FlyDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        imports = kernel.gen_imports()

        self.assertIn("import torch", imports)
        self.assertIn("import flydsl.compiler as flyc", imports)
        self.assertIn("import flydsl.expr as fx", imports)

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
    )
    def test_flydsl_gemm_transposed_rhs_e2e(self):
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        def fn(a, b):
            return torch.mm(a, b.t())

        cases = [
            (32, 128, 128),
            (32, 256, 128),
            (48, 96, 96),
        ]
        for dtype in (torch.bfloat16,):
            for m, n, k in cases:
                with self.subTest(dtype=dtype, m=m, n=n, k=k):
                    a = torch.randn(m, k, device="cuda", dtype=dtype)
                    b = torch.randn(n, k, device="cuda", dtype=dtype)

                    compiled_fn = torch.compile(fn, backend="inductor")
                    result, (code,) = run_and_get_code(compiled_fn, a, b)

                    self.assertIn("async_compile.flydsl", code)
                    self.assertNotIn(".run(", code)
                    self.assertIn("TILE_M: fx.Constexpr", code)
                    self.assertIn("STAGES: fx.Constexpr", code)
                    self.assertIn("BLOCK_N_WARPS: fx.Constexpr", code)
                    self.assertIn("BLOCK_K_WARPS: fx.Constexpr", code)
                    self.assertTrue(
                        torch.allclose(result, fn(a, b), atol=3e-2, rtol=3e-2)
                    )

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        max_autotune_gemm_search_space="EXHAUSTIVE",
        flydsl_enable_autotuning=True,
    )
    def test_flydsl_autotune_transposed_rhs_uses_view_tensor(self):
        from torch._inductor.template_heuristics import flydsl as flydsl_heuristics
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        def fn(a, b):
            return torch.mm(a, b.t())

        configs = [
            asdict(config)
            for config in flydsl_heuristics.get_exhaustive_gemm_configs()[:2]
        ]
        a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        with mock.patch.object(
            flydsl_heuristics, "get_gemm_configs", return_value=configs
        ):
            compiled_fn = torch.compile(fn, backend="inductor")
            result, (code,) = run_and_get_code(compiled_fn, a, b)

        self.assertIn("async_compile.flydsl", code)
        self.assertTrue(torch.allclose(result, fn(a, b), atol=3e-2, rtol=3e-2))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
