# Owner(s): ["module: inductor"]
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch._inductor.codegen.common import (
    _pytorch_cpu_scalar_expr_contract_addcmul,
    _pytorch_cpu_vec_intrinsics_contract_addcmul,
)
from torch._inductor.codegen.cpp import CppOverrides, CppVecOverrides
from torch._inductor.codegen.halide import HalideOverrides
from torch._inductor.codegen.mps import MetalOverrides
from torch._inductor.codegen.triton import TritonKernelOverrides
from torch._inductor.ops_handler import list_ops, OP_NAMES, OpsHandler
from torch._inductor.test_case import TestCase
from torch._inductor.virtualized import V


class TestOpCompleteness(TestCase):
    def verify_ops_handler_completeness(self, handler):
        for op in OP_NAMES:
            self.assertIsNot(
                getattr(handler, op),
                getattr(OpsHandler, op),
                msg=f"{handler} must implement {op}",
            )
        extra_ops = list_ops(handler) - OP_NAMES
        if extra_ops:
            raise AssertionError(
                f"{handler} has an extra ops: {extra_ops}, add them to OpHandler class or prefix with `_`"
            )

    def test_triton_overrides(self):
        self.verify_ops_handler_completeness(TritonKernelOverrides)

    def test_cpp_overrides(self):
        self.verify_ops_handler_completeness(CppOverrides)

    def test_cpp_vec_overrides(self):
        self.verify_ops_handler_completeness(CppVecOverrides)

    def test_cpp_vec_addcmul_aten_codegen(self):
        def clear_addcmul_caches():
            _pytorch_cpu_scalar_expr_contract_addcmul.cache_clear()
            _pytorch_cpu_vec_intrinsics_contract_addcmul.cache_clear()

        clear_addcmul_caches()
        with patch(
            "torch.__config__.show",
            return_value=(
                "PyTorch built with:\n  - GCC 11.5\n  - CPU capability usage: AVX512\n"
            ),
        ):
            self.assertEqual(
                CppVecOverrides.addcmul_aten("self", "value_times_t1", "t2"),
                "fmadd(value_times_t1, t2, self)",
            )
            self.assertEqual(
                CppOverrides.addcmul_aten("self", "value_times_t1", "t2"),
                "std::fma(value_times_t1, t2, self)",
            )

        clear_addcmul_caches()
        with patch(
            "torch.__config__.show",
            return_value=(
                "PyTorch built with:\n"
                "  - GCC 4.2\n"
                "  - clang 18.1.8\n"
                "  - CPU capability usage: AVX512\n"
            ),
        ):
            code = CppVecOverrides.addcmul_aten("self", "value_times_t1", "t2")
            self.assertIn('asm volatile("" : "+m"(product));', code)
            self.assertIn("return self + product;", code)
            self.assertNotIn("fmadd", code)
            self.assertEqual(
                CppOverrides.addcmul_aten("self", "value_times_t1", "t2"),
                "std::fma(value_times_t1, t2, self)",
            )

        arm_vector_configs = (
            "PyTorch built with:\n  - GCC 11.5\n  - CPU capability usage: NEON\n",
            "PyTorch built with:\n  - GCC 13.4\n  - CPU capability usage: SVE256\n",
        )
        for config in arm_vector_configs:
            clear_addcmul_caches()
            with patch("torch.__config__.show", return_value=config):
                code = CppVecOverrides.addcmul_aten("self", "value_times_t1", "t2")
                self.assertIn('asm volatile("" : "+m"(product));', code)
                self.assertIn("return self + product;", code)
                self.assertNotIn("fmadd", code)
                self.assertEqual(
                    CppOverrides.addcmul_aten("self", "value_times_t1", "t2"),
                    "std::fma(value_times_t1, t2, self)",
                )

        clear_addcmul_caches()
        with patch(
            "torch.__config__.show",
            return_value=(
                "PyTorch built with:\n  - MSVC 19.44\n  - CPU capability usage: AVX2\n"
            ),
        ):
            code = CppVecOverrides.addcmul_aten("self", "value_times_t1", "t2")
            self.assertIn("reinterpret_cast<const volatile char*>(&product);", code)
            self.assertIn("return self + product;", code)
            self.assertNotIn("fmadd", code)

            code = CppOverrides.addcmul_aten("self", "value_times_t1", "t2")
            self.assertIn("reinterpret_cast<const volatile char*>(&product);", code)
            self.assertIn("return self + product;", code)
            self.assertNotIn("std::fma", code)

        clear_addcmul_caches()
        with (
            patch(
                "torch.__config__.show",
                return_value=(
                    "PyTorch built with:\n"
                    "  - GCC 11.5\n"
                    "  - CPU capability usage: AVX512\n"
                ),
            ),
            V.set_kernel_handler(SimpleNamespace(tail_size=3)),
        ):
            self.assertEqual(
                CppVecOverrides.addcmul_aten("self", "value_times_t1", "t2"),
                "fmadd(value_times_t1, t2, self)",
            )

        clear_addcmul_caches()
        scalar_contracting_tail_configs = (
            (
                "PyTorch built with:\n"
                "  - GCC 4.2\n"
                "  - clang 18.1.8\n"
                "  - CPU capability usage: AVX512\n"
            ),
            "PyTorch built with:\n  - GCC 11.5\n  - CPU capability usage: NEON\n",
            "PyTorch built with:\n  - GCC 13.4\n  - CPU capability usage: SVE256\n",
        )
        for config in scalar_contracting_tail_configs:
            clear_addcmul_caches()
            with (
                patch("torch.__config__.show", return_value=config),
                V.set_kernel_handler(SimpleNamespace(tail_size=3)),
            ):
                self.assertEqual(
                    CppVecOverrides.addcmul_aten("self", "value_times_t1", "t2"),
                    "fmadd(value_times_t1, t2, self)",
                )

        clear_addcmul_caches()
        with (
            patch(
                "torch.__config__.show",
                return_value=(
                    "PyTorch built with:\n"
                    "  - MSVC 19.44\n"
                    "  - CPU capability usage: AVX2\n"
                ),
            ),
            V.set_kernel_handler(SimpleNamespace(tail_size=3)),
        ):
            code = CppVecOverrides.addcmul_aten("self", "value_times_t1", "t2")
            self.assertIn("reinterpret_cast<const volatile char*>(&product);", code)
            self.assertIn("return self + product;", code)
            self.assertNotIn("fmadd", code)

        clear_addcmul_caches()

    def test_halide_overrides(self):
        self.verify_ops_handler_completeness(HalideOverrides)

    @unittest.skip("MPS backend not yet finished")
    def test_metal_overrides(self):
        self.verify_ops_handler_completeness(MetalOverrides)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
