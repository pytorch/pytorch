# Owner(s): ["module: inductor"]
import functools
import textwrap
import unittest

import torch
import torch._dynamo
import torch._inductor.utils
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch.testing._internal.common_utils import IS_JETSON, IS_MACOS, TEST_WITH_ASAN

_HAS_TRITON = torch._inductor.utils.has_triton()
requires_cuda = functools.partial(unittest.skipIf, not _HAS_TRITON, "requires cuda")


class MinifierTests(MinifierTestBase):
    # Generates code that patches CppOverrides/TritonOverrides.
    def _gen_codegen_fn_patch_code(self, device, bug_type):
        assert bug_type in ("compile_error", "runtime_error", "accuracy")
        return f"""\
{torch._dynamo.config.codegen_config()}
{torch._inductor.config.codegen_config()}
torch._inductor.config.{"cpp" if device == "cpu" else "triton"}.inject_relu_bug_TESTING_ONLY = {bug_type!r}
"""

    # Test that compile and accuracy errors after aot can be repro'd (both CPU and CUDA)
    def _test_after_aot(self, device, bug_type, repro_level):
        run_code = textwrap.dedent(
            f"""\
            @torch.compile()
            def inner(x):
                for _ in range(3):
                    x = torch.sin(x)
                x = torch.relu(x)
                for _ in range(3):
                    x = torch.cos(x)
                return x

            inner(torch.randn(20, 20).to("{device}"))
        """
        )
        patch_code = self._gen_codegen_fn_patch_code(device, bug_type)
        self.assertIsNotNone(patch_code)
        test_proc, _, repro_proc = self._run_full_test_nocode(
            run_code, "aot", repro_level, patch_code
        )
        return test_proc.stderr.decode("utf-8"), repro_proc.stderr.decode("utf-8")

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    def test_after_aot_cpu_compile_error(self):
        tb1, tb2 = self._test_after_aot("cpu", "compile_error", 2)
        self.assertIn("CppCompileError", tb1)
        self.assertIn("CppCompileError", tb2)

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    def test_after_aot_cpu_accuracy_error(self):
        tb1, tb2 = self._test_after_aot("cpu", "accuracy", 4)
        self.assertIn("AccuracyError", tb1)
        self.assertIn("AccuracyError", tb2)

    @requires_cuda()
    def test_after_aot_cuda_compile_error(self):
        tb1, tb2 = self._test_after_aot("cuda", "compile_error", 2)
        self.assertIn("SyntaxError", tb1)
        self.assertIn("SyntaxError", tb2)

    @requires_cuda()
    def test_after_aot_cuda_accuracy_error(self):
        tb1, tb2 = self._test_after_aot("cuda", "accuracy", 4)
        self.assertIn("AccuracyError", tb1)
        self.assertIn("AccuracyError", tb2)

    # Test that runtime errors after aot can be repro'd (CPU only for now)
    def _test_after_aot_runtime_error(self, device, bug_type):
        run_code = textwrap.dedent(
            f"""\
            @torch.compile()
            def inner(x):
                for _ in range(3):
                    x = torch.sin(x)
                x = torch.relu(x)
                for _ in range(3):
                    x = torch.cos(x)
                return x

            inner(torch.randn(20, 20).to("{device}"))
        """
        )
        patch_code = self._gen_codegen_fn_patch_code(device, bug_type)
        self.assertIsNotNone(patch_code)

        test_proc, _, repro_proc = self._run_full_test_nocode(
            run_code, "aot", 3, patch_code
        )

        self.assertNotIn("CompilerError", test_proc.stderr.decode("utf-8"))

        self.assertEqual(test_proc.returncode, repro_proc.returncode)
        self.assertNotEqual(test_proc.returncode, 0)

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    def test_after_aot_cpu_runtime_error(self):
        self._test_after_aot_runtime_error("cpu", "runtime_error")

    @requires_cuda()
    def test_after_aot_cuda_runtime_error(self):
        self._test_after_aot_runtime_error("cuda", "runtime_error")


if __name__ == "__main__":
    import sys

    from torch._dynamo.test_case import run_tests

    # Skip CI tests on mac since CPU inductor does not seem to work due to C++ compile errors,
    # also skip on ASAN due to https://github.com/pytorch/pytorch/issues/98262
    # also skip on Py 3.11+ since unhandled exceptions can cause segfaults
    if not IS_MACOS and not TEST_WITH_ASAN and sys.version_info < (3, 11):
        run_tests()
