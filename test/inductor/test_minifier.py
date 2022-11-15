# Owner(s): ["module: inductor"]
import functools
import textwrap
import unittest

import torch
import torch._dynamo
import torch._inductor.utils
from torch._dynamo.test_minifier_common import MinifierTestBase

_HAS_TRITON = torch._inductor.utils.has_triton()
requires_cuda = functools.partial(unittest.skipIf, not _HAS_TRITON, "requires cuda")

CPP_COMPILE_ERROR = """\
def cpp_compile_error(x):
    return "compile error!"
"""

CPP_RUNTIME_ERROR = """\
def cpp_runtime_error(x):
    return f"{x}; throw 1"
"""

CPP_ACCURACY_ERROR = """\
def cpp_accuracy_error(x):
    return f"{x} + 1"
"""

TRITON_COMPILE_ERROR = """\
def triton_compile_error(x):
    return "compile error!"
"""

# NOTE: there is currently not an easy way to cause a triton runtime error.
TRITON_RUNTIME_ERROR = """\
def triton_runtime_error(x):
    return f"{x}; assert?"
"""

TRITON_ACCURACY_ERROR = """\
def triton_accuracy_error(x):
    return f"{x} + 1"
"""


class MinfierTests(MinifierTestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    # Generates code that patches CppOverrides/TritonOverrides.
    def _gen_codegen_fn_patch_code(self, old_fn_name, new_fn_code, device):
        new_fn_name = self._get_fn_name(new_fn_code)
        if new_fn_name is not None:
            patch_code = f"""\
import torch._inductor.codegen.{"cpp" if device == "cpu" else "triton"} as codegen
overrides = codegen.{"CppOverrides" if device == "cpu" else "TritonOverrides"}
{new_fn_code}
overrides.{old_fn_name} = staticmethod({new_fn_name})
"""
        return f"""\
{patch_code}
isolate_fails_code_str = \"\"\"\\
{patch_code}
torch._dynamo.config.debug_dir_root = "{self.DEBUG_DIR}"
\"\"\"
"""

    # Test that compile and accuracy errors after aot can be repro'd (both CPU and CUDA)
    def _test_after_aot(self, device, backend_code, repro_level):
        run_code = textwrap.dedent(
            f"""\
            @torch._dynamo.optimize("inductor")
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
        patch_code = self._gen_codegen_fn_patch_code("relu", backend_code, device)
        self.assertIsNotNone(patch_code)
        (test_proc, _, repro_proc), _ = self._run_full_test(
            run_code, "aot", repro_level, patch_code
        )
        return (
            (test_proc.stderr.decode("utf-8"), repro_proc.stderr.decode("utf-8")),
            (test_proc.returncode, repro_proc.returncode),
        )

    def test_after_aot_cpu_compile_error(self):
        (tb1, tb2), _ = self._test_after_aot("cpu", CPP_COMPILE_ERROR, 2)
        self.assertIn("CppCompileError", tb1)
        self.assertIn("CppCompileError", tb2)

    def test_after_aot_cpu_accuracy_error(self):
        (tb1, tb2), _ = self._test_after_aot("cpu", CPP_ACCURACY_ERROR, 4)
        self.assertIn("AccuracyError", tb1)
        self.assertIn("AccuracyError", tb2)

    @requires_cuda()
    def test_after_aot_cuda_compile_error(self):
        (tb1, tb2), _ = self._test_after_aot("cuda", TRITON_COMPILE_ERROR, 2)
        self.assertIn("SyntaxError", tb1)
        self.assertIn("SyntaxError", tb2)

    @requires_cuda()
    def test_after_aot_cuda_accuracy_error(self):
        (tb1, tb2), _ = self._test_after_aot("cuda", TRITON_ACCURACY_ERROR, 4)
        self.assertIn("AccuracyError", tb1)
        self.assertIn("AccuracyError", tb2)

    # Test that runtime errors after aot can be repro'd (CPU only for now)
    def _test_after_aot_runtime_error(self, device, backend_code):
        run_code = textwrap.dedent(
            f"""\
            @torch._dynamo.optimize("inductor")
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
        patch_code = self._gen_codegen_fn_patch_code("relu", backend_code, device)
        self.assertIsNotNone(patch_code)

        (test_proc, _, repro_proc), _ = self._run_full_test(
            run_code, "aot", 3, patch_code
        )

        self.assertNotIn("CompilerError", test_proc.stderr.decode("utf-8"))

        self.assertEqual(test_proc.returncode, repro_proc.returncode)
        self.assertNotEqual(test_proc.returncode, 0)

    def test_after_aot_cpu_runtime_error(self):
        self._test_after_aot_runtime_error("cpu", CPP_RUNTIME_ERROR)

    # NOTE: there is currently not an easy way to cause a triton runtime error.
    @unittest.skip
    @requires_cuda()
    def test_after_aot_cuda_runtime_error(self):
        self._test_after_aot_runtime_error("cuda", TRITON_RUNTIME_ERROR)

    # Ensure that inductor codegen patches pass when relu is not present.
    def _test_after_aot_backend_passes(self, device, repro_level, backend_code):
        run_code = textwrap.dedent(
            f"""\
            @torch._dynamo.optimize("inductor")
            def inner(x):
                for _ in range(3):
                    x = torch.sin(x)
                for _ in range(3):
                    x = torch.cos(x)
                return x

            inner(torch.randn(20, 20).to("{device}"))
        """
        )
        patch_code = self._gen_codegen_fn_patch_code("relu", backend_code, device)
        self.assertIsNotNone(patch_code)

        test_code = self._gen_test_code(run_code, "aot", repro_level, patch_code)
        proc, repro_dir = self._run_test_code(test_code)
        self.assertEqual(proc.returncode, 0)
        self.assertIsNone(repro_dir)

    def test_after_aot_cpu_compile_backend_passes(self):
        self._test_after_aot_backend_passes("cpu", 2, CPP_COMPILE_ERROR)

    def test_after_aot_cpu_runtime_backend_passes(self):
        self._test_after_aot_backend_passes("cpu", 2, CPP_RUNTIME_ERROR)

    def test_after_aot_cpu_accuracy_backend_passes(self):
        self._test_after_aot_backend_passes("cpu", 4, CPP_ACCURACY_ERROR)

    @requires_cuda()
    def test_after_aot_cuda_compile_backend_passes(self):
        self._test_after_aot_backend_passes("cuda", 2, TRITON_COMPILE_ERROR)

    # NOTE: there is currently not an easy way to cause a triton runtime error.
    @unittest.skip
    @requires_cuda()
    def test_after_aot_cuda_runtime_backend_passes(self):
        self._test_after_aot_backend_passes("cuda", 2, TRITON_RUNTIME_ERROR)

    @requires_cuda()
    def test_after_aot_cuda_accuracy_backend_passes(self):
        self._test_after_aot_backend_passes("cuda", 4, TRITON_ACCURACY_ERROR)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
