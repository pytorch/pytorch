# Owner(s): ["module: dynamo"]
import functools
import os
import re
import shutil
import subprocess
import textwrap
import unittest

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._inductor.utils
from torch._dynamo.debug_utils import TEST_REPLACEABLE_COMMENT

_HAS_TRITON = torch._inductor.utils.has_triton()
requires_cuda = functools.partial(unittest.skipIf, not _HAS_TRITON, "requires cuda")

RELU_COMPILE_ERROR_BACKEND = """\
from torch._dynamo.optimizations.backends import register_backend

class DynamoCompileError(Exception):
    pass

@register_backend
def test_relu_compile_error(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            raise DynamoCompileError("relu found")
    return gm
"""

RELU_RUNTIME_ERROR_BACKEND = """\
import copy
from torch._dynamo.optimizations.backends import register_backend

@register_backend
def test_relu_runtime_error(gm: torch.fx.GraphModule, example_inputs):
    gm = copy.deepcopy(gm)
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            node.target = torch._assert
            node.args = (False, "DynamoRuntimeError")
    gm.recompile()
    return gm
"""

RELU_ACCURACY_ERROR_BACKEND = """\
import copy
from torch._dynamo.optimizations.backends import register_backend

@register_backend
def test_relu_accuracy_error(gm: torch.fx.GraphModule, example_inputs):
    gm = copy.deepcopy(gm)
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            node.target = torch.add
            node.args = (node.args[0], 1)
    gm.recompile()

    return gm
"""

RELU_CUSTOM_ERROR_BACKEND = """\
class CustomError(Exception):
    pass

def test_relu_custom_error(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            raise CustomError("relu found")
    return gm
"""

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

DEBUG_DIR = "/tmp/_torchdynamo_debug_/"

# Search for the name of the first function defined in a code string.
def get_fn_name(code):
    fn_name_match = re.search(r"def (\w+)\(", code)
    if fn_name_match is not None:
        return fn_name_match.group(1)
    return None


# Generates code that patches CppOverrides/TritonOverrides.
def gen_codegen_fn_patch_code(old_fn_name, new_fn_code, device):
    new_fn_name = get_fn_name(new_fn_code)
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
torch._dynamo.config.debug_dir_root = "{DEBUG_DIR}"
\"\"\"
"""

    return None


class MinfierTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(
                torch._dynamo.config,
                "debug_dir_root",
                DEBUG_DIR,
            )
        )
        os.makedirs(DEBUG_DIR, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(torch._dynamo.config.debug_dir_root, ignore_errors=True)
        cls._exit_stack.close()

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    # Run `code` in a separate python process.
    # Returns the completed process state and the directory containing the
    # minifier launcher script, if `code` outputted it.
    def _run_test_code(self, code):
        proc = subprocess.run(
            ["python3", "-c", code], capture_output=True, cwd=DEBUG_DIR
        )

        repro_dir_match = re.search(
            r"(\S+)minifier_launcher.py", proc.stderr.decode("utf-8")
        )
        if repro_dir_match is not None:
            # Print repro directory for debugging generated code.
            # Make sure to comment out `shutil.rmtree...` above as well.
            print("repro dir:", repro_dir_match.group(1))
            return proc, repro_dir_match.group(1)
        return proc, None

    # Patch generated files with testing patches
    def _inject_code(self, patch_code, filename):
        patch_code = f"""\
{patch_code}
torch._dynamo.config.debug_dir_root = "{DEBUG_DIR}"
"""
        with open(filename, "r") as f:
            code = f.read()
        code = code.replace(TEST_REPLACEABLE_COMMENT, patch_code)
        with open(filename, "w") as f:
            f.write(code)
        return code

    # Runs the minifier launcher script in `repro_dir`, patched with `patch_code`.
    def _run_minifier_launcher(self, patch_code, repro_dir):
        self.assertIsNotNone(repro_dir)
        launch_file = os.path.join(repro_dir, "minifier_launcher.py")
        self.assertTrue(os.path.exists(launch_file))
        launch_code = self._inject_code(patch_code, launch_file)

        launch_proc = subprocess.run(
            ["python3", launch_file],
            capture_output=True,
            cwd=repro_dir,
        )

        return launch_proc, launch_code

    # Runs the repro script in `repro_dir`, patched with `patch_code`
    def _run_repro(self, patch_code, repro_dir):
        self.assertIsNotNone(repro_dir)
        repro_file = os.path.join(repro_dir, "repro.py")
        self.assertTrue(os.path.exists(repro_file))
        repro_code = self._inject_code(patch_code, repro_file)

        repro_proc = subprocess.run(
            ["python3", repro_file], capture_output=True, cwd=repro_dir
        )

        return repro_proc, repro_code

    # Template for testing code.
    # `run_code` is the code to run for the test case.
    # `patch_code` is the code to be patched in every generated file.
    def _gen_test_code(self, run_code, repro_after, repro_level, patch_code):
        return f"""\
import torch
import torch._dynamo
{patch_code}
torch._dynamo.config.repro_after = "{repro_after}"
torch._dynamo.config.repro_level = {repro_level}
torch._dynamo.config.debug_dir_root = "{DEBUG_DIR}"
{run_code}
"""

    # Runs a full minifier test.
    # Minifier tests generally consist of 3 stages:
    # 1. Run the problematic code (in a separate process since it could segfault)
    # 2. Run the generated minifier launcher script
    # 3. Run the generated repro script
    def _run_full_test(self, run_code, repro_after, repro_level, patch_code):
        test_code = self._gen_test_code(run_code, repro_after, repro_level, patch_code)
        test_proc, repro_dir = self._run_test_code(test_code)
        self.assertIsNotNone(repro_dir)
        launch_proc, launch_code = self._run_minifier_launcher(patch_code, repro_dir)
        repro_proc, repro_code = self._run_repro(patch_code, repro_dir)
        return ((test_proc, launch_proc, repro_proc), (launch_code, repro_code))

    # Test that compile, runtime, and accuracy errors after dynamo can be repro'd (both CPU and CUDA)
    def _test_after_dynamo(self, device, repro_level, backend_code, error_name):
        run_code = textwrap.dedent(
            f"""\
            @torch._dynamo.optimize("{get_fn_name(backend_code)}")
            def inner(x):
                for _ in range(10):
                    x = torch.sin(x)
                x = torch.relu(x)
                for _ in range(10):
                    x = torch.cos(x)
                return x

            inner(torch.randn(20, 20).to("{device}"))
        """
        )

        (test_proc, _, repro_proc), _ = self._run_full_test(
            run_code, "dynamo", repro_level, backend_code
        )

        self.assertIn(error_name, test_proc.stderr.decode("utf-8"))
        self.assertIn(error_name, repro_proc.stderr.decode("utf-8"))

    def test_after_dynamo_cpu_compile_error(self):
        self._test_after_dynamo(
            "cpu", 2, RELU_COMPILE_ERROR_BACKEND, "DynamoCompileError"
        )

    def test_after_dynamo_cpu_runtime_error(self):
        self._test_after_dynamo(
            "cpu", 2, RELU_RUNTIME_ERROR_BACKEND, "DynamoRuntimeError"
        )

    def test_after_dynamo_cpu_accuracy_error(self):
        self._test_after_dynamo("cpu", 4, RELU_ACCURACY_ERROR_BACKEND, "AccuracyError")

    @requires_cuda()
    def test_after_dynamo_cuda_compile_error(self):
        self._test_after_dynamo(
            "cuda", 2, RELU_COMPILE_ERROR_BACKEND, "DynamoCompileError"
        )

    @requires_cuda()
    def test_after_dynamo_cuda_runtime_error(self):
        self._test_after_dynamo(
            "cuda", 2, RELU_RUNTIME_ERROR_BACKEND, "DynamoRuntimeError"
        )

    @requires_cuda()
    def test_after_dynamo_cuda_accuracy_error(self):
        self._test_after_dynamo("cuda", 4, RELU_ACCURACY_ERROR_BACKEND, "AccuracyError")

    # Ensure that the testing backends pass when relu is not present.
    def _test_after_dynamo_backend_passes(self, device, repro_level, backend_code):
        run_code = textwrap.dedent(
            f"""\
            @torch._dynamo.optimize("{get_fn_name(backend_code)}")
            def inner(x):
                for _ in range(10):
                    x = torch.sin(x)
                for _ in range(10):
                    x = torch.cos(x)
                return x

            inner(torch.randn(20, 20).to("{device}"))
        """
        )

        test_code = self._gen_test_code(run_code, "dynamo", repro_level, backend_code)
        proc, repro_dir = self._run_test_code(test_code)
        self.assertEqual(proc.returncode, 0)
        self.assertIsNone(repro_dir)

    def test_after_dynamo_cpu_compile_backend_passes(self):
        self._test_after_dynamo_backend_passes("cpu", 2, RELU_COMPILE_ERROR_BACKEND)

    def test_after_dynamo_cpu_runtime_backend_passes(self):
        self._test_after_dynamo_backend_passes("cpu", 2, RELU_RUNTIME_ERROR_BACKEND)

    def test_after_dynamo_cpu_accuracy_backend_passes(self):
        self._test_after_dynamo_backend_passes("cpu", 4, RELU_ACCURACY_ERROR_BACKEND)

    @requires_cuda()
    def test_after_dynamo_cuda_compile_backend_passes(self):
        self._test_after_dynamo_backend_passes("cuda", 2, RELU_COMPILE_ERROR_BACKEND)

    @requires_cuda()
    def test_after_dynamo_cuda_runtime_backend_passes(self):
        self._test_after_dynamo_backend_passes("cuda", 2, RELU_RUNTIME_ERROR_BACKEND)

    @requires_cuda()
    def test_after_dynamo_cuda_accuracy_backend_passes(self):
        self._test_after_dynamo_backend_passes("cuda", 4, RELU_ACCURACY_ERROR_BACKEND)

    # Ensure that generated code with a custom backends generates a runnable minifier
    # launcher script that results in a RuntimeError
    def test_after_dynamo_custom_backend(self):
        run_code = textwrap.dedent(
            f"""\
            @torch._dynamo.optimize({get_fn_name(RELU_CUSTOM_ERROR_BACKEND)})
            def inner(x):
                for _ in range(10):
                    x = torch.sin(x)
                x = torch.relu(x)
                for _ in range(10):
                    x = torch.cos(x)
                return x

            inner(torch.randn(20, 20))
        """
        )

        test_code = self._gen_test_code(
            run_code, "dynamo", 2, RELU_CUSTOM_ERROR_BACKEND
        )
        _, repro_dir = self._run_test_code(test_code)
        launch_proc, launch_code = self._run_minifier_launcher("", repro_dir)
        self.assertIn("RuntimeError", launch_proc.stderr.decode("utf-8"))

    # Test that a module with mixed cpu/cuda parts with an error after dynamo can be repro'd
    @requires_cuda()
    def test_cpu_cuda_module_after_dynamo(self):
        backend_name = get_fn_name(RELU_COMPILE_ERROR_BACKEND)

        run_code = textwrap.dedent(
            f"""\
            class CpuCudaModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.m_x = torch.nn.Linear(20, 20).cuda()
                    self.m_y = torch.nn.Linear(20, 20)
                    self.p_x = torch.nn.Parameter(torch.randn(20, 20).cuda())
                    self.p_y = torch.nn.Parameter(torch.randn(20, 20))
                    self.register_buffer("b_x", torch.ones(20, 20).cuda())
                    self.register_buffer("b_y", torch.ones(20, 20))

                def forward(self, x, y):
                    return self.m_x(x) + self.p_x + self.b_x, self.m_y(y) + self.p_y + self.b_y

            mod = CpuCudaModule()

            @torch._dynamo.optimize("{backend_name}")
            def inner(x1, y1):
                x2 = torch.randn(20, 20).cuda()
                y2 = torch.randn(20, 20)
                x3, y3 = mod(x1 + x2, y1 + y2)
                return torch.relu(x3.cpu() + y3)

            inner(torch.randn(20, 20).cuda(), torch.randn(20, 20))
        """
        )

        (test_proc, _, repro_proc), (launch_code, _) = self._run_full_test(
            run_code, "dynamo", 2, RELU_COMPILE_ERROR_BACKEND
        )

        tb1 = test_proc.stderr.decode("utf-8")
        tb2 = repro_proc.stderr.decode("utf-8")

        # Check if generated minifier code covers all cpu/cuda cases
        self.assertIsNotNone(re.search(r"args.*cuda", launch_code))
        self.assertIsNotNone(re.search(r"args.*cpu", launch_code))
        # search for Linear(...).cuda()
        self.assertIsNotNone(re.search(r"Linear.*cuda", launch_code))
        # search for Linear(...)
        self.assertIsNotNone(
            re.search(r"Linear(?!.*cuda.*$)", launch_code, re.MULTILINE)
        )
        self.assertIsNotNone(re.search(r"register_buffer.*cuda", launch_code))
        self.assertIsNotNone(
            re.search(r"register_buffer(?!.*cuda.*$)", launch_code, re.MULTILINE)
        )
        self.assertIsNotNone(re.search(r"Parameter.*cuda", launch_code))
        self.assertIsNotNone(
            re.search(r"Parameter(?!.*cuda.*$)", launch_code, re.MULTILINE)
        )
        # search for
        # <name> = torch.randn(...)
        # ... = <name>.cuda()
        self.assertIsNotNone(
            re.search(r"(\w+) = torch.randn.*\1\.cuda", launch_code, re.DOTALL)
        )
        # search for
        # <name> = torch.randn(...)
        # no followup call to <name>.cuda()
        self.assertIsNotNone(
            re.search(
                r"(\w+) = torch.randn(?!.*\1\.cuda\(\).*$)", launch_code, re.DOTALL
            )
        )

        self.assertIn(backend_name, tb1)
        self.assertIn(backend_name, tb2)

    # Test if we can actually get a minified graph
    def test_if_graph_minified(self):
        backend_name = get_fn_name(RELU_COMPILE_ERROR_BACKEND)

        run_code = textwrap.dedent(
            f"""\
            @torch._dynamo.optimize("{backend_name}")
            def inner(x):
                for _ in range(20):
                    x = torch.sin(x)
                x = torch.relu(x)
                for _ in range(20):
                    x = torch.cos(x)
                return x

            inner(torch.randn(20, 20))
        """
        )

        (test_proc, _, repro_proc), (launch_code, repro_code) = self._run_full_test(
            run_code, "dynamo", 2, RELU_COMPILE_ERROR_BACKEND
        )

        tb1 = test_proc.stderr.decode("utf-8")
        tb2 = repro_proc.stderr.decode("utf-8")

        self.assertIn(backend_name, tb1)
        self.assertIn(backend_name, tb2)

        # compare the length of the forward functions
        match = re.search(r"def forward.*return", launch_code, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertGreater(match.group(0).count("\n"), 40)

        match = re.search(r"def forward.*return", repro_code, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertLess(match.group(0).count("\n"), 5)

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
        patch_code = gen_codegen_fn_patch_code("relu", backend_code, device)
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
        patch_code = gen_codegen_fn_patch_code("relu", backend_code, device)
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
        patch_code = gen_codegen_fn_patch_code("relu", backend_code, device)
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
