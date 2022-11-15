# Owner(s): ["module: dynamo"]
import functools
import re
import textwrap
import unittest

import torch
import torch._dynamo
from torch._dynamo.test_minifier_common import MinifierTestBase

requires_cuda = functools.partial(
    unittest.skipIf, not torch.cuda.is_available(), "requires cuda"
)

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


class MinfierTests(MinifierTestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    # Test that compile, runtime, and accuracy errors after dynamo can be repro'd (both CPU and CUDA)
    def _test_after_dynamo(self, device, repro_level, backend_code, error_name):
        run_code = textwrap.dedent(
            f"""\
            @torch._dynamo.optimize("{self._get_fn_name(backend_code)}")
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
            @torch._dynamo.optimize("{self._get_fn_name(backend_code)}")
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
            @torch._dynamo.optimize({self._get_fn_name(RELU_CUSTOM_ERROR_BACKEND)})
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
        launch_proc, _ = self._run_minifier_launcher("", repro_dir)
        self.assertIn("RuntimeError", launch_proc.stderr.decode("utf-8"))

    # Test that a module with mixed cpu/cuda parts with an error after dynamo can be repro'd
    @requires_cuda()
    def test_cpu_cuda_module_after_dynamo(self):
        backend_name = self._get_fn_name(RELU_COMPILE_ERROR_BACKEND)

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
        backend_name = self._get_fn_name(RELU_COMPILE_ERROR_BACKEND)

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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
