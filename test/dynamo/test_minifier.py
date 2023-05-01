# Owner(s): ["module: dynamo"]
import functools
import re
import textwrap
import unittest

import torch._dynamo
from torch._dynamo.test_minifier_common import MinifierTestBase

requires_cuda = functools.partial(
    unittest.skipIf, not torch.cuda.is_available(), "requires cuda"
)


class MinifierTests(MinifierTestBase):
    # Test that compile, runtime, and accuracy errors after dynamo can be repro'd (both CPU and CUDA)
    def _test_after_dynamo(self, device, repro_level, backend, error_name):
        run_code = textwrap.dedent(
            f"""\
            @torch._dynamo.optimize({backend!r})
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

        test_proc, _, repro_proc = self._run_full_test_nocode(
            run_code, "dynamo", repro_level, ""
        )

        self.assertIn(error_name, test_proc.stderr.decode("utf-8"))
        self.assertIn(error_name, repro_proc.stderr.decode("utf-8"))

    def test_after_dynamo_cpu_compile_error(self):
        self._test_after_dynamo(
            "cpu", 2, "relu_compile_error_TESTING_ONLY", "ReluCompileError"
        )

    def test_after_dynamo_cpu_runtime_error(self):
        self._test_after_dynamo(
            "cpu", 2, "relu_runtime_error_TESTING_ONLY", "ReluRuntimeError"
        )

    def test_after_dynamo_cpu_accuracy_error(self):
        self._test_after_dynamo(
            "cpu", 4, "relu_accuracy_error_TESTING_ONLY", "AccuracyError"
        )

    @requires_cuda()
    def test_after_dynamo_cuda_compile_error(self):
        self._test_after_dynamo(
            "cuda", 2, "relu_compile_error_TESTING_ONLY", "ReluCompileError"
        )

    @requires_cuda()
    def test_after_dynamo_cuda_runtime_error(self):
        self._test_after_dynamo(
            "cuda", 2, "relu_runtime_error_TESTING_ONLY", "ReluRuntimeError"
        )

    @requires_cuda()
    def test_after_dynamo_cuda_accuracy_error(self):
        self._test_after_dynamo(
            "cuda", 4, "relu_accuracy_error_TESTING_ONLY", "AccuracyError"
        )

    # Ensure that the testing backends pass when relu is not present.
    def _test_after_dynamo_backend_passes(self, device, repro_level, backend):
        @torch._dynamo.optimize(backend)
        def inner(x):
            for _ in range(10):
                x = torch.sin(x)
            for _ in range(10):
                x = torch.cos(x)
            return x

        inner(torch.randn(20, 20).to(device))

    def test_after_dynamo_cpu_compile_backend_passes(self):
        self._test_after_dynamo_backend_passes(
            "cpu", 2, "relu_compile_error_TESTING_ONLY"
        )

    def test_after_dynamo_cpu_runtime_backend_passes(self):
        self._test_after_dynamo_backend_passes(
            "cpu", 2, "relu_runtime_error_TESTING_ONLY"
        )

    def test_after_dynamo_cpu_accuracy_backend_passes(self):
        self._test_after_dynamo_backend_passes(
            "cpu", 4, "relu_accuracy_error_TESTING_ONLY"
        )

    @requires_cuda()
    def test_after_dynamo_cuda_compile_backend_passes(self):
        self._test_after_dynamo_backend_passes(
            "cuda", 2, "relu_compile_error_TESTING_ONLY"
        )

    @requires_cuda()
    def test_after_dynamo_cuda_runtime_backend_passes(self):
        self._test_after_dynamo_backend_passes(
            "cuda", 2, "relu_runtime_error_TESTING_ONLY"
        )

    @requires_cuda()
    def test_after_dynamo_cuda_accuracy_backend_passes(self):
        self._test_after_dynamo_backend_passes(
            "cuda", 4, "relu_accuracy_error_TESTING_ONLY"
        )

    # Test that a module with mixed cpu/cuda parts with an error after dynamo can be repro'd
    @requires_cuda()
    def test_cpu_cuda_module_after_dynamo(self):
        backend_name = "relu_compile_error_TESTING_ONLY"
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

            @torch._dynamo.optimize({backend_name!r})
            def inner(x1, y1):
                x2 = torch.randn(20, 20).cuda()
                y2 = torch.randn(20, 20)
                x3, y3 = mod(x1 + x2, y1 + y2)
                return torch.relu(x3.cpu() + y3)

            inner(torch.randn(20, 20).cuda(), torch.randn(20, 20))
        """
        )

        (test_proc, _, repro_proc), (launch_code, _) = self._run_full_test(
            run_code, "dynamo", 2, ""
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
        backend_name = "relu_compile_error_TESTING_ONLY"
        run_code = textwrap.dedent(
            f"""\
            @torch._dynamo.optimize({backend_name!r})
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
            run_code, "dynamo", 2, ""
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
