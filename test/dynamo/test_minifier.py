# Owner(s): ["module: dynamo"]
import os
import re
import shutil
import subprocess
import textwrap
import traceback
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
from torch._dynamo.debug_utils import TEST_REPLACEABLE_COMMENT
import torch._dynamo.test_case
import torch._dynamo.testing


class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        for _ in range(10):
            x = torch.sin(x)
        x = torch._foobar(x)
        for _ in range(10):
            x = torch.cos(x)
        return x


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


RELU_COMPILE_ERROR_BACKEND = textwrap.dedent("""
from torch._dynamo.optimizations.backends import register_backend

class DummyDynamoCompileError(Exception):
    pass

# Backend for testing only: causes compile error when relu is encountered in gm
@register_backend
def test_relu_compile_error(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            raise DummyDynamoCompileError("relu found")
    return gm
""")

RELU_RUNTIME_ERROR_BACKEND = textwrap.dedent("""
from torch._dynamo.optimizations.backends import register_backend

class DummyDynamoRuntimeError(Exception):
    pass

# Backend for testing only: causes runtime error when relu is encountered in gm
@register_backend
def test_relu_runtime_error(gm: torch.fx.GraphModule, example_inputs):
    def compiled_fn(*args, **kwargs):
        for node in gm.graph.nodes:
            if node.target == torch.relu:
                raise DummyDynamoRuntimeError("relu found")
        return gm.forward(*args, **kwargs)

    return compiled_fn
""")


class MinfierTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(
                torch._dynamo.config,
                "debug_dir_root",
                "/tmp/_torchdynamo_debug_/",
            )
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(torch._dynamo.config.debug_dir_root, ignore_errors=True)
        cls._exit_stack.close()

    def setUp(self):
        super().setUp()
        torch._dynamo.utils.debug_dir.setup()

    def tearDown(self):
        torch._dynamo.utils.debug_dir.clear()
        super().tearDown()

    # Triggers minifier, runs the generated script, and returns
    # the original error, the repro'd error,
    # the minifier launcher code, and the repro code.
    def _run_repro(self, fn, fn_args, fn_kwargs, repro_after, backend_code):
        launch_file = torch._dynamo.debug_utils.get_minifier_repro_path()
        repro_file = os.path.join(torch._dynamo.debug_utils.minifier_dir(), "repro.py")

        @patch.object(torch._dynamo.config, "repro_after", repro_after)
        @patch.object(torch._dynamo.config, "repro_level", 2)
        def wrapped_fn():
            exn_tb = None
            try:
                fn(*fn_args, **fn_kwargs)
            except Exception as e:
                exn_tb = traceback.format_exc()
            return exn_tb

        exn_tb = wrapped_fn()

        self.assertTrue(os.path.exists(launch_file))

        def inject_backend(filename):
            with open(filename, 'r') as f:
                code = f.read()
            code = code.replace(TEST_REPLACEABLE_COMMENT, backend_code)
            print(code)
            with open(filename, 'w') as f:
                f.write(code)
            return code

        launch_code = inject_backend(launch_file)

        subprocess.run(["python3", launch_file], capture_output=True)
        self.assertTrue(os.path.exists(repro_file))
        repro_code = inject_backend(repro_file)

        proc = subprocess.run(["python3", repro_file], capture_output=True) 

        # with open(launch_file, 'r') as f:
        #     launch_code = f.read()
        # with open(repro_file, 'r') as f:
        #     repro_code = f.read()

        return exn_tb, proc.stderr.decode("utf-8"), launch_code, repro_code

    # Checks if the correct error is found in both stack traces
    def _check_error_in_tb(self, tb1, tb2, compile_error):
        error_name = f"DummyDynamo{'Compile' if compile_error else 'Runtime'}Error"
        self.assertIn(error_name, tb1)
        self.assertIn(error_name, tb2)

    # Test that compile and runtime errors after dynamo can be repro'd (both CPU and CUDA)
    def _test_after_dynamo(self, device, compile_error):
        backend = f"test_relu_{'compile' if compile_error else 'runtime'}_error"
        backend_code = RELU_COMPILE_ERROR_BACKEND if compile_error else RELU_RUNTIME_ERROR_BACKEND
        exec(backend_code)
        @torch._dynamo.optimize(backend)
        def inner(x):
            for _ in range(10):
                x = torch.sin(x)
            x = torch.relu(x)
            for _ in range(10):
                x = torch.cos(x)
            return x

        tb1, tb2, _, _ = self._run_repro(inner, (torch.randn(20, 20).to(device),), {}, "dynamo", backend_code)

        self._check_error_in_tb(tb1, tb2, compile_error)

    def test_after_dynamo_cpu_compile_error(self):
        self._test_after_dynamo("cpu", True)

    def test_after_dynamo_cpu_runtime_error(self):
        self._test_after_dynamo("cpu", False)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_after_dynamo_cuda_compile_error(self):
        self._test_after_dynamo("cuda", True)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_after_dynamo_cuda_runtime_error(self):
        self._test_after_dynamo("cuda", False)

    # Test that a module with mixed cpu/cuda parts with an error after dynamo can be repro'd
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cpu_cuda_module_after_dynamo(self):
        mod = CpuCudaModule()
        exec(RELU_COMPILE_ERROR_BACKEND)
        @torch._dynamo.optimize("test_relu_compile_error")
        def inner(x1, y1):
            x2 = torch.randn(20, 20).cuda()
            y2 = torch.randn(20, 20)
            x3, y3 = mod(x1 + x2, y1 + y2)
            return torch.relu(x3.cpu() + y3)
        
        tb1, tb2, code, _ = self._run_repro(inner, (torch.randn(20, 20).cuda(), torch.randn(20, 20)), {}, "dynamo", RELU_COMPILE_ERROR_BACKEND)
        
        # check if generated minifier code covers all cpu/cuda cases
        self.assertIsNotNone(re.search(r"args.*cuda", code))
        self.assertIsNotNone(re.search(r"args.*cpu", code))
        # search for Linear(...).cuda()
        self.assertIsNotNone(re.search(r"Linear.*cuda", code))
        # search for Linear(...)
        self.assertIsNotNone(re.search(r"Linear(?!.*cuda.*$)", code, re.MULTILINE))
        self.assertIsNotNone(re.search(r"register_buffer.*cuda", code))
        self.assertIsNotNone(re.search(r"register_buffer(?!.*cuda.*$)", code, re.MULTILINE))
        self.assertIsNotNone(re.search(r"Parameter.*cuda", code))
        self.assertIsNotNone(re.search(r"Parameter(?!.*cuda.*$)", code, re.MULTILINE))
        # search for
        # <name> = torch.randn(...)
        # ... = <name>.cuda()
        self.assertIsNotNone(re.search(
            r"(\w+) = torch.randn.*\1\.cuda", code, re.DOTALL
        ))
        # search for
        # <name> = torch.randn(...)
        # no followup call to <name>.cuda()
        self.assertIsNotNone(re.search(
            r"(\w+) = torch.randn(?!.*\1\.cuda\(\).*$)", code, re.DOTALL
        ))

        self._check_error_in_tb(tb1, tb2, True)

    # test if we can actually get a minified graph
    def test_if_graph_minified(self):
        exec(RELU_COMPILE_ERROR_BACKEND)
        @torch._dynamo.optimize("test_relu_compile_error")
        def inner(x):
            for _ in range(20):
                x = torch.sin(x)
            x = torch.relu(x)
            for _ in range(20):
                x = torch.cos(x)
            return x

        tb1, tb2, launch_code, repro_code = self._run_repro(inner, (torch.randn(20, 20),), {}, "dynamo", RELU_COMPILE_ERROR_BACKEND)

        self._check_error_in_tb(tb1, tb2, True)

        # compare the length of the forward functions
        match = re.search(r"def forward.*return", launch_code, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertGreater(match.group(0).count("\n"), 40)

        match = re.search(r"def forward.*return", repro_code, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertLess(match.group(0).count("\n"), 5)


    # If error_at_aot is True, an error will be produced when AOTAutograd
    # attempts to generate the backward graph.
    # If error_after_aot is False, an error will be produced in inductor.
    def _test_around_aot(self, error_at_aot):
        mod = MockModule()
        opt_mod = torch._dynamo.optimize("inductor")(mod)

        def inner():
            x = torch.randn(4)
            x.requires_grad = error_at_aot
            opt_mod(x)

        inner()
        self._run_repro(inner, (), {}, "dynamo" if error_at_aot else "aot")

    def test_at_aot(self):
        self._test_around_aot(True)

    def test_after_aot(self):
        self._test_around_aot(False)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
