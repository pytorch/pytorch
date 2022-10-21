# Owner(s): ["module: dynamo"]
import os
import shutil
from unittest.mock import patch

import torch

import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.optimizations.backends import create_backend


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


class MinfierTests(torch._dynamo.test_case.TestCase):
    def test_after_dynamo(self):
        @create_backend
        def bad_dynamo_backend(subgraph):
            import sys

            def f(*args):
                # Shifted the forced exception to runtime as this is more common
                # in JIT compilers.
                for node in subgraph.model.graph.nodes:
                    if node.op == "call_function" and node.target is torch._foobar:
                        sys.stdout.write("Dynamo compiled failed\n")
                        raise NotImplementedError("foobar is not implemented")
                return subgraph.model(*args)

            return f

        mod = MockModule()
        opt_mod = torch._dynamo.optimize("bad_dynamo_backend")(mod)
        repro_dir = "/tmp/test_minifier"
        repro_file = os.path.join(repro_dir, "minifier_launcher.py")
        shutil.rmtree(repro_dir, ignore_errors=True)

        @patch.object(torch._dynamo.config, "repro_after", "dynamo")
        @patch.object(torch._dynamo.config, "repro_dir", repro_dir)
        def inner():
            x = torch.randn(4)
            try:
                opt_mod(x)
            except Exception:
                pass

        inner()
        self.assertTrue(os.path.exists(repro_file))

    # If error_at_aot is True, an error will be produced when AOTAutograd
    # attempts to generate the backward graph.
    # If error_after_aot is False, an error will be produced in inductor.
    def _test_around_aot(self, error_at_aot):
        mod = MockModule()
        opt_mod = torch._dynamo.optimize("inductor")(mod)
        repro_dir = "/tmp/test_minifier"
        repro_file = os.path.join(repro_dir, "minifier_launcher.py")
        shutil.rmtree(repro_dir, ignore_errors=True)

        repro_after = "dynamo" if error_at_aot else "aot"

        @patch.object(torch._dynamo.config, "repro_after", repro_after)
        @patch.object(torch._dynamo.config, "repro_dir", repro_dir)
        def inner():
            x = torch.randn(4)
            x.requires_grad = error_at_aot
            try:
                opt_mod(x)
            except Exception:
                pass

        inner()

        self.assertTrue(os.path.exists(repro_file))

    def test_at_aot(self):
        self._test_around_aot(True)

    def test_after_aot(self):
        self._test_around_aot(False)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
