# Owner(s): ["module: inductor"]
import importlib
import os
import sys

import torch
from torch._inductor import config


importlib.import_module("filelock")

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model_gpu,
    TestCase,
)


# TODO: Remove this file.
# This is a temporary test case to test the base functionality of first Intel GPU Inductor integration.
# We are working on reuse and pass the test cases in test/inductor/*  step by step.
# Will remove this file when pass full test in test/inductor/*.


class XpuBasicTests(TestCase):
    common = check_model_gpu
    device = "xpu"

    def test_add(self):
        def fn(a, b):
            return a + b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))

    def test_sub(self):
        def fn(a, b):
            return a - b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))

    def test_mul(self):
        def fn(a, b):
            return a * b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))

    def test_div(self):
        def fn(a, b):
            return a / b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))

    @config.patch(
        {
            "graph_partition": False,
            "triton.cudagraphs": True,
            "triton.cudagraph_trees": True,
        }
    )
    def test_unregistered_graph_tree_backend_fallback(self):
        from torch._dynamo.utils import counters
        from torch._inductor.graph_tree_backend import is_graph_tree_backend_available

        self.assertEqual(torch.accelerator.current_accelerator().type, "xpu")
        self.assertFalse(is_graph_tree_backend_available())
        counters.clear()

        def fn(x):
            return torch.sin(x) + 1

        x = torch.randn(16, device="xpu")
        expected = fn(x)
        actual = torch.compile(fn)(x)

        self.assertEqual(actual, expected)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_XPU_AND_TRITON

    if HAS_XPU_AND_TRITON:
        run_tests(needs="filelock")
