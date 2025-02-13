# Owner(s): ["oncall: export"]

import os
import pathlib
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase


class TestFullgraphPackage(TestCase):
    def setUp(self):
        if not os.path.exists(os.path.expandvars("/tmp/torchinductor_$USER/")):
            os.makedirs(os.path.expandvars("/tmp/torchinductor_$USER/"))

    def tearDown(self):
        super().tearDown()
        pathlib.Path(self.path()).unlink(missing_ok=True)

    def path(self):
        return os.path.expandvars(f"/tmp/torchinductor_$USER/model_{self.id()}.pt2")

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_fullgraph_package_basic_function(self):
        def f(a, b):
            for i in range(1000):
                a = a + b * i
            return a

        inputs = torch.randn(3, device="cuda"), torch.randn(3, device="cuda")
        expected = f(*inputs)
        f = torch.compile(f, fullgraph=True, name=self.id())
        p, *flags = torch.compiler._enable_fullgraph_package(path=self.path())
        results = f(*inputs)

        p._save()

        self.assertEqual(results, expected)
        torch.compiler._disable_fullgraph_package(flags)

        torch._dynamo.reset()
        torch.compiler._load_fullgraph_package(path=self.path(), names=[self.id()])

        results = f(*inputs)
        self.assertEqual(results, expected)

        
if __name__ == "__main__":
    run_tests()
