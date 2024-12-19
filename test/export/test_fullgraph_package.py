# Owner(s): ["oncall: export"]

import os
import pathlib

import torch

from torch.testing._internal.common_utils import TestCase


class TestFullgraphPackage(TestCase):
    def tearDown(self):
        super().tearDown()
        pathlib.Path(self.path()).unlink(missing_ok=True)

    def path(self):
        return os.path.expandvars(f"/tmp/torchinductor_$USER/model_{self.id()}.pt2")

    def test_basic_function(self):
        @torch.compile(fullgraph=True)
        def f(x, y):
            return x + y

        with torch.compiler._fullgraph_package(path=self.path()):
            f(torch.randn(3), torch.randn(3))

        self.assertTrue(os.path.exists(self.path()))

        inputs = torch.randn(3), torch.randn(3)
        with torch.compiler._fullgraph_package(mode="load", path=self.path()):
            results = f(*inputs)

        expected = f(*inputs)

        self.assertEqual(results, expected)

    def test_basic_module(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)

        mod = torch.compile(Module(), fullgraph=True)

        with torch.compiler._fullgraph_package(path=self.path()):
            mod(torch.randn(3))

        self.assertTrue(os.path.exists(self.path()))

        inputs = (torch.randn(3),)
        with torch.compiler._fullgraph_package(mode="load", path=self.path()):
            results = mod(*inputs)

        expected = mod(*inputs)

        self.assertEqual(results, expected)

    def test_basic_function_dynamo(self):
        @torch.compile(fullgraph=True)
        def f(x, y):
            return x + y

        with torch.compiler._fullgraph_package(path=self.path()):
            f(torch.randn(3), torch.randn(3))

        self.assertTrue(os.path.exists(self.path()))

        inputs = torch.randn(3), torch.randn(3)
        with torch.compiler._fullgraph_package(
            mode="load", path=self.path(), frontend="dynamo"
        ):
            results = f(*inputs)

        expected = f(*inputs)

        self.assertEqual(results, expected)

    def test_basic_module_dynamo(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)

        mod = torch.compile(Module(), fullgraph=True)

        with torch.compiler._fullgraph_package(path=self.path()):
            mod(torch.randn(3))

        self.assertTrue(os.path.exists(self.path()))

        inputs = (torch.randn(3),)
        with torch.compiler._fullgraph_package(
            mode="load", path=self.path(), frontend="dynamo"
        ):
            results = mod(*inputs)

        expected = mod(*inputs)

        self.assertEqual(results, expected)

    def test_multiple_compilations(self):
        def f(x, y):
            return x + y

        f0 = torch.compile(f, fullgraph=True)
        f1 = torch.compile(f, fullgraph=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "Multiple compilations to the same model name.*are not supported",
        ):
            with torch.compiler._fullgraph_package(path=self.path()):
                f0(torch.randn(3), torch.randn(3))
                f1(torch.randn(4, 2), torch.randn(4, 2))

    def test_reentry(self):
        @torch.compile(fullgraph=True)
        def f(x, y):
            return x + y

        with self.assertRaisesRegex(RuntimeError, "is already enabled"):
            with torch.compiler._fullgraph_package(path=self.path()):
                with torch.compiler._fullgraph_package(path=self.path() + ".0"):
                    f(torch.randn(4, 2), torch.randn(4, 2))
