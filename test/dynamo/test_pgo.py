# Owner(s): ["module: dynamo"]

import contextlib

import torch._dynamo.config
import torch._dynamo.test_case
import torch.compiler.config
from torch._dynamo.testing import CompileCounter
from torch._inductor.utils import fresh_inductor_cache


class PgoTest(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch.compiler.config.patch(workflow_id=self.id())
        )
        self._test_stack.enter_context(
            torch._dynamo.config.patch(automatic_dynamic_local_pgo=True)
        )
        self._test_stack.enter_context(fresh_inductor_cache())

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()
        self._test_stack.close()

    def test_basic(self):
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            return x * 2

        f(torch.randn(2, 3))
        f(torch.randn(2, 4))
        self.assertEqual(cnts.frame_count, 2)

        torch._dynamo.reset()
        cnts.clear()

        f(torch.randn(2, 5))
        f(torch.randn(2, 6))
        self.assertEqual(cnts.frame_count, 1)

    def test_distinct_compile_id(self):
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            return x * 2

        with torch.compiler.config.patch(workflow_id="foo"):
            f(torch.randn(2, 3))
            f(torch.randn(2, 4))
        self.assertEqual(cnts.frame_count, 2)

        torch._dynamo.reset()
        cnts.clear()

        with torch.compiler.config.patch(workflow_id="bar"):
            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
        self.assertEqual(cnts.frame_count, 2)

        torch._dynamo.reset()
        cnts.clear()

        with torch.compiler.config.patch(workflow_id="foo"):
            f(torch.randn(2, 7))
            f(torch.randn(2, 8))
        self.assertEqual(cnts.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
