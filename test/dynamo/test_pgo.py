# Owner(s): ["module: dynamo"]

import contextlib
import os

import torch._dynamo.config
import torch._dynamo.test_case
import torch._inductor.mock_cache as mock_cache
import torch.compiler.config
import torch.nested
from torch._dynamo.testing import CompileCounter
from torch._inductor.utils import clear_inductor_caches, fresh_inductor_cache


class PgoTest(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(torch.compiler.config.patch(job_id=self.id()))
        self._test_stack.enter_context(
            torch._dynamo.config.patch(automatic_dynamic_local_pgo=True)
        )
        if os.environ.get("INDUCTOR_TEST_DISABLE_FRESH_CACHE") != "1":
            self._test_stack.enter_context(fresh_inductor_cache())
        mock_cache.PatchCaches.setUp()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()
        self._test_stack.close()
        mock_cache.PatchCaches.tearDown()

    def reset(self):
        torch._dynamo.reset()
        clear_inductor_caches()

    def test_basic(self):
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            return x * 2

        f(torch.randn(2, 3))
        f(torch.randn(2, 4))
        self.assertEqual(cnts.frame_count, 2)

        self.reset()
        cnts.clear()

        f(torch.randn(2, 5))
        f(torch.randn(2, 6))
        self.assertEqual(cnts.frame_count, 1)

    def test_njt(self):
        cnts = CompileCounter()

        # NB: PGO doesn't do anything here, the point is to catch pickle
        # problem with nested int

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            return x * 2

        x = torch.nested.nested_tensor_from_jagged(
            torch.randn(10, 3), torch.tensor([0, 3, 7, 10]), torch.tensor([1, 2, 3])
        )
        y = torch.nested.nested_tensor_from_jagged(
            torch.randn(13, 3), torch.tensor([0, 3, 7, 13]), torch.tensor([1, 2, 6])
        )

        f(x)
        f(y)
        self.assertEqual(cnts.frame_count, 1)

        self.reset()
        cnts.clear()

        a = torch.nested.nested_tensor_from_jagged(
            torch.randn(14, 3), torch.tensor([0, 3, 7, 14]), torch.tensor([1, 2, 7])
        )
        b = torch.nested.nested_tensor_from_jagged(
            torch.randn(15, 3), torch.tensor([0, 3, 7, 15]), torch.tensor([1, 2, 8])
        )

        f(a)
        f(b)
        self.assertEqual(cnts.frame_count, 1)

    def test_distinct_compile_id(self):
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            return x * 2

        with torch.compiler.config.patch(job_id="foo"):
            f(torch.randn(2, 3))
            f(torch.randn(2, 4))
        self.assertEqual(cnts.frame_count, 2)

        self.reset()
        cnts.clear()

        with torch.compiler.config.patch(job_id="bar"):
            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
        self.assertEqual(cnts.frame_count, 2)

        torch._dynamo.reset()
        clear_inductor_caches()
        cnts.clear()

        with torch.compiler.config.patch(job_id="foo"):
            f(torch.randn(2, 7))
            f(torch.randn(2, 8))
        self.assertEqual(cnts.frame_count, 1)

    # TODO: to test local need to ensure the local filesystem gets cleared out
    @torch._dynamo.config.patch(
        automatic_dynamic_remote_pgo=True, automatic_dynamic_local_pgo=False
    )
    def test_remote_basic(self):
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            return x * 2

        with mock_cache.PatchCaches():
            f(torch.randn(2, 3))
            f(torch.randn(2, 4))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(
                mock_cache.global_stats.dynamo_pgo, mock_cache.Stats(2, 0, 1)
            )

            self.reset()
            cnts.clear()

            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(
                mock_cache.global_stats.dynamo_pgo, mock_cache.Stats(2, 1, 1)
            )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
