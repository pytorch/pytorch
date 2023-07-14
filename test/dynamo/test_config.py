# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.utils import disable_cache_limit

# NB: do NOT include this test class in test_dynamic_shapes.py


class ConfigTests(torch._dynamo.test_case.TestCase):
    @disable_cache_limit()
    def test_no_automatic_dynamic(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_static = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        ):
            opt_fn = torch._dynamo.optimize(cnt_static)(fn)
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        self.assertEqual(cnt_static.frame_count, 10)

    @disable_cache_limit()
    def test_automatic_dynamic(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=True
        ):
            opt_fn = torch._dynamo.optimize(cnt_dynamic)(fn)
            # NB: must not do 0, 1 as they specialized
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        # two graphs now rather than 10
        self.assertEqual(cnt_dynamic.frame_count, 2)

    @disable_cache_limit()
    def test_no_assume_static_by_default(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=False
        ):
            opt_fn = torch._dynamo.optimize(cnt_dynamic)(fn)
            # NB: must not do 0, 1 as they specialized
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        # one graph now, as we didn't wait for recompile
        self.assertEqual(cnt_dynamic.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
