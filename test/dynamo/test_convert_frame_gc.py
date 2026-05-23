# Owner(s): ["module: dynamo"]

import unittest.mock

import torch
import torch._dynamo.config
import torch._dynamo.test_case


class ConvertFrameGCTests(torch._dynamo.test_case.TestCase):
    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    @torch._dynamo.config.patch(run_gc_after_compile=True)
    def test_run_gc_after_compile_uses_full_gc(self):
        def fn(x):
            return x + 1

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        with unittest.mock.patch("torch._dynamo.convert_frame.gc.collect") as collect:
            opt_fn(torch.ones(2))

        collect.assert_called()
        self.assertIn(unittest.mock.call(), collect.mock_calls)
        self.assertNotIn(unittest.mock.call(1), collect.mock_calls)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
