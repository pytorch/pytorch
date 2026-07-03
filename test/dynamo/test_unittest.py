# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


class TestUnittest(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._prev

    @make_dynamo_test
    def test_SkipTest(self):
        z = 0
        SkipTest = unittest.SkipTest
        try:
            raise SkipTest("abcd")
        except Exception:
            z = 1
        self.assertEqual(z, 1)

    def test_uncaught_SkipTest_propagates(self):
        # An unhandled unittest.SkipTest at the top of a compiled frame is
        # test-runner control flow: it must propagate as a real SkipTest (so
        # the test is reported skipped) instead of a hard graph break.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            raise unittest.SkipTest("skip me")

        with self.assertRaises(unittest.SkipTest):
            fn(torch.randn(3))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
