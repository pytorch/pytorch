# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


class TestUnittest(torch._dynamo.test_case.TestCase):
    def setUp(self):
        self._prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
