# Owner(s): ["module: dynamo"]

import sys
import unittest

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


class TestPEP479(torch._dynamo.test_case.CPythonTestCase):
    # Tests taken from CPython source code in cpython/Lib/test/test_generator_stop.py
    # https://github.com/python/cpython/blob/v3.13.1/Lib/test/test_generator_stop.py
    @unittest.skipIf(sys.version_info < (3, 12), "Test does not work in Python < 3.12")
    @make_dynamo_test
    def test_stopiteration_wrapping(self):
        def f():
            raise StopIteration

        def g():
            yield f()

        with self.assertRaises(RuntimeError) as cm:
            next(g())
        self.assertEqual("generator raised StopIteration", str(cm.exception))

    @unittest.skipIf(sys.version_info < (3, 12), "Test does not work in Python < 3.12")
    @make_dynamo_test
    def test_stopiteration_wrapping_context(self):
        def f():
            raise StopIteration

        def g():
            yield f()

        try:
            next(g())
        except RuntimeError as exc:
            self.assertIs(type(exc.__cause__), StopIteration)
            self.assertIs(type(exc.__context__), StopIteration)
            self.assertTrue(exc.__suppress_context__)
        else:
            self.fail(
                "__cause__, __context__, or __suppress_context__ "
                "were not properly set"
            )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
