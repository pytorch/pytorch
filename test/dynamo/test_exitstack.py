# Owner(s): ["module: dynamo"]
import contextlib
import sys

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@contextlib.contextmanager
def set_default_dtype(dtype):
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(old_dtype)


class TestExitStack(torch._dynamo.test_case.TestCase):
    def setUp(self):
        self._prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self._prev

    def test_exitstack(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            with contextlib.ExitStack() as stack:
                stack.enter_context(set_default_dtype(torch.float64))
                return t.sin()

        t = torch.randn(2, dtype=torch.float64)
        y = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(y.dtype, torch.float64)

    @make_dynamo_test
    def test_ensure_exc_is_active_in_two_contexts(self):
        def raise_exc(exc):
            self.assertIsNotNone(sys.exc_info()[1])
            raise exc

        try:
            with contextlib.ExitStack() as stack:
                stack.callback(raise_exc, IndexError)
                stack.callback(raise_exc, KeyError)
                raise ZeroDivisionError
        except IndexError as exc:
            self.assertIsInstance(exc.__context__, KeyError)
        else:
            self.fail("Expected IndexError, but no exception was raised")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
