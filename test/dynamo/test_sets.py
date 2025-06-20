# Owner(s): ["module: dynamo"]

# TODO: move set tests from test_functions.py/test_misc.py to this file

import math
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import munge_exc
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


class TestSetGuards(LoggingTestCase):
    def test_set_with_function(self):
        s = {
            torch._C._set_grad_enabled,
            "hello",
            torch.amp._exit_autocast,
        }
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x, s):
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(2)
        y = fn(x, s)
        self.assertEqual(y, x.sin())
        self.assertEqual(cnts.frame_count, 1)

        s.remove(torch.amp._exit_autocast)
        s.add(torch._C._set_fwd_grad_enabled)
        y = fn(x, s)
        self.assertEqual(y, x.cos())
        self.assertEqual(cnts.frame_count, 2)

    @make_logging_test(recompiles=True)
    def test_in_guard(self, records):
        s = {
            "Dynamo",
            "Inductor",
            "PyTorch",
            torch.sin,
        }
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x, s):
            if "PyTorch" in s:
                return x.sin()
            return x.cos()

        x = torch.randn(2)
        y = fn(x, s)
        self.assertEqual(y, x.sin())
        self.assertEqual(cnts.frame_count, 1)

        s.remove("PyTorch")
        s.add("Cuda")
        y = fn(x, s)
        self.assertEqual(y, x.cos())
        self.assertEqual(cnts.frame_count, 2)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "set.__contains__")
        self.assertIn(
            """set.__contains__(s, 'PyTorch')""",
            munge_exc(record.getMessage()),
        )

    def test_set_with_tensors(self):
        s = {
            torch.ones(1),
            torch.tensor([1.0]),
            torch.zeros(1),
        }
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x, s):
            z = torch.zeros(1)
            for i in s:
                z += i
            return x + z

        x = torch.tensor([1.0])
        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: fn(x, s),
            """\
Attempted to wrap a set with tensors
  Explanation: Dynamo cannot trace sets of tensors. To get a stable ordering, Dynamo needs to convert the set into a list and the order might not be stable if the set contains tensors.
  Hint: Use a dictionary instead
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: Python set containing torch.Tensor elements


from user code:
   File "test_sets.py", line N, in fn
    for i in s:""",  # noqa: B950
        )

    def test_set_multiple_types(self):
        s = {
            "PyTorch",
            3.3,
            1j,
            math.nan,
        }
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x, s):
            if "PyTorch" in s:
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        y = fn(x, s)
        self.assertEqual(y, x.sin())
        self.assertEqual(cnts.frame_count, 1)

        s.remove("PyTorch")
        y = fn(x, s)
        self.assertEqual(y, x.cos())
        self.assertEqual(cnts.frame_count, 2)

    def test_set_recompile_on_key_pop(self):
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = CompileCounter()

        def fn(x, s):
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, s)
        opt_fn(x, s)
        self.assertEqual(res, fn(x, s))
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # Pop a value
        s.remove(torch.amp._exit_autocast)

        res = opt_fn(x, s)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, s))

    def test_set_recompile_on_key_change(self):
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = CompileCounter()

        def fn(x, s):
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, s)
        opt_fn(x, s)
        self.assertEqual(res, fn(x, s))
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # Pop a value
        s.remove(torch.amp._exit_autocast)
        # Add a different value
        s.add(torch._C._set_autograd_fallback_mode)

        res = opt_fn(x, s)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, s))

    @unittest.skip("random failures on Python 3.9")
    def test_set_guard_on_keys_change(self):
        # This test guarantee that we're not triggering any of the dict guards
        # on sets
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = CompileCounter()

        def fn(x, s):
            for e in s:
                x = x * len(str(e))
            return x

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(torch.randn(4), s)
        opt_fn(torch.randn(4), s)
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # pop and add the same item
        s.remove(torch.amp._exit_autocast)
        # It is not guaranteed that _exit_autocast will be in a specific order
        s.add(torch.amp._exit_autocast)

        x = torch.randn(4)
        res = opt_fn(x, s)
        # Check Dynamo don't recompile
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, fn(x, s))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
