# Owner(s): ["module: dynamo"]

# TODO: move set tests from test_functions.py/test_misc.py to this file

import math

import torch
import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounter


class SetWithGeneratorTests(torch._dynamo.test_case.TestCase):
    def test_isdisjoint_with_generator(self):
        n = 0

        def gen():
            nonlocal n
            n += 1
            yield 1
            n += 2
            yield 2
            n += 3
            yield 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            nonlocal n
            s = {2, 4, 5}
            s.isdisjoint(gen())
            if n == 3:
                return x.sin()
            return x.cos()

        x = torch.randn(1)
        y = fn(x)
        self.assertEqual(y, x.sin())


class SetGuardsSet(torch._dynamo.test_case.TestCase):
    def test_set_with_tensor(self):
        s = {
            torch._C._set_grad_enabled,
            torch.randn(2),
            torch.amp._exit_autocast,
        }

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, s):
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(2)
        y = fn(x, s)
        self.assertEqual(y, x.sin())

        s.clear()
        y = fn(x, s)
        self.assertEqual(y, x.cos())

    def test_set_with_tensors_2(self):
        s = {
            torch.tensor(1.0),
            torch.randn(2),
            torch.zeros(4),
        }

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, s):
            if len(s) == 3:
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        y = fn(x, s)
        self.assertEqual(y, x.sin())

        s.clear()
        y = fn(x, s)
        self.assertEqual(y, x.cos())

    def test_set_with_str_and_tensor(self):
        s = {
            "PyTorch",
            torch.tensor(1.0),
        }

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, s):
            if "PyTorch" in s:
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        y = fn(x, s)
        self.assertEqual(y, x.sin())

        s.remove("PyTorch")
        y = fn(x, s)
        self.assertEqual(y, x.cos())

    def test_set_multiple_types(self):
        s = {
            "PyTorch",
            torch.tensor(1.0),
            3.3,
            1j,
            math.nan,
        }

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, s):
            if "PyTorch" in s:
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        y = fn(x, s)
        self.assertEqual(y, x.sin())

        s.remove("PyTorch")
        y = fn(x, s)
        self.assertEqual(y, x.cos())

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
        s.add(torch.amp._exit_autocast)

        x = torch.randn(4)
        res = opt_fn(x, s)
        # Check Dynamo don't recompile
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, fn(x, s))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
