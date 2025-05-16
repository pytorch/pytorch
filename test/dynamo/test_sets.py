# Owner(s): ["module: dynamo"]

import operator

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)


class SetSubclass(set):
    pass


class Int(int):
    def __init__(self, x):
        self.x = x


class TestGuards(torch._dynamo.test_case.TestCase):
    def test_set_recompile_on_key_pop(self):
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = torch._dynamo.testing.CompileCounter()

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

        cnts = torch._dynamo.testing.CompileCounter()

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

        cnts = torch._dynamo.testing.CompileCounter()

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

    def test_set_guard_on_keys_change_2(self):
        # This test guarantee that we're not triggering any of the dict guards
        # on sets
        s = {Int(1), Int(2), Int(3)}

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, s):
            for e in s:
                x *= e.x
            return x

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(torch.randn(4), s)
        opt_fn(torch.randn(4), s)
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # pop and add the same item
        e = s.pop()
        e.x = 100
        s.add(e)

        x = torch.randn(4)
        res = opt_fn(x, s)
        # Check Dynamo recompiles
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, s))


class TestSetMethods(torch._dynamo.test_case.TestCase):
    def setUp(self):
        self._prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self._prev
        super().tearDown()

    @parametrize("_type", [set, frozenset, SetSubclass], name_fn=lambda x: x.__name__)
    @parametrize("op", ["and_", "or_", "sub", "xor"])
    @make_dynamo_test
    def test_set_binary_ops(self, _type, op):
        s1 = _type({"a", "b", "c"})
        s2 = _type({"b", "c", "d"})
        set_op = {
            "and_": "intersection",
            "or_": "union",
            "sub": "difference",
            "xor": "symmetric_difference",
        }.get(op)
        r1 = getattr(s1, set_op)(s2)
        r2 = getattr(operator, op)(s1, s2)
        self.assertEqual(r1, r2)

    @parametrize("_type", [set, SetSubclass], name_fn=lambda x: x.__name__)
    @parametrize("op", ["iand", "ior", "isub", "ixor"])
    @make_dynamo_test
    def test_set_inplace_binary_ops(self, _type, op):
        s1 = _type({"a", "b", "c"})
        s2 = _type({"b", "c", "d"})
        set_op = {
            "iand": "intersection_update",
            "ior": "update",
            "isub": "difference_update",
            "ixor": "symmetric_difference_update",
        }.get(op)
        r1 = s1.copy()
        r2 = s1.copy()
        getattr(r1, set_op)(s2)
        getattr(operator, op)(r2, s2)
        self.assertTrue(r1 == r2)

    @parametrize("_type", [set, frozenset, SetSubclass], name_fn=lambda x: x.__name__)
    @make_dynamo_test
    def test_set___eq__(self, _type):
        s1 = _type({"a", "b", "c"})
        s2 = _type({"b", "c", "d"})
        self.assertTrue(s1 == s1)
        self.assertFalse(s1 == s2)


instantiate_parametrized_tests(TestSetMethods)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
