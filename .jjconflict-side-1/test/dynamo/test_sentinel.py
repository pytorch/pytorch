# Owner(s): ["module: dynamo"]

import builtins
import functools
import operator
import sys
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounter


if sys.version_info >= (3, 15):
    MODULE_SENTINEL = builtins.sentinel("MODULE_SENTINEL")


@unittest.skipIf(sys.version_info < (3, 15), "Sentinel not supported in python < 3.15")
class SentinelTests(torch._dynamo.test_case.TestCase):
    def test_is(self):
        s1 = builtins.sentinel("s1")
        s2 = builtins.sentinel("s2")

        def f():
            return s1 is s2, s1 is s1

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(opt_f(), f())

    @unittest.expectedFailure
    def test_eq(self):
        s1 = builtins.sentinel("s1")
        s2 = builtins.sentinel("s2")

        def f():
            return s1 == s2, s1 == s1

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(opt_f(), f())

    def test_attrs(self):
        s1 = builtins.sentinel("s1")

        def f(x):
            return x.cos(), s1.__name__, s1.__module__

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x), f(x))

    def test_name_ro(self):
        s1 = builtins.sentinel("s1")

        def f():
            s1.__name__ = "fail"

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        # TODO should raise AttributeError
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            opt_f()

    def test_recompile(self):
        s1 = builtins.sentinel("s1")
        s2 = builtins.sentinel("s2")

        def f(x, s):
            if s is s1:
                return x.sin()
            return x.cos()

        cnts = CompileCounter()
        opt_f = torch.compile(f, backend=cnts, fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, s1), f(x, s1))
        self.assertEqual(opt_f(x, s2), f(x, s2))
        self.assertEqual(cnts.frame_count, 2)

    def test_is_not(self):
        s1 = builtins.sentinel("s1")
        s2 = builtins.sentinel("s2")

        def f():
            return s1 is not s2, s1 is not s1

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(opt_f(), f())

    def test_missing_default(self):
        # The canonical sentinel use case: marker for "argument not provided"
        MISSING = builtins.sentinel("MISSING")

        def g(x, y=MISSING):
            if y is MISSING:
                return x.sin()
            return x + y

        def f(x, y):
            return g(x), g(x, y)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        self.assertEqual(opt_f(x, y), f(x, y))

    def test_truthiness(self):
        s1 = builtins.sentinel("s1")

        def f(x):
            if s1:
                x = x.sin()
            if not s1:
                x = x + 100
            return x

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x), f(x))

    def test_dict_get_default(self):
        s1 = builtins.sentinel("s1")

        def f(x, d):
            v = d.get("k", s1)
            if v is s1:
                return x.sin()
            return x + v

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, {}), f(x, {}))
        self.assertEqual(opt_f(x, {"k": x}), f(x, {"k": x}))

    def test_graph_break(self):
        s1 = builtins.sentinel("s1")

        def f(x, s):
            y = x.sin()
            torch._dynamo.graph_break()
            return y, s is s1

        opt_f = torch.compile(f, backend="eager")
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, s1), f(x, s1))

    def test_return_sentinel(self):
        s1 = builtins.sentinel("s1")

        def f(x):
            return x.sin(), s1, [s1]

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        ref, res = f(x), opt_f(x)
        self.assertEqual(ref, res)
        self.assertIs(res[1], s1)
        self.assertIs(res[2][0], s1)

    def test_guard_stability(self):
        s1 = builtins.sentinel("s1")
        s1_dup = builtins.sentinel("s1")  # same name, distinct object

        def f(x, s):
            if s is s1:
                return x.sin()
            return x.cos()

        cnts = CompileCounter()
        opt_f = torch.compile(f, backend=cnts, fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, s1), f(x, s1))
        self.assertEqual(opt_f(x, s1), f(x, s1))
        self.assertEqual(cnts.frame_count, 1)
        # identity (not name) drives specialization
        self.assertEqual(opt_f(x, s1_dup), f(x, s1_dup))
        self.assertEqual(cnts.frame_count, 2)

    def test_global_sentinel(self):
        def f(x, s):
            if s is MODULE_SENTINEL:
                return x.sin()
            return x.cos()

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, MODULE_SENTINEL), f(x, MODULE_SENTINEL))

    def test_attr_sentinel(self):
        class Cfg:
            MISSING = builtins.sentinel("MISSING")

        cfg = Cfg()

        def f(x, s):
            if s is cfg.MISSING:
                return x.sin()
            return x.cos()

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, cfg.MISSING), f(x, cfg.MISSING))

    def test_container_input(self):
        s1 = builtins.sentinel("s1")
        s2 = builtins.sentinel("s2")

        def f(x, lst):
            if lst[0] is s1:
                return x.sin()
            return x.cos()

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, [s1]), f(x, [s1]))
        self.assertEqual(opt_f(x, [s2]), f(x, [s2]))

    def test_create_inside_graph_breaks(self):
        # Creating a sentinel inside the compiled region is out of scope;
        # it should graph break, not crash.
        def f(x):
            s = builtins.sentinel("inner")
            return x.sin(), s

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            opt_f(x)

    def test_reduce_sentinel_default(self):
        # functools.reduce's `initial` C-default is a sentinel in 3.15. Unlike
        # user-function defaults (which are source-backed), it reaches Dynamo
        # through SourcelessBuilder via the reduce polyfill.
        def f(x):
            return functools.reduce(operator.add, [x, x.sin(), x.cos()])

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x), f(x))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
