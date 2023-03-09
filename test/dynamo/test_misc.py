# Owner(s): ["module: dynamo"]
import abc
import collections
import copy
import dataclasses
import dis
import enum
import logging
import math
import os
import sys
import typing
import unittest
import unittest.mock as mock
import weakref
from unittest.mock import patch

import numpy as np
import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch.onnx.operators
from torch._dynamo import bytecode_transformation, graph_break
from torch._dynamo.eval_frame import enable_cache_lookup_profiler
from torch._dynamo.output_graph import OutputGraph
from torch._dynamo.testing import (
    CompileCounter,
    requires_static_shapes,
    same,
    unsupported,
)

from torch._dynamo.utils import ifunspec
from torch.nn import functional as F
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FUSED_SDPA,
    SM80OrLater,
)
from torch.testing._internal.common_utils import freeze_rng_state
from torch.testing._internal.jit_utils import JitTestCase

mytuple = collections.namedtuple("mytuple", ["a", "b", "ab"])


def my_custom_function(x):
    return x + 1


class MyPickledModule(torch.nn.Module):
    def __init__(self, z):
        super().__init__()
        self.z = z

    def forward(self, x, y):
        return x * x * x + y + self.z


class MiscTests(torch._dynamo.test_case.TestCase):
    def test_boolarg(self):
        def boolarg(aa, bb, flag):
            if flag:
                return aa - bb
            else:
                return bb - aa

        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        correct1 = boolarg(a, b, True)
        correct2 = boolarg(a, b, False)
        correct3 = boolarg(a, b, None)
        counter = CompileCounter()
        opt_boolarg = torch._dynamo.optimize_assert(counter)(boolarg)
        val1 = opt_boolarg(a, b, True)
        val2 = opt_boolarg(a, b, False)
        val3 = opt_boolarg(a, b, None)
        val4 = opt_boolarg(a, b, True)
        self.assertTrue(same(val1, correct1))
        self.assertTrue(same(val2, correct2))
        self.assertTrue(same(val3, correct3))
        self.assertTrue(same(val4, correct1))
        self.assertEqual(counter.frame_count, 3)

    def test_callpacked(self):
        def call_packed(args):
            a, b, c = args
            return a - b * c

        counter = CompileCounter()
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        c = torch.randn(10, 10)
        correct = call_packed([a, b, c])
        opt_call_packed = torch._dynamo.optimize_assert(counter)(call_packed)
        val1 = opt_call_packed([a, b, c])
        val2 = opt_call_packed((a, b, c))
        val3 = opt_call_packed([a, b, c])
        val4 = opt_call_packed((a, b, c))
        self.assertTrue(same(val1, correct))
        self.assertTrue(same(val2, correct))
        self.assertTrue(same(val3, correct))
        self.assertTrue(same(val4, correct))
        self.assertEqual(counter.frame_count, 2)

    def test_raises(self):
        def fn(a, b, c, cls):
            x = a + b - c * 10
            raise cls(str(x))

        counter = CompileCounter()
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        c = torch.randn(10, 10)
        opt_fn = torch._dynamo.optimize(counter)(fn)
        self.assertRaises(AssertionError, lambda: opt_fn(a, b, c, AssertionError))
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 3)

    def test_inplace(self):
        def inplace1(a, b):
            o = torch.empty((10, 10))
            o.copy_(a)
            o -= b
            return o

        torch._dynamo.testing.standard_test(self, inplace1, 2, expected_ops=3)

    def test_unpack4(self):
        def unpack4(a, b):
            a = a[:5, :]
            b = b[:5, :]
            x, y = a.size()
            o = torch.empty((x, y))
            o.copy_(a / b)
            return o

        torch._dynamo.testing.standard_test(
            self, unpack4, 2, expected_ops=5, expected_ops_dynamic=8
        )

    def test_unpack5(self):
        def unpack5(a, b):
            a = a[:5, :]
            b = b[:5, :]
            x, y = a.shape
            o = torch.empty((x, y))
            o.copy_(a / b)
            return o

        torch._dynamo.testing.standard_test(
            self, unpack5, 2, expected_ops=5, expected_ops_dynamic=8
        )

    def test_matmul1(self):
        def matmul_op1(a, b):
            return a @ b

        # TODO(jansel): FX doesn't support this, should add upstream support
        torch._dynamo.testing.standard_test(self, matmul_op1, 2, expected_ops=1)

    def test_int_shape_binops(self):
        def fn(x):
            # Test reversal by putting int arg first.
            y = 15 - x.shape[0]
            y = 4 + y
            y = 5 * y
            y = 2 % y
            y = 3**y
            y = 10 // y
            y = pow(2, y)
            y = 10 / y
            return x + y

        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=11
        )

    def test_shape_int_inplace_binops(self):
        def fn(x):
            p = x.shape[0]
            p += 2
            p -= 2
            p **= 2
            p /= 2
            p *= 2
            p //= 2
            p %= 2
            return x + p

        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=10
        )

    def test_int_shape_inplace_binops(self):
        def fn(x):
            p = x.shape[0]
            # Test reversal by putting constant first
            y = 2
            y += p
            y = 2
            y -= p
            y = 2
            y **= p
            y = 2
            y /= p
            y = 2
            y *= p
            y = 2
            y //= p
            y = 2
            y %= p
            return x + y

        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=10
        )

    def test_int_int_comparisons(self):
        def fn(x):
            if 2 != 2:
                out = 1
            elif 2 < 1:
                out = 1
            elif 1 > 2:
                out = 1
            elif 1 >= 2:
                out = 1
            elif 2 <= 1:
                out = 1
            elif 2 == 2:
                out = 2
            else:
                out = 1
            return x + out

        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_shape_int_comparisons(self):
        def fn(x):
            a = x.shape[0]
            # Ensure support for constant on right side
            if a != 10:
                out = 1
            elif a < 2:
                out = 1
            elif a > 12:
                out = 1
            elif a >= 12:
                out = 1
            elif a <= 2:
                out = 1
            elif a == 10:
                out = 2
            else:
                out = 1
            return x + out

        # expect for dynamic: size, index, 6 comparison ops, add
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=9
        )

    def test_int_shape_comparisons(self):
        def fn(x):
            a = x.shape[0]
            # Ensure support for constant on left side
            if 10 != a:
                out = 1
            elif 12 < a:
                out = 1
            elif 2 > a:
                out = 1
            elif 2 >= a:
                out = 1
            elif 12 <= a:
                out = 1
            elif 10 == a:
                out = 2
            else:
                out = 1
            return x + out

        # expect for dynamic: size, index, 6 comparison ops, add
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=9
        )

    def test_param_shape_binops(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(15))

            def forward(self, x):
                # Test reversal by putting param shape arg first.
                p = self.param.shape[0]
                y = p - x.shape[0]
                y = p + y
                y = p * y
                y = p % y
                y = p**y
                y = p // y
                y = pow(p, y)
                y = p / y
                return x + y

        counts = torch._dynamo.testing.CompileCounter()
        mod = MyModule()
        optimized_mod = torch._dynamo.optimize(counts, nopython=True)(mod)

        x = torch.randn(3)
        ref = mod(x)
        res = optimized_mod(x)

        self.assertTrue(same(ref, res))
        self.assertEqual(counts.frame_count, 1)
        expected_op_count = 13 if torch._dynamo.testing.config.dynamic_shapes else 1
        self.assertEqual(counts.op_count, expected_op_count)

    def test_user_defined_binop(self):
        class MyClass:
            def __init__(self, value):
                self.value = value

            def __radd__(self, other):
                return self.value + other

        def fn(x, c):
            y = x.shape[0] + c
            return x + y

        counts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(counts)(fn)

        x = torch.randn(3)
        c = MyClass(4)
        ref = fn(x, c)
        res = opt_fn(x, c)

        self.assertTrue(same(ref, res))
        self.assertEqual(counts.frame_count, 1)
        expected_op_count = 4 if torch._dynamo.testing.config.dynamic_shapes else 1
        self.assertEqual(counts.op_count, expected_op_count)

    def test_compare_shapes_eq(self):
        def compare_shapes(a, b, to_list):
            x = list(a.unsqueeze(-1).shape) if to_list else a.shape
            y = list(b.unsqueeze(-1).shape) if to_list else b.shape
            if x == y:
                return a + 1
            else:
                return a + 2

        # Test both ListVariable and ShapeVariable
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=True), 2
        )
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=False), 2
        )

    def test_compare_shapes_tuple_eq(self):
        def compare_shapes(a, b):
            x = tuple(a.unsqueeze(-1).shape)
            y = tuple(b.unsqueeze(-1).shape)
            if x == y:
                return a + 1
            else:
                return a + 2

        torch._dynamo.testing.standard_test(self, lambda a, b: compare_shapes(a, b), 2)

    def test_compare_shapes_tuple_neq(self):
        def compare_shapes(a, b):
            x = tuple(a.unsqueeze(-1).shape)
            y = tuple(b.unsqueeze(-1).shape)
            if x != y:
                return a + 1
            else:
                return a + 2

        torch._dynamo.testing.standard_test(self, lambda a, b: compare_shapes(a, b), 2)

    def test_compare_shapes_neq(self):
        def compare_shapes(a, b, to_list):
            x = list(a.unsqueeze(-1).shape) if to_list else a.shape
            y = list(b.unsqueeze(-1).shape) if to_list else b.shape
            if x != y:
                return a + 1
            else:
                return a + 2

        # Test both ListVariable and ShapeVariable
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=True), 2
        )
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=False), 2
        )

    @patch.object(torch._dynamo.config, "dynamic_shapes", True)
    def test_compare_shapes_with_constant(self):
        def compare_shapes(a):
            x = a.shape
            if x[0] != 3:
                return a * 4
            return a * 3

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(compare_shapes)
        opt_fn(torch.randn([3, 4]))
        opt_fn(torch.randn([4, 3]))
        self.assertEqual(guard_failure.reason, "a.size()[0] == 3")

    def test_builtin_isinstance(self):
        def fn(x):
            t = torch.arange(1, 3)
            a = isinstance(x, torch.Tensor)
            b = isinstance(t, torch.Tensor)
            c = isinstance(x, int)
            d = isinstance(3, int)
            e = isinstance([1, 2, 3], list)
            f = isinstance({"foo": 1, "bar": 2}, dict)
            res = [a, b, c, d, e, f]
            # Can't run yet due to other unimplemented instructions
            # res += [isinstance(torch.nn.LazyLinear(2, 3), torch.nn.Linear)]
            return res

        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_fold(self):
        def fn(a):
            return a + math.sqrt(63)

        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_shape_unpack(self):
        def fn(x):
            a, b = x.size()
            return x * b

        i = torch.randn(5, 10)
        r1 = fn(i)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        r2 = opt_fn(i)
        self.assertTrue(same(r1, r2))

    def test_tensor_iter(self):
        def fn(x):
            for y in x:
                y.add_(1.0)
            return y

        # expect extra size node for dynamic
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=20, expected_ops_dynamic=21
        )

    def test_empty_list(self):
        def fn(x, ll):
            if len(ll) == 0 and not ll and ll is not None:
                return x + 1

        i = torch.randn(5, 10)
        r1 = fn(i, [])
        opt_fn = torch._dynamo.optimize("eager")(fn)
        r2 = opt_fn(i, [])
        r3 = opt_fn(i, tuple())
        self.assertTrue(same(r1, r2))
        self.assertTrue(same(r1, r3))

    def test_min_max_over_iterable(self):
        def get_test_fn(func):
            def _fn(a, b, func=func):
                # try all of list, iterator, tuple, vararg.
                lst = [a.shape[0] + 1, 8, a.shape[0]]
                x = func(lst)
                y = func(iter(lst))
                z = func(tuple(lst))
                w = func(*lst)
                return a + (x + y + z + w)

            return _fn

        # expect for dynamic:
        # 2 * (size, getitem) ops +
        # 1 add op +
        # 4 * 2 min / max ops +
        # 4 final add ops = 17
        torch._dynamo.testing.standard_test(
            self, get_test_fn(func=min), 2, expected_ops=1, expected_ops_dynamic=17
        )
        torch._dynamo.testing.standard_test(
            self, get_test_fn(func=max), 2, expected_ops=1, expected_ops_dynamic=17
        )

    def test_config_obj(self):
        class Cfg:
            def __init__(self):
                self.val = 0.5
                self.count = 3

        def fn(x, cfg):
            for i in range(cfg.count):
                x = x + cfg.val
            return x

        cfg1 = Cfg()
        cfg1.val = 1.0
        cfg2 = Cfg()
        v = torch.zeros(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        v = opt_fn(v, cfg1)  # 3
        v = opt_fn(v, cfg2)  # 4.5
        cfg2.count = 1
        v = opt_fn(v, cfg2)  # 5
        cfg2.val = 2.0
        v = opt_fn(v, cfg2)  # 7
        self.assertEqual(v[0], 7)
        self.assertEqual(cnts.op_count, 8)

    def test_config_getattr_default(self):
        class Cfg:
            def __init__(self):
                self.val = 0.5
                self.count = 10

        def fn(x, cfg):
            if getattr(cfg, "just_add_7", False):
                return x + 7
            for i in range(cfg.count):
                x = x + cfg.val
            return x

        cfg1 = Cfg()
        v = torch.zeros(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        cfg1.just_add_7 = True
        self.assertEqual(opt_fn(v, cfg1)[0], 7)
        self.assertEqual(opt_fn(v, cfg1)[0], 7)
        cfg1.just_add_7 = False
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        self.assertEqual(cnts.frame_count, 3)

    def test_size_input(self):
        def fn(x, s):
            a, b = s
            return x + (a - b)

        v = torch.zeros(10, 20)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertEqual(opt_fn(v, v.size())[0, 0], -10)
        self.assertEqual(opt_fn(v, (10, 20))[0, 0], -10)
        self.assertEqual(opt_fn(v, [10, 20])[0, 0], -10)
        self.assertEqual(cnts.op_count, 2)

    def test_cell_output1(self):
        out = None

        def fn(a, b):
            nonlocal out
            out = a + b * 10

        v = torch.Tensor([100])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertIsNone(opt_fn(v, v))
        self.assertEqual(out[0], 1100)
        self.assertEqual(cnts.op_count, 2)

    def test_cell_output2(self):
        out = None

        def fn(a, b):
            nonlocal out
            c = unsupported(a, b)
            out = a + b * 10 + c

        v = torch.Tensor([100])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertIsNone(opt_fn(v, v))
        self.assertEqual(out[0], 1200)
        self.assertEqual(cnts.op_count, 3)

    def test_return_nested_function(self):
        out = None

        def fn(a, b):
            nonlocal out
            c = a + b
            d = a + 1.0

            def fn2(f: int = 7, g: float = 9.0):
                nonlocal out
                out = a + b * 10
                return c * f - d * g

            return fn2

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        opt_fn_ret = torch._dynamo.optimize(cnts)(opt_fn(v1, v2))
        self.assertEqual(opt_fn_ret(1.5)[0], -459)
        self.assertEqual(out[0], 2100)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 7)

    def test_tensor_dict1(self):
        def fn(inputs):
            return inputs["a"] - inputs["b"] * 1.5

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertEqual(opt_fn({"a": v1, "b": v2})[0], -200)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_tensor_dict2(self):
        def fn1(inputs):
            total = torch.zeros(1)
            for k, v in inputs.items():
                total += v
            return total

        def fn2(inputs):
            total = torch.zeros(1)
            for v in inputs.values():
                total += v
            return total

        def fn3(inputs):
            total = torch.zeros(1)
            for k in inputs.keys():
                total += inputs[k]
            return total

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch._dynamo.optimize(cnts)(fn1)
        opt_fn2 = torch._dynamo.optimize(cnts)(fn2)
        opt_fn3 = torch._dynamo.optimize(cnts)(fn3)
        self.assertEqual(opt_fn1({"a": v1, "b": v2})[0], 300)
        self.assertEqual(opt_fn2({"a": v1, "b": v2})[0], 300)
        self.assertEqual(opt_fn3({"a": v1, "b": v2})[0], 300)
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 9)

    def test_dictcomp(self):
        def fn1(inputs):
            return {k: v + 1 for k, v in inputs.items()}

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch._dynamo.optimize(cnts)(fn1)
        self.assertEqual(opt_fn1({"a": v1, "b": v2})["a"], 101)
        self.assertEqual(opt_fn1({"a": v1, "b": v2})["b"], 201)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_listcomp(self):
        def fn2(inputs):
            return torch.sum(torch.cat([v + 1 for k, v in inputs.items()], 0))

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn2 = torch._dynamo.optimize(cnts)(fn2)
        self.assertEqual(opt_fn2({"a": v1, "b": v2}), 302)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_is_floating_point(self):
        def fn(a, b):
            x = a + 1.0
            if torch.is_floating_point(b):
                x = x + b
            return x + 2.0

        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_floating_point2(self):
        def fn(a, b):
            x = a + 1.0
            if b.is_floating_point():
                x = x + b
            return x + 2.0

        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_tensor(self):
        def fn(a, b):
            x = a + 1.0
            if torch.is_tensor(b):
                x = x + b
            return x + 2.0

        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_tensor2(self):
        def fn(x):
            if torch.is_tensor(x):
                return x + 1
            else:
                return torch.ones([2, 3])

        x1 = {"input": torch.rand(2, 3)}
        x2 = torch.rand(2, 3)
        ref1 = fn(x1)
        ref2 = fn(x2)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res1 = opt_fn(x1)
        res2 = opt_fn(x2)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_numel(self):
        def fn(a):
            return (a + a.numel() + torch.numel(a), a + a.nelement())

        return torch._dynamo.testing.standard_test(
            self, fn=fn, nargs=1, expected_ops=3, expected_ops_dynamic=6
        )

    def test_pair(self):
        def fn(a):
            return (
                torch.zeros(torch.nn.modules.utils._pair(a.size()))
                + a
                + torch.ones(torch.nn.modules.utils._ntuple(3)(3)).sum()
            )

        return torch._dynamo.testing.standard_test(
            self, fn=fn, nargs=1, expected_ops=5, expected_ops_dynamic=8
        )

    @patch.object(torch._dynamo.config, "dynamic_shapes", True)
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_tensor_item_capture(self):
        def fn(a, b):
            return (a + b).sum().item()

        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        correct = fn(v1, v2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize((cnts))(fn)
        self.assertEqual(opt_fn(v1, v2), correct)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    @patch.object(torch._dynamo.config, "dynamic_shapes", True)
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", False)
    def test_tensor_item_no_capture(self):
        def fn(a, b):
            return (a + b).sum().item()

        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        correct = fn(v1, v2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize((cnts))(fn)
        self.assertEqual(opt_fn(v1, v2), correct)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_namedtuple1(self):
        def fn(a, b):
            tmp = mytuple(a, b, a + b)
            return mytuple(tmp.a, tmp[1], tmp.ab + b)

        v1 = torch.Tensor([10])
        v2 = torch.Tensor([20])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertEqual(opt_fn(v1, v2).ab, 50)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_namedtuple2(self):
        def fn(packed):
            a, b, c = packed
            if hasattr(packed, "b"):
                b = packed.b + 1
            c = packed[2]
            return a + b + c

        v1 = torch.Tensor([1])
        v2 = torch.Tensor([2])
        v3 = torch.Tensor([3])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertEqual(opt_fn(mytuple(v1, v2, v3))[0], 7)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_namedtuple3(self):
        def fn(x, packed):
            if isinstance(packed, mytuple):
                return x + 1
            else:
                return x - 1

        x = torch.rand([2, 3])
        packed = mytuple(1, 2, 3)
        ref = fn(x, packed)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res = opt_fn(x, packed)
        self.assertTrue(same(ref, res))

    def test_range_input(self):
        def fn(a, rng):
            x = a
            for i in rng:
                x = x + i
            return x

        def fn1(a):
            return fn(a, rng=range(3))

        return torch._dynamo.testing.standard_test(
            self, fn=fn1, nargs=1, expected_ops=3
        )

    def test_range_with_shape(self):
        def fn(a):
            for i in range(1, a.shape[0]):
                a += 1
            return a

        # expect 1 more op (size call) for dynamic
        return torch._dynamo.testing.standard_test(
            self, fn=fn, nargs=1, expected_ops=9, expected_ops_dynamic=10
        )

    def test_no_grad(self):
        def fn1(a, b):
            x = a + 1
            # redundant no_grad should get ignored
            with torch.no_grad():
                x = x + b
            x = x + 2
            return x

        def fn2(a, b):
            x = a + 1
            with torch.set_grad_enabled(False):
                x = x + b
            x = x + 2
            return x

        def fn3(a, b):
            x = a + 1
            with torch.enable_grad():
                x = x + b
            x = x + 2
            return x

        def fn4(a, b):
            x = a + 1
            with torch.set_grad_enabled(True):
                if torch.is_grad_enabled():
                    x = x + b
            x = x + 2
            return x

        with torch.no_grad():
            torch._dynamo.testing.standard_test(self, fn=fn1, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn2, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn3, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn4, nargs=2, expected_ops=5)
        with torch.enable_grad():
            torch._dynamo.testing.standard_test(self, fn=fn1, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn2, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn3, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn4, nargs=2, expected_ops=5)

    def test_grad_mode_guard(self):
        def fn(a, b):
            prev_grad = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
            a = a + 1
            a.tolist()  # graph break
            ret = a + b
            torch.set_grad_enabled(prev_grad)
            return ret

        a = torch.randn([3, 4])
        b = torch.randn([3, 4])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        for _ in range(10):
            opt_fn(a, b)
        self.assertEqual(cnts.frame_count, 2)

    def test_nested_grad_mode_graph_break(self):
        def fn(x):
            before = torch.is_grad_enabled()
            with torch.set_grad_enabled(False):
                torch._dynamo.graph_break()
                with torch.set_grad_enabled(True):
                    x = torch.mul(x, 5)
                    torch._dynamo.graph_break()
                    x = torch.sqrt(x)
                    assert torch.is_grad_enabled()
                assert not torch.is_grad_enabled()
            assert torch.is_grad_enabled() == before
            return x

        a = torch.randn([3, 4])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        for _ in range(10):
            opt_fn(a)
        self.assertEqual(cnts.frame_count, 3)

    def test_build_tuple_unpack(self):
        def fn1(a, b, c):
            return a - b / c

        def fn2(a, b, c):
            tmp1 = (a,)
            tmp2 = (b, c)
            args = (*tmp1, *tmp2)
            return fn1(*args)

        def fn3(a, *args):
            return fn1(a, *args)

        torch._dynamo.testing.standard_test(self, fn=fn2, nargs=3, expected_ops=2)
        torch._dynamo.testing.standard_test(self, fn=fn3, nargs=3, expected_ops=2)

    def test_list_mul(self):
        def fn(count):
            head_mask = count * [None] * count
            return head_mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertEqual(opt_fn(2), [None] * 4)
        # TODO: the captured frame here is a bit goofy, because we don't
        # output anything and none of the traced operations have side
        # effects.  Probably need better heuristic for bailing on
        # dynamo if there are no outputs
        self.assertEqual(cnts.frame_count, ifunspec(1, 0))
        self.assertEqual(cnts.op_count, ifunspec(2, 0))

    def test_list_slice_mul(self):
        def fn(count):
            a = [1, 2, 3]
            head_mask = count * a[1:] * count
            return head_mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertEqual(opt_fn(2), [2, 3] * 4)
        self.assertEqual(cnts.frame_count, ifunspec(1, 0))
        self.assertEqual(cnts.op_count, ifunspec(14, 0))

    def test_tuple_mul(self):
        def fn(count):
            head_mask = count * (2, 3) * count
            return head_mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertEqual(opt_fn(2), (2, 3) * 4)
        self.assertEqual(cnts.frame_count, ifunspec(1, 0))
        self.assertEqual(cnts.op_count, ifunspec(14, 0))

    def test_tuple_mul_with_shape(self):
        def fn(a):
            x = a.shape[0]
            y = 2 * (x, 3) * 2
            return a + y[4]

        # expect 3 ops post folding for dynamic case: size, index, add
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=3
        )

    def test_tuple_iadd_with_shape(self):
        def fn(a):
            output = (a + a.shape[0], a - a.shape[0])
            # tuple += tuple
            output += (a - a.shape[0], a + a.shape[0])
            # tuple += constant tuple
            output += (2, 3)
            return output

        # expect 4 add / subs for static, 4 * 3 (size, index, math op) for dynamic
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=4, expected_ops_dynamic=12
        )

    def test_list_iadd_with_shape(self):
        def fn(a):
            output = [a + a.shape[0], a - a.shape[0]]
            # list += list
            output += [a - a.shape[0], a + a.shape[0]]
            # list += tuple
            output += (a + a.shape[0], a - a.shape[0])
            return output

        # expect 6 add / subs for static, 6 * 3 (size, index, math op) for dynamic
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=6, expected_ops_dynamic=18
        )

    def test_user_getattr1(self):
        class MyConfig(dict):
            def __getattr__(self, name):
                return self[name]

        def fn(cfg, x, y):
            return x + y + cfg.offset

        x = torch.randn(10)
        cfg = MyConfig(offset=5)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertTrue(same(opt_fn(cfg, x, x), 2 * x + 5))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_user_getattr2(self):
        class MyConfig:
            defined_on_class = 1

            def __init__(self):
                self.defined_on_object = 2

            def __getattr__(self, name):
                return 3

        def fn(cfg, x):
            return x + cfg.defined_on_class - cfg.defined_on_object + cfg.not_defined

        x = torch.randn(10)
        cfg = MyConfig()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertTrue(same(opt_fn(cfg, x), x + 1 - 2 + 3))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_user_getattribute(self):
        class MyObject:
            def __init__(self):
                self.custom_dict = {"a": torch.rand((2, 2))}
                self.my_number = 42

            def __getattribute__(self, name):
                custom_dict = super().__getattribute__("custom_dict")
                if name in custom_dict:
                    return custom_dict[name]
                return super().__getattribute__(name)

            def run(self, x):
                return self.my_number * x + self.a * x

        def fn(obj, x):
            return obj.run(x)

        obj = MyObject()
        x = torch.rand((2, 2))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertTrue(same(opt_fn(obj, x), fn(obj, x)))

    def test_nn_module_getattr(self):
        class MyMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.custom_dict = {"queue": [torch.rand((2, 2)) for _ in range(3)]}
                self.other_attr = torch.rand((2, 2))

            def __getattr__(self, name):
                custom_dict = self.custom_dict
                if name in custom_dict:
                    return custom_dict[name]
                return super().__getattr__(name)

            def forward(self, x):
                return x @ self.other_attr + self.queue[-1]

        x = torch.rand((2, 2))
        mod = MyMod()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_mod = torch._dynamo.optimize(cnts)(mod)
        self.assertTrue(same(opt_mod(x), mod(x)))
        self.assertTrue(cnts.frame_count, 1)
        self.assertTrue(cnts.op_count, 2)

    def test_nn_module_getattribute(self):
        class MyMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.my_number = 42

            def __getattribute__(self, name):
                if name == "special_attr":
                    return torch.tensor([[1, 2], [3, 4]])
                return super().__getattribute__(name)

            def forward(self, x):
                return self.my_number * x + self.special_attr * x

        def fn(mod, x):
            return mod(x)

        mod = MyMod()
        x = torch.rand((2, 2))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertTrue(same(opt_fn(mod, x), fn(mod, x)))

    def test_user_property(self):
        class MyConfig:
            @property
            def prop5(self):
                return 5

        def fn(cfg, x, y):
            return x + y + cfg.prop5

        x = torch.randn(10)
        cfg = MyConfig()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertTrue(same(opt_fn(cfg, x, x), 2 * x + 5))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_dataclass_fields(self):
        @dataclasses.dataclass
        class MyDataClass:
            a: torch.Tensor
            b: torch.Tensor = None
            c: torch.Tensor = None
            d: torch.Tensor = None
            e: torch.Tensor = None

        def fn(obj):
            class_fields = dataclasses.fields(obj)
            assert len(class_fields)
            assert all(field.default is None for field in class_fields[1:])
            other_fields_are_none = all(
                getattr(obj, field.name) is None for field in class_fields[1:]
            )
            assert not other_fields_are_none

            total = getattr(obj, class_fields[0].name)
            for field in class_fields[1:]:
                v = getattr(obj, field.name)
                if v is not None:
                    total += v

            return total

        obj1 = MyDataClass(torch.randn(10), torch.randn(10), torch.randn(10))
        obj2 = MyDataClass(torch.randn(10), e=torch.randn(10))
        correct1 = fn(obj1)
        correct2 = fn(obj2)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertTrue(same(opt_fn(obj1), correct1))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertTrue(same(opt_fn(obj2), correct2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    @requires_static_shapes
    def test_tensor_build_list_unpack(self):
        def fn(x):
            # seen in fastNLP_Bert
            return torch.cat([*x], dim=-1)

        val = torch.randn([1, 1, 473, 768])
        correct = fn(val)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertTrue(same(opt_fn(val), correct))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_numpy_int_constant(self):
        def fn(x, a, b):
            return x + (a % b)

        args = [torch.randn(10), 4096, np.int64(8)]
        correct = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertTrue(same(opt_fn(*args), correct))
        self.assertTrue(same(opt_fn(*args), correct))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_dict_mutation_side_effect(self):
        def fn(d):
            d["c"] = d["a"] + d.pop("b")
            return d

        args1 = {"a": torch.randn(10), "b": torch.randn(10)}
        args2 = dict(args1)
        assert fn(args1) is args1
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertIs(opt_fn(args2), args2)
        self.assertTrue(same(args1, args2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    def test_module_deepcopy(self):
        m1 = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )
        m2 = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

        def fn(m, x):
            m_copy = copy.deepcopy(m)
            return m_copy(x)

        v = torch.randn(10)
        correct1 = fn(m1, v)
        correct2 = fn(m2, v)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        for _ in range(10):
            self.assertTrue(same(opt_fn(m1, v), correct1))
        for _ in range(10):
            self.assertTrue(same(opt_fn(m2, v), correct2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_type_copy(self):
        def fn(seq):
            a, b = seq
            return type(seq)([a + 1, b + 2, a + b])

        args1 = [torch.randn(10), torch.randn(10)]
        args2 = (torch.randn(10), torch.randn(10))
        correct1 = fn(args1)
        correct2 = fn(args2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertTrue(same(opt_fn(args1), correct1))
        self.assertTrue(same(opt_fn(args2), correct2))
        self.assertIsInstance(opt_fn(args1), list)
        self.assertIsInstance(opt_fn(args2), tuple)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 6)

    def test_setattr_mutation1(self):
        class MyObj:  # noqa: B903
            def __init__(self, a, b):
                self.a = a
                self.b = b

        def fn(obj):
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            obj.c = obj.a * obj.b + 4
            obj.b = obj.a * obj.c + 5
            obj.a = obj.b * obj.c + 6
            return obj

        x1 = torch.randn(10)
        x2 = torch.randn(10)
        obj1 = MyObj(x1, x2)
        obj2 = MyObj(x1, x2)
        fn(obj2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertIs(opt_fn(obj1), obj1)
        self.assertTrue(same(obj1.a, obj2.a))
        self.assertTrue(same(obj1.b, obj2.b))
        self.assertTrue(same(obj1.c, obj2.c))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 12)

    def test_setattr_mutation2(self):
        class MyObj:
            def __init__(self, x):
                self.a = x + 1
                self.b = x + 2

        def fn(x):
            x = x / 3.0
            obj = MyObj(x)
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            return obj

        x1 = torch.randn(10)
        obj2 = fn(x1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        obj1 = opt_fn(x1)
        self.assertTrue(same(obj1.a, obj2.a))
        self.assertTrue(same(obj1.b, obj2.b))
        self.assertTrue(same(obj1.c, obj2.c))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 9)

    def test_setattr_mutation3(self):
        # TODO(jansel): dead code eliminate the object creation
        class MyObj:
            def __init__(self, x):
                super().__init__()
                self.a = x + 1
                self.b = x + 2

        def fn(x):
            x = x / 3.0
            obj = MyObj(x)
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            return obj.a, obj.b, obj.c

        x1 = torch.randn(10)
        obj2 = fn(x1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        obj1 = opt_fn(x1)
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 9)

    def test_user_defined_class_name(self):
        class MyClassFoo:
            pass

        def fn1(a, b, c):
            tmp = MyClassFoo()
            if tmp.__class__.__name__ == "MyClassFoo":
                return a - b / c

        torch._dynamo.testing.standard_test(self, fn=fn1, nargs=3)

    def test_user_defined_class_python_type(self):
        class MyClass1:
            pass

        class ExampleMeta(type):
            pass

        class MyClass2(metaclass=ExampleMeta):
            pass

        def fn(x, c):
            if isinstance(c, MyClass1):
                return x + 1
            elif isinstance(c, MyClass2):
                return x + 2
            else:
                return x + 3

        x = torch.rand(3)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        for c in [MyClass1, MyClass2]:
            ref = fn(x, c)
            res = opt_fn(x, c)
            self.assertTrue(same(ref, res))

    def test_super_calling_with_metaclass(self):
        class ExampleMeta(type):
            pass

        class MyClass1(metaclass=ExampleMeta):
            @classmethod
            def add(cls, x):
                return x + 1

        class MyClass2(MyClass1):
            @classmethod
            def add(cls, x):
                torch._dynamo.graph_break()
                return x + super().add(x)

        def fn(x, obj):
            return x + obj.add(x)

        x = torch.rand(3)
        obj = MyClass2()
        opt_fn = torch._dynamo.optimize("eager")(fn)
        ref = fn(x, obj)
        res = opt_fn(x, obj)
        self.assertTrue(same(ref, res))

    def test_manual_seed(self):
        def fn(a, b):
            x = a + b
            torch.manual_seed(9000)
            return x + 1

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_usr_cls_staticmethod(self):
        class Foo:
            @staticmethod
            def bar(a, b):
                return a + b

        def fn(a, b):
            return Foo.bar(a, b) - 1

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=2)

    def test_usr_cls_classmethod(self):
        class Foo:
            @classmethod
            def bar(cls, a, b):
                return a + b

        def fn(a, b):
            return Foo.bar(a, b) - 1

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=2)

    def test_dunder_methods(self):
        class Foo:
            def __init__(self, val):
                super().__init__()
                self.val = val

            def __add__(self, other):
                return Foo(self.val + other.val)

            def __mul__(self, other):
                return Foo(self.val * other.val)

            def __truediv__(self, other):
                return Foo(self.val / other.val)

            def __sub__(self, other):
                return Foo(self.val - other.val)

        def fn(a, b, c):
            return Foo(a) + Foo(b) * Foo(c) / Foo(a) - Foo(b)

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=3, expected_ops=4)

    def test_function_annotation(self):
        class Variable:
            pass

        def fn(x):
            x = x / 3.0

            def inner(y: typing.List[Variable]):
                return x + 1

            return inner

        x1 = torch.randn(10)
        obj2 = fn(x1)([])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        opt_fn_inner = torch._dynamo.optimize_assert(cnts)(opt_fn(x1))
        obj1 = opt_fn_inner([])
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 2)

    def test_nested_closure(self):
        v0 = torch.randn(10)

        def fn1():
            v1 = torch.randn(10)

            def fn2(*args, **kwargs):
                assert len(args) == 1
                assert len(kwargs) == 1
                v2 = torch.randn(10) + args[0] + kwargs["b"]

                def fn3(v3=torch.randn(10)):
                    def fn4():
                        return v0 + v1 + v2 + v3 + 1

                    return fn4

                return fn3

            return fn2(1, b=2)()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch._dynamo.optimize_assert(cnts)(fn1)
        tmp1 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        tmp2 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        self.assertTrue(tmp1().shape, (10,))
        self.assertTrue(same(tmp1(), tmp1()))
        self.assertFalse(same(tmp1(), tmp2()))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 9)

    def test_nested_closure_mutation(self):
        def fn1():
            v1 = torch.randn(10)

            def fn2():
                v2 = torch.randn(10)

                def fn3():
                    nonlocal v1, v2
                    v1 += 1
                    v2 += 2
                    return v1 + v2

                return fn3

            rv = fn2()
            rv()
            rv()
            return rv

        torch.manual_seed(9000)
        counter1 = fn1()
        result1 = [counter1(), counter1(), counter1()]

        torch.manual_seed(9000)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch._dynamo.optimize_assert(cnts)(fn1)
        counter2 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        result2 = [counter2(), counter2(), counter2()]
        result1.append(counter1())
        result2.append(counter2())

        self.assertTrue(same(result1, result2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 11)

    def test_write_to_closures_in_inlining(self):
        out = []
        for use_dynamo in [False, True]:

            def make_counter():
                x = torch.randn(10)

                def counter():
                    nonlocal x
                    x = x + 1
                    return x

                return counter

            torch.manual_seed(0)
            counter = make_counter()
            if not use_dynamo:
                out.append(counter() + counter())
            else:
                cnts = torch._dynamo.testing.CompileCounter()

                @torch._dynamo.optimize(cnts, nopython=True)
                def fn(counter):
                    return counter() + counter()

                out.append(fn(counter))
                self.assertEqual(cnts.frame_count, 1)
                self.assertEqual(cnts.op_count, 3)
                self.assertFalse(same(counter() + counter(), out[-1]))

        self.assertTrue(same(out[0], out[1]))

    def test_top_package_import(self):
        def fn(x):
            import torch.fx

            assert not isinstance(x, torch.fx.Proxy)
            return torch.sin(x)

        x = torch.randn(4, 5)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_optimize_on_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def custom_member(self):
                # Just for checking that Dynamo returned mod object can redirect
                # to this method
                pass

            def forward(self, x):
                return self.relu(x)

        cnts1 = torch._dynamo.testing.CompileCounter()
        mod = MockModule()
        optimized_mod = torch._dynamo.optimize(cnts1, nopython=True)(mod)

        a = torch.randn(10)
        ref = mod(a)
        res = optimized_mod(a)

        optimized_mod.custom_member()

        self.assertTrue(same(ref, res))

    def test_nested_optimize_decorator(self):
        cnts2 = torch._dynamo.testing.CompileCounter()
        cnts3 = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.run()
        def fn1(x):
            return torch.sin(x) * 10

        @torch._dynamo.optimize(cnts2, nopython=True)
        def fn2(x):
            return fn1(x) + 1

        @torch._dynamo.optimize(cnts3, nopython=True)
        def fn3(x):
            return torch.relu(fn2(x))

        fn3(torch.randn(4, 5))
        self.assertEqual(cnts2.frame_count, 0)
        self.assertEqual(cnts3.frame_count, 1)
        self.assertEqual(cnts3.op_count, 4)

    def test_nested_optimize_run(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnts, nopython=True)
        def fn(x):
            return torch.relu(torch.cos(x) + torch.sin(x))

        fn(torch.randn(4))
        self.assertEqual(cnts.frame_count, 1)

        fn(torch.randn(4, 4))
        self.assertEqual(cnts.frame_count, 2)

        # Test that run works on a decorated fn
        fn = torch._dynamo.run(fn)
        fn(torch.randn(4, 4, 4))
        self.assertEqual(cnts.frame_count, 2)

    def test_nested_optimize(self):
        cnts1 = torch._dynamo.testing.CompileCounter()
        cnts2 = torch._dynamo.testing.CompileCounter()

        def fn(x):
            return torch.relu(torch.cos(x) + torch.sin(x))

        fn1 = torch._dynamo.optimize(cnts1, nopython=True)(fn)
        fn2 = torch._dynamo.optimize(cnts2, nopython=True)(fn1)

        # The first optimize in the nesting should be ignored
        fn2(torch.randn(4))
        self.assertEqual(cnts2.frame_count, 1)
        self.assertEqual(cnts1.frame_count, 0)

        # Since the fn code object is already compiled, calling fn1 should
        # directly call the compiled_fn callable.
        torch._dynamo.run()(fn1)(torch.randn(4))
        self.assertEqual(cnts1.frame_count, 0)

        # Test same behavior by reversing the calls
        torch._dynamo.reset()
        cnts1 = torch._dynamo.testing.CompileCounter()
        cnts2 = torch._dynamo.testing.CompileCounter()
        fn1 = torch._dynamo.optimize(cnts1, nopython=True)(fn)
        fn2 = torch._dynamo.optimize(cnts2, nopython=True)(fn1)
        fn1(torch.randn(4))
        self.assertEqual(cnts1.frame_count, 1)
        torch._dynamo.run()(fn2)(torch.randn(4))
        self.assertEqual(cnts2.frame_count, 0)

    @patch.object(torch._dynamo.config, "suppress_errors", True)
    def test_nested_disable_decorator(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.disable()
        def fn1(x):
            return torch.sin(x) * 10

        @torch._dynamo.optimize(cnts)
        def fn2(x):
            x = x + 1
            x = x + 1
            x = fn1(x)  # graph break
            x = x + 1
            x = x + 1
            return x

        @torch._dynamo.optimize(cnts, nopython=True)
        def fn3(x):
            return fn2(x)

        fn2(torch.randn(4, 5))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

        try:
            fn3(torch.randn(4, 5))
            self.assertFalse(True)
        except torch._dynamo.exc.Unsupported as e:
            self.assertIn("call torch._dynamo.disable() wrapped function", str(e))

    def test_graph_break(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnts)
        def fn(x):
            x = torch.cos(x)
            x = torch.cos(x)
            torch._dynamo.graph_break()
            x = torch.cos(x)
            x = torch.cos(x)
            graph_break()
            x = torch.cos(x)
            x = torch.cos(x)
            return x

        fn(torch.randn(4, 5))
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 6)

    def test_torch_size(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            output_size = torch.Size([10, 10])
            x = x.view(*output_size)
            return (x,)

        x = torch.randn(100, requires_grad=True)
        x_clone = x.clone()
        ref = fn(x)

        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x_clone)

        self.assertTrue(same(ref, res))

    def test_torch_seed(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            attention_seed = int(torch.seed() % sys.maxsize)
            torch.manual_seed(attention_seed)
            return (x,)

        x = torch.randn(100, requires_grad=True)
        ref = fn(x)

        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))

    def test_is_tensor_like(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x):
            if torch.overrides.is_tensor_like(x):
                return (x * 2,)
            return (torch.ones(10) + x,)

        x = torch.randn(10)
        ref0 = f(x)
        ref1 = f(4)
        opt_f = torch._dynamo.optimize(cnts, nopython=True)(f)
        res0 = opt_f(x)
        res1 = opt_f(4)
        self.assertTrue(same(ref0, res0))
        self.assertTrue(same(ref1, res1))

    def test_is_tensor_like2(self):
        class MyTensor:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                if func is torch.max:
                    return torch.tensor(123)
                return func(*args, **kwargs)

        def fn(x):
            if torch.overrides.is_tensor_like(x):
                return torch.max(x)
            else:
                return torch.zeros(1)

        x = MyTensor()
        ref0 = fn(x)
        ref1 = fn(4)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res0 = opt_fn(x)
        res1 = opt_fn(4)
        self.assertTrue(same(ref0, res0))
        self.assertTrue(same(ref1, res1))

    def test_tensor_data(self):
        def fn(x, y):
            return x[y.data]

        x = torch.rand(8)
        y = torch.ones(8).to(torch.int)
        ref = fn(x, y)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_tensor_layout(self):
        def fn(x):
            return torch.zeros(
                [x.size()[0], x.size()[1]],
                dtype=x.dtype,
                layout=x.layout,
                device=x.device,
            )

        x = torch.rand(2, 3)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_version_ci(self):
        # temporary test to check that the ci torch version is set correctly
        self.assertTrue(hasattr(torch, "_subclasses"))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_rand(self):
        cnts = torch._dynamo.testing.CompileCounter()
        device = "cuda"

        def fn():
            return torch.randn(10, device=device)

        torch.manual_seed(10)
        ref_run1 = fn()

        torch.manual_seed(10)
        ref_run2 = fn()
        self.assertTrue(same(ref_run1, ref_run2))

        torch.manual_seed(10)
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn()

        self.assertTrue(same(res, ref_run1))

    def test_slice_input(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def getitem(a, idx):
            if isinstance(idx, slice):
                return (
                    torch.zeros(1),
                    a[idx]
                    + [
                        100,
                    ],
                )
            else:
                return (torch.zeros(1), a[idx])

        layers = list(range(10))
        ref0 = getitem(layers, slice(0, 2, 1))
        ref1 = getitem(layers, 2)
        ref2 = getitem(layers, slice(3, 8, 2))
        opt_getitem = torch._dynamo.optimize(cnts, nopython=True)(getitem)
        res0 = opt_getitem(layers, slice(0, 2, 1))
        res1 = opt_getitem(layers, 2)
        res2 = opt_getitem(layers, slice(3, 8, 2))

        self.assertTrue(ref0 == res0)
        self.assertTrue(ref1 == res1)
        self.assertTrue(ref2 == res2)

    def test_grad(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(a, b):
            out = a * b
            out.sum().backward()
            real_out = torch.sigmoid(a.grad + b)
            return real_out

        inps = [torch.randn(4, requires_grad=True) for _ in range(2)]
        for inp in inps:
            inp.grad = None
        ref = fn(*inps)

        for inp in inps:
            inp.grad = None
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(*inps)

        self.assertTrue(same(ref, res))

    @unittest.skipIf(sys.version_info < (3, 10), "use linetable when python >= 3.10")
    def test_linetable_writer(self):
        def fn():
            a = 10
            b = 20
            c = a + b
            f = "linetable_writer"
            return f"Test if {f} generates correct co_linetable: {c}"

        inst = dis.get_instructions(fn)
        result = bytecode_transformation.assemble(inst, fn.__code__.co_firstlineno)
        self.assertTrue(result[1] == fn.__code__.co_linetable)

    @unittest.skipIf(sys.version_info >= (3, 10), "use lnotab when python < 3.10")
    def test_lnotab_writer(self):
        def fn():
            a = 10
            b = 20
            c = a + b
            f = "lnotab_writer"
            return f"Test if {f} generates correct co_lnotab: {c}"

        inst = dis.get_instructions(fn)
        result = bytecode_transformation.assemble(inst, fn.__code__.co_firstlineno)
        self.assertTrue(result[1] == fn.__code__.co_lnotab)

    def test_torch_profiler(self):
        # wrap torch.profiler.* as NullContextVariable and do nothing
        def fn(x):
            y = x**2
            with torch.profiler.profile():
                y = y + 2
                with torch.profiler.record_function("my_function"):
                    z = y**3
                    z.tolist()  # graph break
                    z = z + 1
            return z

        x = torch.randn((2, 2), requires_grad=True)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_autograd_profiler(self):
        # wrap torch.autograd.profiler.* as NullContextVariable and do nothing
        def fn(x):
            y = x**2
            with torch.autograd.profiler.profile():
                y = y + 2
                with torch.autograd.profiler.record_function("my_function"):
                    z = y**3
                    z.tolist()  # graph break
                    z = z + 1
            return z

        x = torch.randn((2, 2), requires_grad=True)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_profiler_cache_lookup(self):
        def fn(x):
            y = x**2
            y = y + 2
            z = y**3
            return z

        x = torch.randn((2, 2), requires_grad=True)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="aot_eager")

        # warmup
        opt_fn(x)

        enable_cache_lookup_profiler(True)
        with torch.autograd.profiler.profile() as prof:
            res = opt_fn(x)
        events = list(
            filter(
                lambda event: event.name == "TorchDynamo Cache Lookup",
                prof.function_events,
            )
        )

        self.assertTrue(same(ref, res))
        self.assertTrue(
            len(events) == 1, "Expected one lookup profiler event for one opt_fn run"
        )

        enable_cache_lookup_profiler(False)
        with torch.autograd.profiler.profile() as prof:
            res = opt_fn(x)
        events = list(
            filter(
                lambda event: event.name == "TorchDynamo Cache Lookup",
                prof.function_events,
            )
        )

        self.assertTrue(same(ref, res))
        self.assertTrue(len(events) == 0, "Expected disabled profiling")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_context_manager1(self):
        def fn(x):
            s = torch.cuda.Stream()
            x = torch.mul(x, 5)
            x = torch.add(x, 2)
            with torch.cuda.stream(s):
                x = torch.relu(x)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2))
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 9)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_context_manager2(self):
        def fn(x, s):
            x = torch.mul(x, 5)
            x = torch.add(x, 2)
            with torch.cuda.stream(s):
                x = torch.relu(x)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2))
        s = torch.cuda.Stream()
        ref = fn(x, s)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x, s)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 8)

    def test_autograd_profiler_enabled(self):
        def fn(x):
            if torch.autograd._profiler_enabled():
                return x + 1
            else:
                return x - 1

        x = torch.randn((2, 2), requires_grad=True)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        if torch.autograd._profiler_enabled():
            torch.autograd._disable_profiler()
        assert not torch.autograd._profiler_enabled()
        ref = fn(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

        with torch.autograd.profiler.profile():
            assert torch.autograd._profiler_enabled()
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))

    def test_tensor_is_contiguous(self):
        def fn(x):
            input = torch.randn((1, 16, 1, 1))
            weight = torch.randn((8, 16, 3, 3))
            weight = weight.to(memory_format=x)
            output = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
            return output.is_contiguous(memory_format=x)

        opt_fn = torch._dynamo.optimize("eager")(fn)
        for x in [torch.contiguous_format, torch.channels_last]:
            self.assertEqual(fn(x), opt_fn(x))

    def test_python_slice(self):
        def f1(input):
            y = 0
            for i, x in enumerate(input[2:], 1):
                y = y + x
            return y

        def f2(input):
            y = 0
            for i, x in enumerate(input.shape[2:], 1):
                y = y + x
            return y

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f1 = torch._dynamo.optimize(cnts)(f1)
        opt_f2 = torch._dynamo.optimize(cnts)(f2)
        res1 = opt_f1([1, 2, 3, 5])
        res2 = opt_f2(torch.rand([2, 3, 4, 5]))

        self.assertEqual(res1, 8)
        self.assertEqual(res2, 9)

    def test_enum_as_dict_key(self):
        class MyEnum(enum.Enum):
            FOO = 10
            BAR = 20

        def fn(x):
            y = x + 2
            z = {
                MyEnum.FOO: torch.tensor(1),
                MyEnum.BAR: 10,
                "MyEnum.BAR": torch.tensor(8),
                5: torch.rand(3),
            }
            torch._dynamo.graph_break()
            a = z[MyEnum.FOO] + z["MyEnum.BAR"]
            b = y * 2
            return a, b

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        for _ in range(10):
            x = torch.rand(3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_const_dict_variable_python_type(self):
        from torch._dynamo.variables import ConstantVariable, ConstDictVariable

        d1 = {"a": ConstantVariable(10), "b": ConstantVariable(20)}
        d2 = collections.OrderedDict(
            [("x", ConstantVariable(12)), ("y", ConstantVariable(22))]
        )
        self.assertEqual(ConstDictVariable(d1, dict).python_type(), dict)
        self.assertEqual(
            ConstDictVariable(d2, collections.OrderedDict).python_type(),
            collections.OrderedDict,
        )

    def test_builtin_subclasses_as_method_on_class_type(self):
        class Foo:
            def __init__(self, name):
                self.ame_ = name

            def get_name(self):
                return "Foo " + self.name_

        class Bar(Foo):
            def __init__(self, name):
                self.name_ = name

            def get_name(self):
                return "Bar " + self.name_

        class Baz(Foo):
            def __init__(self, name):  # noqa: B903
                self.name_ = name

            def get_name(self):
                return "Baz " + self.name_

        subs_of_foo_reg = Foo.__subclasses__()

        counter = CompileCounter()

        @torch._dynamo.optimize_assert(counter)
        def fn():
            return Foo.__subclasses__()

        subs_of_foo_optim = fn()

        self.assertEqual(len(subs_of_foo_reg), 2)
        self.assertEqual(subs_of_foo_reg, subs_of_foo_optim)

    def test_builtin_subclasses_as_method_on_var(self):
        class Foo:
            def __init__(self, name):
                self.name_ = name

            def get_name(self):
                return "Foo " + self.name_

        class Bar(Foo):
            def __init__(self, name):
                self.name_ = name

            def get_name(self):
                return "Bar " + self.name_

        class Baz(Bar):
            def __init__(self, name):
                self.name_ = name

            def get_name(self):
                return "Baz " + self.name_

        subs_of_foo_reg = Foo.__subclasses__()
        sub_of_foo_subclass_var_reg = subs_of_foo_reg[0].__subclasses__()

        sub_of_foo_subclass_var_optim = list()
        counter = CompileCounter()

        @torch._dynamo.optimize_assert(counter)
        def fn():
            return Foo.__subclasses__()

        @torch._dynamo.optimize_assert(counter)
        def fn_single(subs_of_foo_optim):
            return subs_of_foo_optim[0].__subclasses__()

        subs_of_foo_optim = fn()
        sub_of_foo_subclass_var_optim = fn_single(subs_of_foo_optim)

        self.assertEqual(len(sub_of_foo_subclass_var_optim), 1)
        self.assertEqual(sub_of_foo_subclass_var_optim, sub_of_foo_subclass_var_reg)

    def test_enum_no_graphbreaks(self):
        class Foo(enum.Enum):
            FOO = 0
            BAR = 1

        def fn(x, foo):
            if foo is Foo.FOO:
                x = torch.add(x, 1.0)
            x = torch.mul(x, 1.0)
            return x

        x = torch.randn(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        opt_fn(x, Foo.FOO)
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        opt_fn(x, Foo.BAR)
        self.assertEqual(cnts.op_count, 1)

    def test_id_of_nn_module(self):
        class M(torch.nn.Module):
            def forward(self, x, ref_id):
                self_id = id(self)
                if self_id == ref_id:
                    x = torch.mul(x, 1.0)
                x = torch.add(x, 1.0)
                return x

        m = M().eval()
        data = torch.randn(1)
        cnts = torch._dynamo.testing.CompileCounter()
        correct_ref_id = id(m)
        opt_m = torch._dynamo.optimize(cnts, nopython=True)(m)
        opt_m(data, correct_ref_id)
        # Extra op is the recorded equality test (although once
        # the trace is flattened this is dead!)
        self.assertEqual(cnts.op_count, ifunspec(3, 2))

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        incorrect_ref_id = id(m) + 1
        opt_m = torch._dynamo.optimize(cnts, nopython=True)(m)
        opt_m(data, incorrect_ref_id)
        self.assertEqual(cnts.op_count, ifunspec(2, 1))

    def test_inline_func_jump_on_tensor_condition(self):
        def f1(input):
            if input == 0:
                return input + 1
            else:
                return input + 2

        def f2(input):
            return f1(input)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f2 = torch._dynamo.optimize(cnts)(f2)
        res1 = opt_f2(torch.tensor([1.0]))
        res2 = opt_f2(torch.tensor([0.0]))

        self.assertEqual(res1, 3)
        self.assertEqual(res2, 1)

    def test_frozenset_torch_func_contains(self):
        funcs = frozenset([torch.add])

        def fn(x, func):
            if func in funcs:
                x = torch.add(x, 1.0)
            x = torch.mul(x, 1.0)
            return x

        x = torch.randn(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        opt_fn(x, torch.add)
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        opt_fn(x, torch.mul)
        self.assertEqual(cnts.op_count, 1)

    def test_inline_list_mutation(self):
        def f1(x):
            x.append(torch.ones(8))
            return x

        def f2():
            x = [torch.ones(6)]
            f1(x)
            return x

        res1 = f2()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f2 = torch._dynamo.optimize(cnts)(f2)
        res2 = opt_f2()
        self.assertTrue(same(res1, res2))

    def test_inline_dict_mutation(self):
        def f1(d):
            d["c"] = d["a"] + d.pop("b")
            return d

        def f2():
            d = {"a": torch.ones(5), "b": torch.ones(5)}
            f1(d)
            return d

        res1 = f2()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f2 = torch._dynamo.optimize(cnts)(f2)
        res2 = opt_f2()
        self.assertTrue(same(res1, res2))

    def test_recursive_inline_list_mutation(self):
        def f1(x, y):
            x.append(torch.tensor([1.1]))
            y.append(torch.tensor([1.2]))
            return x, y

        def f2(x, y):
            x.append(torch.tensor([2.1]))
            y.append(torch.tensor([2.2]))
            f1(x, y)
            return x, y

        def f3(x):
            x.append(torch.tensor([3.1]))
            y = [torch.tensor([3.2])]
            f2(x, y)
            return x, y

        def f4():
            x = [torch.tensor([4.1])]
            return f3(x)

        res1 = f4()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f4 = torch._dynamo.optimize(cnts)(f4)
        res2 = opt_f4()
        self.assertTrue(same(res1, res2))

    def test_disallow_in_graph(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnts)
        def fn(a):
            x = torch.add(a, 1)
            x = torch.add(x, 1)
            x = torch.sub(x, 1)
            x = torch.add(x, 1)
            x = torch.add(x, 1)
            return x

        torch._dynamo.disallow_in_graph(torch.sub)
        fn(torch.randn(10))
        torch._dynamo.allow_in_graph(torch.sub)

        # check for graph break on sub
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

    def test_allow_in_graph(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnts)
        def fn(a):
            x = torch.add(a, 1)
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            x = torch.add(x, 1)
            return x

        torch._dynamo.allow_in_graph(my_custom_function)
        fn(torch.randn(10))
        torch._dynamo.disallow_in_graph(my_custom_function)

        # check for no graph break
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

    def test_sample_input(self):
        from torch.testing._internal.common_methods_invocations import SampleInput

        def fn(sample):
            if isinstance(sample.input, torch.Tensor):
                return sample.input * 2
            return torch.zeros(())

        sample = SampleInput(torch.ones(2))
        ref = fn(sample)

        opt_fn = torch._dynamo.optimize("eager")(fn)
        res = opt_fn(sample)

        self.assertTrue(same(ref, res))

    def test_release_input_memory(self):
        x = torch.rand([4])
        x_ref = weakref.ref(x)

        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnts)
        def foo(x):
            return x + x

        out = foo(x)
        self.assertTrue(same(out, x + x))
        del x
        self.assertIs(x_ref(), None)

    def test_release_module_memory(self):
        mod = torch.nn.Linear(10, 10)
        x = torch.rand([10, 10])
        mod_weight_ref = weakref.ref(mod.weight)
        mod_ref = weakref.ref(mod)

        # Modules that are passed into torch._dynamo optimized functions
        # will normally be held onto through the generated GraphModule,
        # which contains the modules. remove the reference in this backend
        # and test that no additional references are being held.
        class NoLeakBackend:
            def __call__(self, gm: torch.fx.GraphModule, example_inputs):
                gm.mod = None

                def foo(*args, **kwargs):
                    return (1,)

                return foo

        no_leak_backend = NoLeakBackend()

        @torch._dynamo.optimize(no_leak_backend)
        def foo(mod, x):
            return mod(x)

        foo(mod, x)
        del mod
        del x
        self.assertIsNone(mod_ref(), None)
        self.assertIsNone(mod_weight_ref(), None)

    def test_update_locals_and_stack_uses_shared_cache(self):
        def fn(x):
            perm = [0, 3, 5]
            perm = list(range(min(perm))) + perm
            perm.extend(i for i in range(x.dim()) if i not in perm)
            return perm

        x = torch.rand([2, 2, 2, 2, 2, 2])
        res1 = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    def test_dict_reconstruct_keeps_original_order(self):
        def fn():
            modules = collections.OrderedDict([("act", torch.nn.ReLU())])
            module_dict = torch.nn.ModuleDict(modules)

            next_modules = {"fc4": torch.nn.Linear(5, 6), "act3": torch.nn.Sigmoid()}
            modules.update(next_modules.items())
            module_dict.update(next_modules)
            return modules, module_dict

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        modules, module_dict = opt_fn()

        self.assertEqual(len(module_dict), len(modules))
        for k1, m2 in zip(modules, module_dict.children()):
            self.assertTrue(modules[k1] is m2)

    def test_side_effects_codegen_update_mutated(self):
        # codegen to update mutated variables with side effect
        # should after stack value's codegen
        def f1(x):
            alist = [x]
            alist.append(x + 1)
            alist[0].sum().item()  # graph break
            res = alist.pop()
            res.sum().item()  # graph break
            return res

        def f2(a, b):
            d = {"a": a + 1, "b": b + 2}
            x = d.pop("b")
            x.sum().item()  # graph break
            y = d["a"] + x
            y.sum().item()  # graph break
            d["c"] = y
            return d

        x = torch.rand([2, 3])
        a = torch.rand([5, 6])
        b = torch.rand([5, 6])
        res11 = f1(x)
        res21 = f2(a, b)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f1 = torch._dynamo.optimize(cnts)(f1)
        opt_f2 = torch._dynamo.optimize(cnts)(f2)
        res12 = opt_f1(x)
        res22 = opt_f2(a, b)
        self.assertTrue(same(res11, res12))
        self.assertTrue(same(res21, res22))

    def test_list_append_return_none(self):
        def fn(x):
            alist = []
            blist = alist.append(x + 1)
            return alist, blist

        x = torch.tensor([2.3])
        res = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res2 = opt_fn(x)
        self.assertEqual(res, res2)

    def test_tensor_types(self):
        def fn(dtype, tensor_type):
            x = torch.empty(4, dtype=dtype)
            assert isinstance(x, tensor_type)

        opt_fn = torch._dynamo.optimize("eager")(fn)
        opt_fn(torch.float32, torch.FloatTensor)
        opt_fn(torch.float64, torch.DoubleTensor)
        opt_fn(torch.float16, torch.HalfTensor)
        opt_fn(torch.bfloat16, torch.BFloat16Tensor)
        opt_fn(torch.uint8, torch.ByteTensor)
        opt_fn(torch.int8, torch.CharTensor)
        opt_fn(torch.int64, torch.LongTensor)
        opt_fn(torch.int, torch.IntTensor)
        opt_fn(torch.int16, torch.ShortTensor)
        opt_fn(torch.bool, torch.BoolTensor)

    def test_nan(self):
        def f(x, n):
            return x * 2 + n

        x = torch.randn(4)
        n = float("nan")

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f = torch._dynamo.optimize(cnts)(f)
        opt_f(x, n)
        opt_f(x, n)
        self.assertEqual(cnts.frame_count, 1)

    @patch.object(torch._dynamo.config, "dynamic_shapes", True)
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_item(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                z = torch.max(x)
                return z.int().item()

        x = torch.tensor([[10.6763, 11.7445, -2.2369]])
        model = MyMod()
        y = torch._dynamo.optimize("eager", nopython=True)(model)(x)

        self.assertEqual(y, 11)

    @patch.object(torch._dynamo.config, "dynamic_shapes", True)
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_item_changes(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                z = torch.max(x)
                return z.int().item()

        x = torch.tensor([[10.6763, 11.7445, -2.2369]])
        model = MyMod()
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)
        y = opt_model(x)
        z = opt_model(torch.tensor([[y - 5, y + 10, y + 50]]))

        self.assertEqual(y, 11)
        self.assertEqual(z, 61)

    @patch.object(torch._dynamo.config, "dynamic_shapes", True)
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_item_changes_new_shape(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                z = torch.max(x)
                return z.int().item()

        x = torch.tensor([[10.6763, 11.7445, -2.2369]])
        model = MyMod()
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)
        y = opt_model(x)
        z = opt_model(torch.tensor([[y - 5, y + 50], [y + 5, y - 50]]))

        self.assertEqual(y, 11)
        self.assertEqual(z, 61)

    def test_cross_entropy_loss_fancy_ctor(self):
        output = None
        rand_5 = torch.randn(5)
        rand_3_5 = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        loss = torch.nn.CrossEntropyLoss(
            weight=rand_5, reduce=False, label_smoothing=0.5
        )
        opt_loss = torch._dynamo.optimize("eager", nopython=True)(loss)
        input = rand_3_5
        dynamo_output = opt_loss(input, target)

        loss = torch.nn.CrossEntropyLoss(
            weight=rand_5, reduce=False, label_smoothing=0.5
        )
        input = rand_3_5
        output = loss(input, target)

        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_cross_entropy_loss_simple_ctor(self):
        output = None
        rand_3_5 = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        loss = torch.nn.CrossEntropyLoss()
        opt_loss = torch._dynamo.optimize("eager", nopython=True)(loss)
        input = rand_3_5
        dynamo_output = opt_loss(input, target)

        loss = torch.nn.CrossEntropyLoss()
        input = rand_3_5
        output = loss(input, target)

        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_nn_functional_reduction(self):
        def fn(loss, reduction):
            reduction_enum = F._Reduction.get_enum(reduction)
            if reduction_enum == 0:
                return loss
            elif reduction_enum == 1:
                return loss.mean()
            elif reduction_enum == 2:
                return loss.sum()

        x = torch.rand([3, 5])
        y = "mean"
        ref = fn(x, y)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x, y)
        self.assertTrue(torch.allclose(ref, res))

    def test_large_reduction_list(self):
        dtype = torch.float32
        device = "cpu"

        def check_sum_all(tensor: torch.Tensor) -> None:
            pylist = tensor.reshape(-1).tolist()
            self.assertTrue(same(tensor.sum(), torch.tensor(sum(pylist))))

        check_sum_all(torch.randn(200000, dtype=dtype, device=device))

    def test_raise_on_backend_error(self):
        def my_compiler(gm, _):
            raise RuntimeError("duck!")

        @torch._dynamo.optimize(my_compiler)
        def fn(a, b):
            return a + b / (a - b)

        self.assertRaises(
            torch._dynamo.exc.BackendCompilerFailed,
            lambda: fn(torch.randn(10), torch.randn(10)),
        )

    def test_named_parameters(self):
        n_embd = 768
        block_size = 128
        vocab_size = 65
        embd_pdrop = 0.1

        class MyModel2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                self.drop = torch.nn.Dropout(embd_pdrop)

            def forward(self, x):
                return x

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                self.drop = torch.nn.Dropout(embd_pdrop)
                self.submod2 = MyModel2()

            def forward(self, x):
                return x

        # Regular
        params = []
        mod = MyModel()
        actual_params = list(mod.named_parameters())

        @torch._dynamo.optimize("eager", nopython=True)
        def fn():
            return list(mod.named_parameters())

        params = fn()

        self.assertEqual(len(actual_params), len(params))
        for idx in range(len(params)):
            k_a, v_a = actual_params[idx]
            k, v = params[idx]
            self.assertEqual(k_a, k)
            self.assertTrue(torch.allclose(v_a, v))

        # Prefix
        params = []
        mod = MyModel()
        actual_params = list(mod.named_parameters(prefix="foo"))

        @torch._dynamo.optimize("eager", nopython=True)
        def fn1():
            return list(mod.named_parameters(prefix="foo"))

        params = fn1()

        self.assertEqual(len(actual_params), len(params))
        for idx in range(len(params)):
            k_a, v_a = actual_params[idx]
            k, v = params[idx]
            self.assertEqual(k_a, k)
            self.assertTrue(torch.allclose(v_a, v))

    def test_module_complex_iter(self):
        n_embd = 768
        block_size = 128
        vocab_size = 65
        embd_pdrop = 0.1

        class FakeGPT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                self.drop = torch.nn.Dropout(embd_pdrop)
                self.ln_f = torch.nn.LayerNorm(n_embd)
                self.head = torch.nn.Linear(n_embd, vocab_size, bias=False)

                self.block_size = block_size
                self.names = []

            def forward(self, idx, targets=None):
                b, t = idx.size()
                assert (
                    t <= self.block_size
                ), "Cannot forward, model block size is exhausted."

                # forward the GPT model
                token_embeddings = self.tok_emb(
                    idx
                )  # each index maps to a (learnable) vector
                position_embeddings = self.pos_emb[
                    :, :t, :
                ]  # each position maps to a (learnable) vector
                x = self.drop(token_embeddings + position_embeddings)
                x = self.blocks(x)
                x = self.ln_f(x)
                logits = self.head(x)

                # if we are given some desired targets also calculate the loss
                loss = None
                if targets is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )

                return logits, loss

            def foo(self, memo=None, prefix="", remove_duplicate=False):
                for mn, m in self.named_modules(
                    memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
                ):
                    for pn, p in self.named_parameters():
                        fpn = "%s.%s" % (mn, pn) if mn else pn
                        self.names.append(fpn)

        # Test plain recurse
        model_a = FakeGPT()
        model_a.foo()
        a_names = model_a.names

        model_b = FakeGPT()
        opt_model_b = torch._dynamo.optimize("eager", nopython=True)(model_b)
        opt_model_b.foo()

        self.assertEqual(a_names, model_b.names)

        # Test with prefix
        model_a = FakeGPT()
        model_a.foo(prefix="abc")
        a_names = model_a.names

        model_b = FakeGPT()
        opt_model_b = torch._dynamo.optimize("eager", nopython=True)(model_b)
        opt_model_b.foo(prefix="abc")

        self.assertEqual(a_names, model_b.names)

    def test_numpy_variable_isinstance(self):
        def fn(x, m):
            if isinstance(m, np.ndarray):
                return x + 1
            else:
                return x - 1

        x = torch.tensor([2.3])
        m = np.array([1, 2, 3])
        ref = fn(x, m)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x, m)
        self.assertEqual(ref, res)

    def test_tensor_dot_grad_no_graph_break(self):
        def fn(a, b):
            y = 3 * a**3 - b**2
            y.backward(gradient=torch.tensor([1.0, 1.0]))
            b.grad.zero_()
            return a.grad, b.grad

        a = torch.tensor([2.0, 3.0], requires_grad=True)
        b = torch.tensor([6.0, 4.0], requires_grad=True)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        _, b_grad = opt_fn(a, b)
        self.assertTrue(same(b_grad, torch.tensor([0.0, 0.0])))
        self.assertEqual(cnts.frame_count, 2)

    def test_torch_nn_parameter_isinstance(self):
        def fn(x):
            a = torch.nn.Parameter(torch.rand(2, 3))
            if isinstance(a, torch.Tensor):
                return x + 1
            else:
                return x - 1

        x = torch.tensor([2.5])
        ref = fn(x)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(raise_on_backend_change=True)
    def test_change_backends(self):
        @torch._dynamo.optimize("eager", nopython=True)
        def fn1():
            return x + 1

        @torch._dynamo.optimize("ts")
        def fn2():
            return x + 2

        @torch._dynamo.optimize("eager", nopython=False)
        def fn3():
            return x + 1

        x = torch.tensor([3, 5])

        fn1()
        fn1()
        fn3()
        self.assertRaises(torch._dynamo.exc.ResetRequired, fn2)
        fn1()
        torch._dynamo.reset()
        fn2()
        fn2()
        self.assertRaises(torch._dynamo.exc.ResetRequired, fn1)
        self.assertRaises(torch._dynamo.exc.ResetRequired, fn3)
        fn2()

    def test_dynamo_min_operator_with_shape(self):
        @torch._dynamo.optimize("eager", nopython=True)
        def f(x, a):
            return min(x.shape[0], a)

        result = f(torch.ones(6), 3)
        self.assertEqual(result, 3)

    @patch.object(torch._dynamo.config, "dynamic_shapes", True)
    def test_onnx_shape_as_tensor(self):
        @torch._dynamo.optimize("eager", nopython=True)
        def f(x):
            return 1 + torch._shape_as_tensor(x)[0]

        gm, _ = torch._dynamo.export(f, torch.ones(6))

        input_one_dim = torch.ones(6)
        input_two_dims = torch.ones(7, 4)
        self.assertEqual(f(input_one_dim), 7)
        self.assertEqual(f(input_two_dims), 8)
        self.assertEqual(f(input_two_dims), 8)

        @torch._dynamo.optimize("eager", nopython=True)
        def f_onnx(x):
            return 1 + torch.onnx.operators.shape_as_tensor(x)[0]

        self.assertEqual(f_onnx(input_one_dim), 7)
        self.assertEqual(f_onnx(input_two_dims), 8)
        self.assertEqual(f_onnx(input_two_dims), 8)

    def test_cond(self):
        from functorch.experimental.control_flow import cond

        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        def f(pred, x):
            return cond(pred, true_fn, false_fn, [x])

        opt_fn = torch._dynamo.optimize("eager")(f)
        a = opt_fn(torch.tensor(False), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.cos(torch.tensor([0.25, 0.25])), a))
        b = opt_fn(torch.tensor(True), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), b))

    @unittest.expectedFailure
    def test_cond_side_effects(self):
        from functorch.experimental.cond import cond

        c = 0

        def true_fn(x):
            return x - c

        def false_fn(x):
            return x + c

        def f(pred, x):
            nonlocal c
            c = 1
            return cond(pred, true_fn, false_fn, [x])

        opt_fn = torch._dynamo.optimize("eager")(f)
        c = 0
        a = opt_fn(torch.tensor(False), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.tensor([1.25, 1.25]), a))

    def test_map_side_effects(self):
        from functorch.experimental.control_flow import map

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.tensor(1)

            def forward(self, xs):
                def body(x):
                    self.w += 1
                    return x

                return map(body, xs)

        mod = Module()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Graph state change detected",
        ):
            opt_fn = torch._dynamo.optimize("eager", nopython=True)(mod)
            opt_fn(torch.randn(3, 2))

    def test_cond_nested(self):
        from functorch.experimental.control_flow import cond

        def true_fn_nested(x):
            return x * 10

        def false_fn_nested(x):
            return x * -1

        def true_fn(pred2, x):
            return x.sin()

        def false_fn(pred2, x):
            return x + cond(pred2, true_fn_nested, false_fn_nested, [x])

        def f(pred, pred2, x):
            return cond(pred, true_fn, false_fn, [pred2, x])

        cc = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cc)(f)
        true_true_sin = opt_fn(
            torch.tensor(True), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_true_sin))

        true_false_sin = opt_fn(
            torch.tensor(True), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_false_sin))

        false_true_sum_mult = opt_fn(
            torch.tensor(False), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([2.75, 2.75]), false_true_sum_mult)
        )  # * 10 then add x

        false_false_sum_neg = opt_fn(
            torch.tensor(False), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([0.0, 0.0]), false_false_sum_neg)
        )  # * -1 then add x
        self.assertTrue(cc.frame_count, 2)

    def test_cond_export(self):
        from functorch.experimental.control_flow import cond

        def true_fn_nested(x):
            return x * 10

        def false_fn_nested(x):
            return x * -1

        def true_fn(pred2, x):
            return x.sin()

        def false_fn(pred2, x):
            return x + cond(pred2, true_fn_nested, false_fn_nested, [x])

        def f(pred, pred2, x):
            return cond(pred, true_fn, false_fn, [pred2, x])

        graph, guard = torch._dynamo.export(
            f, torch.tensor(False), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        true_true_sin = graph(
            torch.tensor(True), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_true_sin))

        true_false_sin = graph(
            torch.tensor(True), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_false_sin))

        false_true_sum_mult = graph(
            torch.tensor(False), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([2.75, 2.75]), false_true_sum_mult)
        )  # * 10 then add x

        false_false_sum_neg = graph(
            torch.tensor(False), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([0.0, 0.0]), false_false_sum_neg)
        )  # * -1 then add x

    def test_cond_export_single_arg(self):
        from functorch.experimental.control_flow import cond

        def true_fn(x):
            return x

        def false_fn(x):
            return x.sin()

        def f(pred, x):
            return cond(pred, true_fn, false_fn, [x])

        graph, guard = torch._dynamo.export(
            f, torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        true_mirror = graph(torch.tensor(True), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.tensor([0.25, 0.25]), true_mirror))
        true_mirror_2 = graph(torch.tensor(True), torch.tensor([0.33, 0.33, 0.33]))
        self.assertTrue(same(torch.tensor([0.33, 0.33, 0.33]), true_mirror_2))

        false_sin = graph(torch.tensor(False), torch.tensor([0.5, 0.5]))
        self.assertTrue(same(torch.sin(torch.tensor([0.5, 0.5])), false_sin))

    def test_disable_optimize(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnt, disable=True)
        def f1(x):
            return x + 1

        f1(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

        @torch._dynamo.optimize(cnt, disable=True)
        def f2(x):
            return x + 1

        f2(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

        with patch.dict(os.environ, {"TORCHDYNAMO_DISABLE": "1"}):

            @torch._dynamo.optimize(cnt)
            def f3(x):
                return x + 1

            f3(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

    def test_config_log_level(self):
        @torch._dynamo.optimize("eager")
        def fn(a, b):
            return a + b

        with self.assertLogs(logger="torch._dynamo", level=logging.DEBUG) as log:
            torch._dynamo.config.log_level = logging.DEBUG
            fn(torch.randn(10), torch.randn(10))
            cur_len = len(log)
            self.assertGreater(cur_len, 0)

            torch._dynamo.config.log_level = logging.WARNING
            fn(torch.randn(10), torch.randn(10))
            self.assertEqual(cur_len, len(log))

    @patch.object(torch._dynamo.config, "print_graph_breaks", True)
    def test_duplicate_graph_break_warning(self):
        @torch._dynamo.optimize("eager")
        def f1(a, b):
            f2(a, b)

        def f2(a, b):
            c = a + b
            print("break")
            return a + b + c

        @torch._dynamo.optimize("eager")
        def g1(a, b):
            g2(a, b)

        def g2(a, b):
            c = a + b
            print("break")
            return a + b + c

        def count_graph_break_msgs(msgs):
            return sum(msg.find("Graph break") != -1 for msg in msgs)

        with self.assertLogs(logger="torch._dynamo", level=logging.WARNING) as log:
            torch._dynamo.config.verbose = True
            f1(torch.randn(10), torch.randn(10))
            self.assertGreater(count_graph_break_msgs(log.output), 1)

        with self.assertLogs(logger="torch._dynamo", level=logging.WARNING) as log:
            torch._dynamo.config.verbose = False
            g1(torch.randn(10), torch.randn(10))
            self.assertEqual(count_graph_break_msgs(log.output), 1)

    def test_inplace_param_update(self):
        def fn(param, y):
            prev_grad = torch.is_grad_enabled()
            try:
                torch.set_grad_enabled(False)
                torch.set_grad_enabled(True)
                torch.set_grad_enabled(False)
                param.add_(y)
            finally:
                torch.set_grad_enabled(prev_grad)

        y = torch.randn(4)
        x = torch.nn.Parameter(torch.randn(4))
        fn(x, y)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        opt_fn(x, y)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast(self):
        if not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest("requires bf16")

        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")
                d_float32 = torch.rand((8, 8), device="cuda")

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, guards = torch._dynamo.export(module, torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "cuda")
        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.bfloat16)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater,
        "Can't run fused SDPA on this platform",
    )
    def test_autocast_sdpa(self):
        class MyModule(torch.nn.Module):
            def forward(self, query, key, value):
                with torch.autocast("cpu"):
                    with torch.autocast("cuda", dtype=torch.float32):
                        out = F.scaled_dot_product_attention(
                            query, key, value, None, 0.5, True
                        )
                return out

        dtype = torch.float32
        seq_len_q = 1
        seq_len_k = 1
        head_dim = 8
        query = torch.ones(
            1, 8, seq_len_q, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        key = torch.ones(
            1, 8, seq_len_k, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        value = torch.ones(
            1, 8, seq_len_k, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )

        module = MyModule()
        real = module(query, key, value)
        real_device = real.device
        real_dtype = real.dtype

        opt_mod = torch._dynamo.optimize("inductor")(module)
        compiled = opt_mod(query, key, value)

        self.assertEqual(compiled.device, real_device)
        self.assertEqual(compiled.dtype, real_dtype)

        self.assertEqual(compiled.device.type, "cuda")
        self.assertEqual(compiled.device.index, 0)
        self.assertEqual(compiled.dtype, torch.float32)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater,
        "Can't run fused SDPA on this platform",
    )
    def test_parsing_sdpa(self):
        class MyModule(torch.nn.Module):
            def forward(self, query, key, value):
                out = F.scaled_dot_product_attention(query, key, value, None, 0, True)
                out = F.scaled_dot_product_attention(
                    query, key, value, None, 0, True, scale=8
                )
                out = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=0,
                    is_causal=True,
                )
                out = F.scaled_dot_product_attention(
                    query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=0,
                    is_causal=True,
                )
                out = F.scaled_dot_product_attention(
                    query, key, value, None, dropout_p=0, is_causal=True
                )
                out = F.scaled_dot_product_attention(query, key, value, None, scale=8)
                return out

        device = "cuda"
        dtype = torch.float16
        seq_len_q = 1
        seq_len_k = 1
        head_dim = 8
        query = torch.ones(
            1, 8, seq_len_q, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        key = torch.ones(
            1, 8, seq_len_k, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        value = torch.ones(
            1, 8, seq_len_k, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        module = MyModule()
        opt_mod = torch._dynamo.optimize("inductor")(module)
        opt_mod(query, key, value)

    def test_autocast_cpu(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")
                d_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, guards = torch._dynamo.export(module, torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "cpu")
        self.assertEqual(exported.dtype, torch.bfloat16)

    def test_autocast_cpu_graph_break(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")
                torch._dynamo.graph_break()
                d_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    torch._dynamo.graph_break()
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        opt = torch._dynamo.optimize("eager")(module)
        res = opt(torch.tensor([0.5]))
        self.assertEqual(res.device, real_device)
        self.assertEqual(res.dtype, real_dtype)

        self.assertEqual(res.device.type, "cpu")
        self.assertEqual(res.dtype, torch.bfloat16)

    def test_autocast_cpu_graph_break_2(self):
        # Regression for: https://github.com/pytorch/pytorch/issues/93890
        def fn(x):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                x = torch.mm(x, x)
                torch._dynamo.graph_break()
                x = torch.relu(x)
            return x

        x = torch.rand([4, 4])
        self.assertEqual(x.dtype, torch.float32)
        res = fn(x)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        opt_res = opt_fn(x)
        self.assertTrue(torch.allclose(res, opt_res))
        self.assertEqual(res.dtype, torch.bfloat16)
        self.assertEqual(opt_res.dtype, torch.bfloat16)

    def test_autocast_cpu_graph_break_inner_fn(self):
        class MyModule(torch.nn.Module):
            @staticmethod
            def mm_breaks(x, y):
                torch._dynamo.graph_break()
                return torch.mm(x, y)

            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    torch._dynamo.graph_break()
                    with torch.autocast(
                        device_type="cpu", dtype=torch.bfloat16, enabled=False
                    ):
                        torch._dynamo.graph_break()
                        g_float32 = torch.mm(a_float32, b_float32)
                        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                            # Check that nested with non-inlineable function with graph break
                            torch._dynamo.graph_break()
                            f_float16_1 = self.mm_breaks(a_float32, b_float32)
                    # We remember to exit the inner autocast correctly to outer
                    # even after graph breaks
                    f_float16 = self.mm_breaks(a_float32, b_float32)
                    assert f_float16.dtype == f_float16_1.dtype
                return f_float16, g_float32

        module = MyModule()
        real_16, real_32 = module(torch.tensor([0.5]))
        real_device_16 = real_16.device
        real_dtype_16 = real_16.dtype
        real_device_32 = real_32.device
        real_dtype_32 = real_32.dtype

        graph = torch._dynamo.optimize("eager")(module)
        out_16, out_32 = graph(torch.tensor([0.5]))
        self.assertEqual(out_16.device, real_device_16)
        self.assertEqual(out_16.dtype, real_dtype_16)
        self.assertEqual(out_32.device, real_device_32)
        self.assertEqual(out_32.dtype, real_dtype_32)

        self.assertEqual(out_16.device.type, "cpu")
        self.assertEqual(out_16.dtype, torch.bfloat16)
        self.assertEqual(out_32.device.type, "cpu")
        self.assertEqual(out_32.dtype, torch.float32)

    def test_autocast_graph_break_method(self):
        class MyModule(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.bias = bias

            def mm_not_break(self, x, y):
                return torch.mm(x, y) + self.bias

            def mm_breaks(self, x, y):
                torch._dynamo.graph_break()
                return torch.mm(x, y) + self.bias

            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    with torch.autocast(
                        device_type="cpu", dtype=torch.bfloat16, enabled=False
                    ):
                        g_float32 = torch.mm(a_float32, b_float32)
                    f_float16 = self.mm_breaks(a_float32, b_float32)

                    assert (
                        f_float16[0][0] == self.mm_not_break(a_float32, b_float32)[0][0]
                    )
                return f_float16, g_float32

        module = MyModule(bias=torch.rand((8, 8), device="cpu", dtype=torch.bfloat16))

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            # Autocast doesn't work on addition, so we need the bias to be `bfloat16`
            res = torch.rand((8, 8), device="cpu", dtype=torch.float32) + torch.rand(
                (8, 8), device="cpu", dtype=torch.bfloat16
            )
            self.assertEqual(res.dtype, torch.float32)

        real_16, real_32 = module(torch.tensor([0.5]))
        real_device_16 = real_16.device
        real_dtype_16 = real_16.dtype
        real_device_32 = real_32.device
        real_dtype_32 = real_32.dtype

        graph = torch._dynamo.optimize("eager")(module)
        out_16, out_32 = graph(torch.tensor([0.5]))
        self.assertEqual(out_16.device, real_device_16)
        self.assertEqual(out_16.dtype, real_dtype_16)
        self.assertEqual(out_32.device, real_device_32)
        self.assertEqual(out_32.dtype, real_dtype_32)

        self.assertEqual(out_16.device.type, "cpu")
        self.assertEqual(out_16.dtype, torch.bfloat16)
        self.assertEqual(out_32.device.type, "cpu")
        self.assertEqual(out_32.dtype, torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast_float64(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")
                d_float32 = torch.rand((8, 8), device="cuda")

                with torch.autocast(device_type="cuda", dtype=torch.float64):
                    e_float64 = torch.mm(a_float32, b_float32)
                    f_float64 = torch.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, guards = torch._dynamo.export(module, torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.float64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast_device(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")
                d_float32 = torch.rand((8, 8), device="cuda")

                with torch.autocast("cuda"):
                    e_float64 = torch.mm(a_float32, b_float32)
                    f_float64 = torch.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, guards = torch._dynamo.export(module, torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.torch.float16)

    def test_generate_tensor_from_list_of_numpy_primitive_type(self):
        # Test sth like torch.LongTensor(list(np.int64, np.int64, ...))
        def fn():
            x = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
            y = [x[0], x[2], x[4]]
            z = torch.LongTensor(y)
            return z

        ref = fn()
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res = opt_fn()
        self.assertTrue(same(ref, res))

    def test_autograd_function_equivalence(self):
        for i in range(1, 5):
            model = globals()[f"Module{i}"]()
            opt_model = torch._dynamo.optimize("eager", nopython=True)(model)
            self.assertTrue(
                torch.allclose(opt_model(torch.ones(2, 3)), torch.tensor([2.0]))
            )

    def test_autograd_function_has_graph_break(self):
        x = torch.randn(10)
        for model in [Module5(), Module6()]:
            torch._dynamo.reset()
            cnts = torch._dynamo.testing.CompileCounter()
            opt_model = torch._dynamo.optimize(cnts)(model)
            for _ in range(3):
                ref = model(x)
                res = opt_model(x)
                self.assertTrue(torch.allclose(ref, res))
            self.assertEqual(cnts.frame_count, 2)

    def test_object_classmethod(self):
        class C:
            @classmethod
            def fn(cls, x):
                return x + x

        @torch._dynamo.optimize("eager", nopython=True)
        def f():
            return C().fn(torch.ones(2, 3))

        self.assertTrue(torch.allclose(f(), torch.tensor([2.0])))

    def test_object_staticmethod(self):
        class C:
            @staticmethod
            def fn(x):
                return x + x

        @torch._dynamo.optimize("eager", nopython=True)
        def f():
            return C().fn(torch.ones(2, 3))

        self.assertTrue(torch.allclose(f(), torch.tensor([2.0])))

    def test_user_function_variable_supports_enum_argument(self):
        class Foo(enum.Enum):
            FOO = 0
            BAR = 1

        def gn(x, y=Foo.FOO):
            if y is Foo.FOO:
                return x
            else:
                return x + 1

        def fn(x):
            return gn(x)

        x = torch.randn(2, 3)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_user_function_variable_supports_type_abcmeta_argument(self):
        class Foo(metaclass=abc.ABCMeta):
            @abc.abstractclassmethod
            def read(self):
                pass

        class Bar(Foo):
            def read(self):
                return "Hello World!"

        class Baz:
            pass

        def gn(x, tys=(Bar, Baz)):
            if Bar in tys:
                return x - 1
            else:
                return x + 1

        def fn(x):
            return gn(x)

        x = torch.randn(2, 3)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_user_function_variable_supports_function_argument(self):
        def add1(x):
            return x + 1

        def add2(x):
            return x + 2

        def gn(x, f=add1):
            if f is add1:
                return x + 1
            else:
                return x + 2

        def fn(x, f):
            return gn(x, f)

        x = torch.randn(2, 3)
        ref = fn(x, add2)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x, add2)
        self.assertTrue(torch.allclose(ref, res))

    def test_typing_variable_isinstance(self):
        def fn(x, m):
            if isinstance(m, typing.Mapping):
                return x + 1
            else:
                return x - 1

        x = torch.randn(2, 3)
        m = {"x": torch.randn(3)}
        ref = fn(x, m)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res = opt_fn(x, m)
        self.assertTrue(torch.allclose(ref, res))

    def test_repro_graph_breaks_in__get_item_by_idx(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = torch.nn.Sequential(
                    torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
                )

            def forward(self, x):
                return self.mod[0](x)

        m = Mod()
        graph, _ = torch._dynamo.export(m, torch.randn(3, 3))

    def test_nn_sequential_invocation(self):
        with freeze_rng_state():

            class TestModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linears = torch.nn.Sequential(
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                    )

                def forward(self, x):
                    all_but_last = self.linears[:-1]
                    return all_but_last(x)

            m = TestModel()
            x = torch.rand((2, 2))
            real = m(x)
            graph, _ = torch._dynamo.export(m, x)
            dynamo_result = graph(x)
            self.assertTrue(same(real, dynamo_result))

    def test_nn_sequential_invocation_reposition_indices(self):
        with freeze_rng_state():

            class TestModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linears = torch.nn.Sequential(
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                        torch.nn.Linear(2, 2),
                    )

                def forward(self, x):
                    all_but_last = self.linears[1:3]
                    return all_but_last(x)

            m = TestModel()
            x = torch.rand((2, 2))
            real = m(x)
            graph, _ = torch._dynamo.export(m, x)
            dynamo_result = graph(x)
            self.assertTrue(same(real, dynamo_result))

    def test_error_on_nested_fx_trace(self):
        input = torch.rand(2, 3)

        def f(x):
            x + x

        real = f(input)

        optimized = torch._dynamo.optimize("eager")(f)
        self.assertTrue(same(optimized(input), real))

        with self.assertRaisesRegex(RuntimeError, "Detected that you are using FX"):
            gm = torch.fx.symbolic_trace(optimized)

    @patch.object(torch._dynamo.config, "error_on_nested_fx_trace", False)
    def test_no_error_on_nested_fx_trace(self):
        input = torch.rand(2, 3)

        def f(x):
            x + x

        real = f(input)

        optimized = torch._dynamo.optimize("eager")(f)
        self.assertTrue(same(optimized(input), real))

        # should not error
        gm = torch.fx.symbolic_trace(optimized)
        self.assertTrue(same(gm(input), real))

    def test_not_dynamic_scope(self):
        def f(y):
            x = 1

            def g():
                x = 2
                return lambda: x

            return y + g()()

        input = torch.zeros(1)
        real = f(input)
        optimized = torch._dynamo.optimize("eager")(f)
        opt = optimized(input)
        self.assertTrue(same(opt, real))

    def test_inference_mode(self):
        @torch.inference_mode()
        def func(x, y):
            return x.add(1.0) + y

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=True)
        ref = func(x, y)
        opt_func = torch._dynamo.optimize("eager")(func)

        x1 = torch.ones(4, requires_grad=True)
        res = opt_func(x1, y)
        self.assertTrue(same(ref, res))
        self.assertTrue(same(x, x1))

    def test_if_cond_nn_mod(self):
        class MockModule(torch.nn.Module):
            def __init__(self, output_relu=True):
                super().__init__()
                self.relu = torch.nn.ReLU() if output_relu else None

            def forward(self, x):
                x = torch.sin(x)
                if self.relu:
                    x = self.relu(x)
                return x

        model = MockModule()
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)

        x = torch.rand(4)
        ref = model(x)
        res = opt_model(x)
        self.assertTrue(same(ref, res))

        model = MockModule(output_relu=False)
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)

        x = torch.rand(4)
        ref = model(x)
        res = opt_model(x)
        self.assertTrue(same(ref, res))

    def test_if_cond_user_defined_object(self):
        # obj.__bool__ is not existed
        class A:  # noqa: B903
            def __init__(self, x):
                self.x = x

        # obj.__bool__ is function and returns bool type
        class B:
            def __init__(self, x):
                self.x = x

            def __bool__(self):
                return self.x > 0

        # obj.__bool__ is non-function
        class C:
            def __init__(self, x):
                self.x = x
                self.__bool__ = False

        def fn(x, obj):
            if not obj:
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        obj1 = A(0.5)
        obj2 = B(0.5)
        obj3 = B(-0.5)
        obj4 = C(0.5)
        for obj in [obj1, obj2, obj3, obj4, obj3, obj2]:
            ref = fn(x, obj)
            res = opt_fn(x, obj)
            self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 4)

    def test_if_cond_user_defined_object2(self):
        # obj.__bool__ is function and returns non-bool type
        class MyObj:
            def __init__(self, x):
                self.x = x

            def __bool__(self):
                self.x = 1
                return self.x

        def fn(a, obj):
            if not obj:
                return a + obj.x
            else:
                return a - obj.x

        x = torch.rand(4)
        obj = MyObj(0.5)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        try:
            opt_fn(x, obj)
            self.assertFalse(True)
        except TypeError as e:
            self.assertIn("__bool__ should return bool, returned int", str(e))

    def test_class_has_instancecheck_method(self):
        class A:
            pass

        class ExampleMeta(type):
            def __instancecheck__(cls, instance):
                return True

        class B(metaclass=ExampleMeta):
            pass

        def fn(x, obj):
            if isinstance(obj, B):
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        obj = A()
        ref = fn(x, obj)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x, obj)
        self.assertTrue(same(ref, res))

    def test_torch_cuda_is_available(self):
        def fn(x):
            if torch.cuda.is_available():
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "requires cudnn")
    def test_torch_cudnn_is_acceptable(self):
        def fn(x):
            if torch.backends.cudnn.is_acceptable(tensor=x):
                return x + 1
            return x

        x = torch.rand(4).cuda()
        ref = fn(x)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "requires cudnn")
    def test_torch_cudnn_is_acceptable_bad_inputs(self):
        def fn1(x):
            if torch.backends.cudnn.is_acceptable("invalid"):
                return x + 1
            return x

        def fn2(x):
            if torch.backends.cudnn.is_acceptable(x, 3.14):
                return x + 1
            return x

        with self.assertRaisesRegex(
            AssertionError, "Expect input to cudnn.is_acceptable to be a tensor"
        ):
            x1 = torch.rand(4).cuda()
            opt_fn1 = torch._dynamo.optimize("eager", nopython=True)(fn1)
            res1 = opt_fn1(x1)

        with self.assertRaisesRegex(
            AssertionError, "Expect 1 input to cudnn.is_acceptable"
        ):
            x2 = torch.rand(4).cuda()
            opt_fn2 = torch._dynamo.optimize("eager", nopython=True)(fn2)
            res = opt_fn2(x2)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_get_device(self):
        def fn(x, y):
            x = x + 1
            y = y + 1
            return x.get_device(), y.get_device()

        x = torch.rand(4, device="cuda")
        y = torch.rand(4, device="cpu")
        ref = fn(x, y)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_disable_flag(self):
        cnt = torch._dynamo.testing.CompileCounter()

        with patch.dict(os.environ, {"TORCH_COMPILE_DISABLE": "1"}):

            def fn(x, y):
                x = x + 1
                y = y + 1

            opt_fn = torch._dynamo.optimize(cnt)

        self.assertEqual(cnt.frame_count, 0)

    def test_is_compiling(self):
        def f():
            if torch._dynamo.is_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        opt_f = torch._dynamo.optimize("eager")(f)

        self.assertEqual(f(), torch.zeros(2, 2))
        self.assertEqual(opt_f(), torch.ones(2, 2))

    def test_guard_failure_fn(self):
        def fn(x, y, k):
            x = x + 1
            y = y + 1
            return x * y * k

        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([1.0, 1.0])

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        x2 = torch.tensor([0.5, 0.5, 1.0])
        y2 = torch.tensor([0.5, 0.5, 0.5])

        opt_fn(x, y, 3)
        opt_fn(x2, y2, 5)

        if (
            torch._dynamo.config.dynamic_shapes
            and not torch._dynamo.config.specialize_int
            and not torch._dynamo.config.assume_static_by_default
        ):
            # we didn't actually test guard_failure_fn here but whatever,
            # nice to see no guard failure on the test
            self.assertTrue(guard_failure is None)
        else:
            self.assertTrue(guard_failure is not None)
            if not torch._dynamo.config.dynamic_shapes:
                self.assertEqual(guard_failure[0], "k == 3")

    @patch.object(torch._dynamo.config, "dynamic_shapes", True)
    def test_guard_failure_fn_shape_control(self):
        def fn(x, y):
            if x.shape[0] < 3:
                if y.shape[0] < 3:
                    return x * y
                else:
                    return x + y
            else:
                return -1

        x = torch.randn([2, 2])
        y = torch.randn([2, 2])

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        x2 = torch.randn([5, 5])
        y2 = torch.randn([5, 5])

        opt_fn(x, y)
        opt_fn(x2, y2)

        self.assertTrue(guard_failure is not None)
        if torch._dynamo.config.assume_static_by_default:
            self.assertEqual(guard_failure[0], "x.size()[0] == 2")
        else:
            self.assertEqual(guard_failure[0], "x.size()[0] < 3")

    def test_guard_failure_fn2(self):
        def fn(x, y):
            x = x + 1
            y = y + 1
            return x * y

        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([1.0, 1.0])

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        x2 = torch.tensor([0.5, 0.5, 1.0])
        y2 = torch.tensor([0.5, 0.5, 0.5])

        opt_fn(x, y)
        opt_fn(x2, y2)

        if torch._dynamo.config.dynamic_shapes:
            if torch._dynamo.config.assume_static_by_default:
                self.assertEqual(
                    guard_failure[0],
                    "x.size()[0] == 2",
                )
            else:
                self.assertTrue(guard_failure is None)
        else:
            self.assertTrue(guard_failure is not None)
            self.assertEqual(
                guard_failure[0],
                "tensor 'x' size mismatch at index 0. expected 2, actual 3",
            )

    def test_guard_failure_fn_tensor_iter(self):
        def fn(x):
            for y in x:
                y.add_(1.0)
            return y

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        args1 = torch.randn(10, 10)
        out = fn(args1)
        opt_out = opt_fn(args1)
        self.assertTrue(same(out, opt_out))

        args2 = torch.randn(9, 10)
        out = fn(args2)
        opt_out = opt_fn(args2)
        self.assertTrue(same(out, opt_out))

        # guard is expected for both static and dynamic shapes
        self.assertTrue(guard_failure is not None)
        self.assertEqual(guard_failure[0], "len(x) == 10")

    def test_restore_graphstate(self):
        # This function does some guard accumulation,
        # and then rolls back due to control flow.
        # The idea is that if one were printing guards as they appear,
        # they would see this insert a guard that does not show up in the final set of
        # guards as we rolled back from it.
        def nested_fn(s):
            if x[0] < 10:
                return s * s
            return s

        def fn(x, y):
            x = x + 1
            y = nested_fn(y)
            y = y + 10
            return x * y

        all_guards = []

        def guard_export_print(guards):
            nonlocal all_guards
            all_guards.extend(guards)

        opt_fn = torch._dynamo.optimize("eager", guard_export_fn=guard_export_print)(fn)

        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([1.0, 1.0])
        opt_fn(x, y)

        if torch._dynamo.config.dynamic_shapes:
            self.assertEqual(len(all_guards), 13)
        else:
            self.assertEqual(len(all_guards), 9)
        for guard in all_guards:
            # This guard was created
            self.assertTrue(guard.name != "nested_fn.__closure__[0].cell_contents")

    # Note - here be mild dragons.
    # This test relies a ton on internal implementation. Future refactor efforts
    # are welcome to delete it if necessary, rewriting this test constantly is a chore, not
    # a feature. We kept it around with some amount of saddness, as it was extremely useful in debugging.
    def test_restore_graphstate_internals(self):
        def fn(x, y):
            x = x + 1
            y = y + 1
            return x * y

        _, guards = torch._dynamo.export(
            fn, torch.tensor([0.25, 0.25]), torch.tensor([0.25, 0.25])
        )
        # Dummy ctor
        graph = OutputGraph(
            f_globals={},
            code_options={},
            compiler_fn=None,
            root_tx=None,
            export=False,
        )
        # Contrived property so as not to have it be None
        graph.nn_modules = {}
        # Contrived generation timestamp
        graph.timestamp = 4
        # Contrived guards
        graph.tracing_context.guards_context.dynamo_guards = guards

        # Save the state
        state = graph.copy_graphstate()
        # Saving increments the generation
        self.assertEqual(graph.timestamp, 5)

        # Assure that the saved state is valid
        self.assertEqual(state.timestamp, 4)

        # Ensure that the guards reflect the expected state
        self.assertEqual(graph.tracing_context.guards_context.dynamo_guards, guards)
        self.assertEqual(graph.guards, guards)

        # Mess around with the state
        graph.tracing_context.guards_context.dynamo_guards = set()
        self.assertEqual(graph.guards, set())

        # Restore the state
        graph.restore_graphstate(state)

        # Make sure it restored correctly
        self.assertEqual(graph.timestamp, 4)
        self.assertEqual(graph.guards, guards)
        self.assertEqual(graph.tracing_context.guards_context.dynamo_guards, guards)

    def test_call_parent_non_class_methods_from_child(self):
        class A:
            def add(self, x):
                return x + 10

            def mul(self, x):
                return x * 0.1

        class B(A):
            def add(self, x):
                return x + 20

            def mul(self, x):
                return x * 0.2

        class C(B):
            def add(self, x):
                y = A.add(self, x)
                z = B.mul(self, y)
                return z + 30

        x = torch.rand(4)
        fn = C().add
        ref = fn(x)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_builder_for_class_with_metaclass(self):
        class ExampleMeta(type):
            pass

        class MyClass(metaclass=ExampleMeta):
            pass

        def fn(x, y):
            if isinstance(y, MyClass):
                return x + 1
            else:
                return x - 1

        x = torch.rand([4, 4])
        y = MyClass()
        ref = fn(x, y)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_torch_package_working_with_trace(self):
        # from torch._dynamo.test_case import run_tests

        inputs = [torch.randn([2, 2]), torch.randn([2, 2])]

        optimized_model = torch._dynamo.optimize(backend="eager")(
            MyPickledModule(torch.randn([2, 2]))
        )
        from torch import package

        path = "/tmp/MyPickledModule.pt"
        package_name = "MyPickledModule"
        resource_name = "MyPickledModule.pkl"

        model = MyPickledModule(torch.randn([2, 2]))

        with package.PackageExporter(path) as exp:
            exp.extern("**")
            exp.save_pickle(package_name, resource_name, model)

        imp = package.PackageImporter(path)
        loaded_model = imp.load_pickle(package_name, resource_name)

        optimized_loaded_model = torch._dynamo.optimize("eager")(loaded_model)(*inputs)

    def test_shape_and_tuple_equality(self):
        def fn(x, y, t):
            z = x * y
            if x.size() == t:
                return z.cos()
            return z.sin()

        torch._dynamo.optimize("eager", nopython=True)(fn)(
            torch.randn([4, 4]), torch.randn([4, 4]), (4, 4)
        )

    # specifically test for tensor.attribute -> torch.something()
    def test_real_imag_tensor_attribute(self):
        def fn(x, y):
            a = x.real
            b = x.imag
            return torch.mul(torch.add(a, y), b)

        x_real = torch.rand((4, 4))
        x_imag = torch.rand((4, 4))
        x = torch.complex(x_real, x_imag)
        y = torch.rand((4, 4))

        ref = fn(x, y)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_T_tensor_attribute(self):
        def fn(x, y):
            a = x.T
            return torch.add(a, y)

        x = torch.rand((4, 4))
        y = torch.rand((4, 4))

        ref = fn(x, y)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_recursive_tensor_attribute(self):
        def fn(x, y):
            a = x.real.T
            b = x.imag
            return torch.mul(torch.add(a, y), b)

        x_real = torch.rand((4, 4))
        x_imag = torch.rand((4, 4))
        x = torch.complex(x_real, x_imag)
        y = torch.rand((4, 4))

        ref = fn(x, y)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_tagging_tensors_simple(self):
        def foo(x, y):
            return x * y, x, y

        a = torch.randn([3, 3])
        a.tag = "a"
        a.frog = "ribbity ribbit"
        b = torch.randn([3, 3])
        b.tag = "b"
        b.frog = "ribbit"

        exported = torch._dynamo.export(foo, a, b)
        out_graph = exported[0]

        nodes = list(out_graph.graph.nodes)
        placeholders = [node for node in nodes if node.op == "placeholder"]
        all_tags = []
        all_frogs = []
        for placeholder in placeholders:
            if "tensor_dict" in placeholder.meta:
                all_tags.append(placeholder.meta["tensor_dict"]["tag"])
                all_frogs.append(placeholder.meta["tensor_dict"]["frog"])

        self.assertEqual(all_tags, ["a", "b"])
        self.assertEqual(all_frogs, ["ribbity ribbit", "ribbit"])

    def test_tagging_tensors_mix_used_unused_structure(self):
        def pre_attention_state_ops(input, mems, state):
            lc_key = state[0]
            lc_val = state[1]
            bar = []
            for i in range(0, 4):
                bar2 = []
                for j in range(0, 3):
                    bar2.append(
                        lc_key + lc_val + torch.tensor([0.1, 0.25, 0.4, 0.5, 0.1])
                    )
                bar.append(bar2)

            return bar

        mems = torch.tensor([[[1.8364, 0.2724, -1.4917, -0.4367, 0.8640]]])
        state = [
            torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
            torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
        ]
        i = torch.tensor(
            [
                [0.0313, -0.1487, -0.3846, -0.5321],
                [-1.7073, 1.3331, -0.0890, -1.4935],
                [-0.8314, -0.1862, -0.5935, 1.5232],
            ]
        )

        mems.tag = "MEMS"
        i.tag = "FOO"
        state[0].tag = "STATE_0"
        state[1].tag = "HMMM"

        exported = torch._dynamo.export(pre_attention_state_ops, i, mems, state)
        out_graph = exported[0]

        nodes = list(out_graph.graph.nodes)
        placeholders = [node for node in nodes if node.op == "placeholder"]
        all_tags = []
        for placeholder in placeholders:
            if "tensor_dict" in placeholder.meta:
                all_tags.append(placeholder.meta["tensor_dict"]["tag"])

        self.assertEqual(all_tags, ["STATE_0", "HMMM"])

    def test_get_custom_tensor_attribute(self):
        def fn(x):
            return x.custom_attr * x

        x = torch.rand((2, 2))
        x.custom_attr = 3.14
        ref = fn(x)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_set_custom_tensor_attribute(self):
        def fn(x):
            x.custom_attr = 3.14
            return x.custom_attr * x

        x = torch.rand((2, 2))
        ref = fn(x)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(sys.version_info < (3, 11), "requires Python 3.11+")
    def test_py311_jump_offset(self):
        new_inst = bytecode_transformation.create_instruction
        consts = (None, 1, 2, 3, 4)

        def create_test_code(jump_opname, target_idx):
            targets = [
                new_inst("LOAD_CONST", 1),
                new_inst("LOAD_CONST", 3),
            ]
            jump_to_target_inst = new_inst(jump_opname, target=targets[target_idx])
            """
            pseudocode of generated bytecode:
            def test_py311_fn():
                goto target1
            target0:
                return 1
            target1:
                goto [target0/target2] (via fwd or bwd jump)
                return 2
            target2:
                return 3
                return 4
            """
            # test with LOAD_GLOBAL since it has a different instruction size
            insts = [
                new_inst("RESUME", 0),
                new_inst("JUMP_FORWARD", target=jump_to_target_inst),
                targets[0],
                new_inst("LOAD_GLOBAL", argval="print"),
                new_inst("POP_TOP"),
                new_inst("RETURN_VALUE"),
                jump_to_target_inst,
                new_inst("LOAD_CONST", 2),
                new_inst("LOAD_GLOBAL", argval="print"),
                new_inst("POP_TOP"),
                new_inst("RETURN_VALUE"),
                targets[1],
                new_inst("RETURN_VALUE"),
                new_inst("LOAD_CONST", 4),
                new_inst("RETURN_VALUE"),
            ]
            code_options = collections.OrderedDict(
                [
                    ("co_argcount", 0),
                    ("co_posonlyargcount", 0),
                    ("co_kwonlyargcount", 0),
                    ("co_nlocals", 0),
                    ("co_stacksize", 2),
                    ("co_flags", 3),
                    ("co_code", b""),
                    ("co_consts", consts),
                    ("co_names", ("print",)),
                    ("co_varnames", ()),
                    ("co_filename", __file__),
                    ("co_name", "test_py311_fn"),
                    ("co_qualname", "test_py311_fn"),
                    ("co_firstlineno", 1),
                    ("co_linetable", b""),
                    ("co_exceptiontable", b""),
                    ("co_freevars", ()),
                    ("co_cellvars", ()),
                ]
            )
            return bytecode_transformation.clean_and_assemble_instructions(
                insts,
                list(code_options.keys()),
                code_options,
            )

        # format: jump_opname, target_idx, expected forward jump, expected return value
        test_args = (
            ("JUMP_FORWARD", 0, False, 1),
            ("JUMP_FORWARD", 1, True, 3),
            ("JUMP_BACKWARD", 0, False, 1),
            ("JUMP_BACKWARD", 1, True, 3),
        )

        for test in test_args:
            insts, code = create_test_code(test[0], test[1])
            # check if offset of latest jump instruction is forward/backward
            for inst in reversed(insts):
                if inst.opname.startswith("JUMP"):
                    if test[2]:
                        self.assertIn("FORWARD", inst.opname)
                    else:
                        self.assertIn("BACKWARD", inst.opname)
                    break
            # run the code and check result

            def dummy_fn():
                pass

            dummy_fn.__code__ = code
            self.assertEqual(dummy_fn(), test[3])

            # TODO should also pass the code object back into dynamo again, but
            # dynamo is not enabled for Python 3.11 yet.

    def test_ordered_dict_alias_reconstruct(self):
        od = collections.OrderedDict

        def fn():
            d1 = dict()
            d1["a"] = 1
            d2 = od(d1)
            d2["b"] = 2
            torch._dynamo.graph_break()
            if isinstance(d2, od):
                return d2["a"] + d2["b"]
            else:
                return 0

        dis.dis(fn)
        self.assertEqual(torch._dynamo.optimize("eager")(fn)(), 3)

    @torch._dynamo.config.patch(dynamic_shapes=True)
    def test_raise_guard_full_constraint(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] == 3:
                return x.sin()
            return x.cos()

        torch._dynamo.optimize("eager")(my_dyn_fn)(y)
        torch._dynamo.mark_dynamic(y, 0)

        torch._dynamo.reset()
        with self.assertRaises(
            torch._dynamo.exc.InternalTorchDynamoError,
        ):
            torch._dynamo.optimize("eager")(my_dyn_fn)(y)

    @torch._dynamo.config.patch(dynamic_shapes=True)
    def test_no_raise_guard_partial_constraint(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] > 3:
                return x.sin()
            return x.cos()

        torch._dynamo.optimize("eager")(my_dyn_fn)(y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        torch._dynamo.optimize("eager")(my_dyn_fn)(y)

    @torch._dynamo.config.patch(dynamic_shapes=True)
    def test_no_raise_guard_partial_constraint_across_break(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            z = x * y

            torch._dynamo.graph_break()
            if z.shape[0] > 2:
                return z.cos()

            return x.cos()

        torch._dynamo.optimize("eager")(my_dyn_fn)(y, y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        torch._dynamo.optimize("eager")(my_dyn_fn)(y, y)

    # Sadly, this does not throw - we do not prop correctly across the graph break
    @unittest.expectedFailure
    @torch._dynamo.config.patch(dynamic_shapes=True)
    def test_raise_guard_partial_constraint_across_break(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            z = x * y

            torch._dynamo.graph_break()
            if z.shape[0] == 3:
                return z.cos()

            return x.cos()

        torch._dynamo.optimize("eager")(my_dyn_fn)(y, y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            Exception,
        ):
            torch._dynamo.optimize("eager")(my_dyn_fn)(y, y)

    @torch._dynamo.config.patch(dynamic_shapes=True)
    def test_raise_guard_partial_constraint_no_graph_break(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            z = x * y

            if z.shape[0] == 3:
                return z.cos()

            return x.cos()

        torch._dynamo.optimize("eager")(my_dyn_fn)(y, y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        with self.assertRaises(
            torch._dynamo.exc.InternalTorchDynamoError,
        ):
            torch._dynamo.optimize("eager")(my_dyn_fn)(y, y)

    def test_cannot_trace_mark_dynamic(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            torch._dynamo.mark_dynamic(x, 0)
            return x * x

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            torch._dynamo.optimize("eager")(my_dyn_fn)(y)

    def test_cannot_trace_mark_dynamic_safe_unreached(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] == 3:
                return x
            print("Running", torch._dynamo.mark_dynamic(x, 0))
            return x * x

        torch._dynamo.optimize("eager")(my_dyn_fn)(y)

    @torch._dynamo.config.patch(dynamic_shapes=False)
    def test_no_dynamic_shapes_mark_dynamic_illegal(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] > 3:
                return x.sin()
            return x.cos()

        torch._dynamo.optimize("eager")(my_dyn_fn)(y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            AssertionError,
            "mark_dynamic usage with dynamic_shapes=False is not yet supported",
        ):
            torch._dynamo.optimize("eager")(my_dyn_fn)(y)

    @torch._dynamo.config.patch(dynamic_shapes=False)
    def test_parameter_mark_dynamic_illegal(self):
        y = torch.nn.Parameter(torch.tensor([0.25, 0.25]))
        x = torch.tensor([0.5, 0.5])

        class encoder(torch.nn.Module):
            def __init__(self, y):
                super().__init__()
                self.register_parameter("param", y)

            @torch._dynamo.disable
            def helper(self, x, y):
                return x * y

            def forward(self, a, *args):
                x = a + a
                return self.helper(x, self.param)

        e = encoder(y)
        torch._dynamo.optimize("eager")(e)(x)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        e = encoder(y)
        with self.assertRaisesRegex(
            AssertionError,
            "mark_dynamic on parameter, parameters are always static today",
        ):
            torch._dynamo.optimize("eager")(e)(x)

    @torch._dynamo.config.patch(dynamic_shapes=True)
    def test_py_guards_mark_dynamic(self):
        x = torch.randn([3, 3, 3])

        def my_dyn_fn(a):
            if a.shape[0] > 2:
                return a.cos()
            return a.sin()

        torch._dynamo.mark_dynamic(x, 0)
        counter = CompileCounter()
        # Run with dynamic
        torch._dynamo.optimize(counter)(my_dyn_fn)(x)
        self.assertEqual(counter.frame_count, 1)
        delattr(x, "_dynamo_dynamic_indices")

        torch._dynamo.optimize(counter)(my_dyn_fn)(x)
        # Run without dynamic, no recompile
        self.assertEqual(counter.frame_count, 1)

        # Mark a new dim, 1, as dynamic
        torch._dynamo.mark_dynamic(x, 1)
        torch._dynamo.optimize(counter)(my_dyn_fn)(x)
        # Recompile triggered because we marked a new dym as dynamic
        self.assertEqual(counter.frame_count, 2)

        # Mark an existing dim, 1, as dynamic
        torch._dynamo.mark_dynamic(x, 1)
        torch._dynamo.optimize(counter)(my_dyn_fn)(x)
        # No Recompile triggered because we marked an existing dym as dynamic
        self.assertEqual(counter.frame_count, 2)

        # Reset
        torch._dynamo.reset()
        # Reset counter
        counter = CompileCounter()
        # Clear dynamic
        delattr(x, "_dynamo_dynamic_indices")

        # Run with dynamic 1
        torch._dynamo.mark_dynamic(x, 1)
        torch._dynamo.optimize(counter)(my_dyn_fn)(x)
        self.assertEqual(counter.frame_count, 1)

        # Clear dynamic
        delattr(x, "_dynamo_dynamic_indices")
        # Run with dynamic 0, not subset
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.optimize(counter)(my_dyn_fn)(x)
        self.assertEqual(counter.frame_count, 2)

        # Clear dynamic
        delattr(x, "_dynamo_dynamic_indices")
        # Run with dynamic 0, 1, 2, not subset
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        torch._dynamo.mark_dynamic(x, 2)
        torch._dynamo.optimize(counter)(my_dyn_fn)(x)
        self.assertEqual(counter.frame_count, 3)

        # Clear dynamic
        delattr(x, "_dynamo_dynamic_indices")
        # Run with dynamic 0, 2, subset!
        torch._dynamo.mark_dynamic(x, 2)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.optimize(counter)(my_dyn_fn)(x)
        self.assertEqual(counter.frame_count, 3)

    def test_torch_compile_ctx_on_forward_and_training_step(self):
        class MyModel(torch.nn.Module):
            def forward(self):
                ...

            def training_step(self):
                self()

        model = MyModel()
        compiled_model = torch.compile(model)

        model.forward = compiled_model.dynamo_ctx(model.forward)
        model.training_step = compiled_model.dynamo_ctx(model.training_step)

        model.training_step()

    def test_torch_guards_stack_frame_register(self):
        y = torch.nn.Parameter(torch.tensor([0.25, 0.25]))
        x = torch.tensor([0.5, 0.5])

        class encoder(torch.nn.Module):
            def __init__(self, y):
                super().__init__()
                self.register_parameter("param", y)

            @torch._dynamo.disable
            def helper(self, x, y):
                return x * y

            def forward(self, a, *args):
                x = a + a
                return self.helper(x, self.param)

        e = encoder(y)

        seen_frames = []
        import contextlib

        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            torch._dynamo.optimize("eager")(e)(x)

        self.assertEqual(len(seen_frames), 1)
        self.assertEqual(seen_frames[0].line, "def forward(self, a, *args):")

    def test_torch_guards_stack_frame_register_inlining(self):
        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([0.75, 0.75, 0.75, 0.75])
        z = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        def uwu_inline_me(x, y, z):
            r = torch.cat((x, x)) + y
            r2 = torch.cat((y, y)) + z
            return r, r2

        def fn(x, y, z):
            r, r2 = uwu_inline_me(x, y, z)
            return torch.mul(r, r), torch.mul(r2, r2)

        seen_frames = []
        import contextlib

        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            torch._dynamo.optimize("eager")(fn)(x, y, z)

        self.assertEqual(len(seen_frames), 2)
        self.assertEqual(seen_frames[0].name, "fn")
        self.assertEqual(seen_frames[1].line, "def uwu_inline_me(x, y, z):")


class CustomFunc1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        return foo + foo

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class CustomFunc2(torch.autograd.Function):
    # the forward function can be staticmethod or classmethod
    @classmethod
    def forward(cls, ctx, foo):
        return foo + foo

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class CustomFunc3(torch.autograd.Function):
    # Test there is graph break in forward function
    @staticmethod
    def forward(ctx, foo):
        result = foo + foo
        torch._dynamo.graph_break()
        result = result + foo
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * math.sqrt(result.numel())


class Module1(torch.nn.Module):
    def forward(self, foo):
        return CustomFunc1().apply(foo)


class Module2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = CustomFunc1.apply

    def forward(self, foo):
        return self.fn(foo)


class Module3(torch.nn.Module):
    def forward(self, foo):
        return CustomFunc2().apply(foo)


class Module4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = CustomFunc2.apply

    def forward(self, foo):
        return self.fn(foo)


class Module5(torch.nn.Module):
    def forward(self, foo):
        return CustomFunc3().apply(foo)


class Module6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = CustomFunc3.apply

    def forward(self, foo):
        return self.fn(foo)


class TestTracer(JitTestCase):
    def test_jit_save(self):
        def fn():
            class Foo(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.a = 3

                @torch.jit.export
                def __getstate__(self):
                    return (3, self.training)

                @torch.jit.export
                def __setstate__(self, state):
                    self.a = state[0]
                    self.training = state[1]

                def forward(self, x):
                    return x + self.a

            f = Foo()

            return torch.jit.trace(f, (torch.rand(3, 4),))

        fn()
        opt_fn = torch._dynamo.optimize("eager")(fn)
        opt_fn()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
