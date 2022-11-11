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
import weakref
from unittest.mock import patch

import numpy as np
import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch.onnx.operators
from torch._dynamo import bytecode_transformation, graph_break
from torch._dynamo.testing import (
    CompileCounter,
    requires_static_shapes,
    same,
    unsupported,
)
from torch.testing._internal.common_utils import freeze_rng_state
from torch.testing._internal.jit_utils import JitTestCase

mytuple = collections.namedtuple("mytuple", ["a", "b", "ab"])


def my_custom_function(x):
    return x + 1


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

    def test_numel(self):
        def fn(a):
            return a + a.numel() + torch.numel(a)

        return torch._dynamo.testing.standard_test(
            self, fn=fn, nargs=1, expected_ops=2, expected_ops_dynamic=4
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
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(cnts.op_count, 0)

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
        # wrap torch.profiler.* as ProfilerContextWrapperVariable and do nothing
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
        # wrap torch.autograd.profiler.* as ProfilerContextWrapperVariable and do nothing
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

    def test_const_dict_variable_python_type(self):
        from torch._dynamo.variables import ConstDictVariable

        d1 = {"a": 10, "b": 20}
        d2 = collections.OrderedDict([("x", 12), ("y", 22)])
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
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        incorrect_ref_id = id(m) + 1
        opt_m = torch._dynamo.optimize(cnts, nopython=True)(m)
        opt_m(data, incorrect_ref_id)
        self.assertEqual(cnts.op_count, 1)

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

    @patch.object(torch._dynamo.config, "fake_tensor_propagation", True)
    def test_unsupported_fake_tensor(self):
        def f(x):
            return torch.quantize_per_tensor(x, 0.1, 10, torch.quint8)

        x = torch.randn(2, 2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f = torch._dynamo.optimize(cnts)(f)
        opt_f(x)
        self.assertEqual(cnts.op_count, 0)

        torch._dynamo.reset()
        with patch.object(torch._dynamo.config, "fake_tensor_propagation", False):
            opt_f = torch._dynamo.optimize_assert(
                torch._dynamo.testing.CompileCounter()
            )(f)
            opt_f(x)

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
                from torch.nn import functional as F

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
        from functorch.experimental.cond import cond

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

    def test_cond_nested(self):
        from functorch.experimental.cond import cond

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

    @patch.object(torch._dynamo.config, "fake_tensor_propagation", False)
    def test_cond_nested_fake_tensor_off(self):
        from functorch.experimental.cond import cond

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
        self.assertTrue(cc.frame_count, 1)

    @patch.object(torch._dynamo.config, "fake_tensor_propagation", False)
    def test_cond_export(self):
        from functorch.experimental.cond import cond

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

    @patch.object(torch._dynamo.config, "fake_tensor_propagation", False)
    def test_cond_export_single_arg(self):
        from functorch.experimental.cond import cond

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

                with torch.autocast(device_type="cuda"):
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
        m1 = Module1()

        @torch._dynamo.optimize("eager", nopython=True)
        def f1():
            return m1(torch.ones(2, 3))

        self.assertTrue(torch.allclose(f1(), torch.tensor([2.0])))

        m2 = Module2()

        @torch._dynamo.optimize("eager", nopython=True)
        def f2():
            return m2(torch.ones(2, 3))

        self.assertTrue(torch.allclose(f2(), torch.tensor([2.0])))

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


class CustomFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        return foo + foo

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Module1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, foo):
        return CustomFunc().apply(foo)


class Module2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = CustomFunc.apply

    def forward(self, foo):
        return self.fn(foo)


class TestTracer(JitTestCase):
    def test_jit_save(self):
        def fn():
            class Foo(torch.nn.Module):
                def __init__(self):
                    super(Foo, self).__init__()
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
