# Owner(s): ["module: dynamo"]
# ruff: noqa: F841
import abc
import builtins
import collections
import collections.abc
import copy
import dataclasses
import dis
import enum
import functools
import gc
import importlib
import itertools
import json
import logging
import math
import operator
import os
import pickle
import random
import re
import sys
import tempfile
import threading
import traceback
import types
import typing
import unittest
import unittest.mock as mock
import warnings
import weakref
from unittest.mock import patch

import numpy as np

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils._pytree as python_pytree
import torch.utils.cpp_extension
from torch import Tensor
from torch._C import FileCheck
from torch._dynamo import allow_in_graph
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
from torch._dynamo.exc import Unsupported
from torch._dynamo.source import ConstantSource, GetItemSource, LocalSource
from torch._dynamo.testing import (
    CompileCounter,
    CompileCounterWithBackend,
    expectedFailureDynamic,
    same,
    skipIfNotPy311,
    unsupported,
)
from torch._dynamo.utils import call_size, counters, ifdynstaticdefault
from torch._dynamo.variables import builder
from torch._inductor.codecache import WritableTempFile
from torch._inductor.utils import fresh_cache, run_and_get_code
from torch.ao.quantization import MinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.fx.experimental.recording import NotEqualError, replay_shape_env_events
from torch.fx.experimental.symbolic_shapes import (
    _constrain_range_for_size,
    constrain_range,
    constrain_unify,
    ConstraintViolationError,
    expect_true,
    guard_or_false,
    guard_size_oblivious,
    ShapeEnv,
)
from torch.nn import functional as F
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    SM80OrLater,
    TEST_CUDA,
    TEST_MULTIGPU,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import (
    sample_inputs_take_along_dim,
)
from torch.testing._internal.common_utils import (
    freeze_rng_state,
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    scoped_load_inline,
    set_default_dtype,
    skipIfHpu,
    skipIfNNModuleInlined,
    skipIfWindows,
    subtest,
    TEST_HPU,
    TEST_XPU,
    wrapDeterministicFlagAPITest,
)
from torch.testing._internal.jit_utils import JitTestCase


pytree_modules = {
    "python": python_pytree,
}
if python_pytree._cxx_pytree_dynamo_traceable:
    import torch.utils._cxx_pytree as cxx_pytree

    pytree_modules["cxx"] = cxx_pytree
    pytree_modules["native_optree"] = cxx_pytree.optree
else:
    cxx_pytree = None

parametrize_pytree_module = parametrize(
    "pytree",
    [subtest(module, name=name) for name, module in pytree_modules.items()],
)

MyTuple = collections.namedtuple("MyTuple", ["a", "b", "ab"])
T = typing.TypeVar("T")


# Defined in CPython's Include/object.h
TPFLAGS_MAPPING = 1 << 6

GLOBAL_INT = 1


# Specializes a test to run only if translation validation is set.
def onlyIfTranslationValidation(fn: typing.Callable) -> typing.Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import torch.fx.experimental.validator

        if torch.fx.experimental.validator.translation_validation_enabled():
            return fn(*args, **kwargs)
        raise unittest.SkipTest(f"only works when TV is True.")

    return wrapper


class MyPickledModule(torch.nn.Module):
    def __init__(self, z):
        super().__init__()
        self.z = z

    def forward(self, x, y):
        return x * x * x + y + self.z


# These are used for test_{cond/map}_with_quantization
default_symmetric_fake_quant = FakeQuantize.with_args(
    observer=MinMaxObserver, qscheme=torch.per_tensor_symmetric, dtype=torch.quint8
)
default_weight_symmetric_fake_quant = FakeQuantize.with_args(
    observer=MinMaxObserver, qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
)
uniform_qconfig_8bit = QConfig(
    activation=default_symmetric_fake_quant,
    weight=default_weight_symmetric_fake_quant.with_args,
)
qconfig_dict = {"object_type": [(torch.nn.Linear, uniform_qconfig_8bit)]}


def closure_adder(val):
    def inner(x):
        return torch.sin(x + val)

    return inner


class UserDefineSetAttr:
    setup = False

    def __setattr__(self, key, value):
        assert torch.compiler.is_dynamo_compiling() or UserDefineSetAttr.setup
        super().__setattr__(f"pfx_{key}", value)

    def __getattr__(self, key, c=1):
        assert torch.compiler.is_dynamo_compiling() or UserDefineSetAttr.setup
        # c is added to force a guard on __defaults__ and checks the source for __getattr__
        if c:
            return self.__dict__[f"pfx_{key}"]
        else:
            return None


class MiscTests(torch._inductor.test_case.TestCase):
    def test_get_cache_entry(self):
        def f(x):
            return x + 1

        torch.compile(f)(torch.randn(5, 5, 5))
        entries = _debug_get_cache_entry_list(f)
        self.assertTrue(len(entries) > 0)

        def g(x):
            return x + 2

        entries = _debug_get_cache_entry_list(g)
        self.assertTrue(len(entries) == 0)

        try:
            _debug_get_cache_entry_list(1)
        except TypeError as e:
            self.assertIn("expected a code object!", str(e))

        # test get cache entry on skipped code object
        def h(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 1

        torch.compile(h)(torch.randn(3, 3))

        entries = _debug_get_cache_entry_list(torch._dynamo.graph_break)
        self.assertEqual(len(entries), 0)

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

    def test_dynamo_inside_custom_op(self):
        cnt = torch._dynamo.testing.InductorAndRecordGraphs()
        cnt1 = torch._dynamo.testing.InductorAndRecordGraphs()

        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor x) -> Tensor")

            def inner(x):
                return x.sin().cos()

            def foo_impl(x):
                return torch.compile(inner, fullgraph=True, dynamic=True, backend=cnt)(
                    x
                )

            m.impl("foo", foo_impl, "CompositeExplicitAutograd")

            @torch.compile(fullgraph=True, dynamic=True, backend=cnt1)
            def f(x):
                return torch.ops.mylib.foo.default(x)

            x = torch.randn(3)
            res = f(x)
            res1 = f(x)
            res2 = f(x)
            expected = x.sin().cos()
            self.assertEqual(res, expected)
            self.assertEqual(res1, expected)
            self.assertEqual(res2, expected)
            self.assertTrue(len(cnt.inductor_graphs), 1)
            self.assertTrue(len(cnt1.inductor_graphs), 1)
            self.assertExpectedInline(
                str(cnt.inductor_graphs[0].graph).strip(),
                """\
graph():
    %arg0_1 : [num_users=0] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%arg1_1,), kwargs = {})
    %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%sin,), kwargs = {})
    return (cos,)""",
            )
            self.assertExpectedInline(
                str(cnt1.inductor_graphs[0].graph).strip(),
                """\
graph():
    %arg0_1 : [num_users=0] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %foo : [num_users=1] = call_function[target=torch.ops.mylib.foo.default](args = (%arg1_1,), kwargs = {})
    return (foo,)""",
            )

    @torch._dynamo.config.patch(accumulated_recompile_limit=1)
    def test_dynamo_disabled_in_custom_op_kernels(self):
        counters.clear()

        @torch.library.custom_op("mylib::foo9", mutates_args={})
        def foo(x: torch.Tensor) -> torch.Tensor:
            torch._dynamo.graph_break()
            return x.clone()

        foo.register_fake(torch.clone)

        @torch.compile(backend="eager")
        def f(x):
            return foo._opoverload(x)

        x = torch.randn(2)
        f(x)
        x = torch.randn(3)
        # Recompile hits the cache size limit, which will cause Dynamo to
        # recurse into the frames. The only frame is the implementation
        # of foo. If Dynamo was not turned off correctly, then
        # we'll see a graph break
        f(x)
        self.assertEqual(len(counters["graph_break"]), 0)

        counters.clear()

        called = 0

        # test register_kernel
        @foo.register_kernel("cpu")
        def _(x):
            nonlocal called
            called += 1
            torch._dynamo.graph_break()
            return x.clone()

        f(x)
        self.assertEqual(called, 1)
        self.assertEqual(len(counters["graph_break"]), 0)

        # test torch.library.register_kernel
        counters.clear()
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo2(Tensor x) -> Tensor")

            @torch.library.register_fake("mylib::foo2", lib=m)
            def _(x):
                return x.clone()

            @torch.library.register_kernel("mylib::foo2", "cpu", lib=m)
            def _(x):
                torch._dynamo.graph_break()
                return x.clone()

            @torch.compile(backend="eager")
            def g(x):
                return torch.ops.mylib.foo2.default(x)

            x = torch.randn(2)
            g(x)  # compiles
            x = torch.randn(3)
            g(x)  # dynamo falls back on the outermost frame
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_invalid_args_builtin(self):
        @torch.compile(backend="eager")
        def fn(x):
            x = x.sin()
            if isinstance(x, torch.Tensor, invalid=True):
                x = x.sin()
            return x

        with self.assertRaises(TypeError):
            fn(torch.randn(16))

    def test_scalar_device_movement(self):
        if not torch._dynamo.config.assume_static_by_default:
            self.skipTest("Doesn't work with symints")

        def add_fn(a, b, out):
            res = torch.add(a, b, out=out)
            return res

        res = add_fn(2, 3, torch.tensor(0.0))
        add_fn = torch.compile(add_fn, backend="eager", fullgraph=True)
        res_compiled = add_fn(2, 3, torch.tensor(0.0))
        self.assertEqual(res, res_compiled)

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
        opt_fn = torch.compile(fn, backend=counter)
        self.assertRaises(AssertionError, lambda: opt_fn(a, b, c, AssertionError))
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 3)

    def test_module_not_callable(self):
        def fn(x):
            return torch.fft(x)

        counter = CompileCounter()
        a = torch.randn(10, 10)
        opt_fn = torch.compile(fn, backend=counter)
        self.assertRaisesRegex(
            TypeError, "'module' object is not callable", lambda: opt_fn(a)
        )

    def test_inplace(self):
        def inplace1(a, b):
            o = torch.empty((10, 10))
            o.copy_(a)
            o -= b
            return o

        torch._dynamo.testing.standard_test(self, inplace1, 2, expected_ops=3)

    def test_inplace_desugaring(self):
        def inplace_on_literals(y):
            x0 = 1
            x0 += y
            x1 = 1
            x1 -= y
            return x0, x1

        torch._dynamo.testing.standard_test(
            self, inplace_on_literals, 1, expected_ops=2
        )

    def test_unpack4(self):
        def unpack4(a, b):
            a = a[:5, :]
            b = b[:5, :]
            x, y = a.size()
            o = torch.empty((x, y))
            o.copy_(a / b)
            return o

        torch._dynamo.testing.standard_test(
            self,
            unpack4,
            2,
            expected_ops=5,
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
            self,
            unpack5,
            2,
            expected_ops=5,
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
            self, fn, 1, expected_ops=1, expected_ops_dynamic=ifdynstaticdefault(1, 9)
        )

    @torch._dynamo.config.patch(only_allow_pt2_compliant_ops=True)
    def test_pt2_compliant_ops_are_allowed(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::bar",
                "(Tensor x) -> Tensor",
                lib=lib,
                tags=(torch.Tag.pt2_compliant_tag,),
            )
            torch.library.impl(
                "mylib::bar", "CompositeImplicitAutograd", torch.sin, lib=lib
            )
            assert torch.Tag.pt2_compliant_tag in torch.ops.mylib.bar.default.tags

            def f(x):
                return torch.ops.mylib.bar(x)

            overload = torch.ops.mylib.bar.default

            def g(x):
                return overload(x)

            x = torch.randn(3)

            counts = torch._dynamo.testing.CompileCounter()
            optimized_f = torch.compile(f, backend=counts, fullgraph=True)
            _ = optimized_f(x)

            optimized_g = torch.compile(f, backend=counts, fullgraph=True)
            _ = optimized_g(x)

    @torch._dynamo.config.patch(only_allow_pt2_compliant_ops=True)
    def test_non_pt2_compliant_ops_graph_break(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define("mylib::bar2", "(Tensor x) -> Tensor", lib=lib)
            torch.library.impl(
                "mylib::bar2", "CompositeImplicitAutograd", torch.sin, lib=lib
            )
            assert torch.Tag.pt2_compliant_tag not in torch.ops.mylib.bar2.default.tags

            def f(x):
                return torch.ops.mylib.bar2(x)

            overload = torch.ops.mylib.bar2.default

            def g(x):
                return overload(x)

            x = torch.randn(3)

            counts = torch._dynamo.testing.CompileCounter()
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "not PT2 compliant"
            ):
                optimized_f = torch.compile(f, backend=counts, fullgraph=True)
                y = optimized_f(x)

            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "not PT2 compliant"
            ):
                optimized_g = torch.compile(f, backend=counts, fullgraph=True)
                y = optimized_g(x)

    @torch._dynamo.config.patch(only_allow_pt2_compliant_ops=True)
    def test_pt2_compliant_overload(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::bar3.tensor",
                "(Tensor x) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )
            torch.library.define(
                "mylib::bar3.int", "(Tensor x, int dim) -> Tensor", lib=lib
            )

            torch.library.impl(
                "mylib::bar3.tensor",
                "CompositeImplicitAutograd",
                torch.sin,
                lib=lib,
            )
            torch.library.impl(
                "mylib::bar3.int", "CompositeImplicitAutograd", torch.sum, lib=lib
            )

            def f(x):
                return torch.ops.mylib.bar3(x)

            def g(x):
                return torch.ops.mylib.bar3(x, 1)

            def h(x):
                return torch.ops.mylib.bar3(x, x, x)

            x = torch.randn(3)

            counts = torch._dynamo.testing.CompileCounter()
            optimized_f = torch.compile(f, backend=counts, fullgraph=True)
            optimized_g = torch.compile(g, backend=counts, fullgraph=True)
            optimized_h = torch.compile(h, backend=counts, fullgraph=True)

            # No error: the overload is PT2 compliant
            optimized_f(x)

            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "not PT2 compliant"
            ):
                y = optimized_g(x)

            # graph break on incorrect parsing
            with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "failed to"):
                y = optimized_h(x)

    def test_user_defined_setattr1(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            obj.y = obj.x + 1

        obj = UserDefineSetAttr()
        with patch.object(UserDefineSetAttr, "setup", True):
            obj.x = torch.randn(8)
        fn(obj)
        with patch.object(UserDefineSetAttr, "setup", True):
            self.assertEqual(obj.y, obj.x + 1)
        self.assertEqual(obj.__dict__.keys(), {"pfx_x", "pfx_y"})

    def test_user_defined_setattr2(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            obj = UserDefineSetAttr()
            obj.x = x
            obj.y = obj.x + 1
            return obj

        x = torch.randn(8)
        obj = fn(x)
        with patch.object(UserDefineSetAttr, "setup", True):
            self.assertIs(obj.x, x)
            self.assertEqual(obj.y, x + 1)
        self.assertEqual(obj.__dict__.keys(), {"pfx_x", "pfx_y"})

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_repeat_cat(self):
        def f(x, n):
            m = x.item()
            x = torch.empty(x).repeat(n)  # s0*u0
            return torch.cat([x, x], dim=0)

        fn = torch.compile(f, backend="eager", dynamic=True, fullgraph=True)
        fn(torch.tensor([5]), 5)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_cond_runtime_assert_generation(self):
        def fn(x):
            y = x.nonzero()  # unbacked binding u0
            torch._check(y.shape[0] % 4 == 0)

            return torch.randn(y.shape[0])

        @torch.compile(dynamic=True, backend="aot_eager")
        def foo(x):
            b = torch.cond(
                pred=(x.shape[0] % 4 == 0),
                true_fn=lambda: fn(x),
                false_fn=lambda: fn(x),
            )

            return b

        foo(torch.randn(4, 4))
        with self.assertRaisesRegex(
            RuntimeError, "Runtime assertion failed for expression Eq(Mod(u1, 4), 0)*"
        ):
            foo(torch.randn(5, 5))

    def test_tensor_setattr_getset_descriptor(self):
        # Tensor attribute `real` has special getter/setter for complex dtype.
        def f(x):
            x.real = 10
            return x + 1

        opt_f = torch.compile(f, backend="eager", fullgraph=False)
        x = torch.ones(5, dtype=torch.cfloat)

        res = opt_f(x)
        ref = f(x)
        self.assertEqual(res, ref)

    def test_newly_constructed_tensor_attr_mutation(self):
        def f(x):
            y = x + 10
            y.grad = x
            y.foo = 42
            return y

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.ones(5)

        res = opt_f(x)
        ref = f(x)
        self.assertEqual(res, ref)
        self.assertEqual(res.grad, ref.grad)
        self.assertEqual(res.foo, ref.foo)

    def test_closure_recompiles(self):
        cnt = CompileCounter()

        def fn(x, other_fn):
            return other_fn(x + 1) - 1

        opt = torch.compile(fn, backend=cnt, fullgraph=True)

        x = torch.randn(8)
        for f in (
            closure_adder(5),
            closure_adder(5),
            closure_adder(torch.randn(8)),
            closure_adder(torch.randn(8)),
        ):
            self.assertEqual(opt(x, f), fn(x, f))

        self.assertEqual(cnt.frame_count, 2)

    def test_generate_trivial_abstract_impl(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor x, Tensor[] y, Tensor(a!)? z, SymInt w) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y, z, w):
                x + y[0] + w
                return

            def f(x, y, z, w):
                return torch.ops.mylib.foo(x, y, z, 2)

            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            w = torch.randn(3)
            args = (x, y, z, w)

            output = torch.compile(f, backend="eager", fullgraph=True)(*args)
            self.assertEqual(output, None)

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
            self, fn, 1, expected_ops=1, expected_ops_dynamic=ifdynstaticdefault(1, 6)
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
            self, fn, 1, expected_ops=1, expected_ops_dynamic=ifdynstaticdefault(1, 2)
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

        # TODO: Test the guards maybe?
        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

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

        # TODO: Test the guards maybe?
        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_param_shape_binops(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
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
        optimized_mod = torch.compile(mod, backend=counts, fullgraph=True)

        x = torch.randn(3)
        ref = mod(x)
        res = optimized_mod(x)

        self.assertTrue(same(ref, res))
        self.assertEqual(counts.frame_count, 1)

        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(counts.op_count, """1""")
        else:
            self.assertExpectedInline(counts.op_count, """9""")

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
        opt_fn = torch.compile(fn, backend=counts)

        x = torch.randn(3)
        c = MyClass(4)
        ref = fn(x, c)
        res = opt_fn(x, c)

        self.assertTrue(same(ref, res))
        self.assertEqual(counts.frame_count, 1)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(counts.op_count, """1""")
        else:
            self.assertExpectedInline(counts.op_count, """2""")

    def test_user_defined_iter(self):
        class Mod:
            def __init__(self) -> None:
                self.a = [torch.randn(2, 2), torch.randn(2, 2)]

            def __iter__(self):
                return iter(self.a)

        def f(mod):
            ret = []
            for x in mod:
                ret.append(x + 1)
            return ret

        mod = Mod()
        counts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(f, backend=counts, fullgraph=True)
        ref = f(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        self.assertTrue(same(ref, res))
        self.assertEqual(counts.frame_count, 1)

        mod.a.append(torch.randn(2, 2))
        # `for x in mod` is inlined, where iter(m.a) creates a guard on the list length of m.a
        # Mutating length of mod.a causes a re-compilation.
        ref2 = f(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        self.assertTrue(same(ref2, res2))
        self.assertEqual(counts.frame_count, 2)

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
        self.assertIn(
            """tensor 'a' size mismatch at index 0. expected 3, actual 4""",
            guard_failure.reason,
        )

    def test_recompile_message_on_parameter(self):
        def guard_failures(failure):
            self.assertIn("torch._dynamo.config.force_parameter_static_shapes", failure)

        @torch._dynamo.optimize("eager", guard_fail_fn=guard_failures)
        def fn(x):
            return torch.cos(x)

        x1 = torch.nn.Parameter(torch.rand(32, 16))
        x2 = torch.nn.Parameter(torch.rand(8, 4, 3, 3))
        x3 = torch.nn.Parameter(torch.rand(8, 8, 3, 3))
        fn(x1)
        fn(x2)
        fn(x3)

    def test_builtin_abs(self):
        def fn(x, y):
            return abs(x) + abs(y)

        sample = torch.randn(10, 10)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        for sample in [
            (torch.randn(10, 10), torch.randn(10, 10)),
            (-10, make_tensor(10, dtype=torch.int64, device="cpu")),
            (-0.1, torch.randn(10)),
        ]:
            expect = fn(*sample)
            actual = opt_fn(*sample)
            self.assertEqual(expect, actual)

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

    def test_os_environ_get(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x):
            if os.environ.get("OS_ENVIRON_TEST") == "1":
                return x + 1
            else:
                return x - 1

        x = torch.ones(2, 3)
        try:
            original = os.environ.get("OS_ENVIRON_TEST", None)

            os.environ["OS_ENVIRON_TEST"] = "1"
            res1 = fn(x)
            self.assertEqual(res1, x + 1)
            self.assertEqual(cnts.frame_count, 1)
            os.environ["OS_ENVIRON_TEST"] = "0"
            res2 = fn(x)
            self.assertEqual(res2, x - 1)
            # Ensure re-compile if os.environ items updated
            self.assertEqual(cnts.frame_count, 2)
        finally:
            if original is None:
                del os.environ["OS_ENVIRON_TEST"]
            else:
                os.environ["OS_ENVIRON_TEST"] = original

    def test_os_environ_set_graph_break(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=False)
        def fn(x):
            x = x + 1
            os.environ["OS_ENVIRON_TEST"] = "0"
            return torch.sin(x)

        x = torch.ones(2, 3)
        try:
            original = os.environ.get("OS_ENVIRON_TEST", None)

            os.environ["OS_ENVIRON_TEST"] = "1"
            res1 = fn(x)
            self.assertEqual(res1, torch.sin(x + 1))
            self.assertEqual(os.environ["OS_ENVIRON_TEST"], "0")
            # Ensure we graph break on os.environ.__setitem__
            self.assertEqual(cnts.frame_count, 2)
        finally:
            if original is None:
                del os.environ["OS_ENVIRON_TEST"]
            else:
                os.environ["OS_ENVIRON_TEST"] = original

    def test_sys_modules(self):
        def fn(x, y):
            mod_a = sys.modules.get("aaaaaaaa")
            assert mod_a is None
            assert "bbbbbbbb" not in sys.modules

            assert "operator" in sys.modules
            operator = sys.modules["operator"]
            builtins = sys.modules.get("builtins")
            operator2 = sys.modules.get("cccccccc", operator)

            return operator.add(x, y), operator2.neg(builtins.abs(x))

        torch._dynamo.testing.standard_test(self, fn, 2, expected_ops=3)

        x = torch.randn(10, 10)
        _, guards = torch._dynamo.export(fn, x, x)
        guard_code = []
        for guard in guards:
            if guard.code_list:
                guard_code += guard.code_list

        # Filter out id-matches that won't reproduce run to run
        guard_code = filter(
            lambda line: "id" not in line and "lookup_backend" not in line,
            sorted(guard_code),
        )
        guard_code_str = "\n".join(guard_code)

        for line in """\
2 <= L['x'].size()[0]
L['x'] is L['y']
L['x'].ndimension() == 2
L['x'].requires_grad == False
L['x'].size()[1] == L['x'].size()[0]
L['x'].storage_offset() == 0
___dict_contains('operator', G['sys'].modules)
___dict_contains('operator', G['sys'].modules)
hasattr(L['x'], '_dynamo_dynamic_indices') == False
not ___dict_contains('aaaaaaaa', G['sys'].modules)
not ___dict_contains('bbbbbbbb', G['sys'].modules)
not ___dict_contains('cccccccc', G['sys'].modules)
str(L['x'].device) == 'cpu'
str(L['x'].dtype) == 'torch.float32'
utils_device.CURRENT_DEVICE == None""".split("\n"):
            self.assertIn(
                line,
                guard_code_str,
            )

    def test_fold(self):
        def fn(a):
            return a + math.sqrt(63)

        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_getattr_dict(self):
        def fn(x):
            from torch.masked.maskedtensor._ops_refs import _MASKEDTENSOR_FUNCTION_TABLE

            return x * len(_MASKEDTENSOR_FUNCTION_TABLE)

        i = torch.randn(5)
        r1 = fn(i)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        r2 = opt_fn(i)
        self.assertEqual(r1, r2)

    def test_tensor_hasattr(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            if hasattr(x, "test"):
                return x + 2
            else:
                return x + 1

        self.assertEqual(torch.ones(2, 2) + 1, fn(torch.ones(2, 2)))

        inp = torch.ones(2, 2)
        inp.test = None
        self.assertEqual(torch.ones(2, 2) + 2, fn(inp))

    def test_mro_type_tensor_no_source(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            z = []
            input_type = type(torch.ones(2, 2))
            for cls in input_type.__mro__:
                z.append(cls.__name__)

            return x, input_type, z

        inp = torch.ones(2, 2)
        fn(inp)

    def test_tensor_dynamic_method(self):
        def add_one(x):
            return x + 1

        t = torch.nn.Parameter(torch.ones(1))
        t.add_one = add_one

        @torch.compile(fullgraph=True)
        def fn(x):
            return t.add_one(t) + x

        result = fn(torch.ones(1))
        self.assertEqual(torch.ones(1) + 2, result)

    def test_shape_unpack(self):
        def fn(x):
            a, b = x.size()
            return x * b

        i = torch.randn(5, 10)
        r1 = fn(i)
        opt_fn = torch.compile(fn, backend="eager")
        r2 = opt_fn(i)
        self.assertTrue(same(r1, r2))

    def test_typing_dict(self):
        def fn(d):
            return d[T]

        d = {T: torch.randn(3)}
        r1 = fn(d)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        r2 = opt_fn(d)
        self.assertEqual(r1, r2)

    def test_tensor__iter__(self):
        def fn(x):
            it = x.__iter__()
            for y in it:
                y.add_(1.0)
            return y

        torch._dynamo.testing.standard_test(
            self,
            fn,
            1,
            expected_ops=20,
        )

    def test_tensor_iter(self):
        def fn(x):
            for y in x:
                y.add_(1.0)
            return y

        torch._dynamo.testing.standard_test(
            self,
            fn,
            1,
            expected_ops=20,
        )

    def test_empty_list(self):
        def fn(x, ll):
            if len(ll) == 0 and not ll and ll is not None:
                return x + 1

        i = torch.randn(5, 10)
        r1 = fn(i, [])
        opt_fn = torch.compile(fn, backend="eager")
        r2 = opt_fn(i, [])
        r3 = opt_fn(i, ())
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

        torch._dynamo.testing.standard_test(
            self,
            get_test_fn(func=min),
            2,
            expected_ops=1,
            expected_ops_dynamic=ifdynstaticdefault(1, 7),
        )
        torch._dynamo.testing.standard_test(
            self,
            get_test_fn(func=max),
            2,
            expected_ops=1,
            expected_ops_dynamic=ifdynstaticdefault(1, 7),
        )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_bound_shape_checks(self):
        def f1(x, y):
            b = x.item()
            torch._check(b >= 0)
            torch._check(b < y.shape[0])
            return y[:b]

        fn1 = torch.compile(f1, fullgraph=True, backend="eager")
        fn1(torch.tensor(4), torch.ones(10))

        def f2(x, index):
            idx = index.item()
            torch._check(idx >= 0)
            torch._check(idx < x.size(0))
            return x[idx]

        A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        index = torch.tensor(1, dtype=torch.int64)
        fn2 = torch.compile(f2, fullgraph=True, backend="eager")
        fn2(A, index)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_arange_length_with_float32_dtype(self):
        @torch.compile(fullgraph=True)
        def f(x):
            y = x.item()
            r = torch.arange(y, dtype=torch.float32)

            if r.size(0) == y:
                return r + 1

            return r

        x = torch.tensor([300])
        r = f(x)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_torch_check(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            y = x.item()
            torch._check(y >= 0)
            return torch.arange(0, y)

        f(torch.tensor([3]))
        f(torch.tensor([4]))
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_torch_check_symbolic_shape_rel(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            y = x.item()
            torch._check(x.shape[0] == 1)
            torch._check(x.shape[0] != 2)
            torch._check(x.shape[0] >= 0)
            torch._check(x.shape[0] > 0)
            torch._check(x.shape[0] < 4)
            torch._check(x.shape[0] <= 3)
            return torch.arange(0, y)

        f(torch.tensor([3]))
        f(torch.tensor([4]))
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    # Translation validation changes the exception type, don't run with it
    @torch.fx.experimental._config.patch(translation_validation=False)
    def test_torch_check_nonnegative(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            y = x.item()
            torch._check(y >= 0)
            # Cannot conditional on unbacked SymInt
            if y == 0:
                assert False
            else:
                return torch.arange(0, y)

        self.assertRaises(torch._dynamo.exc.UserError, lambda: f(torch.tensor([3])))

    def test_check_compiles_when_predicate_true_and_message_has_no_closure(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_constant_and_message_has_no_closure(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(4)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_constant_and_message_None(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(4)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_and_message_None(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_and_message_has_global(self):
        global GLOBAL_INT
        GLOBAL_INT = 1

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: f"{GLOBAL_INT} is not greater than 3")
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_raises_at_runtime_when_predicate_false_and_message_has_global(self):
        global GLOBAL_INT
        GLOBAL_INT = 1

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: f"{GLOBAL_INT} is not greater than 3")
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(
            RuntimeError, f"{GLOBAL_INT} is not greater than 3"
        ):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_and_message_None(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, None):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_constant_and_message_None(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(3)

        with self.assertRaisesRegex(RuntimeError, None):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_and_message_has_no_closure(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, "Shape is not greater than 3"):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_constant_and_message_has_no_closure(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(3)

        with self.assertRaisesRegex(RuntimeError, "Shape is not greater than 3"):
            f(x)

    def test_check_assert_error_at_runtime_when_predicate_false_and_message_has_closure(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: f"{x.shape[0]} is not greater than 3")
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Can't extract message from torch._check()"
        ):
            f(x)

    def test_check_assert_error_at_runtime_when_predicate_true_and_message_has_closure(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: f"{x.shape[0]} is not greater than 3")
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Can't extract message from torch._check()"
        ):
            f(x)

    def test_assert(self):
        @torch.compile
        def fn1(x):
            assert x.shape != x.shape

        with self.assertRaises(AssertionError):
            a = torch.randn(10)
            fn1(a)

        def fn2(x):
            assert x.shape == x.shape
            return x.abs()

        torch._dynamo.testing.standard_test(self, fn=fn2, nargs=1, expected_ops=1)

    # When we unspecialize float, we wobble this test by changing
    # the op count since previously we would just specialize and constant
    # fold floats into the graph, whereas when we unspecialize we will have
    # ops for item, add, and all other tensorified operations. Since this
    # test really isn't testing that, we purposely specialize floats here.
    @torch._dynamo.config.patch(specialize_float=True)
    def test_config_obj(self):
        class Cfg:
            def __init__(self) -> None:
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
        opt_fn = torch.compile(fn, backend=cnts)
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
            def __init__(self) -> None:
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
        opt_fn = torch.compile(fn, backend=cnts)
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
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v, v.size())[0, 0], -10)
        self.assertEqual(opt_fn(v, (10, 20))[0, 0], -10)
        self.assertEqual(opt_fn(v, [10, 20])[0, 0], -10)
        # One recompile per differing input type
        self.assertEqual(cnts.frame_count, 3)

    def test_cell_output1(self):
        out = None

        def fn(a, b):
            nonlocal out
            out = a + b * 10

        v = torch.Tensor([100])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
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
        opt_fn = torch.compile(fn, backend=cnts)
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
        opt_fn = torch.compile(fn, backend=cnts)
        opt_fn_ret = torch.compile(opt_fn(v1, v2), backend=cnts)
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
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(opt_fn({"a": v1, "b": v2})[0], -200)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_tensor_dict3(self):
        def fn(inputs_a, inputs_b):
            total = torch.zeros(1)
            input_keys = inputs_a.keys() | inputs_b.keys()
            for k in input_keys:
                if k in inputs_a:
                    total += inputs_a[k]
                if k in inputs_b:
                    total += inputs_b[k]
            return total

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(
            opt_fn({"a": v1, "b": v2}, {"b": v1, "c": v2}),
            fn({"a": v1, "b": v2}, {"b": v1, "c": v2}),
        )
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

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
        opt_fn1 = torch.compile(fn1, backend=cnts, fullgraph=True)
        opt_fn2 = torch.compile(fn2, backend=cnts, fullgraph=True)
        opt_fn3 = torch.compile(fn3, backend=cnts, fullgraph=True)
        self.assertEqual(opt_fn1({"a": v1, "b": v2})[0], 300)
        self.assertEqual(opt_fn2({"a": v1, "b": v2})[0], 300)
        self.assertEqual(opt_fn3({"a": v1, "b": v2})[0], 300)
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 9)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_user_code_statically_known(self):
        from torch.fx.experimental.symbolic_shapes import (
            has_static_value,
            statically_known_true,
        )

        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            # At this point, this isn't statically known, only the hint says so.
            if statically_known_true(x.shape[0] > 9):
                raise Exception()
            torch._check(x.shape[0] >= 10)
            # But now it is.
            return statically_known_true(x.shape[0] > 9), has_static_value(x.shape[0])

        x = torch.zeros(10)
        torch._dynamo.mark_dynamic(x, 0)
        self.assertEqual(f(x), (True, False))

        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def g(x, y):
            n = x.item()
            torch._check(n == 3)
            return has_static_value(4.0), has_static_value(n)

        out = g(torch.tensor([3]), torch.zeros(1))
        self.assertEqual(out, (True, True))

    def test_dictcomp(self):
        def fn1(inputs):
            return {k: v + 1 for k, v in inputs.items()}

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch.compile(fn1, backend=cnts)
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
        opt_fn2 = torch.compile(fn2, backend=cnts)
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
        opt_fn = torch.compile(fn, backend="eager")
        res1 = opt_fn(x1)
        res2 = opt_fn(x2)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_numel(self):
        def fn(a):
            return (a + a.numel() + torch.numel(a), a + a.nelement())

        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=1,
            expected_ops=3,
            expected_ops_dynamic=ifdynstaticdefault(3, 4),
        )

    def test_pair(self):
        def fn(a):
            return (
                torch.zeros(torch.nn.modules.utils._pair(a.size()))
                + a
                + torch.ones(torch.nn.modules.utils._ntuple(3)(3)).sum()
            )

        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=1,
            expected_ops=5,
            expected_ops_dynamic=5,
        )

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_tensor_item_capture(self):
        def fn(a, b):
            return (a + b).sum().item()

        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        correct = fn(v1, v2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v1, v2), correct)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", False)
    def test_tensor_item_no_capture(self):
        def fn(a, b):
            return (a + b).sum().item()

        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        correct = fn(v1, v2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v1, v2), correct)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_namedtuple1(self):
        def fn(a, b):
            tmp = MyTuple(a, b, a + b)
            return MyTuple(tmp.a, tmp[1], tmp.ab + b)

        v1 = torch.Tensor([10])
        v2 = torch.Tensor([20])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v1, v2).ab, 50)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_namedtuple2(self):
        def fn(packed):
            a, b, c = packed
            if hasattr(packed, "b"):
                b = packed.b + 1
            c = packed[2]
            d = len(packed._fields)
            return a + b + c + d

        v1 = torch.Tensor([1])
        v2 = torch.Tensor([2])
        v3 = torch.Tensor([3])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(MyTuple(v1, v2, v3))[0], 10)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_namedtuple3(self):
        def fn(x, packed):
            if isinstance(packed, MyTuple):
                return x + 1
            else:
                return x - 1

        x = torch.rand([2, 3])
        packed = MyTuple(1, 2, 3)
        ref = fn(x, packed)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, packed)
        self.assertTrue(same(ref, res))

    def test_namedtuple_with_custom_getitem(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(my_tuple):
            return my_tuple.a + 1

        class MyTuple(typing.NamedTuple):
            a: torch.Tensor
            b: torch.Tensor

            def __getitem__(self, index):
                return MyTuple(a[index], b[index])

        a = torch.randn(2)
        b = torch.randn(2)

        out = f(MyTuple(a, b))
        self.assertTrue(same(a + 1, out))

        # Test guard evaluation in the second call
        out = f(MyTuple(a, b))
        self.assertTrue(same(a + 1, out))

    def test_namedtuple_source_dynamic_attributes(self):
        class MyNamedTuple(typing.NamedTuple):
            a: torch.Tensor
            b: torch.Tensor

        class MyNamedTupleSubclass(MyNamedTuple):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def f(tup):
            c = torch.tensor(3.0)
            tup.c = c  # Add dynamic attribute
            return tup

        extended_tup = MyNamedTupleSubclass(a=torch.tensor([1.0]), b=torch.tensor(2.0))
        result = f(extended_tup)
        # Verify the tuple has the expected structure
        self.assertEqual(result.a, torch.tensor([1.0]))
        self.assertEqual(result.b, torch.tensor(2.0))
        self.assertTrue(hasattr(result, "c"))
        self.assertEqual(result.c, torch.tensor(3.0))

    def test_namedtuple_sourceless_dynamic_attributes(self):
        class MyNamedTuple(typing.NamedTuple):
            a: torch.Tensor
            b: torch.Tensor

        class MyNamedTupleSubclass(MyNamedTuple):
            pass

        @torch.compile(backend="eager")
        def f():
            # Create namedtuple inside function (sourceless)
            tup = MyNamedTupleSubclass(a=torch.tensor([1.0]), b=torch.tensor(2.0))
            # Add dynamic attribute
            tup.c = torch.tensor(3.0)
            return tup

        result = f()
        # Verify the tuple has the expected structure
        self.assertEqual(result.a, torch.tensor([1.0]))
        self.assertEqual(result.b, torch.tensor(2.0))
        # Verify the dynamic attribute is preserved
        self.assertTrue(hasattr(result, "c"))
        self.assertEqual(result.c, torch.tensor(3.0))

    def test_structseq1(self):
        def fn(x, y):
            return torch.return_types.max((x, y))

        x = torch.randn(3, 2)
        y = torch.randn(2, 4)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True)(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    def test_structseq2(self):
        def fn(x, y):
            return tuple(torch.return_types.qr((2 * x, y - 1)))

        x = torch.randn(3, 2)
        y = torch.randn(2, 4)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True)(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

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

        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=1,
            expected_ops=9,
        )

    def test_range_iter_guards(self):
        @torch.compile()
        def func():
            @torch._dynamo.disable(recursive=False)
            def run(n):
                # For python <= 3.11, list comprehension is implemented by
                # desugaring to:
                # 1. creation of an iterator object
                # 2. calling a new `listcomp` function with (1)
                #
                # In this test we force Dynamo to trace through (2) as the root
                # frame, thereby ensuring we have the right guards for range
                # iterators.
                xs = [torch.ones(1) for i in range(n)]
                return torch.concat(xs)

            return run(2), run(3)

        res2, res3 = func()
        self.assertTrue(same(res2, torch.ones(2)))
        self.assertTrue(same(res3, torch.ones(3)))

    def test_range___iter__(self):
        def func(x):
            it = range(3).__iter__()
            return x + next(it)

        opt_func = torch.compile(func, backend="eager", fullgraph=True)
        x = torch.randn(3)
        self.assertTrue(same(func(x), opt_func(x)))

    def test_range_iter_side_effects(self):
        @torch.compile(backend="eager", fullgraph=True)
        def run(x, it):
            n = next(it)
            return x + n

        it = iter(range(1, 3))
        res = run(torch.zeros(1), it)
        self.assertTrue(same(res, torch.ones(1)))
        self.assertEqual(next(it), 2)

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
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(2), [None] * 4)
        # TODO: the captured frame here is a bit goofy, because we don't
        # output anything and none of the traced operations have side
        # effects.  Probably need better heuristic for bailing on
        # dynamo if there are no outputs
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")
        else:
            self.assertExpectedInline(cnts.frame_count, """1""")
            self.assertExpectedInline(cnts.op_count, """2""")

    def test_list_slice_mul(self):
        def fn(count):
            a = [1, 2, 3]
            head_mask = count * a[1:] * count
            return head_mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(2), [2, 3] * 4)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")
        else:
            self.assertExpectedInline(cnts.frame_count, """1""")
            self.assertExpectedInline(cnts.op_count, """2""")

    def test_tuple_mul(self):
        def fn(count):
            head_mask = count * (2, 3) * count
            return head_mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(2), (2, 3) * 4)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")
        else:
            self.assertExpectedInline(cnts.frame_count, """1""")
            self.assertExpectedInline(cnts.op_count, """2""")

    def test_tuple_mul_with_shape(self):
        def fn(a):
            x = a.shape[0]
            y = 2 * (x, 3) * 2
            return a + y[4]

        # expect 3 ops post folding for dynamic case: size, index, add
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=1
        )

    def test_tuple_iadd_with_shape(self):
        def fn(a):
            output = (a + a.shape[0], a - a.shape[0])
            # tuple += tuple
            output += (a - a.shape[0], a + a.shape[0])
            # tuple += constant tuple
            output += (2, 3)
            return output

        # expect 4 add / subs for static
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=4, expected_ops_dynamic=4
        )

    def test_list_iadd_with_shape(self):
        def fn(a):
            output = [a + a.shape[0], a - a.shape[0]]
            # list += list
            output += [a - a.shape[0], a + a.shape[0]]
            # list += tuple
            output += (a + a.shape[0], a - a.shape[0])
            return output

        # expect 6 add / subs for static

        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=6, expected_ops_dynamic=6
        )

    def test_list_iadd_side_effect(self):
        def fn(a, b):
            a += [b]
            torch._dynamo.graph_break()
            return a

        a = [1, 2, 3]
        b = torch.ones(2, 2)

        opt_fn = torch.compile(fn, backend="eager")

        exp = fn(a, b)

        a = [1, 2, 3]
        b = torch.ones(2, 2)
        act = opt_fn(a, b)

        self.assertEqual(exp, act)

    def test_class_binop(self):
        class Foo:
            def __init__(self, x):
                self.x = x

            def __add__(self, other):
                return Foo(self.x + other.x)

        def fn(a, b):
            return a + b

        x = torch.randn(2)
        a, b = Foo(x), Foo(x + 1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(a, b).x, 2 * x + 1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

        def fn(a, b):
            return a - b

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertRaises(torch._dynamo.exc.Unsupported, opt_fn, a, b)

    def test_user_getattr1(self):
        class MyConfig(dict):
            def __getattr__(self, name):
                return self[name]

        def fn(cfg, x, y):
            return x + y + cfg.offset

        x = torch.randn(10)
        cfg = MyConfig(offset=5)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(cfg, x, x), 2 * x + 5))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_user_getattr2(self):
        class MyConfig:
            defined_on_class = 1

            def __init__(self) -> None:
                self.defined_on_object = 2

            def __getattr__(self, name):
                return 3

        def fn(cfg, x):
            return x + cfg.defined_on_class - cfg.defined_on_object + cfg.not_defined

        x = torch.randn(10)
        cfg = MyConfig()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(cfg, x), x + 1 - 2 + 3))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_getset_descriptor(self):
        def fn(g, x):
            # Just to make Dynamo not skip the frame
            torch.sin(x)
            return g.__get__(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        g = torch.Tensor.shape

        res = opt_fn(g, torch.ones(2, 2))
        exp_res = fn(g, torch.ones(2, 2))
        self.assertEqual(res, exp_res)

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            res = opt_fn(g, torch.ones(2, 2))

    def test_set_descriptor(self):
        class Field:
            def __set__(self, obj, value):
                obj.__dict__["field"] += value * 2

        class Foo:
            field = Field()

            def __init__(self):
                self.__dict__["field"] = 0

        def fn(x, foo):
            foo.field = 10
            return x + foo.field

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        x = torch.zeros(2)
        foo1, foo2 = Foo(), Foo()

        ref = fn(x, foo1)
        res = opt_fn(x, foo2)
        self.assertEqual(ref, res)
        self.assertEqual(foo1.field, foo2.field)

    def test_get_attr_function(self):
        def fn(g, x):
            return g(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        g = torch.Tensor.shape.__get__

        res = opt_fn(g, torch.ones(2, 2))
        exp_res = fn(g, torch.ones(2, 2))
        self.assertEqual(res, exp_res)

    def test_user_getattribute(self):
        class MyObject:
            def __init__(self) -> None:
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
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj, x), fn(obj, x)))

    def test_nn_module_getattr(self):
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
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
        opt_mod = torch.compile(mod, backend=cnts)
        self.assertTrue(same(opt_mod(x), mod(x)))
        self.assertTrue(cnts.frame_count, 1)
        self.assertTrue(cnts.op_count, 2)

    def test_nn_module_getattribute(self):
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
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
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(mod, x), fn(mod, x)))

    def test_constant_getattr(self):
        # https://github.com/pytorch/pytorch/issues/97480
        def fn():
            return getattr(None, "arg", 3)

        cnt = torch._dynamo.testing.CompileCounter()
        optimized_fn = torch.compile(fn, backend=cnt)
        res = optimized_fn()
        self.assertTrue(same(res, 3))

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
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(cfg, x, x), 2 * x + 5))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_data_access_in_inference_mode(self):
        @torch.compile(fullgraph=True)
        def f(x):
            y = x.data
            return y

        with torch.inference_mode():
            x = torch.randn(3)
            y = f(x)
        self.assertEqual(y, x)

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

            if not hasattr(obj, "a"):
                return -1
            if hasattr(obj, "z"):
                return -2

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
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj1), correct1))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj2), correct2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

        # guard failure
        obj2.z = True
        self.assertEqual(opt_fn(obj2), -2)

    def test_dataclass_local_hasattr(self):
        cnt = CompileCounter()
        x = torch.randn(10)

        @dataclasses.dataclass
        class MyDataClass:
            a: torch.Tensor
            b: torch.Tensor

        @torch.compile(backend=cnt, fullgraph=True)
        def fn():
            obj = MyDataClass(x + 1, x - 1)
            if not hasattr(obj, "a"):
                return -1
            if hasattr(obj, "z"):
                return -2
            return obj

        result = fn()
        self.assertIsInstance(result, MyDataClass)
        self.assertEqual(result.a, x + 1)
        self.assertEqual(result.b, x - 1)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_catch_watchings1(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            with warnings.catch_warnings(record=True):
                return x.sin()

        x = torch.randn(8)
        self.assertEqual(fn(x), x.sin())
        self.assertEqual(cnt.frame_count, 1)

    def test_catch_watchings2(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return x.sin(), warnings.catch_warnings(record=True)

        x = torch.randn(8)
        _, a = fn(x)
        _, b = fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertIsInstance(a, warnings.catch_warnings)
        self.assertIsInstance(b, warnings.catch_warnings)
        self.assertIsNot(a, b)

    def test_tensor_build_list_unpack(self):
        def fn(x):
            # seen in fastNLP_Bert
            return torch.cat([*x], dim=-1)

        val = torch.randn([1, 1, 473, 768])
        correct = fn(val)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(val), correct))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_numpy_int_constant(self):
        def fn(x, a, b):
            return x + (a % b)

        args = [torch.randn(10), 4096, np.int64(8)]
        correct = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, dynamic=True, fullgraph=True)
        self.assertTrue(same(opt_fn(*args), correct))
        self.assertTrue(same(opt_fn(*args), correct))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_numpy_subdtype(self):
        def fn(x, n):
            return np.issubdtype(type(n), np.integer) + x

        args = [torch.randn(10), 4096]
        correct = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(opt_fn(*args), correct)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_take_along_axis(self):
        def fn(x, i, a):
            return np.take_along_axis(x, i, a)

        def sample_to_args(s):
            args = (s.input, *sample.args)
            return tuple(a.numpy() if isinstance(a, torch.Tensor) else a for a in args)

        samples = list(
            sample_inputs_take_along_dim(
                None, "cpu", torch.float32, requires_grad=False
            )
        )
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        i = 1
        for sample in samples:
            args = sample_to_args(sample)
            if len(args) < 3:
                # if axis is None, second argument is treated as 1d array
                args = (args[0], np.ravel(args[1]), None)
            self.assertEqual(fn(*args), opt_fn(*args))
            self.assertEqual(cnts.frame_count, i)
            i += 1

    def test_numpy_torch_operators(self):
        def fn(op, t1, t2):
            return op(t1, t2)

        from torch._dynamo.variables.builtin import BuiltinVariable

        operators = BuiltinVariable._fx_graph_functions()

        for op, t1_np, t2_np in itertools.product(
            operators, (True, False), (True, False)
        ):
            if op in [operator.eq, operator.ne]:
                # returns equivalent of torch.eq/ne
                continue
            if op is operator.getitem:
                # skip
                # Did you know that tensor[ndarray_of_floats] works?
                continue
            if op is operator.imatmul and (t1_np or t2_np):
                # skip
                # in numpy, in place matmul does not work single
                # dimensional arrays
                continue
            t1 = torch.rand(5)
            if t1_np:
                t1 = t1.numpy()
            t2 = torch.rand(5)
            if t2_np:
                t2 = t2.numpy()
            try:
                # TODO try a bit harder
                result = op(t1, t2)
            except (RuntimeError, TypeError, IndexError):
                continue
            cnts = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnts)
            self.assertEqual(result, opt_fn(op, t1, t2), msg=f"{op=} {t1_np=} {t2_np=}")
            self.assertEqual(cnts.frame_count, 1, msg=f"{op=} {t1_np=} {t2_np=}")
            torch._dynamo.reset()

    def test_numpy_ndarray_graph_break(self):
        def fn(x):
            a = x.numpy()
            b = a.real
            torch._dynamo.graph_break()
            c = np.multiply(b, 2.0)
            return c

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.randn(3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_numpy_ndarray_graph_break_with_multiple_outputs(self):
        def fn(x, y):
            a = x.numpy()
            b = y.numpy()
            torch._dynamo.graph_break()
            return np.add(a, 1), np.add(b, 1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.randn([1, 3])
            y = torch.randn([1, 3])
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_numpy_force(self):
        def fn(x):
            return x.numpy(force=False)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        x = torch.randn(3)
        res = opt_fn(x)
        self.assertEqual(type(res), np.ndarray)
        self.assertEqual(cnts.frame_count, 1)

        def fn(x):
            return x.numpy(force=True)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        x = torch.randn(3, requires_grad=True)
        res = opt_fn(x)
        self.assertEqual(type(res), np.ndarray)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_recompilation_scalar(self):
        def fn(x, a):
            return np.where(x < 0.5, a, x)

        x = np.random.randn(8)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, dynamic=True)

        ref = fn(x, 3)
        res = opt_fn(x, 3)
        self.assertEqual(ref, res)

        ref = fn(x, 4)
        res = opt_fn(x, 4)
        self.assertEqual(ref, res)

        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_interacts_with_numpy_ndarray(self):
        def fn(x, y):
            a = x.numpy()
            b = y.numpy()
            c = np.ones_like(a)
            d = np.ones_like(b)
            torch._dynamo.graph_break()
            return np.add(a, c), np.add(b, d)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.randn([1, 3])
            y = torch.randn([1, 3])
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_numpy_ndarray_works_with_builtin_function(self):
        def fn(x):
            v = x.sum() / len(x)
            return v

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        for _ in range(10):
            x = np.random.randn(2, 3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_array_of_arrays(self):
        def fn(x, y):
            return np.array([x, y])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        x, y = np.float64(1), np.float64(2)
        res = opt_fn(x, y)
        self.assertEqual(res, np.array([1, 2], dtype=float))
        self.assertEqual(type(res), np.ndarray)
        self.assertEqual(cnts.frame_count, 1)

        x, y = np.arange(2), np.arange(2) + 2
        res = opt_fn(x, y)
        self.assertEqual(res, np.array([[0, 1], [2, 3]]))
        self.assertEqual(type(res), np.ndarray)
        self.assertEqual(cnts.frame_count, 2)

    def test_numpy_readonly(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return x

        x = np.broadcast_to(np.arange(3), (2, 3))
        self.assertFalse(x.flags.writeable)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.simplefilter("ignore", category=DeprecationWarning)  # from asyncio
            y = fn(x)
        self.assertTrue(y.flags.writeable)  # XXX: differs from numpy

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_numpy_tolist(self):
        def fn(x):
            return x.tolist()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        x = np.arange(5)
        r = opt_fn(x)

        self.assertEqual(r, [0, 1, 2, 3, 4])
        self.assertEqual(type(r), list)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_size_attr(self):
        def fn(x):
            return x.size + x

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        x = np.arange(5)
        r = opt_fn(x)

        self.assertEqual(r, fn(x))
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_no_raise(self):
        def _inf_nan_preprocess(t, t_np):
            t_np = np.nan_to_num(t_np)
            return t, t_np

        def fn():
            # shape, dims format
            test_cases = (
                (3, 3),
                (4, 4),
                (5, 5),
            )

            for shape in test_cases:
                t = torch.randn(shape, dtype=torch.complex64)
                t_np = np.random.randn(*shape).astype(np.complex64)

                _, t_np = _inf_nan_preprocess(t, t_np)
                print(t, t_np)  # Just a side effect so that compilation kicks in

        cnt = CompileCounterWithBackend("inductor")
        fn = torch.compile(fn, backend=cnt)
        fn()
        self.assertEqual(cnt.frame_count, ifdynstaticdefault(2, 1))

    def test_mandelbrot_numpy(self):
        def mandelbrot_numpy(max_iter):
            # Define the boundaries of the complex plane
            xn = 450
            yn = 375
            xmin = -2.25
            xmax = 0.75
            ymin = -1.25
            ymax = 1.25

            # Create the grid of complex numbers
            x_values = np.linspace(xmin, xmax, xn, dtype=np.float64)
            y_values = np.linspace(ymin, ymax, yn, dtype=np.float64)
            rx, iy = np.meshgrid(x_values, y_values, indexing="xy")

            x = rx.copy()
            y = iy.copy()
            mask = np.zeros_like(x)
            for i in range(max_iter):
                x_prev = x
                y_prev = y
                x = x_prev**2 - y_prev**2 + rx
                y = 2 * x_prev * y_prev + iy
                inside = np.sqrt(x**2 + y**2) <= 2
                mask += inside
            return mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(mandelbrot_numpy, backend=cnts, fullgraph=True)
        n_iter = torch._dynamo.config.recompile_limit - 2
        for i in range(n_iter):
            x = i + 3
            ref = mandelbrot_numpy(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)
        # We need to specialise the number as it's in a forloop
        self.assertEqual(cnts.frame_count, n_iter)

    def test_numpy_as_global(self):
        global x
        x = np.arange(10)

        @torch.compile(fullgraph=True)
        def fn(y):
            return y + x + x

        r = fn(np.arange(10))
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(r, x * 3)
        del x

    def test_numpy_gt(self):
        x = np.arange(10)

        @torch.compile
        def fn(y):
            return y >= 3

        r = fn(x)
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(r, x >= 3)

    def test_numpy_min(self):
        x = np.arange(10)

        @torch.compile
        def fn(y):
            return min(y, 3), min(y, y - 1)

        r1, r2 = fn(x)
        self.assertEqual(type(r1), np.ndarray)
        self.assertEqual(type(r2), np.ndarray)
        self.assertEqual(r1, np.minimum(x, 3))
        self.assertEqual(r2, np.minimum(x, x - 1))

    def test_graph_break_correctly_when_passing_numpy_ndarray_to_torch_function(self):
        # from transformers/models/big_bird/modeling_big_bird.py
        def fn(x: int, y: torch.Tensor):
            ndarray_list = [np.ones([2, x])]
            ndarray = np.stack(ndarray_list, axis=0)
            tensor = torch.tensor(ndarray, dtype=torch.long)
            tensor.unsqueeze_(0)
            return tensor + y

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for x in range(1, 10):
            y = torch.randn([1, 2, x])
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertEqual(ref, res)
        # It's all traced once with x = 1 and then x = ks0
        # For dynamic it's x=ks0
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(str(cnts.frame_count), """2""")
        else:
            self.assertExpectedInline(str(cnts.frame_count), """2""")

    @skipIfWindows(
        msg="AssertionError: Object comparison failed: dtype('int64') != <class 'int'>"
    )
    def test_numpy_with_builtin_type(self):
        x = np.random.rand(5)

        def fn(x):
            return (x * 5).astype(bool).astype(float).astype(int) + 8

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        r = opt_fn(x)
        self.assertEqual(r.dtype, int)
        self.assertEqual(cnts.frame_count, 1)

    def test_with_builtin_type(self):
        x = torch.randn(5)

        def fn(x):
            return (x * 5).to(bool).to(float).to(int) + 8

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        r = opt_fn(x)
        self.assertEqual(r.dtype, torch.int64)
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_unique_consecutive(self):
        x = torch.tensor([1, 1, 2, 2, 1, 3])

        def fn(x):
            return torch.unique_consecutive(x)

        expected = fn(x)
        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        result = opt_fn(x)
        self.assertEqual(result, expected)

    def test_numpy_unique_f16(self):
        def fn():
            x = np.asarray([1, 1, 2, 2, 3], dtype=np.float16)
            return np.unique(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        r = opt_fn()
        self.assertEqual(r.dtype, np.float16)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_fallback_on_eager(self):
        def fn():
            return np.asarray(["L", "U"])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        r = opt_fn()
        self.assertEqual(cnts.frame_count, 0)  # graph break
        self.assertEqual(r, np.asarray(["L", "U"]))

        # repeat with a different function
        def fn2():
            return np.random.choice(["L", "U"])

        cnts2 = torch._dynamo.testing.CompileCounter()
        opt_fn2 = torch.compile(fn2, backend=cnts2)

        r2 = fn2()
        self.assertEqual(cnts.frame_count, 0)
        assert r2 in ("L", "U")

    def test_trace_ndarray_frame(self):
        def fn(x):
            x = x**2
            print("graph break.")
            return 2 * x

        counter = CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter)

        x = np.arange(8)
        self.assertEqual(fn(x), compiled_fn(x))
        self.assertEqual(counter.frame_count, 2)

    @skipIfWindows(
        msg="AssertionError: The values for attribute 'dtype' do not match: torch.int32 != torch.int64."
    )
    def test_trace_ndarray_frame_2(self):
        # no tensors/ndarray as inputs in the frame
        def fn(x):
            print("graph break.")
            return 2 * np.arange(x)

        counter = CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter)

        x = 8
        self.assertEqual(fn(x), compiled_fn(x))
        self.assertEqual(counter.frame_count, 1)

    def test_numpy_non_torch_dtype(self):
        # test that we gracefully graph break on dtypes
        # that do not have pytorch equivalents.
        def fn(x):
            return isinstance(x, torch.Tensor)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        # torch does not have the `uint16` dtype
        for x in [np.array([42], dtype=np.uint16), np.uint16(42), np.dtype("uint16")]:
            r = opt_fn(x)

            self.assertEqual(r, False)
            self.assertEqual(cnts.frame_count, 0)  # graph break

    def test_numpy_iter(self):
        # test that iteration over an ndarray produces ndarrays not bare tensors
        def fn(x):
            return [bm for bm in x]

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        proba_map = np.arange(3)[:, None]
        res = opt_fn(proba_map)

        self.assertEqual([type(r) for r in res], [np.ndarray, np.ndarray, np.ndarray])
        self.assertEqual(res, [np.array([0]), np.array([1]), np.array([2])])
        self.assertEqual(cnts.frame_count, 1)

    # cache size limit needs to be larger than the `dtypes` list size
    @torch._dynamo.config.patch(recompile_limit=12)
    def test_dtypes_no_graphbreaks(self):
        dtypes = [
            # floats
            float,
            np.float64,
            "float64",
            np.float32,
            "float32",
            # np.dtype('float64')   # XXX: this is not supported, yet
            # integers
            int,
            "int",
            np.intp,
            np.int32,
            np.uint8,
            # np.dtype('int')       # XXX: as above
        ]

        def fn(dt):
            return np.arange(5, dtype=dt)

        for dtyp in dtypes:
            cnts = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnts)

            val = fn(dtyp)
            opt_val = opt_fn(dtyp)

            self.assertEqual(cnts.frame_count, 1)  # no graph break

    # setting the config value makes the PRNG identical to numpy's
    # NB this may involve a graph break
    @torch._dynamo.config.patch(use_numpy_random_stream=True)
    def test_numpy_random_config_to_numpy(self):
        @torch.compile
        def fn():
            return np.random.uniform(size=13)

        self.assertEqual(fn().shape, (13,))

    def test_inplace_view_on_graph_input(self):
        # graph break when calling methods with inplace_view tag on graph input
        func_args_map = {
            lambda x: x.resize_(6).mul_(2): torch.ones(4),
            lambda x: x.t_().mul_(2): torch.rand(2, 3),
            lambda x: x.transpose_(0, 1).mul_(2): torch.rand(2, 3),
            lambda x: x.squeeze_().mul_(2): torch.rand(1, 2, 3),
            lambda x: x.unsqueeze_(0).mul_(2): torch.rand(2, 3),
            lambda x: x.resize_as_(torch.rand(200, 300)): torch.rand(2, 3),
            lambda x: x.swapaxes_(0, 1).mul_(2): torch.rand(2, 3),
            lambda x: x.swapdims_(0, 1).mul_(2): torch.rand(2, 3),
            lambda x: x.rename_("N", "C").mul_(2): torch.zeros(2, 3),
            lambda x: x.as_strided_((3, 2), (2, 1)).mul_(2): torch.zeros(2, 3),
            lambda x: x.detach_().mul_(2): torch.zeros(2, 3),
        }
        for func, args in func_args_map.items():
            args_clone = args.clone()
            cnts = torch._dynamo.testing.CompileCounter()
            opt_f = torch.compile(func, backend=cnts)
            self.assertTrue(same(func(args).shape, opt_f(args_clone).shape))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 1)  # mul_

    def test_out_variants_with_resizing_on_graph_inputs(self):
        def fn(x, y):
            return torch.cosh(x, out=y) + 1

        x = torch.rand(2, 3)
        y = torch.rand(4)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(fn(x, y), opt_fn(x.clone(), y.clone())))
        self.assertEqual(cnts.frame_count, 1)

    def test_out_variants_with_resizing_on_graph_inputs_with_dynamic(self):
        # https://github.com/pytorch/pytorch/issues/120482
        class CustomModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, inputs):
                return torch.outer(**inputs)

        compile_fn = torch.compile(CustomModel(), backend="eager", fullgraph=True)

        shapes = [(2, 1), (6, 1), (4, 1)]
        for shape in shapes:
            vec1, vec2 = shape
            input_tensor1 = torch.randn(vec1)
            input_tensor2 = torch.randn(vec2)
            out_tensor = torch.empty(shape)
            args = {"input": input_tensor1, "vec2": input_tensor2, "out": out_tensor}
            res = compile_fn(args)
            opt_res = res.clone()  # cuz this is out and we mutate it
            res = CustomModel()(args)
            self.assertEqual(res, opt_res)

    def test_out_variants_with_resizing_on_graph_inputs_with_dynamic1(self):
        mv_op = torch.mv

        def mv_out_op(a, b, c):
            torch.mv(b, c, out=a)
            return a

        def fn(op, *args):
            return op(*args)

        opt_fn = torch.compile(fn, backend="eager")

        ref = fn(mv_op, torch.ones(3, 3), torch.ones(3))
        res = opt_fn(mv_op, torch.ones(3, 3), torch.ones(3))
        self.assertEqual(ref, res)

        ref = fn(mv_out_op, torch.empty(0), torch.ones(3, 3), torch.ones(3))
        res = opt_fn(mv_out_op, torch.empty(0), torch.ones(3, 3), torch.ones(3))
        self.assertEqual(ref, res)

    def test_mutable_mapping_multiple_inheritance(self):
        class MyWeirdDict(collections.abc.MutableMapping, torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self._items = kwargs

            def keys(self):
                return self._items.keys()

            def __getitem__(self, item):
                return self._items[item]

            def __setitem__(self, key, value):
                self._items[key] = value

            def __delitem__(self, item):
                del self._items[item]

            def __len__(self):
                return len(self._items)

            def __iter__(self):
                yield from self._items

            def __hash__(self):
                return hash(id(self))

            def items(self):
                for k, v in self._items.items():
                    yield (k, v)

        @torch.compile(fullgraph=True)
        def to_weird_dict(td):
            return MyWeirdDict(**td)

        d = MyWeirdDict(a=1, b=2, c=3)
        res = to_weird_dict(d)
        self.assertEqual(tuple(d.items()), tuple(res.items()))

    def test_dunder_new_function_inlining(self):
        # https://github.com/pytorch/pytorch/issues/107460

        counters.clear()

        class ModelA(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.tanh(x + 1)

        class ModelB(torch.nn.Module):
            def __new__(cls):
                return ModelA()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Linear(2, 2)

            def forward(self, x):
                other = ModelB()
                return self.layer(x) + other(x)

        x = torch.rand(2, 2)
        m = Model()

        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        ref = m(x)
        res = opt_m(x)
        self.assertTrue(same(ref, res))

    def test_dunder_new_function_inlining1(self):
        class Mock:
            def __new__(cls):
                return super().__new__(cls)

            def __init__(self):
                self.c = 5

            def run(self, x):
                return x * self.c

        def fn(x):
            mock = Mock()
            return mock.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)

        self.assertEqual(fn(x), opt_fn(x))

    def test_dunder_new_function_inlining2(self):
        class Vehicle:
            def __new__(cls, *args, **kwargs):
                return super(Vehicle, cls).__new__(cls)

            def __init__(self, make, model, year):
                self.make = make
                self.model = model
                self.year = year

        class Car(Vehicle):
            def __new__(cls, *args, **kwargs):
                return super(Car, cls).__new__(cls)

            def __init__(self, make, model, year, num_doors):
                super(Car, self).__init__(make, model, year)
                self.num_doors = num_doors

        class ElectricCar(Car):
            def __new__(cls, *args, **kwargs):
                return super(ElectricCar, cls).__new__(cls)

            def __init__(self, make, model, year, num_doors, battery_capacity):
                super(ElectricCar, self).__init__(make, model, year, num_doors)
                self.battery_capacity = battery_capacity

            def run(self, x):
                return torch.sin(x)

        def fn(x):
            ev = ElectricCar("Tesla", "Model S", 2022, 4, "100 kWh")
            return ev.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.randn(4)

        self.assertEqual(fn(x), opt_fn(x))

    def test_dunder_new_function_inlining3(self):
        class Foo:
            def __new__(cls):
                instance = object.__new__(cls)
                instance.a = 3
                return instance

            def __init__(self):
                self.a = 5

            def run(self, x):
                return torch.sin(x) * self.a

        class Bar:
            def __new__(cls):
                instance = object.__new__(Foo)  # not returning a new instance of Bar
                instance.a = 7
                return instance

            def __init__(self):
                self.a = 11  # not called in Bar()

            def run(self, x):
                return torch.sin(x) * self.a

        def fn(x):
            bar = Bar()
            return bar.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_dunder_new_function_inlining4(self):
        class Mock(object):
            def __new__(cls, *args):
                return object.__new__(cls)

            def __init__(self):
                self.a = 5

            def run(self, x):
                return torch.sin(x) * self.a

        def fn(x):
            mock = Mock()
            return mock.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_user_defined_object_class_interaction(self):
        class Foo:
            x = 5

        class Mock:
            # This is a class variable
            class_variable = Foo()

            @classmethod
            def get_class_variable(cls):
                # Accessing the class variable using the cls parameter
                return cls.class_variable.x

            def run(self, x):
                return self.get_class_variable() * x

        def fn(x):
            mock = Mock()
            return mock.run(x)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_multiple_inheritance(self):
        class Base1:
            def __new__(cls):
                return super().__new__(cls)

            def __init__(self):
                super().__init__()
                if not hasattr(self, "base2"):
                    raise ValueError("Wrong MRO tracing")
                self.base1 = 3

        class Base2:
            def __new__(cls):
                return super().__new__(cls)

            def __init__(self):
                super().__init__()
                self.base2 = 5

        class Derived(Base1, Base2):
            def __new__(cls):
                return super().__new__(cls)

            def __init__(self):
                super().__init__()
                self.derived = 7

            def run(self, x):
                return self.base1 * self.base2 * self.derived * x

        def fn(x):
            o = Derived()
            return o.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_class_duner_mro(self):
        class ModuleA(torch.nn.Module):
            pass

        class ModuleB(ModuleA):
            pass

        def fn(x, mod):
            if ModuleA in type(mod).__mro__:
                return x + 1
            else:
                return x - 1

        x = torch.rand(2, 3)
        mod = ModuleB()
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        ref = fn(x, mod)
        res = opt_fn(x, mod)
        self.assertTrue(same(ref, res))

    def test_class_duner_flags(self):
        class ModuleA(torch.nn.ModuleDict, collections.abc.MutableMapping):
            def __hash__(self):
                return id(self)

        def fn(x, mod_class):
            if mod_class.__flags__ & TPFLAGS_MAPPING:
                return x + 1
            else:
                return x - 1

        x = torch.rand(2, 3)
        mod_class = ModuleA
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        ref = fn(x, mod_class)
        res = opt_fn(x, mod_class)
        self.assertTrue(same(ref, res))

        def fn(x, mod):
            if type(mod).__flags__ & TPFLAGS_MAPPING:
                return x + 1
            else:
                return x - 1

        x = torch.rand(2, 3)
        mod = ModuleA()
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        ref = fn(x, mod)
        res = opt_fn(x, mod)
        self.assertTrue(same(ref, res))

    def test_nested_wraps(self):
        def foo(x, y):
            def add(x, y):
                return x + y

            @functools.wraps(add)
            def wrapped_call(x, y):
                return add(x, y)

            return wrapped_call(x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        o = torch.compile(foo, fullgraph=True, backend="eager")(x, y)
        self.assertEqual(o, x + y)

        def foo(x, y):
            def nested_call(x, y):
                def mul(x, y):
                    return x * y

                @functools.wraps(mul)
                def double_nested_call(x, y):
                    return mul(x, y)

                return double_nested_call(x, y)

            return nested_call(x, y)

        o = torch.compile(foo, fullgraph=True, backend="eager")(x, y)
        self.assertEqual(o, x * y)

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
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            self.assertTrue(same(opt_fn(m1, v), correct1))
        for _ in range(10):
            self.assertTrue(same(opt_fn(m2, v), correct2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_global_state_guard_serialization(self):
        GlobalStateGuard = torch._C._dynamo.guards.GlobalStateGuard
        guards = GlobalStateGuard()
        serialized_guards = guards.__getstate__()
        json_guards = json.loads(serialized_guards)

        samples = []
        # Test on non autocast state and autocast cache states.
        self.assertIn("autocast_state", json_guards)
        for key, value in json_guards.items():
            if type(value) is int:
                variant = value + 1
            elif type(value) is bool:
                variant = not value
            elif isinstance(value, dict) and key == "autocast_state":
                variant = value.copy()
                variant["cached_enabled"] = not variant["cached_enabled"]
                continue
            else:
                self.fail(f"Unknown global state type {key}: {value}")
            new_dict = json_guards.copy()
            new_dict[key] = variant
            samples.append(new_dict)

        for sample in samples:
            guards.__setstate__(json.dumps(sample))
            self.assertFalse(guards.check())

        guards.__setstate__(json.dumps(json_guards))
        self.assertTrue(guards.check())

        # Test on autocast states.
        def _test_autocast(dtype):
            with torch.autocast("cpu", dtype):
                guards = GlobalStateGuard()
                serialized_guards = guards.__getstate__()
                json_guards = json.loads(serialized_guards)

                for i, enabled in enumerate(json_guards["autocast_state"]["enabled"]):
                    if enabled:
                        self.assertEqual(
                            type(json_guards["autocast_state"]["dtype"][i]), int
                        )
                        json_guards["autocast_state"]["dtype"][i] += 1
                        guards.__setstate__(json.dumps(json_guards))
                        self.assertFalse(guards.check())

        _test_autocast(torch.float16)
        _test_autocast(torch.float32)
        _test_autocast(torch.float64)
        _test_autocast(torch.bfloat16)

    def test_type_copy(self):
        def fn(seq):
            a, b = seq
            return type(seq)([a + 1, b + 2, a + b])

        args1 = [torch.randn(10), torch.randn(10)]
        args2 = (torch.randn(10), torch.randn(10))
        correct1 = fn(args1)
        correct2 = fn(args2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
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
        opt_fn = torch.compile(fn, backend=cnts)
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
        opt_fn = torch.compile(fn, backend=cnts)
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
        opt_fn = torch.compile(fn, backend=cnts)
        obj1 = opt_fn(x1)
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 9)

    def test_nesteduserfunction_setattr(self):
        x = 0

        def update(y):
            def wrapper():
                x += y

            return wrapper

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            w = update(123)
            w.__wrapped__ = x
            return t.sin(), w

        t = torch.randn(2)
        y, w = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(w.__wrapped__, x)

    def test_object_setattr(self):
        @dataclasses.dataclass
        class A:
            x: torch.Tensor

        def fn1(x) -> None:
            a = A(x)
            object.__setattr__(a, "x", x + 2)
            return a

        x1 = torch.randn(10)
        obj11 = fn1(x1.clone())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch.compile(fn1, backend=cnts, fullgraph=True)
        obj12 = opt_fn1(x1.clone())
        self.assertTrue(same(obj11.x, x1 + 2))
        self.assertTrue(same(obj12.x, x1 + 2))
        self.assertTrue(same(obj11.x, obj12.x))
        self.assertEqual(cnts.frame_count, 1)

        @dataclasses.dataclass(frozen=True)
        class B:
            x: torch.Tensor

        def fn2(x) -> None:
            b = B(x)
            return b

        x2 = torch.randn(10)
        obj21 = fn2(x2.clone())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn2 = torch.compile(fn2, backend=cnts, fullgraph=True)
        obj22 = opt_fn2(x2.clone())
        self.assertTrue(same(obj21.x, x2))
        self.assertTrue(same(obj22.x, x2))
        self.assertTrue(same(obj21.x, obj22.x))
        self.assertEqual(cnts.frame_count, 0)

        @dataclasses.dataclass(frozen=True)
        class C:
            x: torch.Tensor

        def fn3(x) -> None:
            c = C(x)
            object.__setattr__(c, "x", x + 2)
            return c

        x3 = torch.randn(10)
        obj31 = fn3(x3.clone())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn3 = torch.compile(fn3, backend=cnts, fullgraph=True)
        obj32 = opt_fn3(x3.clone())
        self.assertTrue(same(obj31.x, x3 + 2))
        self.assertTrue(same(obj32.x, x3 + 2))
        self.assertTrue(same(obj31.x, obj32.x))
        self.assertEqual(cnts.frame_count, 1)

        @dataclasses.dataclass(frozen=True)
        class D:
            x: torch.Tensor

            def __post_init__(self):
                object.__setattr__(self, "y", self.x + 2)

        def fn4(x) -> None:
            d = D(x)
            return d

        x4 = torch.randn(10)
        obj41 = fn4(x4.clone())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn4 = torch.compile(fn4, backend=cnts, fullgraph=True)
        obj42 = opt_fn4(x4.clone())
        self.assertTrue(same(obj41.x, x4))
        self.assertTrue(same(obj42.x, x4))
        self.assertTrue(same(obj41.x, obj42.x))
        self.assertTrue(same(obj41.y, x4 + 2))
        self.assertTrue(same(obj42.y, x4 + 2))
        self.assertTrue(same(obj41.y, obj42.y))
        self.assertEqual(cnts.frame_count, 1)

    def test_thread_local_setattr(self):
        from threading import local

        loc = local()

        @torch.compile(fullgraph=True)
        def fn(x, l):
            l.x = x
            return x + 1

        x = torch.ones(2, 2)
        fn(x, loc)

        self.assertTrue(loc.x is x)

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
        opt_fn = torch.compile(fn, backend="eager")
        for c in [MyClass1, MyClass2]:
            ref = fn(x, c)
            res = opt_fn(x, c)
            self.assertTrue(same(ref, res))

    def test_super_calling_with_metaclass(self):
        class ExampleMeta(type):
            pass

        class MyClass1(metaclass=ExampleMeta):
            coeff = 4  # Force the constant guard to test source in guards

            @classmethod
            def add(cls, x):
                return x + 1

        class MyClass2(MyClass1):
            @classmethod
            def add(cls, x):
                torch._dynamo.graph_break()
                return x + super().add(x) + super().coeff

        def fn(x, obj):
            return x + obj.add(x)

        x = torch.rand(3)
        obj = MyClass2()
        opt_fn = torch.compile(fn, backend="eager")
        ref = fn(x, obj)
        res = opt_fn(x, obj)
        self.assertTrue(same(ref, res))

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

    def test_function_generic_alias_annotation(self):
        class Variable:
            pass

        def fn(x):
            x = x / 3.0

            def inner(y: list[Variable]):
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

                @torch.compile(backend=cnts, fullgraph=True)
                def fn(counter):
                    return counter() + counter()

                out.append(fn(counter))
                self.assertEqual(cnts.frame_count, 1)
                self.assertEqual(cnts.op_count, 3)
                self.assertFalse(same(counter() + counter(), out[-1]))

        self.assertTrue(same(out[0], out[1]))

    # When we unspecialize float, we wobble this test by changing
    # the op count since previously we would just specialize and constant
    # fold floats into the graph, whereas when we unspecialize we will have
    # ops for item, add, and all other tensorified operations. Since this
    # test really isn't testing that, we purposely specialize floats here.
    @torch._dynamo.config.patch(specialize_float=True)
    def test_closure_out_of_scope_cell(self):
        cell1 = torch.rand(1).item()
        cell2 = torch.rand(3, 3)

        def indirect():
            return direct()

        def direct():
            def inner():
                return cell1 + 1, cell2 + 3

            return inner()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(indirect, backend=cnts)
        result1, result2 = opt_fn()
        self.assertAlmostEqual(cell1 + 1, result1)
        self.assertTrue(torch.allclose(cell2 + 3, result2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    # When we unspecialize float, we wobble this test by changing
    # the op count since previously we would just specialize and constant
    # fold floats into the graph, whereas when we unspecialize we will have
    # ops for item, add, and all other tensorified operations. Since this
    # test really isn't testing that, we purposely specialize floats here.
    @torch._dynamo.config.patch(specialize_float=True)
    def test_closure_out_of_scope_cell_with_mutation(self):
        cell1 = torch.rand(1).item()
        orig1 = cell1
        cell2 = torch.rand(3, 3)
        orig2 = cell2.clone()

        def indirect():
            return direct()

        def direct():
            def inner():
                nonlocal cell1, cell2
                x = cell2 + 1
                cell1 += 1
                cell2 += 10
                x = x + cell2
                return cell1, cell2, x

            return inner()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(indirect, backend=cnts, fullgraph=True)
        for i in range(1, 4):
            result1, result2, _ = opt_fn()
            self.assertAlmostEqual(orig1 + 1 * i, result1)
            self.assertTrue(torch.allclose(orig2 + 10 * i, result2))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 3)
            cnts.clear()

    def test_closure_with_mutation_and_graph_break(self):
        def fn():
            x = torch.zeros(1)

            def subfunc():
                x[0] = backup

            if x[0] >= -1e5:
                pass

            backup = 1
            subfunc()
            return x

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        expected = fn()
        actual = opt_fn()
        self.assertTrue(same(expected, actual))
        self.assertEqual(cnts.frame_count, 2)

    def test_closure_out_of_scope_cell_with_cond(self):
        # Test closure with out-of-scope cell variable, used in a cond
        # where the two branches read different closure variables
        from functorch.experimental.control_flow import cond

        def g(x):
            return x

        class ModuleCondDeep(torch.nn.Module):
            def forward(self, pred, x):
                return self._indirection(pred, x)

            def _indirection(self, pred, x):
                return self.indirection(pred, x)

            def indirection(self, pred, x):
                def true_fn(y):
                    return y + 2

                def false_fn(y):
                    return y - 2

                def shallow(x):
                    return x * 2

                def deep(x):
                    # y = g(x)
                    y = x
                    return cond(
                        x[0][0] > 0,
                        true_fn,
                        false_fn,
                        [y],
                    )

                return cond(pred, shallow, deep, [x])

        mod = ModuleCondDeep()
        opt_mod = torch.compile(mod, backend="eager")
        inp = torch.randn(3, 3)
        exp1 = mod(torch.tensor(False), inp)
        actual1 = opt_mod(torch.tensor(False), inp)
        exp2 = mod(torch.tensor(True), inp)
        actual2 = opt_mod(torch.tensor(True), inp)
        self.assertTrue(torch.allclose(exp1, actual1))
        self.assertTrue(torch.allclose(exp2, actual2))

    def test_closure_write_across_functions(self):
        z = 1
        k = 2

        def create_fn():
            def fn(x):
                nonlocal k, z
                k = z

            return fn

        def update_z_and_run_fn(fn, x):
            nonlocal z
            z = 3
            fn(x)
            return x.cos()

        @torch.compile(backend="eager")
        def foo(x):
            fn = create_fn()
            return update_z_and_run_fn(fn, x)

        x = torch.randn(1)
        foo(x)
        self.assertEqual(3, z)
        self.assertEqual(3, k)

    def test_free_var_and_local_name_collision(self):
        x = 10

        def make_func():
            def func():
                return x

            return func

        @torch.compile(backend="eager")
        def root(t):
            x = 0
            func = make_func()
            res = func()
            return t + 1, x, res

        res = root(torch.ones(1))
        self.assertTrue(torch.allclose(torch.ones(1) + 1, res[0]))
        self.assertEqual(0, res[1])
        self.assertEqual(10, res[2])

    def test_cell_captured_by_existing_func_but_not_root_frame(self):
        x = torch.ones(1)

        def get_inner():
            def inner():
                return x + x

            # Calling `inner` so Dynamo won't skip this frame.
            return inner(), inner

        @torch.compile
        def root():
            return get_inner()

        res, inner = root()
        self.assertTrue(torch.allclose(x + x, res))
        self.assertTrue(torch.allclose(inner(), res))

    def test_writes_to_cells_across_frames1(self):
        # This regression test was added when Dynamo accidentally had both
        # unboxed and normal modeling for pre-existing cells, and failed to
        # account for buffered writes when we read from the unboxed value.
        x = 0

        def inc_x():
            nonlocal x
            x += 1

        class MyObj:
            def inc_x_then_return_x(self, fn):
                fn()
                return x

        @torch.compile(backend="eager")
        def root(t):
            obj = MyObj()
            res = obj.inc_x_then_return_x(inc_x)
            return t + 1, res

        res = root(torch.zeros(1))
        self.assertTrue(torch.allclose(res[0], torch.ones(1)))
        self.assertEqual(res[1], 1)
        self.assertEqual(x, 1)

    def test_writes_to_cells_across_frames2(self):
        # This regression test was added when Dynamo didn't fully account for
        # already established `CellVariable` instance for pre-existing cell,
        # while encountering the same cell again (we should reuse the instance
        # rather than creating a new one). This caused buffered writes to escape
        # the newly created `CellVariable`.
        x = 0

        def inc_x_and_get_x(obj):
            nonlocal x
            x += 1
            return obj.get_x()

        class MyObj:
            def get_x(self):
                return x

        @torch.compile(backend="eager")
        def root(t):
            obj = MyObj()
            res = inc_x_and_get_x(obj)
            return t + 1, res

        res = root(torch.zeros(1))
        self.assertTrue(torch.allclose(res[0], torch.ones(1)))
        self.assertEqual(res[1], 1)
        self.assertEqual(x, 1)

    def test_write_to_cells_with_name_shadowing(self):
        x = 0
        y = x

        def make_x_get_set():
            # NOTE: this `x` is a different cell object than the outer `x`.
            x = y

            def set_x(v):
                nonlocal x
                x = v

            def get_x():
                return x

            return get_x, set_x

        get_x, set_x = make_x_get_set()

        @torch.compile(fullgraph=True)
        def fn(t):
            set_x(42)  # This sets the `x` created within `make_x_get_set`
            res = t + x  # This uses the `x` outside `make_x_get_set`.
            return res

        result = fn(torch.ones(1))
        inner_x = get_x()
        self.assertTrue(torch.allclose(result, torch.ones(1)))
        self.assertEqual(inner_x, 42)

    def test_existing_func_that_creates_capturing_nested_func(self):
        x = 0  # Captured by both `make_get_x` and `root`

        def make_get_x():
            def get_x():
                return x

            return get_x

        @torch.compile(backend="eager", fullgraph=True)
        def root(t):
            get_x = make_get_x()
            res = t + x
            return res, get_x

        res, get_x = root(torch.ones(1))
        self.assertTrue(torch.allclose(res, torch.ones(1)))
        self.assertEqual(0, get_x())
        x += 1
        self.assertEqual(1, get_x())

    def test_input_cell_mutation(self):
        def fn(x):
            x = x.cos()

            def inner():
                return x.sin()

            return inner()

        x = torch.ones(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(res, ref)

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

    def test_typing_typevar(self):
        def fn(x):
            def sumt(y: torch.Tensor) -> torch.Tensor:
                return torch.sum(y)

            def foo(c: typing.Callable[[T], T], y: T) -> T:
                return c(y)

            return foo(sumt, x)

        x = torch.randn(3)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)

    def test_typing_union_and_optional(self):
        def fn(x):
            a = torch.jit.annotate(typing.Dict[str, typing.Optional[torch.Tensor]], {})
            b = torch.jit.annotate(
                typing.Dict[str, typing.Union[torch.Tensor, None]], {}
            )
            return a, b, x + 1

        x = torch.randn(3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_tying_union_new_syntax(self):
        def fn(x):
            def inner1(y: torch.Tensor | None):
                return y

            def inner2(y: None | torch.Tensor):
                return y

            def inner3(y: torch.Tensor | list[int]):
                return y

            return x + 1

        torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(3))

    @unittest.expectedFailure
    def test_typing_union_new_syntax_reconstruct(self):
        def fn(x):
            return (
                x + 1,
                torch.Tensor | None,
                None | torch.Tensor,
                torch.Tensor | list[int],
            )

        torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(3))

    def test_optimize_on_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
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
        optimized_mod = torch.compile(mod, backend=cnts1, fullgraph=True)

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

        @torch.compile(backend=cnts2, fullgraph=True)
        def fn2(x):
            return fn1(x) + 1

        @torch.compile(backend=cnts3, fullgraph=True)
        def fn3(x):
            return torch.relu(fn2(x))

        fn3(torch.randn(4, 5))
        self.assertEqual(cnts2.frame_count, 0)
        self.assertEqual(cnts3.frame_count, 1)
        self.assertEqual(cnts3.op_count, 4)

    def test_nested_optimize_run(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
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

        fn1 = torch.compile(fn, backend=cnts1, fullgraph=True)
        fn2 = torch.compile(fn1, backend=cnts2, fullgraph=True)

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
        fn1 = torch.compile(fn, backend=cnts1, fullgraph=True)
        fn2 = torch.compile(fn1, backend=cnts2, fullgraph=True)
        fn1(torch.randn(4))
        self.assertEqual(cnts1.frame_count, 1)
        torch._dynamo.run()(fn2)(torch.randn(4))
        self.assertEqual(cnts2.frame_count, 0)

    def test_torch_size(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            output_size = torch.Size([10, 10])
            x = x.view(*output_size)
            return (x,)

        x = torch.randn(100, requires_grad=True)
        x_clone = x.clone()
        ref = fn(x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x_clone)

        self.assertTrue(same(ref, res))

    def test_torch_size_numel(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn():
            return torch.Size([10, 8]).numel()

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        num = torch.Size([10, 8]).numel()
        self.assertEqual(opt_fn(), num)

    def test_torch_size_numel_dynamic(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            return x.size().numel()

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.rand(10, 1, 8, 1)
        expect = fn(x)
        self.assertEqual(opt_fn(x), expect)

    def test_shape_type(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            return x + (type(x.shape) == torch.Size)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.zeros(())
        self.assertEqual(opt_fn(x), fn(x))

    def test_size_dim(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, dim):
            return x.size(dim=dim)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.empty([4, 9, 8])
        self.assertEqual(opt_fn(x, 1), 9)
        self.assertEqual(opt_fn(x, -2), 9)

    def test_stride_dim(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, dim):
            return x.stride(dim=dim)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.empty([4, 9, 8])
        self.assertEqual(opt_fn(x, 0), 72)
        self.assertEqual(opt_fn(x, -2), 8)

    def test_torch_seed(self):
        from torch._dynamo.utils import counters

        cnts = torch._dynamo.testing.CompileCounter()
        counters.clear()

        def fn(x):
            attention_seed = int(torch.seed() % sys.maxsize)
            torch.manual_seed(attention_seed)
            return (x,)

        x = torch.randn(10, requires_grad=True)
        ref = fn(x)

        # Python code is needed here, since torch.manual_seed graph-breaks.
        # Refs: https://github.com/pytorch/pytorch/issues/107187
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))
        # Only the torch.seed call is turned into an FX graph.
        self.assertEqual(cnts.op_count, 1)
        self.assertEqual(cnts.frame_count, 1)
        # Graph breaks at manual_seed.
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_is_tensor_like(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x):
            if torch.overrides.is_tensor_like(x):
                return (x * 2,)
            return (torch.ones(10) + x,)

        x = torch.randn(10)
        ref0 = f(x)
        ref1 = f(4)
        opt_f = torch.compile(f, backend=cnts, fullgraph=True)
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
        opt_fn = torch.compile(fn, backend="eager")
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
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
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
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_version_ci(self):
        # temporary test to check that the ci torch version is set correctly
        self.assertTrue(hasattr(torch, "_subclasses"))

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
        opt_getitem = torch.compile(getitem, backend=cnts, fullgraph=True)
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
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(*inps)

        self.assertTrue(same(ref, res))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_source_non_input_grad_access(self):
        # This test creates a model, and accesses the grads
        # from its parameter. This means that within dynamo,
        # the tensor we are reading the grad from HAS a source,
        # but is not known to graphargs.
        cnts = torch._dynamo.testing.CompileCounter()

        class TrivialModel(torch.nn.Module):
            def __init__(self) -> None:
                super(TrivialModel, self).__init__()
                self.linear = torch.nn.Linear(2, 1)

            def forward(self, x):
                return self.linear(x)

        def fn(a, b):
            outs = []
            for param in model.parameters():
                outs.append(torch.ones(param.grad.size()))
            return outs, param.grad + 1

        model = TrivialModel()
        # Eager
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        out = model(a)
        out_sum = out.sum()
        out_sum.backward()
        ref = fn(a, b)

        # Compiled
        model = TrivialModel()
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        out = model(a)
        out_sum = out.sum()
        out_sum.backward()

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(a, b)

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_intermediary_tensor_grad_access(self):
        # This test creates a model, and accesses the grads
        # from its parameters and an entirely intermediary tensor.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(a, b):
            intermediary = torch.ones(2, 2)
            c = a + intermediary
            outs = []
            outs.append(intermediary.grad)
            return outs

        # Eager
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        ref = fn(a, b)

        # Compiled
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(a, b)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_clone_sparse_input(self):
        for layout in [
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ]:
            for sparse_input in self.generate_simple_inputs(
                layout,
                device="cpu",
                dtype=torch.float64,
                index_dtype=torch.int64,
            ):
                # Invoke the dynamo clone input method directly.
                sparse_copy = torch._dynamo.utils.clone_input(sparse_input)
                # Make sure sparse clone is successful.
                self.assertEqual(sparse_input, sparse_copy)

    def test_tensor_is_contiguous(self):
        def fn(x):
            input = torch.randn((1, 16, 1, 1))
            weight = torch.randn((8, 16, 3, 3))
            weight = weight.to(memory_format=x)
            output = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
            return output.is_contiguous(memory_format=x)

        opt_fn = torch.compile(fn, backend="eager")
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
        opt_f1 = torch.compile(f1, backend=cnts)
        opt_f2 = torch.compile(f2, backend=cnts)
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
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.rand(3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_enum_as_dict_key_with_overloaded_str(self):
        class MyEnum(enum.Enum):
            FOO = 10
            BAR = 20

            def __str__(self):
                return self.value

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
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.rand(3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_const_dict_variable_python_type(self):
        from torch._dynamo.variables import ConstantVariable, ConstDictVariable

        make_key = ConstantVariable.create

        d1 = {
            make_key("a"): ConstantVariable.create(10),
            make_key("b"): ConstantVariable.create(20),
        }
        d2 = collections.OrderedDict(
            [
                (make_key("x"), ConstantVariable.create(12)),
                (make_key("y"), ConstantVariable.create(22)),
            ]
        )
        self.assertEqual(ConstDictVariable(d1).python_type(), dict)
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

        sub_of_foo_subclass_var_optim = []
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

    def test_builtin_str_on_user_defined_function(self):
        def another_fn():
            pass

        def fn():
            return "another_fn" in str(another_fn)

        opt_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(opt_fn())

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
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, Foo.FOO)
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, Foo.BAR)
        self.assertEqual(cnts.op_count, 1)

    def test_repeat_interleave_graphbreaks(self):
        def fn_no_breaks(x):
            # no breaks on self_int
            x += 1
            x = torch.repeat_interleave(x, 2, 3)
            x += 1
            return x

        def fn_has_breaks(x):
            # breaks on self_Tensor
            x += 1
            x = torch.repeat_interleave(x, torch.tensor(2), 3)
            x += 1
            return x

        x = torch.randn([4, 16, 1, 64])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn_no_breaks, backend=cnts)
        opt_fn(x)
        self.assertEqual(cnts.frame_count, 1)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn_has_breaks, backend=cnts)
        opt_fn(x)
        self.assertEqual(cnts.frame_count, 2)

    def test_id_guarded_class(self):
        class MyClass1:
            pass

        class MyClass2:
            pass

        def fn(x, y):
            return x + id(y) // 100000

        cnts = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(backend=cnts, fullgraph=True)(fn)
        x = torch.randn(3)
        y = MyClass1
        self.assertEqual(fn(x, y), compiled_fn(x, y))
        self.assertEqual(cnts.frame_count, 1)

        # No recompile if still pass in the original class (MyClass1)
        x = torch.randn(3)
        y = MyClass1
        self.assertEqual(fn(x, y), compiled_fn(x, y))
        self.assertEqual(cnts.frame_count, 1)

        # Have to recompile if pass in new class (MyClass2)
        x = torch.randn(3)
        y = MyClass2
        self.assertEqual(fn(x, y), compiled_fn(x, y))
        self.assertEqual(cnts.frame_count, 2)

    def test_id_guarded_object(self):
        class UserDefinedObject:
            @torch.compile(backend="eager")
            def call(self, x, ref_id):
                self_id = id(self)
                if self_id == ref_id:
                    x = torch.mul(x, 1.0)
                else:
                    x = torch.mul(x, 0)
                return x

        # Make sure we do recompile when id(self) is executed on
        # different self objects.
        x = torch.ones(2)
        obj1 = UserDefinedObject()
        obj1_id = id(obj1)
        self.assertEqual(obj1.call(x, obj1_id), torch.ones(2))

        obj2 = UserDefinedObject()
        # if we do not install ID_MATCH: ___check_obj_id(L['self'], xxx) this fails.
        self.assertEqual(obj2.call(x, obj1_id), torch.zeros(2))

    def test_id_guarded_module(self):
        class M(torch.nn.Module):
            def forward(self, x, ref_id):
                self_id = id(self)
                if self_id == ref_id:
                    x = torch.mul(x, 1.0)
                else:
                    x = torch.mul(x, 0)
                return x

        cnts = torch._dynamo.testing.CompileCounter()

        # Make sure we do recompile when id(self) is executed on
        # different self objects.
        x = torch.ones(2)
        m1 = M()
        m1_id = id(m1)
        opt_m1 = torch.compile(m1, backend=cnts, fullgraph=True)
        self.assertEqual(opt_m1(x, m1_id), torch.ones(2))
        self.assertEqual(opt_m1(x, m1_id), torch.ones(2))

        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

        m2 = M()
        opt_m2 = torch.compile(m2, backend=cnts, fullgraph=True)
        # if we do not install ID_MATCH: ___check_obj_id(L['self'], xxx) this fails.
        self.assertEqual(opt_m2(x, m1_id), torch.zeros(2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 2)

    def test_id_tensor(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.y1 = torch.ones(2)
                self.y2 = torch.zeros(2)
                self.ref_y1_id = id(self.y1)
                self.ref_y2_id = id(self.y2)

            def forward(self, x, ref_id):
                if ref_id == id(self.y1):
                    x = torch.mul(x, self.y1)
                else:
                    x = torch.mul(x, self.y2)
                return x

        cnts = torch._dynamo.testing.CompileCounter()

        x = torch.ones(2)
        m = M()
        opt_m = torch.compile(m, backend=cnts, fullgraph=True)

        self.assertEqual(opt_m(x, m.ref_y1_id), torch.ones(2))
        self.assertEqual(cnts.frame_count, 1)

        self.assertEqual(opt_m(x, m.ref_y2_id), torch.zeros(2))
        self.assertEqual(cnts.frame_count, 2)

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
        opt_m = torch.compile(m, backend=cnts, fullgraph=True)
        opt_m(data, correct_ref_id)
        # Extra op is the recorded equality test (although once
        # the trace is flattened this is dead!)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.op_count, """2""")
        else:
            self.assertExpectedInline(cnts.op_count, """2""")

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        incorrect_ref_id = id(m) + 1
        opt_m = torch.compile(m, backend=cnts, fullgraph=True)
        opt_m(data, incorrect_ref_id)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.op_count, """1""")
        else:
            self.assertExpectedInline(cnts.op_count, """1""")

    def test_inline_func_jump_on_tensor_condition(self):
        def f1(input):
            if input == 0:
                return input + 1
            else:
                return input + 2

        def f2(input):
            return f1(input)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f2 = torch.compile(f2, backend=cnts)
        res1 = opt_f2(torch.tensor([1.0]))
        res2 = opt_f2(torch.tensor([0.0]))

        self.assertEqual(res1, 3)
        self.assertEqual(res2, 1)

    def test_set_discard(self):
        def fn(y):
            x = set(["bar"])
            x.discard("bar")
            x.discard("foo")
            return y + len(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.randn(3)
        self.assertEqual(opt_fn(x), x)
        self.assertEqual(cnts.op_count, 1)

    def test_set_update(self):
        @torch.compile(backend="eager", fullgraph=True)
        def run(x, int_set, int_list):
            int_set.update(map(int, int_list))
            return x + 1

        int_set = set()
        int_list = [1, 2, 1]
        res = run(torch.ones(1), int_set, int_list)
        self.assertTrue(same(res, torch.ones(1) + 1))
        self.assertEqual(int_set, set([1, 2]))
        self.assertEqual(int_list, [1, 2, 1])

    def test_frozenset_torch_func_contains(self):
        funcs = frozenset([torch.add])

        def fn(x, func):
            if func in funcs:
                x = torch.add(x, 1.0)
            x = torch.mul(x, 1.0)
            return x

        x = torch.randn(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, torch.add)
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
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
        opt_f2 = torch.compile(f2, backend=cnts)
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
        opt_f2 = torch.compile(f2, backend=cnts)
        res2 = opt_f2()
        self.assertTrue(same(res1, res2))

    def test_inline_local_dict_clear(self):
        def f(d):
            d.clear()
            return d

        inp = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}
        out = torch.compile(f, backend="eager", fullgraph=True)(inp)
        self.assertEqual(len(out), 0)
        self.assertEqual(len(inp), 0)

    def test_inline_module_attr_dict_clear(self):
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}

            def forward(self):
                self.a.clear()
                return self.a

        m = MyMod()
        out = torch.compile(m, backend="eager", fullgraph=True)()
        self.assertEqual(len(out), 0)
        self.assertEqual(len(m.a), 0)

    def test_inline_user_defined_dict_attr_clear(self):
        class MyMod:
            def __init__(self) -> None:
                self.a = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}

        def f(obj, inp):
            ret = len(obj.a) + inp
            obj.a.clear()
            return obj.a, ret

        m = MyMod()
        before_len = len(m.a)
        t_inp = torch.ones(1)
        d, ret = torch.compile(f, backend="eager", fullgraph=True)(m, t_inp)
        self.assertEqual(len(m.a), 0)
        self.assertEqual(len(d), 0)
        self.assertEqual(ret, t_inp + before_len)

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
        opt_f4 = torch.compile(f4, backend=cnts)
        res2 = opt_f4()
        self.assertTrue(same(res1, res2))

    def test_sample_input(self):
        from torch.testing._internal.common_methods_invocations import SampleInput

        def fn(sample):
            if isinstance(sample.input, torch.Tensor):
                return sample.input * 2
            return torch.zeros(())

        sample = SampleInput(torch.ones(2))
        ref = fn(sample)

        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(sample)

        self.assertTrue(same(ref, res))

    @skipIfWindows(
        msg="TODO(xuhancn): confirm, AssertionError: tensor([0.0290, 0.4019, 0.2598, 0.3666]) is not None"
    )
    def test_release_input_memory(self):
        x = torch.rand([4])
        x_ref = weakref.ref(x)

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def foo(x):
            return x + x

        out = foo(x)
        self.assertTrue(same(out, x + x))
        del x
        self.assertIs(x_ref(), None)

    @skipIfWindows(
        msg="TODO: (xuhancn) conform, AssertionError: Linear(in_features=10, out_features=10, bias=True) is not None"
    )
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

        @torch.compile(backend=no_leak_backend)
        def foo(mod, x):
            return mod(x)

        foo(mod, x)
        del mod
        del x
        self.assertIsNone(mod_ref(), None)
        self.assertIsNone(mod_weight_ref(), None)

    @skipIfWindows(msg="TODO: (xuhancn) conform, AssertionError: False is not true")
    def test_release_scope_memory(self):
        def inner(y):
            y

        inner = torch.compile(inner, backend="eager")

        p_ref = None

        x = torch.randn((10, 10))
        inner(x)

        p_ref = weakref.ref(x)
        self.assertTrue(p_ref() is not None)
        del x
        self.assertTrue(p_ref() is None)

    def test_update_locals_and_stack_uses_shared_cache(self):
        def fn(x):
            perm = [0, 3, 5]
            perm = list(range(min(perm))) + perm
            perm.extend(i for i in range(x.dim()) if i not in perm)
            return perm

        x = torch.rand([2, 2, 2, 2, 2, 2])
        res1 = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

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
        opt_f1 = torch.compile(f1, backend=cnts)
        opt_f2 = torch.compile(f2, backend=cnts)
        res12 = opt_f1(x)
        res22 = opt_f2(a, b)
        self.assertTrue(same(res11, res12))
        self.assertTrue(same(res21, res22))

    def test_replay_side_effects_config(self):
        # Test that replay_side_effects config controls mutation replay
        def fn(x, lst):
            lst.append(x + 1)
            return x * 2

        x = torch.tensor([5.0])

        # Test with replay enabled (default)
        lst_with_replay = []
        opt_fn_with_replay = torch.compile(fn, backend="eager")
        result1 = opt_fn_with_replay(x, lst_with_replay)
        self.assertEqual(len(lst_with_replay), 1)  # Mutation should be replayed
        self.assertTrue(same(result1, x * 2))

        torch._dynamo.reset()

        # Test with replay disabled
        lst_without_replay = []
        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="warn"
        ):
            opt_fn_without_replay = torch.compile(fn, backend="eager")
            result2 = opt_fn_without_replay(x, lst_without_replay)
            self.assertEqual(
                len(lst_without_replay), 0
            )  # Mutation should NOT be replayed
            self.assertTrue(same(result2, x * 2))

        torch._dynamo.reset()
        lst_without_replay = []
        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="error"
        ):
            opt_fn_without_replay = torch.compile(fn, backend="eager")
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape(
                    "While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: [\"L['lst']\"]"
                ),
            ):
                _ = opt_fn_without_replay(x, lst_without_replay)

    def test_replay_side_effects_model_attr(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = 4

            def forward(self, x):
                return x.cos()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = 4
                self.tensor = None
                self.bar = Bar()

            def forward(self, x):
                self.const = 5
                self.tensor = x.sin()
                res = self.bar(x)
                return x.cos() + res.sum() + self.tensor

        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="error"
        ):
            foo = Foo()
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape(
                    "While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: [\"L['self']\"]"
                ),
            ):
                torch.compile(foo, fullgraph=True)(torch.randn(4, 4))

        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="silent"
        ):
            foo_v2_compile = Foo()
            foo_v2_eager = Foo()
            inp = torch.randn(4, 4)
            res = torch.compile(foo_v2_compile, fullgraph=True)(torch.randn(4, 4))
            self.assertEqual(foo_v2_compile.tensor, None)
            self.assertEqual(foo_v2_compile.const, 4)
            self.assertEqual(foo_v2_compile.bar.const, 4)
            same(res, foo_v2_eager(inp))

    def test_replay_side_effects_input_mut(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = 4
                self.tensor = None

            def forward(self, x):
                x.add_(5)
                return x.cos()

        # This is ok because we actually capture the graph which
        # has mutation. In export, we never retrace the actual
        # gm so we won't see any mutation applied to inputs
        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="error"
        ):
            foo = Foo()
            torch.compile(foo, fullgraph=True)(torch.randn(4, 4))

    def test_list_append_return_none(self):
        def fn(x):
            alist = []
            blist = alist.append(x + 1)
            return alist, blist

        x = torch.tensor([2.3])
        res = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x)
        self.assertEqual(res, res2)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_tensor_ctor_list_of_tensor(self):
        def fn(x):
            return torch.tensor([x], dtype=torch.int64)

        x = torch.tensor(20)
        res = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x)
        self.assertEqual(res, res2)
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_types(self):
        def fn(dtype, tensor_type):
            x = torch.empty(4, dtype=dtype)
            assert isinstance(x, tensor_type)

        opt_fn = torch.compile(fn, backend="eager")
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
        opt_f = torch.compile(f, backend=cnts)
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
        y = torch.compile(model, backend="eager", fullgraph=True)(x)

        self.assertEqual(y, 11)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_item_changes(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                z = torch.max(x)
                return z.int().item()

        x = torch.tensor([[10.6763, 11.7445, -2.2369]])
        model = MyMod()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
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
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        y = opt_model(x)
        z = opt_model(torch.tensor([[y - 5, y + 50], [y + 5, y - 50]]))

        self.assertEqual(y, 11)
        self.assertEqual(z, 61)

    @unittest.skip("https://github.com/pytorch/pytorch/issues/99726")
    def test_cross_entropy_loss_fancy_ctor1(self):
        rand_5 = torch.randn(5)
        rand_3_5 = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        loss = torch.nn.CrossEntropyLoss(
            weight=rand_5, reduce=False, label_smoothing=0.5
        )
        opt_loss = torch.compile(loss, backend="eager", fullgraph=True)
        input = rand_3_5
        dynamo_output = opt_loss(input, target)

        loss = torch.nn.CrossEntropyLoss(
            weight=rand_5, reduce=False, label_smoothing=0.5
        )
        input = rand_3_5
        output = loss(input, target)

        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_cross_entropy_loss_fancy_ctor2(self):
        rand_3_5 = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        loss = torch.nn.CrossEntropyLoss(reduce=False, label_smoothing=0.5)
        opt_loss = torch.compile(loss, backend="eager", fullgraph=True)
        input = rand_3_5
        dynamo_output = opt_loss(input, target)

        loss = torch.nn.CrossEntropyLoss(reduce=False, label_smoothing=0.5)
        input = rand_3_5
        output = loss(input, target)

        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_cross_entropy_loss_simple_ctor(self):
        output = None
        rand_3_5 = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        loss = torch.nn.CrossEntropyLoss()
        opt_loss = torch.compile(loss, backend="eager", fullgraph=True)
        input = rand_3_5
        dynamo_output = opt_loss(input, target)

        loss = torch.nn.CrossEntropyLoss()
        input = rand_3_5
        output = loss(input, target)

        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_repr(self):
        class Config:
            def __repr__(self):
                return "Config()"

        def forward(x, config):
            return x * len(repr(config))

        config = Config()
        x = torch.randn(2, 2)

        compiled = torch.compile(forward, fullgraph=True)
        compiled(x, config)

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
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
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

        @torch.compile(backend=my_compiler)
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
            def __init__(self) -> None:
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                self.drop = torch.nn.Dropout(embd_pdrop)

            def forward(self, x):
                return x

        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
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

        @torch.compile(backend="eager", fullgraph=True)
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

        @torch.compile(backend="eager", fullgraph=True)
        def fn1():
            return list(mod.named_parameters(prefix="foo"))

        params = fn1()

        self.assertEqual(len(actual_params), len(params))
        for idx in range(len(params)):
            k_a, v_a = actual_params[idx]
            k, v = params[idx]
            self.assertEqual(k_a, k)
            self.assertTrue(torch.allclose(v_a, v))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_module_complex_iter(self):
        n_embd = 768
        block_size = 128
        vocab_size = 65
        embd_pdrop = 0.1

        class FakeGPT(torch.nn.Module):
            def __init__(self) -> None:
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
                assert t <= self.block_size, (
                    "Cannot forward, model block size is exhausted."
                )

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
                        fpn = f"{mn}.{pn}" if mn else pn
                        self.names.append(fpn)

        # Test plain recurse
        model_a = FakeGPT()
        model_a.foo()
        a_names = model_a.names

        model_b = FakeGPT()
        opt_model_b = torch.compile(model_b, backend="eager", fullgraph=True)
        opt_model_b.foo()

        self.assertEqual(a_names, model_b.names)

        # Test with prefix
        model_a = FakeGPT()
        model_a.foo(prefix="abc")
        a_names = model_a.names

        model_b = FakeGPT()
        opt_model_b = torch.compile(model_b, backend="eager", fullgraph=True)
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
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x, m)
        self.assertEqual(ref, res)

        # Test now the other path
        ref = fn(x, x)
        res = opt_fn(x, x)
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
        opt_fn = torch.compile(fn, backend=cnts)
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
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def _optimize_then_check_exp(
        self, foo, args, cnt, exp_out, exp_frame_count, exp_n_cached_backend
    ):
        opt_out = torch.compile(foo, backend=cnt)(*args)
        self.assertEqual(exp_out, opt_out)
        self.assertEqual(cnt.frame_count, exp_frame_count)

    def test_backend_match_guard(self):
        x = torch.randn([3, 4])

        def foo(x):
            return x.sin() + x.cos()

        def foo_graph_break(x):
            a = x.sin()
            torch._dynamo.graph_break()
            b = x.cos()
            return a + b

        eager_record_backend = torch._dynamo.testing.EagerAndRecordGraphs()
        backends = [eager_record_backend, "eager"]

        # We intentionally don't reset dynamo for each backend so that we can test
        # 1. dynamo doesn't recompile when backend stays the same, i.e. frame_count doesn't increase
        # 2. dynamo recompiles when backend changes, i.e. frame_count is non-zero for next backend
        def test_recompile(foo, *, exp_frame_count):
            eager_result = foo(x)
            for i, backend in enumerate(backends):
                cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
                # Run opt_f multiple times to make sure dynamo doesn't recompile.
                # Specifically, frame_count doesn't increase
                # the number of cached backends is i + 2 because we have the optimizing backend + None
                self._optimize_then_check_exp(
                    foo, (x,), cnt, eager_result, exp_frame_count, i + 2
                )
                self._optimize_then_check_exp(
                    foo, (x,), cnt, eager_result, exp_frame_count, i + 2
                )
                self._optimize_then_check_exp(
                    foo, (x,), cnt, eager_result, exp_frame_count, i + 2
                )

        test_recompile(foo, exp_frame_count=1)
        torch._dynamo.reset()
        test_recompile(foo_graph_break, exp_frame_count=2)

    def test_backend_match_guard_multi_threads(self):
        x = torch.randn([3, 4])

        def foo(x):
            return x.sin() + x.cos()

        def compile_then_check_exp(foo, args, cnt, eager_result, exp_frame_count):
            for i in range(3):
                opt_out = torch.compile(foo, backend=cnt)(*args)
                self.assertEqual(opt_out, eager_result)
            self.assertEqual(cnt.frame_count, exp_frame_count)
            thread_success[threading.current_thread()] = True

        eager_record_backend = torch._dynamo.testing.EagerAndRecordGraphs()
        backends = [eager_record_backend, "eager"]

        # Test dynamo recompiles but only caches a single backend for each thread
        eager_result = foo(x)
        # cnt and None
        exp_frame_count = 1
        threads = []
        thread_success = {}
        for i, backend in enumerate(backends):
            cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
            thread = threading.Thread(
                target=compile_then_check_exp,
                args=(
                    foo,
                    (x,),
                    cnt,
                    eager_result,
                    exp_frame_count,
                ),
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        self.assertEqual(len(thread_success), len(threads))

    def test_dynamo_min_operator_with_shape(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x, a):
            return min(x.shape[0], a)

        result = f(torch.ones(6), 3)
        self.assertEqual(result, 3)

    def test_onnx_shape_as_tensor(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return 1 + torch._shape_as_tensor(x)[0]

        gm, _ = torch._dynamo.export(f)(torch.ones(6))

        input_one_dim = torch.ones(6)
        input_two_dims = torch.ones(7, 4)
        self.assertEqual(f(input_one_dim), 7)
        self.assertEqual(f(input_two_dims), 8)
        self.assertEqual(f(input_two_dims), 8)

        @torch.compile(backend="eager", fullgraph=True)
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

        opt_fn = torch.compile(f, backend="eager")
        a = opt_fn(torch.tensor(False), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.cos(torch.tensor([0.25, 0.25])), a))
        b = opt_fn(torch.tensor(True), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), b))

    def test_cond_with_quantization(self):
        from functorch.experimental.control_flow import cond

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                example_inputs = (torch.randn(5, 5),)
                self.model = torch.nn.Linear(5, 5)
                self.quantized_model = prepare_qat_fx(
                    self.model, qconfig_dict, example_inputs=example_inputs
                )

            def forward(self, pred, x):
                def true_fn(x):
                    return x.sin() + self.quantized_model(x)

                def false_fn(x):
                    return x.cos() + self.model(x)

                return cond(pred, true_fn, false_fn, [x])

        module = MyModule()
        opt_m = torch.compile(module, backend="eager", fullgraph=True)
        x = torch.rand((5, 5))
        pred = torch.tensor(True)
        self.assertTrue(same(module(pred, x), opt_m(pred, x)))
        pred = torch.tensor(False)
        self.assertTrue(same(module(pred, x), opt_m(pred, x)))

    def test_map_with_quantization(self):
        from functorch.experimental.control_flow import map

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                example_inputs = (torch.randn(5, 5),)
                self.model = torch.nn.Linear(5, 5)
                self.quantized_model = prepare_qat_fx(
                    self.model, qconfig_dict, example_inputs=example_inputs
                )

            def forward(self, x):
                def body(x):
                    return x.sin() + self.quantized_model(x)

                return map(body, x)

        module = MyModule()
        opt_m = torch.compile(module, backend="eager", fullgraph=True)
        x = torch.rand((5, 5))
        self.assertTrue(same(module(x), opt_m(x)))

    def test_cond_side_effects(self):
        from functorch.experimental.control_flow import cond

        c = 0

        def true_fn(x):
            return x - c

        def false_fn(x):
            return x + c

        def f(pred, x):
            nonlocal c
            c = 1
            return cond(pred, true_fn, false_fn, [x])

        opt_fn = torch.compile(f, backend="eager")
        c = 0
        a = opt_fn(torch.tensor(False), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.tensor([1.25, 1.25]), a))

    def test_map_side_effects(self):
        from functorch.experimental.control_flow import map

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.tensor(1)

            def forward(self, xs):
                def body(x):
                    self.w += 1
                    return x

                return map(body, xs)

        mod = Module()

        error_message = ""
        if torch._dynamo.config.inline_inbuilt_nn_modules:
            error_message = (
                "map doesn't work unless it is captured completely with torch.compile"
            )
        else:
            error_message = "Can't inplace modify module params/buffers"

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError, error_message
        ):
            opt_fn = torch.compile(mod, backend="eager", fullgraph=True)
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
        opt_fn = torch.compile(f, backend=cc)
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

        graph, guard = torch._dynamo.export(f)(
            torch.tensor(False), torch.tensor(True), torch.tensor([0.25, 0.25])
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
            return x.clone()

        def false_fn(x):
            return x.sin()

        def f(pred, x):
            return cond(pred, true_fn, false_fn, [x])

        graph, guard = torch._dynamo.export(f)(
            torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        true_mirror = graph(torch.tensor(True), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.tensor([0.25, 0.25]), true_mirror))
        true_mirror_2 = graph(torch.tensor(True), torch.tensor([0.33, 0.33, 0.33]))
        self.assertTrue(same(torch.tensor([0.33, 0.33, 0.33]), true_mirror_2))

        false_sin = graph(torch.tensor(False), torch.tensor([0.5, 0.5]))
        self.assertTrue(same(torch.sin(torch.tensor([0.5, 0.5])), false_sin))

    def test_enum_guards(self):
        class MyEnum(enum.Enum):
            FOO = 10
            BAR = 20

        def fn(x, y):
            if y == MyEnum.FOO:
                return x + 1
            else:
                return x - 1

        x = torch.rand(3)
        y = MyEnum.BAR
        ref = fn(x, y)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_enum_method(self):
        class Bool(enum.IntEnum):
            TRUE = enum.auto()
            FALSE = enum.auto()

            def is_true(self, x):
                # Return `x + 1` to make sure Dynamo actually traced into this,
                # rather than invoking it.
                return self == Bool.TRUE, x + 1

        def f(x, e):
            cond, y = e.is_true(x)
            if cond:
                return y + 2
            else:
                return y - 2

        opt_f = torch.compile(fullgraph=True)(f)
        args = [torch.zeros(1), Bool.TRUE]
        ref_out = f(*args)
        opt_out = opt_f(*args)
        self.assertTrue(same(ref_out, opt_out))

    def test_enum_subclass(self):
        # Copied from inspect.py

        class _ParameterKind(enum.IntEnum):
            POSITIONAL_ONLY = "positional-only"

            def __new__(cls, description):
                value = len(cls.__members__)
                member = int.__new__(cls, value)
                member._value_ = value
                member.description = description
                return member

            def __str__(self):
                return self.name

        _POSITIONAL_ONLY = _ParameterKind.POSITIONAL_ONLY

        def fn(x):
            _ParameterKind(_POSITIONAL_ONLY)
            return torch.cos(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_duplicate_graph_break_log(self):
        torch._logging.set_logs(graph_breaks=True)

        @torch.compile(backend="eager")
        def f1(a, b):
            f2(a, b)

        def f2(a, b):
            c = a + b
            print("break")
            return a + b + c

        @torch.compile(backend="eager")
        def g1(a, b):
            g2(a, b)

        def g2(a, b):
            c = a + b
            print("break")
            return a + b + c

        def count_graph_break_msgs(msgs):
            return sum("Graph break in user code" in msg for msg in msgs)

        with (
            self.assertLogs(logger="torch._dynamo", level=logging.DEBUG) as log,
            torch._dynamo.config.patch(verbose=True),
        ):
            f1(torch.randn(10), torch.randn(10))
            self.assertGreater(count_graph_break_msgs(log.output), 1)

        with (
            self.assertLogs(logger="torch._dynamo", level=logging.DEBUG) as log,
            torch._dynamo.config.patch(verbose=False),
        ):
            g1(torch.randn(10), torch.randn(10))
            self.assertEqual(count_graph_break_msgs(log.output), 1)

        # reset logging state
        torch._logging.set_logs()

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
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, y)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_generate_tensor_from_list_of_numpy_primitive_type(self):
        # Test sth like torch.LongTensor(list(np.int64, np.int64, ...))
        def fn():
            x = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
            y = [x[0], x[2], x[4]]
            return torch.LongTensor(y)

        ref = fn()
        res = torch.compile(fullgraph=True)(fn)()
        self.assertEqual(ref, res)

    def test_object_classmethod(self):
        class C:
            @classmethod
            def fn(cls, x):
                return x + x

        @torch.compile(backend="eager", fullgraph=True)
        def f():
            return C().fn(torch.ones(2, 3))

        self.assertTrue(torch.allclose(f(), torch.tensor([2.0])))

    def test_object_staticmethod(self):
        class C:
            @staticmethod
            def fn(x):
                return x + x

        @torch.compile(backend="eager", fullgraph=True)
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
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_user_function_variable_supports_type_abcmeta_argument(self):
        class Foo(metaclass=abc.ABCMeta):
            @abc.abstractclassmethod
            def read(self):  # noqa: B027
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
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_user_function_variable_supports_function_argument(self):
        # Test user defined function default arguments can be:
        # 1, user defined functions (e.g, add1)
        # 2, torch functions (e.g, torch.sin)
        # 3, python builtin functions (e.g, operator.neg)
        def add1(x):
            return x + 1

        def gn(x, f1=add1, f2=torch.sin, f3=operator.neg):
            return f3(f2(f1(x)))

        def fn(x):
            return gn(x)

        x = torch.randn(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
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
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, m)
        self.assertTrue(torch.allclose(ref, res))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_repro_graph_breaks_in__get_item_by_idx(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod = torch.nn.Sequential(
                    torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
                )

            def forward(self, x):
                return self.mod[0](x)

        m = Mod()
        graph, _ = torch._dynamo.export(m)(torch.randn(3, 3))

    @torch._dynamo.config.patch(guard_nn_modules=True)
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
            graph, _ = torch._dynamo.export(m)(x)
            dynamo_result = graph(x)
            self.assertTrue(same(real, dynamo_result))

    @torch._dynamo.config.patch(guard_nn_modules=True)
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
            graph, _ = torch._dynamo.export(m)(x)
            dynamo_result = graph(x)
            self.assertTrue(same(real, dynamo_result))

    def test_error_on_nested_fx_trace(self):
        input = torch.rand(2, 3)

        def f(x):
            x + x

        real = f(input)

        optimized = torch.compile(f, backend="eager")
        self.assertTrue(same(optimized(input), real))

        with self.assertRaisesRegex(RuntimeError, "Detected that you are using FX"):
            gm = torch.fx.symbolic_trace(optimized)

    @patch.object(torch._dynamo.config, "error_on_nested_fx_trace", False)
    def test_no_error_on_nested_fx_trace(self):
        input = torch.rand(2, 3)

        def f(x):
            x + x

        real = f(input)

        optimized = torch.compile(f, backend="eager")
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
        optimized = torch.compile(f, backend="eager")
        opt = optimized(input)
        self.assertTrue(same(opt, real))

    def test_inference_mode(self):
        @torch.inference_mode()
        def func(x, y):
            return x.add(1.0) + y

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=True)
        ref = func(x, y)
        opt_func = torch.compile(func, backend="eager")

        x1 = torch.ones(4, requires_grad=True)
        res = opt_func(x1, y)
        self.assertTrue(same(ref, res))
        self.assertTrue(same(x, x1))

    def test_inference_mode_param(self):
        def fn(x):
            p = torch.nn.Parameter(x, requires_grad=False)
            return x * p

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        with torch.inference_mode():
            x = torch.rand(4)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

    def test_if_cond_nn_mod1(self):
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
        opt_model = torch.compile(model, backend="eager", fullgraph=True)

        x = torch.rand(4)
        ref = model(x)
        res = opt_model(x)
        self.assertTrue(same(ref, res))

        model = MockModule(output_relu=False)
        opt_model = torch.compile(model, backend="eager", fullgraph=True)

        x = torch.rand(4)
        ref = model(x)
        res = opt_model(x)
        self.assertTrue(same(ref, res))

    def test_if_cond_nn_mod2(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Sequential()

            def forward(self, x):
                if self.layer:
                    return x + 1
                else:
                    return x - 1

        model = MockModule()
        x = torch.rand(4)
        ref = model(x)
        opt_model = torch.compile(backend="eager")(model)
        res = opt_model(x)
        self.assertTrue(same(ref, res))

    def test_if_cond_nn_mod3(self):
        def fn(x):
            if torch.nn.ModuleList():
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        ref = fn(x)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x)
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
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
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
                self.x = 1.2
                return self.x

        def fn(a, obj):
            if not obj:
                return a + obj.x
            else:
                return a - obj.x

        x = torch.rand(4)
        obj = MyObj(0.5)
        opt_fn = torch.compile(fn, backend="eager")
        try:
            opt_fn(x, obj)
            self.assertFalse(True)
        except TypeError as e:
            self.assertIn("__bool__ should return bool, returned float", str(e))

    def test_unpack_tensor_shape_mismatch(self):
        @torch.compile(backend="eager")
        def f1(x):
            a, b = x
            return torch.sin(a + b)

        x = torch.tensor(2.0)
        with self.assertRaisesRegex(AssertionError, "Can't unpack scalar tensors"):
            f1(x)

        x = torch.tensor([2.0])
        with self.assertRaisesRegex(
            AssertionError, "Can't unpack a tensor of 1 rows into a tuple of 2 elements"
        ):
            f1(x)

        @torch.compile(backend="eager")
        def f2(x):
            (a,) = x
            return torch.sin(a + 1)

        x = torch.tensor(2.0)
        with self.assertRaisesRegex(AssertionError, "Can't unpack scalar tensors"):
            f2(x)

        x = torch.tensor([2.0])
        self.assertTrue(same(f2(x), torch.sin(x[0] + 1)))

    def test_if_cond_user_defined_object3(self):
        # obj.__bool__ is not existed, but obj.__len__ exists
        class A:  # noqa: B903
            def __init__(self, x):
                self.x = x

            def __len__(self):
                return len(self.x)

        # obj.__bool__ takes precedence over obj.__len__
        class B:
            def __init__(self, x):
                self.x = x

            def __bool__(self):
                return False

            def __len__(self):
                return len(self.x)

        def fn(x, obj):
            if not obj:
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        obj1 = A([1, 2, 3])
        obj2 = A([])
        obj3 = B([1, 2, 3])
        obj4 = B([])
        for obj in [obj1, obj2, obj3, obj4]:
            ref = fn(x, obj)
            res = opt_fn(x, obj)
            self.assertTrue(same(ref, res))

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
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, obj)
        self.assertTrue(same(ref, res))

    def test_variable_tracker_recursively_contains(self):
        # VariableTracker.recursively_contains should be updated correctly when mutation happens
        def fn(x):
            data = [[None] * 3] * 3
            for i in range(3):
                if i == 0:
                    data[0][i] = x
                else:
                    data[0][i] = data[0][i - 1] + 1
            return data[0][-1]

        x = torch.rand(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_disable_flag(self):
        cnt = torch._dynamo.testing.CompileCounter()

        with patch.dict(os.environ, {"TORCH_COMPILE_DISABLE": "1"}):

            def fn(x, y):
                x = x + 1
                y = y + 1

            opt_fn = torch.compile(backend=cnt)

        self.assertEqual(cnt.frame_count, 0)

    def test_is_compiling(self):
        def f1():
            if torch._dynamo.is_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        def f2():
            if torch._utils.is_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        def f3():
            if torch.compiler.is_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        def f4():
            if torch.compiler.is_dynamo_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        for f in [f1, f2, f3, f4]:
            opt_f = torch.compile(f, backend="eager")

            self.assertEqual(f(), torch.zeros(2, 2))
            self.assertEqual(opt_f(), torch.ones(2, 2))

    def test_torch_generator_set_state(self):
        def fn():
            default_state = torch.default_generator.get_state()
            x = torch.rand([2, 3])
            if default_state.dtype != "float32":
                x = x * 2
            torch._dynamo.graph_break()
            torch.default_generator.set_state(default_state)
            y = torch.rand([2, 3])
            return x, y

        opt_fn = torch.compile(fn, backend="eager")
        x, y = opt_fn()
        self.assertEqual(x, y * 2)

    def test_torch_distributions_lazy_property(self):
        def fn(x):
            return torch.distributions.Categorical(probs=x).entropy()

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.rand([4, 4])
        self.assertEqual(opt_fn(x), fn(x))

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
            not torch._dynamo.config.specialize_int
            and not torch._dynamo.config.assume_static_by_default
        ):
            # we didn't actually test guard_failure_fn here but whatever,
            # nice to see no guard failure on the test
            self.assertTrue(guard_failure is None)
        else:
            self.assertTrue(guard_failure is not None)

    def test_guard_failure_fn_shape_control(self):
        def fn(x, y):
            if x.shape[0] < 4:
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
        first_guard_failure = guard_failure[0].partition("\n")[0]
        self.assertIn(
            """tensor 'x' size mismatch at index 0. expected 2, actual 5""",
            first_guard_failure,
        )

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

        if torch._dynamo.config.assume_static_by_default:
            self.assertIn(
                """tensor 'x' size mismatch at index 0. expected 2, actual 3""",
                guard_failure[0],
            )
        else:
            self.assertTrue(guard_failure is None)

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
        self.assertIn(
            """size mismatch at index 0. expected 10, actual 9""",
            guard_failure[0],
        )

    def test_no_guard_for_unused_sym_node_fstring(self):
        def fn(x):
            f"{x.shape[0]}"
            return x.sin()

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", guard_fail_fn=guard_failures, dynamic=True
        )(fn)
        args1 = torch.randn(10, 11)
        out = fn(args1)
        opt_out = opt_fn(args1)
        self.assertEqual(out, opt_out)

        # We change x.shape[0] to test whether it's guarded
        args2 = torch.randn(9, 11)
        out = fn(args2)
        opt_out = opt_fn(args2)
        self.assertEqual(out, opt_out)
        self.assertEqual(guard_failure, None)

    def test_guard_sym_node_fstring_when_used(self):
        def fn(x):
            # assign fstring to a variable causes the fstring to be used,
            # which realizes the variable tracker.
            f_str = f"{x.shape[0]}"
            return x.sin(), f_str

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", guard_fail_fn=guard_failures, dynamic=True
        )(fn)
        args1 = torch.randn(10, 11)
        out = fn(args1)
        opt_out = opt_fn(args1)
        self.assertEqual(out, opt_out)

        # We change x.shape[0] to test whether it's guarded
        args2 = torch.randn(9, 11)
        out = fn(args2)
        opt_out = opt_fn(args2)
        self.assertEqual(out, opt_out)
        self.assertTrue(guard_failure is not None)
        self.assertIn("""tensor 'x' size mismatch at index 0""", guard_failure[0])

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

        for guard in all_guards:
            # This guard was created
            self.assertTrue(guard.name != "nested_fn.__closure__[0].cell_contents")

    @unittest.skipIf(not TEST_MULTIGPU, "need multiple GPU")
    def test_symint_as_device_kwarg_multi_gpu(self):
        def fn(rank):
            # -2 to make device id smaller for easier testing on CI
            return torch.ones(10, device=rank.size(0) - 2)

        x = torch.randn(2)
        out = fn(torch.randn(2))

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", guard_fail_fn=guard_failures, dynamic=True
        )(fn)
        self.assertEqual(out, opt_fn(x))

        x = torch.randn(3)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertTrue(guard_failure is not None)
        self.assertIn("""tensor 'rank' size mismatch at index 0""", guard_failure[0])

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "Test requires CUDA or XPU.")
    def test_symint_as_device_kwarg_non_strict_export(self):
        class Mod(torch.nn.Module):
            def forward(self, x):
                # -2 to make device id 0 for easier testing on CI
                return torch.ones(10, device=x.size(0) - 2)

        x = torch.randn(2)
        m = Mod()
        d1 = torch.export.Dim("d1", max=2048)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError, r"Constraints violated \(d1\)"
        ):
            ep = torch.export.export(
                m, (x,), dynamic_shapes={"x": {0: d1}}, strict=False
            )

    def test_call_parent_non_class_methods_from_child(self):
        class A:
            a = 4

            def add(self, x):
                return x + 10

            def mul(self, x):
                return x * 0.1

        class B(A):
            coeff = 4

            def add(self, x):
                return x + 20

            @classmethod
            def cube(cls, x):
                return cls.coeff * x * x * x

            def mul(self, x):
                return super().mul(x) * x * 0.2

        class C(B):
            def add(self, x):
                b = super().cube(x)
                c = A.add(self, x)
                d = B.mul(self, x)
                e = super(B, self).add(x)
                f = super().a * x
                return b + c + d + e + f

        x = torch.rand(4)
        fn = C().add
        ref = fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnt.frame_count, 1)

        # Check recompilation
        A.a = 5
        ref = fn(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        # Ensure that super guard checks are working as expected
        res = opt_fn(x)
        self.assertEqual(cnt.frame_count, 2)

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
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_tuple_from_tuple_iter(self):
        def inner_fn(*args):
            acc = torch.ones(10, 10)
            for arg in args:
                acc.add_(arg)

            return acc

        @torch.compile(backend="eager")
        def fn(inputs, params):
            y = tuple(inputs) + tuple(params)
            return inner_fn(*y)

        inputs = [torch.randn(10, 10) for _ in range(3)]

        fn(inputs, iter(tuple(inputs)))

        def fn(params):
            y = tuple(params)
            return inner_fn(*y)

        opt_fn = torch.compile(fn, backend="eager")
        inputs = [torch.randn(10, 10) for _ in range(3)]
        self.assertTrue(same(fn(iter(tuple(inputs))), opt_fn(iter(tuple(inputs)))))

        # Force recompilation
        inputs = [torch.randn(10, 10) for _ in range(4)]
        self.assertTrue(same(fn(iter(tuple(inputs))), opt_fn(iter(tuple(inputs)))))

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_argwhere_with_dynamic_shapes(self):
        def fn(
            tensor: torch.Tensor,
            mapping: torch.Tensor,
        ) -> torch.Tensor:
            xx, yy = torch.meshgrid(mapping, tensor, indexing="ij")
            indices = torch.argwhere(xx == yy)

            mapped_values = torch.zeros_like(tensor)
            mapped_values[indices[:, 1]] = indices[:, 0]

            return mapped_values

        tensor = torch.tensor([1, 2, 3, 5, 6, 7])
        mapping = torch.tensor([0, 3, 4, 5, 7])
        opt = torch.compile(fn, fullgraph=True)
        self.assertEqual(fn(tensor, mapping), opt(tensor, mapping))

    def test_torch_package_working_with_trace(self):
        # from torch._dynamo.test_case import run_tests

        inputs = [torch.randn([2, 2]), torch.randn([2, 2])]

        optimized_model = torch.compile(
            MyPickledModule(torch.randn([2, 2])), backend="eager"
        )
        from torch import package

        tmp_root = tempfile.gettempdir()
        path = os.path.join(tmp_root, "MyPickledModule.pt")
        package_name = "MyPickledModule"
        resource_name = "MyPickledModule.pkl"

        model = MyPickledModule(torch.randn([2, 2]))

        with package.PackageExporter(path) as exp:
            exp.extern("**")
            exp.save_pickle(package_name, resource_name, model)

        imp = package.PackageImporter(path)
        loaded_model = imp.load_pickle(package_name, resource_name)

        optimized_loaded_model = torch.compile(loaded_model, backend="eager")(*inputs)

    def test_precompile_entry_hit(self):
        from torch._C._dynamo.eval_frame import (
            _load_precompile_entry,
            _reset_precompile_entries,
        )

        def fn(x):
            return x + 1

        def injected(x):
            return x + 42

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn)
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(),
            injected.__code__,
        )
        self.assertEqual(compiled_fn(*args), injected(*args))
        _reset_precompile_entries(fn.__code__)

        self.assertEqual(compiled_fn(*args), fn(*args))

    def test_precompile_entry_miss(self):
        from torch._C._dynamo.eval_frame import _load_precompile_entry

        def fn(x):
            return x + 1

        guard_manager = torch._dynamo.guards.RootGuardManager()
        guard_manager.add_lambda_guard(lambda L: isinstance(L["x"], int), [])

        def injected(x):
            return x + 42

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn)
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager),
            injected.__code__,
        )
        self.assertEqual(compiled_fn(*args), fn(*args))

    def test_precompile_entries(self):
        from torch._C._dynamo.eval_frame import (
            _load_precompile_entry,
            _reset_precompile_entries,
        )

        def fn(x):
            return x + 1

        guard_manager_bool = torch._dynamo.guards.RootGuardManager()
        guard_manager_bool.add_lambda_guard(lambda L: isinstance(L["x"], bool), [])

        def injected_bool(x: bool):
            return x + 102

        guard_manager_int = torch._dynamo.guards.RootGuardManager()
        guard_manager_int.add_lambda_guard(lambda L: isinstance(L["x"], int), [])

        def injected_int(x: int):
            return x + 42

        guard_manager_tensor = torch._dynamo.guards.RootGuardManager()
        guard_manager_tensor.add_lambda_guard(
            lambda L: isinstance(L["x"], torch.Tensor), []
        )

        def injected_tensor(x: torch.Tensor):
            return x + 100

        guard_manager_str = torch._dynamo.guards.RootGuardManager()
        guard_manager_str.add_lambda_guard(lambda L: isinstance(L["x"], str), [])

        def injected_str(x: str):
            return x + "1"

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn)
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_bool),
            injected_bool.__code__,
        )

        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_int),
            injected_int.__code__,
        )

        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_tensor),
            injected_tensor.__code__,
        )

        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_str),
            injected_str.__code__,
        )

        self.assertEqual(compiled_fn(*args), injected_tensor(*args))
        self.assertEqual(compiled_fn(True), injected_bool(True))
        self.assertEqual(compiled_fn(10), injected_int(10))
        self.assertEqual(compiled_fn("10"), injected_str("10"))
        _reset_precompile_entries(fn.__code__)

        self.assertEqual(compiled_fn(*args), fn(*args))

    def test_precompile_fail_on_recompile(self):
        from torch._C._dynamo.eval_frame import _load_precompile_entry

        @torch.compiler.disable
        def graph(x, s0):
            return x + s0

        def fn(x):
            nonlocal graph  # Forcing fn and injected to have the same closure.
            return x - 1

        def injected(x):
            s0 = call_size(x, 0)
            return graph(x, s0)

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn)
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(),
            injected.__code__,
        )
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(*args), injected(*args))

    def test_fail_on_recompile_error_message(self):
        from torch._C._dynamo.eval_frame import (
            _load_precompile_entry,
            _reset_precompile_entries,
        )

        def fn(x):
            return x + 1

        guard_manager_bool = torch._dynamo.guards.RootGuardManager()
        guard_manager_bool.add_lambda_guard(
            lambda L: isinstance(L["x"], bool), ["isinstance(L['x'], bool)"]
        )

        def injected_bool(x: bool):
            return x + 102

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn)
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_bool),
            injected_bool.__code__,
        )

        try:
            with torch.compiler.set_stance("fail_on_recompile"):
                with self.assertRaisesRegex(
                    RuntimeError, "Failed on the following precompiled guards:"
                ):
                    compiled_fn(*args)
        finally:
            _reset_precompile_entries(fn.__code__)

    def test_shape_and_tuple_equality(self):
        def fn(x, y, t):
            z = x * y
            if x.size() == t:
                return z.cos()
            return z.sin()

        torch.compile(fn, backend="eager", fullgraph=True)(
            torch.randn([4, 4]), torch.randn([4, 4]), (4, 4)
        )

    def test_int_list(self):
        # if assume_static_by_default == True: spec int list
        # otherwise: unspec int list
        def fn(x, y):
            return torch.sin(x + y[1] % 2)

        x = torch.randn(6)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        for i in range(10, 25, 3):
            y = [i, i + 1, i + 2]
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertTrue(same(ref, res))
        if torch._dynamo.config.assume_static_by_default:
            if torch._dynamo.config.automatic_dynamic_shapes:
                self.assertExpectedInline(cnt.frame_count, """2""")
            else:
                self.assertExpectedInline(cnt.frame_count, """5""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")

    def test_patched_builtin_functions(self):
        import builtins

        # Cache the original builtin function ids
        torch._dynamo.trace_rules._builtin_function_ids()

        class MyClass:
            pass

        builtin_isinstance = builtins.isinstance

        def patched_isinstance(obj, classinfo) -> bool:
            if builtin_isinstance(obj, MyClass):
                return False
            else:
                return builtin_isinstance(obj, classinfo)

        def fn(x, y):
            if isinstance(y, MyClass):
                return x + 1
            else:
                return x - 1

        x = torch.ones(2, 3)
        y = MyClass()

        try:
            ref = fn(x, y)
            # Monkey patch builtin function
            builtins.isinstance = patched_isinstance
            opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
            res = opt_fn(x, y)
            self.assertTrue(same(ref, x + 1))
            self.assertTrue(same(res, x - 1))
        finally:
            builtins.isinstance = builtin_isinstance

        # check recompilation because builtins is now unpatched
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(res, x + 1))

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
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_cast(self):
        from typing import cast

        def fn(x):
            return cast(torch.Tensor, torch.add(x, 1.0))

        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        ref = fn(torch.ones(2, 2))
        res = opt_fn(torch.ones(2, 2))

        self.assertTrue(same(ref, res))

    def test_T_tensor_attribute(self):
        def fn(x, y):
            a = x.T
            return torch.add(a, y)

        x = torch.rand((4, 4))
        y = torch.rand((4, 4))

        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
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
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_assigning_function_to_object_attribute(self):
        # user-defined functions which are object's attributes are not converted to bound methods
        def my_add(*args):
            a, b = args
            return a + b

        class MyClass:
            def __init__(self, func):
                self.add = func

        obj = MyClass(my_add)

        def fn(x):
            return obj.add(x, 2)

        x = torch.rand(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_assigning_function_to_class_attribute(self):
        # user-defined functions which are class's attributes are converted to bound methods
        def my_add(*args):
            obj, a, b = args
            return obj.x + a + b

        class MyClass:
            add = my_add

            def __init__(self, x):
                self.x = x

        obj = MyClass(0.5)

        def fn(x):
            return obj.add(x, 2)

        x = torch.rand(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_tagging_tensors_simple(self):
        def foo(x, y):
            return x * y, x, y

        a = torch.randn([3, 3])
        a.tag = "a"
        b = torch.randn([3, 3])
        b.tag = "b"

        exported = torch._dynamo.export(foo)(a, b)
        out_graph = exported[0]

        nodes = list(out_graph.graph.nodes)
        placeholders = [node for node in nodes if node.op == "placeholder"]
        all_tags = []
        for placeholder in placeholders:
            if "tensor_dict" in placeholder.meta:
                all_tags.append(placeholder.meta["tensor_dict"]["tag"])

        self.assertEqual(all_tags, ["a", "b"])

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

        exported = torch._dynamo.export(pre_attention_state_ops)(i, mems, state)
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
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_set_custom_tensor_attribute(self):
        def fn(x):
            x.custom_attr = 3.14
            return x.custom_attr * x

        x = torch.rand((2, 2))
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_unhandled_exception_in_dynamo(self):
        # traceback.format_exc() approximates an unhandled exception
        def f(a):
            a += 1
            raise RuntimeError("smoge")
            return a

        opt_fn = torch.compile(f, backend="eager")
        try:
            opt_fn(torch.ones(2))
        except RuntimeError as e:
            self.assertIn("smoge", traceback.format_exc())

    def test_unhandled_exception_in_dynamo2(self):
        # segfaults in python 3.11 if shadow frame is freed improperly
        from torch.testing import make_tensor

        def fn():
            # test that the errors are the same for dense and sparse versions
            def test1(*, is_sparse):
                # shapes must be compatible for matrix multiplication
                a = make_tensor((2, 3), dtype=torch.float32, device="cpu")
                if is_sparse:
                    a_sparse = a.to_sparse_csr()
                    return torch.addmm(a, a_sparse, a)
                else:
                    return torch.addmm(a, a, a)

            try:
                test1(is_sparse=False)
            except RuntimeError as msg:
                try:
                    test1(is_sparse=True)
                except RuntimeError as msg2:
                    raise RuntimeError("smoge")

        opt_fn = torch.compile(fn, backend="eager")
        try:
            opt_fn()
        except RuntimeError:
            self.assertIn("smoge", traceback.format_exc())

    def test_variable_access_in_exception(self):
        def fn():
            x = torch.ones(1)
            try:
                raise RuntimeError("bad")
            except RuntimeError:
                x += 1
            return x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), torch.tensor([2.0]))

    def test_nested_sequential_with(self):
        def fn(x):
            with torch.set_grad_enabled(True):
                with torch.set_grad_enabled(False):
                    x = x + 1
                with torch.set_grad_enabled(True):
                    x = x + 1
                return x

        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(opt_fn(torch.ones(1)), torch.tensor([3.0]))

    def test_nested_sequential_try(self):
        def fn(x):
            try:
                try:
                    x = x + 1
                except:
                    pass
                try:
                    try:
                        x = x + 1
                    except:
                        pass
                except:
                    pass
            except:
                pass
            return x

        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(opt_fn(torch.ones(1)), torch.tensor([3.0]))

    def test_sparse_output_inductor_should_break(self) -> None:
        # See https://github.com/pytorch/pytorch/issues/164823
        # We want consistent semantics here
        def forward(x: torch.Tensor) -> torch.Tensor:
            x_sparse = x.to_sparse()
            return x_sparse * 2

        test_tensor = torch.randn(10, 10)
        pt = forward(test_tensor)
        aot_eager = torch.compile(forward, backend="aot_eager")(test_tensor)
        self.assertEqual(pt, aot_eager)
        inductor = torch.compile(forward, backend="inductor")(test_tensor)

    def test_nested_sequential_try_with(self):
        def fn(x):
            with torch.set_grad_enabled(True):
                try:
                    x = x + 1
                except:
                    pass
                try:
                    with torch.set_grad_enabled(False):
                        x = x + 1
                except:
                    pass
            return x

        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(opt_fn(torch.ones(1)), torch.tensor([3.0]))

    def test_nested_sequential_try_with_graph_break(self):
        def fn(x, n):
            with torch.set_grad_enabled(True):
                with torch.set_grad_enabled(False):
                    x = x + 1
                    torch._dynamo.graph_break()
                try:
                    with torch.set_grad_enabled(False):
                        x = x + 1
                        if n == 0:
                            torch._dynamo.graph_break()
                except:
                    pass
                with torch.set_grad_enabled(False):
                    x = x + 1
                    torch._dynamo.graph_break()
                x = x + 1
            return x

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)
        self.assertEqual(opt_fn(torch.ones(1), 0), torch.tensor([5.0]))
        self.assertEqual(counter.frame_count, 1)

        torch._dynamo.reset()
        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)
        self.assertEqual(opt_fn(torch.ones(1), 1), torch.tensor([5.0]))
        self.assertEqual(counter.frame_count, 3)

    def test_ordered_dict_alias_reconstruct(self):
        od = collections.OrderedDict

        def fn():
            d1 = dict()  # noqa: C408
            d1["a"] = 1
            d2 = od(d1)
            d2["b"] = 2
            torch._dynamo.graph_break()
            if isinstance(d2, od):
                return d2["a"] + d2["b"]
            else:
                return 0

        dis.dis(fn)
        self.assertEqual(torch.compile(fn, backend="eager")(), 3)

    # NOTE this test can be removed once multiline errors are in Python.
    # See https://github.com/python/cpython/issues/106922
    # Covered by test_logging.py:test_trace_call* tests in 3.13+
    @skipIfNotPy311
    @unittest.skipIf(sys.version_info >= (3, 13), "feature landed in 3.13")
    def test_get_instruction_source_311(self):
        def f():
            # flake8: noqa
            # fmt: off
            # test binary ops
            a = ( b   )   +   c
            a = (a + b) // (c - d)
            a = b    \
         +\
               c  # test
            a = (
                (b  # test +
                    )  \
                # +
            << (

                c  # test
                \
            )  # test
            )

            # test slice
            a = bbb   [  ccc    ]
            b = bbbbb \
                [  ccc # test

                 + ddd  \

                ] # test
            a = bbb[ccc][ddd][eee]

            # test nested and multiline function calls
            a = g(g(g(b)))
            a = g(h(
                g(b),
                c
            ))

            # test chained function calls
            a = (g(x).y)(
                z
            )(1)(2)

            # test unicode (match traceback behavior)
            a = ("🔥🔥🔥" +
                + "🔥🔥") + b

        from torch._dynamo.utils import get_instruction_source_311

        if sys.version_info >= (3, 12):
            # Offsets changed in 3.12, e.g. due to removal of PRECALL inst
            offsets = (3, 11, 15, 19, 23, 29, 35, 44, 53, 65)
        else:
            offsets = (3, 11, 15, 19, 23, 29, 35, 46, 58, 74)
        insts = list(dis.get_instructions(f))
        # uncomment to determine offsets
        # print(*enumerate(insts), sep="\n")
        all_sources = "\n".join(
            get_instruction_source_311(f.__code__, insts[offset]) for offset in offsets
        )
        self.assertExpectedInline(
            all_sources,
            """\
            a = ( b   )   +   c
                ~~~~~~~~~~^~~~~

            a = (a + b) // (c - d)
                ~~~~~~~~^^~~~~~~~~

            a = b    \\
                ~~~~~~
         +\\
         ^~
               c  # test
               ~

                (b  # test +
                ~~~~~~~~~~~~
                    )  \\
                    ~~~~
                # +
                ~~~
            << (
            ^^~~


                c  # test
                ~~~~~~~~~
                \\
                ~
            )  # test
            ~

            a = bbb   [  ccc    ]
                ~~~~~~^^^^^^^^^^^

            b = bbbbb \\
                ~~~~~~~
                [  ccc # test
                ^^^^^^^^^^^^^


                 + ddd  \\
                 ^^^^^^^^


                ] # test
                ^

            a = bbb[ccc][ddd][eee]
                ~~~~~~~~^^^^^

            a = g(g(g(b)))
                  ~^^^^^^

            a = g(h(
                  ~^
                g(b),
                ^^^^^
                c
                ^
            ))
            ^

            a = (g(x).y)(
                ~~~~~~~~~
                z
                ~
            )(1)(2)
            ~^^^
""",
        )
        # test unicode (since assertExpectedInline doesn't support unicode)
        op_offset = 74 if sys.version_info >= (3, 12) else 84
        self.assertEqual(
            get_instruction_source_311(f.__code__, insts[op_offset]),
            """\
            a = ("🔥🔥🔥" +
                ~~~~~~~~
                + "🔥🔥") + b
                ~~~~~~~~^~~
""",
        )

    def test_float_speculation_log_divergence(self):
        def fn(x, y, z):
            a = F.interpolate(x, scale_factor=z, mode="bilinear", align_corners=False)
            b = F.interpolate(y, scale_factor=z, mode="bilinear", align_corners=False)
            return a * b

        cnt = CompileCounterWithBackend("inductor")
        fn_opt = torch.compile(fn, backend=cnt)
        y = torch.randn(3, 3, 3, 4)

        self.assertEqual(fn(y, y, 1.0), fn_opt(y, y, 1.0))
        self.assertEqual(fn(y, y, 2.0), fn_opt(y, y, 2.0))

    def test_raise_guard_full_constraint(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] == 3:
                return x.sin()
            return x.cos()

        torch._dynamo.mark_dynamic(y, 0)
        with self.assertRaises(ConstraintViolationError):
            torch.compile(my_dyn_fn, backend="eager")(y)

    def test_raise_guard_indirect_full_constraint(self):
        y = torch.randn([3, 3, 3])

        def dyn_fn(x):
            if x.shape[0] > 3:
                return x.cos()
            if x.shape[0] < 3:
                return x * 2
            return x.sin()

        torch._dynamo.mark_dynamic(y, 0)
        with self.assertRaises(ConstraintViolationError):
            torch.compile(dyn_fn, backend="eager")(y)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_empty_tensor(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            n = x.item()
            return torch.empty((n - 1) // 2)

        self.assertEqual(fn(torch.tensor([4])).size(0), 1)
        self.assertEqual(fn(torch.tensor([1])).size(0), 0)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_sym_and_terms(self):
        from torch.fx.experimental.symbolic_shapes import sym_and

        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def fn(xs):
            u0, u1 = xs.tolist()
            torch._check(sym_and(u0 >= 3, u0 <= 10, u1 >= 2))

            # test individual checks
            n = 0
            if u0 >= 3:
                n += 1
            if u0 <= 11:
                n += 1
            if u1 >= 1:
                n += 1
            return u0 + u1 + n

        fn(torch.tensor([5, 6]))
        fn(torch.tensor([8, 7]))
        with self.assertRaises(RuntimeError):
            fn(torch.tensor([9, 0]))

    def test_unbacked_2d_expand(self):
        @torch.compile(fullgraph=True, dynamic=True, backend="inductor")
        def func(a, b):
            a.expand(b.shape)
            return a * 10

        a = torch.rand(1, 1)
        b = torch.rand(1, 1)

        torch._dynamo.decorators.mark_unbacked(a, 0)
        torch._dynamo.decorators.mark_unbacked(a, 1)
        torch._dynamo.decorators.mark_unbacked(b, 0)
        torch._dynamo.decorators.mark_unbacked(b, 1)
        func(a, b)
        func(torch.rand(4, 5), torch.rand(4, 5))
        # This does not raise an error right now because of a recompilation.
        # https://github.com/pytorch/pytorch/issues/163785
        # with self.assertRaises(AssertionError):
        #     func(torch.rand(1, 1), torch.rand(2, 1))
        func(torch.rand(1, 1), torch.rand(2, 1))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_sym_constrain_range_on_replaced_unbacked_symbol(self):
        # Tests the following case:
        # Deferred runtime asserts adds sym_constrain_range(u0).
        # However, u0 is replaced with s0 + s1.
        # So, now we have sym_constrain_range(s0 + s1).
        def fn(x, y, z):
            z += 7  # to avoid creating unspecified symbol instead of unbacked symbol
            u0 = z.item()
            s0 = x.size(0)
            s1 = y.size(0)
            torch._check(s0 < 100)
            torch._check(s1 < 100)
            torch._check(u0 == s0 + s1)
            return x, y, z

        inputs = (
            x := torch.randn(16, 10),
            y := torch.randn(16, 10),
            torch.tensor(32 - 7),
        )
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)
        opt = torch.compile(fn, fullgraph=True)
        opt(*inputs)
        with self.assertRaises(RuntimeError):
            inputs = (
                x := torch.randn(16, 10),
                y := torch.randn(16, 10),
                torch.tensor(32),
            )
            opt(*inputs)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @torch._dynamo.config.patch(assume_static_by_default=True)
    def test_symint_copy_into_unbacked_slice(self):
        @torch.compile()
        def fn(a, x):
            u0 = torch.tensor(x[0].to(torch.int64).item()).item()
            B, H, T, D = a.shape
            a_padding = torch.zeros((B, H, u0, D), dtype=torch.float64)
            b = torch.cat([a, a_padding], dim=2)
            c = torch.randn(B, H, 152, D)
            b[:, :, :152, :] = c
            return b

        x = torch.tensor([0])
        torch._dynamo.decorators.mark_unbacked(x, 0)
        a = torch.zeros((1, 16, 152, 96))

        # Previously would crash with guard on data dependent error
        fn(a, x)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_symint_fold_nontrivial_product_modulo(self):
        @torch.compile(fullgraph=True)
        def f(x):
            u0, u1 = x.tolist()
            # The condition should fold to true.
            if ((u0 + 10) * (u0 + 10)) % (u0 + 10) == 0:
                return torch.tensor(True)
            return torch.tensor(False)

        res = f(torch.tensor([20, 21]))
        self.assertEqual(torch.tensor(True), res)

    # Translation validation changes the exception type, don't run with it
    @torch.fx.experimental._config.patch(translation_validation=False)
    def test_mark_dynamic_with_ranges(self):
        y = torch.randn([8, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] == 3:
                return x.sin()
            return x.cos()

        torch._dynamo.mark_dynamic(y, 0, min=2, max=5)
        with self.assertRaises(ConstraintViolationError):
            torch.compile(my_dyn_fn, backend="eager")(y)

    def test_mark_static(self):
        counter = CompileCounter()

        def my_dyn_fn(x):
            return x.cos()

        y = torch.randn([3])
        torch._dynamo.mark_static(y, 0)
        torch.compile(my_dyn_fn, backend=counter)(y)

        z = torch.randn([4])
        torch.compile(my_dyn_fn, backend=counter)(z)

        self.assertEqual(counter.frame_count, 2)

    def test_no_raise_guard_partial_constraint(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] > 3:
                return x.sin()
            return x.cos()

        torch.compile(my_dyn_fn, backend="eager")(y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        torch.compile(my_dyn_fn, backend="eager")(y)

    def test_no_raise_guard_partial_constraint_across_break(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            z = x * y

            torch._dynamo.graph_break()
            if z.shape[0] > 2:
                return z.cos()

            return x.cos()

        torch.compile(my_dyn_fn, backend="eager")(y, y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        torch.compile(my_dyn_fn, backend="eager")(y, y)

    def test_tolist(self):
        # This should compile with no faluire.
        cnt = CompileCounterWithBackend("inductor")

        @torch.compile(fullgraph=False, backend=cnt)
        def func(a):
            a = a * 100
            u0, u1, u2, u3, u4 = a.tolist()
            return a * u0 * u1

        func(torch.tensor([1, 2, 3, 4, 5]))
        self.assertEqual(cnt.frame_count, 2)

    # Sadly, this does not throw - we do not prop correctly across the graph break
    @unittest.expectedFailure
    def test_raise_guard_partial_constraint_across_break(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            z = x * y

            torch._dynamo.graph_break()
            if z.shape[0] == 3:
                return z.cos()

            return x.cos()

        torch.compile(my_dyn_fn, backend="eager")(y, y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            Exception,
        ):
            torch.compile(my_dyn_fn, backend="eager")(y, y)

    def test_raise_guard_partial_constraint_no_graph_break(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            z = x * y

            if z.shape[0] == 3:
                return z.cos()

            return x.cos()

        torch._dynamo.mark_dynamic(y, 0)
        with self.assertRaises(ConstraintViolationError):
            torch.compile(my_dyn_fn, backend="eager")(y, y)

    @torch._dynamo.config.patch(force_parameter_static_shapes=True)
    @torch._dynamo.config.patch(force_nn_module_property_static_shapes=True)
    @torch.compiler.config.patch(
        dynamic_sources="L['x'],L['y'],L['self']._modules['y'].x,L['self']._modules['y']._modules['c']._parameters['weight'],L['self']._modules['y']._modules['c']._parameters['bias']"
    )
    def test_dynamic_sources_force_parameter_static_shapes_and_property_static_shapes_override(
        self,
    ):
        builder._DYNAMIC_SOURCES = None

        counter = CompileCounter()

        class Y(torch.nn.Module):
            def __init__(self, n_input, n_output):
                super().__init__()
                self.c = torch.nn.Linear(n_input, n_output)
                self.x = n_input

            def forward(self, x):
                return self.c(x) * self.x

        class M(torch.nn.Module):
            def __init__(self, n_input, n_output):
                self.n_input = n_input
                self.n_output = n_output
                super().__init__()
                self.y = Y(n_input, n_output)

            @torch.compile(backend=counter)
            def forward(self, x, y):
                return self.y(x) * y

        model = M(3210, 30)
        model(torch.randn(1, 3210), 2)
        model = M(3211, 30)
        model(torch.randn(1, 3211), 3)
        model = M(3212, 30)
        model(torch.randn(1, 3212), 4)

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L['self']._modules['inner'].x")
    def test_dynamic_sources_precedence_over_int_specialization(self):
        builder._DYNAMIC_SOURCES = None

        counter = CompileCounter()

        class Model(torch.nn.Module):
            def __init__(self, x) -> None:
                super().__init__()
                self.inner = torch.nn.Linear(10, 10)
                # attach attribute to builtin nn module.
                self.inner.x = x

            @torch.compile(fullgraph=True, backend=counter)
            def forward(self, a):
                return a * self.inner.x

        m1 = Model(50)
        m2 = Model(60)
        with fresh_cache():
            m1(torch.rand(1, 2, 3))
            m2(torch.rand(1, 2, 3))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L['x']")
    def test_dynamic_sources_int(self):
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            return torch.randn(5) * x

        fn(1)
        fn(2)
        fn(3)

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L['x']")
    def test_dynamic_sources_tensor(self):
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            return x * x

        fn(torch.randn(2))
        fn(torch.randn(3))
        fn(torch.randn(4))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(unbacked_sources="L['x']")
    def test_unbacked_sources_tensor(self):
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            return x * x

        fn(torch.randn(0))
        fn(torch.randn(1))
        fn(torch.randn(2))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(unbacked_sources="L['x']")
    def test_unbacked_sources_scalar(self):
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            return x * x

        fn(0)
        fn(1)
        fn(2)

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L['x']")
    def test_dynamic_sources_graph_break(self):
        counter = CompileCounter()

        def foo(x):
            return x * x

        @torch.compile(backend=counter)
        def fn(x):
            x = x * x
            torch._dynamo.graph_break()
            return foo(x)

        fn(torch.randn(2))
        fn(torch.randn(3))
        fn(torch.randn(4))

        # 2 since graph break produces 2 graphs. NB: there are no recompiles
        self.assertEqual(counter.frame_count, 2)

    @torch.compiler.config.patch(dynamic_sources="L['x'], L['y']")
    def test_dynamic_sources_dynamic_override(self):
        counter = CompileCounter()

        @torch.compile(dynamic=False, backend=counter)
        def fn(x, y):
            return x * y

        fn(2, torch.randn(2))
        fn(3, torch.randn(3))
        fn(4, torch.randn(4))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L\\['x.*'\\], L\\['y.*'\\]")
    def test_dynamic_sources_dynamic_override_regex(self):
        counter = CompileCounter()

        @torch.compile(dynamic=False, backend=counter)
        def fn(x1, y1):
            return x1 * y1

        fn(2, torch.randn(2))
        fn(3, torch.randn(3))
        fn(4, torch.randn(4))

        self.assertEqual(counter.frame_count, 1)

    def test_cannot_trace_mark_dynamic(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            torch._dynamo.mark_dynamic(x, 0)
            return x * x

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            torch.compile(my_dyn_fn, backend="eager")(y)

    def test_cannot_trace_mark_dynamic_safe_unreached(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] == 3:
                return x
            print("Running", torch._dynamo.mark_dynamic(x, 0))
            return x * x

        torch.compile(my_dyn_fn, backend="eager")(y)

    def test_anomaly_aot_autograd(self):
        def fail():
            raise AssertionError("fail")

        @allow_in_graph
        def h(a):
            r = a.sum()
            # Trigger an exception in backwards
            r.register_hook(lambda x: fail())
            return r

        @torch.compile(backend="aot_eager")
        def f(a):
            return h(a)

        with (
            warnings.catch_warnings(record=True) as w,
            self.assertRaises(torch._dynamo.exc.BackendCompilerFailed),
        ):
            f(torch.randn(2, 2, requires_grad=True))

        # Suppress unrelated pkg_resources warnings
        self.assertIn("forward call that caused the error", str(w[-1].message))

    def test_py_guards_mark_dynamic(self):
        def my_dyn_fn(a):
            if a.shape[0] > 2:
                return a.cos()
            return a.sin()

        counter = CompileCounter()

        # Run with dynamic
        x0 = torch.randn([3, 3, 3])
        torch._dynamo.mark_dynamic(x0, 0)
        torch.compile(my_dyn_fn, backend=counter)(x0)
        self.assertEqual(counter.frame_count, 1)

        # Run without dynamic, no recompile
        x = torch.randn([3, 3, 3])
        torch.compile(my_dyn_fn, backend=counter)(x)
        self.assertEqual(counter.frame_count, 1)

        # Mark a new dim, 1, as dynamic
        x1 = torch.randn([3, 3, 3])
        torch._dynamo.mark_dynamic(x1, 1)
        torch.compile(my_dyn_fn, backend=counter)(x1)
        # Recompile triggered because we marked a new dym as dynamic
        self.assertEqual(counter.frame_count, 2)

        # Reset
        torch._dynamo.reset()
        # Reset counter
        counter = CompileCounter()

        # Run with dynamic 1
        torch.compile(my_dyn_fn, backend=counter)(x1)
        self.assertEqual(counter.frame_count, 1)

        # Run with dynamic 0, not subset
        torch.compile(my_dyn_fn, backend=counter)(x0)
        self.assertEqual(counter.frame_count, 2)

        # Run with dynamic 0, 1, 2, not subset
        x012 = torch.randn([3, 3, 3])
        torch._dynamo.mark_dynamic(x012, 0)
        torch._dynamo.mark_dynamic(x012, 1)
        torch._dynamo.mark_dynamic(x012, 2)
        torch.compile(my_dyn_fn, backend=counter)(x012)
        self.assertEqual(counter.frame_count, 3)

    def test_recompile_on_global_state_change(self):
        last_state = []
        cnt = 0

        def my_compiler(gm, _):
            nonlocal cnt
            cnt += 1
            state = read_state()

            def inner(*args):
                last_state[:] = state
                return gm(*args)

            return inner

        def read_state():
            return [
                torch.is_grad_enabled(),
                torch.are_deterministic_algorithms_enabled(),
                torch._C._get_cublas_allow_tf32(),
            ]

        def write_state(state):
            torch.set_grad_enabled(state[0])
            torch.use_deterministic_algorithms(state[1])
            torch._C._set_cublas_allow_tf32(state[2])

        @torch.compile(backend=my_compiler)
        def fn(x):
            return x + 1

        initial_state = read_state()
        y = torch.randn(10)
        try:
            for round in range(3):
                for i in range(len(initial_state)):
                    new_state = [False] * len(initial_state)
                    new_state[i] = True
                    write_state(new_state)
                    assert read_state() == new_state
                    last_state.clear()
                    fn(y)
                    assert last_state == new_state
                    if round == 0:
                        assert cnt == i + 1
                    else:
                        assert cnt == len(initial_state)
        finally:
            write_state(initial_state)

    def test_grad_state_mutated(self):
        prior = torch.is_grad_enabled()
        value = None
        cnt = CompileCounter()

        @torch._dynamo.allow_in_graph
        def check_state():
            nonlocal value
            value = torch.is_grad_enabled()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            check_state()
            torch.set_grad_enabled(False)
            return x + 1

        try:
            torch.set_grad_enabled(True)
            fn(torch.randn(10))
            assert value is True
            assert torch.is_grad_enabled() is False

            value = None
            torch.set_grad_enabled(True)
            fn(torch.randn(10))
            assert value is True
            assert torch.is_grad_enabled() is False

            assert cnt.frame_count == 1
        finally:
            torch.set_grad_enabled(prior)

    def test_deterministic_algorithms_mutated(self):
        prior = torch.are_deterministic_algorithms_enabled()
        prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        value = None
        warn_only = None
        cnt = CompileCounter()

        @torch._dynamo.allow_in_graph
        def check_state():
            nonlocal value
            nonlocal warn_only
            value = torch.are_deterministic_algorithms_enabled()
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            check_state()
            torch.use_deterministic_algorithms(False, warn_only=False)
            return x + 1

        def run_fn():
            torch.use_deterministic_algorithms(True, warn_only=True)
            fn(torch.randn(10))
            assert value is True
            assert warn_only is True
            assert torch.are_deterministic_algorithms_enabled() is False
            assert torch.is_deterministic_algorithms_warn_only_enabled() is False

        try:
            run_fn()
            value, warn_only = None, None
            run_fn()
            assert cnt.frame_count == 1
        finally:
            torch.use_deterministic_algorithms(prior, warn_only=prior_warn_only)

    def test_torch_compile_ctx_on_forward_and_training_step(self):
        class MyModel(torch.nn.Module):
            def forward(self): ...

            def training_step(self):
                self()

        model = MyModel()
        compiled_model = torch.compile(model)

        model.forward = compiled_model.dynamo_ctx(model.forward)
        model.training_step = compiled_model.dynamo_ctx(model.training_step)

        model.training_step()

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
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            torch.compile(fn, backend="eager")(x, y, z)

        self.assertEqual(len(seen_frames), 1)
        self.assertEqual(seen_frames[0].name, "fn")
        self.assertEqual(seen_frames[0].line, "r, r2 = uwu_inline_me(x, y, z)")

    def test_fullgraph_capture(self):
        from torch._dynamo.convert_frame import fullgraph_capture
        from torch._dynamo.utils import dynamo_timed, get_metrics_context

        def foo(x):
            if x.shape[1] >= 3:
                return x + x.shape[0]
            else:
                return x - x.shape[0]

        x = torch.randn(4, 3)
        with (
            get_metrics_context(),
            dynamo_timed(""),
        ):
            capture_output = fullgraph_capture(foo, (x,))
            graph_capture_output = capture_output.graph_capture_output
            fn = graph_capture_output.build_guards(foo.__code__)

            for guard in graph_capture_output.output_graph.guards:
                if guard.source == torch._guards.GuardSource.SHAPE_ENV:
                    dynamic = guard.code_list is not None
                    if dynamic:
                        self.assertEqual(
                            guard.code_list,
                            [
                                "L['x'].stride()[0] == L['x'].size()[1]",
                                "2 <= L['x'].size()[0]",
                                "3 <= L['x'].size()[1]",
                            ],
                        )
                        self.assertTrue(
                            fn.guard_manager.check({"x": torch.randn(3, 3)})
                        )
                        self.assertTrue(
                            fn.guard_manager.check({"x": torch.randn(4, 4)})
                        )
                    else:
                        self.assertFalse(
                            fn.guard_manager.check({"x": torch.randn(3, 3)})
                        )
                        self.assertFalse(
                            fn.guard_manager.check({"x": torch.randn(4, 4)})
                        )
                    self.assertFalse(fn.guard_manager.check({"x": torch.randn(4, 2)}))
                    self.assertFalse(fn.guard_manager.check({"x": torch.randn(1, 3)}))
                    break

            backend_input = capture_output.backend_input
            self.assertTrue(fn.guard_manager.check({"x": x}))
        import_sources = {
            alias: importlib.import_module(module_name)
            for alias, module_name in graph_capture_output.import_sources.items()
        }
        self.assertEqual(
            foo(x),
            types.FunctionType(
                graph_capture_output.bytecode,
                {
                    **import_sources,
                    backend_input.backend_id: backend_input.graph_module,
                },
            )(x),
        )

    def test_torch_guards_stack_frame_register_inlining_deep(self):
        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([0.75, 0.75, 0.75, 0.75])
        z = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        def uwu_inline_me_deep(x, y):
            return torch.cat((x, x)) + y

        def uwu_inline_me(x, y, z):
            r = uwu_inline_me_deep(x, y)
            r2 = uwu_inline_me_deep(y, z)
            return r, r2

        def fn(x, y, z):
            r, r2 = uwu_inline_me(x, y, z)
            return torch.mul(r, r), torch.mul(r2, r2)

        seen_frames = []
        import contextlib

        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            torch.compile(fn, backend="eager")(x, y, z)

        self.assertEqual(len(seen_frames), 3)
        self.assertEqual(seen_frames[0].name, "fn")
        self.assertEqual(seen_frames[1].name, "uwu_inline_me")
        self.assertEqual(seen_frames[2].line, "r2 = uwu_inline_me_deep(y, z)")

    def test_recompile_on_disable_1(self):
        # fix https://github.com/pytorch/pytorch/issues/157399
        @torch.compile(backend="eager")
        def fn(x):
            @torch._dynamo.disable
            def inner(x):
                return x + 10

            return inner(x) + 1

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            try:
                for i in range(5):
                    fn(torch.rand(2, 3))
            except torch._dynamo.exc.RecompileError as e:
                self.fail("RecompileError raised unexpectedly: " + str(e))

    def test_recompile_on_disable_2(self):
        def outer(x, cond):
            @torch._dynamo.disable()
            def fn0(y):
                return y + 1

            @torch._dynamo.disable()
            def fn1(y):
                return y + 2

            if cond:
                f = fn0
            else:
                f = fn1

            torch._dynamo.graph_break()
            # there will be a resume function here
            return f(x)

    def test_error_on_recompile(self):
        @torch.compile(backend="eager")
        def fn(a, b):
            return a + b

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            with self.assertRaises(torch._dynamo.exc.RecompileError):
                fn(torch.rand(2, 3), torch.rand(2, 3))
                fn(torch.rand(2, 3), (1, 2, 3))

    def test_guards_strip_function_call(self):
        from torch._dynamo.guards import strip_function_call

        test_case = [
            ("___odict_getitem(a, 1)", "a"),
            ("a.layers[slice(2)][0]._xyz", "a"),
            ("getattr(a.layers[slice(2)][0]._abc, '0')", "a"),
            ("getattr(getattr(a.x[3], '0'), '3')", "a"),
            ("a.layers[slice(None, -1, None)][0]._xyz", "a"),
            ("a.layers[func('offset', -1, None)][0]._xyz", "a"),
        ]
        # strip_function_call should extract the object from the string.
        for name, expect_obj in test_case:
            self.assertEqual(strip_function_call(name), expect_obj)

    def test_int_neg(self):
        def int_neg(a, b):
            x = a.shape[0]
            y = b.shape[0]
            return -x * -y * a * b

        torch._dynamo.testing.standard_test(self, int_neg, 2)

    def test_hash_getitem_slice(self):
        s = GetItemSource(LocalSource("foo"), slice(None, -1, None))
        s2 = GetItemSource(LocalSource("foo"), slice(None, -1, None))
        s3 = GetItemSource(LocalSource("foo"), slice(None, -1, 2))
        some_set = set()

        self.assertTrue(s not in some_set)
        self.assertTrue(s2 not in some_set)
        self.assertTrue(s3 not in some_set)

        some_set.add(s)

        self.assertTrue(s in some_set)
        # s and s2 should hash the  same
        self.assertTrue(s2 in some_set)
        # s3 should be different
        self.assertTrue(s3 not in some_set)

        self.assertTrue(s == s2)
        self.assertTrue(s != s3)

    def test_inline_dict_function(self):
        def _result_type_dict(dtype):
            return {bool: torch.float32}[dtype]

        @torch.compile
        def f():
            return torch.ones(3, dtype=_result_type_dict(bool))

        self.assertEqual(f(), torch.ones(3, dtype=torch.float32))

    def test_inline_dict_function_passed_as_arg(self):
        @torch.compile
        def fn(d, x, y):
            if d[x] is torch.float32:
                return y.cos()
            else:
                return y.sin()

        dd = {bool: torch.float32, int: torch.int64}
        self.assertEqual(fn(dd, bool, torch.ones(4)), torch.ones(4).cos())
        self.assertEqual(fn(dd, int, torch.ones(4)), torch.ones(4).sin())

    def test_add_sizes(self):
        def func(x):
            y = x.size()
            return y + y

        eager_out = func(torch.ones(10, 10, 3))
        compile_out = torch.compile(func, backend="eager")(torch.ones(10, 10, 3))
        self.assertTrue(isinstance(compile_out, torch.Size))
        self.assertEqual(eager_out, compile_out)

    def test_nested_function_resuming_with_correct_globals(self):
        # https://github.com/pytorch/pytorch/issues/99665
        try:
            from .utils import outer_func
        except ImportError:
            from utils import outer_func

        def gn(x, y):
            return x + y

        def fn(x, y):
            return outer_func(gn)(x, y)

        x = torch.rand([3])
        y = torch.rand([3])
        opt_fn = torch.compile(backend="eager")(fn)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_recursion_depth_guards(self):
        @torch.compile(dynamic=True)
        def foo(*args, **kwargs):
            if sum(args) == 0:
                return 0
            return 1

        args = list(range(2000))
        foo(*args)
        # Previously would have crashed

    @dataclasses.dataclass
    class CSETestCase:
        expr: str
        preface: typing.List[str] = dataclasses.field(default_factory=list)
        expected: typing.Optional[str] = None

    def test_guards_cse_pass_single(self):
        from torch._dynamo.guards import PyExprCSEPass

        testcase = self.CSETestCase
        testcases = [
            # Nothing gets CSE-d, since the only repeated sub-expression is 'x'.
            # i.e. not a node type we are interested on.
            testcase(expr="x[0].a"),
            testcase(expr="x[1].a"),
            testcase(expr="x[2].a"),
            # 'a.b.c' gets CSE-d, since it's a sub-expression used more than 'PyExprCSEPass.USE_THRESHOLD'.
            testcase(
                expr="a.b.c[0].d.e",
                preface=["_var0 = a.b", "_var1 = _var0.c"],
                expected="_var1[0].d.e",
            ),
            testcase(expr="a.b.c[1].d.e", expected="_var1[1].d.e"),
            testcase(expr="a.b.c[2].d.e", expected="_var1[2].d.e"),
            # 'm.n[0]' gets CSE-d, since it is a sub-expression used more than 'PyExprCSEPass.USE_THRESHOLD'.
            testcase(
                expr="f(m.n[0], '0').x.y.z",
                preface=["_var2 = m.n", "_var3 = _var2[0]"],
                expected="f(_var3, '0').x.y.z",
            ),
            testcase(expr="f(m.n[0], '1').x.y.z", expected="f(_var3, '1').x.y.z"),
            testcase(expr="f(m.n[0], '2').x.y.z", expected="f(_var3, '2').x.y.z"),
            # The whole expression gets CSE-d, as well as all of its sub-expressions.
            testcase(
                expr="self.g(a, b).k",
                preface=["_var4 = self.g", "_var5 = _var4(a, b)", "_var6 = _var5.k"],
                expected="_var6",
            ),
            testcase(expr="self.g(a, b).k", expected="_var6"),
            testcase(expr="self.g(a, b).k", expected="_var6"),
        ]
        csepass = PyExprCSEPass()
        csepass.count([t.expr for t in testcases])

        for t in testcases:
            preface, expr = csepass.replace(t.expr)
            self.assertEqual(preface, t.preface)
            expected = t.expected if t.expected is not None else t.expr
            self.assertEqual(expr, expected)

    def test_guards_cse_pass_multiple(self):
        from torch._dynamo.guards import PyExprCSEPass

        testcase = self.CSETestCase
        testcases = [
            testcase(
                expr="x[0].a < x[1].a * (3 - x[2].a)",
                expected="x[0].a < x[1].a * (3 - x[2].a)",
            ),
            testcase(
                expr="a.b.c[0].d.e + a.b.c[1].d.e * a.b.c[2].d.e > 0",
                preface=["_var0 = a.b", "_var1 = _var0.c"],
                expected="_var1[0].d.e + _var1[1].d.e * _var1[2].d.e > 0",
            ),
            testcase(
                expr="f(m.n[0], '0').x.y.z * f(m.n[0], '1').x.y.z * f(m.n[0], '2').x.y.z < 512",
                preface=["_var2 = m.n", "_var3 = _var2[0]"],
                expected="f(_var3, '0').x.y.z * f(_var3, '1').x.y.z * f(_var3, '2').x.y.z < 512",
            ),
            testcase(
                expr="self.g(a, b).k + (1 - self.g(a, b).k) <= m[0].a + self.g(a, b).k",
                preface=["_var4 = self.g", "_var5 = _var4(a, b)", "_var6 = _var5.k"],
                expected="_var6 + (1 - _var6) <= m[0].a + _var6",
            ),
        ]

        csepass = PyExprCSEPass()
        csepass.count([t.expr for t in testcases])

        for t in testcases:
            preface, expr = csepass.replace(t.expr)
            self.assertEqual(preface, t.preface)
            expected = t.expected
            expected = expected if expected is not None else t.expr
            self.assertEqual(expr, expected)

    def test_guard_function_builder_with_cse(self):
        from torch._dynamo.guards import build_guard_function

        exprs = [
            "x[0].a < x[1].a * (3 - x[2].a)",
            "a.b.c[0].d.e + a.b.c[1].d.e * a.b.c[2].d.e > 0",
            "f(m.n[0], '0').x.y.z * f(m.n[0], '1').x.y.z * f(m.n[0], '2').x.y.z < 512",
            "self.g(a, b).k + (1 - self.g(a, b).k) <= m[0].a + self.g(a, b).k",
        ]

        _, pycode = build_guard_function(exprs, "")
        expected = """\
def ___make_guard_fn():
    def guard(L):
        if not (x[0].a < x[1].a * (3 - x[2].a)):
            return False
        _var0 = a.b
        _var1 = _var0.c
        if not (_var1[0].d.e + _var1[1].d.e * _var1[2].d.e > 0):
            return False
        _var2 = m.n
        _var3 = _var2[0]
        if not (f(_var3, '0').x.y.z * f(_var3, '1').x.y.z * f(_var3, '2').x.y.z < 512):
            return False
        _var4 = self.g
        _var5 = _var4(a, b)
        _var6 = _var5.k
        if not (_var6 + (1 - _var6) <= m[0].a + _var6):
            return False
        return True
    return guard
"""

        self.assertEqual(expected, pycode)

    def test_dynamo_compiling_fake_tensor_to_vararg_int(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                # use numpy int so it's wrapped as fake tensor in dynamo
                shape = np.int_(16)
                # test shape as fake tensor, which param type is
                # Sequence[Union[_int, SymInt]]
                return x.reshape(shape)

        x = torch.rand([4, 4])
        model = MyModule()
        orig_out = model(x)
        opt_model = torch.compile(MyModule(), backend="eager")
        opt_out = opt_model(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_scalar_tensor_is_equivalent_to_symint_argument(self):
        class GumbelTopKSampler(torch.nn.Module):
            def __init__(self, T, k):
                super().__init__()
                self.T = torch.nn.Parameter(
                    torch.tensor(T, dtype=torch.float32), requires_grad=False
                )
                self.k = torch.nn.Parameter(
                    torch.tensor(k, dtype=torch.int32), requires_grad=False
                )

            def sample_discrete(self, logits):
                threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
                samples = torch.ge(logits.squeeze(1), threshold).float()
                return samples

            def forward(self, logits):
                dsamples = self.sample_discrete(logits)
                return dsamples

        x = torch.rand([4, 4, 4, 4])
        m = GumbelTopKSampler(T=4, k=4)
        orig_out = m(x)
        opt_m = torch.compile(backend="eager")(m)
        opt_out = opt_m(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_scalar_tensor_is_equivalent_to_symint_list_argument(self):
        class Jitter(torch.nn.Module):
            def __init__(self, jitter_val):
                super().__init__()
                self.jitter_val = jitter_val

            def roll_tensor(self, input):
                h_shift = self.jitter_val - 1
                w_shift = self.jitter_val + 1
                return torch.roll(
                    torch.roll(input, shifts=h_shift, dims=2), shifts=w_shift, dims=3
                )

            def forward(self, input):
                return self.roll_tensor(input)

        x = torch.rand([4, 4, 4, 4])
        m = Jitter(jitter_val=4)
        orig_out = m(x)
        opt_m = torch.compile(backend="eager")(m)
        opt_out = opt_m(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_scalar_tensor_is_equivalent_to_int_list_argument(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                permute = torch.tensor([0, 2, 1])
                x = input.permute(*permute)
                return x

        x = torch.randn(2, 3, 4)
        m = MyModel()
        orig_out = m(x)
        opt_m = torch.compile(backend="eager")(m)
        opt_out = opt_m(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_torch_variable_hasattr(self):
        def fn(x):
            if hasattr(torch.nn, "Module"):
                return x * x
            return x + 1

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        x = torch.rand([4, 4])
        fn_out = fn(x)
        compiled_out = compiled_fn(x)
        self.assertTrue(same(fn_out, compiled_out))

    def test_list_hasattr1(self):
        def fn(x):
            if hasattr(x, "foo"):
                return x[0] + 1
            return x[0] - 1

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        x = [torch.randn(3)]
        fn_out = fn(x)
        compiled_out = compiled_fn(x)
        self.assertTrue(same(fn_out, compiled_out))

    def test_list_hasattr2(self):
        def fn():
            x = [torch.zeros(3)]
            if hasattr(x, "__len__"):
                return x[0] + 1
            return x[0] - 1

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertTrue(same(fn_out, compiled_out))

    def test_tuple_hasattr(self):
        def fn(x):
            if hasattr(x, "foo"):
                return x[0] + 1
            return x[1] - 1

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        x = (torch.randn(3), torch.randn(3))
        fn_out = fn(x)
        compiled_out = compiled_fn(x)
        self.assertTrue(same(fn_out, compiled_out))

    def test_fn_hasattr__name__1(self):
        def fn():
            foo = lambda x: x + 1
            return hasattr(foo, "__name__")

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertTrue(fn_out)

    def test_fn_hasattr__name__2(self):
        def bar(x):
            return torch.sin(x)

        def fn():
            return hasattr(bar, "__name__")

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertTrue(fn_out)

    def test_fn_hasattr__name__3(self):
        def bar(x, y):
            return torch.sin(x) + torch.cos(y)

        baz = functools.partial(bar, y=4)

        def fn():
            return hasattr(baz, "__name__")

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertFalse(fn_out)

    def test_torch_objects_as_keys(self):
        remap = {torch.float16: torch.float32}

        def fn():
            return torch.randn(3, dtype=remap[torch.float16])

        opt = torch.compile(fn, backend="eager")
        opt()

    def test_dynamic_one_hot(self):
        def fn(x):
            x = x + 1
            # graph break from data-dependent output shape
            x = torch.nn.functional.one_hot(x)
            x = x + 1
            return x

        inp = torch.arange(20) % 4
        counter = CompileCounter()
        real_out = fn(inp)
        comp_out = torch.compile(fn, backend=counter)(inp)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(counter.op_count, 2)

    def test_jacfwd_one_hot_dynamic_compile(self):
        import torch.nn.functional as F

        MAX, BATCH = 3, 37

        def func(x, idxs):
            return x.square() * F.one_hot(idxs, MAX)

        def jacfunc(x, idxs):
            return torch.func.jacfwd(func, argnums=(0,))(x, idxs)

        idxs = torch.randint(MAX, (BATCH,), dtype=torch.int64)
        x = torch.rand((BATCH, MAX), dtype=torch.float64)
        eager = jacfunc(x, idxs)

        compiled = torch.compile(jacfunc, backend="eager", dynamic=True)
        out_comp = compiled(x, idxs)
        self.assertEqual(eager[0], out_comp[0])

    def test_tracing_nested_py_tree_mixed_all(self):
        def fn(xs):
            flat_xs, spec = python_pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            return python_pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsa = (xs, xs)
        xsb = {"aa": xsa, "ab": xs}
        xsl = {
            "a": xs,
            "b": xsa,
            "c": xsb,
        }

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 18)

    def test_any_all_symnode(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(x):
            t = x.size(0) >= 10
            f = x.size(0) >= 100
            if any([]) or any([f]) or any([f, f]):
                return x - 1
            if all([f]) or all([t, f]) or all([f, t]) or all([f, f]):
                return x - 2
            if not (all([]) and all([t]) and all([t, t])):
                return x - 3
            if not (any([t]) and any([t, f]) and any([f, t])):
                return x - 4
            return x + 1

        y1 = torch.randn(16)
        y2 = torch.randn(18)
        self.assertEqual(fn(y1), y1 + 1)
        self.assertEqual(fn(y2), y2 + 1)
        self.assertEqual(cnt.frame_count, 1)
        y3 = torch.randn(5)
        self.assertEqual(fn(y3), y3 - 3)
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_symint_split(self):
        @torch.compile(backend="eager")
        def f(lengths, values):
            sizes = lengths.tolist()
            return torch.split(values, sizes)

        f(torch.tensor([2, 3, 4]), torch.randn(9))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_out_variant_custom_op(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "split_with_sizes_copy(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()"
            )

            @torch.library.impl(lib, "split_with_sizes_copy", "Meta")
            @torch.library.impl(lib, "split_with_sizes_copy", "CPU")
            def split_with_sizes_copy(
                all_gather_output: torch.Tensor,
                all_gather_input_split_sizes: typing.List[int],
                dim: int,
                out: typing.List[torch.Tensor],
            ) -> None:
                torch.split_with_sizes_copy(
                    all_gather_output, all_gather_input_split_sizes, dim=dim, out=out
                )

            @torch.compile(backend="eager", fullgraph=True)
            def f1(all_gather_output, all_gather_input_split_sizes, dim, out):
                return torch.ops.mylib.split_with_sizes_copy(
                    all_gather_output, all_gather_input_split_sizes, dim, out=out
                )

            all_gather_output = torch.randn(2, 272)
            all_gather_input_split_sizes = [128, 8, 128, 8]
            dim = 1
            out = [
                torch.empty(2, 128),
                torch.empty(2, 8),
                torch.empty(2, 128),
                torch.empty(2, 8),
            ]
            f1(all_gather_output, all_gather_input_split_sizes, dim, out)

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "chunk_cat(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> ()"
            )

            @torch.library.impl(lib, "chunk_cat", "Meta")
            @torch.library.impl(lib, "chunk_cat", "CPU")
            def chunk_cat(
                tensors: typing.List[torch.Tensor],
                dim: int,
                num_chunks: int,
                out: torch.Tensor,
            ) -> None:
                torch._chunk_cat(tensors, dim, num_chunks, out=out)

            @torch.compile(backend="eager", fullgraph=True)
            def f2(tensors, dim, num_chunks, out):
                return torch.ops.mylib.chunk_cat(tensors, dim, num_chunks, out=out)

            x = torch.zeros(100, dtype=torch.int64)
            tensors = [
                torch.randn(16, 16),
                torch.randn(16),
                torch.randn(16, 16),
                torch.randn(16),
            ]
            dim = 0
            num_chunks = 2
            out = torch.empty(2, 272)
            f2(tensors, dim, num_chunks, out)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_runtime_assert_replacement(self):
        @torch.compile(backend="eager")
        def fn(x, y):
            z = y.item()
            torch._check(z == 3)
            return x + z

        fn(torch.randn(4), torch.tensor([3]))
        self.assertRaises(RuntimeError, lambda: fn(torch.randn(4), torch.tensor([4])))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cat_unbacked(self):
        @torch.compile(backend="eager")
        def fn(x, y):
            z = y.item()
            return torch.cat([x, torch.ones(z)])

        self.assertRaises(
            RuntimeError, lambda: fn(torch.randn(2, 3), torch.tensor([0]))
        )
        self.assertRaises(
            RuntimeError, lambda: fn(torch.randn(2, 3), torch.tensor([1]))
        )

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_aot_autograd_propagate_unbacked_symints_shape(self):
        @torch.compile(backend="aot_eager")
        def f(x):
            return torch.nonzero(x)

        f(torch.tensor([1, 0, 3, 2, 0]))

    def test_simple_set_usage(self):
        def foo(x, y):
            setty = {x, y}
            return setty.pop() * setty.pop()

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter, fullgraph=True)
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        foo(x, y)
        self.assertEqual(counter.frame_count, 1)

    def test_add_to_set(self):
        def foo(x, y):
            setty = set()
            setty.add(x[0])
            setty.add(x[1])
            setty.add(x[2])
            setty.add(y)
            return y * len(setty)

        x = torch.randn(10, 10)
        y = torch.randn(2, 2)
        eager_result = foo([x, x, x, x, y], y)

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter, fullgraph=True)
        result = foo([x, x, x, x, y], y)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(result, eager_result)

    def test_remove_set(self):
        def fn(x):
            set_a = set((4, 5))
            set_a.remove(4)
            return x * len(set_a)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_iter_set(self):
        def foo(x, y):
            setty = set()
            for t in x:
                setty.add(t)
            return y * len(setty)

        x = torch.randn(10, 10)
        y = torch.randn(2, 2)
        eager_result = foo([x, x, x, x, y], y)

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter, fullgraph=True)
        result = foo([x, x, x, x, y], y)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(result, eager_result)

    def test_reconstruct_set_across_graph_break(self):
        def foo(x, y):
            setty = set()
            for t in x:
                setty.add(t)
            print("Break!")
            return y * len(setty)

        x = torch.randn(10, 10)
        y = torch.randn(2, 2)

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter)
        result = foo([x, x, x, x, y], y)

    def test_set_aliasing_recompiles(self):
        g1 = torch.randn(10)
        g2 = torch.randn(10)
        g3 = torch.randn(10)
        g4 = torch.randn(10)

        def foo(a, b, c):
            myset = {g1, a, b, c}
            return a + len(myset)

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter)
        # first call with no aliasing
        foo(g2, g3, g4)
        self.assertEqual(counter.frame_count, 1)

        # no aliasing again
        foo(g3, g2, g4)
        # assert no recompile
        self.assertEqual(counter.frame_count, 1)

        # aliasing changes, we should recompile
        foo(g2, g2, g2)
        self.assertEqual(counter.frame_count, 2)

        # same aliasing, different tensor
        foo(g3, g3, g3)
        self.assertEqual(counter.frame_count, 2)

        # aliasing between global and arg, should recompile again
        foo(g1, g1, g1)
        self.assertEqual(counter.frame_count, 3)

        # Reset
        torch._dynamo.reset()

        # aliasing between global and arg, first call
        foo(g1, g1, g1)
        self.assertEqual(counter.frame_count, 4)

        # same aliasing, different tensor, all local, recompile
        foo(g3, g3, g3)
        self.assertEqual(counter.frame_count, 5)

        # aliasing same tensor, we shouldn't recompile
        foo(g2, g2, g2)
        self.assertEqual(counter.frame_count, 5)

        # No aliasing
        foo(g2, g3, g4)
        self.assertEqual(counter.frame_count, 6)

        # No aliasing again
        foo(g3, g2, g4)
        # assert no recompile
        self.assertEqual(counter.frame_count, 6)

    def test_str_format_return1(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(img):
            x = torch.sin(img)
            y = f"shape {img.shape[-2:]} batch size {img.shape[0]}"
            return img + x, y

        img1 = torch.randn(1, 1, 8, 8)
        res, msg = fn(img1)
        self.assertEqual(msg, "shape torch.Size([8, 8]) batch size 1")
        self.assertEqual(res, img1 + torch.sin(img1))

    def test_str___iter__(self):
        def fn(x):
            s = "a"
            if next(s.__iter__()) == "a":
                return x + 1
            else:
                return x

        x = torch.randn(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_str_format_return2(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(img):
            x = torch.sin(img)
            y = "shape {} batch size {y:.2f}".format(img.shape[-2:], y=img.shape[0])
            return img + x, y

        img1 = torch.randn(1, 1, 8, 8)
        res, msg = fn(img1)
        self.assertEqual(msg, "shape torch.Size([8, 8]) batch size 1.00")
        self.assertEqual(res, img1 + torch.sin(img1))

    def test_sourceless_namedtuple(self):
        from collections import namedtuple

        CustomDtype = namedtuple("CustomDtype", ["dtype", "higher_dtype"])

        class CustomTensor(torch.Tensor):
            _data: torch.Tensor
            custom_dtype: CustomDtype
            __torch_function__ = torch._C._disabled_torch_function_impl
            __slots__ = [
                "_data",
                "custom_dtype",
            ]

            def __new__(
                cls,
                data: torch.Tensor,
                custom_dtype: CustomDtype,
            ):
                self = torch.Tensor._make_wrapper_subclass(
                    cls,
                    data.size(),
                    strides=data.stride(),
                    storage_offset=data.storage_offset(),
                    dtype=custom_dtype.dtype,
                    layout=data.layout,
                    requires_grad=data.requires_grad,
                    device=data.device,
                )
                self._data = data
                self.custom_dtype = custom_dtype
                return self

            def __tensor_flatten__(self):
                meta = {
                    "custom_dtype": self.custom_dtype,
                }
                return ["_data"], meta

            @staticmethod
            def __tensor_unflatten__(
                inner_tensors: dict, metadata, outer_size, outer_stride
            ):
                return CustomTensor(
                    inner_tensors["_data"],
                    metadata["custom_dtype"],
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs={}):
                return func(*args, **kwargs)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = CustomTensor(x, CustomDtype(torch.float32, torch.bfloat16))
            return y, y.custom_dtype

        fn(torch.ones(2, 2, device="cpu"))

    # Compiling autograd.Function traces fwd function twice, but the same unbacked symints were not identified
    # as the same across the two tracings. This is an unlikely situation in real use cases, so we add another
    # `test_validate_outputs_unbacked_by_custom_op` to mitigate it and keep this one as expected failure
    # until we have a proper fix.
    @unittest.expectedFailure
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_validate_outputs_unbacked(self):
        class SillyCat(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x0, x1, i):
                ctx.save_for_backward(i)
                return torch.cat([x0, x1])

            @staticmethod
            def backward(ctx, grad_out):
                (i,) = ctx.saved_tensors
                i0, i1 = i.tolist()
                g_x0, g_x1 = grad_out.split([i0, i1])
                return g_x0, g_x1, None

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, i):
            i0, i1 = i.tolist()
            x0, x1 = x.split([i0, i1])
            return SillyCat.apply(x0, x1, i)

        f(torch.randn(9, requires_grad=True), torch.tensor([3, 6]))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_validate_outputs_unbacked_by_custom_op(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b) -> (Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch.library.register_fake("mylib::foo")
            def foo_impl(x, y):
                return torch.cat([x, y])

            @torch.compile(backend="aot_eager", fullgraph=True)
            def f(x, i):
                i0, i1 = i.tolist()
                x0, x1 = x.split([i0, i1])
                return torch.ops.mylib.foo(x0, x1)

            f(torch.randn(9, requires_grad=True), torch.tensor([3, 6]))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_dim_order(self):
        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def f(x):
            x = x.permute(3, 0, 2, 1)
            return x, x.dim_order()

        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def g(x):
            return x.dim_order()

        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def h0(xs, ambiguity_check=False):
            u0, u1, u2 = xs.tolist()
            torch._check(u2 >= u0)
            torch._check(u1 >= u0)
            # stride ordering still isn't unique here, should raise
            y = torch.empty_strided([4, 4, 4], [u0, u1, u2])
            return y.dim_order(ambiguity_check=ambiguity_check)

        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def h1(xs, ambiguity_check=False):
            u0, u1, u2 = xs.tolist()
            y = torch.empty_strided([4, 4, 4], [u0, u0, u0])  # no ordering
            return y.dim_order(ambiguity_check=ambiguity_check)

        # check that for functions permuting contiguous input, the original stride is recovered with dim_order.
        def test(x):
            stride_inp = tuple(x.stride())
            f_out, f_order = f(x)
            self.assertEqual(stride_inp, tuple(f_out.stride(i) for i in f_order))

        # shape: [4, u0, 5, u1]
        x0 = torch.randn(4, 1, 5, 2)
        torch._dynamo.decorators.mark_unbacked(x0, 1)
        torch._dynamo.decorators.mark_unbacked(x0, 3)
        test(x0)

        # shape: [u0, u1, u2, u3]
        x1 = torch.randn(4, 1, 5, 2)
        for i in range(x1.ndim):
            torch._dynamo.decorators.mark_unbacked(x1, i)
        test(x1)

        # custom strides (all integers)
        x2 = torch.randn(10000)
        x2 = x2.as_strided([4, 4, 4, 4], [1, 2, 4, 8])
        assert g(x2) == (3, 2, 1, 0)

        # custom unbacked strides with no ordering: ambiguity check should raise
        xs = torch.tensor([2, 3, 4])
        h0(xs)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            r"The tensor does not have unique dim order.",
        ):
            h0(xs, ambiguity_check=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            r"The tensor does not have unique dim order.",
        ):
            h1(xs, ambiguity_check=True)

    def test_str_format_assert1(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(img):
            x = torch.sin(img)
            val = x.shape[-2:]
            torch._assert(len(val) == 2, f"shape {img.shape}")
            return img + x

        img1 = torch.randn(1, 1, 8, 8)
        res = fn(img1)
        self.assertEqual(res, img1 + torch.sin(img1))

    def test_str_format_assert2(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(img):
            x = torch.sin(img)
            torch._assert(
                img.shape[-2] == 8 and img.shape[-1] == 16, f"shape {img.shape}"
            )
            return img + x

        img1 = torch.randn(1, 3, 8, 16)
        res = fn(img1)
        self.assertEqual(res, img1 + torch.sin(img1))
        self.assertEqual(cnt.frame_count, 1)

        # trigger a recompile and graph break
        img2 = torch.randn(1, 3, 8, 15)
        self.assertRaises(AssertionError, lambda: fn(img2))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tolist_scalar(self):
        def fn(x):
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor([3])
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tolist_1d(self):
        def fn(x):
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor([2, 1])
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tolist_kd(self):
        def fn(x):
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor([[[2, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [2, 1]]])
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @patch.object(torch._dynamo.config, "specialize_int", True)
    def test_tolist_0d(self):
        def fn(x):
            new_list = []
            i = x.tolist()
            new_list.append(i * 4)
            return new_list

        x = torch.tensor(42)
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
    def test_tolist_kd_dynamic(self):
        def fn(x):
            new_list = []
            i = x.tolist()
            new_list.append(i * 4)
            return new_list, x * 10

        x = torch.randint(3, 5, [5, 5])
        eager = fn(x)
        counter = CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter, fullgraph=False)
        compiled = compiled_fn(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

        # Value change, no recompiles
        x = torch.randint(7, 9, [5, 5])
        compiled_fn(x)
        self.assertEqual(counter.frame_count, 1)

        # Size change, forced recompiles
        x = torch.randint(3, 5, [3, 3])
        compiled_fn(x)
        self.assertEqual(counter.frame_count, 2)

    def test_tolist_float(self):
        def fn(x):
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor(
            [[[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]], [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]]
        )
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter)(x)
        self.assertEqual(eager, compiled)
        # Nothing to compile here
        self.assertEqual(counter.frame_count, 0)

    def test_inline_closure_not_loaded_by_parent(self):
        def outer(a):
            return a + 1

        def indirect(x):
            return direct(x)

        def direct(x):
            def deep2(c):
                return outer(c)

            def deep(c):
                return deep2(c)

            return deep(x)

        x = torch.randn(3)
        eager = indirect(x)
        counter = CompileCounter()
        compiled = torch.compile(indirect, backend=counter)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    def test_inline_closure_returned_by_another_function_and_captures(self):
        x = torch.ones(1)

        def fn():
            def inner():
                return x + 2

            return inner

        @torch.compile
        def start():
            # Obtain the `inner` function, which holds reference to `x`.
            inner = fn()

            # When we call `inner`, we end up looking up `x` from our inlining
            # tracer, Dynamo must make sure it still has some modeling of `x` at
            # that point.
            res = inner()
            return res

        res = start()
        self.assertEqual(torch.ones(1) * 3, res)

    def test_deque_input(self):
        a = torch.randn([2, 3])
        b = torch.randn([2, 3])
        d1 = collections.deque(["foo", a, b])
        d2 = d1.copy()

        def fn(q):
            a = q.pop()
            b = q.pop()
            return a * b

        eager = fn(d1)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(d2)
        self.assertEqual(d1, d2)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    def test_deque_append_left(self):
        d1 = collections.deque(["foo", 10, 10])
        d2 = d1.copy()

        def fn(q, a, b):
            q.appendleft(a)
            q.appendleft(b)
            return q.popleft() * q.popleft()

        a = torch.randn([3, 3])
        b = torch.randn([3, 3])
        eager = fn(d1, a, b)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(d2, a, b)
        self.assertEqual(d1, d2)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)
        self.assertTrue(isinstance(compiled, torch.Tensor))

    def test_yield_from(self):
        def yield_from_fn(t_list, k):
            def yield_from_gen(l):
                l2 = [t * k for t in l]
                yield from l2

            return [t * k for t in yield_from_gen(t_list)]

        t_list = [torch.randn([2, 3]) for _ in range(3)]
        eager = yield_from_fn(t_list, 2)
        counter = CompileCounter()
        compiled = torch.compile(yield_from_fn, backend=counter)(t_list, 2)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    def test_yield_from_in_a_loop(self):
        def gen2():
            yield 1

        def gen1():
            for value in range(5):
                yield from gen2()

        def fn(x):
            c = 0
            for i in gen1():
                c = c + i
            return x + c

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.zeros(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_yield_gen_and_from(self):
        def populate_and_multiply_sequence(n, multiplier):
            # Inline generator
            def tensor_generator():
                for i in range(n):
                    yield torch.tensor([i])

            # Use 'yield from' to iterate over tensors and multiply
            t_list = [tensor * multiplier for tensor in tensor_generator()]

            def yield_from_gen():
                yield from t_list

            return [t for t in yield_from_gen()]

        multiplier = torch.tensor([10])
        eager = populate_and_multiply_sequence(5, multiplier)
        counter = CompileCounter()
        compiled = torch.compile(populate_and_multiply_sequence, backend=counter)(
            5, multiplier
        )
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    def test_yield_from_user_stop_iteration(self):
        class MyIter:
            def __init__(self, seq):
                self.seq = seq
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.index += 1
                if self.index <= len(self.seq):
                    return self.seq[self.index - 1]
                raise StopIteration(self.index)

        def yield_from_iter_fn(seq):
            def gen(seq):
                yield from MyIter(seq)

            return [i for i in gen(seq)]

        seq = [torch.randn([2, 3]) for _ in range(3)]
        eager = yield_from_iter_fn(seq)
        counter = CompileCounter()
        compiled = torch.compile(yield_from_iter_fn, backend=counter)(seq)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 0)

    # just to be sure in case anyone tries to run this in older versions of Python
    def test_pep0479_convert_stopiteration(self):
        # https://peps.python.org/pep-0479/
        def generator_with_stop_iteration():
            yield 1
            # Explicitly raising StopIteration inside the generator
            raise StopIteration("StopIteration raised within generator")
            yield 2  # This should never be reached

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                # Try to consume the generator
                gen = generator_with_stop_iteration()
                next(gen)
                next(gen)
            except RuntimeError as e:
                # Check that StopIteration was converted to RuntimeError
                # See STOPITERATION_ERROR opcode in symbolic_convert.py
                return 100
            except StopIteration:
                return 200

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, 100)

    def test_yield_send_to_subgenerator_graph_break(self):
        def subgenerator(tensor):
            multiplier = yield
            yield tensor * multiplier

        def main_generator(t_list):
            for tensor in t_list:
                subgen = subgenerator(tensor)
                next(subgen)
                yield from subgen.send(torch.tensor([10]))

        t_list = [torch.tensor([i]) for i in range(5)]
        eager = list(main_generator(t_list))

        counter = CompileCounter()
        compiled_fn = torch.compile(main_generator, backend=counter)
        compiled = list(compiled_fn(t_list))

        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 0)

    def test_derpy_nn_module_usage(self):
        def ff1(x):
            self = mod1
            return torch.sigmoid(self.mod2(x) + self.param1)

        def ff2(x):
            self = mod2
            return torch.cos(torch.sin(x) * self.param2 + 10)

        mod1 = torch.nn.Module()
        mod2 = torch.nn.Module()
        mod1.register_module("mod2", mod2)
        mod1.register_parameter("param1", torch.nn.Parameter(torch.randn(10)))
        mod1.forward = ff1
        mod2.register_parameter("param2", torch.nn.Parameter(torch.randn(10)))
        mod2.forward = ff2
        mod1.eval()

        x = torch.randn(10)
        expected = mod1(x)
        counter = CompileCounter()
        actual = torch.compile(mod1, backend=counter, fullgraph=True)(x)
        self.assertEqual(actual, expected)
        self.assertEqual(counter.op_count, 6)

    def test_default_args_device_dtype(self):
        class Foo:
            def __init__(
                self,
                dtype: torch.dtype = torch.float16,
                device: torch.device = torch.device("cpu"),
            ) -> None:
                self.value = torch.tensor(10, dtype=dtype, device=device)

        def fn():
            return Foo().value + 1

        opt_func = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn()
        res = opt_func()
        self.assertEqual(ref, res)

    def test_torch_dtype_python_type(self):
        def fn(target):
            target_dtype = target.dtype
            a = torch.zeros(2, 3, dtype=target_dtype)
            # Constant assert at trace time
            assert isinstance(target_dtype, torch.dtype)
            b = torch.zeros(2, 3, dtype=target_dtype)
            c = torch.zeros(2, 3, dtype=target_dtype)
            return a + b + c

        from torch._dynamo.variables import ConstantVariable

        dtype = torch.float16
        expected_variable = ConstantVariable(dtype)
        self.assertEqual(expected_variable.python_type(), type(dtype))

        opt_func = torch.compile(fn, backend="eager", fullgraph=True)
        a = torch.tensor([2, 3], dtype=dtype)
        res = opt_func(a)
        self.assertIsInstance(res, torch.Tensor)

    def test_iterator_limit(self):
        def fn(x):
            def gen():
                while True:
                    yield x

            return list(gen())

        x = torch.randn([0, 1, 2, 3, 4, 5])
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "infinite generator"
        ):
            compiled_fn(x)

    def test_itertools_islice(self):
        counters.clear()

        def fn(x):
            return itertools.islice(x, 2, 5, 2)

        x = torch.randn([0, 1, 2, 3, 4, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_islice_default_step(self):
        counters.clear()

        def fn(x):
            return itertools.islice(x, 2, 5)

        x = torch.randn([0, 1, 2, 3, 4, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_islice_default_end(self):
        counters.clear()

        def fn(x):
            return itertools.islice(x, 2)

        x = torch.randn([0, 1, 2, 3, 4, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_repeat(self):
        counters.clear()

        def fn(x):
            r = itertools.repeat(100.0, 5)
            for i in r:
                x += i
            return x

        x = torch.randn([2, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_infinite_repeat(self):
        counters.clear()

        def fn(x):
            r = itertools.repeat(100.0)
            idx = 0
            for i in r:
                x += i
                idx += 1
                if idx > 10:
                    break
            return x

        x = torch.randn([2, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_infinite_repeat_mutation(self):
        counters.clear()

        def fn(x):
            r = itertools.repeat(x)
            idx = 0
            for i in r:
                x += i
                i += 1
                idx += 1
                if idx > 10:
                    break
            return x

        x = torch.randn([2, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_infinite_count(self):
        for args in ([], [10], [5, -1]):
            counters.clear()

            def fn(x):
                r = itertools.count(*args)
                idx = 0
                for i in r:
                    x += i
                    idx += 1
                    if idx > 10:
                        break
                return x

            x = torch.randn([2, 5])
            eager = fn(x)

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            compiled = compiled_fn(x)

            self.assertEqual(list(eager), list(compiled))
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_infinite_cycle(self):
        counters.clear()

        def fn(x):
            for iterator in (
                iter([]),
                iter([10, 11.0]),
                itertools.repeat(-1, 3),
                itertools.count(10),
            ):
                r = itertools.cycle(iterator)
                idx = 0
                x += 1
                for i in r:
                    x += i
                    idx += 1
                    if idx > 10:
                        break
            return x

        x = torch.randn([2, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_accumulate_symint_default_sum(self):
        # https://github.com/pytorch/pytorch/issues/110287
        counters.clear()

        def fn(x):
            r = itertools.accumulate([x.size(0), x.size(1)])
            for i in r:
                x *= i
            return x

        x = torch.randn(2, 3)
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_accumulate_tensors_default_sum(self):
        counters.clear()

        def fn(a, b, c, d, x):
            l = [a, b, c, d, x]
            for i, t in enumerate(l):
                l[i] = t * x
            return itertools.accumulate(l)

        t_list = [torch.tensor([i + 1]) for i in range(4)]
        x = torch.tensor([[1, 2], [3, 4]])
        eager = fn(*t_list, x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(*t_list, x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_accumulate_tensors_builtins(self):
        for builtin_op in [operator.mul, operator.sub, operator.pow]:
            counters.clear()

            def fn(a, b, c, d, x):
                l = [a, b, c, d, x]
                for i, t in enumerate(l):
                    l[i] = t * x
                return itertools.accumulate(l, builtin_op)

            t_list = [torch.tensor([i + 1]) for i in range(4)]
            x = torch.tensor([[1, 2], [3, 4]])
            eager = fn(*t_list, x)

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            compiled = compiled_fn(*t_list, x)

            self.assertEqual(list(eager), list(compiled))
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_accumulate_tensors_kwargs(self):
        from torch._dynamo.utils import counters

        for kwargs in [
            {"func": operator.mul},
            {"initial": 100},
            {"func": operator.sub, "initial": -1},
        ]:
            counters.clear()

            def fn(a, b, c, d, x):
                l = [a, b, c, d, x]
                for i, t in enumerate(l):
                    l[i] = t * x
                return itertools.accumulate(l, **kwargs)

            t_list = [torch.tensor([i + 1]) for i in range(4)]
            x = torch.tensor([[1, 2], [3, 4]])

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            compiled = compiled_fn(*t_list, x)
            eager = fn(*t_list, x)

            self.assertEqual(list(eager), list(compiled))
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_packaging_version_parse(self):
        from packaging import version

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            x = torch.zeros(1)
            if version.parse(torch.__version__) >= version.parse("2.0.0"):
                return x + 1
            return x

        self.assertEqual(fn().item(), 1)

    def test_itertools_accumulate_tensors_user_defined(self):
        def udo_fn_0(a, b):
            return -1

        rando = random.randint(0, 1)

        def udo_fn_1(a, b):
            return a * rando + b * rando

        seen = []

        def udo_fn_2(a, b):
            seen.append(a)
            seen.append(b)
            return a * len(seen)

        for udo_fn in [udo_fn_0, udo_fn_1, udo_fn_2]:
            counters.clear()
            torch._dynamo.reset()

            def fn(a, b, c, d, x):
                l = [a, b, c, d, x]
                for i, t in enumerate(l):
                    l[i] = t * x
                return itertools.accumulate(l, udo_fn)

            t_list = [torch.tensor([i]) for i in range(4)]
            x = torch.tensor([[1, 2], [3, 4]])
            eager = fn(*t_list, x)

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            compiled = compiled_fn(*t_list, x)

            self.assertEqual(list(eager), list(compiled))
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_pure_python_accumulate(self):
        def accumulate(iterable, func=lambda x, y: x + y):
            it = iter(iterable)
            try:
                # Initialize the accumulator with the first value from the iterable
                accumulator = next(it)
            except StopIteration:
                # If the iterable is empty, return an empty generator
                return
            yield accumulator

            for element in it:
                accumulator = func(accumulator, element)
                yield accumulator

        def fn(it):
            return accumulate(it)

        t_list = [torch.tensor([i]) for i in range(4)]
        eager = fn(t_list)

        counter = CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter)
        compiled = compiled_fn(t_list)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(counter.frame_count, 1)

    def test_itertools_groupby_pure_python_default_identify_func(self):
        counters.clear()

        def fn(l):
            return [(k, list(g)) for k, g in itertools.groupby(l)]

        l = [1, 2, 2, 3, 4, 4, 4, 1, 2]
        eager = fn(l)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(l)

        self.assertEqual(eager, compiled)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_groupby_pure_python_key_func(self):
        counters.clear()

        def fn(l):
            return [(k, list(g)) for k, g in itertools.groupby(l, key=operator.neg)]

        l = [1, 2, -2, 3, 4, 4, -4, 0, -2]
        eager = fn(l)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(l)

        self.assertEqual(eager, compiled)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_tee(self):
        counters.clear()

        def fn(l):
            a, b = itertools.tee(l)
            return list(a), list(b)

        l = [1, 2, 2, 3, 4, 4, 4, 1, 2]
        eager = fn(l)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(l)

        self.assertEqual(eager, compiled)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_list_iterator_contains(self):
        def fn(x):
            it = iter(["my_weight", "not_my_weight"])
            next(it)
            if "my_weight" in it:
                return x + 2
            return x + 1

        x = torch.zeros(3)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)

        self.assertEqual(fn(x), compiled_fn(x))

    def test_storage_return(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = torch.sin(x + 1)
            storage = x.untyped_storage()
            storage.resize_(0)
            y = torch.cos(y)
            return y, storage

        x = torch.randn(10)
        expected = torch.cos(torch.sin(x + 1))
        y, s = fn(x)
        self.assertEqual(y, expected)
        self.assertEqual(x.untyped_storage().size(), 0)
        self.assertIs(s, x.untyped_storage())

    def test_flat_name_to_original_fqn(self):
        class FooBarModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter("0", torch.nn.Parameter(torch.randn(3, 4)))
                self.test_buf = torch.nn.Buffer(torch.randn(3, 4))
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )

            def forward(self, x):
                return ((x + self.test_buf) * getattr(self, "0")) / self.test_param

        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo_bar = FooBarModule()
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )
                self.test_buf = torch.nn.Buffer(torch.randn(3, 4))

            def forward(self, x):
                return (self.foo_bar(x) + self.test_param) * self.test_buf

        gm, _ = torch._dynamo.export(TestModule(), torch.randn(3, 4))
        self.assertIn("dynamo_flat_name_to_original_fqn", gm.meta)
        expected_fqn = {
            "L__self___test_param": "test_param",
            "L__self___test_buf": "test_buf",
            "L__self___foo_bar_0": "foo_bar.0",
            "L__self___foo_bar_test_param": "foo_bar.test_param",
            "L__self___foo_bar_test_buf": "foo_bar.test_buf",
        }
        self.assertEqual(expected_fqn, gm.meta["dynamo_flat_name_to_original_fqn"])

    def test_proxy_frozen_dataclass(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor

        @allow_in_graph
        def inner_fn(dc):
            return dc.x + dc.y

        def fn(x, y):
            dc = TestDataClass(x, y)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True)(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_reconstruct_frozen_dataclass(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor

        def fn(x, y):
            dc = TestDataClass(x, y)
            torch._dynamo.graph_break()
            return dc.x + dc.y

        fn_opt = torch.compile()(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

    def test_nested_dataclass_reconstruct(self):
        @dataclasses.dataclass(frozen=True)
        class NestedDataClass:
            x: int = 2

        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            y: torch.Tensor
            ndc: NestedDataClass = NestedDataClass()

        def fn(y):
            dc = TestDataClass(y)
            z = dc.y + dc.ndc.x
            return z, dc

        fn_opt = torch.compile()(fn)
        inps = (torch.ones(2, 2),)
        actual = fn_opt(*inps)
        expected = fn(*inps)

    def test_frozen_dataclass_default_value(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor
            z: int = dataclasses.field(default=5)
            a: int = 6

        @allow_in_graph
        def inner_fn(dc):
            return dc.x + dc.y + dc.z + dc.a

        def fn(x, y):
            dc = TestDataClass(x, y)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True)(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_default_factory(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor
            z: int = dataclasses.field(default_factory=list)
            a: int = dataclasses.field(default_factory=lambda: [5])

        @allow_in_graph
        def inner_fn(dc):
            return dc.x + dc.y + dc.a[0]

        def fn(x, y):
            dc = TestDataClass(x, y)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True)(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_kw_only(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor
            z: int = dataclasses.field(kw_only=True)
            a: int = dataclasses.field(kw_only=True)

        @allow_in_graph
        def inner_fn(dc):
            return dc.x + dc.y + dc.a + dc.z

        def fn(x, y):
            dc = TestDataClass(x, y, z=5, a=2)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True)(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_attr_access(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor
            z: int
            a: int

        def inner_fn(dc):
            return dc.x + dc.y + dc.a + dc.z

        def fn(x, y):
            dc = TestDataClass(x, y, z=5, a=2)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True)(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_hashable(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: float
            y: float
            z: int
            a: int

        def inner_fn(dc, x, y):
            d = {}
            d[dc] = 2
            return dc.x + dc.y + d[dc] + x + y

        def fn(x, y):
            dc = TestDataClass(x=3.2, y=2.5, z=5, a=2)
            return inner_fn(dc, x, y)

        fn_opt = torch.compile(fullgraph=True)(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)
        self.assertEqual(actual, expected)

    def test_nested_frozen_dataclass_hashable(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClassInner:
            x: float
            y: float

        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            b: TestDataClassInner
            z: int
            a: int

        def inner_fn(dc, x, y):
            d = {}
            d[dc] = 2
            return dc.b.x + dc.b.y + d[dc] + x + y

        def fn(x, y):
            dc = TestDataClass(b=TestDataClassInner(2.4, 4.4), z=5, a=2)
            return inner_fn(dc, x, y)

        fn_opt = torch.compile(fullgraph=True)(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)
        self.assertEqual(actual, expected)

    def test_shape_env_no_recording(self):
        main = ShapeEnv(should_record_events=False)

        # The main ShapeEnv should have no event recorded.
        self.assertEqual(len(main.events), 0)

        # Call create_symbolic_sizes_strides_storage_offset on both of them.
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # Create a guard: size[0] == 3 (call evaluate_expr)
        #   - +1 guard entry
        #   - +1 replacement entry
        size = r[0]
        bool(size[0] == 3)

        # The main ShapeEnv should remain with no event recorded.
        self.assertEqual(len(main.events), 0)

        if torch.fx.experimental.validator.translation_validation_enabled():
            from torch.fx.experimental.symbolic_shapes import (
                CURRENT_NODE_KEY,
                SHAPEENV_EVENT_KEY,
            )

            # Check that we don't store any recording metadata on nodes
            # from the symbolic shape FX graph.
            for n in main.graph.nodes:
                self.assertFalse(SHAPEENV_EVENT_KEY in n.meta)
                self.assertFalse(CURRENT_NODE_KEY in n.meta)

    def _replay_and_check(self, shape_env: ShapeEnv):
        if shape_env.should_record_events:
            replayed = replay_shape_env_events(shape_env.events)
            shape_env.check_equal(replayed)

    def test_shape_env_equal_empty(self):
        main, other = ShapeEnv(), ShapeEnv()
        main.check_equal(other)
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_constructor(self):
        main, other = ShapeEnv(allow_scalar_outputs=False), ShapeEnv()
        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> settings: values don't match.
  >  Left: ShapeEnvSettings(allow_scalar_outputs=False, allow_dynamic_output_shape_ops=True, assume_static_by_default=False, specialize_zero_one=True, duck_shape=True, prefer_deferred_runtime_asserts_over_guards=False, trace_asserts=False)
  > Right: ShapeEnvSettings(allow_scalar_outputs=True, allow_dynamic_output_shape_ops=True, assume_static_by_default=False, specialize_zero_one=True, duck_shape=True, prefer_deferred_runtime_asserts_over_guards=False, trace_asserts=False)
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_create_symbolic_sizes_strides_storage_offset(self):
        main, other = ShapeEnv(), ShapeEnv()
        main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> name_to_node: values don't match.
  >  Left: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {}
==> source_to_symbol: values don't match.
  >  Left: {x.size()[0]: x.size()[0], x.size()[1]: x.size()[1], x.storage_offset(): x.storage_offset(), x.stride()[0]: x.stride()[0], x.stride()[1]: x.stride()[1]}
  > Right: {}
==> source_to_var: values don't match.
  >  Left: {x.size()[0]: s93, x.size()[1]: s44}
  > Right: {}
==> unique_ids: values don't match.
  >  Left: {44, 93}
  > Right: {}
==> val_to_var: values don't match.
  >  Left: {2: s44, 3: s93}
  > Right: {}
==> var_to_range: values don't match.
  >  Left: {s44: VR[2, int_oo], s93: VR[2, int_oo]}
  > Right: {}
==> var_to_sources: values don't match.
  >  Left: {s44: [TensorPropertySource(base=ConstantSource(source_name='x'), prop=<TensorProperty.SIZE: 0>, idx=1)], s93: [TensorPropertySource(base=ConstantSource(source_name='x'), prop=<TensorProperty.SIZE: 0>, idx=0)]}
  > Right: {}
==> var_to_val: values don't match.
  >  Left: {s44: 2, s93: 3}
  > Right: {}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_unbacked(self):
        main, other = ShapeEnv(), ShapeEnv()
        main.create_unbacked_symint()
        main.create_unbacked_symfloat()
        main.create_unbacked_symbool()
        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> name_to_node: values don't match.
  >  Left: {u0, u1, zuf0}
  > Right: {}
==> unbacked_symfloat_counter: values don't match.
  >  Left: 1
  > Right: 0
==> unbacked_symint_counter: values don't match.
  >  Left: 2
  > Right: 0
==> var_to_range: values don't match.
  >  Left: {u0: VR[-int_oo, int_oo], u1: VR[0, 1], zuf0: VR[-oo, oo]}
  > Right: {}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_evaluate_expr_divisible(self):
        main, other = ShapeEnv(), ShapeEnv()

        # Call create_symbolic_sizes_strides_storage_offset on both of them.
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        other.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # Create a guard: size[0] % 3 == 0 (only in the main ShapeEnv)
        #   - +1 guard entry
        #   - +1 divisible entry
        size = r[0]
        bool(size[0] % 3 == 0)

        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> axioms: values don't match.
  >  Left: {(Mod(s93, 3)) < 0: False, (Mod(s93, 3)) <= 0: True, 0 < (Mod(s93, 3)): False, 0 <= (Mod(s93, 3)): True, Eq(0, Mod(s93, 3)): True, Eq(Mod(s93, 3), 0): True, Ne(0, Mod(s93, 3)): False, Ne(Mod(s93, 3), 0): False}
  > Right: {}
==> divisible: values don't match.
  >  Left: {Mod(s93, 3)}
  > Right: {}
==> guards: values don't match.
  >  Left: [Eq(Mod(s93, 3), 0)]
  > Right: []
==> name_to_node: values don't match.
  >  Left: {_assert, eq, mod, x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_evaluate_expr_replacement(self):
        main, other = ShapeEnv(), ShapeEnv()

        # Call create_symbolic_sizes_strides_storage_offset on both of them.
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        other.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # Create a guard: size[0] == 3 (only in the main ShapeEnv)
        #   - +1 guard entry
        #   - +1 replacement entry
        size = r[0]
        bool(size[0] == 3)

        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> axioms: values don't match.
  >  Left: {False: False, True: True}
  > Right: {}
==> guards: values don't match.
  >  Left: [Eq(s93, 3)]
  > Right: []
==> name_to_node: values don't match.
  >  Left: {_assert, eq, x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
==> replacements: values don't match.
  >  Left: {s93: 3}
  > Right: {}
==> var_to_range: values don't match.
  >  Left: {s44: VR[2, int_oo], s93: VR[3, 3]}
  > Right: {s44: VR[2, int_oo], s93: VR[2, int_oo]}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_evaluate_expr_refinement(self):
        main, other = ShapeEnv(), ShapeEnv()

        # Call create_symbolic_sizes_strides_storage_offset on both of them.
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        other.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # Create a guard: size[0] >= 3 (only in the main ShapeEnv)
        #   - +1 guard entry
        #   - +1 var_to_guard entry
        #   - Change: var_to_range
        size = r[0]
        bool(size[0] >= 3)

        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> axioms: values don't match.
  >  Left: {3 <= s93: True, s93 < 3: False}
  > Right: {}
==> guards: values don't match.
  >  Left: [s93 >= 3]
  > Right: []
==> name_to_node: values don't match.
  >  Left: {_assert, ge, x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
==> var_to_range: values don't match.
  >  Left: {s44: VR[2, int_oo], s93: VR[3, int_oo]}
  > Right: {s44: VR[2, int_oo], s93: VR[2, int_oo]}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_runtime_assert(self):
        main, other = ShapeEnv(), ShapeEnv()

        # Call create_unbacked_symint on both of them.
        r = main.create_unbacked_symint()
        other.create_unbacked_symint()

        # Create a runtime assert: r % 3 == 0 (only in the main ShapeEnv)
        #   - +1 deferred_runtime_asserts entry
        #   - Change: num_deferred_runtime_asserts
        expect_true(r % 3 == 0)

        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> axioms: values don't match.
  >  Left: {(PythonMod(u0, 3)) < 0: False, (PythonMod(u0, 3)) <= 0: True, 0 < (PythonMod(u0, 3)): False, 0 <= (PythonMod(u0, 3)): True, Eq(0, PythonMod(u0, 3)): True, Eq(PythonMod(u0, 3), 0): True, Ne(0, PythonMod(u0, 3)): False, Ne(PythonMod(u0, 3), 0): False}
  > Right: {}
==> deferred_runtime_asserts: values don't match.
  >  Left: {u0: [Eq(PythonMod(u0, 3), 0)]}
  > Right: {}
==> name_to_node: values don't match.
  >  Left: {_assert, eq, mod, u0}
  > Right: {u0}
==> num_deferred_runtime_asserts: values don't match.
  >  Left: 1
  > Right: 0
""",
        )
        self._replay_and_check(main)

    def test_shape_env_recorded_function_fallback(self):
        # Make sure the record/replay mechanism for ShapeEnv will fallback
        # if no ShapeEnv instance is found.
        constrain_range(5, min=2, max=10)
        constrain_unify(5, 5)

        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: _constrain_range_for_size(5, min=2, max=10),
            """can only constrain range for SymInt""",
        )

    def test_default_dtype_change(self):
        @torch.compile
        def foo():
            def inner(a, b, res_dtype):
                print(a, b, res_dtype)
                self.assertEqual(torch.result_type(a, b), res_dtype)

            inner(torch.tensor(1, device="cpu"), 1.0, torch.get_default_dtype())

        with set_default_dtype(torch.float):
            foo()
        with set_default_dtype(torch.double):
            foo()

    def test_numpy_ufunc_out(self):
        @torch.compile(backend="eager")
        def foo():
            x = np.arange(5)
            out = np.empty((x.shape[0], x.shape[0]))
            res_out = np.sin(x, out=out)
            assert res_out is out

        foo()

    # Unfortunately, we don't currently preserve the ids of
    # res_out and out correctly across the graph break
    @unittest.expectedFailure
    def test_numpy_ufunc_out_graph_break(self):
        @torch.compile(backend="eager")
        def foo():
            x = np.arange(5)
            out = np.empty((x.shape[0], x.shape[0]))
            res_out = np.sin(x, out=out)
            torch._dynamo.graph_break()
            assert res_out is out

        foo()

    @wrapDeterministicFlagAPITest
    def test_backward_deterministic_mode_mismatch_warning(self):
        @torch.compile
        def func(a, b):
            return a + b

        for forward_deterministic, backward_deterministic in itertools.product(
            [True, False], [True, False]
        ):
            torch.use_deterministic_algorithms(forward_deterministic)
            a = torch.randn(10, requires_grad=True)
            res = func(a, 1)
            grad = torch.ones_like(res)
            torch.use_deterministic_algorithms(backward_deterministic)

            if not forward_deterministic and backward_deterministic:
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^This compiled backward function is being run with torch\.use_deterministic_algorithms",
                ):
                    res.backward(grad)

            else:
                res.backward(grad)

    @skipIfWindows(
        msg="AssertionError: False is not true : Encountered an unexpected fallback to 'aten pow' in dynamo compiled code"
    )
    @unittest.skipIf(
        torch._inductor.config.cpu_backend != "cpp",
        "Skip for non cpp backend CPU as comments contain 'aten.pow' ",
    )
    def test_torch_dynamo_codegen_pow(self):
        def pow(x):
            return x**2

        x = np.arange(8)
        pow_opt = torch.compile(pow)

        actual, source_code = run_and_get_code(pow_opt, x)
        expect = pow(x)

        self.assertEqual(expect, actual)

        self.assertTrue(
            all("aten.pow" not in code for code in source_code),
            msg="Encountered an unexpected fallback to 'aten pow' in dynamo compiled code",
        )

    def test_graph_break_compilation_metrics(self):
        def fn(x):
            x.cos()
            torch._dynamo.graph_break()
            x.sin()
            torch._dynamo.graph_break()
            return x.cos()

        torch._dynamo.utils.clear_compilation_metrics()
        x = torch.rand((4, 4))
        f = torch.compile(fn, backend="eager")
        f(x)
        metrics = torch._dynamo.utils.get_compilation_metrics()
        # Should only be one restart per event
        (restart_reason,) = metrics[0].restart_reasons
        self.assertTrue(
            "User-inserted graph break" in restart_reason,
            "Should have logged graph break reason",
        )
        self.assertTrue(
            metrics[0].dynamo_time_before_restart_s
            <= metrics[0].entire_frame_compile_time_s
        )

        (restart_reason,) = metrics[1].restart_reasons
        self.assertTrue(
            "User-inserted graph break" in restart_reason,
            "Should have logged graph break reason",
        )
        self.assertTrue(
            metrics[1].dynamo_time_before_restart_s
            <= metrics[1].entire_frame_compile_time_s
        )

        # No restarts
        self.assertTrue(
            len(metrics[2].restart_reasons) == 0, "Last compile has no graph break"
        )
        self.assertTrue(metrics[2].dynamo_time_before_restart_s == 0)

    def test_graph_break_compilation_metrics_on_failure(self):
        def fn(x):
            return x.sin()

        def broken_backend(gm, example_inputs):
            raise RuntimeError("broken backend")

        x = torch.rand((4, 4))
        f = torch.compile(fn, backend=broken_backend)
        with unittest.mock.patch("torch._dynamo.config.suppress_errors", True):
            torch._dynamo.utils.clear_compilation_metrics()
            f(x)
            metrics = torch._dynamo.utils.get_compilation_metrics()
            for metric in metrics:
                self.assertTrue(metric.dynamo_time_before_restart_s > 0)
                self.assertTrue(
                    "RuntimeError: broken backend" in metric.fail_reason,
                    "Should have logged fail reason",
                )

    def test_compilation_metrics_size_limit(self):
        def fn1(x):
            return x.relu()

        def fn2(x):
            return x.cos()

        def fn3(x):
            return x.sin()

        def fn4(x):
            return x.exp()

        import contextlib

        @contextlib.contextmanager
        def metrics_limit_ctx():
            try:
                torch._dynamo.utils.set_compilation_metrics_limit(3)
                yield
            finally:
                torch._dynamo.utils.set_compilation_metrics_limit(
                    torch._dynamo.utils.DEFAULT_COMPILATION_METRICS_LIMIT
                )

        x = torch.rand((4, 4))
        torch._dynamo.reset()
        torch.compile(fn1, backend="eager")(x)
        torch.compile(fn2, backend="eager")(x)
        torch.compile(fn3, backend="eager")(x)
        torch.compile(fn4, backend="eager")(x)

        with metrics_limit_ctx():
            torch._dynamo.utils.clear_compilation_metrics()
            torch._dynamo.reset()
            self.assertEqual(0, len(torch._dynamo.utils.get_compilation_metrics()))
            torch.compile(fn1, backend="eager")(x)
            self.assertEqual(1, len(torch._dynamo.utils.get_compilation_metrics()))
            torch.compile(fn2, backend="eager")(x)
            self.assertEqual(2, len(torch._dynamo.utils.get_compilation_metrics()))
            torch.compile(fn3, backend="eager")(x)
            self.assertEqual(3, len(torch._dynamo.utils.get_compilation_metrics()))
            torch.compile(fn4, backend="eager")(x)
            self.assertEqual(3, len(torch._dynamo.utils.get_compilation_metrics()))

    @skipIfWindows(
        msg="TypeError: sequence item 0: expected str instance, NoneType found"
    )
    def test_funcname_cache(self):
        src = """\
import torch
if True:
    test = 3

class AAA:
    class DUMMY:
        class DUMMY2:
            pass

    def dummy(self):
        def dummy2():
            pass
    class BBB:
        @staticmethod
        def CCC():
            class DDD:
                if True:
                    @staticmethod
                    def EEE():
                        x = [torch.ones(3, 3) for _ in range(5)]
                        return x
            return DDD
def fn():
    return 3
"""
        with WritableTempFile(mode="w") as f:
            f.write(src)
            f.flush()
            from torch._dynamo.funcname_cache import get_funcname

            names = [get_funcname(f.name, i + 1) for i in range(src.count("\n") + 1)]

        self.assertExpectedInline(
            "\n".join(names),
            """\




AAA
AAA.DUMMY
AAA.DUMMY.DUMMY2
AAA.DUMMY.DUMMY2
AAA.DUMMY.DUMMY2
AAA.dummy
AAA.dummy.dummy2
AAA.dummy.dummy2
AAA.BBB
AAA.BBB
AAA.BBB.CCC
AAA.BBB.CCC.DDD
AAA.BBB.CCC.DDD
AAA.BBB.CCC.DDD
AAA.BBB.CCC.DDD.EEE
AAA.BBB.CCC.DDD.EEE
AAA.BBB.CCC.DDD.EEE
AAA.BBB.CCC
fn
fn
""",
        )

    def test_return_dict_with_graph_break_and_update(self):
        def create():
            torch._dynamo.graph_break()
            return {0: torch.tensor(3)}

        def fn():
            return {**create()}

        opt_fn = torch.compile(backend="eager")(fn)
        result = opt_fn()
        self.assertIn(0, result)
        self.assertTrue(same(result[0], torch.tensor(3)))

    def test_dynamo_reset_clears_cache(self):
        """Test that dynamo bytecode cache is freed
        when dynamo reset is called
        """

        def fn(x):
            return torch.sin(x)

        opt_fn = torch.compile(backend="eager")(fn)
        opt_fn(torch.randn(3, 3))

        c1 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c1), 1)

        torch._dynamo.reset()
        c2 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c2), 0)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_check_simplification(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            u0, u1 = x.tolist()
            torch._check((2 * u0) // (u0 + u1) != 0)
            if (2 * u0) // (u0 + u1) == 0:
                return torch.tensor(True)
            else:
                return torch.tensor(False)

        fn(torch.tensor([3, 3]))

    @torch._dynamo.config.patch(assume_static_by_default=True)
    def test_mark_unbacked_strict(self):
        @torch.compile()
        def fn(x, y):
            return torch.mul(x, y)

        x = torch.ones(5, 5)
        torch._dynamo.decorators.mark_unbacked(x, 0, strict=True)
        torch._dynamo.decorators.mark_unbacked(x, 1, strict=True)
        y = torch.randn(5, 5)

        with self.assertRaisesRegex(RuntimeError, "specialized"):
            fn(x, y)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_infer_unbacked_size_gt_zero(self):
        # This code, in fact, does NOT work in eager
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = torch.zeros(x.item())
            if y.size(0) < 0:
                assert False
            return y

        self.assertEqual(fn(torch.tensor([0])), torch.zeros(0))

    @torch.fx.experimental._config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_strict_mode(self):
        @torch.compile()
        def fn(x, y):
            if x.shape[0] == 5:
                return torch.randn(5)
            return torch.mul(x, y)

        x = torch.ones(5, 5)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)
        y = torch.randn(5, 5)
        with self.assertRaisesRegex(
            RuntimeError, "Could not guard on data-dependent expression"
        ):
            fn(x, y)

    def test_guard_size_oblivious_backed(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            y = x.size(0)
            # This doesn't actually do anything
            if guard_size_oblivious(y == 0):
                return torch.randn(1)
            else:
                return torch.randn(2)

        # Should not fail in either case
        self.assertEqual(f(torch.randn(0)).shape, (1,))
        self.assertEqual(f(torch.randn(2)).shape, (2,))

    def _test_compile_model_free(self, model_inp_ctr, weakref_watch):
        """
        Args:
        model_inp_ctr
            - constructor that returns a new model and inputs to that model
        weakref_watch
            - function that returns a layer of the model for weakref to
              finalize on, so we can check that the layer is freed after
              the model goes out of scope
        """
        cleared = False

        def finalize():
            nonlocal cleared
            cleared = True

        def run():
            mod, inp = model_inp_ctr()
            weakref.finalize(weakref_watch(mod), finalize)
            torch.compile(mod, backend="eager")(inp)

        run()
        gc.collect()
        self.assertTrue(cleared)

    def test_custom_module_free(self):
        """Test that a model is freed when it goes out of scope"""

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super(Mod, self).__init__()
                self.fc = torch.nn.Linear(100, 100)

            def forward(self, out):
                return self.fc(out)

        self._test_compile_model_free(
            lambda: (Mod(), torch.randn(100, 100)),
            lambda mod: mod.fc,
        )

    def test_sequential_module_free(self):
        self._test_compile_model_free(
            lambda: (
                torch.nn.Sequential(
                    torch.nn.Linear(100, 100),
                    torch.nn.ReLU(),
                ),
                torch.randn(100, 100),
            ),
            lambda mod: mod[0],
        )

    def test_linear_module_free(self):
        self._test_compile_model_free(
            lambda: (torch.nn.Linear(100, 100), torch.randn(100, 100)),
            lambda mod: mod,
        )

    def test_outside_linear_module_free(self):
        # Compared to test_linear_module_free, the linear
        # layer is not the code object that is directly compiled.

        # This test does not use _test_compile_model_free because of difficulty
        # in handling variable fc.

        cleared = False

        def finalize():
            nonlocal cleared
            cleared = True

        def run():
            fc = torch.nn.Linear(100, 100)

            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.fc_ref = fc

                def forward(self, x):
                    return self.fc_ref(x)

            mod = Mod()
            inp = torch.randn(100, 100)
            weakref.finalize(fc, finalize)
            torch.compile(mod, backend="eager")(inp)

        run()
        # del fc  # This should delete all the references
        gc.collect()
        self.assertTrue(cleared)

    def test_parameter_free(self):
        def model_inp_ctr():
            param = torch.nn.Parameter(torch.randn(100, 100))

            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.param = param

                def forward(self, x):
                    return self.param * x[0]

            # return param to keep it alive in _test_compile_model_free
            return Mod(), (torch.randn(100, 100), param)

        self._test_compile_model_free(model_inp_ctr, lambda mod: mod.param)

    def test_conditional_list_comp_in_context(self):
        def fn(inp):
            try:
                return [torch.sin(x) for x in inp if x is not None]
            except Exception:
                pass

        inp = [torch.randn(3, 3) for _ in range(3)] + [None]
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(inp)

    def test_312_binary_slice_with_graph_break1(self):
        l1 = torch.nn.Linear(5, 5)
        l2 = torch.nn.Linear(5, 5)

        def fn(x):
            # causes a graph break with items in the stack
            n = torch.nn.Sequential(l1, l2)
            out = n[1:](x)
            return out

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(5, 5))

    def test_312_binary_slice_with_graph_break2(self):
        class Foo:
            def __setitem__(self, key, val):
                pass

            def __getitem__(self, key):
                torch._dynamo.graph_break()
                return 1

        foo = Foo()

        def fn(x):
            # graph break in a STORE_SLICE instruction
            foo[:] = x
            # graph break in BINARY_SLICE with has_backedge check
            x = x + foo[:]
            if x is None:
                x = x + 1
            else:
                x = x + 1
            return x

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(5, 5))

    def test_super_after_graph_break(self):
        class Foo(torch.nn.Sequential):
            def __init__(self, layers):
                torch._dynamo.graph_break()
                super().__init__(*layers)

        def fn(x):
            layers = [torch.nn.Linear(3, 3) for _ in range(3)]
            mod = Foo(layers)
            return mod(x)

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(3, 3))

    def test_load_fast_and_clear_graph_break(self):
        # Can result in a segfault in 3.12+ if LOAD_FAST_AND_CLEAR
        # is not handled properly in a graph break
        def fn():
            out = torch.cat([torch.randn(r, 5) for r in range(3)])
            torch._dynamo.graph_break()
            out = torch.cat([torch.randn(r, 5) for r in range(3)])
            return out

        self.assertEqual(torch.compile(fn, backend="eager")().shape, (3, 5))

    def test_raises_importerror1(self):
        @torch.compile(backend="eager")
        def fn(x):
            try:
                import some_module_that_surely_does_not_exist

                return
            except ImportError:
                pass
            return x.sin()

        x = torch.randn(8)
        self.assertEqual(fn(x), x.sin())

    def test_raises_importerror2(self):
        @torch.compile(backend="eager")
        def fn(x):
            import some_module_that_surely_does_not_exist

            return x + 1

        x = torch.randn(8)
        with self.assertRaises(ImportError):
            fn(x)

    def test_dynamo_cache_move_to_front(self):
        def fn(x, const):
            return x + const

        # dynamic=False forces Dynamo to recompile
        opt_fn = torch.compile(fn, backend="eager", dynamic=False)

        inp = torch.randn(3, 3)

        # NOTE: assumes that each cache entry is guarded
        # on unique Mod instance
        opt_fn(inp, 1)
        opt_fn(inp, 2)
        opt_fn(inp, 3)

        c1 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c1), 3)

        # move cache entry to front
        opt_fn(inp, 2)
        c2 = _debug_get_cache_entry_list(fn.__code__)
        self.assertIs(c1[1], c2[0])

    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=False)
    @skipIfWindows(msg="TODO: (xuhancn) conform, AssertionError: False is not true")
    def test_dynamo_cache_invalidate(self):
        DeletedGuardManagerWrapper = torch._dynamo.guards.DeletedGuardManagerWrapper

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super(Mod, self).__init__()
                self.fc = torch.nn.Linear(3, 3)

            def forward(self, out):
                return self.fc(out)

        def fn(x, mod):
            return mod(x)

        opt_fn = torch.compile(fn, backend="eager")

        m1 = Mod()
        m2 = Mod()
        m3 = Mod()
        inp = torch.randn(3, 3)

        # NOTE: assumes that each cache entry is guarded
        # on unique Mod instance
        opt_fn(inp, m1)
        opt_fn(inp, m2)
        opt_fn(inp, m3)

        c1 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c1), 3)

        # move cache entry to front
        opt_fn(inp, m2)
        c2 = _debug_get_cache_entry_list(fn.__code__)
        self.assertIs(c1[1], c2[0])

        # delete center of cache
        del m3
        c3 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c3), 3)
        self.assertTrue(isinstance(c3[2].guard_manager, DeletedGuardManagerWrapper))

        # delete end of cache
        del m1
        c4 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c4), 3)
        self.assertTrue(isinstance(c4[1].guard_manager, DeletedGuardManagerWrapper))
        self.assertTrue(isinstance(c4[2].guard_manager, DeletedGuardManagerWrapper))

        del m2
        c5 = _debug_get_cache_entry_list(fn.__code__)
        self.assertTrue(isinstance(c5[0].guard_manager, DeletedGuardManagerWrapper))
        self.assertTrue(isinstance(c5[1].guard_manager, DeletedGuardManagerWrapper))
        self.assertTrue(isinstance(c5[2].guard_manager, DeletedGuardManagerWrapper))

    def test_inspect_signature_bind(self):
        import inspect

        def inner(a, b, *ar, c=10, d=11, **kw):
            pass

        def fn(x, apply_defaults):
            sig = inspect.signature(inner)
            bound = sig.bind(1, 2, 3, d=12, e=15)
            bound.arguments["d"] = 13
            if apply_defaults:
                bound.apply_defaults()
            return (
                sig,
                bound.signature,
                bound,
                bound.arguments,
                bound.args,
                bound.kwargs,
                x + 1,
            )

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        for apply_defaults in (True, False):
            _, _, bound0, arguments0, args0, kwargs0, _ = fn(
                torch.ones(3, 3), apply_defaults
            )
            _, _, bound1, arguments1, args1, kwargs1, _ = opt_fn(
                torch.ones(3, 3), apply_defaults
            )

            self.assertEqual(bound0, bound1)
            self.assertEqual(arguments0, arguments1)
            self.assertEqual(args0, args1)
            self.assertEqual(kwargs0, kwargs1)
            self.assertTrue(args1)
            self.assertTrue(kwargs1)

    def test_inspect_signature_bind_non_user_function(self):
        import inspect

        class Foo:
            def __init__(self, a, b, *ar, c=10, d=11, **kw):
                pass

        def fn(x):
            sig = inspect.signature(Foo)
            bound = sig.bind(1, 2, 3, d=12, e=15)
            return bound, x + 1

        opt_fn = torch.compile(fn, backend="eager")
        bound0, _ = fn(torch.ones(3, 3))
        bound1, _ = opt_fn(torch.ones(3, 3))

        self.assertEqual(bound0, bound1)

        import traceback

        # choose a function that is skipped but has defaults
        self.assertTrue(hasattr(traceback.print_exc, "__kwdefaults__"))
        self.assertIs(
            torch._dynamo.trace_rules.lookup(traceback.print_exc),
            torch._dynamo.variables.SkipFunctionVariable,
        )

        def gn(x):
            sig = inspect.signature(traceback.print_exc)
            bound = sig.bind()
            return bound, x + 1

        opt_gn = torch.compile(gn, backend="eager", fullgraph=True)
        bound0, _ = gn(torch.ones(3, 3))
        bound1, _ = opt_gn(torch.ones(3, 3))

        self.assertEqual(bound0, bound1)

    def test_inspect_signature_parameters(self):
        import inspect

        def fn(x, gn):
            d = inspect.signature(gn).parameters
            if d["a"].default is inspect.Parameter.empty:
                return torch.sin(x + 1)
            else:
                return torch.cos(x + 1)

        def gn(a: torch.Tensor, b: int) -> torch.Tensor:
            return a + b

        x = torch.randn(2, 3)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        self.assertEqual(fn(x, gn), opt_fn(x, gn))

    def test_grad_none(self):
        def fn(x, y):
            x.grad = torch.abs(y)
            x.grad.add_(y)
            return torch.abs(y)

        y = torch.arange(4).reshape(2, 2).to(torch.float)
        x = torch.randn(2, 2)
        x.grad = None

        z = fn(x, y)
        ref_y = torch.clone(z).detach()
        ref_x_grad = torch.clone(x.grad).detach()

        y = torch.arange(4).reshape(2, 2).to(torch.float)
        x = torch.randn(2, 2)
        x.grad = None

        opt_fn = torch.compile(fn, backend="eager")
        z = opt_fn(x, y)
        self.assertEqual(z, ref_y)
        self.assertEqual(x.grad, ref_x_grad)

    def test_grad_non_none(self):
        def fn(x, y):
            x.grad.add_(y)
            return torch.abs(y)

        y = torch.ones(2, 2)
        x = torch.randn(2, 2)
        x.grad = torch.arange(4).reshape(2, 2).to(torch.float)

        z = fn(x, y)
        ref_y = torch.clone(z).detach()
        ref_x_grad = torch.clone(x.grad).detach()

        y = torch.ones(2, 2)
        x = torch.randn(2, 2)
        x.grad = torch.arange(4).reshape(2, 2).to(torch.float)

        cnt = torch._dynamo.testing.CompileCounterWithBackend("eager")
        opt_fn = torch.compile(fn, backend=cnt)
        z = opt_fn(x, y)

        # Ensure that the generated graph returns only one output. We want the
        # add_ on the grad to be part of the graph itself, so that inductor can
        # theoretically move the add_ and resulting copy_ nodes at the right
        # place to free memory.
        self.assertEqual(len(list(cnt.graphs[0].graph.nodes)[-1].all_input_nodes), 1)
        self.assertEqual(z, ref_y)
        self.assertEqual(x.grad, ref_x_grad)

    def test_new_with_int_list(self):
        # Make sure torch.Tensor.new(int argument list) behaves the same on dynamo.
        def fn(x):
            return x.new(*x.size()) + 5

        optfn = torch.compile(backend="eager")(fn)

        x = torch.arange(10).view(2, 5)

        expected = fn(x)
        actual = optfn(x)

        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.stride(), actual.stride())
        self.assertEqual(expected.storage_offset(), actual.storage_offset())

    def test_dynamic_shapes_as_strided(self):
        def fn(t, new_size, new_stride):
            tmp = t.as_strided(new_size, new_stride)
            tmp = tmp.view(-1)
            return t * tmp.sum()

        optfn = torch.compile(backend="eager", dynamic=True)(fn)

        x = torch.randn(3)
        new_size = [0, 3]
        new_stride = [3, 1]

        expected = fn(x, new_size, new_stride)
        actual = optfn(x, new_size, new_stride)

        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.stride(), actual.stride())
        self.assertEqual(expected.storage_offset(), actual.storage_offset())

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_hasattr_nn_module_guard(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.nn.Linear(3, 3)

            def forward(self, x):
                if hasattr(self, "a"):
                    return self.a(x)
                else:
                    return x

        m = M()
        x = torch.randn(3, 3)
        ref = m(x)

        opt_m = torch.compile(backend="eager")(m)
        res = opt_m(x)
        self.assertEqual(ref, res)

    def test_ordered_dict_move_to_end(self):
        d = {
            "foo": 1,
            "bar": 2,
        }

        d = collections.OrderedDict(d)
        d.move_to_end("foo")

        @torch.compile(backend="eager")
        def fn(x, d):
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_defaultdict(self):
        d = collections.defaultdict()
        d["foo"] = 1
        d["bar"] = 2

        @torch.compile(backend="eager")
        def fn(x, d):
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_custom_dict(self):
        class MyDict(dict):
            pass

        d = {
            "foo": 1,
            "bar": 2,
        }

        d = MyDict(d)

        @torch.compile(backend="eager")
        def fn(x, d):
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_hash_hop(self):
        associative_scan = importlib.import_module(
            "torch._higher_order_ops.associative_scan"
        )

        @torch.compile(fullgraph=True)
        def fn(y, s):
            d = dict()
            d[s] = y
            return d[s] + 1.0

        fn(torch.ones(2, 2, device="cpu"), associative_scan.AssociativeScanOp())

    def test_iter_type(self):
        @torch.compile(fullgraph=True)
        def fn(y):
            x = iter([])
            if isinstance(x, list):
                return y + 1
            else:
                return y + 2

        res = fn(torch.ones(2))
        self.assertEqual(torch.ones(2) + 2, res)

    def test_descriptor(self):
        class lazy_property:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __get__(self, instance, obj_type=None):
                value = self.wrapped(instance)
                setattr(instance, self.wrapped.__name__, value)
                return value

        class UserDefined:
            def __init__(self) -> None:
                self.a = 3

            @lazy_property
            def length(self):
                return 3

            def run(self, x):
                return x * self.length

        obj = UserDefined()

        def fn(x):
            return obj.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        # Opt_fn is deliberately called first to trigger the __get__ function.
        # Otherwise, the setattr removes the lazy property.
        ref = opt_fn(x)
        res = fn(x)
        self.assertEqual(ref, res)
        ref = opt_fn(x)
        res = fn(x)
        self.assertEqual(ref, res)

    def test_descriptor_side_effect(self):
        # This pattern (readonly descriptor but writable value in `__dict__`) is
        # from scipy `_make_tuple_bunch`:
        # https://github.com/scipy/scipy/blob/maintenance/1.9.x/scipy/_lib/_bunch.py#L32-L226
        def fget(obj):
            return obj.__dict__["field"]

        class MyClass:
            def __init__(self, n):
                self.__dict__["field"] = n

            field = property(fget)

        def fn(x):
            obj = MyClass(42)
            return x + obj.field, obj

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref_t, ref_obj = fn(x)
        res_t, res_obj = opt_fn(x)
        self.assertEqual(ref_t, res_t)
        self.assertEqual(ref_obj.field, res_obj.field)

    def test_assert_size_stride(self):
        x = torch.randn(2, 3, 4)
        with self.assertRaisesRegex(
            AssertionError,
            "expected size 2==5, stride 12==9 at dim=0; expected size 3==6, stride 4==9 at dim=1; expected size 4==7, stride 1==10 at dim=2",
        ):
            torch._C._dynamo.guards.assert_size_stride(x, (5, 6, 7), (9, 9, 10))

    def test_frozen_dict(self):
        # A pattern from StableDiffusion
        class FrozenDict(collections.OrderedDict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                for key, value in self.items():
                    setattr(self, key, value)

                self.__frozen = True

            def __delitem__(self, *args, **kwargs):
                raise Exception(
                    f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
                )

            def setdefault(self, *args, **kwargs):
                raise Exception(
                    f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
                )

            def pop(self, *args, **kwargs):
                raise Exception(
                    f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
                )

            def update(self, *args, **kwargs):
                raise Exception(
                    f"You cannot use ``update`` on a {self.__class__.__name__} instance."
                )

            def __setattr__(self, name, value):
                if hasattr(self, "__frozen") and self.__frozen:
                    raise Exception(
                        f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance."
                    )
                super().__setattr__(name, value)

            def __setitem__(self, name, value):
                if hasattr(self, "__frozen") and self.__frozen:
                    raise Exception(
                        f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance."
                    )
                super().__setitem__(name, value)

        d = {"a": 1}
        frozen_d = FrozenDict(d)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            dict(frozen_d).items()
            return torch.sin(x)

        fn(torch.randn(4))

    def test_tuple_class(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            updated_x = []
            for v in x:
                updated_x.append(v + 1)
            return x.__class__(updated_x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        d1 = torch.zeros(2, 2)
        d2 = torch.ones(2, 2)

        r = opt_fn((d1, d2))
        self.assertEqual(r.__class__, tuple)
        r1, r2 = r
        self.assertEqual(r1, torch.ones(2, 2))
        self.assertEqual(r2, torch.ones(2, 2) + 1)
        self.assertEqual(cnts.frame_count, 1)

    def test_list_class(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            updated_x = []
            for v in x:
                updated_x.append(v + 1)
            return x.__class__(updated_x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        d1 = torch.zeros(2, 2)
        d2 = torch.ones(2, 2)

        r = opt_fn([d1, d2])
        self.assertEqual(r.__class__, list)
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], torch.ones(2, 2))
        self.assertEqual(r[1], torch.ones(2, 2) + 1)
        self.assertEqual(cnts.frame_count, 1)

    def test_namedtuple_class(self):
        import collections

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            updated_x = []
            for v in x:
                updated_x.append(v + 1)
            return x.__class__(*updated_x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        d1 = torch.zeros(2, 2)
        d2 = torch.ones(2, 2)
        point = collections.namedtuple("Point", ["x", "y"])
        p = point(d1, d2)

        r = opt_fn(p)
        self.assertEqual(r.__class__, point)
        self.assertEqual(r.x, torch.ones(2, 2))
        self.assertEqual(r.y, torch.ones(2, 2) + 1)
        self.assertEqual(cnts.frame_count, 1)

    def test_getattrvariable_as_python_constant(self):
        from torch._dynamo.variables.misc import GetAttrVariable

        @torch.compile(backend="eager")
        def fn(x, rand1):
            random.Random().setstate(rand1.getstate())
            return x + rand1.random()

        def get_rng():
            rand1 = random.Random(1)
            orig_random = rand1.random
            rand1.random = lambda: orig_random()
            return rand1

        x = torch.randn(3, 3)
        expected = fn.__wrapped__(x, get_rng())

        with patch.object(GetAttrVariable, "as_python_constant", autospec=True) as po:
            actual = fn(x, get_rng())

        self.assertEqual(expected, actual)
        self.assertGreater(po.call_count, 0)

    def test_data_ptr_graph_break_builtin(self):
        def f(a, b):
            # builtin + not implemented for DataPtrVariable
            return a.data_ptr() + b.data_ptr()

        a = torch.randn(4)
        b = torch.randn(5)

        # make sure there is a graph break
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(f, backend="eager", fullgraph=True)(a, b)

        torch._dynamo.reset()

        expected = f(a, b)
        actual = torch.compile(f, backend="eager")(a, b)

        self.assertEqual(expected, actual)

    def test_data_ptr_graph_break_aten(self):
        def f(a):
            # torch.add not implemented for DataPtrVariable
            return torch.add(a, a.data_ptr())

        a = torch.randn(4)

        counters.clear()

        expected = f(a)
        actual = torch.compile(f, backend="eager")(a)

        self.assertEqual(expected, actual)
        self.assertTrue(len(counters["graph_break"]) > 0)
        counters.clear()

    class AssertNumOutputBackend:
        """
        A backend that checks the number of output for compiled graph, and
        return the graph as is.
        """

        def __init__(self, test_case, expected_num_output: int):
            self.test_case = test_case
            self.expected_num_output = expected_num_output

        def __call__(self, gm: torch.fx.GraphModule, example_inputs):
            outputs = gm(*example_inputs)
            self.test_case.assertEqual(self.expected_num_output, len(outputs))
            return gm

    def test_returning_nested_func_with_captured_tensor(self):
        @torch.compile(backend=self.AssertNumOutputBackend(self, 2))
        def test():
            x = torch.rand(1)

            def func():
                return x + x

            # Returning `func` forces dynamo to output `x` in the compiled
            # graph, so that we can store it as `func`'s closure. The output of
            # compiled graph would be `(x, x + x)`.
            return func, func()

        test()

    def test_running_nested_func_with_captured_tensor(self):
        @torch.compile(backend=self.AssertNumOutputBackend(self, 1))
        def test():
            x = torch.rand(1)

            def func():
                return x + x

            # `x` is no longer needed after running the compiled graph, so we
            # shouldn't return it. The output of compiled graph would be `(x +
            # x,)`.
            return func()

        test()

    def test_returning_func_with_captured_func_and_tensor(self):
        @torch.compile(backend=self.AssertNumOutputBackend(self, 2))
        def test():
            x = torch.rand(1)

            def nested():
                return x + x

            def func():
                return nested()

            # Returning `func` forces dynamo to output `x` in the compiled
            # graph, so that we can store it as `func`'s closure. The output of
            # compiled graph would be `(x, x + x)`.
            return func, func()

        test()

    def test_running_func_with_captured_func_and_tensor(self):
        @torch.compile(backend=self.AssertNumOutputBackend(self, 1))
        def test():
            x = torch.rand(1)

            def nested():
                return x + x

            def func():
                return nested()

            # `x` is no longer needed after running the compiled graph, so we
            # shouldn't return it. The output of compiled graph would be `(x)`.
            return func()

        test()

    def test_escaping_closure_var_with_backward_hook(self):
        @torch.compile(backend=self.AssertNumOutputBackend(self, 2))
        def fn(x):
            temp = x * x
            captured_var = temp + 1

            # This is where the lambda escapes the lifetime of `fn`, so
            # dynamo must generate proper bytecode to update `captured_var`.
            x.register_hook(lambda _: captured_var)

            # The output of compiled graph would be `(x * x, x * x + 1)`.
            return temp

        ones = torch.ones(4, requires_grad=True)
        fn(ones).sum().backward()

    def test_escaping_closure_var_with_nonlocal_var(self):
        nonlocal_fn = None

        @torch.compile(backend=self.AssertNumOutputBackend(self, 2))
        def fn(x):
            temp = x * x
            captured_var = x + 1

            def inner():
                return captured_var

            # This is where `inner` escapes the lifetime of `fn`, so dynamo must
            # generate proper bytecode to update `captured_var`.
            nonlocal nonlocal_fn
            nonlocal_fn = inner

            # The output of compiled graph would be `(x * x, x * x + 1)`.
            return temp

        ones = torch.ones(4, requires_grad=True)
        fn(ones)
        nonlocal_fn()

    def test_compare_tensor_with_none(self):
        @torch.compile()
        def f(x):
            return torch.tensor(x == None)

        res = f(torch.tensor(1))
        self.assertEqual(torch.tensor(False), res)

    def test_dataclass(self):
        @dataclasses.dataclass(frozen=True)
        class Foo:
            x: int

        @torch.compile(backend="eager", fullgraph=True)
        def run(x, foo0):
            if dataclasses.is_dataclass(foo0):
                foo1 = dataclasses.replace(foo0, **{"x": 1})
                return x + 1, foo1
            return x + 2, foo0

        res, foo = run(torch.zeros(1), Foo(0))
        self.assertTrue(res, torch.ones(1))
        self.assertEqual(foo.x, 1)

    def test_frozenset_of_non_literals(self):
        class Foo:
            pass

        foo = Foo()
        foo.x = 0
        s = frozenset([foo])

        @torch.compile(backend="eager")
        def run(x, s, foo0):
            # Dynamo must have the same representation for `foo0` and `foo1`,
            # otherwise the update to `foo0.x` won't be reflected in the read of
            # `foo1.x`.
            foo1 = list(s)[0]
            foo0.x += 1
            return x + 1, foo1.x

        res = run(torch.ones(1), s, foo)
        self.assertTrue(same(res[0], torch.ones(1) + 1))
        self.assertEqual(res[1], 1)

    def test_ne_operator_with_custom_eq(self):
        class Foo:
            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                return self.x == other.x

        @torch.compile(fullgraph=True, backend="eager")
        def run(x):
            f1 = Foo(0)
            f2 = Foo(0)
            # `x + 1` prevents Dynamo from skipping this frame.
            return x + 1, f1 != f2

        _, ne = run(torch.ones(1))
        self.assertFalse(ne)

    def test_ne_operator_with_custom_ne(self):
        class Foo:
            def __init__(self, x):
                self.x = x
                self.ne_called = False

            def __ne__(self, other):
                # ne_called attr is later checked to ensure that overridden
                # `__ne__` is traced
                self.ne_called = True
                return not self.__eq__(other)

            def __eq__(self, other):
                return self.x == other.x

        f1 = Foo(0)
        f2 = Foo(0)

        @torch.compile(fullgraph=True, backend="eager")
        def run(x):
            # `x + 1` prevents Dynamo from skipping this frame.
            return x + 1, f1 != f2

        _, ne = run(torch.ones(1))
        self.assertFalse(ne)
        self.assertTrue(f1.ne_called)

    def test_ne_operator_with_custom_graphbreak_eq(self):
        counters.clear()

        class Foo:
            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                # This allows us to check that Dynamo actually traced into the
                # custom eq method.
                torch._dynamo.graph_break()
                return self.x == other.x

        @torch.compile(backend="eager")
        def run(x):
            f1 = Foo(0)
            f2 = Foo(0)
            # `x + 1` prevents Dynamo from skipping this frame.
            return x + 1, f1 != f2

        _, ne = run(torch.ones(1))
        self.assertFalse(ne)
        self.assertEqual(len(counters["graph_break"]), 1)

    @unittest.skipIf(sys.version_info < (3, 11), "Python 3.11+")
    def test_RAISE_VARARGS_0(self):
        def foo():
            try:
                raise ValueError
            except:
                raise

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                foo()
            except ValueError:
                return t.sin()
            except Exception:
                return t.cos()

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())

    def test_overridden_getattribute(self):
        class Bar:
            def __init__(self, v):
                self.v = v

        class Foo:
            attribute_map = {}

            def __init__(self):
                self.attribute_map = {
                    "a_premap": "a",
                }
                # `bar` attribute requires propagating sources correctly through
                # object.__getattribute__
                self.bar = Bar(5)

            def __setattr__(self, key, value):
                if key in super().__getattribute__("attribute_map"):
                    key = super().__getattribute__("attribute_map")[key]
                super().__setattr__(key, value)

            def __getattribute__(self, key):
                if key == "sentinel":
                    raise AttributeError()
                if key != "attribute_map" and key in super().__getattribute__(
                    "attribute_map"
                ):
                    key = super().__getattribute__("attribute_map")[key]
                return super().__getattribute__(key)

            def __getattr__(self, key):
                if key == "sentinel":
                    return 5
                raise AttributeError()

        def get_foo():
            f = Foo()
            f.a_premap = 2
            f.b = 3
            return f

        def fn(x, f):
            return x * f.a_premap * f.a * f.b * f.sentinel * f.bar.v

        x = torch.randn(4)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, get_foo()), opt_fn(x, get_foo()))

    def test_dunder_weakref(self):
        class Foo:
            pass

        def fn(x):
            foo = Foo()
            # tests isgetsetdescriptor
            if foo.__weakref__:
                return torch.cos(x)
            return torch.sin(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_guard_filter_fn_by_id(self):
        def guard_filter_fn(entries):
            return [entry.guard_type != "ID_MATCH" for entry in entries]

        @torch.compile(fullgraph=True, options={"guard_filter_fn": guard_filter_fn})
        def fn(x):
            return id(x)

        inputs = (torch.randn(3, 2),)
        fn(*inputs)

        inputs_1 = (torch.randn(3, 2),)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(fn(*inputs_1), id(inputs[0]))

    def test_guard_filter_fn_by_is_global(self):
        def guard_filter_fn(entries):
            return [not entry.is_global for entry in entries]

        global GLOBAL_INT

        @torch.compile(fullgraph=True, options={"guard_filter_fn": guard_filter_fn})
        def fn(x):
            return x + GLOBAL_INT

        GLOBAL_INT = 1
        fn(torch.randn(3, 2))

        GLOBAL_INT = 2
        inputs = (torch.randn(3, 2),)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(fn(*inputs), inputs[0] + 1)

    def test_guard_filter_fn_by_name_and_value(self):
        def guard_filter_fn(entries):
            return [
                not (entry.name == "y" and entry.value is None) for entry in entries
            ]

        @torch.compile(fullgraph=True, options={"guard_filter_fn": guard_filter_fn})
        def fn(x, y):
            if y is not None:
                x += y
            return x

        fn(torch.randn(3, 2), None)

        inputs = (torch.randn(3, 2), torch.tensor(1))
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(fn(*inputs), inputs[0])

    def test_guard_filter_inbuilt_nn_modules(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                return self.norm(x)

        mod = Mod()
        opt_mod = torch.compile(
            mod,
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe
            },
        )

        x = torch.rand(4, 8)
        opt_mod(x)

        mod.norm.eps = 1e-02
        # Since the guards are skipped on inbuilt nn modules, we should not recompile
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            opt_mod(x)

    def test_guard_filter_nn_modules(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = 2
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                return self.norm(x) + self.c

        mod = Mod()
        opt_mod = torch.compile(
            mod,
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe
            },
        )

        x = torch.rand(4, 8)
        opt_mod(x)

        mod.c = 3
        # Since the guards are skipped on all nn modules, we should not recompile
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            opt_mod(x)

    def test_guard_filter_tensors(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = 2.0
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                return self.norm(x) + self.c

        mod = Mod()
        opt_mod = torch.compile(
            mod,
            options={
                "guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe,
            },
        )

        x = torch.rand(4, 8)
        opt_mod(x)

        mod.c = 3.0
        # Since the guards are skipped on all tensors
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            opt_mod(x)

    def test_guard_filter_globals(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = 2
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                return self.norm(x) + self.c + GLOBAL_INT

        mod = Mod()
        opt_mod = torch.compile(
            mod,
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_globals_unsafe,
            },
        )

        global GLOBAL_INT
        GLOBAL_INT = 1
        x = torch.rand(4, 8)
        opt_mod(x)

        GLOBAL_INT = 2
        # Since the guards are skipped on globals, we should not recompile
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            opt_mod(x)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_builtin_bool_on_symint(self):
        def f(x):
            return bool(x.item())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randint(10, (1,))

        ref = f(x)
        res = opt_f(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_builtin_bool_on_symfloat(self):
        def f(x):
            return bool(x.item())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(1)

        ref = f(x)
        res = opt_f(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_builtin_bool_on_symbool(self):
        def f(x):
            return bool(x.item())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(1) == 1

        ref = f(x)
        res = opt_f(x)
        self.assertEqual(ref, res)

    def test_builtin_complex(self):
        def f(x):
            c = (
                complex(),
                complex(1),
                complex(2, 3),
                complex(imag=2),
                complex(real=1),
                complex(imag=1, real=2),
                complex("1+2j"),
                complex(1, 2).conjugate(),
            )
            return [x + z for z in c]

        x = torch.randn(1)
        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        res = opt_f(x)
        ref = f(x)
        self.assertEqual(res, ref)

    def test_builtin_complex_args(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(*args, **kwargs):
            return torch.tensor(complex(*args, **kwargs))

        self.assertRaises(Unsupported, f, 1, 1, 1)
        self.assertRaises(Unsupported, f, 1, 1, fake_arg=1)
        self.assertRaises(Unsupported, f, fake_arg=1)
        self.assertRaises(Unsupported, f, [])
        self.assertRaises(Unsupported, f, "1 + j")

    def test_compiled_class_graph_break(self):
        counter = CompileCounter()

        @torch.compile(backend=counter, fullgraph=False)
        def f(x):
            x += 1

            class C:
                pass

            return x.sin()

        x = torch.randn(3)
        f(x)
        self.assertEqual(counter.frame_count, 2)


class MiscTestsPyTree(torch._inductor.test_case.TestCase):
    @parametrize_pytree_module
    def test_tracing_pytree(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]

        counter = CompileCounter()
        torch.compile(fn, backend=counter, fullgraph=True)(xs)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 3)

    @parametrize_pytree_module
    def test_tracing_nested_pytree(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsl = [xs, xs, xs, xs]

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 12)

    @parametrize_pytree_module
    def test_tracing_nested_tuples(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsl = (xs, xs, xs, xs)

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 12)

    @parametrize_pytree_module
    def test_tracing_nested_dicts(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsl = {
            "a": xs,
            "b": xs,
            "c": xs,
        }

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 9)

    @parametrize_pytree_module
    def test_tracing_nested_mixed_all(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsa = (xs, xs)
        xsb = {"aa": xsa, "ab": xs}
        xsl = {
            "a": xs,
            "b": xsa,
            "c": xsb,
        }

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 18)

    @parametrize_pytree_module
    def test_tracing_nested_tensor_subclass(self, pytree):
        from torch.testing._internal.two_tensor import TwoTensor
        from torch.utils.checkpoint import checkpoint

        def fn(xs):
            nested_xs = [[xs]]
            flat_xs, spec = pytree.tree_flatten(xs)
            return flat_xs[0].clone()

        # use checkpoint to trigger a "sourceless" tensor subclass
        def checkpoint_fn(xs):
            return checkpoint(fn, xs, use_reentrant=True)

        xs = TwoTensor(torch.ones(2, 2), torch.ones(2, 2))

        counter = CompileCounter()
        torch.compile(checkpoint_fn, backend=counter, fullgraph=True)(xs)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 2)

    @parametrize_pytree_module
    def test_pytree_tree_leaves(self, pytree):
        def fn(x):
            tree = {
                "a": [x, x - 1],
                "b": x + 2,
                "c": (
                    x,
                    3.0,
                    collections.deque([0.0, -x, 1, 2], maxlen=3),
                ),
                "d": collections.OrderedDict(
                    {
                        "e": torch.return_types.qr((2 * x, None)),
                        "f": MyTuple(x, x + 1, torch.zeros(4, 3)),
                    },
                ),
            }
            leaves = pytree.tree_leaves(tree)
            return leaves

        x = torch.randn(3, 2)
        expected = fn(x)
        fn_opt = torch.compile(fullgraph=True)(fn)
        actual = fn_opt(x)

        self.assertEqual(actual, expected)

    @parametrize_pytree_module
    def test_pytree_tree_flatten_unflatten(self, pytree):
        def fn(x, y):
            tree = {
                "a": [x, x - 1],
                "b": x + 2,
                "c": (
                    x,
                    3.0,
                    collections.deque([0.0, -x, 1, 2], maxlen=3),
                ),
                "d": collections.OrderedDict(
                    {
                        "e": torch.return_types.qr((2 * x, None)),
                        "f": MyTuple(x, x + 1, torch.zeros(4, 3)),
                    },
                ),
            }
            leaves, treespec = pytree.tree_flatten(tree)
            new_leaves = [
                x - 1,
                y,
                x * y,
                3.0,
                y - 2,
                1,
                torch.zeros(2, 2),
                2 * y,
                -y,
                x + y,
                x - y,
                torch.ones(3, 2),
                1,
            ]
            if pytree.__name__ == "optree":
                # `None` is a internal node rather than leaf in default OpTree / JAX PyTree
                new_leaves.pop()
                # The treespec argument comes first in OpTree / JAX PyTree
                new_tree = pytree.tree_unflatten(treespec, new_leaves)
            else:
                new_tree = pytree.tree_unflatten(new_leaves, treespec)
            return leaves, new_tree

        x = torch.randn(3, 2)
        y = torch.randn(3, 2)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True)(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    @parametrize_pytree_module
    def test_pytree_tree_map(self, pytree):
        def fn(x, y):
            tree1 = {
                "a": [x, x - 1],
                "b": x + 2,
                "c": (
                    x,
                    3.0,
                    collections.deque([0.0, -x, 1, 2], maxlen=3),
                ),
                "d": collections.OrderedDict(
                    {
                        "e": torch.return_types.qr((2 * x, None)),
                        "f": MyTuple(x, x + 1, torch.zeros(4, 3)),
                    },
                ),
            }
            tree2 = collections.OrderedDict(
                [
                    ("c", (y, 3.0, collections.deque([1, -y, 10.0]))),
                    ("a", [y, y + 1]),
                    ("b", y + 2),
                    (
                        "d",
                        {
                            "f": MyTuple(torch.ones(4, 3), -y, y + 1),
                            "e": torch.return_types.qr((2 * y, None)),
                        },
                    ),
                ],
            )
            return pytree.tree_map(lambda u, v: (u, v), tree1, tree2)

        x = torch.randn(3, 2)
        y = torch.randn(3, 2)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True)(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    @parametrize_pytree_module
    def test_pytree_tree_map_dict_order(self, pytree):
        def fn(tree):
            new_tree = pytree.tree_map(lambda x: x, tree)
            return list(new_tree.keys()), list(new_tree.values())

        x = torch.randn(3, 2)
        fn_opt = torch.compile(fullgraph=True)(fn)

        tree1 = {"b": x + 2, "a": x, "c": x - 1}
        expected1 = fn(tree1)
        actual1 = fn_opt(tree1)
        self.assertEqual(actual1, expected1)

        tree2 = collections.OrderedDict([("b", x + 2), ("a", x), ("c", x - 1)])
        expected2 = fn(tree2)
        actual2 = fn_opt(tree2)
        self.assertEqual(actual2, expected2)

        tree3 = collections.defaultdict(int, {"b": x + 2, "a": x, "c": x - 1})
        expected3 = fn(tree3)
        actual3 = fn_opt(tree3)
        self.assertEqual(actual3, expected3)

    @parametrize_pytree_module
    def test_pytree_tree_map_only(self, pytree):
        if not callable(getattr(pytree, "tree_map_only", None)):
            # OpTree and JAX PyTree do not have `tree_map_only`
            return

        def fn(xs):
            def mapper(x):
                return x.clone()

            y = pytree.tree_map_only(torch.Tensor, mapper, xs)
            return y

        xs = [torch.tensor(i) for i in range(3)] + ["hi"]
        xsa = (xs, xs)
        xsb = {"aa": xsa, "ab": xs}

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsb)
        real_out = fn(xsb)

        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 9)

    def test_pytree_register_constant_with_side_effect(self):
        class Foo:
            pass

        class Bar:
            def __eq__(self, other):
                return super().__eq__(other)

            def __hash__(self):
                return 0

        python_pytree.register_constant(Bar)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, obj):
            obj.attr = {3: Bar()}
            return x + 1

        inp = torch.ones(3)
        self.assertEqual(fn(inp, Foo()), inp + 1)


class TestTracer(JitTestCase):
    def test_jit_save(self):
        def fn():
            class Foo(torch.nn.Module):
                def __init__(self) -> None:
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
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn()


class TestCustomFunction(torch.testing._internal.common_utils.TestCase):
    def test_autograd_function_with_matmul_folding_at_output(self):
        """
        When tensor folding occurs during matmul operation returned tensor is a view.
        This can cause issues when matmul is used inside a custom function
        and such view is then returned as output. Then it cannot be modified inplace
        and causes errors.
        It can be especially problematic when after such function inplace allreduce
        is performed. This test recreates this behaviour.
        Issue is resolved when unsafe_view is returned from matmul instead.
        """

        class CustomFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp1, inp2):
                ctx.save_for_backward(inp2)
                ctx.output_shape = inp1.size()
                return torch.matmul(inp1, inp2)

            @staticmethod
            def backward(ctx, grad_output):
                output_shape = ctx.output_shape
                (inp2,) = ctx.saved_tensors
                return (
                    torch.mm(grad_output.squeeze(), inp2.t()).view(output_shape),
                    None,
                )

        def outer_function(inp1, inp2):
            res = CustomFunction.apply(inp1, inp2)
            res.add_(1.0)
            return res.sum()

        def usual_function(inp1, inp2) -> torch.Tensor:
            res = torch.matmul(inp1, inp2)
            res.add_(1.0)
            return res.sum()

        inp1_custom = torch.randn(4, 1, 2, requires_grad=True)
        inp1_usual = inp1_custom.detach().clone().requires_grad_(True)

        inp2 = torch.randn(2, 4)
        c_custom_func = torch.compile(outer_function)
        c_usual_func = torch.compile(usual_function)

        result_custom = c_custom_func(inp1_custom, inp2)
        result_custom.backward()
        result_usual = c_usual_func(inp1_usual, inp2)
        result_usual.backward()

        torch.allclose(inp1_custom.grad, inp1_usual.grad)

    def test_retain_grad(self):
        def fn(x, y):
            y.retain_grad()
            return torch.sin(y) + x

        opt_fn = torch.compile(fn, backend="aot_eager")
        x = torch.randn(4, requires_grad=True)
        y = torch.cos(x)
        opt_fn(x, y).sum().backward()
        self.assertTrue(y.grad is not None)


class MiscTestsDevice(torch._inductor.test_case.TestCase):
    def test_rand(self, device):
        cnts = torch._dynamo.testing.CompileCounter()
        device = device

        def fn():
            return torch.randn(10, device=device)

        torch.manual_seed(10)
        ref_run1 = fn()

        torch.manual_seed(10)
        ref_run2 = fn()
        self.assertTrue(same(ref_run1, ref_run2))

        torch.manual_seed(10)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn()

        self.assertTrue(same(res, ref_run1))

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Can't run fused SDPA on this platform",
    )
    def test_parsing_sdpa(self, device):
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

        device = device
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
        opt_mod = torch.compile(module, backend="inductor")
        opt_mod(query, key, value)

    def test_torch_device_is_available(self, device):
        def fn(x):
            if torch.accelerator.is_available():
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "requires cudnn")
    def test_torch_cudnn_is_acceptable(self, device):
        def fn(x):
            if torch.backends.cudnn.is_acceptable(tensor=x):
                return x + 1
            return x

        x = torch.rand(4).to(device)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "requires cudnn")
    def test_torch_cudnn_is_acceptable_bad_inputs(self, device):
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
            x1 = torch.rand(4).to(device)
            opt_fn1 = torch.compile(fn1, backend="eager", fullgraph=True)
            res1 = opt_fn1(x1)

        with self.assertRaisesRegex(
            AssertionError, "Expect 1 input to cudnn.is_acceptable"
        ):
            x2 = torch.rand(4).to(device)
            opt_fn2 = torch.compile(fn2, backend="eager", fullgraph=True)
            res = opt_fn2(x2)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @torch._dynamo.config.patch(recompile_limit=999)
    def test_legacy_cuda_tensor(self):
        typs = [
            torch.cuda.FloatTensor,
            torch.cuda.DoubleTensor,
            torch.cuda.HalfTensor,
            torch.cuda.BFloat16Tensor,
            torch.cuda.ByteTensor,
            torch.cuda.CharTensor,
            torch.cuda.IntTensor,
            torch.cuda.ShortTensor,
            torch.cuda.LongTensor,
        ]

        def f2(typ):
            return typ([1, 2, 3])

        compiled_f2 = torch.compile(f2, backend="eager", fullgraph=True)
        for typ in typs:
            output = compiled_f2(typ)
            expected = f2(typ)
            self.assertEqual(output, expected)

    def test_get_device(self, device):
        def fn(x, y):
            x = x + 1
            y = y + 1
            return x.get_device(), y.get_device()

        x = torch.rand(4, device=device)
        y = torch.rand(4, device="cpu")
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_symint_as_device_kwarg(self, device):
        def f(rank):
            # -2 to make device id 0 for easier testing on CI
            return torch.ones(10, device=rank.size(0) - 2)

        x = torch.randn(2)
        out = f(torch.randn(2))
        opt_out = torch.compile(backend="eager", dynamic=True, fullgraph=True)(f)(x)
        self.assertEqual(out, opt_out)

    @unittest.skipIf(not TEST_MULTIGPU, "need multiple GPU")
    def test_gpu_set_device(self, device):
        def fn():
            a = torch.ones(2, device=device)
            torch.get_device_module(device).set_device(1)
            return a + 1

        with torch.get_device_module(device).device(0):
            counter = CompileCounter()
            opt_fn = torch.compile(fn, backend=counter)
            res = opt_fn()
            self.assertTrue(res.device.type in device)
            self.assertEqual(res.device.index, 0)
            self.assertEqual(counter.frame_count, 2)

    def test_torch_device_python_type(self, device):
        device_type = torch.device(device).type
        for device, device_type, index in [
            ("cpu", "cpu", None),
            (device, device_type, 0),
        ]:

            def fn(target):
                target_device = target.device
                a = torch.zeros(2, 3, device=target_device)
                # Constant assert at trace time
                assert isinstance(target_device, torch.device)
                assert target_device.type == device_type
                assert target_device.index == index
                b = torch.zeros(2, 3, device=target_device)
                c = torch.zeros(2, 3, device=target_device)
                return a + b + c

            from torch._dynamo.variables import ConstantVariable

            device = torch.device(device)
            expected_variable = ConstantVariable(device)
            self.assertEqual(expected_variable.python_type(), type(device))

            opt_func = torch.compile(fn, backend="eager", fullgraph=True)
            a = torch.tensor([2, 3], device=device)
            res = opt_func(a)
            self.assertIsInstance(res, torch.Tensor)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True)
    def test_interpolate_propagate_real_tensors(self, device):
        @torch.compile(backend="eager", fullgraph=True)
        def f(mask, box):
            # u0, u1 = mask.tolist()
            mask = torch.randn(1, 1, 30, 30, device=device)
            h, w = box.tolist()
            return torch.nn.functional.interpolate(
                mask, (h, w), mode="bilinear", align_corners=False
            )

        f(torch.tensor([30, 30], device=device), torch.tensor([68, 32], device=device))

    def test_scalar_isin_decomposition(self):
        def f():
            x = torch.tensor(0)
            return torch.isin(x, x)

        opt_f = torch.compile(f, backend="inductor", fullgraph=True)
        ref = f()
        res = opt_f()
        self.assertEqual(ref, res)

    def test_randint_no_graphbreak(self):
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(actions, n_act, epsilon=0.1):
            actions_random = torch.randint_like(actions, n_act)

            return actions_random

        x = torch.ones([1], dtype=torch.int64)
        y = torch.tensor(5)
        f(x, y)

    def test_full_graph_capture_scalar_outputs(self):
        @torch.compile(fullgraph=True)
        def foo(a):
            return torch.randn(5) * a.item()

        # We expect to no longer raise here
        foo(torch.tensor(2.0))

    def test_full_graph_capture_dynamic_output_shape_ops(self):
        def fn(x):
            nz = torch.nonzero(x)
            squared = nz * nz
            sliced = torch.ops.aten.slice.Tensor(squared, dim=1, start=-2, end=None)
            view = sliced.unsqueeze(dim=0)
            return view.squeeze(dim=0)

        example_inputs = (torch.randn(1, 1, 1, 1),)
        # we expect to no longer raise here
        torch.compile(fn, fullgraph=True)(*example_inputs)

    def test_dynamic_fill_diagonal_(self):
        @torch.compile(dynamic=True)
        def f(x):
            x.fill_diagonal_(True)

        x = torch.zeros(4, 4)
        f(x)

    def test_dynamic_float_scalar_tensor_coersion(self):
        # Minified version of https://github.com/pytorch/pytorch/issues/158376#issuecomment-3079591367
        class Foo:
            def __init__(self):
                self.config = type(
                    "Config", (), {"pad_val": 1123581321.0, "tolerance": 1e-6}
                )

            @torch.compile(fullgraph=True)
            def forward(self, input):
                outputs = torch.where(
                    torch.abs(input - self.config.pad_val) < self.config.tolerance,
                    torch.tensor(
                        self.config.pad_val, dtype=input.dtype, device=input.device
                    ),
                    torch.tensor(
                        self.config.pad_val + 1, dtype=input.dtype, device=input.device
                    ),
                )
                return outputs

        foo = Foo()
        inputs = torch.randn(3, 4)
        result = foo.forward(inputs)

        original_pad_val = foo.config.pad_val
        foo.config.pad_val += 1.0
        result2 = foo.forward(inputs)

        # Previously would crash with:
        #   RuntimeError: value cannot be converted to type at::Half without overflow


instantiate_parametrized_tests(MiscTestsPyTree)

devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    MiscTestsDevice, globals(), only_for=devices, allow_xpu=True
)


class DynamoOpPromotionTests(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(not TEST_CUDA, "This test requires a CUDA device")
    def test_symbool_tensor_mul(self):
        def symbool_mul_fn(x_bool, sentinel):
            result = x_bool * sentinel
            return result

        x_true = torch.tensor([True], device="cuda")
        x_false = torch.tensor([False], device="cuda")
        sentinel = torch.tensor(2.0, requires_grad=True, device="cuda")
        eager_result_true = symbool_mul_fn(x_true, sentinel)
        eager_result_false = symbool_mul_fn(x_false, sentinel)
        compiled_fn = torch.compile(symbool_mul_fn, fullgraph=True, dynamic=True)
        compiled_result_true = compiled_fn(x_true, sentinel)
        compiled_result_false = compiled_fn(x_false, sentinel)
        self.assertEqual(eager_result_true, compiled_result_true)
        self.assertEqual(eager_result_false, compiled_result_false)
        self.assertEqual(compiled_result_true.item(), 2.0)
        self.assertEqual(compiled_result_false.item(), 0.0)

    @unittest.skipIf(not TEST_CUDA, "This test requires a CUDA device")
    def test_symbool_guard_or_false(self):
        def symbool_guard_fn(a_bool_tensor, b):
            u0 = a_bool_tensor.item()
            # Make sure guard_or_false still handles SymBool produced by .item()
            if guard_or_false(u0):
                return b * 10
            else:
                return b * 100

        compiled_guard_fn = torch.compile(
            symbool_guard_fn, backend="eager", dynamic=True
        )
        a_true = torch.tensor(True, device="cuda")
        a_false = torch.tensor(False, device="cuda")
        b = torch.randn(6, device="cuda")
        eager_res_true = symbool_guard_fn(a_true, b)
        compiled_res_true = compiled_guard_fn(a_true, b)
        self.assertEqual(eager_res_true, compiled_res_true)
        eager_res_false = symbool_guard_fn(a_false, b)
        compiled_res_false = compiled_guard_fn(a_false, b)
        self.assertEqual(eager_res_false, compiled_res_false)
        self.assertEqual(compiled_res_true, b * 10)
        self.assertEqual(compiled_res_false, b * 100)

    @unittest.skipIf(not TEST_CUDA, "This test requires a CUDA device")
    def test_symbool_tensor_mul_does_not_fail(self):
        def fuzzed_program(arg_0, sentinel):
            var_node_2 = arg_0
            var_node_1 = torch.squeeze(var_node_2)
            var_node_0 = var_node_1.item()
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        sentinel = torch.tensor(1.0, requires_grad=True, device="cuda")
        arg_0 = torch.tensor([True], dtype=torch.bool, device="cuda")
        args = (arg_0,) + (sentinel,)
        try:
            compiled_program = torch.compile(
                fuzzed_program, fullgraph=True, dynamic=True
            )
            compiled_program(*args)
        except Exception as e:
            self.fail(f"torch.compile failed with error: {e}")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
