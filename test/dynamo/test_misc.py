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
import subprocess
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
from torch._dynamo.comptime import comptime
from torch._dynamo.decorators import _DimRange
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
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    PYTORCH_CUDA_MEMCHECK,
)
from torch.testing._internal.common_methods_invocations import (
    sample_inputs_take_along_dim,
)
from torch.testing._internal.common_utils import (
    freeze_rng_state,
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    recover_orig_fp32_precision,
    scoped_load_inline,
    set_default_dtype,
    skipCUDAMemoryLeakCheckIf,
    skipIfHpu,
    skipIfNNModuleInlined,
    skipIfWindows,
    subtest,
    TEST_HPU,
    TEST_XPU,
    wrapDeterministicFlagAPITest,
)
from torch.testing._internal.jit_utils import JitTestCase
from torch.utils._sympy.numbers import int_oo


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


# Module-level function used as a torch._check message (UserFunctionVariable,
# as opposed to a nested closure defined inside the compiled region).
def global_check_message():
    return "global check message"


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
        assert torch.compiler.is_dynamo_compiling() or UserDefineSetAttr.setup  # noqa: S101
        super().__setattr__(f"pfx_{key}", value)

    def __getattr__(self, key, c=1):
        assert torch.compiler.is_dynamo_compiling() or UserDefineSetAttr.setup  # noqa: S101
        # c is added to force a guard on __defaults__ and checks the source for __getattr__
        if c:
            return self.__dict__[f"pfx_{key}"]
        else:
            return None


class MiscTests(torch._inductor.test_case.TestCase):
    def test_get_cache_entry(self):
        def f(x):
            return x + 1

        torch.compile(f, backend="eager")(torch.randn(5, 5, 5))
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

        torch.compile(h, backend="eager")(torch.randn(3, 3))

        entries = _debug_get_cache_entry_list(torch._dynamo.graph_break)
        self.assertEqual(len(entries), 0)

    @torch.testing._internal.common_utils.scoped_load_inline
    def test_pybind11_enum_conversion(self, load_inline):
        cpp_source = """
        #include <torch/extension.h>

        enum class E { A = 0, B = 1 };

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            py::enum_<E>(m, "E")
                .value("A", E::A)
                .value("B", E::B);
        }
        """
        mod = load_inline(name="pybind11_enum_test", cpp_sources=cpp_source)
        e = mod.E.A
        self.assertEqual(
            torch.compile(lambda x: int(x), backend="eager", fullgraph=True)(e), 0
        )
        self.assertEqual(
            torch.compile(lambda x: float(x), backend="eager", fullgraph=True)(e), 0.0
        )
        self.assertEqual(
            torch.compile(lambda x: [10, 20][x], backend="eager", fullgraph=True)(
                mod.E.B
            ),
            20,
        )

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

    @unittest.skipIf(not TEST_CUDA, "cuda needed")
    def test_assume_32_bit_indexing(self):
        @torch.compile(backend="inductor")
        def func(a, b):
            # Multiple concat operations
            x = torch.concat([a, b], dim=0)
            y = torch.concat([a, b], dim=1)

            # Reshape to create indexing patterns
            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)

            # Take the smaller one and expand
            min_size = min(x_flat.shape[0], y_flat.shape[0])
            x_trunc = x_flat[:min_size]
            y_trunc = y_flat[:min_size]

            # Combine and compute
            result = (x_trunc + y_trunc) * 10

            # Cumulative operations create complex indexing
            cumsum = result.cumsum(dim=0)

            return cumsum.sum()

        a = torch.rand(100, 30, device="cuda")
        b = torch.rand(100, 30, device="cuda")

        torch._dynamo.decorators.mark_unbacked(a, 0)
        torch._dynamo.decorators.mark_unbacked(a, 1)
        torch._dynamo.decorators.mark_unbacked(b, 0)
        torch._dynamo.decorators.mark_unbacked(b, 1)

        source_code = run_and_get_code(func, a, b)[1]
        # Check that int64 indexing is used (either 1D [:] or 2D [:, None] form)
        self.assertTrue(
            "tl.arange(0, XBLOCK)[:].to(tl.int64)" in str(source_code)
            or "tl.arange(0, XBLOCK)[:, None].to(tl.int64)" in str(source_code)
        )
        # Check that 32-bit indexing is NOT used
        self.assertFalse(
            "tl.arange(0, XBLOCK)[:]\n" in str(source_code)
            and ".to(tl.int64)" not in str(source_code)
        )

        torch._dynamo.reset()

        with torch._inductor.config.patch(assume_32bit_indexing=True):
            source_code = run_and_get_code(func, a, b)[1]
            # Check that int64 indexing is NOT used when assume_32bit_indexing=True
            self.assertFalse(
                "tl.arange(0, XBLOCK)[:].to(tl.int64)" in str(source_code)
                or "tl.arange(0, XBLOCK)[:, None].to(tl.int64)" in str(source_code)
            )

    def test_dynamo_side_effect(self):
        class GlobalContext:
            def __init__(self):
                self._tensors = {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class Module(torch.nn.Module):
            def forward(self, x):
                with GlobalContext() as ctx:
                    z = x + 1
                    ctx._tensors["6"] = x + 2
                return z, ctx

        mod = Module()
        inp = torch.randn(4, 4)
        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="warn"
        ):
            val = torch.compile(mod, backend="eager")(inp)
            # Verify new object is properly initialized
            self.assertIn("6", val[1]._tensors)
            self.assertEqual(val[1]._tensors["6"], inp + 2)

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
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=0] = placeholder[target=arg1_1]
    %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%arg0_1,), kwargs = {})
    %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%sin,), kwargs = {})
    return (cos,)""",
            )
            self.assertExpectedInline(
                str(cnt1.inductor_graphs[0].graph).strip(),
                """\
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=0] = placeholder[target=arg1_1]
    %foo : [num_users=1] = call_function[target=torch.ops.mylib.foo.default](args = (%arg0_1,), kwargs = {})
    return (foo,)""",
            )

    def test_compile_non_infra_inside_compile(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(func, backend=backend, fullgraph=True)(
                    *args, **kwargs
                )
                return out

        x = torch.randn(5)
        with YoloMode():
            out = torch.add(x, x)

        self.assertEqual(len(backend.graphs), 1)

    def test_compile_non_infra_empty(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return torch.ops.aten.mul.Tensor(args[0], args[1])

        x = torch.ones(5)
        with YoloMode():
            out = torch.compile(torch.add, backend=backend, fullgraph=False)(x, x)
        self.assertEqual(out.sum().item(), 5.0)
        self.assertEqual(len(backend.graphs), 0)

        with YoloMode():
            with self.assertRaisesRegex(RuntimeError, "found no compiled frames"):
                torch.compile(torch.add, backend=backend, fullgraph=True)(x, x)

    def test_dispatch_key_set_preserved_across_compile(self):
        # compile_wrapper snapshots and restores the local dispatch key set
        # around every compiled call. Make sure cache-hit calls, and a call that
        # raises a user error, leave the include/exclude sets unchanged.
        def f(x):
            return x + 1

        cf = torch.compile(f, backend="eager")
        x = torch.randn(3)
        cf(x)  # warm up / compile

        include = torch._C._dispatch_tls_local_include_set()
        exclude = torch._C._dispatch_tls_local_exclude_set()
        for _ in range(5):
            cf(x)
        self.assertEqual(include, torch._C._dispatch_tls_local_include_set())
        self.assertEqual(exclude, torch._C._dispatch_tls_local_exclude_set())

        def raises(x):
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            torch.compile(raises, backend="eager")(x)
        self.assertEqual(include, torch._C._dispatch_tls_local_include_set())
        self.assertEqual(exclude, torch._C._dispatch_tls_local_exclude_set())

    def test_dispatch_key_set_stack_balanced_on_fullgraph_error(self):
        # compile_wrapper pushes the local dispatch key set onto a C++ stack and
        # must pop it even when the wrapper's own finally raises. The
        # fullgraph=True "found no compiled frames" error is raised from inside
        # that finally, so a naive restore-at-end-of-finally would be skipped and
        # leak a stack entry. Verify the save/restore stack stays balanced.
        from torch.utils._python_dispatch import TorchDispatchMode

        class YoloMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return torch.ops.aten.mul.Tensor(args[0], args[1])

        x = torch.ones(5)
        with YoloMode():
            for _ in range(3):
                with self.assertRaisesRegex(RuntimeError, "found no compiled frames"):
                    torch.compile(torch.add, backend="eager", fullgraph=True)(x, x)

        # A normal compiled call after the error loop must still work. This
        # guards against an over-restore / double-pop regression, which would
        # leave the stack empty and make the wrapper's own restore raise.
        cf = torch.compile(lambda t: t + 1, backend="eager")
        self.assertEqual(cf(x), x + 1)

        # If any push had leaked, the stack would be non-empty here. When
        # balanced it is empty, so a manual restore raises instead of silently
        # popping a stale entry.
        with self.assertRaisesRegex(RuntimeError, "empty stack"):
            torch._C._dynamo_restore_local_dispatch_key_set()

    def test_tracing_context_tls_defaults_present(self):
        # The _guards thread-local defaults tracing_context/compile_context to
        # None per thread, so the hot-path try_get() calls (once per compiled
        # call) hit a present attribute instead of a slow getattr-miss on a
        # thread that never ran compilation itself (e.g. a worker thread running
        # code compiled on another thread).
        import threading

        from torch._guards import _TLS, CompileContext, TracingContext

        result = {}

        def worker():
            result["tc_present"] = "tracing_context" in _TLS.__dict__
            result["cc_present"] = "compile_context" in _TLS.__dict__
            result["tc"] = TracingContext.try_get()
            result["cc"] = CompileContext.try_get()
            # Off-context (compile_context defaulted to None), get() still raises
            # -- the default makes this the "not set" AssertionError rather than
            # an AttributeError, on every thread regardless of history.
            try:
                CompileContext.get()
                result["get_raised"] = None
            except AssertionError:
                result["get_raised"] = "AssertionError"

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertTrue(result["tc_present"])
        self.assertTrue(result["cc_present"])
        self.assertIsNone(result["tc"])
        self.assertIsNone(result["cc"])
        self.assertEqual(result["get_raised"], "AssertionError")

        # Exercise the set -> restore cycle against the new storage: try_get()
        # sees the context inside tracing() and returns to the None default after.
        from torch._guards import tracing

        self.assertIsNone(TracingContext.try_get())
        tc = TracingContext(None)
        with tracing(tc):
            self.assertIs(TracingContext.try_get(), tc)
        self.assertIsNone(TracingContext.try_get())

    def test_compile_non_infra_empty_with_disalloed_dispatch_mode(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode(TorchDispatchMode):
            @classmethod
            def _should_skip_dynamo(cls):
                return False

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return torch.ops.aten.mul.Tensor(args[0], args[1])

        x = torch.ones(5)
        with YoloMode():
            out = torch.compile(torch.add, backend=backend, fullgraph=True)(x, x)

        self.assertEqual(len(backend.graphs), 1)

    def test_compile_non_infra_multiple(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        backend3 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode2(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(
                    lambda x, y: func(x, y), backend=backend3, fullgraph=True
                )(*args, **kwargs)
                return out

        class YoloMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                def random_fn(func, *args, **kwargs):
                    return func(*args, **kwargs)

                random_fn(func, *args, **kwargs)
                out = torch.compile(torch.add, backend=backend2, fullgraph=True)(
                    args[0], args[1]
                )
                return out

        x = torch.ones(5)
        with YoloMode(), YoloMode2():
            torch.compile(
                lambda x, y: torch.add(x, y), fullgraph=True, backend=backend
            )(x, x)

        self.assertEqual(len(backend2.graphs), 1)
        self.assertEqual(len(backend3.graphs), 0)
        self.assertEqual(len(backend.graphs), 0)

    def test_compile_non_infra_multiple_compile_internal(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        backend3 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode2(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(
                    lambda x, y: func(x, y), backend=backend3, fullgraph=True
                )(*args, **kwargs)
                return out

            @classmethod
            def ignore_compile_internals(cls):
                return True

        class YoloMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(torch.add, backend=backend2, fullgraph=True)(
                    args[0], args[1]
                )
                return out

            @classmethod
            def ignore_compile_internals(cls):
                return True

        x = torch.ones(5)
        with YoloMode(), YoloMode2():
            torch.compile(
                lambda x, y: torch.add(x, y), fullgraph=True, backend=backend
            )(x, x)

        self.assertEqual(len(backend2.graphs), 1)
        self.assertEqual(len(backend3.graphs), 1)
        self.assertEqual(len(backend.graphs), 1)

    def test_compile_non_infra_multiple_compile_internal_mixed(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        backend3 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode2(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(
                    lambda x, y: func(x, y), backend=backend3, fullgraph=True
                )(*args, **kwargs)
                return out

            @classmethod
            def ignore_compile_internals(cls):
                return True

        class YoloMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(torch.add, backend=backend2, fullgraph=True)(
                    args[0], args[1]
                )
                return out

        x = torch.ones(5)
        with YoloMode(), YoloMode2():
            torch.compile(
                lambda x, y: torch.add(x, y), fullgraph=True, backend=backend
            )(x, x)

        self.assertEqual(len(backend2.graphs), 1)
        self.assertEqual(len(backend3.graphs), 0)
        self.assertEqual(len(backend.graphs), 0)

        torch._dynamo.reset()

        backend3 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        with YoloMode2(), YoloMode():
            torch.compile(
                lambda x, y: torch.add(x, y), fullgraph=True, backend=backend
            )(x, x)

        self.assertEqual(len(backend2.graphs), 1)
        self.assertEqual(len(backend3.graphs), 1)
        self.assertEqual(len(backend.graphs), 0)

    @torch._dynamo.config.patch(inline_torch_dispatch_torch_compile=False)
    def test_compile_non_infra_disabled_config(self):
        """Test that setting inline_torch_dispatch_torch_compile=False reverts to old behavior."""
        from torch.utils._python_dispatch import TorchDispatchMode

        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        # When the config is False, torch.compile inside __torch_dispatch__ should
        # be skipped (old behavior) because we are inside a dispatch mode.
        class YoloMode(TorchDispatchMode):
            @classmethod
            def _should_skip_dynamo(cls):
                return False

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(func, backend=backend, fullgraph=False)(
                    *args, **kwargs
                )
                return out

        x = torch.randn(5)
        with YoloMode():
            out = torch.add(x, x)

        # With the config disabled, compile should be skipped, so 0 graphs captured
        self.assertEqual(len(backend.graphs), 0)

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

    def test_bool_literal_iand_unspecialized_float(self):
        @torch.compile(backend="eager")
        def mul(a: torch.Tensor, b: torch.Tensor, c: float | int | None):
            has_bias = c is not None
            has_bias &= c != 0.0

            if has_bias:
                return a * b + c
            return a * b

        a = torch.randn(10)
        b = torch.randn(10)

        self.assertEqual(mul(a, b, c=0.0), a * b)
        self.assertEqual(mul(a, b, c=0.5), a * b + 0.5)

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
            if torch.Tag.pt2_compliant_tag not in torch.ops.mylib.bar.default.tags:
                raise AssertionError("Expected pt2_compliant_tag in bar.default.tags")

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
            if torch.Tag.pt2_compliant_tag in torch.ops.mylib.bar2.default.tags:
                raise AssertionError(
                    "Expected pt2_compliant_tag not in bar2.default.tags"
                )

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

    def test_input_tensor_custom_attr_mutation(self):
        def f(x, flag):
            x.offloading_activation = flag
            return x + 1

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.ones(5)

        res = opt_f(x, True)
        self.assertEqual(res, torch.ones(5) + 1)
        self.assertTrue(x.offloading_activation)

    def test_intermediate_tensor_custom_attr_mutation(self):
        def f(x, flag):
            y = x + 1
            y.offloading_activation = flag
            return y

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.ones(5)

        res = opt_f(x, True)
        self.assertEqual(res, torch.ones(5) + 1)
        self.assertTrue(res.offloading_activation)

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
        "requires Hopper+ (SM >= 9.0) for TMA",
    )
    @unittest.skipIf(
        not torch.utils._triton.has_triton()
        or not hasattr(__import__("triton"), "set_allocator"),
        "requires triton with set_allocator support",
    )
    def test_triton_set_allocator_no_graph_break(self):
        """set_allocator inside torch.compile does not graph break and
        replays correctly at runtime (including cache hits)."""
        import triton
        import triton.language as tl
        from triton.runtime._allocation import NullAllocator

        @triton.jit
        def tma_copy_kernel(
            x_ptr,
            out_ptr,
            M,
            N,
            stride_m,
            stride_n,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            pid = tl.program_id(0)
            desc = tl.make_tensor_descriptor(
                x_ptr,
                shape=[M, N],
                strides=[stride_m, stride_n],
                block_shape=[BLOCK_M, BLOCK_N],
            )
            block = tl.load_tensor_descriptor(desc, [pid * BLOCK_M, 0])
            out_desc = tl.make_tensor_descriptor(
                out_ptr,
                shape=[M, N],
                strides=[stride_m, stride_n],
                block_shape=[BLOCK_M, BLOCK_N],
            )
            tl.store_tensor_descriptor(out_desc, [pid * BLOCK_M, 0], block)

        M, N, BLOCK_M, BLOCK_N = 128, 64, 64, 64

        def run_kernel(x):
            out = torch.empty_like(x)
            tma_copy_kernel[(M // BLOCK_M,)](
                x,
                out,
                M,
                N,
                x.stride(0),
                x.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            return out

        x = torch.randn(M, N, device="cuda")

        from contextlib import contextmanager

        from triton.runtime._allocation import _allocator

        @contextmanager
        def triton_allocator(allocator):
            prev = _allocator.get()
            triton.set_allocator(allocator)
            try:
                yield
            finally:
                triton.set_allocator(prev)

        def fn_with_set_allocator(x):
            triton.set_allocator(
                lambda size, alignment, stream: torch.empty(
                    size, device="cuda", dtype=torch.int8
                )
            )
            return run_kernel(x)

        opt_fn = torch.compile(
            fn_with_set_allocator, backend="aot_eager", fullgraph=True
        )

        # set_allocator inside compiled region does NOT graph break
        with triton_allocator(NullAllocator()):
            out = opt_fn(x)
            self.assertEqual(out, x)

            # Verify set_allocator replays on cache hit (not just tracing)
            triton.set_allocator(NullAllocator())
            out2 = opt_fn(x)
            self.assertEqual(out2, x)

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
                "mylib::foo_generate_trivial_abstract_impl",
                "(Tensor x, Tensor[] y, Tensor(a!)? z, SymInt w) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl(
                "mylib::foo_generate_trivial_abstract_impl", "cpu", lib=lib
            )
            @torch._dynamo.disable
            def foo_impl(x, y, z, w):
                x + y[0] + w
                return

            def f(x, y, z, w):
                return torch.ops.mylib.foo_generate_trivial_abstract_impl(x, y, z, 2)

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

    def test_builtin_eval_constant_expr(self):
        def fn(x):
            values = eval("{'scale': 2, 'offset': (3, 4)}")
            return x * values["scale"] + eval(" 1+1\n") + eval("1/2")

        x = torch.randn(4)
        cnt = CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_builtin_eval_rejects_non_constant_expr(self):
        def fn(x):
            return x + eval("len([1])")

        with self.assertRaisesRegex(Unsupported, "Failed to trace builtin operator"):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(1))

    def test_builtin_eval_propagates_syntax_error(self):
        def fn(x):
            try:
                eval("1+")
            except SyntaxError:
                return x + 1
            return x - 1

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

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
            assert mod_a is None  # noqa: S101
            assert "bbbbbbbb" not in sys.modules  # noqa: S101

            assert "operator" in sys.modules  # noqa: S101
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
            guard_code,
        )
        guard_code_str = "\n".join(guard_code)

        # Make sure that the dict_contains are present in the order of added
        self.assertExpectedInline(
            guard_code_str,
            """\
L['x'].size()[1] == L['x'].size()[0]
L['x'].storage_offset() == 0
2 <= L['x'].size()[0]
utils_device.CURRENT_DEVICE == None
str(L['x'].dtype) == 'torch.float32'
str(L['x'].device) == 'cpu'
L['x'].requires_grad == False
L['x'].ndimension() == 2
hasattr(L['x'], '_dynamo_dynamic_indices') == False
hasattr(L['x'], '_dynamo_weak_dynamic_indices') == False
hasattr(L['x'], '_dynamo_unbacked_indices') == False
hasattr(L['x'], '_dynamo_strict_unbacked_indices') == False
hasattr(L['x'], '_dynamo_static_indices') == False
L['x'] is L['y']
not ___dict_contains('aaaaaaaa', G['sys'].modules)
not ___dict_contains('bbbbbbbb', G['sys'].modules)
___dict_contains('operator', G['sys'].modules)
not ___dict_contains('cccccccc', G['sys'].modules)""",
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
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            if hasattr(x, "test"):
                return x + 2
            else:
                return x + 1

        self.assertEqual(torch.ones(2, 2) + 1, fn(torch.ones(2, 2)))

        inp = torch.ones(2, 2)
        inp.test = None
        self.assertEqual(torch.ones(2, 2) + 2, fn(inp))

    def test_tensor_call_obj_hasattr_view(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            output3 = getattr(x, "view", None)(10)
            return output3

        x = torch.randn(10)
        self.assertEqual(x.view(10), fn(x))

    def test_mro_type_tensor_no_source(self):
        @torch.compile(fullgraph=True, backend="eager")
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

        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            return t.add_one(t) + x

        result = fn(torch.ones(1))
        self.assertEqual(torch.ones(1) + 2, result)

    def test_known_tensor_methods_traced(self):
        # Verify that known tensor methods (in all_tensor_attrs) are still
        # traced into the graph via the generic proxy path.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return x.abs().cos()

        result = fn(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_tensor_subclass_method_traced(self):
        # Methods defined on the actual tensor class (including dynamically
        # added ones) should be proxied through the generic call_method path,
        # not graph-broken.  This validates that the guard uses the concrete
        # class_type rather than the static all_tensor_attrs dict.
        def _dynamo_test_method(self):
            return self + 1

        with unittest.mock.patch.object(
            torch.Tensor, "_dynamo_test_method", _dynamo_test_method, create=True
        ):
            cnt = CompileCounterWithBackend("eager")

            @torch.compile(backend=cnt)
            def fn(x):
                y = x._dynamo_test_method()
                return y + 1

            result = fn(torch.randn(4))
            self.assertEqual(cnt.frame_count, 1)
            # Verify _dynamo_test_method appears as a call_method in the FX graph
            call_method_targets = [
                n.target for n in cnt.graphs[0].graph.nodes if n.op == "call_method"
            ]
            self.assertIn("_dynamo_test_method", call_method_targets)

    def test_unknown_tensor_method_graph_break(self):
        # Truly unknown methods raise AttributeError during tracing at
        # LOAD_ATTR time (dynamic_getattr), ensuring dynamo does not
        # silently proxy them into the compiled graph.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            y = x._nonexistent_test_method_xyz()
            return y + 1

        with self.assertRaises(AttributeError):
            fn(torch.randn(4))

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

    def test_tensor_share_memory(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 64
                self.num_layers = 2

            def forward(self, x):
                batch_size = x.size(0)
                h = torch.zeros(
                    self.num_layers, batch_size, self.hidden_size
                ).share_memory_()
                c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
                return x + h.sum() + c.sum()

        model = Model()
        x = torch.randn(4, 10)
        expected = model(x)
        compiled_model = torch.compile(model, fullgraph=False, backend="eager")
        actual = compiled_model(x)
        self.assertEqual(expected, actual)

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

    def test_min_max_tuple_const(self):
        # min/max on tuple constants should use ConstantVariable.create
        # to properly handle the tuple return type
        def fn():
            a = min((1, 2), (3, 4))
            b = max((1, 2), (3, 4))
            return a, b

        opt_fn = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        result = opt_fn()
        self.assertEqual(result, ((1, 2), (3, 4)))

    def test_min_max_key_and_default(self):
        # Items are non-constant (tensors), so the fully-constant fold path is
        # skipped; the key maps each to a compile-time-constant (ndim), which is
        # what the key comparison requires. Comparison is on key(item) but the
        # original tensor is returned.
        def fn(x, y, z):
            tensors = [x, y, z]
            return (
                max(tensors, key=lambda t: t.ndim).sum(),
                min(tensors, key=lambda t: t.ndim).sum(),
                max(tensors, key=lambda t: t.ndim, default=x).sum(),
                max([], default=y).sum(),
                min([x + 1], default=z).sum(),
                max(x, z, key=lambda t: t.ndim).sum(),
                min(x, z, key=lambda t: t.ndim).sum(),
            )

        cnt = CompileCounter()
        opt = torch.compile(fn, backend=cnt, fullgraph=True)
        x, y, z = torch.randn(2), torch.randn(2, 2), torch.randn(2, 2, 2)
        self.assertEqual(opt(x, y, z), fn(x, y, z))
        self.assertEqual(cnt.frame_count, 1)

    def test_min_max_key_none(self):
        # key=None behaves like no key argument at all.
        def fn(x):
            return x + max([4, 1, 7], key=None) + min([4, 1, 7], key=None)

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(3)
        self.assertEqual(opt(x), fn(x))

    def test_min_max_default_none_sentinel(self):
        # A real default=None must be honored for empty iterables and not be
        # confused with the internal "omitted" sentinel (which raises).
        def fn(x):
            return (
                max((), default=None),
                min((), default=None),
                max((), default=0),
                x + 1,
            )

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(3)
        self.assertEqual(opt(x), fn(x))
        self.assertEqual(opt(x)[:3], (None, None, 0))

    def test_min_max_default_varargs_type_error(self):
        def fn(a, b):
            return max(a, b, default=0)

        opt = torch.compile(fn, backend="eager")
        with self.assertRaises(TypeError):
            fn(1, 2)
        with self.assertRaises(TypeError):
            opt(1, 2)

    def test_min_max_empty_no_default_value_error(self):
        def fn(xs):
            return max(xs)

        opt = torch.compile(fn, backend="eager")
        with self.assertRaises(ValueError):
            fn([])
        with self.assertRaises(ValueError):
            opt([])

    def test_int_base_indexable(self):
        # int(x, base) resolves base via __index__ (PyNumber_AsSsize_t), so a
        # non-constant __index__-able base must be accepted. Previously this
        # raised an "invalid call to builtin op handler" graph break because
        # call_int ignored the base argument entirely.
        class MyIndexable:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        def fn(x):
            a = int("101", base=MyIndexable(2))
            b = int("101", MyIndexable(36))
            c = int(b"ff", base=MyIndexable(16))
            return x + a + b + c

        cnt = CompileCounter()
        opt = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.randn(3)
        self.assertEqual(opt(x), fn(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_int_base_out_of_range_value_error(self):
        class MyIndexable:
            def __index__(self):
                return 37

        def fn(xs):
            return int("43", base=MyIndexable())

        opt = torch.compile(fn, backend="eager")
        with self.assertRaises(ValueError):
            fn([])
        with self.assertRaises(ValueError):
            opt([])

    def test_int_base_non_string_type_error(self):
        def fn(n):
            return int(n, base=16)

        opt = torch.compile(fn, backend="eager")
        with self.assertRaises(TypeError):
            fn(5)
        with self.assertRaises(TypeError):
            opt(5)

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
        @torch.compile(fullgraph=True, backend="eager")
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
                assert False  # noqa: B011, S101
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

    def test_check_compiles_when_predicate_true_and_message_string(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, "Shape is not greater than 3")
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_raises_at_runtime_when_predicate_false_and_message_string(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, "Shape is not greater than 3")
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, "Shape is not greater than 3"):
            f(x)

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

    def test_check_compiles_when_predicate_true_and_message_module_level_fn(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, global_check_message)
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_raises_at_runtime_when_predicate_false_and_message_module_level_fn(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, global_check_message)
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, "global check message"):
            f(x)

    def test_check_compiles_when_predicate_true_and_message_non_str_constant(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, 12345)
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_raises_at_runtime_when_predicate_false_and_message_non_str_constant(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, 12345)
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, "12345"):
            f(x)

    def test_check_graph_breaks_when_message_is_unsupported_type(self):
        # A tensor message is neither a constant nor a function.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, x)
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Can't extract message from torch._check"
        ):
            f(x)

    def test_check_graph_breaks_when_message_is_bound_method(self):
        # A bound method is a function-VT subclass whose reconstructed function
        # drops the receiver, so it is rejected rather than mis-evaluated.
        class M:
            def msg(self):
                return "bound method message"

            def forward(self, x):
                torch._check(x.shape[0] > 3, self.msg)
                return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Can't extract message from torch._check"
        ):
            torch.compile(M().forward, backend="eager", fullgraph=True)(x)

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
            torch._dynamo.exc.Unsupported, "Can't .* torch._check"
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
            torch._dynamo.exc.Unsupported, "Can't .* torch._check"
        ):
            f(x)

    def test_check_with_constant_true(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check_with(ValueError, x.shape[0] > 3, lambda: "bad shape")
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

    def test_check_with_constant_false_raises(self):
        @torch.compile(backend="eager")
        def f(x):
            torch._check_with(ValueError, x.shape[0] > 3, lambda: "bad shape")
            return x + 1

        with self.assertRaises((ValueError, RuntimeError)):
            f(torch.randn(3))

    def test_check_with_try_except(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            try:
                torch._check_with(ValueError, False, lambda: "expected")
            except ValueError:
                pass
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

    def test_check_value(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check_value(x.shape[0] > 3, lambda: "bad value")
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

        @torch.compile(backend="eager")
        def f_fail(x):
            torch._check_value(x.shape[0] > 3, lambda: "bad value")
            return x + 1

        # RuntimeError because graph lowers to torch._check
        with self.assertRaises((ValueError, RuntimeError)):
            f_fail(torch.randn(3))

    def test_check_index(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check_index(x.shape[0] > 3, lambda: "bad index")
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

        @torch.compile(backend="eager")
        def f_fail(x):
            torch._check_index(x.shape[0] > 3, lambda: "bad index")
            return x + 1

        with self.assertRaises((IndexError, RuntimeError)):
            f_fail(torch.randn(3))

    def test_check_type(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check_type(x.shape[0] > 3, lambda: "bad type")
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

        @torch.compile(backend="eager")
        def f_fail(x):
            torch._check_type(x.shape[0] > 3, lambda: "bad type")
            return x + 1

        with self.assertRaises((TypeError, RuntimeError)):
            f_fail(torch.randn(3))

    def test_check_not_implemented(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check_not_implemented(x.shape[0] > 3, lambda: "not impl")
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

        @torch.compile(backend="eager")
        def f_fail(x):
            torch._check_not_implemented(x.shape[0] > 3, lambda: "not impl")
            return x + 1

        with self.assertRaises((NotImplementedError, RuntimeError)):
            f_fail(torch.randn(3))

    def test_check_tensor_all(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check_tensor_all(x > 0, lambda: "all must be positive")
            return x + 1

        x = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(f(x), x + 1)

    def test_check_tensor_all_with(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check_tensor_all_with(
                ValueError, x > 0, lambda: "all must be positive"
            )
            return x + 1

        x = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(f(x), x + 1)

    def test_check_tensor_predicate_raises_clear_error(self):
        @torch.compile(backend="eager")
        def f(x):
            torch._check(x > 0)
            return torch.log(x)

        with self.assertRaisesRegex(
            TypeError, r"cond must be a bool.*torch[.]_check_tensor_all"
        ) as cm:
            f(torch.rand(1))

        self.assertNotIn("FakeTensor", str(cm.exception))

    def test_check_with_closure_constant(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            msg = "shape check failed"
            torch._check_with(ValueError, x.shape[0] > 3, lambda: msg)
            return x + 1

        x = torch.randn(4)
        self.assertEqual(f(x), x + 1)

    def test_assert(self):
        @torch.compile(backend="eager")
        def fn1(x):
            assert x.shape != x.shape  # noqa: S101

        with self.assertRaises(AssertionError):
            a = torch.randn(10)
            fn1(a)

        def fn2(x):
            assert x.shape == x.shape  # noqa: S101
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

    def test_namedtuple___eq__(self):
        class MyNamedTuple(typing.NamedTuple):
            a: int
            b: int

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            t1 = MyNamedTuple(a=1, b=2)
            t2 = (1, 2)
            return x.sin(), (t1 == t2)

        x = torch.randn(2)
        res = f(x)
        self.assertTrue(res[1])

    def test_structseq1(self):
        def fn(x, y):
            return torch.return_types.max((x, y))

        x = torch.randn(3, 2)
        y = torch.randn(2, 4)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    def test_structseq2(self):
        def fn(x, y):
            return tuple(torch.return_types.linalg_qr((2 * x, y - 1)))

        x = torch.randn(3, 2)
        y = torch.randn(2, 4)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    def test_structseq_repr(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            result = torch.max(x, dim=0)
            s = repr(result)
            return result.values

        x = torch.randn(3, 2)

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, fullgraph=True, backend="eager")(x)

        opt_fn = torch._dynamo.optimize(cnts)(fn)
        result = opt_fn(x)
        self.assertEqual(cnts.frame_count, 1)

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
        @torch.compile(backend="eager")
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
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")

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
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")

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
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")

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

    def test_member_descriptor_set_delete_slot(self):
        # member_descriptor (a __slots__ field) exposes __set__/__delete__ via
        # the shared tp_descr_set slot; explicit calls should trace.
        class Slotted:
            __slots__ = ("a",)

        def set_fn(x):
            o = Slotted()
            type(o).a.__set__(o, 9)
            return x + o.a

        def del_fn(x):
            o = Slotted()
            o.a = 4
            type(o).a.__delete__(o)
            return x + (0 if hasattr(o, "a") else 100)

        for fn in (set_fn, del_fn):
            opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
            self.assertEqual(opt_fn(torch.zeros(2)), fn(torch.zeros(2)))

    def test_tuplegetter_readonly_slot(self):
        # namedtuple field accessors are read-only; __set__/__delete__ raise
        # AttributeError through the tp_descr_set slot.
        P = collections.namedtuple("P", ["x", "y"])

        def fn(t):
            p = P(1, 2)
            try:
                type(p).x.__set__(p, 5)
                return t + 1
            except AttributeError as e:
                return t, str(e)

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        self.assertEqual(opt_fn(torch.zeros(2)), fn(torch.zeros(2)))

    def test_dict_with_descriptor(self):
        class MyDescriptor:
            def __get__(self, obj, objtype=None):
                if obj is None:
                    return self
                return obj.__dict__.get("_value", 0)

            def __set__(self, obj, value):
                obj.__dict__["_value"] = value

        class MyClass:
            prop = MyDescriptor()

            def __init__(self):
                self.prop = 42

        def fn(obj):
            obj.__dict__["extra"] = 99
            return obj.prop + obj.__dict__["extra"]

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = MyClass()
        res = fn(obj)

        obj2 = MyClass()
        out = opt_fn(obj2)
        self.assertEqual(res, out)
        self.assertEqual(obj2.__dict__["_value"], 42)
        self.assertEqual(obj2.__dict__["extra"], 99)

    def test_dict_with_slots(self):
        class SlottedClass:
            __slots__ = ("x",)

            def __init__(self, x):
                self.x = x

        def fn(obj):
            # SlottedClass doesn't have __dict__, so this should fail or be handled
            return obj.x * 2

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = SlottedClass(5)
        res = fn(obj)
        out = opt_fn(obj)
        self.assertEqual(res, out)

    def test_dict_with_slots_and_dict(self):
        class SlottedWithDict:
            __slots__ = ("x", "__dict__")

            def __init__(self, x):
                self.x = x

        def fn(obj):
            obj.__dict__["custom"] = 100
            return obj.x + obj.__dict__["custom"]

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = SlottedWithDict(7)
        res = fn(obj)

        obj2 = SlottedWithDict(7)
        out = opt_fn(obj2)
        self.assertEqual(res, out)
        self.assertEqual(obj2.__dict__["custom"], 100)

    def test_dict_descriptor_interaction(self):
        class DescriptorWithDict:
            def __get__(self, obj, objtype=None):
                if obj is None:
                    return self
                return obj.__dict__.get("_internal", "default")

            def __set__(self, obj, value):
                obj.__dict__["_internal"] = f"desc_{value}"

        class MyClass:
            desc = DescriptorWithDict()

            def __init__(self):
                pass

        def fn(obj):
            obj.desc = "hello"
            obj.__dict__["_internal"] = "direct"
            return obj.desc

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = MyClass()
        ref = fn(obj)

        obj2 = MyClass()
        out = opt_fn(obj2)
        self.assertEqual(ref, out)
        self.assertEqual(out, "direct")

    def test_mutable_mapping_dict_update(self):
        """Test that MutableMappingVariable handles __dict__ updates correctly."""
        from collections import OrderedDict

        class CustomMapping(OrderedDict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(mapping):
            mapping.__dict__["custom_attr"] = 42
            return mapping.__dict__["custom_attr"]

        mapping = CustomMapping()
        out = fn(mapping)
        self.assertEqual(out, 42)
        self.assertIn("custom_attr", mapping.__dict__)
        self.assertEqual(mapping.__dict__["custom_attr"], 42)

    def test_mutable_mapping_dict_access_pattern(self):
        """Test accessing attributes through __dict__ on MutableMapping subclasses."""
        from collections import OrderedDict

        class TrackedDict(OrderedDict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(d):
            # Pattern: check if attribute exists in __dict__
            if "tracker" not in d.__dict__:
                d.__dict__["tracker"] = []
            d.__dict__["tracker"].append(1)
            return len(d.__dict__["tracker"])

        d = TrackedDict()
        out = fn(d)
        self.assertEqual(out, 1)
        self.assertIn("tracker", d.__dict__)
        self.assertEqual(d.__dict__["tracker"], [1])

    def test_mutable_mapping_lazy_dict_initialization(self):
        """Test lazy initialization pattern with MutableMapping __dict__."""
        from collections import defaultdict

        class LazyMapping(dict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(mapping):
            # Lazy initialization in __dict__
            if not hasattr(mapping, "_cache"):
                mapping.__dict__["_cache"] = {}
            mapping.__dict__["_cache"]["key"] = "value"
            return mapping.__dict__["_cache"]["key"]

        m = LazyMapping()
        out = fn(m)
        self.assertEqual(out, "value")
        self.assertIn("_cache", m.__dict__)
        self.assertEqual(m.__dict__["_cache"]["key"], "value")

    def test_mutable_mapping_dict_with_property_setter(self):
        """Test MutableMapping with property setters that access __dict__."""
        from collections import OrderedDict

        class PropertyMapping(OrderedDict):
            def __init__(self):
                super().__init__()
                self.__dict__["_value"] = 0

            @property
            def value(self):
                return self.__dict__.get("_value", 0)

            @value.setter
            def value(self, v):
                self.__dict__["_value"] = v

        @torch.compile(fullgraph=True, backend="eager")
        def fn(m):
            m.value = 100
            return m.value

        m = PropertyMapping()
        out = fn(m)
        self.assertEqual(out, 100)
        self.assertIn("_value", m.__dict__)
        self.assertEqual(m.__dict__["_value"], 100)

    def test_mutable_mapping_dict_multiple_accesses(self):
        """Test multiple accesses and mutations to MutableMapping __dict__."""
        from collections import OrderedDict

        class MultiAccessMapping(OrderedDict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(mapping):
            # Multiple accesses to __dict__
            mapping.__dict__["a"] = 1
            mapping.__dict__["b"] = mapping.__dict__["a"] + 1
            mapping.__dict__["c"] = mapping.__dict__["b"] + 1
            return mapping.__dict__["a"] + mapping.__dict__["b"] + mapping.__dict__["c"]

        m = MultiAccessMapping()
        out = fn(m)
        self.assertEqual(out, 6)  # 1 + 2 + 3
        self.assertIn("a", m.__dict__)
        self.assertIn("b", m.__dict__)
        self.assertIn("c", m.__dict__)
        self.assertEqual(m.__dict__["a"], 1)
        self.assertEqual(m.__dict__["b"], 2)
        self.assertEqual(m.__dict__["c"], 3)

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

    def test_nn_module_getattribute_simple_delegation(self):
        # Test that nn.Module with __getattribute__ that overrides a
        # single attribute name compiles without graph break.
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = 3.0

            def __getattribute__(self, name):
                if name == "my_scale":
                    return super().__getattribute__("scale")
                return super().__getattribute__(name)

            def forward(self, x):
                return x * self.my_scale

        mod = MyMod()
        x = torch.randn(2, 4)
        expected = mod(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(mod, backend=cnts)
        result = opt_fn(x)
        self.assertTrue(same(result, expected))

    def test_nn_module_getattribute_graph_break(self):
        # __getattribute__ that Dynamo cannot trace produces correct results
        # via eager fallback instead of crashing.
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def __getattribute__(self, name):
                if name == "my_attr":
                    return eval("len([1])")
                return super().__getattribute__(name)

            def forward(self, x):
                a = self.my_attr
                return self.linear(x) + a

        mod = MyMod()
        x = torch.randn(2, 4)
        expected = mod(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(mod, backend=cnts)
        result = opt_fn(x)
        self.assertTrue(same(result, expected))

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
        @torch.compile(fullgraph=True, backend="eager")
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
            assert len(class_fields)  # noqa: S101
            assert all(field.default is None for field in class_fields[1:])  # noqa: S101
            other_fields_are_none = all(
                getattr(obj, field.name) is None for field in class_fields[1:]
            )
            assert not other_fields_are_none  # noqa: S101

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
            self.assertEqual(
                result,
                opt_fn(op, t1, t2),
                msg=lambda msg: f"{msg}\n{op=} {t1_np=} {t2_np=}",
            )
            self.assertEqual(
                cnts.frame_count, 1, msg=lambda msg: f"{msg}\n{op=} {t1_np=} {t2_np=}"
            )
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
        @torch.compile(fullgraph=True, backend="eager")
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

        cnt = CompileCounterWithBackend("eager")
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

        @torch.compile(fullgraph=True, backend="eager")
        def fn(y):
            return y + x + x

        r = fn(np.arange(10))
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(r, x * 3)
        del x

    def test_numpy_gt(self):
        x = np.arange(10)

        @torch.compile(backend="eager")
        def fn(y):
            return y >= 3

        r = fn(x)
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(r, x >= 3)

    def test_numpy_min(self):
        x = np.arange(10)

        @torch.compile(backend="eager")
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
        if r2 not in ("L", "U"):
            raise AssertionError(f"Expected r2 to be 'L' or 'U', got {r2}")

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
        @torch.compile(backend="eager")
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

        @torch.compile(fullgraph=True, backend="eager")
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

    def test_deepcopy_dict(self):
        MY_DICT = {"a": 1, "b": 2.0, "c": None}

        def fn(x):
            d = copy.deepcopy(MY_DICT)
            d["b"] = 3.0
            return x + d["b"]

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x)
        self.assertEqual(result, correct)

    def test_deepcopy_nested_dict(self):
        NESTED = {"a": {"b": 1.0}, "c": [2.0, 3.0]}

        def fn(x):
            d = copy.deepcopy(NESTED)
            return x + d["a"]["b"] + d["c"][0]

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x)
        self.assertEqual(result, correct)

    def test_deepcopy_list(self):
        MY_LIST = [1.0, 2.0, 3.0]

        def fn(x):
            lst = copy.deepcopy(MY_LIST)
            lst[0] = 5.0
            return x + lst[0]

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x)
        self.assertEqual(result, correct)

    def test_deepcopy_user_defined_object(self):
        class MyConfig:
            def __init__(self, hidden_size=64):
                self.hidden_size = hidden_size

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MyConfig()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, x):
                cfg = copy.deepcopy(self.config)
                return self.linear(x) * cfg.hidden_size

        m = MyModule()
        x = torch.randn(2, 64)
        result = torch.compile(m, fullgraph=True, backend="eager")(x)
        self.assertEqual(m(x), result)

    def test_deepcopy_user_defined_object_with_containers(self):
        class Config:
            def __init__(self):
                self.sizes = [1, 2, 3]
                self.mapping = {"a": 10, "b": 20}
                self.flags = (True, False)

        def fn(x, cfg):
            c = copy.deepcopy(cfg)
            c.sizes[0] = 99
            c.mapping["a"] = 77
            return x + c.sizes[0] + c.mapping["a"]

        cfg = Config()
        x = torch.randn(4)
        correct = fn(x, cfg)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x, cfg)
        self.assertEqual(result, correct)
        # Verify deepcopy didn't mutate original
        self.assertEqual(cfg.sizes[0], 1)
        self.assertEqual(cfg.mapping["a"], 10)

    def test_copy_namedtuple_and_user_object(self):
        # namedtuple is a tuple subclass with __slots__ = () and no __dict__; it
        # must reduce via __getnewargs__ rather than obj.__dict__. Also exercise
        # a plain user object (has __dict__) to guard the common path.
        from collections import namedtuple

        NT = namedtuple("NT", "x y z")

        class Plain:
            def __init__(self):
                self.a = 1
                self.b = [10, 20]

        def fn(o):
            return copy.copy(o), copy.deepcopy(o)

        cfn = torch.compile(fn, fullgraph=True, backend="eager")

        nt = NT(10, 20, 30)
        c, d = cfn(nt)
        self.assertEqual(c, nt)
        self.assertEqual(d, nt)
        self.assertIs(type(c), NT)
        self.assertIs(type(d), NT)
        self.assertEqual(c._fields, nt._fields)

        p = Plain()
        c, d = cfn(p)
        self.assertEqual(c.a, 1)
        self.assertEqual(c.b, [10, 20])
        self.assertIs(c.b, p.b)  # shallow copy shares the list
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, [10, 20])
        self.assertIsNot(d.b, p.b)  # deep copy clones the list
        self.assertIs(type(d), Plain)

    def test_deepcopy_set(self):
        MY_SET = {1, 2, 3}

        def fn(x):
            s = copy.deepcopy(MY_SET)
            return x + len(s)

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x)
        self.assertEqual(result, correct)

    def test_deepcopy_frozenset(self):
        MY_FROZENSET = frozenset([1, 2, 3])

        def fn(x):
            s = copy.deepcopy(MY_FROZENSET)
            return x + len(s)

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x)
        self.assertEqual(result, correct)

    def test_deepcopy_user_defined_object_with_method(self):
        class MyConfig:
            def __init__(self, hidden_size=64):
                self.hidden_size = hidden_size

            def get_size(self):
                return self.hidden_size

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MyConfig()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, x):
                cfg = copy.deepcopy(self.config)
                return self.linear(x) * cfg.get_size()

        m = MyModule()
        x = torch.randn(2, 64)
        correct = m(x)
        result = torch.compile(m, fullgraph=True, backend="eager")(x)
        self.assertEqual(result, correct)

    def test_deepcopy_nested_user_defined_object(self):
        class Inner:
            def __init__(self, scale):
                self.scale = scale

        class Outer:
            def __init__(self):
                self.inner = Inner(2.0)
                self.bias = 1.0

        def fn(x, cfg):
            c = copy.deepcopy(cfg)
            c.inner.scale = 3.0
            return x * c.inner.scale + c.bias

        cfg = Outer()
        x = torch.randn(4)
        correct = fn(x, cfg)
        result = torch.compile(fn, fullgraph=True, backend="eager")(x, cfg)
        self.assertEqual(result, correct)
        # Verify deepcopy didn't mutate original
        self.assertEqual(cfg.inner.scale, 2.0)

    def test_deepcopy_with_getattribute_override(self):
        # Regression test: classes that override __getattribute__ (like
        # HuggingFace PretrainedConfig) caused a graph break on
        # __reduce_ex__ because SuperVariable.call_method for
        # object.__getattribute__ bypassed the polyfill detection in
        # resolve_type_attr.
        class Config:
            attribute_map = {}

            def __init__(self, hidden_size=768, num_layers=6):
                self.hidden_size = hidden_size
                self.num_layers = num_layers

            def __getattribute__(self, key):
                if key != "attribute_map" and key in super().__getattribute__(
                    "attribute_map"
                ):
                    key = super().__getattribute__("attribute_map")[key]
                return super().__getattribute__(key)

        def fn(x, config):
            c = copy.deepcopy(config)
            return x * c.hidden_size + c.num_layers

        x = torch.randn(3)
        config = Config()
        correct = fn(x, config)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x, config)
        self.assertEqual(result, correct)

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

        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, l):
            l.x = x
            return x + 1

        x = torch.ones(2, 2)
        fn(x, loc)

        self.assertTrue(loc.x is x)

    def test_setattr_wrong_args_raises(self):
        # setattr() with wrong arg count should raise TypeError during tracing,
        # not silently graph-break (a graph break would just defer the same
        # failure to eager execution).
        @torch.compile(backend="eager")
        def fn(x):
            setattr(x, "foo")
            return x

        with self.assertRaises(TypeError):
            fn(torch.randn(4))

    def test_user_defined_class_name(self):
        class MyClassFoo:
            pass

        def fn1(a, b, c):
            tmp = MyClassFoo()
            if tmp.__class__.__name__ == "MyClassFoo":
                return a - b / c

        torch._dynamo.testing.standard_test(self, fn=fn1, nargs=3)

    def test_class_reassignment_graph_break(self):
        class BaseClass:
            def __init__(self, x):
                self.x = x

        class DerivedClass(BaseClass):
            def __init__(self, x):
                super().__init__(x)
                self.y = x * 2

        def fn(x):
            obj = BaseClass(5)
            obj.__class__ = DerivedClass
            is_derived = isinstance(obj, DerivedClass)
            has_y = hasattr(obj, "y")
            return x + 1, is_derived, has_y

        x = torch.ones(1)
        eager_result = fn(x)
        compiled_result = torch.compile(fn, backend="eager")(x)
        self.assertEqual(eager_result, compiled_result)

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
                assert len(args) == 1  # noqa: S101
                assert len(kwargs) == 1  # noqa: S101
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

        @torch.compile(backend="eager")
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

        @torch.compile(fullgraph=True, backend="eager")
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

            assert not isinstance(x, torch.fx.Proxy)  # noqa: S101
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

    def test_torch_size_tensor_index_scalar_constant(self):
        def fn(x):
            idx = torch.tensor(1)
            dim_size = x.shape[idx]
            return x.reshape(-1, dim_size)

        x = torch.randn(2, 3, 4)
        ref = fn(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)

    def test_torch_size_tensor_index_non_scalar(self):
        def fn(x):
            idx = torch.tensor([1, 1])
            try:
                dim_size = x.shape[idx]
                return x * dim_size
            except TypeError:
                return x.sum()

        x = torch.randn(2, 3, 4)
        ref = fn(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))

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

    def test_torch_generator_manual_seed(self):
        from torch._dynamo.utils import counters

        cnts = torch._dynamo.testing.CompileCounter()
        counters.clear()

        def fn(x, gen):
            gen.manual_seed(3)
            return x + 1

        x = torch.randn(10)
        ref = fn(x, torch.Generator())

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        res = opt_fn(x, torch.Generator())

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.op_count, 1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_torch_generator_initial_seed(self):
        from torch._dynamo.utils import counters

        cnts = torch._dynamo.testing.CompileCounter()
        counters.clear()

        def fn(x):
            return x + 1, torch.default_generator.initial_seed()

        x = torch.randn(10)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.op_count, 1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_torch_generator_get_state_fullgraph(self):
        def fn():
            return torch.default_generator.get_state()

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)()

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

    @torch._dynamo.config.patch(graph_break_on_nn_param_ctor=False)
    def test_param_grad_in_forward(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            w = torch.nn.Parameter(torch.ones(4, 4))
            saw_none = w.grad is None
            if saw_none:
                w.grad = torch.zeros_like(w)
            return saw_none, w.grad + x

        x = torch.randn(4, 4)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)

        self.assertTrue(res[0])
        self.assertEqual(ref[1], res[1])
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(
        graph_break_on_nn_param_ctor=False, trace_autograd_ops=True
    )
    def test_param_autograd_grad_in_forward(self):
        def fn(x):
            w = torch.nn.Parameter(torch.ones(4, 4))
            y = (x @ w).sum()
            (grad,) = torch.autograd.grad(y, w)
            return grad is not None, w.grad is None, grad

        x = torch.randn(4, 4)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
        res = opt_fn(x)

        self.assertTrue(res[0])
        self.assertTrue(res[1])
        self.assertEqual(ref[2], res[2])

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

    @unittest.skipIf(not TEST_CUDA, "pinned CPU memory requires CUDA")
    @unittest.skipIf(
        PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property"
    )
    def test_clone_input_preserves_pinned_cpu_tensor_metadata(self):
        base = torch.randn(4, 6, pin_memory=True)
        x = base[:, ::2].requires_grad_()
        x.grad = torch.ones_like(x).pin_memory()
        x._dynamo_dynamic_indices = {1}

        cloned = torch._dynamo.utils.clone_input(x)

        self.assertTrue(cloned.is_pinned())
        self.assertEqual(cloned.stride(), x.stride())
        self.assertEqual(cloned, x)
        self.assertNotEqual(cloned.data_ptr(), x.data_ptr())
        self.assertTrue(cloned.requires_grad)
        self.assertIsNotNone(cloned.grad)
        self.assertTrue(cloned.grad.is_pinned())
        self.assertEqual(cloned.grad, x.grad)
        self.assertEqual(cloned._dynamo_dynamic_indices, x._dynamo_dynamic_indices)
        self.assertIsNot(cloned._dynamo_dynamic_indices, x._dynamo_dynamic_indices)

        expanded = torch.arange(3, dtype=torch.float32, pin_memory=True).expand(2, 3)
        expanded_cloned = torch._dynamo.utils.clone_input(expanded)
        self.assertTrue(expanded_cloned.is_pinned())
        self.assertEqual(expanded_cloned, expanded)

    def test_clone_input_does_not_check_pinned_memory_for_wrapper_subclass(self):
        class WrapperSubclass(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_wrapper_subclass(
                    cls,
                    elem.size(),
                    strides=elem.stride(),
                    storage_offset=elem.storage_offset(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    requires_grad=elem.requires_grad,
                    device=elem.device,
                )

            def __init__(self, elem):
                self.elem = elem

            def __tensor_flatten__(self):
                return ["elem"], None

            @staticmethod
            def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
                if metadata is not None:
                    raise AssertionError("Expected metadata to be None")
                return WrapperSubclass(
                    inner_tensors["elem"].as_strided(outer_size, outer_stride)
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if func is torch.ops.aten.is_pinned.default:
                    raise AssertionError(
                        "clone_input should not check pinning on wrapper subclasses"
                    )
                if kwargs is None:
                    kwargs = {}
                args = python_pytree.tree_map_only(
                    WrapperSubclass, lambda x: x.elem, args
                )
                kwargs = python_pytree.tree_map_only(
                    WrapperSubclass, lambda x: x.elem, kwargs
                )
                out = func(*args, **kwargs)
                return python_pytree.tree_map_only(torch.Tensor, WrapperSubclass, out)

        x = WrapperSubclass(torch.randn(4, 6))

        cloned = torch._dynamo.utils.clone_input(x)

        self.assertIsInstance(cloned, WrapperSubclass)
        self.assertEqual(cloned.elem, x.elem)
        self.assertNotEqual(cloned.elem.data_ptr(), x.elem.data_ptr())

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

    def test_const_dict_variable_python_type(self):
        def fn(x):
            d = {"a": 1, "b": 2}
            od = collections.OrderedDict([("x", 3), ("y", 4)])
            return x + d["a"] + od["x"]

        eager_result = fn(torch.ones(3))
        compiled_result = torch.compile(fn, backend="eager", fullgraph=True)(
            torch.ones(3)
        )
        self.assertEqual(eager_result, compiled_result)

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

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        self.assertTrue(opt_fn())

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
                torch.compile(foo, fullgraph=True, backend="eager")(torch.randn(4, 4))

        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="silent"
        ):
            foo_v2_compile = Foo()
            foo_v2_eager = Foo()
            inp = torch.randn(4, 4)
            res = torch.compile(foo_v2_compile, fullgraph=True, backend="eager")(
                torch.randn(4, 4)
            )
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
            torch.compile(foo, fullgraph=True, backend="eager")(torch.randn(4, 4))

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

    def test_function_return_none_creates_constant_variable(self):
        """
        Test that functions returning None properly return ConstantVariable.create(None)
        instead of raw None, which would violate the stack's type contract.

        Regression test for: Avoid using Optional[VariableTracker]
        """

        def gn(x):
            return

        torch._dynamo.config.reorderable_logging_functions.add(gn)

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            if gn(x) is None:
                return x + 2
            return x + 4

        # If this doesn't crash, the test passes
        fn(torch.ones(3))

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
            assert isinstance(x, tensor_type)  # noqa: S101

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

    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_or_union_type_opaque_class(self):
        # Test that or_ on opaque class types (e.g. Shard | _StridedShard)
        # doesn't cause a graph break.
        from torch.distributed.tensor import Shard
        from torch.distributed.tensor.placement_types import _StridedShard

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            _ = Shard | _StridedShard
            return x + 1

        x = torch.randn(4)
        result = fn(x)
        self.assertEqual(result, x + 1)
        self.assertEqual(cnt.frame_count, 1)

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

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_foreach_tensor_scalar(self):
        # Test that foreach ops with tensor scalar values can be traced fullgraph
        # when capture_scalar_outputs is enabled. The scalar's .item() creates an
        # unbacked symbol that should be properly ignored since it only affects
        # tensor values, not shapes.
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = torch.randn(4, 4)
        val = torch.tensor(0.5, dtype=torch.float64)

        # Test _foreach_addcmul with value arg
        def fn_addcmul(a, b, c, val):
            return torch._foreach_addcmul([a], [b], [c], value=1 - val)

        expected = fn_addcmul(a, b, c, val)
        actual = torch.compile(fn_addcmul, backend="eager", fullgraph=True)(
            a, b, c, val
        )
        self.assertEqual(expected, actual)

        # Test _foreach_addcdiv with value arg
        def fn_addcdiv(a, b, c, val):
            return torch._foreach_addcdiv([a], [b], [c], value=1 - val)

        expected = fn_addcdiv(a, b, c, val)
        actual = torch.compile(fn_addcdiv, backend="eager", fullgraph=True)(
            a, b, c, val
        )
        self.assertEqual(expected, actual)

        # Test _foreach_add with scalar arg
        def fn_add(a, val):
            return torch._foreach_add([a], 1 - val)

        expected = fn_add(a, val)
        actual = torch.compile(fn_add, backend="eager", fullgraph=True)(a, val)
        self.assertEqual(expected, actual)

        # Test _foreach_mul with scalar arg
        def fn_mul(a, val):
            return torch._foreach_mul([a], 1 - val)

        expected = fn_mul(a, val)
        actual = torch.compile(fn_mul, backend="eager", fullgraph=True)(a, val)
        self.assertEqual(expected, actual)

        # Test _foreach_pow with scalar exponent
        def fn_pow(a, val):
            return torch._foreach_pow([a.abs()], 1 - val.item())

        expected = fn_pow(a, val)
        actual = torch.compile(fn_pow, backend="eager", fullgraph=True)(a, val)
        self.assertEqual(expected, actual)

    # Test that ops with Scalar arguments (alpha, beta, value) work with
    # 0-d tensor inputs under fullgraph=True. The C++ arg parser calls
    # .item() on 0-d tensors to convert them to Scalars, creating unbacked
    # symbols that should be ignored since they only affect tensor values,
    # not shapes.
    @parametrize(
        "op",
        [
            "add_method",
            "add_positional",
            "add_func",
            "addcmul",
            "addcmul_positional",
            "addcdiv",
            "addcdiv_positional",
            "baddbmm",
        ],
    )
    def test_scalar_arg_0d_tensor(self, op):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = torch.randn(4, 4)
        alpha = torch.tensor(2.0)
        beta = torch.tensor(0.5)
        batch1 = torch.randn(2, 4, 3)
        batch2 = torch.randn(2, 3, 4)
        m_batch = torch.randn(2, 4, 4)

        cases = {
            "add_method": lambda: a.add(b, alpha=alpha),
            "add_positional": lambda: a.add(alpha, b),
            "add_func": lambda: torch.add(a, b, alpha=alpha),
            "addcmul": lambda: a.addcmul(b, c, value=beta),
            "addcmul_positional": lambda: a.addcmul(beta, b, c),
            "addcdiv": lambda: a.addcdiv(b, c.abs() + 1, value=beta),
            "addcdiv_positional": lambda: a.addcdiv(beta, b, c.abs() + 1),
            "baddbmm": lambda: m_batch.baddbmm(batch1, batch2, alpha=alpha, beta=beta),
        }

        fn = cases[op]
        expected = fn()
        actual = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(expected, actual)

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

    def test_cross_entropy_loss_prob_target_dynamic(self):
        shapes = [(3, 5), (4, 7)]
        compiled = torch.compile(F.cross_entropy, dynamic=True, backend="eager")
        for N, C in shapes:
            x = torch.randn(N, C)
            target = torch.rand(N, C).softmax(dim=1)
            weight = torch.rand(C)
            for reduction in ["none", "mean", "sum"]:
                for label_smoothing in [0.0, 0.5]:
                    for w in [None, weight]:
                        expected = F.cross_entropy(
                            x,
                            target,
                            weight=w,
                            reduction=reduction,
                            label_smoothing=label_smoothing,
                        )
                        actual = compiled(
                            x,
                            target,
                            weight=w,
                            reduction=reduction,
                            label_smoothing=label_smoothing,
                        )
                        self.assertEqual(expected, actual)

    def test_repr(self):
        class Config:
            def __repr__(self):
                return "Config()"

        def forward(x, config):
            return x * len(repr(config))

        config = Config()
        x = torch.randn(2, 2)

        compiled = torch.compile(forward, fullgraph=True, backend="eager")
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
                assert t <= self.block_size, (  # noqa: S101
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

    @torch._dynamo.config.patch(trace_autograd_ops=True)
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
        self.assertEqual(cnts.frame_count, 1)

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

    def test_multithread_compile_dynamic(self):
        def f(x):
            comptime.assert_static(x.shape[0])
            return x * x

        def _do_test(func):
            success = True

            def run(offset):
                for i in range(20):
                    print(func(torch.randn(i * 2 + offset)))

            t1 = threading.Thread(target=run, args=[0])
            t2 = threading.Thread(target=run, args=[1])

            def exc_hook(x):
                nonlocal success
                success = False

            try:
                threading.excepthook = exc_hook
                t1.start()
                t2.start()

                t1.join()
                t2.join()
            finally:
                threading.excepthook = threading.__excepthook__
            self.assertTrue(success)

        _do_test(torch.compile(f, backend="eager", dynamic=False))
        torch._dynamo.reset()

        f_opt = torch.compile(f, backend="eager")

        def g(x):
            with torch._dynamo.config.patch(
                automatic_dynamic_shapes=False, assume_static_by_default=True
            ):
                f_opt(x)

        _do_test(g)

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
     