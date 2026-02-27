# Owner(s): ["module: autograd"]
# ruff: noqa: F841

import collections
import contextlib
import functools
import gc
import io
import math
import operator
import os
import pickle
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
import warnings
import weakref
from collections import OrderedDict
from copy import deepcopy
from functools import partial, reduce
from itertools import product
from operator import mul
from typing import TYPE_CHECKING

import torch
import torch.autograd._functions
import torch.autograd.forward_ad as fwAD
from torch import inf, nan, nn
from torch.autograd import (
    _calculate_shape,
    detect_anomaly,
    Function,
    kineto_available,
    Variable,
)
from torch.autograd.function import InplaceFunction, once_differentiable
from torch.autograd.graph import GradientEdge
from torch.autograd.profiler import emit_itt, emit_nvtx, profile, record_function
from torch.autograd.profiler_util import (
    _format_time,
    EventList,
    FunctionEvent,
    FunctionEventAvg,
)
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    dtypes,
    dtypesIfCUDA,
    dtypesIfMPS,
    expectedFailureMPS,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    skipMeta,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_methods_invocations import mask_not_all_zeros
from torch.testing._internal.common_utils import (
    disable_gc,
    gradcheck,
    gradgradcheck,
    instantiate_parametrized_tests,
    IS_MACOS,
    IS_WINDOWS,
    parametrize,
    run_tests,
    scoped_load_inline,
    set_warn_always_context,
    skipCUDANonDefaultStreamIf,
    skipIfMPS,
    skipIfNoLapack,
    skipIfSlowGradcheckEnv,
    skipIfTorchDynamo,
    skipIfWindows,
    skipIfXpu,
    slowTest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import (
    checkpoint,
    checkpoint_sequential,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)
from torch.utils.flop_counter import FlopCounterMode


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


def graph_desc(fn):
    if fn is None:
        return "None"
    result = type(fn).__name__ + "("
    next_functions = fn.next_functions
    for next_fn, _ in next_functions:
        result += graph_desc(next_fn)
        result += ", "
    if next_functions:
        result = result[:-2]
    return result + ")"


class TestAutograd(TestCase):
    def tearDown(self):
        torch.autograd._force_original_view_tracking(False)
        super(TestCase, self).tearDown()

    def test_copy_slices_graph_task_updates(self):
        def f1(x, y):
            out = x.clone().view(-1)
            out += y
            return out

        def f2(x, y):
            out = x.clone().view(-1)
            b = out * 2
            out += y
            return out + b

        x = torch.rand(2, requires_grad=True)
        y = torch.rand(2, requires_grad=True)

        y_safe = torch._C._functions.DelayedError("Boom!", 1)(y)

        for f in [f1, f2]:
            # Ensure that the error Node works
            out = f(x, y_safe)
            with self.assertRaisesRegex(RuntimeError, "Boom!"):
                out.sum().backward()

            out = f(x, y_safe)
            with self.assertRaisesRegex(RuntimeError, "Boom!"):
                torch.autograd.grad(out.sum(), y)

            # Ensure that if we don't ask for y, it doesn't crash
            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), x)

            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), y_safe)

            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), (x, y_safe))

        # Ensure that we don't run extra view Node
        def f3(x, y):
            out = x.clone().view(-1)

            def hook(*args):
                # This should never be called!
                self.assertTrue(False)

            out.register_hook(hook)

            b = out + y
            out += y
            return out + b, b

        out, b = f3(x, y_safe)
        torch.autograd.grad(out.sum(), (b, y_safe))

    def test_grad_mode_class_decoration(self):
        # Decorating class is deprecated and should not be used
        with self.assertWarnsRegex(FutureWarning, "Decorating classes is deprecated"):

            @torch.no_grad()
            class Foo:
                def __init__(self) -> None:
                    if torch.is_grad_enabled():
                        raise AssertionError("expected grad to be disabled")

                def foo(self):
                    # Not applied to methods
                    if not torch.is_grad_enabled():
                        raise AssertionError("expected grad to be enabled")

            # Show that we can actually construct the class
            foo = Foo()
            foo.foo()

        # Decorating functions or methods is fine though
        with warnings.catch_warnings(record=True) as w:

            @torch.no_grad()
            def foo():
                if torch.is_grad_enabled():
                    raise AssertionError("expected grad to be disabled")

            foo()

            class Foo2:
                @torch.no_grad()
                def __init__(self) -> None:
                    if torch.is_grad_enabled():
                        raise AssertionError("expected grad to be disabled")

                @torch.no_grad()
                def foo(self):
                    if torch.is_grad_enabled():
                        raise AssertionError("expected grad to be disabled")

            foo2 = Foo2()
            foo2.foo()

        self.assertEqual(len(w), 0)

    def test_tensor_grad_warnings(self):
        dummy = torch.empty(1)

        with warnings.catch_warnings(record=True) as w:
            # Accessing .grad on leaf
            dummy.requires_grad_()
            foo = dummy.grad
            self.assertEqual(len(w), 0)

            # Accessing .grad on non-leaf
            dummy = dummy.clone()
            foo = dummy.grad
            self.assertEqual(len(w), 1)

            # Accessing .grad on non-leaf that retains gradients
            dummy.retain_grad()
            foo = dummy.grad
            self.assertEqual(len(w), 1)

    def _function_test(self, cls):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        result = cls.apply(x, 2, y)
        go = torch.ones((), requires_grad=True)
        result.sum().backward(go, create_graph=True)

        self.assertEqual(x.grad, y + torch.ones(5, 5))
        self.assertEqual(y.grad, x + torch.ones(5, 5) * 2)
        self.assertIsNotNone(x.grad.grad_fn)
        self.assertIsNotNone(y.grad.grad_fn)

        return x, y

    def test_function(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            @staticmethod
            def backward(ctx, grad_output):
                var1, var2 = ctx.saved_tensors
                # NOTE: self is the test case here
                self.assertIsInstance(var1, torch.Tensor)
                self.assertIsInstance(var2, torch.Tensor)
                self.assertIsInstance(grad_output, torch.Tensor)
                return (
                    grad_output + grad_output * var2,
                    None,
                    grad_output * ctx.pyscalar + grad_output * var1,
                )

        x, y = self._function_test(MyFunction)

        x_grad_desc = graph_desc(x.grad.grad_fn)
        y_grad_desc = graph_desc(y.grad.grad_fn)
        self.assertExpected(x_grad_desc, "x_grad_desc")
        self.assertExpected(y_grad_desc, "y_grad_desc")

        # Avoid leaking memory
        x.grad = None
        y.grad = None

    def test_once_differentiable(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            @staticmethod
            @once_differentiable
            def backward(ctx, grad_output):
                self.assertFalse(torch.is_grad_enabled())
                t1, t2 = ctx.saved_tensors
                return (
                    grad_output + grad_output * t2,
                    None,
                    grad_output * ctx.pyscalar + grad_output * t1,
                )

        x, y = self._function_test(MyFunction)
        self.assertEqual(
            graph_desc(x.grad.grad_fn),
            "CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))",
        )
        self.assertEqual(
            graph_desc(y.grad.grad_fn),
            "CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))",
        )

        # Avoid leaking memory
        x.grad = None
        y.grad = None

    def test_function_returns_input(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad * 2

        for shape in [(1,), ()]:
            v = torch.ones(shape, requires_grad=True)
            MyFunction.apply(v).backward()
            self.assertEqual(v.grad, torch.full(shape, 2.0))

            with torch.no_grad():
                v.grad.zero_()
            MyFunction.apply(v.clone()).backward()
            self.assertEqual(v.grad, torch.full(shape, 2.0))

    def test_function_returns_undefined_tensor(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                return None

        # Test that undefined tensors returned from custom backward function
        # are propagated as undefined and not tensor full of zeroes
        x = torch.ones(1, requires_grad=True)

        MyFunction.apply(x).backward()
        self.assertIsNone(x.grad)

        MyFunction.apply(x**2).backward()
        self.assertIsNone(x.grad)

        MyFunction.apply(x).sum().backward()
        self.assertIsNone(x.grad)

        self.assertIsNone(
            torch.autograd.grad(MyFunction.apply(x), x, allow_unused=True)[0]
        )

    def test_materialize_grads(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                self.assertEqual(grad, torch.zeros(1))
                return grad

        x = torch.ones(1, requires_grad=True)
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    def test_dont_materialize_grads(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.set_materialize_grads(False)
                return x

            @staticmethod
            def backward(ctx, grad):
                self.assertIsNone(grad)
                return grad

        x = torch.ones(1, requires_grad=True)
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    @skipIfTorchDynamo("compile tested in test/dynamo/test_autograd_function.py")
    def test_set_materialize_non_diff_grads(self):
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out0 = x.clone()
                out1 = x.clone()
                ctx.mark_non_differentiable(out1)
                ctx._materialize_non_diff_grads = False
                return out0, out1

            @staticmethod
            def backward(ctx, g0, g1):
                self.assertIsNone(g1)
                return g0

        a = torch.tensor(1.0, requires_grad=True)
        out = Func.apply(a)[0]
        out.backward()

    def test_unused_grad_requires_grad_with_materialize(self):
        x = torch.ones(10, requires_grad=True)
        y = torch.ones(10, requires_grad=True)
        z = (x**2).sum()

        g = torch.autograd.grad(
            z, (x, y), allow_unused=True, materialize_grads=True, create_graph=False
        )

        self.assertFalse(g[0].requires_grad)
        self.assertFalse(g[1].requires_grad)

    def test_legacy_function_deprecation_exception(self):
        # Trigger exception
        class MyFunction(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

        # Check exception occurs
        with self.assertRaisesRegex(
            RuntimeError,
            "Legacy autograd function with non-static forward method is deprecated",
        ):
            MyFunction()(torch.randn(3, 4))

    class SimulateBackwardError(Function):
        @staticmethod
        def forward(ctx, input):
            return input.clone()

        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            raise Exception("Simulate error on backward pass")  # noqa: TRY002

    def test_custom_function_exception(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

        tmp = (t1 + t2) * (t1 + t2)
        t3 = TestAutograd.SimulateBackwardError.apply(tmp)
        with self.assertRaisesRegex(Exception, "Simulate error on backward pass"):
            t3.sum().backward()

    def test_custom_function_non_tensor_inputs_outputs(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale

                # Save scale
                ctx.scale = scale
                ctx.save_for_backward(t1, t2, t3)
                return scale, t4, None, True, t5, "bar", t1

            @staticmethod
            @once_differentiable
            def backward(ctx, *grads):
                # Verify grads
                self.assertEqual(7, len(grads))
                self.assertIsNone(grads[0])
                self.assertIsNone(grads[2])
                self.assertIsNone(grads[3])
                self.assertIsNone(grads[5])

                scale = ctx.scale
                var1, var2, var3 = ctx.saved_tensors
                return (
                    grads[1] * scale + grads[4] * var2 * scale + grads[6],
                    grads[1] * var3 * scale + grads[4] * var1 * scale,
                    None,
                    grads[1] * var2 * scale + grads[4] * scale,
                )

        t1 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t2 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t3 = torch.rand(10, dtype=torch.double)
        scale = random.randint(0, 10)
        res = MyFunction.apply(t1, t2, scale, t3)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

        # Validate running backward.
        torch.autograd.backward([res[1].sum(), res[4].sum(), res[6].sum()])
        self.assertIsNotNone(t1.grad)
        self.assertIsNotNone(t2.grad)
        self.assertIsNone(t3.grad)

        # Test gradcheck
        def foo(t1, t2, t3):
            res = MyFunction.apply(t1, t2, scale, t3)
            return res[1], res[4], res[6]

        gradcheck(foo, (t1, t2, t3))

    def test_custom_function_no_tensors(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale
                return scale, t4, None, True, t5, "bar", t1

            @staticmethod
            @once_differentiable
            def backward(ctx, *args):
                return (args[0], args[1], None, args[2])

        t1 = random.random()
        t2 = random.random()
        t3 = random.random()
        scale = random.randint(0, 10)
        res = MyFunction.apply(t1, t2, scale, t3)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

    def test_invalid_gradients(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                return torch.randn(10, dtype=torch.float)

        with self.assertRaisesRegex(RuntimeError, "expected shape"):
            input = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
            MyFunction.apply(input).sum().backward()

    def test_unrelated_inputs(self):
        # test to ensure grad(grad)check runs successfully even if there is an
        # unrelated (but differentiable) inputs

        def my_function(x, y):
            return x * x

        x = torch.rand(10, dtype=torch.double, requires_grad=True)
        y = torch.rand(10, dtype=torch.double, requires_grad=True)

        gradcheck(my_function, (x, y))
        gradgradcheck(my_function, (x, y))

    def test_not_implemented_grad(self):
        a = torch.rand(2, requires_grad=True)
        # if grad for nextafter ends up being implemented, this should be changed
        y = torch.nextafter(a, a).sum()
        with self.assertRaisesRegex(
            NotImplementedError, "the derivative for .* is not implemented"
        ):
            y.backward()

    def test_not_implemented_fwad(self):
        x = torch.randn(3)
        v = torch.rand(3)

        with fwAD.dual_level():
            dual_x = fwAD.make_dual(x, v)

            err_msg = r"Trying to use forward AD with .* that does not support it"
            hint_msg = "Running forward AD for an OP that does not implement it should raise a NotImplementedError"

            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                # if forward AD ends up being implemented for torch.igamma, choose a different op
                torch.igamma(dual_x, dual_x)

    def test_saved_tensor_hooks_extra_exit_during_bw_no_crash(self):
        # This usage of saved tensor is not supported, but should not crash
        def unpack(x):
            ctx_1.__exit__()
            return x

        ctx_1 = torch.autograd.graph.saved_tensors_hooks(lambda x: x, unpack)
        ctx_2 = torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x)

        for _ in range(10):
            with ctx_2:
                ctx_1.__enter__()
                x = torch.randn(3, 3, requires_grad=True)
                x.sin().sum().backward()

        # Clean up
        for _ in range(10):
            ctx_1.__exit__()

        # Validate there are no more hooks on the stack
        a = torch.tensor(1.0, requires_grad=True)
        y = a.exp()
        y.grad_fn._raw_saved_result.register_hooks(lambda x: x, lambda x: x)

    def test_saved_tensor_hooks_extra_enter_during_bw_no_leak(self):
        # This usage of saved tensor is not supported, but should not leak
        def scope():
            def unpack(x):
                weak_ctx_1().__enter__()
                return x

            ctx_1 = torch.autograd.graph.saved_tensors_hooks(lambda x: x, unpack)
            weak_ctx_1 = weakref.ref(ctx_1)

            x = torch.randn(3, 3, requires_grad=True)
            with ctx_1:
                x.sin().sum().backward()
            return weakref.ref(unpack)

        with disable_gc():
            unpack_hook_ref = scope()
            self.assertIsNone(unpack_hook_ref())

    def test_will_engine_execute_node(self):
        counter = [0]

        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, gO):
                return gO * 2

        def get_grad_fn(t):
            if t.requires_grad and t.grad_fn is None:
                return t.clone().grad_fn.next_functions[0][0]
            else:
                return t.grad_fn

        a = torch.randn(2, 3, 4, requires_grad=True)
        a2 = torch.randn(2, 3, 4, requires_grad=True)
        b = a * a2
        b2 = b.cos()
        c = MyFunction.apply(b)

        should_execute = list(map(get_grad_fn, (a, b, c)))
        should_not_execute = list(map(get_grad_fn, (a2, b2)))

        def fn(x):
            counter[0] += 1

            for g in should_execute:
                self.assertTrue(torch._C._will_engine_execute_node(g))

            for g in should_not_execute:
                self.assertFalse(torch._C._will_engine_execute_node(g))

        h1 = b.register_hook(fn)
        h2 = c.register_hook(fn)

        # .backward(inputs=) is OK
        out = c.sum()
        torch.autograd.backward(out, inputs=(a, b), retain_graph=True)
        self.assertEqual(counter[0], 2)

        # .backward() is OK
        should_execute = list(map(get_grad_fn, (a, a2, b, c)))
        should_not_execute = list(map(get_grad_fn, (b2,)))
        torch.autograd.backward(out, retain_graph=True)

        # .grad is NOT OK when leaf is passed (this is the current state, subject to change)
        with self.assertRaisesRegex(
            RuntimeError, "are currently running autograd.grad()"
        ):
            torch.autograd.grad(out, (a,))

        # .grad is OK when non-leaf is passed
        a = torch.randn(1, 2, 3, requires_grad=True) * 2
        b = a * 2

        def fn(x):
            # Check a non-leaf
            counter[0] += 1
            self.assertTrue(torch._C._will_engine_execute_node(b.grad_fn))

        h3 = b.register_hook(fn)
        counter[0] = 0
        torch.autograd.grad(b.sum(), (a,))
        self.assertEqual(counter[0], 1)

        # Verify other errors are raised
        with self.assertRaisesRegex(RuntimeError, "during the backward pass"):
            torch._C._will_engine_execute_node(out.grad_fn)

        with self.assertRaisesRegex(RuntimeError, "expects an grad_fn"):
            torch._C._will_engine_execute_node(out)

        # Ensure we don't leak memory
        h1.remove()
        h2.remove()
        h3.remove()

    def test_custom_function_vmap_defaults(self):
        class MySquare(Function):
            @staticmethod
            def forward(x):
                return x**2

            @staticmethod
            def setup_context(ctx, inputs, output):
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                return gO * 2 * x

        self.assertFalse(MySquare.generate_vmap_rule)
        self.assertTrue(hasattr(MySquare, "vmap"))

    def test_custom_function_setup_context_simple(self):
        class MySquare(Function):
            @staticmethod
            def forward(x):
                return x**2

            @staticmethod
            def setup_context(ctx, inputs, output):
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                return gO * 2 * x

        x = torch.randn([], requires_grad=True)
        y = MySquare.apply(x)
        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, 2 * x)

    def test_custom_function_setup_context_multi_output(self):
        # Multiple outputs with some non-Tensor outputs.
        class MySquare(Function):
            @staticmethod
            def forward(x):
                two_x = x.item() * 2
                return x**2, two_x

            @staticmethod
            def setup_context(ctx, inputs, output):
                (x,) = inputs
                _, two_x = output
                ctx.two_x = two_x

            @staticmethod
            @once_differentiable
            def backward(ctx, gO, _):
                return gO * ctx.two_x

        x = torch.randn([], requires_grad=True)
        y, _ = MySquare.apply(x)
        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, 2 * x)

    def test_custom_function_setup_context_multi_input(self):
        class MyReshape(Function):
            @staticmethod
            def forward(x, shape, scale_forward, scale_backward):
                return x.reshape(shape) * scale_forward

            @staticmethod
            def setup_context(ctx, inputs, output):
                x, shape, scale_forward, scale_backward = inputs
                ctx.scale_backward = scale_backward
                ctx.x_shape = x.shape

            @staticmethod
            def backward(ctx, gO):
                return gO.reshape(ctx.x_shape) * ctx.scale_backward, None, None, None

        class MyReshapeRef(Function):
            @staticmethod
            def forward(ctx, x, shape, scale_forward, scale_backward):
                ctx.scale_backward = scale_backward
                ctx.x_shape = x.shape
                return x.reshape(shape) * scale_forward

            @staticmethod
            def backward(ctx, gO):
                return gO.reshape(ctx.x_shape) * ctx.scale_backward, None, None, None

        def test(x, shape, scale_forward, scale_backward):
            y = MyReshape.apply(x, shape, scale_forward, scale_backward).sum()
            (gx,) = torch.autograd.grad(y, x)

            y_expected = MyReshapeRef.apply(
                x, shape, scale_forward, scale_backward
            ).sum()
            (gx_expected,) = torch.autograd.grad(y_expected, x)

            self.assertEqual(y_expected, y)
            self.assertEqual(gx_expected, gx)

        test(torch.randn(24, requires_grad=True), (3, 8), 7, 11)
        test(torch.randn(2, 3, 4, requires_grad=True), (6, 4), -1, 2)

    def test_multiple_insert_removal_caching(self):
        torch._C._set_cached_tensors_enabled(True)
        try:
            x = torch.rand([4])

            torch._C._add_cached_tensor(x)
            self.assertTrue(torch._C._is_cached_tensor(x))

            torch._C._add_cached_tensor(x)
            torch._C._remove_cached_tensor(x)

            self.assertFalse(torch._C._is_cached_tensor(x))
        finally:
            torch._C._set_cached_tensors_enabled(False)

    def test_accumulate_grad(self):
        grad_output = torch.ones(5, 5)

        def compute_grad(create_graph):
            x = torch.randn(5, 5, requires_grad=True)
            y = x + 2
            y.backward(grad_output, retain_graph=True)
            x_grad = x.grad
            x_grad_clone = x.grad.clone()
            y.backward(grad_output, create_graph=create_graph)
            return x_grad, x_grad_clone

        # Accumulate in-place when create_graph is False
        x_grad, x_grad_clone = compute_grad(create_graph=False)
        self.assertEqual(x_grad, x_grad_clone * 2)

        # Accumulate out-of-place when create_graph is True
        x_grad, x_grad_clone = compute_grad(create_graph=True)
        self.assertEqual(x_grad, x_grad_clone)

    def test_accumulate_grad_tensor_reference(self):
        def _test_grad_tensor(
            params_grad_tensor,
            backward_grad_tensor,
            should_preserve_reference,
            create_graph,
        ):
            params = torch.tensor([1.5, 1.5]).requires_grad_()
            params.grad = params_grad_tensor
            grad_saved = params.grad
            params.backward(backward_grad_tensor, create_graph=create_graph)
            self.assertEqual(
                id(grad_saved) == id(params.grad), should_preserve_reference
            )

        for create_graph in (False, True):
            # Accumulate dense gradient to sparse gradient will change the `params.grad` reference
            _test_grad_tensor(
                torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])
                ),
                torch.tensor([1.5, 1.5]),
                False,  # never accumulates in-place
                create_graph,
            )

            # Accumulate dense gradient to dense gradient will preserve the `params.grad` reference,
            # but only if create_graph=False.
            _test_grad_tensor(
                torch.tensor([1.5, 1.5]),
                torch.tensor([1.5, 1.5]),
                not create_graph,
                create_graph,
            )

            # Accumulate sparse gradient to sparse gradient will preserve the `params.grad` reference,
            # but only if create_graph=False.
            _test_grad_tensor(
                torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])
                ),
                torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])
                ),
                not create_graph,
                create_graph,
            )

    def test_accumulate_grad_with_zero_numel_grad(self):
        a = torch.rand(4, 0, requires_grad=True)
        b = torch.rand(4, 1, requires_grad=True)
        c = a + b
        if c.shape != (4, 0):
            raise AssertionError(f"expected shape (4, 0), got {c.shape}")
        c.sum().backward()

        self.assertEqual(b.grad, torch.zeros(4, 1))
        self.assertEqual(a.grad, torch.zeros(4, 0))

    def test_hessian_vector(self):
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)

        z = x**2 + y * x + y**2
        z.backward(torch.ones(2, 2), create_graph=True)

        with torch.no_grad():
            x_grad = 2 * x + y
            y_grad = x + 2 * y
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        grad_sum = 2 * x.grad + y.grad
        grad_sum.backward(torch.ones(2, 2))
        x_hv = torch.ones(2, 2) * 5
        y_hv = torch.ones(2, 2) * 4
        self.assertEqual(x.grad, x_grad + x_hv)
        self.assertEqual(y.grad, y_grad + y_hv)

        # Avoid leaking memory
        x.grad = None
        y.grad = None

    def test_grad(self):
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        z = x**2 + y * x + y**2
        z.backward(torch.ones(2, 2), create_graph=True)

        x_grad = 2 * x + y
        y_grad = x + 2 * y
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        grad_sum = 2 * x.grad + y.grad
        x_hv = torch.autograd.grad(
            outputs=[grad_sum],
            grad_outputs=[torch.ones(2, 2)],
            inputs=[x],
            create_graph=True,
        )
        expected_x_hv = torch.ones(2, 2) * 5
        expected_y_hv = torch.ones(2, 2) * 4

        self.assertEqual(x_hv[0], expected_x_hv)
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        # Avoid leaking memory
        x.grad = None
        y.grad = None

        # Test that grad_outputs and outputs have the same shape
        grad_out = torch.ones(2)
        try:
            torch.autograd.grad(
                outputs=[grad_sum],
                grad_outputs=[grad_out],
                inputs=[x],
                create_graph=True,
            )
            self.assertFail()
        except RuntimeError as error:
            self.assertEqual(
                str(error),
                "Mismatch in shape: grad_output[0] has a shape of "
                + str(grad_out.shape)
                + " and output[0] has a shape of "
                + str(grad_sum.shape)
                + ".",
            )

    def test_grad_to_node(self):
        def check_matches(out, inp):
            ref = torch.autograd.grad(out.sum(), inp)

            edge = torch.autograd.graph.get_gradient_edge(inp)
            new = torch.autograd.grad(out.sum(), edge)
            self.assertEqual(ref, new)

        # We need to ensure that our main types of Node work (regular cpp Nodes,
        # AccumulateGrad Nodes and custom Function)
        x = torch.rand(2, requires_grad=True)
        out = x.clone()
        check_matches(out, x)

        x = x.clone()
        out = x.clone()
        check_matches(out, x)

        x = torch.autograd._functions.Resize.apply(x, (2,))
        out = x.clone()
        check_matches(out, x)

        x = torch.var_mean(x)[1]
        out = x.clone()
        check_matches(out, x)

    def test_grad_to_node_set(self):
        x = torch.rand(2, requires_grad=True)
        x_edge = torch.autograd.graph.get_gradient_edge(x)
        out = x.clone()

        with torch.no_grad():
            x.set_(torch.rand_like(x))

        with self.assertRaisesRegex(RuntimeError, "to not have been used in the graph"):
            torch.autograd.grad(out.sum(), x)

        # Works
        torch.autograd.grad(out.sum(), x_edge)

    def test_grad_to_node_inplace(self):
        x = torch.rand(2, requires_grad=True).clone()
        x_edge = torch.autograd.graph.get_gradient_edge(x)
        x *= 2

        g_old, g_new = torch.autograd.grad(x.sum(), (x_edge, x))
        self.assertEqual(g_old, 2 * torch.ones_like(x))
        self.assertEqual(g_new, torch.ones_like(x))

    def test_grad_to_node_multi(self):
        x = torch.rand(2, requires_grad=True).clone()
        y = torch.rand(2, requires_grad=True).clone()

        out = x + y

        ref = torch.autograd.grad(out.sum(), (x, y))

        inp_edges = (
            GradientEdge(x.grad_fn, x.output_nr),
            GradientEdge(y.grad_fn, y.output_nr),
        )
        new = torch.autograd.grad(out.sum(), inp_edges)

        self.assertEqual(ref, new)

    def test_grad_to_node_materialize(self):
        x = torch.rand(2, requires_grad=True).clone()
        edge_x = GradientEdge(x.grad_fn, x.output_nr)
        y = torch.rand(2, requires_grad=True).clone()
        edge_y = GradientEdge(y.grad_fn, y.output_nr)

        out = x.clone()

        # Works
        torch.autograd.grad(
            out.sum(), (edge_x, y), allow_unused=True, materialize_grads=True
        )
        torch.autograd.grad(
            out.sum(), (x, y), allow_unused=True, materialize_grads=True
        )
        torch.autograd.grad(out.sum(), (x, edge_y), allow_unused=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "materialize_grads cannot be used when the given input is a GradientEdge",
        ):
            torch.autograd.grad(
                out.sum(), (x, edge_y), allow_unused=True, materialize_grads=True
            )

    def test_backward_to_node(self):
        x = torch.rand(2, requires_grad=True).clone()
        edge_x = GradientEdge(x.grad_fn, x.output_nr)
        y = torch.rand(2, requires_grad=True).clone()
        edge_y = GradientEdge(y.grad_fn, y.output_nr)

        out = x.clone()

        # All should work in this case
        torch.autograd.backward(out.sum(), inputs=(edge_x, y))
        torch.autograd.backward(out.sum(), inputs=(x, y))
        torch.autograd.backward(out.sum(), inputs=(x, edge_y))
        torch.autograd.backward(out.sum(), inputs=(edge_x, edge_y))

    def test_grad_fn_input_metadata(self):
        x = torch.rand(2, requires_grad=True, dtype=torch.float32)
        y = torch.rand(2, requires_grad=True, dtype=torch.float32)
        z = x * y
        z_metadata = z.grad_fn._input_metadata[0]
        self.assertEqual(z_metadata.shape, (2,))
        self.assertEqual(z_metadata.dtype, torch.float32)

        # Multiple outputs
        b = torch.rand(3, 3, requires_grad=True)
        var, _ = torch.var_mean(b, dim=0)

        metadata_0 = var.grad_fn._input_metadata[0]
        metadata_1 = var.grad_fn._input_metadata[1]
        self.assertEqual(metadata_0.shape, (3,))
        self.assertEqual(metadata_1.shape, (3,))

        # Preserves symints
        nt = torch.nested.nested_tensor(
            [torch.randn(3, 2), torch.randn(2, 2)],
            layout=torch.jagged,
            requires_grad=True,
        )

        nt_metadata = nt.clone().grad_fn._input_metadata[0]

        self.assertIsInstance(nt_metadata.shape[1], torch.SymInt)
        self.assertEqual(nt_metadata.shape, nt.shape)
        self.assertTrue(nt_metadata.is_nested_tensor)
        self.assertFalse(nt_metadata.is_cpp_nested_tensor)
        self.assertEqual(nt_metadata.dtype, nt.dtype)

        class Test(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        x = torch.randn(3, 3, requires_grad=True)
        x = Test.apply(x)
        metadata = x.grad_fn._input_metadata[0]
        self.assertEqual(metadata.shape, (3, 3))

    def test_gradient_edge_output(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)

        def fn(x, reduce=True):
            tmp = x.sin().cos()
            if reduce:
                tmp = tmp.sum()
            out = tmp.exp().clone().sin().sum()
            tmp_edge = torch.autograd.graph.get_gradient_edge(tmp)
            return out, tmp_edge

        # Compute fn backward in two steps
        out, tmp_edge = fn(x)
        (tmp_grad,) = torch.autograd.grad(out, (tmp_edge,))

        (x_grad,) = torch.autograd.grad(tmp_edge, (x,), grad_outputs=(tmp_grad,))

        # Compare with as if we did it in one go.
        out, _ = fn(x)
        (x_grad_ref,) = torch.autograd.grad(out, (x,))
        self.assertEqual(x_grad, x_grad_ref)

        # Incorrect case: grad_outputs not passed/implicitly None and output is
        # not a scalar
        out, tmp_edge = fn(x, reduce=False)
        with self.assertRaisesRegex(
            RuntimeError, "grad can be implicitly created only for scalar output"
        ):
            torch.autograd.grad(tmp_edge, (x,))

        # grad_outputs is None, and output is a scalar is fine
        out, tmp_edge = fn(x, reduce=True)
        torch.autograd.grad(tmp_edge, (x,))

        # Incorrect case: grad_outputs wrong size
        out, tmp_edge = fn(x)
        with self.assertRaisesRegex(RuntimeError, "Mismatch in shape"):
            torch.autograd.grad(
                tmp_edge, (x,), grad_outputs=torch.tensor([1.0, 2.0, 3.0, 4.0])
            )

        # Incorrect case: wrong dtype
        out, tmp_edge = fn(x)
        (tmp_grad,) = torch.autograd.grad(out, (tmp_edge,))
        with self.assertRaisesRegex(RuntimeError, "required to have the same dtype"):
            torch.autograd.grad(
                tmp_edge,
                (x,),
                grad_outputs=torch.rand_like(tmp_grad, dtype=torch.complex64),
            )

        # Run with .backward() and compare with .grad()
        out, tmp_edge = fn(x)
        torch.autograd.backward(tmp_edge, retain_graph=True)
        (x_grad_ref,) = torch.autograd.grad(tmp_edge, (x,), retain_graph=True)
        self.assertEqual(x.grad, x_grad_ref)

        # Pass a tuple of GradientEdges
        x.grad = None
        torch.autograd.backward((tmp_edge,), retain_graph=True)
        self.assertEqual(x.grad, x_grad_ref)

        # Mixing GradientEdge and Tensors
        out1, tmp_edge1 = fn(x)
        out2, tmp_edge2 = fn(x)
        (x_grad_ref,) = torch.autograd.grad((tmp_edge1, out2), (x,), retain_graph=True)
        x.grad = None
        torch.autograd.backward((tmp_edge1, out2), retain_graph=True)
        self.assertEqual(x.grad, x_grad_ref)

        # .backward(): wrong shape
        out, tmp_edge = fn(x)
        with self.assertRaisesRegex(RuntimeError, "Mismatch in shape"):
            torch.autograd.backward(
                tmp_edge, inputs=(x,), grad_tensors=torch.tensor([1.0, 2.0, 3.0, 4.0])
            )

    def test_gradient_edge_graph_ownership(self):
        # Ensure we own the graph properly
        class Clone(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gX):
                return gX.clone()

        inp = torch.rand(1, requires_grad=True).clone()

        # C++ Node
        out = inp.clone()
        edge = torch.autograd.graph.get_gradient_edge(out)
        torch.autograd.backward(edge)
        del out
        torch.autograd.backward(edge)

        # python Node
        out = Clone.apply(inp)
        edge = torch.autograd.graph.get_gradient_edge(out)
        torch.autograd.backward(edge)
        del out
        torch.autograd.backward(edge)

    def test_grad_nonleaf(self):
        x_init = torch.randn(2, 2, requires_grad=True)
        x = x_init
        y = torch.randn(2, 2, requires_grad=True)
        grad_output = torch.ones(2, 2)

        def fn(x):
            return x**2 + y * x + y**2

        for _ in range(5):
            (grad_x,) = torch.autograd.grad(
                fn(x), x, grad_outputs=grad_output, create_graph=True
            )

            grad_x_expected = 2 * x + y
            self.assertIsNone(y.grad)
            self.assertIsNone(x.grad)
            self.assertEqual(grad_x, grad_x_expected)

            x = x + 0.05 * grad_x

        val_init = fn(x_init).sum()
        val_final = fn(x).sum()
        self.assertGreater(val_final, val_init)

        x.backward(grad_output)
        self.assertIsNotNone(y.grad)
        self.assertIsNotNone(x_init.grad)

    def test_grad_nonleaf_many_outputs(self):
        # This checks an edge case for function callbacks
        # We want to capture two grads of a function, but can only
        # register a single callback.
        x = torch.randn(4, 2, requires_grad=True)
        a, b = x.chunk(2)

        def hook(*grads):
            hook_called[0] = True

        hook_called = [False]
        x.register_hook(hook)

        go = torch.randn(2, 2)
        grad_a, grad_b = torch.autograd.grad(
            (a + 2 * b), [a, b], grad_outputs=go, create_graph=True
        )

        self.assertEqual(grad_a, go)
        self.assertEqual(grad_b, go * 2)
        self.assertFalse(hook_called[0])
        self.assertIsNone(x.grad)

    def test_grad_nonleaf_register_hook(self):
        # This checks an edge case for register_hook.
        # We want to capture grad of a nonleaf tensor,
        # but avoid segfault during backward of other nonleaf tensors
        x = torch.randn(5, requires_grad=True)
        x_list = x.unbind()

        x0 = x_list[0]
        hook_results = [None]

        def hook(grad):
            hook_results[0] = grad

        x0.register_hook(hook)

        x_list[0].backward()
        self.assertEqual(hook_results[0], torch.tensor(1.0))
        expected_grad = torch.tensor([1.0, 0, 0, 0, 0])
        self.assertEqual(x.grad, expected_grad)
        self.assertIsNone(x_list[0].grad)

        for i in range(1, 5, 1):
            x_list[i].backward()
            self.assertEqual(hook_results[0], None)
            expected_grad[i] = 1.0
            self.assertEqual(x.grad, expected_grad)
            self.assertIsNone(x_list[i].grad)

    def test_grad_materialize_grads(self):
        x = torch.tensor(0.5, requires_grad=True)
        a = torch.tensor(1.0, requires_grad=True)
        y = x * a
        dydx = torch.autograd.grad(y, x, create_graph=True)
        d2ydx2_none = torch.autograd.grad(dydx, x, create_graph=True, allow_unused=True)
        d2ydx2 = torch.autograd.grad(
            dydx, x, create_graph=True, allow_unused=True, materialize_grads=True
        )
        # `allow_unused` set to True implicitly
        d3ydx3 = torch.autograd.grad(d2ydx2, x, materialize_grads=True)
        self.assertIsNone(d2ydx2_none[0])
        self.assertEqual(d2ydx2[0].item(), 0)
        self.assertEqual(d3ydx3[0].item(), 0)
        with self.assertRaisesRegex(
            ValueError, "Expected allow_unused to be True or not passed when"
        ):
            torch.autograd.grad(y, x, allow_unused=False, materialize_grads=True)

    def test_post_accumulate_grad_hook_on_non_leaf(self):
        def hook(tensor):
            tensor.sub_(1.0)

        leaf = torch.rand(3, requires_grad=True)
        non_leaf = 2.0 * leaf

        with self.assertRaisesRegex(
            RuntimeError,
            "post accumulate grad hooks cannot be registered on non-leaf tensors",
        ):
            non_leaf.register_post_accumulate_grad_hook(hook)

    def test_post_accumulate_grad_hook_multiple_hooks(self):
        def hook1(tensor):
            tensor.sub_(tensor.grad)

        def hook2(tensor):
            tensor.mul_(4.0)

        tensor = torch.rand(3, requires_grad=True)
        tensor_ref = tensor.detach().clone()
        tensor.register_post_accumulate_grad_hook(hook1)
        tensor.register_post_accumulate_grad_hook(hook2)
        sum = tensor.sum()
        sum.backward()
        # both hooks should be called, in order
        self.assertEqual(4.0 * (tensor_ref - 1.0), tensor)

    def test_post_accumulate_grad_hook_multiple_tensors(self):
        def hook(tensor):
            tensor.sub_(tensor.grad)

        tensor1 = torch.rand(3, requires_grad=True)
        tensor1_ref = tensor1.detach().clone()
        tensor2 = torch.rand(5, requires_grad=True)
        tensor2_ref = tensor2.detach().clone()
        tensor1.register_post_accumulate_grad_hook(hook)
        tensor2.register_post_accumulate_grad_hook(hook)
        tensor1.sum().backward()
        tensor2.sum().backward()
        # both tensors should have been modified
        self.assertEqual(tensor1_ref - 1.0, tensor1)
        self.assertEqual(tensor2_ref - 1.0, tensor2)

    def test_post_accumulate_grad_hook_returns_not_None(self):
        def bad_hook(tensor):
            return tensor.grad

        tensor = torch.rand(2, 3, requires_grad=True)
        tensor.register_post_accumulate_grad_hook(bad_hook)
        # should error!
        with self.assertRaisesRegex(RuntimeError, "hooks should return None."):
            tensor.sum().backward()

    def test_post_accumulate_grad_hook_e2e(self):
        def setup_optim_in_bwd(model):
            optims = {}
            handles = []

            def optim_step_hook(param):
                optims[param].step()
                optims[param].zero_grad()

            for p in model.parameters():
                optims[p] = torch.optim.Adam([p])
                handles.append(p.register_post_accumulate_grad_hook(optim_step_hook))

            return handles

        model = torch.nn.Linear(3, 2)
        input = torch.rand(2, 3)
        handles = setup_optim_in_bwd(model)

        # make a copy for reference
        model_copy = deepcopy(model)
        optim_copy = torch.optim.Adam(model_copy.parameters())

        iters = 5

        for _ in range(iters):
            loss = model(input).sum()
            loss.backward()

            loss_copy = model_copy(input).sum()
            loss_copy.backward()
            optim_copy.step()
            optim_copy.zero_grad()

        params_copy = []  # freeze a copy of the params to compare later
        for p_reference, p in zip(model_copy.parameters(), model.parameters()):
            self.assertEqual(p_reference, p)
            params_copy.append(p_reference.detach().clone())

        # After removing the handle, the model should no longer update.
        for h in handles:
            h.remove()

        for _ in range(iters):
            loss = model(input).sum()
            loss.backward()

            loss_copy = model_copy(input).sum()
            loss_copy.backward()
            optim_copy.step()
            optim_copy.zero_grad()

        for p_static, p_reference, p in zip(
            params_copy, model_copy.parameters(), model.parameters()
        ):
            self.assertEqual(p_static, p)
            self.assertNotEqual(p_reference, p)

    def test_post_accumulate_grad_hook_gets_cleaned_up(self):
        def fun_stuff_with_hook():
            thing_to_put_in_hook = torch.rand(3)

            def hook(tensor):
                tensor.sub_(tensor.grad)
                tensor.add_(thing_to_put_in_hook)

            tensor = torch.rand(3, requires_grad=True)
            tensor.register_post_accumulate_grad_hook(hook)
            tensor.sum().backward()
            ref = weakref.ref(thing_to_put_in_hook)
            gc.collect()
            return tensor, ref

        with disable_gc():
            tensor, ref = fun_stuff_with_hook()
            self.assertIsNotNone(
                ref()
            )  # thing_to_put_in_hook should be kept alive by tensor

            del tensor
            gc.collect()
            self.assertIsNone(ref())  # thing_to_put_in_hook should be cleaned

    def test_post_accumulate_grad_hook_ordering(self):
        tensor = torch.rand(3, requires_grad=True)

        def pre_hook(grad):
            return grad.sub(2.0)

        def acc_grad_node_pre_hook(grad_out):
            return (grad_out[0].div(5.0),)

        def post_acc_grad_hook(tensor):
            tensor.grad.add_(0.5)

        def acc_grad_node_post_hook(grad_in, grad_out):
            tensor.grad = grad_out[0].mul(10)

        acc_grad = tensor.view_as(tensor).grad_fn.next_functions[0][0]
        tensor.register_hook(pre_hook)
        acc_grad.register_prehook(acc_grad_node_pre_hook)
        tensor.register_post_accumulate_grad_hook(post_acc_grad_hook)
        acc_grad.register_hook(acc_grad_node_post_hook)
        tensor.sum().backward()

        # the hooks should run in the order of:
        #   1. tensor prehook
        #   2. acc_grad prehook
        #   3. tensor post acc_grad hook
        #   4. acc_grad posthook
        # so that would be ((1 - 2) / 5 + 0.5) * 10 = 3
        self.assertEqual(torch.tensor([3.0, 3.0, 3.0]), tensor.grad)

    def test_hook_with_no_name(self):
        # Create a hook that do not have a __name__ attribute
        class MyHookClass:
            def __call__(self, grad):
                return grad.clone()

        x = torch.randn(5, requires_grad=True).clone()
        x.register_hook(MyHookClass())
        x.sum().backward()
        # Should run fine

    def test_prehook_ordering(self):
        # Hooks registered to tensor are ordered before those
        # that are registered to grad_fn
        log = []

        def hook1(g):
            log.append(1)
            return g * 3

        def hook2(gs):
            log.append(2)
            return tuple(g * 2 for g in gs)

        a = torch.tensor(1.0, requires_grad=True)
        b = a.clone()

        b.grad_fn.register_prehook(hook2)
        b.register_hook(hook1)
        b.grad_fn.register_prehook(hook2)

        acc = b.grad_fn.next_functions[0][0]
        a.register_hook(hook1)
        acc.register_prehook(hook2)
        a.register_hook(hook1)

        b.sum().backward(retain_graph=True)
        self.assertEqual(log, [1, 2, 2, 1, 1, 2])

        # grad also runs hooks on accumulate grad nodes, even though
        # the accumulate grad nodes are not actually executed
        log = []
        torch.autograd.grad(b.sum(), inputs=(a,), retain_graph=True)
        self.assertEqual(log, [1, 2, 2, 1, 1])

        log = []
        b.sum().backward(inputs=(b,))
        self.assertEqual(log, [1, 2, 2])
        # retains_grad hooks would not observe modifications by all pre hooks
        # because they are executed after
        self.assertEqual(b.grad.item(), 3)

    def test_retains_grad_can_always_observe_tensor_prehook(self):
        def tensor_prehook(g):
            return g * 2

        a = torch.tensor(1.0, requires_grad=True)
        b = a.clone()
        b.register_hook(tensor_prehook)
        b.retain_grad()
        b.register_hook(tensor_prehook)

        b.clone().backward()
        self.assertEqual(b.grad.item(), 4)

        a = torch.tensor(1.0, requires_grad=True)
        b = a.clone()
        b.retain_grad()
        b.register_hook(tensor_prehook)

        b.clone().backward()
        self.assertEqual(b.grad.item(), 2)

    def test_accumulate_grad_posthooks_can_observe_tensor_prehook(self):
        # Post hooks on accumulate should be able to observe changes to
        # grad made by tensor prehooks
        a = torch.tensor(1.0, requires_grad=True)

        def tensor_prehook(g):
            return g * 2

        def posthook(gO, gI):
            self.assertTrue(torch.allclose(gI[0], a * 2))
            self.assertEqual(len(gO), 0)

        def prehook(gI):
            self.assertTrue(torch.allclose(gI[0], a * 2))
            self.assertEqual(len(gI), 1)

        b = a.clone()
        acc = b.grad_fn.next_functions[0][0]
        acc.register_hook(posthook)
        acc.register_prehook(prehook)
        a.register_hook(tensor_prehook)

        b.backward()

    def test_accumulate_grad_posthooks_should_not_execute(self):
        def tensor_prehook(g):
            raise RuntimeError

        def posthook(gO, gI):
            raise RuntimeError

        a = torch.tensor(1.0, requires_grad=True)
        a.register_hook(tensor_prehook)
        b = torch.tensor(1.0, requires_grad=True)
        c = a.clone()
        acc = c.grad_fn.next_functions[0][0]
        acc.register_hook(posthook)

        out = a + b + c
        out.sum().backward(inputs=[b])

    def test_hook_edge_case_when_called_with_grad(self):
        # grad executes the tensor hooks of the next node but not
        # grad_fn pre hooks or the post hooks
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        c = b * 2

        tensor_hook_count = [0]
        prehook_count = [0]
        posthook_count = [0]

        def reset_counts():
            nonlocal tensor_hook_count, prehook_count, posthook_count
            tensor_hook_count = [0]
            prehook_count = [0]
            posthook_count = [0]

        def tensor_prehook(g):
            tensor_hook_count[0] += 1

        def prehook(g):
            prehook_count[0] += 1

        def posthook(gI, gO):
            posthook_count[0] += 1

        a.register_hook(tensor_prehook)
        b.register_hook(tensor_prehook)
        acc = b.grad_fn.next_functions[0][0]
        acc.register_hook(posthook)
        acc.register_prehook(prehook)
        b.grad_fn.register_hook(posthook)
        b.grad_fn.register_prehook(prehook)

        torch.autograd.grad(c, inputs=(b), retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 1)
        self.assertEqual(posthook_count[0], 0)
        self.assertEqual(prehook_count[0], 0)
        reset_counts()

        torch.autograd.grad(c, inputs=(a, b), retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 2)
        self.assertEqual(posthook_count[0], 1)
        self.assertEqual(prehook_count[0], 1)
        reset_counts()

        c.backward(retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 2)
        self.assertEqual(posthook_count[0], 2)
        self.assertEqual(prehook_count[0], 2)
        reset_counts()

        c.backward(inputs=(a, b), retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 2)
        self.assertEqual(posthook_count[0], 2)
        self.assertEqual(prehook_count[0], 2)

    def test_sharded_grad(self):
        leaves = [torch.zeros(5, 5, requires_grad=True) for _ in range(10)]
        intermediates = [l * i + l * l for i, l in enumerate(leaves)]
        loss = sum(v * i for i, v in enumerate(intermediates)).sum()

        # define a helper for dividing intermediates into groups
        def group(l, group_size):
            return (l[i : i + group_size] for i in range(0, len(l), group_size))

        # Compute the d loss / d intermediates in chunks of shard_size
        shard_size = 2
        d_intermediates = [
            d_i
            for intermediates_batch in group(intermediates, shard_size)
            for d_i in torch.autograd.grad(loss, intermediates_batch)
        ]
        # Compute rest of backward pass
        torch.autograd.backward(intermediates, d_intermediates)

        for i, l in enumerate(leaves):
            self.assertEqual(l.grad, i * i * (1 + l))

    def test_backward_badcalls(self):
        x = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            x.backward()

    def test_grad_badcalls(self):
        x = torch.ones(1)
        y = x**2
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.autograd.grad(x, y)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.autograd.grad(y, x)

        x = torch.ones(1, requires_grad=True)
        y = x**2
        torch.autograd.grad(y, x)  # this should succeed now

    def test_grad_empty_inputs(self):
        x = torch.tensor([1.0], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "grad requires non-empty inputs."):
            torch.autograd.grad(2 * x, [], grad_outputs=torch.tensor([1.0]))

    def test_grad_fn_badcalls(self):
        error_regex = "expected .* arguments, got .* instead"
        x = torch.ones(1, requires_grad=True)
        y = x**2
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn(x.detach(), x.detach())  # too many
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn()  # too few

        y.grad_fn(x.detach())  # this should succeed

    def test_grad_unreachable(self):
        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        # Make sure x and y have grad accumulators allocated
        z = x * 2
        w = y * 2

        grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_y)

        # This is slightly different than the case above, because z doesn't even
        # have a grad accumulator allocated.
        z = torch.ones(1, requires_grad=True)
        grad_x, grad_z = torch.autograd.grad(x * 2, [x, z], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_z)

        # allow_unused=False, but grads contains None inside, should throw
        with self.assertRaisesRegex(RuntimeError, "Set allow_unused=True"):
            grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=False)

    def test_grad_unreachable_discovery(self):
        # Test that certain nodes are not erroneously executed when an input
        # is unreachable. See #39784
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, x):
                self.fail("This node should not be executed!")

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        (gY,) = torch.autograd.grad(x, (y,), allow_unused=True)
        self.assertIsNone(gY)

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        z = torch.randn(1, requires_grad=True)
        (gY, gZ) = torch.autograd.grad(x + z, (y, z), allow_unused=True)
        self.assertIsNone(gY)
        self.assertIsNotNone(gZ)

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        torch.autograd.backward(x, inputs=(y,))  # allow_unused is implicitly True!
        self.assertIsNone(y.grad)

    def test_grad_batched_grad(self):
        x = torch.randn(2, 2, requires_grad=True)

        out = x.clone()  # Size([2, 2])
        batched_grad = (
            torch.arange(3).expand(2, 2, 3).transpose(0, 2)
        )  # Size([3, 2, 2])
        (grad,) = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
        self.assertEqual(
            grad, torch.arange(3).expand(2, 2, 3).transpose(0, 2).to(dtype=grad.dtype)
        )

        # Detect shape mismatch
        grad_out = torch.ones(2, 2)
        with self.assertRaisesRegex(
            RuntimeError, "If `is_grads_batched=True`, we interpret the first"
        ):
            torch.autograd.grad(
                outputs=out,
                grad_outputs=(grad_out,),
                inputs=(x,),
                is_grads_batched=True,
            )

        # Scalar outputs
        out = x.sum()  # Size([])
        batched_grad = torch.arange(3)  # Size([3])
        (grad,) = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
        self.assertEqual(
            grad, torch.arange(3).expand(2, 2, 3).transpose(0, 2).to(dtype=grad.dtype)
        )

        # We consider scalar and sized-1 to be a mismatch. This is consistent with current non-batched behavior.
        grad_out = torch.ones(2).unsqueeze(1)
        with self.assertRaisesRegex(
            RuntimeError, "If `is_grads_batched=True`, we interpret the first"
        ):
            torch.autograd.grad(
                outputs=out,
                grad_outputs=(grad_out,),
                inputs=(x,),
                is_grads_batched=True,
            )

    def test_hooks(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        y.requires_grad_(True)

        counter = [0]

        def bw_hook(inc, grad):
            self.assertIsInstance(grad, torch.Tensor)
            counter[0] += inc

        z = x**2 + x * 2 + x * y + y
        x.register_hook(lambda *args: bw_hook(0, *args))
        test = z.register_hook(lambda *args: bw_hook(1, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 1)

        test2 = z.register_hook(lambda *args: bw_hook(2, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 4)

        test2.remove()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 5)

        def bw_hook_modify(grad):
            return grad.mul(2)

        test.remove()
        z.register_hook(bw_hook_modify)
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(y.grad, (x + 1) * 2)

        y.register_hook(bw_hook_modify)
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5))
        self.assertEqual(y.grad, (x + 1) * 4)

    def _get_mul2(self, use_custom_function):
        if use_custom_function:

            class Mul2(Function):
                @staticmethod
                def forward(ctx, x):
                    return x * 2

                @staticmethod
                def backward(ctx, gO):
                    return gO * 2

            return Mul2.apply
        else:
            return lambda x: x * 2

    def test_grad_fn_prehooks(self):
        for use_custom_function in (True, False):
            mul2 = self._get_mul2(use_custom_function)

            a = torch.tensor([1.0], requires_grad=True)
            b = mul2(a)

            post_counter = [0]
            pre_counter = [0]

            def posthook(grad_input, grad_output):
                self.assertEqual(pre_counter[0], 3)
                self.assertTrue(torch.allclose(grad_output[0], torch.ones(1) * 8))
                self.assertTrue(torch.allclose(grad_input[0], torch.ones(1) * 16))
                post_counter[0] += 1
                return grad_input

            def prehook(grad_output):
                pre_counter[0] += 1
                return (grad_output[0] * 2,)

            # register posthook x 2
            b.grad_fn.register_hook(posthook)
            b.grad_fn.register_hook(posthook)
            # register prehook x 3
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(lambda x: None)
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(lambda x: x)
            b.grad_fn.register_prehook(lambda x: None)

            b.sum().backward()

            self.assertEqual(post_counter[0], 2)
            self.assertEqual(pre_counter[0], 3)

            # Return None
            a = torch.rand(3, 3, requires_grad=True)
            b = mul2(a)

            def prehook(grad_output):
                pre_counter[0] += 1
                return None

            b.grad_fn.register_prehook(prehook)
            b.sum().backward()
            self.assertEqual(pre_counter[0], 4)
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))

    def test_grad_fn_prehooks_multiple_outputs(self):
        # Compute gradients without hooks
        b = torch.rand(3, 3, requires_grad=True)
        var, mean = torch.var_mean(b, dim=0)
        (var + mean).sum().backward()

        # Compute gradients with hooks
        a = b.detach().requires_grad_()
        counter = [0]

        def prehook(grad_output):
            gvar, gmean = grad_output
            counter[0] += 1
            return (gvar * 2, gmean * 2)

        var, mean = torch.var_mean(a, dim=0)
        mean.grad_fn.register_prehook(prehook)
        (var + mean).sum().backward()

        self.assertEqual(counter[0], 1)
        # Compare
        self.assertTrue(torch.allclose(a.grad, b.grad * 2))

        # Test with custom Function
        class DoubleMul2(Function):
            @staticmethod
            def forward(ctx, x, a, y):
                ctx.a = a
                return a * x * 2, a, a * y * 2

            @staticmethod
            def backward(ctx, g1, _a, g2):
                return ctx.a * g1 * 2, None, ctx.a * g2 * 2

        counter = [0]

        def prehook(grad_output):
            g1, ga, g2 = grad_output
            self.assertIsNone(ga)
            counter[0] += 1
            return (g1 * 2, None, g2 * 2)

        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        k = 3
        c, _, d = DoubleMul2.apply(a, k, b)
        c.grad_fn.register_prehook(prehook)
        (c + d).sum().backward()

        self.assertEqual(counter[0], 1)
        self.assertTrue(torch.allclose(a.grad, torch.ones(1) * 4 * k))
        self.assertTrue(torch.allclose(b.grad, torch.ones(1) * 4 * k))

    def test_grad_fn_prehooks_remove_hooks(self):
        for use_custom_function in (True, False):
            mul2 = self._get_mul2(use_custom_function)

            # Simply remove hooks

            a = torch.rand(3, 3, requires_grad=True)
            b = mul2(a)
            counter = [0]

            def prehook(grad_output):
                counter[0] += 1
                return None

            handle = b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(prehook)
            handle.remove()
            b.sum().backward()
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))
            self.assertEqual(counter[0], 1)

            # Remove hooks during backward
            a = torch.rand(3, 3, requires_grad=True)
            b = mul2(a)
            counter = [0]

            def prehook1(grad_output):
                handle2.remove()
                # Remove hook that is already removed is OK
                handle3.remove()
                return None

            def prehook2(grad_output):
                counter[0] += 1
                return None

            # Hooks that registered first run first
            b.grad_fn.register_prehook(prehook1)
            handle2 = b.grad_fn.register_prehook(prehook2)
            handle3 = b.grad_fn.register_prehook(prehook2)
            handle3.remove()
            b.sum().backward()
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))
            self.assertEqual(counter[0], 1)

    def test_node_post_hook_registered_during_unpack_hook(self):
        """
        Test that post hooks registered during one of the node's
        unpack hooks are properly restricted and will run properly.
        """
        test_case = self

        class RegisterPostNodeHook(torch.autograd.graph.saved_tensors_hooks):
            def __init__(self) -> None:
                def pack_tensor(tensor: torch.Tensor) -> torch.Tensor:
                    return tensor

                def unpack_tensor(tensor: torch.Tensor) -> torch.Tensor:
                    node = torch._C._current_autograd_node()

                    def hook(outputs, inputs):
                        # Assert that inputs passed in are None
                        test_case.assertTrue(all(i is None for i in inputs))
                        halved_outputs = tuple(
                            o / 2.0 if o is not None else None for o in outputs
                        )
                        return halved_outputs

                    node.register_hook(hook)
                    return tensor

                super().__init__(pack_tensor, unpack_tensor)

        a = torch.rand(3, 3, requires_grad=True)

        def model():
            var, mean = torch.var_mean(a, dim=0)
            loss = (var + mean).sum()
            loss.backward()

        model()
        ref_grad = a.grad.clone()

        with RegisterPostNodeHook():
            model()

        # Verify that the post hook got called and the grad propagation worked
        self.assertEqual(ref_grad / 2.0 + ref_grad, a.grad)

    def test_hooks_cpp(self):
        # Tests hooks for autograd function implemented in C++
        bn = torch.nn.BatchNorm1d(5, affine=False)
        bn.double()
        bn.eval()

        counter = [0]

        def bw_hook(grad):
            counter[0] += 1
            return grad * 2

        x = torch.ones(5, 5, dtype=torch.double, requires_grad=True)
        z = bn(x)
        z.register_hook(bw_hook)
        z.sum().backward()

        self.assertEqual(counter[0], 1, msg="bw_hook not called")
        self.assertEqual(
            x.grad, torch.ones(5, 5, dtype=torch.double) * 2, atol=1e-5, rtol=0
        )

    def test_hook_none(self):
        # WARNING: this is a test for autograd internals.
        # You should never have to use such things in your code.
        class NoneGradientFunction(Function):
            @staticmethod
            def forward(ctx, x, y):
                if not ctx.needs_input_grad[0]:
                    raise AssertionError("expected needs_input_grad[0] to be True")
                if ctx.needs_input_grad[1]:
                    raise AssertionError("expected needs_input_grad[1] to be False")
                return x, y

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                return grad_x, None

        was_called = [False]

        def hook(grad):
            self.assertIsNotNone(grad)
            was_called[0] = True

        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5)
        rx, ry = NoneGradientFunction.apply(x, y)
        rx.register_hook(hook)
        ry.register_hook(hook)
        sum(rx, ry).sum().backward()
        self.assertTrue(was_called[0])

    def test_retain_grad(self):
        input = torch.rand(1, 3, requires_grad=True)
        h1 = input * 3
        out = (h1 * h1).sum()

        # It should be possible to call retain_grad() multiple times
        h1.retain_grad()
        h1.retain_grad()

        # Gradient should be accumulated
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 2, h1.grad)
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 4, h1.grad)

        with torch.no_grad():
            input.grad.zero_()
        # It should be a no-op for leaves
        input.retain_grad()
        input.retain_grad()
        out.backward()
        self.assertEqual(input * 18, input.grad)

    # NB: See test/cpp/api/autograd.cpp for more tests on the interaction between
    #     retains_grad and hooks in cpp
    def test_retain_grad_inplace(self):
        a = torch.tensor([1.0], requires_grad=True).clone()
        a.retain_grad()
        a.mul_(2)
        a.sum().backward()
        self.assertEqual(a.grad, torch.tensor([1.0]))

        a = torch.tensor([1.0], requires_grad=True).clone()
        a.retain_grad()
        # Inplace multiple times is OK
        a.mul_(2)
        a.mul_(2)
        a.sum().backward()
        self.assertEqual(a.grad, torch.tensor([1.0]))

        # When in-place over view is done, the retains_grad hooks should be
        # moved from base's original grad_fn to the copyslices node.
        x = torch.tensor([1.0], requires_grad=True).clone()
        x.retain_grad()
        x_view = x[:]
        x_view *= 2
        x *= 2
        x.sum().backward()
        # The grad is 1, not 4, because we are computing grad wrt the latest
        # version of x.
        self.assertEqual(a.grad, torch.tensor([1.0]))

        # If the base did not originally require grad, there should be no hook
        # to move. Make sure this case runs without error.
        x = torch.zeros(4)
        y = x.view(2, 2)
        y.add_(torch.randn(2, 2, requires_grad=True))

    def test_retains_grad_inplace_multiple_outputs(self):
        class DoubleMul(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2, x * 3

            @staticmethod
            def backward(ctx, g1, g2):
                return g1 * 2 + g2 * 3

        var_mean = partial(torch.var_mean, dim=0)

        for fn in (DoubleMul.apply, var_mean):
            b = torch.rand(3, 3, requires_grad=True)
            var, mean = fn(b)
            var.retain_grad()
            mean.retain_grad()
            # node has two retains_grad hooks
            var.mul_(2)
            # the retain_grad hook multi-output node refers should now be a nullptr
            (var + mean).sum().backward()
            gvar = var.grad
            gmean = mean.grad

            a = b.detach().requires_grad_(True)
            var, mean = fn(a)
            var.mul_(2)
            out = (var + mean).sum()
            gvar_expected, gmean_expected = torch.autograd.grad(out, inputs=(var, mean))
            self.assertTrue(torch.allclose(gvar, gvar_expected))
            self.assertTrue(torch.allclose(gmean, gmean_expected))

    def test_retain_grad_inplace_over_view(self):
        base = torch.tensor([1.0], requires_grad=True).clone()
        view = base[:]
        view2 = base[:]
        view.retain_grad()
        view2.retain_grad()
        view.mul_(2)
        (view + view2).sum().backward()

        # The old grad_fn, slice, wouldn't be part of the graph during backward
        # so if the retains grad were not properly updated to the new grad_fn,
        # the grad would still be None
        self.assertEqual(view.grad, view2.grad)
        self.assertEqual(view.grad, torch.tensor([1.0]))

    def test_tensor_hooks_inplace(self):
        # Check that the second hook gets registered to the new version of tensor
        count1 = [0]
        count2 = [0]

        def fn1(grad):
            count1[0] += 1
            # x2 from mul, x2 from fn2
            self.assertEqual(grad, torch.tensor([4.0]))
            return grad * 2

        def fn2(grad):
            count2[0] += 1
            self.assertEqual(grad, torch.tensor([1.0]))
            return grad * 2

        a = torch.tensor([1.0], requires_grad=True)
        b = a.clone()
        b.register_hook(fn1)
        b.mul_(2)
        b.register_hook(fn2)
        b.sum().backward()
        self.assertEqual(count1[0], 1)
        self.assertEqual(count2[0], 1)
        self.assertEqual(a.grad, torch.tensor([8.0]))

        count3 = [0]

        def fn3(grad):
            count3[0] += 1
            self.assertEqual(grad, torch.tensor([4.0]))
            return grad * 2

        a = torch.tensor([1.0], requires_grad=True)
        b = a.clone()
        b.register_hook(fn3)
        # Inplace multiple times is OK
        b.mul_(2)
        b.mul_(2)
        b.sum().backward()
        self.assertEqual(count1[0], 1)
        self.assertEqual(a.grad, torch.tensor([8.0]))

    def test_tensor_hooks_inplace_multiple_outputs(self):
        class DoubleMul(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2, x * 3

            @staticmethod
            def backward(ctx, g1, g2):
                return g1 * 2 + g2 * 3

        var_mean = partial(torch.var_mean, dim=0)

        for fn in (DoubleMul.apply, var_mean):
            counts = [0, 0, 0]

            def fn0(grad):
                counts[0] += 1
                self.assertEqual(grad, torch.ones_like(out1) * 2)

            def fn1(grad):
                counts[1] += 1
                self.assertEqual(grad, torch.ones_like(out1) * 3)

            def fn2(grad):
                counts[2] += 1
                self.assertEqual(grad, torch.ones_like(out1))

            b = torch.rand(3, 3, requires_grad=True)
            out1, out2 = fn(b)
            h1 = out1.register_hook(fn0)
            h2 = out2.register_hook(fn1)
            # node refers to two hook dicts
            # out1 no longer no longer points to its old hook dict
            out1.mul_(2)
            # fn2 is registered to out1's new hook dict
            h3 = out1.register_hook(fn2)
            (out1 + out2 * 3).sum().backward()
            self.assertEqual(counts, [1, 1, 1])

            # Avoid leaking memory
            h1.remove()
            h2.remove()
            h3.remove()

    def test_tensor_hooks_inplace_over_view(self):
        # There might be a better UX here, but this is the way it is now
        count = [0]

        def fn0(grad):
            self.fail()

        def fn1(grad):
            self.fail()

        def fn2(grad):
            count[0] += 1
            self.assertEqual(grad, torch.tensor([1.0]))

        base = torch.tensor([1.0], requires_grad=True).clone()
        view = base[:]
        view2 = base[:]
        view.register_hook(fn0)
        view2.register_hook(fn1)
        view.mul_(2)
        # We need to explicitly trigger an update to view to update its grad_fn
        view2.grad_fn
        view2.register_hook(fn2)
        (view + view2).sum().backward()
        # The hooks originally registered to view are not fired, one must explicitly
        # trigger an update to the view's grad_fn, and then register a new hook
        self.assertEqual(count[0], 1)

    def test_retain_grad_cycle(self):
        x = torch.ones(5, 5, requires_grad=True)

        def run_test():
            y = x * 2
            y.retain_grad()

            return y / 2, torch._C._WeakTensorRef(y)

        z, ref = run_test()
        self.assertTrue(ref.expired())
        z.sum().backward()

    def test_backward(self):
        v = torch.randn(5, 5, requires_grad=True)
        x = torch.randn(5, 5, requires_grad=True)
        y = (torch.rand(5, 5) + 0.1).requires_grad_(True)
        z = torch.randn(5, 5, requires_grad=True)
        grad_output = torch.randn(5, 5)

        v.backward(grad_output)
        self.assertEqual(v.grad, grad_output)

        a = x + (y * z) + 4 * z**2 * x / y
        a.backward(grad_output)
        x_grad = 4 * z.pow(2) / y + 1
        y_grad = z - 4 * x * z.pow(2) / y.pow(2)
        z_grad = 8 * x * z / y + y
        self.assertEqual(x.grad, x_grad * grad_output)
        self.assertEqual(y.grad, y_grad * grad_output)
        self.assertEqual(z.grad, z_grad * grad_output)

    def test_to_sparse_backward(self):
        to_attr_names = (
            "to_dense",
            "to_sparse",
            "to_sparse_csr",
            "to_sparse_csc",
            "to_sparse_bsr",
            "to_sparse_bsc",
        )
        to_params = ((), (), (), (), (2,), (2,))
        to_attr_names_params = dict(zip(to_attr_names, to_params))

        def check_inversion_possible(
            t, layout1, layout1_params, layout2, layout2_params
        ):
            l = (layout1, layout2)
            p = (layout1_params, layout2_params)
            for l1, l2, p1, p2 in ((*l, *p), (*l[::-1], *p[::-1])):
                try:
                    to_l1 = getattr(t, l1)(*p1)
                    to_l2 = getattr(to_l1, l2)(*p2)
                except RuntimeError:
                    return False

            return True

        self_strided = torch.rand(4, 4, dtype=torch.double) + 1
        grad_strided = torch.rand(4, 4, dtype=torch.double) + 1

        for from_to_attr in to_attr_names:
            from_params = to_attr_names_params[from_to_attr]
            self_from = getattr(self_strided, from_to_attr)(
                *from_params
            ).requires_grad_(True)

            for to_to_attr in to_attr_names[1:]:
                to_params = to_attr_names_params[to_to_attr]

                if check_inversion_possible(
                    self_strided, from_to_attr, from_params, to_to_attr, to_params
                ):
                    self_to = getattr(self_from, to_to_attr)(*to_params)
                    grad_to = getattr(grad_strided, to_to_attr)(*to_params)

                    # No gradcheck support for BSR/BSC, so the grads are checked explicitly
                    grad_res = torch.autograd.grad(self_to, self_from, grad_to)[0]

                    self.assertEqual(grad_res.layout, self_from.layout)
                    self.assertEqual(grad_res.to_dense(), grad_strided)

    def test_sparse_mm_backward(self):
        size = (3, 3)

        mm_test_cases = product(*(([False, True],) * 4))

        for a_req_grad, a_is_sparse, b_req_grad, b_is_sparse in mm_test_cases:
            # We should only be testing cases with sparse inputs, and at least one
            # input needs to require grad so we can call a backward pass
            if not ((a_is_sparse or b_is_sparse) and (a_req_grad or b_req_grad)):
                continue
            a = torch.randn(size)
            if a_is_sparse:
                # detaching as `a` needs to be a leaf
                a = a.to_sparse().detach()
            b = torch.randn(size)
            if b_is_sparse:
                # detaching as `b` needs to be a leaf
                b = b.to_sparse().detach()

            a = a.requires_grad_(a_req_grad)
            b = b.requires_grad_(b_req_grad)

            r = a.mm(b)
            s = r.sum().backward()
            a_grad = None if a.grad is None else a.grad.detach().clone()
            b_grad = None if b.grad is None else b.grad.detach().clone()

            # Redo with only dense tensors
            a = (
                (a.to_dense() if a.is_sparse else a)
                .clone()
                .detach()
                .requires_grad_(a_req_grad)
            )
            b = (
                (b.to_dense() if b.is_sparse else b)
                .clone()
                .detach()
                .requires_grad_(b_req_grad)
            )

            r = a.mm(b)
            r.sum().backward()

            self.assertEqual(a_grad, a.grad)
            self.assertEqual(b_grad, b.grad)

    def test_multi_backward(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        q = torch.randn(5, 5, requires_grad=True)

        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        q2 = q * 2
        z = x + y + q2
        c = a * b + q2
        grad_z = torch.randn(5, 5)
        grad_c = torch.randn(5, 5)
        torch.autograd.backward([z, c], [grad_z, grad_c])

        self.assertEqual(x.grad, grad_z)
        self.assertEqual(y.grad, grad_z)
        self.assertEqual(a.grad, grad_c * b)
        self.assertEqual(b.grad, grad_c * a)
        self.assertEqual(q.grad, (grad_c + grad_z) * 2)

    def test_multi_backward_no_grad(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=False)

        z = x + y
        q = y * 2

        # NB: we currently raise an exception if any arguments to backwards
        # have requires_grad=False and don't have a grad_fn. We may want to
        # relax that check to a warning.
        def call_backwards():
            torch.autograd.backward([z, q], [torch.ones(5, 5), torch.ones(5, 5)])

        self.assertRaises(RuntimeError, call_backwards)

    def test_backward_with_inputs(self):
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        def fn():
            return x**2 + y * x + y**2

        gradient = torch.ones(2, 2)
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y

        @torch.no_grad()
        def reset_grad():
            x.grad.zero_()
            y.grad.zero_()

        torch.autograd.backward(fn(), gradient, inputs=[x, y])
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(y.grad, y_grad_expected)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=[x])
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(y.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=[y])
        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=y)
        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        self.assertRaisesRegex(
            RuntimeError,
            "cannot be empty",
            lambda: torch.autograd.backward(fn(), gradient, inputs=[]),
        )

    def test_backward_with_scalar_input(self):
        x = torch.randn([], dtype=torch.double, requires_grad=True)
        out = x**2
        out.backward(inputs=x)
        self.assertEqual(x.grad, 2 * x)

    def test_backward_with_nonleaf_inputs(self):
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        x_nonleaf = x * 1
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        out = x_nonleaf**2 + y * x_nonleaf + y**2

        out.backward(
            torch.ones(2, 2, dtype=torch.double),
            create_graph=True,
            inputs=[x, y, x_nonleaf],
        )
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y
        x_non_leaf_expected = 2 * x_nonleaf + y

        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(x_nonleaf.grad, x_non_leaf_expected)

        # backward doesn't have an allow_unused flag, so the behavior of backward
        # when variable is not part of the graph is as if allow_used were true
        # x.grad will simply be None.
        out.backward(
            torch.ones(2, 2, dtype=torch.double), create_graph=True, inputs=[z]
        )
        self.assertIsNone(z.grad)

        # Avoid leaking memory
        x.grad = None
        y.grad = None
        x_nonleaf.grad = None

    def test_dependent_backward(self):
        x = torch.randn(10, requires_grad=True)
        y = x**2
        z = y**3

        go_y = torch.randn(10)
        go_z = torch.randn(10)
        torch.autograd.backward([y, z], [go_y, go_z])

        xd = x
        self.assertEqual(x.grad, 2 * xd * go_y + 6 * xd.pow(5) * go_z)

    def test_save_output_nr(self):
        x = torch.randn(10, requires_grad=True)

        class MultiOutputFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x[:5], x[5:]

            @staticmethod
            def backward(ctx, *grad):
                return torch.cat(grad)

        a, b = MultiOutputFn.apply(x)
        self.assertEqual(b.output_nr, 1)

        class TestFn(Function):
            @staticmethod
            def forward(ctx, b):
                ctx.save_for_backward(b)
                return b * 2

            @staticmethod
            def backward(ctx, grad_b):
                (b,) = ctx.saved_tensors
                self.assertEqual(b.output_nr, 1)

        TestFn.apply(b).sum().backward()

    def test_first_grad_fn_access_in_no_grad_mode(self):
        a = torch.tensor([1 + 1j], requires_grad=True).clone()
        v = a.real
        a.add_(1)
        with torch.autograd.grad_mode.no_grad():
            v.grad_fn

    @skipIfTorchDynamo("too slow")
    def test_free_deep_graph(self):
        def scope():
            depth = 150000
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # build a "chain" computation graph
            for _ in range(depth):
                y = y + y * 0.000001

            # graph deletion occurs when the above locals go out of scope.
            # In this case `del y` will trigger it but it's easier to leave
            # it to Python to delete the locals.

        # Should not stack overflow
        scope()

    @skipIfTorchDynamo("too slow")
    def test_free_deep_graph_complicated(self):
        def scope():
            depth = 100000
            randchoice = torch.randint(2, [depth, 2])
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # Hold the two previous values
            prev_values = [None, None]

            # Build a "chain with skip connections" graph
            for _ in range(depth):
                prev_tensors = [
                    tensor for tensor in prev_values[:-1] if tensor is not None
                ]
                prev_values.append(y)
                prev_values.pop(0)

                # Definitely pick one tensor to add
                y += y * 0.000001

                # Possibly add other tensors
                nprev = len(prev_tensors)
                if nprev == 2:
                    y += randchoice[depth].mul(torch.cat(prev_tensors)).sum()

            # graph deletion occurs when the above locals go out of scope.

        # Should not stack overflow
        scope()

    @skipIfTorchDynamo("too slow")
    def test_free_deep_graph_pyfunction(self):
        class MyOp(Function):
            @staticmethod
            def forward(ctx, tensor1, tensor2):
                return tensor1 + tensor2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        def scope():
            depth = 150000
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # build deeply nested computation graph
            for _ in range(depth):
                y = MyOp.apply(y, y)

            # graph deletion occurs when the above locals go out of scope.

        # Should not stack overflow
        scope()

    def test_no_unnecessary_save(self):
        # If we kept x in the derivative Function of x * 2 we would
        # get an error in the backward that would complain that we've
        # modified x, which was needed for gradient computation.
        # Since we should elide unnecessary saves, this test should pass.
        mu = torch.ones(1, requires_grad=True)
        x = torch.empty(1)
        loss = 0
        for i in range(3):
            x.detach_()
            x.copy_(mu + i)
            ft = torch.tensor([float(i)])
            multiplied = x * ft
            s = multiplied.sum()
            loss += s
        loss.backward()

    def test_no_grad(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        with torch.no_grad():
            w = x + y

        def adder(x, y):
            return x + y

        adders = [torch.no_grad()(adder), torch.no_grad(adder)]

        for adder in adders:
            z = adder(x, y)

            self.assertFalse(w.requires_grad)
            self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))
            self.assertIsNone(w.grad_fn)
            self.assertFalse(z.requires_grad)
            self.assertRaises(RuntimeError, lambda: z.backward(torch.ones(5, 5)))
            self.assertIsNone(z.grad_fn)

        # test nested decorator and with-statement on no_grad
        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            w = adder(x, y)
            self.assertFalse(torch.is_grad_enabled())

    def test_enable_grad_decorator_no_paren(self):
        x = torch.ones(1, requires_grad=True)

        @torch.enable_grad
        def doubler(x):
            return x * 2

        with torch.no_grad():
            z = doubler(x)
        self.assertTrue(z.requires_grad)

    def test_set_grad_generator_functions(self):
        @torch.no_grad()
        def gen_no_grad():
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), False)
                yield i

        with torch.enable_grad():
            for _ in gen_no_grad():
                self.assertEqual(torch.is_grad_enabled(), True)

        @torch.enable_grad()
        def gen_enable_grad():
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), True)
                yield i

        with torch.no_grad():
            for _ in gen_enable_grad():
                self.assertEqual(torch.is_grad_enabled(), False)

    def test_set_grad_generator_functions_recursive(self):
        # enable_grad_decorator_recursive and no_grad_decorator_recursive call each other
        # recursively, to ensure that the decorators preserve the caller's setting
        @torch.enable_grad()
        def enable_grad_decorator_recursive(depth):
            self.assertTrue(torch.is_grad_enabled())
            if depth > 0:
                no_grad_decorator_recursive(depth - 1)
                self.assertTrue(torch.is_grad_enabled())

        @torch.no_grad()
        def no_grad_decorator_recursive(depth):
            self.assertFalse(torch.is_grad_enabled())
            if depth > 0:
                enable_grad_decorator_recursive(depth - 1)
                self.assertFalse(torch.is_grad_enabled())

        # enable_grad_context_manager_recursive and no_grad_context_manager_recursive call
        # each other recursively, to ensure that the decorators preserve the caller's setting
        def enable_grad_context_manager_recursive(depth):
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())
                if depth > 0:
                    no_grad_context_manager_recursive(depth - 1)
                    self.assertTrue(torch.is_grad_enabled())

        def no_grad_context_manager_recursive(depth):
            with torch.no_grad():
                self.assertFalse(torch.is_grad_enabled())
                if depth > 0:
                    enable_grad_context_manager_recursive(depth - 1)
                    self.assertFalse(torch.is_grad_enabled())

        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertTrue(torch.is_grad_enabled())

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertFalse(torch.is_grad_enabled())

    def test_set_grad_coroutines(self):
        @torch.no_grad()
        def coro_no_grad(n=10):
            self.assertFalse(torch.is_grad_enabled())
            for i in range(n):
                self.assertFalse(torch.is_grad_enabled())
                r = yield i
                self.assertFalse(torch.is_grad_enabled())
                self.assertEqual(i, r)
            self.assertFalse(torch.is_grad_enabled())

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            self.assertTrue(torch.is_grad_enabled())
            for i in range(n):
                self.assertTrue(torch.is_grad_enabled())
                r = yield i
                self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(i, r)
            self.assertTrue(torch.is_grad_enabled())

        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            coro, r = coro_no_grad(), None
            try:
                while True:
                    self.assertTrue(torch.is_grad_enabled())
                    r = coro.send(r)
                    self.assertTrue(torch.is_grad_enabled())

            except StopIteration:
                pass

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            coro, r = coro_enable_grad(), None
            try:
                while True:
                    self.assertFalse(torch.is_grad_enabled())
                    r = coro.send(r)
                    self.assertFalse(torch.is_grad_enabled())

            except StopIteration:
                pass

    def test_set_grad_coroutines_benign_exceptions(self):
        class RecoverableException(Exception):
            pass

        @torch.no_grad()
        def coro_no_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except RecoverableException:
                    self.assertFalse(torch.is_grad_enabled())
                    has_raised = True

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except RecoverableException:
                    self.assertTrue(torch.is_grad_enabled())
                    has_raised = True

        with torch.enable_grad():
            coro = coro_no_grad()
            if 0 != next(coro):
                raise AssertionError("expected next(coro) == 0")
            try:
                while True:
                    r = coro.throw(RecoverableException)
                    self.assertLess(r, 0)

            except StopIteration:
                pass

        with torch.no_grad():
            coro = coro_enable_grad()
            if 0 != next(coro):
                raise AssertionError("expected next(coro) == 0")
            try:
                while True:
                    r = coro.throw(RecoverableException)
                    self.assertLess(r, 0)

            except StopIteration:
                pass

    def test_set_grad_coroutines_critical_exceptions(self):
        class UnrecoverableException(Exception):
            pass

        class SecondaryException(Exception):
            pass

        @torch.no_grad()
        def coro_no_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except UnrecoverableException:
                    self.assertFalse(torch.is_grad_enabled())
                    raise SecondaryException from None

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except UnrecoverableException:
                    self.assertTrue(torch.is_grad_enabled())
                    raise SecondaryException from None

        with torch.enable_grad():
            coro = coro_no_grad()
            if 0 != next(coro):
                raise AssertionError("expected next(coro) == 0")
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)

        with torch.no_grad():
            coro = coro_enable_grad()
            if 0 != next(coro):
                raise AssertionError("expected next(coro) == 0")
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)

    def test_set_grad_coroutines_exit(self):
        @torch.no_grad()
        def coro_no_grad(state):
            for i in range(10):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield i

                except GeneratorExit:
                    self.assertFalse(torch.is_grad_enabled())
                    state.add("GeneratorExit")
                    raise

        @torch.enable_grad()
        def coro_enable_grad(state):
            for i in range(10):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield i

                except GeneratorExit:
                    self.assertTrue(torch.is_grad_enabled())
                    state.add("GeneratorExit")
                    raise

        state = set()
        with torch.enable_grad():
            coro = coro_no_grad(state)
            for _ in range(5):
                next(coro)

            coro.close()
        self.assertTrue("GeneratorExit" in state)

        state = set()
        with torch.no_grad():
            coro = coro_enable_grad(state)
            for _ in range(5):
                next(coro)

            coro.close()
        self.assertTrue("GeneratorExit" in state)

    def test_no_grad_python_function(self):
        """Python Functions should respect grad mode."""
        x = torch.ones(5, 5, requires_grad=True)

        class MyOp(Function):
            @staticmethod
            def forward(self, x):
                return x + 1

            @staticmethod
            def backward(self, dy):
                return dy

        with torch.no_grad():
            y = MyOp.apply(x)
        self.assertFalse(y.requires_grad)

    def test_indexing(self):
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        def compare(x, y, idx, indexed_tensor, indexed_var):
            indexed_var_t = indexed_var.data
            if not isinstance(indexed_tensor, torch.Tensor):
                indexed_var_t = indexed_var_t[0]
            self.assertEqual(indexed_tensor, indexed_var_t)

            indexed_var.sum().backward()
            expected_grad = torch.empty(x.size()).fill_(0)
            expected_grad[idx] = 1
            self.assertEqual(y.grad, expected_grad)

        def check_index(x, y, idx):
            if y.grad is not None:
                with torch.no_grad():
                    y.grad.zero_()
            indexed_tensor = x[idx]
            indexed_var = y[idx]
            compare(x, y, idx, indexed_tensor, indexed_var)

        check_index(x, y, 1)
        check_index(x, y, (1, 1))
        check_index(x, y, slice(1, None))
        check_index(x, y, slice(None, 2))
        check_index(x, y, (slice(None, 2), 2))
        check_index(x, y, (slice(1, 2), 2))
        check_index(x, y, (1, slice(2, None)))
        check_index(x, y, (slice(None, None), slice(2, None)))
        check_index(x, y, torch.LongTensor([0, 2]))
        check_index(x, y, torch.rand(4, 4).bernoulli().bool())
        check_index(x, y, (Ellipsis, slice(2, None)))
        check_index(x, y, ([0], [0]))
        check_index(x, y, ([1, 2, 3], [0]))
        check_index(x, y, ([1, 2], [2, 1]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, ((slice(None), [2, 3])))
        check_index(x, y, (([2, 3], slice(None))))

        # advanced indexing, with less dim, or ellipsis
        check_index(x, y, ([0]))
        check_index(x, y, ([0],))

        x = torch.arange(1.0, 49).view(4, 3, 4)
        y = Variable(x, requires_grad=True)

        check_index(x, y, (slice(None), [0], [0]))
        check_index(x, y, ([0], [0], slice(None)))
        check_index(x, y, (slice(None), [0, 1, 2], [0]))
        check_index(x, y, ([0, 1, 2], [0], slice(None)))
        check_index(x, y, (slice(None), [1, 2], [2, 1]))
        check_index(x, y, ([1, 2], [2, 1], slice(None)))
        check_index(x, y, (slice(None), [[1, 2], [2, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 2]], slice(None)))
        check_index(x, y, (slice(None), slice(None), [2, 1]))
        check_index(x, y, (slice(None), [2, 1], slice(None)))
        check_index(x, y, ([2, 1], slice(None), slice(None)))

        # advanced indexing, with less dim, or ellipsis
        check_index(x, y, ([0],))
        check_index(x, y, ([0], slice(None)))
        check_index(x, y, ([0], Ellipsis))
        check_index(x, y, ([1, 2], [0, 1]))
        check_index(x, y, ([1, 2], [0, 1], Ellipsis))
        check_index(x, y, (Ellipsis, [1, 2], [0, 1]))

        # advanced indexing, with a tensor wrapped in a variable
        z = torch.LongTensor([0, 1])
        zv = Variable(z, requires_grad=False)
        seq = (z, Ellipsis)
        seqv = (zv, Ellipsis)

        if y.grad is not None:
            with torch.no_grad():
                y.grad.zero_()
        indexed_tensor = x[seq]
        indexed_var = y[seqv]
        compare(x, y, seq, indexed_tensor, indexed_var)

    def test_indexing_duplicates(self):
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        idx = torch.LongTensor([1, 1, 3, 2, 1, 2])
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4)
        for i in idx:
            expected_grad[i] += 1
        self.assertEqual(y.grad, expected_grad)

        # with advanced indexing
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        idx = ([1, 1, 3, 2, 1, 2], [0])
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4)
        for i in idx[0]:
            for j in idx[1]:
                expected_grad[i][j] += 1

        self.assertEqual(y.grad, expected_grad)

        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)
        idx = ([[1, 2], [0, 0]], [[0, 1], [1, 1]])
        y[idx].sum().backward()
        expected_grad = torch.tensor(
            [
                [0.0, 2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.assertEqual(y.grad, expected_grad)

        x = torch.arange(1.0, 65).view(4, 4, 4)
        y = Variable(x, requires_grad=True)

        idx = ([1, 1, 1], slice(None), slice(None))
        y[idx].sum().backward()
        expected_grad = torch.empty(4, 4, 4).zero_()
        expected_grad[1].fill_(3)
        self.assertEqual(y.grad, expected_grad)

    def test_index_backward_does_not_save_tensor(self):
        # Example from https://github.com/pytorch/pytorch/issues/24853.
        # if `index(tensor, indices)` saves `tensor` for backwards, then it will
        # trigger a version check on `tensor` during the backward pass, which
        # will cause the following code to error because `tensor` gets modified
        # by the indexing line.
        a = torch.tensor([1.0, 0, 0])
        b = torch.zeros(3, requires_grad=True)
        tensor = b + 0
        tensor[a != 0] = tensor[a != 0]
        tensor.backward(torch.zeros_like(tensor))

    def test_volatile_deprecated(self):
        v = torch.autograd.torch.randn(3, 3)
        with warnings.catch_warnings(record=True) as w:
            self.assertFalse(v.volatile)
        self.assertIn("volatile", str(w[0].message))

    def test_saved_variables_deprecated(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, tensor1, tensor2):
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + tensor2

            @staticmethod
            def backward(ctx, grad_output):
                var1, var2 = ctx.saved_variables
                return (grad_output, grad_output)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            x = torch.randn((3, 3), requires_grad=True)
            y = torch.randn((3, 3), requires_grad=True)
            MyFunction.apply(x, y).sum().backward()

            has_deprecated = (
                "deprecated" in str(warn) and "saved_variables" in str(warn)
                for warn in warns
            )
            has_deprecated = reduce(lambda x, y: x or y, has_deprecated)
            self.assertTrue(has_deprecated)

    def test_requires_grad(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        z = torch.randn(5, 5, requires_grad=True)
        a = x + y
        self.assertFalse(a.requires_grad)
        b = a + z
        self.assertTrue(b.requires_grad)

        def error():
            raise RuntimeError

        # Make sure backward isn't called on these
        a._backward_hooks = OrderedDict()
        x._backward_hooks = OrderedDict()
        y._backward_hooks = OrderedDict()
        a._backward_hooks["test"] = error
        x._backward_hooks["test"] = error
        y._backward_hooks["test"] = error
        b.backward(torch.ones(5, 5))

    def test_requires_grad_(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)
        self.assertIs(x, x.requires_grad_())
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_())
        self.assertTrue(y.requires_grad)
        self.assertIs(x, x.requires_grad_(True))
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_(True))
        self.assertTrue(y.requires_grad)
        z = x * y
        self.assertRaises(RuntimeError, lambda: z.requires_grad_(False))
        self.assertIs(z, z.requires_grad_())
        self.assertTrue(z.requires_grad)
        self.assertIs(z, z.requires_grad_(True))
        self.assertTrue(z.requires_grad)

        self.assertIs(x, x.requires_grad_(False))
        self.assertFalse(x.requires_grad)
        self.assertIs(y, y.requires_grad_(False))
        self.assertFalse(y.requires_grad)

    def test_requires_grad_inplace(self):
        a = torch.randn(5, 5)
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)

        # non-leaf
        a = torch.randn(5, 5) + 0
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)

    def test_no_requires_grad_inplace(self):
        # basic case, should be able to modify inplace while requires_grad is False
        a = torch.randn(2, 3)
        a.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad, torch.ones(2, 3))

        # same but with a view
        a = torch.randn(2, 3)
        b = a[:]
        b.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad, torch.ones(2, 3))

        # should fail if requires_grad = True when we modify inplace
        a = torch.randn(2, 3)
        b = a[:]
        a.requires_grad = True
        with self.assertRaises(RuntimeError):
            a.add_(5)
        with self.assertRaises(RuntimeError):
            b.add_(5)

    def test_attribute_deletion(self):
        x = torch.randn((5, 5), requires_grad=True)
        del x.grad
        self.assertIsNone(x.grad)
        with self.assertRaises(RuntimeError):
            del x.data
        with self.assertRaises(TypeError):
            x.data = None
        with self.assertRaises(RuntimeError):
            del x.requires_grad
        with self.assertRaises(RuntimeError):
            del x._grad_fn
        with self.assertRaises(RuntimeError):
            del x._backward_hooks

    def test_duplicate_backward_root(self):
        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        x = a * b
        grad_output = torch.randn_like(x)
        torch.autograd.backward([x, x], [grad_output, grad_output])

        self.assertEqual(a.grad, b * grad_output * 2)
        self.assertEqual(b.grad, a * grad_output * 2)

    def test_backward_no_grad(self):
        a = torch.randn(5, 5, requires_grad=True)
        b = a + 2
        with self.assertRaises(RuntimeError):
            torch.autograd.backward([b], [None])

    def test_backward_twice_with_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        self.assertRaisesRegex(
            RuntimeError,
            "Specify retain_graph=True",
            lambda: c.backward(torch.tensor([1, 1, 1], dtype=torch.double)),
        )

    def test_backward_twice_retained_graph_with_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_twice_without_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = b + 1
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_twice_retained_graph_without_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_create_graph_warns(self):
        with set_warn_always_context(True):
            b = torch.randn(3, requires_grad=True, dtype=torch.double)
            c = b * b
            with warnings.catch_warnings(record=True) as ws:
                c.backward(torch.ones_like(c), create_graph=True)
            b.grad = None
            self.assertTrue(
                any(
                    "Using backward() with create_graph=True" in str(w.message)
                    for w in ws
                )
            )

            # Should not warn for grad
            with warnings.catch_warnings(record=True) as ws:
                torch.autograd.grad(c, b, torch.ones_like(c), create_graph=True)
            self.assertFalse(
                any(
                    "Using backward() with create_graph=True" in str(w.message)
                    for w in ws
                )
            )

    def test_next_functions(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        a = x + y
        self.assertIsNotNone(a.grad_fn)
        next_functions = a.grad_fn.next_functions
        self.assertEqual(len(next_functions), 2)
        self.assertIsInstance(next_functions[0][0], torch._C._functions.AccumulateGrad)
        self.assertEqual(next_functions[0][1], 0)
        self.assertIsInstance(next_functions[1][0], torch._C._functions.AccumulateGrad)
        self.assertEqual(next_functions[1][1], 0)

        b = a + 5
        next_functions = b.grad_fn.next_functions
        self.assertEqual(len(next_functions), 2)
        self.assertIs(next_functions[0][0], a.grad_fn)
        self.assertIs(next_functions[1][0], None)

    def test_inplace(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4, requires_grad=True)

        z = x * y
        q = z + y
        w = z * y
        z.add_(2)
        # Add doesn't need it's inputs to do backward, so it shouldn't raise
        q.backward(torch.ones(5, 5), retain_graph=True)
        # Mul saves both inputs in forward, so it should raise
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))

        z = x * y
        q = z * y
        r = z + y
        w = z.add_(y)
        # w is a the last expression, so this should succeed
        w.backward(torch.ones(5, 5), retain_graph=True)
        # r doesn't use the modified value in backward, so it should succeed
        r.backward(torch.ones(5, 5), retain_graph=True)
        # q uses dirty z, so it should raise
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        with torch.no_grad():
            x.grad.zero_()
        m = x / 2
        z = m + y / 8
        q = z * y
        r = z + y
        prev_version = z._version
        w = z.exp_()
        self.assertNotEqual(z._version, prev_version)
        r.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad, torch.ones(5, 5) / 2)
        w.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad, torch.empty(5, 5).fill_((1 + math.e) / 2))
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        leaf = torch.ones(5, 5, requires_grad=True)
        x = leaf.clone()
        x.add_(10)
        self.assertEqual(x, torch.ones(5, 5) * 11)
        # x should be still usable
        y = x + 2
        y.backward(torch.ones(5, 5))
        self.assertEqual(leaf.grad, torch.ones(5, 5))
        z = x * y
        x.add_(2)
        self.assertRaises(RuntimeError, lambda: z.backward(torch.ones(5, 5)))

    def test_mark_non_differentiable(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, input):
                output = input > 0
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                return (grad_output * 0).to(torch.double)

        x = torch.randn(5, 5, requires_grad=True)
        mask = MyFunction.apply(x)
        self.assertFalse(mask.requires_grad)
        y = x.masked_fill(mask, 0)
        y.sum().backward()

    @skipIfTorchDynamo("compile tested in test/dynamo/test_autograd_function.py")
    def test_mark_non_differentiable_mixed(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, input):
                a = input + 1
                b = input + 2
                ctx.mark_non_differentiable(a)
                return a, b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                self.assertTrue((grad_a == 0).all())
                self.assertTrue((grad_b == 1).all())
                return grad_b

        x = torch.randn(5, 5, requires_grad=True)
        a, b = MyFunction.apply(x)
        self.assertFalse(a.requires_grad)
        self.assertTrue(b.requires_grad)
        b.sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5))

    def test_mark_non_differentiable_none(self):
        # This used to segfault because MyFunction would send back null
        # gradients to MulBackward, which is implemented in C++. C++
        # implemented functions expect incoming grad_outputs to be non-null.
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, input):
                output = input.clone()
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                return None

        x = torch.randn(5, 5, requires_grad=True)
        r = MyFunction.apply(x * x)
        (r * x).sum().backward()

    def test_return_duplicate(self):
        class DoubleDuplicate(Function):
            @staticmethod
            def forward(ctx, x):
                output = x * 2
                return output, output

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1 * 2 + grad2 * 2

        def fn(x):
            a, b = DoubleDuplicate.apply(x)
            self.assertIs(a, b)
            return a + b

        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        gradcheck(fn, [x])
        gradgradcheck(fn, [x])

    def test_return_duplicate_inplace(self):
        class DoubleInplace(Function):
            @staticmethod
            def forward(ctx, x):
                x.mul_(2)
                ctx.mark_dirty(x)
                return x, x

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1 * 2 + grad2 * 2

        def inplace_fn(x):
            a, b = DoubleInplace.apply(x.clone())
            self.assertIs(a, b)
            return a + b

        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        gradcheck(inplace_fn, [x])
        gradgradcheck(inplace_fn, [x])

        # Can't modify leaf variables in-place
        self.assertRaises(RuntimeError, lambda: InplaceFunction.apply(x))
        # Functions which modify views in-place must return only one output
        self.assertRaises(RuntimeError, lambda: InplaceFunction.apply(x.clone()[0]))

    def _test_setitem(self, size, index):
        x = torch.ones(*size, requires_grad=True)
        y = x + 2
        y_version = y._version
        y[index] = 2
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad = torch.ones(*size)
        expected_grad[index] = 0
        self.assertEqual(x.grad, expected_grad)

    def _test_setitem_tensor(self, size, index):
        x = torch.ones(*size, requires_grad=True)
        y = x + 2
        y_version = y._version
        value = x.new(x[index].size()).fill_(7)
        value.requires_grad = True
        y[index] = value
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad_input = torch.ones(*size)
        expected_grad_input[index] = 0
        self.assertEqual(x.grad, expected_grad_input)
        self.assertEqual(value.grad, torch.ones_like(value))

        # case when x broadcasts to as y[1]
        x = torch.randn(4, requires_grad=True)
        y = torch.zeros(2, 3, 4)
        y[1] = x
        y.backward(torch.randn(2, 3, 4))
        self.assertEqual(x.size(), x.grad.size())

    def test_setitem(self):
        self._test_setitem((5, 5), 1)
        self._test_setitem((5,), 1)
        self._test_setitem((1,), 0)
        self._test_setitem((10,), ([0, 4, 2]))
        self._test_setitem((5, 5), ([0, 4], [2, 2]))
        self._test_setitem((5, 5, 5), (slice(None), slice(None), [1, 3]))
        self._test_setitem((5, 5, 5), (slice(None), [1, 3], slice(None)))
        self._test_setitem((5, 5, 5), ([1, 3], slice(None), slice(None)))
        self._test_setitem((5, 5, 5), (slice(None), [2, 4], [1, 3]))
        self._test_setitem((5, 5, 5), ([1, 3], [2, 4], slice(None)))
        self._test_setitem_tensor((5, 5), 3)
        self._test_setitem_tensor((5, 5), ([0, 1], [1, 0]))
        self._test_setitem_tensor((5,), 3)
        self._test_setitem_tensor(
            (5,), Variable(torch.LongTensor([3]), requires_grad=False).sum()
        )
        self._test_setitem_tensor((5,), [[0, 1, 2, 3]])
        self._test_setitem_tensor((5, 5, 5), (slice(None), slice(None), [1, 3]))
        self._test_setitem_tensor((5, 5, 5), (slice(None), [1, 3], slice(None)))
        self._test_setitem_tensor((5, 5, 5), ([1, 3], slice(None), slice(None)))
        self._test_setitem_tensor((5, 5, 5), (slice(None), [2, 4], [1, 3]))
        self._test_setitem_tensor((5, 5, 5), ([1, 3], [2, 4], slice(None)))
        self._test_setitem_tensor(
            (5, 5, 5),
            (
                Variable(torch.LongTensor([1, 3]), requires_grad=False),
                [2, 4],
                slice(None),
            ),
        )

    def test_setitem_mask(self):
        mask = torch.BoolTensor(5, 5).bernoulli_()
        self._test_setitem((5, 5), Variable(mask))
        self._test_setitem((5,), Variable(mask[0]))
        self._test_setitem((1,), Variable(mask[0, 0:1]))
        self._test_setitem_tensor((5, 5), Variable(mask))
        self._test_setitem_tensor((5,), Variable(mask[0]))

    def test_select_sum(self):
        # both select and sum return Scalars in ATen; ensure they work together.
        x = torch.randn(10, dtype=torch.double, requires_grad=True)

        def func(x):
            return x.select(0, 1).sum()

        gradcheck(func, [x])
        gradgradcheck(func, [x])

    def test_diagonal_expanded_v(self):
        value = torch.rand([])
        v_expanded = torch.tensor(value).expand(10)
        a = torch.rand(10, 10, dtype=torch.double, requires_grad=True)
        (result,) = torch.autograd.grad(a.diagonal(), a, v_expanded)
        self.assertEqual(result, torch.eye(10, dtype=torch.double) * value)

    def test_select_expanded_v(self):
        v_expanded = torch.rand(10).expand(10, 10)
        a = torch.rand(10, 10, 10, requires_grad=True)
        (result,) = torch.autograd.grad(a[0], a, v_expanded)
        expected = torch.zeros(10, 10, 10)
        expected[0] = v_expanded
        self.assertEqual(result, expected)

    def test_slice_expanded_v(self):
        v_expanded = torch.rand(10, 1).expand(2, 10, 10)
        a = torch.rand(10, 10, 10, requires_grad=True)
        (result,) = torch.autograd.grad(a[3:5], a, v_expanded)
        expected = torch.zeros(10, 10, 10)
        expected[3:5] = v_expanded
        self.assertEqual(result, expected)

    def test_unused_output(self):
        x = torch.randn(10, 10, requires_grad=True)
        outputs = x.chunk(5)
        o = outputs[2]
        o = o * 4 + 2
        o.sum().backward()
        expected_grad = torch.zeros(10, 10)
        expected_grad[4:6] = 4
        self.assertEqual(x.grad, expected_grad)

        with torch.no_grad():
            x.grad.zero_()
        grad_output = torch.randn(2, 10)
        outputs = x.chunk(5)
        outputs[0].backward(grad_output)
        expected_grad = torch.zeros(10, 10)
        expected_grad[:2] = grad_output
        self.assertEqual(x.grad, expected_grad)

    # TODO: opinfo this or move to the sparse test suite
    def _test_sparse_gather(self, size_x, size_ind, dim):
        x = torch.randn(size_x, requires_grad=True)
        if len(size_ind) > 0 and len(size_x) > 0:
            ind = torch.randint(x.size(dim), size_ind)
        else:
            ind = torch.zeros(size_ind, dtype=torch.int64)
        out = torch.gather(x, dim, ind, sparse_grad=False)
        grad = torch.rand_like(out)
        out.backward(grad)
        grad_dense = x.grad.clone()
        x.grad = None
        out = torch.gather(x, dim, ind, sparse_grad=True)
        out.backward(grad)
        self.assertEqual(grad_dense, x.grad.to_dense())

    def test_sparse_gather_dim0(self):
        self._test_sparse_gather((10, 10), (5, 10), 0)

    def test_sparse_gather_dim1(self):
        self._test_sparse_gather((10, 10, 5), (10, 5, 5), 1)

    def test_sparse_gather_dim_neg(self):
        self._test_sparse_gather((10, 10, 5), (10, 10, 2), -1)

    def test_sparse_gather_ind_scalar(self):
        self._test_sparse_gather((10,), (), 0)

    def test_sparse_gather_x_scalar(self):
        self._test_sparse_gather((), (2,), 0)

    def test_sparse_gather_both_scalar(self):
        self._test_sparse_gather((), (), 0)

    @skipIfTorchDynamo("grad_dtype not supported in compile")
    def test_grad_dtype(self):
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        # Default to tensor's dtype
        self.assertEqual(leaf.grad_dtype, torch.float32)
        leaf.grad_dtype = torch.float16
        self.assertEqual(leaf.grad_dtype, torch.float16)
        leaf.grad_dtype = None  # Allow any dtype
        self.assertIsNone(leaf.grad_dtype)

        # get/set grad_dtype is only allowed on leaf tensors
        non_leaf = leaf * 2
        self.assertFalse(non_leaf.is_leaf)
        with self.assertRaisesRegex(
            RuntimeError, "grad_dtype can only be accessed on leaf tensors"
        ):
            _ = non_leaf.grad_dtype
        with self.assertRaisesRegex(
            RuntimeError, "grad_dtype can only be set on leaf tensors"
        ):
            non_leaf.grad_dtype = torch.float16

        # Manual setting
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        grad_match = torch.tensor([1.0, 1.0])
        x.grad = grad_match
        self.assertEqual(x.grad.dtype, torch.float32)

        x.grad = None
        x.grad_dtype = torch.float16
        grad_mismatch = torch.tensor([1.0, 1.0])
        with self.assertRaisesRegex(
            RuntimeError,
            "attempting to assign a gradient with dtype.*float.*to a tensor with grad_dtype.*Half",
        ):
            x.grad = grad_mismatch

        # When grad_dtype is None, any dtype is allowed
        x.grad = None
        x.grad_dtype = None
        grad_any = torch.tensor([1.0, 1.0], dtype=torch.float64)
        x.grad = grad_any
        self.assertEqual(x.grad.dtype, torch.float64)

        # Incoming gradient case
        class MismatchedGradientFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp * 2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.to(torch.float64)

        d = torch.tensor([1.0, 2.0], requires_grad=True)
        output = MismatchedGradientFunction.apply(d)
        loss = output.sum()
        loss.backward()
        # Default behavior is to cast to tensor dtype
        self.assertEqual(d.grad.dtype, torch.float32)
        self.assertTrue(torch.allclose(d.grad, torch.tensor([1.0, 1.0])))

        e = torch.tensor([3.0, 4.0], requires_grad=True)
        e.grad_dtype = None
        output_e = MismatchedGradientFunction.apply(e)
        loss_e = output_e.sum()
        loss_e.backward()
        # No casting is done if set to None.
        self.assertTrue(
            torch.allclose(e.grad, torch.tensor([1.0, 1.0], dtype=torch.float64))
        )

        f = torch.tensor([5.0, 6.0], requires_grad=True)
        f.grad_dtype = torch.float16  # Expect float16 gradients
        output_f = MismatchedGradientFunction.apply(f)
        loss_f = output_f.sum()
        loss_f.backward()
        self.assertTrue(
            torch.allclose(f.grad, torch.tensor([1.0, 1.0], dtype=torch.float16))
        )

        # Setting grad_dtype when gradient already exists
        g = torch.tensor([1.0, 2.0], requires_grad=True)
        g.grad = torch.tensor([1.0, 1.0])
        g.grad_dtype = torch.float32
        self.assertEqual(g.grad_dtype, torch.float32)
        with self.assertRaisesRegex(
            RuntimeError, "Cannot set grad_dtype.*because there is already a gradient"
        ):
            g.grad_dtype = torch.float16
        g.grad_dtype = None
        self.assertIsNone(g.grad_dtype)
        g.grad = None
        g.grad_dtype = torch.float16
        self.assertEqual(g.grad_dtype, torch.float16)

        # Test the case where there is an existing accumulate grad
        h = torch.tensor([1.0, 2.0], requires_grad=True)
        _ = h.clone()
        h.grad_dtype = None
        output = MismatchedGradientFunction.apply(h)
        output.sum().backward()
        self.assertEqual(h.grad.dtype, torch.float64)

        # Mixed accumulation cases
        k = torch.tensor([1.0, 2.0], requires_grad=True)
        k.grad_dtype = None
        y = k * 2
        y.sum().backward()
        k.grad = k.grad.to(torch.bfloat16)
        y2 = k * 3
        # Doesn't type promote to float32, always coerce to current .grad's dtype.
        # This is because the accumulation is done in-place on the existing grad.
        self.assertEqual(k.grad.dtype, torch.bfloat16)

        l = torch.tensor([3.0, 4.0], requires_grad=True, dtype=torch.bfloat16)
        l.grad_dtype = None
        z = l * 2
        z.sum().backward()
        l.grad = l.grad.to(torch.float32)
        z2 = l * 3
        z2.sum().backward()
        self.assertEqual(l.grad.dtype, torch.float32)

    def test_gc_in_destructor(self):
        """
        Previously, if a Function destructor triggered a garbage collection,
        the Variable's tp_dealloc handler would get called twice leading to a
        segfault.
        """

        class CollectOnDelete(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

            def __del__(self):
                gc.collect()

        for _ in range(10):
            CollectOnDelete().forward(torch.randn(1, requires_grad=True)).backward()

    def test_naughty_autograd_function_attribute_access(self):
        class Id(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad_x):
                return grad_x

        with self.assertWarnsRegex(DeprecationWarning, "should not be instantiated"):
            f = Id()

        # After raising warning, should still return an instance
        self.assertIsInstance(f, Id)
        x = torch.zeros(1, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "non-static forward method is deprecated"
        ):
            f(x)
        t = Id.apply(x)
        self.assertEqual(t.grad_fn.name(), "IdBackward")

        # THPFunction is the base class of both grad_fn and autograd functions,
        # which means that a lot of accessors on them may segfault. Test that we
        # properly error in this case.
        t = torch.ones(1, requires_grad=True)
        t._backward_hooks = {}
        with self.assertRaisesRegex(
            RuntimeError, "Attribute '_register_hook_dict' is invalid"
        ):
            f._register_hook_dict(t)
        with self.assertRaisesRegex(
            RuntimeError, "Attribute 'register_hook' is invalid"
        ):
            f.register_hook(lambda x, y: None)
        with self.assertRaisesRegex(
            RuntimeError, "Attribute 'next_functions' is invalid"
        ):
            f.next_functions
        with self.assertRaisesRegex(RuntimeError, "Attribute 'name' is invalid"):
            f.name()
        with self.assertRaisesRegex(
            RuntimeError, "Attribute '_sequence_nr' is invalid"
        ):
            f._sequence_nr()
        with self.assertRaisesRegex(
            RuntimeError, "Attribute '_set_sequence_nr' is invalid"
        ):
            f._set_sequence_nr(2)
        with self.assertRaisesRegex(
            RuntimeError, "Attribute '_input_metadata' is invalid"
        ):
            f._input_metadata
        with self.assertRaisesRegex(
            RuntimeError, "underlying PyNode has already been deallocated"
        ):
            f.metadata

    @unittest.expectedFailure
    def test_naughty_anomaly_access(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, g):
                return g

        x = torch.zeros(1, requires_grad=True)
        y = MyFunction.apply(x)
        y.backward()
        y.grad_fn.metadata
        g = y.grad_fn
        del y
        g.metadata  # this currently fails, but shouldn't

    def test_naughty_autograd_function_stashing_ctx(self):
        saved_ctx = []

        class Id(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, grad_x):
                saved_ctx.append(ctx)
                return ctx.saved_tensors

        p = torch.zeros(1, requires_grad=True)
        loss = Id.apply(p)
        loss.backward(retain_graph=True)
        del loss
        # At this point in time, it complains that the graph has been freed
        # (which indeed true, although a somewhat indirect way of stating the
        # problem).
        self.assertRaises(RuntimeError, lambda: saved_ctx[0].saved_tensors)

    def test_custom_autograd_repeated_grad_grad(self):
        # This test failed the equality check in PR #22983; it's an interesting
        # and different test case worth enshrining.  mult1 is not testing
        # anything that interesting, but mult2 is the interesting case.

        def mult1(x):
            return x.prod(dim=-1).prod(dim=-1)

        class Mult(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = mult1(x)
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return (grad_output * y)[:, None, None] / x

        mult2 = Mult.apply

        def check_gradgrad_repeated(x, y):
            (gy,) = torch.autograd.grad(y[0], x, create_graph=True)
            (ggy_1,) = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            (gy,) = torch.autograd.grad(y[0], x, create_graph=True)
            (ggy_2,) = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            self.assertEqual(ggy_1[0, 0, 1], ggy_2[0, 0, 1])

        x = torch.ones(2, 4, 4).requires_grad_()
        check_gradgrad_repeated(x, mult1(x))
        check_gradgrad_repeated(x, mult2(x))

    def test_custom_autograd_no_early_free(self):
        # This test failed complaining that buffers had already been freed
        # prior to #22983.  Also pretty interesting test case.
        class Double(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x**2
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, _ = ctx.saved_tensors
                return grad_output * 2 * x

        # this is equivalent, but uses the output of .forward() in .backward()
        class Double2(Double):
            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * 2 * y / x

        double = Double.apply
        double2 = Double2.apply

        x = torch.tensor(2).double().requires_grad_()

        self.assertTrue(gradcheck(double, x))
        self.assertTrue(gradgradcheck(double, x))
        self.assertTrue(gradcheck(double2, x))
        self.assertTrue(gradgradcheck(double2, x))

        y = double(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)

        y = double2(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)  # should not error!

    def test_custom_autograd_ac_early_stop(self):
        refs = []

        class Test(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x.clone()
                ctx.save_for_backward(y)
                refs.append(weakref.ref(y))
                return y

            @staticmethod
            def backward(ctx, *args):
                _ = ctx.saved_tensors
                return None

        def fn(inp):
            return Test.apply(inp)

        inp = torch.randn(5, 5, requires_grad=True)

        def scope():
            # Early-stop is true by default in non-reentrant torch.utils.checkpoint
            out = torch.utils.checkpoint.checkpoint(fn, inp, use_reentrant=False)
            out.sum().backward()

        with disable_gc():
            scope()

            for ref in refs:
                self.assertIsNone(ref())

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    def test_checkpoint_compile_no_recompile(self):
        # Check for ambient TorchFunctionMode, e.g. when PYTORCH_TEST_WITH_CROSSREF=1
        expect_fail = len(torch.overrides._get_current_function_mode_stack()) > 0

        @torch.compile(backend="aot_eager")
        def fn(x):
            return x.sin().cos()

        x = torch.rand(10, 10, requires_grad=True)

        def run():
            out = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
            out.sum().backward()

            torch._dynamo.reset()

            prev = torch.get_default_device()
            try:
                # Using torch.device("cuda") directly doesn't work here because
                # it has some issues. In particular, unlike set_default_device or
                # invoking the TorchFunctionMode directly, it doesn't update the
                # global state dynamo references for guards:
                # torch.utils._device.CURRENT_DEVICE
                torch.set_default_device("cuda")
                out = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
                out.sum().backward()
            finally:
                torch.set_default_device(prev)

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            if expect_fail:
                with self.assertRaises(RuntimeError):
                    run()
            else:
                run()

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    def test_checkpoint_device_context_fn(self):
        @contextlib.contextmanager
        def apply_device(device):
            try:
                prev = torch.get_default_device()
                torch.set_default_device(device)
                yield
            finally:
                torch.set_default_device(prev)

        def context_fn():
            return contextlib.nullcontext(), apply_device("cuda")

        def fn(x):
            return x.sin().cos()

        with apply_device("cuda"):
            a = torch.tensor(1.0, requires_grad=True)
            out = torch.utils.checkpoint.checkpoint(
                fn, a, context_fn=context_fn, use_reentrant=False
            )
            out.backward()

    def test_detach(self):
        x = torch.randn(10, 10, requires_grad=True)
        y = x + 2
        y = y.detach()
        z = y * 4 + 2
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

        x = torch.randn(10, 10, requires_grad=True)
        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)
        z = x + y
        z.sum().backward()
        # This is an incorrect gradient, but we assume that's what the user
        # wanted. detach() is an advanced option.
        self.assertEqual(x.grad, torch.ones(10, 10))

        # in-place detach
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        a = x * 2
        (y + a).sum().backward(retain_graph=True)
        a.detach_()
        self.assertFalse(a.requires_grad)
        (y + a).sum().backward()  # this won't backprop to x
        self.assertEqual(x.grad, torch.ones(10, 10) * 2)
        self.assertEqual(y.grad, torch.ones(10, 10) * 2)

        # in-place detach on a view raises an exception
        view = x.narrow(0, 1, 4)
        self.assertRaisesRegex(RuntimeError, "view", lambda: view.detach_())

    def test_detach_base(self):
        "detaching base does not detach view"
        x = torch.randn(10, 10, requires_grad=True)
        view = x.narrow(0, 1, 4)
        x.detach_()
        self.assertFalse(x.requires_grad)
        self.assertTrue(view.requires_grad)
        self.assertIsNotNone(view.grad_fn)
        self.assertIs(view._base, x)

    def test_detach_then_inplace_raises_in_autograd(self):
        x = torch.randn([], requires_grad=True)
        orig_x = x.detach().clone()

        y = x**2  # saves x
        z = x.detach()
        z.zero_()
        with self.assertRaisesRegex(RuntimeError, "has been modified by an inplace"):
            y.backward()

    def _test_type_conversion_backward(self, t):
        fvar = Variable(t(torch.randn(5, 5).float()), requires_grad=True)
        fvar.double().sum().backward()
        self.assertEqual(fvar.grad, torch.ones_like(fvar))
        self.assertEqual(type(fvar.grad), type(fvar))
        dvar = Variable(t(torch.randn(5, 5).double()), requires_grad=True)
        dvar.float().sum().backward()
        self.assertEqual(dvar.grad, torch.ones_like(dvar))
        self.assertEqual(type(dvar.grad), type(dvar))

    def test_type_conversions(self):
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.int(), torch.IntTensor)
        if torch.cuda.is_available():
            self.assertIsInstance(x.float().cuda(), torch.cuda.FloatTensor)
            self.assertIsInstance(x.int().cuda(), torch.cuda.IntTensor)
            self.assertIsInstance(x.int().cuda().cpu(), torch.IntTensor)
            if torch.cuda.device_count() >= 2:
                x2 = x.float().cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                x2 = x.float().cuda()
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 0)
                x2 = x2.cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                y = Variable(torch.randn(5).cuda(1), requires_grad=True)
                y.cpu().sum().backward()
                self.assertIs(y.grad.get_device(), 1)
                self.assertIs(y.long().get_device(), 1)

        for t in [
            torch.DoubleTensor,
            torch.FloatTensor,
            torch.IntTensor,
            torch.ByteTensor,
        ]:
            for y_var in (True, False):
                y = torch.randint(5, (5, 5), dtype=t.dtype)
                y = Variable(y) if y_var else y
                self.assertIsInstance(x.type(t), t)
                self.assertIsInstance(x.type_as(y), t)
                # TODO: t.dtype should work
                t_dtype = t().dtype
                self.assertIsInstance(x.type(t_dtype), t)
                self.assertIs(t_dtype, x.type(t_dtype).dtype)
                self.assertEqual(y.data_ptr(), y.type(t).data_ptr())
                if torch.cuda.is_available():
                    for x_cuda in (True, False):
                        for y_cuda in (True, False):
                            x_c = x.cuda() if x_cuda else x
                            y_c = y.cuda() if y_cuda else y
                            _, y_type = y_c.type().rsplit(".", 1)
                            y_typestr = ("torch.cuda." if y_cuda else "torch.") + y_type
                            self.assertEqual(y_c.type(), x_c.type(y_typestr).type())
                            self.assertIs(y_c.dtype, x_c.type(y_c.dtype).dtype)
                            self.assertEqual(
                                y_c.data_ptr(),
                                y_c.cuda().data_ptr() if y_cuda else y_c.data_ptr(),
                            )

        self._test_type_conversion_backward(lambda x: x)
        if torch.cuda.is_available():
            self._test_type_conversion_backward(lambda x: x.cuda())
            if torch.cuda.device_count() >= 2:
                # one of these has to be the non-default device
                self._test_type_conversion_backward(lambda x: x.cuda(0))
                self._test_type_conversion_backward(lambda x: x.cuda(1))

    def test_isolated_node(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        a = x + y
        b = torch.max(a, 1, True)[1].repeat(1, 5).double()
        o = (b + a).sum()
        o.backward()

    def test_shape(self):
        x = torch.randn(3, 4)
        self.assertEqual(2, len(x.shape))
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(x.shape[1], 4)

    def test_numpy_requires_grad(self):
        x = torch.randn(2, 2, requires_grad=True)
        err_msg_outputs = r"Can't call numpy\(\) on Tensor that requires grad. Use tensor.detach\(\).numpy\(\) instead."
        with self.assertRaisesRegex(RuntimeError, err_msg_outputs):
            x.numpy()

        with torch.no_grad():
            x.numpy()

        x = torch.randn(2, 2)
        x.numpy()

        with torch.no_grad():
            x.numpy()

    def test_return_leaf(self):
        class Identity(Function):
            @staticmethod
            def forward(ctx, a, b):
                return a, a + b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a + grad_b, grad_b

        hook_called = [False]
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        q, p = Identity.apply(x, y)

        # Make sure hooks only receive grad from usage of q, not x.
        def hook(grad):
            hook_called[0] = True
            self.assertEqual(grad, torch.ones(5, 5))

        q.register_hook(hook)
        (q + p + x).sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5) * 3)
        self.assertEqual(y.grad, torch.ones(5, 5))
        self.assertTrue(hook_called[0])

    def test_return_leaf_inplace(self):
        class Inplace(InplaceFunction):
            @staticmethod
            def forward(ctx, a, b):
                ctx.mark_dirty(a)
                return a.add_(b), b + 2

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a, grad_a + grad_b

        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)

        q, p = Inplace.apply(x, y)
        self.assertIs(q, x)
        self.assertIs(q.grad_fn.__class__, Inplace._backward_cls)
        self.assertTrue(q.requires_grad)
        q.sum().backward()
        self.assertEqual(y.grad, torch.ones(5, 5))

    def test_leaf_assignment(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, requires_grad=True)
        z = torch.randn(5, requires_grad=True)

        x[0] = y
        x[1] = 2 * z
        self.assertTrue(x.requires_grad)
        self.assertIsNot(x.grad_fn, None)
        x.sum().backward()
        self.assertEqual(y.grad, torch.ones(5))
        self.assertEqual(z.grad, torch.ones(5) * 2)

    def test_no_grad_assignment(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5)
        with torch.no_grad():
            x[0] = y

        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad_fn)

    def test_no_grad_modifies_version(self):
        x = torch.randn(5, requires_grad=True)
        y = torch.randn(5, requires_grad=True)
        z = (x * y).sum()
        with torch.no_grad():
            x *= 2
        self.assertRaisesRegex(
            RuntimeError, "modified by an inplace operation", lambda: z.backward()
        )

    def test_increment_version(self):
        a = torch.rand(5, requires_grad=True)
        v = a._version
        torch.autograd.graph.increment_version(a)
        self.assertEqual(a._version, v + 1)

        a = torch.zeros(5, dtype=torch.int)
        v = a._version
        torch.autograd.graph.increment_version(a)
        self.assertEqual(a._version, v + 1)

        with torch.inference_mode():
            a = torch.rand(5, requires_grad=True)
            # does not error
            torch.autograd.graph.increment_version(a)

        # does not error
        torch.autograd.graph.increment_version(a)

    def test_no_grad_input(self):
        class MyFunction(Function):
            @staticmethod
            def forward(self, x):
                return x

            @staticmethod
            def backward(self, grad_output):
                return grad_output

        x = torch.randn(5, requires_grad=True)
        with torch.no_grad():
            y = MyFunction.apply(x)

        self.assertTrue(x.requires_grad)
        self.assertIsNone(y.grad_fn)

    def test_backward_copy(self):
        # This tests checks backward engine for a very subtle bug that appeared
        # in one of the initial versions of autograd. Gradients tensors were
        # simply stored in lists while the function waited for all its gradients
        # to be computed. However, sometimes an output was used multiple times,
        # so the gradients needed to be summed. Engine used to keep a need_copy
        # set of tensors that will need a clone upon next addition and removed
        # them from the set as soon as the clone was performed. However, this
        # could lead to incorrect results if the same gradient tensor was
        # buffered in three places in the graph:
        # 1. When accumulating gradients in one of these places it was cloned
        #    and removed from need_copy set.
        # 2. When accumulating in second place, it wasn't in the need_copy set,
        #    so the gradients were simply accumulated in-place (which already
        #    modified the grad in 3rd place)
        # 3. When accumulating in the third place, it wasn't in the need_copy set
        #    as well, so the incoming gradient was summed in-place, yielding
        #    incorrect results in all functions, except the first one.
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5, requires_grad=True)
        # Simulate that we're in the middle of the graph
        a = x + 2
        b = y + 2
        c = x + 2
        # This op will just return grad_output two times in backward
        add1 = a + b
        add2 = add1 + c
        # Simulate a long branch, so grad_output will get buffered.
        for _ in range(4):
            a = a * 2
            b = b * 2
            c = c * 2
        branch = a + b + c
        out = add2 + branch
        # expected gradients are:
        # for x: 34 (16 from final a, 16 from final c, 2 from add2)
        # for y: 17 (16 from final b, 1 from add2)
        grad_output = torch.ones(5, 5)
        out.backward(grad_output)
        self.assertEqual(x.grad, torch.ones(5, 5) * 34)
        self.assertEqual(y.grad, torch.ones(5, 5) * 17)

    def test_save_none_for_backward(self):
        test_case = self

        class MyFn(Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(None, input, None)
                return input * input

            @staticmethod
            def backward(ctx, grad_output):
                n1, input, n2 = ctx.saved_tensors
                test_case.assertIsNone(n1)
                test_case.assertIsNone(n2)
                return 2 * input * grad_output

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, 2 * x)

    def test_too_many_grads(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None, None

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, torch.ones_like(x))

    def test_pickle(self):
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=False)

        def assert_strict_equal(var1, var2):
            self.assertEqual(var1, var2)
            self.assertEqual(var1.requires_grad, var2.requires_grad)

        serialized = [pickle.dumps([x, y], protocol=p) for p in range(3)]
        for dump in serialized:
            xc, yc = pickle.loads(dump)
            assert_strict_equal(xc, x)
            assert_strict_equal(yc, y)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_autograd_function.py")
    def test_dep_nograd(self):
        class F1(Function):
            @staticmethod
            def forward(ctx, input):
                out = torch.randn(input.size())
                ctx.mark_non_differentiable(out)
                return input, out

            @staticmethod
            def backward(ctx, grad_output, ignored):
                return grad_output

        class F2(Function):
            @staticmethod
            def forward(ctx, input, ignored):
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None

        x = torch.randn(5, requires_grad=True)
        a, b = F1.apply(x)
        b = b + 1  # separate F1 from F2 by another op
        self.assertTrue(a.requires_grad)
        self.assertFalse(b.requires_grad)
        c = F2.apply(a, b)
        c.backward(torch.ones(c.size()))
        self.assertEqual(x.grad, torch.ones(x.size()))

    def test_set_grad_enabled(self):
        x = torch.tensor([1.0], requires_grad=True)
        with torch.set_grad_enabled(False):
            y = x * 2
        self.assertFalse(y.requires_grad)
        with torch.set_grad_enabled(True):
            y = x * 2
        self.assertTrue(y.requires_grad)
        with torch.set_grad_enabled(False):
            torch.set_grad_enabled(True)
            y = x * 2
        self.assertTrue(y.requires_grad)

    def test_set_grad_enabled_wraps(self):
        for decorator in [True, False]:
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())

                if decorator:
                    # This should not mutate the global grad mode!
                    @torch.set_grad_enabled(False)
                    def inner_func(x):
                        return x.sin()

                else:

                    def inner_func(x):
                        return x.sin()

                    # This is non-idiomatic usage!
                    # More idiomatic usage: torch.set_grad_enabled(False)(inner_func)
                    obj = torch.set_grad_enabled(False)
                    self.assertTrue(not torch.is_grad_enabled())

                    # this will consume the set_grad_enabled global mutation!
                    inner_func = obj(inner_func)
                    self.assertTrue(torch.is_grad_enabled())

                self.assertTrue(torch.is_grad_enabled())

                x = torch.zeros(1, requires_grad=True)
                self.assertTrue(not inner_func(x).requires_grad)

    def test_simple_reentrant(self):
        y_data = torch.randn(2, 2)

        class Reenter(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x, requires_grad=True)
                    ctx.y = Variable(y_data, requires_grad=True)
                    ctx.output_var = ctx.x * ctx.y
                return ctx.output_var.detach()

            @staticmethod
            def backward(ctx, grad_output):
                with torch.enable_grad():
                    ctx.output_var.sum().backward()
                return ctx.x.grad * grad_output

        # Reentrant starts on CPU thread, finishes on GPU thread
        x = torch.randn(2, 2, requires_grad=True)
        out = Reenter.apply(x)
        out.sum().backward()
        self.assertEqual(x.grad, y_data)

    def test_reentrant_child_error(self):
        # Parent graph.
        a = torch.rand(3, 3, requires_grad=True)
        c = a * a

        # Reentrant child graph.
        b = torch.rand(3, 3, requires_grad=True)
        e = b * b
        f = TestAutograd.SimulateBackwardError.apply(e)
        reentrant_root = f.sum()

        class ReentrantFunc(Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                # Reentrant backward in child will throw an error.
                reentrant_root.backward()
                return grad

        d = ReentrantFunc.apply(c)
        with self.assertRaisesRegex(Exception, "Simulate error"):
            d.sum().backward()

    def test_var_mean_differentiable(self):
        dim = [2, 4]
        keepdim = False
        input1 = torch.randn(3, 4, 5, 6, 2, 3, requires_grad=True)
        input2 = deepcopy(input1)
        var1, mean1 = torch.var_mean(input1, dim=dim, keepdim=keepdim)
        var2 = input2.var(dim=dim, keepdim=keepdim)
        mean2 = input2.mean(dim=dim, keepdim=keepdim)
        grad = torch.randn(3, 4, 6, 3, requires_grad=True)

        r1 = var1 * var1 * mean1 * mean1
        r2 = var2 * var2 * mean2 * mean2
        self.assertEqual(r1, r2, rtol=0.01, atol=0.0)

        torch.autograd.backward(r1, grad)
        torch.autograd.backward(r2, grad)
        self.assertEqual(input1.grad, input2.grad, rtol=0.01, atol=0.0)

    @skipIfSlowGradcheckEnv
    @skipIfNoLapack
    def test_lobpcg(self):
        def func(k, A, largest=True, B=None):
            X_shape = list(A.shape)
            X_shape[-1] = k
            X = torch.eye(A.size(-2), k, dtype=A.dtype, device=A.device)
            if A.dim() > 2:
                X = X.expand(X_shape)

            D, U = torch.lobpcg(A=A, k=k, B=B, X=X, largest=largest)

            # LOBPCG uses a random initial eigenspace approximation
            # if parameter `X` is not provided.
            # This may cause a non-deterministic behavior
            # when it comes to the sign of an eigenvector
            # (note if v is an eigenvector, so is -v),
            # hence we eliminate this non-determinism
            # by making sure that each column of U
            # gets multiplied by the sign of its max (in absolute value) element.
            # Also, gradcheck changes the content of the input by +/- eps (default to 1e-06)
            # to compute the numerical gradient which can also cause the signs to flip.
            _, idx = U.abs().max(-2, keepdim=True)
            sign = U.gather(-2, idx).sign()
            U = U * sign
            return D, U

        # TODO: review if this can be ported to OpInfos or moved to test_linalg.py
        def run_symeig_test(k, sizes, largest=True):
            A = torch.rand(*sizes).double()
            A = (A @ A.mT) / 10
            A.requires_grad_(True)

            gradcheck(lambda A: func(k, A, largest), A, check_batched_grad=False)

            # Custom gradient vectors for better stability due to some
            # non-determinism in the lobpcg's forward.
            # Note it is not required if symeig is in forward instead (tested).
            D_grad = torch.rand(*A.shape[:-2], k) / 100
            U_grad = torch.rand(*A.shape[:-1], k) / 100
            gradgradcheck(
                lambda A: func(k, A, largest),
                A,
                [D_grad, U_grad],
                atol=1e-4,
                check_batched_grad=False,
            )

            # check whether A.grad is symmetric
            A = A.detach().requires_grad_(True)
            D, U = func(k, A, largest)
            (D.sum() + U.sum()).backward()
            self.assertEqual(A.grad, A.grad.mT)

        for largest in [True, False]:
            run_symeig_test(1, (6, 6), largest=largest)
            run_symeig_test(1, (2, 6, 6), largest=largest)
            run_symeig_test(1, (2, 2, 6, 6), largest=largest)
            run_symeig_test(2, (6, 6), largest=largest)
            run_symeig_test(2, (2, 6, 6), largest=largest)
            run_symeig_test(2, (2, 2, 6, 6), largest=largest)
            run_symeig_test(3, (9, 9), largest=largest)
            run_symeig_test(3, (2, 9, 9), largest=largest)
            run_symeig_test(3, (2, 2, 9, 9), largest=largest)

    def test_variable_traverse(self):
        def get_out_and_unrefed_cycle():
            inp = torch.randn(10, requires_grad=True)
            tmp = inp.view(10, 1)
            out = tmp.view(10)

            # Create a reference cycle that contains an
            # intermediary Variable in the graph
            my_list = []
            my_list.append(tmp)
            my_list.append(my_list)

            return out

        out = get_out_and_unrefed_cycle()
        gc.collect()
        # This will segfault if things have been erroneously released
        out.backward(torch.randn(out.size()))

    # TODO: review porting these to OpInfo tests
    def test_pow_zero_tensor_gradient(self):
        def run_test(input_size, exponent):
            input = torch.zeros(*input_size, requires_grad=True)
            input.pow(exponent).sum().backward()
            self.assertEqual(input.grad.abs().sum(), 0)

        run_test((10,), torch.zeros(10))
        run_test((10, 10), torch.zeros(10, 10))
        run_test((10,), 0)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_node_ordering_when_none_returned(self):
        class Matmul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w):
                # x: [M, N]
                # w: [N, K]
                ctx.save_for_backward(x, w)
                return x @ w

            @staticmethod
            def backward(ctx, g_out):
                # g_out: [M, K]
                x, w = ctx.saved_tensors
                g_x = g_out @ w.T
                g_w = x.T @ g_out
                w.main_grad = g_w.float()
                return g_x, None

        executed = []

        class HookFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, g):
                executed.append("A")
                return g

        def hook(*args, **kwargs):
            executed.append("B")

        x = torch.randn((3, 3), dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x = HookFunction.apply(x)
        w = torch.randn((3, 3), dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w.register_hook(hook)
        o = Matmul.apply(x, w)
        o.sum().backward()

        self.assertEqual(executed, ["B", "A"])

    def test_current_graph_task_id(self):
        id = [-1]

        def hook(_):
            id[0] = torch._C._current_graph_task_id()

        t = torch.tensor(1.0, requires_grad=True).clone()
        t.register_hook(hook)

        t.backward(retain_graph=True)
        base = id[0]
        t.backward(retain_graph=True)
        self.assertEqual(id[0] - base, 1)
        t.backward(retain_graph=True)
        self.assertEqual(id[0] - base, 2)

        self.assertEqual(torch._C._current_graph_task_id(), -1)

    @skipIfTorchDynamo(
        "_current_graph_task_execution_order requires active backward pass"
    )
    def test_current_graph_task_execution_order(self):
        predicted = [None]
        all_hooks = []

        def hook(_):
            predicted[0] = torch._C._current_graph_task_execution_order()

        def names(nodes):
            return ", ".join([node.name().split(" ")[-1] for node in nodes]) + "\n"

        def grad_fns(*tensors):
            # or grad accumulator
            out = []
            for t in tensors:
                if t.requires_grad and t.grad_fn is None:
                    out.append(t.clone().grad_fn.next_functions[0][0])
                else:
                    out.append(t.grad_fn)
            return out

        actual = []

        def register_logging_hooks(*tensors):
            # register hooks that log the order in which they are called
            def get_hook(i):
                def hook(t_):
                    actual.append(tensors[i])

                return hook

            for i, t in enumerate(tensors):
                all_hooks.append(t.register_hook(get_hook(i)))

        # Basic example: single path
        t = torch.tensor(1.0, requires_grad=True).clone().sin().exp()
        all_hooks.append(t.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            t.backward()
        self.assertExpectedInline(
            names(predicted[0]),
            """\
ExpBackward0, SinBackward0, CloneBackward0, torch::autograd::AccumulateGrad
""",
        )

        # We don't exactly follow sequence_nr order
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(2.0, requires_grad=True)
        c = b.sin()
        d = a.cos()
        out = c * d
        register_logging_hooks(a, b, c, d, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            out.backward()
        self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Accumulate grad node has more than one input
        a = torch.tensor(1.0, requires_grad=True)
        b = a.sin()
        c = a.cos()
        out = b * c
        register_logging_hooks(a, b, c, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            out.backward()
        self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Multiple roots are also OK
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        out = b.sin()
        out2 = b.cos()
        out3 = b.cos()
        register_logging_hooks(a, b, out, out2, out3)
        all_hooks.append(out3.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.grad((out, out3, out2), inputs=(a,))
        self.assertExpectedInline(
            names(predicted[0]),
            """\
CosBackward0, CosBackward0, SinBackward0, MulBackward0, torch::autograd::AccumulateGrad
""",
        )
        # TODO: Uncomment after update to hooks behavior
        # self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Case where next node is nullptr
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        out = b.sin()
        register_logging_hooks(a, b, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            out.backward()
        self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Case where two `inputs` on the same path
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        out = b.sin()
        register_logging_hooks(a, b, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.grad((out,), inputs=(a, b))
        self.assertEqual(
            names(predicted[0]),
            """\
SinBackward0, MulBackward0, torch::autograd::AccumulateGrad
""",
        )
        # TODO: Uncomment after update to hooks behavior
        # self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Case where `inputs` specifies a subgraph
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)
        c = a * b
        out = c.sin()
        register_logging_hooks(a, b, c, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.grad((out,), inputs=(a,))
        self.assertEqual(
            names(predicted[0]),
            """\
SinBackward0, MulBackward0, torch::autograd::AccumulateGrad
""",
        )
        # TODO: Uncomment after update to hooks behavior
        # self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Errors when not called in a backward
        with self.assertRaisesRegex(
            RuntimeError, "should only be called during the backward pass"
        ):
            torch._C._current_graph_task_execution_order()

        # Errors when context manager not enabled
        t = torch.tensor(1.0, requires_grad=True).clone().sin().exp()
        all_hooks.append(t.register_hook(hook))
        with self.assertRaisesRegex(
            RuntimeError,
            "expects the current backward to be executed with multithreading disabled",
        ):
            t.backward()

        # Avoid leaking memory
        for h in all_hooks:
            h.remove()

    @skipIfWindows(msg="node name demangling inconsistent on windows")
    def test_backward_hook_relative_ordering(self):
        order = []

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        x = torch.randn(10, 10, requires_grad=True)
        module = MyModule()
        module.register_full_backward_hook(
            lambda _1, _2, _3: order.append(
                "module_full_backward_hook_BackwardHookFunctionBackward0"
            )
        )

        def make_pre_hook(id):
            return lambda _: order.append(f"pre_hook_{id}")

        def make_post_hook(id):
            return lambda _1, _2: order.append(f"post_hook_{id}")

        count = 0

        def register_hooks_on_all_nodes(nodes):
            nonlocal count
            for node, _ in nodes:
                count += 1
                id = f"{node.name()}_{count}"
                node.register_prehook(make_pre_hook(id))
                node.register_hook(make_post_hook(id))
                register_hooks_on_all_nodes(node.next_functions)

        loss = module(x).sum()
        register_hooks_on_all_nodes(((loss.grad_fn, None),))

        def make_tensor_pre_hook(id):
            return lambda _: order.append(f"tensor_pre_hook_{id}")

        def make_post_acc_grad_hook(id):
            return lambda _: order.append(f"post_acc_grad_hook_{id}")

        x.register_hook(make_tensor_pre_hook("x"))
        module.linear.weight.register_hook(make_tensor_pre_hook("weight"))
        module.linear.bias.register_hook(make_tensor_pre_hook("bias"))

        x.register_post_accumulate_grad_hook(make_post_acc_grad_hook("x"))
        module.linear.weight.register_post_accumulate_grad_hook(
            make_post_acc_grad_hook("weight")
        )
        module.linear.bias.register_post_accumulate_grad_hook(
            make_post_acc_grad_hook("bias")
        )

        loss.backward()

        expected_order = [
            "pre_hook_SumBackward0_1",
            "post_hook_SumBackward0_1",
            "pre_hook_BackwardHookFunctionBackward_2",
            "post_hook_BackwardHookFunctionBackward_2",
            "pre_hook_AddmmBackward0_3",
            "post_hook_AddmmBackward0_3",
            "tensor_pre_hook_bias",
            "pre_hook_torch::autograd::AccumulateGrad_4",
            "post_acc_grad_hook_bias",
            "post_hook_torch::autograd::AccumulateGrad_4",
            "pre_hook_TBackward0_7",
            "post_hook_TBackward0_7",
            "tensor_pre_hook_weight",
            "pre_hook_torch::autograd::AccumulateGrad_8",
            "post_acc_grad_hook_weight",
            "post_hook_torch::autograd::AccumulateGrad_8",
            "pre_hook_BackwardHookFunctionBackward_5",
            "module_full_backward_hook_BackwardHookFunctionBackward0",
            "post_hook_BackwardHookFunctionBackward_5",
            "tensor_pre_hook_x",
            "pre_hook_torch::autograd::AccumulateGrad_6",
            "post_acc_grad_hook_x",
            "post_hook_torch::autograd::AccumulateGrad_6",
        ]

        self.assertEqual(len(expected_order), len(order))
        for expected, actual in zip(expected_order, order):
            self.assertEqual(expected, actual)

    def test_view_replay_enabled(self):
        def f(x):
            out = x.clone().view(-1)
            # mutate the view, triggering autograd view-replay logic
            out.add_(1)
            return out

        x = torch.ones(2, 2, requires_grad=True)

        # Test as a context manager
        with torch.autograd._force_original_view_tracking(False):
            out = f(x)
            self.assertTrue("AsStridedBackward" in str(out.grad_fn))
            self.assertFalse(torch.autograd.is_view_replay_enabled())
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        with torch.autograd._force_original_view_tracking(True):
            out = f(x)
            self.assertTrue("ViewBackward" in str(out.grad_fn))
            self.assertTrue(torch.autograd.is_view_replay_enabled())
        out = f(x)
        self.assertTrue("AsStridedBackward" in str(out.grad_fn))
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        with torch.autograd._force_original_view_tracking(False):
            torch.autograd._force_original_view_tracking(True)
            out = f(x)
            self.assertTrue("ViewBackward" in str(out.grad_fn))
            self.assertTrue(torch.autograd.is_view_replay_enabled())
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        # Test as a function
        torch.autograd._force_original_view_tracking(False)
        out = f(x)
        self.assertTrue("AsStridedBackward" in str(out.grad_fn))
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        torch.autograd._force_original_view_tracking(True)
        out = f(x)
        self.assertTrue("ViewBackward" in str(out.grad_fn))
        self.assertTrue(torch.autograd.is_view_replay_enabled())

    def test_unsafe_set_version_counter(self):
        x = torch.ones(2, requires_grad=True).clone()
        x.add_(1)
        x.add_(2)
        self.assertEqual(2, x._version)
        with torch.autograd._unsafe_preserve_version_counter(x):
            x.mul_(2)
            x.mul_(3)
        # version counter doesn't change inside of the context manager
        self.assertEqual(2, x._version)

        torch._C._autograd._unsafe_set_version_counter((x,), (0,))
        self.assertEqual(0, x._version)
        with self.assertRaisesRegex(RuntimeError, "Cannot set"):
            torch._C._autograd._unsafe_set_version_counter((x,), (-1,))

        y = torch.ones(2, requires_grad=True).clone()
        with torch.autograd._unsafe_preserve_version_counter((x, y)):
            x.mul_(2)
            y.mul_(3)
        # version counter doesn't change inside of the context manager
        self.assertEqual(0, x._version)
        self.assertEqual(0, y._version)

    def test_current_node(self):
        pr = []

        class MyMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args, kwargs=None):
                node = torch._C._current_autograd_node()
                # Don't use node.name() here as it is not consistent on windows
                node_name = node.__class__.__name__ if node else "None"
                pr.append(f"Running {func} from within {node_name}")
                return func(*args, **(kwargs or {}))

        with MyMode():
            pr.append("FW")
            a = torch.rand(10, requires_grad=True)
            b = a.mul(2).div(3).sum()
            pr.append("BW")
            b.backward()
            pr.append("Done")

        self.assertExpectedInline(
            "\n".join(pr),
            """\
FW
Running aten.rand.default from within None
Running aten.mul.Tensor from within None
Running aten.div.Tensor from within None
Running aten.sum.default from within None
BW
Running aten.ones_like.default from within None
Running aten.expand.default from within SumBackward0
Running aten.div.Tensor from within DivBackward0
Running aten.mul.Tensor from within MulBackward0
Running aten.detach.default from within AccumulateGrad
Done""",
        )

    def test_profiler(self):
        x = torch.randn(10, 10)

        with profile(use_kineto=kineto_available()) as p:
            self.assertTrue(torch.autograd._profiler_enabled())
            y = x * 2 + 4

        self.assertFalse(torch.autograd._profiler_enabled())

        names = ["aten::mul", "aten::add"]
        found_indices = set()
        for evt in p.function_events:
            if evt.name in names:
                found_indices.add(names.index(evt.name))
        self.assertEqual(len(found_indices), len(names))

    def test_profiler_seq_nr(self):
        with profile(use_kineto=kineto_available()) as p:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = z.sum(dim=None)
            s.backward()
        print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
        # expecting aten::add, aten::sum to have the sequence numbers,
        # expecting the corresponding backward nodes to have the same numbers
        # as the forward ops
        autograd_ops = {
            ("aten::add", "Add"): [],
            ("aten::sum", "Sum"): [],
        }
        accumulate_ops = []
        found_empty = False
        for e in p.function_events:
            for (fwd_name, bwd_name), ops in autograd_ops.items():
                if e.name == fwd_name or (bwd_name in e.name and "Backward" in e.name):
                    ops.append(e)

            if "AccumulateGrad" in e.name:
                accumulate_ops.append(e)

            # check that nested ops (e.g. empty) don't have
            # sequence number
            if e.name == "aten::empty":
                self.assertEqual(e.sequence_nr, -1)
                found_empty = True

        for idx, ((fwd_name, bwd_name), ops) in enumerate(autograd_ops.items()):
            self.assertEqual(len(ops), 3)
            self.assertEqual(ops[0].name, fwd_name)
            self.assertEqual(
                ops[1].name,
                f"autograd::engine::evaluate_function: {bwd_name}Backward{idx}",
            )
            self.assertEqual(ops[2].name, f"{bwd_name}Backward{idx}")
            self.assertGreaterEqual(ops[0].sequence_nr, 0)
            self.assertEqual(ops[1].sequence_nr, ops[0].sequence_nr)
            self.assertEqual(ops[2].sequence_nr, ops[0].sequence_nr)
            self.assertEqual(ops[0].fwd_thread, 0)
            self.assertEqual(ops[1].fwd_thread, ops[0].thread)
            self.assertEqual(ops[2].fwd_thread, ops[0].thread)
        self.assertTrue(found_empty)

    def test_profiler_unboxed_only(self):
        x = torch.rand(3, 4)

        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            x.resize_([3, 2])

    def test_profiler_propagation(self):
        def foo(x):
            with record_function("in_foo") as rf:
                return x * 2

        x = torch.rand(3, 4)
        traced_foo = torch.jit.trace(foo, x)

        def bar(x):
            with record_function("in_bar") as rf:
                # we expect that profiler will be able
                # propagate across fork
                fut = torch.jit._fork(traced_foo, x)
                y = torch.jit._wait(fut)
                # note: continuation (and rf's end) can
                # be executed in a different thread
                with record_function("in_bar_after_wait") as rf2:
                    y = y * 2
                return y

        traced_bar = torch.jit.trace(bar, x)

        with profile(use_kineto=kineto_available()) as p:
            traced_bar(x)

        found_foo = False
        found_bar = False
        found_bar_after_wait = False
        for info in p.function_events:
            if info.name == "in_foo":
                self.assertFalse(found_foo)
                found_foo = True
            elif info.name == "in_bar":
                self.assertFalse(found_bar)
                found_bar = True
            elif info.name == "in_bar_after_wait":
                self.assertFalse(found_bar_after_wait)
                found_bar_after_wait = True
        self.assertTrue(found_foo)
        self.assertTrue(found_bar)
        self.assertTrue(found_bar_after_wait)

    def test_record_function_callbacks(self):
        x = torch.randn(10, 10)
        with profile(use_kineto=kineto_available()) as p:
            with record_function("foo"):
                y = x * 2 + 4

        function_events = p.function_events
        foo_event = next(event for event in function_events if "foo" in event.name)
        self.assertEqual(foo_event.count, 1)

    def test_record_function_legacy(self):
        # Test the new _record_function ops work
        # Note: Remove once record_function uses these directly
        x = torch.randn(10, 10)
        with profile(use_kineto=kineto_available()) as p:
            handle = torch.ops.profiler._record_function_enter("bar", None)
            try:
                y = x * 2 + 4
            finally:
                torch.ops.profiler._record_function_exit(handle)

        function_events = p.function_events
        foo_event = next(event for event in function_events if "bar" in event.name)
        self.assertEqual(foo_event.count, 1)

    def test_profiler_aggregation_fake(self):
        events = EventList()
        id = [0]

        def get_id():
            id[0] = id[0] + 1
            return id[0]

        # [[thread_id, [(start, end, id), ....]], ...]
        # Using list instead of a dict so order is guaranteed for any Python
        # version
        threads = [
            [1, [(0, 1, get_id()), (1, 2, get_id())]],
            [0, [(0, 2, get_id()), (1, 2, get_id()), (1, 3, get_id())]],
        ]
        for thread, ranges in threads:
            for range in ranges:
                if len(range) != 3:
                    raise AssertionError(f"expected len(range) == 3, got {len(range)}")
                events.append(
                    FunctionEvent(
                        id=range[2],
                        node_id=0,
                        name="",
                        thread=thread,
                        start_us=range[0],
                        end_us=range[1],
                    )
                )

        events._populate_cpu_children()

        # Note that [1, 3] pushes out [0, 2] first. Then we record [1, 2]
        # as a child of [1, 3]
        res = [[], [], [], [], [4]]

        def get_children_ids(event):
            return [child.id for child in event.cpu_children]

        if [get_children_ids(event) for event in events] != res:
            raise AssertionError("children ids mismatch")

    def test_profiler_aggregation_table(self):
        """
        Test if the profiling result is aggregated for `str(prof)`

        See: https://github.com/pytorch/pytorch/issues/37500
        """

        x = torch.randn(1024)
        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            torch.einsum("i->", x)

        prof_str = str(prof)
        prof_table = prof.table()

        self.assertEqual(prof_table, prof_str)

    def test_profiler_function_event_avg(self):
        avg = FunctionEventAvg()
        avg.add(
            FunctionEvent(id=0, node_id=0, name="foo", thread=0, start_us=10, end_us=15)
        )
        avg.add(
            FunctionEvent(id=1, node_id=0, name="foo", thread=0, start_us=20, end_us=30)
        )
        avg.add(avg)
        self.assertEqual(avg.key, "foo")

        # aggregate stats
        self.assertEqual(avg.count, 4)
        self.assertEqual(avg.cpu_time_total, 30)
        self.assertEqual(avg.self_cpu_time_total, 30)
        self.assertEqual(avg.device_time_total, 0)

        # average stats
        self.assertEqual(avg.cpu_time, 7.5)
        self.assertEqual(avg.device_time_total, 0)

    def test_profiler_shapes(self):
        print()
        layer1 = torch.nn.Linear(20, 30)
        layer2 = torch.nn.Linear(30, 40)
        input = torch.randn(128, 20)
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            layer2(layer1(input))

        print(prof.function_events)

        linear_expected_shapes = [
            [[128, 20], [30, 20], [30]],
            [[128, 30], [40, 30], [40]],
        ]

        found_indices = set()
        for event in prof.function_events:
            if event.name == "aten::linear":
                self.assertTrue(event.input_shapes in linear_expected_shapes)
                found_indices.add(linear_expected_shapes.index(event.input_shapes))
        self.assertEqual(len(found_indices), len(linear_expected_shapes))

    def test_profiler_aggregation_lstm(self):
        print()
        rnn = torch.nn.LSTM(10, 20, 2)
        total_time_s = 0
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            for _ in range(20):
                input = torch.randn(5, 3, 10)
                h = torch.randn(2, 3, 20)
                c = torch.randn(2, 3, 20)
                start = time.time()
                rnn(input, (h, c))
                end = time.time()
                total_time_s += end - start

        print(prof.table(sort_by="self_cpu_time_total", row_limit=10, header="TEST"))
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total", row_limit=10
            )
        )
        print(
            prof.table(
                sort_by="self_cpu_time_total",
                row_limit=10,
                max_src_column_width=300,
                header="TEST",
                top_level_events_only=True,
            )
        )
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total", row_limit=10, top_level_events_only=True
            )
        )

        total_time_us = (
            total_time_s * 1000.0 * 1000.0
        )  # make it us which is profiler default
        print("Total time based on python measurements: ", _format_time(total_time_us))
        print(
            f"CPU time measurement python side overhead: {(total_time_us / prof.self_cpu_time_total - 1.0) * 100.0:.2f}%"
        )

        if sys.platform != "win32":
            with tempfile.NamedTemporaryFile() as trace_file:
                prof.export_chrome_trace(trace_file.name)

    def test_record_function(self):
        x = torch.randn(10, 10)

        def forward(x):
            with record_function("outer"):
                y = x * 2 + 4
                with record_function("inner"):
                    y = y - 1
            y = y / 1

        forward(x)

        with profile(use_kineto=kineto_available()) as p:
            forward(x)

        events = p.function_events
        important_events = [
            "outer",
            "aten::mul",
            "aten::add",
            "inner",
            "aten::sub",
            "aten::div",
        ]
        idx = 0
        for info in events:
            if info.name == important_events[idx]:
                idx = idx + 1
            if idx == len(important_events):
                break
        self.assertEqual(idx, len(important_events))

        # We can also use record_function to decorate arbitrary function
        @record_function("my_func")
        def f(x, y):
            return x + y

        with profile(use_kineto=kineto_available()) as p:
            f(1, 2)

        self.assertTrue("my_func" in str(p))

    def test_record_function_multithreaded(self):
        rf = record_function("outer")
        rf.__enter__()
        with record_function("inner"):
            # test that exiting the record function after starting another one
            # doesn't throw.
            rf.__exit__(None, None, None)

        with record_function("inner"):
            rf.__enter__()
        # test that exiting the record function after ending another one
        # doesn't throw.
        rf.__exit__(None, None, None)

    def test_dir(self):
        x = torch.randn(10, 10)
        keys = dir(x)
        self.assertIn("shape", keys)

        # real and imag are only implemented for complex tensors.
        y = torch.randn(10, 10, dtype=torch.cfloat)
        imag_key = "imag"
        self.assertRaises(RuntimeError, lambda: hasattr(x, imag_key))
        self.assertTrue(hasattr(y, imag_key))
        keys.remove(imag_key)

        for key in keys:
            self.assertTrue(hasattr(x, key))

    def test_inplace_on_view_saved_output(self):
        # Test an in-place operation on a view in which the in-place op saves
        # its output. Previously, this created a reference cycle.
        dealloc = [0]

        class IncrementOnDelete:
            def __del__(self):
                dealloc[0] += 1

        def test():
            root = torch.randn(3, 3, requires_grad=True)
            copy = root.clone()
            copy.grad_fn.register_hook(IncrementOnDelete())
            view = copy.view(9)
            torch.nn.functional.relu(view, inplace=True)

        test()
        self.assertEqual(dealloc[0], 1)

    def test_inplace_on_view_leaf_errors(self):
        # Issue #21875: Fail faster (when we try to modify the view vs. in backward())
        x = torch.zeros(1, requires_grad=True)
        y = x.view_as(x)
        with self.assertRaisesRegex(
            RuntimeError,
            "a view of a leaf Variable that "
            "requires grad is being used in "
            "an in-place operation.",
        ):
            y.add_(1)

    def test_inplace_on_view_backward(self):
        # Issue #10532: Make sure that this does not raise RuntimeError.
        net = nn.Sequential(nn.InstanceNorm2d(2), nn.ReLU(True))

        x = torch.tensor([[[[1.0, 1.0]]]], requires_grad=True)
        (g,) = torch.autograd.grad(
            net(x).pow(2), [x], grad_outputs=x.new_ones(x.shape), create_graph=True
        )
        torch.autograd.grad(g.sum(), [x])
        self.assertEqual(x, torch.tensor([[[[1.0, 1.0]]]]))

        # https://discuss.pytorch.org/t/freeing-buffer-strange-behavior/31955/8
        inputs = torch.ones((1, 3, 256, 256), requires_grad=True)

        tmp1 = (inputs + 1).view_as(inputs)
        tmp2 = torch.nn.functional.threshold(tmp1, 0.0, 0.0, True)
        prob_interpolated = torch.sigmoid(tmp2)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=inputs,
            grad_outputs=torch.ones(prob_interpolated.size()),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient_penalty = gradients.sum()
        gradient_penalty.backward()

        fn = gradient_penalty.grad_fn.next_functions[0][0].next_functions[1][0]
        self.assertEqual(fn.name(), "ThresholdBackwardBackward0")

    def test_inplace_on_view_weak_grad_fn(self):
        # Issue 23502: Test that b's grad_fn is preserved.
        a = torch.arange(10.0, requires_grad=True)

        b = a.narrow(0, 0, 2).clone().view(-1)
        b.relu_()

        c = b.clone()
        del b
        gc.collect()

        s = c.sum()
        s.backward()
        self.assertEqual(s, torch.tensor(1.0))

        # Issue #21875: Fail faster (when we try to modify the view vs. in backward())
        a = torch.rand(10, requires_grad=True).narrow(0, 0, 10)
        with self.assertRaises(RuntimeError):
            b = a.relu_()

    def test_out_variant_raises_when_inputs_require_grad(self):
        a = torch.randn(2, 2, requires_grad=True)
        b = torch.randn(2, 2, requires_grad=True)
        x = torch.zeros_like(a)

        # out=... functions don't support automatic differentiation currently
        self.assertRaisesRegex(RuntimeError, "out=", lambda: torch.mul(a, b, out=x))

        # the inputs can require grad if we're in no_grad() mode
        with torch.no_grad():
            torch.mul(a, b, out=x)
            self.assertEqual(x, a * b)

        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        x = torch.zeros(2, 2, requires_grad=True)
        # we should throw an exception if the output requires grad
        self.assertRaisesRegex(RuntimeError, "out=", lambda: torch.mul(a, b, out=x))

    def test_anomaly_detect_nan(self):
        size = 10

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                ctx.fail_0th = fail_0th
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                gI = gO.clone().expand(size)
                gI[0] = 0
                gI[0] /= 0  # Generate a nan
                if ctx.fail_0th:
                    return gI, None, None
                else:
                    return None, gI, None

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        out.backward()  # Should not fail

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        with self.assertRaisesRegex(
            RuntimeError,
            "Function 'MyFuncBackward' returned nan values in its 0th output.",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out.backward()
            self.assertIn("No forward pass information", str(w[0].message))

        inp = torch.rand(size, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "Function 'MyFuncBackward' returned nan values in its 1th output.",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out = MyFunc.apply(inp, inp, False)
                    out.backward()
            self.assertIn("MyFunc.apply", str(w[0].message))

    def test_calculate_shape_util(self):
        out = torch.randn(10, 5, requires_grad=True)
        grad = torch.randn(5, 10, requires_grad=True)
        out_shape, grad_shape = _calculate_shape(out, grad, False)

        if out_shape != torch.Size([10, 5]):
            raise AssertionError(f"expected out_shape == (10, 5), got {out_shape}")
        if grad_shape != torch.Size([5, 10]):
            raise AssertionError(f"expected grad_shape == (5, 10), got {grad_shape}")

        out = torch.nested.as_nested_tensor(
            [
                torch.randn(10, 5, requires_grad=True),
                torch.randn(10, 5, requires_grad=True),
                torch.randn(10, 5, requires_grad=True),
            ]
        )
        grad = torch.nested.as_nested_tensor(
            [
                torch.randn(5, 10, requires_grad=True),
                torch.randn(5, 10, requires_grad=True),
            ]
        )
        out_shape, grad_shape = _calculate_shape(out, grad, False)

        if not torch.equal(out_shape, torch.tensor([[10, 5], [10, 5], [10, 5]])):
            raise AssertionError("out_shape mismatch")
        if not torch.equal(grad_shape, torch.tensor([[5, 10], [5, 10]])):
            raise AssertionError("grad_shape mismatch")

    def test_nested_anomaly_detect_nan(self):
        size = 10

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, inp1, fail_0th):
                ctx.fail_0th = fail_0th
                ctx.save_for_backward(inp1)
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                (inp,) = ctx.saved_tensors
                fail_0th = ctx.fail_0th
                g = gO.clone().expand(size)
                gI = MyFunc2.apply(g * inp, g + inp, fail_0th)
                return gI, None

        class MyFunc2(Function):
            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                ctx.fail_0th = fail_0th
                return inp1 * 2.0 + inp2

            @staticmethod
            def backward(ctx, gO):
                fail_0th = ctx.fail_0th
                g1 = gO.clone()
                g2 = gO.clone()
                g1[0] = 0
                g2[0] = 0
                # generate a nan
                if fail_0th:
                    g1[0] /= 0
                else:
                    g2[0] /= 0
                return g1, g2, None

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, True)
        (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
        gsum = ginp.sum()
        gsum.backward()  # should not fail

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, True)
        (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
        gsum = ginp.sum()
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaisesRegex(
                RuntimeError,
                "Function 'MyFunc2Backward' returned nan values in its 0th output.",
            ):
                with detect_anomaly():
                    gsum.backward()
        self.assertIn("No forward pass information", str(w[1].message))

        inp = torch.rand(size, requires_grad=True)
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaisesRegex(
                RuntimeError,
                "Function 'MyFunc2Backward' returned nan values in its 1th output.",
            ):
                with detect_anomaly():
                    out = MyFunc.apply(inp, False)
                    (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
                    gsum = ginp.sum()
                    gsum.backward()
        self.assertIn("MyFunc2.apply", str(w[1].message))
        self.assertIn("MyFunc.apply", str(w[2].message))

    def test_anomaly_grad_warnings(self):
        # PyTorch won't throw warnings if there is an error
        # but we'd want to at least see them in stderr

        class StdErrDiverter:
            def __enter__(self):
                self.stderr_orig = sys.stderr
                self.stderr_new = io.StringIO()
                sys.stderr = self.stderr_new
                return self

            def __exit__(self, *args):
                self.captured = self.stderr_new.getvalue()
                sys.stderr = self.stderr_orig

        # if the warnings don't throw, they will be handled as regular warnings
        with self.assertRaisesRegex(
            RuntimeError,
            "one of the variables needed for gradient computation has been "
            "modified by an inplace operation",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    a = torch.randn(5, requires_grad=True)
                    d1 = a + 1
                    d2 = d1**2
                    d1 += 1
                    torch.autograd.grad(d2.sum(), a)

        self.assertEqual(len(w), 2)
        self.assertIn("Anomaly Detection has been enabled", str(w[0].message))
        self.assertIn("Error detected in PowBackward0", str(w[1].message))

        # if the warning throws, it will be printed to sys.stderr
        with self.assertRaisesRegex(
            RuntimeError,
            "one of the variables needed for gradient computation has been "
            "modified by an inplace operation",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    warnings.simplefilter("error")
                    with StdErrDiverter() as s:
                        a = torch.randn(5, requires_grad=True)
                        d1 = a + 1
                        d2 = d1**2
                        d1 += 1
                        torch.autograd.grad(d2.sum(), a)

        self.assertEqual(len(w), 1)
        self.assertIn("Anomaly Detection has been enabled", str(w[0].message))
        self.assertIn("Error detected in PowBackward0", s.captured)

    def test_anomaly_assign_parent_cleanup(self):
        # Test that python objects created are properly cleaned up when assign_parent is called

        def get_ref():
            # we use torch.exp here but any function that will construct a new node in its
            # backward call in grad mode will work
            x = torch.randn(2, 2, requires_grad=True)
            t = x.exp()

            # ExpBackward calls mul, creating the MulBackward node when create_graph=True.
            # In anomaly mode, a PyObject referencing MulBackward's "parent" ExpBackward is added to
            # MulBackward's anomaly metadata dict, creating the following reference chain:
            #
            # grad -> MulBackward -> PyObject -> ExpBackward
            #
            with detect_anomaly():
                grad = torch.autograd.grad(t, x, torch.ones_like(t), create_graph=True)

            # We add a weak reference to a new Foo object, which we insert into ExpBackward's metadata dict
            #
            # (PyObject) -> ExpBackward -> dict -> *Foo*
            #            t ----^        WeakRef ---^
            #
            # We want to test that when grad goes out of scope at the end of this function that PyObject is destroyed
            # We can test this by seeing whether Foo is not kept alive once t is destroyed
            class Foo:
                pass

            my_obj = Foo()
            meta_dict = t.grad_fn.metadata
            meta_dict[0] = my_obj
            ref = weakref.ref(my_obj)
            return t, ref

        t, ref = get_ref()
        self.assertIsNotNone(ref())
        del t
        self.assertIsNone(ref())

    def test_nested_anomaly_printstack_cleanup(self):
        # Test if metadata dict PyObject is properly destroyed
        def get_ref():
            # This is similar to the construction in test_anomaly_assign_parent_cleanup:
            #
            # MyFuncBackward2 -> PyObject -> MyFuncBackward -> dict -> Foo
            #                               out ---^         WeakRef ---^
            #
            # We want to check that Foo is still properly destroyed even when MyFunc2Backward's
            # AnomalyMetadata calls printstack, which does some python object manipulation.
            #
            # You might be wondering why we still have to test_anomaly_assign_parent_cleanup,
            # since if PyObject is not destroyed here, wouldn't this test would detect that also?
            # The answer is that custom function's PyObject (THPFunction) actually only hold
            # a weak reference to the c++ node!
            class MyFunc(Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gO):
                    (x,) = ctx.saved_tensors
                    return MyFunc2.apply(x)

            class MyFunc2(Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO + float("NaN")

            inp = torch.rand(1, requires_grad=True)
            out = MyFunc.apply(inp)
            (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)

            with warnings.catch_warnings(record=True) as w:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Function 'MyFunc2Backward' returned nan values in its 0th output.",
                ):
                    with detect_anomaly():
                        ginp.backward()

            class Foo:
                pass

            my_obj = Foo()
            meta_dict = out.grad_fn.metadata
            meta_dict[0] = my_obj
            ref = weakref.ref(my_obj)
            return out, ref

        t, ref = get_ref()
        self.assertIsNotNone(ref())
        del t
        self.assertIsNone(ref())

    def test_anomaly_mode_no_check_nan(self):
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, gO):
                return torch.tensor(float("nan")).expand(10, 10)

        def run_fn(a):
            out = MyFunc.apply(a)
            return out.sum()

        with warnings.catch_warnings(record=True) as w:
            with torch.autograd.detect_anomaly(check_nan=False):
                inp = torch.rand(10, 10, requires_grad=True)
                out = run_fn(inp)
                out.backward(retain_graph=True)

                with torch.autograd.detect_anomaly(check_nan=True):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Function 'MyFuncBackward' returned nan values in its 0th output.",
                    ):
                        out.backward(retain_graph=True)

                out.backward()

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_forward_traceback_preserves_exception_with_checkpoint(self):
        # Regression test: gatherForwardTraceback() must not clear a pending
        # Python exception.  See combined_traceback.cpp for the fix.
        #
        # Ingredients: (1) CUDA memory history recording with context="all"
        # so allocator callbacks fire on free, (2) a custom library op whose
        # forward goes through THPFunction_apply (via Generated in
        # torch._library.autograd), (3) non-reentrant checkpoint.
        #
        # During backward recomputation, _StopRecomputationError is raised
        # from the checkpoint pack_hook inside _save_variables.  The exception
        # stays pending in Python thread state while C++ stack-unwinds (the
        # default python_error ctor does not persist).  Destroying the output
        # THPObjectPtr during unwinding frees the recomputed output tensor's
        # CUDA storage, triggering the allocator callback ->
        # CapturedTraceback::gather() -> gatherForwardTraceback().  On
        # Python < 3.13, without the PyErr_Fetch/PyErr_Restore fix, the
        # PyDict_GetItemRef compat shim clears the pending exception ->
        # SystemError.
        with torch.library._scoped_library("_test_autograd", "FRAGMENT"):

            @torch.library.custom_op("_test_autograd::sin_op", mutates_args=())
            def sin_op(x: torch.Tensor) -> torch.Tensor:
                return x.sin()

            def setup_context(ctx, inputs, output):
                (x,) = inputs
                ctx.save_for_backward(x)

            def backward(ctx, grad):
                (x,) = ctx.saved_tensors
                return grad * x.cos()

            torch.library.register_autograd(
                "_test_autograd::sin_op",
                backward,
                setup_context=setup_context,
            )

            def fn(x):
                return torch.ops._test_autograd.sin_op(x)

            try:
                torch.cuda.memory._record_memory_history("all", stacks="python")
                x = torch.randn(4, device="cuda", requires_grad=True)
                y = checkpoint(fn, x, use_reentrant=False)
                y.sum().backward()
            finally:
                torch.cuda.memory._record_memory_history(None)

    def test_no_grad_copy(self):
        # create autograd function that saves grad pointer as class static
        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad.data_ptr()
                return grad, grad

        class NonContGradFunc(Function):
            @staticmethod
            def forward(ctx, inp1):
                ctx.size = inp1.size()
                return torch.tensor([1.0])

            @staticmethod
            def backward(ctx, grad):
                return torch.ones(1).expand(ctx.size)

        a = torch.randn(5, 6, requires_grad=True)
        b = torch.randn(5, 6, requires_grad=True)
        # non-contiguous grad should be copied
        NonContGradFunc.apply(MyFunc.apply(a, b)).backward()
        self.assertFalse(a.grad.data_ptr() == MyFunc.static_grad_ptr)
        self.assertFalse(b.grad.data_ptr() == MyFunc.static_grad_ptr)
        # test case that should trigger no copy for one of a,b
        a.grad = b.grad = None
        MyFunc.apply(a, b)[1][0].backward()
        p_g = MyFunc.static_grad_ptr
        p_a = a.grad.data_ptr()
        p_b = b.grad.data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # check one of them is using the computed buffer
        self.assertTrue(p_a == p_g or p_b == p_g)

    def test_no_grad_copy_sparse(self):
        # create autograd function that saves grad pointer as class static
        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad._values().data_ptr()
                return grad, grad

        class NonContGradFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                # Create a sparse tensor with non-contiguous indices and values
                # and return as grad.
                v = torch.rand(1, 3)
                i = torch.ones(1, 1, dtype=torch.long)
                nv = v.expand(8, 3)
                ni = i.expand(1, 8)
                ngrad = torch.sparse_coo_tensor(ni, nv, (10, 3), dtype=torch.float32)
                NonContGradFunc.static_grad_ptr = ngrad._values().data_ptr()
                return ngrad, ngrad

        a = torch.randn(10, 3, requires_grad=True)
        b = torch.randn(10, 3, requires_grad=True)
        input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.tensor([0, 4])
        import torch.nn.functional as F

        # test case that should trigger no copy for one of a,b
        emb_matrix = MyFunc.apply(a, b)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        loss.backward(retain_graph=True)
        p_g = MyFunc.static_grad_ptr
        p_a = a.grad._values().data_ptr()
        p_b = b.grad._values().data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # check one of them is using the computed buffer
        self.assertTrue(p_a == p_g or p_b == p_g)

        # Run backwards multiple times to ensure accumulation works.
        for _ in range(10):
            loss.backward(retain_graph=True)

        # non-contiguous indices and value, we should trigger a copy.
        a.grad = b.grad = None
        emb_matrix = NonContGradFunc.apply(a, b)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        loss.backward(retain_graph=True)
        p_g = NonContGradFunc.static_grad_ptr
        p_a = a.grad._values().data_ptr()
        p_b = b.grad._values().data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # Verify we cloned both grads.
        self.assertFalse(p_a == p_g)
        self.assertFalse(p_b == p_g)

        # Run backwards multiple times to ensure accumulation works.
        for _ in range(10):
            loss.backward(retain_graph=True)

    def test_gradcheck_single_input(self):
        def check(fast_mode):
            def f(inp):
                return inp.mul(5)

            gradcheck(
                f,
                torch.rand(10, dtype=torch.float64, requires_grad=True),
                fast_mode=fast_mode,
            )
            gradgradcheck(
                f,
                torch.rand(10, dtype=torch.float64, requires_grad=True),
                fast_mode=fast_mode,
            )

        check(fast_mode=True)
        check(fast_mode=False)

    @parametrize(
        "layout",
        (
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ),
    )
    def test_gradcheck_input(self, layout):
        if layout in {torch.sparse_bsr, torch.sparse_bsc}:
            blocksize = (2, 2)
            size = (4, 8)
        else:
            blocksize = None
            size = (2, 2)

        def check(fast_mode, masked):
            def fn(sparse):
                return torch.sum(sparse)

            gradcheck(
                fn,
                torch.rand(size, dtype=torch.double)
                .to_sparse(layout=layout, blocksize=blocksize)
                .requires_grad_(),
                masked=masked,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )

        for fast_mode, masked in product(*[(True, False)] * 2):
            check(fast_mode=fast_mode, masked=masked)

    def test_gradcheck_nondeterministic(self):
        class NonDetFunc(Function):
            @staticmethod
            def forward(ctx, x, jitter=0.0):
                ctx._jitter = jitter
                return x

            @staticmethod
            def backward(ctx, grad_out):
                return (
                    NonDetFunc.apply(grad_out, ctx._jitter)
                    * (1 + torch.rand_like(grad_out) * ctx._jitter),
                    None,
                )

        def check(fast_mode):
            inp = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
            gradcheck(
                lambda x: NonDetFunc.apply(x, 0.0),
                inp,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )
            with self.assertRaisesRegex(RuntimeError, "Backward is not reentrant"):
                gradcheck(
                    lambda x: NonDetFunc.apply(x, 1e-6),
                    inp,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            with self.assertRaisesRegex(RuntimeError, "Backward is not reentrant"):
                gradgradcheck(
                    lambda x: NonDetFunc.apply(x, 1e-12),
                    inp,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            gradcheck(
                lambda x: NonDetFunc.apply(x, 0.0),
                inp,
                nondet_tol=1e-5,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )
            gradcheck(
                lambda x: NonDetFunc.apply(x, 1e-6),
                inp,
                nondet_tol=1e-5,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )
            gradgradcheck(
                lambda x: NonDetFunc.apply(x, 1e-12),
                inp,
                nondet_tol=1e-5,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_validates_inputs(self):
        def check(fast_mode):
            x = torch.rand(10, requires_grad=True).to_sparse()
            self.assertTrue(
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    check_batched_grad=False,
                    atol=1e-1,
                    fast_mode=fast_mode,
                    masked=True,
                )
            )
            self.assertFalse(
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    masked=False,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )
            self.assertTrue(
                gradcheck(
                    lambda x: x.to_dense(masked_grad=False),
                    (x,),
                    masked=False,
                    atol=1e-1,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

            # when none of the inputs require grad (always raises even if raise_exception=False)
            x = torch.rand(10, requires_grad=False)
            with self.assertRaisesRegex(
                ValueError, "at least one input tensor to require gradient"
            ):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)

            # (warning) when inputs are not double precision
            x = torch.ones(1, dtype=torch.float32, requires_grad=True)
            with self.assertWarnsRegex(
                UserWarning, "Input #0 requires gradient and is not a double precision"
            ):
                self.assertTrue(
                    gradcheck(lambda x: x, (x,), atol=1e-1, fast_mode=fast_mode)
                )

            # when layout is not mkldnn(aka has strides) and input has a dimension with stride 0. (always raises
            # even if raise_exception=False)
            x = torch.ones(1, dtype=torch.float64, requires_grad=True)
            x = x.expand((2, 2))
            with self.assertRaisesRegex(
                RuntimeError, "The 0th input has a dimension with stride 0"
            ):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)

        check(fast_mode=True)
        check(fast_mode=False)

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_gradcheck_validates_input_mkldnn(self):
        # when mkldnn inputs, forward mode testing is not allowed
        # Update tolerances below to make sure the gradient match even in single precision floats
        # Use the warning assert to hide the float32 warning
        x = torch.ones(1).to_mkldnn().requires_grad_()
        with self.assertWarnsRegex(
            UserWarning, "Input #0 requires gradient and is not a double precision"
        ):
            with self.assertRaisesRegex(
                ValueError, "MKLDNN inputs are not support for forward AD gradcheck."
            ):
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    raise_exception=False,
                    fast_mode=False,
                    check_forward_ad=True,
                    atol=1e-1,
                    rtol=1e-1,
                )

        with self.assertWarnsRegex(
            UserWarning, "Input #0 requires gradient and is not a double precision"
        ):
            with self.assertRaisesRegex(
                ValueError, "MKLDNN inputs are not support for forward AD gradcheck."
            ):
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    raise_exception=False,
                    fast_mode=True,
                    check_forward_ad=True,
                    atol=1e-1,
                    rtol=1e-1,
                )

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_gradcheck_test_outputs(self):
        def check(fast_mode):
            # when sparse outputs (always raise even if raise_exception=False)
            x = torch.rand(10, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(
                ValueError, "Sparse output is not supported at gradcheck yet"
            ):
                gradcheck(
                    lambda x: x,
                    (x,),
                    masked=True,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )

            # when mkldnn outputs (always raise even if raise_exception=False)
            root = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
            with self.assertRaisesRegex(
                ValueError, "MKLDNN output is not supported at gradcheck yet"
            ):
                gradcheck(
                    lambda x: x.to_mkldnn(),
                    (root,),
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_check_no_differentiable_outputs(self):
        def check(fast_mode):
            # When none of the outputs are differentiable, but numerical gradient is not zero
            x = torch.ones((1,), requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError, "Numerical gradient for function expected to be zero"
            ):
                gradcheck(lambda x: torch.tensor([x]), x)
            self.assertFalse(
                gradcheck(
                    lambda x: torch.tensor([x]),
                    x,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

            # succeed when no outputs at all
            self.assertTrue(gradcheck(lambda x: (), (x,), fast_mode=fast_mode))

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_check_batched_grad(self):
        def check(fast_mode):
            x = torch.rand(10, dtype=torch.double, requires_grad=True).to_sparse()
            # runtime error while compute batched grad (print big error)
            with self.assertRaisesRegex(
                RuntimeError,
                "gradcheck or gradgradcheck failed while testing batched gradient",
            ):
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    masked=True,
                    check_batched_grad=True,
                    fast_mode=fast_mode,
                )
            self.assertFalse(
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    masked=True,
                    check_batched_grad=True,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_backward_mul_by_grad_output(self):
        # when grad_input is sparse and has incorrect sparse_dim/dense_dim
        def check(fast_mode):
            def fn(x):
                def hook(grad):
                    if grad is not None:
                        return grad.to_dense().to_sparse(1)
                    return grad

                y = x.clone()
                y.register_hook(hook)
                return y.to_dense()

            x = torch.ones((2, 2), dtype=torch.double, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(
                RuntimeError, "grad is sparse tensor, but has incorrect sparse_dim"
            ):
                gradcheck(
                    fn,
                    (x,),
                    atol=1e-1,
                    masked=True,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            self.assertFalse(
                gradcheck(
                    fn,
                    (x,),
                    atol=1e-1,
                    masked=True,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

            # when backward not multiplied by grad_output (non-sparse case)
            def fn2(x):
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError, "backward not multiplied by grad_output"
            ):
                gradcheck(fn2, (x,), atol=1e-1, fast_mode=fast_mode)
            self.assertFalse(
                gradcheck(
                    fn2, (x,), atol=1e-1, raise_exception=False, fast_mode=fast_mode
                )
            )

            # when backward not multiplied by grad_output (sparse case)
            def fn3(x):
                y = x.clone().to_dense()
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(1, dtype=torch.double, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(
                RuntimeError, "backward not multiplied by grad_output"
            ):
                gradcheck(
                    fn3,
                    (x,),
                    atol=1e-1,
                    masked=True,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            self.assertFalse(
                gradcheck(
                    fn3,
                    (x,),
                    atol=1e-1,
                    masked=True,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

            # when layout of grad_input is not the same as input
            class Test(Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, x):
                    return x.to_sparse()

            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, "grad is incorrect layout"):
                gradcheck(
                    Test.apply, (x,), check_batched_grad=False, fast_mode=fast_mode
                )
            self.assertFalse(
                gradcheck(
                    Test.apply,
                    (x,),
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

        check(fast_mode=True)
        check(fast_mode=False)

    # There are two issues:
    # 1. Dynamo uses real fake tensor when speculating so we never trace
    #    the x is none branch.
    # 2. torch.autograd.gradcheck wraps grads with UndefinedGrad which
    #    gets resulted as Zero tensors when getting passed into custom
    #    autograd function in the runtime. Apply materialize grad is tricky,
    #    because user function (dynamo in this case) needs to handle None case
    # this is fine in normal torch.compile case because aot_autograd would
    # never receive None tensors. But it will be a problem when we are directly
    # tracing autograd.grad into graph because now you will get different
    # result from eager. One potential fix is to detect x is None in dynamo
    # bytecode level but that is too complicated so we just YOLO.
    @skipIfTorchDynamo("branching on grad")
    def test_gradcheck_undefined_grad(self):
        def check(fast_mode):
            # when encounter runtime error while running backward
            def fn(x):
                def hook(x):
                    if x is None:
                        raise RuntimeError("x is undefined")

                y = x.clone()
                y.register_hook(hook)
                return y

            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertWarnsRegex(
                UserWarning,
                "Backwards compatibility: New undefined gradient support checking feature",
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Expected backward function to handle undefined output grads",
                ):
                    gradcheck(fn, (x,), fast_mode=fast_mode)
                self.assertFalse(
                    gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode)
                )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_jacobian_mismatch(self):
        def check(fast_mode):
            def fn(x):  # R -> R, C -> C
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            self.assertFalse(
                gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode)
            )

            x_c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
            with self.assertRaisesRegex(
                RuntimeError,
                "While considering the imaginary part of complex outputs only",
            ):
                gradcheck(fn, (x_c,), fast_mode=False)
            self.assertFalse(
                gradcheck(fn, (x_c,), raise_exception=False, fast_mode=False)
            )

            def fn2(x):  # R -> C
                y = torch.complex(x, x)
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError,
                "While considering the imaginary part of complex outputs only",
            ):
                gradcheck(fn2, (x,), fast_mode=False)
            self.assertFalse(
                gradcheck(fn2, (x,), raise_exception=False, fast_mode=False)
            )

            def fn3(x):  # C -> R
                y = torch.real(x)
                y.register_hook(lambda x: x + 1e-2)
                return y

            with self.assertRaisesRegex(
                RuntimeError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn3, (x_c,), fast_mode=False)
            self.assertFalse(
                gradcheck(fn3, (x_c,), raise_exception=False, fast_mode=False)
            )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_dense_and_sparse_inputs(self):
        def check(fast_mode):
            def fn(x, y):
                return x * y.coalesce().to_dense()

            a = torch.rand(2, 2, dtype=torch.double, requires_grad=True)
            b = torch.rand(2, 2, dtype=torch.double).to_sparse().requires_grad_(True)
            self.assertTrue(
                gradcheck(
                    fn,
                    (a, b),
                    masked=True,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            )

        check(fast_mode=True)
        check(fast_mode=False)

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_gradcheck_multiple_mkldnn_inputs(self):
        def check(fast_mode):
            def fn(x, y):
                return x + y.to_dense()

            a = torch.rand(10, requires_grad=True)
            b = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            self.assertTrue(
                gradcheck(
                    fn, (a, b), atol=1e-1, check_batched_grad=False, fast_mode=fast_mode
                )
            )

            def fn2(x, y):
                return x.to_dense() + y.to_dense()

            c = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            self.assertTrue(
                gradcheck(
                    fn, (a, c), atol=1e-1, check_batched_grad=False, fast_mode=fast_mode
                )
            )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_output_shape_or_dtype_depend_on_values(self):
        def check(fast_mode):
            def fn(x):
                if torch.all(x >= 1):
                    return torch.cat([x, x])
                else:
                    return x

            a = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(
                AssertionError,
                "return outputs with the same shape when inputs are perturbed",
            ):
                self.assertTrue(gradcheck(fn, (a,), fast_mode=fast_mode))

            def fn2(x):
                if torch.all(x >= 1):
                    return x.to(torch.float32)
                else:
                    return x

            with self.assertRaisesRegex(
                AssertionError,
                "return outputs with the same dtype when inputs are perturbed",
            ):
                self.assertTrue(gradcheck(fn2, (a,), fast_mode=fast_mode))

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_complex_non_complex_outputs(self):
        def fn(x, y):
            z = torch.complex(x, y)
            return z, x + 1

        a = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        self.assertTrue(gradcheck(fn, (a, b)))

        def fn2(z):
            return z, torch.real(z)

        c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
        self.assertTrue(gradcheck(fn2, (c)))

    @unittest.skipIf(TEST_CUDA, "CPU-only test")
    def test_gradcheck_adjusted_atol_complex_inputs(self):
        # Regression test for incorrect atol transformation for
        # complex inputs, allowing fast gradcheck to fail and slow gradcheck to pass.
        # See issue: https://github.com/pytorch/pytorch/issues/166385

        # this particular seed on CPU triggers a specific input tangent vector u in [0,1)^d such that
        # v^T(j_n - j_a)u is interesting.
        torch.manual_seed(97)

        def sample_func(z):
            return 1.0 / torch.norm(z)

        # Input needs to be at least 2-dim. to trigger
        # in gradcheck an input projection vector u
        # that is not all 1s.
        eps = 10e-3  # eps distance factor from origin, to get
        # some interesting numerical vs analytic discrepancy.
        z = eps * torch.rand(
            2,
            dtype=torch.complex128,
            requires_grad=True,
        )
        atol = 8.3e-6
        rtol = 1e-9

        # check both fast and slow gradcheck pass after the fix to _adjusted_atol()
        self.assertTrue(
            gradcheck(sample_func, (z,), fast_mode=True, atol=atol, rtol=rtol)
        )
        self.assertTrue(
            gradcheck(sample_func, (z,), fast_mode=False, atol=atol, rtol=rtol)
        )

    def test_gradcheck_get_numerical_jacobian(self):
        # get_numerical_jacobian is deprecated and no longer used internally by gradcheck
        from torch.autograd.gradcheck import get_numerical_jacobian

        def fn(inputs):
            # get_numerical_jacobian requires fn to take inputs as a tuple
            # and returns the jacobian wrt the first output
            x = inputs[0]
            y = inputs[1]
            return 2 * x + y, x + 2 * y

        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)

        with self.assertWarnsRegex(
            FutureWarning, "`get_numerical_jacobian` was part of PyTorch's private API"
        ):
            jacobian = get_numerical_jacobian(fn, (a, b), target=a, eps=1e-6)
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))

        with self.assertWarnsRegex(
            FutureWarning, "`get_numerical_jacobian` was part of PyTorch's private API"
        ):
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-6)
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))
        self.assertEqual(jacobian[1], 1 * torch.eye(4, dtype=torch.double))

        with self.assertRaisesRegex(ValueError, "Expected grad_out to be 1.0"):
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-6, grad_out=2.0)

    def test_gradcheck_get_analytical_jacobian(self):
        from torch.autograd.gradcheck import get_analytical_jacobian

        def fn(x, y):
            return 2 * x + y, x + 2 * y

        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)

        outputs = fn(a, b)
        with self.assertWarnsRegex(
            FutureWarning, "`get_analytical_jacobian` was part of PyTorch's private API"
        ):
            (
                jacobians,
                reentrant,
                correct_grad_sizes,
                correct_grad_types,
            ) = get_analytical_jacobian((a, b), outputs[0])
        self.assertEqual(jacobians[0], 2 * torch.eye(4, dtype=torch.double))
        self.assertEqual(jacobians[1], 1 * torch.eye(4, dtype=torch.double))
        self.assertTrue(reentrant)

        class NonDetFunc(Function):
            @staticmethod
            def forward(ctx, x, jitter=0.0):
                ctx._jitter = jitter
                return x

            @staticmethod
            def backward(ctx, grad_out):
                return (
                    NonDetFunc.apply(grad_out, ctx._jitter)
                    * (1 + torch.rand_like(grad_out) * ctx._jitter),
                    None,
                )

        outputs = NonDetFunc.apply(a, 1e-6)
        with self.assertWarnsRegex(
            FutureWarning, "`get_analytical_jacobian` was part of PyTorch's private API"
        ):
            (
                jacobians,
                reentrant,
                correct_grad_sizes,
                correct_grad_types,
            ) = get_analytical_jacobian((a,), outputs)
        self.assertFalse(reentrant)

        with self.assertRaisesRegex(ValueError, "Expected grad_out to be 1.0"):
            jacobians, _, _, _ = get_analytical_jacobian((a,), outputs, grad_out=2.0)

    def test_gradcheck_custom_error(self):
        from torch.autograd.gradcheck import GradcheckError

        def check(fast_mode):
            def fn(x):
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(
                GradcheckError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            with self.assertRaisesRegex(
                RuntimeError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            self.assertFalse(
                gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode)
            )

            def fn2(x):
                raise RuntimeError("Not a GradcheckError!")

            # Checks that when raise_exception=False, non-GradcheckErrors are not caught by gradcheck
            with self.assertRaisesRegex(RuntimeError, "Not a GradcheckError!"):
                gradcheck(fn2, (x,), fast_mode=fast_mode, raise_exception=False)

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_forward_ad(self):
        def fn(x, y):
            return x + y, y

        def bad_fn(x, y):
            # Hacky way to check if we're currently inside a forward ad level
            is_running_forward_ad = fwAD._current_level >= 0

            if is_running_forward_ad:
                y_p, y_d = fwAD.unpack_dual(y)
                y = fwAD.make_dual(y_p, y_d * 1.1)

            return x + y, y

        err_msg = "Jacobian computed with forward mode mismatch for output 0 with respect to input 1"

        for fast_mode in [True, False]:
            # Test for all inputs and outputs being real
            x = torch.rand(2, dtype=torch.double, requires_grad=True)
            y = torch.rand(2, dtype=torch.double, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

            def basic_mul(x):
                return torch.view_as_real(torch.resolve_conj(x * 1j))

            gradcheck(basic_mul, x, check_forward_ad=True, fast_mode=fast_mode)

            # Test for one input and one output being complex
            x = torch.rand(2, dtype=torch.cdouble, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

            # Test for all inputs and outputs being complex
            y = torch.rand(2, dtype=torch.cdouble, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

    def test_gradcheck_forward_ad_runs_with_no_requires_grad(self):
        # Currently requires_grad is used as a easy way for gradcheck to know
        # which inputs of the function are meant to be differentiable
        # This test checks that when the inputs are passed to the function they should not have
        # requires_grad=True even though they may have requires_grad=True when passed
        # to gradcheck
        class UserFn(Function):
            @staticmethod
            def forward(ctx, x, y):
                if fwAD._current_level >= 0:
                    self.assertFalse(x.requires_grad)
                    self.assertFalse(y.requires_grad)
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_t, y_t):
                return x_t, y_t

        x = torch.rand(2, dtype=torch.double, requires_grad=True)
        y = torch.rand(2, dtype=torch.double, requires_grad=True)

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=False,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )

        x = torch.rand(2, dtype=torch.double, requires_grad=True)
        y = torch.rand(2, dtype=torch.double, requires_grad=False)
        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )

    def test_gradcheck_forward_ad_respects_requires_grad(self):
        # Currently requires_grad is used as a easy way for gradcheck to know
        # which inputs of the function are meant to be differentiable
        jvp_count = [0]

        class UserFn(Function):
            @staticmethod
            def forward(ctx, x, y):
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_t, y_t):
                jvp_count[0] += 1
                return x_t, y_t

        # NB: In slow gradcheck we need to loop through numel times so use numel = 1 to ensure
        #     that fast and slow have the same counts
        x = torch.rand(1, dtype=torch.double, requires_grad=True)
        y = torch.rand(1, dtype=torch.double, requires_grad=True)
        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=False,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )
        self.assertEqual(jvp_count[0], 2)  # (2) once per input
        jvp_count = [0]

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )
        self.assertEqual(
            jvp_count[0], 6
        )  # (+4): (once with normal ZT (+1), once with efficient ZT (+1)) for each input (x2)
        jvp_count = [0]

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )
        self.assertEqual(
            jvp_count[0], 12
        )  # (+6): (compute batch of 2 with vmap (+1), with a loop (+2)) for each input (x2)
        jvp_count = [0]

        # Repeat the previous test except we mark one input with requires_grad=False
        # NB: _test_undefined_forward_mode is only (+1), when function has single differentiable input, not (+2)!
        #     Otherwise, other counts are halved.
        x = torch.rand(1, dtype=torch.double, requires_grad=True)
        y = torch.rand(1, dtype=torch.double, requires_grad=False)
        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )
        self.assertEqual(jvp_count[0], 5)  # 1 + 1 + 3

    def test_gradcheck_check_forward_or_backward_only(self):
        """Depending on settings for check_forward_ad and check_backward_ad, the
        correct codepaths should be reached (or not reached)
        """
        fwd_fail_err_msg = "FAIL FWD"
        bwd_fail_err_msg = "FAIL BWD"

        class UserFn(Function):
            @staticmethod
            def forward(ctx, foo, fwd_bad, bwd_bad):
                ctx.fwd_bad = fwd_bad
                ctx.bwd_bad = bwd_bad
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                if ctx.bwd_bad:
                    raise RuntimeError(bwd_fail_err_msg)
                else:
                    return 2 * gO, None, None

            @staticmethod
            def jvp(ctx, gI, _1, _2):
                if ctx.fwd_bad:
                    raise RuntimeError(fwd_fail_err_msg)
                else:
                    return 2 * gI

        for fast_mode in (True, False):
            for check_forward_ad in (True, False):
                for check_backward_ad in (True, False):
                    for fwd_bad in (True, False):
                        for bwd_bad in (True, False):
                            fwd_should_fail = fwd_bad and check_forward_ad
                            bwd_should_fail = bwd_bad and check_backward_ad

                            def run():
                                gradcheck(
                                    UserFn.apply,
                                    (x, fwd_bad, bwd_bad),
                                    check_forward_ad=check_forward_ad,
                                    check_backward_ad=check_backward_ad,
                                    check_undefined_grad=check_backward_ad,
                                    check_batched_grad=check_backward_ad,
                                    fast_mode=fast_mode,
                                )

                            x = torch.rand(2, dtype=torch.double, requires_grad=True)

                            if not check_forward_ad and not check_backward_ad:
                                with self.assertRaisesRegex(
                                    AssertionError, "Expected at least one of"
                                ):
                                    run()
                                continue

                            if not fwd_should_fail and not bwd_should_fail:
                                run()
                            else:
                                # If both fail, backward AD failure "hides" forward AD failure
                                if fwd_should_fail:
                                    fail_msg = fwd_fail_err_msg
                                if bwd_should_fail:
                                    fail_msg = bwd_fail_err_msg
                                with self.assertRaisesRegex(RuntimeError, fail_msg):
                                    run()

    def test_gradcheck_forward_ad_batched_grad(self):
        x = torch.rand(2, dtype=torch.double, requires_grad=True)

        # multiple inputs and outputs with non-tensors inputs
        def fn1(a: torch.Tensor, b: int):
            return a.clone(), a + 1

        gradcheck(
            fn1,
            (x, 1),
            check_forward_ad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_undefined_grad=False,
            check_batched_forward_grad=True,
        )

        # unrelated inputs: tangent for c is None
        def fn2(a: torch.Tensor, c: torch.Tensor):
            return a.clone()

        gradcheck(
            fn2,
            (x, x.clone()),
            check_forward_ad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_undefined_grad=False,
            check_batched_forward_grad=True,
        )

        class Fn(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                return gO * 2

            @staticmethod
            def jvp(ctx, gI):
                torch.randn_like(gI)
                return gI * 2

        msg = "vmap: We do not yet support calling random operations inside of vmap"
        with self.assertRaisesRegex(RuntimeError, msg):
            gradcheck(
                Fn.apply, (x,), check_forward_ad=True, check_batched_forward_grad=True
            )

    def test_version_counter(self):
        x = torch.randn(1, 2)

        # In-place op bumps version
        x_saved_version = x._version
        x.add_(1).add_(1)
        self.assertTrue(x._version > x_saved_version)

        # Differentiable view shares version counter
        xz = x[:]
        self.assertTrue(x._version == xz._version)
        xz.add_(1)
        self.assertTrue(x._version == xz._version)

        # `x.data = y` preserves version counter of `x`
        x_saved_version = x._version
        x.data = torch.randn(2, 3)
        self.assertTrue(x._version == x_saved_version)
        x.add_(1)
        self.assertTrue(x._version > x_saved_version)
        # Make sure `x` is still using the same version counter it shares with `xz`
        self.assertTrue(x._version == xz._version)

        # In-place op on `xz` also updates version of `x`,
        # because they share the version counter
        xz.add_(1)
        self.assertTrue(x._version == xz._version)

    def test_set_data_tensorimpl_type(self):
        # Dense tensor has impl of type `TensorImpl`, while sparse tensor has impl
        # of type `SparseTensorImpl`.
        x = torch.randn(1, 2)
        x_s = torch.sparse_coo_tensor(torch.zeros([1, 1]), torch.ones([1]))
        with self.assertRaisesRegex(RuntimeError, "incompatible tensor type"):
            x.data = x_s

    def test_set_data_preserve_pyobj(self):
        a = torch.randn(1, 2)
        b = torch.randn(1, 2)
        b_id_saved = id(b)
        b.data = a
        self.assertTrue(b_id_saved == id(b))

    def test_set_data_self_requires_grad(self):
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(2.0)
        c = torch.tensor(3, dtype=torch.int64)
        a.data = b
        with self.assertRaisesRegex(
            RuntimeError, "must be floating point or complex dtype"
        ):
            a.data = c

    @unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
    def test_thread_shutdown(self):
        code = """import torch
from torch.autograd import Function
class MyFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad

# Run on cuda if it is available to ensure that the worker thread
# is properly initialized by the time we exit.
device = "cuda" if torch.cuda.is_available() else "cpu"

for shape in [(1,), ()]:
    v = torch.ones(shape, requires_grad=True, device=device)
    MyFunction.apply(v).backward()
"""
        s = TestCase.runWithPytorchAPIUsageStderr(code)
        # The autograd engine creates worker threads only when GPU devices are present.
        # So make sure that we do shutdown threads when we're testing cuda and make sure
        # that there is no thread to shutdown when we're not using cuda.
        if TEST_CUDA or torch.backends.mps.is_available() or torch.xpu.is_available():
            self.assertRegex(s, "PYTORCH_API_USAGE torch.autograd.thread_shutdown")
        else:
            self.assertNotRegex(s, "PYTORCH_API_USAGE torch.autograd.thread_shutdown")

    @unittest.skipIf(
        IS_MACOS,
        "Fails with SIGBUS on macOS; https://github.com/pytorch/pytorch/issues/25941",
    )
    def test_deep_reentrant(self):
        class DeepReentrant(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    DeepReentrant.apply(ctx.x).sum().backward()
                return x

        # Test stack overflow escape mechanism
        v = torch.tensor(2000.0, requires_grad=True)
        # This will cause stack overflow if reentrant calls are handled
        # in the same thread recursively
        DeepReentrant.apply(v).sum().backward()

        # Test stack overflow escape mechanism multiple times
        # to ensure reusing workers in the pool works fine
        v2 = torch.tensor(200.0, requires_grad=True)
        DeepReentrant.apply(v2).sum().backward()

    def test_reentrant_priority(self):
        order = []

        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, x):
                order.append("MyFunction")
                return x

        class Reentrant(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                order.append("Reentrant")
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    Reentrant.apply(ctx.x).backward()
                return x

        a = MyFunction.apply(torch.tensor(6.0, requires_grad=True))
        b = Reentrant.apply(torch.tensor(9.0, requires_grad=True))
        v = a * b
        v.backward()
        # The tasks for the Reentrant and MyFunction backward() will be added
        # to the queue in the autograd engine at the same time. The backward
        # for Reentrant will be executed first, which will then add other
        # backward tasks to the queue. We want to ensure all the reentrant tasks
        # are prioritized over the MyFunction backward task regardless of their
        # sequence numbers
        self.assertEqual(len(order), 11)
        self.assertEqual(order.count("Reentrant"), 10)
        self.assertEqual(order[-1], "MyFunction")

    @slowTest
    def test_checkpointing(self):
        num_inp = 2000
        nz_inp = 10
        nz_out = 10
        nz_bottleneck = 1000

        # small proxy network for some complex reasoning we want to do per input
        module = nn.Sequential(
            nn.Linear(nz_inp, nz_bottleneck),
            nn.ReLU(),
            nn.Linear(nz_bottleneck, nz_inp),
        )

        feat_combined = []
        for _ in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = True
            feat_r = checkpoint(module, data_r, use_reentrant=True)
            feat_combined.append(feat_r)

        # compute mean as a proxy for some joint reasoning
        mean_combined = torch.stack(feat_combined).mean()
        mean_combined.backward()

    def _test_checkpointing_non_reentrant_autocast(self, device_type):
        for enabled in [True, False]:

            def foo(x, y, z):
                # torch.mm is on autocast's list of ops that should run in
                # the autocast precision
                x = torch.mm(x, y)
                y = torch.mm(x, z)
                z = torch.mm(z, z)
                expected_dtype = torch.float32 if not enabled else torch.bfloat16
                self.assertEqual(expected_dtype, z.dtype)
                return z

            x = torch.randn(3, 3, requires_grad=True)
            y = torch.randn(3, 3, requires_grad=True)
            z = torch.randn(3, 3, requires_grad=True)
            if device_type == "cuda":
                x = x.cuda()
                y = y.cuda()
                z = z.cuda()

            with torch.autocast(
                enabled=enabled, device_type=device_type, dtype=torch.bfloat16
            ):
                loss = checkpoint(foo, x, y, z, use_reentrant=False)
                loss = loss.sum()

            # Without saving + recasting the autocast type, would raise error in autograd
            # about mismatched dtypes.
            loss.backward()  # triggers recomputation to check it runs in bfloat

    def test_checkpointing_non_reentrant_autocast_cpu(self):
        """
        Test that autocast args such as the dtype are preserved during non-reentrant
        checkpoint recomputation on CPU.
        """
        self._test_checkpointing_non_reentrant_autocast(device_type="cpu")

    @unittest.skipIf(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        "Test requires CUDA bf16 support",
    )
    def test_checkpointing_non_reentrant_autocast_gpu(self):
        """
        Test that autocast args/kwargs such as the dtype are preserved during
        non-reentrant checkpoint recomputation on GPU.
        """
        self._test_checkpointing_non_reentrant_autocast(device_type="cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    @slowTest
    def test_checkpointing_without_reentrant_memory_savings(self):
        class MyModel(nn.Module):
            def __init__(self, n, use_checkpoint, use_reentrant):
                super().__init__()
                self.n = n
                self.use_checkpoint = use_checkpoint
                self.use_reentrant = use_reentrant
                self.layers = nn.ModuleList()
                for _ in range(self.n):
                    layer = nn.Sequential(
                        nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 256)
                    )
                    self.layers.append(layer)
                # pre-allocate the grad so that increased memory usage is mainly
                # due to activations.
                for layer in self.layers:
                    for lin in layer:
                        lin.weight.grad = torch.ones_like(lin.weight)
                        lin.bias.grad = torch.ones_like(lin.bias)

            def forward(self, x):
                for i in range(self.n):
                    if not self.use_checkpoint:
                        x = self.layers[i](x)
                    else:
                        x = checkpoint(
                            self.layers[i], x, use_reentrant=self.use_reentrant
                        )

                return x

        model_no_checkpoint = MyModel(
            8, use_checkpoint=False, use_reentrant=False
        ).cuda()
        model_reentrant_checkpoint = MyModel(
            8, use_checkpoint=True, use_reentrant=True
        ).cuda()
        model_no_reentrant_checkpoint = MyModel(
            8, use_checkpoint=True, use_reentrant=False
        ).cuda()

        x = torch.randn(100, 256, requires_grad=True, device="cuda")

        torch.cuda.reset_peak_memory_stats()
        loss = model_no_checkpoint(x.clone()).sum()
        loss.backward()
        mem_no_checkpoint = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        loss = model_reentrant_checkpoint(x.clone()).sum()
        loss.backward()
        mem_reentrant_checkpoint = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        loss = model_no_reentrant_checkpoint(x.clone()).sum()
        loss.backward()
        mem_no_reentrant_checkpoint = torch.cuda.max_memory_allocated()

        self.assertTrue(mem_reentrant_checkpoint < mem_no_checkpoint)
        self.assertTrue(mem_no_reentrant_checkpoint < mem_no_checkpoint)

    def test_checkpointing_without_reentrant_custom_function_works(self):
        msg = "Unpack is being triggered for a tensor that was already unpacked once"

        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y, z):
                w = x * y * z
                out = w + w
                ctx.save_for_backward(x, y, z, w, out)
                return out

            @staticmethod
            def backward(ctx, grad_out):
                x, y, z, w, out = ctx.saved_tensors
                # Accessing the saved Tensors a second time will raise because
                # recomputed tensors get cleared as soon as they are unpacked.
                # A recomputation is only triggered if your backward has a new
                # graph-task id.
                with self.assertRaisesRegex(RuntimeError, msg):
                    x_2, y_2, z_2, w_2, out_2 = ctx.saved_tensors
                return x, y, z

        x = torch.tensor(1.0, requires_grad=True)
        y = torch.tensor(2.0, requires_grad=True)
        z = torch.tensor(3.0, requires_grad=True)

        def foo(x, y, z):
            x = x * y * z
            y = y * y * z
            z = z * z
            out = MyFunc.apply(x, y, z)
            return out

        out = checkpoint(foo, x, y, z, use_reentrant=False)
        out.sum().backward()

    def test_checkpointing_without_reentrant_with_context_fn(self):
        class VerboseTorchDispatchMode(TorchDispatchMode):
            def __init__(self) -> None:
                self.operators = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                self.operators.append(func.__name__)
                return func(*args, **kwargs)

        x = torch.tensor(1.0, requires_grad=True)
        verbose_mode = VerboseTorchDispatchMode()

        def context_fn():
            return verbose_mode, contextlib.nullcontext()

        out = checkpoint(
            lambda x: x.exp(), x, use_reentrant=False, context_fn=context_fn
        )
        self.assertEqual(verbose_mode.operators, ["exp.default"])

        verbose_mode.operators = []

        def context_fn():
            return contextlib.nullcontext(), verbose_mode

        out = checkpoint(
            lambda x: x.exp(), x, use_reentrant=False, context_fn=context_fn
        )
        out.backward()
        self.assertEqual(verbose_mode.operators, ["exp.default", "detach.default"])

        with self.assertRaisesRegex(
            Exception, "only supported when use_reentrant=False"
        ):
            out = checkpoint(
                lambda x: x.sin(), x, use_reentrant=True, context_fn=context_fn
            )

    def test_checkpoint_warns_if_use_reentrant_not_passed_explcitly(self):
        a = torch.randn(1, requires_grad=True)

        # Passing explicitly should not warn
        self.assertNotWarn(lambda: checkpoint(lambda x: x, a, use_reentrant=False))

        # Not passing explicitly warns
        with self.assertWarnsOnceRegex(
            UserWarning, ".*the use_reentrant parameter should be passed explicitly.*"
        ):
            checkpoint(lambda x: x, a)

    def test_checkpoint_sequential_warns_if_use_reentrant_not_passed_explcitly(self):
        a = torch.randn(3, requires_grad=True)
        modules_list = [
            torch.nn.Linear(3, 3),
            torch.nn.Linear(3, 3),
            torch.nn.Linear(3, 3),
        ]

        # Passing explicitly should not warn
        self.assertNotWarn(
            lambda: checkpoint_sequential(modules_list, 3, a, use_reentrant=False)
        )

        # Not passing explicitly warns
        with self.assertWarnsOnceRegex(
            UserWarning, ".*the use_reentrant parameter should be passed explicitly.*"
        ):
            checkpoint_sequential(modules_list, 3, a)

    @skipIfTorchDynamo("GraphExecGroup does not support compile")
    def test_checkpoint_graph_execution_group(self):
        def run(use_graph_execution_group):
            counter = [0]

            def fn(x):
                counter[0] += 1
                y = x.sin().cos()
                z = y.sin().cos()
                return y, z

            x = torch.randn(3, 3, requires_grad=True)

            y, z = checkpoint(fn, x, use_reentrant=False)

            group = torch.utils.checkpoint.GraphExecGroup()

            ctx = contextlib.nullcontext()
            if use_graph_execution_group:
                ctx = group

            with ctx:
                (grad_y,) = torch.autograd.grad(
                    z, inputs=(y,), grad_outputs=(torch.ones(3, 3),)
                )

                (grad_x,) = torch.autograd.grad(
                    y,
                    inputs=(x,),
                    grad_outputs=(grad_y,),
                )

            if use_graph_execution_group:
                self.assertEqual(counter[0], 2)
            else:
                self.assertEqual(counter[0], 3)

        run(use_graph_execution_group=True)
        run(use_graph_execution_group=False)

        # Test the not actually disjoint case (using retain_graph=True since
        # otherwise autograd itself will catch this)
        def fn(x):
            return x.sin().cos()

        x = torch.randn(3, 3, requires_grad=True)
        out = checkpoint(fn, x, use_reentrant=False)
        with torch.utils.checkpoint.GraphExecGroup():
            # Under this context, we will enforce that two backward are disjoint
            # even if retain_graph=True.
            out.sum().backward(retain_graph=True)
            with self.assertRaisesRegex(
                RuntimeError, "Performing two backward calls that overlap"
            ):
                out.sum().backward()

    def test_checkpoint_detects_non_determinism(self):
        def save_3_tensors(x):
            out = x.sin().exp()
            out = out.sin()
            return out

        def save_2_tensors(x):
            return x.sin().exp()

        def save_2_tensors_alt(x):
            return x.sin() * torch.tensor([1.0, 2.0])

        def get_non_det_fn(orig_fn, recompute_fn):
            counter = [0]

            def fn(x):
                if counter[0] == 0:
                    counter[0] += 1
                    return orig_fn(x)
                else:
                    return recompute_fn(x)

            return fn

        a = torch.randn(1, requires_grad=True)

        # Save fewer tensors during recompute
        fn = get_non_det_fn(orig_fn=save_3_tensors, recompute_fn=save_2_tensors)
        with self.assertRaisesRegex(
            RuntimeError, "A different number of tensors was saved"
        ):
            out = checkpoint(fn, a, use_reentrant=False)
            out.backward()

        # Save more tensors during recompute
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_3_tensors)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            with self.assertRaisesRegex(
                RuntimeError, "trying to save more tensors during recomputation"
            ):
                out = checkpoint(fn, a, use_reentrant=False)
                out.backward()

        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_3_tensors)
        # If early stopping is enabled, we would not raise (the results would be correct anyway)
        out = checkpoint(fn, a, use_reentrant=False)
        out.backward()

        # Save the same number of tensors but the shape is different
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)
        with self.assertRaisesRegex(RuntimeError, "tensors have different metadata"):
            out = checkpoint(fn, a, use_reentrant=False)
            out.backward()

        # Get the debug message if debug=True
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)

        with self.assertRaisesRegex(
            RuntimeError,
            "You are seeing this error because you passed `debug=True` to checkpoint",
        ):
            out = checkpoint(fn, a, use_reentrant=False, debug=True)
            out.backward()

        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)

        with self.assertRaisesRegex(
            RuntimeError,
            "You are seeing this error because you passed `debug=True` to checkpoint",
        ):
            with torch.utils.checkpoint.set_checkpoint_debug_enabled(True):
                out = checkpoint(fn, a, use_reentrant=False, debug=False)
                out.backward()

        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)

        with self.assertRaisesRegex(
            RuntimeError, "Recomputed values for the following tensors have different"
        ):
            with torch.utils.checkpoint.set_checkpoint_debug_enabled(False):
                out = checkpoint(fn, a, use_reentrant=False, debug=True)
                out.backward()

    def test_access_saved_tensor_twice_without_recomputation_works(self):
        count = [0]

        def foo(a):
            count[0] += 1
            b = a * a
            c = a * b
            d = torch.exp(a)
            return d

        a = torch.randn(5, requires_grad=True)
        d = checkpoint(foo, a, use_reentrant=False)
        self.assertEqual(count[0], 1)
        # Recomputed variables only persist within a particular backward call.
        # If _saved_result is accessed outside of a backward, it will trigger
        # a recompute. And afterwards, those recomputed results are immediately
        # cleared.
        d.grad_fn._saved_result
        self.assertEqual(count[0], 2)
        # Second access will trigger another recompute
        d.grad_fn._saved_result
        self.assertEqual(count[0], 3)
        # Backward clears the saved variable
        d.sum().backward()
        self.assertEqual(count[0], 4)
        # Now it raises an error
        with self.assertRaisesRegex(
            RuntimeError,
            "or directly access saved tensors after they have already been freed",
        ):
            d.grad_fn._saved_result

    @slowTest
    @parametrize("input_requires_grad", [True, False])
    def test_checkpointing_without_reentrant(self, input_requires_grad):
        """
        Basic test for checkpoint without reentrant autograd.
        """
        num_inp = 2000
        nz_inp = 10
        nz_out = 10
        nz_bottleneck = 1000

        # small proxy network for some complex reasoning we want to do per input
        module = nn.Sequential(
            nn.Linear(nz_inp, nz_bottleneck),
            nn.ReLU(),
            nn.Linear(nz_bottleneck, nz_inp),
        )

        # Module holder for testing activation checkpointing with no_reentrant
        # supports kwargs.
        class MyModule(nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.module = mod

            def forward(self, data):
                return self.module(data)

        module = MyModule(mod=module)

        # Run model with and without checkpointing and verify gradients are
        # equivalent, regardless of if inputs require grads or not.
        module_copy = deepcopy(module)

        feat_combined = []
        feat_combined_no_checkpoint = []
        for _ in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = input_requires_grad
            data_r_copy = data_r.clone()
            feat_r = checkpoint(module, data=data_r, use_reentrant=False)
            feat_combined.append(feat_r)
            feat_r_no_checkpoint = module_copy(data_r)
            feat_combined_no_checkpoint.append(feat_r_no_checkpoint)

        # compute mean as a proxy for some joint reasoning
        mean_combined = torch.stack(feat_combined).mean()
        mean_combined.backward()
        mean_combined_no_checkpoint = torch.stack(feat_combined_no_checkpoint).mean()
        mean_combined_no_checkpoint.backward()

        for checkpoint_param, param in zip(
            module.parameters(), module_copy.parameters()
        ):
            self.assertEqual(checkpoint_param.grad, param.grad)

    def test_checkpoint_valid_reset_on_error(self):
        a = torch.randn(2, 2, requires_grad=True)

        with self.assertRaisesRegex(
            Exception, "torch.utils.checkpoint is incompatible"
        ):
            b = checkpoint(torch.exp, a, use_reentrant=True).sum()
            torch.autograd.grad(b, (a,))

        c = checkpoint(torch.exp, a, use_reentrant=True).sum()
        c.backward()

    @parametrize("use_reentrant", [True, False])
    def test_checkpointing_without_reentrant_detached_tensor(self, use_reentrant):
        class NoGradModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)
                self.lin2 = nn.Linear(2, 2, bias=False)

            def forward(self, x):
                with torch.no_grad():
                    return self.lin2(self.linear(x))

        module = NoGradModule()

        err_ctx = (
            self.assertRaisesRegex(
                RuntimeError, "none of output has requires_grad=True"
            )
            if use_reentrant
            else contextlib.nullcontext()
        )

        a = torch.randn(2, 2, requires_grad=True)
        for _ in range(3):
            with err_ctx:
                # out does not require grad
                out = checkpoint(module, a, use_reentrant=use_reentrant)
                # Make loss require grad, otherwise we would run into
                # "element 0 of tensors does not require grad and does not have a grad_fn"
                out += a
                out.sum().backward()

    def test_checkpointing_without_reentrant_saved_object_identity(self):
        x_backward = None

        class Test(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(y)
                return x

            @staticmethod
            def backward(ctx, x):
                nonlocal x_backward
                (x_backward,) = ctx.saved_tensors
                return x, None

        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=False)

        Test.apply(a, b).backward()
        self.assertIs(b, x_backward)

        x_backward = None
        checkpoint(Test.apply, a, b, use_reentrant=False).backward()
        self.assertIs(b, x_backward)

    def test_checkpointing_without_reentrant_correct_grad(self):
        """
        Verifies that correct gradients are calculated for checkpoint
        without reentrant autograd, for both backward() and autograd.grad().
        """
        a = torch.randn(2, 2, requires_grad=True)

        b = torch.exp(a).sum()
        b.backward()
        b_grad = a.grad

        a.grad = None
        c = checkpoint(torch.exp, a, use_reentrant=False).sum()
        c.backward()
        c_grad = a.grad

        a.grad = None
        d = checkpoint(torch.exp, a, use_reentrant=False).sum()
        (d_grad,) = torch.autograd.grad(d, (a,))

        self.assertEqual(b_grad, c_grad)
        self.assertEqual(b_grad, d_grad)

    @skipIfXpu(msg="torch._C._scatter Not implemented on XPU, issue #143239")
    def test_checkpointing_without_reentrant_dataparallel(self):
        """
        Verifies gradient correctness when checkpoint without reentrant autograd
        is used in conjunction with DataParallel.
        """

        class LinearModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)

            def forward(self, inp):
                return self.linear(inp)

        a = torch.randn(2, 2, requires_grad=True)
        if torch.cuda.is_available():
            a = a.cuda()

        model = LinearModule()
        if torch.cuda.is_available():
            model = model.cuda()

        b = deepcopy(model)(a).sum()
        b.backward()
        b_grad = a.grad

        a.grad = None

        module = torch.nn.DataParallel(deepcopy(model))
        c = checkpoint(module, a, use_reentrant=False).sum()
        c.backward()
        c_grad = a.grad

        self.assertEqual(b_grad, c_grad)

    def test_checkpointing_without_reentrant_parameter_used_in_an_out(self):
        """
        Ensures that gradient hooks are only called once per tensor.
        """
        w = torch.randn(10, 10, requires_grad=True)
        count = 0

        def hook(grad):
            nonlocal count
            count += 1

        w.register_hook(hook)
        x = torch.rand(10, 10, requires_grad=True)
        h = w * x  # Using w outside the checkpoint
        out = checkpoint(
            lambda x: w * x, h, use_reentrant=False
        )  # Using w inside the checkpoint

        out.sum().backward()
        # should only call hook once
        self.assertEqual(count, 1)

    def test_checkpointing_without_reentrant_arbitrary_input_output(self):
        """
        Ensures checkpointing without reentrant autograd works with functions
        with arbitrary input/output structures.
        """

        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Linear(5, 5, bias=False)

            def forward(self, dict_input):
                tensor = dict_input["tensor"]
                return {"result": self.layer(tensor)}

        model_no_checkpoint = MyModel()
        model_checkpoint_without_reentrant = deepcopy(model_no_checkpoint)

        inp = {"tensor": torch.randn(5, 5)}

        out_no_checkpoint = model_no_checkpoint(inp)["result"].sum()

        out_checkpoint = checkpoint(
            model_checkpoint_without_reentrant, inp, use_reentrant=False
        )["result"].sum()

        self.assertEqual(out_checkpoint, out_no_checkpoint)

        out_no_checkpoint.backward()
        out_checkpoint.backward()

        for param, checkpoint_param in zip(
            model_no_checkpoint.parameters(),
            model_checkpoint_without_reentrant.parameters(),
        ):
            self.assertEqual(param.grad, checkpoint_param.grad)

    def test_callback_adds_callback(self):
        called = [0]

        def callback_final():
            called[0] += 1

        def callback_adds_callback():
            called[0] += 1
            Variable._execution_engine.queue_callback(callback_final)

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, grad):
                Variable._execution_engine.queue_callback(callback_adds_callback)
                return grad

        a = torch.rand((3, 3), requires_grad=True)
        b = MyFunc.apply(a)
        b.sum().backward()

        self.assertEqual(called[0], 2)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_callback_propagates_errors_from_device_thread(self):
        def callback():
            raise RuntimeError("blah")

        def hook_with_callback(*args):
            torch.autograd.Variable._execution_engine.queue_callback(callback)

        t = torch.tensor([1.0, 2.0], requires_grad=True, device=torch.device("cuda"))
        t.register_hook(hook_with_callback)
        output = t**2
        loss = output.sum()

        with self.assertRaisesRegex(RuntimeError, "blah"):
            loss.backward()

    def _test_reentrant_with_callbacks(self, install_callbacks_in_depths):
        counter = {}
        counter["inner"] = 0
        counter["outer"] = 0

        def inc_inner_counter():
            counter["inner"] += 1

        def inc_outer_counter():
            counter["outer"] += 1

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                if 1 in install_callbacks_in_depths:
                    # Add a callback to execute.
                    Variable._execution_engine.queue_callback(inc_inner_counter)

                return input

        class MyReentrantFunc(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                if 0 in install_callbacks_in_depths:
                    # Add a callback to execute.
                    Variable._execution_engine.queue_callback(inc_outer_counter)
                # Reentrant backward call.
                tmp_inp = input.detach().requires_grad_()
                with torch.enable_grad():
                    tmp_out = (MyFunc.apply(tmp_inp)).sum()
                tmp_out.backward()
                return input

        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = MyReentrantFunc.apply(t1)
        t3 = t2.sum()
        torch.autograd.backward([t3])

        return counter

    def test_reentrant_with_callbacks_depth_0(self):
        # Verify callback is called only once.
        ret = self._test_reentrant_with_callbacks([0])
        self.assertEqual(ret["outer"], 1)
        self.assertEqual(ret["inner"], 0)

    def test_reentrant_with_callbacks_depth_1(self):
        # Verify callback is called only once.
        ret = self._test_reentrant_with_callbacks([1])
        self.assertEqual(ret["outer"], 0)
        self.assertEqual(ret["inner"], 1)

    def test_reentrant_with_callbacks_both_depths(self):
        # Verify callback is called twice.
        ret = self._test_reentrant_with_callbacks([0, 1])
        self.assertEqual(ret["outer"], 1)
        self.assertEqual(ret["inner"], 1)

    def test_reentrant_with_leaf_variable_hook(self):
        handle = None
        param = torch.rand(10, requires_grad=True)

        def add_gradient_penalty_to_grad(grad):
            handle.remove()
            old_param_grad = grad
            param.grad = None
            # Add some sort of gradient penalty by directly updating the gradients
            with torch.enable_grad():
                g = grad.detach().requires_grad_()
                new_param = param.detach().requires_grad_()
                out = ((g * 2) + new_param).sum()
                out.backward()
            res = g.grad + grad
            param.grad = old_param_grad
            return res

        handle = param.register_hook(add_gradient_penalty_to_grad)
        # Forward pass
        tmp = param * param
        loss = tmp.sum()
        # Compute the gradients
        loss.backward()

    def test_reentrant_with_non_leaf_variable_hook(self):
        handle = None
        param = torch.rand(10, requires_grad=True)

        def manual_increase_gradient(grad):
            handle.remove()
            # Add some sort of gradient penalty by directly updating the gradients
            with torch.enable_grad():
                g = grad.detach().requires_grad_()
                out = ((g * 2) + 5).sum()
                out.backward()
            res = g.grad + grad
            return res

        # Forward pass
        tmp = param * param
        handle = tmp.register_hook(manual_increase_gradient)
        loss = tmp.sum()
        # Compute the gradients
        loss.backward()
        self.assertEqual(param.grad, 6 * param)

    def test_grad_fn_attr_bindings(self):
        # Check that the getter of each type returns what we want
        # See `gen_autograd_functions.py` for how the getters are generated
        #
        # This test is only meant to check if the codegen'd bindings work
        # Please help update this test if you update the names of any the fields we check!
        #
        a = torch.ones(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        out1 = torch.stack([a, b], dim=0)
        out2 = (a * 2) * b
        # TODO: I don't think we have a backward saving a list of tensors
        #       at the moment. It used to be stack, but for no reason...
        #       see discussion in #84993
        # self.assertEqual(out.grad_fn._saved_tensors, (a, b))              # TewnsorList -> Tuple[Tensor]
        self.assertEqual(out2.grad_fn._saved_self, a * 2)
        self.assertIsInstance(out2.grad_fn._saved_self, torch.Tensor)
        self.assertIsInstance(
            out2.grad_fn._raw_saved_self, torch._C._autograd.SavedTensor
        )
        self.assertEqual(out1.grad_fn._saved_dim, 0)  # int64_t -> int
        self.assertIsInstance(out1.grad_fn._saved_dim, int)

        out2.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)

        out2.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            out2.grad_fn._saved_self
        # TODO: interestingly, this only happens if indexing into a list grad_fn._raw_saved_tensors[0],
        #       not when using a saved tensor, see discussion in #84993
        # with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
        #     out2.grad_fn._raw_saved_self
        self.assertEqual(out1.grad_fn._saved_dim, 0)

        a = torch.ones(2, 2, requires_grad=True)
        indices = torch.tensor([0, 1])
        out = a[:, indices]
        self.assertEqual(
            out.grad_fn._saved_indices, (None, indices)
        )  # c10::List<std::optional<Tensor>> -> Tuple[Tensor?]
        self.assertIsInstance(out.grad_fn._saved_indices[1], torch.Tensor)
        self.assertIsInstance(
            out.grad_fn._raw_saved_indices[1], torch._C._autograd.SavedTensor
        )
        self.assertEqual(
            out.grad_fn._saved_self_sym_sizes, a.shape
        )  # SymIntArrayRef -> Tuple[SymInt]
        self.assertIsInstance(out.grad_fn._saved_self_sym_sizes[0], int)

        out.grad_fn._raw_saved_indices[1].register_hooks(lambda x: x, lambda x: x)
        with self.assertRaisesRegex(RuntimeError, "None is forbidden"):
            out.grad_fn._raw_saved_indices[0].register_hooks(lambda x: x, lambda x: x)

        out = a.mean()
        self.assertEqual(
            out.grad_fn._saved_self_sym_sizes, a.shape
        )  # IntArrayRef -> Tuple[int]

        a = torch.ones(2, 2, requires_grad=True)
        out = a * a
        out.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)
        out.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "after it has been freed"):
            out.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)

        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.nn.functional.interpolate(a, 4, mode="linear")
        self.assertEqual(
            out.grad_fn._saved_output_size, (4,)
        )  # std::optional<IntArrayRef> -> int[]?
        self.assertIsInstance(out.grad_fn._saved_output_size[0], int)
        self.assertEqual(out.grad_fn._saved_align_corners, False)  # bool -> bool
        self.assertIsInstance(out.grad_fn._saved_align_corners, bool)
        if hasattr(out.grad_fn, "_saved_scale_factors"):
            self.assertIsNone(
                out.grad_fn._saved_scale_factors
            )  # std::optional<ArrayRef<double>> -> float[]?
        else:
            self.assertIsNone(
                out.grad_fn._saved_scales
            )  # std::optional<ArrayRef<double>> -> float[]?

        a = torch.ones(1, 1, 3, 3, requires_grad=True)
        out = nn.Conv2d(1, 1, 3)(a)
        self.assertEqual(
            out.grad_fn._saved_bias_sym_sizes_opt, (1,)
        )  # std::optional<SymIntArrayRef> -> SymInt[]?
        out = nn.Conv2d(1, 1, 3, bias=False)(a)
        # TODO: This is BAD! we converted a std::nullopt into a (0,)
        self.assertEqual(out.grad_fn._saved_bias_sym_sizes_opt, (0,))

        a = torch.ones(1, 3, 3, requires_grad=True)
        out = torch.addbmm(a.squeeze(0), a, a)
        self.assertEqual(out.grad_fn._saved_batch1_sym_argsize_0, 1)  # int64_t
        self.assertEqual(out.grad_fn._saved_batch1_sym_argsize_1, 3)  # int64_t

        a = torch.ones(1, 1, 3, 3, requires_grad=True)
        out = torch.nn.functional.unfold(a, 3)
        self.assertEqual(out.grad_fn._saved_self_sym_argsize_minus_2, 3)  # SymInt
        self.assertEqual(out.grad_fn._saved_self_sym_argsize_minus_1, 3)  # SymInt

        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.nn.functional.interpolate(a, scale_factor=0.5, mode="linear")
        self.assertEqual(out.grad_fn._saved_scales, 0.5)

        a = torch.ones(2, 2, requires_grad=True)
        out = torch.pdist(a, p=1)
        self.assertEqual(out.grad_fn._saved_p, 1.0)  # double -> float
        self.assertIsInstance(out.grad_fn._saved_p, float)

        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.logit(a, 1.0)
        self.assertEqual(out.grad_fn._saved_eps, 1.0)  # c10:optional<double> -> float?
        self.assertIsInstance(out.grad_fn._saved_eps, float)
        out = torch.logit(a)
        self.assertIsNone(out.grad_fn._saved_eps)

        if torch._C.has_lapack:
            a = torch.ones(1, 1, requires_grad=True)
            q, r = torch.linalg.qr(a, mode="reduced")
            self.assertEqual(q.grad_fn._saved_mode, "reduced")  # std::string -> str

        a = torch.tensor([1.0], requires_grad=True)
        out = torch.div(a, 2.0, rounding_mode="trunc")
        self.assertEqual(
            out.grad_fn._saved_rounding_mode, "trunc"
        )  # std::optional<std::string> -> str?
        out = torch.div(a, 2.0, rounding_mode=None)
        self.assertIsNone(
            out.grad_fn._saved_rounding_mode
        )  # std::optional<std::string> -> str?

        x = torch.zeros(5, requires_grad=True)
        out = torch.threshold(x, threshold=(1 + 0j), value=(1 + 0j))
        self.assertIsInstance(
            out.grad_fn._saved_threshold, complex
        )  # Scalar(complex double) -> complex
        cfloat = torch.tensor(1 + 0j, dtype=torch.complex64)
        out = torch.threshold(x, threshold=cfloat, value=(1 + 0j))
        self.assertIsInstance(
            out.grad_fn._saved_threshold, complex
        )  # Scalar(complex float) -> complex
        out = torch.threshold(x, threshold=1.0, value=1.0)
        self.assertIsInstance(
            out.grad_fn._saved_threshold, float
        )  # Scalar(floating point) -> float
        out = torch.threshold(x, threshold=1, value=1)
        self.assertIsInstance(
            out.grad_fn._saved_threshold, int
        )  # Scalar(integral) -> int
        out = torch.threshold(x, threshold=False, value=False)
        self.assertIsInstance(
            out.grad_fn._saved_threshold, bool
        )  # Scalar(bool) -> bool

        a = torch.ones(2, 2, requires_grad=True)
        out = a.as_strided((3,), (1,), 1)
        self.assertEqual(
            out.grad_fn._saved_storage_offset, 1
        )  # c10:optional<int64_t> -> int?
        self.assertIsInstance(out.grad_fn._saved_storage_offset, int)
        out = a.as_strided((3,), (1,))
        self.assertIsNone(out.grad_fn._saved_storage_offset)

        a = torch.ones(2, requires_grad=True)
        out = torch.tanh(a)
        self.assertEqual(out, out.grad_fn._saved_result)  # saved variable when output

        a = torch.randn(3, 5, requires_grad=True)
        b = torch.tensor([1, 0, 4])
        loss = nn.NLLLoss()
        out = loss(a, b)
        self.assertIsNone(out.grad_fn._saved_weight)
        loss = nn.NLLLoss(weight=torch.ones((5,)))
        out = loss(a, b)
        self.assertEqual(
            out.grad_fn._saved_weight, torch.ones((5,))
        )  # c10:optional<Tensor> -> Tensor?

        out.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            out.grad_fn._saved_weight

        num_tensors = 3
        input_tensors = [
            torch.ones(2, 2, requires_grad=True) for _ in range(num_tensors)
        ]
        scalars = [
            0.0 for _ in range(num_tensors)
        ]  # ArrayRef<Scalar> -> Tuple[Scalar, ...]
        results = torch._foreach_maximum(input_tensors, scalars)
        for t in results:
            self.assertEqual(t.grad_fn._saved_scalars, scalars)

    def test_get_data_and_hooks_from_raw_saved_variable(self):
        def pack_hook(t):
            return t

        def unpack_hook(t):
            return t

        a = torch.tensor(2.0, requires_grad=True)

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            b = a**2

        c = b.exp()
        d = c**2

        pow_sv = b.grad_fn._raw_saved_self
        exp_sv = c.grad_fn._raw_saved_result
        pow2_sv = d.grad_fn._raw_saved_self

        # Returns the packed object as-is
        self.assertTrue(pow_sv.data is a)
        self.assertTrue(pow_sv.unpack_hook is unpack_hook)
        # Returns the detached data when the output/leaf is saved
        self.assertFalse(exp_sv.data is c)
        self.assertIsNone(exp_sv.unpack_hook)
        # Returns the un-detached data when input is saved
        self.assertTrue(pow2_sv.data is c)
        self.assertIsNone(pow2_sv.unpack_hook)

    def test_saved_tensor_constructor_with_hooks(self):
        pack_count = [0]
        unpack_count = [0]

        def pack_hook(x):
            pack_count[0] += 1
            return x

        def unpack_hook(x):
            unpack_count[0] += 1
            return x

        a = torch.randn(5, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            saved = torch._C._autograd._make_saved_tensor(a, is_output=False)
            self.assertEqual(pack_count[0], 1)
            self.assertEqual(unpack_count[0], 0)

            unpacked = saved.unpack()
            self.assertEqual(pack_count[0], 1)
            self.assertEqual(unpack_count[0], 1)

            unpacked2 = saved.unpack()
            self.assertEqual(pack_count[0], 1)
            self.assertEqual(unpack_count[0], 2)

        # Check tensor equality outside the hooks context to avoid
        # triggering additional pack hooks from assertEqual internals
        self.assertEqual(unpacked, a)
        self.assertEqual(unpacked2, a)

    def test_saved_tensor_constructor_forbidden_without_flag(self):
        a = torch.randn(5, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "Trying to create a SavedTensor object from Python is forbidden",
        ):
            torch.autograd.SavedTensor()

    def test_custom_function_saved_tensors(self):
        def getFn(save=True):
            class MyFn(Function):
                @staticmethod
                def forward(ctx, x):
                    if save:
                        ctx.save_for_backward(x, None)
                    return x

                @staticmethod
                def backward(ctx, g):
                    return g

            return MyFn

        a = torch.randn(5, requires_grad=True)

        y = getFn(True).apply(a)

        self.assertEqual((a, None), y.grad_fn.saved_tensors)
        saved = y.grad_fn._raw_saved_tensors
        self.assertIsInstance(saved[0], torch._C._autograd.SavedTensor)
        # We can't tell the underlying tensor is None without unpacking it
        self.assertIsInstance(saved[1], torch._C._autograd.SavedTensor)

        # We catch that error when the user calls register_hooks on it
        with self.assertRaisesRegex(RuntimeError, "None is forbidden"):
            saved[1].register_hooks(lambda x: x, lambda x: x)

        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            saved[0].register_hooks(lambda x: x)
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            saved[0].register_hooks(1, 1)
        saved[0].register_hooks(lambda x: x, lambda x: x)
        with self.assertRaisesRegex(RuntimeError, "already been set"):
            saved[0].register_hooks(lambda x: x, lambda x: x)
        y.sum().backward()

        # Using a reference to the SavedTensor object after the
        # saved variables have been released can lead to undefined behavior
        del saved
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            y.grad_fn._raw_saved_tensors
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            y.grad_fn.saved_tensors

        y = getFn(False).apply(a)
        self.assertEqual(y.grad_fn.saved_tensors, ())
        self.assertEqual(y.grad_fn._raw_saved_tensors, ())

    @skipIfTorchDynamo("dynamo accesses saved_tensors multiple times")
    def test_clear_saved_tensors_on_access(self):
        class MyFn(Function):
            clear_saved_tensors_on_access = True

            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                ref = weakref.ref(x)
                del x
                # Local variable should be the only remaining reference
                self.assertIsNone(ref())
                return grad_output

        x = torch.randn(3, requires_grad=True)
        y = MyFn.apply(x.clone())
        y.sum().backward()

    @skipIfTorchDynamo("test tests an error that dynamo does not reproduce")
    def test_clear_saved_tensors_on_access_double_access_error(self):
        class MyFn(Function):
            clear_saved_tensors_on_access = True

            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                _ = ctx.saved_tensors  # first access
                _ = ctx.saved_tensors  # second access - should raise
                return grad_output

        x = torch.randn(3, requires_grad=True)
        y = MyFn.apply(x)
        with self.assertRaisesRegex(RuntimeError, "can only be accessed once"):
            y.sum().backward()

    def test_autograd_node_isinstance(self):
        # Node is a "virtual" base class of codegen'd nodes. This means that
        # isinstance and issubclass are overridden, but mro is unchanged
        Node = torch.autograd.graph.Node

        a = torch.rand(3, 3, requires_grad=True)
        b = a.exp()

        # Some nodes have codegened registrations to the torch._C._function module
        self.assertIsInstance(b.grad_fn, Node)
        self.assertTrue(issubclass(type(b.grad_fn), Node))
        self.assertTrue(Node not in type(b.grad_fn).mro())

        # Other nodes have manual registrations to the torch._C._function module
        self.assertNotIsInstance(torch._C._functions.AccumulateGrad, Node)
        self.assertTrue(issubclass(torch._C._functions.AccumulateGrad, Node))
        self.assertIsInstance(b.grad_fn.next_functions[0][0], Node)
        self.assertTrue(issubclass(torch._C._functions.DelayedError, Node))

        # Special cases
        self.assertNotIsInstance(None, Node)
        self.assertNotIsInstance(1, Node)
        self.assertNotIsInstance(Node, Node)
        self.assertTrue(issubclass(Node, Node))

        # Custom function case
        self.assertTrue(issubclass(torch.autograd.function.BackwardCFunction, Node))

        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                self.assertIsInstance(ctx, Node)
                return x

            @staticmethod
            def backward(ctx, x):
                self.assertIsInstance(ctx, Node)
                return x

        out = Func.apply(a)
        self.assertIsInstance(out.grad_fn, Node)
        self.assertTrue(issubclass(type(out.grad_fn), Node))
        self.assertTrue(Node not in type(out.grad_fn).mro())
        out.sum().backward()

    def test_autograd_views_codegen(self):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This test checks the behavior of two codegen functions (view_as and unbind)
        # with respect to view tracking and inplace operation on the output.

        def run_test(grad_mode, requires_grad, is_view, should_raise_tuple):
            def maybe_check_raise(fn, should_raise):
                self.assertTrue(should_raise is None or isinstance(should_raise, str))
                if should_raise is not None:
                    with self.assertRaisesRegex(RuntimeError, should_raise):
                        fn()
                else:
                    fn()

            inp = torch.rand(2, requires_grad=requires_grad).clone()
            with torch.set_grad_enabled(grad_mode):
                out = inp.view_as(inp)
            # Are they differentiable views?
            self.assertTrue(out._is_view() == is_view)
            # Are inplace allowed?
            maybe_check_raise(lambda: out.add_(1), should_raise_tuple[0])

            inp = torch.rand(2, requires_grad=requires_grad).clone()
            with torch.set_grad_enabled(grad_mode):
                out = inp.unbind()
            # Are they differentiable views?
            self.assertTrue(out[0]._is_view() == is_view)
            self.assertTrue(out[1]._is_view() == is_view)
            # Are inplace allowed?
            maybe_check_raise(lambda: out[0].add_(1), should_raise_tuple[1])
            maybe_check_raise(lambda: out[1].add_(1), should_raise_tuple[2])

        # should_raise contains None if it should not raise
        # should_raise contains a string of the error if it should raise
        # The 3 elements are for view_as, first output of unbind and second output of unbind
        run_test(
            grad_mode=True,
            requires_grad=False,
            is_view=True,
            should_raise_tuple=(None, None, None),
        )
        inp_change_err = (
            "Output {} of UnbindBackward0 is a view and is being modified inplace."
        )
        run_test(
            grad_mode=True,
            requires_grad=True,
            is_view=True,
            should_raise_tuple=(
                None,
                inp_change_err.format("0"),
                inp_change_err.format("1"),
            ),
        )
        leaf_grad_err = (
            "A view was created in no_grad mode and is being modified inplace"
        )
        run_test(
            grad_mode=False,
            requires_grad=True,
            is_view=True,
            should_raise_tuple=(leaf_grad_err, leaf_grad_err, leaf_grad_err),
        )
        run_test(
            grad_mode=False,
            requires_grad=False,
            is_view=True,
            should_raise_tuple=(None, None, None),
        )

    def test_inplace_not_requires_grad(self):
        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx, grad):
                return grad

        # Original Tensor does not require grad
        a = torch.rand(1, 2)

        # Tensor being written does require grad
        b = torch.rand(1, requires_grad=True)

        # Take an invalid view on 'a' that should raise an error (warns during deprecation)
        view_a = MyFn.apply(a)

        with self.assertRaisesRegex(
            RuntimeError, "This view was created inside a custom Function"
        ):
            view_a += b

        # Extra test for copy_ that is a manual implementation and could be easily
        # forgotten when the codegen is updated (warns during deprecation)
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = MyFn.apply(a)

        with self.assertRaisesRegex(
            RuntimeError, "This view was created inside a custom Function"
        ):
            view_a.copy_(b)

        # Functions that should throw must properly throw
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = a.unbind()[0]
        with self.assertRaisesRegex(
            RuntimeError,
            "This view is the output of a function that returns multiple views.",
        ):
            view_a.copy_(b)

        # Sanity check that views that should work still work
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        a.select(1, 0).copy_(b)

    def _do_test_autograd_simple_views_python(self, dtype):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This checks the autograd.Function behavior when we return one or multiple outputs
        # while one of these is an input, a view of an input or of a temporary tensor.

        # This indicator is used to track how many times the backward function was called
        bw_called = [0]
        # This indicator is used to check if the argument `ga` contains non-zero values
        ga_nz = [False]

        class IdOneOutput(Function):
            @staticmethod
            def forward(ctx, a, make_view, pure_view):
                ctx._is_pure_view = pure_view
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                return a

            @staticmethod
            def backward(ctx, ga):
                bw_called[0] += 1
                return ga, None, None

        class IdTwoOutput(Function):
            @staticmethod
            def forward(ctx, a, b, make_view, pure_view):
                ctx._is_pure_view = pure_view
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                return a, a + b

            @staticmethod
            def backward(ctx, ga, gab):
                bw_called[0] += 1
                if ga.eq(0).all():
                    ga_nz[0] = False
                else:
                    ga_nz[0] = True
                return ga + gab, gab, None, None

        class ViewOfTemp(Function):
            @staticmethod
            def forward(ctx, a, make_view, pure_view):
                ctx._is_pure_view = pure_view
                ctx.save_for_backward(a)
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                b = a.clone()
                return b.select(0, 0)

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                (a,) = ctx.saved_tensors
                res = torch.zeros_like(a)
                res.select(0, 0).copy_(grad)
                return res, None, None

        fn_id_to_inplace_on_view_err_msg = {
            "one_output": (
                "Output 0 of IdOneOutputBackward is a view and is being "
                "modified inplace. This view was created inside a custom Function"
            ),
            "two_output": (
                "Output 0 of IdTwoOutputBackward is a view and is being modified inplace."
                " This view is the output of a function that returns multiple views.",
                "Pure view custom Function can only have one input Tensor and one output Tensor."
                " Open an issue if you need to support more.",
            ),
            "view_of_temp": (
                "Output 0 of ViewOfTempBackward is a view and is being "
                "modified inplace. This view was created inside a custom Function",
                "a view of a leaf Variable that requires grad is being used in an in-place operation",
            ),
        }

        for fn_id in ["one_output", "two_output", "view_of_temp"]:
            for inplace in [True, False]:
                for make_view in [True, False]:
                    for pure_view in [True, False]:
                        # Used for special casing the tests below
                        output_is_a_view = make_view or fn_id == "view_of_temp"

                        def fn(a, b):
                            # never modify a, b inplace for gracheck
                            a = a.clone()
                            b = b.clone()
                            if fn_id == "two_output":
                                tmp1, tmp2 = IdTwoOutput.apply(
                                    a, b, make_view, pure_view
                                )
                                if inplace:
                                    tmp1 += 3
                                    tmp2 += 3
                                else:
                                    tmp1 = tmp1 + 3
                                    tmp2 = tmp2 + 3
                                tmp = tmp1 * tmp2
                            else:
                                if fn_id == "one_output":
                                    tmp = IdOneOutput.apply(a, make_view, pure_view)
                                else:
                                    tmp = ViewOfTemp.apply(a + b, make_view, pure_view)
                                if inplace:
                                    tmp += 3
                                else:
                                    tmp = tmp + 3

                            return tmp.sum()

                        a = torch.ones(2, dtype=dtype, requires_grad=True)
                        b = torch.ones(2, dtype=dtype, requires_grad=True)

                        err_msg = fn_id_to_inplace_on_view_err_msg[fn_id][
                            int(pure_view)
                        ]

                        will_raise_error = (
                            (pure_view and fn_id == "two_output")
                            or (pure_view and fn_id == "view_of_temp" and inplace)
                            or (not pure_view and inplace and output_is_a_view)
                        )

                        if will_raise_error:
                            with self.assertRaisesRegex(RuntimeError, err_msg):
                                gradcheck(fn, (a, b), check_batched_grad=False)
                        else:
                            gradcheck(fn, (a, b), check_batched_grad=False)

                        # Was the custom backward called properly
                        bw_called[0] = 0
                        ga_nz[0] = True  # For the case where the backward is called

                        expected_called = 1
                        expected_ga_nz = True

                        if will_raise_error:
                            expected_called = 0
                            with self.assertRaisesRegex(RuntimeError, err_msg):
                                fn(a, b)
                        else:
                            fn(a, b).abs().backward()

                        if (
                            fn_id == "one_output"
                            and inplace
                            and output_is_a_view
                            and pure_view
                        ):
                            # We expect the op to have been replayed and we leveraged the pure view
                            # to re-create the graph, so the original backward was not called
                            expected_called = 0

                        self.assertTrue(bw_called[0] == expected_called)
                        self.assertTrue(ga_nz[0] == expected_ga_nz)

    def test_autograd_simple_views_python(self):
        self._do_test_autograd_simple_views_python(torch.double)
        self._do_test_autograd_simple_views_python(torch.cdouble)

    def test_autograd_inplace_views_creation_meta(self):
        # Tests creation_meta properly handled for inplace views

        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.view_as(x)

            @staticmethod
            def backward(ctx, x):
                return x

        view_custom = Func.apply

        def run_test(
            fn, fn_type, grad_mode_view, grad_mode_iview, requires_grad, error1, error2
        ):
            # This test checks the behavior of inplace-view functions when
            # the views are created in grad mode or not
            base = torch.rand(2, 3, requires_grad=requires_grad).clone()
            # 1. Create a view with `grad_mode=grad_mode_view`
            with torch.set_grad_enabled(grad_mode_view):
                if fn_type == "multi_view":
                    inp = base.unbind()[0]
                elif fn_type == "custom":
                    inp = view_custom(base)
                else:
                    inp = base.view_as(base)

            # 2. Perform inplace view with `grad_mode=grad_mode_iview`
            with torch.set_grad_enabled(grad_mode_iview):
                if error1 is not None:
                    with self.assertRaisesRegex(RuntimeError, error1):
                        fn(inp)
                    return
                else:
                    # If error is None, check that runs without error
                    fn(inp)
            # 3. Do inplace on the (new) view
            if error2 is not None:
                with self.assertRaisesRegex(RuntimeError, error2):
                    inp.add_(1)
            else:
                # If error is None, check that runs without error
                inp.add_(1)

        no_grad_err = "A view was created in no_grad mode"
        multi_view_err = "function that returns multiple views"
        custom_err = "view was created inside a custom Function"

        def run_tests(fn):
            for fn_type in ("normal", "multi_view", "custom"):
                for grad_mode_view in (True, False):
                    for grad_mode_iview in (True, False):
                        for requires_grad in (True, False):
                            error1 = None  # expected error when we do inplace_view on original view
                            error2 = None  # expected error when we do inplace on the resulting view

                            if requires_grad:
                                if not grad_mode_view and grad_mode_iview:
                                    error1 = no_grad_err
                                if not grad_mode_view and not grad_mode_iview:
                                    error2 = no_grad_err

                                if fn_type == "multi_view":
                                    if grad_mode_view and grad_mode_iview:
                                        error1 = multi_view_err
                                    if grad_mode_view and not grad_mode_iview:
                                        error2 = multi_view_err

                                if fn_type == "custom":
                                    if grad_mode_view and grad_mode_iview:
                                        error1 = custom_err
                                    if grad_mode_view and not grad_mode_iview:
                                        error2 = custom_err

                            run_test(
                                fn,
                                fn_type,
                                grad_mode_view,
                                grad_mode_iview,
                                requires_grad,
                                error1,
                                error2,
                            )

        # This list was created by logging gen_inplace_or_view_type.py
        #   detach_ is excluded for this test because it cannot be applied to
        #   views and thus does not return a view
        run_tests(lambda v: v.as_strided_((1, 0), (2, 2)))
        run_tests(lambda v: v.transpose_(0, 0))
        run_tests(lambda v: v.t_())
        run_tests(lambda v: v.squeeze_(0))
        run_tests(lambda v: v.unsqueeze_(0))
        run_tests(lambda v: v.swapdims_(0, 0))
        run_tests(lambda v: v.swapaxes_(0, 0))

    def test_autograd_print_tensor(self):
        a = torch.ones(1, requires_grad=True)
        a_clone = a.clone()
        self.assertEqual(repr(a), "tensor([1.], requires_grad=True)")
        self.assertEqual(repr(a_clone), "tensor([1.], grad_fn=<CloneBackward0>)")

        with torch.no_grad():
            b = a[:]
            b *= 2

        # Special handling for printing view created in no-grad and modified
        # in-placed in no-grad.
        self.assertEqual(repr(b), "tensor([2.], grad_fn=<Invalid>)")

        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, x):
                return x

        c = Func.apply(a)
        self.assertEqual(repr(c), "tensor([2.], grad_fn=<FuncBackward>)")

    def test_autograd_inplace_view_of_view(self):
        x = torch.zeros(2)
        with torch.no_grad():
            y = x.view(2)
        y.requires_grad_(True)
        z = y.view(2)
        with self.assertRaisesRegex(
            RuntimeError, "a view of a view .* is being .* inside the no_grad block"
        ):
            z /= 2

        x = torch.zeros(2)
        with torch.inference_mode():
            y = x.view(2)
        y.requires_grad_(True)
        z = y.view(2)
        with self.assertRaisesRegex(
            RuntimeError, "a view of a view .* is being .* inside the inference_mode"
        ):
            z /= 2

    # TODO This is not the correct behavior -
    # See https://github.com/pytorch/pytorch/issues/49825#issuecomment-794466627
    def test_autograd_inplace_views_cross_dtype(self):
        # This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        a_orig = torch.rand(3, 3, requires_grad=True, dtype=torch.complex64)
        a = a_orig.clone()
        b = torch.view_as_real(a)
        b = b.transpose(0, 1)
        b += 1
        b.backward(torch.arange(0, 18, dtype=torch.float).view(3, 3, 2))
        non_inplace_grad = a_orig.grad

        a_orig = torch.rand(3, 3, requires_grad=True, dtype=torch.complex64)
        a = a_orig.clone()
        b = torch.view_as_real(a)
        b.transpose_(0, 1)
        b += 1
        b.backward(torch.arange(0, 18, dtype=torch.float).view(3, 3, 2))
        inplace_grad = a_orig.grad

        # TODO: this is a bug!
        # once this is fixed, it should have the transpose removed:
        # self.assertEqual(non_inplace_grad, inplace_grad)
        self.assertEqual(non_inplace_grad.T, inplace_grad)

    def test_autograd_multiple_views_python(self):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This checks that multiples views in the forward are properly traced and how they
        # behave with respect to inplace operations.

        # This indicator is used to track how many times the backward function was called
        bw_called = [0]

        class ComplexView(Function):
            @staticmethod
            def forward(ctx, a, idx):
                res = a.narrow(0, idx, 1)
                res = a.select(0, idx)
                ctx.save_for_backward(a)
                ctx.idx = idx
                return res

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                (a,) = ctx.saved_tensors
                res = torch.zeros_like(a)
                res.select(0, ctx.idx).copy_(grad)
                return res, None

        a = torch.ones(2, requires_grad=True)
        idx = 1

        bw_called[0] = 0
        out = ComplexView.apply(a.clone(), idx)
        out.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        out = ComplexView.apply(a.clone(), idx)
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of ComplexViewBackward is a view and is being modified inplace",
        ):
            out += 1

    def test_autograd_python_custom_function_inplace(self):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This test checks custom autograd.Function that perform inplace operations

        bw_called = [0]

        # I) Single output
        class MyAdder(Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                return grad, grad

        a = torch.ones(2, requires_grad=True)
        b = torch.ones(2, requires_grad=True)

        # No extra inplace
        c = MyAdder.apply(a.clone(), b)
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # With extra inplace on the output
        bw_called[0] = 0
        c = MyAdder.apply(a.clone(), b)
        c += 2
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # The input is a view
        bw_called[0] = 0
        c = MyAdder.apply(a.clone().view_as(a), b)
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # Should not give non-inputs to mark_dirty
        class MyAdderBad(Function):
            @staticmethod
            def forward(ctx, a, b):
                c = 3 * a
                c.add_(b)
                ctx.mark_dirty(c)
                return c

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                grad = 3 * grad
                return grad, grad

        a = torch.ones(2, requires_grad=True)
        b = torch.ones(2, requires_grad=True)

        with warnings.catch_warnings(record=True) as w:
            MyAdderBad.apply(a.clone(), b)
        self.assertEqual(len(w), 1)

        # II) Multiple outputs
        class MyBadAdder(Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a, a + b

            @staticmethod
            def backward(ctx, ga, gab):
                bw_called[0] += 1
                return ga + gab, ga + gab

        # No extra inplace
        bw_called[0] = 0
        c, d = MyBadAdder.apply(a.clone(), b)
        (c * d).sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # With extra inplace on the output
        bw_called[0] = 0
        c, d = MyBadAdder.apply(a.clone(), b)
        c += 2
        (c * d).sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # The input is a view
        inplace_on_view_err = (
            "your Function modifies inplace an input that is a view of another Tensor"
        )
        with self.assertRaisesRegex(RuntimeError, inplace_on_view_err):
            c, d = MyBadAdder.apply(a.clone().view_as(a), b)

        # III) Inplace + other op
        class MyOutPlaceAdder(Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a.clone(), a + b

            @staticmethod
            def backward(ctx, ga, gab):
                bw_called[0] += 1
                return ga + gab, ga + 2 * gab

        # We don't reuse the input
        def fn(a, b):
            orig_a = a.clone().view_as(a)
            c, d = MyOutPlaceAdder.apply(orig_a, b)
            return (c * d).sum()

        bad_mark_dirty_err = "Some elements marked as dirty during the forward method were not returned as output."
        with self.assertRaisesRegex(RuntimeError, bad_mark_dirty_err):
            fn(a, b)

    def test_custom_function_mark_dirty_not_differentiable(self):
        def get_custom_fn(jvp_err):
            class InplaceMul(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    result = x.mul_(2)
                    ctx.mark_dirty(result)
                    return result

                @staticmethod
                def backward(ctx, grad_output):
                    pass

                @staticmethod
                def jvp(ctx, x_t):
                    if jvp_err:
                        return x_t
                    else:
                        return x_t.mul_(2)

            return InplaceMul

        for requires_grad, jvp_err in product([True, False], repeat=2):
            InplaceMul = get_custom_fn(jvp_err)
            # Make sure that tensor is always returned as-is if marked dirty
            z = torch.tensor(1.0, requires_grad=requires_grad)
            x = z.clone()
            y = InplaceMul.apply(x)
            self.assertTrue(x is y)
            self.assertEqual(x, z * 2)

            # jvp must properly modify the input grad if mark_dirty is set
            with fwAD.dual_level():
                x_tangent = torch.ones_like(x)
                x_dual = fwAD.make_dual(x, x_tangent)

                if jvp_err:
                    bad_mark_dirty_err = (
                        "jvp function must modify the corresponding gradient inplace"
                    )
                    with self.assertRaisesRegex(RuntimeError, bad_mark_dirty_err):
                        InplaceMul.apply(x_dual)
                else:
                    out_dual = InplaceMul.apply(x_dual)
                    _, out_tangent = fwAD.unpack_dual(out_dual)
                    self.assertTrue(out_dual is x_dual)
                    self.assertTrue(out_tangent is x_tangent)

    def test_custom_function_mark_output_view_of_intermediate(self):
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                out = inp.clone().view_as(inp)
                ctx.mark_dirty(out)
                return out

            @staticmethod
            def backward(ctx, gO):
                pass

        a = torch.tensor([1.0], requires_grad=True)
        a_clone = a.clone()

        with self.assertRaisesRegex(
            RuntimeError, "received a tensor that was not an input."
        ):
            Func.apply(a_clone)

    def test_custom_function_inplace_on_non_default_view(self):
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                inp.add_(1)
                ctx.mark_dirty(inp)
                return inp

            @staticmethod
            def backward(ctx, gO):
                pass

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        a_clone = a.clone()
        b, c = a.split_with_sizes([1, 1], dim=0)

        with self.assertRaisesRegex(
            RuntimeError, "output of a function that returns multiple view"
        ):
            Func.apply(b)

    def test_custom_function_inplace_on_view_of_leaf(self):
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                inp.add_(1)
                ctx.mark_dirty(inp)
                return inp

            @staticmethod
            def backward(ctx, gO):
                pass

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = a.view_as(a)

        with self.assertRaisesRegex(
            RuntimeError, "a view of a leaf Variable that requires grad"
        ):
            Func.apply(b)

    def test_named_tensor_for_complex_views(self):
        names = ["batch", "height", "width", "complex"]
        z = torch.ones((2, 1, 2, 2), requires_grad=True)
        z_named = z.refine_names(*names)
        z_complex = torch.view_as_complex(z_named.rename(None)).refine_names(
            *names[:-1]
        )
        z_complex.sum().abs().backward()
        expected = torch.ones_like(z_complex).rename(None)
        abs_1_1j = abs(1 + 1j)
        expected.fill_(complex(abs_1_1j / 2, abs_1_1j / 2))
        self.assertEqual(z.grad, torch.view_as_real(expected))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO and sys.version_info >= (3, 14), "Fails in python 3.14.2"
    )
    def test_custom_function_saving_mutated_view_no_leak(self):
        class Test(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.mark_dirty(x)
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, grad):
                pass

        def scope():
            x = torch.tensor(1.0, requires_grad=True).clone()
            x = x.view_as(x)
            y = Test.apply(x)
            return weakref.ref(x)

        ref = scope()
        self.assertIsNone(ref())

    def test_custom_function_return_view_in_nograd(self):
        class Alias(Function):
            @staticmethod
            def forward(ctx, x):
                return x[:]

            @staticmethod
            def backward(ctx, gx):
                return gx

        inp = torch.rand(2, requires_grad=True)

        with torch.no_grad():
            output = Alias.apply(inp)

        with torch.no_grad():
            expected_output = inp[:]

        # Calling the custom function should operate as if we called an equivalent op
        self.assertEqual(output.requires_grad, expected_output.requires_grad)

        # Check that in-place modification on view throws
        leaf_grad_err = (
            "A view was created in no_grad mode and is being modified inplace"
        )
        with self.assertRaisesRegex(RuntimeError, leaf_grad_err):
            output.zero_()

    def test_custom_function_preserve_torch_function_when_return_as_is(self):
        class Custom(torch.Tensor):
            def __init__(self, data):
                super().__init__()
                self._data = data

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                kwargs = {} if kwargs is None else kwargs
                args = tuple(a._data if isinstance(a, cls) else a for a in args)
                out = func(*args, **kwargs)
                if isinstance(out, torch.Tensor):
                    out = cls(out)
                return out

        class Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx):
                pass

        x = Custom(torch.randn(2, 3))
        y = Fn.apply(x)
        self.assertTrue(isinstance(y, Custom))

    def test_grad_mode_restored_reentrant(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, go):
                original = torch._C.is_grad_enabled()
                with torch.enable_grad():
                    self.assertTrue(torch._C.is_grad_enabled())
                    foo = torch.rand(go.size(), requires_grad=True)
                    (grad,) = torch.autograd.grad(foo**3, foo, grad_outputs=go)
                    self.assertTrue(torch._C.is_grad_enabled())
                self.assertTrue(torch._C.is_grad_enabled() == original)
                return grad

        inp = torch.rand(3, requires_grad=True)

        # Case where original==False
        MyFunction.apply(inp).sum().backward()
        # Case where original==True
        MyFunction.apply(inp).sum().backward(create_graph=True)

    def test_power_function(self):
        a = torch.tensor([0.0, 0.0, 0.0])
        b = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        c = torch.sum(a**b)
        c.backward()
        self.assertEqual(b.grad, torch.tensor([-inf, 0.0, 0.0]))

        s = 0
        b = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        c = torch.sum(s**b)
        c.backward()
        self.assertEqual(b.grad, torch.tensor([-inf, 0.0, 0.0]))

    def test_custom_function_error(self):
        class BadFw(Function):
            @staticmethod
            def backward(ctx, foo):
                return foo

        class BadBw(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo.clone()

        class BadBw2(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo.clone()

            @staticmethod
            def backward(ctx, foo):
                return foo

            @staticmethod
            def vjp(ctx, foo):
                return foo

        class BadJvp(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo.clone()

        inp = torch.rand(1, requires_grad=True)
        with self.assertRaisesRegex(NotImplementedError, "must implement the forward"):
            BadFw.apply(inp)

        with self.assertRaisesRegex(RuntimeError, "must implement either the backward"):
            BadBw.apply(inp).sum().backward()

        with self.assertRaisesRegex(
            RuntimeError, "Implementing both 'backward' and 'vjp'"
        ):
            BadBw2.apply(inp).sum().backward()

        with self.assertRaisesRegex(RuntimeError, "must implement the jvp function"):
            with fwAD.dual_level():
                d = fwAD.make_dual(inp, torch.rand_like(inp))
                res = BadJvp.apply(d)

    def test_custom_function_forward_mode_view_checks(self):
        flag_to_error = {
            "ok": None,
            "not_a_view": "jvp is not returning a view",
            "not_a_view_of_inp": "jvp is not returning a view of the given",
            "not_a_view_of_inp_base": "jvp is not returning a view of the same base",
        }

        class ViewFn(Function):
            @staticmethod
            def forward(ctx, foo, flag):
                ctx.flag = flag
                ctx.size = foo.size()
                return foo.narrow(0, 0, 2)

            @staticmethod
            def vjp(ctx, gO):
                gI = gO.new_zeros(ctx.size)
                gI.narrow(0, 0, 2).copy_(gO)
                return gI, None

            @staticmethod
            def jvp(ctx, gI, _):
                res = gI.narrow(0, 0, 2)
                if ctx.flag != "ok":
                    # Break the view in the gradients!
                    res = res.clone()
                if ctx.flag in ["not_a_view_of_inp", "not_a_view_of_inp_base"]:
                    # Result should be a view, just of the wrong thing
                    res = res.view_as(res)
                return res

        inp = torch.rand(4, 4, dtype=torch.double, requires_grad=True)

        for flag, msg in flag_to_error.items():

            def test_fn(inp):
                if flag == "not_a_view_of_inp_base":
                    inp = inp.view_as(inp)
                return ViewFn.apply(inp, flag)

            if msg is None:
                gradcheck(test_fn, inp, check_forward_ad=True)
            else:
                with self.assertRaisesRegex(RuntimeError, msg):
                    gradcheck(test_fn, inp, check_forward_ad=True)

    def test_custom_function_forward_mode_inplace_checks(self):
        class InplaceFn(Function):
            @staticmethod
            def forward(ctx, foo, flag):
                ctx.mark_dirty(foo)
                ctx.flag = flag
                foo.mul_(2)
                return foo

            @staticmethod
            def vjp(ctx, gO):
                return 2 * gO, None

            @staticmethod
            def jvp(ctx, gI, _):
                if ctx.flag:
                    # Don't do the change inplace
                    return 2 * gI
                else:
                    gI.mul_(2)
                    return gI

        inp = torch.rand(4, 4, dtype=torch.double, requires_grad=True)

        def test_fn(inp, flag):
            inp = inp.clone()
            return InplaceFn.apply(inp, flag)

        gradcheck(test_fn, (inp, False), check_forward_ad=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "inplace custom Function is not modifying the forward mode gradients inplace",
        ):
            gradcheck(test_fn, (inp, True), check_forward_ad=True)

    def test_custom_function_forward_mode_wrong_formula(self):
        class UserFn(Function):
            @staticmethod
            def forward(ctx, foo, should_fail):
                ctx.should_fail = should_fail
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                return 2 * gO, None

            @staticmethod
            def jvp(ctx, gI, _):
                if ctx.should_fail:
                    # Wrong gradient formula
                    return 3 * gI
                else:
                    return 2 * gI

        inp = torch.rand(10, dtype=torch.double, requires_grad=True)
        gradcheck(UserFn.apply, (inp, False), check_forward_ad=True)

        with self.assertRaisesRegex(
            RuntimeError, "Jacobian computed with forward mode mismatch for output 0"
        ):
            gradcheck(UserFn.apply, (inp, True), check_forward_ad=True)

    def test_custom_function_forward_mode_non_tensor_before_tensor_args(self):
        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, nt, x, nt2, y):
                return x * 2 + y * 3

            @staticmethod
            def jvp(ctx, nt, x_t, nt2, y_t):
                self.assertIsNone(nt)
                self.assertIsNone(nt2)
                return x_t * 2 + y_t * 3

        x = torch.tensor(1.0, dtype=torch.double)
        t = torch.tensor(1.0, dtype=torch.double)
        y = torch.tensor(1.0, dtype=torch.double)

        with fwAD.dual_level():
            dual_x = fwAD.make_dual(x, t)
            MyFn.apply(1, dual_x, 1, y)

        gradcheck(
            MyFn.apply,
            (1, x.requires_grad_(True), 1, y.requires_grad_(True)),
            check_forward_ad=True,
            check_backward_ad=False,
            check_batched_grad=False,
        )

    def test_custom_function_forward_mode_forward_is_no_op(self):
        error_regex = (
            "A custom Function's forward is returning a view \\(or an input as-is\\)"
        )

        return_lambdas = {
            # If we return an input as-is in forward, that is treated
            # as if self.view_as(self) is performed. If jvp returns x.view_as(x),
            # this is OK.
            "view_as": lambda x: x.view_as(x),
            # Expect this to raise an error
            "self": lambda x: x,
            # Expect this to raise the same error
            "mul_by_2": lambda x: x * 2,
        }

        for k, fn in return_lambdas.items():

            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y):
                    return x + y, x

                @staticmethod
                def vjp(ctx, gO1, gO2):
                    return gO1 + gO2, gO1

                @staticmethod
                def jvp(ctx, x_t, y_t):
                    return x_t + y_t, fn(x_t)

            a = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            t = torch.tensor(1.0, dtype=torch.double)
            b = torch.tensor(1.0, dtype=torch.double, requires_grad=True)

            c = torch.tensor(1.0, dtype=torch.double)
            t2 = torch.tensor(1.0, dtype=torch.double)
            d = torch.tensor(1.0, dtype=torch.double)

            with fwAD.dual_level():
                a_dual = fwAD.make_dual(a, t)
                c_dual = fwAD.make_dual(c, t2)

                if k == "view_as":
                    _, out2 = MyFn.apply(a_dual, b)
                    self.assertTrue(fwAD.unpack_dual(out2).tangent._base is t)

                    _, out2 = MyFn.apply(c_dual, d)
                    self.assertTrue(fwAD.unpack_dual(out2).tangent._base is t2)
                else:
                    with self.assertRaisesRegex(RuntimeError, error_regex):
                        MyFn.apply(a_dual, b)

                    with self.assertRaisesRegex(RuntimeError, error_regex):
                        MyFn.apply(c_dual, d)

            if k == "view_as":
                gradcheck(MyFn.apply, (a, c), check_forward_ad=True)
            else:
                with self.assertRaisesRegex(RuntimeError, error_regex):
                    gradcheck(MyFn.apply, (a, c), check_forward_ad=True)

    def test_custom_function_save_for_forward(self):
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
                ctx.save_for_backward(x, y)
                ctx.save_for_forward(x, y)
                ctx.z = z
                ctx.prod = x * y
                return z * ctx.prod

            @staticmethod
            def jvp(ctx, x_t, y_t, _):
                x_p, y_p = ctx.saved_tensors
                z = ctx.z
                return z * (y_p * x_t + x_p * y_t)

            @staticmethod
            def vjp(ctx, grad_out):
                x, y = ctx.saved_tensors
                z = ctx.z
                return z * grad_out * y, z * grad_out * x, None

        a = torch.tensor(1.0, requires_grad=True, dtype=torch.double)
        t = torch.tensor(1.0, dtype=torch.double)
        b = torch.tensor(2.0, requires_grad=True, dtype=torch.double)
        c = 4

        with fwAD.dual_level():
            a_dual = fwAD.make_dual(a, t)
            out = Func.apply(a_dual, b, c)
            out.backward()

        gradcheck(Func.apply, (a, b, c), check_forward_ad=True)

        # When saved for backward, but not saved for forward
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x: torch.Tensor):
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def jvp(ctx, x_t):
                self.assertEqual(len(ctx.saved_tensors), 0)
                return x_t

            @staticmethod
            def vjp(ctx, grad_out):
                (x,) = ctx.saved_tensors
                self.assertEqual(len(ctx.saved_tensors), 1)
                return grad_out

        with fwAD.dual_level():
            a_dual = fwAD.make_dual(a, t)
            out = Func.apply(a_dual)
            out.backward()

        gradcheck(Func.apply, (a,), check_forward_ad=True)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_autograd_function.py")
    def test_custom_function_forward_mode_non_differentiable(self):
        # returns differentiable type, marked non-differentiable
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                out = y.clone()
                ctx.mark_non_differentiable(out)
                return x.clone(), out

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                return x_tangent, None

        x = torch.tensor(2.0)
        x_tangent = torch.tensor(1.0)
        y = torch.tensor(3.0)

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            _, out2_dual = Func.apply(x_dual, y)
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, None)

        y = torch.tensor(3)

        # returns non-differentiable type, NOT marked non-differentiable
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                self.assertIsNone(y_tangent)
                return x_tangent, None

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            _, out2_dual = Func.apply(x_dual, y)
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, None)

        class FuncWrong(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                out = y.clone()
                ctx.mark_non_differentiable(out)
                return x.clone(), out

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                return x_tangent, x_tangent.clone()

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            with self.assertRaisesRegex(
                RuntimeError, "You should return None at that position instead"
            ):
                FuncWrong.apply(x_dual, y)

        # returns non-tensor
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone(), object(), x.clone()

            @staticmethod
            def jvp(ctx, x_tangent):
                return x_tangent, None, x_tangent

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            out_dual, _, out2_dual = Func.apply(x_dual)
            self.assertEqual(fwAD.unpack_dual(out_dual).tangent, x_tangent)
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, x_tangent)

    def test_custom_function_local_inplace(self):
        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp, inplace):
                view = inp.clone()[:3]
                if inplace:
                    view += 2
                return view

            @staticmethod
            def backward(ctx, grad):
                return grad, None

        base = torch.rand(10, requires_grad=True)

        foo = MyFn.apply(base, False)
        self.assertEqual(foo.grad_fn.__class__.__name__, "MyFnBackward")

        foo = MyFn.apply(base, True)
        self.assertEqual(foo.grad_fn.__class__.__name__, "MyFnBackward")

    def test_integer_outputs(self):
        inp = torch.rand(4, requires_grad=True)

        out = inp.argmax()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        out = inp.argmin()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        out = inp.argsort()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        val = torch.rand((), requires_grad=True)

        out = torch.searchsorted(inp, val)
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        bins = torch.linspace(0, 1.0, steps=100, requires_grad=True)
        vals = torch.rand(5, 5, requires_grad=True)
        out = torch.bucketize(vals, bins)
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        val = torch.empty(5).requires_grad_()
        out = val.count_nonzero()
        self.assertFalse(out.requires_grad)

        def assert_only_first_requires_grad(res):
            if not isinstance(res, tuple):
                res = (res,)
            self.assertTrue(res[0].requires_grad)
            for out in res[1:]:
                if out is not None:
                    self.assertFalse(out.requires_grad)

        for sort in [True, False]:
            for return_inverse in [True, False]:
                for return_counts in [True, False]:
                    res = torch.unique(
                        inp,
                        sorted=sort,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                    )
                    assert_only_first_requires_grad(res)

                    res = torch.unique(
                        inp,
                        sorted=sort,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                        dim=0,
                    )
                    assert_only_first_requires_grad(res)

                    res = torch.unique_consecutive(
                        inp, return_inverse=return_inverse, return_counts=return_counts
                    )
                    assert_only_first_requires_grad(res)

                    res = torch.unique_consecutive(
                        inp,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                        dim=0,
                    )
                    assert_only_first_requires_grad(res)

                    # Here we test the internal functions to make sure all of them are
                    # covered on top of the public API
                    res = torch._unique(inp, sorted=sort, return_inverse=return_inverse)
                    assert_only_first_requires_grad(res)

                    # This looks public but is actually manually deleted from the
                    # torch namespace in torch/functional.py
                    res = torch._VF.unique_dim(
                        inp,
                        dim=0,
                        sorted=sort,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                    )
                    assert_only_first_requires_grad(res)

                    # We don't test `unique_dim_consecutive` here.
                    # It looks public but the python binding is actually manually disabled in
                    # tools/autograd/gen_python_functions.py

                    res = torch._unique2(
                        inp,
                        sorted=sort,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                    )
                    assert_only_first_requires_grad(res)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO and sys.version_info >= (3, 14), "Fails in python 3.14.2"
    )
    def test_custom_function_cycle(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, x, metadata):
                x = x.clone()
                ctx.meta = metadata
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                self.assertEqual(x, 3.14)
                self.assertEqual(ctx.meta["foo"], 3.14)
                return gO * x, None

        def get_refs(with_backward):
            a = torch.tensor(3.14, requires_grad=True)

            metadata = {}
            out = MyFn.apply(a, metadata)

            metadata["foo"] = out

            if with_backward:
                out.sum().backward()
                self.assertEqual(a.grad, a)

            return torch._C._WeakTensorRef(out)

        with disable_gc():
            ref = get_refs(False)
            self.assertFalse(ref.expired())
        gc.collect()
        self.assertTrue(ref.expired())

        # The backward clears the saved_variables but not the __dict__
        with disable_gc():
            ref = get_refs(True)
            self.assertFalse(ref.expired())
        gc.collect()
        self.assertTrue(ref.expired())

    def test_create_graph_and_full_backward_hook_cycle(self):
        # If BackwardHook saves grad_output, it can create a cycle when we perform backward
        # with create_graph=True
        #
        #   grad_output -> grad_output.grad_fn -> graph -> hook -> grad_output
        #
        class TestCls:
            # Dummy class for the purpose of creating a weakref
            pass

        def get_ref(input_requires_grad, nb_hooks):
            t = torch.randn(10, requires_grad=input_requires_grad)
            a = torch.tensor(1.0, requires_grad=True)

            class Test(nn.Module):
                def forward(self, x):
                    return x**2 * a**2

            mod = Test()

            for _ in range(nb_hooks):
                mod.register_full_backward_hook(lambda a, b, c: None)

            tmp = mod(t)

            # Save dummy object to graph and get a weak ref to it
            test = TestCls()
            ref = weakref.ref(test)
            tmp.grad_fn.metadata["a"] = test

            with set_warn_always_context(True):
                with warnings.catch_warnings(record=True) as w:
                    tmp.exp().sum().backward(create_graph=True)
                    self.assertTrue(w)
                    found = 0
                    for warning in w:
                        if "Using backward() with create_graph=True" in str(
                            warning.message
                        ):
                            found += 1
                    self.assertEqual(found, 1)

            # Remove the backward + create_graph=True cycle
            a.grad = None
            t.grad = None

            return ref

        for nb_hooks in (1, 2, 3):
            for input_requires_grad in (True, False):
                ref_ = get_ref(
                    input_requires_grad=input_requires_grad,
                    nb_hooks=nb_hooks,
                )
                gc.collect()
                self.assertIsNone(ref_())

    @parametrize("use_custom_function", [True, False])
    @parametrize("use_tensor_hook", [True, False])
    def test_hook_closure_cycle(self, use_custom_function, use_tensor_hook):
        # This creates a cycle between the hook and grad_fn_b
        # hook -> closure -> grad_fn_b (python) -> grad_fn (cpp) -> hook (cpp)
        # -> dict -> hook
        #
        # This test is testing that the grad_fn_b (python) only traverses the
        # dict if it is the only one holding a reference to the grad_fn_b (cpp)
        # shared_ptr
        #
        # See: https://github.com/pytorch/pytorch/issues/102174
        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad

        class Test:
            pass

        count = [0]

        def scope():
            a = torch.tensor(1.0, requires_grad=True)
            if use_custom_function:
                b = Function.apply(a)
            else:
                b = a.clone()
            grad_fn_b = b.grad_fn
            obj = Test()

            def hook(*args):
                # Make sure this hook's closure holds onto grad_fn_b
                # This forms a cycle between the hook and grad_fn_b
                # We also hold onto a sentinel object 'obj' to track
                # whether this cycle is still alive. See 'ref' below.
                grad_fn_b
                obj
                count[0] += 1

            if use_tensor_hook:
                b.register_hook(hook)
            else:
                b.grad_fn.register_hook(hook)
            c = b.clone()
            ref = weakref.ref(obj)
            return c, ref

        with disable_gc():
            out, ref = scope()
            out.backward(retain_graph=True)

            gc.collect()

            # Make sure gc does not clear the cycle noted above.
            # e.g. the hook is alive and gets fired even after gc runs
            out.backward(retain_graph=True)
            self.assertEqual(count[0], 2)

            # ref is still alive because the use_count of the cpp grad_fn
            # shared_ptr > 1 since (1) the python grad_fn is alive, and (2) the
            # rest of the graph holds onto the shared_ptr
            self.assertIsNotNone(ref())

            # Then delete the rest of the graph and check that ref is dead
            del out
            gc.collect()
            self.assertIsNone(ref())

    def test_full_backward_hook_double_backward(self):
        x = torch.rand(1, requires_grad=True)
        y = torch.rand_like(x)

        func = torch.nn.MSELoss()
        counter = [0]

        def hook(module, grad_input, grad_output):
            counter[0] += 1

        func.register_full_backward_hook(hook)

        f = func(x, y)

        (gradx_f,) = torch.autograd.grad(f, x, create_graph=True)
        self.assertEqual(counter[0], 1)
        _ = torch.autograd.grad(gradx_f, x)
        # We should not error, and counter should not be incremented
        self.assertEqual(counter[0], 1)

    def test_input_buffer_accum(self):
        leaf = torch.rand(2, 2, requires_grad=True)

        # An op that returns sparse gradients
        ind = torch.tensor([[0, 0]], dtype=torch.long)
        out2 = leaf.gather(0, ind, sparse_grad=True)

        # An op that returns the gradients as-is
        out1 = leaf.clone()

        grad_out1_original = torch.rand_like(out1)
        grad_out1 = grad_out1_original.clone()
        grad_out2 = torch.rand_like(out2)

        torch.autograd.backward((out1, out2), (grad_out1, grad_out2))

        # Given gradients should not be modified inplace
        self.assertEqual(grad_out1, grad_out1_original)

    def test_no_unnecessary_unwrapping(self):
        a = torch.randn(5, requires_grad=True)
        a_orig = a.detach().clone()
        b = a * a
        c = a * b
        d = torch.exp(a)

        # a is leaf
        self.assertIs(b.grad_fn._saved_self, a)
        self.assertIs(b.grad_fn._saved_other, a)
        self.assertIs(c.grad_fn._saved_self, a)

        # b is not an output
        self.assertIs(c.grad_fn._saved_other, b)

        # d is an output
        self.assertEqual(d.grad_fn._saved_result, d)
        self.assertIsNot(d.grad_fn._saved_result, d)

        c.sum().backward()

        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            c.grad_fn._saved_self

        # a is left untouched
        self.assertEqual(a, a_orig)

    def test_saved_variable_version_counter(self):
        a = torch.rand(2, requires_grad=True)

        b = torch.exp(a)

        b_unpacked = b.grad_fn._saved_result
        self.assertEqual(b, b_unpacked)
        self.assertEqual(b._version, b_unpacked._version)

        with torch.no_grad():
            b += 1

        self.assertEqual(b, b_unpacked)
        self.assertEqual(b._version, b_unpacked._version)

    def test_saved_variable_packing_unpacking_saved_original_with_hooks(self):
        # Tests that packing/unpacking a SavedVariable works correctly with user-defined hooks
        # The saved_original / did_not_save_original distinction corresponds to the `save_original`
        # attribute of `SavedVariable`.

        def test(get_input, is_leaf):
            a = get_input()
            grad_fn = a.grad_fn
            y = a * a
            y.grad_fn._raw_saved_self.register_hooks(lambda x: 2 * x, lambda x: x / 2)
            self.assertEqual(a, y.grad_fn._saved_self)
            if not is_leaf:
                self.assertIs(grad_fn, y.grad_fn._saved_self.grad_fn)
                y.sum().backward()
            else:
                y.sum().backward()
                self.assertEqual(2 * a, a.grad)

            a = get_input()
            grad_fn = a.grad_fn
            y = a * a
            y.grad_fn._raw_saved_self.register_hooks(lambda x: 2 * x, lambda x: x)
            self.assertEqual(2 * a, y.grad_fn._saved_self)
            if not is_leaf:
                self.assertIs(grad_fn, y.grad_fn._saved_self.grad_fn)
                y.sum().backward()
            else:
                y.sum().backward()
                self.assertEqual(3 * a, a.grad)

            # double backward
            a = get_input()
            grad_fn = a.grad_fn
            y = a**3
            y.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)
            s = torch.sum(y)
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            if not is_leaf:
                self.assertIs(grad_fn, y.grad_fn._saved_self.grad_fn)
                g.sum().backward()
            else:
                g.sum().backward()
                self.assertEqual(6 * a, a.grad)

            a = get_input()
            y = a * a
            y.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: 1)
            with self.assertRaisesRegex(
                TypeError, "Output of saved tensor unpack_hook expected to be a Tensor"
            ):
                print(y.grad_fn._saved_self)

            a = get_input()
            y = a * a
            with self.assertRaisesRegex(
                TypeError, "missing 1 required positional argument"
            ):
                y.grad_fn._raw_saved_self.register_hooks(lambda x, b: x, lambda x: x)

            a = get_input()
            y = a * a
            with self.assertRaisesRegex(
                TypeError, "missing 1 required positional argument"
            ):
                y.grad_fn._raw_saved_self.register_hooks(
                    lambda x, b: (x, b), lambda x: x
                )

            def inplace_double(x):
                x *= 2
                return x

            a = get_input()
            t = a * a

            with self.assertRaisesRegex(
                RuntimeError,
                "A saved tensor pack hook is modifying its input in place.",
            ):
                t.grad_fn._raw_saved_self.register_hooks(
                    inplace_double, lambda x: x / 2
                )

        # leaf
        test(lambda: torch.randn(5, requires_grad=True), True)

        # not leaf, not output
        test(lambda: (1 + torch.randn(5, requires_grad=True)), False)

    def test_saved_variable_saved_original_inplace_detach(self):
        # Detaching a tensor that is saved input raises
        a = torch.tensor(1.0, requires_grad=True).clone()
        b = a.sin()
        a.detach_()
        with self.assertRaisesRegex(
            RuntimeError, "Trying to use a saved tensor that has been detached"
        ):
            b.backward()

        # Detaching a tensor that is saved as output is OK
        a = torch.tensor(1.0, requires_grad=True).clone()
        b = a.exp()
        a.detach_()
        b.backward()

    def test_saved_variable_packing_unpacking_did_not_save_original_with_hooks(self):
        # Tests that packing/unpacking a SavedVariable works correctly with user-defined hooks
        # The saved_original / did_not_save_original distinction corresponds to the `save_original`
        # attribute of `SavedVariable`.

        a = torch.randn(5, requires_grad=True)
        y = torch.exp(a)
        y.grad_fn._raw_saved_result.register_hooks(lambda x: x, lambda x: x)
        self.assertEqual(y, y.grad_fn._saved_result)
        self.assertIs(y.grad_fn, y.grad_fn._saved_result.grad_fn)
        y.sum().backward()
        self.assertEqual(a.grad, y)

    def test_saved_variable_packing_unpacking_saved_original_with_default_hooks(self):
        # Tests that default hooks are properly registered, used and reset
        # The saved_original / did_not_save_original distinction corresponds to the `save_original`
        # attribute of `SavedVariable`.
        # See also:
        #  - test_saved_variable_packing_unpacking_saved_original_with_hooks

        def pack(x):
            warnings.warn("pack")
            return x

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            a = torch.ones(5, requires_grad=True)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                y = a * a
                # should raise two warnings from a being saved twice
                self.assertEqual(len(w), 2)

        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a * a
            self.assertEqual(a, y.grad_fn._saved_self)
            self.assertEqual(a, y.grad_fn._saved_other)
            y.sum().backward()
            self.assertEqual(2 * a, a.grad)

        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x / 2):
            a = torch.randn(5, requires_grad=True)
            y = a * a
            self.assertEqual(a, y.grad_fn._saved_self)
            self.assertEqual(a, y.grad_fn._saved_other)
            y.sum().backward()
            self.assertEqual(2 * a, a.grad)

        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a * a
            self.assertEqual(2 * a, y.grad_fn._saved_self)
            self.assertEqual(2 * a, y.grad_fn._saved_other)
            y.sum().backward()
            self.assertEqual(4 * a, a.grad)

        # Exited hooks correctly
        a = torch.randn(5, requires_grad=True)
        y = a * a
        self.assertEqual(a, y.grad_fn._saved_self)
        self.assertEqual(a, y.grad_fn._saved_other)
        y.sum().backward()
        self.assertEqual(2 * a, a.grad)

    def test_saved_variable_packing_unpacking_did_not_save_original_with_default_hooks(
        self,
    ):
        # See also test_saved_variable_packing_unpacking_did_not_save_original_with_hooks

        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = torch.exp(a)
            self.assertEqual(y, y.grad_fn._saved_result)
            y.sum().backward()
            self.assertEqual(a.grad, y)

    def test_setting_default_saved_variable_hooks_twice_should_not_fail(self):
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                pass

    def test_setting_default_saved_variable_hooks_twice_should_use_inner(self):
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 3 * x, lambda x: 3 * x):
            b = torch.randn(5, requires_grad=True)
            with torch.autograd.graph.saved_tensors_hooks(
                lambda x: 5 * x, lambda x: 5 * x
            ):
                a = torch.randn(5, requires_grad=True)
                y = a * a
            z = b * b
        y.sum().backward()
        z.sum().backward()
        self.assertEqual(2 * 5 * 5 * a, a.grad)
        self.assertEqual(2 * 3 * 3 * b, b.grad)

    def test_disabling_saved_tensor_hooks(self):
        with torch.autograd.graph.disable_saved_tensors_hooks("error message"):
            with self.assertRaisesRegex(RuntimeError, "error message"):
                with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                    pass

        self.assertTrue(torch._C._autograd._saved_tensors_hooks_is_enabled())

        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            with self.assertRaisesRegex(RuntimeError, "error message"):
                with torch.autograd.graph.disable_saved_tensors_hooks("error message"):
                    pass

        self.assertTrue(torch._C._autograd._saved_tensors_hooks_is_enabled())

    def test_disabling_saved_tensor_hooks_nested(self):
        with torch.autograd.graph.disable_saved_tensors_hooks("outer"):
            with torch.autograd.graph.disable_saved_tensors_hooks("inner"):
                with self.assertRaisesRegex(RuntimeError, "inner"):
                    with torch.autograd.graph.saved_tensors_hooks(
                        lambda x: x, lambda x: x
                    ):
                        pass

            self.assertFalse(torch._C._autograd._saved_tensors_hooks_is_enabled())

        self.assertTrue(torch._C._autograd._saved_tensors_hooks_is_enabled())

    def test_saved_tensor_hooks_custom_error_propagation(self):
        class CustomError(Exception):
            pass

        class error_on_pack_hook(torch.autograd.graph.saved_tensors_hooks):
            def __init__(self) -> None:
                def pack_hook(x):
                    raise CustomError("pack")

                super().__init__(pack_hook, lambda x: x)

        class error_on_unpack_hook(torch.autograd.graph.saved_tensors_hooks):
            def __init__(self) -> None:
                def unpack_hook(x):
                    raise CustomError("unpack")

                super().__init__(lambda x: x, unpack_hook)

        a = torch.tensor(1.0, requires_grad=True)

        with error_on_pack_hook():
            with self.assertRaisesRegex(CustomError, "pack"):
                out = torch.sin(a)

        with error_on_unpack_hook():
            out = torch.sin(a)
            with self.assertRaisesRegex(CustomError, "unpack"):
                out.backward()

    def test_saved_tensor_hooks_custom_function_intermediates(self):
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                intermediate = x.exp()
                ctx.save_for_backward(
                    intermediate.clone().detach_().requires_grad_(True)
                )
                return x.exp()

            @staticmethod
            def backward(ctx, grad_out):
                (intermediate,) = ctx.saved_tensors
                return grad_out * intermediate

        a = torch.tensor(1.0, requires_grad=True)

        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            out = Func.apply(a)
        out.backward()

    def test_unpack_hooks_exec_count(self):
        def f(x, y):
            return x * y

        pack_count = 0
        unpack_count = 0

        def pack_hook(x):
            nonlocal pack_count
            pack_count += 1
            return x

        # unpack hook shouldn't run during compilation, while we trace the forward
        def unpack_hook(x):
            nonlocal unpack_count
            unpack_count += 1
            return x

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out_test = f(x, y)
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 0)
            out_test.sum().backward()
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 1)

    def test_saved_tensors_hook_version_counter_not_shared(self):
        class Test(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.sin()

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                before = a._version
                x.add_(1)
                self.assertEqual(a._version, before)
                return grad_output

        a = torch.tensor(1.0, requires_grad=True)
        a_replacement = a.clone()

        def pack_hook(x):
            return a_replacement

        def unpack_hook(x):
            return x

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            b = Test.apply(a)

        b.backward()

    def test_save_on_cpu_and_checkpoint(self):
        a = torch.randn(2, 2, requires_grad=True)

        b = a.pow(2).pow(2).pow(2).pow(2)
        b.sum().backward()
        b_grad = a.grad.clone()
        a.grad.zero_()

        with torch.autograd.graph.save_on_cpu():
            h = a.pow(2)
            h = checkpoint(lambda x: x.pow(2).pow(2), h, use_reentrant=False)
            c = h.pow(2)
        c.sum().backward()
        c_grad = a.grad.clone()
        a.grad.zero_()

        def f(a):
            h = a.pow(2)
            with torch.autograd.graph.save_on_cpu():
                h = h.pow(2).pow(2)
            return h.pow(2)

        d = checkpoint(f, a, use_reentrant=False)
        d.sum().backward()
        d_grad = a.grad.clone()

        self.assertEqual(b_grad, c_grad)
        self.assertEqual(b_grad, d_grad)

    def test_pack_hook_with_inplace_modification_should_fail(self):
        a = torch.randn(5, requires_grad=True)

        def inc(x):
            x += 1
            return x

        with torch.autograd.graph.saved_tensors_hooks(inc, lambda x: x):
            with self.assertRaisesRegex(
                RuntimeError,
                "A saved tensor pack hook is modifying its input in place.",
            ):
                y = torch.exp(a)

        y = torch.exp(a)
        with self.assertRaisesRegex(
            RuntimeError, "A saved tensor pack hook is modifying its input in place."
        ):
            y.grad_fn._raw_saved_result.register_hooks(inc, lambda x: x)

    def test_saving_variable_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:

            def pack(x):
                name = os.path.join(tmp_dir, str(uuid.uuid4()))
                torch.save(x, name)
                return name

            def unpack(name):
                return torch.load(name)

            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                a = torch.ones(5, requires_grad=True)
                y = a * a
                self.assertEqual(a, y.grad_fn._saved_self)

                y.sum().backward()
                self.assertEqual(2 * a, a.grad)

    def test_default_saved_tensors_hooks_double_backward(self):
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a**3
            s = torch.sum(y)
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            g.sum().backward()
            self.assertEqual(6 * a, a.grad)

        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a**3
            s = torch.sum(y)
        (g,) = torch.autograd.grad(s, (a,), create_graph=True)
        g.sum().backward()
        # factor 2 because only a is saved once
        self.assertEqual(6 * 2 * a, a.grad)

        a = torch.randn(5, requires_grad=True)
        y = a**3
        s = torch.sum(y)
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            g.sum().backward()
            # factor 4 because pow_backward is grad * (exp * self.pow(exp - 1))
            # so grad is saved and self (i.e. a) is saved
            self.assertEqual(6 * 4 * a, a.grad)

        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a**3
            s = torch.sum(y)
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            g.sum().backward()
            # combining the two above blocks: 2 * 4 = 8
            # note that in that sense, a is saved twice
            self.assertEqual(6 * 8 * a, a.grad)

    def test_wrapped_number_saved_tensors_hooks(self):
        def err_hook(x):
            raise RuntimeError("this hook should not be called")

        with torch.autograd.graph.saved_tensors_hooks(err_hook, err_hook):
            a = torch.randn(5, requires_grad=True)
            out = (a * 3).sum()
            # 3 is saved as a saved tensor because it is a wrapped number, but
            # wrapped numbers should be special cased to not trigger saved variable hooks
            torch.autograd.grad(out, (a,))

    def test_graph_save_on_cpu(self):
        def test(get_input, cuda, pin_memory):
            with torch.autograd.graph.save_on_cpu(pin_memory):
                a = get_input()
                if cuda:
                    a.cuda()
                y = a * a
                self.assertEqual(a, y.grad_fn._saved_self)
                self.assertEqual(a, y.grad_fn._saved_other)
                self.assertEqual(a.dtype, y.grad_fn._saved_self.dtype)
                self.assertEqual(a.layout, y.grad_fn._saved_self.layout)
                if y.is_sparse:
                    y = y.to_dense()
                y.sum().backward()

                actual = 2 * a
                expected = a.grad
                if a.is_sparse:
                    actual = actual.coalesce()
                    expected = expected.coalesce()

                self.assertEqual(actual, expected)

        for cuda in [False] + ([True] if torch.cuda.is_available() else []):
            for pin_memory in [True, False]:
                # FloatTensor
                test(lambda: torch.randn(5, requires_grad=True), cuda, pin_memory)
                # DoubleTensor
                test(
                    lambda: torch.randn(5, requires_grad=True, dtype=torch.double),
                    cuda,
                    pin_memory,
                )
                # Sparse tensor
                x = torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(),
                    torch.tensor([1.0, 1.0]),
                    requires_grad=True,
                )
                test(lambda: x, cuda, pin_memory)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_graph_save_on_cpu_cuda(self):
        def f(x):
            a = x + 1
            return a * a

        # with grad
        a = torch.ones(1, requires_grad=True, device="cuda")
        y = f(a)
        memory_with_grad = torch.cuda.memory_allocated()

        del a
        del y

        # without grad
        a = torch.ones(1, requires_grad=True, device="cuda")
        with torch.no_grad():
            y = f(a)
        memory_without_grad = torch.cuda.memory_allocated()

        self.assertGreater(memory_with_grad, memory_without_grad)

        del a
        del y

        # with hooks
        with torch.autograd.graph.save_on_cpu():
            a = torch.ones(1, requires_grad=True, device="cuda")
            y = f(a)
            memory_with_hooks = torch.cuda.memory_allocated()
            self.assertEqual(memory_with_hooks, memory_without_grad)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_scalar_grad_mixed_device(self):
        x = torch.tensor(1.0, requires_grad=True)
        y = torch.randn(2, 2, device="cuda")
        out = x * y
        out.sum().backward()

    @scoped_load_inline
    def test_multi_grad_all_hooks(self, load_inline):
        t1 = torch.rand(2, requires_grad=True)
        t2 = torch.rand(2, requires_grad=True)
        t3 = torch.rand(2, requires_grad=True)
        t4 = torch.rand(2, requires_grad=True)

        # Ensure we properly detect all types of Nodes here
        # C++ Node
        t1 = t1.mul(2)

        # Python custom Function
        class Foo(Function):
            @staticmethod
            def forward(ctx, a):
                return a.clone()

            @staticmethod
            def backward(ctx, gO):
                return gO

        t2 = Foo.apply(t2)

        # C++ Node
        t3 = torch._C._functions.UndefinedGrad()(t3)

        # C++ Custom Op
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x.clone();
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(test_multigrad_all_hooks, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        module = load_inline(
            name="test_multigrad_all_hooks",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        t4 = torch.ops.test_multigrad_all_hooks.custom_op_backed_by_autograd_fn(t4)

        res = [None] * 4
        count = [0]

        def hook(grads):
            nonlocal res
            count[0] += 1
            res = [g is not None for g in grads]

        handle = torch.autograd.graph.register_multi_grad_hook((t1, t2, t3, t4), hook)

        out = t2 * t3

        out.sum().backward(inputs=(t2, t3), retain_graph=True)
        self.assertEqual(count[0], 1)
        self.assertEqual(res, [False, True, True, False])

        out.sum().backward(inputs=(t1, t4), retain_graph=True)
        self.assertEqual(count[0], 1)

        out.sum().backward(inputs=(t1, t3), retain_graph=True)
        self.assertEqual(count[0], 2)
        self.assertEqual(res, [False, False, True, False])

        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, gO):
                raise RuntimeError("error message")

        out = Func.apply(t2) * t3
        with self.assertRaisesRegex(RuntimeError, "error message"):
            out.sum().backward(inputs=(t2, t3), retain_graph=True)
        self.assertEqual(count[0], 2)

        handle.remove()
        out.sum().backward(inputs=(t1, t3), retain_graph=True)
        self.assertEqual(count[0], 2)

    def test_multi_grad_any_hooks(self):
        hook_id = 0
        any_hook_handles: list[RemovableHandle] = []

        class MultiOutputModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(3, 3)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                z = self.lin(x)
                out = torch.sin(z), torch.cos(z)
                nonlocal hook_id
                z.register_hook(partial(hook, hook_id))
                hook_id += 1
                any_hook_handles.append(
                    torch.autograd.graph.register_multi_grad_hook(
                        out, partial(hook, hook_id), mode="any"
                    )
                )
                hook_id += 1
                return out

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod1 = MultiOutputModule()
                self.mod2 = MultiOutputModule()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.mod1(x)
                z = y[0] + y[1]
                return self.mod2(z)

        hook_order: list[int] = []
        hook_count = 0

        def hook(hook_id: int, *unused):
            nonlocal hook_count
            nonlocal hook_order
            hook_count += 1
            hook_order.append(hook_id)

        # Any hooks: IDs 1 and 3; regular hooks: IDs 0 and 2
        model = Model()
        inp = torch.randn((2, 3))
        out = model(inp)
        (out[0] + out[1]).sum().backward()
        # Check that the any-hook runs only once and before the regular hook
        # for each module
        self.assertEqual(len(any_hook_handles), 2)
        self.assertEqual(hook_order, [3, 2, 1, 0])

        hook_id = 0
        hook_order.clear()
        any_hook_handles.clear()
        out = model(inp)
        for handle in any_hook_handles:
            handle.remove()
        (out[0] + out[1]).sum().backward()
        # Check that the any-hook does not run if removed
        self.assertEqual(hook_order, [2, 0])

    def test_multi_grad_hooks_invalid_mode(self):
        t1 = torch.rand(2, requires_grad=True)
        t2 = torch.rand(2, requires_grad=True)
        regex = r"Expects mode to be one of \('all', 'any'\) but got foo"
        with self.assertRaisesRegex(ValueError, regex):
            torch.autograd.graph.register_multi_grad_hook(
                (t1, t2), lambda _: None, mode="foo"
            )

    def test_pynode_destruction_deadlock(self):
        script = """
import torch

class Foo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def forward(ctx, gO):
        return gO.clone()

def get_out():
    inp = torch.rand(2, requires_grad=True)

    # The python function is first so that it runs
    # last in the backward pass
    right = Foo.apply(inp)

    # An op that creates new memory
    left1 = inp.clone()
    # An op that saves its input
    left2 = left1 ** 2

    # Inplace modify so that the backward for
    # left2 always raises an error
    left1 += 1

    # An op that takes both side as input.
    # After running, both side's last op will be in
    # the ready queue
    # And the op for left will run first as it was
    # executed last during the forward
    out = left2 + right

    return out

# Nothing should be global variables here as, from what
# I can see, python leaks all the global objects
get_out().sum().backward()

# This used to deadlock when the PyNode is being destroyed after
# the error is raised.
"""
        try:
            subprocess.check_output(
                [sys.executable, "-c", script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
                # It is ok to have an extra long timeout here as a timeout means the test failed
                timeout=20,
            )
        except subprocess.TimeoutExpired as e:
            self.fail(
                msg="Example code timed out! See the code sample in the test for details."
            )
        except subprocess.CalledProcessError as e:
            if e.returncode < 0:
                # Sometimes we segfault instead of deadlocking
                self.fail("Subprocess exited with a fatal signal")
            else:
                err_msg = (
                    "RuntimeError: one of the variables needed for gradient computation"
                )
                self.assertTrue(err_msg in e.output.decode("utf-8"))

    def test_view_func_replay(self):
        with torch.autograd._force_original_view_tracking(True):

            def _assert_match_metadata(a, b):
                self.assertEqual(a.size(), b.size())
                self.assertEqual(a.stride(), b.stride())
                self.assertEqual(a.storage_offset(), b.storage_offset())
                self.assertEqual(a.device, b.device)
                self.assertEqual(a.dtype, b.dtype)

            def _test_fn(fn, inp, *args, use_unsafe_view_func=False):
                outs = fn(inp, *args)
                # handle functions that return multiple views (e.g. split)
                if isinstance(outs, torch.Tensor):
                    outs = [outs]

                for out in outs:
                    self.assertTrue(out._is_view())
                    self.assertTrue(out._base is inp)

                    # forward view_func
                    new_inp = inp.clone()
                    _assert_match_metadata(new_inp, inp)
                    if use_unsafe_view_func:
                        new_out = out._view_func_unsafe(new_inp)
                    else:
                        new_out = out._view_func(new_inp)
                    _assert_match_metadata(new_out, out)
                    self.assertEqual(new_out, out)

                    # reverse view_func
                    new_out = out.detach()
                    new_inp = out._rev_view_func_unsafe(new_out)
                    _assert_match_metadata(new_inp, inp)
                    self.assertTrue(new_inp._is_view())
                    self.assertTrue(new_inp._base is new_out)

            # test individual view ops
            _test_fn(torch.ops.aten.alias.default, torch.rand(2, 2))
            _test_fn(torch.as_strided, torch.rand(2, 2), (4,), (1,))
            _test_fn(torch.chunk, torch.rand(2, 4), 2, -1)
            _test_fn(torch.diagonal, torch.rand(4, 4))
            _test_fn(torch.ops.aten.expand.default, torch.rand(4, 1), (-1, 3))
            _test_fn(torch.narrow, torch.rand(2, 2), 0, 1, 1)
            _test_fn(torch.permute, torch.rand(2, 3, 4), (1, 0, 2))
            _test_fn(torch.select, torch.rand(2, 2), 0, 0)
            _test_fn(torch.ops.aten.slice.Tensor, torch.rand(2, 2), 1, 1, 2)
            _test_fn(torch.split, torch.rand(2, 2), 1)
            _test_fn(torch.split_with_sizes, torch.rand(2, 4), [1, 3], -1)
            _test_fn(torch.squeeze, torch.rand(2, 1, 4))
            _test_fn(torch.squeeze, torch.rand(2, 1, 4), 1)
            _test_fn(torch.squeeze, torch.rand(2, 1, 1, 4), [1, 2])
            _test_fn(torch.t, torch.rand(2, 4))
            _test_fn(torch.transpose, torch.rand(2, 4), 0, 1)
            _test_fn(torch.unbind, torch.rand(1, 5))
            _test_fn(torch.ops.aten.unfold.default, torch.rand(1, 5), 1, 3, 2)
            _test_fn(torch.unsqueeze, torch.rand(2, 4), -2)
            _test_fn(torch.ops.aten.view.default, torch.rand(2, 10), (-1, 5, 2))
            _test_fn(torch.view_as_complex, torch.rand(2, 2))
            _test_fn(torch.view_as_real, torch.rand(2, 2, dtype=torch.cfloat))

            # test view chains
            _test_fn(
                lambda x: x.unsqueeze(-1).transpose(-1, -2).squeeze(1),
                torch.randn(2, 4),
            )
            _test_fn(
                lambda x: x.chunk(2, -1)[0].transpose(0, 1).unsqueeze(-1),
                torch.randn(2, 3, 4),
            )
            _test_fn(
                lambda x: x.split_with_sizes([1, 3], -1)[0].chunk(2, 0),
                torch.randn(2, 3, 4),
            )

            # chains with missing view_func()s use as_strided() to cover the gaps
            def chain_with_only_parent_view_func(x):
                with torch.autograd._force_original_view_tracking(True):
                    x = x.split_with_sizes([1, 3], -1)[0]

                with torch.autograd._force_original_view_tracking(False):
                    x = x.chunk(2, 0)

                return x

            _test_fn(chain_with_only_parent_view_func, torch.randn(2, 3, 4))

            def chain_with_only_current_view_func(x):
                with torch.autograd._force_original_view_tracking(False):
                    x = x.split_with_sizes([1, 3], -1)[0]

                with torch.autograd._force_original_view_tracking(True):
                    x = x.chunk(2, 0)

                return x

            _test_fn(chain_with_only_current_view_func, torch.randn(2, 3, 4))

            # TODO: Move this somewhere else
            # test NT views
            from torch.nested._internal.nested_tensor import (
                nested_view_from_values_offsets,
            )

            values = torch.randn(10, 5)
            offsets = torch.tensor([0, 3, 6, 10])
            _test_fn(nested_view_from_values_offsets, values, offsets)

            nt = nested_view_from_values_offsets(values, offsets).detach().clone()
            _test_fn(
                torch.ops.aten._nested_get_values.default, nt, use_unsafe_view_func=True
            )

            def chain_nt_to_dense_back_and_forth(nt):
                # NJT1 -> dense -> NJT2 -> dense
                offsets2 = nt.offsets().detach().clone()
                return nested_view_from_values_offsets(nt.values(), offsets2).values()

            _test_fn(chain_nt_to_dense_back_and_forth, nt, use_unsafe_view_func=True)

            def chain_dense_to_nt_back_and_forth(values, offsets):
                offsets2 = offsets.detach().clone()
                # dense -> NJT1 -> dense -> NJT2
                return nested_view_from_values_offsets(
                    nested_view_from_values_offsets(values, offsets).values(), offsets2
                )

            _test_fn(
                chain_dense_to_nt_back_and_forth,
                values,
                offsets,
                use_unsafe_view_func=True,
            )

    def test_view_func_replay_with_modified_state(self):
        with torch.autograd._force_original_view_tracking(True):
            base = torch.randn(3, 4, 5)
            view = base.select(1, 2)

            def symint_visitor_fn(x):
                # modify saved index
                return x + 1

            # ensure modifying state changes view replay
            new_base = torch.randn_like(base)
            new_view = view._view_func(new_base, symint_visitor_fn=symint_visitor_fn)
            self.assertEqual(new_view, new_base.select(1, 3))

            # ensure saved state reverts back afterwards
            self.assertEqual(view._view_func(new_base), new_base.select(1, 2))

            # check modifying tensor state. currently, slice_inverse() is the only
            # view that saves a tensor
            base = torch.randn(3, 4, 5)
            sliced = base[:, 2:3, :].detach()
            view = torch.ops.aten.slice_inverse(sliced, base, 1, 2, 3, 1)

            replacement_shape = (1, 2, 3)

            def tensor_visitor_fn(x):
                # return tensor with a smaller shape than the saved one
                return torch.randn(*replacement_shape)

            # ensure modifying state changes view replay
            new_sliced = torch.ones_like(base)[:, 2:3, :].detach()
            new_view = view._view_func(new_sliced, tensor_visitor_fn=tensor_visitor_fn)
            self.assertEqual(new_view.shape, replacement_shape)
            self.assertEqual(
                new_view, new_sliced.as_strided(replacement_shape, (6, 3, 1))
            )

            # ensure saved state reverts back afterwards
            self.assertEqual(view._view_func(sliced), base)

    def test_setup_context_when_forward_has_default_args(self):
        class PowFunction(Function):
            @staticmethod
            def forward(x, y=3):
                return torch.pow(x, y)

            @staticmethod
            def setup_context(ctx, inputs, output):
                x, y = inputs
                ctx.save_for_backward(x)
                ctx.y = y

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                y = ctx.y
                return gO * y * torch.pow(x, y - 1), None

        class PowFunctionWithClassmethod(Function):
            @classmethod
            def forward(cls, x, y=3):
                return torch.pow(x, y)

            @classmethod
            def setup_context(cls, ctx, inputs, output):
                x, y = inputs
                ctx.save_for_backward(x)
                ctx.y = y

            @classmethod
            def backward(cls, ctx, gO):
                (x,) = ctx.saved_tensors
                y = ctx.y
                return gO * y * torch.pow(x, y - 1), None

        x = torch.tensor(2.0, requires_grad=True)

        y = torch.tensor(8.0)
        y_expected = torch.tensor(12.0)

        y1 = PowFunction.apply(x)
        (y1_expected,) = torch.autograd.grad(y1, x)

        y2 = PowFunctionWithClassmethod.apply(x)
        (y2_expected,) = torch.autograd.grad(y2, x)

        self.assertEqual(y, y1)
        self.assertEqual(y_expected, y1_expected)
        self.assertEqual(y, y2)
        self.assertEqual(y_expected, y2_expected)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_gradcheck_default_device_placement_context(self):
        # During gradcheck with fast_mode=True, we create a random vector on the CPU device using a CPU generator.
        # This test ensures that this still works when the default device is set to something else by the user.
        with torch.device("cuda"):
            x = torch.randn(3, dtype=torch.double, requires_grad=True)

            def func(inp):
                return inp**2.0

            self.assertTrue(gradcheck(func, x, fast_mode=True))

    def test_grad_thread_safety(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        NUM_ITERS = 10
        NUM_THREADS = 4

        # Concurrent calls to tensor.untyped_storage()
        def access_grad(tensor, barrier):
            barrier.wait()
            return weakref.ref(tensor.grad)

        for i in range(NUM_ITERS):
            tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
            (tensor**2).sum().backward()

            barrier = threading.Barrier(NUM_THREADS)
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = [
                    executor.submit(access_grad, tensor, barrier)
                    for _ in range(NUM_THREADS)
                ]

                # Check that all the grad tensors returned were the same
                for future in futures:
                    self.assertEqual(future.result()(), tensor.grad)
                self.assertIsNotNone(tensor.grad)


def index_perm_variable(shape, max_indices):
    if not isinstance(shape, tuple):
        shape = (shape,)

    index = torch.randperm(max_indices).narrow(0, 0, reduce(mul, shape)).view(shape)
    return index


def bernoulli_scalar():
    return torch.tensor(0, dtype=torch.uint8).bernoulli_()


class TestAutogradForwardModeBatchedGrad(TestCase):
    def test_out_of_place_basic(self):
        a = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        b = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        self.assertTrue(
            gradcheck(
                torch.sin,
                a,
                check_forward_ad=True,
                check_batched_grad=True,
                check_batched_forward_grad=True,
            )
        )
        self.assertTrue(
            gradcheck(
                torch.add,
                (a, b),
                check_forward_ad=True,
                check_batched_grad=True,
                check_batched_forward_grad=True,
            )
        )

    def test_out_of_place_not_same_layout(self):
        input = torch.zeros([2, 2]).transpose(0, 1)
        tangent = torch.zeros([2, 2, 2])

        def jvp(tangent):
            with fwAD.dual_level():
                x = fwAD.make_dual(input, tangent)
                return fwAD.unpack_dual(x)[1]

        x_tangent = torch._vmap_internals._vmap(jvp, 0, 0)(tangent)

        self.assertIsNot(x_tangent, tangent)

    def test_inplace_on_view_same_layout(self):
        input = torch.zeros([2, 2])
        tangent = torch.zeros([2, 2, 2])
        base = torch.zeros([2, 2])
        view = base.view_as(base)

        def jvp(tangent):
            with fwAD.dual_level():
                x = fwAD.make_dual(input, tangent)
                view.copy_(x)
                return (
                    fwAD.unpack_dual(x)[1],
                    fwAD.unpack_dual(view)[1],
                    fwAD.unpack_dual(view._base)[1],
                )

        x_tangent, view_tangent, base_tangent = torch._vmap_internals._vmap(jvp, 0, 0)(
            tangent
        )

        self.assertFalse(
            view_tangent._is_view()
        )  # Optimization to share the same tensor!
        self.assertIs(view_tangent, base_tangent)
        self.assertIs(x_tangent, tangent)

    def test_inplace_on_view_not_same_layout(self):
        input = torch.zeros([2, 2])
        tangent = torch.zeros([2, 2, 2])
        view = torch.zeros([2, 2]).transpose(0, 1)

        def jvp(tangent):
            with fwAD.dual_level():
                x = fwAD.make_dual(input, tangent)
                view.copy_(x)
                return (
                    fwAD.unpack_dual(x)[1],
                    fwAD.unpack_dual(view)[1],
                    fwAD.unpack_dual(view._base)[1],
                )

        x_tangent, view_tangent, base_tangent = torch._vmap_internals._vmap(jvp, 0, 0)(
            tangent
        )

        self.assertIs(view_tangent._base, base_tangent)
        self.assertIs(x_tangent, tangent)
        self.assertIsNot(view_tangent, tangent)

    def test_metadata_check_for_storage_numel_skipped(self):
        # See: test_metadata_check_checks_storage_numel for the reverse of this test
        primal = torch.randn(5)[:4].detach()
        self.assertEqual(len(primal.storage()), 5)
        tangent = torch.randn(10, 4)

        def jvp(tangent):
            with fwAD.dual_level():
                dual = fwAD.make_dual(primal, tangent)
                _, unpacked_tangent = fwAD.unpack_dual(dual)

                # No copy is made
                self.assertIs(tangent, unpacked_tangent)

                # as_strided raises
                with self.assertRaisesRegex(
                    RuntimeError, "can access memory outside of `tensor`"
                ):
                    dual.as_strided((5,), (1,), 0)
            return unpacked_tangent

        torch._vmap_internals._vmap(jvp, 0, 0)(tangent)


class TestAutogradForwardMode(TestCase):
    def tearDown(self):
        # Ensure that a failing test won't make others fail
        while fwAD._current_level >= 0:
            fwAD.exit_dual_level()

        super().tearDown()

    def test_forward_level_cleanup(self):
        def get_tensor_and_weak_ref():
            # Create a new Tensor and weak reference
            t = torch.rand(2, requires_grad=True)
            return t, torch._C._WeakTensorRef(t)

        # Sanity check that the helper function works as expected
        t, t_ref = get_tensor_and_weak_ref()
        self.assertFalse(t_ref.expired())

        del t
        self.assertTrue(t_ref.expired())

        # Main test code
        foo = torch.rand(2)

        with fwAD.dual_level():
            tangent, tangent_ref = get_tensor_and_weak_ref()
            self.assertFalse(tangent_ref.expired())

            dual = fwAD.make_dual(foo, tangent)
            self.assertFalse(tangent_ref.expired())

            # Make sure that the tangent we provided has been reused as is
            self.assertTrue(fwAD.unpack_dual(dual)[1] is tangent)

            # Make sure that dual is keeping the tangent alive
            del tangent
            self.assertFalse(tangent_ref.expired())

            # Make sure that the dual level does not keep the c++
            # version of the tangent alive
            del dual
            self.assertTrue(tangent_ref.expired())

    def test_size_check(self):
        foo = torch.rand(2)
        tangent = torch.rand(3)

        with fwAD.dual_level():
            with self.assertRaisesRegex(
                RuntimeError,
                "Trying to set a forward gradient that has a different size",
            ):
                dual = fwAD.make_dual(foo, tangent)

            dual = fwAD.make_dual(foo, tangent[1:])

    def test_metadata_check_checks_storage_numel(self):
        primal = torch.randn(5)[:4].detach()
        self.assertEqual(len(primal.storage()), 5)
        tangent = torch.randn(4)

        with fwAD.dual_level():
            dual = fwAD.make_dual(primal, tangent)
            _, unpacked_tangent = fwAD.unpack_dual(dual)

            # # Verify that mutating unpacked tangent does not affect the original tangent
            tangent_clone = tangent.clone()
            unpacked_tangent *= 2
            self.assertTrue(torch.allclose(tangent_clone, tangent))

            # as_strided runs without error
            dual.as_strided((5,), (1,), 0)

    def test_metadata_check_checks_ignores_size_zero(self):
        a = torch.ones(0).as_strided((0, 1), (1, 1), 0)
        b = torch.ones(0).as_strided((0, 1), (1, 0), 0)

        with fwAD.dual_level():
            dual = fwAD.make_dual(a, b)
            torch.diagonal(dual, offset=0)

        input = torch.rand([0, 1], dtype=torch.complex128, requires_grad=True)
        func = partial(torch.diagonal, offset=0)
        torch.autograd.gradcheck(func, (input,), check_forward_ad=True)

    def test_metadata_check_when_primal_has_conj_bit(self):
        # Make sure the _has_same_storage_numel is a fallthrough, so that
        # conj bit does not materialize. If it materializes it would
        # cause the layout check to fail for views that do not index the
        # the entire storage.
        a = torch.randn(2, 2, dtype=torch.cdouble).conj()
        b = torch.rand_like(a)

        self.assertTrue(torch.is_conj(a))
        self.assertEqual(len(a.storage()), len(b.storage()))

        with fwAD.dual_level():
            dual = fwAD.make_dual(a, b)
            dual[1:]

    def test_metadata_check_when_primal_has_neg_bit(self):
        # Make sure the _has_same_storage_numel is a fallthrough, so that
        # conj bit does not materialize. If it materializes it would
        # cause the layout check to fail for views that do not index the
        # the entire storage.
        a = torch.randn(2, 2, dtype=torch.cdouble).conj().imag
        b = torch.randn(2, 2, dtype=torch.cdouble).imag

        self.assertTrue(torch.is_neg(a))
        self.assertEqual(len(a.storage()), len(b.storage()))

        with fwAD.dual_level():
            dual = fwAD.make_dual(a, b)
            dual[1:]

    def test_metadata_check_check_conj(self):
        keys = {
            "NEITHER": lambda x: x,
            "CONJ": lambda x: x.conj(),
            "NEG": lambda x: x._neg_view(),
        }

        for primal_key, tangent_key in product(keys, keys):
            x = keys[primal_key](torch.randn(2, 3, 4, dtype=torch.cdouble))
            t = keys[tangent_key](torch.randn(2, 3, 4, dtype=torch.cdouble))

            if primal_key == tangent_key:
                with fwAD.dual_level():
                    dual = fwAD.make_dual(x, t)
                    self.assertTrue(fwAD.unpack_dual(dual).tangent is t)
                    torch.real(dual)
                    torch.imag(dual)
            else:
                with fwAD.dual_level():
                    dual = fwAD.make_dual(x, t)
                    self.assertTrue(fwAD.unpack_dual(dual).tangent is not t)
                    torch.real(dual)
                    torch.imag(dual)

    def test_metadata_check_ignore_storage_offset_for_zero_numel_tensor(self):
        # See https://github.com/pytorch/pytorch/issues/80507
        a = torch.tensor([1.0]).as_strided((0,), (1,), 1)
        b = torch.tensor([1.0]).as_strided((0,), (1,), 2)

        with fwAD.dual_level():
            dual_input = fwAD.make_dual(a, b)
            # Check that no copy is made
            self.assertIs(fwAD.unpack_dual(dual_input).tangent, b)

        a = torch.tensor([1.0]).as_strided((1,), (2,), 0)
        b = torch.tensor([1.0]).as_strided((1,), (1,), 0)

        with fwAD.dual_level():
            dual_input = fwAD.make_dual(a, b)
            dual_input[1:]

    # The following test functions want to ensure all the following behaviors:
    #   - Ensure that default level system in the python binding works
    #   - Ensure that only level 0 exists and nesting is properly disabled
    #   - Ensure that printing works fine
    #   - Ensure that basic packing/unpacking works
    #   - Ensure that advanced packing/unpacking works
    #     - For memory / version counter share
    #     - For backward AD (regular ops)
    #   - Ensure that view + inplace for both modes work fine
    #   - Ensure we do proper cleanup on exit of a level

    def test_default_level(self):
        foo = torch.rand(2)
        bar = torch.rand(2)

        with fwAD.dual_level():
            baz = fwAD.make_dual(foo, bar)
            baz_primal, baz_tangent = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        # We don't actually need to enforce that these two are the exact same python
        # object, feel free to relax in the future
        self.assertIs(baz_tangent, bar)

        baz_primal, baz_tangent = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        self.assertEqual(baz_tangent, None)

    def test_fwd_grad_enabled(self):
        # Tests some private helper functions to enable/disable fwd grad mode
        enabled = fwAD._is_fwd_grad_enabled()
        self.assertTrue(enabled)

        try:
            torch._C._set_fwd_grad_enabled(False)
            enabled = fwAD._is_fwd_grad_enabled()
            self.assertFalse(enabled)
        finally:
            torch._C._set_fwd_grad_enabled(True)

        enabled = fwAD._is_fwd_grad_enabled()
        self.assertTrue(enabled)

    def test_set_fwd_grad_enabled(self):
        # Tests a private helper function
        try:
            torch._C._set_fwd_grad_enabled(False)
            enabled = fwAD._is_fwd_grad_enabled()
            self.assertFalse(enabled)

            with fwAD._set_fwd_grad_enabled(True):
                enabled = fwAD._is_fwd_grad_enabled()
                self.assertTrue(enabled)

            enabled = fwAD._is_fwd_grad_enabled()
            self.assertFalse(enabled)
        finally:
            torch._C._set_fwd_grad_enabled(True)

    def test_nested_level(self):
        with fwAD.dual_level() as level:
            # For now only level 0 exists
            self.assertEqual(level, 0)

        with fwAD.dual_level():
            with self.assertRaisesRegex(
                RuntimeError, "Nested forward mode AD is not supported at the moment"
            ):
                nest_level = fwAD.enter_dual_level()

    def test_set_fw_grad_having_own_fw_grad_at_same_level(self):
        foo = torch.rand(2)
        bar = torch.rand(2)
        baz = torch.rand(2)

        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)
            with self.assertRaisesRegex(
                RuntimeError, "has a forward gradient at the same level"
            ):
                fwAD.make_dual(baz, dual)

    def test_codegen_ignores_undefined_outputs(self):
        # This test checks that codegen silently ignores undefined outputs
        # Below, grad_input is specified as False in grad_output_mask, so
        # convolution backward will return a undefined tensor in that position.
        # Note that for this test to work we need to make sure either grad_output
        # or weight to be a dual tensor, so grad_input requires forward grad
        weight = torch.randn(6, 1, 30, 30)
        inp = torch.rand((1, 1, 32, 32))
        out = torch.nn.functional.conv2d(inp, weight)
        grad_out = torch.ones_like(out)

        with fwAD.dual_level():
            dual_weight = fwAD.make_dual(weight, torch.ones_like(weight))
            grad_input, _, _ = torch.ops.aten.convolution_backward(
                grad_out,
                inp,
                dual_weight,
                (0,),
                (1, 1),
                (0, 0),
                (1, 1),
                False,
                (0, 0),
                1,
                (False, True, False),
            )
        self.assertIsNone(grad_input)

    def test_make_dual_inference_tensor_in_inference_mode(self):
        with torch.inference_mode():
            foo = torch.rand(2)
            bar = torch.rand(2)
            foo_copy = foo.clone()

            with fwAD.dual_level():
                dual = fwAD.make_dual(foo, bar)
                self.assertFalse(dual._is_view())

                dual += 1
                self.assertFalse(torch.allclose(foo, foo_copy))

    def test_make_dual_torch_dispatch(self):
        counter = [0]

        class MySubclass(torch.Tensor):
            def __new__(cls, data=None):
                return torch.Tensor._make_subclass(cls, data)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if func.overloadpacket == torch.ops.aten.alias:
                    counter[0] += 1

                    # Make sure we can re-enable autograd here
                    with torch.overrides.enable_reentrant_dispatch():
                        foo = torch.rand(1, requires_grad=True)
                        self.assertIsNotNone(foo.exp().grad_fn)

                with no_dispatch():
                    return func(*args, **kwargs)

        a = torch.tensor(1.0)
        s = MySubclass(a)

        with fwAD.dual_level():
            # Only the primal has "alias" called on it
            fwAD.make_dual(s, torch.rand_like(s))
            self.assertEqual(counter[0], 1)
            fwAD.make_dual(torch.rand_like(s), s)
            self.assertEqual(counter[0], 1)

    def test_make_dual_forbid_integral_dtype(self):
        primal_f = torch.ones(2, 2, dtype=torch.float)
        primal_l = torch.ones(2, 2, dtype=torch.long)

        tangent_f = torch.ones(2, 2, dtype=torch.float)
        tangent_l = torch.ones(2, 2, dtype=torch.long)

        with fwAD.dual_level():
            # Float Primal and Long Tangent
            with self.assertRaisesRegex(
                ValueError, "Expected tangent to be floating point or complex"
            ):
                fwAD.make_dual(primal_f, tangent_l)

            # Long Primal and Long Tangent
            with self.assertRaisesRegex(
                ValueError, "Expected primal to be floating point or complex"
            ):
                fwAD.make_dual(primal_l, tangent_l)

            # Long Primal and Float Tangent
            with self.assertRaisesRegex(
                ValueError, "Expected primal to be floating point or complex"
            ):
                fwAD.make_dual(primal_l, tangent_f)

    def test_print(self):
        with fwAD.dual_level() as level:
            a = torch.rand(3)
            self.assertFalse("tangent=" in str(a))

            b = fwAD.make_dual(a, torch.rand(3))
            self.assertFalse("tangent=" in str(a))
            self.assertTrue("tangent=" in str(b))

            b_primal, b_tangent = fwAD.unpack_dual(b)
            self.assertFalse("tangent=" in str(b_primal))
            self.assertFalse("tangent=" in str(b_tangent))

    def test_basic_packing_unpacking(self):
        foo = torch.rand(2)
        bar = torch.rand(2)

        with fwAD.dual_level():
            baz = fwAD.make_dual(foo, bar)
            baz_primal, baz_tangent = fwAD.unpack_dual(baz)
            self.assertEqual(baz_primal, foo)
            self.assertIs(baz_tangent, bar)

            # Check unpacked dual is returned as a named tuple
            # NB: Every invocation of unpack_dual returns a new tensor view
            self.assertIsNot(baz_primal, fwAD.unpack_dual(baz).primal)
            self.assertEqual(baz_primal, fwAD.unpack_dual(baz).primal)
            self.assertIs(baz_tangent, fwAD.unpack_dual(baz).tangent)

            # Check that packing/unpacking did not change the input
            foo_primal, foo_tangent = fwAD.unpack_dual(foo)
            self.assertEqual(foo_primal, foo)
            self.assertIsNone(foo_tangent)

    def test_advanced_packing_unpacking(self):
        foo = torch.rand(2)
        bar = torch.ones(2)

        # Memory and version counter check
        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)

            # Ensure that they are sharing memory and version counter
            self.assertEqual(dual.storage().data_ptr(), foo.storage().data_ptr())

            # Ensure we properly share the version counter
            self.assertEqual(foo._version, dual._version)
            foo.add_(1)
            self.assertEqual(foo._version, dual._version)

            # Unpacking should only create aliases as well
            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            self.assertEqual(dual_primal.storage().data_ptr(), foo.storage().data_ptr())
            self.assertEqual(
                dual_tangent.storage().data_ptr(), bar.storage().data_ptr()
            )
            # And the tangent is actually reused as-is so it is still the same Tensor
            self.assertIs(dual_tangent, bar)

            # Ensure we properly share the version counter
            self.assertEqual(foo._version, dual_primal._version)
            foo.add_(1)
            self.assertEqual(foo._version, dual_primal._version)
            self.assertEqual(bar._version, dual_tangent._version)
            bar.add_(1)
            self.assertEqual(bar._version, dual_tangent._version)

        # backward mode check
        with fwAD.dual_level():
            foo.requires_grad_()
            bar.requires_grad_()

            # Check that backward gradients properly propagates through packing/unpacking
            dual = fwAD.make_dual(foo, bar)
            p, t = fwAD.unpack_dual(dual)

            gfoo, gbar = torch.autograd.grad(
                p.sum(), (foo, bar), retain_graph=True, allow_unused=True
            )
            self.assertEqual(gfoo, torch.ones_like(foo))
            self.assertIsNone(gbar)

            gfoo, gbar = torch.autograd.grad(
                t.sum(), (foo, bar), retain_graph=True, allow_unused=True
            )
            self.assertIsNone(gfoo)
            self.assertEqual(gbar, torch.ones_like(bar))

            # Check that forward gradients are impacted by detach()
            detached_dual = dual.detach()
            out = detached_dual * 2
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertEqual(p, foo * 2)
            self.assertIsNone(t)

            # Check that forward gradients are not impacted by no_grad
            with torch.no_grad():
                out = dual * 3
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertFalse(t.requires_grad)
            self.assertEqual(p, foo * 3)
            self.assertEqual(t, bar * 3)

            # Check that forward gradients are not impacted by inplace detach
            dual = dual.clone()
            dual.detach_()
            out = dual * 2
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertEqual(p, foo * 2)
            self.assertIsNone(t)

    def test_view_inplace_non_differentiable_views(self):
        original_foo = torch.rand(2, dtype=torch.double)
        original_bar = torch.ones(2, dtype=torch.double)

        # Do clones to be able to compare the values updated inplace
        # with the original content of these Tensors
        foo = original_foo.clone()
        bar = original_bar.clone()

        with fwAD.dual_level():
            # Note that in this test, we use "update" to mean computing the right tangent for the dual
            # All the inplace operations here are expected to update the primal value of the Tensors but
            # not always their tangents.
            # Also all mentions of "non differentiable view" here means non forward differentiable view
            # unless specified otherwise.
            # See note [Forward Grad View/inplace] for more details on how these views work.

            # Check that inplace ops do not update non-differentiable views
            # Non differentiable view
            dual = fwAD.make_dual(foo, bar)
            dual *= 2
            # Check that non differentiable view's tangent was not updated
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # Check that the computed result is correct
            self.assertEqual(bar, original_bar * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            self.assertEqual(foo, original_foo * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 2)
            # Other non differentiable view
            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            self.assertIsNone(fwAD.unpack_dual(dual_primal)[1])
            self.assertIsNone(fwAD.unpack_dual(dual_tangent)[1])
            dual_primal *= 2
            # Ensure dual's tangent did not change
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            dual_tangent *= 2
            # Ensure dual's primal did not change
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 4)

    def test_view_inplace_differentiable_views(self):
        original_foo = torch.rand(2)
        original_bar = torch.ones(2)

        # Do clones to be able to compare the values updated inplace
        # with the original content of these Tensors
        foo = original_foo.clone()
        bar = original_bar.clone()

        with fwAD.dual_level():
            # Check that inplace ops do update differentiable view but stop at non differentiable ones
            # A non differentiable view
            dual = fwAD.make_dual(foo, bar)
            # A differentiable view
            view = dual.narrow(0, 0, 1)
            view *= 2
            # Check that non differentiable view was not updated
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # Check that differentiable view was updated
            self.assertEqual(fwAD.unpack_dual(dual)[1], torch.tensor([2.0, 1.0]))
            self.assertEqual(fwAD.unpack_dual(view)[1], torch.tensor([2.0]))

            # Check that we track differentiable view even for Tensors that are not dual
            baz = torch.rand(2)
            baz += dual
            self.assertEqual(fwAD.unpack_dual(baz)[1], fwAD.unpack_dual(dual)[1])
            # Updates on view should as well
            baz = torch.rand(2)
            baz[0] = dual[0]
            self.assertEqual(fwAD.unpack_dual(baz)[1][0], fwAD.unpack_dual(dual)[1][0])
            # Unused values get a gradient of 0
            self.assertEqual(fwAD.unpack_dual(baz)[1][1], 0.0)

            # Check that forward non-differentiable views do prevent gradient update
            baz = torch.rand(2)
            view = baz.detach()
            view += dual
            self.assertIsNone(fwAD.unpack_dual(baz)[1])

    def test_view_inplace_always_creates_a_view(self):
        # See https://github.com/pytorch/pytorch/issues/67800
        # The codepath may depend on the op. At the time writing, when self is not a dual tensor
        # the resulting forward grad for self for...
        # - add_ has the same layout as self
        # - mul_ has the same layout as other
        # This is kind of fragile because the above depends on how the forward grad expression
        # is written. For add and mul at least, the output inherits the layout of LHS.
        # We want to handle at least these two cases.
        inplace_binary_ops = (  # Add more to this list?
            lambda x, y: x.add_(y),
            lambda x, y: x.mul_(y),
            lambda x, y: x.copy_(y),
        )

        for inplace_binary_op in inplace_binary_ops:
            base = torch.randn(2, 2)
            view = base.transpose(0, 1)

            primal = torch.randn(2, 2)
            tangent = torch.randn(2, 2)

            with fwAD.dual_level():
                dual = fwAD.make_dual(primal, tangent)
                inplace_binary_op(view, dual)

                # Verify that a view relationship is created for both the primal and tangent
                p, t = fwAD.unpack_dual(base)
                p_clone = p.clone()
                t_clone = t.clone()
                view *= 2
                p, t = fwAD.unpack_dual(base)

                self.assertTrue(torch.allclose(p_clone * 2, p))
                self.assertTrue(torch.allclose(t_clone * 2, t))

    def test_grad_cleanup(self):
        foo = torch.rand(2)
        bar = torch.rand(2)
        baz = torch.rand(2)

        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            self.assertIs(fwAD.unpack_dual(dual)[1], bar)

        self.assertIsNone(fwAD.unpack_dual(dual)[1])

        with fwAD.dual_level():
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            new_dual = fwAD.make_dual(foo, baz)

            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            new_dual_primal, new_dual_tangent = fwAD.unpack_dual(new_dual)
            self.assertEqual(dual_primal, new_dual_primal)
            self.assertIsNone(dual_tangent)
            self.assertEqual(new_dual_tangent, baz)

    def test_detach_view_tracking(self):
        # Default detach is both forward and backward non-differentiable
        foo = torch.rand(2)
        foo_weak = torch._C._WeakTensorRef(foo)

        out = foo.detach()

        del foo
        self.assertTrue(foo_weak.expired())

    def test_out_variant(self):
        with fwAD.dual_level():
            foo = fwAD.make_dual(torch.rand(2), torch.rand(2))
            bar = torch.rand(2)

            with self.assertRaisesRegex(RuntimeError, "out= function"):
                torch.add(bar, bar, out=foo)

            with self.assertRaisesRegex(RuntimeError, "out= function"):
                torch.add(foo, bar, out=bar)

    def test_non_differentiable(self):
        with fwAD.dual_level():
            foo = fwAD.make_dual(torch.rand(2), torch.rand(2))
            bar = torch.rand(2)

            # No differentiable outputs, shouldn't error
            eq = foo == bar

            # Inplace
            foo.eq_(bar)

    def test_create_new_zeros_with_same_meta(self):
        new_zeroes_fn = torch.ops.aten._new_zeros_with_same_feature_meta

        def check(a, b):
            def assert_same_meta(t, target):
                for num_bdim in range(t.dim()):
                    result = new_zeroes_fn(t, target, self_num_batch_dims=num_bdim)

                    self.assertEqual(result.dim(), target.dim() + num_bdim)

                    # Check size/strides match for feature dims only
                    for i in range(num_bdim, result.dim()):
                        self.assertEqual(result.size()[i], target.size()[i - num_bdim])
                        self.assertEqual(
                            result.stride()[i], target.stride()[i - num_bdim]
                        )

                    # Check that we generate strides reasonably
                    if target.is_contiguous():
                        self.assertTrue(result.is_contiguous())

                    self.assertEqual(result.storage_offset(), target.storage_offset())

                    prod_of_t_bdims = reduce(operator.mul, t.size()[:num_bdim], 1)
                    self.assertEqual(
                        len(result.storage()), len(target.storage()) * prod_of_t_bdims
                    )

                    # TensorOptions is same
                    self.assertEqual(result.dtype, target.dtype)

            assert_same_meta(a, b)
            assert_same_meta(b, a)

        a = torch.randn(5, dtype=torch.float)
        b = torch.randn(2, 3, 4, dtype=torch.double)
        check(a, b)

        # non-contiguous case
        a = torch.randn(2, 3, 4).transpose(0, 1).contiguous().transpose(0, 1)
        b = torch.randn(2, 3, 4)
        check(a, b)

        a = torch.randn(5).narrow(0, 1, 2)
        b = torch.randn(2)
        check(a, b)

        # tensor is not a view, but still does not index entirety of storage
        a = torch.randn(5).resize_(4)
        b = torch.randn(4)
        check(a, b)

        # Zero-numel tensors
        a = torch.randn(1, 0, 2)
        b = torch.randn(1, 2)
        check(a, b)

        # Scalar tensor
        a = torch.tensor(1.0)
        b = torch.randn(1, 2)
        check(a, b)

    def test_backward_graph_destruction(self):
        def fn():
            a = torch.rand(10, requires_grad=True)

            da = fwAD.make_dual(torch.rand_like(a), a)

            # Create an object with a c++ cycle as:
            # db -> AutogradMeta -> ForwardGrad -> db's grad
            # db's grad -> AutogradMeta -> MulBackward
            # MulBackward -> SavedVariable -> db
            db = da.exp()

        with fwAD.dual_level():
            fn()
        # This test make sure that we don't deadlock on exit of this
        # context manager. If you do, there is something wrong with the
        # locking of the forward ad level most likely


# Generic device type autograd tests.
class TestAutogradDeviceType(TestCase):
    def test_min_max_median_backprops_to_all_values(self, device):
        for f in [torch.min, torch.max, torch.median, torch.nanmedian]:
            x1 = torch.tensor(
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], device=device, requires_grad=True
            )
            x2 = torch.tensor(
                [float("nan"), float("nan"), float("nan")], requires_grad=True
            )
            for x in [x1, x2]:
                y = f(x)
                y.backward()
                self.assertEqual(x.grad.sum(), 1.0)
                self.assertEqual((x.grad == 1 / 3).sum(), 3)

    def test_scatter_index_reduce_amin_amax_backprops_to_all_values(self, device):
        # tests that gradients are evenly distributed when there are multiple max/min values
        # tested here instead of adding a SampleInput as the backward for this case is non-differentiable for gradgrad
        # as is the case for test_min_max_median_backprops_to_all_values above
        fns = (torch.scatter_reduce, torch.index_reduce)
        reduces = ("amin", "amax")
        for fn, reduction in product(fns, reduces):
            input = torch.randn(
                (2, 3), device=device, dtype=torch.float64, requires_grad=True
            )
            src = input.clone().detach_().requires_grad_(True)
            idx = torch.arange(2).to(dtype=torch.long, device=device)
            if fn == torch.scatter_reduce:
                idx = idx.unsqueeze(-1).expand((2, 3))

            gradcheck(fn, (input, 0, idx, src, reduction), check_batched_grad=False)

    def test_scatter_index_reduce_prod_gradgrad_error(self, device):
        # test that double backward raises an error for the case where 2 zeros in src
        # are scattered to the same position in self
        input = torch.tensor(
            [1.0], device=device, dtype=torch.float64, requires_grad=True
        )
        src = torch.tensor(
            [0.0, 0.0], device=device, dtype=torch.float64, requires_grad=True
        )
        idx = torch.tensor([0, 0], device=device, dtype=torch.long)

        for fn in (torch.scatter_reduce, torch.index_reduce):
            # check that this case passes on gradcheck
            gradcheck(fn, (input, 0, idx, src, "prod"), check_batched_grad=False)
            with self.assertRaisesRegex(
                RuntimeError, "Double backward is unsupported for"
            ):
                gradgradcheck(fn, (input, 0, idx, src, "prod"))

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_parameter_resize(self, device):
        asd = torch.nn.Parameter(torch.ones(16, dtype=torch.double, device=device))

        for _ in range(2):
            with torch.no_grad():
                asd.set_(asd[1:])
                asd.grad = None

            m = torch.cat((asd, asd))
            m.sum().backward()

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    @dtypes(torch.double, torch.cdouble)
    def test_sparse_ctor_getter_backward(self, device, dtype):
        # See NOTE [ Sparse: autograd and API ] on the expected behavior of this test
        def _test(size, sparse_dim, nnz, device):
            v_size = [nnz] + list(size[sparse_dim:])
            i = torch.rand(sparse_dim, nnz)
            i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
            i = i.to(torch.long)

            inp = torch.randn(
                v_size, dtype=torch.double, device=device, requires_grad=True
            )
            other = self.genSparseTensor(
                size, sparse_dim, nnz, is_uncoalesced=True, device=device, dtype=dtype
            )[0]

            def fn(v):
                x = torch.sparse_coo_tensor(i, v, size, dtype=dtype, device=device)
                y = (x + other).coalesce()
                yv = y.values()
                new_v = yv.tanh()
                z = torch.sparse_coo_tensor(y.indices(), new_v, y.size())
                return z.coalesce().values()

            gradcheck(fn, (inp,), check_batched_grad=False)
            # FIXME: make gradgradcheck work.
            # gradgradcheck(fn, (inp,), check_batched_grad=False)

            # assert that _values is non-differentiable
            with self.assertRaisesRegex(RuntimeError, "does not have a grad_fn"):
                other.detach().requires_grad_()._values().backward(
                    torch.ones_like(other._values())
                )

        for empty_i, empty_v, empty_nnz in product([True, False], repeat=3):
            sparse_size = [] if empty_i else [2, 1]
            dense_size = [1, 0, 2] if empty_v else [1, 2]
            nnz = 0 if empty_nnz else 5
            _test(sparse_size + dense_size, len(sparse_size), nnz, device)

    @skipMeta
    @skipIfMPS
    @dtypes(torch.double, torch.cdouble)
    def test_sparse_backward(self, device, dtype):
        class FixedGradientFunction(Function):
            @staticmethod
            def forward(ctx, x, grad_x):
                ctx.save_for_backward(grad_x)
                return x

            @staticmethod
            def backward(ctx, grad_x):
                (saved_grad_x,) = ctx.saved_tensors
                return saved_grad_x, None

        size = torch.Size([6, 3, 2])
        i1 = torch.tensor([[0, 3, 4], [0, 2, 2]], dtype=torch.long)
        v1 = make_tensor([3, 2], dtype=dtype, device=device)
        sparse_grad1 = torch.sparse_coo_tensor(i1, v1, size, dtype=dtype, device=device)
        i2 = torch.tensor([[0, 1, 3, 4], [0, 1, 2, 2]], dtype=torch.long)
        v2 = make_tensor([4, 2], dtype=dtype, device=device)
        sparse_grad2 = torch.sparse_coo_tensor(i2, v2, size, dtype=dtype, device=device)
        dense_grad = torch.rand(size, device=device, dtype=dtype)
        fn = FixedGradientFunction

        # sparse first
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (
            fn.apply(x, sparse_grad1)
            + fn.apply(x, dense_grad)
            + fn.apply(x, sparse_grad2)
        ).sum().abs().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        # dense first
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (
            fn.apply(x, dense_grad)
            + fn.apply(x, sparse_grad1)
            + fn.apply(x, sparse_grad2)
        ).sum().abs().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        # sparse only
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (fn.apply(x, sparse_grad1) + fn.apply(x, sparse_grad2)).sum().abs().backward()
        self.assertEqual(x.grad, sparse_grad1 + sparse_grad2)

    @skipIfMPS
    def test_sparse_mask_autograd(self, device):
        tensor = torch.randn(3, requires_grad=True, device=device)
        mask = torch.ones(3, device=device)
        mask[1] = 0
        mask = mask.to_sparse()
        converted = tensor.sparse_mask(mask).to_dense()
        converted.sum().backward()
        self.assertEqual(tensor.grad, mask.to_dense())

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_pyscalar_conversions(self, device):
        def _test_pyscalar_conversions(t, integral_conv):
            # integral -> integral
            l = t(torch.zeros(1, 1, 1, dtype=torch.long))
            pyscalar = -12345
            l[0] = pyscalar
            self.assertEqual(integral_conv(l), pyscalar)

            # floating point -> floating point
            f = Variable(t(torch.randn(1, 1, dtype=torch.double)))
            pyscalar = -12345.1
            f[0] = pyscalar
            self.assertEqual(float(f), pyscalar)
            f[0] = nan
            self.assertTrue(math.isnan(float(f)))
            f[0] = inf
            self.assertEqual(float(f), inf)
            f[0] = -inf
            self.assertEqual(float(f), -inf)

            # integral -> floating point
            # check we can convert something that loses precision
            pyscalar = 1234567890123456789
            self.assertNotEqual(pyscalar, integral_conv(float(pyscalar)))
            l[0] = pyscalar
            self.assertEqual(float(l), float(pyscalar))

            # floating point -> integral
            f[0] = nan
            self.assertRaises(ValueError, lambda: integral_conv(f[0]))
            f[0] = inf
            self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
            f[0] = -inf
            self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
            f[0] = sys.float_info.max
            self.assertEqual(integral_conv(f), sys.float_info.max)

            # bool, nonzero
            def test_nonzero(tensor, value, expected):
                tensor[0] = value
                self.assertEqual(expected, bool(tensor))
                self.assertEqual(expected, bool(tensor))

            test_nonzero(l, 0, False)
            test_nonzero(l, -2, True)
            test_nonzero(f, 0.0, False)
            test_nonzero(f, sys.float_info.min, True)
            test_nonzero(f, nan, bool(nan))
            test_nonzero(f, inf, bool(inf))
            test_nonzero(f, -inf, bool(-inf))

        _test_pyscalar_conversions(lambda x: x.to(device), lambda x: int(x))

    @dtypesIfMPS(torch.float32)
    @dtypesIfCUDA(
        torch.half,
        torch.float,
        torch.double,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    @dtypes(
        torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64
    )
    def test_set_requires_grad_only_for_floats(self, device, dtype):
        def f1():
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad_()

        def f2():
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad = True

        def f3():
            torch.ones(1, dtype=dtype, device=device, requires_grad=True)

        a = torch.ones(1, dtype=dtype, device=device)
        a.requires_grad = False  # should always work
        a.requires_grad_(False)

        for f in [f1, f2, f3]:
            if dtype.is_floating_point:
                f()
            else:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "floating point",
                    msg=f"dt: {a.dtype} device: {a.device}",
                ):
                    f()

    @onlyCUDA
    def test_advanced_indexing_backwards_large(self, device):
        # See https://github.com/pytorch/pytorch/issues/22843
        n = 1 << 16
        x = torch.rand(n, 1, device=device, requires_grad=True)
        a = x[:, [0]]
        a.sum().backward()
        self.assertEqual(x.grad, torch.ones(n, 1, device=device))

    def test_advanced_indexing_backwards_memory_format(self, device):
        # See https://github.com/pytorch/pytorch/issues/36956
        shape = (2, 8, 1, 2)
        i = torch.randint(1, shape, device=device).contiguous(
            memory_format=torch.channels_last
        )
        x = torch.randn(shape, requires_grad=True, device=device)
        x[i].sum().backward()

    def _test_reentrant_parent_error_on_cpu(self, device):
        t1 = torch.rand([3, 3], requires_grad=True)
        t2 = torch.rand([3, 3], device=device, requires_grad=True)
        t3 = torch.rand([3, 3], device=device, requires_grad=True)

        # Parent graph cpu graph.
        t4 = t1 * t1
        t5 = TestAutograd.SimulateBackwardError.apply(t4)

        # Child gpu graph (much longer than parent graph).
        prev = t2 * t2
        for _ in range(10):
            prev = prev * t2
        reentrant_root = prev

        class ReentrantFunc(Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                # Reentrant backward in child will take much longer.
                reentrant_root.backward()
                return grad

        # Parent gpu graph.
        t6 = ReentrantFunc.apply(t3)
        t7 = t6 * t6

        # Parent graph will error out first, while child graph will continue executing.
        with self.assertRaisesRegex(Exception, "Simulate error"):
            torch.autograd.backward([t5.sum(), t7.sum()])

        # No grads should be accumulated since child graph will stop execution
        # after parent receives error.
        self.assertIsNone(t2.grad)
        self.assertIsNone(t1.grad)
        self.assertIsNone(t3.grad)

    @onlyCUDA
    def test_reentrant_parent_error_on_cpu(self, device):
        def _get_cuda_memory_usage():
            # we don't need CUDA synchronize because the statistics are not tracked at
            # actual freeing, but at when marking the block as free.
            num_devices = torch.cuda.device_count()
            gc.collect()
            return tuple(torch.cuda.memory_allocated(i) for i in range(num_devices))

        before = _get_cuda_memory_usage()

        # Run as separate function so that gc can clean up everything when we
        # check for memory usage.
        self._test_reentrant_parent_error_on_cpu(device)

        # Wait for autograd thread to cleanup failed tasks.
        after = _get_cuda_memory_usage()
        start = time.time()
        while before != after and time.time() - start < 30:
            time.sleep(0.1)
            after = _get_cuda_memory_usage()

        self.assertEqual(before, after)

    @skipIfMPS  # the test doesn't work on MPS
    # TODO: see if these tests can be ported to OpInfos or moved to where's test suite
    def test_where_functional(self, device):
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        def where(cond, x, y):
            return torch.where(cond, x, y)

        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, device=device)])

        x = torch.randn(5, 1, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, 1, dtype=torch.double, device=device, requires_grad=True)
        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, 5, device=device)])

    @skipIfMPS  # the test doesn't work on MPS
    def test_where_scalar(self, device):
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        scalar = 4.0
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        def where_scalar_first(cond, x):
            return torch.where(cond, scalar, x)

        def where_scalar_second(cond, x):
            return torch.where(cond, x, scalar)

        gradcheck(where_scalar_first, (cond, x))
        gradgradcheck(where_scalar_first, (cond, x))

        gradcheck(where_scalar_second, (cond, x))
        gradgradcheck(where_scalar_second, (cond, x))

    @onlyCUDA
    def test_free_unneeded_tensor(self, device):
        x = torch.randn(2, 3, 10, 10, device=device, requires_grad=True)
        m = torch.randn(1, 3, 1, 1, device=device)

        z = x.sum()
        base_mem = torch.cuda.memory_allocated()
        z = ((x + 2) * m).sum()
        end_mem = torch.cuda.memory_allocated()

        # In the end the memory usage should remain equal, because neither of
        # (x + 2) and ((x + 2) * m) should be kept alive for backward, while the
        # previous allocation of z had the same size as the current one.
        self.assertEqual(base_mem, end_mem)

    @onlyCUDA
    def test_pin_memory(self, device):
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        self.assertEqual(x, x.pin_memory())
        self.assertIsNot(x, x.pin_memory())
        self.assertTrue(x.pin_memory().requires_grad)
        gradcheck(lambda x: x.pin_memory(), [x])
        gradgradcheck(lambda x: x.pin_memory(), [x])

    @onlyCUDA
    def test_profiler_emit_nvtx(self, device):
        # This test is not intended to ensure correctness of nvtx ranges.
        # That would require something a great deal more complex (you'd have to create a
        # profile in a subprocess, open it, and parse the sql somehow).
        # This test is merely intended to catch if emit_nvtx breaks on construction.
        a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
        with torch.cuda.profiler.profile():
            with emit_nvtx():
                a.add(1.0)

    @onlyCUDA
    def test_rnn_backward_to_input_but_not_parameters(self, device):
        # this checks whether it is possible to not require
        # weight parameters, but require inputs, see #7722
        l = torch.nn.LSTM(2, 3).to(device)
        for p in l.parameters():
            p.requires_grad = False
        s = torch.randn(1, 1, 2, requires_grad=True, device=device)
        out, _ = l(s)
        out.sum().backward()
        self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    @unittest.skipIf(not torch.profiler.itt.is_available(), "ITT is required")
    def test_profiler_emit_itt(self, device):
        # This test is not intended to ensure correctness of itt ranges.
        # That would require something a great deal more complex (you'd have to create a
        # profile in a subprocess, open it, and parse the sql somehow).
        # This test is merely intended to catch if emit_itt breaks on construction.
        a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
        with emit_itt():
            a.add(1.0)

    @skipIfMPS  # the test doesn't work as randn is not supported with type long
    @deviceCountAtLeast(1)
    def test_grad_assignment(self, devices):
        x = torch.randn(5, 5, device=devices[0])

        # Tests that the wrong type raises
        with self.assertRaisesRegex(TypeError, "expected to be a Tensor or None"):
            x.grad = 0

        # Tests that the wrong shape raises
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(2, 2, device=devices[0])

        # Tests that the wrong dtype raises
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(5, 5, dtype=torch.long, device=devices[0])

        # Tests that self-assignment raises
        with self.assertRaises(RuntimeError):
            x.grad = x

        # Tests device -> cpu grad assignment raises
        if self.device_type != "cpu":
            with self.assertRaises(RuntimeError):
                t_cpu = torch.rand(5, 5)
                t_cpu.grad = torch.randn(5, 5, device=devices[0])

        # Tests half type on CUDA
        if self.device_type == "cuda":
            x = x.to(dtype=torch.half, device=devices[0])
            x.grad = torch.zeros_like(x)

        # Tests cross-device assignment raises
        if len(devices) > 1:
            x = torch.randn(5, 5, device=devices[0])
            with self.assertRaises(RuntimeError):
                x.grad = torch.randn(5, 5, device=devices[1])

    @dtypesIfMPS(torch.float32)
    @deviceCountAtLeast(1)
    @dtypes(torch.float, torch.double)
    def test_requires_grad_factory(self, devices, dtype):
        fns = [torch.ones_like, torch.randn_like]
        x = torch.randn(2, 3, dtype=dtype, device=devices[0])

        for fn in fns:
            for requires_grad in [True, False]:
                output = fn(
                    x, dtype=dtype, device=devices[0], requires_grad=requires_grad
                )
                self.assertEqual(requires_grad, output.requires_grad)
                self.assertIs(dtype, output.dtype)
                self.assertEqual(devices[0], str(x.device))

    @deviceCountAtLeast(2)
    def test_unused_output_device(self, devices):
        from torch.nn.parallel._functions import Broadcast

        x = torch.randn(5, 5, dtype=torch.float, device=devices[0], requires_grad=True)
        outputs = Broadcast.apply(list(range(len(devices))), x)
        y = outputs[-1] * 2
        y.sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5) * 2)

    @deviceCountAtLeast(2)
    def test_backward_device(self, devices):
        # check that current device matches the variable's device
        device = [None]

        class Identity(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                device[0] = grad_output.device
                return grad_output.clone()

        v = torch.randn(1, device=devices[1], requires_grad=True)
        Identity.apply(v).backward()
        self.assertEqual(str(device[0]), devices[1])

    @deviceCountAtLeast(2)
    def test_inputbuffer_add_multidevice(self, devices):
        input = torch.randn(1, device=devices[0], requires_grad=True)
        output = input.to(device=devices[1]) + input.to(device=devices[1])
        output.backward()

    @onlyCPU
    def test_copy_(self, device):
        # At the time of writing this test, copy_ is not generated from native_functions.yaml
        # there was a bug that bfloat16 was not recognized as floating.
        x = torch.randn(10, device=device, requires_grad=True)
        floating_dt = floating_types_and(torch.half, torch.bfloat16)
        for dt in floating_dt:
            y = torch.empty(10, device=device, dtype=dt)
            y.copy_(x)
            self.assertTrue(y.requires_grad)
            z = x.to(torch.bfloat16)
            self.assertTrue(z.requires_grad)

    def test_copy_forward_ad_broadcasting(self, device):
        # copy_ allows the src to have a different shape from self as long as src is
        # broadcastable to self. Make sure forward AD handles this case.
        primal = torch.rand(3, 3, device=device)
        tangent = torch.rand(3, 3, device=device)
        non_dual = torch.rand(1, 3, 3, device=device)

        with fwAD.dual_level():
            dual = fwAD.make_dual(primal, tangent)
            non_dual.copy_(dual)

    def test_copy_forward_ad_same_layout_copies_grad(self, device):
        primal = torch.tensor([[3.0], [4.0]], device=device)
        tangent = torch.tensor([[5.0], [6.0]], device=device)

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(primal, tangent)
            non_dual = torch.tensor([[1.0], [2.0]])
            non_dual.copy_(x_dual)
            self.assertTrue(fwAD.unpack_dual(non_dual).tangent is not tangent)

    @onlyCUDA
    def test_simple_reentrant_cross_device(self, device):
        class ReentrantFunc(Function):
            _cpu_mode = True

            @staticmethod
            def forward(ctx, x):
                return x * (x + 2)

            @staticmethod
            def backward(ctx, grad_output):
                with torch.enable_grad():
                    if ReentrantFunc._cpu_mode:
                        new_param = torch.randn(2, 2, requires_grad=True)
                        (new_param**2).sum().backward()
                    else:
                        new_param = torch.randn(2, 2, device=device, requires_grad=True)
                        (new_param**2).sum().backward()
                return grad_output

        # Reentrant starts on GPU thread, finishes on GPU thread
        x = torch.randn(2, 2, device=device, requires_grad=True)
        out = ReentrantFunc.apply(x)
        out.sum().backward()

        # Reentrant starts on CPU thread, finishes on GPU thread
        x = torch.randn(2, 2, requires_grad=True)
        # set ReentrantFunc node to GPU to emit tasks to GPU queue
        ReentrantFunc._cpu_mode = False
        out = ReentrantFunc.apply(x)
        out.sum().backward()

        # Reentrant starts on GPU thread, finishes on CPU thread
        x = torch.randn(2, 2, device=device, requires_grad=True)
        # set ReentrantFunc node to CPU to emit tasks to CPU queue
        ReentrantFunc._cpu_mode = True
        out = ReentrantFunc.apply(x)
        out.sum().backward()

    @onlyCUDA
    def test_cross_device_reentrant_autograd(self, device):
        # Output on gpu so that this task will be associated with the gpu thread
        def fn_on_gpu(inp):
            # Artificially increase the priority of the next op to make sure it runs
            # as soon as we reach it before the ops of branch1.
            dummy = inp * 2 * 2 * 2 * 2
            return inp.to(device=device)

        def parent_on_cpu(inp):
            # Slow branch of ops on gpu so that the work queue for the gpu thread
            # won't empty too quickly. They also have smaller priorities than the
            # ones created by fn_on_gpu
            branch1 = inp.to(device=device)
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            # Perform checkpoint on cpu tensors. So the last op performed in the reentrant
            # autograd is an AccumulateGrad that runs on the cpu thread for the gpu thread.
            # So the cpu thread will notify the gpu thread with an empty NodeTask.
            branch2 = checkpoint(fn_on_gpu, inp, use_reentrant=True)
            out = branch2 + branch1
            return out

        inp = torch.rand(2, requires_grad=True)
        out = parent_on_cpu(inp)
        # This will segfault if the empty NodeTask is not handled properly in the
        # gpu thread ReadyQueue
        out.sum().backward()

    def test_inplace_on_view_backprop_base(self, device):
        # modify view and back-prop through base
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v1.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[2, 2], [1, 1]])

    def test_inplace_on_view_backprop_view_of_view(self, device):
        # modify view and backprop through view-of-view
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = x.narrow(0, 0, 1)
        v1.mul_(2)
        v2.sum().backward()
        self.assertEqual(root.grad.tolist(), [[2, 2], [0, 0]])

    def test_inplace_on_view_of_view(self, device):
        # modify view-of-view and backprop through base
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()

        v1 = x.narrow(0, 0, 1)
        v2 = v1.narrow(1, 1, 1)
        v2.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1]])

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_inplace_on_view_then_no_grad(self, device):
        # Perform an in-place operation on a view of a non-leaf variable.
        a = torch.ones(3, 1, dtype=torch.double, device=device, requires_grad=True)
        b = a * 2
        c = b.view_as(b)
        c[0][0] = 3

        # Force a graph update with grad disabled.
        with torch.no_grad():
            c.grad_fn

        c.sum().backward()

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_inplace_on_view_gradcheck(self, device):
        # gradcheck modifications to views
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=True)
        b = torch.randn(2, 2, dtype=torch.double, device=device, requires_grad=True)

        def func(root, b):
            x = root.clone()
            x.narrow(1, 2, 2).narrow(0, 1, 2).mul_(b)
            x.narrow(1, 0, 2).narrow(0, 1, 2).mul_(b)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(
            a.size(), dtype=torch.double, device=device, requires_grad=True
        )
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_on_view_multiple_outputs(self, device):
        root = torch.arange(9.0, dtype=torch.double).reshape(3, 3).requires_grad_()
        x = root.clone()
        v1 = x.unbind()
        with self.assertRaises(RuntimeError):
            v1[0].mul_(2)

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_inplace_on_view_of_multiple_output_view(self, device):
        a = torch.rand(
            10, dtype=torch.double, device=device, requires_grad=True
        ).clone()
        b = a.unbind(0)
        c = b[0].view_as(b[0])
        with self.assertRaises(RuntimeError):
            c.mul_(2)

    @skipIfMPS  # MPS backend doesn't support double types
    def test_inplace_multiple_output_view_of_view(self, device):
        a = torch.rand(
            10, dtype=torch.double, device=device, requires_grad=True
        ).clone()
        b = a.view_as(a)
        c = b.unbind(0)
        with self.assertRaises(RuntimeError):
            c[0].mul_(2)

    @skipIfMPS  # MPS backend doesn't support double types
    def test_inplace_on_view_makes_base_require_grad(self, device):
        # in-place modification to view makes base require grad
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=False)
        b = torch.randn(4, 2, dtype=torch.double, device=device, requires_grad=True)

        def func(root, b):
            x = root.clone()
            self.assertFalse(x.requires_grad)
            x.narrow(1, 2, 2).mul_(b)
            self.assertTrue(x.requires_grad)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(
            a.size(), dtype=torch.double, device=device, requires_grad=True
        )
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_on_view_backprop_view(self, device):
        # modify view and backprop through view
        a = torch.tensor([2.0, 5.0], device=device, requires_grad=False)
        b = torch.tensor([3.0], device=device, requires_grad=True)
        res = a.narrow(0, 1, 1).mul_(b)
        res.sum().backward()
        self.assertEqual(b.grad.tolist(), [5])
        self.assertIsNone(a.grad)

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_inplace_on_view_modify_base(self, device):
        # Test that an in-place operation on a base that forced it to require
        # grad also forces any previous views to require grad and backprop
        # correctly
        r = torch.ones(1, dtype=torch.double, device=device, requires_grad=True)

        def fn(r):
            x = torch.ones(5, dtype=torch.double, device=device)
            v = x.select(0, 1)
            self.assertFalse(v.requires_grad)
            self.assertIsNone(v.grad_fn)
            x.add_(r)  # v is now dependent on r due to the in-place op on x
            self.assertTrue(v.requires_grad)
            return v

        gradcheck(fn, [r])
        gradgradcheck(fn, [r])

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_inplace_on_view_python(self, device):
        # in-place modifications of Python-autograd created view
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=True)
        b = torch.randn(2, 2, dtype=torch.double, device=device, requires_grad=True)

        class PyAdd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.mark_dirty(x)
                x.add_(y)
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad, grad

        def func(root, b):
            x = root.clone()
            PyAdd.apply(x.narrow(1, 2, 2).narrow(0, 1, 2), b)
            PyAdd.apply(x.narrow(1, 0, 2).narrow(0, 1, 2), b)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(
            a.size(), dtype=torch.double, device=device, requires_grad=True
        )
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_on_view_non_contig(self, device):
        root = torch.ones(2, 3, 2, device=device).select(2, 1).t().requires_grad_(True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = v1.narrow(1, 1, 1)
        v2.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1], [1, 1]])

    def test_inplace_on_view_multi_output_unsafe(self, device):
        for f in [
            lambda t: t.unsafe_split(1),
            lambda t: t.unsafe_split_with_sizes((1, 1, 1)),
            lambda t: t.unsafe_chunk(3),
        ]:
            a = torch.randn(3, 3, device=device, requires_grad=True)
            b = a + a
            s1, s2, s3 = f(b)
            s1.mul_(s2)
            s1.sum().backward()

    def test_inplace_on_view_multi_output_safe(self, device):
        for f in [
            lambda t: t.split(1),
            lambda t: t.split_with_sizes((1, 1, 1)),
            lambda t: t.chunk(3),
        ]:
            a = torch.randn(3, 3, device=device, requires_grad=True)
            b = a + a
            s1, s2, s3 = f(b)
            error_msg = (
                "This view is the output of a function that returns multiple views."
            )
            with self.assertRaisesRegex(RuntimeError, error_msg):
                s1.mul_(s2)

    def test_inplace_on_view_undefined_grad_output(self, device):
        a = torch.tensor([1.0], requires_grad=True)
        c = a.clone()
        v = c[:]
        b = torch.tensor(1.0, requires_grad=True)

        class InplaceFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, other):
                ctx.mark_dirty(x)
                return x.mul_(2)

            @staticmethod
            def backward(ctx, grad):
                return grad * 2, None

        out = InplaceFunc.apply(v, b)
        out.backward()
        self.assertIsNone(b.grad)
        self.assertEqual(a.grad.item(), 2)

    @skipIfMPS  # the test doesn't work on MPS as double types are not supported
    def test_mv_grad_stride_0(self, device):
        # Reference: https://github.com/pytorch/pytorch/issues/38315
        mat = torch.randn(2, 2, dtype=torch.double, device=device)
        vec = torch.randn(1, dtype=torch.double, device=device).requires_grad_(True)

        def fn(vec):
            # Expand inside the function to make sure the input to
            # gradcheck does not have overlapping memory
            vec = vec.expand(2)
            return (mat @ vec).sum()

        gradcheck(fn, (vec))
        gradgradcheck(fn, (vec))

    @onlyCUDA
    def test_gradcheck_input_output_different_device(self, device):
        x = torch.ones((1,), dtype=torch.double, device="cuda", requires_grad=True)
        gradcheck(lambda x: x.to("cpu"), (x,))

        x = torch.ones((1,), dtype=torch.double, device="cpu", requires_grad=True)
        gradcheck(lambda x: x.to("cuda"), (x,))

    def test_strided_leaf_grad_layout(self, device):
        # (1) If leaf is non-overlapping and dense, grad's layout should match its leaf.
        for fmt_a in (torch.contiguous_format, torch.channels_last):
            for fmt_b in (torch.contiguous_format, torch.channels_last):
                a = torch.rand((2, 3, 4, 5), device=device).to(memory_format=fmt_a)
                b = torch.rand((2, 3, 4, 5), device=device).to(memory_format=fmt_b)
                a.requires_grad_()
                b.requires_grad_()
                # checks (1) for broadcasted gradients
                a.sum().backward()
                self.assertEqual(a.grad.stride(), a.stride())
                b.sum().backward()
                self.assertEqual(b.grad.stride(), b.stride())
                # checks (1) for non-broadcasted gradients
                a.grad = None
                b.grad = None
                (a * b).sum().backward()
                self.assertEqual(a.grad.stride(), a.stride())
                self.assertEqual(b.grad.stride(), b.stride())

        # (2) If leaf isn't dense, checks that grads are rowmajor contiguous.
        c = torch.empty_strided((2, 2), (4, 2), device=device).copy_(
            torch.rand((2, 2), device=device)
        )
        c.requires_grad_()
        d = torch.rand((2, 2), device=device)
        # checks (2) for broadcasted gradients
        c.sum().backward()
        self.assertEqual(c.grad.stride(), (2, 1))
        # checks (2) for non-broadcasted gradients
        c.grad = None
        (c * d).sum().backward()
        self.assertEqual(c.grad.stride(), (2, 1))

    @skipIfMPS
    def test_copy_r_to_c(self, device):
        out_c = torch.empty(3, 2, dtype=torch.cdouble, device=device)
        inp_r = torch.randn(3, 2, dtype=torch.double, device=device, requires_grad=True)

        def do_test():
            out_c.copy_(inp_r)
            out_c_inter = out_c.sum()
            out_c_inter.abs().backward()
            with torch.no_grad():
                self.assertEqual(
                    inp_r.grad, torch.ones_like(inp_r) * torch.sgn(out_c_inter).real
                )

        self.assertNotWarn(do_test)

    def test_to_r_to_c(self, device):
        def do_test():
            inp_r = torch.randn(
                3, 2, dtype=torch.double, device=device, requires_grad=True
            )
            out = inp_r.to(torch.complex128)
            out_inter = out.sum()
            out_inter.abs().backward()
            with torch.no_grad():
                self.assertEqual(
                    inp_r.grad, torch.ones_like(inp_r) * torch.sgn(out_inter).real
                )

        self.assertNotWarn(do_test)

    def test_non_differentiable_ops(self, device):
        # Just make sure the op doesn't raise an error
        # and resulting tensor has requires_grad=False.
        x = torch.tensor([[1, 2], [3, 4.0]], requires_grad=True, device=device)
        out = torch.isin(x, torch.tensor([2, 3], device=device))
        self.assertFalse(out.requires_grad)

        x = torch.randn(3, 3, requires_grad=True)
        out = torch.signbit(x)
        self.assertFalse(out.requires_grad)

    def test_warning_in_backward(self, device):
        # Test warning during backward are always propagated as python warnings (gh-50209)
        # NOTE: For device=cuda, warning gets propagated from a worker thread
        a = torch.zeros((), device=device, requires_grad=True)
        b = torch._C._nn._test_warn_in_autograd(a)

        with self.assertWarnsRegex(UserWarning, "Warn from backward"):
            b.backward()

    def test_complex_scalar_backward(self, device):
        a = torch.zeros(1, device=device, requires_grad=True)
        b = a * 0.5j

        msg = "grad can be implicitly created only for real scalar outputs"
        with self.assertRaisesRegex(RuntimeError, msg):
            b.backward()

        with self.assertRaisesRegex(RuntimeError, msg):
            torch.autograd.grad(b, a)

    def test_pow_real_negative_base_complex_exponent(self, device):
        # OpInfo doesn't naturally support input of mixed types, hence this test here.
        base = -torch.ones(2, device=device, dtype=torch.double)
        exponent = torch.randn(
            2, device=device, dtype=torch.cdouble, requires_grad=True
        )

        def fn(exponent):
            return torch.pow(base, exponent)

        torch.autograd.gradcheck(fn, (exponent,))

        def fn(exponent):
            return torch.pow(-1, exponent)

        torch.autograd.gradcheck(fn, (exponent,))

    def test_resize_version_bump(self, device):
        x = torch.rand((1,), device=device)
        y = torch.randn((3,), device=device)
        x.resize_((1, 2))
        self.assertEqual(x._version, 1)
        x.resize_as_(y)
        self.assertEqual(x._version, 2)

        # In the following cases, `resize` is no-op,
        # so no version bumps.
        x.resize_((3,))
        self.assertEqual(x._version, 2)

        x.resize_as_(y)
        self.assertEqual(x._version, 2)

    @unittest.skipIf(not torch.accelerator.is_available(), "requires accelerator")
    def test_zero_dim_param_mixed_device_grad(self, device):
        # cpu 0-dim params with an accelerator device grad
        # https://github.com/pytorch/pytorch/issues/160084
        class RegressionModel(torch.nn.Module):
            def __init__(self, a=0, b=0):
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor(a).float())
                self.b = torch.nn.Parameter(torch.tensor(b).float())

            def forward(self, x):
                return x * self.a + self.b

        # Keep the model on cpu as we do want to test the mixed cpu/accelerator behavior here
        model = RegressionModel()
        inputs = torch.randn(4, 10, device=device)
        out = model(inputs)
        out.sum().backward()
        self.assertIsNotNone(model.a.grad)
        self.assertIsNotNone(model.b.grad)
        self.assertEqual(model.a.grad.device, torch.device("cpu"))
        self.assertEqual(model.b.grad.device, torch.device("cpu"))


class TestAllowMutationOnSaved(TestCase):
    def assertClonedLenEqual(self, ctx, n):
        self.assertEqual(len(list(ctx.cloned.items())), n)

    def assertTIDMapLenEqual(self, ctx, n):
        self.assertEqual(len(list(ctx.tid_to_weakhandle.items())), n)

    def test_basic(self):
        a = torch.rand(2, 3, requires_grad=True)

        def fn(a):
            b = a.clone()
            out = (b**2).sum()
            b.sin_()
            out.sum().backward()
            return a.grad

        msg = (
            "variables needed for gradient computation has been modified by an inplace"
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(a)

        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            da = fn(a)

        self.assertTrue(torch.allclose(a * 2, da))
        self.assertClonedLenEqual(ctx, 0)

    def test_views(self):
        a = torch.rand(2, 3, requires_grad=True)

        def fn(a):
            b = a.clone()
            c = b.view_as(b)
            out = (b**2).sum()  # How does this work?
            c.sin_()
            out.sum().backward()
            return a.grad

        msg = (
            "variables needed for gradient computation has been modified by an inplace"
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(a)

        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            da = fn(a)

        self.assertClonedLenEqual(ctx, 0)
        self.assertTrue(torch.allclose(a * 2, da))

    def test_save_base_and_modify_view(self):
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            b = a.clone()
            c = b[:1]
            out = b**2
            # modify the view
            c *= 10
            # self.assertClonedLenEqual(ctx, 1)
            out.sum().backward()
            self.assertClonedLenEqual(ctx, 0)

        self.assertClonedLenEqual(ctx, 0)
        self.assertTrue(torch.allclose(a * 2, a.grad))

    def test_save_view_modify_base(self):
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            b = a.clone()
            c = b[:]
            out = (c**2).sum()
            b *= 2
            out.backward()
            self.assertTrue(torch.allclose(a * 2, a.grad))

    def test_double_backward(self):
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            b = a.clone()
            out = (b**2).sum()
            b.sin_()
            torch.autograd.grad(out, a, create_graph=True)
            (da,) = torch.autograd.grad(out, a, create_graph=True)
            (d2a,) = torch.autograd.grad(da.sum(), a)

        self.assertTrue(torch.allclose(torch.ones_like(a) * 2, d2a))
        self.assertClonedLenEqual(ctx, 0)

    def test_saved_but_not_anymore(self):
        # Make sure we don't clone if the tensor was once saved, but
        # by the time we do in-place, it is no longer saved
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.randn(2, 3, requires_grad=True).clone()
            out = (a**2).sum()
            self.assertTIDMapLenEqual(ctx, 1)
            self.assertClonedLenEqual(ctx, 0)
            out.backward()
            a.sin_()
            self.assertClonedLenEqual(ctx, 0)
            out = (a**2).sum()
            a.sin_()
            self.assertClonedLenEqual(ctx, 1)
            del out
            self.assertClonedLenEqual(ctx, 0)

    def test_saved_same_tensor_many_times(self):
        # We should only clone once
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.randn(2, 3, requires_grad=True).clone()
            b = a**2
            c = a**2
            a.sin_()
            self.assertClonedLenEqual(ctx, 1)
            del b, c
            self.assertClonedLenEqual(ctx, 0)

    def test_saved_same_tensor_different_versions(self):
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.randn(2, 3, requires_grad=True).clone()
            b = a**2
            a.sin_()
            c = a**2
            a.sin_()
            self.assertClonedLenEqual(ctx, 2)
            del b
            self.assertClonedLenEqual(ctx, 1)
            del c
            self.assertClonedLenEqual(ctx, 0)

    def test_with_math_views(self):
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.tensor([1 + 1j], requires_grad=True).clone()
            b = a.conj()
            out = (b**2).sum()
            a.sin_()
            out.abs().backward()

            a = torch.tensor([1 + 1j], requires_grad=True).clone()
            b = a.conj()
            out = (b**2).sum()
            # in this case, it is no longer a view it seems
            b.sin_()
            out.abs().backward()

    def test_with_out_variant(self):
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.tensor([1.0], requires_grad=True)
            b = torch.tensor([1.0])
            c = torch.tensor([2.0])
            out = a * b
            self.assertTIDMapLenEqual(ctx, 1)
            torch.sin(c, out=b)
            self.assertClonedLenEqual(ctx, 1)
            out.backward()
            self.assertClonedLenEqual(ctx, 0)

    def test_backward_out_of_context(self):
        # Out of context
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            out = (a**2).sum()

        msg = "Trying to backward outside of the 'allow_mutation_on_saved_tensors' context"
        with self.assertRaisesRegex(AssertionError, msg):
            out.backward()

        # Different context
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            out = (a**2).sum()

        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            with self.assertRaisesRegex(AssertionError, msg):
                out.backward()

    def test_disallow_nesting(self):
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            msg = "allow_mutation_on_saved_tensors contexts cannot be nested"
            with self.assertRaisesRegex(RuntimeError, msg):
                with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
                    pass

    def test_inplace_foreach(self):
        with torch.autograd.graph.allow_mutation_on_saved_tensors():
            a = [
                torch.tensor(1.0, requires_grad=True),
                torch.tensor(1.0, requires_grad=True),
            ]
            b = torch._foreach_exp(a)
            torch._foreach_add_(b, 1)
            (b[0] + b[1]).backward()

        self.assertEqual([a[0].grad, a[1].grad], torch._foreach_exp(a))


class TestAutogradInferenceMode(TestCase):
    def _is_inference_tensor(self, tensor):
        try:
            err_msg = "Inference tensors do not track version counter"
            with self.assertRaisesRegex(RuntimeError, err_msg):
                tensor._version
            return True
        except AssertionError as e:
            return False

    def test_inference_mode_context_manager(self):
        self.assertFalse(torch.is_inference_mode_enabled())
        with torch.inference_mode():
            self.assertTrue(torch.is_inference_mode_enabled())
            with torch.inference_mode(False):
                self.assertFalse(torch.is_inference_mode_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())

    def test_inference_mode_decorator(self):
        def func(x):
            self.assertEqual(torch.is_inference_mode_enabled(), mode)
            return x * x

        for mode, use_kwarg in product((True, False, None), (True, False)):
            if mode is None:
                if use_kwarg:
                    decorated = torch.inference_mode(mode=func)
                else:
                    decorated = torch.inference_mode(func)
                mode = True
            else:
                if use_kwarg:
                    decorated = torch.inference_mode(mode=mode)(func)
                else:
                    decorated = torch.inference_mode(mode)(func)

            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)
                d = decorated(c)
                self.assertTrue(not mode or torch.is_inference(d))
                self.assertEqual(d.requires_grad, requires_grad and not mode)

    def test_inference_mode_tensor_creation(self):
        with torch.inference_mode():
            # new tensors created through constructors are inference tensors
            c = torch.ones(1, 2, 3)
            self.assertFalse(c.requires_grad)
            self.assertTrue(torch.is_inference(c))

            # requires_grad doesn't change inference tensor behavior in InferenceMode
            tmp = torch.ones(1, 2, 3, requires_grad=True)
            self.assertTrue(tmp.requires_grad)
            self.assertTrue(torch.is_inference(tmp))

            tmp = torch.ones(1, 2, 3).requires_grad_(False)
            self.assertFalse(tmp.requires_grad)
            self.assertTrue(torch.is_inference(tmp))

    def test_inference_mode_existing_autograd_session(self):
        s = torch.ones(1, 2, 3, requires_grad=True)
        a = s.clone()

        # `a` gets saved outside of inference mode
        out = a * a
        with torch.inference_mode():
            a.add_(2)

        self.assertFalse(torch.is_inference(a))
        # tensors created outside of inference mode aren't
        # inference tensors, so they will still have their
        # version counters tracked
        err_msg = (
            "one of the variables needed for gradient computation has been "
            "modified by an inplace operation"
        )
        with self.assertRaisesRegex(RuntimeError, err_msg):
            out.backward(torch.ones_like(out))

    def test_inference_mode_inf_tensor_in_inf_mode_functional_op(self):
        def functional_op(x):
            return x * x

        with torch.inference_mode():
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # performing a non-view operation produces a inference tensor
                # that does not require grad
                func_out = functional_op(c)
                self.assertTrue(torch.is_inference(func_out))
                self.assertFalse(func_out.requires_grad)

    def test_inference_mode_inf_tensor_in_inf_mode_inplace_op(self):
        @torch.inference_mode()
        def run_test(fn):
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # after performing inplace operation, tensor is still
                # an inference tensor
                fn(c)
                self.assertTrue(torch.is_inference(c))
                self.assertEqual(c.requires_grad, requires_grad)

        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

        # inplace ops with manual kernel for ADInplaceOrView key in VariableTypeManual.cpp
        run_test(lambda x: x.resize_(1, 2))
        run_test(lambda x: x.resize_as_(torch.ones(1, 2)))
        run_test(lambda x: x.copy_(torch.ones(1, 2, 3)))

    def test_inference_mode_inf_tensor_in_inf_mode_view_op(self):
        with torch.inference_mode():
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # perform view operation produces inference tensor
                # that does not require grad
                view_out = c.view(-1)
                self.assertTrue(torch.is_inference(view_out))
                self.assertFalse(view_out.requires_grad)

    def test_inference_mode_inf_tensor_in_normal_mode_functional_op(self):
        def functional_op(x):
            return x * x

        for requires_grad in (True, False):
            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

        func_out = functional_op(c)
        self.assertFalse(torch.is_inference(func_out))
        self.assertFalse(func_out.requires_grad)
        self.assertTrue(func_out.is_leaf)

    @skipIfTorchDynamo(
        "exception from ill-formed graph module is not propagated with eager_noexcept"
    )
    def test_inference_mode_inf_tensor_in_normal_mode_inplace_op(self):
        def run_test(fn):
            for requires_grad in (False, True):
                with torch.inference_mode():
                    c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                if requires_grad:
                    # leaf variable that requires grad is being used in an inplace
                    # operation when requires_grad=True
                    pass
                else:
                    err_msg = "Inplace update to inference tensor outside InferenceMode"
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        fn(c)

        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    def test_inference_mode_inf_tensor_in_normal_mode_view_op(self):
        for requires_grad in (True, False):
            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

            out = c.view(-1)
            self.assertTrue(torch.is_inference(out))
            self.assertFalse(out.requires_grad)
            self.assertFalse(out._is_view())
            self.assertTrue(out.is_leaf)

    def test_normal_tensor_inplace_output_in_inference_mode(self):
        def run_test(fn):
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()

                with torch.inference_mode():
                    fn(a)
                    self.assertFalse(torch.is_inference(a))
                    self.assertEqual(a.requires_grad, requires_grad)

                    # inplace -> inplace
                    fn(a)
                    self.assertFalse(torch.is_inference(a))
                    self.assertEqual(a.requires_grad, requires_grad)

                    # inplace -> inplace -> view
                    view_out = a.view(-1)
                    self.assertFalse(torch.is_inference(view_out))
                    self.assertEqual(view_out.requires_grad, requires_grad)

        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    def test_normal_tensor_inplace_output_in_normal_mode(self):
        def run_test(fn):
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()

                with torch.inference_mode():
                    fn(a)
                    self.assertFalse(torch.is_inference(a))
                    self.assertEqual(a.requires_grad, requires_grad)

                fn(a)
                self.assertFalse(torch.is_inference(a))
                self.assertEqual(a.requires_grad, requires_grad)

                # inplace -> inplace
                fn(a)
                self.assertFalse(torch.is_inference(a))
                self.assertEqual(a.requires_grad, requires_grad)

                # inplace -> inplace -> view
                view_out = a.view(-1)
                self.assertFalse(torch.is_inference(view_out))
                self.assertEqual(view_out.requires_grad, requires_grad)
            run_test(lambda x: x.add_(2))
            run_test(lambda x: x.transpose_(0, 1))

    def test_normal_tensor_view_output_in_inference_mode(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                out = a.view(-1)
                self.assertFalse(torch.is_inference(out))
                self.assertEqual(out.requires_grad, requires_grad)
                self.assertTrue(out._is_view())

                # view -> view
                tmp = out.view(-1)
                self.assertFalse(torch.is_inference(tmp))
                self.assertEqual(tmp.requires_grad, requires_grad)
                self.assertTrue(tmp._is_view())
                self.assertTrue(tmp.is_leaf)

                # view -> view -> inplace
                self.assertTrue(torch.is_inference_mode_enabled())
                tmp.add_(2)
                self.assertFalse(torch.is_inference(tmp))
                self.assertEqual(tmp.requires_grad, requires_grad)
                # Accessing is_leaf in python tries to update grad_fn and raises:
                # A view was created in inference mode and its base or
                # another view of its base has been modified inplace in normal mode
                # tmp.is_leaf
                self.assertEqual(a._version, tmp._version)

    def test_normal_tensor_view_output_in_normal_mode(self):
        def functional_op(x):
            return x * x

        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                out = a.view(-1)
                self.assertFalse(torch.is_inference(out))
                self.assertEqual(out.requires_grad, requires_grad)
                self.assertTrue(out._is_view())
                self.assertTrue(out.is_leaf)

            tmp = functional_op(out)
            self.assertFalse(torch.is_inference(tmp))
            self.assertEqual(tmp.requires_grad, requires_grad)

            if requires_grad:
                err_msg = (
                    "A view was created in inference mode and is being modified inplace"
                )
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    out.add_(2)
            else:
                out.add_(2)

            tmp = out.view(2, 3)
            self.assertFalse(torch.is_inference(tmp))
            self.assertEqual(tmp.requires_grad, requires_grad)

    def test_mix_inference_and_normal_tensor_functional_op(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)

            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

            # add is safe since it doesn't save any variable for backward
            out = c.add(s)
            self.assertFalse(torch.is_inference(out))
            self.assertEqual(out.requires_grad, requires_grad)
            if requires_grad:
                # leaf inference tensor with requires_grad=True can still have gradient
                out.backward(torch.ones_like(out))
                self.assertEqual(c.grad, torch.ones_like(c))

            if requires_grad:
                err_msg = "Inference tensors cannot be saved for backward"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    c * s

                # TODO: Test this with an autograd.Function when it works
                #       stack stopped capturing a TensorList input
                # # inference tensor in TensorList input
                # inputs = [s, c]
                # with self.assertRaisesRegex(RuntimeError, err_msg):
                #     torch.stack(inputs)

    def test_mix_inference_and_normal_tensor_inplace_op(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                c = torch.ones(1, 2, 3)

            self.assertTrue(torch.is_inference(c))
            if requires_grad:
                err_msg = "Inference tensors cannot be saved for backward"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    a.mul_(c)

                # inference tensor in TensorList input
                err_msg = (
                    "out=... arguments don't support automatic differentiation, "
                    "but one of the arguments requires grad"
                )
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.mul(s, s, out=c)
            else:
                a.mul_(c)
                err_msg = "Inplace update to inference tensor outside InferenceMode is not allowed"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.mul(s, s, out=c)

    def test_mix_inference_and_normal_tensor_view_op(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)

            with torch.inference_mode():
                c = torch.ones(1, 2, 3)

            # view_as is a composite op which calls view with only one
            # tensor argument. So there isn't a mixed inference and normal
            # tensor inputs for view ops
            tmp1 = c.view_as(s)
            self.assertTrue(torch.is_inference(tmp1))
            self.assertFalse(tmp1.requires_grad)

            # this is fine since its equivalent as s.view(c.sizes()) which
            # isn't a mixed input scenario
            tmp2 = s.view_as(c)
            self.assertFalse(torch.is_inference(tmp2))
            self.assertEqual(tmp2.requires_grad, requires_grad)

    def test_inference_mode_handle_direct_view_on_rebase(self):
        def run_test(fn):
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()

                with torch.inference_mode():
                    view_out = a.view_as(a)

                if requires_grad:
                    err_msg = "A view was created in inference mode and is being modified inplace"
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        fn(view_out)
                else:
                    fn(view_out)

        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    def test_inference_mode_handle_indirect_view_on_rebase(self):
        def run_test(fn):
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()

                with torch.inference_mode():
                    view_out = a.view(-1)

                fn(a)
                if requires_grad:
                    err_msg = "A view was created in inference mode and its base or another view "
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        view_out.grad_fn
                else:
                    view_out.grad_fn

        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))


NUM_GPU_CYCLES_IN_ONE_SEC = 2_000_000_000


@contextlib.contextmanager
def _set_device_index(target_device):
    orig_device = torch.accelerator.current_device_index()
    try:
        torch.accelerator.set_device_index(target_device)
        yield
    finally:
        torch.accelerator.set_device_index(orig_device)


def _sleep_if_cuda(cycles):
    if "cuda" == torch.accelerator.current_accelerator().type:
        return torch.cuda._sleep(cycles)
    else:
        # Update this if non-cuda accelerators support something like sleep
        return


def _get_device_name(idx):
    return f"{torch.accelerator.current_accelerator().type}:{idx}"


# Although this is written to be generic over all accelerators, non-cuda accelerators
# are not fully tested since sleep is only supported on cuda.
class TestAutogradStreamSynchronization(TestCase):
    def get_default_streams(self, num_devices=1):
        out = []
        for i in range(num_devices):
            with _set_device_index(i):
                acc = torch.accelerator.current_accelerator()
                out.append(torch.get_device_module(acc).default_stream())
        return tuple(out)

    def synchronize_all_devices(self, num_devices=1):
        for i in range(num_devices):
            torch.accelerator.synchronize(i)

    def assert_all_streams_default(self, num_devices=1):
        # Sanity check
        default_streams = self.get_default_streams(num_devices)
        for i in range(num_devices):
            with _set_device_index(i):
                acc = torch.accelerator.current_accelerator()
                # Do this instead of using torch.accelerator.current_stream(i)
                # Otherwise, e.g. in the case of cuda, we'd be trying to compare
                # torch.cuda.Stream with torch.Stream
                self.assertEqual(
                    torch.get_device_module(acc).current_stream(), default_streams[i]
                )

    # AttributeError: module 'torch.mps' has no attribute 'default_stream'
    @expectedFailureMPS
    @skipCUDANonDefaultStreamIf(True)
    def test_consumer_to_single_producer_case_2_correctness(self, device):
        if device == "cpu":
            self.skipTest("requires accelerator")

        #                          Device    Stream
        # Consumer (MulBackward):  cuda:0    s0
        # Producer              :  cuda:0    s1
        # Gradient              :  cuda:0    s1
        class Producer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                out = gO.clone()
                _sleep_if_cuda(NUM_GPU_CYCLES_IN_ONE_SEC // 2)
                out.add_(1)
                return out

        def test():
            self.synchronize_all_devices()
            self.assert_all_streams_default()

            with torch.Stream(0) as s0:
                a = torch.ones(256, 256, requires_grad=True, device=_get_device_name(0))
                b = a * 2

            with torch.Stream(0) as s1:
                s1.wait_stream(s0)
                out = Producer.apply(b)

                with torch.autograd.grad_mode.set_multithreading_enabled(False):
                    out.sum().backward()

            self.synchronize_all_devices()

            # Expected result: a.grad = (grad_out + 1) * 2 = 4
            self.assertEqual(a.grad, torch.full_like(a, 4))

        # Run an extra time to warm up
        for _ in range(2):
            test()

    def _test_consumer_to_single_producer_case_3_correctness(
        self, non_default_ambient_stream
    ):
        #                          Device    Stream
        # Consumer (MulBackward):  cuda:0    s0
        # Producer              :  cuda:1    cuda:1 default
        # Gradient              :  cuda:0    cuda:0 default
        class Producer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # The node's canonical stream is the current stream
                # of the device of the first output.
                ctx.node_stream = torch.accelerator.current_stream(1)
                return x.to(_get_device_name(1))

            @staticmethod
            def backward(ctx, gO):
                out = gO.to(_get_device_name(0))
                with _set_device_index(0):
                    _sleep_if_cuda(NUM_GPU_CYCLES_IN_ONE_SEC // 2)
                # It's the node's responsibility to sync back to its canonical stream.
                out.add_(1)
                ctx.node_stream.wait_stream(torch.accelerator.current_stream(0))
                return out

        def test():
            self.synchronize_all_devices(2)
            self.assert_all_streams_default(2)

            (default_stream_0,) = self.get_default_streams()

            # Ensure consumer node happens on non-default stream so that
            # when FuncBackward produces a gradient on a default stream
            # a sync is necessary.
            with torch.Stream(0) as s0:
                a = torch.ones(256, 256, requires_grad=True, device="cuda")
                b = a * 2

            default_stream_0.wait_stream(s0)
            out = Producer.apply(b)

            def call_backward(x):
                with torch.autograd.grad_mode.set_multithreading_enabled(False):
                    x.sum().backward()

            if non_default_ambient_stream:
                with torch.Stream(0) as s1:
                    s1.wait_stream(default_stream_0)
                    call_backward(out)
            else:
                call_backward(out)

            self.synchronize_all_devices(2)

            # Expected result: a.grad = (grad_out + 1) * 2 = 4
            self.assertEqual(a.grad, torch.full_like(a, 4))

        # Run an extra time to warm up
        for _ in range(2):
            test()

    # AttributeError: module 'torch.mps' has no attribute 'default_stream'
    @expectedFailureMPS
    @skipCUDANonDefaultStreamIf(True)
    @unittest.skipIf(
        torch.accelerator.device_count() < 2, "accelerator count is less than 2"
    )
    def test_consumer_to_single_producer_case_3_correctness_non_default_ambient_stream(
        self, device
    ):
        if device == "cpu":
            self.skipTest("requires accelerator")
        self._test_consumer_to_single_producer_case_3_correctness(
            non_default_ambient_stream=True
        )

    # AttributeError: module 'torch.mps' has no attribute 'default_stream'
    @expectedFailureMPS
    @skipCUDANonDefaultStreamIf(True)
    @unittest.skipIf(
        torch.accelerator.device_count() < 2, "accelerator count is less than 2"
    )
    def test_consumer_to_single_producer_case_3_correctness(self, device):
        if device == "cpu":
            self.skipTest("requires accelerator")
        self._test_consumer_to_single_producer_case_3_correctness(
            non_default_ambient_stream=False
        )

    # AttributeError: module 'torch.mps' has no attribute 'default_stream'
    @expectedFailureMPS
    @skipCUDANonDefaultStreamIf(True)
    @unittest.skipIf(
        torch.accelerator.device_count() < 2, "accelerator count is less than 2"
    )
    def test_consumer_to_single_producer_case_4_correctness(self, device):
        if device == "cpu":
            self.skipTest("requires accelerator")

        #           Device    Stream
        # Consumer: cuda:0    cuda:0 default
        # Producer: cuda:1    s1
        # Gradient: cuda:1    s1
        class Producer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                out = gO.clone()
                _sleep_if_cuda(NUM_GPU_CYCLES_IN_ONE_SEC // 2)
                return out.add_(1)

        class Consumer(torch.autograd.Function):
            # In the multi-output case, the node's canonical device and stream correspond to
            # that of its first output. This is required to induce cases 4/5.
            @staticmethod
            def forward(ctx, x):
                return x.clone(), x.to(_get_device_name(1))

            @staticmethod
            def backward(ctx, gO_0, gO_1):
                # gO_1 is on s1, but we're currently doing compute in cuda:1 default
                # It's the user's responsibility to sync to consumer (.to() should do this
                # already.)
                # Things would work out if the engine sync'd s1 with consumer.
                # Ignore grad wrt first arg because we don't use it.
                return gO_1.to(_get_device_name(0))

        def test():
            self.synchronize_all_devices(2)
            self.assert_all_streams_default(2)

            _, default_stream_1 = self.get_default_streams(2)
            a = torch.ones(256, 256, requires_grad=True, device=_get_device_name(0))
            _unused, b = Consumer.apply(a)

            with torch.Stream(1) as s1:
                s1.wait_stream(default_stream_1)
                out = Producer.apply(b)

                with torch.autograd.grad_mode.set_multithreading_enabled(False):
                    out.sum().backward()

            self.synchronize_all_devices(2)

            # Expected result: a.grad = grad_out + 1 = 2
            self.assertEqual(a.grad, torch.full_like(a, 2))

        # Run an extra time to warm up
        for _ in range(2):
            test()

    # AttributeError: module 'torch.mps' has no attribute 'default_stream'
    @expectedFailureMPS
    @skipCUDANonDefaultStreamIf(True)
    @unittest.skipIf(
        torch.accelerator.device_count() < 2, "accelerator count is less than 2"
    )
    def test_consumer_to_multi_producer_case_4_correctness(self, device):
        if device == "cpu":
            self.skipTest("requires accelerator")

        #             Device    Stream
        # Consumer  : cuda:0    cuda:0 default
        #
        # Producer 1: cuda:1    s1
        # Gradient 1: cuda:1    s1
        #
        # Producer 2: cuda:1    s2
        # Gradient 2: cuda:1    s2
        #
        # Accumulation stream: s2 since it is scheduled first
        class ProducerFast(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                out = gO.clone()
                return out * 2

        class ProducerSlow(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                out = gO.clone()
                _sleep_if_cuda(NUM_GPU_CYCLES_IN_ONE_SEC // 2)
                return out.mul_(2)

        class Consumer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.node_stream = torch.accelerator.current_stream(x.device)
                return x.clone(), x.to(_get_device_name(1))

            @staticmethod
            def backward(ctx, gO_0, gO_1):
                torch.accelerator.current_stream(gO_1.device).wait_stream(
                    ctx.node_stream
                )
                return (gO_1 * 2).to(_get_device_name(0))

        def test():
            self.synchronize_all_devices(2)
            self.assert_all_streams_default(2)

            default_stream_0, default_stream_1 = self.get_default_streams(2)

            a = torch.ones(256, 256, requires_grad=True, device=_get_device_name(0))
            _unused, b = Consumer.apply(a)

            with torch.Stream(1) as s1:
                s1.wait_stream(default_stream_1)
                out1 = ProducerFast.apply(b)

            with torch.Stream(1) as s2:
                s2.wait_stream(default_stream_1)
                out2 = ProducerSlow.apply(b)

            default_stream_1.wait_stream(s1)
            default_stream_1.wait_stream(s2)

            with torch.autograd.grad_mode.set_multithreading_enabled(False):
                (out1 + out2).sum().backward()

            self.synchronize_all_devices(2)

            # If the accumulation stream does not wait for the slow producer stream
            # the in-place mul-by-2 is performed on the accumulated buffer AFTER
            # ProducerFast has already accumulated!
            #
            # Correct: (1.mul_(2) + 2) * 2 = 8
            # Incorrect: (1 + 2).mul_(2) * 2 = 12
            self.assertEqual(a.grad, torch.full_like(a, 8))

        # Run an extra time to warm up
        for _ in range(2):
            test()

    # This test may spuriously fail on non-cuda accelerators (since we won't
    # be calling sleep)
    @onlyCUDA
    @skipCUDANonDefaultStreamIf(True)
    def test_side_stream_backward_overlap(self, device):
        # In case 2/3, we would designate the consumer as the accumulation
        # stream and naively, one might have the consumer wait for the producer
        # as soon as we've added to the InputBuffer the first time.
        #
        # However, in the case where the stream of the consumer also happens to
        # be the stream of the producer, this is suboptimal because it would
        # prevent the computation of the two producers from being overlapped.
        # what you really want to do is to have that sync between the producer
        # and consumer to be delayed until right before the accumulation.
        # Note that this doesn't address N=3, but the side-stream N=2 case is
        # the common case.
        events = {
            "main_backward_start": None,
            "side_backward_start": None,
            "side_backward_end": None,
        }

        class Main(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, gO):
                # Record when main backward starts
                evt = torch.Event(enable_timing=True)
                evt.record()
                events["main_backward_start"] = evt
                return gO

        class Side(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, gO):
                evt = torch.Event(enable_timing=True)
                evt.record()
                events["side_backward_start"] = evt

                _sleep_if_cuda(NUM_GPU_CYCLES_IN_ONE_SEC // 2)
                result = gO.clone()

                evt = torch.Event(enable_timing=True)
                evt.record()
                events["side_backward_end"] = evt
                return result

        def populate_events():
            self.synchronize_all_devices()
            self.assert_all_streams_default()

            (default_stream_0,) = self.get_default_streams()

            a = torch.ones(256, 256, requires_grad=True, device=_get_device_name(0))
            b = a.clone()  # not a leaf, does it matter?

            evt = torch.Event()
            evt.record()

            # Overlap during forward
            c_main = Main.apply(b)

            with torch.Stream(0) as s0:
                s0.wait_event(evt)
                c_side = Side.apply(b)

            default_stream_0.wait_stream(s0)

            with torch.autograd.grad_mode.set_multithreading_enabled(False):
                (c_main + c_side).sum().backward()

            self.synchronize_all_devices()

        def check_ordering():
            # Sanity check: side backward's end happens after start
            self.assertTrue(
                events["side_backward_start"].elapsed_time(events["side_backward_end"])
                > 0
            )
            # Overlap check: side's backward starts before side backward ends
            self.assertTrue(
                events["main_backward_start"].elapsed_time(events["side_backward_end"])
                > 0
            )

        # Warmup
        for _ in range(2):
            populate_events()

        # Reset events (not really necessary but OK)
        events["side_backward_start"] = None
        events["side_backward_end"] = None
        events["main_backward_start"] = None

        # Test
        populate_events()
        check_ordering()

    @expectedFailureMPS
    def test_warn_on_accumulate_grad_stream_mismatch_flag(self, device):
        if device == "cpu":
            self.skipTest("requires accelerator")

        def do_test(suppress_warn, keep_grad_acc):
            def _test():
                with set_warn_always_context(True):
                    with warnings.catch_warnings(record=True) as warns:
                        warnings.simplefilter("always")

                        with torch.Stream(0) as s0:
                            a = torch.ones(8, 8, device=device, requires_grad=True)
                            if keep_grad_acc:
                                # create grad_acc under s1 and keep alive with b
                                b = a.clone()

                        with torch.Stream(0) as s1:
                            s1.wait_stream(s0)
                            c = a.sum()

                        c.backward()

                    filter_str = "set_warn_on_accumulate_grad_stream_mismatch"
                    return sum([filter_str in str(w.message) for w in warns]) > 0

            if suppress_warn:
                try:
                    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(
                        False
                    )
                    actual_warn = _test()
                finally:
                    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(
                        True
                    )
            else:
                actual_warn = _test()

            expect_warn = not suppress_warn and keep_grad_acc
            self.assertEqual(actual_warn, expect_warn)

        # Warn by default
        self.assertTrue(torch._C._warn_on_accumulate_grad_stream_mismatch())

        for suppress_warn in (True, False):
            for keep_grad_acc in (True, False):
                do_test(suppress_warn=suppress_warn, keep_grad_acc=keep_grad_acc)


class TestMultithreadAutograd(TestCase):
    def _run_py_multithread_fn(
        self, fn, args=(), num_threads=10, kwargs=None, pass_idx=False
    ):
        class PropagatingThread(threading.Thread):
            """Helper class to propagate exception from child
            thread to main thread on join.

            Reference: https://stackoverflow.com/a/31614591/5602957
            """

            def run(self):
                self.exception = None
                try:
                    self.ret = super().run()
                except Exception as e:
                    self.exception = e

            def join(self, timeout=None):
                super().join(timeout)
                if self.exception:
                    raise self.exception from self.exception
                return self.ret

        threads = []
        for idx in range(num_threads):
            p = PropagatingThread(target=fn, args=((idx, *args) if pass_idx else args))
            p.start()
            threads.append(p)

        for p in threads:
            p.join()

    def test_multithreaded_exception_propagation(self):
        # Test whether exception in child thread
        # are propagated to main thread.
        def fn():
            self.assertTrue(False)

        with self.assertRaises(AssertionError):
            self._run_py_multithread_fn(fn)

    def test_simple_backward(self):
        # simple multithreaded backward that create threads in the beginning of training
        # and everything else is training separately, i.e. inputs, operations, etc.
        def train_fn():
            x = torch.ones(5, 5, requires_grad=True)
            y = (x + 3) * (x + 4) * 0.5
            y.sum().backward()
            self.assertEqual(x.grad, x + 3.5)

        self._run_py_multithread_fn(train_fn)

    def test_simple_backward_same_input(self):
        # simple multithreaded backward with only shared inputs (i.e. This is common
        # for things like Hogwild multithreaded training with multiple CPU threads)
        def train_fn_backward(x):
            y = (x + 3) * (x + 4) * 0.5
            y.sum().backward()

        x = torch.ones(5, 5, requires_grad=True)
        self._run_py_multithread_fn(train_fn_backward, (x,))
        # Since we are calling backward from multiple threads
        # and all threads share the same input, when we do backward
        # concurrently, different backwards will all accumulate to
        # the same .grad for each input, and the gradients should
        # be equal to num_threads * gradient
        self.assertEqual(x.grad, 10 * (x + 3.5))

        def train_fn_grad(x):
            y = (x + 3) * (x + 4) * 0.5
            grads = torch.autograd.grad(y.sum(), x)
            self.assertEqual(len(grads), 1)
            self.assertEqual(grads[0], x + 3.5)

        # since we use functional grad() api, gradients will not
        # be accumulate to the same place and should be the same
        self._run_py_multithread_fn(train_fn_grad, (x,))

    def test_multi_grad_all_hooks(self):
        # Multihooks should behave independently per execution of backward
        # Test that the hook fired the number of times we ran backward
        # even if those executions occur concurrently on different threads
        t1 = torch.rand(2, requires_grad=True)
        t2 = torch.rand(2, requires_grad=True)
        t3 = torch.rand(2, requires_grad=True)
        t4 = torch.rand(2, requires_grad=True)

        res = None
        count = [0]
        hook_lock = threading.Lock()

        def hook(grads):
            nonlocal res
            with hook_lock:
                count[0] += 1
                grad_is_none = [g is not None for g in grads]
                if res is None:
                    res = grad_is_none
                else:
                    self.assertEqual(res, grad_is_none)

        handle = torch.autograd.graph.register_multi_grad_hook((t1, t2, t3, t4), hook)

        out = (t2 * t3).sum()

        def backward_retain_graph(out, t2, t3):
            out.backward(inputs=(t2, t3), retain_graph=True)

        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)

        self.assertEqual(count[0], 5)
        self.assertEqual(res, [False, True, True, False])

        # Leave one hook partially applied
        res = None
        count = [0]
        err_count = [0]
        bw_count = [0]
        bw_count_lock = threading.Lock()
        err_count_lock = threading.Lock()

        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, gO):
                with bw_count_lock:
                    bw_count[0] += 1
                    if bw_count[0] == 1:
                        raise RuntimeError("error message")
                    else:
                        return gO

        out = (Func.apply(t2) * t3).sum()

        def backward_retain_graph(out, t2, t3):
            try:
                out.backward(inputs=(t2, t3), retain_graph=True)
            except RuntimeError:
                with err_count_lock:
                    err_count[0] += 1

        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)

        self.assertEqual(count[0], 4)
        self.assertEqual(err_count[0], 1)
        self.assertEqual(res, [False, True, True, False])

        handle.remove()

    def test_multi_grad_any_hooks(self):
        # Multihooks should behave independently per execution of backward
        # Test that the hook fired the number of times we ran backward
        # even if those executions occur concurrently on different threads
        t1 = torch.rand(2, requires_grad=True)
        t2 = torch.rand(2, requires_grad=True)
        t3 = torch.rand(2, requires_grad=True)
        t4 = torch.rand(2, requires_grad=True)

        res = None
        count = [0]
        hook_lock = threading.Lock()

        def hook(grad):
            nonlocal res
            with hook_lock:
                count[0] += 1
                if res is None:
                    res = "foo"
                else:
                    self.assertEqual(res, "foo")

        torch.autograd.graph.register_multi_grad_hook(
            (t1, t2, t3, t4), hook, mode="any"
        )

        out = (t2 * t3).sum()

        def backward_retain_graph(out, t2, t3):
            out.backward(inputs=(t2, t3), retain_graph=True)

        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)
        self.assertEqual(count[0], 5)
        self.assertEqual(res, "foo")

        # Raise an error in one thread's backward
        res = None
        count = [0]
        err_count = [0]
        bw_count = [0]
        bw_count_lock = threading.Lock()
        err_count_lock = threading.Lock()

        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, gO):
                with bw_count_lock:
                    bw_count[0] += 1
                    if bw_count[0] == 1:
                        raise RuntimeError("error message")
                    else:
                        return gO

        out = (Func.apply(t2) * t3).sum()

        def backward_retain_graph(out, t2, t3):
            try:
                out.backward(inputs=(t2, t3), retain_graph=True)
            except RuntimeError:
                with err_count_lock:
                    err_count[0] += 1

        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)

        # Expect all 5 threads to increment count since the hook runs before
        # the custom backward
        self.assertEqual(count[0], 5)
        self.assertEqual(err_count[0], 1)
        self.assertEqual(res, "foo")

    def test_dataparallel_saved_tensors_hooks(self):
        def pack(x):
            warnings.warn("pack")
            return x

        _self = self

        class Model(torch.nn.Module):
            def forward(self, x):
                with warnings.catch_warnings(record=True) as w:
                    y = x * x
                    if torch.cuda.device_count() >= 2:
                        # DataParallel is calling the forward in different threads
                        # without propagating TLS, so hooks should not be called here
                        _self.assertEqual(len(w), 0)
                    else:
                        # DataParallel only uses one thread
                        # so hooks should be called here
                        _self.assertGreater(len(w), 0)

        x = torch.ones(5, 5, requires_grad=True)
        model = torch.nn.DataParallel(Model())

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            model(x)
            with warnings.catch_warnings(record=True) as w:
                y = x * x
                # hooks should be called here
                _self.assertGreater(len(w), 0)

    def test_python_thread_in_middle(self):
        # User might write a network that starts on one CPU thread, then runs its second half
        # concurrently with other threads (either via python threading or fork/join calls),
        # then calls backward()/grad() on BOTH threads, like a Y pattern from input at the
        # bottom to output at the top. This way part of the GraphTask is being shared across
        # different threads and we need to ensure user specify retain_graph=True, otherwise
        # error out with the correct error message

        # Case 1: multiple backward with python threads, retain_graph=False
        # should throw error in some threads with no retain_graph.
        success_vs_raises = [0, 0]

        def train_fn_no_retain_graph(x):
            y = x + x**2
            try:
                y.sum().backward()
                success_vs_raises[0] += 1
            except RuntimeError as error:
                success_vs_raises[1] += 1
                self.assertRegex(str(error), "Specify retain_graph=True")

        x_no_retain = torch.ones(5, 5, requires_grad=True)
        y_no_retain = x_no_retain + x_no_retain**2
        self._run_py_multithread_fn(
            train_fn_no_retain_graph, (y_no_retain,), num_threads=5
        )
        # at least one thread will be success in this case, all other threads should raise
        # with the error that throw to user to recommend them specify retain_graph=True
        self.assertTrue(success_vs_raises[0] >= 1)

        # multiple backward with python threads, no error with retain_graph=True
        def train_fn_retain_graph(x):
            y = x + x**2
            y.sum().backward(retain_graph=True)

        x_retain = torch.ones(5, 5, requires_grad=True)
        y_retain = x_retain + x_retain**2
        self._run_py_multithread_fn(train_fn_retain_graph, (y_retain,), num_threads=5)
        # result should equal to num_thread * gradients
        self.assertEqual(
            x_retain.grad,
            5 * (4 * x_retain**3 + 6 * (x_retain**2) + 4 * x_retain + 1),
        )

    def test_fork_join_in_middle(self):
        # multiple backward with jit threads (fork/join primitive)
        # similar to test_python_thread_in_middle, we test with retain_graph=False/True

        # Case 1: multiple grad() calls with jit threads, retain_graph=False
        # should throw error in some threads with no retain_graph.
        @torch.jit.script
        def train_fn_jit_no_retain(middle, orig_x):
            y = middle + middle**2
            return torch.autograd.grad([y.sum()], [orig_x])

        @torch.jit.script
        def train_fn_fork_join_calls_no_retain(x):
            y_no_retain = (x + 3) * (x + 4) * 0.5

            fut = torch.jit._fork(train_fn_jit_no_retain, y_no_retain, x)
            grad_hat = train_fn_jit_no_retain(y_no_retain, x)
            grad = torch.jit._wait(fut)
            return grad, grad_hat

        try:
            train_fn_fork_join_calls_no_retain(torch.randn(5, 5, requires_grad=True))
        except RuntimeError as error:
            self.assertRegex(str(error), "Specify retain_graph=True")

        # Case 2: no error with retain_graph=True
        @torch.jit.script
        def train_fn_jit_retain(middle, orig_x):
            y = middle + middle**2
            return torch.autograd.grad([y.sum()], [orig_x], retain_graph=True)

        @torch.jit.script
        def train_fn_fork_join_calls_retain(x):
            y_retain = (x + 3) * (x + 4) * 0.5
            fut1 = torch.jit._fork(train_fn_jit_retain, y_retain, x)
            fut2 = torch.jit._fork(train_fn_jit_retain, y_retain, x)
            grad = train_fn_jit_retain(y_retain, x)
            grad1 = torch.jit._wait(fut1)
            grad2 = torch.jit._wait(fut2)
            return grad, grad1, grad2

        grad, grad1, grad2 = train_fn_fork_join_calls_retain(
            torch.randn(5, 5, requires_grad=True)
        )
        self.assertEqual(grad, grad1)
        self.assertEqual(grad, grad2)

    def test_preserve_backtrace(self):
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx, *grad):
                raise ValueError("something")

        t = torch.rand(10, requires_grad=True)
        try:
            Foo.apply(t).sum().backward()
        except Exception:
            import traceback

            tb = sys.exc_info()[2]
            tb_str = "\n".join(traceback.format_tb(tb))
            self.assertTrue('raise ValueError("something")' in tb_str)

    # TODO(@anjali411): add an OpInfo based test for torch.cat
    # Issue: https://github.com/pytorch/pytorch/issues/51627
    #        https://github.com/pytorch/pytorch/issues/75852
    def test_cat_stack_r_to_c(self):
        inp_c = torch.rand(3, 2, dtype=torch.cdouble, requires_grad=True)
        inp_r = torch.randn(3, 2, dtype=torch.double, requires_grad=True)

        def fn(x1, x2):
            return torch.cat((x1, x2), dim=-1)

        def fn2(x1, x2):
            return torch.stack((x1, x2), dim=-1)

        torch.autograd.gradcheck(fn, [inp_r, inp_c], check_forward_ad=True)
        torch.autograd.gradcheck(fn, [inp_c, inp_r], check_forward_ad=True)

        torch.autograd.gradcheck(fn2, [inp_r, inp_c], check_forward_ad=True)
        torch.autograd.gradcheck(fn2, [inp_c, inp_r], check_forward_ad=True)

    def test_set_multithreading_enabled_as_context_manager_and_function(self):
        # Test as a context manager
        with torch.autograd.set_multithreading_enabled(False):
            self.assertFalse(torch.autograd.is_multithreading_enabled())
        self.assertTrue(torch.autograd.is_multithreading_enabled())

        with torch.autograd.set_multithreading_enabled(True):
            self.assertTrue(torch.autograd.is_multithreading_enabled())
        self.assertTrue(torch.autograd.is_multithreading_enabled())

        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.set_multithreading_enabled(True)
            self.assertTrue(torch.autograd.is_multithreading_enabled())
        self.assertTrue(torch.autograd.is_multithreading_enabled())

        torch.autograd.set_multithreading_enabled(False)
        self.assertFalse(torch.autograd.is_multithreading_enabled())

        torch.autograd.set_multithreading_enabled(True)
        self.assertTrue(torch.autograd.is_multithreading_enabled())

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_custom_function_propagates_errors_from_device_thread(self):
        class MyFunc(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, gO):
                raise RuntimeError("blah")
                return gO

        t = torch.tensor([1.0, 2.0], requires_grad=True, device=torch.device("cuda"))
        out = MyFunc.apply(t).sum()

        with self.assertRaisesRegex(RuntimeError, "blah"):
            out.backward()


class TestNestedCheckpoint(TestCase):
    @staticmethod
    def grad(fn):
        def wrapper(x):
            with torch.enable_grad():
                out = fn(x)
                (grad_input,) = torch.autograd.grad(out, inputs=(x,), create_graph=True)
            return grad_input

        return wrapper

    @staticmethod
    def sum(fn):
        def wrapped(x):
            return fn(x).sum()

        return wrapped

    @staticmethod
    def checkpoint(fn):
        def wrapped(*args, **kwargs):
            return torch.utils.checkpoint.checkpoint(
                fn, *args, use_reentrant=False, **kwargs
            )

        return wrapped

    def get_tests(self, fn):
        grad, c = self.grad, self.checkpoint

        tests = (
            # function <> tuple of function arbitrarily wrapped in checkpoint in various ways
            (fn, (c(fn), c(c(fn)))),
            (grad(fn), (grad(c(fn)), grad(c(c(fn))))),
            (
                grad(grad(fn)),
                (grad(c(grad(fn))), c(grad(grad(c(fn)))), grad(c(grad(c(fn))))),
            ),
            (
                grad(grad(grad(fn))),
                (grad(c(grad(grad(c(fn))))), grad(c(grad(c(grad(c(fn))))))),
            ),
        )
        return tests

    def check_graph_dies(self, fn):
        def iter_graph(roots):
            if not roots:
                return
            seen = set()
            q = collections.deque()
            for node in roots:
                if node is not None:
                    seen.add(node)
                    q.append(node)

            while q:
                node = q.popleft()
                for fn, _idx in node.next_functions:
                    if fn in seen or fn is None:
                        continue
                    seen.add(fn)
                    q.append(fn)

                yield node

        class Handle:
            __slot__ = ["node_name"]

            def __init__(self, node_name):
                self.node_name = node_name

        def scope():
            a = torch.randn((), requires_grad=True)
            out = fn(a)
            refs = []
            for node in iter_graph([out.grad_fn]):
                handle = Handle(node.name())
                refs.append(weakref.ref(handle))
                node.metadata["blah"] = handle
            return refs

        refs = scope()
        node_names = [ref().node_name for ref in refs if ref() is not None]
        if len(node_names) > 0:
            print("Nodes still alive:", node_names)

        self.assertEqual(len(node_names), 0)

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint(self, early_stop):
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            x = torch.randn((), requires_grad=True)

            def f(x):
                out = x.sin().exp().sin()
                return out

            def g(x):
                a = x.sin().exp().sin()
                b = x.sin().exp().sin()
                (ga,) = torch.autograd.grad(a, x)
                (gb,) = torch.autograd.grad(b, x)
                return x.sin()

            for fn in (f, g):
                for expected_fn, actual_fns in self.get_tests(fn):
                    expected = expected_fn(x)

                    for actual_fn in actual_fns:
                        actual = actual_fn(x)
                        self.assertTrue(torch.allclose(expected, actual))
                        self.check_graph_dies(actual_fn)

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_two_children(self, early_stop):
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            grad, sum, c = self.grad, self.sum, self.checkpoint

            def f(x):
                return x.sin().exp().sin()

            def g(x):
                return x.cos().sin().exp()

            def hc(x):
                return c(g)(c(f)(x))

            def h(x):
                return g(f(x))

            a = torch.randn(3, 3, requires_grad=True)
            expected = grad(sum(grad(sum(h))))(a)
            actual = grad(sum(grad(sum(c(hc)))))(a)
            self.assertTrue(torch.allclose(expected, actual))

            actual = grad(sum(c(grad(sum(c(hc))))))(a)
            self.assertTrue(torch.allclose(expected, actual))

            self.check_graph_dies(grad(c(hc)))
            self.check_graph_dies(grad(sum(grad(sum(c(hc))))))
            self.check_graph_dies(grad(sum(c(grad(sum(c(hc)))))))

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_non_tensor_inputs_and_outputs(self, early_stop):
        def fn(k, a, b, f):
            return f(k * a * b.exp()), 1, "abcd"

        k = 3
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)

        def f(x):
            return x.sin()

        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            out, _unused1, _unused2 = checkpoint(fn, k, a, b, f, use_reentrant=False)
        actual_grads = torch.autograd.grad(out, (a, b))

        out, _unused1, _unused2 = fn(k, a, b, f)
        expected_grads = torch.autograd.grad(out, (a, b))
        for actual, expected in zip(actual_grads, expected_grads):
            self.assertTrue(torch.allclose(actual, expected))

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_kwargs(self, early_stop):
        def fn(a, blah=None):
            out = a.sin().exp()
            if blah is not None:
                out = out * blah
            return out.sin().exp()

        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)

        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            out = checkpoint(fn, a, blah=b, use_reentrant=False)
            actual_grads = torch.autograd.grad(out, (a, b))

            out = fn(a, blah=b)
            expected_grads = torch.autograd.grad(out, (a, b))
            for actual, expected in zip(actual_grads, expected_grads):
                self.assertTrue(torch.allclose(actual, expected))

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_same_graph(self, early_stop):
        counter = [0]

        def hook(*_unused_args):
            counter[0] += 1

        def fn(a):
            return a.sin().cos().sin()

        a = torch.tensor(1.0, requires_grad=True)

        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            out = checkpoint(fn, a, use_reentrant=False)
        # The hook is registered on the original graph
        out.grad_fn.next_functions[0][0].register_hook(hook)
        # And backward is performed on the original graph
        out.backward()

        self.assertEqual(counter[0], 1)

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_reentrant_backwards(self, early_stop):
        def fn(a):
            x = a.sin().cos()
            out = x.sin()
            return x, out

        def hook(*_unused_args):
            # do backward again, but skip over the part of the graph where
            # the hook was registered
            x.backward(retain_graph=True)

        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            x, out = checkpoint(fn, a, use_reentrant=False)
        out.grad_fn.register_hook(hook)
        out.backward(retain_graph=True)

    def test_nested_checkpoint_set_early_stop(self):
        counter = [0]

        def clone(x):
            counter[0] += 1
            return x.clone()

        def fn(x):
            # Since clone does not save anything, it is not recomputed iff
            # early stop is enabled.
            return clone(x.sin().cos())

        # Test default
        # Early stopping is enabled by default
        a = torch.tensor(1.0, requires_grad=True)
        out = checkpoint(fn, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 1)

        # Test local setting
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        out = checkpoint(fn, a, use_reentrant=False, early_stop=False)
        out.backward()
        self.assertEqual(counter[0], 2)

        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        out = checkpoint(fn, a, use_reentrant=False, early_stop=True)
        out.backward()
        self.assertEqual(counter[0], 1)

        # Test context manager
        # Expect early stopping to be disabled for all checkpoints ran under
        # the context manager, even though context manager is no longer active
        # when backward/recomputation is performed.
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            out = checkpoint(fn, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 2)

        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(True):
            out = checkpoint(fn, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 1)

        # Test context manager nesting
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            with torch.utils.checkpoint.set_checkpoint_early_stop(True):
                out = checkpoint(fn, a, use_reentrant=False, early_stop=False)
        out.backward()
        self.assertEqual(counter[0], 1)

        # Test precedence
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            out = checkpoint(fn, a, use_reentrant=False, early_stop=True)
        out.backward()
        self.assertEqual(counter[0], 2)

        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(True):
            out = checkpoint(fn, a, use_reentrant=False, early_stop=False)
        out.backward()
        self.assertEqual(counter[0], 1)

    def test_nested_checkpoint_set_early_stop_no_recompution_needed(self):
        # Case 1: We have one tensor saved and its the input

        # We have two different counters here because in this case we actually
        # do call into x.sin() at the python level during recomputation whether
        # or not early stop is enabled. This is because the early stopping
        # only happens at the autograd level (preventing us from reaching the
        # backend).
        python_dispatch_counter = [0]
        counter = [0]

        class SinCounterMode(TorchDispatchMode):
            def __init__(self) -> None:
                self.count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = {} if kwargs is None else kwargs
                if func is torch.ops.aten.sin.default:
                    self.count += 1
                return func(*args, **kwargs)

        def fn(x):
            counter[0] += 1
            return x.sin()

        # With early stopping (enabled by default)
        a = torch.tensor(1.0, requires_grad=True)
        with SinCounterMode() as python_dispatch_counter:  # noqa: F811
            out = checkpoint(fn, a, use_reentrant=False)
            out.backward()
        self.assertEqual(counter[0], 2)
        self.assertEqual(python_dispatch_counter.count, 1)

        # Without early stopping
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with SinCounterMode() as python_dispatch_counter:
            with torch.utils.checkpoint.set_checkpoint_early_stop(False):
                out = checkpoint(fn, a, use_reentrant=False)
            out.backward()
        self.assertEqual(counter[0], 2)
        self.assertEqual(python_dispatch_counter.count, 2)

        # Case 2: Forward saves no tensors

        # Since unpack isn't even called, counter is 1 whether or not early stop
        # is enabled!
        counter = [0]

        def fn2(x):
            counter[0] += 1
            return x.clone()

        # With early stopping (enabled by default)
        a = torch.tensor(1.0, requires_grad=True)
        out = checkpoint(fn2, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 1)

        # Without early stopping
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            out = checkpoint(fn2, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 1)


class TestSelectiveActivationCheckpoint(TestCase):
    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_flops_and_mem(self):
        # From https://github.com/pytorch/pytorch/pull/126320
        def get_act_mem(f):
            out = f()
            out.backward()
            # Why do one forward and backward?
            start_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
            out = f()
            cur_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
            act_mem = (cur_mem - start_mem) / (1024 * 1024)
            out.backward()
            return act_mem

        def get_bw_flops(f):
            # Normalized so that a 512 square matmul returns 1
            f().backward()
            out = f()
            # NB: FlopCounterMode is pushed onto the mode stack before CachedMode, so
            # it will be able to observe whether an op is cached or not.
            with FlopCounterMode(display=False) as mode:
                out.backward()
            return mode.get_total_flops() / (512**3 * 2)

        x = torch.randn(512, 512, requires_grad=True, device="cuda")
        y = torch.randn(512, 512, requires_grad=True, device="cuda")

        def fn(x, y):
            return torch.mm(x.cos(), y).sin().sum()

        def fn_ac(x, y):
            return checkpoint(fn, x, y, use_reentrant=False)

        def fn_sac(x, y):
            context_fn = functools.partial(
                create_selective_checkpoint_contexts,
                [torch.ops.aten.mm.default],
            )
            out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
            return out

        def policy_fn(ctx, op, *args, **kwargs):
            if op == torch.ops.aten.mm.default:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        def fn_sac2(x, y):
            context_fn = functools.partial(
                create_selective_checkpoint_contexts,
                policy_fn,
            )
            out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
            return out

        def policy_fn_bool(ctx, op, *args, **kwargs):
            return op == torch.ops.aten.mm.default

        def fn_sac3(x, y):
            context_fn = functools.partial(
                create_selective_checkpoint_contexts,
                policy_fn_bool,
            )
            out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
            return out

        act_mem_noac = get_act_mem(lambda: fn(x, y))
        bw_flops_noac = get_bw_flops(lambda: fn(x, y))

        self.assertEqual(act_mem_noac, 2.0)
        self.assertEqual(bw_flops_noac, 2.0)

        act_mem_ac = get_act_mem(lambda: fn_ac(x, y))
        bw_flops_ac = get_bw_flops(lambda: fn_ac(x, y))

        self.assertEqual(act_mem_ac, 0.0)
        self.assertEqual(bw_flops_ac, 3.0)

        act_mem_sac = get_act_mem(lambda: fn_sac(x, y))
        bw_flops_sac = get_bw_flops(lambda: fn_sac(x, y))

        self.assertEqual(act_mem_sac, 1.0)
        self.assertEqual(bw_flops_sac, 2.0)

        act_mem_sac2 = get_act_mem(lambda: fn_sac2(x, y))
        bw_flops_sac2 = get_bw_flops(lambda: fn_sac2(x, y))

        self.assertEqual(act_mem_sac2, 1.0)
        self.assertEqual(bw_flops_sac2, 2.0)

        act_mem_sac3 = get_act_mem(lambda: fn_sac3(x, y))
        bw_flops_sac3 = get_bw_flops(lambda: fn_sac3(x, y))

        self.assertEqual(act_mem_sac3, 1.0)
        self.assertEqual(bw_flops_sac3, 2.0)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_output_already_has_autograd_meta(self):
        # View of tensor of non-differentiable dtype still has AutogradMeta
        def fn(x, y):
            return x.view(-1), y.sin().cos()

        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        y = torch.randn(3, requires_grad=True)

        context_fn = functools.partial(
            create_selective_checkpoint_contexts,
            [torch.ops.aten.view.default],
        )
        out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
        out[1].sum().backward()

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_subclass_dispatching_sizes(self):
        # Test that we ignore ops that grab metadata like torch.ops.aten.sym_size.default
        # Caching such metadata ops can be problematic when the following are satisfied:
        #
        # 1. size/strides are dispatched upon
        # 2. our policy saves sizes
        ta = torch.randn(6, 2)

        class CustomSizeDynamicShapesTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, inner):
                return torch.Tensor._make_wrapper_subclass(
                    # TODO: right now, _make_wrapper_subclass's dynamic shape interaction is not great.
                    # Calling the overload that has kwargs causes us to go down the first overload path,
                    # which will **always** specialize sizes.
                    # We should probably eventually fix this so that the first overload can just handle dynamic shapes.
                    cls,
                    inner.size(),
                    inner.stride(),
                    None,
                    None,
                    inner.dtype,
                    inner.layout,
                    inner.device,
                    False,
                    inner.requires_grad,
                    "sizes",
                )

            def __init__(self, inner):
                self.inner = inner

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                if kwargs is None:
                    kwargs = {}
                args_inner = torch.utils._pytree.tree_map_only(
                    cls, lambda x: x.inner, args
                )
                out_inner = func(*args_inner, **kwargs)
                return torch.utils._pytree.tree_map_only(
                    torch.Tensor, lambda x: cls(x), out_inner
                )

        def policy_fn(ctx, op, *args, **kwargs):
            if op is torch.ops.aten.sym_size.default:
                # Silently ignored!
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            # We avoid the following case
            #
            # saved     :[4, 3], [], [], [4, 3], [4, 3], [4, 3], [12]
            # forward   :sum   ,sum,mul, mul   , mul   ,view   , view
            # recompute :sum   ,sum,mul, view  , view
            #
            # Views save the shape of their input, so we expect the second
            # view to save 12, but because during AC packing during forward
            # saves the shapes of the input for metadata checks later,
            # we would save the wrong shape during the recompute.
            view_out = (x * x.sum()).view(-1).view(4, 3)
            self.assertEqual(view_out.grad_fn._saved_self_sym_sizes, [12])
            return view_out.exp()

        x = torch.randn(4, 3, requires_grad=True)
        x_wrapper = CustomSizeDynamicShapesTensor(x)
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        out = checkpoint(fn, x_wrapper, use_reentrant=False, context_fn=context_fn)
        out.sum().backward()

    def test_bad_inputs(self):
        bad_op_list1 = [2]

        with self.assertRaisesRegex(
            ValueError, "Expected op in `op_list` to be an OpOverload"
        ):
            create_selective_checkpoint_contexts(bad_op_list1)

        bad_op_list2 = [torch.ops.aten.sin]

        with self.assertRaisesRegex(
            ValueError, "update the OpOverloadPacket to a specific OpOverload"
        ):
            create_selective_checkpoint_contexts(bad_op_list2)

        with self.assertRaisesRegex(TypeError, "either a function or a list of ops."):
            create_selective_checkpoint_contexts(2)

    # Dynamo fails for various reasons:
    # - some tests using custom op that does not implement Fake
    # - dynamo is trying to trace into saved variable hooks unpack hook for some reason
    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_policy_with_state(self):
        # If I have a stateful callable, state is shared between the original
        # forward and the recompute.
        counters = []

        class Policy:
            def __init__(self) -> None:
                self.counter = [0]
                self.recompute_counter = [0]

            def __call__(self, ctx, func, *args, **kwargs):
                counter = self.recompute_counter if ctx.is_recompute else self.counter
                counter[0] += 1
                counters.append(counter[0])
                if counter == 1 and func is torch.ops.aten.mm.default:
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            return x.sin().sin().sin()

        x = torch.randn(3, requires_grad=True)
        context_fn = functools.partial(
            create_selective_checkpoint_contexts,
            Policy(),
            allow_cache_entry_mutation=True,
        )
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
        out.sum().backward()
        # 1. counter properly reset to 0 for the recompute
        # 2. due to early-stop we do not recompute the final op
        self.assertEqual(counters, [1, 2, 3, 1, 2])

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_storage_lifetime(self):
        from torch.utils._python_dispatch import _get_current_dispatch_mode
        from torch.utils.checkpoint import (
            _CachedTorchDispatchMode,
            _CachingTorchDispatchMode,
        )

        def policy_fn(ctx, op, *args, **kwargs):
            return CheckpointPolicy.MUST_SAVE

        ref = None

        def fn(x):
            nonlocal ref

            self.assertIsInstance(
                _get_current_dispatch_mode(),
                (_CachingTorchDispatchMode, _CachedTorchDispatchMode),
            )

            out = x.cos().exp()

            if isinstance(_get_current_dispatch_mode(), _CachingTorchDispatchMode):
                raw_val = (
                    _get_current_dispatch_mode()
                    .storage[torch.ops.aten.exp.default][0]
                    .val
                )
                # ref should've been detached
                # to avoid graph -> the saved variable hooks -> recompute_context -> storage -> graph
                self.assertFalse(raw_val.requires_grad)
                ref = weakref.ref(raw_val)

            # Careful for early-stop
            return out.sin()

        with disable_gc():
            # Case 1: If graph goes away without backward, make sure there's no reference cycle
            #         keeping storage alive.
            x = torch.randn(3, requires_grad=True)
            context_fn = functools.partial(
                create_selective_checkpoint_contexts, policy_fn
            )
            out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
            self.assertIsNotNone(ref())
            del out
            self.assertIsNone(ref())

            # Case 2: After backward, even if retain_graph=True, the storage should go away
            x = torch.randn(3, requires_grad=True)
            context_fn = functools.partial(
                create_selective_checkpoint_contexts, policy_fn
            )
            out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
            self.assertIsNotNone(ref())
            out.sum().backward(retain_graph=True)
            # The dispatch mode's storage should still be alive, but the entries should've
            # been cleared.
            self.assertIsNone(ref())

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_version_counter(self):
        def policy_fn(ctx, op, *args, **kwargs):
            if op == torch.ops.aten.sin.default:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            return x.sin().mul_(2).cos().exp()

        x = torch.randn(3, requires_grad=True)
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)

        # 1) Error because the output of sin is saved and mutated by mul_
        with self.assertRaisesRegex(RuntimeError, "has been mutated"):
            out.sum().backward()

        x = torch.randn(3, requires_grad=True)
        context_fn = functools.partial(
            create_selective_checkpoint_contexts,
            policy_fn,
            allow_cache_entry_mutation=True,
        )
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)

        # 2) No longer should be an error because of allow_cache_entry_mutation
        out.sum().backward()

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_function_with_more_than_one_output(self):
        # maybe there is a more systematic way:
        counter = [0]

        def policy_fn(ctx, op, *args, **kwargs):
            if op == torch.ops.aten.var_mean.correction:
                counter[0] += 1
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        # var_mean has two outputs
        def fn(x):
            a, b = torch.var_mean(x)
            return a * b

        x = torch.randn(3, requires_grad=True)
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
        x_grad = torch.autograd.grad(out.sum(), (x,))
        x_grad_ref = torch.autograd.grad(fn(x).sum(), (x,))
        self.assertEqual(x_grad, x_grad_ref)
        self.assertEqual(counter[0], 2)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_function_with_non_tensor_output(self):
        # When SAC is enabled, the op is not computed a second time
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            counter = [0]

            @torch.library.custom_op("mylib::sin_with_extra", mutates_args=())
            def sin_with_extra(x: torch.Tensor) -> tuple[torch.Tensor, int]:
                counter[0] += 1
                return x.sin(), 2

            def setup_context(ctx, inputs, output) -> torch.Tensor:
                (x,) = inputs
                ctx.save_for_backward(x)

            def backward(ctx, grad, _unused):
                (x,) = ctx.saved_tensors
                return grad * x.cos()

            torch.library.register_autograd(
                "mylib::sin_with_extra", backward, setup_context=setup_context
            )

            x = torch.randn(3, requires_grad=True)

            def fn(x):
                return (torch.ops.mylib.sin_with_extra(x)[0] * x.sin().exp()).sin()

            ops_list = [torch.ops.mylib.sin_with_extra.default]

            x = torch.randn(3, requires_grad=True)
            context_fn = functools.partial(
                create_selective_checkpoint_contexts, ops_list
            )
            out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
            x_grad = torch.autograd.grad(out.sum(), (x,))
            self.assertEqual(counter[0], 1)
            x_grad_ref = torch.autograd.grad(fn(x).sum(), (x,))
            self.assertEqual(x_grad, x_grad_ref)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_can_only_trigger_recompute_once(self):
        # We don't support this to avoid adding extra complexity for now.
        # If there's a need, we could probably do some kind of use_count tracking.
        # TODO: have a nice error message here.
        def policy_fn(ctx, op, *args, **kwargs):
            if op == torch.ops.aten.sin.default:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            return x.sin().cos().exp()

        x = torch.randn(3, requires_grad=True)
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
        out.sum().backward(retain_graph=True)

        with self.assertRaisesRegex(RuntimeError, "Trying to backward an extra time"):
            out.sum().backward(retain_graph=True)


class TestAutogradMultipleDispatch(TestCase):
    def test_autograd_multiple_dispatch_registrations(self, device):
        t = torch.randn(3, 3, device=device, requires_grad=True)
        # using _test_autograd_multiple_dispatch.fullcoverage which has
        # registrations in derivatives.yaml for Default, AutogradCUDA and NestedTensorAutograd
        out = torch._test_autograd_multiple_dispatch(t)
        grad = torch.randn(3, 3, device=device)
        out.backward(grad)

        if "cuda" not in device:
            # bogus default gradient registered for Autograd is grad + 1
            self.assertEqual(t.grad, grad + 1)
        else:
            # bogus gradient registered for AutogradCUDA is grad * 2
            self.assertEqual(t.grad, grad * 2)

        # test registered AutogradNestedTensor formula
        a = (
            torch.arange(6, dtype=torch.float, device=device)
            .reshape(2, 3)
            .requires_grad_(True)
        )
        b = (
            torch.arange(8, dtype=torch.float, device=device)
            .reshape(2, 4)
            .requires_grad_(True)
        )
        nt = torch.nested.as_nested_tensor([a, b], dtype=torch.float, device=device)

        nt_out = torch._test_autograd_multiple_dispatch(nt)
        c = torch.randn(2, 3, device=device)
        d = torch.randn(2, 4, device=device)
        nt_grad = torch.nested.nested_tensor([c, d], dtype=torch.float, device=device)
        nt_out.backward(nt_grad)

        # bogus gradient for AutogradNestedTensor is grad * grad
        self.assertEqual(a.grad, c * c)
        self.assertEqual(b.grad, d * d)

    def test_autograd_composite_implicit_and_dispatch_registration(self, device):
        t = torch.randn(3, 3, device=device, requires_grad=True)
        # using _test_autograd_multiple_dispatch.ntonly
        # which has registrations in derivatives.yaml for NestedTensorAutograd and otherwise is CompositeImplicit
        out = torch._test_autograd_multiple_dispatch(t, True)
        grad = torch.randn(3, 3, device=device)
        out.backward(grad)

        # t.grad is just out.grad by composite op since _test_autograd_multiple_dispatch is just a clone
        self.assertEqual(t.grad, grad)

        # test registered AutogradNestedTensor formula
        a = (
            torch.arange(6, dtype=torch.float, device=device)
            .reshape(2, 3)
            .requires_grad_(True)
        )
        b = (
            torch.arange(8, dtype=torch.float, device=device)
            .reshape(2, 4)
            .requires_grad_(True)
        )
        nt = torch.nested.as_nested_tensor([a, b], dtype=torch.float, device=device)

        nt_out = torch._test_autograd_multiple_dispatch(nt, True)
        c = torch.randn(2, 3, device=device)
        d = torch.randn(2, 4, device=device)
        nt_grad = torch.nested.nested_tensor([c, d], dtype=torch.float, device=device)
        nt_out.backward(nt_grad)

        # bogus gradient for AutogradNestedTensor is grad * grad + grad
        self.assertEqual(a.grad, c * c + c)
        self.assertEqual(b.grad, d * d + d)

    def test_foward_mode_AD(self, device):
        # check that forward mode AD is only registered for the Default
        # dispatch for _test_autograd_multiple_dispatch.fullcoverage and not AutogradCUDA

        primal = torch.randn(3, device=device)
        tangent = torch.randn(3, device=device)

        with fwAD.dual_level():
            dual_input = fwAD.make_dual(primal, tangent)

            err_msg = r"Trying to use forward AD with .* that does not support it"
            hint_msg = "Running forward AD for an OP that does not implement it should raise a NotImplementedError"

            if "cuda" in device:
                with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                    torch._test_autograd_multiple_dispatch(dual_input)
            else:
                torch._test_autograd_multiple_dispatch(dual_input)

    def test_view_copy(self, device):
        # tests that view_copy derivative formulas are also generated per dispatch key
        # from their respective view ops in derivatives.yaml
        t = torch.randn(2, 2, device=device, requires_grad=True)
        t_ref = t.detach().clone().requires_grad_()
        # _test_autograd_multiple_dispatch_view does a .view(-1) on the input
        t_view = torch._test_autograd_multiple_dispatch_view(t_ref)
        t_view_copy = torch._test_autograd_multiple_dispatch_view_copy(t)

        grad = torch.randn(4, device=device)
        t_view_copy.backward(grad)
        t_view.backward(grad.clone())

        # forward and backward give the same shape + result
        self.assertEqual(t_view_copy, t_view)
        self.assertEqual(t.grad, t_ref.grad)
        # backward results are per-dispatch-key in derivatives.yaml
        if "cuda" in device:
            # gradient registered to AutogradCUDA is grad.reshape_as(self) + 1
            self.assertEqual(t.grad, grad.reshape_as(t) + 1)
        else:
            # Default gradient registered is grad.reshape_as(self)
            self.assertEqual(t.grad, grad.reshape_as(t))

    @onlyCPU
    def test_per_dispatch_key_input_saving(self, device):
        # Tests that sum.dim_IntList's input is not saved for regular tensors but is saved for nested tensors
        def foo(x):
            # Don't modify the input inplace
            x = x.clone()
            res = x.sum(-1, keepdim=True)
            x.add_(x)
            return res

        inp = torch.rand(2, device=device, requires_grad=True)
        # sum's input is not saved for regular Tensors
        foo(inp).backward()

        # sum's input is saved for Nested Tensors
        nt = torch.nested.nested_tensor(
            [torch.rand(2), torch.rand(2)], device=device, requires_grad=True
        )
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            foo(nt).backward(
                torch.nested.nested_tensor(
                    [torch.rand(1), torch.rand(1)], device=device
                )
            )

    @onlyCUDA
    def test_backward_single_threaded(self):
        threads_eq = None

        class TestFn(Function):
            @staticmethod
            def forward(ctx, x, self):
                ctx.self = self
                ctx.tid = threading.get_ident()
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                nonlocal threads_eq
                threads_eq = ctx.tid == threading.get_ident()
                return gO, None

        inp = torch.rand(10, device="cuda", requires_grad=True)

        with torch.autograd.set_multithreading_enabled(False):
            TestFn.apply(inp, None).sum().backward()
        self.assertTrue(threads_eq)

        TestFn.apply(inp, None).sum().backward()
        self.assertFalse(threads_eq)

    @onlyCUDA
    def test_backward_tls_stash(self):
        local = threading.local()
        local.my_obj = {}
        local.my_obj[10] = 10
        test_self = self
        torch._C._stash_obj_in_tls("my_obj", local.my_obj)

        class TestFn(Function):
            @staticmethod
            def forward(ctx, x, self):
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                test_self.assertTrue(torch._C._is_key_in_tls("my_obj"))
                test_self.assertTrue(torch._C._get_obj_in_tls("my_obj")[10] == 10)
                torch._C._get_obj_in_tls("my_obj")[10] = 5
                return gO, None

        inp = torch.rand(10, device="cuda", requires_grad=True)

        TestFn.apply(inp, None).sum().backward()
        self.assertEqual(local.my_obj[10], 5)

    def test_is_retain_graph(self):
        retain_graph_set = False

        class TestFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                nonlocal retain_graph_set
                retain_graph_set = (
                    torch._C._autograd._get_current_graph_task_keep_graph()
                )
                return gO, None

        inp = torch.rand(10, requires_grad=True)

        out = TestFn.apply(inp)
        self.assertFalse(retain_graph_set)
        out.sum().backward(retain_graph=True)
        self.assertTrue(retain_graph_set)
        out.sum().backward(retain_graph=False)
        self.assertFalse(retain_graph_set)

    def test_set_sequence_nr(self):
        x = torch.randn((10,), dtype=torch.float32, requires_grad=True)
        y = torch.randn((10,), dtype=torch.float32, requires_grad=True)
        z = torch.randn((10,), dtype=torch.float32, requires_grad=True)

        a = x + y
        b = y + z
        c = a + b

        self.assertIsNotNone(a.grad_fn)
        self.assertIsNotNone(b.grad_fn)
        self.assertIsNotNone(c.grad_fn)

        a.grad_fn._set_sequence_nr(100)
        b.grad_fn._set_sequence_nr(99)
        c.grad_fn._set_sequence_nr(98)

        self.assertEqual(a.grad_fn._sequence_nr(), 100)
        self.assertEqual(b.grad_fn._sequence_nr(), 99)
        self.assertEqual(c.grad_fn._sequence_nr(), 98)

        def log_grad_order(grad: torch.Tensor, name: str, order):
            order.append(name)
            return grad

        order = []
        a.register_hook(partial(log_grad_order, name="a", order=order))
        b.register_hook(partial(log_grad_order, name="b", order=order))
        c.register_hook(partial(log_grad_order, name="c", order=order))

        c.sum().backward()

        # Expect to see that even though c has the smallest sequence number, it is still the first node to get run in autograd.
        # Also check that although a comes first during the forward, after giving it priority with sequence_nr,
        # its autograd node is run before that of b.
        self.assertEqual(order, ["c", "a", "b"])

        self.assertEqual(x.grad, torch.ones_like(x))
        self.assertEqual(y.grad, 2 * torch.ones_like(x))
        self.assertEqual(z.grad, torch.ones_like(x))

    def test_atan2_zero_gradient(self):
        x = torch.tensor([0.0], requires_grad=True)
        y = torch.tensor([0.0], requires_grad=True)
        z = torch.atan2(x, y)
        z.backward()
        self.assertEqual(x.grad, torch.zeros_like(x))
        self.assertEqual(y.grad, torch.zeros_like(y))

    # Test that torch.autograd.backward respects __torch_function__ on tensor subclasses.
    def test_backward_respects_torch_function(self):
        backward_called_with_subclass = [False]

        class AsyncTensorLike(torch.Tensor):
            """Tensor subclass that tracks when backward is called with it."""

            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_wrapper_subclass(
                    cls,
                    data.size(),
                    strides=data.stride(),
                    storage_offset=data.storage_offset(),
                    dtype=data.dtype,
                    layout=data.layout,
                    device=data.device,
                    requires_grad=data.requires_grad,
                )

            def __init__(self, data):
                # store the inner tensor
                self._data = data

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func is torch.autograd.backward:
                    backward_called_with_subclass[0] = True
                    # unwrap inner tensors and call the real backward
                    new_args = []
                    for arg in args:
                        if isinstance(arg, tuple):
                            new_args.append(
                                tuple(a._data if isinstance(a, cls) else a for a in arg)
                            )
                        elif isinstance(arg, cls):
                            new_args.append(arg._data)
                        else:
                            new_args.append(arg)
                    return func(*new_args, **kwargs)
                return func(
                    *tuple(a._data if isinstance(a, cls) else a for a in args), **kwargs
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                def unwrap(t):
                    return t._data if isinstance(t, cls) else t

                return func(
                    *torch.utils._pytree.tree_map(unwrap, args),
                    **torch.utils._pytree.tree_map(unwrap, kwargs or {}),
                )

        x = torch.randn(3, requires_grad=True)
        y = x * 2
        wrapped = AsyncTensorLike(y)
        torch.autograd.backward(wrapped, torch.ones_like(y))

        self.assertTrue(
            backward_called_with_subclass[0],
            "backward() should invoke __torch_function__ on tensor subclasses",
        )
        self.assertEqual(x.grad, 2 * torch.ones_like(x))

    def test_trace_backward_nonsquare_171704(self):
        # https://github.com/pytorch/pytorch/issues/171704
        for shape in [(5, 2), (7, 3), (4, 1), (1, 5), (2, 5)]:
            with self.subTest(shape=shape):
                x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
                torch.trace(x).backward()
                expected = torch.zeros(shape, dtype=torch.float64)
                for i in range(min(shape)):
                    expected[i, i] = 1.0
                self.assertEqual(x.grad, expected)


# Import test cases from below autograd/ here. These are found
# implicitly by the loader, so Flake8 thinks they are unused, hence
# the suppressions.

from autograd.test_complex import TestAutogradComplex  # noqa: F401
from autograd.test_functional import TestAutogradFunctional  # noqa: F401
from autograd.test_logging import TestAutogradLogging  # noqa: F401


# e.g., TestAutogradDeviceTypeCPU and TestAutogradDeviceTypeCUDA
instantiate_device_type_tests(TestAutogradDeviceType, globals(), except_for=None)

instantiate_device_type_tests(
    TestAutogradMultipleDispatch, globals(), only_for=("cpu", "cuda")
)
instantiate_device_type_tests(
    TestAutogradStreamSynchronization, globals(), except_for=None
)

instantiate_parametrized_tests(TestAutograd)
instantiate_parametrized_tests(TestNestedCheckpoint)

if __name__ == "__main__":
    run_tests()
