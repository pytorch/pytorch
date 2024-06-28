# Owner(s): ["module: dynamo"]
import copy
import re
import unittest
from textwrap import dedent
from unittest.mock import patch

import torch

import torch._dynamo
import torch._dynamo.test_case
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._dynamo.testing import CompileCounter, expectedFailureDynamic, rand_strided
from torch._functorch.aot_autograd import _aot_export_function, create_functional_call
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.profiler import profile
from torch.testing._internal.common_utils import compare_equal_outs_and_grads


def maybe_dupe_op(x):
    y = x + 1
    z = x + 2
    if x.numel() < 5:
        return y, y
    else:
        return y, z


aten = torch.ops.aten
lib = torch.library.Library("custom", "DEF")  # noqa: TOR901
lib.define("maybe_dupe_op(Tensor a) -> (Tensor, Tensor)")
lib.impl("maybe_dupe_op", maybe_dupe_op, "CPU")
lib.impl("maybe_dupe_op", maybe_dupe_op, "Meta")


class AotAutogradFallbackTests(torch._dynamo.test_case.TestCase):
    def test_LSTM(self):
        # https://github.com/pytorch/torchdynamo/issues/1147
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_mod_model_lstm_lstm = torch.nn.LSTM(
                    64, 64, num_layers=2, bidirectional=True
                )

            def forward(self, permute: torch.Tensor):
                self_mod_model_lstm_lstm = self.self_mod_model_lstm_lstm(permute)
                return (self_mod_model_lstm_lstm,)

        mod = Repro()

        aot_mod = torch._dynamo.optimize("aot_eager")(mod)

        args = [((92, 4, 64), (1, 5888, 92), torch.float32, "cpu", False)]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        eager_result = mod(*args)
        aot_result = aot_mod(*args)
        self.assertTrue(torch._dynamo.testing.same(eager_result, aot_result))

    def test_mutation(self):
        # https://github.com/pytorch/torchdynamo/issues/1301
        def fn(param, y):
            prev_grad = torch.is_grad_enabled()
            try:
                torch.set_grad_enabled(False)
                param.add_(y)
            finally:
                torch.set_grad_enabled(prev_grad)
            return y

        y = torch.randn(4)
        x = torch.nn.Parameter(torch.randn(4))
        aot_fn = torch._dynamo.optimize("aot_eager")(fn)
        # This should not error: we mutated an autograd leaf under no_grad mode.
        aot_fn(x, y)

    def test_mutation1(self):
        def fn(_stack0: torch.Tensor, diagonal_chunked_attention_scores: torch.Tensor):
            getitem = diagonal_chunked_attention_scores[
                (
                    slice(None, None, None),
                    slice(None, None, None),
                    slice(None, 256, None),
                    slice(None, 257, None),
                )
            ]
            _stack0[
                (
                    slice(None, None, None),
                    slice(None, -1, None),
                    slice(None, None, None),
                    slice(256, None, None),
                )
            ] = getitem
            view = _stack0.view(1, 12, 1024, 513)
            return (view,)

        x = torch.randn(torch.Size([12, 4, 256, 513]))
        y = torch.randn(torch.Size([12, 3, 512, 513]))
        aot_fn = torch._dynamo.optimize("aot_eager")(fn)
        aot_fn(x, y)

    def test_negative_testing_mutation(self):
        def fn(_stack0: torch.Tensor, diagonal_chunked_attention_scores: torch.Tensor):
            getitem = diagonal_chunked_attention_scores[
                (
                    slice(None, None, None),
                    slice(None, None, None),
                    slice(None, 256, None),
                    slice(None, 257, None),
                )
            ]
            _stack0 = torch.sin(_stack0)
            _stack0[
                (
                    slice(None, None, None),
                    slice(None, -1, None),
                    slice(None, None, None),
                    slice(256, None, None),
                )
            ] = getitem
            view = _stack0.view(1, 12, 1024, 513)
            return (view,)

        x = torch.randn(torch.Size([12, 4, 256, 513]))
        y = torch.randn(torch.Size([12, 3, 512, 513]))
        aot_fn = torch._dynamo.optimize("aot_eager")(fn)
        aot_fn(x, y)

    def test_negative_testing(self):
        def fn(x, y):
            return torch.sin(x).add_(y)

        y = torch.randn(4)
        x = torch.randn(4)
        aot_fn = torch._dynamo.optimize("aot_eager")(fn)
        aot_fn(x, y)

    def test_call_fn_with_non_const_inputs_aot_safe(self):
        class ModuleSpecialFwd(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=20, kernel_size=(5, 5)
                )

            def _conv_forward(self, x):
                return self.conv._conv_forward(x, self.conv.weight, self.conv.bias)

            def forward(self, x):
                return self._conv_forward(x)

        # Init mod
        mod = ModuleSpecialFwd()
        rx = torch.randn([3, 10, 10])

        # Run it for real
        real = mod(rx)

        # Run it in export
        graph, _ = torch._dynamo.export(mod)(rx)

        # Run exported graph with AOT
        self.assertTrue(torch._dynamo.testing.same(real, graph(rx)))

        aot_fn = torch._dynamo.optimize("aot_eager")(graph)
        aot_fn(rx)

    def test_call_fn_with_non_const_inputs_aot_unsafe(self):
        class ModuleSpecialFwd(torch.nn.Module):
            def _some_bad_fwd(self, param, y):
                prev_grad = torch.is_grad_enabled()
                try:
                    torch.set_grad_enabled(False)
                    param.add_(y)
                finally:
                    torch.set_grad_enabled(prev_grad)
                return y

            def forward(self, x, y):
                return self._some_bad_fwd(x, y)

        # Init mod
        mod = ModuleSpecialFwd()
        x = torch.nn.Parameter(torch.randn(4))
        y = torch.randn([4])

        # Run it for real
        real = mod(x, y)

        # Run it in export
        graph, _ = torch._dynamo.export(mod)(x, y)

        # Assert equal
        self.assertTrue(torch._dynamo.testing.same(real, graph(x, y)))

        # Run exported graph with AOT
        aot_fn = torch._dynamo.optimize("aot_eager")(graph)
        # This should not error: we mutated an autograd leaf under no_grad mode.
        aot_fn(x, y)

    def test_call_fn_with_non_const_inputs_aot_unsafe_control_flow(self):
        class ModuleSpecialFwd(torch.nn.Module):
            def _some_bad_fwd(self, param, y):
                if y[0][0] < 3:
                    return y + param
                return param * y

            def forward(self, x, y):
                a = x * y
                a = self._some_bad_fwd(a, a)
                b = x + y
                return a * b

        # Init mod
        mod = ModuleSpecialFwd()
        x = torch.nn.Parameter(torch.randn([2, 2]))
        y = torch.randn([2, 2])

        # Run it for real
        real = mod(x, y)

        # Run it through optimize, with our capturing fn

        gms = []
        counter = CompileCounter()

        def capturing_fn(gm, inputs):
            nonlocal gms
            gms.append(gm)
            return counter(gm, inputs)

        optimized_mod = torch._dynamo.optimize(capturing_fn)(mod)

        # Assert equal
        self.assertTrue(torch._dynamo.testing.same(real, optimized_mod(x, y)))

        # Uncomment to reproduce commented out graphs below.
        # for gm in gms:
        #     print("GM CODE", gm.code)

        self.assertEqual(counter.frame_count, 4)
        self.assertEqual(counter.op_count, 7)
        # Graph 1
        # def forward(self, x : torch.nn.parameter.Parameter, y : torch.Tensor):
        #     mul = x * y;  x = y = None
        #     return (mul,)
        # BREAK
        # Graph 2
        # def forward(self, y : torch.Tensor):
        #     getitem = y[0];  y = None
        #     getitem_1 = getitem[0];  getitem = None
        #     lt = getitem_1 < 3;  getitem_1 = None
        #     return (lt,)
        # BREAK
        # Graph 3
        # def forward(self, param : torch.Tensor, y : torch.Tensor):
        #     add = y + param;  y = param = None
        #     return (add,)
        # BREAK
        # Graph 4
        # def forward(self, _stack0 : torch.Tensor, x : torch.nn.parameter.Parameter, y : torch.Tensor):
        #     add = x + y;  x = y = None
        #     mul = _stack0 * add;  _stack0 = add = None
        #     return (mul,)

        # Run fn with AOT
        torch._dynamo.reset()

        aot_fn = torch._dynamo.optimize("aot_eager")(optimized_mod)
        aot_fn(x, y)

    # Note: Dynamo recompilation guarding invalid grad
    #
    # This test is a spiritual equivalent to test_invalid_requires_grad_fake in test_autodispatch.py
    # The point of this test is to invoke aot_autograd in a way that would normally trigger an assertion
    # (This is what test_invalid_requires_grad_fake) does. However, the point of this test is to prove
    # that we do not hit this assertion, as dynamo recompiles correctly and protects this condition.
    #
    # Subnote: The reason for us having test_invalid_requires_grad_fake utilizing fake tensors
    # is because dynamo sends fake tensors down to aot_autograd.
    @patch("torch._functorch.config.debug_assert", True)
    def test_requires_grad_fake_via_dynamo_recompiles(self):
        class F(torch.nn.Module):
            def forward(self, x, y):
                return (x + y,)

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        z = torch.randn(3, 3, requires_grad=False)

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        failure_reason = None

        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        fxy = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        compare_equal_outs_and_grads(self, F(), fxy, (x, y))
        compare_equal_outs_and_grads(self, F(), fxy, (x, z))
        self.assertIn(
            """tensor 'L['y']' requires_grad mismatch. expected requires_grad=1""",
            failure_reason,
        )

        # Reset failure reason
        failure_reason = None

        self.assertEqual(cc.frame_count, 2)

        torch._dynamo.reset()  # for new backend
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        fxz = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        compare_equal_outs_and_grads(self, F(), fxz, (x, z))
        compare_equal_outs_and_grads(self, F(), fxz, (x, z))
        self.assertEqual(cc.frame_count, 1)
        self.assertTrue(failure_reason is None)

    def test_double_backward_errors(self):
        # Remove this test after we get double backward to actually work
        for grad_output in (torch.tensor(1.0, requires_grad=True), None):
            x = torch.tensor(1.0, requires_grad=True)
            err = "torch.compile with aot_autograd does not currently support double backward"

            # The following cases should be equivalent:

            # (1) double backward entirely inside compiled function
            def f1(x):
                y = x.sin().exp()
                (gx,) = torch.autograd.grad(
                    y, x, create_graph=True, grad_outputs=grad_output
                )
                torch.autograd.grad(gx, x)
                return gx

            compiled_f1 = torch.compile(backend="aot_eager")(f1)
            f1(x)
            with self.assertRaisesRegex(RuntimeError, err):
                compiled_f1(x)

            # (2) the second half of double backward outside compiled function
            def f2(x):
                y = x.sin().exp()
                (gx,) = torch.autograd.grad(
                    y, x, create_graph=True, grad_outputs=grad_output
                )
                return gx

            compiled_f2 = torch.compile(backend="aot_eager")(f2)
            gx = compiled_f2(x)
            with self.assertRaisesRegex(RuntimeError, err):
                torch.autograd.grad(gx, x)

            # (3) double backward entirely outside compiled function
            def f3(x):
                y = x.sin().exp()
                return y

            compiled_f3 = torch.compile(backend="aot_eager")(f3)
            y = compiled_f3(x)
            (gx,) = torch.autograd.grad(
                y, x, create_graph=True, grad_outputs=grad_output
            )
            with self.assertRaisesRegex(RuntimeError, err):
                torch.autograd.grad(gx, x)

        # create_graph=False
        def f4(x):
            y = x.sin().exp()
            return y

        compiled_f4 = torch.compile(backend="aot_eager")(f4)
        x = torch.tensor(1.0, requires_grad=True)
        y = compiled_f4(x)
        (gx,) = torch.autograd.grad(y, x, create_graph=False, grad_outputs=grad_output)

    @patch("torch._functorch.config.debug_assert", True)
    def test_arg_dupe_via_dynamo_recompiles(self):
        class F(torch.nn.Module):
            def forward(self, x, y):
                x = x.trunc_()
                y = y.trunc_()
                return (x + y,)

        x = torch.randn(3, 3, requires_grad=True)
        x1, x2, x3, x4 = x.clone(), x.clone(), x.clone(), x.clone()
        y = torch.randn(3, 3, requires_grad=True)
        y1, y2, y4 = y.clone(), y.clone(), y.clone()

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        failure_reason = None

        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        fxy = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        # Note: to prevent a recompilation between the two calls,
        # we need to clone x and y on each use.
        # fxy mutates the input's metadata, so otherwise dynamo will end up recompiling.
        fxy(x1, y1)
        fxy(x2, y2)

        self.assertTrue(failure_reason is None)

        # Reset failure reason
        failure_reason = None

        self.assertEqual(cc.frame_count, 1)

        torch._dynamo.reset()  # for new backend
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        fxx = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        fxx(x3, x3)
        fxx(x4, y4)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn("""L['x'] is L['y']""", failure_reason)

    @patch("torch._functorch.config.debug_assert", True)
    def test_arg_dupe_via_dynamo_recompiles_many_args_param_non_tensor_arg(self):
        class F(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, a, b, e, f):
                a.trunc_()
                b.trunc_()
                return (a + b + self.mean) * e * f

        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        a1, a2 = a.clone(), a.clone()
        b1, b2 = b.clone(), b.clone()

        failure_reason = None

        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        self.assertTrue(failure_reason is None)

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a1, a1, 2, 2)
        f(a2, b2, 2, 2)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn(
            """L['a'] is L['b']""",
            failure_reason,
        )

        torch._dynamo.reset()

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        c3, c4 = c.clone(), c.clone()
        d3, d4 = d.clone(), d.clone()

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(c3, c3, 3, 3)
        f(c4, d4, 3, 3)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn("""L['a'] is L['b']""", failure_reason)

    @patch("torch._functorch.config.debug_assert", True)
    def test_arg_dupe_via_dynamo_recompiles_many_with_global(self):
        z = None

        class F(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, a, b, e, f):
                a.trunc_()
                b.trunc_()
                return (a + b + z + self.mean) * e * f

        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        z = a
        a1, a2 = a.clone(), a.clone()
        b1, b2 = b.clone(), b.clone()

        failure_reason = None

        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        self.assertTrue(failure_reason is None)

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a1, a1, 2, 2)
        f(a2, b2, 2, 2)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn(
            """L['a'] is L['b']""",
            failure_reason,
        )

    @patch("torch._functorch.config.debug_assert", True)
    def test_arg_dupe_via_dynamo_recompiles_many_args_param_non_tensor_arg_list(self):
        class F(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, e, f, a, b):
                a.trunc_()
                b.trunc_()
                return (a + b + self.mean) * e[0] * f[0]

        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        a1, a2 = a.clone(), a.clone()
        b1, b2 = b.clone(), b.clone()

        failure_reason = None

        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        self.assertTrue(failure_reason is None)

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f([3, 2, 1], [4, 5, 6], a1, a1)
        f([3, 2, 1], [4, 5, 6], a2, b2)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn(
            """L['a'] is L['b']""",
            failure_reason,
        )

        torch._dynamo.reset()

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        c3, c4 = c.clone(), c.clone()
        d3, d4 = d.clone(), d.clone()

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f([3, 2, 1], [4, 5, 6], c3, c3)
        f([3, 2, 1], [4, 5, 6], c4, d4)
        self.assertEqual(cc.frame_count, 2)

    @patch("torch._functorch.config.debug_assert", True)
    def test_arg_dupe_via_dynamo_recompiles_many_args_param(self):
        class F(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, a, b):
                a.trunc_()
                b.trunc_()
                return a + b + self.mean

        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        a1, a2 = a.clone(), a.clone()
        b1, b2 = b.clone(), b.clone()

        failure_reason = None

        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        self.assertTrue(failure_reason is None)

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a1, a1)
        f(a2, b2)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn(
            """L['a'] is L['b']""",
            failure_reason,
        )

        torch._dynamo.reset()

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        c3, c4 = c.clone(), c.clone()
        d3, d4 = d.clone(), d.clone()

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(c3, c3)
        f(c4, d4)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn("""L['a'] is L['b']""", failure_reason)

    @patch("torch._functorch.config.debug_assert", True)
    def test_arg_dupe_via_dynamo_recompiles_many_args(self):
        class F(torch.nn.Module):
            def forward(self, a, b, c, d):
                a.trunc_()
                b.trunc_()
                c.trunc_()
                d.trunc_()
                return (a + b + c + d,)

        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        a1, a2, a3, a4 = a.clone(), a.clone(), a.clone(), a.clone()
        b1, b2, b3, b4 = b.clone(), b.clone(), b.clone(), b.clone()

        failure_reason = None

        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        self.assertTrue(failure_reason is None)

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a1, a1, a1, a1)
        f(a2, b2, b2, b2)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn(
            """L['a'] is L['b']""",
            failure_reason,
        )

        torch._dynamo.reset()

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        c3, c4 = c.clone(), c.clone()
        d3, d4 = d.clone(), d.clone()

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a3, b3, c3, c3)
        f(a4, b4, c4, d4)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn("""L['c'] is L['d']""", failure_reason)

    def test_alias_inputs(self):
        def fn():
            a = torch.tensor([1])
            a = a[0:1]
            b = a.squeeze()
            a[0] = 0
            if a[0] < 1e5:
                pass
            a[0] = 2
            return b

        ref_output = fn()
        aot_fn = torch._dynamo.optimize("aot_eager")(fn)
        actual_output = aot_fn()
        self.assertEqual(ref_output, actual_output)

    def test_grad_inputs_alias_inputs(self):
        class Test(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx, grad):
                (x,) = ctx.saved_tensors
                return x, grad

        def fn(x, y):
            return Test.apply(x, y)

        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="aot_eager")
        out = compiled_fn(x, y)
        out.sum().backward()

    @expectedFailureDynamic  # https://github.com/pytorch/pytorch/issues/103539
    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    @patch("torch._functorch.config.debug_assert", True)
    def test_multiple_aot_autograd_calls_dupe_args(self):
        # this is just dealing with the fact that
        # aot_module_simplified expects submods to always return tuples/lists
        class WrapperModule(torch.nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.mod = mod

            def forward(self, *args):
                out = self.mod(*args)
                if isinstance(out, (list, tuple)):
                    return out
                return (out,)

        def compile_submod(input_mod, args):
            from functorch.compile import nop
            from torch._functorch.aot_autograd import aot_module_simplified

            class WrapperModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.original = input_mod
                    self.submod = aot_module_simplified(input_mod, args, nop)

                def forward(self, *args):
                    return self.submod(*args)

            return WrapperModule()

        def test_compile(fx_g, example_inps):
            split_gm = torch.fx.passes.split_module.split_module(
                fx_g, None, lambda node: 1 if "mul" in str(node) else 0
            )
            submod_1_inps = split_gm.submod_0(*example_inps)
            split_gm.submod_0 = compile_submod(
                WrapperModule(split_gm.submod_0), example_inps
            )
            split_gm.submod_1 = compile_submod(
                WrapperModule(split_gm.submod_1), submod_1_inps
            )
            return split_gm

        @torch._dynamo.optimize(test_compile)
        def f(a):
            b, c = torch.ops.custom.maybe_dupe_op(a)
            return (b.mul_(c),)

        f(torch.ones(4))
        f(torch.ones(6))

    def test_nn_parameter_construction(self):
        # https://github.com/pytorch/pytorch/issues/99569
        def fn(x):
            y = x.sin()
            z = torch.nn.Parameter(torch.ones(1))
            return y + z

        x = torch.rand((4, 4))

        opt_fn = torch._dynamo.optimize("aot_eager")(fn)
        self.assertTrue(torch._dynamo.testing.same(fn(x), opt_fn(x)))

    def test_aot_sequence_nr(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(1, 1),
                    stride=1,
                    padding="same",
                    bias=True,
                )
                self.bn1 = torch.nn.BatchNorm2d(num_features=16)
                self.relu1 = torch.nn.ReLU()
                self.fc1 = torch.nn.Linear(in_features=1638400, out_features=1)
                self.loss_fn = torch.nn.L1Loss()

            def forward(self, x, target):
                y = x
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = x + y
                x = torch.flatten(x)
                x = self.fc1(x)
                output = self.loss_fn(x, target)

                return (output,)

        mod = Model()
        mod.train()
        x = torch.rand(100, 16, 32, 32, requires_grad=True)
        target = torch.rand(1)

        # Use dynamo export to get the fx graph module
        g_mod, _ = torch._dynamo.export(mod, x, target)

        def _prepare_model_args():
            named_parameters = dict(g_mod.named_parameters(remove_duplicate=False))
            named_buffers = dict(g_mod.named_buffers(remove_duplicate=False))
            params_and_buffers = {
                **dict(named_parameters),
                **dict(named_buffers),
            }
            params_and_buffers_flat, params_spec = pytree.tree_flatten(
                params_and_buffers
            )
            params_len = len(params_and_buffers_flat)
            functional_call = create_functional_call(g_mod, params_spec, params_len)
            return params_and_buffers_flat, functional_call

        full_args, fn_to_trace = _prepare_model_args()
        param_and_buf_len = len(full_args)
        full_args.extend([x, target])

        # aot_export requires a graph mod input of fwd graph
        # returns the full fwd/bwd graph in graph mod format
        with torch.enable_grad(), fx_traceback.preserve_node_meta():
            fx_g, _, _, _ = _aot_export_function(
                fn_to_trace,
                full_args,
                decompositions=None,
                num_params_buffers=param_and_buf_len,
                no_tangents=True,
            )

        # Walk all the nodes in fx graph.
        # Write the resulting ops to a table
        min_seq_nr = -1
        seq_table = "SeqNr|OrigAten|SrcFn\n"
        for node in fx_g.graph.nodes:
            if "call_" in node.op and "getitem" not in str(node.target):
                seq_nr = node.meta.get("seq_nr", -1)
                if seq_nr < 0:
                    continue
                if min_seq_nr < 0:
                    min_seq_nr = seq_nr
                source_fn_stack = node.meta.get("source_fn_stack", [])
                orig_aten = node.meta.get("original_aten", "")
                mod_name = ""
                if len(source_fn_stack) > 0:
                    mod_name = source_fn_stack[-1][0]
                # Make all seq_nr relative so it starts at 0
                seq_nr = seq_nr - min_seq_nr
                seq_table = seq_table + f"{seq_nr}|{orig_aten}|{mod_name}\n"

        self.maxDiff = None
        self.assertExpectedInline(
            seq_table,
            dedent(
                """\
SeqNr|OrigAten|SrcFn
0|aten.convolution.default|l__self___conv1
0|aten.add.Tensor|l__self___bn1
1|aten._native_batch_norm_legit_functional.default|l__self___bn1
2|aten.relu.default|l__self___relu1
2|aten.detach.default|l__self___relu1
2|aten.detach.default|l__self___relu1
3|aten.add.Tensor|add
4|aten.view.default|flatten
5|aten.view.default|l__self___fc1
6|aten.t.default|l__self___fc1
7|aten.addmm.default|l__self___fc1
8|aten.view.default|l__self___fc1
9|aten.sub.Tensor|l__self___loss_fn
10|aten.abs.default|l__self___loss_fn
11|aten.mean.default|l__self___loss_fn
11|aten.ones_like.default|
11|aten.expand.default|
11|aten.div.Scalar|
10|aten.sgn.default|
10|aten.mul.Tensor|
8|aten.view.default|
7|aten.t.default|
7|aten.mm.default|
7|aten.t.default|
7|aten.mm.default|
7|aten.t.default|
7|aten.sum.dim_IntList|
7|aten.view.default|
6|aten.t.default|
5|aten.view.default|
4|aten.view.default|
2|aten.detach.default|
2|aten.detach.default|
2|aten.threshold_backward.default|
1|aten.native_batch_norm_backward.default|
0|aten.convolution_backward.default|
11|aten.add.Tensor|
"""
            ),
        )

    def test_split_with_sizes_aot_autograd_cleans_up_traceback_meta(self):
        from torch._functorch.aot_autograd import setup_stacktrace_preservation_hooks

        def fn(result, split_sizes):
            rs = torch.ops.aten.split_with_sizes(result, split_sizes.tolist())
            return rs

        example_inputs = (
            torch.randn(32, requires_grad=True),
            torch.tensor((7, 16, 9)),
        )
        outs = fn(*example_inputs)
        setup_stacktrace_preservation_hooks([out.grad_fn for out in outs])
        with fx_traceback.preserve_node_meta():
            (outs[0].sum() + outs[1].sum() + outs[2].sum()).backward()

        self.assertNotIn("grad_fn_seq_nr", fx_traceback.current_meta)
        self.assertNotIn("in_grad_fn", fx_traceback.current_meta)

    # https://github.com/pytorch/pytorch/issues/110121
    def test_aot_export_joint_simple_repro(self):
        class Mod(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.linear = torch.nn.Linear(5, 7)

            def forward(self, x):
                return self.linear(x)

        def mini_backend(gm, sample_inputs):
            from torch._functorch.aot_autograd import aot_export_joint_simple

            fake_mode = torch._dynamo.utils.detect_fake_mode(sample_inputs)

            with patch.object(fake_mode, "allow_non_fake_inputs", True), fake_mode:
                return aot_export_joint_simple(gm, sample_inputs, trace_joint=False)

        sample_inputs = [torch.rand((3, 4, 5))]
        model = Mod()
        m_compiled = torch.compile(model, backend=mini_backend)

        out_ref = model(*sample_inputs)
        out_test = m_compiled(*sample_inputs)
        self.assertEqual(out_ref, out_test)

    def test_eager_sequence_nr(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(1, 1),
                    stride=1,
                    padding="same",
                    bias=True,
                )
                self.bn1 = torch.nn.BatchNorm2d(num_features=16)
                self.relu1 = torch.nn.ReLU()
                self.fc1 = torch.nn.Linear(in_features=1638400, out_features=1)
                self.loss_fn = torch.nn.L1Loss()

            def forward(self, x, target):
                y = x
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = x + y
                x = torch.flatten(x)
                x = self.fc1(x)
                output = self.loss_fn(x, target)

                return (output,)

        def grad_with_create_graph(mod, x, target):
            y = mod(x, target)
            # Set create_graph=True to ensure that the sequence_nr
            # for backward ops continues to count down.
            (gx,) = torch.autograd.grad(
                y[0], x, create_graph=True, grad_outputs=grad_output
            )
            return gx

        x = torch.rand(100, 16, 32, 32, requires_grad=True)
        target = torch.rand(1)
        mod = Model()
        args = [mod, x, target]
        grad_output = torch.tensor(1.0, requires_grad=True)
        compiled_f1 = torch.compile(backend="aot_eager")(grad_with_create_graph)
        model_instance = compiled_f1
        with profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
        ) as kineto_prof:
            res = model_instance(*args)
        bwd_set = set()
        prof_str = "SeqNr|Thread|FwdThread|Name\n"
        for event in kineto_prof.events():
            if event.sequence_nr >= 0:
                prof_str = (
                    prof_str + f"{event.sequence_nr}|{event.thread}"
                    f"|{event.fwd_thread}|{event.name}|\n"
                )
                if re.search(r"Backward[01]", event.name):
                    bwd_set.add(event.sequence_nr)
        self.assertTrue(len(bwd_set), 13)

    def test_aot_grad_mode_mutation(self):
        for compiler in ["aot_eager", "inductor"]:

            def f(x):
                y = x * x
                torch.set_grad_enabled(False)
                return y.clone(), y

            f_compiled = torch.compile(f, backend=compiler, fullgraph=True)

            torch.set_grad_enabled(True)
            x = torch.ones(3, requires_grad=True) * 3
            y_ref = f(x)
            self.assertEqual(torch.is_grad_enabled(), False)
            torch.set_grad_enabled(True)
            y = f_compiled(x)
            self.assertEqual(torch.is_grad_enabled(), False)
            torch.set_grad_enabled(True)
            self.assertEqual(y_ref, y)

            self.assertIsNone(y_ref[0].grad_fn)
            self.assertIsNone(y[0].grad_fn)

            self.assertIsNotNone(y_ref[1].grad_fn)
            self.assertIsNotNone(y[1].grad_fn)

            # Check that the grad computed for the inputs, given the input, is the same
            # The tangent to `y[0]`, which has grad_required=False, is irrelevant
            self.assertEqual(
                sum(y_ref[1].grad_fn(torch.tensor([-1.0, 2.0, 0.0]))),
                sum(
                    x
                    for x in y[1].grad_fn.apply(None, torch.tensor([-1.0, 2.0, 0.0]))
                    if x is not None
                ),
            )

    def test_aot_autograd_raises_invalid_leaf_set(self):
        @torch.compile
        def f(x):
            x.set_(torch.ones(2))

        # We still want to make sure that this raises
        x = torch.ones(2, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "is being used in an in-place operation"
        ):
            f(x)

    def test_aot_autograd_expand_mutation_functionalizes(self):
        def fn(x):
            y = x.expand(3, *x.shape)
            y[0, 0].add_(5)
            return y

        opt_fn = torch.compile(fn, backend="aot_eager")

        x = torch.arange(6)
        x_opt = x.clone().detach()
        self.assertEqual(fn(x), opt_fn(x_opt))
        self.assertEqual(x, x_opt)

    def test_aot_autograd_expand_mutation_backwards(self):
        def fn(x, z):
            y = x.expand(3, *x.shape)
            y[1, 1].mul_(5)
            ret = y * z
            return ret

        opt_fn = torch.compile(fn, backend="aot_eager")

        x = torch.arange(6, dtype=torch.float)
        z = x.clone().detach()
        x_opt = x.clone().detach()
        z_opt = x.clone().detach()

        z.requires_grad = True
        z_opt.requires_grad = True

        res = fn(x, z)
        opt_res = opt_fn(x_opt, z_opt)

        self.assertEqual(res, opt_res)

        res.sum().backward()
        opt_res.sum().backward()

        self.assertEqual(x, x_opt)
        self.assertEqual(z.grad, z_opt.grad)

    def test_data_ptr_access_copy(self):
        import torch._functorch.config as _config

        with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
            with FakeTensorMode():
                x = torch.randn(3)
                y = copy.copy(x)
        self.assertEqual(y.shape, x.shape)

    def test_data_ptr_access_fails_in_forward(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define("mylib::foo", "(Tensor x) -> Tensor", lib=lib)

            @torch.library.impl("mylib::foo", "CompositeImplicitAutograd", lib=lib)
            def _(x):
                x.data_ptr()
                return x.clone()

            x = torch.randn(3)

            def data_ptr_graph_input(x):
                r0 = torch.ops.mylib.foo(x)
                return r0

            def data_ptr_graph_intermediate(x):
                y = x.clone()
                r0 = torch.ops.mylib.foo(y)
                return r0

            tests = [data_ptr_graph_input, data_ptr_graph_intermediate]

            def ctx():
                return self.assertRaisesRegex(
                    RuntimeError, "Cannot access data pointer"
                )

            for f in tests:
                with ctx():
                    make_fx(f, tracing_mode="fake")(x)
                with ctx():
                    make_fx(f, tracing_mode="symbolic")(x)
                with ctx():
                    torch.compile(f, backend="eager", fullgraph=True)(x)

    def test_data_ptr_access_fails_in_backward(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define("mylib::foo", "(Tensor x) -> Tensor", lib=lib)

            backward_called = False

            class Foo(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x.clone()

                @staticmethod
                def backward(ctx, grad):
                    nonlocal backward_called
                    backward_called = True
                    grad.data_ptr()
                    return grad.clone()

            @torch.library.impl("mylib::foo", "CompositeImplicitAutograd", lib=lib)
            def _(x):
                return Foo.apply(x)

            def f(x):
                return torch.ops.mylib.foo(x)

            x = torch.randn(3, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, "Cannot access data pointer"):
                y = torch.compile(f, backend="aot_eager", fullgraph=True)(x)
            self.assertTrue(backward_called)

    # We don't know how to catch multiple mutations to the same memory location
    @unittest.expectedFailure
    def test_aot_autograd_expand_mutation_error(self):
        def fn(x):
            y = x.expand(3, *x.shape)
            y[0:3, 0].add_(5)
            return y

        opt_fn = torch.compile(fn, backend="aot_eager")

        x = torch.arange(6)
        x_opt = x.clone().detach()
        with self.assertRaises(Exception):
            fn(x)
        with self.assertRaises(Exception):
            opt_fn(x_opt)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
