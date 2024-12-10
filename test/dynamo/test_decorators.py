# Owner(s): ["module: dynamo"]
import functools
import operator
import os
import unittest.mock as mock
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.exc import IncorrectUsage
from torch._dynamo.utils import counters


def my_custom_function(x):
    return x + 1


class DecoratorTests(torch._dynamo.test_case.TestCase):
    def test_disallow_in_graph(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
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

    def test_disable_for_custom_op(self):
        import torch.library
        from torch.library import Library

        foo = Library("foo", "DEF")  # noqa: TOR901
        foo.define("custom(Tensor self) -> Tensor")

        # Dynamic shape data dependent operator. For static shape compilation, Dynamo
        # should graph break on it. But, the meta kernel is not implemented properly.
        @torch.library.impl(foo, "custom", "CPU")
        def foo_cpu(x):
            return x.nonzero()

        # Disallow does not work because of extra python frames with torch.library python API
        torch.ops.foo.custom = torch._dynamo.disable(torch.ops.foo.custom)

        def fn(x):
            a = torch.nn.functional.relu(x)
            b = torch.ops.foo.custom(a)
            c = torch.cos(b)
            return c

        x = torch.randint(2, (100,))
        ref = fn(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(ref, res)

    def test_disable_ignores_outer_wraps(self):
        def orig_inner():
            pass

        def inner():
            pass

        inner._torchdynamo_orig_callable = orig_inner

        @functools.wraps(inner)
        def wrapper():
            raise AssertionError("wrapper called")

        # This behavior is not ideal, but supporting it would add overhead
        # to callsites of eval_frame.innermost_fn. A warning would also be very noisy.
        w = torch._dynamo.disable(fn=wrapper, recursive=True)

    def test_disable_nn_modules_forward_hook(self):
        class SimpleLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                return self.layer0(torch.sigmoid(inp))

        class SimpleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = SimpleLinear()
                self.layer1 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                z = self.layer0(torch.sin(inp))
                return self.layer1(z)

        def hook(module, args):
            inp = args[0].sigmoid()
            return (inp,)

        model = SimpleModel()
        model.layer0.register_forward_pre_hook(hook)

        # Disable my monkeypatching
        model.layer0 = torch._dynamo.disable(model.layer0)

        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")
        opt_model = torch.compile(model, backend=cnts)
        opt_model(torch.randn(4))

        # check for no graph break
        self.assertEqual(cnts.frame_count, 2)

        gm0 = cnts.graphs[0]
        # Check that the first graph has sin node, and no sigmoid
        self.assertTrue(any(node.target is torch.sin for node in gm0.graph.nodes))
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm0.graph.nodes)
        )

        gm1 = cnts.graphs[1]
        # Check that the first graph does not have sigmoid. sigmoid is used in
        # both hook and disabled module.
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm1.graph.nodes)
        )

    def test_disable_nn_module_with_class_decorator(self):
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")

        @torch._dynamo.disable
        class SimpleLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                return self.layer0(torch.sigmoid(inp))

        @torch.compile(backend=cnts)
        class SimpleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = SimpleLinear()
                self.layer1 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                z = self.layer0(torch.sin(inp))
                return self.layer1(z)

        def hook(module, args):
            inp = args[0].sigmoid()
            return (inp,)

        model = SimpleModel()
        model.layer0.register_forward_pre_hook(hook)

        model(torch.randn(4))

        # check for no graph break
        self.assertEqual(cnts.frame_count, 2)

        gm0 = cnts.graphs[0]
        # Check that the first graph has sin node, and no sigmoid
        self.assertTrue(any(node.target is torch.sin for node in gm0.graph.nodes))
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm0.graph.nodes)
        )

        gm1 = cnts.graphs[1]
        # Check that the first graph does not have sigmoid. sigmoid is used in
        # both hook and disabled module.
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm1.graph.nodes)
        )

    def test_allow_in_graph(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
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

    def test_incorrect_usage_disallow_in_graph(self):
        with self.assertRaises(IncorrectUsage):

            @torch._dynamo.disallow_in_graph
            def fn1(x):
                return x.cos()

    def test_graph_break(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            x = torch.cos(x)
            x = torch.cos(x)
            torch._dynamo.graph_break()
            x = torch.cos(x)
            x = torch.cos(x)
            torch._dynamo.graph_break()
            x = torch.cos(x)
            x = torch.cos(x)
            return x

        fn(torch.randn(4, 5))
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 6)

    def test_skip(self):
        def fn2(x):
            return x.sin()

        @torch._dynamo.disable(recursive=False)
        def fn1(x):
            x = x.sigmoid()
            return fn2(x.cos())

        def fn(x):
            return fn1(x.tan())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        opt_fn(torch.randn(4))
        self.assertEqual(cnts.frame_count, 2)

    def test_substitute_in_graph(self):
        counters.clear()

        # NB: Choose another C function for test when we support operator.indexOf
        #     out of the box
        cnts = torch._dynamo.testing.CompileCounter()
        fn = operator.indexOf
        opt_fn = torch.compile(fn, backend=cnts)
        out = fn([1, 2, 3, 4, 5], 3)
        opt_out = opt_fn([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, opt_out)
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(len(counters["graph_break"]), 1)

        torch._dynamo.reset()
        counters.clear()

        with self.assertRaisesRegex(TypeError, "Signature mismatch"):

            @torch._dynamo.substitute_in_graph(operator.indexOf)
            def _(sequence, x):
                for i, item in enumerate(sequence):
                    if item is x or item == x:
                        return i
                raise ValueError("sequence.index(x): x not in sequence")

        @torch._dynamo.substitute_in_graph(operator.indexOf)
        def polyfill(a, b):
            for i, item in enumerate(a):
                if item is b or item == b:
                    return i
            raise ValueError("sequence.index(x): x not in sequence")

        cnts = torch._dynamo.testing.CompileCounter()
        fn = operator.indexOf
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        out = fn([1, 2, 3, 4, 5], 3)
        opt_out = opt_fn([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, opt_out)
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(len(counters["graph_break"]), 0)

        torch._dynamo.reset()
        counters.clear()

        cnts = torch._dynamo.testing.CompileCounter()
        fn = polyfill
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        out = fn([1, 2, 3, 4, 5], 3)
        opt_out = opt_fn([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, opt_out)
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(len(counters["graph_break"]), 0)

    @patch.object(torch._dynamo.config, "suppress_errors", True)
    def test_nested_disable_decorator(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.disable()
        def fn1(x):
            return torch.sin(x) * 10

        @torch.compile(backend=cnts)
        def fn2(x):
            x = x + 1
            x = x + 1
            x = fn1(x)  # graph break
            x = x + 1
            x = x + 1
            return x

        @torch.compile(backend=cnts, fullgraph=True)
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

    def test_disable_optimize(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, disable=True)
        def f1(x):
            return x + 1

        f1(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

        @torch.compile(backend=cnt, disable=True)
        def f2(x):
            return x + 1

        f2(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

        with patch.dict(os.environ, {"TORCHDYNAMO_DISABLE": "1"}):

            @torch.compile(backend=cnt)
            def f3(x):
                return x + 1

            f3(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

    def test_torch_guards_stack_frame_register_inlining_disable(self):
        x = torch.tensor([0.5, 0.5])

        class encoder(torch.nn.Module):
            def __init__(self, y):
                super().__init__()
                self.a = y

            @torch._dynamo.disable
            def helper(self, x, y):
                return x * y

            def forward(self, a, *args):
                x = a + a
                return self.helper(x, self.a)

        e = encoder(2.0)

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
            torch.compile(e, backend="eager")(x)

        self.assertEqual(len(seen_frames), 0)

    def test_torch_guards_stack_frame_register_inlining_partially_disable(self):
        y = torch.nn.Parameter(torch.tensor([0.25, 0.25]))
        x = torch.tensor([0.5, 0.5])

        class encoder(torch.nn.Module):
            def __init__(self, y):
                super().__init__()
                self.register_parameter("param", y)

            @torch._dynamo.disable
            def helper_disabled(self, x, y):
                return x.sin() * y.cos()

            def helper(self, x, y):
                return x * y

            def forward(self, a, *args):
                x = a + a
                return self.helper(x, self.param) + self.helper_disabled(x, self.param)

        e = encoder(y)

        cnt = torch._dynamo.testing.CompileCounter()
        torch.compile(e, backend=cnt)(x)

        # first frame is before disable, second frame is after disable
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 3)

    def _test_mark_static_address(self, guarded):
        # This test verifies that dynamo properly marks inputs as static
        # when using the mark_static_address API.
        # On 1st compile, we expect the input to be marked as static, with guarded
        # set depending on the `guarded` flag.
        # On 2nd compile, we expect the input to be unmarked
        # if inlining NN modules, we expect metadata to be present on the tensor, indicating
        # the static address type of the input
        # if not inlining NN modules, we expect the tensor to be present in the buffers attribute
        # of the graph.

        compiles_with_buffers = 0
        compiles = 0

        def debug_compiler(gm, _):
            nonlocal compiles_with_buffers
            nonlocal compiles
            if torch._dynamo.config.inline_inbuilt_nn_modules:
                input_node = [
                    n
                    for n in gm.graph.nodes
                    if n.op == "placeholder" and n.name == "l_x_"
                ]
                self.assertEqual(len(input_node), 1)
                input_node = input_node[0]
                if compiles == 0:
                    self.assertEqual(
                        input_node.meta["tensor_dict"]["_dynamo_static_input_type"],
                        "guarded" if guarded else "unguarded",
                    )
                elif compiles == 1:
                    self.assertFalse(
                        "_dynamo_static_input_type" in input_node.meta["tensor_dict"]
                    )
                else:
                    raise RuntimeError(f"Unexpected number of compiles: {compiles}")
            else:
                compiles_with_buffers += len(gm._buffers) > 0
            compiles += 1
            return gm

        @torch.compile(backend=debug_compiler)
        def fn(x):
            return x + 1

        inp = torch.ones(2)

        torch._dynamo.mark_static_address(inp, guard=guarded)

        fn(inp)
        if not torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertEqual(compiles_with_buffers, 1)

        inp2 = torch.ones(2)

        # if guarded, should trigger another recompile
        # since it was not marked static, compiles with buffers
        # should not be incremented
        fn(inp2)

        if not torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertEqual(compiles_with_buffers, 1)

        self.assertEqual(compiles, 2 if guarded else 1)

    def test_mark_static_address_guarded(self):
        with torch._dynamo.config.patch("inline_inbuilt_nn_modules", True):
            self._test_mark_static_address(guarded=True)

        self._test_mark_static_address(guarded=True)

    def test_mark_static_address_unguarded(self):
        with torch._dynamo.config.patch("inline_inbuilt_nn_modules", True):
            self._test_mark_static_address(guarded=False)

        self._test_mark_static_address(guarded=False)

    def test_class_methods(self):
        class A:
            @classmethod
            def my_class_method(cls, arg1):
                return cls, arg1

            @staticmethod
            def my_static_method(arg1):
                return None, arg1

            def my_regular_method(self, arg1):
                return self, arg1

        class B(A):
            def my_class_method(self, arg1):
                return super().my_class_method(arg1)

            def my_static_method(self, arg1):
                return super().my_static_method(arg1)

        class C(A):
            @classmethod
            def my_class_method(cls, arg1):
                return super().my_class_method(arg1)

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(a, b, c):
            # We want a function that does not graph break but
            # does generate custom bytecode
            v1 = a.my_class_method(1)
            v2 = A.my_class_method(2)
            v3 = a.my_static_method(3)
            v4 = A.my_static_method(4)
            v5 = a.my_regular_method(5)
            v6 = b.my_class_method(6)
            v7 = b.my_static_method(7)
            v8 = c.my_class_method(8)
            v9 = C.my_class_method(9)
            torch.rand(2)
            return v1, v2, v3, v4, v5, v6, v7, v8, v9

        a, b, c = A(), B(), C()
        v1, v2, v3, v4, v5, v6, v7, v8, v9 = fn(a, b, c)

        self.assertEqual(v1, (A, 1))
        self.assertEqual(v2, (A, 2))
        self.assertEqual(v3, (None, 3))
        self.assertEqual(v4, (None, 4))
        self.assertEqual(v5, (a, 5))
        # TODO fix me: we do not resolve classmethods properly
        # from a regular method
        # self.assertEqual(v6, (B, 6))
        self.assertEqual(v7, (None, 7))
        self.assertEqual(v8, (C, 8))
        self.assertEqual(v9, (C, 9))

        self.assertEqual(cnt.frame_count, 1)

    def test_assume_constant_result_on_user_defined_fn(self):
        @torch._dynamo.assume_constant_result
        def const_fn(n, s):
            return torch.full([n], s)

        def fn(B):
            B = const_fn(B.size(0), 13)
            X = B * 2
            return X.tolist()

        B_list = [8] * 32

        B = torch.tensor(B_list, dtype=torch.int32)
        torch._dynamo.decorators.mark_static(B, 0)

        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        self.assertEqual(
            fn(B), torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)(B)
        )

    def test_assume_constant_result_on_computation_with_graph_input(self):
        @torch._dynamo.assume_constant_result
        def check(y):
            return y[0].item() == 1

        def fn(x, y):
            if check(y):
                return x + 2
            else:
                return x + 1

        y = torch.tensor([1])
        x = torch.tensor(1)

        self.assertEqual(fn(x, y), torch.compile(fn)(x, y))

    @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
    def test_mark_static_nn_module(self):
        @torch._dynamo.mark_static
        class Mock(torch.nn.Module):
            def __init__(self, c):
                super().__init__()
                self.c = c

            def forward(self, x):
                return x * self.c

        cnts = torch._dynamo.testing.CompileCounter()
        mod1 = Mock(10)
        mod2 = Mock(20)
        mod3 = Mock(30)
        opt_mod1 = torch.compile(mod1, backend=cnts, fullgraph=True)
        opt_mod2 = torch.compile(mod2, backend=cnts, fullgraph=True)
        opt_mod3 = torch.compile(mod3, backend=cnts, fullgraph=True)

        x = torch.randn(4, 4)
        opt_mod1(x)
        opt_mod2(x)
        opt_mod3(x)

        # Must be 3 compilations. If not marked static there would be 2, because self.c would be converted to symints.
        self.assertEqual(cnts.frame_count, 3)

    def test_set_stance_force_eager(self):
        @torch.compile(backend="eager")
        def a(x):
            if torch._dynamo.is_compiling():
                return x + 1
            return x + 2

        @torch.compiler.set_stance("force_eager")
        def b(x):
            return a(x)

        def c(x):
            out0 = a(x)
            with torch.compiler.set_stance("force_eager"):
                out1 = a(x)
            return out0, out1, a(x)

        inp = torch.ones(3)
        # test that decorating b has no overall side effect
        self.assertEqual(a(inp), inp + 1)

        self.assertEqual(b(inp), inp + 2)
        self.assertEqual(c(inp), (inp + 1, inp + 2, inp + 1))

        torch.compiler.set_stance("force_eager")
        self.assertEqual(a(inp), inp + 2)
        torch.compiler.set_stance("default")
        self.assertEqual(a(inp), inp + 1)

    def test_set_stance_eager_on_recompile(self):
        @torch.compile(backend="eager", dynamic=False)
        def a(x, n):
            if torch._dynamo.is_compiling():
                return x + n + 1
            return x + n + 2

        inp = torch.ones(3)
        out1 = a(inp, 1)
        with torch.compiler.set_stance("eager_on_recompile"):
            out2 = a(inp, 1)
            out3 = a(inp, 2)

        self.assertEqual(out1, inp + 2)
        self.assertEqual(out2, inp + 2)
        self.assertEqual(out3, inp + 4)

    def test_set_stance_fail_on_recompile(self):
        @torch.compile(backend="eager", dynamic=False)
        def a(x, n):
            if torch._dynamo.is_compiling():
                return x + n + 1
            return x + n + 2

        inp = torch.ones(3)
        out1 = a(inp, 1)
        with torch.compiler.set_stance("fail_on_recompile"):
            out2 = a(inp, 1)
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                a(inp, 2)

        self.assertEqual(out1, inp + 2)
        self.assertEqual(out2, inp + 2)

    def test_set_stance_fail_on_recompile_with_disable(self):
        @torch.compiler.disable
        def inner(x):
            return x

        @torch.compile(backend="eager")
        def f(x):
            return inner(x)

        f(torch.randn(3, 3))
        # should not raise error
        with torch.compiler.set_stance("fail_on_recompile"):
            f(torch.randn(3, 3))

    def test_set_stance_forbid_in_graph(self):
        @torch.compiler.set_stance("force_eager")
        def a(x):
            return x + 1

        @torch.compile(backend="eager")
        def b(x):
            return a(x)

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            b(torch.ones(3))

        @torch.compile(backend="eager")
        def c(x):
            with torch.compiler.set_stance("force_eager"):
                return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            c(torch.ones(3))

        @torch.compile(backend="eager")
        @torch.compiler.set_stance("force_eager")
        def d(x):
            return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            d(torch.ones(3))

        @torch.compile(backend="eager")
        def e(x):
            with torch._dynamo.set_stance("force_eager"):
                return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            e(torch.ones(3))

        @torch.compile(backend="eager")
        def f(x):
            torch._dynamo.eval_frame._set_stance("force_eager")
            return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            f(torch.ones(3))

        @torch.compile(backend="eager")
        def g(x):
            # cause a skipped frame
            try:
                torch._dynamo.graph_break()
            except Exception:
                pass
            # NOTE: torch._dynamo.is_compiling() will get traced
            # and return true. torch.compiler.is_compiling() is skipped
            # and will return false.
            if torch.compiler.is_compiling():
                raise RuntimeError("Expect this frame to be skipped")
            # should not be traced, but eval frame callback is still set
            with torch.compiler.set_stance("force_eager"):
                return x + 1

        with self.assertRaisesRegex(RuntimeError, "set_stance in a torch.compile"):
            g(torch.ones(3))

    def test_set_stance_force_backend(self):
        @torch.compile
        def a(x):
            return x + 1

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compiler.set_stance("default", force_backend=cnts)
        def b(x):
            return a(x)

        b(torch.ones(3))

        self.assertEqual(cnts.frame_count, 1)

        @torch.compiler.set_stance("default", force_backend="eager")
        def c(x):
            return a(x)

        # just make sure this doesn't crash
        c(torch.ones(3))

        with self.assertRaisesRegex(RuntimeError, "force_backend"):

            @torch.compiler.set_stance("force_eager", force_backend="eager")
            def d(x):
                pass

    def test_set_stance_force_backend_with_disable(self):
        @torch.compiler.disable
        def inner(x):
            return x

        @torch.compile(backend="eager")
        def f(x):
            return inner(x)

        f(torch.randn(3, 3))

        def fail_backend(gm, ex):
            raise RuntimeError("fail!")

        # should not raise error
        with torch.compiler.set_stance("default", force_backend=fail_backend):
            f(torch.randn(3, 3))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
