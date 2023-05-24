# Owner(s): ["module: dynamo"]
import os
import unittest.mock as mock
from unittest.mock import patch

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.exc import IncorrectUsage


def my_custom_function(x):
    return x + 1


class DecoratorTests(torch._dynamo.test_case.TestCase):
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

    def test_disable_for_custom_op(self):
        import torch.library
        from torch.library import Library

        foo = Library("foo", "DEF")
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
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(ref, res)

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

    def test_incorrect_usage_disallow_in_graph(self):
        with self.assertRaises(IncorrectUsage):

            @torch._dynamo.disallow_in_graph
            def fn1(x):
                return x.cos()

    def test_graph_break(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnts)
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
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        opt_fn(torch.randn(4))
        self.assertEqual(cnts.frame_count, 2)

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

    def test_torch_guards_stack_frame_register_inlining_disable(self):
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
                return x * y

            def helper(self, x, y):
                return x * y

            def forward(self, a, *args):
                x = a + a
                return self.helper(x, self.param) + self.helper_disabled(x, self.param)

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
        self.assertEqual(seen_frames[0].name, "forward")
        self.assertEqual(
            seen_frames[0].line,
            "return self.helper(x, self.param) + self.helper_disabled(x, self.param)",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
