# Owner(s): ["module: dynamo"]
import contextvars
import threading

import torch
import torch._dynamo
import torch._dynamo.testing
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter


class TestContextVars(TestCase):
    def test_get_with_constructor_default(self):
        cv = contextvars.ContextVar("precision", default="fp32")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            p = cv.get()
            if p == "fp32":
                return x + 1
            return x + 2

        x = torch.randn(4)
        ref = x + 1
        self.assertEqual(fn(x), ref)

    def test_get_with_explicit_default(self):
        cv = contextvars.ContextVar("test_explicit")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            val = cv.get("fallback")
            if val == "fallback":
                return x * 2
            return x * 3

        x = torch.randn(4)
        self.assertEqual(fn(x), x * 2)

    def test_get_rejects_keyword_default(self):
        cv = contextvars.ContextVar("test_keyword")

        @torch.compile(backend="eager")
        def fn(x):
            cv.get(default="kw_fallback")
            return x

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            fn(x)

    def test_get_rejects_extra_positional_args(self):
        cv = contextvars.ContextVar("test_extra_args")

        @torch.compile(backend="eager")
        def fn(x):
            cv.get(1, 2)
            return x

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            fn(x)

    def test_get_with_set_value(self):
        cv = contextvars.ContextVar("test_set", default="default")
        token = cv.set("active")

        try:

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                val = cv.get()
                if val == "active":
                    return x + 10
                return x

            x = torch.randn(4)
            self.assertEqual(fn(x), x + 10)
        finally:
            cv.reset(token)

    def test_get_no_default_no_value(self):
        cv = contextvars.ContextVar("no_default")

        @torch.compile(backend="eager")
        def fn(x):
            cv.get()
            return x

        x = torch.randn(4)
        with self.assertRaises(LookupError):
            fn(x)

    def test_recompilation_on_value_change(self):
        cv = contextvars.ContextVar("recompile_test", default="a")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            val = cv.get()
            if val == "a":
                return x + 1
            return x + 2

        x = torch.randn(4)

        self.assertEqual(fn(x), x + 1)
        self.assertEqual(cnt.frame_count, 1)

        token = cv.set("b")
        try:
            self.assertEqual(fn(x), x + 2)
            self.assertEqual(cnt.frame_count, 2)
        finally:
            cv.reset(token)

    def test_no_recompilation_same_value(self):
        cv = contextvars.ContextVar("no_recompile", default="stable")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            cv.get()
            return x + 1

        x = torch.randn(4)
        fn(x)
        fn(x)
        fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_multiple_cvs(self):
        cv1 = contextvars.ContextVar("cv1", default=1)
        cv2 = contextvars.ContextVar("cv2", default=2)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            a = cv1.get()
            b = cv2.get()
            return x + a + b

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 3)

    def test_fullgraph_get_only(self):
        cv = contextvars.ContextVar("fullgraph_ok", default="yes")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            cv.get()
            return x + 1

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)

    def test_fullgraph_set_errors(self):
        cv = contextvars.ContextVar("fullgraph_set", default="val")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            cv.set("new")
            return x

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            fn(x)

    def test_set_graph_breaks(self):
        cv = contextvars.ContextVar("set_breaks", default="old")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            x = x + 1
            cv.set("new")
            return x + 2

        x = torch.randn(4)
        fn(x)
        self.assertEqual(cnt.frame_count, 2)

    def test_reset_graph_breaks(self):
        cv = contextvars.ContextVar("reset_breaks", default="val")
        token = cv.set("tmp")

        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            x = x + 1
            cv.reset(token)
            return x + 2

        x = torch.randn(4)
        fn(x)
        self.assertEqual(cnt.frame_count, 2)

    def test_cv_name_attribute(self):
        cv = contextvars.ContextVar("my_var_name", default=0)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            name = cv.name
            if name == "my_var_name":
                return x + 1
            return x

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)

    def test_nn_module_forward(self):
        cv = contextvars.ContextVar("module_cv", default="train")

        class MyModule(torch.nn.Module):
            def forward(self, x):
                mode = cv.get()
                if mode == "train":
                    return x * 2
                return x

        mod = MyModule()
        compiled = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(compiled(x), x * 2)

    def test_cv_across_graph_break(self):
        cv = contextvars.ContextVar("graph_break_cv", default="before")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            val = cv.get()
            if val == "before":
                x = x + 1
            print("graph break")
            val2 = cv.get()
            if val2 == "before":
                x = x + 2
            return x

        x = torch.randn(4)
        result = fn(x)
        self.assertEqual(result, x + 3)
        self.assertEqual(cnt.frame_count, 2)

    def test_cv_as_function_arg(self):
        cv = contextvars.ContextVar("arg_cv", default="hello")

        def read_cv(c):
            return c.get()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            val = read_cv(cv)
            if val == "hello":
                return x + 5
            return x

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 5)

    def test_cv_in_closure(self):
        cv = contextvars.ContextVar("closure_cv", default="closed")

        def make_fn():
            def fn(x):
                val = cv.get()
                if val == "closed":
                    return x - 1
                return x

            return fn

        compiled = torch.compile(make_fn(), backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(compiled(x), x - 1)

    def test_tensor_value(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        cv = contextvars.ContextVar("tensor_cv", default=t)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            bias = cv.get()
            return x + bias

        x = torch.randn(3)
        self.assertEqual(fn(x), x + t)

    def test_guard_none_value(self):
        cv = contextvars.ContextVar("none_cv", default=None)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            val = cv.get()
            if val is None:
                return x + 1
            return x + 2

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)

    def test_list_value(self):
        cv = contextvars.ContextVar("list_cv", default=[10, 20])  # noqa: B039

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            items = cv.get()
            return x + len(items)

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 2)

    def test_list_default_graph_breaks(self):
        cv = contextvars.ContextVar("list_default")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            x = x + 1
            val = cv.get([1, 2, 3])
            return x + len(val)

        x = torch.randn(4)
        fn(x)
        self.assertEqual(cnt.frame_count, 2)

    def test_recompilation_on_context_run(self):
        cv = contextvars.ContextVar("ctx_run", default="original")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            val = cv.get()
            if val == "original":
                return x + 1
            return x + 2

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)
        self.assertEqual(cnt.frame_count, 1)

        ctx = contextvars.copy_context()
        ctx.run(cv.set, "changed")
        result = ctx.run(fn, x)
        self.assertEqual(result, x + 2)
        self.assertEqual(cnt.frame_count, 2)

    def test_recompilation_across_threads(self):
        cv = contextvars.ContextVar("thread_cv", default="main")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            val = cv.get()
            if val == "main":
                return x + 1
            return x + 2

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)
        self.assertEqual(cnt.frame_count, 1)

        results = []
        errors = []

        def thread_fn():
            try:
                cv.set("thread")
                results.append(fn(x))
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=thread_fn)
        t.start()
        t.join()
        self.assertEqual(len(errors), 0, f"Thread raised: {errors}")
        self.assertEqual(results[0], x + 2)
        self.assertEqual(cnt.frame_count, 2)


if __name__ == "__main__":
    run_tests()
