# Owner(s): ["module: dynamo"]
import contextvars
import threading

import torch
import torch._dynamo
import torch._dynamo.testing
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import HardwareClassification


class TestContextVars(TestCase):
    hw_classification = HardwareClassification.GENERIC

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

    def test_fullgraph_set_and_reset(self):
        cv = contextvars.ContextVar("fullgraph_set", default="val")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            token = cv.set("new")
            if cv.get() == "new":
                x = x + 1
            cv.reset(token)
            return x

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)
        self.assertEqual(cv.get(), "val")

    def test_get_with_explicit_token_missing_binding(self):
        cv = contextvars.ContextVar("token_missing_binding", default="fallback")
        cv.set(contextvars.Token.MISSING)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            if cv.get() is contextvars.Token.MISSING:
                return x + 1
            return x + 2

        self.assertEqual(fn(torch.tensor(0)), torch.tensor(1))
        self.assertIs(cv.get(), contextvars.Token.MISSING)

    def test_reset_to_constructor_default_list(self):
        default = []
        cv = contextvars.ContextVar("list_default_after_reset", default=default)

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            token = cv.set("tmp")
            cv.reset(token)
            items = cv.get()
            items.append("x")
            return len(items)

        self.assertEqual(fn(), 1)
        self.assertEqual(default, ["x"])
        self.assertIs(cv.get(), default)
        self.assertEqual(cv.get(), ["x"])

    def test_reset_to_constructor_default_tensor(self):
        default = torch.tensor([0.0])
        cv = contextvars.ContextVar("tensor_default_after_reset", default=default)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            token = cv.set(x + 1)
            cv.reset(token)
            return cv.get() + 2

        self.assertEqual(fn(torch.tensor([3.0])), torch.tensor([2.0]))
        self.assertIs(cv.get(), default)
        self.assertEqual(cv.get(), default)

    def test_reset_to_constructor_default_custom_object(self):
        class Box:
            def __init__(self) -> None:
                self.values = []

        default = Box()
        cv = contextvars.ContextVar("object_default_after_reset", default=default)

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            token = cv.set("tmp")
            cv.reset(token)
            obj = cv.get()
            obj.values.append("x")
            return len(obj.values)

        self.assertEqual(fn(), 1)
        self.assertEqual(default.values, ["x"])
        self.assertIs(cv.get(), default)
        self.assertEqual(cv.get().values, ["x"])

    def test_set_and_get_do_not_graph_break(self):
        cv = contextvars.ContextVar("set_breaks", default="old")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            token = cv.set("new")
            if cv.get() == "new":
                x = x + 1
            cv.reset(token)
            return x + 2

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 3)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cv.get(), "old")

    def test_return_dict_key_from_pre_mutation_get(self):
        cv = contextvars.ContextVar("dict_key_before_set", default="old")

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            key = cv.get()
            cv.set("new")
            return {key: 1}

        self.assertEqual(fn(), {"old": 1})
        self.assertEqual(cv.get(), "new")

    def test_return_set_element_from_pre_mutation_get(self):
        cv = contextvars.ContextVar("set_elem_before_set", default="old")

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            elem = cv.get()
            cv.set("new")
            return {elem}

        self.assertEqual(fn(), {"old"})
        self.assertEqual(cv.get(), "new")

    def test_return_dict_key_from_token_old_value_after_rebind(self):
        cv = contextvars.ContextVar("token_old_value_dict_key", default="old")
        cv.set("old")

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            token = cv.set("new")
            old_value = token.old_value
            cv.set("newer")
            return {old_value: 1}

        self.assertEqual(fn(), {"old": 1})
        self.assertEqual(cv.get(), "newer")

    def test_fullgraph_reset_external_token_errors(self):
        cv = contextvars.ContextVar("external_token", default="root")
        token = cv.set("tmp")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            cv.reset(token)
            return x

        x = torch.randn(4)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "ContextVar.reset\\(\\) on external token not supported",
        ):
            fn(x)

    def test_reset_used_external_token_does_not_replay_later_side_effects(self):
        cv = contextvars.ContextVar("used_token", default="root")
        token = cv.set("tmp")
        cv.reset(token)

        @torch.compile(backend="eager")
        def fn(x):
            try:
                cv.reset(token)
            except RuntimeError:
                return x
            x.add_(1)
            return x

        x = torch.tensor([1.0])
        out = fn(x)
        self.assertEqual(out, torch.tensor([1.0]))
        self.assertEqual(x, torch.tensor([1.0]))

    def test_reset_other_context_external_token_matches_try_except(self):
        cv = contextvars.ContextVar("other_ctx", default="root")
        token = contextvars.Context().run(cv.set, "tmp")

        @torch.compile(backend="eager")
        def fn(x):
            try:
                cv.reset(token)
            except ValueError:
                return x + 1
            return x + 2

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)

    def test_replay_side_effects_false_skips_contextvar_replay(self):
        cv = contextvars.ContextVar("no_replay_cv", default="root")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            cv.set("compiled")
            if cv.get() == "compiled":
                return x + 1
            return x + 2

        x = torch.randn(4)
        with torch._dynamo.config.patch(replay_side_effects=False):
            self.assertEqual(fn(x), x + 1)
        self.assertEqual(cv.get(), "root")

    def test_replay_side_effects_false_non_fullgraph_set_falls_back_eager(self):
        cv = contextvars.ContextVar("no_replay_graph_break", default="root")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            cv.set("compiled")
            print("graph break")
            if cv.get() == "compiled":
                return x + 1
            return x + 2

        x = torch.randn(4)
        with torch._dynamo.config.patch(replay_side_effects=False):
            self.assertEqual(fn(x), x + 1)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cv.get(), "compiled")

    def test_replay_side_effects_false_return_token_errors(self):
        cv = contextvars.ContextVar("no_replay_token", default="root")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x + 1, cv.set("compiled")

        x = torch.randn(4)
        with torch._dynamo.config.patch(replay_side_effects=False):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "ContextVar token escape requires side-effect replay",
            ):
                fn(x)

    def test_replay_side_effects_false_non_fullgraph_reset_falls_back_eager(self):
        cv = contextvars.ContextVar("no_replay_reset", default="root")
        token = cv.set("tmp")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            x = x + 1
            cv.reset(token)
            print("graph break")
            return x + (1 if cv.get() == "root" else 2)

        out = None
        with torch._dynamo.config.patch(replay_side_effects=False):
            out = fn(torch.tensor(0))
        self.assertEqual(out, torch.tensor(2))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cv.get(), "root")

    def test_preexisting_external_value_internal_set_reset(self):
        cv = contextvars.ContextVar("external_then_internal", default="root")
        cv.set("external")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            token = cv.set("internal")
            cv.reset(token)
            return x + (1 if cv.get() == "external" else 2)

        out = fn(torch.tensor(0))
        self.assertEqual(out, torch.tensor(1))
        self.assertEqual(cv.get(), "external")

    def test_set_then_graph_break_get_after_break(self):
        cv = contextvars.ContextVar("set_then_break", default="old")

        @torch.compile(backend="eager")
        def fn(x):
            cv.set("new")
            print("graph break")
            return x + (1 if cv.get() == "new" else 2)

        out = fn(torch.tensor(0))
        self.assertEqual(out, torch.tensor(1))
        self.assertEqual(cv.get(), "new")

    def test_reset_used_token_checks_runtime_error_before_wrong_context(self):
        cv1 = contextvars.ContextVar("cv1", default="root")
        cv2 = contextvars.ContextVar("cv2", default="root")
        token = cv1.set("tmp")
        cv1.reset(token)

        @torch.compile(backend="eager")
        def fn(x):
            try:
                cv2.reset(token)
            except RuntimeError:
                return x + 1
            except ValueError:
                return x + 2
            return x + 3

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)

    def test_non_fullgraph_reset_external_token_graph_breaks_then_succeeds(self):
        cv = contextvars.ContextVar("external_reset", default="root")
        token = cv.set("tmp")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            x = x + 1
            cv.reset(token)
            return x + 2

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 3)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cv.get(), "root")

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

    def test_nested_set_reset_order(self):
        cv = contextvars.ContextVar("nested_cv", default="root")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            token1 = cv.set("a")
            token2 = cv.set("b")
            if cv.get() == "b":
                x = x + 1
            cv.reset(token2)
            if cv.get() == "a":
                x = x + 2
            cv.reset(token1)
            if cv.get() == "root":
                x = x + 4
            return x

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 7)
        self.assertEqual(cv.get(), "root")

    def test_non_lifo_reset_order(self):
        cv = contextvars.ContextVar("non_lifo_cv", default="root")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            token1 = cv.set("a")
            token2 = cv.set("b")
            cv.reset(token1)
            if cv.get() == "root":
                x = x + 1
            cv.reset(token2)
            if cv.get() == "a":
                x = x + 2
            return x

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 3)
        self.assertEqual(cv.get(), "a")

    def test_return_old_value_after_set(self):
        cv = contextvars.ContextVar("old_value", default="orig")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            old = cv.get()
            cv.set("new")
            return x + 1, old

        x = torch.randn(4)
        out, old = fn(x)
        self.assertEqual(out, x + 1)
        self.assertEqual(old, "orig")
        self.assertEqual(cv.get(), "new")

    def test_return_old_value_twice_after_set(self):
        cv = contextvars.ContextVar("old_value_twice", default="orig")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            old = cv.get()
            cv.set("new")
            return x + 1, old, old

        x = torch.randn(4)
        out, old1, old2 = fn(x)
        self.assertEqual(out, x + 1)
        self.assertEqual(old1, "orig")
        self.assertEqual(old2, "orig")
        self.assertEqual(cv.get(), "new")

    def test_return_old_value_after_set_across_graph_break(self):
        cv = contextvars.ContextVar("old_value_gb", default="orig")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            old = cv.get()
            cv.set("new")
            print("graph break")
            return x + 1, old

        x = torch.randn(4)
        out, old = fn(x)
        self.assertEqual(out, x + 1)
        self.assertEqual(old, "orig")
        self.assertEqual(cv.get(), "new")

    def test_return_derived_old_value_after_set_across_graph_break(self):
        cv = contextvars.ContextVar("derived_old_value")

        class Box:
            def __init__(self, tag):
                self.tag = tag

        cv.set(Box("old"))

        @torch.compile(backend="eager")
        def fn(x):
            old_tag = cv.get().tag
            cv.set(Box("new"))
            print("graph break")
            return x + (1 if old_tag == "old" else 2)

        out = fn(torch.tensor(0))
        self.assertEqual(out, torch.tensor(1))
        self.assertEqual(cv.get().tag, "new")

    def test_contextvar_mutation_with_allow_in_graph_stale_read(self):
        # Opaque runtime callables observe pre-mutation cv state (deferred replay).
        # Same staleness semantics as globals and random state.
        cv = contextvars.ContextVar("allow_in_graph_cv", default="old")

        @torch.compiler.allow_in_graph
        def read_cv():
            return torch.tensor(1) if cv.get() == "new" else torch.tensor(2)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            cv.set("new")
            return x + read_cv()

        self.assertEqual(fn(torch.tensor(0)), torch.tensor(2))

    def test_contextvar_mutation_with_nonstrict_trace_stale_read(self):
        cv = contextvars.ContextVar("nonstrict_trace_cv", default="old")

        @torch.compiler.nonstrict_trace
        def read_cv():
            return torch.tensor(1) if cv.get() == "new" else torch.tensor(2)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            token = cv.set("new")
            try:
                return x + read_cv()
            finally:
                cv.reset(token)

        self.assertEqual(fn(torch.tensor(0)), torch.tensor(2))

    def test_contextvar_mutation_with_leaf_function_stale_read(self):
        cv = contextvars.ContextVar("leaf_function_cv", default="old")

        from torch._dynamo.decorators import leaf_function

        @leaf_function
        def read_cv(x):
            return x + (1 if cv.get() == "new" else 2)

        @read_cv.register_fake
        def fake_read_cv(x):
            return x + 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            token = cv.set("new")
            try:
                return read_cv(x)
            finally:
                cv.reset(token)

        self.assertEqual(fn(torch.tensor(0)), torch.tensor(2))

    def test_mutated_original_object_survives_rebind(self):
        cv = contextvars.ContextVar("mutated_old_object", default=None)
        old = set()
        cv.set(old)

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            items = cv.get()
            items.add("old")
            cv.set(set())

        fn()
        current = cv.get()
        self.assertEqual(old, {"old"})
        self.assertEqual(current, set())
        self.assertIsNot(current, old)

    def test_rebind_to_original_object_after_temporary_set(self):
        cv = contextvars.ContextVar("restore_old_object", default=None)
        old = {"start"}
        cv.set(old)

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            items = cv.get()
            cv.set("tmp")
            cv.set(items)

        fn()
        self.assertIs(cv.get(), old)
        self.assertEqual(cv.get(), {"start"})

    def test_set_new_object_with_mutation(self):
        cv = contextvars.ContextVar("new_object_cv", default=None)

        class Box:
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            obj = Box()
            obj.value = "ok"
            cv.set(obj)
            return x + 1, obj

        x = torch.randn(4)
        result, obj = fn(x)
        self.assertEqual(result, x + 1)
        self.assertIs(cv.get(), obj)
        self.assertIsInstance(obj, Box)
        self.assertEqual(obj.value, "ok")

    def test_return_token_from_compiled_function(self):
        cv = contextvars.ContextVar("return_token", default="base")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            token = cv.set("compiled")
            return x + 1, token

        x = torch.randn(4)
        result, token = fn(x)
        try:
            self.assertEqual(result, x + 1)
            self.assertEqual(cv.get(), "compiled")
        finally:
            cv.reset(token)
        self.assertEqual(cv.get(), "base")

    def test_token_var_and_old_value_attributes(self):
        cv = contextvars.ContextVar("token_attrs")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            t = cv.set("x")
            return t.var is cv, t.old_value is contextvars.Token.MISSING, x + 1

        self.assertEqual(fn(torch.tensor(1)), (True, True, torch.tensor(2)))

    def test_token_old_value_missing_return(self):
        cv = contextvars.ContextVar("token_old_value_missing_return")

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            tok = cv.set("inner")
            return tok.old_value

        out = fn()
        self.assertIs(out, contextvars.Token.MISSING)

    def test_token_old_value_explicit_token_missing_binding(self):
        cv = contextvars.ContextVar(
            "token_old_value_explicit_missing", default="fallback"
        )
        cv.set(contextvars.Token.MISSING)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            tok = cv.set("inner")
            cv.reset(tok)
            return x + (1 if cv.get() is contextvars.Token.MISSING else 2)

        self.assertEqual(fn(torch.tensor(0)), torch.tensor(1))
        self.assertIs(cv.get(), contextvars.Token.MISSING)

    def test_token_old_value_recompiles_on_external_change(self):
        cv = contextvars.ContextVar("token_old_value_cache")
        cv.set(1)
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            tok = cv.set(10)
            return x + tok.old_value

        self.assertEqual(fn(torch.tensor(0)), torch.tensor(1))
        cv.set(2)
        self.assertEqual(fn(torch.tensor(0)), torch.tensor(2))
        self.assertEqual(cnt.frame_count, 2)

    def test_token_missing_old_value_recompiles_on_external_change(self):
        cv = contextvars.ContextVar("token_missing_cache")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            tok = cv.set("inner")
            return x + (1 if tok.old_value is contextvars.Token.MISSING else 2)

        self.assertEqual(fn(torch.tensor(0)), torch.tensor(1))
        cv.set("outer")
        self.assertEqual(fn(torch.tensor(0)), torch.tensor(2))
        self.assertEqual(cnt.frame_count, 2)

    def test_token_old_value_mutation_aliases_preexisting_object(self):
        cv = contextvars.ContextVar("token_old_value_alias")
        old = []
        cv.set(old)

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            tok = cv.set("tmp")
            tok.old_value.append("x")
            cv.reset(tok)

        fn()
        self.assertEqual(old, ["x"])
        self.assertIs(cv.get(), old)
        self.assertEqual(cv.get(), ["x"])

    def test_token_old_value_mutable_return_alias(self):
        cv = contextvars.ContextVar("token_old_value_mutable_return")
        old = ["a"]
        cv.set(old)

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            tok = cv.set("inner")
            return tok.old_value

        out = fn()
        self.assertIs(out, old)
        self.assertEqual(out, ["a"])

    def test_token_old_value_tensor_value(self):
        cv = contextvars.ContextVar("token_old_value_tensor")
        cv.set(torch.tensor(3))

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            tok = cv.set(torch.tensor(4))
            return x + tok.old_value

        self.assertEqual(fn(torch.tensor(1)), torch.tensor(4))

    def test_token_from_other_compiled_region_fullgraph_errors(self):
        cv = contextvars.ContextVar("cross_region_token", default="root")

        @torch.compile(backend="eager", fullgraph=True)
        def set_fn(x):
            return x + 1, cv.set("compiled")

        @torch.compile(backend="eager", fullgraph=True)
        def reset_fn(token, x):
            cv.reset(token)
            return x + 2

        x = torch.randn(4)
        _, token = set_fn(x)
        self.assertEqual(cv.get(), "compiled")
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "ContextVar.reset\\(\\) on external token not supported",
        ):
            reset_fn(token, x)

    def test_contextvar_set_in_hop_body_errors(self):
        cv = contextvars.ContextVar("hop_test", default="original")

        def true_fn(x):
            cv.set("mutated")
            return x + 1

        def false_fn(x):
            return x - 1

        @torch.compile(backend="eager")
        def fn(x):
            return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        x = torch.randn(4).abs() + 1
        with self.assertRaises(torch._dynamo.exc.UncapturedHigherOrderOpError):
            fn(x)

    def test_set_graph_break_reset_internal_token(self):
        cv = contextvars.ContextVar("cross_break", default="root")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            token = cv.set("new")
            x = x + 1
            torch._dynamo.graph_break()
            cv.reset(token)
            if cv.get() == "root":
                return x + 10
            return x + 20

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 11)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cv.get(), "root")

    def test_multiple_contextvars_interleaved_set_reset(self):
        cv1 = contextvars.ContextVar("cv1", default="a")
        cv2 = contextvars.ContextVar("cv2", default="b")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            t1 = cv1.set("x")
            t2 = cv2.set("y")
            if cv1.get() == "x" and cv2.get() == "y":
                x = x + 1
            cv1.reset(t1)
            cv2.reset(t2)
            if cv1.get() == "a" and cv2.get() == "b":
                x = x + 2
            return x

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 3)
        self.assertEqual(cv1.get(), "a")
        self.assertEqual(cv2.get(), "b")

    def test_set_overrides_explicit_get_default(self):
        cv = contextvars.ContextVar("override_default", default="cd")
        cv.set("active")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x + (1 if cv.get("ignored") == "active" else 2)

        self.assertEqual(fn(torch.tensor(0)), torch.tensor(1))

    def test_chained_set_old_value(self):
        cv = contextvars.ContextVar("chained", default="root")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            cv.set("first")
            t2 = cv.set("second")
            return x + (1 if t2.old_value == "first" else 2)

        self.assertEqual(fn(torch.tensor(0)), torch.tensor(1))

    def test_reset_to_unbound_raises_lookup_error(self):
        cv = contextvars.ContextVar("unbound_after_reset")

        @torch.compile(backend="eager")
        def fn(x):
            t = cv.set("tmp")
            cv.reset(t)
            try:
                cv.get()
                return x + 2
            except LookupError:
                return x + 1

        self.assertEqual(fn(torch.tensor(0)), torch.tensor(1))

    def test_internal_token_reuse_raises_runtime_error(self):
        cv = contextvars.ContextVar("reuse_internal", default="root")

        @torch.compile(backend="eager")
        def fn(x):
            t = cv.set("tmp")
            cv.reset(t)
            try:
                cv.reset(t)
                return x + 2
            except RuntimeError:
                return x + 1

        self.assertEqual(fn(torch.tensor(0)), torch.tensor(1))

    def test_old_value_after_set_reset_set_unset_state(self):
        cv = contextvars.ContextVar("reset_then_set", default="root")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            t1 = cv.set("a")
            cv.reset(t1)
            t2 = cv.set("b")
            return x + (1 if t2.old_value is contextvars.Token.MISSING else 2)

        self.assertEqual(fn(torch.tensor(0)), torch.tensor(1))

    def test_reset_eager_apply_restores_real_cv(self):
        cv = contextvars.ContextVar("eager_reset_cv")
        cv.set("original")
        fake_reads: list = []

        with torch.library._scoped_library("_test_eager_reset", "DEF") as lib:
            lib.define("read_cv(Tensor x) -> Tensor")

            @torch.library.register_kernel("_test_eager_reset::read_cv", "cpu", lib=lib)
            def _(x):
                return x.clone()

            @torch.library.register_fake("_test_eager_reset::read_cv", lib=lib)
            def _(x):
                fake_reads.append(cv.get())
                return torch.empty_like(x)

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                token = cv.set("modified")
                cv.reset(token)
                return torch.ops._test_eager_reset.read_cv(x)

            fn(torch.tensor(10))
            self.assertEqual(fake_reads[0], "original")
            self.assertEqual(cv.get(), "original")


if __name__ == "__main__":
    run_tests()
