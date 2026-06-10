# Owner(s): ["module: dynamo"]
import warnings
from types import ModuleType, SimpleNamespace
from typing_extensions import ParamSpec
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.eval_frame import _debug_get_cache_entry_list, reset_code
from torch._dynamo.testing import CompileCounter
from torch._dynamo.utils import CleanupHook, CleanupManager, counters
from torch.testing._internal.common_utils import AlwaysWarnTypedStorageRemoval


_variable = 0
_variable_2 = 0
_condition_dependent_skip_flag = False
_P = ParamSpec("_P")
_paramspec_module = ModuleType("_paramspec_module")
_paramspec_module._P = ParamSpec("_paramspec_module._P")  # type: ignore[attr-defined]


def user_function():
    return torch.compiler.is_compiling()


def user_generator():
    for _ in range(1):
        yield torch.compiler.is_compiling()
    return


class MyModule(torch.nn.Module):
    def __init__(self, mode: int):
        super().__init__()
        self.mode = mode
        self.register_forward_pre_hook(self.pre_forward, with_kwargs=True)

    def pre_forward(self, module, args, kwargs):
        if self.mode == 5:
            if user_function():
                global _variable
                _variable += 1
        return args, kwargs

    def forward(self, x):
        global _variable, _variable_2

        if self.mode == 1:
            if torch.compiler.is_compiling():
                _variable += 1
            else:
                _variable_2 += 1
        elif self.mode == 2:
            if user_function():
                _variable += 1
        elif self.mode == 3:
            lambda_f = lambda: torch.compiler.is_compiling()  # noqa: E731
            if lambda_f():
                _variable += 1
        elif self.mode == 4:
            for cond in user_generator():
                if cond:
                    _variable += 1
        elif self.mode == 5:
            x += 1
        elif self.mode == 6:
            if user_function():
                torch._dynamo.graph_break()
                _variable += 1
        return x


class SkipNonTensorTests(torch._dynamo.test_case.TestCase):
    def test_add_tensor1(self):
        def fn(a, b):
            return a + b

        counter = CompileCounter()
        x = torch.randn(4)
        y = 5
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        opt_fn(x, y)

        if counter.op_count != 1:
            raise AssertionError(f"Expected op_count 1, got {counter.op_count}")

    def test_add_tensor2(self):
        def fn(a, b):
            return torch.add(a, b)

        counter = CompileCounter()

        x = torch.randn(4)
        y = 5
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        opt_fn(x, y)

        if counter.op_count != 1:
            raise AssertionError(f"Expected op_count 1, got {counter.op_count}")

    def test_add_tensor_list(self):
        def fn(lst):
            return lst[0] + lst[1]

        counter = CompileCounter()
        x = torch.randn(4)
        y = 5
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        opt_fn([x, y])

        if counter.op_count != 1:
            raise AssertionError(f"Expected op_count 1, got {counter.op_count}")

    def test_add_tensor_dict(self):
        def fn(dt):
            return dt["a"] + dt["b"]

        counter = CompileCounter()
        x = torch.randn(4)
        y = 5
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        opt_fn({"a": x, "b": y})

        if counter.op_count != 1:
            raise AssertionError(f"Expected op_count 1, got {counter.op_count}")

    def test_add_skip(self):
        def fn(a, b):
            return a + b

        counter = CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        x = 4
        y = 5
        opt_fn(x, y)

        if counter.op_count != 0:
            raise AssertionError(f"Expected op_count 0, got {counter.op_count}")

    def test_condition_dependent_graph_break_skip_does_not_poison_code(self):
        def fn(x, n):
            if n == 0:
                try:
                    torch._dynamo.graph_break()
                finally:
                    pass
            if torch.compiler.is_compiling():
                return x + 1
            return x - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)
        x = torch.ones(3)
        preexisting_builtins_keys = {
            name for name in fn.__globals__ if name.startswith("__builtins_dict___")
        }

        self.assertEqual(opt_fn(x, 0), x - 1)
        self.assertEqual(counter.frame_count, 0)
        entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(entries), 1)
        self.assertIsNot(entries[0].code, fn.__code__)
        self.assertTrue(
            entries[0].trace_annotation.startswith("Torch-Compiled Eager Fallback")
        )

        self.assertEqual(opt_fn(x, 0), x - 1)
        self.assertEqual(counter.frame_count, 0)
        self.assertEqual(len(_debug_get_cache_entry_list(fn.__code__)), 1)

        self.assertEqual(opt_fn(x, 1), x + 1)
        self.assertEqual(counter.frame_count, 1)
        entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(entries), 2)
        self.assertTrue(any(entry.code is not fn.__code__ for entry in entries))
        self.assertTrue(
            any(
                entry.trace_annotation.startswith("Torch-Compiled Eager Fallback")
                for entry in entries
            )
        )

        self.assertEqual(opt_fn(x, 0), x - 1)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(len(_debug_get_cache_entry_list(fn.__code__)), 2)

        new_builtins_keys = {
            name for name in fn.__globals__ if name.startswith("__builtins_dict___")
        } - preexisting_builtins_keys
        self.assertTrue(new_builtins_keys)
        del entries
        torch._dynamo.reset()
        self.assertEqual(
            {name for name in fn.__globals__ if name.startswith("__builtins_dict___")},
            preexisting_builtins_keys,
        )

    def test_condition_dependent_empty_graph_skip_does_not_poison_code(self):
        def fn(x, flag):
            if flag:
                if torch.compiler.is_compiling():
                    return x + 1
                return x - 1
            return x

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)
        x = torch.ones(3)

        self.assertEqual(opt_fn(x, False), x)
        self.assertEqual(counter.frame_count, 0)
        entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(entries), 1)
        self.assertIsNot(entries[0].code, fn.__code__)
        self.assertTrue(
            entries[0].trace_annotation.startswith("Torch-Compiled Eager Fallback")
        )

        self.assertEqual(opt_fn(x, True), x + 1)
        self.assertEqual(counter.frame_count, 1)
        entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(entries), 2)
        self.assertTrue(any(entry.code is not fn.__code__ for entry in entries))
        self.assertTrue(
            any(
                entry.trace_annotation.startswith("Torch-Compiled Eager Fallback")
                for entry in entries
            )
        )

        self.assertEqual(opt_fn(x, False), x)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(len(_debug_get_cache_entry_list(fn.__code__)), 2)

    def test_condition_dependent_skip_reset_code_cleans_globals(self):
        def fn(x, n):
            if n == 0:
                try:
                    torch._dynamo.graph_break()
                finally:
                    pass
            if torch.compiler.is_compiling():
                return x + 1
            return x + 1

        opt_fn = torch.compile(fn, backend="eager", dynamic=False)
        x = torch.ones(3)
        preexisting_builtins_keys = {
            name for name in fn.__globals__ if name.startswith("__builtins_dict___")
        }

        self.assertEqual(opt_fn(x, 0), x + 1)
        new_builtins_keys = {
            name for name in fn.__globals__ if name.startswith("__builtins_dict___")
        } - preexisting_builtins_keys
        self.assertTrue(new_builtins_keys)

        reset_code(fn.__code__)
        self.assertEqual(
            {name for name in fn.__globals__ if name.startswith("__builtins_dict___")},
            preexisting_builtins_keys,
        )

    def test_condition_dependent_skip_cleanup_hooks_are_idempotent(self):
        def fn():
            pass

        scope = {}
        cleanup_count = CleanupManager.count
        hook = CleanupHook.create(scope, "__tmp_cleanup", object())
        self.assertEqual(CleanupManager.count, cleanup_count + 1)

        key = fn.__code__.replace(co_varnames=fn.__code__.co_varnames)
        CleanupManager.instance[key] = [hook, hook]
        CleanupManager.instance.cleanup(key)
        CleanupManager.instance.cleanup(key)

        self.assertNotIn("__tmp_cleanup", scope)
        self.assertEqual(CleanupManager.count, cleanup_count)

    def test_reset_inside_compiled_frame_preserves_resume_globals(self):
        def fn(x):
            torch._dynamo.reset()
            return x + 1

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.ones(3)

        self.assertEqual(opt_fn(x), x + 1)
        self.assertEqual(opt_fn(x), x + 1)

    def test_explicit_skip_frame_cleans_failed_trace_globals(self):
        def fn(x):
            try:
                torch._dynamo.skip_frame()
            finally:
                pass
            return x + 1

        opt_fn = torch.compile(fn, backend="eager", dynamic=False)
        x = torch.ones(3)
        preexisting_builtins_keys = {
            name for name in fn.__globals__ if name.startswith("__builtins_dict___")
        }

        self.assertEqual(opt_fn(x), x + 1)
        self.assertEqual(
            {name for name in fn.__globals__ if name.startswith("__builtins_dict___")},
            preexisting_builtins_keys,
        )

    def test_condition_dependent_skip_paramspec_guard_detection(self):
        from torch._dynamo.convert_frame import _guard_targets_paramspec_attr

        def fail_getattr(name):
            raise AssertionError("module __getattr__ should not run")

        lazy_module = ModuleType("lazy_module")
        lazy_module.__getattr__ = fail_getattr  # type: ignore[attr-defined]

        output = SimpleNamespace(
            global_scope={
                "_P": _P,
                "_paramspec_module": _paramspec_module,
                "lazy_module": lazy_module,
            },
            local_scope={},
        )

        self.assertTrue(_guard_targets_paramspec_attr("G['_P'].args", output))
        self.assertTrue(
            _guard_targets_paramspec_attr("G['_paramspec_module']._P.args", output)
        )
        self.assertFalse(
            _guard_targets_paramspec_attr("G['lazy_module'].missing.args", output)
        )

    def test_condition_dependent_skip_with_global_tensor_factory(self):
        def fn(n):
            if n == 0:
                try:
                    torch._dynamo.graph_break()
                finally:
                    pass
                return torch.ones(3) - 1
            if torch.compiler.is_compiling():
                return torch.ones(3) + 1
            return torch.ones(3) - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)

        self.assertEqual(opt_fn(0), torch.zeros(3))
        self.assertEqual(counter.frame_count, 0)

        self.assertEqual(opt_fn(1), torch.ones(3) + 1)
        self.assertEqual(counter.frame_count, 1)

    def test_condition_dependent_skip_with_other_torch_factory(self):
        def fn(n):
            if n == 0:
                try:
                    torch._dynamo.graph_break()
                finally:
                    pass
                return torch.empty_strided((3,), (1,)) - 1
            if torch.compiler.is_compiling():
                return torch.empty_strided((3,), (1,)) + 1
            return torch.empty_strided((3,), (1,)) - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)

        self.assertEqual(opt_fn(0).shape, (3,))
        self.assertEqual(counter.frame_count, 0)

        self.assertEqual(opt_fn(1).shape, (3,))
        self.assertEqual(counter.frame_count, 1)

    def test_condition_dependent_skip_with_tensor_constructor(self):
        def fn(n):
            if n == 0:
                try:
                    torch._dynamo.graph_break()
                finally:
                    pass
                return torch.Tensor([1.0]) - 1
            if torch.compiler.is_compiling():
                return torch.Tensor([1.0]) + 1
            return torch.Tensor([1.0]) - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)

        self.assertEqual(opt_fn(0), torch.zeros(1))
        self.assertEqual(counter.frame_count, 0)

        self.assertEqual(opt_fn(1), torch.full((1,), 2.0))
        self.assertEqual(counter.frame_count, 1)

    def test_condition_dependent_skip_with_legacy_tensor_constructor(self):
        def fn(n):
            if n == 0:
                try:
                    torch._dynamo.graph_break()
                finally:
                    pass
                return torch.FloatTensor([1.0]) - 1
            if torch.compiler.is_compiling():
                return torch.FloatTensor([1.0]) + 1
            return torch.FloatTensor([1.0]) - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)

        self.assertEqual(opt_fn(0), torch.zeros(1))
        self.assertEqual(counter.frame_count, 0)

        self.assertEqual(opt_fn(1), torch.full((1,), 2.0))
        self.assertEqual(counter.frame_count, 1)

    def test_condition_dependent_skip_with_direct_torch_import(self):
        from torch import ones
        from torch._dynamo import graph_break
        from torch.compiler import is_compiling

        def fn(n):
            if n == 0:
                try:
                    graph_break()
                finally:
                    pass
                return ones(3) - 1
            if is_compiling():
                return ones(3) + 1
            return ones(3) - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)

        self.assertEqual(opt_fn(0), torch.zeros(3))
        self.assertEqual(counter.frame_count, 0)

        self.assertEqual(opt_fn(1), torch.ones(3) + 1)
        self.assertEqual(counter.frame_count, 1)

    def test_condition_dependent_skip_ignores_storage_factory(self):
        def fn(n):
            if n == 0:
                try:
                    torch._dynamo.graph_break()
                finally:
                    pass
            return torch.FloatStorage(1)

        opt_fn = torch.compile(fn, backend="eager", dynamic=False)

        with AlwaysWarnTypedStorageRemoval(True):
            with warnings.catch_warnings(record=True) as w:
                warnings.resetwarnings()
                opt_fn(0)
                self.assertEqual(len(w), 1, msg=str([str(a) for a in w]))
                self.assertIn("TypedStorage is deprecated", str(w[0].message))

    def test_condition_dependent_skip_with_sequence_length_guard(self):
        def fn(xs):
            if len(xs) == 0:
                try:
                    torch._dynamo.graph_break()
                finally:
                    pass
                return torch.zeros(3)
            if torch.compiler.is_compiling():
                return xs[0] + 1
            return xs[0] - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)
        x = torch.ones(3)

        self.assertEqual(opt_fn([]), torch.zeros(3))
        self.assertEqual(counter.frame_count, 0)

        self.assertEqual(opt_fn([x]), x + 1)
        self.assertEqual(counter.frame_count, 1)

    def test_condition_dependent_skip_with_global_guard(self):
        global _condition_dependent_skip_flag

        def fn(x):
            if _condition_dependent_skip_flag:
                try:
                    torch._dynamo.graph_break()
                finally:
                    pass
            if torch.compiler.is_compiling():
                return x + 1
            return x - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)
        x = torch.ones(3)

        try:
            _condition_dependent_skip_flag = True
            self.assertEqual(opt_fn(x), x - 1)
            self.assertEqual(counter.frame_count, 0)

            _condition_dependent_skip_flag = False
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(counter.frame_count, 1)
        finally:
            _condition_dependent_skip_flag = False

    def test_condition_dependent_skip_with_tensor_shape_guard(self):
        def fn(x):
            if x.shape[0] == 3:
                try:
                    torch._dynamo.graph_break()
                finally:
                    pass
            if torch.compiler.is_compiling():
                return x + 1
            return x - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)

        x3 = torch.ones(3)
        x4 = torch.ones(4)
        self.assertEqual(opt_fn(x3), x3 - 1)
        self.assertEqual(counter.frame_count, 0)

        self.assertEqual(opt_fn(x4), x4 + 1)
        self.assertEqual(counter.frame_count, 1)

    def test_condition_dependent_skip_with_tensor_len_guard(self):
        def fn(x):
            if len(x) == 3:
                try:
                    torch._dynamo.graph_break()
                finally:
                    pass
            if torch.compiler.is_compiling():
                return x + 1
            return x - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)

        x3 = torch.ones(3)
        x4 = torch.ones(4)
        self.assertEqual(opt_fn(x3), x3 - 1)
        self.assertEqual(counter.frame_count, 0)

        self.assertEqual(opt_fn(x4), x4 + 1)
        self.assertEqual(counter.frame_count, 1)

    def test_condition_dependent_skip_with_tensor_state_guard(self):
        def fn(x):
            if x.requires_grad:
                try:
                    torch._dynamo.graph_break()
                finally:
                    pass
            if torch.compiler.is_compiling():
                return x + 1
            return x - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)

        x_grad = torch.ones(3, requires_grad=True)
        x_no_grad = torch.ones(3)
        self.assertEqual(opt_fn(x_grad), x_grad - 1)
        self.assertEqual(counter.frame_count, 0)

        self.assertEqual(opt_fn(x_no_grad), x_no_grad + 1)
        self.assertEqual(counter.frame_count, 1)

    def test_condition_dependent_skip_with_tensor_predicate_guard(self):
        def fn(x):
            if x.is_complex():
                try:
                    torch._dynamo.graph_break()
                finally:
                    pass
                return x.real - 1
            if torch.compiler.is_compiling():
                return x + 1
            return x - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)

        x_complex = torch.ones(3, dtype=torch.complex64)
        x_float = torch.ones(3)
        self.assertEqual(opt_fn(x_complex), x_complex.real - 1)
        self.assertEqual(counter.frame_count, 0)

        self.assertEqual(opt_fn(x_float), x_float + 1)
        self.assertEqual(counter.frame_count, 1)

    def test_condition_dependent_graph_break_after_call_does_not_poison_code(self):
        def fn(x, n):
            y = x + 10
            if n == 0:
                torch._dynamo.graph_break()
            return y + 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=False)
        x = torch.ones(3)
        counters.clear()

        self.assertEqual(opt_fn(x, 0), x + 11)
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(sum(counters["graph_break"].values()), 1)

        self.assertEqual(opt_fn(x, 1), x + 11)
        self.assertEqual(counter.frame_count, 3)

    def test_nested_explicit_graph_break_counted_once(self):
        def inner(x):
            torch._dynamo.graph_break()
            return x + 1

        def outer(x):
            return inner(x + 1) + 1

        opt_fn = torch.compile(outer, backend="eager")
        counters.clear()

        self.assertEqual(opt_fn(torch.ones(2)), torch.ones(2) + 3)
        self.assertEqual(sum(counters["graph_break"].values()), 1)

    def test_backend_skip_frame_preserves_fullgraph_failure(self):
        def backend(gm, args):
            raise torch._dynamo.exc.SkipFrame("backend skip")

        def fn(x):
            return x + 1

        preexisting_builtins_keys = {
            name for name in fn.__globals__ if name.startswith("__builtins_dict___")
        }
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with self.assertRaisesRegex(
            RuntimeError, "torch.compile with fullgraph=True found no compiled frames"
        ):
            opt_fn(torch.ones(2))
        self.assertEqual(len(_debug_get_cache_entry_list(fn.__code__)), 0)
        self.assertEqual(
            {name for name in fn.__globals__ if name.startswith("__builtins_dict___")},
            preexisting_builtins_keys,
        )

    def test_backend_skip_frame_still_skips_code_object(self):
        backend_calls = 0

        def backend(gm, args):
            nonlocal backend_calls
            backend_calls += 1
            raise torch._dynamo.exc.SkipFrame("backend skip")

        def fn(x, n):
            if n == 0:
                return x + 1
            return x + 2

        opt_fn = torch.compile(fn, backend=backend, dynamic=False)
        x = torch.ones(2)

        self.assertEqual(opt_fn(x, 0), x + 1)
        self.assertEqual(backend_calls, 1)
        self.assertEqual(opt_fn(x, 1), x + 2)
        self.assertEqual(backend_calls, 1)

    @patch.object(torch._dynamo.config, "raise_on_ctx_manager_usage", False)
    def test_recursive_list(self):
        def fn(x):
            return x

        counter = CompileCounter()

        x = []
        x.append(x)
        with torch._dynamo.optimize_assert(counter):
            fn(x)

        if counter.op_count != 0:
            raise AssertionError(f"Expected op_count 0, got {counter.op_count}")

    @patch.object(torch._dynamo.config, "raise_on_ctx_manager_usage", False)
    def test_custom_list(self):
        def fn(x):
            return x[0] + x[1]

        counter = CompileCounter()

        class Foo(list):
            def __iter__(self):
                raise Exception  # noqa: TRY002

            def __len__(self):
                raise Exception  # noqa: TRY002

        x = Foo()
        x.append(torch.randn(4))
        x.append(torch.randn(4))
        with torch._dynamo.optimize_assert(counter):
            fn(x)

        if counter.op_count != 0:
            raise AssertionError(f"Expected op_count 0, got {counter.op_count}")

    def test_do_not_skip_side_effects(self):
        # https://github.com/pytorch/pytorch/issues/110765

        # By invoking torch.compiler.is_compiling(),
        # there may be side-effects inconsistent with eager when
        # compiling. Thus we force dynamo to commit the graph,
        # even if it does not perform any tensor operation
        global _variable, _variable_2

        for mode in range(1, 7):
            torch._dynamo.reset()

            _variable = 0
            _variable_2 = 0

            mod = MyModule(mode=mode)
            model = torch.compile(mod, backend="eager", fullgraph=mode != 6)
            if _variable != 0:
                raise AssertionError(f"Expected _variable 0, got {_variable}")
            if _variable_2 != 0:
                raise AssertionError(f"Expected _variable_2 0, got {_variable_2}")

            model(torch.tensor([1]))
            if _variable != 1:
                raise AssertionError(f"Expected _variable 1, got {_variable}")
            if _variable_2 != 0:
                raise AssertionError(f"Expected _variable_2 0, got {_variable_2}")

            model(torch.tensor([1]))
            if _variable != 2:
                raise AssertionError(f"Expected _variable 2, got {_variable}")
            if _variable_2 != 0:
                raise AssertionError(f"Expected _variable_2 0, got {_variable_2}")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
