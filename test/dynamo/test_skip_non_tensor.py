# Owner(s): ["module: dynamo"]
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
from torch._dynamo.testing import CompileCounter


_variable = 0
_variable_2 = 0


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

        self.assertTrue(
            {name for name in fn.__globals__ if name.startswith("__builtins_dict___")}
            - preexisting_builtins_keys
        )
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
