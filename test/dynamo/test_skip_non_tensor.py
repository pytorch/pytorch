# Owner(s): ["module: dynamo"]
from unittest.mock import patch

import torch

import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounter

_variable = 0
_variable_2 = 0


def user_function():
    return torch._utils.is_compiling()


def user_generator():
    for _ in range(1):
        yield torch._utils.is_compiling()
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
            if torch._utils.is_compiling():
                _variable += 1
            else:
                _variable_2 += 1
        elif self.mode == 2:
            if user_function():
                _variable += 1
        elif self.mode == 3:
            lambda_f = lambda: torch._utils.is_compiling()  # noqa: E731
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

        assert counter.op_count == 1

    def test_add_tensor2(self):
        def fn(a, b):
            return torch.add(a, b)

        counter = CompileCounter()

        x = torch.randn(4)
        y = 5
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        opt_fn(x, y)

        assert counter.op_count == 1

    def test_add_tensor_list(self):
        def fn(lst):
            return lst[0] + lst[1]

        counter = CompileCounter()
        x = torch.randn(4)
        y = 5
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        opt_fn([x, y])

        assert counter.op_count == 1

    def test_add_tensor_dict(self):
        def fn(dt):
            return dt["a"] + dt["b"]

        counter = CompileCounter()
        x = torch.randn(4)
        y = 5
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        opt_fn({"a": x, "b": y})

        assert counter.op_count == 1

    def test_add_skip(self):
        def fn(a, b):
            return a + b

        counter = CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        x = 4
        y = 5
        opt_fn(x, y)

        assert counter.op_count == 0

    @patch.object(torch._dynamo.config, "raise_on_ctx_manager_usage", False)
    def test_recursive_list(self):
        def fn(x):
            return x

        counter = CompileCounter()

        x = []
        x.append(x)
        with torch._dynamo.optimize_assert(counter):
            fn(x)

        assert counter.op_count == 0

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

        assert counter.op_count == 0

    def test_do_not_skip_side_effects(self):
        # https://github.com/pytorch/pytorch/issues/110765

        # By invoking torch._utils.is_compiling(),
        # there may be side-effects inconsistent with eager when
        # compiling. Thus we force dynamo to commit the graph,
        # even if it does not perform any tensor operation
        global _variable, _variable_2

        for mode in range(1, 7):
            torch._dynamo.reset()

            _variable = 0
            _variable_2 = 0

            mod = MyModule(mode=mode)
            model = torch._dynamo.optimize(backend="eager", nopython=mode != 6)(mod)
            assert _variable == 0
            assert _variable_2 == 0

            model(torch.tensor([1]))
            assert _variable == 1
            assert _variable_2 == 0

            model(torch.tensor([1]))
            assert _variable == 2
            assert _variable_2 == 0


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
