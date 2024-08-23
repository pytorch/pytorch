# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo
import torch._dynamo.test_case


class GuardHooksTest(torch._dynamo.test_case.TestCase):
    def test_basic(self):
        from torch._dynamo.hooks import keep_tensor_guards

        def fn(x, y):
            return x + y

        opt_fn = torch.compile(
            fn, backend="eager", dynamo_options={"guard_filter_fn": keep_tensor_guards}
        )

        x = torch.randn(3)

        opt_fn(x, 1)

        with torch._dynamo.config.patch("error_on_recompile", True):
            opt_fn(x, 1)

    def test_dynamic_guards(self):
        from torch._dynamo.hooks import keep_tensor_guards, skip_all_symbolic_guards

        def fn(x, y):
            return x + y

        opt_fn = torch.compile(
            fn,
            backend="eager",
            dynamic=True,
            dynamo_options={
                "guard_filter_fn": keep_tensor_guards,
                "symbolic_guard_filter_fn": skip_all_symbolic_guards,
            },
        )

        x = torch.randn(3)
        y = torch.randn(3)

        opt_fn(x, y)

        # symbolic guards are skipped
        with torch._dynamo.config.patch("error_on_recompile", True):
            opt_fn(x, torch.randn(1))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
