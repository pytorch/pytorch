# Owner(s): ["module: dynamo"]

from unittest import mock

import torch
import torch._dynamo
import torch._dynamo.test_case


class ExplainTests(torch._dynamo.test_case.TestCase):
    def test_dynamic_option(self):
        def fn(x):
            y = torch.zeros(x.shape[0], device=x.device)
            return x + y

        x = torch.randn(4)
        explain_output = torch._dynamo.explain(fn, dynamic=True)(x)

        self.assertEqual(explain_output.graph_count, 1)
        self.assertIn("torch.SymInt", explain_output.graphs[0].code)

    def test_other_optimize_kwargs_are_passed_to_optimize(self):
        def fn(x):
            return x + 1

        with mock.patch("torch._dynamo.eval_frame.optimize") as optimize_mock:
            optimize_mock.return_value.return_value = lambda *args, **kwargs: None
            torch._dynamo.explain(
                fn,
                isolate_recompiles=True,
                recompile_limit=1,
            )(torch.randn(4))

        self.assertEqual(optimize_mock.call_count, 1)
        self.assertEqual(
            optimize_mock.call_args.kwargs,
            {
                "nopython": False,
                "guard_export_fn": mock.ANY,
                "isolate_recompiles": True,
                "recompile_limit": 1,
            },
        )

    def test_shapes_spec_option_is_passed_to_optimize(self):
        def fn(x):
            return x + 1

        with self.assertRaisesRegex(
            ValueError,
            "`dynamic` and `shapes_spec` cannot both be set",
        ):
            torch._dynamo.explain(
                fn,
                dynamic=True,
                shapes_spec={},
            )(torch.randn(4))

    def test_deprecated_direct_invocation_still_accepts_model_args(self):
        def fn(x, scale=1):
            return x * scale

        x = torch.randn(4)
        with self.assertWarnsRegex(FutureWarning, "deprecated"):
            explain_output = torch._dynamo.explain(fn, x, scale=3)

        self.assertEqual(explain_output.graph_count, 1)

    def test_deprecated_kwarg_only_invocation_still_accepts_model_args(self):
        def fn(x, scale=1):
            return x * scale

        with self.assertWarnsRegex(FutureWarning, "deprecated"):
            explain_output = torch._dynamo.explain(fn, x=torch.randn(4), scale=3)

        self.assertEqual(explain_output.graph_count, 1)

    def test_model_kwarg_named_dynamic_uses_inner_call(self):
        def fn(x, dynamic=False):
            if dynamic:
                return x + 1
            return x - 1

        explain_output = torch._dynamo.explain(fn)(torch.randn(4), dynamic=True)

        self.assertEqual(explain_output.graph_count, 1)

    def test_ambiguous_outer_kwargs_raise(self):
        def fn(x):
            return x + 1

        with self.assertRaisesRegex(
            TypeError,
            "received both optimization kwargs .*dynamic.* and function kwargs .*x",
        ):
            torch._dynamo.explain(fn, dynamic=True, x=torch.randn(4))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
