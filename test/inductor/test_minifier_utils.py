# Owner(s): ["module: inductor"]
import torch
from torch._dynamo.repro.aoti import (
    AOTIMinifierError,
    export_for_aoti_minifier,
    get_module_string,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class MinifierUtilsTests(TestCase):
    def test_invalid_output(self):
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                # return a graph module
                return self.linear

        model = SimpleModel()
        # Here we obtained a graph with invalid output by symbolic_trace for simplicity,
        # it can also obtained from running functorch.compile.minifier on an exported graph.
        traced = torch.fx.symbolic_trace(model)
        for strict in [True, False]:
            gm = export_for_aoti_minifier(traced, (torch.randn(2, 2),), strict=strict)
            self.assertTrue(gm is None)

    def test_non_exportable(self):
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x.sum()

        model = SimpleModel()
        # Force export failure by providing an input with in-compatible shapes
        inputs = (torch.randn(2), torch.randn(2))
        for strict in [True, False]:
            gm = export_for_aoti_minifier(
                model, inputs, strict=strict, skip_export_error=True
            )
            print(gm)
            self.assertTrue(gm is None)

            with self.assertRaises(AOTIMinifierError):
                export_for_aoti_minifier(
                    model, inputs, strict=strict, skip_export_error=False
                )

    def test_convert_module_to_string(self):
        class M(torch.nn.Module):
            def forward(self, x, flag):
                flag = flag.item()

                def true_fn(x):
                    return x.clone()

                return torch.cond(flag > 0, true_fn, true_fn, [x])

        inputs = (
            torch.rand(28, 28),
            torch.tensor(1),
        )

        model = M()
        gm = torch.export.export(model, inputs, strict=False).module()

        # TODO: make NNModuleToString.convert() generate string for nested submodules.
        model_string = get_module_string(gm)
        self.assertExpectedInline(
            model_string.strip(),
            """\
# from torch.nn import *
# class Repro(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.true_graph_0 = <lambda>()
#         self.false_graph_0 = <lambda>()



#     def forward(self, x, flag):
#         x, flag, = fx_pytree.tree_flatten_spec(([x, flag], {}), self._in_spec)
#         item = torch.ops.aten.item.default(flag);  flag = None
#         gt = item > 0;  item = None
#         true_graph_0 = self.true_graph_0
#         false_graph_0 = self.false_graph_0
#         cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, [x]);  gt = true_graph_0 = false_graph_0 = x = None
#         getitem = cond[0];  cond = None
#         return pytree.tree_unflatten((getitem,), self._out_spec)""",
        )


if __name__ == "__main__":
    run_tests()
