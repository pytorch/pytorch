# Owner(s): ["oncall: export"]
import unittest

import torch
from torch._dynamo.eval_frame import is_dynamo_supported
from torch.export import Dim
from torch.export.experimental import _ExportPackage
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not is_dynamo_supported(), "dynamo isn't supported")
class TestPackage(TestCase):
    def test_basic(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        self.assertEqual(
            package._exporter("fn", fn)(x),
            fn(x),
        )
        self.assertEqual(len(package.methods), 1)
        self.assertEqual(len(package.methods["fn"].fallbacks), 1)
        self.assertEqual(len(package.methods["fn"].overloads), 0)

    def test_more_than_once(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        exporter = package._exporter("fn", fn)
        exporter(x)
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot export .* more than once",
        ):
            exporter(x)

    def test_error(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        exporter = package._exporter("fn", fn, fallback="error")
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot export fallback .* when fallback policy is set to 'error'",
        ):
            exporter(x)

    def test_overloads(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.shape[0] == 4:
                    return x + 1
                elif x.shape[0] == 3:
                    return x - 1
                else:
                    return x + 2

        fn = Module()
        x = torch.randn(3, 2)
        x2 = torch.randn(4, 2)
        x3 = torch.randn(5, 2)

        def spec(self, x):
            assert x.shape[0] == 3

        def spec2(self, x):
            assert x.shape[0] == 4

        def spec3(self, x):
            assert x.shape[0] >= 5
            return {"x": (Dim("batch", min=5), Dim.STATIC)}

        package = _ExportPackage()
        exporter = (
            package._exporter("fn", fn)
            ._define_overload("spec", spec)
            ._define_overload("spec2", spec2)
            ._define_overload("spec3", spec3)
        )
        self.assertEqual(exporter(x), x - 1)
        self.assertEqual(exporter(x2), x2 + 1)
        self.assertEqual(exporter(x3), x3 + 2)
        self.assertEqual(len(package.methods), 1)
        self.assertEqual(len(package.methods["fn"].overloads), 3)

    def test_autofallback(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        package = _ExportPackage()
        exporter = package._exporter("fn", fn, fallback="auto")
        exporter(torch.randn(3, 2))
        exporter(torch.randn(2, 3))
        method = package.methods["fn"]
        self.assertEqual(len(method.fallbacks), 1)
        ep, example_inputs = next(iter(method.fallbacks.values()))
        self.assertEqual(len(example_inputs._examples), 2)
        self.assertExpectedInline(
            str(ep.graph_module.graph).strip(),
            """\
graph():
    %args_0 : [num_users=1] = placeholder[target=args_0]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%args_0, 1), kwargs = {})
    return (add,)""",
        )

        rand = torch.randn(4, 4)
        self.assertEqual(ep.module()(rand), fn(rand))

    def test_autofallback_num_elem_change(self):
        def fn(x: torch.Tensor, **kwargs) -> torch.Tensor:
            y = kwargs["y"]
            z = kwargs.get("z", None)
            if z is None:
                return x + y
            return x + y + z

        package = _ExportPackage()
        exporter = package._exporter("fn", fn, fallback="auto")
        input = torch.randn(3, 2)
        out1 = exporter(input, y=input)
        eager_out1 = fn(input, y=input)
        self.assertEqual(out1, eager_out1)
        out2 = exporter(input, y=input, z=input)
        eager_out2 = fn(input, y=input, z=input)
        self.assertEqual(out2, eager_out2)
        method = package.methods["fn"]
        self.assertEqual(len(method.fallbacks), 2)

    def test_autofallback_dtype_change(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x.sin()

        package = _ExportPackage()
        exporter = package._exporter("fn", fn, fallback="auto")
        input1 = torch.randn(3, 2, dtype=torch.float32)
        out1 = exporter(input1)
        eager_out1 = fn(input1)
        self.assertEqual(out1, eager_out1)

        input2 = input1.to(int)
        out2 = exporter(input2)
        eager_out2 = fn(input2)
        self.assertEqual(out2, eager_out2)

        input3 = input1.to("cuda")
        out3 = exporter(input3)
        eager_out3 = fn(input3)
        self.assertEqual(out3, eager_out3)
        method = package.methods["fn"]
        self.assertEqual(len(method.fallbacks), 3)

    def test_autofallback_discrete_inputs(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] > 3:
                return x.sin()
            return x.cos()

        input1 = torch.randn(4, 2, dtype=torch.float32)
        package = _ExportPackage()
        exporter = package._exporter("fn", fn, fallback="auto")
        out1 = exporter(input1)
        eager_out1 = fn(input1)
        self.assertEqual(out1, eager_out1)

        input2 = torch.randn(2, 2, dtype=torch.float32)
        with self.assertRaisesRegex(
            RuntimeError, r"Expected input at \*args\[0\].shape\[0\] to be <= 3"
        ):
            _ = exporter(input2)


if __name__ == "__main__":
    run_tests()
