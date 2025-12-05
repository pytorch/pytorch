# Owner(s): ["module: dynamo"]
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.testing._internal.common_utils import run_tests, TestCase


@dataclass
class InputData:
    count: int
    values: Tensor


torch.utils._pytree.register_dataclass(InputData)


@dataclass
class InputInvalid:
    count: int
    values: Tensor

# No pytree for InputInvalid


@dataclass
class OutputData:
    result1: Tensor
    result2: Tensor


torch.utils._pytree.register_dataclass(OutputData)

@dataclass
class OutputInvalid:
    result1: Tensor
    result2: Tensor

# No pytree for OutputInvalid

class TestDataclassFunction(TestCase):
    def test_simple(self):
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i_count: int, i_values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            output_tensor = a + b * i_count * i_values

            result1 = output_tensor + i_count
            result2 = output_tensor * (i_count + 1)

            return output_tensor, result1, result2

        def fn(i: InputData) -> Tensor:
            x = torch.sin(i.values)
            y, z_result1, z_result2 = gn(i.count, i.values)
            return x + y + z_result1 + z_result2

        count = 5
        values = torch.randn(4, 4)
        i = InputData(count, values)
        ref = fn(i)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(i)
        self.assertEqual(ref, res)

    def test_dataclass_input(self):
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i: InputData) -> tuple[Tensor, Tensor, Tensor]:
            output_tensor = a + b * i.count * i.values

            result1 = output_tensor + i.count
            result2 = output_tensor * (i.count + 1)

            return output_tensor, result1, result2

        def fn(i: InputData) -> Tensor:
            x = torch.sin(i.values)
            y, z_result1, z_result2 = gn(i)
            return x + y + z_result1 + z_result2

        count = 5
        values = torch.randn(4, 4)
        i = InputData(count, values)
        ref = fn(i)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(i)
        self.assertEqual(ref, res)

    def test_invalid_input(self):
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i: InputInvalid) -> tuple[Tensor, Tensor, Tensor]:
            output_tensor = a + b * i.count * i.values

            result1 = output_tensor + i.count
            result2 = output_tensor * (i.count + 1)

            return output_tensor, result1, result2

        def fn(i: InputInvalid) -> Tensor:
            x = torch.sin(i.values)
            y, z_result1, z_result2 = gn(i)
            return x + y + z_result1 + z_result2

        count = 5
        values = torch.randn(4, 4)
        i = InputInvalid(count, values)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Invalid input type for nonstrict_trace-ed function",
        ):
            res = opt_fn(i)

    def test_dataclass_output(self):
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i_count: int, i_values: Tensor) -> tuple[Tensor, OutputData]:
            output_tensor = a + b * i_count * i_values

            result1 = output_tensor + i_count
            result2 = output_tensor * (i_count + 1)
            out = OutputData(result1, result2)

            return output_tensor, out

        def fn(i: InputData) -> Tensor:
            x = torch.sin(i.values)
            y, z = gn(i.count, i.values)
            return x + y + z.result1 + z.result2

        count = 5
        values = torch.randn(4, 4)
        i = InputData(count, values)
        ref = fn(i)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(i)
        self.assertEqual(ref, res)

    def test_invalid_output(self):
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(i_count: int, i_values: Tensor) -> tuple[Tensor, OutputInvalid]:
            output_tensor = a + b * i_count * i_values

            result1 = output_tensor + i_count
            result2 = output_tensor * (i_count + 1)
            out = OutputInvalid(result1, result2)

            return output_tensor, out

        def fn(i: InputData) -> Tensor:
            x = torch.sin(i.values)
            y, z = gn(i.count, i.values)
            return x + y + z.result1 + z.result2

        count = 5
        values = torch.randn(4, 4)
        i = InputData(count, values)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Unsupported output type for nonstrict_trace-ed function",
        ):
            res = opt_fn(i)


if __name__ == "__main__":
    run_tests()
