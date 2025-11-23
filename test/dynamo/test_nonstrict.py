import torch
from dataclasses import dataclass
from torch.testing._internal.common_utils import run_tests, TestCase


@dataclass
class OutputData:
    result1: torch.Tensor
    result2: torch.Tensor


torch.utils._pytree.register_dataclass(OutputData)


class TestDataclassFunction(TestCase):
    def test_process_module_and_data(self):
        a = 4
        b = torch.randn(4, 4)

        @torch._dynamo.nonstrict_trace
        def gn(
            count, values
        ) -> tuple[torch.Tensor, OutputData]:
            output_tensor = a + b * count * values

            output_data = OutputData(
                result1=output_tensor + count,
                result2=output_tensor * (count + 1)
            )

            return output_tensor, output_data

        def fn(count, values):
            x = torch.sin(values)
            y, y_data = gn(count, values)
            return x + y + y_data.result1 + y_data.result2


        count = 5
        values = torch.randn(4, 4)
        ref = fn(count, values)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(count, values)
        self.assertEqual(ref, res)


if __name__ == "__main__":
    run_tests()
