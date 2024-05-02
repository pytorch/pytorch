# Owner(s): ["oncall: jit"]

import re

import torch
import torch._lazy.metrics as metrics
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import run_tests, TestCase

torch._lazy.ts_backend.init()

NODE_TYPE_PATTERN = re.compile(r", NodeType=[^\n]+")


class LazyFuncionalizationTest(TestCase):
    def test_lazy_init_with_view(self):
        def f(device, reset_storage=False):
            torch.manual_seed(2023)

            if device == "lazy":
                metrics.reset()

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(4, 2, bias=False)

                def forward(self, x):
                    return x @ self.fc1.weight.transpose(0, 1)

            with torch.device(device):
                model = Model()

                if device == "lazy":
                    if reset_storage:
                        torch._C._unsafe_reset_storage(model.fc1.weight)

                    torch._lazy.mark_step()

                    sync_tensors = metrics.counter_value("SyncedTensorsWithIR")
                    if reset_storage:
                        assert sync_tensors == 1
                    else:
                        # There is an extra tensor being unnecessarily synced if
                        # the functional storage is not reset.
                        assert sync_tensors == 2

                x = torch.ones(4)
                out = model(x)

                if device == "lazy":
                    torch._lazy.mark_step()

                return out

        cpu_out = f("cpu")
        lazy_out_1 = f("lazy", reset_storage=False)
        lazy_out_2 = f("lazy", reset_storage=True)

        self.assertEqual(cpu_out, lazy_out_1.to("cpu"))
        self.assertEqual(cpu_out, lazy_out_2.to("cpu"))

    def test_data_assign(self):
        def text(lazyt):
            raw = torch._C._lazy._get_tensors_text([lazyt])
            return NODE_TYPE_PATTERN.sub("", raw)

        origin = torch.rand(3, dtype=torch.float32)
        tensor = origin.to("lazy")

        self.assertExpectedInline(
            text(tensor),
            """\
IR {
  %0 = [Float[3]] lazy_tensors::device_data(), device=CPU0, ROOT=0
}
""",
        )

        # Modify the data-type of tensor, and assign it to 'data'.
        # This should update the inner tensor of FunctionalTensorWrapper,
        # changing the corresponding IR node.
        modified_tensor = tensor.to(torch.bfloat16)
        tensor.data = modified_tensor

        self.assertExpectedInline(
            text(tensor),
            """\
IR {
  %0 = [Float[3]] lazy_tensors::device_data(), device=CPU0
  %1 = [BFloat16[3]] aten::_to_copy(%0), dtype=BFloat16, layout=null, device=null, pin_memory=null, non_blocking=0, memory_format=null, ROOT=0
}
""",  # noqa: B950
        )


if __name__ == "__main__":
    run_tests()
