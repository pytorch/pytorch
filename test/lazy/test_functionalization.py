# Owner(s): ["oncall: jit"]

import torch
import torch._lazy.metrics as metrics
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import run_tests, TestCase

torch._lazy.ts_backend.init()


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


if __name__ == "__main__":
    run_tests()
