# Owner(s): ["oncall: distributed"]


import torch
import torch.nn as nn
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestMultiThread, get_devtype
from torch.testing._internal.common_utils import run_tests


c10d_ops = torch.ops.c10d
funcol = torch.ops.c10d_functional


device_type = torch.device(get_devtype())


class TestReplicateForwardInputs(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(1)
    def test_root_move_forward_input_to_device(self):
        device = torch.device(device_type.type, 0)

        class ParamlessModule(nn.Module):
            def forward(self, x: torch.Tensor, ys: tuple[torch.Tensor, ...]):
                # Check that Replicate moved the inputs to GPU, including recursing
                # into the tuple data structure
                assert x.device == device, f"Expects {device} but got {x.device}"
                assert ys[0].device == device, (
                    f"Expects {device} but got {ys[0].device}"
                )
                assert ys[1].device == device, (
                    f"Expects {device} but got {ys[1].device}"
                )
                y = ys[0] + ys[1]
                return x + y + 1

        model = ParamlessModule().to(device)
        replicate(model).to(device)
        x = torch.randn((3,))
        ys = (torch.randn((3,)), torch.randn((3,)))
        self.assertEqual(x.device, torch.device("cpu"))
        self.assertEqual(ys[0].device, torch.device("cpu"))
        self.assertEqual(ys[1].device, torch.device("cpu"))
        model(x, ys)


if __name__ == "__main__":
    run_tests()
