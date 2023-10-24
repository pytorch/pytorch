# Owner(s): ["oncall: distributed"]
import torch
from torch.distributed._tensor.experimental.tp_transform import (
    tensor_parallel_transformation,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class MLPListModule(torch.nn.Module):
    """
    A dummy model with list of MLPs.
    """

    def __init__(self, num_mlps=3):
        super().__init__()
        self.mlps = torch.nn.ModuleList()
        for _ in range(num_mlps):
            self.mlps.append(
                torch.nn.Sequential(
                    torch.nn.Linear(6, 18),
                    torch.nn.ReLU(),
                    torch.nn.Linear(18, 6),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.chunk(x, 2, dim=1)[0]
        for mlp in self.mlps:
            x = mlp(x)
        return x + torch.ones_like(x)


class TensorParallelTest(DTensorTestBase):
    def setUp(self) -> None:
        super().setUp()

    @with_comms
    def test_tensor_parallel_transformation_e2e(self):
        torch.manual_seed(0)
        model = MLPListModule(2).to(device=self.device_type)
        inputs = (torch.randn((10, 12)).to(device=self.device_type),)
        with torch.inference_mode():
            res = model(*inputs)
        exported_program = torch._export.export(
            model,
            inputs,
            constraints=None,
        )
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
        )
        tp_model = tp_exported_program.module()
        with torch.inference_mode():
            tp_res = tp_model(*inputs)
        self.assertEqual(res, tp_res)
