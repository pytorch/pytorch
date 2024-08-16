# Owner(s): ["oncall: distributed"]
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler, OptState
from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests


class TestFullyShardGradientScaler(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_gradient_scaler_no_infs(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            *[nn.Linear(4, 4, device="cuda", bias=False) for _ in range(2)]
        )
        for layer in model:
            fully_shard(layer)
        fully_shard(model)
        scaler = GradScaler(init_scale=2.0, enabled=True)
        input = torch.randn([4, 4], device="cuda")
        loss = model(input).sum()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        scaler.scale(loss).backward()
        inv_scale = scaler._scale.double().reciprocal().float().item()
        inital_grad = opt.param_groups[0]["params"][0].grad.to_local().clone()

        scaler.unscale_(opt)
        for found_inf in scaler._per_optimizer_states[id(opt)][
            "found_inf_per_device"
        ].values():
            self.assertEqual(found_inf.item(), 0)
        self.assertEqual(
            scaler._per_optimizer_states[id(opt)]["stage"].value,
            OptState.UNSCALED.value,
        )
        unscaled_grad = opt.param_groups[0]["params"][0].grad.to_local().clone()
        [row, col] = unscaled_grad.shape

        for row_idx in range(row):
            for col_idx in range(col):
                if inital_grad[row_idx][col_idx].item() != float("inf"):
                    self.assertEqual(
                        unscaled_grad[row_idx][col_idx].item(),
                        inital_grad[row_idx][col_idx].item() * inv_scale,
                    )

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=True)
        scaler.step(opt)
        scaler.update()

    @skip_if_lt_x_gpu(2)
    def test_gradient_scaler_with_infs(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            *[nn.Linear(4, 4, device="cuda", bias=False) for _ in range(2)]
        )
        for layer in model:
            fully_shard(layer)
        fully_shard(model)
        scaler = GradScaler(init_scale=2.0, enabled=True)
        input = torch.randn([4, 4], device="cuda")
        loss = model(input).sum()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        scaler.scale(loss).backward()
        inv_scale = scaler._scale.double().reciprocal().float().item()
        if opt.param_groups[0]["params"][0].grad._local_tensor.device.index == 0:
            opt.param_groups[0]["params"][0].grad._local_tensor[0, 0].fill_(
                float("inf")
            )
        inital_grad = opt.param_groups[0]["params"][0].grad.to_local().clone()

        scaler.unscale_(opt)
        for found_inf in scaler._per_optimizer_states[id(opt)][
            "found_inf_per_device"
        ].values():
            self.assertEqual(found_inf.item(), 1)

        unscaled_grad = opt.param_groups[0]["params"][0].grad.to_local().clone()
        [row, col] = unscaled_grad.shape
        for row_idx in range(row):
            for col_idx in range(col):
                if inital_grad[row_idx][col_idx].item() != float("inf"):
                    self.assertEqual(
                        unscaled_grad[row_idx][col_idx].item(),
                        inital_grad[row_idx][col_idx].item() * inv_scale,
                    )


if __name__ == "__main__":
    run_tests()
