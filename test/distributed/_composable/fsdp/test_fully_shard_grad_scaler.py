# Owner(s): ["oncall: distributed"]
import copy

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler, OptState
from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests


class TestFullyShardGradientScaler(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_gradient_scaler(self):
        self.run_subtests(
            {"has_inf": [True, False]},
            self._test_gradient_scaler,
        )

    def _test_gradient_scaler(self, has_inf: bool):
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
        inv_scale = scaler._scale.double().reciprocal().float()
        if (
            has_inf is True
            and opt.param_groups[0]["params"][0].grad._local_tensor.device.index == 0
        ):
            opt.param_groups[0]["params"][0].grad._local_tensor[0, 0].fill_(
                float("inf")
            )
        inital_grad = opt.param_groups[0]["params"][0].grad.to_local().clone()

        scaler.unscale_(opt)
        for found_inf in scaler._per_optimizer_states[id(opt)][
            "found_inf_per_device"
        ].values():
            self.assertEqual(found_inf, has_inf)
        self.assertEqual(
            scaler._per_optimizer_states[id(opt)]["stage"].value,
            OptState.UNSCALED.value,
        )
        unscaled_grad = opt.param_groups[0]["params"][0].grad.to_local().clone()
        self.assertEqual(unscaled_grad, inital_grad * inv_scale)
        initial_scale = scaler.get_scale()
        initial_state = copy.copy(opt.state)

        scaler.step(opt)
        steped_state = copy.copy(opt.state)
        if has_inf:
            # assert parameters are the same before/after
            self.assertEqual(steped_state, initial_state)
        else:
            # new parameters here if no inf found during .unscale_()
            self.assertNotEqual(steped_state.items(), initial_state.items())

        scaler.update()
        updated_scale = scaler.get_scale()
        if has_inf:
            # assert scale is updated
            backoff_factor = scaler.get_backoff_factor()
            self.assertEqual(updated_scale, initial_scale * backoff_factor)
        else:
            # scale is not updated
            self.assertEqual(updated_scale, initial_scale)


if __name__ == "__main__":
    run_tests()
