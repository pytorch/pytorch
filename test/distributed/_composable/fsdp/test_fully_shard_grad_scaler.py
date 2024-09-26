# Owner(s): ["oncall: distributed"]
import copy

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler, OptState
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, MLP
from torch.testing._internal.common_utils import run_tests, skipIfRocm


class TestFullyShardGradientScaler(FSDPTest):
    @skip_if_lt_x_gpu(4)
    @skipIfRocm
    def test_gradient_scaler(self):
        self.run_subtests(
            {"has_inf": [True, False], "test_2d": [True, False]},
            self._test_gradient_scaler,
        )

    def _test_gradient_scaler(self, has_inf: bool, test_2d: bool):
        torch.manual_seed(0)
        model = nn.Sequential(
            *[nn.Linear(4, 4, device="cuda", bias=False) for _ in range(2)]
        )
        for layer in model:
            fully_shard(layer)
        fully_shard(model)
        input = torch.randn([4, 4], device="cuda")

        if test_2d:
            mesh_2d = init_device_mesh(
                "cuda", (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
            )
            dp_mesh, tp_mesh = mesh_2d["dp"], mesh_2d["tp"]
            model = nn.Sequential(MLP(2), MLP(2), MLP(2))
            tp_parallelize_plan = {
                "0.in_proj": ColwiseParallel(),
                "0.out_proj": RowwiseParallel(),
                "1.in_proj": ColwiseParallel(),
                "1.out_proj": RowwiseParallel(),
                "2.in_proj": ColwiseParallel(),
                "2.out_proj": RowwiseParallel(),
            }
            model = parallelize_module(
                model,
                device_mesh=tp_mesh,
                parallelize_plan=tp_parallelize_plan,
            )
            for module in model:
                fully_shard(module, mesh=dp_mesh)
            fully_shard(model, mesh=dp_mesh)
            input = torch.randn((2,), device="cuda")

        loss = model(input).sum()
        scaler = GradScaler(init_scale=2.0, enabled=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        scaler.scale(loss).backward()
        inv_scale = scaler._scale.double().reciprocal().float()
        if (
            has_inf is True
            and opt.param_groups[0]["params"][0].grad._local_tensor.device.index == 1
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
