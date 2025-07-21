# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed.tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.nn import functional as F
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)


ITER_TIME = 10
LR = 0.001


def _conv_fn(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    for name, param in module.named_parameters():
        dist_spec = [Replicate()]
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, device_mesh, dist_spec)
        )
        name = "_".join(name.split("."))
        module.register_parameter(name, dist_param)


class DistConvolutionOpsTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        # hard code world size to 2
        return 2

    @with_comms
    def test_downsampling_convolution(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(3)]

        input_list = torch.rand(ITER_TIME, 7, 3, 512, 1024)
        grad_output_list = torch.rand(ITER_TIME, 7, 256, 128, 256) * 1e-3

        model = nn.Conv2d(3, 256, kernel_size=4, stride=4, padding=0).to(
            self.device_type
        )
        nn.init.ones_(model.weight)
        nn.init.zeros_(model.bias)
        model_gt = copy.deepcopy(model).to(self.device_type)

        # training with dtensor
        model = distribute_module(
            model, device_mesh, _conv_fn, input_fn=None, output_fn=None
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        for i in range(ITER_TIME):
            optimizer.zero_grad()
            inp = input_list[i].to(self.device_type).requires_grad_()
            inp_dtensor = distribute_tensor(inp, device_mesh, shard_spec)
            output = model(inp_dtensor)
            grad_output = grad_output_list[i].to(self.device_type)
            grad_output_dtensor = distribute_tensor(
                grad_output, device_mesh, shard_spec
            )
            output.backward(grad_output_dtensor)
            optimizer.step()

        # training with plain tensor
        optimizer_gt = torch.optim.SGD(model_gt.parameters(), lr=LR)
        for i in range(ITER_TIME):
            optimizer_gt.zero_grad()
            inp = input_list[i].to(self.device_type).requires_grad_()
            output = model_gt(inp)
            grad_output = grad_output_list[i].to(self.device_type)
            output.backward(grad_output)
            optimizer_gt.step()

        weight_diff_abs = model.weight.to_local() - model_gt.weight
        bias_diff_abs = model.bias.to_local() - model_gt.bias
        weight_diff_rel = weight_diff_abs / (torch.abs(model_gt.weight) + 1e-8)
        bias_diff_rel = bias_diff_abs / (torch.abs(model_gt.bias) + 1e-8)
        weight_mse_abs = torch.mean(weight_diff_abs * weight_diff_abs).item()
        bias_mse_abs = torch.mean(bias_diff_abs * bias_diff_abs).item()
        weight_mse_rel = torch.mean(weight_diff_rel * weight_diff_rel).item()
        bias_mse_rel = torch.mean(bias_diff_rel * bias_diff_rel).item()
        self.assertTrue(
            weight_mse_abs <= 1e-6,
            f"Too large absolute mse for weight tensor, expected less equal 1e-6, got {weight_mse_abs}",
        )
        self.assertTrue(
            bias_mse_abs <= 1e-6,
            f"Too large absolute mse for bias tensor, expected less equal 1e-6, got {bias_mse_abs}",
        )
        self.assertTrue(
            weight_mse_rel <= 1e-6,
            f"Too large relative mse for weight tensor, expected less equal 1e-6, got {weight_mse_rel}",
        )
        self.assertTrue(
            bias_mse_rel <= 1e-6,
            f"Too large relative mse for bias tensor, expected less equal 1e-6, got {bias_mse_rel}",
        )

    # TODO: test_depthwise_convolution is broken in CI with gloo backend.
    # Temporarily disable it to unblock CI.
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_depthwise_convolution(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(3)]

        input_list = torch.rand(ITER_TIME, 7, 256, 128, 256)
        grad_output_list = torch.rand(ITER_TIME, 7, 256, 128, 256) * 1e-3

        model = nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256).to(
            self.device_type
        )
        nn.init.ones_(model.weight)
        nn.init.zeros_(model.bias)
        model_gt = copy.deepcopy(model).to(self.device_type)

        # training with dtensor
        model = distribute_module(
            model, device_mesh, _conv_fn, input_fn=None, output_fn=None
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        for i in range(ITER_TIME):
            optimizer.zero_grad()
            inp = input_list[i].to(self.device_type).requires_grad_()
            inp_dtensor = distribute_tensor(inp, device_mesh, shard_spec)
            output = model(inp_dtensor)
            grad_output = grad_output_list[i].to(self.device_type)
            grad_output_dtensor = distribute_tensor(
                grad_output, device_mesh, shard_spec
            )
            output.backward(grad_output_dtensor)
            optimizer.step()

        # training with plain tensor
        optimizer_gt = torch.optim.SGD(model_gt.parameters(), lr=LR)
        for i in range(ITER_TIME):
            optimizer_gt.zero_grad()
            inp = input_list[i].to(self.device_type).requires_grad_()
            output = model_gt(inp)
            grad_output = grad_output_list[i].to(self.device_type)
            output.backward(grad_output)
            optimizer_gt.step()

        weight_diff_abs = model.weight.to_local() - model_gt.weight
        bias_diff_abs = model.bias.to_local() - model_gt.bias
        weight_diff_rel = weight_diff_abs / (torch.abs(model_gt.weight) + 1e-8)
        bias_diff_rel = bias_diff_abs / (torch.abs(model_gt.bias) + 1e-8)
        weight_mse_abs = torch.mean(weight_diff_abs * weight_diff_abs).item()
        bias_mse_abs = torch.mean(bias_diff_abs * bias_diff_abs).item()
        weight_mse_rel = torch.mean(weight_diff_rel * weight_diff_rel).item()
        bias_mse_rel = torch.mean(bias_diff_rel * bias_diff_rel).item()
        self.assertTrue(
            weight_mse_abs <= 1e-6,
            f"Too large absolute mse for weight tensor, expected less equal 1e-6, got {weight_mse_abs}",
        )
        self.assertTrue(
            bias_mse_abs <= 1e-6,
            f"Too large absolute mse for bias tensor, expected less equal 1e-6, got {bias_mse_abs}",
        )
        self.assertTrue(
            weight_mse_rel <= 1e-6,
            f"Too large relative mse for weight tensor, expected less equal 1e-6, got {weight_mse_rel}",
        )
        self.assertTrue(
            bias_mse_rel <= 1e-6,
            f"Too large relative mse for bias tensor, expected less equal 1e-6, got {bias_mse_rel}",
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_conv_backward_none_grad_inp(self):
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        conv = nn.Conv2d(64, 64, 3, padding=1).train()
        x = torch.randn(1, 64, 32, 32)
        x_dt = DTensor.from_local(x, device_mesh, [Replicate()])
        w = conv.weight
        w_dt = torch.nn.Parameter(DTensor.from_local(w, device_mesh, [Replicate()]))

        b = conv.bias
        b_dt = torch.nn.Parameter(DTensor.from_local(b, device_mesh, [Replicate()]))

        res = F.conv2d(x_dt, w_dt, b_dt, padding=1)
        dres = torch.rand_like(res)
        res.backward(dres)
        self.assertTrue(w_dt.grad is not None)
        self.assertTrue(b_dt.grad is not None)
        self.assertTrue(x_dt.grad is None)


if __name__ == "__main__":
    run_tests()
