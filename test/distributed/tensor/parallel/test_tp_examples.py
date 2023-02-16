# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, Replicate
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    parallelize_module,
    TensorParallelMultiheadAttention,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    NUM_DEVICES,
    skip_unless_torch_gpu,
    with_comms,
)


class MLPModule(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = torch.nn.Linear(10, 16, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(16, 12, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


class MultiheadAttnWrap(nn.Module):
    def __init__(self, embed_dim, num_heads, add_bias_kv=False, device=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, add_bias_kv=add_bias_kv, device=device
        )

    def forward(self, query, key, value):
        return self.attn(query, key, value)


# TODO: replace repeated test code with _check_module
class DistTensorParallelExampleTest(DTensorTestBase):
    @with_comms
    def test_mlp_megatron_e2e(self):
        inp_size = [5, 10]
        # Ensure all tp ranks have same input.
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)
        model_tp = MLPModule(self.device_type)

        # Ensure model are initialized the same way.
        self.assertEqual(model.net1.weight, model_tp.net1.weight)
        self.assertEqual(model.net1.bias, model_tp.net1.bias)
        self.assertEqual(model.net2.weight, model_tp.net2.weight)
        self.assertEqual(model.net2.bias, model_tp.net2.bias)

        # Shard module and initialize optimizer.
        LR = 0.25
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        model_tp = parallelize_module(model_tp, device_mesh, PairwiseParallel())
        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim_tp = torch.optim.SGD(model_tp.parameters(), lr=LR)

        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)

        output.sum().backward()
        output_tp.sum().backward()

        device_mesh = model_tp.net1.weight.device_mesh
        replicate = [Replicate()] * device_mesh.ndim

        # Ensure gradients are same.
        self.assertEqual(
            model.net1.weight.grad,
            model_tp.net1.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.net1.bias.grad,
            model_tp.net1.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.net2.weight.grad,
            model_tp.net2.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.net2.bias.grad,
            model_tp.net2.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        optim.step()
        optim_tp.step()

        # Ensure model weights are still same after update.
        self.assertEqual(
            model.net1.weight,
            model_tp.net1.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.net1.bias,
            model_tp.net1.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.net2.weight,
            model_tp.net2.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        # Due to the trick we use for Partial aggregation, we only check the weight when local_rank = 0.
        if self.rank == 0:
            self.assertEqual(
                model.net2.bias,
                model_tp.net2.bias.redistribute(
                    device_mesh=device_mesh, placements=replicate
                ).to_local(),
            )

        inp = torch.rand(*inp_size, device=self.device_type)
        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)

    # TensorParallelMultiheadAttention == dist_module(TensorParallelMultiheadAttention)
    # baddbmm introduces nan occasionally on CPU: https://github.com/pytorch/pytorch/issues/80588
    @with_comms
    @skip_unless_torch_gpu
    def test_self_attn_megatron_e2e(self):
        inp_size = [8, 12, 16]
        # Ensure all tp ranks have same input.
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)

        # Initialize model using same seed.
        torch.manual_seed(5)
        model = TensorParallelMultiheadAttention(
            16,
            8,
            tp_size=NUM_DEVICES,
            add_bias_kv=True,
            device=self.device_type,
        )
        torch.manual_seed(5)
        model_tp = TensorParallelMultiheadAttention(
            16,
            8,
            tp_size=NUM_DEVICES,
            add_bias_kv=True,
            device=self.device_type,
        )

        # Ensure model are initialized the same way.
        self.assertEqual(model.qkv.weight, model_tp.qkv.weight)
        self.assertEqual(model.qkv.bias, model_tp.qkv.bias)
        self.assertEqual(model.proj.weight, model_tp.proj.weight)
        self.assertEqual(model.proj.bias, model_tp.proj.bias)

        # Shard module and initialize optimizer.
        device_mesh = DeviceMesh(self.device_type, list(range(NUM_DEVICES)))
        parallelize_module(model_tp, device_mesh, PairwiseParallel())

        device_mesh = model_tp.qkv.weight.device_mesh
        replicate = [Replicate()] * device_mesh.ndim
        # Ensure model are initialized the same way.
        self.assertEqual(
            model.qkv.weight,
            model_tp.qkv.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.qkv.bias,
            model_tp.qkv.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.weight,
            model_tp.proj.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.bias,
            model_tp.proj.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        LR = 0.25
        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim_tp = torch.optim.SGD(model_tp.parameters(), lr=LR)

        output = model(inp, inp, inp)
        output_tp = model_tp(inp, inp, inp)
        self.assertEqual(output, output_tp)

        output.sum().backward()
        output_tp.sum().backward()

        device_mesh = model_tp.qkv.weight.device_mesh
        # Ensure gradients are same.
        self.assertEqual(
            model.qkv.weight.grad,
            model_tp.qkv.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.qkv.bias.grad,
            model_tp.qkv.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.weight.grad,
            model_tp.proj.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.bias.grad,
            model_tp.proj.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        optim.step()
        optim_tp.step()

        # Ensure model weights are still same after update.
        self.assertEqual(
            model.qkv.weight,
            model_tp.qkv.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.qkv.bias,
            model_tp.qkv.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.weight,
            model_tp.proj.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.bias,
            model_tp.proj.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        inp = torch.rand(*inp_size, device=self.device_type)
        output = model(inp, inp, inp)
        output_tp = model_tp(inp, inp, inp)
        self.assertEqual(output, output_tp)

    # TensorParallelMultiheadAttention == dist_module(torch.nn.MultiheadAttention)
    # baddbmm introduces nan occasionally on CPU: https://github.com/pytorch/pytorch/issues/80588
    @with_comms
    @skip_unless_torch_gpu
    def test_self_attn_replacement_megatron_e2e(self):
        inp_size = [8, 12, 16]
        # Ensure all tp ranks have same input.
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)

        # TODO: our sharding function cannot shard the root node
        torch.manual_seed(5)
        model = TensorParallelMultiheadAttention(
            16,
            8,
            tp_size=NUM_DEVICES,
            add_bias_kv=True,
            device=self.device_type,
        )
        model_tp = MultiheadAttnWrap(16, 8, add_bias_kv=True, device=self.device_type)

        # TODO: somehow using torch.nn.MultiheadAttention's initial params does not work
        # Use TensorParallelMultiheadAttention parameters instead
        x = model.qkv.weight.clone().detach().requires_grad_()
        model_tp.attn.register_parameter("in_proj_weight", torch.nn.Parameter(x))

        x = model.qkv.bias.clone().detach().requires_grad_()
        model_tp.attn.register_parameter("in_proj_bias", torch.nn.Parameter(x))

        x = model.proj.weight.clone().detach().requires_grad_()
        model_tp.attn.out_proj.register_parameter("weight", torch.nn.Parameter(x))

        x = model.proj.bias.clone().detach().requires_grad_()
        model_tp.attn.out_proj.register_parameter("bias", torch.nn.Parameter(x))

        # check if parameters are same
        self.assertEqual(model.qkv.weight, model_tp.attn.in_proj_weight)
        self.assertEqual(model.qkv.bias, model_tp.attn.in_proj_bias)
        self.assertEqual(model.proj.weight, model_tp.attn.out_proj.weight)
        self.assertEqual(model.proj.bias, model_tp.attn.out_proj.bias)

        # Shard module and initialize optimizer.
        device_mesh = DeviceMesh(self.device_type, list(range(NUM_DEVICES)))
        parallelize_module(model_tp, device_mesh, PairwiseParallel())

        device_mesh = model_tp.attn.qkv.weight.device_mesh
        replicate = [Replicate()] * device_mesh.ndim
        # Ensure model are initialized the same way.
        self.assertEqual(
            model.qkv.weight,
            model_tp.attn.qkv.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.qkv.bias,
            model_tp.attn.qkv.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.weight,
            model_tp.attn.proj.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.bias,
            model_tp.attn.proj.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        LR = 0.25
        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim_tp = torch.optim.SGD(model_tp.parameters(), lr=LR)

        output = model(inp, inp, inp)
        output_tp = model_tp(inp, inp, inp)
        self.assertEqual(output, output_tp)

        output.sum().backward()
        output_tp.sum().backward()

        device_mesh = model_tp.attn.qkv.weight.device_mesh
        # Ensure gradients are same.
        self.assertEqual(
            model.qkv.weight.grad,
            model_tp.attn.qkv.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.qkv.bias.grad,
            model_tp.attn.qkv.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.weight.grad,
            model_tp.attn.proj.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.bias.grad,
            model_tp.attn.proj.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        optim.step()
        optim_tp.step()

        # Ensure model weights are still same after update.
        self.assertEqual(
            model.qkv.weight,
            model_tp.attn.qkv.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.qkv.bias,
            model_tp.attn.qkv.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.weight,
            model_tp.attn.proj.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.bias,
            model_tp.attn.proj.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        inp = torch.rand(*inp_size, device=self.device_type)
        output = model(inp, inp, inp)
        output_tp = model_tp(inp, inp, inp)
        self.assertEqual(output, output_tp)


if __name__ == "__main__":
    run_tests()
