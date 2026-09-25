# Owner(s): ["oncall: distributed"]

import copy
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    DataParallelMeshDims,
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.distributed.pipelining._backward import (
    stage_backward_input,
    stage_backward_weight,
)
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import parametrize, run_tests


if dist._is_spmd_types_available():
    import spmd_types as spmd
    from spmd_types.checker import typecheck


class TwoLinear(nn.Module):
    def __init__(self, device, in_features=4, out_features=4):
        super().__init__()
        self.first = nn.Linear(in_features, out_features, bias=False, device=device)
        self.second = nn.Linear(in_features, out_features, bias=False, device=device)

    def forward(self, inp, use_second=True):
        output = self.first(inp)
        return output + self.second(inp) if use_second else output


class TestFullyShardPendingGrad(FSDPTest):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_grad_dtype_unused_parameters(self, device):
        class Model(TwoLinear):
            def forward(self, inp, use_second):
                return self.second(inp) if use_second else self.first(inp)

        device = torch.device(device).type
        model = Model(device, in_features=1, out_features=3).to(torch.bfloat16)
        model.first.weight.grad_dtype = torch.float32
        model.second.weight.grad_dtype = torch.bfloat16
        fully_shard(
            model,
            mesh=init_device_mesh(device, (self.world_size,)),
            mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32),
        )
        model.set_reduce_scatter_unused_params(True)
        inp = torch.full((1, 1), self.rank + 1, device=device, dtype=torch.bfloat16)
        for first_rank in (0, 1):
            model.zero_grad(set_to_none=True)
            model(inp, use_second=self.rank != first_rank).sum().backward()
            for param, expected in zip(
                model.parameters(), ((first_rank + 1) / 2, (2 - first_rank) / 2)
            ):
                self.assertEqual(param.grad.dtype, param.grad_dtype)
                actual = param.grad.full_tensor()
                self.assertEqual(actual, torch.full_like(actual, expected))

    @skip_if_lt_x_gpu(2)
    def test_grad_dtype_none_unused_parameters(self, device):
        class Model(TwoLinear):
            def __init__(self, device):
                super().__init__(device, in_features=1, out_features=2)
                self.third = nn.Linear(1, 2, bias=False, device=device)

            def forward(self, inp, use_second):
                branch = self.second if use_second else self.third
                return self.first(inp) + branch(inp)

        device = torch.device(device).type
        model = Model(device)
        model.first.weight.grad_dtype = None
        model.first.weight.requires_grad_(False)
        fully_shard(model, mesh=init_device_mesh(device, (self.world_size,)))
        model.set_reduce_scatter_unused_params(True)
        inp = torch.ones((1, 1), device=device)
        # Frozen parameters never need zero gradients, so no reduce_dtype is needed.
        model(inp, use_second=self.rank == 0).sum().backward()
        self.assertIsNone(model.first.weight.grad)
        self.assertIsNotNone(model.second.weight.grad)
        self.assertIsNotNone(model.third.weight.grad)
        # Unfreezing after enabling unused-parameter reduction is still caught
        # on every rank instead of hanging the collective.
        model.zero_grad(set_to_none=True)
        model.first.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "grad_dtype=None requires"):
            model(inp, use_second=self.rank == 0).sum().backward()

    @skip_if_lt_x_gpu(2)
    @parametrize("cpu_offload", [False, True])
    def test_native_accumulation_dtypes(self, device, cpu_offload):
        device = torch.device(device).type
        model = nn.Linear(1, 2, bias=False, device=device)
        fully_shard(
            model,
            mesh=init_device_mesh(device, (self.world_size,)),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
            ),
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if cpu_offload
            else OffloadPolicy(),
        )
        cases = (
            (((True, 256), (False, 1), (True, -256)), 1),
            (((True, 3), (False, 256), (False, 1), (True, -256)), 3),
        )
        for contributions, expected in cases:
            model.zero_grad(set_to_none=True)
            for sync, value in contributions:
                model.set_requires_gradient_sync(sync)
                model(
                    torch.full((1, 1), value, device=device, dtype=torch.bfloat16)
                ).sum().backward()
                self.assertEqual(model.weight.grad.dtype, torch.float32)
                if not sync:
                    self.assertEqual(
                        model.weight.grad.to(device).full_tensor(),
                        torch.full((2, 1), float(contributions[0][1]), device=device),
                    )
                    model.unshard()
                    self.assertEqual(model.weight.grad.dtype, torch.bfloat16)
                    self.assertEqual(model.weight.grad.device.type, device)
                    model.reshard()
            actual = model.weight.grad.to(device).full_tensor()
            # Previously reduced FP32 history stays separate; pending history
            # still rounds in BF16, including 256 + 1 before the final -256.
            self.assertEqual(actual, torch.full_like(actual, expected))

    @skip_if_lt_x_gpu(2)
    @parametrize("cpu_offload", [False, True])
    @parametrize("reshard_after_backward", [False, True])
    def test_native_storage_and_mutations(
        self, device, cpu_offload, reshard_after_backward
    ):
        device = torch.device(device).type
        model = nn.Linear(2, 4, bias=False, device=device)
        fully_shard(
            model,
            mesh=init_device_mesh(device, (self.world_size,)),
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
            ),
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if cpu_offload
            else OffloadPolicy(),
        )
        model.set_reshard_after_backward(reshard_after_backward)
        saved = None
        expected = torch.zeros((4, 2), device=device, dtype=torch.bfloat16)
        for step, value in enumerate((2, 1, 1)):
            sync = step == 2
            model.set_requires_gradient_sync(sync)
            model.set_is_last_backward(sync)
            output = model(
                torch.full((1, 2), value, device=device, dtype=torch.bfloat16)
            )
            if saved is not None:
                self.assertIs(model.weight.grad, saved)
            output.sum().backward()
            expected.add_(value)
            if sync:
                if not reshard_after_backward:
                    self.assertIsNone(model.weight.grad)
                continue
            if reshard_after_backward:
                self.assertIsNone(model.weight.grad)
                model.unshard()
            unreduced = model.weight.grad
            self.assertEqual(unreduced.device.type, device)
            self.assertEqual(unreduced.dtype, torch.bfloat16)
            if saved is not None:
                self.assertIs(unreduced, saved)
                self.assertEqual(unreduced.data_ptr(), saved.data_ptr())
            saved = unreduced
            if step == 0:
                unreduced.square_()
                unreduced[:, 0].add_(1)
                expected.square_()
                expected[:, 0].add_(1)
            elif step == 1:
                unreduced.copy_(torch.full_like(unreduced, 3))
                expected.fill_(3)
            self.assertEqual(unreduced, expected)
            model.reshard()
            self.assertIsNone(model.weight.grad)
        model.reshard()
        self.assertEqual(
            model.weight.grad.device.type, "cpu" if cpu_offload else device
        )
        self.assertEqual(model.weight.grad.to(device).full_tensor(), expected.float())
        model.unshard()
        self.assertIsNone(model.weight.grad)

    @skip_if_lt_x_gpu(2)
    @parametrize("set_to_none", [False, True])
    @parametrize("cpu_offload", [False, True])
    @parametrize("clear_root", [False, True])
    def test_grouped_clear_after_forward(
        self, device, set_to_none, cpu_offload, clear_root
    ):
        device = torch.device(device).type
        model = TwoLinear(device)
        fully_shard(
            [model.first, model.second],
            mesh=init_device_mesh(device, (self.world_size,)),
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if cpu_offload
            else OffloadPolicy(),
        )
        fully_shard(model)
        model.set_reshard_after_backward(False)
        for sync, value in ((True, 4), (False, 2)):
            model.set_requires_gradient_sync(sync)
            model(
                torch.full((1, 4), value, device=device, dtype=torch.bfloat16)
            ).sum().backward()
        output = model(
            torch.ones(1, 4, device=device, dtype=torch.bfloat16, requires_grad=True),
            use_second=False,
        )
        for param in model.parameters():
            self.assertEqual(param.grad, torch.full_like(param.grad, 2))
        (model if clear_root else model.first).zero_grad(set_to_none=set_to_none)
        for module in (model.first, model.second) if clear_root else (model.first,):
            if set_to_none:
                self.assertIsNone(module.weight.grad)
            else:
                unreduced = module.weight.grad
                self.assertEqual(unreduced, torch.zeros_like(unreduced))
        if not clear_root:
            unreduced = model.second.weight.grad
            self.assertEqual(unreduced, torch.full_like(unreduced, 2))
        model.set_requires_gradient_sync(True)
        output.sum().backward()
        model.first.reshard()
        self.assertEqual(
            model.first.weight.grad.to(device).full_tensor(),
            torch.full((4, 4), 5.0, device=device),
        )
        self.assertEqual(
            model.second.weight.grad.to(device).full_tensor(),
            torch.full((4, 4), 4.0 if clear_root else 6.0, device=device),
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("set_to_none", [False, True])
    def test_sharded_clear_preserves_unsharded_grad(self, device, set_to_none):
        device = torch.device(device).type
        model = TwoLinear(device)
        fully_shard(
            [model.first, model.second],
            mesh=init_device_mesh(device, (self.world_size,)),
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        fully_shard(model)
        for sync, value in ((True, 4), (False, 2)):
            model.set_requires_gradient_sync(sync)
            model(
                torch.full((1, 4), value, device=device, dtype=torch.bfloat16)
            ).sum().backward()
        model.first.zero_grad(set_to_none=set_to_none)
        if set_to_none:
            self.assertIsNone(model.first.weight.grad)
        else:
            grad = model.first.weight.grad.to_local()
            self.assertEqual(grad, torch.zeros_like(grad))
        grad = model.second.weight.grad.to_local()
        self.assertEqual(grad, torch.full_like(grad, 4))
        model.first.unshard()
        for param in model.parameters():
            self.assertEqual(param.grad, torch.full_like(param.grad, 2))
        model.set_requires_gradient_sync(True)
        model(torch.ones(1, 4, device=device, dtype=torch.bfloat16)).sum().backward()
        for param, expected in zip(model.parameters(), (3.0, 7.0)):
            self.assertEqual(
                param.grad.full_tensor(), torch.full((4, 4), expected, device=device)
            )

    @skip_if_lt_x_gpu(2)
    @parametrize("set_to_none", [False, True])
    @parametrize("reduce_dtype", [torch.bfloat16, torch.float32])
    def test_grouped_pipeline_split_backward(self, device, set_to_none, reduce_dtype):
        device = torch.device(device).type
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(4, 4, bias=False, device=device),
            nn.Linear(4, 4, bias=False, device=device),
        )
        reference = copy.deepcopy(model).to(torch.bfloat16)
        for param in reference.parameters():
            param.grad_dtype = reduce_dtype
        fully_shard(
            [model[0], model[1]],
            mesh=init_device_mesh(device, (self.world_size,)),
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=reduce_dtype
            ),
        )
        fully_shard(model)
        model.set_reshard_after_backward(False)
        for _ in range(2):
            model.zero_grad(set_to_none=set_to_none)
            reference.zero_grad(set_to_none=set_to_none)
            saved = []
            for step in range(3):
                sync = step == 2
                model.set_requires_gradient_sync(sync)
                model.set_is_last_backward(sync)
                inp = torch.full(
                    (2, 4),
                    self.rank + step + 1.0,
                    device=device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                output = model(inp)
                if sync:
                    output.sum().backward()
                else:
                    _, groups = stage_backward_input(
                        [output], [torch.ones_like(output)], [inp], model.parameters()
                    )
                    stage_backward_weight(model.parameters(), groups)
                reference(inp.detach()).sum().backward()
                if sync:
                    for param in model.parameters():
                        self.assertIsNone(param.grad)
                    continue
                for index, (param, ref_param) in enumerate(
                    zip(model.parameters(), reference.parameters())
                ):
                    unreduced = param.grad
                    self.assertEqual(unreduced.dtype, reduce_dtype)
                    self.assertEqual(unreduced, ref_param.grad)
                    if step:
                        self.assertIs(unreduced, saved[index])
                    else:
                        saved.append(unreduced)
            model[0].reshard()
            for param, ref_param in zip(model.parameters(), reference.parameters()):
                dist.all_reduce(ref_param.grad, op=dist.ReduceOp.AVG)
                self.assertEqual(param.grad.full_tensor(), ref_param.grad.float())

    @skip_if_lt_x_gpu(2)
    @parametrize("cpu_offload", [False, True])
    def test_dtensor_pending_owner(self, device, cpu_offload):
        device = torch.device(device).type
        mesh = init_device_mesh(
            device, (1, self.world_size), mesh_dim_names=("dp", "tp")
        )
        reference = nn.Linear(4, 4, bias=False, device=device)
        model = parallelize_module(
            copy.deepcopy(reference), mesh["tp"], ColwiseParallel()
        )
        fully_shard(
            model,
            mesh=mesh["dp"],
            reshard_after_forward=False,
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if cpu_offload
            else OffloadPolicy(),
        )
        model.set_reshard_after_backward(False)
        saved = None
        for step, value in enumerate((1, 2, 3)):
            sync = step == 2
            model.set_requires_gradient_sync(sync)
            model.set_is_last_backward(sync)
            inp = torch.full((2, 4), value, device=device, dtype=torch.float32)
            reference(inp).sum().backward()
            model(inp).sum().backward()
            if sync:
                continue
            unreduced = model.weight.grad
            self.assertIsInstance(unreduced, DTensor)
            self.assertEqual(unreduced.device_mesh, mesh["tp"])
            self.assertEqual(unreduced.placements, (Shard(0),))
            self.assertEqual(unreduced.device.type, device)
            self.assertEqual(unreduced.full_tensor(), reference.weight.grad)
            if saved is not None:
                self.assertIs(unreduced, saved)
            saved = unreduced
        self.assertIsNone(model.weight.grad)
        model.reshard()
        self.assertEqual(
            model.weight.grad.to(device).full_tensor(), reference.weight.grad
        )

    @skip_if_lt_x_gpu(2)
    def test_accumulation_divide_factor(self, device):
        device = torch.device(device).type
        model = nn.Linear(1, 2, bias=False, device=device)
        fully_shard(model, mesh=init_device_mesh(device, (self.world_size,)))
        model.set_gradient_divide_factor(3)
        for step, sync in enumerate((True, False, True)):
            model.set_requires_gradient_sync(sync)
            model(torch.full((1, 1), 3.0, device=device)).sum().backward()
            self.assertEqual(
                model.weight.grad.full_tensor(),
                torch.full((2, 1), 6.0 if step == 2 else 2.0, device=device),
            )


class TestFullyShardPendingGradHSDP(FSDPTest):
    @property
    def world_size(self):
        return 4

    @skip_if_lt_x_gpu(4)
    @parametrize("cpu_offload", [False, True])
    @parametrize("reduce_dtype", [torch.bfloat16, torch.float32])
    def test_grad_dtype_pending_reductions(self, device, cpu_offload, reduce_dtype):
        device = torch.device(device).type
        model = TwoLinear(device, in_features=1, out_features=5).to(torch.bfloat16)
        model.first.weight.grad_dtype = torch.float32
        model.second.weight.grad_dtype = torch.bfloat16
        mesh = init_device_mesh(device, (2, 2), mesh_dim_names=("replicate", "shard"))
        fully_shard(
            [model.first, model.second],
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(reduce_dtype=reduce_dtype),
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if cpu_offload
            else OffloadPolicy(),
        )
        fully_shard(model)
        for step, value in enumerate((256, 1, -256)):
            model.set_requires_all_reduce(step == 2)
            model(
                torch.full((1, 1), value, device=device, dtype=torch.bfloat16)
            ).sum().backward()
        expected = 0 if reduce_dtype == torch.bfloat16 else 1
        for param in model.parameters():
            self.assertEqual(param.grad.dtype, param.grad_dtype)
            actual = param.grad.to(device).full_tensor()
            self.assertEqual(actual, torch.full_like(actual, expected))

        model.zero_grad(set_to_none=True)
        model.set_requires_all_reduce(False)
        for module, value in ((model.first, 4), (model.second, 5)):
            module(
                torch.full(
                    (1, 1),
                    value,
                    device=device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
            ).sum().backward()
        model.set_requires_all_reduce(True)
        model(
            torch.full((1, 1), 2, device=device, dtype=torch.bfloat16),
            use_second=False,
        ).sum().backward()
        for param, expected in zip(model.parameters(), (6, 5)):
            actual = param.grad.to(device).full_tensor()
            self.assertEqual(actual, torch.full_like(actual, expected))

    @skip_if_lt_x_gpu(4)
    @parametrize("cpu_offload", [False, True])
    @parametrize("clear", [None, False, True])
    def test_native_partial_ownership(self, device, cpu_offload, clear):
        self._test_native_partial_ownership(device, cpu_offload, clear, shard_dim=0)

    @skip_if_lt_x_gpu(4)
    @parametrize("set_to_none", [False, True])
    def test_native_partial_ownership_shard1(self, device, set_to_none):
        self._test_native_partial_ownership(
            device, False, set_to_none, shard_dim=1, clear_sharded=True
        )

    def _test_native_partial_ownership(
        self, device, cpu_offload, clear, shard_dim, clear_sharded=False
    ):
        device = torch.device(device).type
        in_features, out_features = (1, 5) if shard_dim == 0 else (4, 4)
        model = TwoLinear(device, in_features, out_features)
        mesh = init_device_mesh(device, (2, 2), mesh_dim_names=("replicate", "shard"))
        fully_shard(
            [model.first, model.second],
            mesh=mesh,
            shard_placement_fn=lambda _: Shard(shard_dim),
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
            ),
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if cpu_offload
            else OffloadPolicy(),
        )
        fully_shard(model)
        model.set_reshard_after_backward(False)
        model(
            torch.full((1, in_features), 4.0, device=device, dtype=torch.bfloat16)
        ).sum().backward()
        model.set_requires_all_reduce(False)
        for module, value in ((model.first, 2), (model.second, 3)):
            module(
                torch.full(
                    (1, in_features),
                    value,
                    device=device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
            ).sum().backward()
        model.set_requires_gradient_sync(False)
        model.first(
            torch.ones(
                1, in_features, device=device, dtype=torch.bfloat16, requires_grad=True
            )
        ).sum().backward()
        self.assertEqual(model.first.weight.grad.dtype, torch.bfloat16)
        self.assertEqual(model.first.weight.grad.device.type, device)
        # The second parameter only has an internal HSDP partial to synchronize.
        self.assertIsNone(model.second.weight.grad)
        if clear is not None:
            if clear_sharded:
                model.first.reshard()
            clear_module = model.first if clear_sharded else model
            clear_module.zero_grad(set_to_none=clear)
            for param in clear_module.parameters():
                if clear:
                    self.assertIsNone(param.grad)
                elif param.grad is not None:
                    self.assertEqual(param.grad, torch.zeros_like(param.grad))
        model.set_requires_gradient_sync(True)
        model(
            torch.ones(1, in_features, device=device, dtype=torch.bfloat16),
            use_second=False,
        ).sum().backward()
        expected_values = (8, 7)
        if clear is not None:
            expected_values = (4 if clear_sharded else 7, 7)
        model.first.unshard()
        for param in model.parameters():
            self.assertIsNone(param.grad)
        model.first.reshard()
        for param, expected in zip(model.parameters(), expected_values):
            self.assertEqual(
                param.grad.to(device).full_tensor(),
                torch.full((out_features, in_features), float(expected), device=device),
            )
        model(
            torch.ones(1, in_features, device=device, dtype=torch.bfloat16)
        ).sum().backward()
        model.first.reshard()
        for param, expected in zip(model.parameters(), expected_values):
            self.assertEqual(
                param.grad.to(device).full_tensor(),
                torch.full(
                    (out_features, in_features), float(expected + 1), device=device
                ),
            )


class SpmdPendingGradModel(nn.Module):
    def __init__(self, reference, mesh):
        super().__init__()
        self.tp_pg = mesh.get_group("tp")
        self.replicated_weight = nn.Parameter(reference[0].weight.detach().clone())
        self.sharded_weight = nn.Parameter(
            reference[1].weight.detach().chunk(2)[mesh["tp"].get_local_rank()].clone()
        )
        dp_axis = spmd.MeshAxis.of(mesh.get_group("dp"))
        tp_axis = spmd.MeshAxis.of(self.tp_pg)
        spmd.assert_type(self.replicated_weight, {dp_axis: spmd.R, tp_axis: spmd.R})
        spmd.assert_type(self.sharded_weight, {dp_axis: spmd.R, tp_axis: spmd.S(0)})
        for param in self.parameters():
            param.grad_dtype = torch.float32

    def forward(self, inp):
        output = inp @ self.replicated_weight.t()
        output = spmd.redistribute(
            output,
            self.tp_pg,
            src=spmd.S(0),
            dst=spmd.R,
            backward_options={"op_dtype": torch.float32},
        )
        output = output @ self.sharded_weight.t()
        output = spmd.redistribute(
            output,
            self.tp_pg,
            src=spmd.S(1),
            dst=spmd.I,
            backward_options={"op_dtype": torch.float32},
        )
        return output.sum()


@unittest.skipUnless(dist._is_spmd_types_available(), "requires spmd_types")
class TestFullyShardSpmdPendingGrad(FSDPTest):
    @property
    def world_size(self):
        return 4

    @skip_if_lt_x_gpu(4)
    @parametrize("cpu_offload", [False, True])
    def test_native_spmd_accumulation(self, device, cpu_offload):
        device = torch.device(device).type
        mesh = init_device_mesh(device, (2, 2), mesh_dim_names=("dp", "tp"))
        reference = nn.Sequential(
            nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16),
            nn.Linear(4, 8, bias=False, device=device, dtype=torch.bfloat16),
        )
        for param in reference.parameters():
            with torch.no_grad():
                param.fill_(0.25)
            param.grad_dtype = torch.float32
        model = SpmdPendingGradModel(reference, mesh)
        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="dp"),
            mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32),
            reshard_after_forward=False,
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if cpu_offload
            else OffloadPolicy(),
        )
        model.set_reshard_after_backward(False)
        dp_axis = spmd.MeshAxis.of(mesh.get_group("dp"))
        tp_axis = spmd.MeshAxis.of(mesh.get_group("tp"))
        saved = []
        for step in range(3):
            model.set_requires_gradient_sync(step == 2)
            inputs = [
                torch.arange(8, device=device, dtype=torch.bfloat16).view(2, 4) / 8
                + (rank + step + 1) / 2
                for rank in range(self.world_size)
            ]
            loss = (
                sum(
                    reference(torch.cat([inputs[rank] for rank in ranks])).sum()
                    for ranks in mesh.mesh.tolist()
                )
                / mesh["dp"].size()
            )
            loss.backward()
            inp = inputs[self.rank]
            with (
                spmd.set_current_mesh(mesh),
                typecheck(strict_mode="strict", local=False),
            ):
                spmd.assert_type(
                    inp,
                    {dp_axis: spmd.V, tp_axis: spmd.V},
                    partition_spec=spmd.PartitionSpec((dp_axis, tp_axis), None),
                )
                model(inp).backward()
            if step == 2:
                for param in model.parameters():
                    self.assertIsNone(param.grad)
                continue
            for index, param in enumerate(model.parameters()):
                unreduced = param.grad
                self.assertNotIsInstance(unreduced, DTensor)
                self.assertEqual(unreduced.dtype, torch.float32)
                self.assertEqual(unreduced.device.type, device)
                if step:
                    self.assertIs(unreduced, saved[index])
                else:
                    saved.append(unreduced)
        model.reshard()
        for param, ref_param in zip(model.parameters(), reference.parameters()):
            self.assertEqual(param.grad.to(device).full_tensor(), ref_param.grad)


instantiate_device_type_tests(
    TestFullyShardPendingGrad, globals(), only_for=(get_devtype().type,)
)
instantiate_device_type_tests(
    TestFullyShardPendingGradHSDP, globals(), only_for=(get_devtype().type,)
)
instantiate_device_type_tests(
    TestFullyShardSpmdPendingGrad, globals(), only_for=(get_devtype().type,)
)
if __name__ == "__main__":
    run_tests()
