# Owner(s): ["oncall: distributed"]

import copy
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    DataParallelMeshDims,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import parametrize, run_tests


if dist._is_spmd_types_available():
    import spmd_types as spmd
    from spmd_types.checker import typecheck


class TestFullyShardPendingGrad(FSDPTest):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "consumer",
        [
            "clip",
            "clip_foreach",
            "unscale",
            "unscale_inf",
            "mul",
            "zero",
            "zero_none",
        ],
    )
    def test_pending_grad_consumers(self, device, consumer):
        self.run_subtests(
            {
                "mesh_size": [1, self.world_size],
                "reduce_op": ["avg", "sum"],
                "reshard_after_backward": [False, True],
            },
            self._test_pending_grad_consumers,
            torch.device(device).type,
            consumer,
        )

    def _test_pending_grad_consumers(
        self, device, consumer, mesh_size, reduce_op, reshard_after_backward
    ):
        mesh = init_device_mesh(
            device,
            (self.world_size // mesh_size, mesh_size),
            mesh_dim_names=("replicate", "shard"),
        )["shard"]
        ranks = mesh.mesh.tolist()
        ref_model = nn.Linear(4, 2, device=device)
        with torch.no_grad():
            ref_model.weight.fill_(0.25)
            ref_model.bias.fill_(0.5)
        model = copy.deepcopy(ref_model)
        fully_shard(model, mesh=mesh)
        model.set_reshard_after_backward(reshard_after_backward)
        model.set_requires_gradient_sync(False)
        if reduce_op == "sum":
            model.set_gradient_divide_factor(1)
        models = (ref_model, model)
        optims = [torch.optim.SGD(m.parameters(), lr=0.1) for m in models]
        scalers = [torch.amp.GradScaler(device, init_scale=8.0) for _ in models]
        use_scaler = consumer.startswith("unscale")

        def loss(module, rank, step):
            inp = torch.arange(8, device=device).view(2, 4) / 8
            inp = inp + (rank + 1) / 2 + step / 4
            output = module(inp).sum()
            if consumer == "unscale_inf" and step == 0 and rank == ranks[-1]:
                output = output * float("inf")
            return output

        def backward(step):
            ref_loss = sum(loss(ref_model, rank, step) for rank in ranks)
            if reduce_op == "avg":
                ref_loss = ref_loss / mesh_size
            losses = (ref_loss, loss(model, self.rank, step))
            for value, scaler in zip(losses, scalers):
                if use_scaler and step == 0:
                    value = scaler.scale(value)
                value.backward()

        def check_grads(placement):
            for ref_param, param in zip(ref_model.parameters(), model.parameters()):
                if ref_param.grad is None:
                    self.assertIsNone(param.grad)
                    continue
                self.assertIsInstance(param.grad, DTensor)
                self.assertEqual(param.grad.placements, (placement,))
                self.assertEqual(ref_param.grad, param.grad.full_tensor())

        backward(0)
        check_grads(Partial(reduce_op))
        norms = []
        found_inf = []
        for module, optim, scaler in zip(models, optims, scalers):
            if consumer.startswith("clip"):
                norm = nn.utils.clip_grad_norm_(
                    module.parameters(), 1.0, foreach=consumer == "clip_foreach"
                )
                norms.append(norm.full_tensor() if isinstance(norm, DTensor) else norm)
            elif use_scaler:
                scaler.unscale_(optim)
                found_inf.append(scaler._found_inf_per_device(optim))
            elif consumer.startswith("zero"):
                module.zero_grad(set_to_none=consumer == "zero_none")
            else:
                for param in module.parameters():
                    param.grad.mul_(2)
        if norms:
            self.assertEqual(norms[0], norms[1])
        if found_inf:
            self.assertEqual(found_inf[0], found_inf[1])
            for value in found_inf[1].values():
                self.assertEqual(value.item(), float(consumer == "unscale_inf"))
        check_grads(Partial(reduce_op))

        # The unscale cases only check mutation preservation here. Normal
        # GradScaler use unscales after all gradient accumulation is complete.
        model.set_requires_gradient_sync(True)
        backward(1)
        check_grads(Shard(0))

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "optimizer_cls,dtype",
        [
            (torch.optim.SGD, torch.float32),
            (torch.optim.AdamW, torch.float32),
            (torch.optim.SGD, torch.bfloat16),
        ],
    )
    @parametrize("foreach", [False, True])
    def test_optimizer_step_with_pending_grad(
        self, device, optimizer_cls, dtype, foreach
    ):
        self.run_subtests(
            {
                "mesh_size": [1, self.world_size],
                "reshard_after_backward": [False, True],
            },
            self._test_optimizer_step_with_pending_grad,
            torch.device(device).type,
            optimizer_cls,
            dtype,
            foreach,
        )

    def _test_optimizer_step_with_pending_grad(
        self, device, optimizer_cls, dtype, foreach, mesh_size, reshard_after_backward
    ):
        mesh = init_device_mesh(
            device,
            (self.world_size // mesh_size, mesh_size),
            mesh_dim_names=("replicate", "shard"),
        )["shard"]
        ranks = mesh.mesh.tolist()
        ref_model = nn.Linear(4, 2, device=device, dtype=dtype)
        with torch.no_grad():
            ref_model.weight.fill_(0.25)
            ref_model.bias.fill_(0.5)
        model = copy.deepcopy(ref_model)
        if dtype == torch.bfloat16:
            for module in (ref_model, model):
                for param in module.parameters():
                    param.grad_dtype = torch.float32
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32),
        )
        model.set_reshard_after_backward(reshard_after_backward)
        ref_optim, optim = [
            optimizer_cls(module.parameters(), lr=0.125, foreach=foreach, fused=False)
            for module in (ref_model, model)
        ]
        params = list(model.parameters())
        for step in range(4):
            sync = step >= 2
            model.set_requires_gradient_sync(sync)
            inputs = {
                rank: torch.arange(8, device=device, dtype=dtype).view(2, 4) / 8
                + (rank + 1) / 2
                + step / 4
                for rank in ranks
            }
            ref_outputs = {rank: ref_model(inp) for rank, inp in inputs.items()}
            with CommDebugMode() as comm_mode:
                output = model(inputs[self.rank])
            self.assertEqual(output, ref_outputs[self.rank])
            if step == 1 and not reshard_after_backward:
                self.assertEqual(comm_mode.get_total_counts(), 0)
            elif step >= 2 and mesh_size > 1:
                self.assertGreater(comm_mode.get_total_counts(), 0)
            ref_loss = sum(
                value.float().square().mean() for value in ref_outputs.values()
            )
            (ref_loss / mesh_size).backward()
            output.float().square().mean().backward()
            for ref_param, param, original in zip(
                ref_model.parameters(), model.parameters(), params
            ):
                self.assertIs(param, original)
                self.assertIsInstance(param.grad, DTensor)
                self.assertEqual(
                    param.grad.placements, (Shard(0) if sync else Partial("avg"),)
                )
                self.assertEqual(ref_param.grad, param.grad.full_tensor())
            if step in (1, 2):
                ref_optim.step()
                optim.step()
                for ref_param, param in zip(ref_model.parameters(), params):
                    self.assertEqual(ref_param, param.full_tensor())
                    self.assertEqual(
                        set(ref_optim.state[ref_param]), set(optim.state[param])
                    )
                    for key, value in optim.state[param].items():
                        if isinstance(value, DTensor):
                            value = value.full_tensor()
                        self.assertEqual(ref_optim.state[ref_param][key], value)


instantiate_device_type_tests(
    TestFullyShardPendingGrad, globals(), only_for=(get_devtype().type,)
)


class SpmdPendingGradModel(nn.Module):
    def __init__(self, ref_model, mesh):
        super().__init__()
        self.tp_pg = mesh.get_group("tp")
        self.replicated_weight = nn.Parameter(ref_model[0].weight.detach().clone())
        self.sharded_weight = nn.Parameter(
            ref_model[1].weight.detach().chunk(2)[mesh["tp"].get_local_rank()].clone()
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
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_reduced_to_pending_spmd_grad(self, device, dtype):
        device = torch.device(device).type
        meshes = [
            init_device_mesh(
                device,
                (self.world_size // (dp_size * 2), dp_size, 2),
                mesh_dim_names=("replicate", "dp", "tp"),
            )["dp", "tp"]
            for dp_size in (1, 2)
        ]
        self.run_subtests(
            {
                "mesh": meshes,
                "reduce_op": ["avg", "sum"],
                "mutation": ["mul", "zero", "zero_none"],
                "reshard_after_backward": [False, True],
            },
            self._test_reduced_to_pending_spmd_grad,
            device,
            dtype,
        )

    def _test_reduced_to_pending_spmd_grad(
        self, device, dtype, mesh, reduce_op, mutation, reshard_after_backward
    ):
        ref_model = nn.Sequential(
            nn.Linear(4, 4, bias=False, device=device, dtype=dtype),
            nn.Linear(4, 8, bias=False, device=device, dtype=dtype),
        )
        for param in ref_model.parameters():
            with torch.no_grad():
                param.fill_(0.25)
            param.grad_dtype = torch.float32
        model = SpmdPendingGradModel(ref_model, mesh)
        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="dp"),
            mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32),
            reshard_after_forward=False,
        )
        model.set_reshard_after_backward(reshard_after_backward)
        if reduce_op == "sum":
            model.set_gradient_divide_factor(1)
        dp_axis = spmd.MeshAxis.of(mesh.get_group("dp"))
        tp_axis = spmd.MeshAxis.of(mesh.get_group("tp"))
        ranks_by_dp = mesh.mesh.tolist()

        def make_input(rank, step):
            return (
                torch.arange(8, device=device, dtype=dtype).view(2, 4) / 8
                + (rank + 1) / 2
                + step / 4
            )

        def check_grads(pending):
            for index, (ref_param, param) in enumerate(
                zip(ref_model.parameters(), model.parameters())
            ):
                if ref_param.grad is None:
                    self.assertIsNone(param.grad)
                    continue
                self.assertIsInstance(param.grad, DTensor)
                tp_placement = Partial() if pending and index == 0 else Replicate()
                if index == 1:
                    tp_placement = Shard(0)
                self.assertEqual(param.grad.placements[1], tp_placement)
                if pending:
                    self.assertEqual(param.grad.placements[0], Partial(reduce_op))
                else:
                    self.assertEqual(param.grad.placements[0], param.placements[0])
                self.assertEqual(ref_param.grad, param.grad.full_tensor())

        for step, sync in enumerate((True, False, True)):
            model.set_requires_gradient_sync(sync)
            ref_loss = sum(
                ref_model(torch.cat([make_input(rank, step) for rank in ranks])).sum()
                for ranks in ranks_by_dp
            )
            if reduce_op == "avg":
                ref_loss = ref_loss / len(ranks_by_dp)
            ref_loss.backward()
            inp = make_input(self.rank, step)
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
            check_grads(pending=not sync)
            if not sync:
                for module in (ref_model, model):
                    if mutation == "mul":
                        for param in module.parameters():
                            param.grad.mul_(2)
                    else:
                        module.zero_grad(set_to_none=mutation == "zero_none")
                check_grads(pending=True)


instantiate_device_type_tests(
    TestFullyShardSpmdPendingGrad, globals(), only_for=(get_devtype().type,)
)
if __name__ == "__main__":
    run_tests()
