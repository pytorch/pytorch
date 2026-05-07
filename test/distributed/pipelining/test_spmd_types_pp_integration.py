# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from __future__ import annotations

import functools

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining.schedules import Schedule1F1B
from torch.distributed.pipelining.stage import PipelineStage
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_accelerator_dist_backend,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_MULTIACCELERATOR,
)


if dist._is_spmd_types_available():
    import spmd_types as spmd
    import spmd_types._checker
    import spmd_types._type_attr

d_hid = 256
batch_size = 64
n_microbatches = 4

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = dist.get_default_backend_for_device(device_type)


class NormMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


class NormMLPStack(nn.Module):
    def __init__(self, dim: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([NormMLP(dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (output - target).pow(2).mean()


def _requires_multi_gpu(func):
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@skip_but_pass_in_sandcastle_if(
    not dist._is_spmd_types_available(), "requires spmd_types"
)
class TestSpmdTypesPPIntegration(MultiProcContinuousTest):
    """Verify that PP recv preserves spmd_types annotations on activations."""

    world_size = 4

    @classmethod
    def backend_str(cls) -> str:
        return backend

    @classmethod
    def device_type(cls) -> str:
        return device_type

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def init_pg(self) -> None:
        if device_type == "cuda":
            torch.cuda.set_device(self.device)
        from spmd_types._mesh_axis import _reset

        _reset()

    @_requires_multi_gpu
    def test_annotations_survive_pp_recv(self):
        """Stage 0 output carries Replicate annotation; stage 1 should receive it."""
        self.init_pg()

        mesh = init_device_mesh(device_type, (2, 2), mesh_dim_names=("pp", "tp"))
        tp_mesh = mesh["tp"]
        pp_mesh = mesh["pp"]
        pp_group = pp_mesh.get_group()
        pp_rank = pp_mesh.get_local_rank()
        tp_axis = spmd.MeshAxis.of(tp_mesh.get_group())

        torch.manual_seed(0)
        with torch.device(self.device):
            model = NormMLPStack(d_hid, 2)

        # Annotate all params as Replicate on TP axis
        for param in model.parameters():
            spmd._type_attr.set_local_type(param, {tp_axis: spmd.R})

        # Enable typechecking so forward outputs carry annotations
        mesh_axes = frozenset({tp_axis})
        for layer in model.layers:
            orig_fwd = layer.forward

            def _make_typechecked(fwd):
                def _typechecked_fwd(*args, **kwargs):
                    with (
                        spmd.set_current_mesh(mesh_axes),
                        spmd._checker.typecheck(strict_mode="strict"),
                    ):
                        return fwd(*args, **kwargs)

                return _typechecked_fwd

            layer.forward = _make_typechecked(orig_fwd)

        # Hook stage 1 to capture annotation on received activation
        received_types: list[dict | None] = []

        def _capture_annotation(module, args):
            inp = args[0]
            if spmd._checker.has_local_type(inp):
                lt = spmd._type_attr.get_local_type(inp)
                received_types.append(
                    {
                        name: lt.get(spmd.MeshAxis.of(tp_mesh.get_group(name)))
                        for name in tp_mesh.mesh_dim_names
                    }
                )
            else:
                received_types.append(None)

        model.layers[1].register_forward_pre_hook(_capture_annotation)

        stage = PipelineStage(
            submodule=model.get_submodule(f"layers.{pp_rank}"),
            stage_index=pp_rank,
            num_stages=2,
            device=self.device,
            group=pp_group,
            device_mesh=tp_mesh,
        )

        torch.manual_seed(42)
        input_tensor = torch.randn(batch_size, d_hid, device=self.device)
        spmd._type_attr.set_local_type(input_tensor, {tp_axis: spmd.R})
        target_tensor = torch.randn(batch_size, d_hid, device=self.device)

        schedule = Schedule1F1B(stage, n_microbatches=n_microbatches, loss_fn=_loss_fn)

        if stage.is_first:
            schedule.step(input_tensor)
        else:
            schedule.step(target=target_tensor, losses=[])

        # Only stage 1 (pp_rank=1) runs the hook
        if pp_rank == 1:
            self.assertGreater(len(received_types), 0, "Hook was not called on stage 1")
            first = received_types[0]
            self.assertIsNotNone(first, "Activation has no spmd_types annotation")
            self.assertEqual(
                first["tp"], spmd.R, f"Expected Replicate, got {first['tp']}"
            )


if __name__ == "__main__":
    run_tests()
