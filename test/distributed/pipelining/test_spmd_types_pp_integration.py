# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from __future__ import annotations

import copy
import functools
import types
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import DataParallelMeshDims, fully_shard
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

if TYPE_CHECKING:
    from collections.abc import Callable


d_hid = 256
batch_size = 64
n_microbatches = 4

MODEL_SEED = 0
INPUT_SEED = 42
TARGET_SEED = 123

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


def _apply_spmd_tp(block: NormMLP, tp_pg: dist.ProcessGroup) -> None:
    """Shard weights for column/row parallel TP using spmd_types collectives."""
    from spmd_types import all_reduce, convert

    tp_rank = tp_pg.rank()
    tp_size = tp_pg.size()

    with torch.no_grad():
        # Column-parallel: shard fc1 weight on dim 0
        block.fc1.weight = nn.Parameter(
            block.fc1.weight.chunk(tp_size, dim=0)[tp_rank].contiguous()
        )
        # Row-parallel: shard fc2 weight on dim 1
        block.fc2.weight = nn.Parameter(
            block.fc2.weight.chunk(tp_size, dim=1)[tp_rank].contiguous()
        )

    def _tp_forward(self, x):
        # Input arrives as Invariant, broadcast to Replicate for matmul
        x = convert(x, tp_pg, src=spmd.I, dst=spmd.R)
        z = self.fc1(self.norm(x))
        z = self.act(z)
        z = self.fc2(z)
        # All-reduce to go back to Invariant
        z = all_reduce(z, tp_pg, dst=spmd.I)
        return z

    block.forward = types.MethodType(_tp_forward, block)


def _annotate_replicate(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Annotate all parameters as Replicate on the TP axis."""
    tp_axis = spmd.MeshAxis.of(tp_mesh.get_group())
    for param in model.parameters():
        spmd._type_attr.set_local_type(param, {tp_axis: spmd.R})


def _wrap_with_typecheck(model: NormMLPStack, mesh: DeviceMesh) -> None:
    """Wrap each NormMLP layer's forward to run under spmd_types typecheck."""
    mesh_axes = frozenset(
        spmd.MeshAxis.of(mesh.get_group(name)) for name in mesh.mesh_dim_names
    )

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

    def _make_mesh(self) -> DeviceMesh:
        return init_device_mesh(device_type, (2, 2), mesh_dim_names=("pp", "tp"))

    # ------------------------------------------------------------------
    # Reference model: run all layers serially with microbatched fwd/bwd
    # ------------------------------------------------------------------

    def _run_reference_microbatched(
        self,
        ref_model: NormMLPStack,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor | None], dict[int, tuple[torch.Tensor, torch.Tensor]]
    ]:
        """Run the reference model microbatched, capturing per-stage I/O shapes.

        Returns:
            ref_grads: {param_fqn: grad_tensor}
            static_stage_io: {stage_idx: (input_like, output_like)} for static metadata
        """
        ref_model.zero_grad(set_to_none=True)

        input_chunks = torch.tensor_split(input_tensor, n_microbatches)
        target_chunks = torch.tensor_split(target_tensor, n_microbatches)

        # Re-annotate chunks (tensor_split doesn't preserve spmd_types attrs)
        if spmd._checker.has_local_type(input_tensor):
            lt = spmd._type_attr.get_local_type(input_tensor)
            for chunk in input_chunks:
                spmd._type_attr.set_local_type(chunk, lt)

        # Capture first-microbatch I/O shapes per stage
        captured_io: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        for stage_idx, stage_mod in enumerate(ref_model.layers):

            def _capture_io(
                _module: nn.Module,
                args: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                idx: int = stage_idx,
            ) -> None:
                captured_io[idx] = (args[0], output)

            hooks.append(stage_mod.register_forward_hook(_capture_io))

        for mb_idx, (inp_chunk, tgt_chunk) in enumerate(
            zip(input_chunks, target_chunks)
        ):
            output = ref_model(inp_chunk)
            loss = _loss_fn(output, tgt_chunk)
            loss.backward()

            if mb_idx == 0:
                # Capture shapes from first microbatch, then remove hooks
                static_stage_io: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
                for si in range(len(ref_model.layers)):
                    stage_in, stage_out = captured_io[si]
                    static_stage_io[si] = (
                        torch.empty_like(stage_in),
                        torch.empty_like(stage_out),
                    )
                for h in hooks:
                    h.remove()
                hooks = []

        # Average gradients across microbatches
        for param in ref_model.parameters():
            if param.grad is not None:
                param.grad.div_(n_microbatches)

        ref_grads = {
            name: param.grad.clone() if param.grad is not None else None
            for name, param in ref_model.named_parameters()
        }
        return ref_grads, static_stage_io

    # ------------------------------------------------------------------
    # PP stage construction
    # ------------------------------------------------------------------

    def _make_stage(
        self,
        pp_model: NormMLPStack,
        stage_index: int,
        num_stages: int,
        pp_group: dist.ProcessGroup,
        tp_mesh: DeviceMesh,
    ) -> PipelineStage:
        stage_module = pp_model.get_submodule(f"layers.{stage_index}")
        return PipelineStage(
            submodule=stage_module,
            stage_index=stage_index,
            num_stages=num_stages,
            device=self.device,
            group=pp_group,
            device_mesh=tp_mesh,
        )

    # ------------------------------------------------------------------
    # Core test: training correctness
    # ------------------------------------------------------------------

    def _run_training_correctness(
        self,
        apply_parallelism: Callable[[NormMLP, dist.ProcessGroup], None],
        annotate_fn: Callable[[nn.Module, DeviceMesh], None],
        input_spmd_type: object | None = None,
        typecheck: bool = False,
    ) -> None:
        self.init_pg()

        mesh = self._make_mesh()
        tp_mesh = mesh["tp"]
        pp_mesh = mesh["pp"]
        pp_group = pp_mesh.get_group()
        pp_rank = pp_mesh.get_local_rank()
        tp_pg = tp_mesh.get_group()

        n_layers = 2  # 1 stage per PP rank

        # Build two identical models
        torch.manual_seed(MODEL_SEED)
        with torch.device(self.device):
            ref_model = NormMLPStack(d_hid, n_layers)
        pp_model = copy.deepcopy(ref_model)

        # Apply TP parallelism to both
        for layer in ref_model.layers:
            apply_parallelism(layer, tp_pg)
        for layer in pp_model.layers:
            apply_parallelism(layer, tp_pg)

        # Annotate with spmd_types
        annotate_fn(ref_model, tp_mesh)
        annotate_fn(pp_model, tp_mesh)
        if typecheck:
            _wrap_with_typecheck(ref_model, tp_mesh)
            _wrap_with_typecheck(pp_model, tp_mesh)

        # Create inputs (same on all ranks for Replicate; per-rank shard for TP)
        tp_axis = spmd.MeshAxis.of(tp_pg)
        torch.manual_seed(INPUT_SEED)
        input_full = torch.randn(batch_size, d_hid, device=self.device)
        torch.manual_seed(TARGET_SEED)
        target_full = torch.randn(batch_size, d_hid, device=self.device)

        # Reference: microbatched serial execution
        ref_input = input_full.clone().requires_grad_(True)
        if input_spmd_type is not None:
            spmd._type_attr.set_local_type(ref_input, {tp_axis: input_spmd_type})
        ref_target = target_full.clone()
        ref_grads, _ = self._run_reference_microbatched(
            ref_model, ref_input, ref_target
        )

        # PP execution (dynamic metadata inference)
        stage = self._make_stage(
            pp_model=pp_model,
            stage_index=pp_rank,
            num_stages=n_layers,
            pp_group=pp_group,
            tp_mesh=tp_mesh,
        )
        stage.submod.zero_grad(set_to_none=True)

        pp_input = input_full.clone().requires_grad_(True)
        if input_spmd_type is not None:
            spmd._type_attr.set_local_type(pp_input, {tp_axis: input_spmd_type})
        pp_target = target_full.clone()

        schedule = Schedule1F1B(
            stage,
            n_microbatches=n_microbatches,
            loss_fn=_loss_fn,
        )

        if stage.is_first and stage.is_last:
            losses: list[torch.Tensor] = []
            schedule.step(pp_input, target=pp_target, losses=losses)
        elif stage.is_first:
            schedule.step(pp_input)
        elif stage.is_last:
            losses = []
            schedule.step(target=pp_target, losses=losses)
        else:
            schedule.step()

        # Compare gradients
        stage_module = pp_model.get_submodule(f"layers.{pp_rank}")
        for name, param in stage_module.named_parameters():
            param_fqn = f"layers.{pp_rank}.{name}"
            self.assertIsNotNone(param.grad, f"Missing PP grad for {param_fqn}")
            ref_grad = ref_grads[param_fqn]
            self.assertIsNotNone(ref_grad, f"Missing ref grad for {param_fqn}")
            torch.testing.assert_close(
                param.grad,
                ref_grad,
                msg=f"Gradient mismatch for {param_fqn}",
            )

    # ------------------------------------------------------------------
    # Annotation verification: check annotations survive PP forward recv
    # ------------------------------------------------------------------

    def _run_annotation_check(
        self,
        apply_parallelism: Callable[[NormMLP, dist.ProcessGroup], None],
        annotate_fn: Callable[[nn.Module, DeviceMesh], None],
        expected_activation_types: dict[str, object],
    ) -> None:
        """Verify that spmd_types annotations on activations survive PP recv.

        Registers a forward pre-hook on stage 1 to check that the activation
        received from stage 0 carries the expected per-axis spmd type.

        Args:
            expected_activation_types: {mesh_dim_name: expected_type}, e.g.
                {"tp": spmd.R}.
        """
        self.init_pg()

        mesh = self._make_mesh()
        tp_mesh = mesh["tp"]
        pp_mesh = mesh["pp"]
        pp_group = pp_mesh.get_group()
        pp_rank = pp_mesh.get_local_rank()
        tp_pg = tp_mesh.get_group()

        torch.manual_seed(MODEL_SEED)
        with torch.device(self.device):
            model = NormMLPStack(d_hid, 2)

        for layer in model.layers:
            apply_parallelism(layer, tp_pg)
        annotate_fn(model, tp_mesh)

        # Register hook on stage 1 to check forward input annotations
        annotation_results: list[dict] = []

        def check_fwd_annotation(module, args):
            inp = args[0]
            result = {"has_type": spmd._checker.has_local_type(inp)}
            if result["has_type"]:
                lt = spmd.get_local_type(inp)
                result["local_type"] = {
                    name: lt.get(spmd.MeshAxis.of(tp_mesh.get_group(name)))
                    for name in tp_mesh.mesh_dim_names
                }
            annotation_results.append(result)

        model.layers[1].register_forward_pre_hook(check_fwd_annotation)

        torch.manual_seed(INPUT_SEED)
        input_full = torch.randn(batch_size, d_hid, device=self.device)
        torch.manual_seed(TARGET_SEED)
        target_full = torch.randn(batch_size, d_hid, device=self.device)

        stage = self._make_stage(
            pp_model=model,
            stage_index=pp_rank,
            num_stages=2,
            pp_group=pp_group,
            tp_mesh=tp_mesh,
        )

        schedule = Schedule1F1B(
            stage,
            n_microbatches=n_microbatches,
            loss_fn=_loss_fn,
        )

        if stage.is_first:
            schedule.step(input_full)
        elif stage.is_last:
            losses = []
            schedule.step(target=target_full, losses=losses)

        # Only pp_rank=1 (stage 1) runs the hook
        if pp_rank == 1:
            self.assertGreater(
                len(annotation_results),
                0,
                "Forward pre-hook was not called on stage 1",
            )
            first = annotation_results[0]
            self.assertTrue(
                first["has_type"],
                "Activation received by stage 1 has no spmd_types annotation",
            )
            for dim_name, expected_type in expected_activation_types.items():
                actual = first["local_type"].get(dim_name)
                self.assertEqual(
                    actual,
                    expected_type,
                    f"Activation type on '{dim_name}': expected {expected_type}, got {actual}",
                )

    # ------------------------------------------------------------------
    # Test methods
    # ------------------------------------------------------------------

    @_requires_multi_gpu
    def test_replicate(self):
        """PP + spmd_types with all-Replicate annotations and typecheck."""

        def _no_tp(block, tp_pg):
            pass

        self._run_training_correctness(
            _no_tp,
            _annotate_replicate,
            input_spmd_type=spmd.R,
            typecheck=True,
        )

    @_requires_multi_gpu
    def test_tp(self):
        """PP + TP with spmd_types annotations."""
        self._run_training_correctness(
            _apply_spmd_tp,
            _annotate_replicate,
        )

    # ------------------------------------------------------------------
    # FSDP + PP test
    # ------------------------------------------------------------------

    def _run_fsdp_pp_correctness(self) -> None:
        """FSDP + PP: params annotated as Replicate on FSDP axis, then fully_shard'd."""
        self.init_pg()

        mesh = init_device_mesh(device_type, (2, 2), mesh_dim_names=("pp", "fsdp"))
        fsdp_mesh = mesh["fsdp"]
        pp_mesh = mesh["pp"]
        pp_group = pp_mesh.get_group()
        pp_rank = pp_mesh.get_local_rank()
        fsdp_axis = spmd.MeshAxis.of(fsdp_mesh.get_group())

        n_layers = 2

        torch.manual_seed(MODEL_SEED)
        with torch.device(self.device):
            ref_model = NormMLPStack(d_hid, n_layers)
        pp_model = copy.deepcopy(ref_model)

        # Annotate params as Replicate on FSDP axis, then apply fully_shard
        for param in pp_model.parameters():
            spmd._type_attr.set_local_type(param, {fsdp_axis: spmd.R})
        for layer in pp_model.layers:
            fully_shard(
                layer,
                mesh=fsdp_mesh,
                dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
            )
        fully_shard(
            pp_model,
            mesh=fsdp_mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
        )

        # Reference: replicate on all ranks (no FSDP), microbatched
        from torch.distributed._composable import replicate

        replicate(ref_model, device_ids=[self.rank])

        torch.manual_seed(INPUT_SEED)
        input_full = torch.randn(batch_size, d_hid, device=self.device)
        torch.manual_seed(TARGET_SEED)
        target_full = torch.randn(batch_size, d_hid, device=self.device)

        ref_input = input_full.clone().requires_grad_(True)
        ref_target = target_full.clone()
        ref_grads, _ = self._run_reference_microbatched(
            ref_model, ref_input, ref_target
        )

        stage = self._make_stage(
            pp_model=pp_model,
            stage_index=pp_rank,
            num_stages=n_layers,
            pp_group=pp_group,
            tp_mesh=fsdp_mesh,
        )
        stage.submod.zero_grad(set_to_none=True)

        pp_input = input_full.clone().requires_grad_(True)
        pp_target = target_full.clone()

        schedule = Schedule1F1B(
            stage,
            n_microbatches=n_microbatches,
            loss_fn=_loss_fn,
        )

        if stage.is_first and stage.is_last:
            losses: list[torch.Tensor] = []
            schedule.step(pp_input, target=pp_target, losses=losses)
        elif stage.is_first:
            schedule.step(pp_input)
        elif stage.is_last:
            losses = []
            schedule.step(target=pp_target, losses=losses)
        else:
            schedule.step()

        # Compare gradients (FSDP grads are DTensors, compare full tensors)
        stage_module = pp_model.get_submodule(f"layers.{pp_rank}")
        for name, param in stage_module.named_parameters():
            param_fqn = f"layers.{pp_rank}.{name}"
            self.assertIsNotNone(param.grad, f"Missing PP grad for {param_fqn}")
            ref_grad = ref_grads[param_fqn]
            self.assertIsNotNone(ref_grad, f"Missing ref grad for {param_fqn}")
            pp_grad = param.grad
            if hasattr(pp_grad, "full_tensor"):
                pp_grad = pp_grad.full_tensor()
            torch.testing.assert_close(
                pp_grad,
                ref_grad,
                msg=f"Gradient mismatch for {param_fqn}",
            )

    @_requires_multi_gpu
    def test_fsdp_pp(self):
        """FSDP + PP with spmd_types annotations."""
        self._run_fsdp_pp_correctness()


if __name__ == "__main__":
    run_tests()
