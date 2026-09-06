# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from __future__ import annotations

import copy
import functools
from typing import cast, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.pipelining._utils import _DTensorMeta
from torch.distributed.pipelining.schedules import (
    Schedule1F1B,
    ScheduleDualPipeV,
    ScheduleInterleaved1F1B,
    ScheduleZBVZeroBubble,
)
from torch.distributed.pipelining.stage import PipelineStage
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
    SequenceParallel,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skip_but_pass_in_sandcastle_if,
)


if TYPE_CHECKING:
    from collections.abc import Callable


d_hid = 256
batch_size = 64
n_microbatches = 4

StageStaticMeta = tuple[DTensor, DTensor, DTensor | None, DTensor | None]
StageStaticMetaMap = dict[int, StageStaticMeta]

MODEL_SEED = 0
INPUT_SEED = 42
TARGET_SEED = 123


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x
            * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            * self.weight
        )


class NormMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
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


def apply_tp_replicate(block: NormMLP, tp_mesh: DeviceMesh) -> None:
    parallelize_module(
        block,
        tp_mesh,
        {
            "fc1": ColwiseParallel(input_layouts=Replicate(), use_local_output=False),
            "fc2": RowwiseParallel(output_layouts=Replicate(), use_local_output=False),
        },
    )
    for name, param in block.norm.named_parameters():
        block.norm.register_parameter(
            name,
            nn.Parameter(
                DTensor.from_local(param, tp_mesh, [Replicate()], run_check=False)
            ),
        )


def apply_tp_shard(block: NormMLP, tp_mesh: DeviceMesh) -> None:
    parallelize_module(
        block,
        tp_mesh,
        {
            "norm": SequenceParallel(use_local_output=False),
            "fc1": ColwiseParallel(
                input_layouts=Shard(1),
                use_local_output=False,  # type: ignore[arg-type]
            ),
            "fc2": RowwiseParallel(
                output_layouts=Shard(1),
                use_local_output=False,  # type: ignore[arg-type]
            ),
        },
    )


def _loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (output - target).pow(2).mean()


def _requires_multi_gpu(func):
    @skip_but_pass_in_sandcastle_if(
        torch.accelerator.device_count() < 4,
        "Test requires 4+ GPUs",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class DTensorPPIntegrationBase(MultiProcContinuousTest):
    world_size = 4

    @classmethod
    def backend_str(cls) -> str:
        return dist.get_default_backend_for_device(cls.device_type)

    def _rank_device(self, device: str) -> torch.device:
        device_type = torch.device(device).type
        return torch.device(device_type, self.rank)

    def init_pg(self, rank_device: torch.device) -> None:
        if torch.accelerator.is_available():
            torch.accelerator.set_device_index(rank_device)

    def _make_mesh(self) -> DeviceMesh:
        return init_device_mesh(self.device_type, (2, 2), mesh_dim_names=("pp", "tp"))

    def _build_baseline_and_clones(
        self,
        n_layers: int,
        tp_mesh: DeviceMesh,
        apply_tp: Callable,
        *,
        rank_device: torch.device,
    ) -> tuple[NormMLPStack, NormMLPStack]:
        torch.manual_seed(MODEL_SEED)
        with torch.device(rank_device):
            ref_model = NormMLPStack(d_hid, n_layers)

        pp_model = copy.deepcopy(ref_model)

        for layer in ref_model.layers:
            apply_tp(layer, tp_mesh)
        for layer in pp_model.layers:
            apply_tp(layer, tp_mesh)

        return ref_model, pp_model

    def _make_full_batch_tensors(
        self, *, rank_device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(INPUT_SEED)
        input_full = torch.randn(batch_size, d_hid, device=rank_device)
        torch.manual_seed(TARGET_SEED)
        target_full = torch.randn(batch_size, d_hid, device=rank_device)
        return input_full, target_full

    def _run_reference_microbatched(
        self,
        ref_model: NormMLPStack,
        input_dt: DTensor,
        target_dt: DTensor,
        *,
        rank_device: torch.device,
        null_boundary_grads: bool = True,
    ) -> tuple[
        dict[str, torch.Tensor | None],
        StageStaticMetaMap,
    ]:
        ref_model.zero_grad(set_to_none=True)

        input_chunks = torch.tensor_split(input_dt, n_microbatches)
        target_chunks = torch.tensor_split(target_dt, n_microbatches)

        captured_io: dict[int, tuple[DTensor, DTensor]] = {}
        captured_grads: dict[int, dict[str, DTensor | None]] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        def _make_grad_hook(stage_idx: int, key: str) -> Callable[[torch.Tensor], None]:
            def _hook(grad: torch.Tensor) -> None:
                captured_grads[stage_idx][key] = self._clone_preserving_placement(grad)

            return _hook

        for stage_idx, stage_mod in enumerate(ref_model.layers):

            def _capture_io(
                _module: nn.Module,
                args: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                idx: int = stage_idx,
            ) -> None:
                stage_input = cast(DTensor, args[0])
                stage_output = cast(DTensor, output)
                captured_io[idx] = (stage_input, stage_output)
                captured_grads[idx] = {"input": None, "output": None}

                for tensor, key in (
                    (stage_input, "input"),
                    (stage_output, "output"),
                ):
                    if tensor.requires_grad:
                        hooks.append(tensor.register_hook(_make_grad_hook(idx, key)))

            hooks.append(stage_mod.register_forward_hook(_capture_io))

        static_stage_meta: StageStaticMetaMap = {}

        for mb_idx, (input_chunk, target_chunk) in enumerate(
            zip(input_chunks, target_chunks)
        ):
            output = ref_model(input_chunk)
            loss = _loss_fn(output, target_chunk)
            loss.backward()

            if mb_idx == 0:
                for stage_idx in range(len(ref_model.layers)):
                    stage_input, stage_output = captured_io[stage_idx]
                    input_args = self._empty_dt_from(
                        stage_input,
                        stage_input.requires_grad,
                        rank_device=rank_device,
                    )
                    output_args = self._empty_dt_from(
                        stage_output,
                        stage_output.requires_grad,
                        rank_device=rank_device,
                    )
                    input_grad_dt = captured_grads[stage_idx]["input"]
                    output_grad_dt = captured_grads[stage_idx]["output"]
                    input_grads = (
                        self._empty_dt_from(
                            cast(DTensor, input_grad_dt), False, rank_device=rank_device
                        )
                        if input_grad_dt is not None
                        else None
                    )
                    output_grads = (
                        self._empty_dt_from(
                            cast(DTensor, output_grad_dt),
                            False,
                            rank_device=rank_device,
                        )
                        if output_grad_dt is not None
                        else None
                    )

                    if null_boundary_grads:
                        if stage_idx == 0:
                            input_grads = None
                        if stage_idx == (len(ref_model.layers) - 1):
                            output_grads = None

                    static_stage_meta[stage_idx] = (
                        input_args,
                        output_args,
                        input_grads,
                        output_grads,
                    )

                for hook in hooks:
                    hook.remove()
                hooks = []

        for param in ref_model.parameters():
            if param.grad is not None:
                param.grad.div_(n_microbatches)

        ref_grads = {
            name: self._clone_preserving_placement(grad) if grad is not None else grad
            for name, grad in (
                (param_name, param.grad)
                for param_name, param in ref_model.named_parameters()
            )
        }
        return ref_grads, static_stage_meta

    @staticmethod
    def _clone_preserving_placement(grad: torch.Tensor) -> torch.Tensor:
        """Clone a gradient tensor, preserving DTensor placement.

        ``DTensor.clone()`` triggers all-reduce for ``Partial`` placements,
        converting them to ``Replicate``.  This helper clones only the local
        tensor and reconstructs the DTensor with the original placement so
        that downstream comparisons see the true placement.
        """
        if isinstance(grad, DTensor):
            local_clone = grad.to_local().clone()
            return DTensor.from_local(
                local_clone,
                grad.device_mesh,
                list(grad.placements),
                shape=grad.shape,
                stride=grad.stride(),
                run_check=False,
            )
        return grad.clone()

    def _empty_dt_from(
        self, dt: DTensor, requires_grad: bool, *, rank_device: torch.device
    ) -> DTensor:
        meta = _DTensorMeta.from_dtensor(dt)
        empty_local = torch.empty(meta.shape, dtype=meta.dtype, device=rank_device)
        result = DTensor.from_local(
            empty_local,
            dt.device_mesh,
            list(meta.placements),
            shape=meta.global_shape,
            stride=meta.global_stride,
            run_check=False,
        )
        if requires_grad:
            result.requires_grad_(True)
        return result

    def _make_stage_for_pp_run(
        self,
        pp_model: NormMLPStack,
        stage_index: int,
        num_stages: int,
        pp_group: dist.ProcessGroup,
        tp_mesh: DeviceMesh,
        static_stage_meta: StageStaticMetaMap | None,
        *,
        rank_device: torch.device,
    ) -> PipelineStage:
        stage_module = pp_model.get_submodule(f"layers.{stage_index}")
        if static_stage_meta is None:
            return PipelineStage(
                submodule=stage_module,
                stage_index=stage_index,
                num_stages=num_stages,
                device=rank_device,
                group=pp_group,
                get_mesh=lambda _dim_names, _layout: tp_mesh,
            )

        input_args, output_args, input_grads, output_grads = static_stage_meta[
            stage_index
        ]

        return PipelineStage(
            submodule=stage_module,
            stage_index=stage_index,
            num_stages=num_stages,
            device=rank_device,
            group=pp_group,
            input_args=(input_args,),
            output_args=(output_args,),
            input_grads=(input_grads,),
            output_grads=(output_grads,),
            get_mesh=lambda _dim_names, _layout: tp_mesh,
        )

    def _assert_grad_close(
        self,
        pp_grad: torch.Tensor,
        ref_grad: torch.Tensor,
        param_fqn: str,
    ) -> None:
        pp_for_compare = pp_grad
        ref_for_compare = ref_grad

        if isinstance(pp_grad, DTensor) and isinstance(ref_grad, DTensor):
            pp_meta = _DTensorMeta.from_dtensor(pp_grad)
            ref_meta = _DTensorMeta.from_dtensor(ref_grad)
            self.assertEqual(
                pp_meta.get_diff(ref_meta),
                [],
                lambda msg: f"{msg}\nDTensor metadata mismatch for {param_fqn}",
            )
            pp_for_compare = pp_grad.to_local()
            ref_for_compare = ref_grad.to_local()

        torch.testing.assert_close(
            pp_for_compare,
            ref_for_compare,
            msg=f"Gradient mismatch for {param_fqn}",
        )

    def _assert_stage_grad_parity(
        self,
        pp_model: NormMLPStack,
        ref_grads: dict[str, torch.Tensor | None],
        stage_index: int,
    ) -> None:
        stage_module = pp_model.get_submodule(f"layers.{stage_index}")
        for name, param in stage_module.named_parameters():
            param_fqn = f"layers.{stage_index}.{name}"
            self.assertIsNotNone(
                param.grad, lambda msg: f"{msg}\nMissing PP grad for {param_fqn}"
            )
            self.assertIsNotNone(
                ref_grads[param_fqn],
                lambda msg: f"{msg}\nMissing reference grad for {param_fqn}",
            )
            pp_grad = cast(torch.Tensor, param.grad)
            ref_grad = cast(torch.Tensor, ref_grads[param_fqn])
            self._assert_grad_close(pp_grad, ref_grad, param_fqn)

    def _compute_stage_indices(
        self,
        schedule_class: type,
        pp_rank: int,
        pp_size: int,
        num_stages: int,
    ) -> list[int]:
        if schedule_class is Schedule1F1B:
            return [pp_rank]
        if schedule_class is ScheduleInterleaved1F1B:
            return [pp_rank + i * pp_size for i in range(num_stages // pp_size)]
        return [pp_rank, num_stages - 1 - pp_rank]

    def _execute_schedule_step(
        self,
        schedule,
        stages: list[PipelineStage],
        pp_input: DTensor,
        pp_target: DTensor,
    ) -> None:
        has_first = any(stage.is_first for stage in stages)
        has_last = any(stage.is_last for stage in stages)

        if has_first and has_last:
            losses: list[torch.Tensor] = []
            schedule.step(pp_input, target=pp_target, losses=losses)
        elif has_first:
            schedule.step(pp_input)
        elif has_last:
            losses: list[torch.Tensor] = []
            schedule.step(target=pp_target, losses=losses)
        else:
            schedule.step()

    def _run_training_correctness(
        self,
        device: str,
        schedule_class: type,
        apply_tp: Callable,
        placements: list,
        *,
        use_static_metadata: bool = True,
        null_boundary_grads: bool = False,
    ) -> None:
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)

        mesh = self._make_mesh()
        tp_mesh = mesh["tp"]
        pp_mesh = mesh["pp"]
        pp_group = pp_mesh.get_group()
        pp_rank = pp_mesh.get_local_rank()
        pp_size = pp_mesh.size()

        n_layers = 2 if schedule_class is Schedule1F1B else 4

        ref_model, pp_model = self._build_baseline_and_clones(
            n_layers=n_layers,
            tp_mesh=tp_mesh,
            apply_tp=apply_tp,
            rank_device=rank_device,
        )

        input_full, target_full = self._make_full_batch_tensors(rank_device=rank_device)

        ref_input = distribute_tensor(
            input_full.clone(),
            tp_mesh,
            placements,
        )
        ref_input.requires_grad_(True)
        ref_target = distribute_tensor(
            target_full.clone(),
            tp_mesh,
            placements,
        )

        ref_grads, static_stage_meta = self._run_reference_microbatched(
            ref_model,
            ref_input,
            ref_target,
            null_boundary_grads=null_boundary_grads,
            rank_device=rank_device,
        )

        pp_static_stage_meta = static_stage_meta if use_static_metadata else None

        pp_input = distribute_tensor(
            input_full.clone(),
            tp_mesh,
            placements,
        )
        pp_input.requires_grad_(True)
        pp_target = distribute_tensor(
            target_full.clone(),
            tp_mesh,
            placements,
        )

        num_stages = n_layers
        stage_indices = self._compute_stage_indices(
            schedule_class,
            pp_rank,
            pp_size,
            num_stages,
        )

        stages = [
            self._make_stage_for_pp_run(
                pp_model=pp_model,
                stage_index=stage_index,
                num_stages=num_stages,
                pp_group=pp_group,
                tp_mesh=tp_mesh,
                static_stage_meta=pp_static_stage_meta,
                rank_device=rank_device,
            )
            for stage_index in stage_indices
        ]

        for stage in stages:
            stage.submod.zero_grad(set_to_none=True)

        if schedule_class is Schedule1F1B:
            schedule = schedule_class(
                stages[0],
                n_microbatches=n_microbatches,
                loss_fn=_loss_fn,
            )
        else:
            schedule = schedule_class(
                cast(list[PipelineStage], stages),  # type: ignore[arg-type]
                n_microbatches=n_microbatches,
                loss_fn=_loss_fn,
            )

        self._execute_schedule_step(schedule, stages, pp_input, pp_target)

        for stage_index in stage_indices:
            self._assert_stage_grad_parity(
                pp_model=pp_model,
                ref_grads=ref_grads,
                stage_index=stage_index,
            )

    def _run_inference_only_equivalence(
        self,
        device: str,
        apply_tp: Callable,
        placements: list,
        *,
        use_static_metadata: bool = False,
    ) -> None:
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)

        mesh = self._make_mesh()
        tp_mesh = mesh["tp"]
        pp_mesh = mesh["pp"]
        pp_group = pp_mesh.get_group()
        pp_rank = pp_mesh.get_local_rank()

        ref_model, pp_model = self._build_baseline_and_clones(
            n_layers=2,
            tp_mesh=tp_mesh,
            apply_tp=apply_tp,
            rank_device=rank_device,
        )

        input_full, target_full = self._make_full_batch_tensors(rank_device=rank_device)

        ref_input = distribute_tensor(
            input_full.clone(),
            tp_mesh,
            placements,
        )

        static_stage_meta: StageStaticMetaMap | None = None

        if use_static_metadata:
            ref_input_for_meta = distribute_tensor(
                input_full.clone(),
                tp_mesh,
                placements,
            )
            ref_input_for_meta.requires_grad_(True)
            ref_target_for_meta = distribute_tensor(
                target_full.clone(),
                tp_mesh,
                placements,
            )
            _, captured_meta = self._run_reference_microbatched(
                ref_model,
                ref_input_for_meta,
                ref_target_for_meta,
                rank_device=rank_device,
            )

            static_stage_meta = {
                stage_idx: (
                    self._empty_dt_from(stage_meta[0], False, rank_device=rank_device),
                    self._empty_dt_from(stage_meta[1], False, rank_device=rank_device),
                    None,
                    None,
                )
                for stage_idx, stage_meta in captured_meta.items()
            }

            with torch.no_grad():
                ref_output = ref_model(ref_input)
        else:
            with torch.no_grad():
                ref_output = ref_model(ref_input)

        pp_input = distribute_tensor(
            input_full.clone(),
            tp_mesh,
            placements,
        )
        pp_input.requires_grad_(False)

        stage = self._make_stage_for_pp_run(
            pp_model=pp_model,
            stage_index=pp_rank,
            num_stages=2,
            pp_group=pp_group,
            tp_mesh=tp_mesh,
            static_stage_meta=static_stage_meta,
            rank_device=rank_device,
        )
        schedule = Schedule1F1B(
            stage,
            n_microbatches=n_microbatches,
            loss_fn=None,
        )

        with torch.no_grad():
            pp_output = schedule.eval(pp_input)

        if stage.is_last:
            self.assertIsNotNone(pp_output)
            self.assertIsInstance(pp_output, DTensor)
            self.assertIsInstance(ref_output, DTensor)
            self._assert_grad_close(
                cast(torch.Tensor, pp_output),
                cast(torch.Tensor, ref_output),
                "inference_output",
            )


class TestDTensorPPModes(DTensorPPIntegrationBase):
    hw_classification = HardwareClassification.ACCELERATOR

    @_requires_multi_gpu
    def test_static_mode_replicate(self, device):
        self._run_training_correctness(
            device,
            Schedule1F1B,
            apply_tp_replicate,
            [Replicate()],
        )

    @_requires_multi_gpu
    def test_static_mode_none_grad_slots_replicate(self, device):
        self._run_training_correctness(
            device,
            Schedule1F1B,
            apply_tp_replicate,
            [Replicate()],
            null_boundary_grads=True,
        )

    @_requires_multi_gpu
    def test_static_mode_none_grad_slots_shard(self, device):
        self._run_training_correctness(
            device,
            Schedule1F1B,
            apply_tp_shard,
            [Shard(1)],
            null_boundary_grads=True,
        )

    @_requires_multi_gpu
    def test_static_mode_shard(self, device):
        self._run_training_correctness(
            device,
            Schedule1F1B,
            apply_tp_shard,
            [Shard(1)],
        )

    @_requires_multi_gpu
    def test_dynamic_mode_inference_only_replicate(self, device):
        self._run_inference_only_equivalence(
            device,
            apply_tp_replicate,
            [Replicate()],
        )

    @_requires_multi_gpu
    def test_static_mode_inference_only_replicate(self, device):
        self._run_inference_only_equivalence(
            device,
            apply_tp_replicate,
            [Replicate()],
            use_static_metadata=True,
        )

    @_requires_multi_gpu
    def test_dynamic_mode_inference_only_shard(self, device):
        self._run_inference_only_equivalence(
            device,
            apply_tp_shard,
            [Shard(1)],
        )

    @_requires_multi_gpu
    def test_static_mode_inference_only_shard(self, device):
        self._run_inference_only_equivalence(
            device,
            apply_tp_shard,
            [Shard(1)],
            use_static_metadata=True,
        )

    @_requires_multi_gpu
    def test_dynamic_mode_replicate(self, device):
        self._run_training_correctness(
            device,
            Schedule1F1B,
            apply_tp_replicate,
            [Replicate()],
            use_static_metadata=False,
        )

    @_requires_multi_gpu
    def test_dynamic_mode_shard(self, device):
        self._run_training_correctness(
            device,
            Schedule1F1B,
            apply_tp_shard,
            [Shard(1)],
            use_static_metadata=False,
        )

    @_requires_multi_gpu
    def test_static_mode_interleaved1f1b_replicate(self, device):
        self._run_training_correctness(
            device,
            ScheduleInterleaved1F1B,
            apply_tp_replicate,
            [Replicate()],
        )

    @_requires_multi_gpu
    def test_static_mode_zbv_zero_bubble_replicate(self, device):
        self._run_training_correctness(
            device,
            ScheduleZBVZeroBubble,
            apply_tp_replicate,
            [Replicate()],
        )

    @_requires_multi_gpu
    def test_static_mode_dualpipev_replicate(self, device):
        self._run_training_correctness(
            device,
            ScheduleDualPipeV,
            apply_tp_replicate,
            [Replicate()],
        )

    @_requires_multi_gpu
    def test_dynamic_mode_interleaved1f1b_shard(self, device):
        self._run_training_correctness(
            device,
            ScheduleInterleaved1F1B,
            apply_tp_shard,
            [Shard(1)],
            use_static_metadata=False,
        )

    @_requires_multi_gpu
    def test_dynamic_mode_zbv_zero_bubble_shard(self, device):
        self._run_training_correctness(
            device,
            ScheduleZBVZeroBubble,
            apply_tp_shard,
            [Shard(1)],
            use_static_metadata=False,
        )

    @_requires_multi_gpu
    def test_dynamic_mode_dualpipev_shard(self, device):
        self._run_training_correctness(
            device,
            ScheduleDualPipeV,
            apply_tp_shard,
            [Shard(1)],
            use_static_metadata=False,
        )


instantiate_device_type_tests(
    TestDTensorPPModes,
    globals(),
    except_for=["cpu"],
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests()
