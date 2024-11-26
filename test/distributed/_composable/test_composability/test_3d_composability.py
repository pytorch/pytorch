# Owner(s): ["oncall: distributed"]

import copy
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import with_comms


class Test3DTraining(FSDPTest):
    global num_layers
    num_layers = 8

    @with_comms
    @requires_nccl()
    @skip_if_lt_x_gpu(8)
    def test_3d(self):
        self.run_subtests(
            {
                "scheduleClass": [
                    "GPipe",
                    "1F1B",
                    "Interleaved1F1B",
                    "LoopedBFS",
                    "InterleavedZeroBubble",
                ],
            },
            self._test_3d,
        )

    def _test_3d(self, scheduleClass):
        parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            tp=2,
            pp=2,
            world_size=min(8, torch.cuda.device_count()),
            enable_loss_parallel=True,
            schedule=scheduleClass,
        )
        device = torch.device("cuda")
        world_mesh = parallel_dims.build_mesh(device_type="cuda")

        if parallel_dims.pp_enabled:
            pp_mesh = world_mesh["pp"]

        full_model = nn.ModuleList([MLPModule(8) for _ in range(num_layers)])
        model = nn.Sequential(*copy.deepcopy(full_model))
        model.to(device)

        def loss_fn(pred, labels):
            return pred.sum()

        if parallel_dims.pp_enabled:
            stages, models = self.pipeline_llama_manual_split(
                model, pp_mesh, parallel_dims, "cuda"
            )
            schedule_class = get_schedule_class(parallel_dims.schedule)
            if schedule_class in [PipelineScheduleSingle, PipelineScheduleMulti]:
                raise ValueError(
                    f"{schedule_class} is not supported as we do not support custom CSV schedules."
                )
            pp_schedule = schedule_class(
                (
                    stages
                    if issubclass(schedule_class, PipelineScheduleMulti)
                    else stages[0]
                ),
                n_microbatches=2,
                loss_fn=loss_fn,
            )
            for model in models:
                if parallel_dims.tp_enabled:
                    self.apply_tp(
                        model,
                        world_mesh["tp"],
                        loss_parallel=parallel_dims.loss_parallel_enabled,
                        enable_async_tp=False,
                    )

                if parallel_dims.dp_shard_enabled:
                    dp_mesh_dim_names = (
                        ("dp_replicate", "dp_shard")
                        if parallel_dims.dp_replicate_enabled
                        else ("dp",)
                    )
                    dp_mesh = world_mesh[dp_mesh_dim_names]
                    self.apply_fsdp(
                        model,
                        dp_mesh,
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                        tp_enabled=parallel_dims.tp_enabled,
                        pp_enabled=parallel_dims.pp_enabled,
                        cpu_offload=None,
                    )
                model.to_empty(device=device)
                model.train()

        else:
            if parallel_dims.tp_enabled:
                self.apply_tp(
                    model,
                    world_mesh["tp"],
                    loss_parallel=parallel_dims.loss_parallel_enabled,
                    enable_async_tp=False,
                )

            if parallel_dims.dp_shard_enabled:
                dp_mesh_dim_names = (
                    ("dp_replicate", "dp_shard")
                    if parallel_dims.dp_replicate_enabled
                    else ("dp",)
                )
                dp_mesh = world_mesh[dp_mesh_dim_names]
                self.apply_fsdp(
                    model,
                    world_mesh["dp"],
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    tp_enabled=parallel_dims.tp_enabled,
                    pp_enabled=parallel_dims.pp_enabled,
                    cpu_offload=None,
                )
            model.to_empty(device=device)
            model.train()
            models = [model]

        optimizer_kwargs = {
            "lr": 0.01,
            "betas": (0.9, 0.95),
            "weight_decay": 0.1,
            "fused": False,
            "foreach": True,
        }
        optimizers = [
            torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            for model in models
        ]

        for train_step in range(10):
            for optimizer in optimizers:
                optimizer.zero_grad()

            # Generate random data
            data = torch.rand(4, 8, device=device)
            labels = torch.randint(0, 2, (4,), device=device)
            if parallel_dims.pp_enabled:
                is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1
                if pp_mesh.get_local_rank() == 0:
                    pp_schedule.step(data)
                elif is_last_stage:
                    losses = []
                    pp_schedule.step(target=labels, losses=losses)
                else:
                    pp_schedule.step()

                # accumulate losses across pipeline microbatches
                loss = (
                    torch.mean(torch.stack(losses))
                    if is_last_stage
                    else torch.Tensor([-1.0])
                )
            else:
                # Forward
                output = model(data)
                # Compute loss
                loss = loss_fn(output, labels)
                # Backward
                loss.backward()

            self.clip_grad_norm_(
                [param for model in models for param in model.parameters()],
                1.0,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            for optimizer in optimizers:
                optimizer.step()

    def pipeline_llama_manual_split(
        self,
        whole_model: nn.Module,
        pp_mesh: DeviceMesh,
        parallel_dims,
        device,
    ):
        pp_rank = pp_mesh.get_local_rank()
        pp_size = pp_mesh.size()
        splits = self.generate_split_points(parallel_dims.schedule, parallel_dims.pp)

        def _build_stage(
            stage_idx, start_layer, stop_layer, is_first=False, is_last=False
        ):
            model = copy.deepcopy(whole_model)
            if not is_first:
                model.tok_embeddings = None

            start_idx = int(start_layer.split(".")[-1]) if start_layer else 0
            stop_idx = int(stop_layer.split(".")[-1]) if stop_layer else num_layers
            model = model[start_idx:stop_idx]

            if not is_last:
                model.norm = None
                model.output = None

            stage = PipelineStage(
                model,
                stage_idx,
                num_stages,
                device,
                group=pp_mesh.get_group("pp"),
            )
            return stage, model

        num_stages = len(splits) + 1
        stage_idx = pp_rank

        stages = []
        models = []
        stages_per_rank = num_stages // pp_size

        for s in range(stages_per_rank):
            stage_idx = pp_rank + s * pp_size
            start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
            stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
            stage, model_chunk = _build_stage(
                stage_idx,
                start_layer,
                stop_layer,
                is_first=stage_idx == 0,
                is_last=stage_idx == num_stages - 1,
            )
            stages.append(stage)
            models.append(model_chunk)
        return stages, models

    def generate_split_points(self, schedule_class, pp_dim):
        schedule_class = get_schedule_class(schedule_class)
        if issubclass(schedule_class, PipelineScheduleSingle):
            num_stages_per_rank = 1
        elif issubclass(schedule_class, PipelineScheduleMulti):
            # Multi-stage schedules support more than 2 stages per rank, but this is the default if
            # no pipeline split is specified
            num_stages_per_rank = 2
        else:
            raise ValueError("Unsupported pipeline schedule")
        total_stages = pp_dim * num_stages_per_rank

        if total_stages > num_layers:
            raise ValueError("Total stages cannot be greater than the number of layers")

        base_interval = num_layers // total_stages
        extra_layers = num_layers % total_stages

        splits = []
        current_layer = 0
        for i in range(total_stages - 1):
            if i == 0:
                current_layer += base_interval
            else:
                # Middle stages get an extra layer if there are any remaining
                if extra_layers > 0:
                    current_layer += base_interval + 1
                    extra_layers -= 1
                else:
                    current_layer += base_interval
            splits.append("layers." + str(current_layer))
        return splits

    def apply_tp(
        self,
        model: nn.Module,
        tp_mesh: DeviceMesh,
        loss_parallel: bool,
        enable_async_tp: bool,
    ):
        """Apply tensor parallelism."""
        # 1. Parallelize the embedding and shard its outputs (which are the first
        # transformer block's inputs)
        # 2. Parallelize the root norm layer over the sequence dim
        # 3. Parallelize the final linear output layer
        parallelize_module(
            model,
            tp_mesh,
            {
                "tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                "norm": SequenceParallel(),
                "output": ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Shard(-1) if loss_parallel else Replicate(),
                    use_local_output=not loss_parallel,
                ),
            },
        )

        # Parallel styles used for transformer block linear weights and their
        # inputs may be different for float8 linears
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

        # Apply tensor + sequence parallelism to every transformer block
        # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
        #       by folding (and unfolding) the batch dimension and the sequence dimension.
        #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
        for layer_id in range(len(model)):
            transformer_block = model[layer_id]
            layer_plan = {
                "attention_norm": SequenceParallel(),
                "attention": prepare_module_input(
                    input_layouts=(Shard(1), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attention.wq": colwise_parallel(),
                "attention.wk": colwise_parallel(),
                "attention.wv": colwise_parallel(),
                "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
                "ffn_norm": SequenceParallel(),
                "feed_forward": prepare_module_input(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": colwise_parallel(),
                "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                "feed_forward.w3": colwise_parallel(),
            }

            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        if enable_async_tp:
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            torch._inductor.config._micro_pipeline_tp = True
            enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    def apply_fsdp(
        self,
        model: nn.Module,
        dp_mesh: DeviceMesh,
        param_dtype: torch.dtype,
        reduce_dtype: torch.dtype,
        tp_enabled: bool,
        pp_enabled: bool,
        cpu_offload: bool = False,
    ):
        """
        Apply data parallelism to the model. FSDP2 is used here.
        """
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        if cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()

        for layer_id in range(len(model)):
            transformer_block = model[layer_id]
            if pp_enabled:
                # For PP, do not reshard after forward to avoid per-microbatch
                # all-gathers, which can be expensive and non-overlapped
                reshard_after_forward = False
            else:
                # As an optimization, do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately
                reshard_after_forward = int(layer_id) < 7
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)

    def clip_grad_norm_(
        self,
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
        pp_mesh: Optional[DeviceMesh] = None,
    ) -> torch.Tensor:
        """
        Clip the gradient norm of an iterable of parameters.

        Gradient norm clipping requires computing the gradient norm over the entire model.
        `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
        We need to manually reduce the gradient norm across PP stages.
        See https://github.com/pytorch/torchtitan/issues/596 for details.
        """
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(
            grads, norm_type, error_if_nonfinite, foreach
        )

        if pp_mesh is not None:
            if isinstance(total_norm, DTensor):
                # will reach here if PP + other parallelism is used. If only using PP, total_norm will be a local tensor

                # if total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`
                # we can simply reduce the DTensor to get the total norm in this tensor's process group
                # and then convert it to a local tensor
                total_norm = total_norm.full_tensor()

            if math.isinf(norm_type):
                dist.all_reduce(
                    total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group()
                )
            else:
                total_norm **= norm_type
                dist.all_reduce(
                    total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group()
                )
                total_norm **= 1.0 / norm_type

        torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
        return total_norm


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool = True
    schedule: str = "1F1B"

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, tp, pp = (
            self.dp_replicate,
            self.dp_shard,
            self.tp,
            self.pp,
        )
        for d in (dp_replicate, tp, pp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"
        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."

        dp = dp_replicate * dp_shard
        if dp < 0:
            dp = self.world_size // (tp * pp)
            self.dp_shard = dp_shard = dp // dp_replicate

        assert dp_replicate >= 1
        assert dp_shard >= 1
        assert tp >= 1, tp
        assert pp >= 1, pp
        assert dp_replicate * dp_shard * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.tp],
            ["pp", "dp_replicate", "dp_shard", "tp"],
        ):
            if d > 1:
                dims.append(d)
                if (name == "dp_replicate" and self.dp_shard == 1) or (
                    name == "dp_shard" and self.dp_replicate == 1
                ):
                    names.append("dp")
                else:
                    names.append(name)

        # logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)
        # Create all the submesh here to ensure all required process groups are
        # initialized
        if self.dp_replicate > 1 and self.dp_shard > 1:
            mesh["dp_replicate", "dp_shard"]._flatten(mesh_dim_name="dp")

        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def non_data_parallel_size(self):
        return self.tp * self.pp


# MLP Layer
class MLPModule(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


if __name__ == "__main__":
    run_tests()
