# Owner(s): ["oncall: distributed"]

import copy
import logging
import os
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import (
    pipeline,
    PipelineStage,
    Schedule1F1B,
    ScheduleFlexibleInterleaved1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    SplitPoint,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


@dataclass
class ParallelDims:
    dp: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool
    dp_type: str

    def __post_init__(self):
        self.dp_type = self.dp_type.lower()
        self._validate()

    def _validate(self):
        dp, tp, pp = self.dp, self.tp, self.pp
        if dp == -1:
            self.dp = dp = self.world_size // (tp * pp)
        assert dp >= 1, dp
        assert tp >= 1, tp
        assert pp >= 1, pp
        assert (
            dp * tp * pp == self.world_size
        ), f"Invalid parallel dims: dp({dp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        assert self.dp_type in ("fsdp", "ddp")

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp, self.tp], ["pp", "dp", "tp"], strict=True
        ):
            if d > 1:
                dims.append(d)
                names.append(name)
        # logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        return init_device_mesh(device_type, dims, mesh_dim_names=names)

    @property
    def dp_enabled(self):
        return self.dp > 1

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
    def model_parallel_size(self):
        return self.tp * self.pp


class Test3DComposability(FSDPTest):
    logger = logging.getLogger()
    logger.info("Starting job: 3D composability test")
    DeviceType = Union[int, str, torch.device]

    def _init_logger(self):
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        os.environ["KINETO_LOG_LEVEL"] = "5"

    @skip_if_lt_x_gpu(4)
    def test_3d_composability(self):
        self.run_subtests(
            {
                "dp_degree": [
                    1,
                ],
                "tp_degree": [
                    2,
                ],
                "pp_degree": [
                    2,
                ],
            },
            self._test_3d_composability,
        )
        self.logger.info("Finished job: 3D composability test")

    def _test_3d_composability(self, dp_degree, tp_degree, pp_degree):
        TRACE_BUFFER_SIZE = "TORCH_NCCL_TRACE_BUFFER_SIZE"
        TRACE_FILE = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
        DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
        ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
        SKIP_CLEANUP = "3"

        world_size = int(dp_degree * tp_degree * pp_degree)
        parallel_dims = ParallelDims(
            dp=dp_degree,
            tp=tp_degree,
            pp=pp_degree,
            world_size=world_size,
            enable_loss_parallel=False,
            dp_type="fsdp",
        )
        device = torch.device("cuda", index=dist.get_rank())
        torch.cuda.set_device(device)

        def _warn_overwrite_env(env, val):
            if env in os.environ:
                self.logger.warning("ENV will be overridden based on job config")
            os.environ[env] = val

        _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)
        _warn_overwrite_env(TRACE_BUFFER_SIZE, str(0))

        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        torch.manual_seed(0)
        model_args = ModelArgs(dropout_p=0.0)
        model = Transformer(model_args)

        world_mesh = parallel_dims.build_mesh(device_type="cuda")

        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        def loss_fn(pred, labels):
            return torch.nn.functional.cross_entropy(
                pred.flatten(0, 1), labels.flatten(0, 1)
            )

        if parallel_dims.pp_enabled:
            pp_mesh = world_mesh["pp"]
            pp_schedule, model_parts = self._pipeline_llama(
                model, pp_mesh, parallel_dims, device, model_args, loss_fn
            )
            for m in model_parts:
                self._parallelize_llama(m, world_mesh, parallel_dims)
                m.to_empty(device="cuda")
        else:
            self._parallelize_llama(model, world_mesh, parallel_dims)

    def _parallelize_llama(
        self,
        model: nn.Module,
        world_mesh: DeviceMesh,
        parallel_dims: ParallelDims,
    ):
        if parallel_dims.tp_enabled:
            self._apply_tp(
                model,
                world_mesh["tp"],
                loss_parallel=parallel_dims.loss_parallel_enabled,
                enable_float8=False,
                enable_async_tp=False,  # add config here
            )

        self._apply_compile(model)

        if parallel_dims.dp_enabled:
            if parallel_dims.dp_type == "fsdp":
                dp_mesh = world_mesh["dp"] if world_mesh.ndim > 1 else world_mesh
                assert dp_mesh.mesh_dim_names == ("dp",), dp_mesh.mesh_dim_names

                self._apply_fsdp(
                    model,
                    world_mesh["dp"],
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    tp_enabled=parallel_dims.tp_enabled,
                    pp_enabled=parallel_dims.pp_enabled,
                )
            else:
                self._apply_ddp(
                    model,
                    world_mesh,
                    enable_compile=True,
                    enable_compiled_autograd=False,
                )

    def _apply_tp(
        self,
        model: nn.Module,
        tp_mesh: DeviceMesh,
        loss_parallel: bool,
        enable_float8: bool,
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
        if enable_float8:
            from torchao.float8.float8_tensor_parallel import (
                Float8ColwiseParallel,
                Float8RowwiseParallel,
                PrepareFloat8ModuleInput,
            )

            rowwise_parallel, colwise_parallel, prepare_module_input = (
                Float8RowwiseParallel,
                Float8ColwiseParallel,
                PrepareFloat8ModuleInput,
            )
        else:
            rowwise_parallel, colwise_parallel, prepare_module_input = (
                RowwiseParallel,
                ColwiseParallel,
                PrepareModuleInput,
            )

        # Apply tensor + sequence parallelism to every transformer block
        # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
        #       by folding (and unfolding) the batch dimension and the sequence dimension.
        #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
        for layer_id, transformer_block in enumerate(model.layers):
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

        if enable_async_tp:
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            torch._inductor.config._micro_pipeline_tp = True
            enable_symm_mem_for_group(tp_mesh.get_group().group_name)
            torch.distributed.breakpoint()
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )
        self.logger.info(
            f"Applied {'Float8 ' if enable_float8 else ''}{'Async ' if enable_async_tp else ''}"  # noqa: G004
            "Tensor Parallelism to the model"
        )

    def _apply_fsdp(
        self,
        model: nn.Module,
        dp_mesh: DeviceMesh,
        param_dtype: torch.dtype,
        reduce_dtype: torch.dtype,
        tp_enabled: bool,
        pp_enabled: bool,
    ):
        """
        Apply data parallelism to the model. FSDP2 is used here.
        """
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

        for layer_id, transformer_block in enumerate(model.layers):
            if pp_enabled:
                # For PP, do not reshard after forward to avoid per-microbatch
                # all-gathers, which can be expensive and non-overlapped
                reshard_after_forward = False
            else:
                # As an optimization, do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately
                reshard_after_forward = int(layer_id) < len(model.layers) - 1
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)
        self.logger.info("Applied FSDP to the model")

    def _apply_ddp(
        self,
        model: nn.Module,
        dp_mesh: DeviceMesh,
        enable_compile: bool,
        enable_compiled_autograd: bool,
    ):
        """TODO: what does this part do?
        if enable_compile:
            if enable_compiled_autograd:
                torch._dynamo.config.optimize_ddp = (
                    "python_reducer_without_compiled_forward"
                )
            else:
                torch._dynamo.config.optimize_ddp = "ddp_optimizer"
        """
        replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

        self.logger.info("Applied DDP to the model")

    def _apply_compile(self, model: nn.Module):
        """
        Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
        repeated structure. Alternatively one can compile the whole model (after applying DP).
        """
        for layer_id, transformer_block in model.layers.named_children():
            transformer_block = torch.compile(transformer_block, fullgraph=True)
            model.layers.register_module(layer_id, transformer_block)

        self.logger.info("Compiling each TransformerBlock with torch.compile")

    def _pipeline_llama(
        self,
        model: nn.Module,
        pp_mesh: DeviceMesh,
        parallel_dims: ParallelDims,
        device: DeviceType,
        model_config: ModelArgs,
        loss_fn: Callable[..., torch.Tensor],
    ):
        split_mode = "tracer"
        valid_split_modes = ("manual", "tracer")
        if split_mode == "manual":
            stages, models = self._pipeline_llama_manual(
                model, pp_mesh, parallel_dims, device, model_config
            )
        """ TODO: enable tracer
        elif split_mode == "tracer":
            stages, models = self._pipeline_llama_tracer(
                model, pp_mesh, parallel_dims, device, model_config
            )
        """

        schedule_class = [
            "1f1b",
            "gpipe",
            "interleaved_1f1b",
            "flexible_interleaved_1f1b",
        ]
        pp_schedule = self._build_pipeline_schedule(
            parallel_dims.pp, schedule_class[3], stages, loss_fn
        )

        return pp_schedule, models

    def _pipeline_llama_manual(
        self,
        whole_model: nn.Module,
        pp_mesh: DeviceMesh,
        parallel_dims: ParallelDims,
        device: DeviceType,
        model_config: ModelArgs,
    ):
        """
        This API extracts one torch.nn.Module objects for the part of the model configured to run inside this stage.

        It wraps the model chunk in a ManualPipelineStage object and returns both the stage and model objects.

        The stage object is used to create a pipeline schedule, and the model object can be used for applying SPMD
        parallelism.
        """
        pp_rank = pp_mesh.get_local_rank()
        pp_size = pp_mesh.size()
        microbatches = parallel_dims.pp
        splits = ["layers.1", "layers.2", "layers.3"]

        def _build_stage(
            stage_idx, pp_mesh, start_layer, stop_layer, is_first=False, is_last=False
        ):
            model = copy.deepcopy(whole_model)
            if not is_first:
                model.tok_embeddings = None

            mesh_dict = {}
            mesh_id = 0
            for mesh_tensor in pp_mesh.mesh:
                mesh_dict[mesh_id] = mesh_tensor.item()
                mesh_id += 1

            drop_layers = start_layer is not None
            for layer_id, mesh_id in mesh_dict.items():
                # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
                if f"layers.{mesh_id}" == start_layer:
                    drop_layers = False
                if f"layers.{mesh_id}" == stop_layer:
                    drop_layers = True
                if drop_layers:
                    del model.layers[layer_id]
                    break

            if not is_last:
                model.norm = None
                model.output = None

            # TODO(whc) once ManualPipelineStage supports lazy shape inference, we can leave model on meta device longer and
            # get rid of the input shape hardcoded here. For now, it should not be a big deal since we only materialize the
            # layers of the model that map to this stage, not the whole model.
            if parallel_dims.dp_enabled:
                mp_dtype = torch.float32
            else:
                mp_dtype = torch.bfloat16
            batch_size = 8
            local_seq_len = int(2048 // parallel_dims.tp)
            layers_io_shape = (batch_size, local_seq_len, model_config.dim)
            output_layer_shape = (
                batch_size,
                2048,
                model_config.vocab_size,
            )
            if is_first:
                tokens_shape = (8, 2048)
                input = torch.randint(
                    model_config.vocab_size,
                    tokens_shape,
                    dtype=torch.int64,
                    device=device,
                )
            else:
                # later layers (assume all start w/ a transformer layer)
                input = torch.rand(layers_io_shape, dtype=mp_dtype, device=device)

            if is_last:
                output = torch.rand(
                    output_layer_shape, dtype=torch.float32, device=device
                )
            else:
                # earlier layers (assume all end in a transformer layer)
                output = torch.rand(layers_io_shape, dtype=mp_dtype, device=device)

            model.to_empty(device=device)
            stage = PipelineStage(
                model,
                stage_idx,
                num_stages,
                device,
                input_args=input.chunk(microbatches)[0],
                output_args=output.chunk(microbatches)[0],
                group=pp_mesh.get_group("pp"),
            )
            return stage, model

        num_stages = len(splits) + 1
        stage_idx = pp_rank

        stages = []
        models = []
        for stage_idx in self._stage_ids_this_rank(
            pp_rank, pp_size, num_stages, style="loop"
        ):
            start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
            stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
            stage, model_chunk = _build_stage(
                stage_idx,
                pp_mesh,
                start_layer,
                stop_layer,
                is_first=stage_idx == 0,
                is_last=stage_idx == num_stages - 1,
            )
            self.logger.info(
                f"PP rank {pp_rank} is building stage_idx {stage_idx}"  # noqa: G004
                f" with start_layer {start_layer}, stop_layer {stop_layer}: model chunk \n{model_chunk}"  # noqa: G004
            )
            stages.append(stage)
            models.append(model_chunk)
        return stages, models

    def _stage_ids_this_rank(
        self, pp_rank: int, pp_size: int, num_stages: int, style: str = "loop"
    ) -> Tuple[int]:
        """Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule"""
        assert (
            num_stages % pp_size == 0
        ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
        stages_per_rank = num_stages // pp_size
        if style == "loop":
            return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
        elif style == "v":
            assert (
                stages_per_rank == 2
            ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
            stage_v_pairs = list(
                zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1))
            )
            return stage_v_pairs[pp_rank]

    def _pipeline_llama_tracer(
        self,
        model: nn.Module,
        pp_mesh: DeviceMesh,
        parallel_dims: ParallelDims,
        device: DeviceType,
        model_config: ModelArgs,
    ):
        pp_rank = pp_mesh.get_local_rank()
        pp_size = pp_mesh.size()
        microbatches = parallel_dims.pp
        tokens_shape = (8, 2048)
        input = torch.randint(
            model_config.vocab_size,
            tokens_shape,
            dtype=torch.int64,
            device=device,
        )
        stage_idx = pp_rank
        split_spec = {layer_name: SplitPoint.BEGINNING for layer_name in ["layers.1"]}
        num_stages = len(split_spec) + 1
        pipe = pipeline(
            model,
            mb_args=(input.chunk(microbatches)[0],),
            split_spec=split_spec,
        )

        stages = []
        models = []
        for stage_idx in self._stage_ids_this_rank(
            pp_rank, pp_size, num_stages, style="loop"
        ):
            models.append(pipe.get_stage_module(stage_idx))
            stages.append(
                pipe.build_stage(
                    stage_idx,
                    device=device,
                    group=pp_mesh.get_group(),
                )
            )
        return stages, models

    def _build_pipeline_schedule(
        self, pp_dim, pipeline_parallel_schedule, stages, loss_fn
    ):
        looped_schedule = False

        if pipeline_parallel_schedule == "1f1b":
            schedule_class = Schedule1F1B
        elif pipeline_parallel_schedule == "gpipe":
            schedule_class = ScheduleGPipe
        elif pipeline_parallel_schedule == "interleaved_1f1b":
            schedule_class = ScheduleInterleaved1F1B
            looped_schedule = True
        elif pipeline_parallel_schedule == "flexible_interleaved_1f1b":
            schedule_class = ScheduleFlexibleInterleaved1F1B
            looped_schedule = True
        self.logger.info(
            f"Using pipeline schedule {pipeline_parallel_schedule}"  # noqa: G004
        )
        n_microbatches = pp_dim

        return schedule_class(
            stages if looped_schedule else stages[0],
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
        )


if __name__ == "__main__":
    run_tests()
