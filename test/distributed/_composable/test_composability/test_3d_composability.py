# Owner(s): ["oncall: distributed"]

import copy
import functools
import logging
import os
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import (
    PipelineStage,
    Schedule1F1B,
    ScheduleFlexibleInterleaved1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.optim.lr_scheduler import LambdaLR
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
    float8: bool
    async_tp: bool
    schedule_class: str

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


class Float8Handler:
    def __init__(self, parallel_dims: ParallelDims, scale_for_fsdp: bool):
        self.enabled = False

        if not (
            torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
        ):
            # Float8 is only supported on SM89 or later (H100+ GPUs)
            return
        try:
            from torchao.float8 import CastConfig, Float8LinearConfig, ScalingType
        except ImportError as e:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            ) from e

        # Mutates the model inplace replacing instances of torch.nn.Linear with Float8Linear
        enable_fsdp_float8_all_gather = (
            parallel_dims.dp_enabled
            and parallel_dims.dp_type == "fsdp"
            and scale_for_fsdp
        )
        scaling_type_input = ScalingType("dynamic")  # ["dynamic", "delayed"]
        scaling_type_weight = ScalingType("dynamic")  # ["dynamic", "delayed"]
        scaling_type_grad_output = ScalingType("dynamic")  # ["dynamic", "delayed"]
        self.config = Float8LinearConfig(
            enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
            cast_config_input=CastConfig(scaling_type=scaling_type_input),
            cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
            cast_config_grad_output=CastConfig(scaling_type=scaling_type_grad_output),
            enable_pre_and_post_forward=False,
        )

        self.enabled = True

        # for precompute_float8_dynamic_scale_for_fsdp
        self.precompute_scale = enable_fsdp_float8_all_gather and scale_for_fsdp

        # for sync_float8_amax_and_scale_history
        self.delayed_scaling = (
            scaling_type_input == "delayed"
            or scaling_type_weight == "delayed"
            or scaling_type_grad_output == "delayed"
        )
        self._sync_float8_amax_and_scale_history = None
        # self.compile = job_config.training.compile

    def convert_to_float8_training(self, model: nn.Module):
        """
        This function converts the linear layers of `model` to `Float8Linear`.
        Note that today, only dynamic tensor scaling (the default) is supported.
        This will mutate the model inplace.
        """
        if not self.enabled:
            return

        from torchao.float8 import convert_to_float8_training

        # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
        convert_to_float8_training(
            model,
            config=self.config,
            module_filter_fn=lambda mod, fqn: fqn != "output",
        )

    def precompute_float8_dynamic_scale_for_fsdp(
        self, model: Union[nn.Module, List[nn.Module]]
    ):
        if not self.enabled:
            return

        if not self.precompute_scale:
            return

        from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            precompute_float8_dynamic_scale_for_fsdp(m)

    def sync_float8_amax_and_scale_history(
        self, model: Union[nn.Module, List[nn.Module]]
    ):
        if not self.enabled:
            return

        if not self.delayed_scaling:
            return

        from torchao.float8 import sync_float8_amax_and_scale_history

        # TODO(vkuzo): see if precalculating the modules to sync over is going to
        # meaningfully help performance

        if self._sync_float8_amax_and_scale_history is None:
            if self.compile:
                self._sync_float8_amax_and_scale_history = torch.compile(
                    sync_float8_amax_and_scale_history
                )
            else:
                self._sync_float8_amax_and_scale_history = (
                    sync_float8_amax_and_scale_history
                )

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            self._sync_float8_amax_and_scale_history(m)


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

    @skip_if_lt_x_gpu(8)
    def test_3d_composability(self):
        self.run_subtests(
            {
                "name": ["pp_dp_tp"],
                "dp_degree": [
                    2,
                ],
                "tp_degree": [
                    2,
                ],
                "pp_degree": [
                    2,
                ],
                "float8": [False],
                "async_tp": [True, False],
                "schedule_class": ["1f1b", "gpipe", "interleaved_1f1b", "flexible_interleaved_1f1b", "interleaved_zerobubble_1f1b"],
            self._test_3d_composability,
        )
        self.logger.info("Finished job: 3D composability test")

    def _test_3d_composability(
        self, name, dp_degree, tp_degree, pp_degree, float8, async_tp, schedule_class
    ):
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
            float8=float8,
            async_tp=async_tp,
            schedule_class=schedule_class,
        )
        device = torch.device("cuda", index=dist.get_rank())
        torch.cuda.set_device(device)

        def _warn_overwrite_env(env, val):
            if env in os.environ:
                self.logger.warning("ENV will be overridden based on job config")
            os.environ[env] = val

        _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        torch.manual_seed(0)
        model_args = ModelArgs(dropout_p=0.0)
        model = Transformer(model_args)
        world_mesh = parallel_dims.build_mesh(device_type="cuda")

        if parallel_dims.float8:
            scale_for_fsdp = False
            if parallel_dims.dp_enabled and parallel_dims.dp_type == "fsdp":
                scale_for_fsdp = True
            # a no-op hander if float8 is not enabled
            float8_handler = Float8Handler(parallel_dims, scale_for_fsdp)
            # swap to Float8Linear based on float8 configs
            float8_handler.convert_to_float8_training(model)

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
                m.train()
        else:
            self._parallelize_llama(model, world_mesh, parallel_dims)
            model.train()
            model_parts = [model]

        optimizers = self._build_optimizers(model_parts)
        lr_schedulers = self._build_lr_schedulers(optimizers.optimizers)
        train_state_step = 0

        while train_state_step < 10000:
            train_state_step += 1

            # clip gradients
            for m in model_parts:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, foreach=True)

            # sync float8 amaxes and scales
            # float8_handler.sync_float8_amax_and_scale_history(model_parts)

            # optimizer step
            optimizers.step()
            lr_schedulers.step()

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
                enable_float8=parallel_dims.float8,
                enable_async_tp=parallel_dims.async_tp,
            )

        # self._apply_compile(model)

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

            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        if enable_async_tp:
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            torch._inductor.config._micro_pipeline_tp = True
            enable_symm_mem_for_group(tp_mesh.get_group().group_name)

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
        stages, models = self._pipeline_llama_manual(
            model, pp_mesh, parallel_dims, device, model_config
        )

        pp_schedule = self._build_pipeline_schedule(
            parallel_dims.pp, parallel_dims.schedule_class, stages, loss_fn
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

            if parallel_dims.dp_enabled:
                mp_dtype = torch.float32
            else:
                mp_dtype = torch.bfloat16
            batch_size = 8
            local_seq_len = int(128 // parallel_dims.tp)
            layers_io_shape = (batch_size, local_seq_len, model_config.dim)
            output_layer_shape = (
                batch_size,
                128,
                model_config.vocab_size,
            )
            if is_first:
                tokens_shape = (8, 128)
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
        elif pipeline_parallel_schedule == "interleaved_zerobubble_1f1b":
            schedule_class = ScheduleInterleavedZeroBubble
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

    def _build_optimizers(self, model_parts):
        """Wrap one optimizer per model part in an OptimizersContainer which provides a single
        step() and zero_grad() method for all the child optimizers.
        """

        def _build_optimizer(model):
            name = "AdamW"
            lr = 8e-4
            fused = False

            # Common parameters for both optimizers
            optimizer_kwargs = {
                "lr": lr,
                "betas": (0.9, 0.95),
                "weight_decay": 0.1,
                "fused": fused,
                "foreach": not fused,
            }
            if name == "Adam":
                # TODO: make the optimizer options configurable by toml/cmd args
                optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
            elif name == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            else:
                raise NotImplementedError(f"Optimizer {name} not added.")

            return optimizer

        class OptimizersContainer:
            """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages"""

            def __init__(self, optimizers):
                self.optimizers = optimizers

            def step(self):
                for optimizer in self.optimizers:
                    optimizer.step()

            def zero_grad(self):
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

        return OptimizersContainer([_build_optimizer(model) for model in model_parts])

    def _build_lr_schedulers(self, optimizers):
        def _build_lr_scheduler(optimizer):
            """Build a linear warmup and linear decay scheduler"""

            def linear_warmup_linear_decay(
                warmup_steps: int, decay_steps: int, current_step: int
            ) -> float:
                """Computes linear warmup followed by linear decay.
                Per LambdaLR requirement, this is accomplished by returning
                a multiplicative factor to adjust the learning rate to
                create the desired schedule.
                """
                if current_step < warmup_steps:
                    # linear warmup
                    # 0-indexed step, hence + 1 adjustments
                    current_step += 1
                    curr_adjustment = float(current_step / (warmup_steps + 1))

                else:
                    # linear decay
                    normalized_step = decay_steps - (current_step - warmup_steps)
                    curr_adjustment = 1 - (decay_steps - normalized_step) / decay_steps

                return curr_adjustment

            warmup_steps = 200
            decay_steps = float(10000 - warmup_steps)
            lr_lambda = functools.partial(
                linear_warmup_linear_decay, warmup_steps, decay_steps
            )
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            return warmup_scheduler

        class SchedulersContainer:
            """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

            def __init__(self, schedulers):
                self.schedulers = schedulers

            def step(self):
                for schedulers in self.schedulers:
                    schedulers.step()

        return SchedulersContainer(
            [_build_lr_scheduler(optimizer) for optimizer in optimizers]
        )


if __name__ == "__main__":
    run_tests()
