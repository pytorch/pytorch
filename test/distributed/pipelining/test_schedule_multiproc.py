# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import copy
import logging
from dataclasses import dataclass

from model_registry import ModelWithKwargs, MultiMLP, MultiMLPKwargs, MultiMLPWithDw
from schedule_registry import (
    ScheduleUnbalanced,
    ScheduleVShaped,
    ScheduleWithReorderedB,
    ScheduleWithW,
)

import torch
import torch.distributed as dist
from torch.distributed.pipelining import (
    _ScheduleForwardOnly,
    pipeline,
    PipelineStage,
    Schedule1F1B,
    ScheduleDualPipeV,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleLoopedBFS,
    ScheduleZBVZeroBubble,
)
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineContext,
    _PipelineScheduleRuntime,
    _wait_batch_p2p,
    FORWARD,
    OVERLAP_F_B,
)
from torch.distributed.pipelining.stage import _PipelineStageBase  # noqa: TC002
from torch.nn.modules.loss import MSELoss
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    check_leaked_tensors,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_MULTIACCELERATOR,
)


logger = logging.getLogger(__name__)

d_hid = 512
batch_size = 64
torch.manual_seed(0)
device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = dist.get_default_backend_for_device(device_type)


@dataclass
class PipelineTestConfig:
    world_size: int
    device: torch.device
    rank: int


def setup_models_and_data(
    config: PipelineTestConfig, n_layers=None, model_class=MultiMLP
):
    """Setup models, input data, target data, and loss function."""
    if n_layers is None:
        n_layers = config.world_size

    full_mod = model_class(d_hid, n_layers=n_layers)
    full_mod.to(config.device)
    ref_mod = copy.deepcopy(full_mod)

    x = torch.randn(batch_size, d_hid, device=config.device)
    with torch.no_grad():
        y = ref_mod(x)
        target = y + torch.randn(batch_size, d_hid, device=config.device)

    loss_fn = torch.nn.MSELoss(reduction="sum")
    return full_mod, ref_mod, x, target, loss_fn


def create_single_stage_pipeline(
    config: PipelineTestConfig, mod, x, chunks, use_tracer=True
):
    """Create a single-stage pipeline using either tracer or manual stage creation."""
    if use_tracer:
        x_mb = x.chunk(chunks)[0]
        split_spec = mod.split_spec if hasattr(mod, "split_spec") else None
        pipe = pipeline(mod, mb_args=(x_mb,), split_spec=split_spec)
        stage = pipe.build_stage(config.rank, config.device)
        stage_module = pipe.get_stage_module(config.rank)
        return stage, stage_module, [stage_module]
    else:
        # Manual stage creation
        submod_name = f"layers.{config.rank}"
        stage_module = mod.get_submodule(submod_name)
        stage = PipelineStage(
            stage_module, config.rank, config.world_size, config.device
        )
        return stage, stage_module, [stage_module]


def create_multi_stage_pipeline(
    config: PipelineTestConfig, mod, stages_per_rank, n_stages, stage_indices=None
):
    """Create multiple pipeline stages for interleaved schedules."""
    if stage_indices is None:
        stage_indices = [
            config.rank + i * config.world_size for i in range(stages_per_rank)
        ]

    submod_names = [f"layers.{i}" for i in stage_indices]
    stage_modules = [mod.get_submodule(submod_name) for submod_name in submod_names]
    stages = [
        PipelineStage(stage_module, stage_idx, n_stages, config.device)
        for stage_module, stage_idx in zip(stage_modules, stage_indices, strict=True)
    ]
    return stages, stage_modules, submod_names


def run_reference_model(ref_mod, x, target, loss_fn, num_iterations=2, **kwargs):
    """Run reference model for specified iterations and return final output and loss."""
    ref_out = None
    ref_loss = None

    for _ in range(num_iterations):
        ref_mod.zero_grad()
        ref_out = ref_mod(x, **kwargs)
        ref_loss = loss_fn(ref_out, target)
        ref_loss.backward()

    return ref_out, ref_loss


def check_gradients(
    config: PipelineTestConfig,
    stage_modules,
    ref_mod,
    submod_names=None,
    rtol=1e-5,
    atol=4e-5,
):
    """Check that gradients match between pipeline stages and reference model using flexible comparison."""

    def grad_check(grad1, grad2, param_name, rtol, atol, tolerance=0.05):
        if grad1 is None and grad2 is None:
            return
        if grad1 is None or grad2 is None:
            raise AssertionError(
                f"One gradient is None for {param_name}: {grad1} vs {grad2}"
            )
        try:
            torch.testing.assert_close(grad1, grad2, rtol=rtol, atol=atol)
        except AssertionError:
            print(
                f"Numerical issues detected for {param_name}: param grad {grad1} vs ref grad {grad2}"
            )
            raise

    if submod_names is None:
        # Single stage case - need to detect tracer vs manual pipeline
        stage_modules = [stage_modules]

        # Try to detect if this is a tracer-based pipeline by checking if parameter exists in ref_mod
        sample_param_name = next(iter(stage_modules[0].named_parameters()))[0]
        try:
            # Try to get parameter directly from reference model (tracer-based)
            ref_mod.get_parameter(sample_param_name)
            is_tracer_based = True
        except AttributeError:
            # Parameter doesn't exist at root level, must be manual pipeline
            is_tracer_based = False

        if is_tracer_based:
            # Tracer-based pipeline: parameter names are full paths from root model
            for name, p in stage_modules[0].named_parameters():
                ref_p = ref_mod.get_parameter(name)
                grad_check(p.grad, ref_p.grad, name, rtol, atol)
        else:
            # Manual pipeline: parameter names are local to the submodule
            submod_name = f"layers.{config.rank}"
            ref_submod = ref_mod.get_submodule(submod_name)
            for name, p in stage_modules[0].named_parameters():
                ref_p = ref_submod.get_parameter(name)
                grad_check(p.grad, ref_p.grad, f"{submod_name}.{name}", rtol, atol)
    else:
        # Multi-stage case - always use submodule approach
        for stage_module, submod_name in zip(stage_modules, submod_names):
            ref_submod = ref_mod.get_submodule(submod_name)
            for name, p in stage_module.named_parameters():
                ref_p = ref_submod.get_parameter(name)
                grad_check(p.grad, ref_p.grad, f"{submod_name}.{name}", rtol, atol)


def zero_gradients(stage_modules):
    """Zero gradients for all stage modules."""
    if not isinstance(stage_modules, list):
        stage_modules = [stage_modules]
    for stage_module in stage_modules:
        stage_module.zero_grad()


class ScheduleTest(MultiProcContinuousTest):
    world_size = 4

    @classmethod
    def backend_str(cls) -> str:
        # Testing with NCCL backend
        return backend

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    @property
    def config(self) -> PipelineTestConfig:
        """Lazily create and return the pipeline test configuration."""
        return PipelineTestConfig(
            world_size=self.world_size, device=self.device, rank=self.rank
        )

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [_ScheduleForwardOnly])
    @skip_if_lt_x_gpu(4)
    def test_forward_only(self, ScheduleClass):
        mod, mod_ref, x, _, _ = setup_models_and_data(self.config)
        x_clone = x.clone()

        num_microbatches = 2 * self.world_size
        stage, _, _ = create_single_stage_pipeline(
            self.config, mod, x, num_microbatches
        )
        schedule = ScheduleClass(stage, num_microbatches, scale_grads=False)

        # Run forward-only schedule
        out = None
        num_iters = 20
        for _ in range(num_iters):
            if self.rank == 0:
                schedule.step(x)
                dist.recv(x, src=self.world_size - 1)
            elif self.rank == self.world_size - 1:
                out = schedule.step()
                dist.send(out, dst=0)
            else:
                schedule.step()

        # Validate pipelined output matches reference model
        if self.rank == self.world_size - 1:
            for _ in range(num_iters):
                x_clone = mod_ref(x_clone)
            torch.testing.assert_close(x_clone, out)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize(
        "ScheduleClass",
        [
            ScheduleGPipe,
            Schedule1F1B,
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ],
    )
    @skip_if_lt_x_gpu(4)
    def test_eval_inference_mode(self, ScheduleClass):
        num_microbatches = 4
        if ScheduleClass in [
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ]:
            # Multi-stage schedules
            stages_per_rank = 2
            n_stages = stages_per_rank * self.world_size
            mod, _, x, target, loss_fn = setup_models_and_data(
                self.config, n_layers=n_stages
            )

            # Create multi-stage pipeline
            stages, stage_modules, _ = create_multi_stage_pipeline(
                self.config, mod, stages_per_rank, n_stages
            )
            schedule = ScheduleClass(
                stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
            )
        else:
            # Single-stage schedules
            mod, _, x, target, loss_fn = setup_models_and_data(self.config)

            # Create single-stage pipeline
            stage, stage_module, _ = create_single_stage_pipeline(
                self.config, mod, x, num_microbatches
            )
            stage_modules = [stage_module]
            schedule = ScheduleClass(
                stage, num_microbatches, loss_fn=loss_fn, scale_grads=False
            )

        # Clear gradients and run eval
        zero_gradients(stage_modules)
        losses = []

        if self.rank == 0:
            # Support with and without no_grad()
            with torch.no_grad():
                schedule.eval(x)
        elif self.rank == self.world_size - 1:
            schedule.eval(target=target, losses=losses)
        else:
            schedule.eval()

        # Check that gradients were NOT computed during eval
        grad_computed_eval = any(
            param.grad is not None
            for stage_module in stage_modules
            for param in stage_module.parameters()
        )

        # Verify that gradients were not computed during eval
        self.assertFalse(
            grad_computed_eval, "Gradients should not be computed during eval()"
        )

        # Verify that losses are still computed during eval
        if self.rank == self.world_size - 1:
            self.assertTrue(len(losses) > 0, "Losses should be computed during eval()")

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize(
        "ScheduleClass",
        [
            ScheduleGPipe,
            Schedule1F1B,
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ],
    )
    @skip_if_lt_x_gpu(4)
    def test_return_output(self, ScheduleClass):
        num_microbatches = 4
        if ScheduleClass in [
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ]:
            # Multi-stage schedules
            stages_per_rank = 2
            n_stages = stages_per_rank * self.world_size
            mod, _, x, target, loss_fn = setup_models_and_data(
                self.config, n_layers=n_stages
            )

            # Create multi-stage pipeline
            stages, stage_modules, _ = create_multi_stage_pipeline(
                self.config, mod, stages_per_rank, n_stages
            )
            schedule = ScheduleClass(
                stages,
                num_microbatches,
                loss_fn=loss_fn,
                scale_grads=False,
            )
        else:
            # Single-stage schedules
            mod, _, x, target, loss_fn = setup_models_and_data(self.config)

            # Create single-stage pipeline
            stage, stage_module, _ = create_single_stage_pipeline(
                self.config, mod, x, num_microbatches
            )
            schedule = ScheduleClass(
                stage,
                num_microbatches,
                loss_fn=loss_fn,
                scale_grads=False,
            )

        losses = []

        if self.rank == self.world_size - 1:
            output = schedule.step(target=target, losses=losses, return_outputs=False)
        else:
            schedule.step(x)

        # Verify that output is None
        if self.rank == self.world_size - 1:
            self.assertTrue(output is None, "Output should be None")

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    @skip_if_lt_x_gpu(4)
    def test_multi_iter(self, ScheduleClass):
        mod, _, x, target, loss_fn = setup_models_and_data(self.config)
        chunks = 4
        stage, _, _ = create_single_stage_pipeline(self.config, mod, x, chunks)
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

        # Run
        for _ in range(20):
            if self.rank == 0:
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                losses = []
                schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier(device_ids=[self.rank])

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    @skip_if_lt_x_gpu(4)
    def test_kwargs_with_tracer(self, ScheduleClass):
        mod = ModelWithKwargs(d_hid, splits=self.world_size)
        mod.to(self.device)

        x = torch.randn(batch_size, d_hid, device=self.device)
        y = torch.randn(batch_size, d_hid, device=self.device)
        target = torch.randn(batch_size, d_hid, device=self.device)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        chunks = 4
        x_mb = x.chunk(chunks)[0]
        y_mb = y.chunk(chunks)[0]

        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            mb_kwargs={"y": y_mb},
        )

        stage = pipe.build_stage(
            self.rank,
            self.device,
        )

        # Attach to a schedule
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

        # Run
        out = None
        losses = []
        if self.rank == 0:
            schedule.step(x, y=y)
        elif self.rank == self.world_size - 1:
            out = schedule.step(target=target, losses=losses)
        else:
            schedule.step()

        dist.barrier(device_ids=[self.rank])

        # Last rank checks result
        if self.rank == self.world_size - 1:
            ref_out = mod(x, y=y)
            ref_loss = loss_fn(ref_out, target)
            pipe_loss = sum(losses)
            torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=5e-3)
            torch.testing.assert_close(pipe_loss, ref_loss)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    @skip_if_lt_x_gpu(4)
    def test_grad_with_tracer(self, ScheduleClass):
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(self.config)

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create pipeline and schedule
        chunks = 2 * self.world_size
        stage, stage_module, stage_modules = create_single_stage_pipeline(
            self.config, mod, x, chunks
        )
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

        # Run pipeline
        out = None
        losses = []
        for _ in range(2):
            zero_gradients(stage_module)
            if self.rank == 0:
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier(device_ids=[self.rank])

        # Last rank checks result
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(self.config, stage_module, ref_mod)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    @parametrize("shape_inference", [True, False])
    @skip_if_lt_x_gpu(4)
    def test_grad_with_manual(self, ScheduleClass, shape_inference):
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(self.config)

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create manual pipeline stage
        chunks = 2 * self.world_size
        stage, stage_module, _ = create_single_stage_pipeline(
            self.config, mod, x, chunks, use_tracer=False
        )

        # Handle shape inference
        if not shape_inference:
            input_args = (x.chunk(chunks)[0],)
            with torch.no_grad():
                output_args = stage_module(*input_args)
            stage = PipelineStage(
                stage_module,
                self.rank,
                self.world_size,
                self.device,
                input_args=input_args,
                output_args=output_args,
            )

        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

        # Run pipeline
        out = None
        losses = []
        for _ in range(2):
            zero_gradients(stage_module)
            if self.rank == 0:
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier(device_ids=[self.rank])

        # Last rank checks result
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(self.config, stage_module, ref_mod)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize(
        "ScheduleClass",
        [
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ],
    )
    @skip_if_lt_x_gpu(4)
    def test_grad_with_manual_interleaved(self, ScheduleClass):
        stages_per_rank = 2
        n_stages = stages_per_rank * self.world_size
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages
        )

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create multi-stage pipeline
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, stages_per_rank, n_stages
        )
        print(f"Rank {self.rank} stages: {[stage.stage_index for stage in stages]}")

        num_microbatches = (
            ScheduleClass.num_microbatches
            if hasattr(ScheduleClass, "num_microbatches")
            else 2 * self.world_size
        )

        # Create schedule
        schedule = ScheduleClass(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )

        # Run pipeline with tensor leak checking
        out = None
        losses = []
        with check_leaked_tensors() as garbage_tensors:
            for _ in range(2):
                zero_gradients(stage_modules)
                if self.rank == 0:
                    schedule.step(x)
                elif self.rank == self.world_size - 1:
                    out = schedule.step(target=target, losses=losses)
                else:
                    schedule.step()

        self.assertEqual(
            len(garbage_tensors),
            0,
            "Found leaked tensors, check logs above for debug info",
        )
        dist.barrier()

        # Verify results
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients - use relaxed tolerances for interleaved schedules
        # since gradients are small
        check_gradients(
            self.config, stage_modules, ref_mod, submod_names, rtol=5e-3, atol=5e-3
        )

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleInterleavedZeroBubble])
    @skip_if_lt_x_gpu(4)
    def test_schedule_with_weight_update_mlp_e2e(self, ScheduleClass):
        stages_per_rank = 2
        n_stages = stages_per_rank * self.world_size
        full_mod, ref_mod, x, target, _ = setup_models_and_data(
            self.config, n_layers=n_stages, model_class=MultiMLPWithDw
        )
        full_mod.toggle()
        loss_fn = MSELoss()

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create multi-stage pipeline with custom dw_builder
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, full_mod, stages_per_rank, n_stages
        )

        class CustomState:
            def __init__(self, stage_module, stage_idx, rank):
                self.i = 0
                self.stage_module = stage_module
                self.stage_idx = stage_idx
                self.rank = rank

            def dw_builder(self):
                def dw_runner():
                    self.i += 1
                    print(
                        f"[Rank {self.rank}] dw_count={self.i} stage={self.stage_idx}"
                    )
                    self.stage_module.compute_dW()

                return dw_runner

        # Create custom states and rebuild stages with dw_builder
        cs = {}
        stage_indices = [
            self.rank + i * self.world_size for i in range(stages_per_rank)
        ]
        for stage_module, stage_idx in zip(stage_modules, stage_indices):
            cs[stage_idx] = CustomState(stage_module, stage_idx, self.rank)

        stages = [
            PipelineStage(
                stage_module,
                stage_idx,
                n_stages,
                self.device,
                dw_builder=cs[stage_idx].dw_builder,
            )
            for stage_module, stage_idx in zip(stage_modules, stage_indices)
        ]

        schedule = ScheduleClass(stages, 2, loss_fn=loss_fn)

        # Run pipeline
        out = None
        losses = []
        for _ in range(2):
            zero_gradients(stage_modules)
            if self.rank == 0:
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier(device_ids=[self.rank])

        # Verify results
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses) / len(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(self.config, stage_modules, ref_mod, submod_names)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize(
        "schedule_class",
        [ScheduleZBVZeroBubble, ScheduleDualPipeV],
    )
    @skip_if_lt_x_gpu(4)
    def test_v_shape_schedules(self, schedule_class):
        n_stages = 8
        rank_stages = {0: [0, 7], 1: [1, 6], 2: [2, 5], 3: [3, 4]}
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages
        )

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create multi-stage pipeline with custom stage indices
        num_microbatches = 8
        stage_indices = rank_stages[self.rank]
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, len(stage_indices), n_stages, stage_indices
        )

        schedule = schedule_class(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )

        # Run pipeline - special case where first and last stage are on rank 0
        out = None
        losses = []
        for _ in range(2):
            zero_gradients(stage_modules)
            if self.rank == 0:
                out = schedule.step(x, target=target, losses=losses)
            else:
                schedule.step()

        # Verify results (rank 0 has both first and last stages)
        if self.rank == 0:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(self.config, stage_modules, ref_mod, submod_names)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @skip_if_lt_x_gpu(4)
    def test_custom_function_callback(self):
        """Test the custom function callback functionality with _PipelineScheduleRuntime."""
        n_stages = 8
        rank_stages = {0: [0, 7], 1: [1, 6], 2: [2, 5], 3: [3, 4]}
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages
        )

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create multi-stage pipeline with custom stage indices
        num_microbatches = 8
        stage_indices = rank_stages[self.rank]
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, len(stage_indices), n_stages, stage_indices
        )

        # Use DualPipeV schedule as the base schedule
        base_schedule = ScheduleDualPipeV(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )
        base_schedule._prepare_schedule_with_comms(base_schedule.pipeline_order)

        # Track both types of callbacks separately
        forward_calls = []
        overlap_calls = []

        def forward_callback(action: _Action, ctx: _PipelineContext):
            """Custom callback for FORWARD computation that mimics the original implementation."""
            schedule = ctx.schedule_ref
            assert isinstance(schedule, _PipelineScheduleRuntime)
            stage_index_to_stage: dict[int, _PipelineStageBase] = {
                stage.stage_index: stage for stage in schedule._stages
            }
            stage = stage_index_to_stage[action.stage_index]
            stage_index = stage.stage_index
            mb_index = action.microbatch_index
            assert mb_index is not None
            fwd_recv_ops = schedule.fwd_recv_ops
            arg_mbs = ctx.arg_mbs
            kwarg_mbs = ctx.kwarg_mbs

            is_next_stage_on_this_rank = stage_index + 1 in stage_index_to_stage
            is_prev_stage_on_this_rank = stage_index - 1 in stage_index_to_stage

            # used in verification at the end
            forward_calls.append((stage_index, mb_index))

            if (
                not stage.is_first
                # no recv op expected for V-schedule special case (see [Note: V-schedule special case])
                and not is_prev_stage_on_this_rank
            ):
                assert (
                    stage_index,
                    mb_index,
                ) in fwd_recv_ops, f"Computing {action=} before receiving input"
                from torch.distributed.pipelining.schedules import _wait_batch_p2p

                _wait_batch_p2p(fwd_recv_ops.pop((stage_index, mb_index)))

            output = stage.forward_one_chunk(
                mb_index,
                arg_mbs[mb_index],  # type: ignore[index]
                kwarg_mbs[mb_index],  # type: ignore[index]
            )
            schedule._maybe_compute_loss(stage, output, ctx.target_mbs, mb_index)

            # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
            # see [Note: V-schedule special case]
            if is_next_stage_on_this_rank:
                stage_index_to_stage[stage_index + 1].set_local_fwd_input(
                    output, mb_index
                )

        def overlap_callback(action: _Action, ctx: _PipelineContext):
            """Custom callback for OVERLAP_F_B computation that mimics the original implementation."""
            schedule = ctx.schedule_ref
            assert isinstance(schedule, _PipelineScheduleRuntime)
            stage_index_to_stage: dict[int, _PipelineStageBase] = {
                stage.stage_index: stage for stage in schedule._stages
            }
            assert action.sub_actions is not None
            fwd_action = action.sub_actions[0]
            bwd_action = action.sub_actions[1]

            # Forward ========================================================
            forward_callback(fwd_action, ctx)
            overlap_calls.append(
                (
                    fwd_action.stage_index,
                    fwd_action.microbatch_index,
                    bwd_action.stage_index,
                    bwd_action.microbatch_index,
                )
            )

            # Backward ========================================================
            backward_stage_index = bwd_action.stage_index
            backward_stage = stage_index_to_stage[backward_stage_index]
            backward_mb_index = bwd_action.microbatch_index
            assert backward_mb_index is not None
            bwd_recv_ops = schedule.bwd_recv_ops
            is_next_stage_on_this_rank = (
                backward_stage.stage_index + 1 in stage_index_to_stage
            )
            is_prev_stage_on_this_rank = (
                backward_stage.stage_index - 1 in stage_index_to_stage
            )
            if (
                not backward_stage.is_last
                # no recv op expected for V-schedule special case (see [Note: V-schedule special case])
                and not is_next_stage_on_this_rank
            ):
                assert (
                    backward_stage_index,
                    backward_mb_index,
                ) in bwd_recv_ops, (
                    f"Attempted to run compute {action=} before receiving input"
                )
                _wait_batch_p2p(
                    bwd_recv_ops.pop((backward_stage_index, backward_mb_index))
                )
            loss = schedule._maybe_get_loss(backward_stage, backward_mb_index)
            schedule.backward_counter[backward_stage_index] += 1
            last_backward = (
                schedule.backward_counter[backward_stage_index]
                == schedule._n_microbatches
            )
            grad_scale_factor = schedule._n_microbatches if schedule.scale_grads else 1
            backward_stage.backward_one_chunk(
                backward_mb_index,
                loss=loss,
                full_backward=True,
                last_backward=last_backward,
            )
            if last_backward:
                backward_stage.scale_grads(grad_scale_factor)
            # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
            # see [Note: V-schedule special case]
            if is_prev_stage_on_this_rank:
                stage_index_to_stage[backward_stage_index - 1].set_local_bwd_input(
                    backward_stage.get_local_bwd_output(backward_mb_index),
                    backward_mb_index,
                )

        # Add the callback for FORWARD computation type

        base_schedule.register_custom_function(FORWARD, forward_callback)
        base_schedule.register_custom_function(OVERLAP_F_B, overlap_callback)

        # Run pipeline - special case where first and last stage are on rank 0
        out = None
        losses = []
        num_loops = 2
        for _ in range(num_loops):
            zero_gradients(stage_modules)
            if self.rank == 0:
                out = base_schedule.step(x, target=target, losses=losses)
            else:
                base_schedule.step()

        dist.barrier()

        # Verify results (rank 0 has both first and last stages)
        if self.rank == 0:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

            # Verify overlap callbacks were called
            self.assertGreater(
                len(overlap_calls), 0, "OVERLAP_F_B callback should have been called"
            )

            # In a V-schedule with 8 microbatches and 2 stages per rank,
            # rank 0 should have 32 calls (8 microbatches * 2 stages * 2 loops)
            expected_count = num_microbatches * 2 * num_loops
            self.assertEqual(len(forward_calls), expected_count)

            # Verify all callback calls are for stages on this rank
            for stage_idx, _ in forward_calls:
                self.assertIn(
                    stage_idx,
                    stage_indices,
                    f"Callback called for stage {stage_idx} not on rank {self.rank}",
                )

        # Check gradients using helper method
        check_gradients(self.config, stage_modules, ref_mod, submod_names)

    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, "NCCL test requires 2+ GPUs"
    )
    @parametrize(
        "ScheduleClass",
        [ScheduleInterleavedZeroBubble, ScheduleInterleaved1F1B],
    )
    @skip_if_lt_x_gpu(4)
    def test_zero_bubble_with_model_kwargs(self, ScheduleClass):
        stages_per_rank = 2
        n_stages = stages_per_rank * self.world_size
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages, model_class=MultiMLPKwargs
        )
        unused_kwarg = torch.tensor([1.0], device=self.device)

        # Run reference with kwargs
        ref_out, ref_loss = run_reference_model(
            ref_mod, x, target, loss_fn, unused_kwarg=unused_kwarg
        )

        # Create multi-stage pipeline
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, stages_per_rank, n_stages
        )

        num_microbatches = (
            ScheduleClass.num_microbatches
            if hasattr(ScheduleClass, "num_microbatches")
            else 2 * self.world_size
        )
        schedule = ScheduleClass(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )

        # Run pipeline with kwargs
        out = None
        losses = []
        for _ in range(2):
            zero_gradients(stage_modules)
            if self.rank == 0:
                schedule.step(
                    x,
                    unused_kwarg=unused_kwarg.clone()
                    .unsqueeze(0)
                    .expand(num_microbatches, -1),
                )
            elif self.rank == self.world_size - 1:
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier()

        # Verify results
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(
            self.config, stage_modules, ref_mod, submod_names, rtol=3e-5, atol=5e-3
        )


instantiate_parametrized_tests(ScheduleTest)


class CustomSchedulesTest(MultiProcContinuousTest):
    """
    These schedules are from the ScheduleRegistry and require world_size == 2
    The schedules test weird and unconventional schedules for edge cases
    """

    world_size = 2

    @classmethod
    def backend_str(cls) -> str:
        # Testing with NCCL backend
        return backend

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    @property
    def config(self) -> PipelineTestConfig:
        """Lazily create and return the pipeline test configuration."""
        return PipelineTestConfig(
            world_size=self.world_size, device=self.device, rank=self.rank
        )

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize(
        "schedule_class",
        [ScheduleVShaped, ScheduleUnbalanced],
    )
    @skip_if_lt_x_gpu(4)
    def test_non_symmetric_stage_ids(self, schedule_class):
        n_stages = schedule_class.n_stages
        rank_stages = schedule_class.rank_stages

        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages
        )

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create multi-stage pipeline with custom stage indices
        num_microbatches = 1
        stage_indices = rank_stages[self.rank]
        print(f"Rank {self.rank} stages: {stage_indices}")
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, len(stage_indices), n_stages, stage_indices
        )

        schedule = schedule_class(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )

        # Run pipeline - special case where first and last stage are on rank 0
        out = None
        losses = []
        for _ in range(2):
            zero_gradients(stage_modules)
            if self.rank == 0:
                out = schedule.step(x, target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier()

        # Verify results (rank 0 has both first and last stages)
        if self.rank == 0:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(self.config, stage_modules, ref_mod, submod_names)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleWithReorderedB])
    @skip_if_lt_x_gpu(4)
    def test_pipeline_schedule_runtime_custom_sched(self, ScheduleClass):
        n_stages = 2
        stages_per_rank = 1
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages
        )

        # Run reference
        ref_out, ref_loss = run_reference_model(ref_mod, x, target, loss_fn)

        # Create pipeline stages
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, stages_per_rank, n_stages
        )
        print(f"Rank {self.rank} stages: {[stage.stage_index for stage in stages]}")

        num_microbatches = (
            ScheduleClass.num_microbatches
            if hasattr(ScheduleClass, "num_microbatches")
            else 8
        )

        schedule = ScheduleClass(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )
        assert isinstance(schedule, _PipelineScheduleRuntime)

        # Run pipeline with tensor leak checking
        with check_leaked_tensors() as garbage_tensors:
            for _ in range(2):
                zero_gradients(stage_modules)
                if self.rank == 0:
                    schedule.step(x)
                elif self.rank == self.world_size - 1:
                    losses = []
                    out = schedule.step(target=target, losses=losses)
                else:
                    schedule.step()

        self.assertEqual(
            len(garbage_tensors),
            0,
            "Found leaked tensors, check logs above for debug info",
        )
        dist.barrier()

        # Verify results
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        check_gradients(self.config, stage_modules, ref_mod, submod_names)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleWithW])
    @skip_if_lt_x_gpu(4)
    def test_schedule_with_native_zero_bubble(self, ScheduleClass):
        n_stages = ScheduleClass.n_stages
        num_microbatches = ScheduleClass.num_microbatches
        rank_stages = ScheduleClass.rank_stages

        num_steps = 4
        mod, ref_mod, x, target, loss_fn = setup_models_and_data(
            self.config, n_layers=n_stages
        )

        # Create multi-stage pipeline with custom stage indices
        stage_indices = rank_stages[self.rank]
        print(f"Rank {self.rank} stages: {stage_indices}")
        stages, stage_modules, submod_names = create_multi_stage_pipeline(
            self.config, mod, len(stage_indices), n_stages, stage_indices
        )

        schedule = ScheduleClass(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )

        # Run reference model
        ref_x = x.detach().clone().requires_grad_(x.requires_grad)
        torch.testing.assert_close(x, ref_x)
        for _ in range(num_steps):
            ref_out = ref_mod(ref_x)
            ref_loss = loss_fn(ref_out, target)
            ref_loss.backward()

        # Run pipeline with tensor leak checking
        losses = []
        with check_leaked_tensors() as garbage_tensors:
            for _ in range(num_steps):
                if self.rank == 0:
                    schedule.step(x)
                elif self.rank == self.world_size - 1:
                    schedule.step(target=target, losses=losses)
                else:
                    schedule.step()

        self.assertEqual(
            len(garbage_tensors),
            0,
            "Found leaked tensors, check logs above for debug info",
        )

        # Check gradients using helper method
        check_gradients(self.config, stage_modules, ref_mod, submod_names)


instantiate_parametrized_tests(CustomSchedulesTest)


if __name__ == "__main__":
    run_tests()
