# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import copy
import logging
import tempfile

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
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleLoopedBFS,
    ScheduleZBVZeroBubble,
)
from torch.distributed.pipelining.schedules import _PipelineScheduleRuntime
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcContinousTest,
    requires_nccl,
)
from torch.testing._internal.common_utils import (
    check_leaked_tensors,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
)


logger = logging.getLogger(__name__)

d_hid = 512
batch_size = 64
torch.manual_seed(0)
device_type = "cuda"


class ScheduleTest(MultiProcContinousTest):
    world_size = 2

    @classmethod
    def backend_str(cls) -> str:
        # Testing with NCCL backend
        return "nccl"

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _setup_models_and_data(self, n_layers=None, model_class=MultiMLP):
        """Setup models, input data, target data, and loss function."""
        if n_layers is None:
            n_layers = self.world_size

        full_mod = model_class(d_hid, n_layers=n_layers)
        full_mod.to(self.device)
        ref_mod = copy.deepcopy(full_mod)

        x = torch.randn(batch_size, d_hid, device=self.device)
        with torch.no_grad():
            y = ref_mod(x)
            target = y + torch.randn(batch_size, d_hid, device=self.device)

        loss_fn = torch.nn.MSELoss(reduction="sum")
        return full_mod, ref_mod, x, target, loss_fn

    def _create_single_stage_pipeline(self, mod, x, chunks, use_tracer=True):
        """Create a single-stage pipeline using either tracer or manual stage creation."""
        if use_tracer:
            x_mb = x.chunk(chunks)[0]
            split_spec = mod.split_spec if hasattr(mod, "split_spec") else None
            pipe = pipeline(mod, mb_args=(x_mb,), split_spec=split_spec)
            stage = pipe.build_stage(self.rank, self.device)
            stage_module = pipe.get_stage_module(self.rank)
            return stage, stage_module, [stage_module]
        else:
            # Manual stage creation
            submod_name = f"layers.{self.rank}"
            stage_module = mod.get_submodule(submod_name)
            stage = PipelineStage(stage_module, self.rank, self.world_size, self.device)
            return stage, stage_module, [stage_module]

    def _create_multi_stage_pipeline(
        self, mod, stages_per_rank, n_stages, stage_indices=None
    ):
        """Create multiple pipeline stages for interleaved schedules."""
        if stage_indices is None:
            stage_indices = [
                self.rank + i * self.world_size for i in range(stages_per_rank)
            ]

        submod_names = [f"layers.{i}" for i in stage_indices]
        stage_modules = [mod.get_submodule(submod_name) for submod_name in submod_names]
        stages = [
            PipelineStage(stage_module, stage_idx, n_stages, self.device)
            for stage_module, stage_idx in zip(stage_modules, stage_indices)
        ]
        return stages, stage_modules, submod_names

    def _run_reference_model(
        self, ref_mod, x, target, loss_fn, num_iterations=2, **kwargs
    ):
        """Run reference model for specified iterations and return final output and loss."""
        ref_out = None
        ref_loss = None

        for _ in range(num_iterations):
            ref_mod.zero_grad()
            ref_out = ref_mod(x, **kwargs)
            ref_loss = loss_fn(ref_out, target)
            ref_loss.backward()

        return ref_out, ref_loss

    def _check_gradients(
        self, stage_modules, ref_mod, submod_names=None, rtol=1e-5, atol=4e-5
    ):
        """Check that gradients match between pipeline stages and reference model using flexible comparison."""

        def grad_check(grad1, grad2, param_name, rtol, atol, tolerance=0.05):
            if grad1 is None and grad2 is None:
                return
            if grad1 is None or grad2 is None:
                raise AssertionError(
                    f"One gradient is None for {param_name}: {grad1} vs {grad2}"
                )
            torch.testing.assert_close(grad1, grad2, rtol=rtol, atol=atol)

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
                submod_name = f"layers.{self.rank}"
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

    def _zero_gradients(self, stage_modules):
        """Zero gradients for all stage modules."""
        if not isinstance(stage_modules, list):
            stage_modules = [stage_modules]
        for stage_module in stage_modules:
            stage_module.zero_grad()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [_ScheduleForwardOnly])
    def test_forward_only(self, ScheduleClass):
        mod, mod_ref, x, _, _ = self._setup_models_and_data()
        x_clone = x.clone()

        num_microbatches = 2 * self.world_size
        stage, _, _ = self._create_single_stage_pipeline(mod, x, num_microbatches)
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

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
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
            mod, _, x, target, loss_fn = self._setup_models_and_data(n_layers=n_stages)

            # Create multi-stage pipeline
            stages, stage_modules, _ = self._create_multi_stage_pipeline(
                mod, stages_per_rank, n_stages
            )
            schedule = ScheduleClass(
                stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
            )
        else:
            # Single-stage schedules
            mod, _, x, target, loss_fn = self._setup_models_and_data()

            # Create single-stage pipeline
            stage, stage_module, _ = self._create_single_stage_pipeline(
                mod, x, num_microbatches
            )
            stage_modules = [stage_module]
            schedule = ScheduleClass(
                stage, num_microbatches, loss_fn=loss_fn, scale_grads=False
            )

        # Clear gradients and run eval
        self._zero_gradients(stage_modules)
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

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    def test_multi_iter(self, ScheduleClass):
        mod, _, x, target, loss_fn = self._setup_models_and_data()
        chunks = 4
        stage, _, _ = self._create_single_stage_pipeline(mod, x, chunks)
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

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    def test_kwargs_with_tracer(self, ScheduleClass):
        # Model has two stages only, thus limiting group size to 2
        group_size = 2
        group = dist.new_group(list(range(group_size)))
        if self.rank >= group_size:
            return

        mod = ModelWithKwargs(d_hid)
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
            group=group,
        )

        # Attach to a schedule
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

        # Run
        out = None
        losses = []
        if self.rank == 0:
            schedule.step(x, y=y)
        elif self.rank == group_size - 1:
            out = schedule.step(target=target, losses=losses)
        else:
            schedule.step()

        # dist.barrier()

        # Last rank checks result
        if self.rank == group_size - 1:
            ref_out = mod(x, y=y)
            ref_loss = loss_fn(ref_out, target)
            pipe_loss = sum(losses)
            torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=5e-3)
            torch.testing.assert_close(pipe_loss, ref_loss)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    def test_grad_with_tracer(self, ScheduleClass):
        mod, ref_mod, x, target, loss_fn = self._setup_models_and_data()

        # Run reference
        ref_out, ref_loss = self._run_reference_model(ref_mod, x, target, loss_fn)

        # Create pipeline and schedule
        chunks = 2 * self.world_size
        stage, stage_module, stage_modules = self._create_single_stage_pipeline(
            mod, x, chunks
        )
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

        # Run pipeline
        out = None
        losses = []
        for _ in range(2):
            self._zero_gradients(stage_module)
            if self.rank == 0:
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier()

        # Last rank checks result
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        self._check_gradients(stage_module, ref_mod)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    @parametrize("shape_inference", [True, False])
    def test_grad_with_manual(self, ScheduleClass, shape_inference):
        mod, ref_mod, x, target, loss_fn = self._setup_models_and_data()

        # Run reference
        ref_out, ref_loss = self._run_reference_model(ref_mod, x, target, loss_fn)

        # Create manual pipeline stage
        chunks = 2 * self.world_size
        stage, stage_module, _ = self._create_single_stage_pipeline(
            mod, x, chunks, use_tracer=False
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
            self._zero_gradients(stage_module)
            if self.rank == 0:
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier()

        # Last rank checks result
        if self.rank == self.world_size - 1:
            torch.testing.assert_close(out, ref_out)
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Check gradients using helper method
        self._check_gradients(stage_module, ref_mod)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize(
        "ScheduleClass",
        [
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ],
    )
    @parametrize("use_new_runtime", [False, True])
    def test_grad_with_manual_interleaved(self, ScheduleClass, use_new_runtime):
        stages_per_rank = 2
        n_stages = stages_per_rank * self.world_size
        mod, ref_mod, x, target, loss_fn = self._setup_models_and_data(
            n_layers=n_stages
        )

        # Run reference
        ref_out, ref_loss = self._run_reference_model(ref_mod, x, target, loss_fn)

        # Create multi-stage pipeline
        stages, stage_modules, submod_names = self._create_multi_stage_pipeline(
            mod, stages_per_rank, n_stages
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

        # Handle new runtime testing
        if use_new_runtime:
            old_schedule = schedule
            tmp_schedule = _PipelineScheduleRuntime(
                stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
            )
            tmp_schedule._load_actions(old_schedule.pipeline_order)

            # Test CSV round-trip for compute_comms schedule
            schedule = _PipelineScheduleRuntime(
                stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
            )
            with tempfile.NamedTemporaryFile() as f:
                tmp_schedule._dump_csv(f.name)
                f.seek(0)
                schedule._load_csv(f.name, format="compute_comms")

            one_more_schedule = _PipelineScheduleRuntime(
                stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
            )
            one_more_schedule._load_actions(
                schedule.pipeline_order_with_comms, format="compute_comms"
            )

            # Verify schedule consistency
            self.assertEqual(
                len(schedule.pipeline_order_with_comms),
                len(one_more_schedule.pipeline_order_with_comms),
            )
            for rank in schedule.pipeline_order_with_comms:
                self.assertEqual(
                    len(schedule.pipeline_order_with_comms[rank]),
                    len(one_more_schedule.pipeline_order_with_comms[rank]),
                )
                for a, b in zip(
                    schedule.pipeline_order_with_comms[rank],
                    one_more_schedule.pipeline_order_with_comms[rank],
                ):
                    self.assertEqual(a, b)

        # Run pipeline with tensor leak checking
        out = None
        losses = []
        with check_leaked_tensors() as garbage_tensors:
            for _ in range(2):
                self._zero_gradients(stage_modules)
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
        self._check_gradients(
            stage_modules, ref_mod, submod_names, rtol=5e-3, atol=5e-3
        )

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleWithW, ScheduleInterleavedZeroBubble])
    def test_schedule_with_native_zero_bubble(self, ScheduleClass):
        print(ScheduleClass)
        if ScheduleClass is ScheduleInterleavedZeroBubble:
            n_stages = 4
            num_microbatches = 2 * n_stages
            rank_stages = {0: [0, 2], 1: [1, 3]}
        else:
            n_stages = ScheduleClass.n_stages
            num_microbatches = ScheduleClass.num_microbatches
            rank_stages = ScheduleClass.rank_stages

        num_steps = 4
        mod, ref_mod, x, target, loss_fn = self._setup_models_and_data(
            n_layers=n_stages
        )

        # Create multi-stage pipeline with custom stage indices
        stage_indices = rank_stages[self.rank]
        print(f"Rank {self.rank} stages: {stage_indices}")
        stages, stage_modules, submod_names = self._create_multi_stage_pipeline(
            mod, len(stage_indices), n_stages, stage_indices
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
        self._check_gradients(stage_modules, ref_mod, submod_names)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleWithReorderedB])
    def test_pipeline_schedule_runtime_custom_sched(self, ScheduleClass):
        n_stages = 2
        stages_per_rank = 1
        mod, ref_mod, x, target, loss_fn = self._setup_models_and_data(
            n_layers=n_stages
        )

        # Run reference
        ref_out, ref_loss = self._run_reference_model(ref_mod, x, target, loss_fn)

        # Create pipeline stages
        stages, stage_modules, submod_names = self._create_multi_stage_pipeline(
            mod, stages_per_rank, n_stages
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
                self._zero_gradients(stage_modules)
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
        self._check_gradients(stage_modules, ref_mod, submod_names)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize(
        "schedule_class", [ScheduleVShaped, ScheduleUnbalanced, ScheduleZBVZeroBubble]
    )
    @parametrize("use_new_runtime", [False, True])
    def test_non_symmetric_stage_ids(self, schedule_class, use_new_runtime):
        if schedule_class is ScheduleZBVZeroBubble:
            n_stages = 4
            rank_stages = {0: [0, 3], 1: [1, 2]}
        else:
            n_stages = schedule_class.n_stages
            rank_stages = schedule_class.rank_stages

        mod, ref_mod, x, target, loss_fn = self._setup_models_and_data(
            n_layers=n_stages
        )

        # Run reference
        ref_out, ref_loss = self._run_reference_model(ref_mod, x, target, loss_fn)

        # Create multi-stage pipeline with custom stage indices
        num_microbatches = 1
        stage_indices = rank_stages[self.rank]
        print(f"Rank {self.rank} stages: {stage_indices}")
        stages, stage_modules, submod_names = self._create_multi_stage_pipeline(
            mod, len(stage_indices), n_stages, stage_indices
        )

        schedule = schedule_class(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )

        if use_new_runtime:
            old_schedule = schedule
            schedule = _PipelineScheduleRuntime(
                stages, num_microbatches, loss_fn=loss_fn
            )
            schedule._load_actions(old_schedule.pipeline_order)

        # Run pipeline - special case where first and last stage are on rank 0
        out = None
        losses = []
        for _ in range(2):
            self._zero_gradients(stage_modules)
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
        self._check_gradients(stage_modules, ref_mod, submod_names)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleInterleavedZeroBubble])
    def test_schedule_with_weight_update_mlp_e2e(self, ScheduleClass):
        stages_per_rank = 2
        n_stages = stages_per_rank * self.world_size
        full_mod, ref_mod, x, target, loss_fn = self._setup_models_and_data(
            n_layers=n_stages, model_class=MultiMLPWithDw
        )
        full_mod.toggle()

        # Run reference
        ref_out, ref_loss = self._run_reference_model(ref_mod, x, target, loss_fn)

        # Create multi-stage pipeline with custom dw_builder
        stages, stage_modules, submod_names = self._create_multi_stage_pipeline(
            full_mod, stages_per_rank, n_stages
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

        schedule = ScheduleClass(stages, 2, loss_fn=loss_fn, scale_grads=False)

        # Run pipeline
        out = None
        losses = []
        for _ in range(2):
            self._zero_gradients(stage_modules)
            if self.rank == 0:
                schedule.step(x)
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
        self._check_gradients(stage_modules, ref_mod, submod_names)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize(
        "ScheduleClass",
        [ScheduleInterleavedZeroBubble, ScheduleInterleaved1F1B],
    )
    def test_zero_bubble_with_model_kwargs(self, ScheduleClass):
        stages_per_rank = 2
        n_stages = stages_per_rank * self.world_size
        mod, ref_mod, x, target, loss_fn = self._setup_models_and_data(
            n_layers=n_stages, model_class=MultiMLPKwargs
        )
        unused_kwarg = torch.tensor([1.0], device=self.device)

        # Run reference with kwargs
        ref_out, ref_loss = self._run_reference_model(
            ref_mod, x, target, loss_fn, unused_kwarg=unused_kwarg
        )

        # Create multi-stage pipeline
        stages, stage_modules, submod_names = self._create_multi_stage_pipeline(
            mod, stages_per_rank, n_stages
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
            self._zero_gradients(stage_modules)
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
        self._check_gradients(
            stage_modules, ref_mod, submod_names, rtol=1e-5, atol=5e-3
        )


instantiate_parametrized_tests(ScheduleTest)

if __name__ == "__main__":
    run_tests()
