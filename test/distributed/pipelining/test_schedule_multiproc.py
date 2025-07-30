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
batch_size = 256
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

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [_ScheduleForwardOnly])
    def test_forward_only(self, ScheduleClass):
        mod = MultiMLP(d_hid, n_layers=self.world_size)
        mod.to(self.device)

        mod_ref = copy.deepcopy(mod)

        x = torch.randn(batch_size, d_hid, device=self.device)
        x_clone = x.clone()

        num_microbatches = 2 * self.world_size
        x_mb = x.chunk(num_microbatches)[0]

        # Create a pipeline
        split_spec = mod.split_spec if hasattr(mod, "split_spec") else None
        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            split_spec=split_spec,
        )

        stage = pipe.build_stage(
            self.rank,
            self.device,
        )

        # Attach to a schedule
        schedule = ScheduleClass(stage, num_microbatches, scale_grads=False)

        # Run
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

        # Validate pipelined output is the same as reference model
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
        if ScheduleClass in [
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ]:
            # Multi-stage schedules
            stages_per_rank = 2
            n_stages = stages_per_rank * self.world_size
            mod = MultiMLP(d_hid, n_layers=n_stages)
            mod.to(self.device)

            x = torch.randn(batch_size, d_hid, device=self.device)
            target = torch.randn(batch_size, d_hid, device=self.device)
            loss_fn = torch.nn.MSELoss(reduction="sum")

            chunks = 4
            stage_indices = [
                self.rank + i * self.world_size for i in range(stages_per_rank)
            ]
            submod_names = [f"layers.{i}" for i in stage_indices]
            stage_modules = [
                mod.get_submodule(submod_name) for submod_name in submod_names
            ]
            stages = [
                PipelineStage(
                    stage_module,
                    stage_idx,
                    n_stages,
                    self.device,
                )
                for stage_module, stage_idx in zip(stage_modules, stage_indices)
            ]

            # Test with eval() method for inference
            schedule = ScheduleClass(stages, chunks, loss_fn=loss_fn, scale_grads=False)

            # Clear gradients
            for stage_module in stage_modules:
                stage_module.zero_grad()

            if self.rank == 0:
                schedule.eval(x)
            elif self.rank == self.world_size - 1:
                losses = []
                schedule.eval(target=target, losses=losses)
            else:
                schedule.eval()

            # Check that gradients were NOT computed during eval
            grad_computed_eval = False
            for stage_module in stage_modules:
                for param in stage_module.parameters():
                    if param.grad is not None:
                        grad_computed_eval = True
                        break
                if grad_computed_eval:
                    break

            # Verify that gradients were not computed during eval
            self.assertFalse(
                grad_computed_eval,
                "Gradients should not be computed during eval()",
            )

            # Verify that losses are still computed during eval
            if self.rank == self.world_size - 1:
                self.assertTrue(
                    len(losses) > 0, "Losses should be computed during eval()"
                )
        else:
            # Single-stage schedules
            mod = MultiMLP(d_hid, n_layers=self.world_size)
            mod.to(self.device)

            x = torch.randn(batch_size, d_hid, device=self.device)
            target = torch.randn(batch_size, d_hid, device=self.device)
            loss_fn = torch.nn.MSELoss(reduction="sum")

            chunks = 4
            x_mb = x.chunk(chunks)[0]

            # Create a pipeline
            split_spec = mod.split_spec if hasattr(mod, "split_spec") else None
            pipe = pipeline(
                mod,
                mb_args=(x_mb,),
                split_spec=split_spec,
            )

            stage = pipe.build_stage(
                self.rank,
                self.device,
            )

            # Test with eval() method for inference
            schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

            # Get stage module for gradient checking
            stage_module = pipe.get_stage_module(self.rank)
            stage_module.zero_grad()

            if self.rank == 0:
                schedule.eval(x)
            elif self.rank == self.world_size - 1:
                losses = []
                schedule.eval(target=target, losses=losses)
            else:
                schedule.eval()

            # Check that gradients were NOT computed during eval
            grad_computed_eval = False
            for param in stage_module.parameters():
                if param.grad is not None:
                    grad_computed_eval = True
                    break

            # Verify that gradients were not computed during eval
            self.assertFalse(
                grad_computed_eval,
                "Gradients should not be computed during eval()",
            )

            # Verify that losses are still computed during eval
            if self.rank == self.world_size - 1:
                self.assertTrue(
                    len(losses) > 0, "Losses should be computed during eval()"
                )

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    def test_multi_iter(self, ScheduleClass):
        mod = MultiMLP(d_hid, n_layers=self.world_size)
        mod.to(self.device)

        x = torch.randn(batch_size, d_hid, device=self.device)
        target = torch.randn(batch_size, d_hid, device=self.device)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        chunks = 4
        x_mb = x.chunk(chunks)[0]

        # Create a pipeline
        split_spec = mod.split_spec if hasattr(mod, "split_spec") else None
        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            split_spec=split_spec,
        )

        stage = pipe.build_stage(
            self.rank,
            self.device,
        )

        # Attach to a schedule
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
        if self.rank == 0:
            schedule.step(x, y=y)
        elif self.rank == group_size - 1:
            losses = []
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
        mod = MultiMLP(d_hid, n_layers=self.world_size)
        mod.to(self.device)

        ref_mod = copy.deepcopy(mod)
        x = torch.randn(batch_size, d_hid, device=self.device)
        with torch.no_grad():
            y = ref_mod(x)
            # Add a small perturbation
            target = y + torch.randn(batch_size, d_hid, device=self.device)

        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Run reference
        for _ in range(2):
            ref_mod.zero_grad()
            ref_out = ref_mod(x)
            ref_loss = loss_fn(ref_out, target)
            ref_loss.backward()

        # Create a pipeline
        chunks = 2 * self.world_size
        x_mb = x.chunk(chunks)[0]
        split_spec = mod.split_spec if hasattr(mod, "split_spec") else None
        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            split_spec=split_spec,
        )

        stage = pipe.build_stage(
            self.rank,
            self.device,
        )

        # Attach to a schedule
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

        # Run
        stage_module = pipe.get_stage_module(self.rank)
        for _ in range(2):
            # Zero gradients
            stage_module.zero_grad()
            if self.rank == 0:
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                losses = []
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier()

        # Last rank checks result
        if self.rank == self.world_size - 1:
            # Check output
            torch.testing.assert_close(out, ref_out)
            # Check loss
            # Since the reduction used in the loss function above is "sum", we use
            # "sum" here to reduce microbatch losses into a single value too.
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Every rank checks gradients
        for name, p in stage_module.named_parameters():
            ref_p = ref_mod.get_parameter(name)
            try:
                torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)
            except AssertionError:
                print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                raise

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    @parametrize("shape_inference", [True, False])
    def test_grad_with_manual(self, ScheduleClass, shape_inference):
        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        full_mod.to(self.device)

        ref_mod = copy.deepcopy(full_mod)
        x = torch.randn(batch_size, d_hid, device=self.device)
        with torch.no_grad():
            y = ref_mod(x)
            # Add a small perturbation
            target = y + torch.randn(batch_size, d_hid, device=self.device)

        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Run reference
        for _ in range(2):
            ref_mod.zero_grad()
            ref_out = ref_mod(x)
            ref_loss = loss_fn(ref_out, target)
            ref_loss.backward()

        # Get a submodule, e.g. `layers.0` or `layers.1`
        submod_name = f"layers.{self.rank}"
        stage_module = full_mod.get_submodule(submod_name)
        chunks = 2 * self.world_size

        if shape_inference:
            input_args = None
            output_args = None
        else:
            input_args = (x.chunk(chunks)[0],)
            with torch.no_grad():
                output_args = stage_module(*input_args)

        # Create a pipeline stage to wrap that submodule
        stage = PipelineStage(
            stage_module,
            self.rank,
            self.world_size,
            self.device,
            input_args=input_args,
            output_args=output_args,
        )

        # Attach to a schedule
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn, scale_grads=False)

        # Run
        for _ in range(2):
            # Zero gradients
            stage_module.zero_grad()
            if self.rank == 0:
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                losses = []
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier()

        # Last rank checks result
        if self.rank == self.world_size - 1:
            # Check output
            torch.testing.assert_close(out, ref_out)
            # Check loss
            # Since the reduction used in the loss function above is "sum", we use
            # "sum" here to reduce microbatch losses into a single value too.
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Every rank checks gradients
        ref_submod = ref_mod.get_submodule(submod_name)
        for name, p in stage_module.named_parameters():
            ref_p = ref_submod.get_parameter(name)
            try:
                torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)
            except AssertionError:
                print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                raise

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
        full_mod = MultiMLP(d_hid, n_layers=n_stages)
        full_mod.to(self.device)

        ref_mod = copy.deepcopy(full_mod)
        x = torch.randn(batch_size, d_hid, device=self.device)
        with torch.no_grad():
            y = ref_mod(x)
            # Add a small perturbation
            target = y + torch.randn(batch_size, d_hid, device=self.device)

        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Run reference
        for _ in range(2):
            ref_mod.zero_grad()
            ref_out = ref_mod(x)
            ref_loss = loss_fn(ref_out, target)
            ref_loss.backward()

        # Get a submodule, e.g. `layers.0` or `layers.1`
        stage_indices = [
            self.rank + i * self.world_size for i in range(stages_per_rank)
        ]
        print(f"Rank {self.rank} stages: {stage_indices}")
        submod_names = [f"layers.{i}" for i in stage_indices]
        stage_modules = [
            full_mod.get_submodule(submod_name) for submod_name in submod_names
        ]
        # Create a pipeline stage to wrap that submodule
        num_microbatches = (
            ScheduleClass.num_microbatches
            if hasattr(ScheduleClass, "num_microbatches")
            else 2 * self.world_size
        )
        stages = [
            PipelineStage(
                stage_module,
                stage_idx,
                n_stages,
                self.device,
            )
            for stage_module, stage_idx in zip(stage_modules, stage_indices)
        ]

        # Attach to a schedule
        schedule = ScheduleClass(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )
        if use_new_runtime:
            old_schedule = schedule
            tmp_schedule = _PipelineScheduleRuntime(
                stages,
                num_microbatches,
                loss_fn=loss_fn,
                scale_grads=False,
            )
            tmp_schedule._load_actions(old_schedule.pipeline_order)
            # test that csv round-trip works for compute_comms schedule
            schedule = _PipelineScheduleRuntime(
                stages,
                num_microbatches,
                loss_fn=loss_fn,
                scale_grads=False,
            )
            with tempfile.NamedTemporaryFile() as f:
                tmp_schedule._dump_csv(f.name)
                f.seek(0)
                schedule._load_csv(f.name, format="compute_comms")
            one_more_schedule = _PipelineScheduleRuntime(
                stages,
                num_microbatches,
                loss_fn=loss_fn,
                scale_grads=False,
            )
            one_more_schedule._load_actions(
                schedule.pipeline_order_with_comms, format="compute_comms"
            )
            self.assertEqual(
                len(schedule.pipeline_order_with_comms),
                len(
                    one_more_schedule.pipeline_order_with_comms,
                ),
            )
            for rank in schedule.pipeline_order_with_comms:
                self.assertEqual(
                    len(schedule.pipeline_order_with_comms[rank]),
                    len(
                        one_more_schedule.pipeline_order_with_comms[rank],
                    ),
                )
                for a, b in zip(
                    schedule.pipeline_order_with_comms[rank],
                    one_more_schedule.pipeline_order_with_comms[rank],
                ):
                    self.assertEqual(a, b)

        # Run
        with check_leaked_tensors() as garbage_tensors:
            for _ in range(2):
                # Zero gradients
                for stage_module in stage_modules:
                    stage_module.zero_grad()
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

        # Last rank checks result
        if self.rank == self.world_size - 1:
            # Check output
            torch.testing.assert_close(out, ref_out)
            # Check loss
            # Since the reduction used in the loss function above is "sum", we use
            # "sum" here to reduce microbatch losses into a single value too.
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Every rank checks gradients
        for stage_module, submod_name in zip(stage_modules, submod_names):
            # Get corresponding submodule from reference model
            ref_submod = ref_mod.get_submodule(submod_name)
            # Check gradients per parameter
            for name, p in stage_module.named_parameters():
                ref_p = ref_submod.get_parameter(name)
                try:
                    torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=1e-3)
                except AssertionError:
                    print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                    raise

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleWithW, ScheduleInterleavedZeroBubble])
    def test_schedule_with_native_zero_bubble(self, ScheduleClass):
        print(ScheduleClass)
        if ScheduleClass is ScheduleInterleavedZeroBubble:
            n_stages = 4
            num_microbatches = 2 * n_stages
            rank_stages = {
                0: [0, 2],
                1: [1, 3],
            }
        else:
            n_stages = ScheduleClass.n_stages
            num_microbatches = ScheduleClass.num_microbatches
            rank_stages = ScheduleClass.rank_stages

        num_steps = 4
        full_mod = MultiMLP(d_hid, n_layers=n_stages)
        full_mod.to(self.device)

        ref_mod = copy.deepcopy(full_mod)
        x = torch.randn(batch_size, d_hid, device=self.device)
        # x = torch.randn(batch_size, d_hid, device=self.device, requires_grad=True)
        with torch.no_grad():
            y = ref_mod(x)
            # Add a small perturbation
            target = y + torch.randn(batch_size, d_hid, device=self.device)

        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Create a pipeline stage to wrap that submodule
        stage_indices = rank_stages[self.rank]
        print(f"Rank {self.rank} stages: {stage_indices}")
        submod_names = [f"layers.{i}" for i in stage_indices]
        stage_modules = [
            full_mod.get_submodule(submod_name) for submod_name in submod_names
        ]
        stages = [
            PipelineStage(
                stage_module,
                stage_idx,
                n_stages,
                self.device,
            )
            for stage_module, stage_idx in zip(stage_modules, rank_stages[self.rank])
        ]

        # We set scale_grads=False since we use a loss function that sums instead of mean-reduces
        # (note: normally we recommend using mean-reduce loss functions, but we preserve at least one test case
        #        using sum scaling for completeness)
        schedule = ScheduleClass(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )

        # Run reference
        ref_x = x.detach().clone().requires_grad_(x.requires_grad)
        torch.testing.assert_close(x, ref_x)
        for _ in range(num_steps):
            ref_out = ref_mod(ref_x)
            ref_loss = loss_fn(ref_out, target)
            ref_loss.backward()

        with check_leaked_tensors() as garbage_tensors:
            # Run pipelined stages
            for _ in range(num_steps):
                if self.rank == 0:
                    schedule.step(x)
                elif self.rank == self.world_size - 1:
                    losses = []
                    schedule.step(target=target, losses=losses)
                else:
                    schedule.step()
        self.assertEqual(
            len(garbage_tensors),
            0,
            "Found leaked tensors, check logs above for debug info",
        )

        # Every rank checks parameters compared with the reference model
        for stage_module, submod_name in zip(stage_modules, submod_names):
            # Get corresponding submodule from reference model
            ref_submod = ref_mod.get_submodule(submod_name)
            # Check gradients per parameter
            for name, p in stage_module.named_parameters():
                ref_p = ref_submod.get_parameter(name)
                try:
                    torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)
                except AssertionError:
                    print(
                        f"Parameter test failed for {submod_name}.{name}: {p.grad} vs {ref_p.grad}"
                    )
                    raise

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize(
        "ScheduleClass",
        [
            ScheduleWithReorderedB,
        ],
    )
    def test_pipeline_schedule_runtime_custom_sched(self, ScheduleClass):
        n_stages = 2
        num_microbatches = 2
        stages_per_rank = 1
        full_mod = MultiMLP(d_hid, n_layers=n_stages)
        full_mod.to(self.device)

        ref_mod = copy.deepcopy(full_mod)
        x = torch.randn(batch_size, d_hid, device=self.device)
        with torch.no_grad():
            y = ref_mod(x)
            # Add a small perturbation
            target = y + torch.randn(batch_size, d_hid, device=self.device)

        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Run reference
        for _ in range(2):
            ref_mod.zero_grad()
            ref_out = ref_mod(x)
            ref_loss = loss_fn(ref_out, target)
            ref_loss.backward()

        # Get a submodule, e.g. `layers.0` or `layers.1`
        stage_indices = [
            self.rank + i * self.world_size for i in range(stages_per_rank)
        ]
        print(f"Rank {self.rank} stages: {stage_indices}")
        submod_names = [f"layers.{i}" for i in stage_indices]
        stage_modules = [
            full_mod.get_submodule(submod_name) for submod_name in submod_names
        ]
        # Create a pipeline stage to wrap that submodule
        num_microbatches = (
            ScheduleClass.num_microbatches
            if hasattr(ScheduleClass, "num_microbatches")
            else 8
        )
        stages = [
            PipelineStage(
                stage_module,
                stage_idx,
                n_stages,
                self.device,
            )
            for stage_module, stage_idx in zip(stage_modules, stage_indices)
        ]

        # Attach to a schedule
        schedule = ScheduleClass(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )
        assert isinstance(schedule, _PipelineScheduleRuntime)

        # Run
        with check_leaked_tensors() as garbage_tensors:
            for _ in range(2):
                # Zero gradients
                for stage_module in stage_modules:
                    stage_module.zero_grad()
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

        # Last rank checks result
        if self.rank == self.world_size - 1:
            # Check output
            torch.testing.assert_close(out, ref_out)
            # Check loss
            # Since the reduction used in the loss function above is "sum", we use
            # "sum" here to reduce microbatch losses into a single value too.
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Every rank checks gradients
        for stage_module, submod_name in zip(stage_modules, submod_names):
            # Get corresponding submodule from reference model
            ref_submod = ref_mod.get_submodule(submod_name)
            # Check gradients per parameter
            for name, p in stage_module.named_parameters():
                ref_p = ref_submod.get_parameter(name)
                try:
                    torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)
                except AssertionError:
                    print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                    raise

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize(
        "schedule_class", [ScheduleVShaped, ScheduleUnbalanced, ScheduleZBVZeroBubble]
    )
    @parametrize("use_new_runtime", [False, True])
    def test_non_symmetric_stage_ids(self, schedule_class, use_new_runtime):
        if schedule_class is ScheduleZBVZeroBubble:
            n_stages = 4
            rank_stages = {
                0: [0, 3],
                1: [1, 2],
            }
        else:
            n_stages = schedule_class.n_stages
            rank_stages = schedule_class.rank_stages
        full_mod = MultiMLP(d_hid, n_layers=n_stages)
        full_mod.to(self.device)

        ref_mod = copy.deepcopy(full_mod)
        x = torch.randn(batch_size, d_hid, device=self.device)
        with torch.no_grad():
            y = ref_mod(x)
            # Add a small perturbation
            target = y + torch.randn(batch_size, d_hid, device=self.device)

        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Run reference
        for _ in range(2):
            ref_mod.zero_grad()
            ref_out = ref_mod(x)
            ref_loss = loss_fn(ref_out, target)
            ref_loss.backward()

        # Create a pipeline stage to wrap that submodule
        num_microbatches = 1
        stage_indices = rank_stages[self.rank]
        print(f"Rank {self.rank} stages: {stage_indices}")
        submod_names = [f"layers.{i}" for i in stage_indices]
        stage_modules = [
            full_mod.get_submodule(submod_name) for submod_name in submod_names
        ]
        stages = [
            PipelineStage(
                stage_module,
                stage_idx,
                n_stages,
                self.device,
            )
            for stage_module, stage_idx in zip(stage_modules, rank_stages[self.rank])
        ]

        schedule = schedule_class(
            stages,
            num_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
        )
        if use_new_runtime:
            old_schedule = schedule
            schedule = _PipelineScheduleRuntime(
                stages,
                num_microbatches,
                loss_fn=loss_fn,
            )
            schedule._load_actions(old_schedule.pipeline_order)

        # Run
        # TODO how to better specify .step() when first and last stage are on rank 0...
        for _ in range(2):
            # Zero gradients
            for stage_module in stage_modules:
                stage_module.zero_grad()
            if self.rank == 0:
                losses = []
                out = schedule.step(x, target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier()

        # Last rank checks result
        if self.rank == 0:
            # Check output
            torch.testing.assert_close(out, ref_out)
            # Check loss
            # Since the reduction used in the loss function above is "sum", we use
            # "sum" here to reduce microbatch losses into a single value too.
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Every rank checks gradients
        for stage_module, submod_name in zip(stage_modules, submod_names):
            # Get corresponding submodule from reference model
            ref_submod = ref_mod.get_submodule(submod_name)
            # Check gradients per parameter
            for name, p in stage_module.named_parameters():
                ref_p = ref_submod.get_parameter(name)
                try:
                    torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)
                except AssertionError:
                    print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                    raise

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleInterleavedZeroBubble])
    def test_schedule_with_weight_update_mlp_e2e(self, ScheduleClass):
        stages_per_rank = 2
        n_stages = stages_per_rank * self.world_size
        full_mod = MultiMLPWithDw(d_hid, n_layers=n_stages)
        full_mod.to(self.device)

        ref_mod = copy.deepcopy(full_mod)
        x = torch.randn(batch_size, d_hid, device=self.device)
        with torch.no_grad():
            y = ref_mod(x)
            # Add a small perturbation
            target = y + torch.randn(batch_size, d_hid, device=self.device)

        ref_loss_fn = torch.nn.MSELoss(reduction="sum")
        full_loss_fn = torch.nn.MSELoss(reduction="sum")

        full_mod.toggle()

        # Get a submodule, e.g. `layers.0` or `layers.1`
        stage_indices = [
            self.rank + i * self.world_size for i in range(stages_per_rank)
        ]
        submod_names = [f"layers.{i}" for i in stage_indices]
        stage_modules = [
            full_mod.get_submodule(submod_name) for submod_name in submod_names
        ]

        # Run reference
        for _ in range(2):
            ref_stage_modules = [
                ref_mod.get_submodule(submod_name) for submod_name in submod_names
            ]
            for stage_module in ref_stage_modules:
                stage_module.zero_grad()

            ref_mod.zero_grad()
            ref_out = ref_mod(x)
            ref_loss = ref_loss_fn(ref_out, target)
            ref_loss.backward()

        class CustomState:
            def __init__(self, stage_module, stage_idx, rank):
                self.i = 0
                self.stage_module = stage_module
                self.stage_idx = stage_idx
                self.rank = rank

            def dw_builder(self):
                def dw_runner():
                    # This inner function would be called by PipelineStage during `backward_weight_one_chunk`
                    self.i += 1
                    print(
                        f"[Rank {self.rank}] dw_count={self.i} stage={self.stage_idx}"
                    )
                    self.stage_module.compute_dW()

                return dw_runner

        cs = {}
        for stage_module, stage_idx in zip(stage_modules, stage_indices):
            cs[stage_idx] = CustomState(stage_module, stage_idx, self.rank)

        # Create a pipeline stage to wrap that submodule
        chunks = 2
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

        # Attach to a schedule
        schedule = ScheduleClass(
            stages, chunks, loss_fn=full_loss_fn, scale_grads=False
        )

        for _ in range(2):
            # Zero gradients
            for stage_module in stage_modules:
                stage_module.zero_grad()
            if self.rank == 0:
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                losses = []
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier()
        # Last rank checks result
        if self.rank == self.world_size - 1:
            # Check output
            torch.testing.assert_close(out, ref_out)

            # Check loss
            # Since the reduction used in the loss function above is "sum", we use
            # "sum" here to reduce microbatch losses into a single value too.
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Every rank checks gradients
        for stage_module, submod_name in zip(stage_modules, submod_names):
            # Get corresponding submodule from reference model
            ref_submod = ref_mod.get_submodule(submod_name)
            # Check gradients per parameter
            for name, p in stage_module.named_parameters():
                ref_p = ref_submod.get_parameter(name)
                torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize(
        "ScheduleClass",
        [ScheduleInterleavedZeroBubble, ScheduleInterleaved1F1B],
    )
    def test_zero_bubble_with_model_kwargs(self, ScheduleClass):
        stages_per_rank = 2
        n_stages = stages_per_rank * self.world_size
        full_mod = MultiMLPKwargs(d_hid, n_layers=n_stages)
        full_mod.to(self.device)

        ref_mod = copy.deepcopy(full_mod)
        x = torch.randn(batch_size, d_hid, device=self.device)
        unused_kwarg = torch.tensor([1.0], device=self.device)

        with torch.no_grad():
            y = ref_mod(x)
            # Add a small perturbation
            target = y + torch.randn(batch_size, d_hid, device=self.device)

        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Get a submodule, e.g. `layers.0` or `layers.1`
        stage_indices = [
            self.rank + i * self.world_size for i in range(stages_per_rank)
        ]
        submod_names = [f"layers.{i}" for i in stage_indices]
        stage_modules = [
            full_mod.get_submodule(submod_name) for submod_name in submod_names
        ]
        # Run reference
        for _ in range(2):
            ref_stage_modules = [
                ref_mod.get_submodule(submod_name) for submod_name in submod_names
            ]
            for stage_module in ref_stage_modules:
                stage_module.zero_grad()

            ref_mod.zero_grad()
            ref_out = ref_mod(x, unused_kwarg=unused_kwarg)
            ref_loss = loss_fn(ref_out, target)
            ref_loss.backward()

        # Create a pipeline stage to wrap that submodule
        stages = [
            PipelineStage(
                stage_module,
                stage_idx,
                n_stages,
                self.device,
            )
            for stage_module, stage_idx in zip(stage_modules, stage_indices)
        ]

        # Attach to a schedule
        num_microbatches = (
            ScheduleClass.num_microbatches
            if hasattr(ScheduleClass, "num_microbatches")
            else 2 * self.world_size
        )
        schedule = ScheduleClass(
            stages, num_microbatches, loss_fn=loss_fn, scale_grads=False
        )

        for _ in range(2):
            # Zero gradients
            for stage_module in stage_modules:
                stage_module.zero_grad()
            if self.rank == 0:
                schedule.step(
                    x,
                    unused_kwarg=unused_kwarg.clone()
                    .unsqueeze(0)
                    .expand(num_microbatches, -1),
                )
            elif self.rank == self.world_size - 1:
                losses = []
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

        dist.barrier()
        # Last rank checks result
        if self.rank == self.world_size - 1:
            # Check output
            torch.testing.assert_close(out, ref_out)

            # Check loss
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # Every rank checks gradients
        for stage_module, submod_name in zip(stage_modules, submod_names):
            # Get corresponding submodule from reference model
            ref_submod = ref_mod.get_submodule(submod_name)
            # Check gradients per parameter
            for name, p in stage_module.named_parameters():
                ref_p = ref_submod.get_parameter(name)
                try:
                    torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=5e-3)
                except AssertionError:
                    print(
                        f"Gradient test failed for {name}: {p.grad=} vs {ref_p.grad=}"
                    )
                    raise


instantiate_parametrized_tests(ScheduleTest)

if __name__ == "__main__":
    run_tests()
