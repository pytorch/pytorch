# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import copy
import logging
import os
import sys
import tempfile

from model_registry import ModelWithKwargs, MultiMLP, MultiMLPWithDw
from schedule_registry import ScheduleUnbalanced, ScheduleVShaped

import torch
import torch.distributed as dist
from torch.distributed.pipelining import (
    pipeline,
    PipelineStage,
    Schedule1F1B,
    ScheduleFlexibleInterleaved1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleLoopedBFS,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcContinousTest,
    requires_nccl,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skip_but_pass_in_sandcastle_if,
)


logger = logging.getLogger(__name__)

d_hid = 512
batch_size = 256

torch.manual_seed(0)


class ScheduleTest(MultiProcContinousTest):
    @classmethod
    def backend_str(cls) -> str:
        # Testing with NCCL backend
        return "nccl"

    @classmethod
    def setUpClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, before any test starts.
        Set up the device.
        """
        super().setUpClass()
        dev_id = cls.rank % torch.cuda.device_count()
        cls.device = torch.device(f"cuda:{dev_id}")

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
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn)

        # Run
        for _ in range(20):
            if self.rank == 0:
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                losses = []
                out = schedule.step(target=target, losses=losses)
            else:
                schedule.step()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    def test_kwargs_with_tracer(self, ScheduleClass):
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
        )

        # Attach to a schedule
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn)

        # Run
        if self.rank == 0:
            schedule.step(x, y=y)
        elif self.rank == self.world_size - 1:
            losses = []
            out = schedule.step(target=target, losses=losses)
        else:
            schedule.step()

        dist.barrier()

        # Last rank checks result
        if self.rank == self.world_size - 1:
            ref_out = mod(x, y=y)
            ref_loss = loss_fn(ref_out, target)
            pipe_loss = sum(losses)
            torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=5e-3)
            torch.testing.assert_close(pipe_loss, ref_loss)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    @parametrize("ModelClass", [MultiMLP])
    def test_grad_with_tracer(self, ScheduleClass, ModelClass):
        mod = ModelClass(d_hid)
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
        chunks = 4
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
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn)

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
    def test_grad_with_manual(self, ScheduleClass):
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
        chunks = 4
        # Create a pipeline stage to wrap that submodule
        stage = PipelineStage(
            stage_module,
            self.rank,
            self.world_size,
            self.device,
            input_args=x.chunk(chunks)[0],
        )

        # Attach to a schedule
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn)

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
    @parametrize("ScheduleClass", [ScheduleInterleaved1F1B, ScheduleLoopedBFS])
    def test_grad_with_manual_interleaved(self, ScheduleClass):
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
        chunks = 8
        input_args = x.chunk(chunks)[0]
        stages = [
            PipelineStage(
                stage_module,
                stage_idx,
                n_stages,
                self.device,
                input_args=input_args,
            )
            for stage_module, stage_idx in zip(stage_modules, stage_indices)
        ]

        # Attach to a schedule
        schedule = ScheduleClass(stages, chunks, loss_fn=loss_fn)

        # Run
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
                try:
                    torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)
                except AssertionError:
                    print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                    raise

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleVShaped, ScheduleUnbalanced])
    def test_non_symmetric_stage_ids(self, ScheduleClass):
        n_stages = ScheduleClass.n_stages
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
        chunks = 1
        input_args = x.chunk(chunks)[0]
        rank_stages = ScheduleClass.rank_stages
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
                input_args=input_args,
            )
            for stage_module, stage_idx in zip(stage_modules, rank_stages[self.rank])
        ]

        # Attach to a schedule
        stage_index_to_group_rank = {
            value: key for key, values in rank_stages.items() for value in values
        }
        schedule = ScheduleClass(
            stages, chunks, stage_index_to_group_rank, loss_fn=loss_fn
        )

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
    @parametrize("ScheduleClass", [ScheduleFlexibleInterleaved1F1B])
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
        input_args = x.chunk(chunks)[0]
        stages = [
            PipelineStage(
                stage_module,
                stage_idx,
                n_stages,
                self.device,
                input_args=input_args,
                dw_builder=cs[stage_idx].dw_builder,
            )
            for stage_module, stage_idx in zip(stage_modules, stage_indices)
        ]

        # Attach to a schedule
        schedule = ScheduleClass(
            stages, chunks, loss_fn=full_loss_fn, enable_zero_bubble=True
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


instantiate_parametrized_tests(ScheduleTest)


if __name__ == "__main__":
    # Check if GPU and NCCL are available
    if not (
        dist.is_available()
        and dist.is_nccl_available()
        and torch.cuda.device_count() > 1
    ):
        print(
            "c10d NCCL not available or not enough GPUs, skipping tests",
            file=sys.stderr,
        )
        sys.exit(0)

    rank = int(os.getenv("RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 2))

    if rank != -1:
        # Launched with torchrun or other multi-proc launchers. Directly run the test.
        ScheduleTest.run_rank(rank, world_size)
    else:
        # Launched as a single process. Spawn subprocess to run the tests.
        # Also need a rendezvous file for `init_process_group` purpose.
        rdvz_file = tempfile.NamedTemporaryFile(delete=False).name
        torch.multiprocessing.spawn(
            ScheduleTest.run_rank,
            nprocs=world_size,
            args=(world_size, rdvz_file),
        )
