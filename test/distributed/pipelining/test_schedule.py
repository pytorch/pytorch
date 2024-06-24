# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import copy
import logging
import os
import sys
import tempfile
import unittest

from model_registry import ModelWithKwargs, MultiMLP

import torch
import torch.distributed as dist
from torch.distributed.pipelining import (
    pipeline,
    PipelineStage,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleLoopedBFS,
)
from torch.distributed.pipelining.stage import _PipelineStageBase

from torch.distributed.pipelining.utils import (
    format_pipeline_order,
    validate_pipeline_order,
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


class MockPipelineStage(_PipelineStageBase):
    def __init__(self, *args, **kwargs):
        # Mock the necessary attributes
        self.num_stages = kwargs.get("num_stages", 1)
        self.group_size = kwargs.get("group_size", 1)
        self.group_rank = kwargs.get("group_rank", 0)
        self.group = kwargs.get("group", None)

    def _create_grad_recv_info(self, *args, **kwargs):
        return None

    def _prepare_forward_infra(self, n_microbatches):
        pass

    def _prepare_backward_infra(self, n_microbatches):
        pass


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


instantiate_parametrized_tests(ScheduleTest)


class TestSchedulePlan(unittest.TestCase):
    @parametrize("ScheduleClass", [ScheduleInterleaved1F1B, ScheduleLoopedBFS])
    def test_pipeline_order(self, ScheduleClass):
        # Define a list of test cases with varying num_local_stages, num_microbatches, and group_size
        # These should succeed since num_microbatches % group_size == 0
        test_cases = [
            # small number of stages
            (2, 2, 2),
            (2, 4, 4),
            (2, 8, 2),
            (2, 8, 4),
            (2, 8, 8),
            (4, 4, 4),
            (4, 8, 4),
            (4, 8, 8),
            # large microbatches
            (4, 16, 4),
            (4, 32, 4),
            (4, 64, 4),
            # large groups
            (4, 16, 16),
            (4, 32, 32),
            (4, 128, 64),
            # odd num pipeline stages
            (3, 2, 2),
            (3, 8, 2),
            (3, 12, 4),
            # odd group_sizes
            (4, 6, 3),
            (4, 10, 5),
        ]
        for num_local_stages, num_microbatches, group_size in test_cases:
            with self.subTest(
                num_local_stages=num_local_stages,
                num_microbatches=num_microbatches,
                group_size=group_size,
            ):
                print(f"{num_local_stages=} {num_microbatches=} {group_size=}")
                num_stages = num_local_stages * group_size
                stages = [
                    MockPipelineStage(group_size=group_size, num_stages=num_stages)
                    for i in range(num_local_stages)
                ]

                schedule = ScheduleClass(stages, num_microbatches)
                formatted_pipeline_order = format_pipeline_order(
                    schedule.pipeline_order
                )
                # print(formatted_pipeline_order)
                validate_pipeline_order(
                    schedule.pipeline_order, num_microbatches, num_stages
                )


instantiate_parametrized_tests(TestSchedulePlan)

if __name__ == "__main__":
    # Run only the TestSchedulePlan tests (single process)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSchedulePlan)
    runner = unittest.TextTestRunner()
    runner.run(suite)

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
