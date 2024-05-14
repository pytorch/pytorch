# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import os
import sys
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.pipelining import (
    pipe_split,
    pipeline,
    PipelineStage,
    Schedule1F1B,
    ScheduleGPipe,
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


d_hid = 512
batch_size = 256
chunks = 4

torch.manual_seed(0)


class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.register_buffer("cval", torch.randn((d_hid,), requires_grad=False))
        self.lin0 = torch.nn.Linear(d_hid, d_hid)
        self.lin1 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y=torch.zeros(batch_size, d_hid)):
        x = torch.mm(x, self.mm_param0)
        x = x + y
        x = torch.relu(x)
        # try passing a value that doesn't require_grad across skip boundaries
        a_constant = self.cval.clone()
        x = self.lin0(x)
        pipe_split()
        x = torch.relu(x) + a_constant
        x = torch.mm(x, self.mm_param1)
        x = self.lin1(x)
        x = torch.relu(x)
        return x


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
    def test_ec_forward(self):
        # Setting this flag for numerical stability
        torch.distributed.pipelining.microbatch._debug_mask_minibatches = True

        mod = ExampleCode()
        mod.to(self.device)

        x = torch.randn(batch_size, d_hid, device=self.device)
        y = torch.randn(batch_size, d_hid, device=self.device)

        pipe = pipeline(
            mod,
            chunks,
            example_args=(x,),
            example_kwargs={"y": y},
        )

        stage = PipelineStage(
            pipe,
            self.rank,
            device=self.device,
        )

        # Attach to a schedule
        schedule = ScheduleGPipe(stage, chunks)

        # Run
        if self.rank == 0:
            schedule.step(x, y=y)
        else:
            out = schedule.step()

        dist.barrier()

        # Last rank checks result
        if self.rank == self.world_size - 1:
            ref_out = mod(x, y=y)
            torch.testing.assert_close(out, ref_out)

        # Test qualname mapping
        submod_keys = stage.submod.state_dict().keys()
        # Confirm keys are consistent with original model
        old_keys = mod.state_dict().keys()
        assert all(k in old_keys for k in submod_keys)
        # Reset this flag
        torch.distributed.pipelining.microbatch._debug_mask_minibatches = False

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    def test_ec_backward(self, ScheduleClass):
        mod = ExampleCode()
        mod.to(self.device)

        x = torch.randn(batch_size, d_hid, device=self.device)
        y = torch.randn(batch_size, d_hid, device=self.device)
        target = torch.randn(batch_size, d_hid, device=self.device)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        pipe = pipeline(
            mod,
            chunks,
            example_args=(x,),
            example_kwargs={"y": y},
        )

        stage = PipelineStage(
            pipe,
            self.rank,
            device=self.device,
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
