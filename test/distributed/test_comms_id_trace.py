# Owner(s): ["oncall: distributed"]

"""
End-to-end verification that "Comms Id" appears in PyTorch profiler Chrome
trace JSON for NCCL collectives.

Usage:
    pytest test/distributed/test_comms_id_trace.py
"""

import json
import os
import tempfile

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests


class TestCommsIdTrace(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls):
        return "nccl"

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_comms_id_in_trace(self):
        """Verify that Comms Id appears in profiler trace for NCCL collectives."""
        device = self.device

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            tensor = torch.ones(4, 4, device=device) * self.rank
            dist.all_reduce(tensor)
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        trace_dir = tempfile.mkdtemp(prefix="comms_id_trace_")
        trace_path = os.path.join(trace_dir, f"trace_rank{self.rank}.json")
        prof.export_chrome_trace(trace_path)

        with open(trace_path) as f:
            trace = json.load(f)

        comms_ids = []
        for event in trace.get("traceEvents", []):
            args = event.get("args", {})
            if "Comms Id" in args:
                comms_ids.append(args["Comms Id"])

        self.assertGreater(len(comms_ids), 0, "No 'Comms Id' found in trace events")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_comms_id_consistent_across_ranks(self):
        """Verify that the same collective has the same Comms Id on all ranks."""
        device = self.device

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            tensor = torch.ones(4, 4, device=device) * self.rank
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        trace_dir = tempfile.mkdtemp(prefix="comms_id_trace_")
        trace_path = os.path.join(trace_dir, f"trace_rank{self.rank}.json")
        prof.export_chrome_trace(trace_path)

        with open(trace_path) as f:
            trace = json.load(f)

        comms_ids = []
        for event in trace.get("traceEvents", []):
            args = event.get("args", {})
            if "Comms Id" in args:
                comms_ids.append(args["Comms Id"])

        # Gather comms_ids from all ranks to rank 0 for comparison
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, comms_ids)

        # All ranks should have the same set of comms_ids
        if self.rank == 0:
            self.assertGreater(len(gathered[0]), 0, "No Comms Id found on rank 0")
            for r in range(1, self.world_size):
                self.assertEqual(
                    sorted(gathered[0]),
                    sorted(gathered[r]),
                    f"Comms Ids differ between rank 0 and rank {r}",
                )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_comms_id_differs_across_operations(self):
        """Verify that different collectives get different Comms Ids."""
        device = self.device

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            tensor = torch.ones(4, 4, device=device) * self.rank
            dist.all_reduce(tensor)
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        trace_dir = tempfile.mkdtemp(prefix="comms_id_trace_")
        trace_path = os.path.join(trace_dir, f"trace_rank{self.rank}.json")
        prof.export_chrome_trace(trace_path)

        with open(trace_path) as f:
            trace = json.load(f)

        comms_ids = []
        for event in trace.get("traceEvents", []):
            args = event.get("args", {})
            if "Comms Id" in args:
                comms_ids.append(args["Comms Id"])

        # Two allreduce ops should produce at least 2 distinct comms_ids
        unique_ids = set(comms_ids)
        self.assertGreaterEqual(
            len(unique_ids), 2, "Expected different Comms Ids for different operations"
        )


if __name__ == "__main__":
    run_tests()
