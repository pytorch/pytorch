#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Backend-agnostic profiler test base class.

This module provides a reusable ProfilerTest class that can be used by different
backends (nccl, etc.) by passing in the backend name, device, and a
validation function.

Example usage:
    class MyBackendProfilerTest(ProfilerTestBase):
        def __init__(self, *args, **kwargs):
            super().__init__(
                backend="my_backend",
                device=torch.device("cuda"),
                validation_func=self._validate,
                *args,
                **kwargs,
            )

        def _validate(self, per_coll_meta):
            # Backend-specific validation
            self.assertEqual(len(per_coll_meta["barrier"]), 1)
            ...
"""

import json
import os
import tempfile
import unittest
from collections import defaultdict
from collections.abc import Callable

import torch
from torch.comms import new_comm, ReduceOp
from torch.profiler import profile


PROFILER_TEST_TENSOR_COUNT: int = 4


def get_profiler_meta(prof):
    """Extract profiler metadata from trace.

    Torch profiler includes metadata in an inserted operator called "record_param_comms".
    This function exports the trace and extracts those events.
    """
    # Intentionally not a context manager: we need the path after close, and
    # delete=False keeps the file for export_chrome_trace to write into.
    tf = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w+t", suffix=".json", delete=False
    )
    tf.close()
    trace_file = tf.name

    prof.export_chrome_trace(trace_file)
    with open(trace_file) as f:
        events = json.load(f)["traceEvents"]
    print(f"Trace saved to {trace_file}")

    os.remove(trace_file)

    return [e for e in events if e.get("name") == "record_param_comms"]


class ProfilerTestBase(unittest.TestCase):
    """Backend-agnostic profiler test base class.

    This class provides common profiler testing functionality that can be reused
    across different backends by passing in the backend name, device, and a
    validation function.
    """

    def __init__(
        self,
        backend: str,
        device: torch.device,
        validation_func: Callable[[dict[str, list]], None],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._backend = backend
        self._device = device
        self._validation_func = validation_func

    def setUp(self):
        """Set up test environment before each test."""
        self.torchcomm = new_comm(self._backend, self._device, name="comms_test_name")

    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, "torchcomm") and self.torchcomm:
            self.torchcomm = None

    def sanity_check_profiler_meta(self, meta_events):
        """Validate basic fields in profiler events.

        Torch profiler includes metadata in an inserted operator called "record_param_comms".
        This method tests for basic fields in profiler events that correspond to
        communication collectives.
        """
        per_coll_meta = defaultdict(list)
        for e in meta_events:
            args = e.get("args", {})
            collname = args.get("Collective name", "")
            self.assertNotEqual(collname, "")
            self.assertNotEqual(args.get("dtype", ""), "")

            per_coll_meta[collname].append(args)

            self.assertEqual(args["Process Group Name"], "comms_test_name")
            self.assertNotEqual(args["Process Group Ranks"], "")

            self.assertGreaterEqual(args.get("In msg nelems", -1), 0)
            self.assertGreaterEqual(args.get("Out msg nelems", -1), 0)
            self.assertGreaterEqual(args.get("Group size", -1), 0)

        return per_coll_meta

    def run_all_collective_operations(self):
        """Run all collective operations to generate profiler traces."""
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()

        # Calculate device index
        if self._device.type == "cuda":
            device_index = self.rank % torch.cuda.device_count()
            self.device = torch.device(f"cuda:{device_index}")
        else:
            self.device = self._device

        options = {"dtype": torch.float, "device": self.device}

        # Prepare tensors
        send_tensor = torch.ones(PROFILER_TEST_TENSOR_COUNT, **options) * float(
            self.rank + 1
        )
        recv_tensor = torch.zeros(PROFILER_TEST_TENSOR_COUNT, **options)
        tensors_all_ranks = [
            torch.zeros(PROFILER_TEST_TENSOR_COUNT, **options)
            for _ in range(self.num_ranks)
        ]
        input_tensors = [
            torch.ones(PROFILER_TEST_TENSOR_COUNT, **options) * float(self.rank + 1)
            for _ in range(self.num_ranks)
        ]
        recv_tensor_single = torch.zeros(
            PROFILER_TEST_TENSOR_COUNT * self.num_ranks, **options
        )
        send_tensor_single = torch.zeros(
            PROFILER_TEST_TENSOR_COUNT * self.num_ranks, **options
        )

        input_split_sizes = [PROFILER_TEST_TENSOR_COUNT] * self.num_ranks
        output_split_sizes = [PROFILER_TEST_TENSOR_COUNT] * self.num_ranks

        send_rank = (self.rank + 1) % self.num_ranks
        recv_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

        # Execute all operations
        if self.rank % 2 == 0:
            self.torchcomm.send(send_tensor, send_rank, False)
            self.torchcomm.recv(recv_tensor, recv_rank, False)
        else:
            self.torchcomm.recv(recv_tensor, recv_rank, False)
            self.torchcomm.send(send_tensor, send_rank, False)

        self.torchcomm.all_reduce(send_tensor, ReduceOp.SUM, False)
        self.torchcomm.reduce(send_tensor, 0, ReduceOp.SUM, False)

        self.torchcomm.all_gather_single(recv_tensor_single, send_tensor, False)
        self.torchcomm.all_gather(tensors_all_ranks, send_tensor, False)
        self.torchcomm.gather(tensors_all_ranks, send_tensor, 0, False)

        self.torchcomm.reduce_scatter_single(
            recv_tensor, send_tensor_single, ReduceOp.SUM, False
        )
        self.torchcomm.reduce_scatter(
            recv_tensor, tensors_all_ranks, ReduceOp.SUM, False
        )
        self.torchcomm.scatter(recv_tensor, tensors_all_ranks, 0, False)

        self.torchcomm.all_to_all(tensors_all_ranks, input_tensors, False)
        self.torchcomm.all_to_all_single(recv_tensor_single, send_tensor_single, False)
        self.torchcomm.all_to_all_v_single(
            recv_tensor_single,
            send_tensor_single,
            output_split_sizes,
            input_split_sizes,
            False,
        )

        self.torchcomm.broadcast(send_tensor, 0, False)

        work = self.torchcomm.barrier(False)
        work.wait()
        return work

    def test_all_operations(self):
        """Test that all collective operations produce correct profiler output."""
        with profile() as prof:
            self.run_all_collective_operations()
            rank = self.torchcomm.get_rank()
            self.torchcomm.finalize()

        # Synchronize before checking results
        if self._device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

        if rank == 0:
            meta_events = get_profiler_meta(prof)
            self.assertGreater(len(meta_events), 0)

            per_coll_meta = self.sanity_check_profiler_meta(meta_events)
            # Call backend-specific validation
            self._validation_func(per_coll_meta)
