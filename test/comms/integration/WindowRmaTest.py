#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import os
import unittest

from integration.helpers.TorchCommTestHelpers import (
    get_dtype_name,
    TorchCommTestWrapper,
)

import torch
from torch.comms import TorchCommWinAccessType


def _should_skip_rma_test():
    """Check if RMA tests should be skipped.

    RMA window ops are supported by the nccl backend on NCCL 2.29+ (via
    ncclCommWindowRegister / ncclPutSignal / ncclSignal / ncclWaitSignal).
    Returns (should_skip, reason) where should_skip is a bool and reason is the
    skip message (only meaningful when should_skip is True).
    """
    backend = os.getenv("TEST_BACKEND", "").lower()
    if backend == "nccl":
        # NCCL 2.29+ runtime check happens in the backend at new_window() time.
        return False, ""
    return True, "RMA window ops require nccl backend"


_rma_skip, _rma_skip_reason = _should_skip_rma_test()


class WindowRmaTest(unittest.TestCase):
    """Test class for Window RMA operations in TorchComm."""

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()
        self.allocator = self.torchcomm.get_mem_allocator()

        # Probe NCCL symmetric-window support. NCCL_WIN_COLL_SYMMETRIC needs a
        # transport that can expose VMM-backed memory across ranks (NVLink
        # intra-node or IB inter-node). On hosts without either, NCCL leaves
        # the window handle null and the backend now surfaces that as a
        # ncclInvalidUsage. The probe is itself collective, so every rank
        # either succeeds together or skips together.
        if self.torchcomm.get_backend() == "nccl":
            pool = torch.cuda.MemPool(self.allocator)
            with torch.cuda.use_mem_pool(pool):
                probe_buf = torch.empty(1, dtype=torch.float, device=self.device)
            try:
                probe_win = self.torchcomm.new_window()
                probe_win.tensor_register(probe_buf)
                probe_win.tensor_deregister()
            except RuntimeError as e:
                self.skipTest(
                    "NCCL symmetric window registration unsupported on this "
                    f"hardware (need NVLink or InfiniBand): {e}"
                )

    def tearDown(self):
        self.allocator = None
        self.torchcomm = None
        self.wrapper = None

    def _window_put_test(
        self, count, dtype, async_op, async_signal, use_tensor_in_new_window=False
    ):
        """Test window put operation.

        Args:
            count: Number of elements in the tensor.
            dtype: Data type of the tensor.
            async_op: Whether to use async put operation.
            async_signal: Whether to use async signal operation.
            use_tensor_in_new_window: If True, pass tensor directly to new_window()
                instead of calling tensor_register() separately.
        """
        put_stream = torch.cuda.Stream()
        wait_stream = torch.cuda.Stream()

        # Both the source tensor and the destination window must live in the
        # NCCL mempool: backends like the upstream NCCL window APIs require
        # the buffer to be inside a symmetric (VMM-backed) registered
        # segment.
        pool = torch.cuda.MemPool(self.allocator)
        with torch.cuda.use_mem_pool(pool):
            input_tensor = torch.full(
                [count], self.rank, dtype=dtype, device=self.device
            )
            win_buf = torch.ones(
                [count * self.num_ranks], dtype=dtype, device=self.device
            )

        self.torchcomm.barrier(False)

        if use_tensor_in_new_window:
            # New API: pass tensor directly to new_window()
            win = self.torchcomm.new_window(win_buf)
        else:
            # Legacy API: create window then register tensor separately
            win = self.torchcomm.new_window()
            win.tensor_register(win_buf)
        self.torchcomm.barrier(False)

        dst_rank = (self.rank + 1) % self.num_ranks
        src_rank = (self.rank - 1 + self.num_ranks) % self.num_ranks

        # Perform multiple put operations to test repeated usage
        num_iterations = 10
        for iteration in range(num_iterations):
            # Put the tensor to the Window of the next rank
            with torch.cuda.stream(put_stream):
                work = win.put(input_tensor, dst_rank, dst_rank * count, async_op)
                if async_op:
                    work.wait()
            # signal the next rank to proceed
            with torch.cuda.stream(put_stream):
                signal_work = win.signal(dst_rank, async_signal)
                if async_signal:
                    signal_work.wait()

            # wait signal from the previous rank
            with torch.cuda.stream(wait_stream):
                wait_signal_work = win.wait_signal(src_rank, async_signal)
                if async_signal:
                    wait_signal_work.wait()

                local_tensor = win.map_remote_tensor(self.rank)

            # wait for the data from the previous rank to be ready
            wait_stream.synchronize()
            output_tensor = local_tensor[self.rank * count : (self.rank + 1) * count]

            target_tensor = (
                torch.ones(
                    [count],
                    dtype=dtype,
                    device=self.device,
                )
                * src_rank
            )

            torch.testing.assert_close(
                output_tensor,
                target_tensor,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Rank {self.rank} Iteration {iteration} Expected {target_tensor} but got {output_tensor}",
            )

        # Cleanup
        wait_stream.synchronize()
        put_stream.synchronize()
        win.tensor_deregister()
        del win
        del pool
        torch.cuda.synchronize()

    def _map_remote_tensor_device_agnostic_test(self, count, dtype):
        """Helper function to test map_remote_tensor with device-agnostic access."""
        print(
            f"Testing map_remote_tensor_device_agnostic with count={count}, dtype={get_dtype_name(dtype)}"
        )

        pool = torch.cuda.MemPool(self.allocator)
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.arange(
                count * self.num_ranks, dtype=dtype, device=self.device
            )

        win = self.torchcomm.new_window()
        win.tensor_register(win_buf)
        self.torchcomm.barrier(False)

        # Test local access
        local_tensor = win.map_remote_tensor(self.rank)
        self.assertEqual(local_tensor.dtype, win_buf.dtype)
        self.assertEqual(local_tensor.shape, win_buf.shape)
        torch.testing.assert_close(local_tensor, win_buf, rtol=0, atol=0)

        # Test remote access (only for unified memory)
        remote_rank = (self.rank + 1) % self.num_ranks
        win_attr = win.get_attr(remote_rank)
        if win_attr.access_type == TorchCommWinAccessType.WIN_ACCESS_TYPE_UNIFIED:
            remote_tensor = win.map_remote_tensor(remote_rank)
            self.assertEqual(remote_tensor.dtype, dtype)
            self.assertEqual(remote_tensor.shape, win_buf.shape)

            expected_data = torch.arange(
                count * self.num_ranks, dtype=dtype, device=self.device
            )
            torch.testing.assert_close(
                remote_tensor, expected_data, rtol=1e-5, atol=1e-5
            )

        # Cleanup
        win.tensor_deregister()
        del win
        del pool

    @unittest.skipIf(_rma_skip, _rma_skip_reason)
    def test_all_tests(self):
        """Run all tests with all parameter combinations."""
        counts = [4, 1024, 1024 * 1024]
        dtypes = [torch.float, torch.int, torch.int8]
        async_ops = [True, False]
        async_signals = [True, False]

        # Nested loops for all parameter combinations
        for (
            count,
            dtype,
            async_op,
            async_signal,
        ) in itertools.product(
            counts,
            dtypes,
            async_ops,
            async_signals,
        ):
            # Create a descriptive test name for better test output
            test_name = f"Count_{count}_{get_dtype_name(dtype)}_AsyncOp_{async_op}_AsyncSignal_{async_signal}"
            print(f"Running _window_put_test with parameters: {test_name}")

            self._window_put_test(count, dtype, async_op, async_signal)

        # Test map_remote_tensor_device_agnostic with specific dtypes
        dtypes_to_test = [torch.float32, torch.int32, torch.bfloat16]
        count = 1024
        for dtype in dtypes_to_test:
            print("Running _map_remote_tensor_device_agnostic_test")
            self._map_remote_tensor_device_agnostic_test(count, dtype)

    @unittest.skipIf(_rma_skip, _rma_skip_reason)
    def test_new_window_with_tensor(self):
        """Test that new_window() accepts an optional tensor argument.

        This tests the new API where tensor can be passed directly to new_window()
        instead of calling tensor_register() separately.
        """
        counts = [4, 1024]
        dtypes = [torch.float, torch.int]

        for count, dtype in itertools.product(counts, dtypes):
            test_name = (
                f"Count_{count}_{get_dtype_name(dtype)}_with_tensor_in_new_window"
            )
            print(f"Running _window_put_test with tensor in new_window: {test_name}")

            # Test with use_tensor_in_new_window=True
            self._window_put_test(
                count,
                dtype,
                async_op=False,
                async_signal=False,
                use_tensor_in_new_window=True,
            )


if __name__ == "__main__":
    unittest.main()
