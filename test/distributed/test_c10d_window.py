# Owner(s): ["oncall: distributed"]
#
# Backend-agnostic tests for the c10d one-sided window API
# (Backend.new_window / Window.put / signal / wait_signal /
# map_remote_tensor). Modeled on torchcomms' WindowRmaTest; parameterized over
# backends like test_c10d_fault_tolerance.py so additional backends can be
# enabled by extending WINDOW_BACKENDS.

import os
import sys
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


WINDOW_BACKENDS = [
    ("nccl2", "cuda"),
]


class AbstractWindowTest:
    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        return torch.device(f"{self.device_type}:{self.rank}")

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )
        self.backend = dist.get_backend_impl(device=self.device)
        self.assertTrue(dist._supports_window())
        self.assertTrue(self.backend.supports_window)

    def _make_pool(self):
        return torch.cuda.MemPool(self.backend.mem_allocator)

    def _probe_window_support(self, pool):
        # Symmetric-window registration needs a transport that can expose
        # VMM-backed memory across ranks (NVLink intra-node or IB inter-node).
        # The probe is collective, so every rank succeeds or skips together.
        with torch.cuda.use_mem_pool(pool):
            probe_buf = torch.empty(1, dtype=torch.float, device=self.device)
        try:
            probe_win = dist._new_window()
            probe_win.tensor_register(probe_buf)
            probe_win.tensor_deregister()
        except RuntimeError as e:
            self.skipTest(f"symmetric window registration unsupported: {e}")

    def _window_ring_put(self, count, dtype, async_op, use_tensor_in_new_window):
        # Ring put: each rank puts its payload into the next rank's window and
        # signals it; each rank waits for the previous rank's signal and
        # verifies the received slice. Port of torchcomms' _window_put_test.
        pool = self._make_pool()
        self._probe_window_support(pool)
        with torch.cuda.use_mem_pool(pool):
            input_tensor = torch.full(
                [count], self.rank, dtype=dtype, device=self.device
            )
            win_buf = torch.ones(
                [count * self.world_size], dtype=dtype, device=self.device
            )
        dist.barrier()

        if use_tensor_in_new_window:
            win = dist._new_window(win_buf)
        else:
            win = dist._new_window()
            win.tensor_register(win_buf)
        dist.barrier()

        dst_rank = (self.rank + 1) % self.world_size
        src_rank = (self.rank - 1 + self.world_size) % self.world_size

        for iteration in range(4):
            work = win.put(input_tensor, dst_rank, dst_rank * count, async_op)
            if async_op:
                work.wait()
            signal_work = win.signal(dst_rank, async_op)
            if async_op:
                signal_work.wait()

            wait_work = win.wait_signal(src_rank, async_op)
            if async_op:
                wait_work.wait()

            local_tensor = win.map_remote_tensor(self.rank)
            torch.cuda.synchronize()
            received = local_tensor[self.rank * count : (self.rank + 1) * count]
            expected = torch.full([count], src_rank, dtype=dtype, device=self.device)
            self.assertEqual(received, expected, msg=f"iteration {iteration}")

        torch.cuda.synchronize()
        win.tensor_deregister()
        del win
        del pool
        torch.cuda.synchronize()

    def test_put_signal_wait_sync(self):
        self._init_pg()
        self._window_ring_put(1024, torch.float, False, False)

    def test_put_signal_wait_async(self):
        self._init_pg()
        self._window_ring_put(1024, torch.float, True, False)

    def test_put_dtypes_and_sizes(self):
        self._init_pg()
        for count, dtype in [
            (4, torch.float),
            (1024, torch.int),
            (1024 * 64, torch.int8),
        ]:
            self._window_ring_put(count, dtype, False, False)

    def test_new_window_with_tensor(self):
        self._init_pg()
        self._window_ring_put(1024, torch.float, False, True)

    def test_map_remote_tensor_local(self):
        self._init_pg()
        pool = self._make_pool()
        self._probe_window_support(pool)
        count = 1024
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.arange(count, dtype=torch.float, device=self.device)
        win = dist._new_window()
        win.tensor_register(win_buf)
        dist.barrier()

        local_tensor = win.map_remote_tensor(self.rank)
        self.assertEqual(local_tensor.dtype, win_buf.dtype)
        self.assertEqual(local_tensor.shape, win_buf.shape)
        self.assertEqual(local_tensor, win_buf)

        win.tensor_deregister()

    def test_window_attr(self):
        self._init_pg()
        pool = self._make_pool()
        self._probe_window_support(pool)
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.ones(16, dtype=torch.float, device=self.device)
        win = dist._new_window(win_buf)
        dist.barrier()
        from torch._C._distributed_c10d import WindowAccessType

        attr = win.get_attr((self.rank + 1) % self.world_size)
        self.assertIsInstance(attr.access_type, WindowAccessType)
        win.tensor_deregister()

    def test_register_errors(self):
        self._init_pg()
        pool = self._make_pool()
        self._probe_window_support(pool)
        win = dist._new_window()
        # Registering a tensor outside the backend mempool must fail.
        plain = torch.ones(16, dtype=torch.float, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "mempool|MemPool"):
            win.tensor_register(plain)
        # Ops before registration must fail.
        with self.assertRaisesRegex(RuntimeError, "not registered"):
            win.put(plain, 0, 0, False)
        with self.assertRaisesRegex(RuntimeError, "not registered"):
            win.signal(0, False)
        # Double registration must fail.
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.ones(16, dtype=torch.float, device=self.device)
        win.tensor_register(win_buf)
        dist.barrier()
        with self.assertRaisesRegex(
            RuntimeError, "double registration|Double registration"
        ):
            win.tensor_register(win_buf)
        win.tensor_deregister()

    def test_put_out_of_bounds(self):
        self._init_pg()
        pool = self._make_pool()
        self._probe_window_support(pool)
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.ones(16, dtype=torch.float, device=self.device)
            src = torch.ones(16, dtype=torch.float, device=self.device)
        win = dist._new_window(win_buf)
        dist.barrier()
        with self.assertRaisesRegex(RuntimeError, "exceeds the window size"):
            win.put(src, (self.rank + 1) % self.world_size, 8, False)
        win.tensor_deregister()


def _make_window_test_class(backend_name, device_type):
    class WindowTest(AbstractWindowTest, MultiProcessTestCase):
        pass

    WindowTest.backend_name = backend_name
    WindowTest.device_type = device_type
    WindowTest.__name__ = f"{backend_name.capitalize()}WindowTest"
    WindowTest.__qualname__ = WindowTest.__name__
    cls = unittest.skipIf(
        not dist.is_backend_available(backend_name),
        f"{backend_name} backend is not available",
    )(WindowTest)
    if device_type == "cuda":
        cls = unittest.skipIf(
            not TEST_CUDA or torch.cuda.device_count() < 2,
            "window CUDA tests require at least 2 GPUs",
        )(cls)
    return cls


for backend_name, device_type in WINDOW_BACKENDS:
    globals()[f"{backend_name.capitalize()}WindowTest"] = _make_window_test_class(
        backend_name, device_type
    )


if __name__ == "__main__":
    run_tests()
