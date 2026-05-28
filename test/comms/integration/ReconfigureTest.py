#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Tests for the reconfigure() Fault Tolerance API in TorchComms.

This test verifies:
1. Backends that don't implement reconfigure() raise RuntimeError
2. Backends that implement reconfigure() can successfully reconfigure the
   communicator with a new set of peers
3. After successful reconfigure, collective operations are permitted
"""

import os
import unittest
from datetime import timedelta

from integration.helpers.TorchCommTestHelpers import skip_backend, TorchCommTestWrapper

import torch
from torch.distributed import TCPStore


class ReconfigureTest(unittest.TestCase):
    """Test class for reconfigure() fault tolerance API."""

    SUPPORTED_BACKENDS = {"mccl", "gloo", "nccl"}

    _shared_store = None

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        self.backend = os.getenv("TEST_BACKEND", "")

        if self._is_supported_backend():
            self.rank = int(
                os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0))
            )
            self.world_size = int(
                os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
            )

            if self.backend == "gloo":
                self.device = torch.device(os.environ.get("TEST_DEVICE", "cpu"))
            else:
                if not torch.cuda.is_available():
                    self.skipTest("CUDA is not available")

                if self.backend == "mccl":
                    os.environ["NCCL_COMM_STATE_DEBUG_TOPO"] = "nolocal"
                    os.environ["NCCL_IGNORE_TOPO_LOAD_FAILURE"] = "1"

                device_id = self.rank % torch.cuda.device_count()
                self.device = torch.device(f"cuda:{device_id}")

            if ReconfigureTest._shared_store is None:
                master_addr = os.environ.get("MASTER_ADDR", "localhost")
                master_port = int(os.environ.get("MASTER_PORT", "29500"))

                ReconfigureTest._shared_store = TCPStore(
                    host_name=master_addr,
                    port=master_port,
                    world_size=self.world_size,
                    is_master=(self.rank == 0),
                    timeout=timedelta(seconds=30),
                )

            self.store = ReconfigureTest._shared_store
        else:
            self.wrapper = self.get_wrapper()
            self.torchcomm = self.wrapper.get_torchcomm()
            self.rank = self.torchcomm.get_rank()
            self.world_size = self.torchcomm.get_size()
            self.device = self.torchcomm.get_device()

    def tearDown(self):
        """Clean up after each test."""
        if not self._is_supported_backend():
            self.torchcomm = None
            self.wrapper = None

    def _is_supported_backend(self):
        """Check if current backend supports reconfigure()."""
        return self.backend in self.SUPPORTED_BACKENDS

    def _get_store_for_comm(self):
        """Get the store to pass to new_comm for NCCL bootstrap."""
        return getattr(self, "store", None)

    def _skip_if_nccl_too_old(self):
        """Skip test if NCCL < 2.29 (commGrow/commGetUniqueId unavailable)."""
        if self.backend not in ("nccl",):
            return
        if torch.cuda.is_available():
            major, minor, _ = torch.cuda.nccl.version()
            if (major, minor) < (2, 29):
                self.skipTest(f"NCCL {major}.{minor} < 2.29, commGrow not supported")

    def _collect_handles(self, comm, key_prefix):
        """Collect init handles from all ranks."""
        my_handle = comm.get_init_handle()
        key = f"{key_prefix}_{self.rank}"
        self.store.set(key, my_handle)
        handles = []
        for i in range(self.world_size):
            handle = self.store.get(f"{key_prefix}_{i}").decode("utf-8")
            handles.append(handle)
        return handles

    def test_reconfigure_unsupported_backend(self):
        """Test reconfigure() raises RuntimeError for unsupported backends."""
        if self._is_supported_backend():
            self.skipTest(
                f"Backend {self.backend} supports reconfigure(), skipping negative test"
            )

        with self.assertRaises(RuntimeError) as context:
            self.torchcomm.reconfigure(
                uuid=0,
                init_handles=["test_handle"],
                timeout=timedelta(milliseconds=5000),
            )

        self.assertIn("reconfigure not implemented", str(context.exception))
        print(f"[Rank {self.rank}] Expected RuntimeError raised: {context.exception}")

    def test_enable_reconfigure_unsupported_backend(self):
        """Test enable_reconfigure=True raises RuntimeError for unsupported backends."""
        if self._is_supported_backend():
            self.skipTest(
                f"Backend {self.backend} supports reconfigure, skipping negative test"
            )

        import torch.comms

        with self.assertRaises(RuntimeError) as context:
            torch.comms.new_comm(
                self.backend,
                self.device,
                "test_reconfigure_unsupported",
                enable_reconfigure=True,
            )

        print(
            f"[Rank {self.rank}] Expected RuntimeError for enable_reconfigure=True: "
            f"{context.exception}"
        )

    def test_reconfigure_basic(self):
        """Test basic reconfigure with ordered handles (list)."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")

        import torch.comms

        comm = torch.comms.new_comm(
            self.backend,
            self.device,
            "reconfigure_basic",
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
        )

        all_handles = self._collect_handles(comm, "test_reconfigure_basic")
        self.assertGreater(len(all_handles), 0)

        print(f"[Rank {self.rank}] Collected handles: {all_handles}")

        work = comm.reconfigure(
            uuid=0,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        )
        self.assertIsNotNone(work)

        work.wait()

        self.assertGreaterEqual(comm.get_rank(), 0)
        self.assertLess(comm.get_rank(), self.world_size)
        self.assertEqual(comm.get_size(), self.world_size)

        print(f"[Rank {self.rank}] Reconfigure completed successfully")

        comm.finalize()

    def test_reconfigure_then_collective(self):
        """Test that collective operations work after reconfigure."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")

        import torch.comms

        comm = torch.comms.new_comm(
            self.backend,
            self.device,
            "reconfigure_collective",
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
        )

        all_handles = self._collect_handles(comm, "test_reconfigure_collective")

        work = comm.reconfigure(
            uuid=2,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        )
        work.wait()

        if self.world_size > 1:
            my_rank = comm.get_rank()
            count = 4
            send_tensor = torch.ones(count, dtype=torch.float, device=self.device) * (
                my_rank + 1
            )
            recv_tensor = torch.zeros(count, dtype=torch.float, device=self.device)

            send_rank = (my_rank + 1) % self.world_size
            recv_rank = (my_rank - 1 + self.world_size) % self.world_size

            if my_rank % 2 == 0:
                send_work = comm.send(send_tensor, send_rank, async_op=True)
                recv_work = comm.recv(recv_tensor, recv_rank, async_op=True)
            else:
                recv_work = comm.recv(recv_tensor, recv_rank, async_op=True)
                send_work = comm.send(send_tensor, send_rank, async_op=True)

            send_work.wait()
            recv_work.wait()

            if self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()

            expected = torch.ones(count, dtype=torch.float, device="cpu") * (
                recv_rank + 1
            )
            self.assertTrue(
                torch.allclose(recv_tensor.cpu(), expected),
                f"[Rank {my_rank}] Send/recv after reconfigure failed",
            )

            print(f"[Rank {my_rank}] Send/recv after reconfigure succeeded")

        comm.finalize()

    def test_reconfigure_then_allreduce(self):
        """Test that allreduce works after reconfigure."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")

        import torch.comms

        comm = torch.comms.new_comm(
            self.backend,
            self.device,
            "reconfigure_allreduce",
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
        )

        all_handles = self._collect_handles(comm, "test_reconfigure_allreduce")

        work = comm.reconfigure(
            uuid=3,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        )
        work.wait()

        my_rank = comm.get_rank()
        tensor = torch.ones(4, dtype=torch.float, device=self.device) * (my_rank + 1)
        comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=False)

        if self.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

        expected_value = sum(range(1, self.world_size + 1))
        expected = torch.ones(4, dtype=torch.float, device="cpu") * expected_value
        self.assertTrue(
            torch.allclose(tensor.cpu(), expected),
            f"[Rank {my_rank}] AllReduce after reconfigure failed: "
            f"got {tensor.cpu()}, expected {expected}",
        )

        print(f"[Rank {my_rank}] AllReduce after reconfigure succeeded")

        comm.finalize()

    def test_reconfigure_scale_down_up(self):
        """Test that reconfigure works when scaling down and up."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")
        self._skip_if_nccl_too_old()

        import torch.comms

        comm = torch.comms.new_comm(
            self.backend,
            self.device,
            "reconfigure_scale_down_up",
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
        )

        comm.reconfigure(
            uuid=3,
            init_handles=[comm.get_init_handle()],
            timeout=timedelta(milliseconds=30000),
        ).wait()

        all_handles = self._collect_handles(comm, "test_reconfigure_scale_down_up")
        comm.reconfigure(
            uuid=4,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        ).wait()

        comm.reconfigure(
            uuid=5,
            init_handles=[comm.get_init_handle()],
            timeout=timedelta(milliseconds=30000),
        ).wait()

        comm.finalize()

    def test_reconfigure_single_to_all(self):
        """Test reconfigure from single rank to all ranks then allreduce."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")

        import torch.comms

        comm = torch.comms.new_comm(
            self.backend,
            self.device,
            "reconfigure_single_to_all",
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
        )

        comm.reconfigure(
            uuid=3,
            init_handles=[comm.get_init_handle()],
            timeout=timedelta(milliseconds=30000),
        ).wait()

        all_handles = self._collect_handles(comm, "test_reconfigure_single_to_all")
        comm.reconfigure(
            uuid=4,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        ).wait()

        my_rank = comm.get_rank()
        tensor = torch.ones(4, dtype=torch.float, device=self.device) * (my_rank + 1)
        comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=False)

        if self.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

        expected_value = sum(range(1, self.world_size + 1))
        expected = torch.ones(4, dtype=torch.float, device="cpu") * expected_value
        self.assertTrue(
            torch.allclose(tensor.cpu(), expected),
            f"[Rank {my_rank}] AllReduce after single-to-all reconfigure failed: "
            f"got {tensor.cpu()}, expected {expected}",
        )

        comm.finalize()

    def test_reconfigure_identity(self):
        """Test that reconfigure works when world size doesn't change."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")

        import torch.comms

        comm = torch.comms.new_comm(
            self.backend,
            self.device,
            "reconfigure_identity",
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
        )

        all_handles = self._collect_handles(comm, "test_reconfigure_identity1")
        comm.reconfigure(
            uuid=4,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        ).wait()

        all_handles = self._collect_handles(comm, "test_reconfigure_identity2")
        comm.reconfigure(
            uuid=5,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        ).wait()

        comm.finalize()

    def test_reconfigure_late(self):
        """Test that reconfigure works when workers join late."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")
        self._skip_if_nccl_too_old()

        import torch.comms

        comm = torch.comms.new_comm(
            self.backend,
            self.device,
            "reconfigure_late",
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
        )

        initial_world_size = self.world_size // 2

        all_handles = self._collect_handles(comm, "test_reconfigure_late1")
        if self.rank < initial_world_size:
            comm.reconfigure(
                uuid=4,
                init_handles=all_handles[:initial_world_size],
                timeout=timedelta(milliseconds=30000),
            ).wait()

        all_handles = self._collect_handles(comm, "test_reconfigure_late2")
        comm.reconfigure(
            uuid=5,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        ).wait()

        comm.finalize()

    def test_reconfigure_merge_split(self):
        """Test that reconfigure works when there's a split brain condition."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")
        self._skip_if_nccl_too_old()

        import torch.comms

        comm = torch.comms.new_comm(
            self.backend,
            self.device,
            "reconfigure_merge_split",
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
        )

        initial_world_size = self.world_size // 2

        all_handles = self._collect_handles(comm, "test_reconfigure_merge_split1")
        if self.rank < initial_world_size:
            initial_handles = all_handles[:initial_world_size]
            initial_uuid = 4
        else:
            initial_handles = all_handles[initial_world_size:]
            initial_uuid = 5
        comm.reconfigure(
            uuid=initial_uuid,
            init_handles=initial_handles,
            timeout=timedelta(milliseconds=30000),
        ).wait()

        all_handles = self._collect_handles(comm, "test_reconfigure_merge_split2")
        comm.reconfigure(
            uuid=6,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        ).wait()

        comm.finalize()

    @skip_backend("gloo")
    def test_reconfigure_after_abort(self):
        """After abort(), reconfigure() should recover the communicator."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")

        import torch.comms

        comm = torch.comms.new_comm(
            self.backend,
            self.device,
            "reconfigure_after_abort",
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
        )

        all_handles = self._collect_handles(comm, "test_reconfigure_after_abort1")
        comm.reconfigure(
            uuid=10,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        ).wait()

        comm.abort()

        all_handles = self._collect_handles(comm, "test_reconfigure_after_abort2")
        comm.reconfigure(
            uuid=11,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        ).wait()

        tensor = torch.ones(4, dtype=torch.float, device=self.device) * (
            comm.get_rank() + 1
        )
        comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=False)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        expected_value = sum(range(1, self.world_size + 1))
        expected = torch.ones(4, dtype=torch.float, device="cpu") * expected_value
        self.assertTrue(
            torch.allclose(tensor.cpu(), expected),
            f"[Rank {comm.get_rank()}] AllReduce after abort+reconfigure failed: "
            f"got {tensor.cpu()}, expected {expected}",
        )

        comm.finalize()


if __name__ == "__main__":
    unittest.main()
