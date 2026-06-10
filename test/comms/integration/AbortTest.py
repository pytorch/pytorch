#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest
from datetime import timedelta

import torch
import torch.comms
from torch.distributed import TCPStore


class AbortTest(unittest.TestCase):
    """Test the abort() API for TorchComm."""

    SUPPORTED_BACKENDS = {"nccl"}

    _shared_store = None

    def setUp(self):
        self.backend = os.getenv("TEST_BACKEND", "")

        if self.backend not in self.SUPPORTED_BACKENDS:
            self.skipTest(f"Backend {self.backend} does not support abort()")

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        self.rank = int(
            os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        )
        self.world_size = int(
            os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
        )
        device_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{device_id}")

        if AbortTest._shared_store is None:
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = int(os.environ.get("MASTER_PORT", "29500"))
            AbortTest._shared_store = TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=self.world_size,
                is_master=(self.rank == 0),
                timeout=timedelta(seconds=30),
            )
        self.store = AbortTest._shared_store

    def _create_reconfigurable_comm(self, name, uuid):
        comm = torch.comms.new_comm(
            self.backend,
            self.device,
            name,
            enable_reconfigure=True,
            store=self.store,
        )
        my_handle = comm.get_init_handle()
        self.store.set(f"{name}_{self.rank}", my_handle)
        handles = []
        for i in range(self.world_size):
            h = self.store.get(f"{name}_{i}").decode("utf-8")
            handles.append(h)
        work = comm.reconfigure(
            uuid=uuid,
            init_handles=handles,
            timeout=timedelta(seconds=30),
        )
        work.wait()
        return comm

    def test_abort(self):
        """After abort(), subsequent operations should raise RuntimeError."""
        comm = self._create_reconfigurable_comm("abort_error_state", 0)

        comm.abort()
        comm.finalize()


if __name__ == "__main__":
    unittest.main()
