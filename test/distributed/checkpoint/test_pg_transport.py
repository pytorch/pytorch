# Owner(s): ["oncall: distributed"]

import sys
from datetime import timedelta
import unittest
from unittest import TestCase, skipIf, skipUnless
import logging
from typing import Optional

import torch
from torch.distributed import TCPStore
import torch.distributed as dist

from torch.testing._internal.common_utils import run_tests
from torch.distributed.checkpoint._pg_transport import PGTransport

from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcContinousTest,
    requires_nccl,
)
from torch.testing._internal.common_utils import (
    check_leaked_tensors,
    instantiate_parametrized_tests,
    parametrize,
    skip_but_pass_in_sandcastle_if,
)

import torch.nn as nn
from torch.distributed.distributed_c10d import _get_default_group
from datetime import timedelta
import os
from torch.testing._internal.common_device_type import instantiate_device_type_tests



logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def ring_send_recv_checkpoint(transport: PGTransport, state_dict, rank, world_size, step=0):
    """
    Use the transport to send to rank + 1 and receive from rank - 1.
    """
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    if rank == 0:
        transport.send_checkpoint([next_rank], state_dict, 0)
        received_checkpoint = transport.recv_checkpoint(prev_rank, step)
    else:
        received_checkpoint = transport.recv_checkpoint(prev_rank, step)
        transport.send_checkpoint([next_rank], received_checkpoint, step)
    return received_checkpoint

def _test_pg_transport(self, device) -> None:
    # python test/distributed/checkpoint/test_pg_transport.py -k test_pg_transport
    print(f"{self.rank=} pid: {os.getpid()} {device=}")
    print("in test")

    model = SimpleModel().to(device)
    transport = PGTransport(_get_default_group(), timedelta(seconds=10), device)
    original_state_dict = model.state_dict()
    received_checkpoint = ring_send_recv_checkpoint(
        transport=transport,
        state_dict=original_state_dict,
        rank=self.rank,
        world_size=self.world_size
    )
    self.assertEqual(original_state_dict, received_checkpoint)

def _step_mismatch(self, device) -> None:
    transport = PGTransport(_get_default_group(), timedelta(seconds=10), device)
    model = SimpleModel().to(device)
    original_state_dict = model.state_dict()
    with self.assertRaises(Exception):
        if self.rank == 0:
            ring_send_recv_checkpoint(
                transport=transport,
                state_dict=original_state_dict,
                rank=self.rank,
                world_size=self.world_size,
                step=0
            )
        else:
            received_checkpoint = ring_send_recv_checkpoint(
                transport=transport,
                state_dict=original_state_dict,
                rank=self.rank,
                world_size=self.world_size,
                step=1
            )

class PgTransportCPU(MultiProcContinousTest):
    world_size = 8

    @classmethod
    def backend_str(cls) -> Optional[str]:
        return "gloo"

    @classmethod
    def device_type(cls) -> str:
        return "cpu"

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_type())

    def test_pg_transport(self) -> None:
        _test_pg_transport(self, self.device)

    def test_step_mismatch(self) -> None:
        _step_mismatch(self, self.device)

@skipIf(not torch.cuda.is_available(), "CUDA not available")
class PgTransportCUDA(MultiProcContinousTest):
    world_size = 2

    @classmethod
    def backend_str(cls) -> Optional[str]:
        return "nccl"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device(f"{self.device_type()}:{self.rank}")

    def test_pg_transport(self) -> None:
        _test_pg_transport(self, self.device)

    def test_step_mismatch(self) -> None:
        _step_mismatch(self, self.device)

import fbvscode
# fbvscode.attach_debugger()
if __name__ == "__main__":
    run_tests()
