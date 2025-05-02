# Owner(s): ["oncall: distributed"]

# To run:
# TORCH_SYMMMEM=NVSHMEM python test/distributed/test_nvshmem.py
# OR
# TORCH_SYMMMEM=NVSHMEM torchrun --nproc-per-node 4 test/distributed/test_nvshmem.py

import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import (
    MultiProcContinousTest,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import (
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
)


symm_mem_backend = os.getenv("TORCH_SYMMMEM")

if symm_mem_backend != "NVSHMEM":
    print(
        "test_nvshmem requires setting `TORCH_SYMMMEM=NVSHMEM`, skipping tests",
        file=sys.stderr,
    )
    sys.exit(0)


# Decorator
def requires_nvshmem():
    return skip_but_pass_in_sandcastle_if(
        symm_mem_backend != "NVSHMEM",
        "test_nvshmem requires setting `TORCH_SYMMMEM=NVSHMEM`",
    )


# So that tests are written in device-agnostic way
device_type = "cuda"
device_module = torch.get_device_module(device_type)


@requires_nvshmem()
class NVSHMEMSymmetricMemoryTest(MultiProcContinousTest):
    def setUp(self) -> None:
        super().setUp()
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # NOTE: required for nvshmem allocation
        torch.empty(1, device=self.device)

    # Required by MultiProcContinousTest
    @classmethod
    def backend_str(cls) -> str:
        return "nccl"

    @property
    def world_size(self) -> int:
        return device_module.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    @skipIfRocm
    def test_nvshmem_all_to_all(self) -> None:
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel_per_peer = 10
        numel = self.world_size * numel_per_peer
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)

        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)
        torch.ops.symm_mem.nvshmem_all_to_all(inp, out, group_name)

        expected = torch.cat(
            [
                torch.empty(numel_per_peer, dtype=dtype, device=self.device).fill_(i)
                for i in range(self.world_size)
            ]
        )
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    if not device_module.is_available():
        sys.exit(TEST_SKIPS["no_cuda"].exit_code)

    # If launched by torchrun, these values would have been set
    rank = int(os.getenv("RANK", "-1"))
    world_size = int(os.getenv("WORLD_SIZE", "-1"))

    if rank != -1:
        # Launched with torchrun or other multi-proc launchers. Directly run the test.
        NVSHMEMSymmetricMemoryTest.run_rank(rank, world_size)
    else:
        # No external launcher, spawn N processes
        world_size = device_module.device_count()
        # Launched as a single process. Spawn subprocess to run the tests.
        # Also need a rendezvous file for `init_process_group` purpose.
        rdvz_file = tempfile.NamedTemporaryFile(delete=False).name
        torch.multiprocessing.spawn(
            NVSHMEMSymmetricMemoryTest.run_rank,
            nprocs=world_size,
            args=(world_size, rdvz_file),
        )
