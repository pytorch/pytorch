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

    @skipIfRocm
    def test_nvshmem_all_to_all_vdev(self) -> None:
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        # Number of elements for a peer is random between [0, k)
        k = 10
        inp_splits = torch.randint(k, (self.world_size,), device=self.device)
        inp_numel = inp_splits.sum().item()
        # Exchange input splits to get output splits
        out_splits = torch.zeros_like(inp_splits)
        dist.all_to_all_single(out_splits, inp_splits)
        out_numel = out_splits.sum().item()
        # Align up to make it bigger
        align = 16
        out_numel_max = (out_numel + align - 1) // align * align

        inp = symm_mem.empty(inp_numel, dtype=dtype, device=self.device).fill_(
            self.rank
        )
        out = symm_mem.empty(out_numel_max, dtype=dtype, device=self.device).fill_(-1)
        in_out_splits = symm_mem.empty(
            (3, self.world_size), dtype=torch.int64, device=self.device
        )
        # Row 0 is input splits
        in_out_splits[0].copy_(inp_splits)

        torch.ops.symm_mem.nvshmem_all_to_all_vdev(inp, out, in_out_splits, group_name)

        # Check input splits (row 0) -- should not change
        torch.testing.assert_close(in_out_splits[0], inp_splits)

        # Check output splits (row 1)
        torch.testing.assert_close(in_out_splits[1], out_splits)

        # Check output offsets (row 2)
        out_offsets = torch.cumsum(out_splits, dim=0)  # inclusive scan
        # output offsets from `nvshmem_all_to_all_vdev` is exclusive scan
        self.assertEqual(in_out_splits[2][0], 0)
        torch.testing.assert_close(in_out_splits[2][1:], out_offsets[:-1])

        # Check data
        expected = torch.empty(out_numel, dtype=dtype, device=self.device)
        dist.all_to_all_single(expected, inp, out_splits.tolist(), inp_splits.tolist())
        torch.testing.assert_close(out[:out_numel], expected)


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
