# Owner(s): ["module: c10d"]

import torch

import torch.distributed as dist
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed.distributed_c10d import _get_process_group_store

from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
)


def requires_cuda_p2p_access():
    cuda_p2p_access_available = (
        torch.cuda.is_available() and torch.cuda.device_count() >= 2
    )
    num_devices = torch.cuda.device_count()
    for i in range(num_devices - 1):
        for j in range(i + 1, num_devices):
            if not torch.cuda.can_device_access_peer(i, j):
                cuda_p2p_access_available = False
                break
        if not cuda_p2p_access_available:
            break

    return skip_but_pass_in_sandcastle_if(
        not cuda_p2p_access_available,
        "cuda p2p access is not available",
    )


@instantiate_parametrized_tests
@requires_cuda_p2p_access()
class SymmetricMemoryTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        _SymmetricMemory.set_group_info(
            "0",
            self.rank,
            self.world_size,
            _get_process_group_store(dist.GroupMember.WORLD),
        )

    def _verify_symmetric_memory(self, symm_mem):
        self.assertEqual(symm_mem.world_size, 2)

        buf = symm_mem.get_buffer(0, (64, 64), torch.float32)
        if symm_mem.rank == 0:
            symm_mem.wait_signal(src_rank=1)
            self.assertTrue(buf.eq(42).all())
        else:
            buf.fill_(42)
            symm_mem.put_signal(dst_rank=0)

        symm_mem.barrier()

        if symm_mem.rank == 0:
            symm_mem.barrier()
            self.assertTrue(buf.eq(43).all())
        else:
            buf.fill_(43)
            symm_mem.barrier()

        symm_mem.barrier()

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_empty_strided_p2p(self) -> None:
        self._init_process()

        shape = (64, 64)
        stride = (64, 1)
        dtype = torch.float32
        device = self.device
        group_name = "0"
        alloc_args = (shape, stride, dtype, device, group_name)

        t = torch.empty(shape, dtype=dtype, device=device)
        with self.assertRaises(RuntimeError):
            _SymmetricMemory.rendezvous(t)

        t = _SymmetricMemory.empty_strided_p2p(*alloc_args)
        symm_mem = _SymmetricMemory.rendezvous(t)

        del t
        self._verify_symmetric_memory(symm_mem)

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_empty_strided_p2p_persistent(self) -> None:
        self._init_process()

        shape = (64, 64)
        stride = (64, 1)
        dtype = torch.float32
        device = self.device
        alloc_id = 42  # Persistent allocation
        group_name = "0"
        alloc_args = (shape, stride, dtype, device, group_name, alloc_id)

        t = _SymmetricMemory.empty_strided_p2p(*alloc_args)
        data_ptr = t.data_ptr()

        # Verify that persistent allocation would fail if there's an active
        # allocation with the same alloc_id.
        with self.assertRaises(RuntimeError):
            _SymmetricMemory.empty_strided_p2p(*alloc_args)

        # Verify that persistent allocation would succeed in lieu of activate
        # allocations with the same alloc_id, and the returned tensor would
        # have the same data pointer.
        del t
        t = _SymmetricMemory.empty_strided_p2p(*alloc_args)
        self.assertEqual(t.data_ptr(), data_ptr)

        # Verify that get_symmetric_memory would fail if called before
        # rendezvous.
        with self.assertRaises(RuntimeError):
            _SymmetricMemory.get_symmetric_memory(t)

        symm_mem_0 = _SymmetricMemory.rendezvous(t)
        symm_mem_1 = _SymmetricMemory.get_symmetric_memory(t)
        self.assertEqual(id(symm_mem_0), id(symm_mem_1))

        self._verify_symmetric_memory(symm_mem_0)


if __name__ == "__main__":
    run_tests()
