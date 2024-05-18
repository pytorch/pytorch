# Owner(s): ["module: c10d"]
import os
from typing import List

import torch

import torch.distributed as dist
from torch.distributed._cuda_p2p import (
    get_cuda_p2p_backend,
    is_cuda_p2p_group,
    ProcessGroupCudaP2P,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
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


@requires_nccl()
@requires_cuda_p2p_access()
class ProcessGroupCudaP2PTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def ranks(self) -> List[int]:
        return list(range(self.world_size))

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    @property
    def buffer_size(self) -> int:
        return 4 * 1024

    def _init_process_group(self) -> None:
        os.environ["TEST_INTRA_NODE_COMM"] = "1"
        torch.cuda.set_device(self.device)

        # Verify cuda p2p specific APIs on ProcessGroupCudaP2P
        store = dist.FileStore(self.file_name, self.world_size)
        options = ProcessGroupCudaP2P.Options()
        options.buffer_size = self.buffer_size
        dist.init_process_group(
            backend="cuda_p2p",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=options,
        )

    @skip_if_lt_x_gpu(2)
    def test_p2p_apis(self) -> None:
        self._init_process_group()

        # Verify cuda p2p specific APIs on ProcessGroupCudaP2P
        assert is_cuda_p2p_group(dist.group.WORLD)
        backend = get_cuda_p2p_backend(dist.group.WORLD)
        assert isinstance(backend, ProcessGroupCudaP2P)
        assert backend.get_buffer_size() == self.buffer_size

        backend.get_p2p_buffer(self.rank, (self.buffer_size // 4,), torch.float)
        with self.assertRaises(RuntimeError):
            backend.get_p2p_buffer(self.rank, (self.buffer_size // 4 + 1,), torch.float)
        with self.assertRaises(RuntimeError):
            backend.get_p2p_buffer(self.rank, (self.buffer_size // 4,), torch.float, 1)

        # Verify cuda p2p specific APIs on non-cuda p2p process group
        non_cuda_p2p_pg = dist.new_group(backend="nccl")

        assert not is_cuda_p2p_group(non_cuda_p2p_pg)
        with self.assertRaises(TypeError):
            get_cuda_p2p_backend(non_cuda_p2p_pg)

        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_p2p_buffer(self) -> None:
        self._init_process_group()
        rank = self.rank
        world_size = self.world_size

        assert is_cuda_p2p_group(dist.group.WORLD)
        backend = get_cuda_p2p_backend(dist.group.WORLD)
        local_buffer = backend.get_p2p_buffer(
            (rank) % world_size, (self.buffer_size // 4,), torch.float
        )
        remote_buffer = backend.get_p2p_buffer(
            (rank + 1) % world_size, (self.buffer_size // 4,), torch.float
        )

        local_buffer.fill_(rank)
        backend.intra_node_barrier()
        assert remote_buffer.eq((rank + 1) % world_size).all()

        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
