# Owner(s): ["module: c10d"]
import os
from typing import List

import torch

import torch.distributed as dist
from torch.distributed._cuda_p2p import (
    _fused_all_gather_matmul_fallback,
    _fused_matmul_reduce_scatter_fallback,
    get_cuda_p2p_backend,
    get_p2p_buffer_size,
    is_cuda_p2p_group,
    p2p_usage_counter,
    restride_A_for_fused_matmul_reduce_scatter,
    restride_A_shard_for_fused_all_gather_matmul,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
)


def requires_cuda_p2p_access():
    cuda_p2p_access_available = (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
        and dist.is_nccl_available()
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

    def _init_process_group(self, buffer_size: int) -> None:
        os.environ["TEST_INTRA_NODE_COMM"] = "1"
        torch.cuda.set_device(self.device)

        # Verify cuda p2p specific APIs on ProcessGroupCudaP2P
        store = dist.FileStore(self.file_name, self.world_size)
        options = dist.ProcessGroupCudaP2P.Options()
        options.buffer_size = buffer_size
        dist.init_process_group(
            backend="cuda_p2p",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=options,
        )

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_p2p_apis(self) -> None:
        BUFFER_SIZE = 4 * 1024

        self._init_process_group(BUFFER_SIZE)

        # Verify cuda p2p specific APIs on ProcessGroupCudaP2P
        assert is_cuda_p2p_group(dist.group.WORLD)
        assert get_p2p_buffer_size(dist.group.WORLD) == BUFFER_SIZE

        backend = get_cuda_p2p_backend(dist.group.WORLD)
        assert isinstance(backend, dist.ProcessGroupCudaP2P)
        assert backend.get_buffer_size() == BUFFER_SIZE

        backend.get_p2p_buffer(self.rank, (BUFFER_SIZE // 4,), torch.float)
        with self.assertRaises(RuntimeError):
            backend.get_p2p_buffer(self.rank, (BUFFER_SIZE // 4 + 1,), torch.float)
        with self.assertRaises(RuntimeError):
            backend.get_p2p_buffer(self.rank, (BUFFER_SIZE // 4,), torch.float, 1)

        # Verify cuda p2p specific APIs on non-cuda p2p process group
        non_cuda_p2p_pg = dist.new_group(backend="nccl")

        assert not is_cuda_p2p_group(non_cuda_p2p_pg)
        assert get_p2p_buffer_size(non_cuda_p2p_pg) == 0
        with self.assertRaises(TypeError):
            get_cuda_p2p_backend(non_cuda_p2p_pg)

        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_p2p_buffer(self) -> None:
        BUFFER_SIZE = 4 * 1024

        self._init_process_group(BUFFER_SIZE)
        rank = self.rank
        world_size = self.world_size

        assert is_cuda_p2p_group(dist.group.WORLD)
        backend = get_cuda_p2p_backend(dist.group.WORLD)
        local_buffer = backend.get_p2p_buffer(
            (rank) % world_size, (BUFFER_SIZE // 4,), torch.float
        )
        remote_buffer = backend.get_p2p_buffer(
            (rank + 1) % world_size, (BUFFER_SIZE // 4,), torch.float
        )

        local_buffer.fill_(rank)
        backend.intra_node_barrier()
        assert remote_buffer.eq((rank + 1) % world_size).all()

        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    @parametrize("gather_dim", [0, 1])
    def test_fused_all_gather_matmul(self, gather_dim: int) -> None:
        B = 8
        M = 64
        N = 16
        K = 32
        BUFFER_SIZE = B * M * K // self.world_size * 4

        self._init_process_group(BUFFER_SIZE)
        group = dist.group.WORLD
        rank = self.rank
        world_size = self.world_size

        torch.manual_seed(42 + rank)
        A_shard = torch.rand(B, M // self.world_size, K, device="cuda")
        Bs = [torch.rand(K, N, device="cuda") for _ in range(3)]

        ag_output_0, mm_outputs_0 = _fused_all_gather_matmul_fallback(
            A_shard, Bs, gather_dim=gather_dim, group_name=group.group_name
        )
        with p2p_usage_counter() as counter:
            ag_output_1, mm_outputs_1 = torch.ops.cuda_p2p.fused_all_gather_matmul(
                A_shard, Bs, gather_dim=gather_dim, group_name=group.group_name
            )
            assert counter["fused_all_gather_matmul"] == 1

        assert torch.allclose(ag_output_0, ag_output_1)
        assert ag_output_0.stride() == ag_output_1.stride()
        for mm_output_0, mm_output_1 in zip(mm_outputs_0, mm_outputs_1):
            assert torch.allclose(mm_output_0, mm_output_1)
            assert mm_output_0.stride(), mm_output_1.stride()

        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    @parametrize("scatter_dim", [0, 1])
    def test_fused_matmul_reduce_scatter(self, scatter_dim: int) -> None:
        B = 8
        M = 64
        N = 16
        K = 32
        BUFFER_SIZE = B * M * N // self.world_size * 4 * 2

        self._init_process_group(BUFFER_SIZE)
        group = dist.group.WORLD
        rank = self.rank
        world_size = self.world_size

        torch.manual_seed(42 + rank)
        A = torch.rand(B, M, K, device="cuda")
        B = torch.rand(K, N, device="cuda")

        output_0 = _fused_matmul_reduce_scatter_fallback(
            A, B, "avg", scatter_dim=scatter_dim, group_name=group.group_name
        )
        with p2p_usage_counter() as counter:
            output_1 = torch.ops.cuda_p2p.fused_matmul_reduce_scatter(
                A, B, "avg", scatter_dim=scatter_dim, group_name=group.group_name
            )
            assert counter["fused_matmul_reduce_scatter"] == 1

        assert torch.allclose(output_0, output_1)
        assert output_0.stride() == output_1.stride()

        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()

    @parametrize("dim", [0, 1, 2])
    def test_optimal_layout(self, dim: int) -> None:
        t = torch.rand(8, 64, 32, 16)

        x = restride_A_shard_for_fused_all_gather_matmul(t, dim)
        self.assertTrue(x.movedim(dim, 0).is_contiguous())
        self.assertTrue(torch.allclose(x, t))

        x = restride_A_for_fused_matmul_reduce_scatter(t, dim)
        self.assertTrue(x.movedim(dim, 0).is_contiguous())
        self.assertTrue(torch.allclose(x, t))


if __name__ == "__main__":
    run_tests()
