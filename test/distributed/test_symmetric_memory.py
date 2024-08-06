# Owner(s): ["module: c10d"]

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import (
    _fused_all_gather_matmul_fallback,
    _fused_all_gather_scaled_matmul_fallback,
    _fused_matmul_reduce_scatter_fallback,
    _fused_scaled_matmul_reduce_scatter_fallback,
    enable_symm_mem_for_group,
    restride_A_for_fused_matmul_reduce_scatter,
    restride_A_shard_for_fused_all_gather_matmul,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
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
        enable_symm_mem_for_group(dist.group.WORLD.group_name)

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
    def test_cuda_nvlink_connectivity_detection(self) -> None:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _detect_dma_connectivity

        connectivity = _detect_dma_connectivity(DeviceType.CUDA, "nvlink")
        self.assertEqual(connectivity.device_type, DeviceType.CUDA)
        self.assertEqual(connectivity.connection_type, "nvlink")
        self.assertEqual(len(connectivity.matrix), torch.cuda.device_count())
        for row in connectivity.matrix:
            self.assertEqual(len(row), torch.cuda.device_count())

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
        self.assertIsNone(_SymmetricMemory.rendezvous(t))

        t = _SymmetricMemory.empty_strided_p2p(*alloc_args)
        symm_mem = _SymmetricMemory.rendezvous(t)

        del t
        self._verify_symmetric_memory(symm_mem)
        dist.destroy_process_group()

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
        dist.destroy_process_group()

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    @parametrize("gather_dim", [0, 1])
    def test_fused_all_gather_matmul(self, gather_dim: int) -> None:
        self._init_process()

        BATCH = 8
        M = 64
        N = 16
        K = 32
        group = dist.group.WORLD
        rank = self.rank
        world_size = self.world_size

        torch.manual_seed(42 + rank)
        A_shard = torch.rand(BATCH, M // self.world_size, K, device="cuda")
        Bs = [torch.rand(K, N, device="cuda") for _ in range(3)]

        ag_output_0, mm_outputs_0 = _fused_all_gather_matmul_fallback(
            A_shard, Bs, gather_dim=gather_dim, group_name=group.group_name
        )
        ag_output_1, mm_outputs_1 = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, Bs, gather_dim=gather_dim, group_name=group.group_name
        )

        assert torch.allclose(ag_output_0, ag_output_1)
        assert ag_output_0.stride() == ag_output_1.stride()
        for mm_output_0, mm_output_1 in zip(mm_outputs_0, mm_outputs_1):
            assert torch.allclose(mm_output_0, mm_output_1)
            assert mm_output_0.stride(), mm_output_1.stride()

        dist.destroy_process_group()

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    @parametrize("gather_dim", [0, 1])
    def test_fused_all_gather_scaled_matmul(self, gather_dim: int) -> None:
        self._init_process()

        BATCH = 8
        M = 64
        N = 16
        K = 32
        group = dist.group.WORLD
        rank = self.rank
        world_size = self.world_size

        torch.manual_seed(42 + rank)
        A_shard = torch.rand(BATCH, M // self.world_size, K, device="cuda").to(
            torch.float8_e4m3fn
        )
        A_scale = torch.tensor(0.1, device="cuda")
        Bs = [
            torch.rand(N, K, device="cuda").to(torch.float8_e4m3fn).T for _ in range(3)
        ]
        B_scales = [torch.tensor(0.1, device="cuda") for _ in range(3)]
        out_dtypes = [None, torch.bfloat16, torch.float32]

        ag_output_0, mm_outputs_0 = _fused_all_gather_scaled_matmul_fallback(
            A_shard,
            Bs,
            A_scale,
            B_scales,
            gather_dim=gather_dim,
            group_name=group.group_name,
            biases=[None] * len(Bs),
            result_scales=[None] * len(Bs),
            out_dtypes=out_dtypes,
            use_fast_accum=[None] * len(Bs),
        )
        ag_output_1, mm_outputs_1 = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
            A_shard,
            Bs,
            A_scale,
            B_scales,
            gather_dim=gather_dim,
            group_name=group.group_name,
            biases=[None] * len(Bs),
            result_scales=[None] * len(Bs),
            out_dtypes=out_dtypes,
            use_fast_accum=[None] * len(Bs),
        )

        self.assertTrue(
            torch.allclose(
                ag_output_0.to(torch.float32),
                ag_output_1.to(torch.float32),
            )
        )
        self.assertEqual(ag_output_0.stride(), ag_output_1.stride())
        for mm_output_0, mm_output_1 in zip(mm_outputs_0, mm_outputs_1):
            self.assertTrue(
                torch.allclose(
                    mm_output_0.to(torch.float32), mm_output_1.to(torch.float32)
                )
            )
            self.assertEqual(mm_output_0.stride(), mm_output_1.stride())
            self.assertEqual(mm_output_0.dtype, mm_output_1.dtype)

        dist.destroy_process_group()

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    @parametrize("scatter_dim", [0, 1])
    def test_fused_matmul_reduce_scatter(self, scatter_dim: int) -> None:
        self._init_process()

        BATCH = 8
        M = 64
        N = 16
        K = 32
        group = dist.group.WORLD
        rank = self.rank
        world_size = self.world_size

        torch.manual_seed(42 + rank)
        A = torch.rand(BATCH, M, K, device="cuda")
        B = torch.rand(K, N, device="cuda")

        output_0 = _fused_matmul_reduce_scatter_fallback(
            A, B, "avg", scatter_dim=scatter_dim, group_name=group.group_name
        )
        output_1 = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A, B, "avg", scatter_dim=scatter_dim, group_name=group.group_name
        )

        assert torch.allclose(output_0, output_1)
        assert output_0.stride() == output_1.stride()

        dist.destroy_process_group()

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    @parametrize("scatter_dim", [0, 1])
    def test_fused_scaled_matmul_reduce_scatter(self, scatter_dim: int) -> None:
        self._init_process()

        BATCH = 8
        M = 64
        N = 16
        K = 32
        group = dist.group.WORLD
        rank = self.rank
        world_size = self.world_size

        torch.manual_seed(42 + rank)
        A = torch.rand(BATCH, M, K, device="cuda").to(torch.float8_e4m3fn)
        A_scale = torch.tensor(0.1, device="cuda")
        B = torch.rand(N, K, device="cuda").to(torch.float8_e4m3fn).T
        B_scale = torch.tensor(0.1, device="cuda")

        output_0 = _fused_scaled_matmul_reduce_scatter_fallback(
            A,
            B,
            A_scale,
            B_scale,
            "avg",
            scatter_dim,
            group.group_name,
            out_dtype=torch.bfloat16,
        )
        output_1 = torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
            A,
            B,
            A_scale,
            B_scale,
            "avg",
            scatter_dim,
            group.group_name,
            out_dtype=torch.bfloat16,
        )

        assert torch.allclose(output_0, output_1)
        assert output_0.stride() == output_1.stride()

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

    @skip_if_lt_x_gpu(2)
    @parametrize("symm_mem_input", [True, False])
    def test_low_contention_all_gather(self, symm_mem_input: bool) -> None:
        self._init_process()

        if symm_mem_input:
            t = _SymmetricMemory.empty_strided_p2p(
                size=(64, 64),
                stride=(64, 1),
                dtype=torch.float32,
                device=self.device,
                group_name="0",
            ).fill_(self.rank)
        else:
            t = torch.full((64, 64), self.rank, dtype=torch.float32, device=self.device)

        res = torch.ops.symm_mem._low_contention_all_gather(t, "0")
        res = torch.ops._c10d_functional.wait_tensor(res)
        self.assertEqual(res.shape, (64 * self.world_size, 64))

        chunks = res.chunk(self.world_size)
        for r in range(self.world_size):
            self.assertTrue(chunks[r].eq(r).all())

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    @parametrize("reduce_op", ["sum", "avg"])
    @parametrize("symm_mem_input", [True, False])
    def test_low_contention_reduce_scatter(
        self, reduce_op: str, symm_mem_input: bool
    ) -> None:
        self._init_process()

        if symm_mem_input:
            t = _SymmetricMemory.empty_strided_p2p(
                size=(64, 64),
                stride=(64, 1),
                dtype=torch.float32,
                device=self.device,
                group_name="0",
            )
        else:
            t = torch.empty((64, 64), dtype=torch.float32, device=self.device)

        chunks = t.chunk(self.world_size)
        for r in range(self.world_size):
            chunks[r].fill_(r)

        res = torch.ops.symm_mem._low_contention_reduce_scatter(t, reduce_op, "0")
        res = torch.ops._c10d_functional.wait_tensor(res)
        self.assertEqual(res.shape, (64 // self.world_size, 64))

        if reduce_op == "sum":
            expect = self.rank * self.world_size
        elif reduce_op == "avg":
            expect = self.rank
        else:
            raise AssertionError(f"Unexpected reduce_op: {reduce_op}")
        self.assertTrue(res.eq(expect).all())

        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
