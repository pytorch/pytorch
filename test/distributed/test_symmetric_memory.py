# Owner(s): ["module: c10d"]

import os
from unittest import skipIf

import torch
import torch.distributed as dist
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory
from torch._inductor.utils import fresh_inductor_cache, run_and_get_triton_code
from torch.distributed._functional_collectives import all_gather_tensor
from torch.distributed._symmetric_memory import (
    _fused_all_gather_matmul_fallback,
    _fused_all_gather_matmul_native,
    _fused_all_gather_scaled_matmul_fallback,
    _fused_matmul_reduce_scatter_fallback,
    _fused_scaled_matmul_reduce_scatter_fallback,
    enable_symm_mem_for_group,
    restride_A_for_fused_matmul_reduce_scatter,
    restride_A_shard_for_fused_all_gather_matmul,
)
from torch.testing._internal.common_cuda import _get_torch_cuda_version, SM90OrLater
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
    TestCase,
)


def requires_cuda_p2p_access():
    cuda_p2p_access_available = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (8, 0)
        and torch.cuda.device_count() >= 2
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


def requires_multicast_support():
    has_multicast_support = (
        torch.cuda.is_available()
        and _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    )
    return skip_but_pass_in_sandcastle_if(
        not has_multicast_support,
        "multicast support is not available",
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
        torch.manual_seed(42 + self.rank)

    def _get_test_alloc_args(self):
        shape = (64, 64)
        stride = (64, 1)
        dtype = torch.float32
        device = self.device
        group_name = "0"
        return (shape, stride, dtype, device, group_name)

    def _verify_symmetric_memory(self, symm_mem):
        self.assertEqual(symm_mem.world_size, 2)

        buf = symm_mem.get_buffer(0, (symm_mem.buffer_size // 4,), torch.float32)
        self.assertEqual(buf.storage_offset(), 0)
        self.assertEqual(buf.storage().size(), symm_mem.buffer_size // 4)

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

        alloc_args = self._get_test_alloc_args()

        t = torch.empty((64, 64), device=self.device)
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

        alloc_args = self._get_test_alloc_args()

        t = _SymmetricMemory.empty_strided_p2p(*alloc_args, alloc_id=42)
        data_ptr = t.data_ptr()

        # Verify that persistent allocation would fail if there's an active
        # allocation with the same alloc_id.
        with self.assertRaises(RuntimeError):
            _SymmetricMemory.empty_strided_p2p(*alloc_args, alloc_id=42)

        # Verify that persistent allocation would succeed in lieu of activate
        # allocations with the same alloc_id, and the returned tensor would
        # have the same data pointer.
        del t
        t = _SymmetricMemory.empty_strided_p2p(*alloc_args, alloc_id=42)
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
    def test_get_signal_pad(self) -> None:
        self._init_process()

        t = _SymmetricMemory.empty_strided_p2p(*self._get_test_alloc_args())
        symm_mem = _SymmetricMemory.rendezvous(t)
        peer_rank = (self.rank + 1) % self.world_size

        signal_pad = symm_mem.get_signal_pad(self.rank)
        self.assertEqual(signal_pad.data_ptr(), symm_mem.signal_pad_ptrs[symm_mem.rank])

        signal_pad = symm_mem.get_signal_pad(peer_rank)
        self.assertEqual(signal_pad.dtype, torch.uint32)
        self.assertEqual(signal_pad.numel(), symm_mem.signal_pad_size // 4)

        # Only specify sizes
        signal_pad = symm_mem.get_signal_pad(peer_rank, (8, 8))
        self.assertEqual(signal_pad.dtype, torch.uint32)
        self.assertEqual(signal_pad.numel(), 64)

        # Only specify dtype
        signal_pad = symm_mem.get_signal_pad(peer_rank, dtype=torch.uint64)
        self.assertEqual(signal_pad.dtype, torch.uint64)
        self.assertEqual(signal_pad.numel(), symm_mem.signal_pad_size // 8)

        # Specify both sizes and dtype
        signal_pad = symm_mem.get_signal_pad(peer_rank, (8, 8), dtype=torch.uint64)
        self.assertEqual(signal_pad.dtype, torch.uint64)
        self.assertEqual(signal_pad.numel(), 64)

        # Sanity check that writes to buffer doesn't corrupt signal_pad
        t = _SymmetricMemory.empty_strided_p2p(
            (0,),
            (0,),
            torch.float32,
            self.device,
            dist.group.WORLD.group_name,
        )
        symm_mem = _SymmetricMemory.rendezvous(t)
        signal_pad = symm_mem.get_signal_pad(self.rank)
        signal_pad.fill_(42)
        t.fill_(0)
        self.assertTrue(signal_pad.eq(42).all())

        dist.destroy_process_group()

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_barrier_timeout(self) -> None:
        self._init_process()

        alloc_args = self._get_test_alloc_args()

        t = _SymmetricMemory.empty_strided_p2p(*alloc_args)
        symm_mem = _SymmetricMemory.rendezvous(t)

        if self.rank == 0:
            with self.assertRaises(RuntimeError):
                symm_mem.barrier(timeout_ms=1000)
                torch.cuda.synchronize()
        else:
            torch.cuda.synchronize()

        # The device-side timeout triggers a __trap() that causes all
        # subsequent host/device interactions to result in an "unspecified
        # launch failure." Using os._exit(0) to abort the test, as it's
        # impossible to terminate the process in this state.
        os._exit(0)

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_put_signal_timeout(self) -> None:
        self._init_process()

        alloc_args = self._get_test_alloc_args()

        t = _SymmetricMemory.empty_strided_p2p(*alloc_args)
        symm_mem = _SymmetricMemory.rendezvous(t)

        if self.rank == 0:
            with self.assertRaises(RuntimeError):
                # First, put a signal into rank 1's signal pad. Since rank 1
                # doesn't wait on this signal, the subsequent put will timeout.
                symm_mem.put_signal(dst_rank=1)
                symm_mem.put_signal(dst_rank=1, timeout_ms=1000)
                torch.cuda.synchronize()
        else:
            torch.cuda.synchronize()

        # The device-side timeout triggers a __trap() that causes all
        # subsequent host/device interactions to result in an "unspecified
        # launch failure." Using os._exit(0) to abort the test, as it's
        # impossible to terminate the process in this state.
        os._exit(0)

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_wait_signal_timeout(self) -> None:
        self._init_process()

        alloc_args = self._get_test_alloc_args()

        t = _SymmetricMemory.empty_strided_p2p(*alloc_args)
        symm_mem = _SymmetricMemory.rendezvous(t)

        if self.rank == 0:
            with self.assertRaises(RuntimeError):
                symm_mem.wait_signal(src_rank=1, timeout_ms=1000)
                torch.cuda.synchronize()
        else:
            torch.cuda.synchronize()

        # The device-side timeout triggers a __trap() that causes all
        # subsequent host/device interactions to result in an "unspecified
        # launch failure." Using os._exit(0) to abort the test, as it's
        # impossible to terminate the process in this state.
        os._exit(0)

    @skipIfRocm
    @requires_cuda
    def test_allow_overlapping_devices(self) -> None:
        os.environ["TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES"] = "1"
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        group_name = dist.group.WORLD.group_name
        enable_symm_mem_for_group(group_name)

        t = _SymmetricMemory.empty_strided_p2p(
            (64,),
            (1,),
            dtype=torch.uint32,
            device=torch.device("cuda:0"),
            group_name=group_name,
        ).fill_(0)
        symm_mem_handle = _SymmetricMemory.rendezvous(t)

        self.assertEqual(symm_mem_handle.rank, self.rank)
        self.assertEqual(symm_mem_handle.world_size, self.world_size)

        for rank in range(self.world_size):
            buf = symm_mem_handle.get_buffer(rank, (64,), torch.float32)
            if rank == self.rank:
                self.assertEqual(buf.data_ptr(), t.data_ptr())
            else:
                self.assertEqual(buf.device, t.device)

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
    @skipIf(
        not SM90OrLater,
        "_fused_all_gather_matmul_native currently only supports sm>=90",
    )
    @skip_if_lt_x_gpu(2)
    @parametrize("symm_mem_input", [True, False])
    @parametrize("is_b_row_major", [True, False])
    def test_fused_all_gather_matmul_native(
        self, symm_mem_input: bool, is_b_row_major: bool
    ) -> None:
        self._init_process()

        M = 1024
        N = 1024
        K = 1024
        group_name = dist.group.WORLD.group_name

        torch.manual_seed(42 + self.rank)
        if symm_mem_input:
            A_shard = _SymmetricMemory.empty_strided_p2p(
                size=(M // self.world_size, K),
                stride=(K, 1),
                dtype=torch.bfloat16,
                device=self.device,
                group_name="0",
            ).normal_()
        else:
            A_shard = torch.rand(
                M // self.world_size, K, dtype=torch.bfloat16, device="cuda"
            )

        if is_b_row_major:
            B = torch.rand(K, N, dtype=torch.bfloat16, device="cuda")
        else:
            B = torch.rand(N, K, dtype=torch.bfloat16, device="cuda").t()

        ag_baseline, mm_baseline = _fused_all_gather_matmul_fallback(
            A_shard, [B], gather_dim=0, group_name=group_name
        )
        ag_target, mm_target = _fused_all_gather_matmul_native(
            A_shard, B, group_name=group_name
        )

        torch.testing.assert_close(ag_target, ag_baseline)
        torch.testing.assert_close(mm_target, mm_baseline[0])

        dist.destroy_process_group()

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    @parametrize("gather_dim", [0, 1])
    @parametrize(
        "scale_mode", ["tensor-wise", "row-wise-replicated", "row-wise-sharded"]
    )
    def test_fused_all_gather_scaled_matmul(
        self, gather_dim: int, scale_mode: str
    ) -> None:
        self._init_process()

        BATCH = 8
        M = 64
        N = 16
        K = 32
        group = dist.group.WORLD
        rank = self.rank
        world_size = self.world_size

        if gather_dim == 0:
            leading_dims = (BATCH // self.world_size, M)
        elif gather_dim == 1:
            leading_dims = (BATCH, M // self.world_size)
        else:
            raise AssertionError("Invalid scale_mode: {scale_mode}")

        torch.manual_seed(42 + rank)
        A_shard = torch.rand(*leading_dims, K, device="cuda").to(torch.float8_e4m3fn)
        Bs = [
            torch.rand(N, K, device="cuda").to(torch.float8_e4m3fn).T for _ in range(3)
        ]

        if scale_mode == "tensor-wise":
            A_scale = torch.tensor(0.1, device="cuda")
            B_scales = [torch.tensor(0.1, device="cuda") for _ in range(3)]
            out_dtypes = [None, torch.bfloat16, torch.float32]
        elif scale_mode == "row-wise-sharded":
            A_scale = torch.full((*leading_dims, 1), 0.1, device="cuda")
            B_scales = [torch.full((1, N), 0.1, device="cuda") for _ in range(3)]
            out_dtypes = [torch.bfloat16] * 3
        elif scale_mode == "row-wise-replicated":
            A_scale = torch.full((BATCH, M, 1), 0.1, device="cuda")
            B_scales = [torch.full((1, N), 0.1, device="cuda") for _ in range(3)]
            out_dtypes = [torch.bfloat16] * 3
        else:
            raise AssertionError(f"Invalid scale_mode: {scale_mode}")

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
    @parametrize("rowwise", [True, False])
    def test_fused_scaled_matmul_reduce_scatter(
        self, scatter_dim: int, rowwise: bool
    ) -> None:
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
        B = torch.rand(N, K, device="cuda").to(torch.float8_e4m3fn).T

        if rowwise:
            A_scale = torch.full((BATCH, M, 1), 0.1, device="cuda")
            B_scale = torch.full((1, N), 0.1, device="cuda")
        else:
            A_scale = torch.tensor(0.1, device="cuda")
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

    @skipIfRocm
    @parametrize("dim", [0, 1, 2])
    def test_optimal_layout(self, dim: int) -> None:
        t = torch.rand(8, 64, 32, 16)

        x = restride_A_shard_for_fused_all_gather_matmul(t, dim)
        self.assertTrue(x.movedim(dim, 0).is_contiguous())
        self.assertTrue(torch.allclose(x, t))

        x = restride_A_for_fused_matmul_reduce_scatter(t, dim)
        self.assertTrue(x.movedim(dim, 0).is_contiguous())
        self.assertTrue(torch.allclose(x, t))

    @skipIfRocm
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

    @skipIfRocm
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


@instantiate_parametrized_tests
@requires_cuda_p2p_access()
class SymmMemAllReduceTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        # world_size > 2 is needed to verify accumulation order
        return 4

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
        torch.manual_seed(42 + self.rank)

    @skip_if_lt_x_gpu(4)
    @requires_multicast_support()
    @parametrize("dtype", [torch.float, torch.bfloat16])
    @parametrize("align_bytes", [4, 8, 16])
    @parametrize("size_bytes", [4, 8192, 8196])
    def test_multimem_all_reduce(
        self, dtype: torch.dtype, size_bytes: int, align_bytes: int
    ) -> None:
        self._init_process()
        group_name = dist.group.WORLD.group_name

        t = _SymmetricMemory.empty_strided_p2p(
            size=(16384,),
            stride=(1,),
            dtype=dtype,
            device=self.device,
            group_name=group_name,
        ).fill_(0)

        self.assertTrue(t.data_ptr() % 16 == 0)
        self.assertTrue(align_bytes % t.element_size() == 0)
        self.assertTrue(size_bytes % t.element_size() == 0)

        shift = align_bytes // t.element_size()
        numel = size_bytes // t.element_size()
        res = t[shift : shift + numel]
        res.normal_()
        inp = res.clone()

        torch.ops.symm_mem.multimem_all_reduce_(res, "sum", group_name)

        # Head and tail should not be written
        self.assertTrue(t[:shift].eq(0).all().item())
        self.assertTrue(t[shift + numel :].eq(0).all().item())
        self._verify_all_reduce_result(inp, res)

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    @requires_multicast_support()
    @parametrize("dtype", [torch.float, torch.bfloat16])
    @parametrize("align_bytes", [4, 8, 16])
    @parametrize("size_bytes", [4, 8192, 8196])
    def test_multimem_one_shot_all_reduce(
        self, dtype: torch.dtype, size_bytes: int, align_bytes: int
    ) -> None:
        self._init_process()
        group_name = dist.group.WORLD.group_name

        inp = _SymmetricMemory.empty_strided_p2p(
            size=(size_bytes,),
            stride=(1,),
            dtype=dtype,
            device=self.device,
            group_name=group_name,
        ).normal_()

        res = torch.ops.symm_mem.multimem_one_shot_all_reduce(inp, "sum", group_name)

        gathered_inps = all_gather_tensor(inp, 0, "0").view(self.world_size, -1)
        # Only verify that the results are close to the sum of inputs across
        # ranks (see Note [multimem_one_shot_all_reduce]).
        torch.testing.assert_close(
            gathered_inps.sum(dim=0), res, rtol=1e-03, atol=1e-05
        )

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    @parametrize("dtype", [torch.float, torch.bfloat16])
    @parametrize("align_bytes", [4, 8, 16])
    @parametrize("size_bytes", [4, 8192, 8196])
    def test_one_shot_all_reduce(
        self, dtype: torch.dtype, size_bytes: int, align_bytes: int
    ) -> None:
        self._init_process()
        group_name = dist.group.WORLD.group_name

        inp = _SymmetricMemory.empty_strided_p2p(
            size=(size_bytes,),
            stride=(1,),
            dtype=dtype,
            device=self.device,
            group_name=group_name,
        ).normal_()

        res = torch.ops.symm_mem.one_shot_all_reduce(inp, "sum", group_name)
        self._verify_all_reduce_result(inp, res)

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    @parametrize("dtype", [torch.float, torch.bfloat16])
    @parametrize("align_bytes", [4, 8, 16])
    @parametrize("size_bytes", [4, 8192, 8196])
    def test_two_shot_all_reduce(
        self, dtype: torch.dtype, size_bytes: int, align_bytes: int
    ) -> None:
        self._init_process()
        group_name = dist.group.WORLD.group_name

        t = _SymmetricMemory.empty_strided_p2p(
            size=(16384,),
            stride=(1,),
            dtype=dtype,
            device=self.device,
            group_name=group_name,
        ).fill_(0)

        self.assertTrue(t.data_ptr() % 16 == 0)
        self.assertTrue(align_bytes % t.element_size() == 0)
        self.assertTrue(size_bytes % t.element_size() == 0)

        shift = align_bytes // t.element_size()
        numel = size_bytes // t.element_size()
        res = t[shift : shift + numel]
        res.normal_()
        inp = res.clone()

        torch.ops.symm_mem.two_shot_all_reduce_(res, "sum", group_name)

        # Head and tail should not be written
        self.assertTrue(t[:shift].eq(0).all().item())
        self.assertTrue(t[shift + numel :].eq(0).all().item())
        self._verify_all_reduce_result(inp, res)

        dist.destroy_process_group()

    def _verify_all_reduce_result(self, inp, res):
        gathered_res = all_gather_tensor(res, 0, "0").view(self.world_size, -1)
        # Verify that the results across ranks are identical
        self.assertEqual(
            (gathered_res == gathered_res[0, :]).all(dim=0).sum(), inp.numel()
        )

        # Verify that the result are close to the sum of inputs across ranks
        gathered_inps = all_gather_tensor(inp, 0, "0").view(self.world_size, -1)
        torch.testing.assert_close(
            gathered_inps.sum(dim=0), res, rtol=1e-01, atol=1e-01
        )


@instantiate_parametrized_tests
@requires_cuda_p2p_access()
class LoweringTest(MultiProcessTestCase):
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
        torch.manual_seed(42 + self.rank)

        torch._inductor.config._collective.auto_select = True

    @skipIfRocm  # requires registered-buffer support
    @skip_if_lt_x_gpu(2)
    @fresh_inductor_cache()
    def test_lowering_one_shot_all_reduce(self):
        self._init_process()

        arg = torch.rand(4, 4, device=self.device)

        def func_0(x):
            x = x + 1
            x = torch.ops._c10d_functional.all_reduce(x, "sum", "0")
            return torch.ops._c10d_functional.wait_tensor(x)

        compiled_0 = torch.compile(func_0, fullgraph=True)
        code_0 = run_and_get_triton_code(compiled_0, arg)

        self.assertIn("one_shot_all_reduce", code_0)
        self.assertNotIn("return (buf0", code_0)

        # All-reduce on a slice view
        def func_1(x):
            x = x + 1
            x = x[2:]
            x = torch.ops._c10d_functional.all_reduce(x, "sum", "0")
            return torch.ops._c10d_functional.wait_tensor(x)

        compiled_1 = torch.compile(func_1, fullgraph=True)
        code_1 = run_and_get_triton_code(compiled_1, arg)

        self.assertIn("one_shot_all_reduce", code_1)
        self.assertNotIn("return (buf0", code_1)

        # All-reduce on input
        def func_2(x):
            x = torch.ops._c10d_functional.all_reduce(x, "sum", "0")
            return torch.ops._c10d_functional.wait_tensor(x)

        compiled_2 = torch.compile(func_2, fullgraph=True)
        code_2 = run_and_get_triton_code(compiled_2, arg)

        self.assertNotIn("one_shot_all_reduce", code_2)

        # All-reduce on matmul output
        def func_3(x):
            x = x @ x
            x = torch.ops._c10d_functional.all_reduce(x, "sum", "0")
            return torch.ops._c10d_functional.wait_tensor(x)

        compiled_3 = torch.compile(func_3, fullgraph=True)
        code_3 = run_and_get_triton_code(compiled_3, arg)

        self.assertIn("one_shot_all_reduce", code_3)
        self.assertNotIn("return (buf0", code_3)


class SymmMemSingleProcTest(TestCase):
    @skipIfRocm
    @requires_cuda
    @skipIf(
        _get_torch_cuda_version() < (12, 0),
        "stream_write_value32 currently only supports cuda version>=12.0",
    )
    def test_stream_write_value32(self):
        tensor = torch.zeros(4, dtype=torch.uint32, device="cuda")
        expect = torch.tril(torch.ones(4, 4, device="cuda")).to(torch.uint32)

        for i in range(4):
            _SymmetricMemory.stream_write_value32(tensor, i, 1)
            torch.testing.assert_close(tensor, expect[i])

        with self.assertRaises(RuntimeError):
            _SymmetricMemory.stream_write_value32(tensor, offset=-1, val=1)

        with self.assertRaises(RuntimeError):
            _SymmetricMemory.stream_write_value32(tensor, offset=0, val=4294967296)

    @skipIfRocm
    @requires_cuda
    def test_memset32(self):
        t = _SymmetricMemory.empty_strided_p2p(
            (64,),
            (1,),
            dtype=torch.uint32,
            device=torch.device("cuda:0"),
            group_name="0",
        ).fill_(0)

        _SymmetricMemory.memset32(t, offset=32, val=1, count=16)
        self.assertTrue(t[:32].eq(0).all())
        self.assertTrue(t[32:48].eq(1).all())
        self.assertTrue(t[48:].eq(0).all())

        with self.assertRaisesRegex(
            RuntimeError, "input must be a flat, contiguous uint32 tensor"
        ):
            _SymmetricMemory.memset32(t.view(8, 8), offset=0, val=1, count=1)

        with self.assertRaisesRegex(
            RuntimeError, "input must be a flat, contiguous uint32 tensor"
        ):
            _SymmetricMemory.memset32(t.view(torch.float32), offset=0, val=1, count=1)

        with self.assertRaisesRegex(
            RuntimeError, "offset must be greater than or equal to 0"
        ):
            _SymmetricMemory.memset32(t, offset=-1, val=1, count=1)

        with self.assertRaisesRegex(
            RuntimeError, r"val must be in the range of.*\(uint32_t\)"
        ):
            _SymmetricMemory.memset32(t, offset=0, val=4294967296, count=1)

        with self.assertRaisesRegex(RuntimeError, "count must be a positive integer"):
            _SymmetricMemory.memset32(t, offset=0, val=1, count=-1)

        with self.assertRaisesRegex(RuntimeError, "count must be a positive integer"):
            _SymmetricMemory.memset32(t, offset=0, val=1, count=0)

        with self.assertRaisesRegex(
            RuntimeError, r"offset \+ count.*exceeded the numel of the input"
        ):
            _SymmetricMemory.memset32(t, offset=64, val=1, count=1)

        with self.assertRaisesRegex(
            RuntimeError, r"offset \+ count.*exceeded the numel of the input"
        ):
            _SymmetricMemory.memset32(t, offset=0, val=1, count=65)

        _SymmetricMemory.memset32(t, offset=0, val=1, count=64)
        _SymmetricMemory.memset32(t, offset=63, val=1, count=1)


if __name__ == "__main__":
    run_tests()
