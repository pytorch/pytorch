# Owner(s): ["module: dynamo"]

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from torch._higher_order_ops.flex_ep import FlexEPDispatchPlan, flex_ep
from torch._inductor.utils import run_fw_bw_and_get_code
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    PLATFORM_SUPPORTS_SYMM_MEM,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
)
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils._triton import has_triton


BATCH = 2
HIDDEN_DIM = 16
INTERMEDIATE_DIM = 16
LOCAL_EXPERTS = 1
TOPK = 1
TOKEN_ALIGNMENT = 128
EP_TIMEOUT_SECONDS = 120.0


def requires_nvshmem():
    return skip_but_pass_in_sandcastle_if(
        not PLATFORM_SUPPORTS_SYMM_MEM or not symm_mem.is_nvshmem_available(),
        "test_flex_ep_nvshmem requires NVSHMEM with SymmMem platform support",
    )


def _align_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def _compute_max_tokens_recv(B: int, ep_size: int, num_experts: int, topk: int) -> int:
    local_experts = num_experts // ep_size
    max_tokens_received = _align_up(
        B * ep_size * min(local_experts, topk),
        TOKEN_ALIGNMENT,
    )
    return max_tokens_received + TOKEN_ALIGNMENT * local_experts


@dataclass
class NvlSharedBuffer:
    raw: torch.Tensor
    barrier_counter: torch.Tensor
    dispatch_recv_buffer: torch.Tensor
    dispatch_recv_buffer_scaling_factors: torch.Tensor
    dispatch_recv_weights: torch.Tensor
    dispatch_recv_origin_global_token_id: torch.Tensor
    combine_recv_buffer: torch.Tensor
    combine_recv_scale_factors: torch.Tensor
    combine_recv_weights: torch.Tensor
    allgather_expert_counts: torch.Tensor

    @staticmethod
    def tensors_shapes_and_dtypes(
        B: int,
        D: int,
        EP_SIZE: int,
        NUM_EXPERTS: int,
        TOPK: int,
    ) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
        assert NUM_EXPERTS % EP_SIZE == 0  # noqa: S101
        max_tokens_received = _compute_max_tokens_recv(
            B,
            EP_SIZE,
            NUM_EXPERTS,
            TOPK,
        )
        return (
            ("barrier_counter", (16,), torch.int32),
            ("dispatch_recv_buffer", (max_tokens_received, D), torch.uint16),
            (
                "dispatch_recv_buffer_scaling_factors",
                (max_tokens_received, D // 16),
                torch.float8_e8m0fnu,
            ),
            ("dispatch_recv_weights", (max_tokens_received,), torch.float32),
            (
                "dispatch_recv_origin_global_token_id",
                (max_tokens_received,),
                torch.int64,
            ),
            ("combine_recv_buffer", (B, TOPK, D), torch.uint16),
            ("combine_recv_scale_factors", (B, TOPK, D // 16), torch.float8_e8m0fnu),
            ("combine_recv_weights", (B, TOPK), torch.float32),
            ("allgather_expert_counts", (EP_SIZE, NUM_EXPERTS), torch.int64),
        )

    @classmethod
    def get_buffer_size_bytes(cls, *args, **kwargs) -> int:
        total_size = 0
        for _name, shape, dtype in cls.tensors_shapes_and_dtypes(*args, **kwargs):
            total_size = _align_up(total_size, 16)
            numel = 1
            for dim in shape:
                numel *= dim
            total_size += numel * dtype.itemsize
        return total_size

    @classmethod
    def view_from_buffer(
        cls,
        buffer: torch.Tensor,
        *,
        B: int,
        D: int,
        EP_SIZE: int,
        NUM_EXPERTS: int,
        TOPK: int,
    ) -> "NvlSharedBuffer":
        assert buffer.ndim == 1  # noqa: S101
        assert buffer.dtype == torch.uint8  # noqa: S101
        assert buffer.is_contiguous()  # noqa: S101

        offset = 0

        def view_buffer_chunk(
            shape: tuple[int, ...],
            dtype: torch.dtype,
        ) -> torch.Tensor:
            nonlocal offset
            offset = _align_up(offset, 16)
            num_bytes = dtype.itemsize
            for dim in shape:
                num_bytes *= dim
            assert offset + num_bytes <= buffer.numel()  # noqa: S101
            out = buffer[offset : offset + num_bytes].view(dtype).view(shape)
            offset += num_bytes
            return out

        tensors = {
            name: view_buffer_chunk(shape, dtype)
            for name, shape, dtype in cls.tensors_shapes_and_dtypes(
                B=B,
                D=D,
                EP_SIZE=EP_SIZE,
                NUM_EXPERTS=NUM_EXPERTS,
                TOPK=TOPK,
            )
        }
        assert offset == buffer.numel()  # noqa: S101
        return cls(raw=buffer, **tensors)

    def offset_of(self, name: str) -> int:
        return getattr(self, name).data_ptr() - self.raw.data_ptr()


@dataclass(frozen=True)
class RouterOperands:
    raw: torch.Tensor
    buffers_cuda_ptrs: torch.Tensor
    offs_barrier_counter: int
    offs_dispatch_recv_buffer: int
    offs_dispatch_recv_buffer_scaling_factors: int
    offs_dispatch_recv_weights: int
    offs_dispatch_recv_origin_global_token_id: int
    offs_combine_recv_buffer: int
    offs_combine_recv_scale_factors: int
    offs_combine_recv_weights: int
    offs_allgather_expert_counts: int
    ep_rank: int


torch.utils._pytree.register_dataclass(RouterOperands)


_ROUTER_EP_BACKEND_SKIP_REASON: str | None = None

_INDUCTOR_ROUTER_OP_NAMES = {
    "ep_allgather": "flex_ep_allgather",
    "router_dispatch": "flex_ep_router_dispatch",
    "router_combine": "flex_ep_router_combine",
    "barrier_arrive": "flex_ep_barrier_arrive",
    "barrier_wait": "flex_ep_barrier_wait",
    "zfill_ranges_inplace": "flex_ep_zfill_ranges_inplace",
}


def _has_inductor_router_op(name: str) -> bool:
    try:
        getattr(torch.ops.inductor, _INDUCTOR_ROUTER_OP_NAMES[name])
        return True
    except AttributeError:
        return False


def _register_test_router_ep_backend_ops() -> bool:
    global _ROUTER_EP_BACKEND_SKIP_REASON

    required_ops = (
        "ep_allgather",
        "router_dispatch",
        "router_combine",
        "barrier_arrive",
        "barrier_wait",
        "zfill_ranges_inplace",
    )
    if all(_has_inductor_router_op(name) for name in required_ops):
        _ROUTER_EP_BACKEND_SKIP_REASON = None
        return True

    try:
        import triton
        import triton.language as tl

        import torch.distributed._symmetric_memory._shmem_triton as shmem_triton
        from torch.distributed._symmetric_memory._shmem_triton import requires_shmem
    except ImportError as exc:
        _ROUTER_EP_BACKEND_SKIP_REASON = (
            f"test_flex_ep_nvshmem requires Triton RouterEP kernels: {exc}"
        )
        return False

    shmem_backend = shmem_triton.get_shmem_backend_module()

    @requires_shmem
    @triton.jit
    def _barrier_arrive_kernel(flag_ptr, out_ptr):
        shmem_backend.quiet()
        old = tl.atomic_add(
            flag_ptr,
            tl.full([], 1, dtype=tl.int32),
            sem="release",
            scope="sys",
        )
        tl.store(out_ptr, old + 1)

    @triton.jit
    def _barrier_wait_kernel(
        cuda_ptrs_ptr,
        expected_ptr,
        n_flags,
        offs_flag,
        timeout_ns,
        BLOCK: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK)
        mask = offs < n_flags
        expected = tl.load(expected_ptr)
        base_ptrs = tl.load(cuda_ptrs_ptr + offs, mask=mask, other=0)
        flag_ptrs = (base_ptrs + offs_flag).to(tl.pointer_type(tl.int32))
        zero = tl.zeros([BLOCK], dtype=tl.int32)
        start = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;",
            "=l",
            [],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        )
        done = tl.zeros([], dtype=tl.int32)
        while done == 0:
            vals = tl.atomic_add(
                flag_ptrs,
                zero,
                mask=mask,
                sem="acquire",
                scope="sys",
            )
            waiting = tl.sum(tl.where(mask & (vals < expected), 1, 0))
            if waiting == 0:
                done = 1
            else:
                now = tl.inline_asm_elementwise(
                    "mov.u64 $0, %globaltimer;",
                    "=l",
                    [],
                    dtype=tl.int64,
                    is_pure=False,
                    pack=1,
                )
                if (now - start) > timeout_ns:
                    tl.device_print("flex_ep test barrier_wait timed out")
                    tl.inline_asm_elementwise(
                        "trap;",
                        "=r",
                        [],
                        dtype=tl.int32,
                        is_pure=False,
                        pack=1,
                    )
                    done = 1

    @requires_shmem
    @triton.jit
    def _ep_allgather_kernel(
        input_ptr,
        buffers_cuda_ptrs_ptr,
        offs_output,
        ep_rank,
        input_num_bytes: tl.constexpr,
    ):
        peer = tl.program_id(0)
        base_ptr = tl.load(buffers_cuda_ptrs_ptr + ep_rank)
        dst_ptr = (base_ptr + offs_output + ep_rank * input_num_bytes).to(
            tl.pointer_type(tl.uint8)
        )
        shmem_backend.put(dst_ptr, input_ptr, input_num_bytes, peer)
        shmem_backend.quiet()

    @requires_shmem
    @triton.jit
    def _router_dispatch_kernel(
        my_tokens_ptr,
        dest_ranks_ptr,
        dest_offsets_ptr,
        buffers_cuda_ptrs_ptr,
        dispatch_recv_origin_global_token_id_ptr,
        dispatch_recv_weights_ptr,
        offs_recv_tokens,
        offs_recv_origin_global_token_id,
        ep_rank,
        max_B,
        D_BYTES: tl.constexpr,
        TOPK: tl.constexpr,
        WRITE_MAPPING: tl.constexpr,
    ):
        copy_id = tl.program_id(0)
        token_id = copy_id // TOPK
        expert_slot = copy_id - token_id * TOPK
        dest_rank = tl.load(dest_ranks_ptr + copy_id)
        dest_offset = tl.load(dest_offsets_ptr + copy_id)
        valid = dest_rank >= 0

        src_ptr = my_tokens_ptr + copy_id * D_BYTES
        base_ptr = tl.load(buffers_cuda_ptrs_ptr + ep_rank)
        dst_ptr = (base_ptr + offs_recv_tokens + dest_offset * D_BYTES).to(
            tl.pointer_type(tl.uint8)
        )

        if valid:
            shmem_backend.put(dst_ptr, src_ptr, D_BYTES, dest_rank)

        if valid and WRITE_MAPPING:
            id_ptr = (base_ptr + offs_recv_origin_global_token_id).to(
                tl.pointer_type(tl.int64)
            )
            global_token_id = ep_rank * max_B * TOPK + token_id * TOPK + expert_slot
            scratch_id_ptr = dispatch_recv_weights_ptr.to(tl.pointer_type(tl.int64))
            scratch_id_ptr += copy_id
            tl.store(scratch_id_ptr, global_token_id)
            shmem_backend.put(id_ptr + dest_offset, scratch_id_ptr, 1, dest_rank)
        shmem_backend.quiet()

    @requires_shmem
    @triton.jit
    def _router_combine_kernel(
        send_tokens_ptr,
        token_send_end_ptr,
        send_origin_global_token_id_ptr,
        buffers_cuda_ptrs_ptr,
        offs_combine_recv_tokens,
        ep_rank,
        ep_size,
        max_B,
        D_BYTES: tl.constexpr,
        TOPK: tl.constexpr,
    ):
        copy_id = tl.program_id(0)
        token_send_end = tl.load(token_send_end_ptr)
        valid = copy_id < token_send_end
        origin_id = tl.load(
            send_origin_global_token_id_ptr + copy_id, mask=valid, other=-1
        )
        valid = valid & (origin_id >= 0)

        max_B_topk = max_B * TOPK
        from_ep_rank = origin_id // max_B_topk
        dest_idx = origin_id - from_ep_rank * max_B_topk
        valid = valid & (from_ep_rank >= 0) & (from_ep_rank < ep_size)

        src_ptr = send_tokens_ptr + copy_id * D_BYTES
        base_ptr = tl.load(buffers_cuda_ptrs_ptr + ep_rank)
        dst_ptr = (base_ptr + offs_combine_recv_tokens + dest_idx * D_BYTES).to(
            tl.pointer_type(tl.uint8)
        )
        if valid:
            shmem_backend.put(dst_ptr, src_ptr, D_BYTES, from_ep_rank.to(tl.int32))
        shmem_backend.quiet()

    @triton.jit
    def _zfill_ranges_kernel(
        input_ptr,
        begin_ofs_ptr,
        end_ofs_ptr,
        row_num_bytes: tl.constexpr,
        BLOCK_BYTES: tl.constexpr,
    ):
        range_idx = tl.program_id(0)
        row_in_range = tl.program_id(1)
        begin = tl.load(begin_ofs_ptr + range_idx)
        end = tl.load(end_ofs_ptr + range_idx)
        row = begin + row_in_range
        byte_offsets = tl.arange(0, BLOCK_BYTES)
        mask = (row < end) & (byte_offsets < row_num_bytes)
        tl.store(
            input_ptr + row * row_num_bytes + byte_offsets,
            tl.zeros([BLOCK_BYTES], dtype=tl.uint8),
            mask=mask,
        )

    if not _has_inductor_router_op("barrier_arrive"):

        @torch.library.custom_op(
            "inductor::flex_ep_barrier_arrive",
            mutates_args=("flag",),
        )
        def _test_barrier_arrive(flag: torch.Tensor) -> torch.Tensor:
            out = torch.empty(1, dtype=torch.int32, device=flag.device)
            _barrier_arrive_kernel[(1,)](flag, out, num_warps=1)
            return out

    if not _has_inductor_router_op("barrier_wait"):

        @torch.library.custom_op("inductor::flex_ep_barrier_wait", mutates_args=())
        def _test_barrier_wait(
            cuda_ptrs: torch.Tensor,
            offs_flag: int,
            expected: torch.Tensor,
            timeout_s: float = 5.0,
        ) -> None:
            block = triton.next_power_of_2(cuda_ptrs.numel())
            timeout_ns = int(timeout_s * 1e9)
            _barrier_wait_kernel[(1,)](
                cuda_ptrs,
                expected,
                cuda_ptrs.numel(),
                offs_flag,
                timeout_ns,
                BLOCK=block,
                num_warps=1,
            )

    if not _has_inductor_router_op("ep_allgather"):

        @torch.library.custom_op(
            "inductor::flex_ep_allgather",
            mutates_args=("output",),
        )
        def _test_ep_allgather(
            output: torch.Tensor,
            input: torch.Tensor,
            buffers_cuda_ptrs: torch.Tensor,
            offs_output: int,
            ep_rank: int,
        ) -> None:
            input_u8 = input.contiguous().view(torch.uint8)
            _ep_allgather_kernel[(buffers_cuda_ptrs.numel(),)](
                input_u8,
                buffers_cuda_ptrs,
                offs_output,
                ep_rank,
                input_num_bytes=input_u8.numel(),
                num_warps=1,
            )

    if not _has_inductor_router_op("router_dispatch"):

        @torch.library.custom_op(
            "inductor::flex_ep_router_dispatch",
            mutates_args=(
                "dispatch_recv_buffer",
                "dispatch_recv_buffer_scaling_factors",
                "dispatch_recv_origin_global_token_id",
                "dispatch_recv_weights",
            ),
            schema="(Tensor my_tokens, Tensor? my_scaling_factors, "
            "Tensor? my_topk_weights, Tensor dest_ranks, Tensor dest_offsets, "
            "Tensor buffers_cuda_ptrs, Tensor(a!) dispatch_recv_buffer, "
            "Tensor(b!) dispatch_recv_buffer_scaling_factors, "
            "Tensor(c!) dispatch_recv_origin_global_token_id, "
            "Tensor(d!) dispatch_recv_weights, int offs_recv_tokens, "
            "int offs_recv_scaling_factors, int offs_recv_weights, "
            "int offs_recv_origin_global_token_id, int ep_rank, int num_ctas, "
            "int max_B) -> ()",
        )
        def _test_router_dispatch(
            my_tokens: torch.Tensor,
            my_scaling_factors: torch.Tensor | None,
            my_topk_weights: torch.Tensor | None,
            dest_ranks: torch.Tensor,
            dest_offsets: torch.Tensor,
            buffers_cuda_ptrs: torch.Tensor,
            dispatch_recv_buffer: torch.Tensor,
            dispatch_recv_buffer_scaling_factors: torch.Tensor,
            dispatch_recv_origin_global_token_id: torch.Tensor,
            dispatch_recv_weights: torch.Tensor,
            offs_recv_tokens: int,
            offs_recv_scaling_factors: int,
            offs_recv_weights: int,
            offs_recv_origin_global_token_id: int,
            ep_rank: int,
            num_ctas: int,
            max_B: int,
        ) -> None:
            del my_scaling_factors, my_topk_weights
            del dispatch_recv_buffer, dispatch_recv_buffer_scaling_factors
            del offs_recv_scaling_factors, offs_recv_weights, num_ctas
            assert my_tokens.dtype == torch.bfloat16  # noqa: S101
            assert my_tokens.is_contiguous()  # noqa: S101
            B, topk, D = my_tokens.shape
            my_tokens_u8 = my_tokens.view(torch.uint8)
            D_bytes = D * my_tokens.element_size()
            _router_dispatch_kernel[(B * topk,)](
                my_tokens_u8,
                dest_ranks,
                dest_offsets,
                buffers_cuda_ptrs,
                dispatch_recv_origin_global_token_id,
                dispatch_recv_weights,
                offs_recv_tokens,
                offs_recv_origin_global_token_id,
                ep_rank,
                max_B,
                D_BYTES=D_bytes,
                TOPK=topk,
                WRITE_MAPPING=offs_recv_origin_global_token_id >= 0,
                num_warps=1,
            )

    if not _has_inductor_router_op("router_combine"):

        @torch.library.custom_op(
            "inductor::flex_ep_router_combine",
            mutates_args=(
                "combine_recv_buffer",
                "combine_recv_scale_factors",
                "combine_recv_weights",
            ),
            schema="(Tensor send_tokens, Tensor? send_scale_factors, "
            "Tensor? send_weights, Tensor expert_begin_offset_per_ep, "
            "Tensor token_send_end, Tensor send_origin_global_token_id, "
            "Tensor buffers_cuda_ptrs, Tensor(a!) combine_recv_buffer, "
            "Tensor(b!) combine_recv_scale_factors, Tensor(c!) combine_recv_weights, "
            "int offs_combine_recv_tokens, int offs_combine_recv_scale_factors, "
            "int offs_combine_recv_weights, int ep_rank, int B, int TOPK, "
            "int num_ctas, int max_B) -> ()",
        )
        def _test_router_combine(
            send_tokens: torch.Tensor,
            send_scale_factors: torch.Tensor | None,
            send_weights: torch.Tensor | None,
            expert_begin_offset_per_ep: torch.Tensor,
            token_send_end: torch.Tensor,
            send_origin_global_token_id: torch.Tensor,
            buffers_cuda_ptrs: torch.Tensor,
            combine_recv_buffer: torch.Tensor,
            combine_recv_scale_factors: torch.Tensor,
            combine_recv_weights: torch.Tensor,
            offs_combine_recv_tokens: int,
            offs_combine_recv_scale_factors: int,
            offs_combine_recv_weights: int,
            ep_rank: int,
            B: int,
            TOPK: int,
            num_ctas: int,
            max_B: int,
        ) -> None:
            del send_scale_factors, send_weights, expert_begin_offset_per_ep
            del combine_recv_buffer, combine_recv_scale_factors, combine_recv_weights
            del offs_combine_recv_scale_factors, offs_combine_recv_weights
            del B, num_ctas
            assert send_tokens.dtype == torch.bfloat16  # noqa: S101
            assert send_tokens.is_contiguous()  # noqa: S101
            D = send_tokens.shape[-1]
            send_tokens_u8 = send_tokens.view(torch.uint8)
            D_bytes = D * send_tokens.element_size()
            _router_combine_kernel[(send_tokens.shape[0],)](
                send_tokens_u8,
                token_send_end,
                send_origin_global_token_id,
                buffers_cuda_ptrs,
                offs_combine_recv_tokens,
                ep_rank,
                buffers_cuda_ptrs.numel(),
                max_B,
                D_BYTES=D_bytes,
                TOPK=TOPK,
                num_warps=1,
            )

    if not _has_inductor_router_op("zfill_ranges_inplace"):

        @torch.library.custom_op(
            "inductor::flex_ep_zfill_ranges_inplace",
            mutates_args=("input",),
        )
        def _test_zfill_ranges_inplace(
            input: torch.Tensor,
            begin_ofs: torch.Tensor,
            end_ofs: torch.Tensor,
            max_values_per_batch: int,
        ) -> None:
            assert input.is_contiguous()  # noqa: S101
            assert input.ndim == 2  # noqa: S101
            block_bytes = triton.next_power_of_2(input.shape[1])
            _zfill_ranges_kernel[(begin_ofs.numel(), max_values_per_batch)](
                input,
                begin_ofs,
                end_ofs,
                row_num_bytes=input.shape[1],
                BLOCK_BYTES=block_bytes,
                num_warps=1,
            )

    if all(_has_inductor_router_op(name) for name in required_ops):
        _ROUTER_EP_BACKEND_SKIP_REASON = None
        return True
    _ROUTER_EP_BACKEND_SKIP_REASON = (
        "test_flex_ep_nvshmem could not register RouterEP backend kernels"
    )
    return False


def has_router_ep_backend_ops() -> bool:
    return _register_test_router_ep_backend_ops()


def requires_router_ep_backend_ops():
    return skip_but_pass_in_sandcastle_if(
        not has_router_ep_backend_ops(),
        _ROUTER_EP_BACKEND_SKIP_REASON
        or "test_flex_ep_nvshmem requires registered RouterEP backend kernels",
    )


def _view_beginning_as(x: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype):
    num_bytes = dtype.itemsize
    for dim in shape:
        num_bytes *= dim
    return x.view(-1).view(torch.uint8)[:num_bytes].view(dtype).view(shape)


def _router_barrier(
    dependency,
    barrier_counter,
    buffers_cuda_ptrs,
    offs_barrier_counter,
    *,
    nonce,
    clone_result=False,
):
    value = torch.ops._flex_ep.barrier_arrive(
        barrier_counter[:1],
        dependency,
        nonce,
    )
    waited = torch.ops._flex_ep.barrier_wait(
        dependency,
        buffers_cuda_ptrs,
        offs_barrier_counter,
        value,
        EP_TIMEOUT_SECONDS,
    )
    if clone_result:
        return waited
    return dependency


def _make_router_fns(num_experts: int, ep_size: int, num_ctas: int = 152):
    local_experts = num_experts // ep_size
    max_recv_tokens = _compute_max_tokens_recv(
        BATCH,
        ep_size,
        num_experts,
        TOPK,
    )

    def view_buffer(raw):
        return NvlSharedBuffer.view_from_buffer(
            raw,
            B=BATCH,
            D=HIDDEN_DIM,
            EP_SIZE=ep_size,
            NUM_EXPERTS=num_experts,
            TOPK=TOPK,
        )

    def build_dispatch_plan_fn(
        topk_idx,
        operands,
    ):
        buffer = view_buffer(operands.raw)
        barrier_counter = buffer.barrier_counter
        dispatch_recv_origin_global_token_id = (
            buffer.dispatch_recv_origin_global_token_id
        )
        allgather_expert_counts = buffer.allgather_expert_counts

        dispatch_recv_origin_global_token_id = _router_barrier(
            dispatch_recv_origin_global_token_id,
            barrier_counter,
            operands.buffers_cuda_ptrs,
            operands.offs_barrier_counter,
            nonce=0,
        )

        dispatch_recv_origin_global_token_id = torch.ops._flex_ep.fill_i64_inplace(
            dispatch_recv_origin_global_token_id,
            -1,
        )
        expert_count_buffer = torch.zeros(
            num_experts,
            dtype=torch.int64,
            device=topk_idx.device,
        )
        expert_count_buffer.scatter_(
            0,
            topk_idx.flatten().to(torch.int64),
            1,
            reduce="add",
        )
        allgather_expert_counts = torch.ops._flex_ep.ep_allgather(
            allgather_expert_counts,
            expert_count_buffer,
            operands.buffers_cuda_ptrs,
            operands.offs_allgather_expert_counts,
            operands.ep_rank,
        )
        allgather_expert_counts = _router_barrier(
            allgather_expert_counts,
            barrier_counter,
            operands.buffers_cuda_ptrs,
            operands.offs_barrier_counter,
            nonce=1,
            clone_result=True,
        )

        all_offsets, recv_total_tokens, local_experts_start = (
            torch.ops._flex_ep.router_compute_all_expert_offsets(
                allgather_expert_counts,
                operands.ep_rank,
                local_experts,
                TOKEN_ALIGNMENT,
            )
        )
        expert_begin_offset = all_offsets[operands.ep_rank]
        recv_ofs = all_offsets[:, :, operands.ep_rank].reshape(-1)
        dest_ranks, dest_offsets = torch.ops._flex_ep.router_compute_dest_offsets(
            topk_idx,
            recv_ofs,
            ep_size,
        )
        max_recv_tokens_tensor = torch.full(
            (), max_recv_tokens, device=topk_idx.device, dtype=torch.int32
        )
        overflow = local_experts_start[-1] > max_recv_tokens_tensor
        return FlexEPDispatchPlan(
            dispatch_recv_origin_global_token_id[:max_recv_tokens],
            expert_begin_offset,
            dest_ranks,
            dest_offsets,
            local_experts_start,
            max_recv_tokens_tensor,
            recv_total_tokens,
            overflow,
        )

    def dispatch_fn(
        x_expanded,
        plan,
        operands,
    ):
        recv_origin_global_token_id = plan.recv_origin_global_token_id
        expert_begin_offset_per_ep = plan.expert_begin_offset_per_ep
        dest_ranks = plan.dest_ranks
        dest_offsets = plan.dest_offsets
        local_experts_start = plan.local_experts_start

        buffer = view_buffer(operands.raw)
        barrier_counter = buffer.barrier_counter
        dispatch_recv_buffer = buffer.dispatch_recv_buffer
        dispatch_recv_buffer_scaling_factors = (
            buffer.dispatch_recv_buffer_scaling_factors
        )
        dispatch_recv_weights = buffer.dispatch_recv_weights

        dispatch_out = torch.ops._flex_ep.router_dispatch(
            x_expanded,
            None,
            None,
            dest_ranks,
            dest_offsets,
            operands.buffers_cuda_ptrs,
            dispatch_recv_buffer,
            dispatch_recv_buffer_scaling_factors,
            recv_origin_global_token_id,
            dispatch_recv_weights,
            operands.offs_dispatch_recv_buffer,
            operands.offs_dispatch_recv_buffer_scaling_factors,
            operands.offs_dispatch_recv_weights,
            operands.offs_dispatch_recv_origin_global_token_id,
            operands.ep_rank,
            num_ctas,
            BATCH,
        )
        barrier = torch.ops._flex_ep.barrier_arrive(
            barrier_counter[:1],
            dispatch_out[0],
            2,
        )
        recv_x = _view_beginning_as(
            dispatch_recv_buffer,
            (max_recv_tokens, x_expanded.shape[-1]),
            x_expanded.dtype,
        )
        recv_x_u8 = torch.ops._flex_ep.zfill_ranges_inplace(
            recv_x.view(torch.uint8),
            expert_begin_offset_per_ep[:, -1],
            local_experts_start[1:],
            TOKEN_ALIGNMENT,
        )
        recv_x = recv_x_u8.view(x_expanded.dtype).view(recv_x.shape)
        recv_x_u8 = torch.ops._flex_ep.barrier_wait(
            recv_x_u8,
            operands.buffers_cuda_ptrs,
            operands.offs_barrier_counter,
            barrier,
            EP_TIMEOUT_SECONDS,
        )
        recv_x = recv_x_u8.view(x_expanded.dtype).view(recv_x.shape)
        return recv_x

    def combine_fn(
        y3,
        plan,
        operands,
    ):
        recv_origin_global_token_id = plan.recv_origin_global_token_id
        expert_begin_offset_per_ep = plan.expert_begin_offset_per_ep
        local_experts_start = plan.local_experts_start

        buffer = view_buffer(operands.raw)
        barrier_counter = buffer.barrier_counter
        combine_recv_buffer = buffer.combine_recv_buffer
        combine_recv_scale_factors = buffer.combine_recv_scale_factors
        combine_recv_weights = buffer.combine_recv_weights

        combine_recv_buffer = _router_barrier(
            combine_recv_buffer,
            barrier_counter,
            operands.buffers_cuda_ptrs,
            operands.offs_barrier_counter,
            nonce=0,
        )
        (
            combine_recv_buffer,
            _combine_recv_scale_factors,
            _combine_recv_weights,
        ) = torch.ops._flex_ep.router_combine(
            y3,
            None,
            None,
            expert_begin_offset_per_ep,
            local_experts_start[-1:].to(torch.int64),
            recv_origin_global_token_id,
            operands.buffers_cuda_ptrs,
            combine_recv_buffer,
            combine_recv_scale_factors,
            combine_recv_weights,
            operands.offs_combine_recv_buffer,
            operands.offs_combine_recv_scale_factors,
            operands.offs_combine_recv_weights,
            operands.ep_rank,
            BATCH,
            TOPK,
            num_ctas,
            BATCH,
        )
        combine_recv_buffer = _router_barrier(
            combine_recv_buffer,
            barrier_counter,
            operands.buffers_cuda_ptrs,
            operands.offs_barrier_counter,
            nonce=1,
            clone_result=True,
        )
        combined = _view_beginning_as(
            combine_recv_buffer,
            (BATCH, TOPK, y3.shape[-1]),
            y3.dtype,
        )
        return combined.sum(1)

    def combine_bwd_fn(
        dy,
        plan,
        operands,
    ):
        expert_begin_offset_per_ep = plan.expert_begin_offset_per_ep
        dest_ranks = plan.dest_ranks
        dest_offsets = plan.dest_offsets
        local_experts_start = plan.local_experts_start

        buffer = view_buffer(operands.raw)
        barrier_counter = buffer.barrier_counter
        dispatch_recv_buffer = buffer.dispatch_recv_buffer
        dispatch_recv_buffer_scaling_factors = (
            buffer.dispatch_recv_buffer_scaling_factors
        )
        dispatch_recv_weights = buffer.dispatch_recv_weights
        dispatch_recv_origin_global_token_id = (
            buffer.dispatch_recv_origin_global_token_id
        )

        dispatch_recv_buffer = _router_barrier(
            dispatch_recv_buffer,
            barrier_counter,
            operands.buffers_cuda_ptrs,
            operands.offs_barrier_counter,
            nonce=0,
        )
        grad_tokens = dy.unsqueeze(1).expand(BATCH, TOPK, dy.shape[-1]).contiguous()
        dispatch_out = torch.ops._flex_ep.router_dispatch(
            grad_tokens,
            None,
            None,
            dest_ranks,
            dest_offsets,
            operands.buffers_cuda_ptrs,
            dispatch_recv_buffer,
            dispatch_recv_buffer_scaling_factors,
            dispatch_recv_origin_global_token_id,
            dispatch_recv_weights,
            operands.offs_dispatch_recv_buffer,
            operands.offs_dispatch_recv_buffer_scaling_factors,
            operands.offs_dispatch_recv_weights,
            -1,
            operands.ep_rank,
            num_ctas,
            BATCH,
        )
        barrier = torch.ops._flex_ep.barrier_arrive(
            barrier_counter[:1],
            dispatch_out[0],
            1,
        )
        dy3 = _view_beginning_as(
            dispatch_recv_buffer,
            (max_recv_tokens, dy.shape[-1]),
            dy.dtype,
        )
        dy3_u8 = torch.ops._flex_ep.zfill_ranges_inplace(
            dy3.view(torch.uint8),
            expert_begin_offset_per_ep[:, -1],
            local_experts_start[1:],
            TOKEN_ALIGNMENT,
        )
        dy3 = dy3_u8.view(dy.dtype).view(dy3.shape)
        dy3_u8 = torch.ops._flex_ep.barrier_wait(
            dy3_u8,
            operands.buffers_cuda_ptrs,
            operands.offs_barrier_counter,
            barrier,
            EP_TIMEOUT_SECONDS,
        )
        dy3 = dy3_u8.view(dy.dtype).view(dy3.shape)
        return dy3

    def dispatch_bwd_fn(
        dx_recv,
        plan,
        operands,
    ):
        recv_origin_global_token_id = plan.recv_origin_global_token_id
        expert_begin_offset_per_ep = plan.expert_begin_offset_per_ep
        local_experts_start = plan.local_experts_start

        buffer = view_buffer(operands.raw)
        barrier_counter = buffer.barrier_counter
        combine_recv_buffer = buffer.combine_recv_buffer
        combine_recv_scale_factors = buffer.combine_recv_scale_factors
        combine_recv_weights = buffer.combine_recv_weights

        combine_recv_buffer = _router_barrier(
            combine_recv_buffer,
            barrier_counter,
            operands.buffers_cuda_ptrs,
            operands.offs_barrier_counter,
            nonce=0,
        )
        (
            combine_recv_buffer,
            _combine_recv_scale_factors,
            _combine_recv_weights,
        ) = torch.ops._flex_ep.router_combine(
            dx_recv,
            None,
            None,
            expert_begin_offset_per_ep,
            local_experts_start[-1:].to(torch.int64),
            recv_origin_global_token_id,
            operands.buffers_cuda_ptrs,
            combine_recv_buffer,
            combine_recv_scale_factors,
            combine_recv_weights,
            operands.offs_combine_recv_buffer,
            operands.offs_combine_recv_scale_factors,
            operands.offs_combine_recv_weights,
            operands.ep_rank,
            BATCH,
            TOPK,
            num_ctas,
            BATCH,
        )
        combine_recv_buffer = _router_barrier(
            combine_recv_buffer,
            barrier_counter,
            operands.buffers_cuda_ptrs,
            operands.offs_barrier_counter,
            nonce=1,
            clone_result=True,
        )
        return _view_beginning_as(
            combine_recv_buffer,
            (BATCH, TOPK, dx_recv.shape[-1]),
            dx_recv.dtype,
        )

    return (
        build_dispatch_plan_fn,
        dispatch_fn,
        combine_fn,
        combine_bwd_fn,
        dispatch_bwd_fn,
    )


def _reference(x, topk_idx, w13, w2, rank: int, world_size: int):
    gathered_x = [torch.empty_like(x) for _ in range(world_size)]
    gathered_topk_idx = [torch.empty_like(topk_idx) for _ in range(world_size)]
    gathered_w13 = [torch.empty_like(w13) for _ in range(world_size)]
    gathered_w2 = [torch.empty_like(w2) for _ in range(world_size)]
    dist.all_gather(gathered_x, x.detach())
    dist.all_gather(gathered_topk_idx, topk_idx)
    dist.all_gather(gathered_w13, w13.detach())
    dist.all_gather(gathered_w2, w2.detach())

    gathered_x[rank] = x
    gathered_w13[rank] = w13
    gathered_w2[rank] = w2

    local_output = None
    global_loss = x.new_zeros(())
    for src_rank in range(world_size):
        outputs = []
        for b in range(BATCH):
            token_out = x.new_zeros((HIDDEN_DIM,))
            for k in range(TOPK):
                expert = int(gathered_topk_idx[src_rank][b, k].item())
                owner_rank = expert // LOCAL_EXPERTS
                local_expert = expert % LOCAL_EXPERTS
                y1 = gathered_x[src_rank][b : b + 1] @ gathered_w13[owner_rank][
                    local_expert
                ].transpose(-2, -1)
                gate, up = y1.chunk(2, dim=-1)
                y2 = F.silu(gate) * up
                token_out = token_out + (
                    y2 @ gathered_w2[owner_rank][local_expert].transpose(-2, -1)
                ).squeeze(0)
            outputs.append(token_out)
        rank_output = torch.stack(outputs)
        global_loss = global_loss + rank_output.float().sum()
        if src_rank == rank:
            local_output = rank_output
    if local_output is None:
        raise AssertionError("missing local reference output")
    global_loss.backward()
    return local_output


class FlexEpNVSHMEMTest(MultiProcContinuousTest):
    world_size = 2

    @classmethod
    def backend_str(cls):
        return "nccl"

    def _init_device(self) -> None:
        torch.cuda.set_device(self.device)
        symm_mem.set_backend("NVSHMEM")

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _make_inputs(self, requires_grad: bool = False):
        torch.manual_seed(1234 + self.rank)
        x = torch.randn(
            BATCH,
            HIDDEN_DIM,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=requires_grad,
        )
        topk_idx = torch.arange(BATCH, device=self.device, dtype=torch.int64).view(
            BATCH,
            TOPK,
        )
        w13 = torch.randn(
            LOCAL_EXPERTS,
            2 * INTERMEDIATE_DIM,
            HIDDEN_DIM,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=requires_grad,
        )
        w2 = torch.randn(
            LOCAL_EXPERTS,
            HIDDEN_DIM,
            INTERMEDIATE_DIM,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=requires_grad,
        )
        return x, topk_idx, w13, w2

    def _make_router_operands(self):
        kwargs = {
            "B": BATCH,
            "D": HIDDEN_DIM,
            "EP_SIZE": self.world_size,
            "NUM_EXPERTS": LOCAL_EXPERTS * self.world_size,
            "TOPK": TOPK,
        }
        nvl_buf_sz = NvlSharedBuffer.get_buffer_size_bytes(**kwargs)
        raw_storage = symm_mem.empty(
            _align_up(nvl_buf_sz, torch.int64.itemsize) // torch.int64.itemsize,
            device=self.device,
            dtype=torch.int64,
        )
        raw = raw_storage.view(torch.uint8)[:nvl_buf_sz]
        raw.zero_()
        handle = symm_mem.rendezvous(raw_storage, dist.group.WORLD)
        buffer_storages = [
            handle.get_buffer(peer, raw_storage.shape, raw_storage.dtype)
            if peer != self.rank
            else raw_storage
            for peer in range(self.world_size)
        ]
        buffers = [buf.view(torch.uint8)[:nvl_buf_sz] for buf in buffer_storages]
        self._router_raw_storages = buffer_storages
        self._router_raw_buffers = buffers
        my_buffer = NvlSharedBuffer.view_from_buffer(
            buffers[self.rank],
            **kwargs,
        )
        my_buffer.barrier_counter.fill_(0)
        buffers_cuda_ptrs = torch.tensor(
            [buf.data_ptr() for buf in buffers],
            dtype=torch.int64,
            device=self.device,
        )
        dist.barrier()
        _router_barrier(
            my_buffer.barrier_counter,
            my_buffer.barrier_counter,
            buffers_cuda_ptrs,
            my_buffer.offset_of("barrier_counter"),
            nonce=0,
        )
        return RouterOperands(
            raw=raw,
            buffers_cuda_ptrs=buffers_cuda_ptrs,
            offs_barrier_counter=my_buffer.offset_of("barrier_counter"),
            offs_dispatch_recv_buffer=my_buffer.offset_of("dispatch_recv_buffer"),
            offs_dispatch_recv_buffer_scaling_factors=my_buffer.offset_of(
                "dispatch_recv_buffer_scaling_factors"
            ),
            offs_dispatch_recv_weights=my_buffer.offset_of("dispatch_recv_weights"),
            offs_dispatch_recv_origin_global_token_id=my_buffer.offset_of(
                "dispatch_recv_origin_global_token_id"
            ),
            offs_combine_recv_buffer=my_buffer.offset_of("combine_recv_buffer"),
            offs_combine_recv_scale_factors=my_buffer.offset_of(
                "combine_recv_scale_factors"
            ),
            offs_combine_recv_weights=my_buffer.offset_of("combine_recv_weights"),
            offs_allgather_expert_counts=my_buffer.offset_of(
                "allgather_expert_counts"
            ),
            ep_rank=self.rank,
        )

    def _flex_ep_call(self, x, topk_idx, w13, w2, router_operands):
        fns = _make_router_fns(
            num_experts=LOCAL_EXPERTS * self.world_size,
            ep_size=self.world_size,
        )
        return flex_ep(
            x,
            topk_idx,
            w13,
            w2,
            *fns,
            router_operands,
            num_experts=LOCAL_EXPERTS * self.world_size,
            ep_rank=self.rank,
            ep_size=self.world_size,
            max_tokens=BATCH,
            topk=TOPK,
        )

    def _run_with_grads(self, fn, args):
        y = fn(*args)
        y.float().sum().backward()
        return y, args[0].grad, args[2].grad, args[3].grad

    @requires_nvshmem()
    @requires_router_ep_backend_ops()
    @skip_if_lt_x_gpu(2)
    def test_eager_matches_reference(self):
        self._init_device()
        args = (*self._make_inputs(requires_grad=True), self._make_router_operands())
        actual = self._run_with_grads(self._flex_ep_call, args)

        ref_args = self._make_inputs(requires_grad=True)
        expected_y = _reference(*ref_args, self.rank, self.world_size)
        expected = (expected_y, ref_args[0].grad, ref_args[2].grad, ref_args[3].grad)

        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertEqual(actual_tensor, expected_tensor, rtol=1e-1, atol=1.0)

    @requires_nvshmem()
    @requires_router_ep_backend_ops()
    @skip_if_lt_x_gpu(2)
    def test_inductor_issues_router_ep_kernels(self):
        if not HAS_GPU or not has_triton():
            self.skipTest("flex_ep NVSHMEM Inductor test requires CUDA and Triton")

        self._init_device()
        eager_args = (
            *self._make_inputs(requires_grad=True),
            self._make_router_operands(),
        )
        eager = self._run_with_grads(self._flex_ep_call, eager_args)
        dist.barrier()

        compiled_args = (
            *self._make_inputs(requires_grad=True),
            self._make_router_operands(),
        )
        compiled_fn = torch.compile(
            self._flex_ep_call,
            backend="inductor",
            fullgraph=True,
        )
        compiled_y, codes = run_fw_bw_and_get_code(lambda: compiled_fn(*compiled_args))
        compiled = (
            compiled_y,
            compiled_args[0].grad,
            compiled_args[2].grad,
            compiled_args[3].grad,
        )

        for name, actual_tensor, expected_tensor in zip(
            ("y", "dx", "dw13", "dw2"),
            compiled,
            eager,
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    actual_tensor,
                    expected_tensor,
                    msg=f"{name} mismatch",
                    rtol=1e-1,
                    atol=1.0,
                )

        generated_code = "\n".join(codes)
        self.assertIn("_flex_ep.router_dispatch.default", generated_code)
        self.assertIn("_flex_ep.router_combine.default", generated_code)
        self.assertNotIn("all_to_all_vdev_2d.default", generated_code)


if __name__ == "__main__":
    run_tests()
