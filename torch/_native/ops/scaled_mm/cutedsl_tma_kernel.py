"""TMA-staged CUDA-core MXFP8 GEMV for native ``scaled_mm``."""

import hashlib
import operator
from functools import cache
from pathlib import Path

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cuda.bindings import driver as cuda  # pyrefly: ignore[missing-import]
from cutlass.cute.nvgpu import cpasync, tcgen05

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache


TMA_PRODUCER_WARP = 0
_SUPPORTED_CAPABILITIES = {(10, 0), (10, 3)}
_SOURCE_FINGERPRINT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_CONFIGS = {
    (4608, 8192): (4, 2),
    (8192, 2048): (8, 2),
    (4096, 8192): (4, 2),
    (8192, 4096): (8, 3),
}


@cute.jit
def combined_e8m0_to_f32(a, b):
    """Combine two E8M0 exponents before converting to float32."""
    exponent = cutlass.Int32(a) + cutlass.Int32(b) - 254
    bits = cutlass.Uint32(0)
    if exponent >= -126:
        bits = cutlass.Uint32(exponent + 127) << 23
        if exponent > 127:
            bits = cutlass.Uint32(0x7F800000)
    else:
        bits = cutlass.Uint32(0)
        if exponent >= -149:
            bits = cutlass.Uint32(1) << cutlass.Uint32(exponent + 149)
    if a == 0xFF or b == 0xFF:
        bits = cutlass.Uint32(0x7F800001)
    return bits.bitcast(cutlass.Float32)


@cute.jit
def blocked_scale_offset(row, col, col_blocks: cutlass.Constexpr):
    """Map a logical scale coordinate to SWIZZLE_32_4_4 storage."""
    row_block = row // 128
    row_in_block = row % 128
    return (
        (((row_block * col_blocks + col // 4) * 32 + row_in_block % 32) * 4)
        + row_in_block // 32
    ) * 4 + col % 4


class Mxfp8GemmSmallMTma:
    """Compute a raw-layout MXFP8 GEMM with one consumer warp per M row."""

    def __init__(self, m: int, n: int, k: int, block_n: int, num_stages: int):
        if not 1 <= m <= 8:
            raise ValueError("m must be between 1 and 8")
        if k % 1024 != 0:
            raise ValueError("k must be divisible by 1024")
        sf_k = k // 32
        if n % block_n != 0 or block_n > 32:
            raise ValueError("n must be divisible by block_n and block_n <= 32")
        if num_stages not in (2, 3):
            raise ValueError("num_stages must be 2 or 3")
        if sf_k < num_stages * 32:
            raise ValueError("TMA staging requires at least one K tile per stage")
        self.m = m
        self.n = n
        self.sf_k = sf_k
        self.block_n = block_n
        self.num_stages = num_stages
        self.tile_k_u32 = 256
        self.num_k_tiles = sf_k // 32

    def x_smem_layout(self):
        """Return the staged layout for M 1024-byte input tiles."""
        return cute.make_ordered_layout(
            (self.m, self.tile_k_u32, self.num_stages), order=(1, 0, 2)
        )

    def w_smem_layout(self):
        """Return the staged layout for one weight-row tile."""
        return cute.make_ordered_layout(
            (self.block_n, self.tile_k_u32, self.num_stages), order=(1, 0, 2)
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_w: cute.CopyAtom,
        mW_u32: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mX_u32: cute.Tensor,
        mSFW: cute.Tensor,
        mSFX: cute.Tensor,
        mO: cute.Tensor,
    ):
        """Overlap TMA production with typed FP8 conversion and accumulation."""
        tidx, _, _ = cute.arch.thread_idx()
        pid_n, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        lane = tidx % 32
        n0 = pid_n * self.block_n
        chunk_layout = cute.make_ordered_layout((1, 4), order=(1, 0))
        col_a = cute.assume(lane * 8 + (lane >> 2) % 2 * 4, divby=4)
        col_b = cute.assume(lane * 8 + ((lane >> 2) + 1) % 2 * 4, divby=4)
        scale_layout = cute.make_layout(1)
        smem_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Uint32,
            num_bits_per_copy=128,
        )
        input_scale_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float8E8M0FNU,
            num_bits_per_copy=8,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_LAST,
        )
        weight_scale_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float8E8M0FNU,
            num_bits_per_copy=8,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
        )

        smem = cutlass.utils.SmemAllocator()
        barriers = smem.allocate_array(
            cutlass.Int64, self.num_stages * 2, byte_alignment=8
        )
        sX = smem.allocate_tensor(
            cutlass.Uint32, self.x_smem_layout(), byte_alignment=128
        )
        sW = smem.allocate_tensor(
            cutlass.Uint32, self.w_smem_layout(), byte_alignment=128
        )

        if warp == TMA_PRODUCER_WARP:
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_x)
        producer, consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.m),
            tx_count=(self.block_n + self.m) * self.tile_k_u32 * 4,
            barrier_storage=barriers,
            tidx=lane,
        ).make_participants()

        gW = cute.local_tile(
            mW_u32,
            (self.block_n, self.tile_k_u32),
            (pid_n, None),
        )
        gX = cute.local_tile(mX_u32, (self.m, self.tile_k_u32), (0, None))
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            cute.make_layout(1),
            cute.group_modes(sW, 0, 2),
            cute.group_modes(gW, 0, 2),
        )
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom_x,
            0,
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(gX, 0, 2),
        )

        if warp == TMA_PRODUCER_WARP:
            stage = producer.acquire_and_advance()
            cute.copy(
                tma_atom_w,
                tWgW[(None, stage.count)],
                tWsW[(None, stage.index)],
                tma_bar_ptr=stage.barrier,
            )
            cute.copy(
                tma_atom_x,
                tXgX[(None, stage.count)],
                tXsX[(None, stage.index)],
                tma_bar_ptr=stage.barrier,
            )
            stage.commit()

        acc = [cutlass.Float32(0.0) for _ in range(self.block_n)]
        for k_tile in cutlass.range_constexpr(self.num_k_tiles):
            if warp == TMA_PRODUCER_WARP and k_tile < self.num_k_tiles - 1:
                stage = producer.acquire_and_advance()
                cute.copy(
                    tma_atom_w,
                    tWgW[(None, stage.count)],
                    tWsW[(None, stage.index)],
                    tma_bar_ptr=stage.barrier,
                )
                cute.copy(
                    tma_atom_x,
                    tXgX[(None, stage.count)],
                    tXsX[(None, stage.index)],
                    tma_bar_ptr=stage.barrier,
                )
                stage.commit()

            full = consumer.wait_and_advance()
            scale_k = lane + k_tile * 32
            x_frag = cute.make_rmem_tensor((1, 8), cutlass.Uint32)
            cute.copy(
                smem_atom,
                cute.make_tensor(
                    sX.iterator
                    + cute.assume(sX.layout((warp, col_a, full.index)), divby=4),
                    chunk_layout,
                ),
                cute.make_tensor(x_frag.iterator, chunk_layout),
            )
            cute.copy(
                smem_atom,
                cute.make_tensor(
                    sX.iterator
                    + cute.assume(sX.layout((warp, col_b, full.index)), divby=4),
                    chunk_layout,
                ),
                cute.make_tensor(x_frag.iterator + 4, chunk_layout),
            )
            x_values = (
                cute.recast_tensor(x_frag, cutlass.Float8E4M3FN)
                .load()
                .to(cutlass.Float32)
                .reshape((2, 16))
            )
            input_scale = cute.make_rmem_tensor(1, cutlass.Float8E8M0FNU)
            cute.copy(
                input_scale_atom,
                cute.make_tensor(
                    mSFX.iterator + blocked_scale_offset(warp, scale_k, self.sf_k // 4),
                    scale_layout,
                ),
                input_scale,
            )
            input_scale_u8 = cute.recast_tensor(input_scale, cutlass.Uint8)[0]

            for row in cutlass.range_constexpr(self.block_n):
                w_frag = cute.make_rmem_tensor((1, 8), cutlass.Uint32)
                cute.copy(
                    smem_atom,
                    cute.make_tensor(
                        sW.iterator
                        + cute.assume(sW.layout((row, col_a, full.index)), divby=4),
                        chunk_layout,
                    ),
                    cute.make_tensor(w_frag.iterator, chunk_layout),
                )
                cute.copy(
                    smem_atom,
                    cute.make_tensor(
                        sW.iterator
                        + cute.assume(sW.layout((row, col_b, full.index)), divby=4),
                        chunk_layout,
                    ),
                    cute.make_tensor(w_frag.iterator + 4, chunk_layout),
                )
                w_values = (
                    cute.recast_tensor(w_frag, cutlass.Float8E4M3FN)
                    .load()
                    .to(cutlass.Float32)
                    .reshape((2, 16))
                )
                weight_scale = cute.make_rmem_tensor(1, cutlass.Float8E8M0FNU)
                cute.copy(
                    weight_scale_atom,
                    cute.make_tensor(
                        mSFW.iterator
                        + blocked_scale_offset(n0 + row, scale_k, self.sf_k // 4),
                        scale_layout,
                    ),
                    weight_scale,
                )
                product = (x_values * w_values).reduce(
                    cute.ReductionOp.ADD, cutlass.Float32(0.0), (None, 1)
                )
                weight_scale_u8 = cute.recast_tensor(weight_scale, cutlass.Uint8)[0]
                acc[row] += (product[0] + product[1]) * combined_e8m0_to_f32(
                    input_scale_u8, weight_scale_u8
                )
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            full.release()

        if warp == TMA_PRODUCER_WARP:
            producer.tail()
        for row in cutlass.range_constexpr(self.block_n):
            acc[row] = cute.arch.warp_reduction(acc[row], operator.add)
        if lane == 0:
            for row in cutlass.range_constexpr(self.block_n):
                mO[warp, n0 + row] = acc[row].to(cutlass.BFloat16)

    @cute.jit
    def __call__(
        self,
        mW: cute.Tensor,
        mX: cute.Tensor,
        mSFW: cute.Tensor,
        mSFX: cute.Tensor,
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Construct TMA descriptors and launch the kernel."""
        mW_u32 = cute.recast_tensor(mW, cutlass.Uint32)
        mX_u32 = cute.recast_tensor(mX, cutlass.Uint32)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_op,
            mW_u32,
            cute.select(self.w_smem_layout(), mode=[0, 1]),
            (self.block_n, self.tile_k_u32),
        )
        tma_atom_x, tma_tensor_x = cpasync.make_tiled_tma_atom(
            tma_op,
            mX_u32,
            cute.select(self.x_smem_layout(), mode=[0, 1]),
            (self.m, self.tile_k_u32),
        )
        name = f"mxfp8_gemm_tma_m{self.m}_bn{self.block_n}_s{self.num_stages}"
        self.kernel(
            tma_atom_w,
            tma_tensor_w,
            tma_atom_x,
            tma_tensor_x,
            mSFW,
            mSFX,
            mO,
            _name_prefix=name,
        ).launch(
            grid=[self.n // self.block_n, 1, 1],
            block=[self.m * 32, 1, 1],
            stream=stream,
        )


def _blocked_scale_numel(rows: int, k: int) -> int:
    return ((rows + 127) // 128) * 128 * (((k // 32) + 3) // 4) * 4


def _make_fake_tensor(dtype, shape: tuple[int, ...], assumed_align: int):
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=assumed_align,
    )


@cache
def _device_compile_properties(device: int) -> tuple[tuple[int, int], int]:
    with torch.cuda.device(device):
        return (
            torch.cuda.get_device_capability(device),
            cutlass.utils.HardwareInfo().get_max_active_clusters(1),
        )


def _config_for(n: int, k: int) -> tuple[int, int]:
    return _CONFIGS.get((n, k), (4, 2))


@instrumented_cutedsl_cache(
    "aten::_scaled_mm_v2",
    key_fn=lambda device, capability, max_clusters, n, k, block_n, stages, source: (
        f"mxfp8_tma_m1 device={device} capability={capability} "
        f"clusters={max_clusters} N={n} K={k} BN={block_n} S={stages} "
        f"source={source[:12]}"
    ),
)
def _compile_mxfp8_tma_m1(
    device: int,
    capability: tuple[int, int],
    max_active_clusters: int,
    n: int,
    k: int,
    block_n: int,
    num_stages: int,
    source_fingerprint: str,
):
    """Compile one hardware- and shape-specialized M=1 TMA kernel."""
    with torch.cuda.device(device):
        op = Mxfp8GemmSmallMTma(1, n, k, block_n, num_stages)
        return cute.compile(
            op,
            _make_fake_tensor(cutlass.Float8E4M3FN, (n, k), 16),
            _make_fake_tensor(cutlass.Float8E4M3FN, (1, k), 16),
            _make_fake_tensor(cutlass.Float8E8M0FNU, (_blocked_scale_numel(n, k),), 32),
            _make_fake_tensor(cutlass.Float8E8M0FNU, (_blocked_scale_numel(1, k),), 32),
            _make_fake_tensor(cutlass.BFloat16, (1, n), 16),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
            _name_prefix=(f"mxfp8_tma_m1_n{n}_k{k}_bn{block_n}_s{num_stages}"),
        )


def mxfp8_tma_m1_scaled_mm(
    q_input: torch.Tensor,
    weight_t: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write one-row MXFP8 scaled-mm output using the staged TMA kernel."""
    n = weight_t.shape[1]
    k = q_input.shape[1]
    block_n, num_stages = _config_for(n, k)
    device = q_input.device.index
    if device is None:
        device = torch.cuda.current_device()

    def launch() -> None:
        capability, max_active_clusters = _device_compile_properties(device)
        if capability not in _SUPPORTED_CAPABILITIES:
            raise RuntimeError(
                f"M=1 MXFP8 TMA scaled_mm requires SM100 or SM103, got {capability}"
            )
        tensors = (
            (q_input, 16, "q_input"),
            (weight_t, 16, "weight"),
            (input_scale, 32, "input scale"),
            (weight_scale, 32, "weight scale"),
            (output, 16, "output"),
        )
        for tensor, alignment, name in tensors:
            if tensor.data_ptr() % alignment:
                raise RuntimeError(f"{name} must be {alignment}-byte aligned")
        _compile_mxfp8_tma_m1(
            device,
            capability,
            max_active_clusters,
            n,
            k,
            block_n,
            num_stages,
            _SOURCE_FINGERPRINT,
        )(
            weight_t.T,
            q_input,
            weight_scale,
            input_scale,
            output,
        )

    if device == torch.cuda.current_device():
        launch()
    else:
        with torch.cuda.device(device):
            launch()
    return output
