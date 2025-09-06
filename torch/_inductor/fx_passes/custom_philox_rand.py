from typing import Sequence, Optional
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton, register_fake, register_kernel

BLOCK  = 256

@triton.jit
def _pick_lane(u0, u1, u2, u3, lane):
    v = tl.where(lane == 0, u0, u1)
    v = tl.where(lane == 1, u1, v)
    v = tl.where(lane == 2, u2, v)
    v = tl.where(lane == 3, u3, v)
    return v

@triton.jit
def _philox_fill_uniform_gridstride(
    out_ptr, n_elements, seed, offset_blocks, lane_shift,
    threads_per_round,  # = BLOCK * grid_x
    BLOCK: tl.constexpr = BLOCK,
):
    UNROLL = 4
    pid = tl.program_id(0)                       # [0, grid_x)
    tid = pid * BLOCK + tl.arange(0, BLOCK)      # [0, BLOCK*grid_x)
    inv  = 1.0 / 4294967296.0
    half = inv * 0.5

    # rounds_total = ceil(n / (threads_per_round * UNROLL))
    rounds_total = (n_elements + threads_per_round * UNROLL - 1) // (threads_per_round * UNROLL)

    for r in range(rounds_total):
        subseq = (tid).to(tl.uint64)
        lane = ((tid + lane_shift) % 4).to(tl.uint64)

        offblk = tl.full(subseq.shape, (offset_blocks + r), tl.uint64) 
        u0, u1, u2, u3 = tl.philox(
            seed,
            (offblk & 0xFFFFFFFF).to(tl.uint32),
            ((offblk >> 32) & 0xFFFFFFFF).to(tl.uint32),
            (subseq & 0xFFFFFFFF).to(tl.uint32),
            ((subseq >> 32) & 0xFFFFFFFF).to(tl.uint32),
        )
    
        inv  = 1.0 / 4294967296.0  # 2^-32
        half = inv * 0.5

        base   = tid * 4
        stride = threads_per_round
        
        # k=0
        i0 = base + (r * UNROLL) * stride
        m0 = i0 < n_elements
        lane0 = tl.full(tid.shape, (lane_shift + 0) % 4, tl.uint32)
        f0 = _pick_lane(u0, u1, u2, u3, lane0).to(tl.float32) * inv + half
        tl.store(out_ptr + i0, 1.0 - f0, mask=m0)

        # k=1
        i1 = base + 1 + (r * UNROLL) * stride
        m1 = i1 < n_elements
        lane1 = tl.full(tid.shape, (lane_shift + 1) % 4, tl.uint32)
        f1 = _pick_lane(u0, u1, u2, u3, lane1).to(tl.float32) * inv + half
        tl.store(out_ptr + i1, 1.0 - f1, mask=m1)

        # k=2
        i2 = base + 2 + (r * UNROLL) * stride
        m2 = i2 < n_elements
        lane2 = tl.full(tid.shape, (lane_shift + 2) % 4, tl.uint32)
        f2 = _pick_lane(u0, u1, u2, u3, lane2).to(tl.float32) * inv + half
        tl.store(out_ptr + i2, 1.0 - f2, mask=m2)

        # k=3
        i3 = base + 3 + (r * UNROLL) * stride
        m3 = i3 < n_elements
        lane3 = tl.full(tid.shape, (lane_shift + 3) % 4, tl.uint32)
        f3 = _pick_lane(u0, u1, u2, u3, lane3).to(tl.float32) * inv + half
        tl.store(out_ptr + i3, 1.0 - f3, mask=m3)

# ---- host helpers ----
def _compute_grid_x(nelem: int, block: int, device_index: int) -> int:
    prop = torch.cuda.get_device_properties(device_index)
    blocks_per_sm = prop.max_threads_per_multi_processor // block
    max_blocks = prop.multi_processor_count * blocks_per_sm
    need_blocks = (nelem + block - 1) // block
    return min(max_blocks, need_blocks)

def _reserve_seed_and_offset_gridstride(x_device_index: int, nelem: int, block: int):
    UNROLL = 4
    gen = torch.cuda.default_generators[x_device_index]
    seed = int(gen.initial_seed())
    grid_x = _compute_grid_x(nelem, block, x_device_index)
    rounds_per_thread = (nelem + (block * grid_x * UNROLL) - 1) // (block * grid_x * UNROLL)
    counter_offset_per_thread = rounds_per_thread * UNROLL
    used_32 = counter_offset_per_thread #* block * grid_x
    old_off = int(gen.get_offset())
    gen.set_offset(old_off + used_32)  
    return seed, (old_off // 4), (old_off % 4), grid_x

@triton_op("my_triton_op::philox_rand", mutates_args={})
def philox_rand(
    shape: Sequence[int], #*,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    #seed: Optional[int] = None,
    #start_offset: Optional[int] = None,
) -> torch.Tensor:
    #raise NotImplementedError("custom::philox_rand has no eager body; kernels are registered via register_kernel().")
    out = torch.empty(tuple(int(x) for x in shape), dtype=dtype, device=device)
    n = out.numel()
    if n == 0:
        return out

    print(tuple(int(x) for x in shape))
    dev_idx = out.device.index or 0
    # sync to CUDAGenerator
    seed_val, offset_blocks, lane_shift, grid_x = _reserve_seed_and_offset_gridstride(dev_idx, n, BLOCK)

    buf = out if out.dtype == torch.float32 else torch.empty_like(out, dtype=torch.float32)

    grid = lambda meta: (grid_x,)
    wrap_triton(_philox_fill_uniform_gridstride)[grid](
        buf, n, seed_val, offset_blocks, lane_shift,
        threads_per_round=BLOCK * grid_x,
        BLOCK=BLOCK,
    )
    if buf is not out:
        out.copy_(buf.to(out.dtype))
    return out

