"""Shared host-side logic for the fused dropout overrides.

Both DSL kernels replicate aten's ``fused_dropout_kernel_vec`` (VEC=4 path,
``aten/src/ATen/native/cuda/Dropout.cu``) exactly: same launch geometry, same
philox counter mapping (counter=(offset/4 + iter), subsequence=global thread
id), same curand uniform transform and comparison, so results and generator
offset advancement are bit-identical to aten.

RNG state comes from ``Generator.philox_state(increment)``, which mirrors the
C++ PhiloxCudaState protocol: outside capture (HostState) seed/offset are CPU
tensors and the kernels take their values as scalar arguments; during capture
(DevState) they are CUDA tensors aliasing the generator's extragraph state
that ``replay_prologue`` refills on every replay, and the kernels load them
from device memory at run time.
"""

import functools
import struct

import torch


_BLOCK = 256
_UNROLL = 4


@functools.cache
def _device_launch_caps(device_index: int) -> tuple[int, int]:
    props = torch.cuda.get_device_properties(device_index)
    blocks_per_sm = props.max_threads_per_multi_processor // _BLOCK
    return props.multi_processor_count, blocks_per_sm


def launch_plan(n: int, device_index: int) -> tuple[int, int, int]:
    """Mirror of aten's dropout_cuda launch math.

    Returns (grid, counter_offset, num_iters). counter_offset is the philox
    reservation passed to ``philox_state``; num_iters is the per-thread
    curand_uniform4 call count (== counter_offset / 4).
    """
    sm_count, blocks_per_sm = _device_launch_caps(device_index)
    grid = min((n + _BLOCK - 1) // _BLOCK, sm_count * blocks_per_sm)
    counter_offset = ((n - 1) // (_BLOCK * grid * _UNROLL) + 1) * _UNROLL
    return grid, counter_offset, counter_offset // _UNROLL


def keep_prob_and_scale(p: float) -> tuple[float, float]:
    """aten computes p1m = 1.-p in double, casts to accscalar_t (float for
    fp32 inputs), and the kernel computes scale = 1/p1m in float. Reproduce
    the exact float32 values (returned as exactly-representable doubles)."""
    p_keep = struct.unpack("f", struct.pack("f", 1.0 - p))[0]
    scale = struct.unpack("f", struct.pack("f", 1.0 / p_keep if p_keep else 0.0))[0]
    return p_keep, scale


def eligible(x: torch.Tensor, p: float, train) -> bool:
    # Fire only where aten would take the fp32 VEC=4 vectorized path with the
    # philox mapping we replicate; everything else falls through to aten.
    # aten shortcuts train=False and p==1 before touching the generator
    # (train=None counts as training).
    if (train is not None and not train) or p == 1.0:
        return False
    if not x.is_cuda or x.dtype != torch.float32 or x.numel() == 0:
        return False
    if not x.is_contiguous():
        return False
    if x.numel() % _UNROLL != 0 or x.data_ptr() % 16 != 0:
        return False
    return True


def philox_args(x: torch.Tensor, counter_offset: int):
    """Reserve counter_offset from the default generator of x's device and
    return (seed_t, offset_t, intragraph [python int]). seed_t/offset_t are
    CPU tensors in eager (HostState) and CUDA tensors under capture
    (DevState); callers branch on seed_t.is_cuda."""
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    gen = torch.cuda.default_generators[device_index]
    seed_t, offset_t, intra_t = gen.philox_state(counter_offset)
    return seed_t, offset_t, intra_t.item()
