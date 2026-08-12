"""Host-side RNG plumbing shared by the distribution overrides.

These ops are NULLARY (no input tensor: the output's shape is given, and every
element's value comes from the RNG), so they build on the same nin == 0 kernel
plumbing as fill_ / arange. What they add is generator STATE.

Bit-exactness with aten is the whole difficulty. aten's distributions run through
``distribution_nullary_kernel`` (ATen/native/cuda/DistributionTemplates.h), and matching
its output means matching four things exactly:

  * the LAUNCH GEOMETRY, because the grid size feeds the grid-stride loop and therefore
    which philox counter each element draws from (see ``launch_plan``);
  * the philox COUNTER MAPPING that curand_init(seed, subsequence, offset) sets up:
    subsequence = the thread's flat index -> counter words z/w, offset/4 -> words x/y
    (curand_philox4x32_x.h: skipahead_sequence increments ctr.z/w, skipahead ctr.x/y and
    puts offset & 3 in STATE -- aten's offsets are multiples of 4, so STATE is always 0);
  * the ELEMENT MAPPING, which is strided rather than contiguous: iteration `it` of the
    grid-stride loop writes elements `idx + it*stride + ii*blockDim*gridDim` for
    ii in [0, unroll), NOT four contiguous elements (that is dropout's layout, not this
    one);
  * the RESERVATION, counter_offset, which is how far the generator's offset advances --
    get this wrong and every SUBSEQUENT random call diverges even if this one matches.

RNG state itself comes from ``Generator.philox_state(increment)``, mirroring the C++
PhiloxCudaState protocol: outside graph capture (HostState) seed/offset are CPU tensors
whose values the kernel takes as scalar arguments; during capture (DevState) they are CUDA
tensors aliasing the generator's extragraph state, refilled by replay_prologue on every
replay, so the kernel must LOAD them from device memory instead of baking them.
"""

import functools
import struct

import torch


# aten's block_size_bound for the distribution kernels.
_BLOCK = 256

# Grid-stride iterations emitted as a compile-time block by the hybrid loop (see
# cutedsl_kernels.kernel). Exposed as a KNOB, not a hard constant: the best depth is a
# register-pressure/ILP tradeoff that depends on the kernel body and the arch, and the value
# below is a B200 measurement (see _choose_iter_unroll for the data). Kernel count stays
# O(kind x dtype x capture x unroll), so the autotuner may sweep it without the count
# becoming shape-dependent.
_ITER_UNROLL = 4
_ITER_UNROLL_CANDIDATES = (1, 2, 4, 8)


def choose_iter_unroll(hw=None) -> int:
    """Unroll depth for the hybrid grid-stride loop.

    B200, fp32 normal_ (cold compile, then us at 2^24 / 2^26 / 2^28):
        1 -> 0.41s  --    --    (fully dynamic; no ILP across iterations)
        2 -> 0.50s  --    --
        4 -> 0.67s  38.8  127.4  477.8   <- default
        8 -> 0.97s  39.3  127.4  477.6   (ties, +45% compile)
       16 -> 1.58s  39.5  142.5  562.5   (slower; register pressure)
       32 -> 2.98s  39.2  149.6  642.9

    Each iteration keeps ~14 values live (10 u32 of philox state + 4 results), so the useful
    depth is bounded by the register file rather than by the tensor size. A GPU with a
    different budget per thread may prefer another rung, which is why this is a function of
    `hw` rather than a literal -- it just has no non-B200 data to key on yet.
    """
    return _ITER_UNROLL


@functools.cache
def _device_launch_caps(device_index: int) -> tuple[int, int]:
    props = torch.cuda.get_device_properties(device_index)
    return props.multi_processor_count, props.max_threads_per_multi_processor // _BLOCK


def launch_plan(n: int, device_index: int, unroll: int) -> tuple[int, int, int]:
    """Mirror of aten's calc_execution_policy.

    Returns (grid, counter_offset, num_iters). `unroll` is aten's unroll_factor =
    sizeof(dist return)/sizeof(accscalar_t): 4 for a float32 uniform4/normal4, 2 for
    float64. counter_offset is the reservation handed to philox_state, and each
    grid-stride iteration consumes exactly one curand4 call.
    """
    sm_count, blocks_per_sm = _device_launch_caps(device_index)
    grid = min((n + _BLOCK - 1) // _BLOCK, sm_count * blocks_per_sm)
    # max_generator_offsets_per_curand_call is 4 regardless of unroll_factor.
    num_iters = (n - 1) // (_BLOCK * grid * unroll) + 1
    return grid, num_iters * 4, num_iters


def f32(x: float) -> float:
    """Round a python float through float32, so the value the kernel receives is the one
    aten's accscalar_t arithmetic would use (returned as an exactly-representable
    double)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def philox_args(device_index: int, counter_offset: int):
    """Reserve counter_offset from the device's default generator.

    Returns (seed_t, offset_t, intragraph). seed_t/offset_t are CPU tensors in eager
    (HostState) and CUDA tensors under capture (DevState); callers branch on
    seed_t.is_cuda. The kernel-visible offset is offset_t + intragraph.
    """
    gen = torch.cuda.default_generators[device_index]
    seed_t, offset_t, intra_t = gen.philox_state(counter_offset)
    return seed_t, offset_t, intra_t.item()


@functools.cache
def state_dummy(device_index: int) -> torch.Tensor:
    """A never-read placeholder for the device-state slots in eager mode: the compiled
    signature always has them, but HostState passes its values as scalars instead."""
    return torch.zeros(1, dtype=torch.int64, device=f"cuda:{device_index}")
