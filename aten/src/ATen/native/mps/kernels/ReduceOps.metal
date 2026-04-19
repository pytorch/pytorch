#include <ATen/native/mps/kernels/ReduceOps.h>
#include <c10/metal/atomic.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

struct norm_abs_functor {
  template <typename T, enable_if_t<!is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    return static_cast<T>(::precise::abs(x));
  }

  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline float operator()(const T x) {
    const auto abs_2 = ::precise::abs(float2(x));
    return c10::metal::hypot(abs_2.x, abs_2.y);
  }
};

// `reduction_idx` is the index of a particular batch of input elements that all
// get reduced to one output element. `reduction_element_idx` is the index of
// just one input element within its batch.
static uint32_t get_input_offset(
    uint32_t reduction_element_idx,
    uint32_t reduction_idx,
    constant NormParams<>& params) {
  uint32_t input_offset = 0;

  for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
    auto input_dim_size = params.input_sizes[dim];
    auto output_dim_size = params.output_sizes[dim];

    // If the the input and output have the same size for this dim, then this
    // dim is not being reduced, so we index by `reduction_idx`
    if (input_dim_size == output_dim_size) {
      auto index_in_dim = reduction_idx % input_dim_size;
      reduction_idx /= input_dim_size;
      input_offset += index_in_dim * params.input_strides[dim];

      // Otherwise, this dim is being reduced, so we index by
      // `reduction_element_idx`
    } else {
      auto index_in_dim = reduction_element_idx % input_dim_size;
      reduction_element_idx /= input_dim_size;
      input_offset += index_in_dim * params.input_strides[dim];
    }
  }
  return input_offset;
}

// In this kernel, each threadgroup is responsible for calculating one element
// of the output.
// TI - dtype of the input tensor.
// TO - dtype of the output tensor.
template <typename TI, typename TO>
kernel void norm(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant NormParams<>& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simdgroup_size [[threads_per_simdgroup]]) {
  using TA = opmath_t<TO>;
  TA output_val = 0;
  const auto p = static_cast<TA>(params.p);

  if (p == INFINITY) {
    output_val = -INFINITY;
  } else if (p == -INFINITY) {
    output_val = INFINITY;
  }

  // First, all the input elements assigned to the threadgroup are divided
  // between all the threads in the threadgroup, and each thread reduces those
  // elements down to one partial `output_val`.
  for (uint32_t reduction_element_idx = tid;
       reduction_element_idx < params.reduction_size;
       reduction_element_idx += tptg) {
    auto input_elem =
        input[get_input_offset(reduction_element_idx, tgid, params)];
    auto input_abs = static_cast<TA>(norm_abs_functor()(input_elem));

    if (p == INFINITY) {
      output_val = max(input_abs, output_val);

    } else if (p == -INFINITY) {
      output_val = min(input_abs, output_val);

    } else if (p == 0) {
      output_val += (input_abs == 0) ? 0 : 1;

    } else {
      output_val += static_cast<TA>(::precise::pow(input_abs, p));
    }
  }

  // Next, all the threads in a threadgroup reduce their `output_val`s together
  // with a series of SIMD group reductions.
  auto threads_remaining = tptg;
  threadgroup TA shared_outputs[MAX_THREADGROUP_SIZE];

  while (threads_remaining > 1) {
    if (p == INFINITY) {
      output_val = simd_max(output_val);
    } else if (p == -INFINITY) {
      output_val = simd_min(output_val);
    } else {
      output_val = simd_sum(output_val);
    }

    threads_remaining = ceil_div(threads_remaining, simdgroup_size);

    if (threads_remaining > 1) {
      // One thread from each SIMD group writes to a shared buffer
      if (simd_lane_id == 0) {
        shared_outputs[simdgroup_id] = output_val;
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // The remaining threads each read one of the partial outputs from the
      // shared buffer
      if (tid < threads_remaining) {
        output_val = shared_outputs[tid];
      } else {
        return;
      }
    }
  }

  // Finally, one thread in the threadgroup writes the final output
  if (tid == 0) {
    uint32_t output_offset = 0;
    uint32_t reduction_idx = tgid;

    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      auto output_dim_size = params.output_sizes[dim];

      if (output_dim_size > 1) {
        auto index_in_dim = reduction_idx % output_dim_size;
        reduction_idx /= output_dim_size;
        output_offset += index_in_dim * params.output_strides[dim];
      }
    }

    if (p != 0 && p != 1 && p != INFINITY && p != -INFINITY) {
      output_val = static_cast<TA>(::precise::pow(output_val, 1 / p));
    }
    output[output_offset] = static_cast<TO>(output_val);
  }
}

#define REGISTER_NORM(TI, TO)                               \
  template [[host_name("norm_" #TI "_" #TO)]]               \
  kernel void norm<TI, TO>(                                 \
      constant TI * input [[buffer(0)]],                    \
      device TO * output [[buffer(1)]],                     \
      constant NormParams<> & params [[buffer(2)]],         \
      uint tid [[thread_position_in_threadgroup]],          \
      uint tptg [[threads_per_threadgroup]],                \
      uint tgid [[threadgroup_position_in_grid]],           \
      uint simd_lane_id [[thread_index_in_simdgroup]],      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]], \
      uint simdgroup_size [[threads_per_simdgroup]]);

REGISTER_NORM(float, float);
REGISTER_NORM(half, half);
REGISTER_NORM(bfloat, bfloat);
REGISTER_NORM(float2, float);
REGISTER_NORM(half2, half);

#include <c10/metal/reduction_utils.h>

// Load helper: cast to opmath_t, optionally replacing NaN with zero.
template <
    bool NAN_TO_ZERO,
    typename TI,
    ::metal::enable_if_t<!NAN_TO_ZERO, bool> = true>
inline opmath_t<TI> load_val(TI v) {
  return static_cast<opmath_t<TI>>(v);
}

template <
    bool NAN_TO_ZERO,
    typename TI,
    ::metal::enable_if_t<NAN_TO_ZERO, bool> = true>
inline opmath_t<TI> load_val(TI v) {
  auto r = static_cast<opmath_t<TI>>(v);
  if (::metal::isnan(static_cast<float>(r)))
    r = 0;
  return r;
}

// Sum reduction kernel with multiple independent accumulation chains (ILP).
// Each thread maintains NCHAINS independent accumulators to hide ALU latency
// and keep the memory pipeline saturated.
//
// Two internal paths selected per-threadgroup (not per-element):
//   - Single reduced dim (or full reduction): compute input_base + k * stride
//     once per TG, then direct indexing — no per-element dim loop.
//   - Multiple reduced dims: fall back to get_input_offset per element.
// NAN_TO_ZERO: when true, NaN inputs are treated as zero (nansum behavior).
// The compiler eliminates the dead branch when false.
template <
    typename TI,
    typename TO,
    uint NCHAINS = SUM_NCHAINS,
    bool NAN_TO_ZERO = false>
kernel void sum_reduction(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant NormParams<>& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simdgroup_size [[threads_per_simdgroup]]) {
  using TA = opmath_t<TO>;

  // Compute input_base (once per TG) and detect reduction pattern.
  // For single reduced dim: input_base + k * reduction_stride gives
  // the k-th reduction element — no per-element dim loop needed.
  uint32_t input_base = 0;
  uint32_t reduction_stride = 1;
  uint32_t num_reduced_dims = 0;
  {
    uint32_t out_idx = tgid;
    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      if (params.input_sizes[dim] != params.output_sizes[dim]) {
        num_reduced_dims++;
        reduction_stride = params.input_strides[dim];
      } else {
        auto idx = out_idx % params.output_sizes[dim];
        out_idx /= params.output_sizes[dim];
        input_base += idx * params.input_strides[dim];
      }
    }
  }

  // Load helper: cast to accumulator type, optionally replacing NaN with zero

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = 0;
  }

  const uint32_t rsize = params.reduction_size;
  const uint32_t stride = tptg * NCHAINS;
  uint32_t base = tid * NCHAINS;

  if (num_reduced_dims <= 1) {
    // Fast path: direct indexing with base + k * reduction_stride
    for (; base + NCHAINS <= rsize; base += stride) {
      for (uint j = 0; j < NCHAINS; j++) {
        acc[j] += load_val<NAN_TO_ZERO>(
            input[input_base + (base + j) * reduction_stride]);
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] +=
          load_val<NAN_TO_ZERO>(input[input_base + idx * reduction_stride]);
    }
  } else {
    // Generic path: per-element strided offset for multi-dim reductions
    for (; base + NCHAINS <= rsize; base += stride) {
      for (uint j = 0; j < NCHAINS; j++) {
        acc[j] += load_val<NAN_TO_ZERO>(
            input[get_input_offset(base + j, tgid, params)]);
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] +=
          load_val<NAN_TO_ZERO>(input[get_input_offset(idx, tgid, params)]);
    }
  }

  // Collapse chains into a single value
  TA output_val = acc[0];
  for (uint j = 1; j < NCHAINS; j++) {
    output_val += acc[j];
  }

  // SIMD + threadgroup tree reduction
  auto threads_remaining = tptg;
  threadgroup TA shared_outputs[MAX_THREADGROUP_SIZE];

  while (threads_remaining > 1) {
    output_val = c10::metal::simd_sum(output_val);
    threads_remaining = ceil_div(threads_remaining, simdgroup_size);

    if (threads_remaining > 1) {
      if (simd_lane_id == 0) {
        shared_outputs[simdgroup_id] = output_val;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < threads_remaining) {
        output_val = shared_outputs[tid];
      } else {
        return;
      }
    }
  }

  if (tid == 0) {
    uint32_t output_offset = 0;
    uint32_t reduction_idx = tgid;

    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      auto output_dim_size = params.output_sizes[dim];
      if (output_dim_size > 1) {
        auto index_in_dim = reduction_idx % output_dim_size;
        reduction_idx /= output_dim_size;
        output_offset += index_in_dim * params.output_strides[dim];
      }
    }
    // params.p > 0 means "divide the accumulator by p before casting"
    // (used by mean to keep the division in opmath_t precision so the
    // fp32 accumulation isn't lost when TO is fp16/bf16/half2).
    if (params.p > 0) {
      output_val /= static_cast<TA>(params.p);
    }
    output[output_offset] = static_cast<TO>(output_val);
  }
}

// Specialized kernel for reducing a non-innermost dim of a contiguous 2D
// tensor. Each thread handles one column, iterating over all rows with
// coalesced reads. Multiple row-workers per threadgroup reduce via shared
// memory. This avoids the strided-access penalty of the generic kernel for
// dim=0.
//
// Grid: (ceil(N/TG_X), 1) threadgroups, each (TG_X, TG_Y) threads.
// TG_X threads cover adjacent columns (coalesced), TG_Y threads split rows.
template <
    typename TI,
    typename TO,
    uint TG_X = 32,
    uint TG_Y = 32,
    uint NCHAINS = SUM_NCHAINS,
    bool NAN_TO_ZERO = false>
kernel void sum_reduction_outer(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]], // [M, N, output_stride]
    constant float& divisor [[buffer(3)]], // >0 divides accumulator before cast
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = opmath_t<TO>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint out_stride = sizes.z;

  uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= N)
    return;

  // Split rows among TG_Y workers
  uint rows_per_y = ceil_div(M, TG_Y);
  uint row_start = tid_tg.y * rows_per_y;
  uint row_end = min(row_start + rows_per_y, M);

  // Multiple accumulation chains for ILP
  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++)
    acc[j] = 0;

  uint row = row_start;
  for (; row + NCHAINS <= row_end; row += NCHAINS) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] += load_val<NAN_TO_ZERO>(input[(row + j) * N + col]);
    }
  }
  for (; row < row_end; row++) {
    acc[row % NCHAINS] += load_val<NAN_TO_ZERO>(input[row * N + col]);
  }

  TA sum = acc[0];
  for (uint j = 1; j < NCHAINS; j++)
    sum += acc[j];

  // Reduce across TG_Y row-workers via shared memory
  threadgroup TA shmem[TG_Y][TG_X];
  shmem[tid_tg.y][tid_tg.x] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = TG_Y / 2; stride > 0; stride >>= 1) {
    if (tid_tg.y < stride)
      shmem[tid_tg.y][tid_tg.x] += shmem[tid_tg.y + stride][tid_tg.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid_tg.y == 0) {
    TA final_val = shmem[0][tid_tg.x];
    if (divisor > 0) {
      final_val /= static_cast<TA>(divisor);
    }
    output[col * out_stride] = static_cast<TO>(final_val);
  }
}

#define REGISTER_SUM_OUTER_IMPL(TI, TO, PREFIX, NTZ)                 \
  template [[host_name(PREFIX "reduction_outer_" #TI "_" #TO)]]      \
  kernel void sum_reduction_outer<TI, TO, 32, 32, SUM_NCHAINS, NTZ>( \
      constant TI * input [[buffer(0)]],                             \
      device TO * output [[buffer(1)]],                              \
      constant uint3 & sizes [[buffer(2)]],                          \
      constant float& divisor [[buffer(3)]],                         \
      uint2 tid_tg [[thread_position_in_threadgroup]],               \
      uint2 tg_pos [[threadgroup_position_in_grid]]);

#define REGISTER_SUM_OUTER(TI, TO) \
  REGISTER_SUM_OUTER_IMPL(TI, TO, "sum_", false)
#define REGISTER_NANSUM_OUTER(TI, TO) \
  REGISTER_SUM_OUTER_IMPL(TI, TO, "nansum_", true)

REGISTER_SUM_OUTER(float, float);
REGISTER_SUM_OUTER(half, half);
REGISTER_SUM_OUTER(half, float);
REGISTER_SUM_OUTER(bfloat, bfloat);
REGISTER_SUM_OUTER(bfloat, float);
REGISTER_SUM_OUTER(int, int);
REGISTER_SUM_OUTER(int, long);
REGISTER_SUM_OUTER(long, long);
REGISTER_SUM_OUTER(short, short);
REGISTER_SUM_OUTER(short, long);
REGISTER_SUM_OUTER(char, char);
REGISTER_SUM_OUTER(char, long);
REGISTER_SUM_OUTER(uchar, uchar);
REGISTER_SUM_OUTER(uchar, long);
REGISTER_SUM_OUTER(bool, long);
REGISTER_SUM_OUTER(bool, int);
REGISTER_SUM_OUTER(float2, float2);
REGISTER_SUM_OUTER(half2, half2);

// Specialized kernel for reducing the innermost dim of a contiguous tensor.
// Input [M, N] -> output [M], each SIMD group reduces one row of N elements.
// Multiple SIMD groups per TG handle different rows for occupancy.
// No shared memory needed — simd_sum suffices for intra-row reduction.
template <
    typename TI,
    typename TO,
    uint NCHAINS = SUM_NCHAINS,
    bool NAN_TO_ZERO = false>
kernel void sum_reduction_inner(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint2& sizes [[buffer(2)]], // [M, N]
    constant float& divisor [[buffer(3)]], // >0 divides accumulator before cast
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = opmath_t<TO>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_simd_groups = tptg / 32;

  // Each SIMD group handles a different row
  uint row = tgid * num_simd_groups + simdgroup_id;
  if (row >= M)
    return;

  constant TI* row_ptr = input + row * N;

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++)
    acc[j] = 0;

  // Each of 32 lanes reads elements at stride 32, NCHAINS at a time.
  // Align down to full blocks of stride = 32 * NCHAINS elements.
  const uint stride = 32 * NCHAINS;
  const uint aligned_N = (N / stride) * stride;
  uint base = simd_lane_id * NCHAINS;
  for (; base < aligned_N; base += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] += load_val<NAN_TO_ZERO>(row_ptr[base + j]);
    }
  }
  // Tail: remaining elements after last full block, one per lane
  for (uint i = aligned_N + simd_lane_id; i < N; i += 32) {
    acc[0] += load_val<NAN_TO_ZERO>(row_ptr[i]);
  }

  TA sum = acc[0];
  for (uint j = 1; j < NCHAINS; j++)
    sum += acc[j];

  sum = c10::metal::simd_sum(sum);

  if (simd_lane_id == 0) {
    if (divisor > 0) {
      sum /= static_cast<TA>(divisor);
    }
    output[row] = static_cast<TO>(sum);
  }
}

#define REGISTER_SUM_INNER_IMPL(TI, TO, PREFIX, NTZ)            \
  template [[host_name(PREFIX "reduction_inner_" #TI "_" #TO)]] \
  kernel void sum_reduction_inner<TI, TO, SUM_NCHAINS, NTZ>(    \
      constant TI * input [[buffer(0)]],                        \
      device TO * output [[buffer(1)]],                         \
      constant uint2 & sizes [[buffer(2)]],                     \
      constant float& divisor [[buffer(3)]],                    \
      uint tptg [[threads_per_threadgroup]],                    \
      uint tgid [[threadgroup_position_in_grid]],               \
      uint simd_lane_id [[thread_index_in_simdgroup]],          \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define REGISTER_SUM_INNER(TI, TO) \
  REGISTER_SUM_INNER_IMPL(TI, TO, "sum_", false)
#define REGISTER_NANSUM_INNER(TI, TO) \
  REGISTER_SUM_INNER_IMPL(TI, TO, "nansum_", true)

REGISTER_SUM_INNER(float, float);
REGISTER_SUM_INNER(half, half);
REGISTER_SUM_INNER(half, float);
REGISTER_SUM_INNER(bfloat, bfloat);
REGISTER_SUM_INNER(bfloat, float);
REGISTER_SUM_INNER(int, int);
REGISTER_SUM_INNER(int, long);
REGISTER_SUM_INNER(long, long);
REGISTER_SUM_INNER(short, short);
REGISTER_SUM_INNER(short, long);
REGISTER_SUM_INNER(char, char);
REGISTER_SUM_INNER(char, long);
REGISTER_SUM_INNER(uchar, uchar);
REGISTER_SUM_INNER(uchar, long);
REGISTER_SUM_INNER(bool, long);
REGISTER_SUM_INNER(bool, int);
REGISTER_SUM_INNER(float2, float2);
REGISTER_SUM_INNER(half2, half2);

#define REGISTER_SUM_IMPL(TI, TO, PREFIX, NTZ)              \
  template [[host_name(PREFIX "reduction_" #TI "_" #TO)]]   \
  kernel void sum_reduction<TI, TO, SUM_NCHAINS, NTZ>(      \
      constant TI * input [[buffer(0)]],                    \
      device TO * output [[buffer(1)]],                     \
      constant NormParams<> & params [[buffer(2)]],         \
      uint tid [[thread_position_in_threadgroup]],          \
      uint tptg [[threads_per_threadgroup]],                \
      uint tgid [[threadgroup_position_in_grid]],           \
      uint simd_lane_id [[thread_index_in_simdgroup]],      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]], \
      uint simdgroup_size [[threads_per_simdgroup]]);

#define REGISTER_SUM(TI, TO) REGISTER_SUM_IMPL(TI, TO, "sum_", false)
#define REGISTER_NANSUM(TI, TO) REGISTER_SUM_IMPL(TI, TO, "nansum_", true)

REGISTER_SUM(float, float);
REGISTER_SUM(float, half);
REGISTER_SUM(float, bfloat);
REGISTER_SUM(half, half);
REGISTER_SUM(half, float);
REGISTER_SUM(bfloat, bfloat);
REGISTER_SUM(bfloat, float);
REGISTER_SUM(int, int);
REGISTER_SUM(int, long);
REGISTER_SUM(long, long);
REGISTER_SUM(short, short);
REGISTER_SUM(short, long);
REGISTER_SUM(char, char);
REGISTER_SUM(char, long);
REGISTER_SUM(uchar, uchar);
REGISTER_SUM(uchar, long);
REGISTER_SUM(bool, long);
REGISTER_SUM(bool, int);
REGISTER_SUM(float2, float2);
REGISTER_SUM(half2, half2);

// nansum variants (floating-point only — integers can't have NaN)
REGISTER_NANSUM(float, float);
REGISTER_NANSUM(half, half);
REGISTER_NANSUM(half, float);
REGISTER_NANSUM(bfloat, bfloat);
REGISTER_NANSUM(bfloat, float);

REGISTER_NANSUM_OUTER(float, float);
REGISTER_NANSUM_OUTER(half, half);
REGISTER_NANSUM_OUTER(half, float);
REGISTER_NANSUM_OUTER(bfloat, bfloat);
REGISTER_NANSUM_OUTER(bfloat, float);

REGISTER_NANSUM_INNER(float, float);
REGISTER_NANSUM_INNER(half, half);
REGISTER_NANSUM_INNER(half, float);
REGISTER_NANSUM_INNER(bfloat, bfloat);
REGISTER_NANSUM_INNER(bfloat, float);
