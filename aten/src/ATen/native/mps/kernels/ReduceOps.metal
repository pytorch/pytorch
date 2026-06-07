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

// Load modes for sum_reduction: identity (sum), nan-to-zero (nansum),
// or nonzero-as-one (count_nonzero).
enum LoadMode : uint {
  LOAD_IDENTITY = 0,
  LOAD_NAN_TO_ZERO = 1,
  LOAD_NONZERO = 2
};

template <typename T, ::metal::enable_if_t<!is_complex_v<T>, bool> = true>
inline bool load_is_nonzero(T v) {
  return v != T(0);
}

template <typename T, ::metal::enable_if_t<is_complex_v<T>, bool> = true>
inline bool load_is_nonzero(T v) {
  return v.x != 0 || v.y != 0;
}

// Load helper: cast to opmath_t, optionally replacing NaN with zero,
// or map nonzero to 1 for count_nonzero semantics.
template <
    LoadMode MODE,
    typename TI,
    ::metal::enable_if_t<MODE == LOAD_IDENTITY, bool> = true>
inline opmath_t<TI> load_val(TI v) {
  return static_cast<opmath_t<TI>>(v);
}

template <
    LoadMode MODE,
    typename TI,
    ::metal::enable_if_t<MODE == LOAD_NAN_TO_ZERO, bool> = true>
inline opmath_t<TI> load_val(TI v) {
  auto r = static_cast<opmath_t<TI>>(v);
  if (::metal::isnan(static_cast<float>(r)))
    r = 0;
  return r;
}

// LOAD_NONZERO returns uint: MPS tensor numel fits in uint32, so per-TG
// (and per-output-element) non-zero counts cannot overflow. This lets
// count_nonzero accumulate in 32-bit integer instead of 64-bit, which is a
// meaningful speedup for small inputs (especially bool) where compute
// overhead dominates. The final cast back to long happens at the output
// store in the kernel.
template <
    LoadMode MODE,
    typename TI,
    ::metal::enable_if_t<MODE == LOAD_NONZERO, bool> = true>
inline uint load_val(TI v) {
  return load_is_nonzero(v) ? 1u : 0u;
}

// Sum reduction kernel with multiple independent accumulation chains (ILP).
// Each thread maintains NCHAINS independent accumulators to hide ALU latency
// and keep the memory pipeline saturated.
//
// Two internal paths selected per-threadgroup (not per-element):
//   - Single reduced dim (or full reduction): compute input_base + k * stride
//     once per TG, then direct indexing — no per-element dim loop.
//   - Multiple reduced dims: fall back to get_input_offset per element.
// MODE: LOAD_IDENTITY (sum), LOAD_NAN_TO_ZERO (nansum),
// LOAD_NONZERO (count_nonzero — contributes 1 per nonzero element).
// The compiler eliminates dead branches per instantiation.
template <
    typename TI,
    typename TO,
    uint NCHAINS = SUM_NCHAINS,
    LoadMode MODE = LOAD_IDENTITY>
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
  using TA = ::metal::conditional_t<MODE == LOAD_NONZERO, uint, opmath_t<TO>>;

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
        acc[j] +=
            load_val<MODE>(input[input_base + (base + j) * reduction_stride]);
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] +=
          load_val<MODE>(input[input_base + idx * reduction_stride]);
    }
  } else {
    // Generic path: per-element strided offset for multi-dim reductions
    for (; base + NCHAINS <= rsize; base += stride) {
      for (uint j = 0; j < NCHAINS; j++) {
        acc[j] +=
            load_val<MODE>(input[get_input_offset(base + j, tgid, params)]);
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] +=
          load_val<MODE>(input[get_input_offset(idx, tgid, params)]);
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

template <
    typename TI,
    typename TO,
    uint NCHAINS = SUM_NCHAINS,
    LoadMode MODE = LOAD_IDENTITY>
kernel void sum_reduction_strided_pass1(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant NormParams<>& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  using TA = ::metal::conditional_t<MODE == LOAD_NONZERO, uint, opmath_t<TO>>;

  const uint32_t E = params.reduction_size;
  const uint32_t base_flat = tgid * E;

  TA acc = 0;
  for (uint32_t k = tid; k < E; k += tptg) {
    acc += load_val<MODE>(input[get_input_offset(base_flat + k, 0u, params)]);
  }

  threadgroup TA shared[MAX_THREADGROUP_SIZE / 32];
  TA total = c10::metal::threadgroup_sum(shared, acc, tid, tptg);
  if (tid == 0) {
    output[tgid] = static_cast<TO>(total);
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
    LoadMode MODE = LOAD_IDENTITY>
kernel void sum_reduction_outer(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]], // [M, N, output_stride]
    constant float& divisor [[buffer(3)]], // >0 divides accumulator before cast
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = ::metal::conditional_t<MODE == LOAD_NONZERO, uint, opmath_t<TO>>;
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
      acc[j] += load_val<MODE>(input[(row + j) * N + col]);
    }
  }
  for (; row < row_end; row++) {
    acc[row % NCHAINS] += load_val<MODE>(input[row * N + col]);
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

#define REGISTER_SUM_OUTER_IMPL(TI, TO, PREFIX, MODE)                 \
  template [[host_name(PREFIX "reduction_outer_" #TI "_" #TO)]]       \
  kernel void sum_reduction_outer<TI, TO, 32, 32, SUM_NCHAINS, MODE>( \
      constant TI * input [[buffer(0)]],                              \
      device TO * output [[buffer(1)]],                               \
      constant uint3 & sizes [[buffer(2)]],                           \
      constant float& divisor [[buffer(3)]],                          \
      uint2 tid_tg [[thread_position_in_threadgroup]],                \
      uint2 tg_pos [[threadgroup_position_in_grid]]);

#define REGISTER_SUM_OUTER(TI, TO) \
  REGISTER_SUM_OUTER_IMPL(TI, TO, "sum_", LOAD_IDENTITY)
#define REGISTER_NANSUM_OUTER(TI, TO) \
  REGISTER_SUM_OUTER_IMPL(TI, TO, "nansum_", LOAD_NAN_TO_ZERO)
#define REGISTER_COUNT_NONZERO_OUTER(TI) \
  REGISTER_SUM_OUTER_IMPL(TI, long, "count_nonzero_", LOAD_NONZERO)

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
    LoadMode MODE = LOAD_IDENTITY>
kernel void sum_reduction_inner(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint2& sizes [[buffer(2)]], // [M, N]
    constant float& divisor [[buffer(3)]], // >0 divides accumulator before cast
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = ::metal::conditional_t<MODE == LOAD_NONZERO, uint, opmath_t<TO>>;
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
      acc[j] += load_val<MODE>(row_ptr[base + j]);
    }
  }
  // Tail: remaining elements after last full block, one per lane
  for (uint i = aligned_N + simd_lane_id; i < N; i += 32) {
    acc[0] += load_val<MODE>(row_ptr[i]);
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

#define REGISTER_SUM_INNER_IMPL(TI, TO, PREFIX, MODE)           \
  template [[host_name(PREFIX "reduction_inner_" #TI "_" #TO)]] \
  kernel void sum_reduction_inner<TI, TO, SUM_NCHAINS, MODE>(   \
      constant TI * input [[buffer(0)]],                        \
      device TO * output [[buffer(1)]],                         \
      constant uint2 & sizes [[buffer(2)]],                     \
      constant float& divisor [[buffer(3)]],                    \
      uint tptg [[threads_per_threadgroup]],                    \
      uint tgid [[threadgroup_position_in_grid]],               \
      uint simd_lane_id [[thread_index_in_simdgroup]],          \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define REGISTER_SUM_INNER(TI, TO) \
  REGISTER_SUM_INNER_IMPL(TI, TO, "sum_", LOAD_IDENTITY)
#define REGISTER_NANSUM_INNER(TI, TO) \
  REGISTER_SUM_INNER_IMPL(TI, TO, "nansum_", LOAD_NAN_TO_ZERO)
#define REGISTER_COUNT_NONZERO_INNER(TI) \
  REGISTER_SUM_INNER_IMPL(TI, long, "count_nonzero_", LOAD_NONZERO)

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

#define REGISTER_SUM_IMPL(TI, TO, PREFIX, MODE)             \
  template [[host_name(PREFIX "reduction_" #TI "_" #TO)]]   \
  kernel void sum_reduction<TI, TO, SUM_NCHAINS, MODE>(     \
      constant TI * input [[buffer(0)]],                    \
      device TO * output [[buffer(1)]],                     \
      constant NormParams<> & params [[buffer(2)]],         \
      uint tid [[thread_position_in_threadgroup]],          \
      uint tptg [[threads_per_threadgroup]],                \
      uint tgid [[threadgroup_position_in_grid]],           \
      uint simd_lane_id [[thread_index_in_simdgroup]],      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]], \
      uint simdgroup_size [[threads_per_simdgroup]]);

#define REGISTER_SUM_STRIDED_IMPL(TI, TO, PREFIX, MODE)               \
  template [[host_name(PREFIX "reduction_strided_" #TI "_" #TO)]]     \
  kernel void sum_reduction_strided_pass1<TI, TO, SUM_NCHAINS, MODE>( \
      constant TI * input [[buffer(0)]],                              \
      device TO * output [[buffer(1)]],                               \
      constant NormParams<> & params [[buffer(2)]],                   \
      uint tid [[thread_position_in_threadgroup]],                    \
      uint tptg [[threads_per_threadgroup]],                          \
      uint tgid [[threadgroup_position_in_grid]]);

#define REGISTER_SUM(TI, TO)                       \
  REGISTER_SUM_IMPL(TI, TO, "sum_", LOAD_IDENTITY) \
  REGISTER_SUM_STRIDED_IMPL(TI, TO, "sum_", LOAD_IDENTITY)
#define REGISTER_NANSUM(TI, TO)                          \
  REGISTER_SUM_IMPL(TI, TO, "nansum_", LOAD_NAN_TO_ZERO) \
  REGISTER_SUM_STRIDED_IMPL(TI, TO, "nansum_", LOAD_NAN_TO_ZERO)
#define REGISTER_COUNT_NONZERO(TI)                            \
  REGISTER_SUM_IMPL(TI, long, "count_nonzero_", LOAD_NONZERO) \
  REGISTER_SUM_STRIDED_IMPL(TI, long, "count_nonzero_", LOAD_NONZERO)

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

// count_nonzero: output is always long; reuses sum-reduction machinery
// with LOAD_NONZERO mode (1 per nonzero element, 0 otherwise).
REGISTER_COUNT_NONZERO(float);
REGISTER_COUNT_NONZERO(half);
REGISTER_COUNT_NONZERO(bfloat);
REGISTER_COUNT_NONZERO(long);
REGISTER_COUNT_NONZERO(int);
REGISTER_COUNT_NONZERO(short);
REGISTER_COUNT_NONZERO(char);
REGISTER_COUNT_NONZERO(uchar);
REGISTER_COUNT_NONZERO(bool);
REGISTER_COUNT_NONZERO(float2);
REGISTER_COUNT_NONZERO(half2);

REGISTER_COUNT_NONZERO_OUTER(float);
REGISTER_COUNT_NONZERO_OUTER(half);
REGISTER_COUNT_NONZERO_OUTER(bfloat);
REGISTER_COUNT_NONZERO_OUTER(long);
REGISTER_COUNT_NONZERO_OUTER(int);
REGISTER_COUNT_NONZERO_OUTER(short);
REGISTER_COUNT_NONZERO_OUTER(char);
REGISTER_COUNT_NONZERO_OUTER(uchar);
REGISTER_COUNT_NONZERO_OUTER(bool);
REGISTER_COUNT_NONZERO_OUTER(float2);
REGISTER_COUNT_NONZERO_OUTER(half2);

REGISTER_COUNT_NONZERO_INNER(float);
REGISTER_COUNT_NONZERO_INNER(half);
REGISTER_COUNT_NONZERO_INNER(bfloat);
REGISTER_COUNT_NONZERO_INNER(long);
REGISTER_COUNT_NONZERO_INNER(int);
REGISTER_COUNT_NONZERO_INNER(short);
REGISTER_COUNT_NONZERO_INNER(char);
REGISTER_COUNT_NONZERO_INNER(uchar);
REGISTER_COUNT_NONZERO_INNER(bool);
REGISTER_COUNT_NONZERO_INNER(float2);
REGISTER_COUNT_NONZERO_INNER(half2);

// =============================================================================
// value reductions: amin/amax (Op = MinOp/MaxOp on T, identity load) and
// all/any (Op = MinOp/MaxOp on uchar, predicate load).
// any = max-of-bool, all = min-of-bool; the predicate load converts each
// input element to {0, 1} (nonzero, NaN -> 1) before the reduction.
// =============================================================================

// Reduction op functors. Each Op defines identity<T>(), combine(), and
// threadgroup_reduce(). Identity is the "neutral" element for the op (-INF
// for max on floats, numeric_limits<T>::lowest() for integers, etc.).
// combine() and threadgroup_reduce() delegate to c10::metal helpers, which
// propagate NaN for floats.
struct MaxOp {
  template <
      typename T,
      ::metal::enable_if_t<!is_floating_point_v<T>, bool> = true>
  static inline constexpr T identity() {
    return ::metal::numeric_limits<T>::lowest();
  }
  // Float identity is -INFINITY (not -FLT_MAX): max(-INF, x) = x for any
  // finite x including -FLT_MAX, but max(-FLT_MAX, -INFINITY) would
  // incorrectly return -FLT_MAX.
  template <
      typename T,
      ::metal::enable_if_t<is_floating_point_v<T>, bool> = true>
  static inline constexpr T identity() {
    return T(-INFINITY);
  }
  template <typename T>
  static inline T combine(T a, T b) {
    return c10::metal::max(a, b);
  }
  template <typename T>
  static inline T simd_reduce(T val) {
    return c10::metal::simd_max(val);
  }
  template <typename T>
  static inline T threadgroup_reduce(
      threadgroup T* shared,
      T val,
      uint tid,
      uint tptg) {
    return c10::metal::threadgroup_max(shared, val, tid, tptg);
  }
};

struct MinOp {
  template <
      typename T,
      ::metal::enable_if_t<!is_floating_point_v<T>, bool> = true>
  static inline constexpr T identity() {
    return ::metal::numeric_limits<T>::max();
  }
  template <
      typename T,
      ::metal::enable_if_t<is_floating_point_v<T>, bool> = true>
  static inline constexpr T identity() {
    return T(INFINITY);
  }
  template <typename T>
  static inline T combine(T a, T b) {
    return c10::metal::min(a, b);
  }
  template <typename T>
  static inline T simd_reduce(T val) {
    return c10::metal::simd_min(val);
  }
  template <typename T>
  static inline T threadgroup_reduce(
      threadgroup T* shared,
      T val,
      uint tid,
      uint tptg) {
    return c10::metal::threadgroup_min(shared, val, tid, tptg);
  }
};

// Load functors decide how an input element is converted into the
// accumulator type. IdentityLoad casts (min/max keep the value unchanged);
// PredicateLoad maps nonzero (and NaN) -> 1, zero -> 0 (any/all).
struct IdentityLoad {
  template <typename TA, typename TI>
  static inline TA load(TI v) {
    return static_cast<TA>(v);
  }
};

struct PredicateLoad {
  template <typename TA, typename TI>
  static inline TA load(TI v) {
    return load_is_nonzero(v) ? TA(1) : TA(0);
  }
};

// General value reduction: same 2D-via-NormParams layout as sum_reduction,
// parameterised on the reduction op and load mode. For min/max, TI == TO
// and Load = IdentityLoad. For all/any, TO = uchar (a 1-byte alias for the
// bool output buffer) and Load = PredicateLoad. The
// max_total_threads_per_threadgroup hint lets the compiler bound the
// runtime tptg value, which in turn lets c10::metal::threadgroup_min/max
// constant-fold its size-vs-simdgroup_size branch.
template <
    typename Op,
    typename Load,
    typename TI,
    typename TO,
    uint NCHAINS = SUM_NCHAINS>
[[max_total_threads_per_threadgroup(MAX_THREADGROUP_SIZE)]]
kernel void value_reduction(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant NormParams<>& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
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

  using TA = TO;
  const TA identity_val = Op::template identity<TA>();
  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = identity_val;
  }

  const uint32_t rsize = params.reduction_size;
  const uint32_t stride = tptg * NCHAINS;
  uint32_t base = tid * NCHAINS;

  if (num_reduced_dims <= 1) {
    for (; base + NCHAINS <= rsize; base += stride) {
      for (uint j = 0; j < NCHAINS; j++) {
        acc[j] = Op::combine(
            acc[j],
            Load::template load<TA>(
                input[input_base + (base + j) * reduction_stride]));
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] = Op::combine(
          acc[idx % NCHAINS],
          Load::template load<TA>(input[input_base + idx * reduction_stride]));
    }
  } else {
    for (; base + NCHAINS <= rsize; base += stride) {
      for (uint j = 0; j < NCHAINS; j++) {
        acc[j] = Op::combine(
            acc[j],
            Load::template load<TA>(
                input[get_input_offset(base + j, tgid, params)]));
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] = Op::combine(
          acc[idx % NCHAINS],
          Load::template load<TA>(input[get_input_offset(idx, tgid, params)]));
    }
  }

  TA output_val = acc[0];
  for (uint j = 1; j < NCHAINS; j++) {
    output_val = Op::combine(output_val, acc[j]);
  }

  threadgroup TA shared_outputs[MAX_THREADGROUP_SIZE / simdgroup_size];
  output_val = Op::threadgroup_reduce(shared_outputs, output_val, tid, tptg);

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
    output[output_offset] = output_val;
  }
}

// Outer-dim variant: input is logically [M, N], reduce M down so output is
// [N]. TG_X threads cover adjacent output columns (coalesced reads), TG_Y
// threads split the M rows. Mirrors sum_reduction_outer; uses the same
// (Op, Load) abstraction as value_reduction.
template <
    typename Op,
    typename Load,
    typename TI,
    typename TO,
    uint TG_X = 32,
    uint TG_Y = 32,
    uint NCHAINS = SUM_NCHAINS>
[[max_total_threads_per_threadgroup(TG_X * TG_Y)]]
kernel void value_reduction_outer(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]], // [M, N, output_stride]
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = TO;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint out_stride = sizes.z;

  uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= N) {
    return;
  }

  uint rows_per_y = ceil_div(M, TG_Y);
  uint row_start = tid_tg.y * rows_per_y;
  uint row_end = min(row_start + rows_per_y, M);

  const TA identity_val = Op::template identity<TA>();
  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = identity_val;
  }

  uint row = row_start;
  for (; row + NCHAINS <= row_end; row += NCHAINS) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] = Op::combine(
          acc[j], Load::template load<TA>(input[(row + j) * N + col]));
    }
  }
  for (; row < row_end; row++) {
    acc[row % NCHAINS] = Op::combine(
        acc[row % NCHAINS], Load::template load<TA>(input[row * N + col]));
  }

  TA val = acc[0];
  for (uint j = 1; j < NCHAINS; j++) {
    val = Op::combine(val, acc[j]);
  }

  // Reduce across TG_Y row-workers via shared memory.
  threadgroup TA shmem[TG_Y][TG_X];
  shmem[tid_tg.y][tid_tg.x] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = TG_Y / 2; stride > 0; stride >>= 1) {
    if (tid_tg.y < stride) {
      shmem[tid_tg.y][tid_tg.x] = Op::combine(
          shmem[tid_tg.y][tid_tg.x], shmem[tid_tg.y + stride][tid_tg.x]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid_tg.y == 0) {
    output[col * out_stride] = shmem[0][tid_tg.x];
  }
}

// Inner-dim variant: input is logically [M, N], reduce N (innermost dim)
// so output is [M]. One SIMD group (32 lanes) handles one row, multiple
// SIMD groups per TG for occupancy. No shared memory needed since
// simd_reduce suffices for intra-row collapse. Mirrors sum_reduction_inner.
template <
    typename Op,
    typename Load,
    typename TI,
    typename TO,
    uint NCHAINS = SUM_NCHAINS>
kernel void value_reduction_inner(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint2& sizes [[buffer(2)]], // [M, N]
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = TO;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_simd_groups = tptg / simdgroup_size;

  uint row = tgid * num_simd_groups + simdgroup_id;
  if (row >= M) {
    return;
  }

  constant TI* row_ptr = input + row * N;

  const TA identity_val = Op::template identity<TA>();
  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = identity_val;
  }

  const uint stride = simdgroup_size * NCHAINS;
  const uint aligned_N = (N / stride) * stride;
  uint base = simd_lane_id * NCHAINS;
  for (; base < aligned_N; base += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] = Op::combine(acc[j], Load::template load<TA>(row_ptr[base + j]));
    }
  }
  for (uint i = aligned_N + simd_lane_id; i < N; i += simdgroup_size) {
    acc[0] = Op::combine(acc[0], Load::template load<TA>(row_ptr[i]));
  }

  TA val = acc[0];
  for (uint j = 1; j < NCHAINS; j++) {
    val = Op::combine(val, acc[j]);
  }

  val = Op::simd_reduce(val);

  if (simd_lane_id == 0) {
    output[row] = val;
  }
}

#define REGISTER_VALUE_REDUCTION_IMPL(TI, TO, NAME, OP, LOAD)               \
  template [[host_name(NAME "_reduction_" #TI "_" #TO)]]                    \
  kernel void value_reduction<OP, LOAD, TI, TO, SUM_NCHAINS>(               \
      constant TI * input [[buffer(0)]],                                    \
      device TO * output [[buffer(1)]],                                     \
      constant NormParams<> & params [[buffer(2)]],                         \
      uint tid [[thread_position_in_threadgroup]],                          \
      uint tptg [[threads_per_threadgroup]],                                \
      uint tgid [[threadgroup_position_in_grid]]);                          \
  template [[host_name(NAME "_reduction_outer_" #TI "_" #TO)]]              \
  kernel void value_reduction_outer<OP, LOAD, TI, TO, 32, 32, SUM_NCHAINS>( \
      constant TI * input [[buffer(0)]],                                    \
      device TO * output [[buffer(1)]],                                     \
      constant uint3 & sizes [[buffer(2)]],                                 \
      uint2 tid_tg [[thread_position_in_threadgroup]],                      \
      uint2 tg_pos [[threadgroup_position_in_grid]]);                       \
  template [[host_name(NAME "_reduction_inner_" #TI "_" #TO)]]              \
  kernel void value_reduction_inner<OP, LOAD, TI, TO, SUM_NCHAINS>(         \
      constant TI * input [[buffer(0)]],                                    \
      device TO * output [[buffer(1)]],                                     \
      constant uint2 & sizes [[buffer(2)]],                                 \
      uint tptg [[threads_per_threadgroup]],                                \
      uint tgid [[threadgroup_position_in_grid]],                           \
      uint simd_lane_id [[thread_index_in_simdgroup]],                      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define REGISTER_MAX(T) \
  REGISTER_VALUE_REDUCTION_IMPL(T, T, "max", MaxOp, IdentityLoad)
#define REGISTER_MIN(T) \
  REGISTER_VALUE_REDUCTION_IMPL(T, T, "min", MinOp, IdentityLoad)
#define REGISTER_ANY(TI) \
  REGISTER_VALUE_REDUCTION_IMPL(TI, uchar, "any", MaxOp, PredicateLoad)
#define REGISTER_ALL(TI) \
  REGISTER_VALUE_REDUCTION_IMPL(TI, uchar, "all", MinOp, PredicateLoad)

// Numeric types that participate in min/max AND all/any.
#define REGISTER_REDUCTIONS_OPS_FOR_TYPE(T) \
  REGISTER_MAX(T)                           \
  REGISTER_MIN(T)                           \
  REGISTER_ANY(T)                           \
  REGISTER_ALL(T)

// Types that only participate in all/any (bool: no simd_min/max; complex:
// no ordering, but predicate-reduce on its real/imag pair is well-defined).
#define REGISTER_PRED_REDUCTIONS_FOR_TYPE(T) \
  REGISTER_ANY(T)                            \
  REGISTER_ALL(T)

REGISTER_REDUCTIONS_OPS_FOR_TYPE(float);
REGISTER_REDUCTIONS_OPS_FOR_TYPE(half);
REGISTER_REDUCTIONS_OPS_FOR_TYPE(bfloat);
REGISTER_REDUCTIONS_OPS_FOR_TYPE(long);
REGISTER_REDUCTIONS_OPS_FOR_TYPE(int);
REGISTER_REDUCTIONS_OPS_FOR_TYPE(short);
REGISTER_REDUCTIONS_OPS_FOR_TYPE(char);
REGISTER_REDUCTIONS_OPS_FOR_TYPE(uchar);

REGISTER_PRED_REDUCTIONS_FOR_TYPE(bool);
REGISTER_PRED_REDUCTIONS_FOR_TYPE(float2);
REGISTER_PRED_REDUCTIONS_FOR_TYPE(half2);

// ===== prod / welford / argreduce kernels (this PR, on top of malfet sum/value migration) =====
// Product reduction kernels
// ============================================================================

template <typename TI, typename TO, uint NCHAINS = SUM_NCHAINS>
kernel void prod_reduction(
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

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++)
    acc[j] = 1;

  const uint32_t rsize = params.reduction_size;
  const uint32_t stride = tptg * NCHAINS;
  uint32_t base = tid * NCHAINS;

  if (num_reduced_dims <= 1) {
    for (; base + NCHAINS <= rsize; base += stride) {
      for (uint j = 0; j < NCHAINS; j++) {
        acc[j] *=
            static_cast<TA>(input[input_base + (base + j) * reduction_stride]);
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] *=
          static_cast<TA>(input[input_base + idx * reduction_stride]);
    }
  } else {
    for (; base + NCHAINS <= rsize; base += stride) {
      for (uint j = 0; j < NCHAINS; j++) {
        acc[j] *=
            static_cast<TA>(input[get_input_offset(base + j, tgid, params)]);
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] *=
          static_cast<TA>(input[get_input_offset(idx, tgid, params)]);
    }
  }

  TA output_val = acc[0];
  for (uint j = 1; j < NCHAINS; j++)
    output_val *= acc[j];

  auto threads_remaining = tptg;
  threadgroup TA shared_outputs[MAX_THREADGROUP_SIZE];

  while (threads_remaining > 1) {
    output_val = c10::metal::simd_prod(output_val);
    threads_remaining = ceil_div(threads_remaining, simdgroup_size);
    if (threads_remaining > 1) {
      if (simd_lane_id == 0)
        shared_outputs[simdgroup_id] = output_val;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < threads_remaining) {
        output_val = shared_outputs[tid];
      } else {
        output_val = static_cast<TA>(1);
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
    output[output_offset] = static_cast<TO>(output_val);
  }
}

template <
    typename TI,
    typename TO,
    uint TG_X = 32,
    uint TG_Y = 32,
    uint NCHAINS = SUM_NCHAINS>
kernel void prod_reduction_outer(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = opmath_t<TO>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint out_stride = sizes.z;

  uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= N)
    return;

  uint rows_per_y = ceil_div(M, TG_Y);
  uint row_start = tid_tg.y * rows_per_y;
  uint row_end = min(row_start + rows_per_y, M);

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++)
    acc[j] = 1;

  uint row = row_start;
  for (; row + NCHAINS <= row_end; row += NCHAINS) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] *= static_cast<TA>(input[(row + j) * N + col]);
    }
  }
  for (; row < row_end; row++) {
    acc[row % NCHAINS] *= static_cast<TA>(input[row * N + col]);
  }

  TA prod = acc[0];
  for (uint j = 1; j < NCHAINS; j++)
    prod *= acc[j];

  threadgroup TA shmem[TG_Y][TG_X];
  shmem[tid_tg.y][tid_tg.x] = prod;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = TG_Y / 2; s > 0; s >>= 1) {
    if (tid_tg.y < s)
      shmem[tid_tg.y][tid_tg.x] *= shmem[tid_tg.y + s][tid_tg.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid_tg.y == 0) {
    output[col * out_stride] = static_cast<TO>(shmem[0][tid_tg.x]);
  }
}

template <typename TI, typename TO, uint NCHAINS = SUM_NCHAINS>
kernel void prod_reduction_inner(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint2& sizes [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = opmath_t<TO>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_simd_groups = tptg / 32;

  // One SIMD group reduces one row; num_simd_groups rows share a TG (mirrors
  // sum_reduction_inner). Avoids one whole TG per row, which idles most threads
  // and over-subscribes the GPU when there are many short rows.
  uint row = tgid * num_simd_groups + simdgroup_id;
  if (row >= M)
    return;

  constant TI* row_ptr = input + row * N;

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++)
    acc[j] = 1;

  const uint stride = 32 * NCHAINS;
  const uint aligned_N = (N / stride) * stride;
  uint base = simd_lane_id * NCHAINS;
  for (; base < aligned_N; base += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] *= static_cast<TA>(row_ptr[base + j]);
    }
  }
  for (uint i = aligned_N + simd_lane_id; i < N; i += 32) {
    acc[0] *= static_cast<TA>(row_ptr[i]);
  }

  TA prod = acc[0];
  for (uint j = 1; j < NCHAINS; j++)
    prod *= acc[j];
  // Reduce across the 32 lanes of this SIMD group (no shared memory needed).
  prod = c10::metal::simd_prod(prod);

  if (simd_lane_id == 0) {
    output[row] = static_cast<TO>(prod);
  }
}

#define REGISTER_PROD(TI, TO)                               \
  template [[host_name("prod_reduction_" #TI "_" #TO)]]     \
  kernel void prod_reduction<TI, TO, SUM_NCHAINS>(          \
      constant TI * input [[buffer(0)]],                    \
      device TO * output [[buffer(1)]],                     \
      constant NormParams<> & params [[buffer(2)]],         \
      uint tid [[thread_position_in_threadgroup]],          \
      uint tptg [[threads_per_threadgroup]],                \
      uint tgid [[threadgroup_position_in_grid]],           \
      uint simd_lane_id [[thread_index_in_simdgroup]],      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]], \
      uint simdgroup_size [[threads_per_simdgroup]]);

#define REGISTER_PROD_OUTER(TI, TO)                              \
  template [[host_name("prod_reduction_outer_" #TI "_" #TO)]]    \
  kernel void prod_reduction_outer<TI, TO, 32, 32, SUM_NCHAINS>( \
      constant TI * input [[buffer(0)]],                         \
      device TO * output [[buffer(1)]],                          \
      constant uint3 & sizes [[buffer(2)]],                      \
      uint2 tid_tg [[thread_position_in_threadgroup]],           \
      uint2 tg_pos [[threadgroup_position_in_grid]]);

#define REGISTER_PROD_INNER(TI, TO)                           \
  template [[host_name("prod_reduction_inner_" #TI "_" #TO)]] \
  kernel void prod_reduction_inner<TI, TO, SUM_NCHAINS>(      \
      constant TI * input [[buffer(0)]],                      \
      device TO * output [[buffer(1)]],                       \
      constant uint2 & sizes [[buffer(2)]],                   \
      uint tid [[thread_index_in_threadgroup]],               \
      uint tptg [[threads_per_threadgroup]],                  \
      uint tgid [[threadgroup_position_in_grid]],             \
      uint simd_lane_id [[thread_index_in_simdgroup]],        \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

REGISTER_PROD(float, float);
REGISTER_PROD(half, half);
REGISTER_PROD(half, float);
REGISTER_PROD(bfloat, bfloat);
REGISTER_PROD(bfloat, float);
REGISTER_PROD(int, int);
REGISTER_PROD(int, long);
REGISTER_PROD(long, long);
REGISTER_PROD(short, short);
REGISTER_PROD(short, long);
REGISTER_PROD(char, char);
REGISTER_PROD(char, long);
REGISTER_PROD(uchar, uchar);
REGISTER_PROD(uchar, long);
REGISTER_PROD(bool, long);
REGISTER_PROD(bool, int);

REGISTER_PROD_OUTER(float, float);
REGISTER_PROD_OUTER(half, half);
REGISTER_PROD_OUTER(half, float);
REGISTER_PROD_OUTER(bfloat, bfloat);
REGISTER_PROD_OUTER(bfloat, float);
REGISTER_PROD_OUTER(int, int);
REGISTER_PROD_OUTER(int, long);
REGISTER_PROD_OUTER(long, long);
REGISTER_PROD_OUTER(short, short);
REGISTER_PROD_OUTER(short, long);
REGISTER_PROD_OUTER(char, char);
REGISTER_PROD_OUTER(char, long);
REGISTER_PROD_OUTER(uchar, uchar);
REGISTER_PROD_OUTER(uchar, long);
REGISTER_PROD_OUTER(bool, long);
REGISTER_PROD_OUTER(bool, int);

REGISTER_PROD_INNER(float, float);
REGISTER_PROD_INNER(half, half);
REGISTER_PROD_INNER(half, float);
REGISTER_PROD_INNER(bfloat, bfloat);
REGISTER_PROD_INNER(bfloat, float);
REGISTER_PROD_INNER(int, int);
REGISTER_PROD_INNER(int, long);
REGISTER_PROD_INNER(long, long);
REGISTER_PROD_INNER(short, short);
REGISTER_PROD_INNER(short, long);
REGISTER_PROD_INNER(char, char);
REGISTER_PROD_INNER(char, long);
REGISTER_PROD_INNER(uchar, uchar);
REGISTER_PROD_INNER(uchar, long);
REGISTER_PROD_INNER(bool, long);
REGISTER_PROD_INNER(bool, int);

// ============================================================================
// Welford reduction kernels (var / std / var_mean / std_mean)
// ============================================================================

inline float3 simd_welford_combine(float3 stats) {
  for (ushort i = simdgroup_size / 2; i > 0; i /= 2) {
    float3 other;
    other.x = ::metal::simd_shuffle_and_fill_down(stats.x, 0.0f, i);
    other.y = ::metal::simd_shuffle_and_fill_down(stats.y, 0.0f, i);
    other.z = ::metal::simd_shuffle_and_fill_down(stats.z, 0.0f, i);
    stats = welford_combine(stats, other);
  }
  return stats;
}

struct WelfordConfig {
  float correction;
  float compute_std;
  float write_mean;
};

template <typename TI, typename TO>
kernel void welford_reduction(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    device TO* output_mean [[buffer(2)]],
    constant NormParams<>& params [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simdgroup_size_val [[threads_per_simdgroup]]) {
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

  float w_mean = 0, w_m2 = 0, w_count = 0;
  const uint32_t rsize = params.reduction_size;

  if (num_reduced_dims <= 1) {
    for (uint32_t k = tid; k < rsize; k += tptg) {
      float val = static_cast<float>(input[input_base + k * reduction_stride]);
      w_count += 1;
      float delta = val - w_mean;
      w_mean += delta / w_count;
      w_m2 += delta * (val - w_mean);
    }
  } else {
    for (uint32_t k = tid; k < rsize; k += tptg) {
      float val = static_cast<float>(input[get_input_offset(k, tgid, params)]);
      w_count += 1;
      float delta = val - w_mean;
      w_mean += delta / w_count;
      w_m2 += delta * (val - w_mean);
    }
  }

  float3 stats = simd_welford_combine(float3(w_mean, w_m2, w_count));

  threadgroup float3 shared_stats[MAX_THREADGROUP_SIZE / 32];
  uint num_simdgroups = ceil_div(tptg, simdgroup_size_val);

  if (num_simdgroups > 1) {
    if (simd_lane_id == 0) {
      shared_stats[simdgroup_id] = stats;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_simdgroups) {
      stats = shared_stats[tid];
    } else {
      stats = float3(0, 0, 0);
    }
    stats = simd_welford_combine(stats);
  }

  if (tid == 0) {
    float denom = max(stats.z - config.correction, 0.0f);
    float var = (denom > 0) ? stats.y / denom : (stats.y > 0 ? INFINITY : NAN);  // denom==0 (correction>=N): match CPU IEEE (inf), not Metal fast-math nan

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

    output[output_offset] =
        static_cast<TO>(config.compute_std > 0 ? ::precise::sqrt(var) : var);
    if (config.write_mean > 0) {
      output_mean[output_offset] = static_cast<TO>(stats.x);
    }
  }
}

template <typename TI, typename TO, uint TG_X = 32, uint TG_Y = 32>
kernel void welford_reduction_outer(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    device TO* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]) {
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint out_stride = sizes.z;

  uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= N)
    return;

  uint rows_per_y = ceil_div(M, TG_Y);
  uint row_start = tid_tg.y * rows_per_y;
  uint row_end = min(row_start + rows_per_y, M);

  float w_mean = 0, w_m2 = 0, w_count = 0;
  for (uint row = row_start; row < row_end; row++) {
    float val = static_cast<float>(input[row * N + col]);
    w_count += 1;
    float delta = val - w_mean;
    w_mean += delta / w_count;
    w_m2 += delta * (val - w_mean);
  }

  threadgroup float3 shmem[TG_Y][TG_X];
  shmem[tid_tg.y][tid_tg.x] = float3(w_mean, w_m2, w_count);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = TG_Y / 2; s > 0; s >>= 1) {
    if (tid_tg.y < s)
      shmem[tid_tg.y][tid_tg.x] = welford_combine(
          shmem[tid_tg.y][tid_tg.x], shmem[tid_tg.y + s][tid_tg.x]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid_tg.y == 0) {
    float3 stats = shmem[0][tid_tg.x];
    float denom = max(stats.z - config.correction, 0.0f);
    float var = (denom > 0) ? stats.y / denom : (stats.y > 0 ? INFINITY : NAN);  // denom==0 (correction>=N): match CPU IEEE (inf), not Metal fast-math nan
    output[col * out_stride] =
        static_cast<TO>(config.compute_std > 0 ? ::precise::sqrt(var) : var);
    if (config.write_mean > 0) {
      output_mean[col * out_stride] = static_cast<TO>(stats.x);
    }
  }
}

template <typename TI, typename TO>
kernel void welford_reduction_inner(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    device TO* output_mean [[buffer(2)]],
    constant uint2& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_simd_groups = tptg / 32;

  // One SIMD group (32 lanes) reduces one row; num_simd_groups rows share a TG.
  // This keeps every lane busy on small rows and bounds the threadgroup count
  // (M/num_simd_groups) instead of launching one whole TG per row, which idles
  // most threads and over-subscribes the GPU for many short rows.
  uint row = tgid * num_simd_groups + simdgroup_id;
  if (row >= M)
    return;

  constant TI* row_ptr = input + row * N;

  constexpr uint NCHAINS = 4;
  float w_mean[NCHAINS] = {0}, w_m2[NCHAINS] = {0}, w_count[NCHAINS] = {0};
  // Each lane reads NCHAINS consecutive elements, blocks strided by 32*NCHAINS
  // (coalesced across the SIMD group), matching sum_reduction_inner.
  const uint stride = 32 * NCHAINS;
  const uint aligned_N = (N / stride) * stride;
  uint base = simd_lane_id * NCHAINS;
  for (; base < aligned_N; base += stride) {
    for (uint c = 0; c < NCHAINS; c++) {
      float val = static_cast<float>(row_ptr[base + c]);
      w_count[c] += 1;
      float delta = val - w_mean[c];
      w_mean[c] += delta / w_count[c];
      w_m2[c] += delta * (val - w_mean[c]);
    }
  }
  // Tail: leftover elements after the last full block, one per lane.
  for (uint i = aligned_N + simd_lane_id; i < N; i += 32) {
    float val = static_cast<float>(row_ptr[i]);
    w_count[0] += 1;
    float delta = val - w_mean[0];
    w_mean[0] += delta / w_count[0];
    w_m2[0] += delta * (val - w_mean[0]);
  }
  // Merge chains
  float3 merged = float3(w_mean[0], w_m2[0], w_count[0]);
  for (uint c = 1; c < NCHAINS; c++) {
    merged = welford_combine(merged, float3(w_mean[c], w_m2[c], w_count[c]));
  }
  // Reduce across the 32 lanes of this SIMD group (no shared memory needed).
  float3 stats = simd_welford_combine(merged);

  if (simd_lane_id == 0) {
    float denom = max(stats.z - config.correction, 0.0f);
    float var = (denom > 0) ? stats.y / denom : (stats.y > 0 ? INFINITY : NAN);  // denom==0 (correction>=N): match CPU IEEE (inf), not Metal fast-math nan
    output[row] =
        static_cast<TO>(config.compute_std > 0 ? ::precise::sqrt(var) : var);
    if (config.write_mean > 0) {
      output_mean[row] = static_cast<TO>(stats.x);
    }
  }
}

// ============================================================================
// Welford 2-pass for all-reduce: pass1 writes per-TG (mean, M2, count)
// triplet; pass2 combines triplets and produces final var/std/mean.
// Used when output.numel() == 1 and N is large enough to benefit from
// multiple threadgroups working in parallel.
// ============================================================================

template <typename TI>
kernel void welford_reduction_pass1(
    constant TI* input [[buffer(0)]],
    device float3* partials [[buffer(1)]],
    constant uint2& sizes [[buffer(2)]], // .x = elems_per_group, .y = total N
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  const uint32_t elems_per_group = sizes.x;
  const uint32_t total_N = sizes.y;
  const uint32_t group_start = tgid * elems_per_group;
  const uint32_t group_end = min(group_start + elems_per_group, total_N);

  // Accumulate this thread's local sum / sum-of-squares (cheap FMAs, no
  // per-element division), shifted by a data value so the sum-of-squares stays
  // numerically safe regardless of the true mean. Convert to a Welford triplet
  // with a single division; the cross-thread combine below uses the stable
  // Welford merge. This avoids ~N per-element divisions in the hot loop, which
  // is what made the Welford all-reduce slower than the plain sum all-reduce.
  float shift = (group_start + tid < group_end)
      ? static_cast<float>(input[group_start + tid])
      : 0.0f;
  float w_sum = 0, w_sumsq = 0, w_count = 0;
  for (uint32_t k = group_start + tid; k < group_end; k += tptg) {
    float d = static_cast<float>(input[k]) - shift;
    w_sum += d;
    w_sumsq += d * d;
    w_count += 1;
  }
  float w_mean = 0, w_m2 = 0;
  if (w_count > 0) {
    float meand = w_sum / w_count;
    w_mean = shift + meand;
    w_m2 = w_sumsq - w_sum * meand;  // sum((x - mean)^2)
  }

  float3 stats = simd_welford_combine(float3(w_mean, w_m2, w_count));

  threadgroup float3 shared_stats[MAX_THREADGROUP_SIZE / 32];
  uint num_simdgroups = ceil_div(tptg, 32u);
  if (num_simdgroups > 1) {
    if (simd_lane_id == 0) {
      shared_stats[simdgroup_id] = stats;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_simdgroups) {
      stats = shared_stats[tid];
    } else {
      stats = float3(0, 0, 0);
    }
    stats = simd_welford_combine(stats);
  }

  if (tid == 0) {
    partials[tgid] = stats;
  }
}

// Pass2: reduce N partial triplets to one final triplet, then write
// var (or std) and optionally mean. Runs as a single threadgroup.
template <typename TO>
kernel void welford_reduction_pass2(
    device const float3* partials [[buffer(0)]],
    device TO* output [[buffer(1)]],
    device TO* output_mean [[buffer(2)]],
    constant uint& num_partials [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  float3 acc = float3(0, 0, 0);
  for (uint32_t i = tid; i < num_partials; i += tptg) {
    acc = welford_combine(acc, partials[i]);
  }
  float3 stats = simd_welford_combine(acc);
  threadgroup float3 shared_stats[MAX_THREADGROUP_SIZE / 32];
  uint num_simdgroups = ceil_div(tptg, 32u);
  if (num_simdgroups > 1) {
    if (simd_lane_id == 0) {
      shared_stats[simdgroup_id] = stats;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_simdgroups) {
      stats = shared_stats[tid];
    } else {
      stats = float3(0, 0, 0);
    }
    stats = simd_welford_combine(stats);
  }
  if (tid == 0) {
    float denom = max(stats.z - config.correction, 0.0f);
    float var = (denom > 0) ? stats.y / denom : (stats.y > 0 ? INFINITY : NAN);  // denom==0 (correction>=N): match CPU IEEE (inf), not Metal fast-math nan
    output[0] =
        static_cast<TO>(config.compute_std > 0 ? ::precise::sqrt(var) : var);
    if (config.write_mean > 0) {
      output_mean[0] = static_cast<TO>(stats.x);
    }
  }
}

// Wide inner-dim welford: one whole threadgroup (up to 1024 threads) reduces one
// row, via per-thread streaming Welford + shared-memory tree. Used when N is
// large / M is small (few but huge rows), where the simd-per-row variant would
// leave only 32 threads on a giant row (slow AND loses precision from a deep
// per-lane streaming accumulation). Matches the original reduction order.
template <typename TI, typename TO>
kernel void welford_reduction_inner_wide(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    device TO* output_mean [[buffer(2)]],
    constant uint2& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_simd_groups = tptg / 32;
  uint row = tgid;
  if (row >= M)
    return;
  constant TI* row_ptr = input + row * N;
  constexpr uint NCHAINS = 4;
  float w_mean[NCHAINS] = {0}, w_m2[NCHAINS] = {0}, w_count[NCHAINS] = {0};
  uint stride = tptg * NCHAINS;
  uint base = tid;
  for (uint i = base; i + (NCHAINS - 1) * tptg < N; i += stride) {
    for (uint c = 0; c < NCHAINS; c++) {
      float val = static_cast<float>(row_ptr[i + c * tptg]);
      w_count[c] += 1;
      float delta = val - w_mean[c];
      w_mean[c] += delta / w_count[c];
      w_m2[c] += delta * (val - w_mean[c]);
    }
  }
  for (uint i = base + (N / stride) * stride; i < N; i += tptg) {
    float val = static_cast<float>(row_ptr[i]);
    w_count[0] += 1;
    float delta = val - w_mean[0];
    w_mean[0] += delta / w_count[0];
    w_m2[0] += delta * (val - w_mean[0]);
  }
  float3 merged = float3(w_mean[0], w_m2[0], w_count[0]);
  for (uint c = 1; c < NCHAINS; c++) {
    merged = welford_combine(merged, float3(w_mean[c], w_m2[c], w_count[c]));
  }
  float3 stats = simd_welford_combine(merged);
  threadgroup float3 shared_stats[32];
  if (simd_lane_id == 0) {
    shared_stats[simdgroup_id] = stats;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_id == 0) {
    stats = (simd_lane_id < num_simd_groups) ? shared_stats[simd_lane_id]
                                             : float3(0, 0, 0);
    stats = simd_welford_combine(stats);
    if (simd_lane_id == 0) {
      float denom = max(stats.z - config.correction, 0.0f);
      float var = (denom > 0) ? stats.y / denom : (stats.y > 0 ? INFINITY : NAN);
      output[row] =
          static_cast<TO>(config.compute_std > 0 ? ::precise::sqrt(var) : var);
      if (config.write_mean > 0) {
        output_mean[row] = static_cast<TO>(stats.x);
      }
    }
  }
}

#define REGISTER_WELFORD_INNER_WIDE(TI, TO)                 \
  template [[host_name("welford_inner_wide_" #TI "_" #TO)]] \
  kernel void welford_reduction_inner_wide<TI, TO>(         \
      constant TI * input [[buffer(0)]],                    \
      device TO * output [[buffer(1)]],                     \
      device TO * output_mean [[buffer(2)]],                \
      constant uint2 & sizes [[buffer(3)]],                 \
      constant WelfordConfig & config [[buffer(4)]],        \
      uint tid [[thread_index_in_threadgroup]],             \
      uint tptg [[threads_per_threadgroup]],                \
      uint tgid [[threadgroup_position_in_grid]],           \
      uint simd_lane_id [[thread_index_in_simdgroup]],      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);
REGISTER_WELFORD_INNER_WIDE(float, float);
REGISTER_WELFORD_INNER_WIDE(half, half);
REGISTER_WELFORD_INNER_WIDE(half, float);
REGISTER_WELFORD_INNER_WIDE(bfloat, bfloat);
REGISTER_WELFORD_INNER_WIDE(bfloat, float);

#define REGISTER_WELFORD(TI, TO)                            \
  template [[host_name("welford_" #TI "_" #TO)]]            \
  kernel void welford_reduction<TI, TO>(                    \
      constant TI * input [[buffer(0)]],                    \
      device TO * output [[buffer(1)]],                     \
      device TO * output_mean [[buffer(2)]],                \
      constant NormParams<> & params [[buffer(3)]],         \
      constant WelfordConfig & config [[buffer(4)]],        \
      uint tid [[thread_position_in_threadgroup]],          \
      uint tptg [[threads_per_threadgroup]],                \
      uint tgid [[threadgroup_position_in_grid]],           \
      uint simd_lane_id [[thread_index_in_simdgroup]],      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]], \
      uint simdgroup_size_val [[threads_per_simdgroup]]);

#define REGISTER_WELFORD_OUTER(TI, TO)                 \
  template [[host_name("welford_outer_" #TI "_" #TO)]] \
  kernel void welford_reduction_outer<TI, TO, 32, 32>( \
      constant TI * input [[buffer(0)]],               \
      device TO * output [[buffer(1)]],                \
      device TO * output_mean [[buffer(2)]],           \
      constant uint3 & sizes [[buffer(3)]],            \
      constant WelfordConfig & config [[buffer(4)]],   \
      uint2 tid_tg [[thread_position_in_threadgroup]], \
      uint2 tg_pos [[threadgroup_position_in_grid]]);

#define REGISTER_WELFORD_INNER(TI, TO)                 \
  template [[host_name("welford_inner_" #TI "_" #TO)]] \
  kernel void welford_reduction_inner<TI, TO>(         \
      constant TI * input [[buffer(0)]],               \
      device TO * output [[buffer(1)]],                \
      device TO * output_mean [[buffer(2)]],           \
      constant uint2 & sizes [[buffer(3)]],            \
      constant WelfordConfig & config [[buffer(4)]],   \
      uint tid [[thread_index_in_threadgroup]],        \
      uint tptg [[threads_per_threadgroup]],           \
      uint tgid [[threadgroup_position_in_grid]],      \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

REGISTER_WELFORD(float, float);
REGISTER_WELFORD(half, half);
REGISTER_WELFORD(half, float);
REGISTER_WELFORD(bfloat, bfloat);
REGISTER_WELFORD(bfloat, float);

REGISTER_WELFORD_OUTER(float, float);

// 2-pass welford registrations (pass1 templated on TI; pass2 templated on TO).
#define REGISTER_WELFORD_PASS1(TI)                     \
  template [[host_name("welford_pass1_" #TI)]]         \
  kernel void welford_reduction_pass1<TI>(             \
      constant TI * input [[buffer(0)]],               \
      device float3 * partials [[buffer(1)]],          \
      constant uint2 & sizes [[buffer(2)]],            \
      uint tid [[thread_position_in_threadgroup]],     \
      uint tptg [[threads_per_threadgroup]],           \
      uint tgid [[threadgroup_position_in_grid]],      \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define REGISTER_WELFORD_PASS2(TO)                     \
  template [[host_name("welford_pass2_" #TO)]]         \
  kernel void welford_reduction_pass2<TO>(             \
      device const float3* partials [[buffer(0)]],     \
      device TO* output [[buffer(1)]],                 \
      device TO* output_mean [[buffer(2)]],            \
      constant uint& num_partials [[buffer(3)]],       \
      constant WelfordConfig& config [[buffer(4)]],    \
      uint tid [[thread_position_in_threadgroup]],     \
      uint tptg [[threads_per_threadgroup]],           \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

REGISTER_WELFORD_PASS1(float);
REGISTER_WELFORD_PASS1(half);
REGISTER_WELFORD_PASS1(bfloat);

REGISTER_WELFORD_PASS2(float);
REGISTER_WELFORD_PASS2(half);
REGISTER_WELFORD_PASS2(bfloat);

REGISTER_WELFORD_OUTER(half, half);
REGISTER_WELFORD_OUTER(half, float);
REGISTER_WELFORD_OUTER(bfloat, bfloat);
REGISTER_WELFORD_OUTER(bfloat, float);

REGISTER_WELFORD_INNER(float, float);
REGISTER_WELFORD_INNER(half, half);
REGISTER_WELFORD_INNER(half, float);
REGISTER_WELFORD_INNER(bfloat, bfloat);
REGISTER_WELFORD_INNER(bfloat, float);

// ============================================================================
// Arg-reduce kernels (argmax / argmin / max / min with indices)
// ============================================================================

template <typename TI, bool IS_MAX>
kernel void argreduce(
    constant TI* input [[buffer(0)]],
    device long* output_indices [[buffer(1)]],
    device TI* output_values [[buffer(2)]],
    constant NormParams<>& params [[buffer(3)]],
    constant uchar& write_values [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simdgroup_size_val [[threads_per_simdgroup]]) {
  using TA = opmath_t<TI>;

  TA best_val;
  if (IS_MAX) {
    best_val = ::metal::numeric_limits<TA>::lowest();
  } else {
    best_val = ::metal::numeric_limits<TA>::max();
  }
  long best_idx = 0;

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

  const uint32_t rsize = params.reduction_size;

  if (num_reduced_dims <= 1) {
    for (uint32_t k = tid; k < rsize; k += tptg) {
      TA val = static_cast<TA>(input[input_base + k * reduction_stride]);
      bool better = IS_MAX ? (val > best_val) : (val < best_val);
      if (better || ::metal::isnan(static_cast<float>(val))) {
        best_val = val;
        best_idx = k;
      }
    }
  } else {
    for (uint32_t k = tid; k < rsize; k += tptg) {
      TA val = static_cast<TA>(input[get_input_offset(k, tgid, params)]);
      bool better = IS_MAX ? (val > best_val) : (val < best_val);
      if (better || ::metal::isnan(static_cast<float>(val))) {
        best_val = val;
        best_idx = k;
      }
    }
  }

  threadgroup TA arg_data[32];
  threadgroup long idx_data[32];

  long result_idx;
  if (IS_MAX) {
    result_idx = c10::metal::threadgroup_argmax(
        arg_data, idx_data, best_val, best_idx, tid, tptg);
  } else {
    result_idx = c10::metal::threadgroup_argmin(
        arg_data, idx_data, best_val, best_idx, tid, tptg);
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
    output_indices[output_offset] = result_idx;
    if (write_values) {
      uint32_t val_input_offset;
      if (num_reduced_dims <= 1) {
        val_input_offset = input_base + result_idx * reduction_stride;
      } else {
        val_input_offset = get_input_offset(result_idx, tgid, params);
      }
      output_values[output_offset] = input[val_input_offset];
    }
  }
}

// Contiguous inner-dim arg-reduce: one SIMD group (32 lanes) reduces one row of
// N elements; num_simd_groups rows share a threadgroup. Avoids launching one TG
// per row (over-subscription + idle threads) for many short rows. Used when the
// reduction is over the contiguous innermost dim. Same lowest-index/NaN tie
// semantics as the generic kernel.
template <typename TI, bool IS_MAX>
kernel void argreduce_inner(
    constant TI* input [[buffer(0)]],
    device long* output_indices [[buffer(1)]],
    device TI* output_values [[buffer(2)]],
    constant uint2& sizes [[buffer(3)]],  // [M, N]
    constant uchar& write_values [[buffer(4)]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = opmath_t<TI>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_simd_groups = tptg / 32;
  uint row = tgid * num_simd_groups + simdgroup_id;
  if (row >= M)
    return;
  constant TI* row_ptr = input + row * N;

  TA best_val = IS_MAX ? ::metal::numeric_limits<TA>::lowest()
                       : ::metal::numeric_limits<TA>::max();
  long best_idx = 0;
  for (uint k = simd_lane_id; k < N; k += 32) {
    TA val = static_cast<TA>(row_ptr[k]);
    bool better = IS_MAX ? (val > best_val) : (val < best_val);
    if (better || ::metal::isnan(static_cast<float>(val))) {
      best_val = val;
      best_idx = k;
    }
  }
  auto rc = IS_MAX ? c10::metal::simd_argmax(best_val, best_idx)
                   : c10::metal::simd_argmin(best_val, best_idx);
  if (simd_lane_id == 0) {
    output_indices[row] = rc.second;
    if (write_values) {
      output_values[row] = static_cast<TI>(rc.first);
    }
  }
}

// Outer-dim arg-reduce (reduce dim 0 of a contiguous [M, N] tensor): TG_X
// columns x TG_Y row-workers per threadgroup, coalesced column reads, then a
// tree reduction across the TG_Y workers of each column. Avoids the generic
// kernel's one-TG-per-output + strided reads for many output columns.
template <typename TI, bool IS_MAX, uint TG_X, uint TG_Y>
kernel void argreduce_outer(
    constant TI* input [[buffer(0)]],
    device long* output_indices [[buffer(1)]],
    device TI* output_values [[buffer(2)]],
    constant uint2& sizes [[buffer(3)]],  // [M, N]
    constant uchar& write_values [[buffer(4)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = opmath_t<TI>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= N)
    return;

  uint rows_per_y = (M + TG_Y - 1) / TG_Y;
  uint row_start = tid_tg.y * rows_per_y;
  uint row_end = ::metal::min(row_start + rows_per_y, M);

  TA best_val = IS_MAX ? ::metal::numeric_limits<TA>::lowest()
                       : ::metal::numeric_limits<TA>::max();
  long best_idx = M;  // sentinel: a worker with no rows never wins (idx >= M)
  for (uint row = row_start; row < row_end; row++) {
    TA val = static_cast<TA>(input[row * N + col]);
    bool better = IS_MAX ? (val > best_val) : (val < best_val);
    if (best_idx == (long)M || better || ::metal::isnan(static_cast<float>(val))) {
      best_val = val;
      best_idx = row;
    }
  }

  threadgroup TA vmem[TG_Y][TG_X];
  threadgroup long imem[TG_Y][TG_X];
  vmem[tid_tg.y][tid_tg.x] = best_val;
  imem[tid_tg.y][tid_tg.x] = best_idx;
  ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);

  for (uint s = TG_Y / 2; s > 0; s >>= 1) {
    if (tid_tg.y < s) {
      TA cv = vmem[tid_tg.y][tid_tg.x];
      long ci = imem[tid_tg.y][tid_tg.x];
      TA ov = vmem[tid_tg.y + s][tid_tg.x];
      long oi = imem[tid_tg.y + s][tid_tg.x];
      bool c_valid = ci < (long)M, o_valid = oi < (long)M;
      bool c_nan = c_valid && ::metal::isnan(static_cast<float>(cv));
      bool o_nan = o_valid && ::metal::isnan(static_cast<float>(ov));
      bool take;
      if (!o_valid) {
        take = false;
      } else if (!c_valid) {
        take = true;
      } else if (o_nan != c_nan) {
        take = o_nan;  // NaN beats non-NaN (torch returns a NaN index)
      } else if (o_nan && c_nan) {
        take = oi < ci;  // both NaN: lowest index
      } else {
        bool better = IS_MAX ? (ov > cv) : (ov < cv);
        take = better || (ov == cv && oi < ci);  // tie -> lowest index
      }
      if (take) {
        vmem[tid_tg.y][tid_tg.x] = ov;
        imem[tid_tg.y][tid_tg.x] = oi;
      }
    }
    ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
  }

  if (tid_tg.y == 0) {
    output_indices[col] = imem[0][tid_tg.x];
    if (write_values) {
      output_values[col] = static_cast<TI>(vmem[0][tid_tg.x]);
    }
  }
}

// Fused 2-pass arg-reduce for large all-reduce (value+index). pass1: each TG
// reduces a contiguous chunk to a per-group (winning global index, value);
// pass2: a single TG picks the winning group. Only 2 dispatches (vs the
// reshape/gather/index_select chain), which matters at ~1M elems where the
// generic single-TG kernel is dispatch/parallelism-bound. Lowest index wins ties.
template <typename TI, bool IS_MAX>
kernel void argreduce_pass1(
    constant TI* input [[buffer(0)]],
    device TI* partials_val [[buffer(1)]],
    device long* partials_idx [[buffer(2)]],
    constant uint2& sizes [[buffer(3)]],  // [elems_per_group, total_N]
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = opmath_t<TI>;
  const uint epg = sizes.x;
  const uint total_N = sizes.y;
  const uint group_start = tgid * epg;
  const uint group_end = min(group_start + epg, total_N);

  TA best_val = IS_MAX ? ::metal::numeric_limits<TA>::lowest()
                       : ::metal::numeric_limits<TA>::max();
  long best_idx = group_start;
  for (uint k = group_start + tid; k < group_end; k += tptg) {
    TA val = static_cast<TA>(input[k]);
    bool better = IS_MAX ? (val > best_val) : (val < best_val);
    if (better || ::metal::isnan(static_cast<float>(val))) {
      best_val = val;
      best_idx = k;
    }
  }
  threadgroup TA arg_data[32];
  threadgroup long idx_data[32];
  long win = IS_MAX
      ? c10::metal::threadgroup_argmax(arg_data, idx_data, best_val, best_idx, tid, tptg)
      : c10::metal::threadgroup_argmin(arg_data, idx_data, best_val, best_idx, tid, tptg);
  if (tid == 0) {
    partials_idx[tgid] = win;
    partials_val[tgid] = input[win];
  }
}

template <typename TI, bool IS_MAX>
kernel void argreduce_pass2(
    device const TI* partials_val [[buffer(0)]],
    device const long* partials_idx [[buffer(1)]],
    device long* output_idx [[buffer(2)]],
    device TI* output_val [[buffer(3)]],
    constant uint& num_partials [[buffer(4)]],
    constant uchar& write_values [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = opmath_t<TI>;
  TA best_val = IS_MAX ? ::metal::numeric_limits<TA>::lowest()
                       : ::metal::numeric_limits<TA>::max();
  long best_pos = 0;
  for (uint i = tid; i < num_partials; i += tptg) {
    TA val = static_cast<TA>(partials_val[i]);
    bool better = IS_MAX ? (val > best_val) : (val < best_val);
    if (better || ::metal::isnan(static_cast<float>(val))) {
      best_val = val;
      best_pos = (long)i;
    }
  }
  threadgroup TA arg_data[32];
  threadgroup long idx_data[32];
  long win_pos = IS_MAX
      ? c10::metal::threadgroup_argmax(arg_data, idx_data, best_val, best_pos, tid, tptg)
      : c10::metal::threadgroup_argmin(arg_data, idx_data, best_val, best_pos, tid, tptg);
  if (tid == 0) {
    output_idx[0] = partials_idx[win_pos];
    if (write_values) {
      output_val[0] = partials_val[win_pos];
    }
  }
}

#define REGISTER_ARGREDUCE(TI, NAME, IS_MAX)                \
  template [[host_name(NAME "_" #TI)]]                      \
  kernel void argreduce<TI, IS_MAX>(                        \
      constant TI * input [[buffer(0)]],                    \
      device long* output_indices [[buffer(1)]],            \
      device TI* output_values [[buffer(2)]],               \
      constant NormParams<>& params [[buffer(3)]],          \
      constant uchar& write_values [[buffer(4)]],           \
      uint tid [[thread_position_in_threadgroup]],          \
      uint tptg [[threads_per_threadgroup]],                \
      uint tgid [[threadgroup_position_in_grid]],           \
      uint simd_lane_id [[thread_index_in_simdgroup]],      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]], \
      uint simdgroup_size_val [[threads_per_simdgroup]]);

REGISTER_ARGREDUCE(float, "argmax", true);
REGISTER_ARGREDUCE(float, "argmin", false);
REGISTER_ARGREDUCE(half, "argmax", true);
REGISTER_ARGREDUCE(half, "argmin", false);
REGISTER_ARGREDUCE(bfloat, "argmax", true);
REGISTER_ARGREDUCE(bfloat, "argmin", false);
REGISTER_ARGREDUCE(int, "argmax", true);
REGISTER_ARGREDUCE(int, "argmin", false);
REGISTER_ARGREDUCE(long, "argmax", true);
REGISTER_ARGREDUCE(long, "argmin", false);
REGISTER_ARGREDUCE(short, "argmax", true);
REGISTER_ARGREDUCE(short, "argmin", false);
REGISTER_ARGREDUCE(char, "argmax", true);
REGISTER_ARGREDUCE(char, "argmin", false);
REGISTER_ARGREDUCE(uchar, "argmax", true);
REGISTER_ARGREDUCE(uchar, "argmin", false);

#define REGISTER_ARGREDUCE_INNER(TI, NAME, IS_MAX)         \
  template [[host_name(NAME "_inner_" #TI)]]               \
  kernel void argreduce_inner<TI, IS_MAX>(                 \
      constant TI * input [[buffer(0)]],                   \
      device long* output_indices [[buffer(1)]],           \
      device TI* output_values [[buffer(2)]],              \
      constant uint2& sizes [[buffer(3)]],                 \
      constant uchar& write_values [[buffer(4)]],          \
      uint tptg [[threads_per_threadgroup]],               \
      uint tgid [[threadgroup_position_in_grid]],          \
      uint simd_lane_id [[thread_index_in_simdgroup]],     \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);
REGISTER_ARGREDUCE_INNER(float, "argmax", true);
REGISTER_ARGREDUCE_INNER(float, "argmin", false);
REGISTER_ARGREDUCE_INNER(half, "argmax", true);
REGISTER_ARGREDUCE_INNER(half, "argmin", false);
REGISTER_ARGREDUCE_INNER(bfloat, "argmax", true);
REGISTER_ARGREDUCE_INNER(bfloat, "argmin", false);
REGISTER_ARGREDUCE_INNER(int, "argmax", true);
REGISTER_ARGREDUCE_INNER(int, "argmin", false);
REGISTER_ARGREDUCE_INNER(long, "argmax", true);
REGISTER_ARGREDUCE_INNER(long, "argmin", false);
REGISTER_ARGREDUCE_INNER(short, "argmax", true);
REGISTER_ARGREDUCE_INNER(short, "argmin", false);
REGISTER_ARGREDUCE_INNER(char, "argmax", true);
REGISTER_ARGREDUCE_INNER(char, "argmin", false);
REGISTER_ARGREDUCE_INNER(uchar, "argmax", true);
REGISTER_ARGREDUCE_INNER(uchar, "argmin", false);


#define REGISTER_ARGREDUCE_OUTER(TI, NAME, IS_MAX)         \
  template [[host_name(NAME "_outer_" #TI)]]               \
  kernel void argreduce_outer<TI, IS_MAX, 32, 32>(         \
      constant TI * input [[buffer(0)]],                   \
      device long* output_indices [[buffer(1)]],           \
      device TI* output_values [[buffer(2)]],              \
      constant uint2& sizes [[buffer(3)]],                 \
      constant uchar& write_values [[buffer(4)]],          \
      uint2 tid_tg [[thread_position_in_threadgroup]],     \
      uint2 tg_pos [[threadgroup_position_in_grid]]);
REGISTER_ARGREDUCE_OUTER(float, "argmax", true);
REGISTER_ARGREDUCE_OUTER(float, "argmin", false);
REGISTER_ARGREDUCE_OUTER(half, "argmax", true);
REGISTER_ARGREDUCE_OUTER(half, "argmin", false);
REGISTER_ARGREDUCE_OUTER(bfloat, "argmax", true);
REGISTER_ARGREDUCE_OUTER(bfloat, "argmin", false);
REGISTER_ARGREDUCE_OUTER(int, "argmax", true);
REGISTER_ARGREDUCE_OUTER(int, "argmin", false);
REGISTER_ARGREDUCE_OUTER(long, "argmax", true);
REGISTER_ARGREDUCE_OUTER(long, "argmin", false);
REGISTER_ARGREDUCE_OUTER(short, "argmax", true);
REGISTER_ARGREDUCE_OUTER(short, "argmin", false);
REGISTER_ARGREDUCE_OUTER(char, "argmax", true);
REGISTER_ARGREDUCE_OUTER(char, "argmin", false);
REGISTER_ARGREDUCE_OUTER(uchar, "argmax", true);
REGISTER_ARGREDUCE_OUTER(uchar, "argmin", false);


#define REGISTER_ARGREDUCE_2PASS(TI, NAME, IS_MAX)         \
  template [[host_name(NAME "_pass1_" #TI)]]               \
  kernel void argreduce_pass1<TI, IS_MAX>(                 \
      constant TI * input [[buffer(0)]],                   \
      device TI* partials_val [[buffer(1)]],               \
      device long* partials_idx [[buffer(2)]],             \
      constant uint2& sizes [[buffer(3)]],                 \
      uint tid [[thread_position_in_threadgroup]],         \
      uint tptg [[threads_per_threadgroup]],               \
      uint tgid [[threadgroup_position_in_grid]],          \
      uint simd_lane_id [[thread_index_in_simdgroup]],     \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]); \
  template [[host_name(NAME "_pass2_" #TI)]]               \
  kernel void argreduce_pass2<TI, IS_MAX>(                 \
      device const TI* partials_val [[buffer(0)]],         \
      device const long* partials_idx [[buffer(1)]],       \
      device long* output_idx [[buffer(2)]],               \
      device TI* output_val [[buffer(3)]],                 \
      constant uint& num_partials [[buffer(4)]],           \
      constant uchar& write_values [[buffer(5)]],          \
      uint tid [[thread_position_in_threadgroup]],         \
      uint tptg [[threads_per_threadgroup]],               \
      uint simd_lane_id [[thread_index_in_simdgroup]],     \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);
REGISTER_ARGREDUCE_2PASS(float, "argmax", true);
REGISTER_ARGREDUCE_2PASS(float, "argmin", false);
REGISTER_ARGREDUCE_2PASS(half, "argmax", true);
REGISTER_ARGREDUCE_2PASS(half, "argmin", false);
REGISTER_ARGREDUCE_2PASS(bfloat, "argmax", true);
REGISTER_ARGREDUCE_2PASS(bfloat, "argmin", false);
REGISTER_ARGREDUCE_2PASS(int, "argmax", true);
REGISTER_ARGREDUCE_2PASS(int, "argmin", false);
REGISTER_ARGREDUCE_2PASS(long, "argmax", true);
REGISTER_ARGREDUCE_2PASS(long, "argmin", false);
REGISTER_ARGREDUCE_2PASS(short, "argmax", true);
REGISTER_ARGREDUCE_2PASS(short, "argmin", false);
REGISTER_ARGREDUCE_2PASS(char, "argmax", true);
REGISTER_ARGREDUCE_2PASS(char, "argmin", false);
REGISTER_ARGREDUCE_2PASS(uchar, "argmax", true);
REGISTER_ARGREDUCE_2PASS(uchar, "argmin", false);

// Small-M variants: welford_outer / prod_outer with TG_Y matching M.
// Avoids the 75-87% idle-thread waste of TG_Y=32 when M is 8 or 16.

template [[host_name("welford_outer_8_float_float")]]
kernel void welford_reduction_outer<float, float, 128, 8>(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);
template [[host_name("welford_outer_8_half_half")]]
kernel void welford_reduction_outer<half, half, 128, 8>(
    constant half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device half* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);
template [[host_name("welford_outer_8_half_float")]]
kernel void welford_reduction_outer<half, float, 128, 8>(
    constant half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);
template [[host_name("welford_outer_8_bfloat_bfloat")]]
kernel void welford_reduction_outer<bfloat, bfloat, 128, 8>(
    constant bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    device bfloat* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);
template [[host_name("welford_outer_8_bfloat_float")]]
kernel void welford_reduction_outer<bfloat, float, 128, 8>(
    constant bfloat* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);

template [[host_name("welford_outer_16_float_float")]]
kernel void welford_reduction_outer<float, float, 64, 16>(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);
template [[host_name("welford_outer_16_half_half")]]
kernel void welford_reduction_outer<half, half, 64, 16>(
    constant half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device half* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);
template [[host_name("welford_outer_16_half_float")]]
kernel void welford_reduction_outer<half, float, 64, 16>(
    constant half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);
template [[host_name("welford_outer_16_bfloat_bfloat")]]
kernel void welford_reduction_outer<bfloat, bfloat, 64, 16>(
    constant bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    device bfloat* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);
template [[host_name("welford_outer_16_bfloat_float")]]
kernel void welford_reduction_outer<bfloat, float, 64, 16>(
    constant bfloat* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);

template [[host_name("prod_reduction_outer_8_float_float")]]
kernel void prod_reduction_outer<float, float, 128, 8>(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);
template [[host_name("prod_reduction_outer_8_half_half")]]
kernel void prod_reduction_outer<half, half, 128, 8>(
    constant half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);
template [[host_name("prod_reduction_outer_8_bfloat_bfloat")]]
kernel void prod_reduction_outer<bfloat, bfloat, 128, 8>(
    constant bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);

template [[host_name("prod_reduction_outer_16_float_float")]]
kernel void prod_reduction_outer<float, float, 64, 16>(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);
template [[host_name("prod_reduction_outer_16_half_half")]]
kernel void prod_reduction_outer<half, half, 64, 16>(
    constant half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);
template [[host_name("prod_reduction_outer_16_bfloat_bfloat")]]
kernel void prod_reduction_outer<bfloat, bfloat, 64, 16>(
    constant bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]);

// Small-M outer prod (M <= 16): grid-stride over columns, each thread reduces
// COLS columns over all M rows. The generic outer kernel splits the tiny M
// across TG_Y row-workers and pays a shared-memory barrier tree; that overhead
// dominates when M is small. One-thread-per-column instead over-subscribes the
// GPU for large N. sum dim=0 uses a tensor-core simdgemm path that products
// cannot, so prod needs this dedicated layout. COLS=2 won a config sweep.
template <typename TI, typename TO, uint COLS = 2>
kernel void prod_reduction_outer_smallm(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]]) {
  using TA = opmath_t<TO>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint out_stride = sizes.z;
  for (uint col = gid; col < N; col += gsize) {
    TA acc = 1;
    for (uint row = 0; row < M; row++) {
      acc *= static_cast<TA>(input[row * N + col]);
    }
    output[col * out_stride] = static_cast<TO>(acc);
  }
}

#define REGISTER_PROD_OUTER_SMALLM(TI, TO)                             template [[host_name("prod_reduction_outer_smallm_" #TI "_" #TO)]]   kernel void prod_reduction_outer_smallm<TI, TO, 2>(                      constant TI* input [[buffer(0)]],                                    device TO* output [[buffer(1)]],                                     constant uint3& sizes [[buffer(2)]],                                 uint gid [[thread_position_in_grid]],                                uint gsize [[threads_per_grid]]);
REGISTER_PROD_OUTER_SMALLM(float, float);
REGISTER_PROD_OUTER_SMALLM(half, half);
REGISTER_PROD_OUTER_SMALLM(bfloat, bfloat);

// ============================================================================
// simdgroup_matrix-accelerated welford for dim=0 reduction with small M.
// Each simdgroup processes CHUNKS_PER_SG 8-col chunks sequentially, so TG
// covers SG_PER_TG * CHUNKS_PER_SG * 8 cols regardless of N. This keeps the
// TG count proportional to N / 256 instead of N / 32, avoiding the
// TG-scheduling overhead that hurt v1/v2 on huge-N shapes.
//   col_sum[j]  = (ones[8x8] @ x[8x8])[0,j]
//   col_ssq[j]  = (x[8x8]^T @ x[8x8])[j,j]   (diagonal)
// ============================================================================

template <typename TI, typename TO>
kernel void welford_reduction_outer_simdgemm(
    device const TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    device TO* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint tg_x [[threadgroup_position_in_grid]]) {
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint out_stride = sizes.z;

  constexpr uint COLS_PER_CHUNK = 8;
  constexpr uint SG_PER_TG = 4;
  constexpr uint CHUNKS_PER_SG = 16;
  constexpr uint COLS_PER_TG =
      SG_PER_TG * CHUNKS_PER_SG * COLS_PER_CHUNK; // 256

  const uint sg_col_base =
      tg_x * COLS_PER_TG + sg_id * CHUNKS_PER_SG * COLS_PER_CHUNK;

  threadgroup float tg_sum[SG_PER_TG][CHUNKS_PER_SG][8];
  threadgroup float tg_ssq[SG_PER_TG][CHUNKS_PER_SG][8];
  threadgroup float tg_buf[SG_PER_TG][8][8];
  threadgroup TI tg_xscratch[SG_PER_TG][8][8];

  simdgroup_matrix<TI, 8, 8> ones = make_filled_simdgroup_matrix<TI, 8>(TI(1));

  for (uint chunk = 0; chunk < CHUNKS_PER_SG; chunk++) {
    const uint col_base = sg_col_base + chunk * COLS_PER_CHUNK;
    if (col_base >= N)
      break;

    simdgroup_matrix<float, 8, 8> sum_acc =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> ssq_acc =
        make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint row = 0; row + 8 <= M; row += 8) {
      simdgroup_matrix<TI, 8, 8> x_tile;
      simdgroup_load(x_tile, input + row * N + col_base, N);  // single global read
      simdgroup_multiply_accumulate(sum_acc, ones, x_tile, sum_acc);

      // Transpose via threadgroup scratch rather than a second (strided) global
      // load of the same tile — the kernel is memory-bound, so re-reading from
      // DRAM doubles traffic. simdgroup_barrier suffices (intra-simdgroup).
      simdgroup_store(x_tile, (threadgroup TI*)tg_xscratch[sg_id], 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_matrix<TI, 8, 8> x_T;
      simdgroup_load(x_T, (threadgroup TI*)tg_xscratch[sg_id], 8, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(ssq_acc, x_T, x_tile, ssq_acc);
    }

    // Each simdgroup's tg_buf slot is private; no race.
    simdgroup_store(sum_acc, (threadgroup float*)tg_buf[sg_id], 8);
    if (sg_lane < 8) {
      tg_sum[sg_id][chunk][sg_lane] = tg_buf[sg_id][0][sg_lane];
    }
    simdgroup_store(ssq_acc, (threadgroup float*)tg_buf[sg_id], 8);
    if (sg_lane < 8) {
      tg_ssq[sg_id][chunk][sg_lane] = tg_buf[sg_id][sg_lane][sg_lane];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // First 8 threads of each simdgroup walk through their CHUNKS_PER_SG
  // chunks and write outputs.
  if (sg_lane < 8) {
    for (uint chunk = 0; chunk < CHUNKS_PER_SG; chunk++) {
      const uint col_base = sg_col_base + chunk * COLS_PER_CHUNK;
      const uint out_col = col_base + sg_lane;
      if (out_col >= N)
        break;
      float total_sum = tg_sum[sg_id][chunk][sg_lane];
      float total_ssq = tg_ssq[sg_id][chunk][sg_lane];
      float mean = total_sum / float(M);
      float var_num =
          total_ssq - 2.0f * mean * total_sum + float(M) * mean * mean;
      float denom = max(float(M) - config.correction, 0.0f);
      float var = (denom > 0) ? var_num / denom : (var_num > 0 ? INFINITY : NAN);  // match CPU IEEE (inf)
      output[out_col * out_stride] =
          static_cast<TO>(config.compute_std > 0 ? ::precise::sqrt(var) : var);
      if (config.write_mean > 0) {
        output_mean[out_col * out_stride] = static_cast<TO>(mean);
      }
    }
  }
}

template [[host_name("welford_outer_simdgemm_float_float")]]
kernel void welford_reduction_outer_simdgemm<float, float>(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint tg_x [[threadgroup_position_in_grid]]);

template [[host_name("welford_outer_simdgemm_half_half")]]
kernel void welford_reduction_outer_simdgemm<half, half>(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device half* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint tg_x [[threadgroup_position_in_grid]]);

template [[host_name("welford_outer_simdgemm_half_float")]]
kernel void welford_reduction_outer_simdgemm<half, float>(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint tg_x [[threadgroup_position_in_grid]]);

template [[host_name("welford_outer_simdgemm_bfloat_bfloat")]]
kernel void welford_reduction_outer_simdgemm<bfloat, bfloat>(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    device bfloat* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint tg_x [[threadgroup_position_in_grid]]);

template [[host_name("welford_outer_simdgemm_bfloat_float")]]
kernel void welford_reduction_outer_simdgemm<bfloat, float>(
    device const bfloat* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint tg_x [[threadgroup_position_in_grid]]);

