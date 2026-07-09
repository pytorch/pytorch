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

    } else if (p == 1) {
      output_val += input_abs;

    } else if (p == 2) {
      output_val += input_abs * input_abs;

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
      output_val = (p == 2)
          ? static_cast<TA>(::precise::sqrt(output_val))
          : static_cast<TA>(::precise::pow(output_val, 1 / p));
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
// nonzero-as-one (count_nonzero), abs (L1 norm), or square (L2 norm).
enum LoadMode : uint {
  LOAD_IDENTITY = 0,
  LOAD_NAN_TO_ZERO = 1,
  LOAD_NONZERO = 2,
  LOAD_ABS = 3,
  LOAD_SQUARE = 4
};

// Finalize op applied to the accumulator (in opmath_t) before the output cast.
// FINAL_SQRT turns a sum-of-squares reduction into an L2 norm.
enum FinalizeOp : uint { FINAL_NONE = 0, FINAL_SQRT = 1 };

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

template <
    LoadMode MODE,
    typename TI,
    ::metal::enable_if_t<MODE == LOAD_ABS, bool> = true>
inline opmath_t<TI> load_val(TI v) {
  return static_cast<opmath_t<TI>>(
      ::precise::abs(static_cast<opmath_t<TI>>(v)));
}

template <
    LoadMode MODE,
    typename TI,
    ::metal::enable_if_t<MODE == LOAD_SQUARE, bool> = true>
inline opmath_t<TI> load_val(TI v) {
  auto r = static_cast<opmath_t<TI>>(v);
  return r * r;
}

template <
    FinalizeOp FINAL,
    typename T,
    ::metal::enable_if_t<FINAL == FINAL_NONE, bool> = true>
inline T finalize_val(T v) {
  return v;
}

template <
    FinalizeOp FINAL,
    typename T,
    ::metal::enable_if_t<FINAL == FINAL_SQRT, bool> = true>
inline T finalize_val(T v) {
  return static_cast<T>(::precise::sqrt(v));
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
    constant uint4& sizes [[buffer(2)]],
    constant float& divisor [[buffer(3)]], // >0 divides accumulator before cast
    constant uint4& strides [[buffer(4)]],
    uint3 tid_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = ::metal::conditional_t<MODE == LOAD_NONZERO, uint, opmath_t<TO>>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint out_stride = sizes.z;
  const uint num_segs = max(sizes.w, 1u);
  const uint sK = strides.x;
  const uint sB = strides.y;
  const uint batch_in = tg_pos.z * strides.z;

  uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= N)
    return;

  const uint seg_rows = ceil_div(M, num_segs);
  const uint seg_start = tg_pos.y * seg_rows;
  const uint seg_end = min(seg_start + seg_rows, M);
  uint rows_per_y = ceil_div(seg_rows, TG_Y);
  uint row_start = seg_start + tid_tg.y * rows_per_y;
  uint row_end = min(row_start + rows_per_y, seg_end);

  // Multiple accumulation chains for ILP
  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++)
    acc[j] = 0;
  const uint col_off = batch_in + col * sB;

  uint row = row_start;
  for (; row + NCHAINS <= row_end; row += NCHAINS) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] += load_val<MODE>(input[col_off + (row + j) * sK]);
    }
  }
  for (; row < row_end; row++) {
    acc[row % NCHAINS] += load_val<MODE>(input[col_off + row * sK]);
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
    const uint out_idx = (num_segs > 1) ? (tg_pos.y * N + col)
                                        : (tg_pos.z * N + col * out_stride);
    output[out_idx] = static_cast<TO>(final_val);
  }
}

template <
    typename TI,
    typename TO,
    uint TG_SIZE = 256,
    uint NCHAINS = SUM_NCHAINS,
    LoadMode MODE = LOAD_IDENTITY>
kernel void sum_reduction_narrow(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint4& sizes [[buffer(2)]],
    constant float& divisor [[buffer(3)]],
    uint3 tid_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = ::metal::conditional_t<MODE == LOAD_NONZERO, uint, opmath_t<TO>>;
  const uint tid = tid_tg.x;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_segs = max(sizes.w, 1u);

  const uint seg_rows = ceil_div(M, num_segs);
  const uint r0 = tg_pos.y * seg_rows;
  const uint r1 = min(r0 + seg_rows, M);
  const uint base = tg_pos.z * M * N + r0 * N;
  const uint count = (r0 < M) ? (r1 - r0) * N : 0u;

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = 0;
  }
  const uint stride = TG_SIZE * NCHAINS;
  uint k = tid;
  for (; k + (NCHAINS - 1) * TG_SIZE < count; k += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] += load_val<MODE>(input[base + k + j * TG_SIZE]);
    }
  }
  for (; k < count; k += TG_SIZE) {
    acc[0] += load_val<MODE>(input[base + k]);
  }
  TA sum = acc[0];
  for (uint j = 1; j < NCHAINS; j++) {
    sum += acc[j];
  }

  threadgroup TA shmem[TG_SIZE];
  shmem[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < N) {
    TA s = 0;
    for (uint t = tid; t < TG_SIZE; t += N) {
      s += shmem[t];
    }
    if (divisor > 0) {
      s /= static_cast<TA>(divisor);
    }
    const uint out_idx =
        (num_segs > 1) ? (tg_pos.y * N + tid) : (tg_pos.z * N + tid);
    output[out_idx] = static_cast<TO>(s);
  }
}

template <
    typename TI,
    typename TO,
    uint TG_SIZE = 256,
    uint NCHAINS = SUM_NCHAINS,
    LoadMode MODE = LOAD_IDENTITY>
kernel void sum_reduction_narrow_strided(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint4& sizes [[buffer(2)]],
    constant float& divisor [[buffer(3)]],
    constant uint3& strides [[buffer(4)]],
    uint3 tid_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = ::metal::conditional_t<MODE == LOAD_NONZERO, uint, opmath_t<TO>>;
  const uint tid = tid_tg.x;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_segs = max(sizes.w, 1u);
  const uint sK = strides.x;
  const uint sB = strides.y;

  const uint seg_rows = ceil_div(M, num_segs);
  const uint r0 = tg_pos.y * seg_rows;
  const uint r1 = min(r0 + seg_rows, M);
  const uint base = tg_pos.z * strides.z + r0 * sK;
  const uint count = (r0 < M) ? (r1 - r0) * N : 0u;

  const uint col_off = (tid % N) * sB;
  const uint rstep = TG_SIZE / N;
  uint row = tid / N;

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = 0;
  }
  const uint stride = TG_SIZE * NCHAINS;
  uint k = tid;
  for (; k + (NCHAINS - 1) * TG_SIZE < count; k += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] += load_val<MODE>(input[base + (row + j * rstep) * sK + col_off]);
    }
    row += rstep * NCHAINS;
  }
  for (; k < count; k += TG_SIZE) {
    acc[0] += load_val<MODE>(input[base + row * sK + col_off]);
    row += rstep;
  }
  TA sum = acc[0];
  for (uint j = 1; j < NCHAINS; j++) {
    sum += acc[j];
  }

  threadgroup TA shmem[TG_SIZE];
  shmem[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < N) {
    TA s = 0;
    for (uint t = tid; t < TG_SIZE; t += N) {
      s += shmem[t];
    }
    if (divisor > 0) {
      s /= static_cast<TA>(divisor);
    }
    const uint out_idx =
        (num_segs > 1) ? (tg_pos.y * N + tid) : (tg_pos.z * N + tid);
    output[out_idx] = static_cast<TO>(s);
  }
}

#define REGISTER_SUM_OUTER_IMPL(TI, TO, PREFIX, MODE)                       \
  template [[host_name(PREFIX "reduction_narrow_strided_" #TI "_" #TO)]]    \
  kernel void sum_reduction_narrow_strided<TI, TO, 256, SUM_NCHAINS, MODE>( \
      constant TI * input [[buffer(0)]],                                    \
      device TO * output [[buffer(1)]],                                     \
      constant uint4 & sizes [[buffer(2)]],                                 \
      constant float& divisor [[buffer(3)]],                                \
      constant uint3& strides [[buffer(4)]],                                \
      uint3 tid_tg [[thread_position_in_threadgroup]],                      \
      uint3 tg_pos [[threadgroup_position_in_grid]]);                       \
  template [[host_name(PREFIX "reduction_narrow_" #TI "_" #TO)]]            \
  kernel void sum_reduction_narrow<TI, TO, 256, SUM_NCHAINS, MODE>(         \
      constant TI * input [[buffer(0)]],                                    \
      device TO * output [[buffer(1)]],                                     \
      constant uint4 & sizes [[buffer(2)]],                                 \
      constant float& divisor [[buffer(3)]],                                \
      uint3 tid_tg [[thread_position_in_threadgroup]],                      \
      uint3 tg_pos [[threadgroup_position_in_grid]]);                       \
  template [[host_name(PREFIX "reduction_outer_" #TI "_" #TO)]]             \
  kernel void sum_reduction_outer<TI, TO, 32, 32, SUM_NCHAINS, MODE>(       \
      constant TI * input [[buffer(0)]],                                    \
      device TO * output [[buffer(1)]],                                     \
      constant uint4 & sizes [[buffer(2)]],                                 \
      constant float& divisor [[buffer(3)]],                                \
      constant uint4& strides [[buffer(4)]],                                \
      uint3 tid_tg [[thread_position_in_threadgroup]],                      \
      uint3 tg_pos [[threadgroup_position_in_grid]]);                       \
  template [[host_name(PREFIX "reduction_col_" #TI "_" #TO)]]               \
  kernel void sum_reduction_outer<TI, TO, 32, 1, SUM_NCHAINS, MODE>(        \
      constant TI * input [[buffer(0)]],                                    \
      device TO * output [[buffer(1)]],                                     \
      constant uint4 & sizes [[buffer(2)]],                                 \
      constant float& divisor [[buffer(3)]],                                \
      constant uint4& strides [[buffer(4)]],                                \
      uint3 tid_tg [[thread_position_in_threadgroup]],                      \
      uint3 tg_pos [[threadgroup_position_in_grid]]);

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
    LoadMode MODE = LOAD_IDENTITY,
    FinalizeOp FINAL = FINAL_NONE>
kernel void sum_reduction_inner(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint2& sizes [[buffer(2)]], // [M, N]
    constant float& divisor [[buffer(3)]], // >0 divides accumulator before cast
    constant uint2& strides [[buffer(4)]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = ::metal::conditional_t<MODE == LOAD_NONZERO, uint, opmath_t<TO>>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_simd_groups = tptg / 32;
  const uint sK = strides.x;
  uint row = tgid * num_simd_groups + simdgroup_id;
  if (row >= M)
    return;
  const uint row_base = row * strides.y;

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
      acc[j] += load_val<MODE>(input[row_base + (base + j) * sK]);
    }
  }
  // Tail: remaining elements after last full block, one per lane
  for (uint i = aligned_N + simd_lane_id; i < N; i += 32) {
    acc[0] += load_val<MODE>(input[row_base + i * sK]);
  }

  TA sum = acc[0];
  for (uint j = 1; j < NCHAINS; j++)
    sum += acc[j];

  sum = c10::metal::simd_sum(sum);

  if (simd_lane_id == 0) {
    if (divisor > 0) {
      sum /= static_cast<TA>(divisor);
    }
    output[row] = static_cast<TO>(finalize_val<FINAL>(sum));
  }
}

constant constexpr uint SMALL_INNER_SHMEM = 2048;
template <typename TI, typename TO, LoadMode MODE = LOAD_IDENTITY>
kernel void sum_reduction_inner_small(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]],
    constant float& divisor [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  using TA = ::metal::conditional_t<MODE == LOAD_NONZERO, uint, opmath_t<TO>>;
  const uint M = sizes.x;
  const uint K = sizes.y;
  const uint RPT = sizes.z;
  const uint row0 = tgid * RPT;
  if (row0 >= M) {
    return;
  }
  const uint rows = min(RPT, M - row0);
  const uint strip = rows * K;

  using V = vec4type_t<TI>;
  threadgroup TI sh[SMALL_INNER_SHMEM];
  constant V* vin = reinterpret_cast<constant V*>(input + row0 * K);
  const uint nvec = strip / 4;
  for (uint i = tid; i < nvec; i += tptg) {
    const V v = vin[i];
    const uint b = i * 4;
    sh[b] = v.x;
    sh[b + 1] = v.y;
    sh[b + 2] = v.z;
    sh[b + 3] = v.w;
  }
  for (uint i = nvec * 4 + tid; i < strip; i += tptg) {
    sh[i] = input[row0 * K + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint r = tid; r < rows; r += tptg) {
    TA acc = 0;
    const uint base = r * K;
    for (uint j = 0; j < K; j++) {
      acc += load_val<MODE>(sh[base + j]);
    }
    if (divisor > 0) {
      acc /= static_cast<TA>(divisor);
    }
    output[row0 + r] = static_cast<TO>(acc);
  }
}

#define REGISTER_SUM_INNER_SMALL_IMPL(TI, TO, PREFIX, MODE)           \
  template [[host_name(PREFIX "reduction_inner_small_" #TI "_" #TO)]] \
  kernel void sum_reduction_inner_small<TI, TO, MODE>(                \
      constant TI * input [[buffer(0)]],                              \
      device TO * output [[buffer(1)]],                               \
      constant uint3 & sizes [[buffer(2)]],                           \
      constant float& divisor [[buffer(3)]],                          \
      uint tid [[thread_position_in_threadgroup]],                    \
      uint tptg [[threads_per_threadgroup]],                          \
      uint tgid [[threadgroup_position_in_grid]]);

#define REGISTER_SUM_INNER_IMPL(TI, TO, PREFIX, MODE, FINAL)         \
  template [[host_name(PREFIX "reduction_inner_" #TI "_" #TO)]]      \
  kernel void sum_reduction_inner<TI, TO, SUM_NCHAINS, MODE, FINAL>( \
      constant TI * input [[buffer(0)]],                             \
      device TO * output [[buffer(1)]],                              \
      constant uint2 & sizes [[buffer(2)]],                          \
      constant float& divisor [[buffer(3)]],                         \
      constant uint2& strides [[buffer(4)]],                         \
      uint tptg [[threads_per_threadgroup]],                         \
      uint tgid [[threadgroup_position_in_grid]],                    \
      uint simd_lane_id [[thread_index_in_simdgroup]],               \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define REGISTER_SUM_INNER(TI, TO) \
  REGISTER_SUM_INNER_IMPL(TI, TO, "sum_", LOAD_IDENTITY, FINAL_NONE)
#define REGISTER_NANSUM_INNER(TI, TO) \
  REGISTER_SUM_INNER_IMPL(TI, TO, "nansum_", LOAD_NAN_TO_ZERO, FINAL_NONE)
#define REGISTER_COUNT_NONZERO_INNER(TI) \
  REGISTER_SUM_INNER_IMPL(TI, long, "count_nonzero_", LOAD_NONZERO, FINAL_NONE)
#define REGISTER_NORM_INNER(TI, TO)                                 \
  REGISTER_SUM_INNER_IMPL(TI, TO, "norm_l1_", LOAD_ABS, FINAL_NONE) \
  REGISTER_SUM_INNER_IMPL(TI, TO, "norm_l2_", LOAD_SQUARE, FINAL_SQRT)

REGISTER_SUM_INNER(float, float);
REGISTER_SUM_INNER(half, half);
REGISTER_SUM_INNER(half, float);
REGISTER_SUM_INNER(bfloat, bfloat);
REGISTER_SUM_INNER(bfloat, float);
REGISTER_SUM_INNER_SMALL_IMPL(float, float, "sum_", LOAD_IDENTITY);
REGISTER_SUM_INNER_SMALL_IMPL(half, half, "sum_", LOAD_IDENTITY);
REGISTER_SUM_INNER_SMALL_IMPL(half, float, "sum_", LOAD_IDENTITY);
REGISTER_SUM_INNER_SMALL_IMPL(bfloat, bfloat, "sum_", LOAD_IDENTITY);
REGISTER_SUM_INNER_SMALL_IMPL(bfloat, float, "sum_", LOAD_IDENTITY);
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

REGISTER_NORM_INNER(float, float);
REGISTER_NORM_INNER(half, half);
REGISTER_NORM_INNER(bfloat, bfloat);

template <
    typename TI,
    typename TO,
    uint NCHAINS = SUM_NCHAINS,
    LoadMode MODE = LOAD_IDENTITY>
kernel void sum_reduction_vec(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint2& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  using TA = opmath_t<TO>;
  using V = vec4type_t<TI>;
  const uint E = params.y;
  const uint g_base = tgid * E;
  const uint n_vec = E / 4;
  constant V* vin = reinterpret_cast<constant V*>(input + g_base);

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = 0;
  }
  const uint stride = tptg * NCHAINS;
  uint i = tid;
  for (; i + (NCHAINS - 1) * tptg < n_vec; i += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      const V v = vin[i + j * tptg];
      acc[j] += load_val<MODE>(v.x) + load_val<MODE>(v.y) +
          load_val<MODE>(v.z) + load_val<MODE>(v.w);
    }
  }
  for (; i < n_vec; i += tptg) {
    const V v = vin[i];
    acc[0] += load_val<MODE>(v.x) + load_val<MODE>(v.y) + load_val<MODE>(v.z) +
        load_val<MODE>(v.w);
  }
  for (uint k = n_vec * 4 + tid; k < E; k += tptg) {
    acc[0] += load_val<MODE>(input[g_base + k]);
  }
  TA sum = acc[0];
  for (uint j = 1; j < NCHAINS; j++) {
    sum += acc[j];
  }
  threadgroup TA shared[MAX_THREADGROUP_SIZE / 32];
  const TA total = c10::metal::threadgroup_sum(shared, sum, tid, tptg);
  if (tid == 0) {
    output[tgid] = static_cast<TO>(total);
  }
}

#define REGISTER_SUM_VEC_IMPL(TI, TO, PREFIX, MODE)           \
  template [[host_name(PREFIX "reduction_vec_" #TI "_" #TO)]] \
  kernel void sum_reduction_vec<TI, TO, SUM_NCHAINS, MODE>(   \
      constant TI * input [[buffer(0)]],                      \
      device TO * output [[buffer(1)]],                       \
      constant uint2 & params [[buffer(2)]],                  \
      uint tid [[thread_position_in_threadgroup]],            \
      uint tptg [[threads_per_threadgroup]],                  \
      uint tgid [[threadgroup_position_in_grid]]);

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

#define REGISTER_SUM_VEC(TI, TO) \
  REGISTER_SUM_VEC_IMPL(TI, TO, "sum_", LOAD_IDENTITY)
#define REGISTER_NANSUM_VEC(TI, TO) \
  REGISTER_SUM_VEC_IMPL(TI, TO, "nansum_", LOAD_NAN_TO_ZERO)
REGISTER_SUM_VEC(float, float);
REGISTER_SUM_VEC(float, half);
REGISTER_SUM_VEC(float, bfloat);
REGISTER_SUM_VEC(half, half);
REGISTER_SUM_VEC(half, float);
REGISTER_SUM_VEC(bfloat, bfloat);
REGISTER_SUM_VEC(bfloat, float);
REGISTER_NANSUM_VEC(float, float);
REGISTER_NANSUM_VEC(half, half);
REGISTER_NANSUM_VEC(half, float);
REGISTER_NANSUM_VEC(bfloat, bfloat);
REGISTER_NANSUM_VEC(bfloat, float);

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

// Reduction op functors MaxOp / MinOp (identity, replace, combine,
// simd_reduce, threadgroup_reduce) live in c10/metal/reduction_utils.h so the
// inductor MPS codegen can reuse the same identity/replace pair; both are
// pulled in via the file-scope `using namespace c10::metal`.

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
    template <typename> class OpFn,
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
  using Op = OpFn<TA>;
  const TA identity_val = Op::identity();
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

template <
    template <typename> class OpFn,
    typename Load,
    typename TI,
    typename TO,
    uint NCHAINS = SUM_NCHAINS>
kernel void value_reduction_vec(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint2& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  using TA = TO;
  using Op = OpFn<TA>;
  using V = vec4type_t<TI>;
  const uint E = params.y;
  const uint g_base = tgid * E;
  const uint n_vec = E / 4;
  constant V* vin = reinterpret_cast<constant V*>(input + g_base);

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = Op::identity();
  }
  const uint stride = tptg * NCHAINS;
  uint i = tid;
  for (; i + (NCHAINS - 1) * tptg < n_vec; i += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      const V v = vin[i + j * tptg];
      acc[j] = Op::combine(acc[j], Load::template load<TA>(v.x));
      acc[j] = Op::combine(acc[j], Load::template load<TA>(v.y));
      acc[j] = Op::combine(acc[j], Load::template load<TA>(v.z));
      acc[j] = Op::combine(acc[j], Load::template load<TA>(v.w));
    }
  }
  for (; i < n_vec; i += tptg) {
    const V v = vin[i];
    acc[0] = Op::combine(acc[0], Load::template load<TA>(v.x));
    acc[0] = Op::combine(acc[0], Load::template load<TA>(v.y));
    acc[0] = Op::combine(acc[0], Load::template load<TA>(v.z));
    acc[0] = Op::combine(acc[0], Load::template load<TA>(v.w));
  }
  for (uint k = n_vec * 4 + tid; k < E; k += tptg) {
    acc[0] = Op::combine(acc[0], Load::template load<TA>(input[g_base + k]));
  }
  TA output_val = acc[0];
  for (uint j = 1; j < NCHAINS; j++) {
    output_val = Op::combine(output_val, acc[j]);
  }
  threadgroup TA shared_outputs[MAX_THREADGROUP_SIZE / simdgroup_size];
  output_val = Op::threadgroup_reduce(shared_outputs, output_val, tid, tptg);
  if (tid == 0) {
    output[tgid] = output_val;
  }
}

// Outer-dim variant: input is logically [M, N], reduce M down so output is
// [N]. TG_X threads cover adjacent output columns (coalesced reads), TG_Y
// threads split the M rows. Mirrors sum_reduction_outer; uses the same
// (Op, Load) abstraction as value_reduction.
template <
    template <typename> class OpFn,
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
    constant uint4& sizes [[buffer(2)]],
    constant uint4& strides [[buffer(3)]],
    uint3 tid_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = TO;
  using Op = OpFn<TA>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint out_stride = sizes.z;
  const uint num_segs = max(sizes.w, 1u);
  const uint sK = strides.x;
  const uint sB = strides.y;
  const uint batch_in = tg_pos.z * strides.z;

  uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= N) {
    return;
  }

  const uint seg_rows = ceil_div(M, num_segs);
  const uint seg_start = tg_pos.y * seg_rows;
  const uint seg_end = min(seg_start + seg_rows, M);
  uint rows_per_y = ceil_div(seg_rows, TG_Y);
  uint row_start = seg_start + tid_tg.y * rows_per_y;
  uint row_end = min(row_start + rows_per_y, seg_end);

  const TA identity_val = Op::identity();
  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = identity_val;
  }
  const uint col_off = batch_in + col * sB;

  uint row = row_start;
  for (; row + NCHAINS <= row_end; row += NCHAINS) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] = Op::combine(
          acc[j], Load::template load<TA>(input[col_off + (row + j) * sK]));
    }
  }
  for (; row < row_end; row++) {
    acc[row % NCHAINS] = Op::combine(
        acc[row % NCHAINS], Load::template load<TA>(input[col_off + row * sK]));
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
    const uint out_idx = (num_segs > 1) ? (tg_pos.y * N + col)
                                        : (tg_pos.z * N + col * out_stride);
    output[out_idx] = shmem[0][tid_tg.x];
  }
}

template <
    template <typename> class OpFn,
    typename Load,
    typename TI,
    typename TO,
    uint TG_SIZE = 256,
    uint NCHAINS = SUM_NCHAINS>
kernel void value_reduction_narrow(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint4& sizes [[buffer(2)]],
    uint3 tid_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = TO;
  using Op = OpFn<TA>;
  const uint tid = tid_tg.x;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_segs = max(sizes.w, 1u);

  const uint seg_rows = ceil_div(M, num_segs);
  const uint r0 = tg_pos.y * seg_rows;
  const uint r1 = min(r0 + seg_rows, M);
  const uint base = tg_pos.z * M * N + r0 * N;
  const uint count = (r0 < M) ? (r1 - r0) * N : 0u;

  const TA identity_val = Op::identity();
  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = identity_val;
  }
  const uint stride = TG_SIZE * NCHAINS;
  uint k = tid;
  for (; k + (NCHAINS - 1) * TG_SIZE < count; k += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] = Op::combine(
          acc[j], Load::template load<TA>(input[base + k + j * TG_SIZE]));
    }
  }
  for (; k < count; k += TG_SIZE) {
    acc[0] = Op::combine(acc[0], Load::template load<TA>(input[base + k]));
  }
  TA val = acc[0];
  for (uint j = 1; j < NCHAINS; j++) {
    val = Op::combine(val, acc[j]);
  }

  threadgroup TA shmem[TG_SIZE];
  shmem[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < N) {
    TA s = identity_val;
    for (uint t = tid; t < TG_SIZE; t += N) {
      s = Op::combine(s, shmem[t]);
    }
    const uint out_idx =
        (num_segs > 1) ? (tg_pos.y * N + tid) : (tg_pos.z * N + tid);
    output[out_idx] = s;
  }
}

// Inner-dim variant: input is logically [M, N], reduce N (innermost dim)
// so output is [M]. One SIMD group (32 lanes) handles one row, multiple
// SIMD groups per TG for occupancy. No shared memory needed since
// simd_reduce suffices for intra-row collapse. Mirrors sum_reduction_inner.
template <
    template <typename> class OpFn,
    typename Load,
    typename TI,
    typename TO,
    uint NCHAINS = SUM_NCHAINS>
kernel void value_reduction_inner(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint2& sizes [[buffer(2)]], // [M, N]
    constant uint2& strides [[buffer(3)]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = TO;
  using Op = OpFn<TA>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_simd_groups = tptg / simdgroup_size;

  uint row = tgid * num_simd_groups + simdgroup_id;
  if (row >= M) {
    return;
  }
  const uint sK = strides.x;
  const uint row_base = row * strides.y;

  const TA identity_val = Op::identity();
  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = identity_val;
  }

  const uint stride = simdgroup_size * NCHAINS;
  const uint aligned_N = (N / stride) * stride;
  uint base = simd_lane_id * NCHAINS;
  for (; base < aligned_N; base += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] = Op::combine(
          acc[j], Load::template load<TA>(input[row_base + (base + j) * sK]));
    }
  }
  for (uint i = aligned_N + simd_lane_id; i < N; i += simdgroup_size) {
    acc[0] =
        Op::combine(acc[0], Load::template load<TA>(input[row_base + i * sK]));
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

template <
    template <typename> class OpFn,
    typename Load,
    typename TI,
    typename TO>
kernel void value_reduction_inner_small(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  using TA = TO;
  using Op = OpFn<TA>;
  const uint M = sizes.x;
  const uint K = sizes.y;
  const uint RPT = sizes.z;
  const uint row0 = tgid * RPT;
  if (row0 >= M) {
    return;
  }
  const uint rows = min(RPT, M - row0);
  const uint strip = rows * K;

  using V = vec4type_t<TI>;
  threadgroup TA sh[SMALL_INNER_SHMEM];
  constant V* vin = reinterpret_cast<constant V*>(input + row0 * K);
  const uint nvec = strip / 4;
  for (uint i = tid; i < nvec; i += tptg) {
    const V v = vin[i];
    const uint b = i * 4;
    sh[b] = Load::template load<TA>(v.x);
    sh[b + 1] = Load::template load<TA>(v.y);
    sh[b + 2] = Load::template load<TA>(v.z);
    sh[b + 3] = Load::template load<TA>(v.w);
  }
  for (uint i = nvec * 4 + tid; i < strip; i += tptg) {
    sh[i] = Load::template load<TA>(input[row0 * K + i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint r = tid; r < rows; r += tptg) {
    TA acc = Op::identity();
    const uint base = r * K;
    for (uint j = 0; j < K; j++) {
      acc = Op::combine(acc, sh[base + j]);
    }
    output[row0 + r] = acc;
  }
}

#define REGISTER_VALUE_INNER_SMALL_IMPL(TI, TO, NAME, OP, LOAD)      \
  template [[host_name(NAME "_reduction_inner_small_" #TI "_" #TO)]] \
  kernel void value_reduction_inner_small<OP, LOAD, TI, TO>(         \
      constant TI * input [[buffer(0)]],                             \
      device TO * output [[buffer(1)]],                              \
      constant uint3 & sizes [[buffer(2)]],                          \
      uint tid [[thread_position_in_threadgroup]],                   \
      uint tptg [[threads_per_threadgroup]],                         \
      uint tgid [[threadgroup_position_in_grid]]);

#define REGISTER_VALUE_INNER_SMALL(T)                                    \
  REGISTER_VALUE_INNER_SMALL_IMPL(T, T, "max", MaxOp, IdentityLoad)      \
  REGISTER_VALUE_INNER_SMALL_IMPL(T, T, "min", MinOp, IdentityLoad)      \
  REGISTER_VALUE_INNER_SMALL_IMPL(T, uchar, "any", MaxOp, PredicateLoad) \
  REGISTER_VALUE_INNER_SMALL_IMPL(T, uchar, "all", MinOp, PredicateLoad)

REGISTER_VALUE_INNER_SMALL(float);
REGISTER_VALUE_INNER_SMALL(half);
REGISTER_VALUE_INNER_SMALL(bfloat);

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
      constant uint4 & sizes [[buffer(2)]],                                 \
      constant uint4 & strides [[buffer(3)]],                               \
      uint3 tid_tg [[thread_position_in_threadgroup]],                      \
      uint3 tg_pos [[threadgroup_position_in_grid]]);                       \
  template [[host_name(NAME "_reduction_col_" #TI "_" #TO)]]                \
  kernel void value_reduction_outer<OP, LOAD, TI, TO, 32, 1, SUM_NCHAINS>(  \
      constant TI * input [[buffer(0)]],                                    \
      device TO * output [[buffer(1)]],                                     \
      constant uint4 & sizes [[buffer(2)]],                                 \
      constant uint4 & strides [[buffer(3)]],                               \
      uint3 tid_tg [[thread_position_in_threadgroup]],                      \
      uint3 tg_pos [[threadgroup_position_in_grid]]);                       \
  template [[host_name(NAME "_reduction_narrow_" #TI "_" #TO)]]             \
  kernel void value_reduction_narrow<OP, LOAD, TI, TO, 256, SUM_NCHAINS>(   \
      constant TI * input [[buffer(0)]],                                    \
      device TO * output [[buffer(1)]],                                     \
      constant uint4 & sizes [[buffer(2)]],                                 \
      uint3 tid_tg [[thread_position_in_threadgroup]],                      \
      uint3 tg_pos [[threadgroup_position_in_grid]]);                       \
  template [[host_name(NAME "_reduction_inner_" #TI "_" #TO)]]              \
  kernel void value_reduction_inner<OP, LOAD, TI, TO, SUM_NCHAINS>(         \
      constant TI * input [[buffer(0)]],                                    \
      device TO * output [[buffer(1)]],                                     \
      constant uint2 & sizes [[buffer(2)]],                                 \
      constant uint2 & strides [[buffer(3)]],                               \
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

#define REGISTER_VALUE_VEC_IMPL(TI, TO, NAME, OP, LOAD)           \
  template [[host_name(NAME "_reduction_vec_" #TI "_" #TO)]]      \
  kernel void value_reduction_vec<OP, LOAD, TI, TO, SUM_NCHAINS>( \
      constant TI * input [[buffer(0)]],                          \
      device TO * output [[buffer(1)]],                           \
      constant uint2 & params [[buffer(2)]],                      \
      uint tid [[thread_position_in_threadgroup]],                \
      uint tptg [[threads_per_threadgroup]],                      \
      uint tgid [[threadgroup_position_in_grid]]);

#define REGISTER_VALUE_VEC(T)                                    \
  REGISTER_VALUE_VEC_IMPL(T, T, "max", MaxOp, IdentityLoad)      \
  REGISTER_VALUE_VEC_IMPL(T, T, "min", MinOp, IdentityLoad)      \
  REGISTER_VALUE_VEC_IMPL(T, uchar, "any", MaxOp, PredicateLoad) \
  REGISTER_VALUE_VEC_IMPL(T, uchar, "all", MinOp, PredicateLoad)

REGISTER_VALUE_VEC(float);
REGISTER_VALUE_VEC(half);
REGISTER_VALUE_VEC(bfloat);

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

// =============================================================================
// argmax/argmin: per output element find the (linear) index of the max/min
// input element along the reduced dim(s). Output is always int64. NaN
// propagates (first NaN in source order wins); on ties the lowest index wins.
// Mirrors the value_reduction layout but tracks a (value, index) pair instead
// of just a value.
// =============================================================================

// Arg-reductions reuse the same MaxOp / MinOp structs that drive value
// reductions: identity and simd_reduce for the value side, replace for the
// NaN-propagating per-thread scan and outer-kernel shared-memory tree reduce.

// SIMD argmin/argmax with proper pair tie-break (the c10::metal::simd_argmax
// helper ties on lowest LANE, which is wrong when a single lane has scanned
// multiple positions and stored a non-minimal index for the winning value).
// Two-step: simd-reduce the values, then simd_min the indices of lanes whose
// value matched the winner (NaN lanes count as winners on float). Returns
// (winner_value, min_winning_idx).
template <
    template <typename> class OpFn,
    typename TA,
    ::metal::enable_if_t<is_floating_point_v<TA>, bool> = true>
inline c10::metal::pair<TA, uint32_t> simd_arg_reduce(TA val, uint32_t idx) {
  using Op = OpFn<TA>;
  const TA winner = Op::simd_reduce(val);
  const bool is_winner = ::metal::isnan(val) || (val == winner);
  const uint32_t eff_idx =
      is_winner ? idx : ::metal::numeric_limits<uint32_t>::max();
  return {winner, ::metal::simd_min(eff_idx)};
}

template <
    template <typename> class OpFn,
    typename TA,
    ::metal::enable_if_t<!is_floating_point_v<TA>, bool> = true>
inline c10::metal::pair<TA, uint32_t> simd_arg_reduce(TA val, uint32_t idx) {
  using Op = OpFn<TA>;
  const TA winner = Op::simd_reduce(val);
  const uint32_t eff_idx =
      (val == winner) ? idx : ::metal::numeric_limits<uint32_t>::max();
  return {winner, ::metal::simd_min(eff_idx)};
}

// Pair tie-break for the shared-memory tree reduction in the outer-dim
// kernel: cand replaces cur if strictly better OR (equal AND lower idx).
// `Op::replace` already handles NaN propagation; equality here is the
// !replace-either-way fallback, which subsumes both both-NaN and both-equal
// cases.
template <template <typename> class OpFn, typename TA>
inline bool arg_replace(
    TA cand_val,
    uint32_t cand_idx,
    TA cur_val,
    uint32_t cur_idx) {
  using Op = OpFn<TA>;
  if (Op::replace(cand_val, cur_val)) {
    return true;
  }
  if (Op::replace(cur_val, cand_val)) {
    return false;
  }
  return cand_idx < cur_idx;
}

// Generic single-pass argmax/argmin. Each threadgroup computes one output
// element; the per-thread loop scans the reduction with strict-improvement
// updates (so on equal values the earlier index is kept), then a two-stage
// SIMD reduction collapses the per-thread (value, index) pairs.
//
// Lane-to-source-index ordering: tid t processes reduction indices
// {t, t+tptg, t+2*tptg, ...}, so within a simdgroup the lowest lane sees the
// lowest index. Across simdgroups, simdgroup_id 0 contains the lowest tids.
// Tie-break by lowest-lane in simd_argmax therefore matches PyTorch's
// "first occurrence wins" convention.
template <template <typename> class OpFn, typename TI>
[[max_total_threads_per_threadgroup(MAX_THREADGROUP_SIZE)]]
kernel void arg_reduction(
    constant TI* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant NormParams<>& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  using TA = opmath_t<TI>;
  using Op = OpFn<TA>;

  // Compute input_base and detect reduction pattern (mirrors value_reduction).
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

  TA best_val = Op::identity();
  uint32_t best_idx = 0;
  const uint32_t rsize = params.reduction_size;

  if (num_reduced_dims <= 1) {
    for (uint32_t idx = tid; idx < rsize; idx += tptg) {
      const TA val =
          static_cast<TA>(input[input_base + idx * reduction_stride]);
      if (Op::replace(val, best_val)) {
        best_val = val;
        best_idx = idx;
      }
    }
  } else {
    for (uint32_t idx = tid; idx < rsize; idx += tptg) {
      const TA val =
          static_cast<TA>(input[get_input_offset(idx, tgid, params)]);
      if (Op::replace(val, best_val)) {
        best_val = val;
        best_idx = idx;
      }
    }
  }

  // Two-stage SIMD reduction. Stage 1: each simdgroup picks its winner via
  // simd_arg_reduce (proper pair tie-break). If there's only one simdgroup,
  // we're done. Stage 2: all 32 lanes of simdgroup 0 load the per-simdgroup
  // winners (slots past the active count get identity + UINT_MAX idx so they
  // never win the value race and contribute UINT_MAX to the idx race).
  auto rc = simd_arg_reduce<OpFn>(best_val, best_idx);
  uint32_t result_idx = rc.second;

  if (tptg > simdgroup_size) {
    threadgroup TA shared_vals[MAX_THREADGROUP_SIZE / 32];
    threadgroup uint32_t shared_idxs[MAX_THREADGROUP_SIZE / 32];
    const uint32_t nsimd = tptg / simdgroup_size;
    if (tid % simdgroup_size == 0) {
      shared_vals[tid / simdgroup_size] = rc.first;
      shared_idxs[tid / simdgroup_size] = rc.second;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < simdgroup_size) {
      const TA v = (tid < nsimd) ? shared_vals[tid] : Op::identity();
      const uint32_t i = (tid < nsimd)
          ? shared_idxs[tid]
          : ::metal::numeric_limits<uint32_t>::max();
      auto rc2 = simd_arg_reduce<OpFn>(v, i);
      if (tid == 0) {
        shared_idxs[0] = rc2.second;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    result_idx = shared_idxs[0];
  }

  if (tid == 0) {
    uint32_t output_offset = 0;
    uint32_t reduction_idx = tgid;
    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      const auto output_dim_size = params.output_sizes[dim];
      if (output_dim_size > 1) {
        const auto index_in_dim = reduction_idx % output_dim_size;
        reduction_idx /= output_dim_size;
        output_offset += index_in_dim * params.output_strides[dim];
      }
    }
    output[output_offset] = static_cast<long>(result_idx);
  }
}

// Inner-dim arg-reduction: input is logically [M, N] contiguous, reduce N
// (innermost). One SIMD group (32 lanes) per row, multiple SIMD groups per
// TG for occupancy. Lane L scans positions {L, L+32, L+64, ...} of its row
// with strict-improvement updates (so the lane's stored idx is the lowest
// of its scanned positions matching the winning value). The cross-lane
// collapse uses simd_arg_reduce which ties on lowest IDX, not lowest LANE.
template <template <typename> class OpFn, typename TI>
kernel void arg_reduction_inner(
    constant TI* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant uint2& sizes [[buffer(2)]], // [M, N]
    constant uint2& strides [[buffer(3)]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = opmath_t<TI>;
  using Op = OpFn<TA>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_simd_groups = tptg / simdgroup_size;

  const uint row = tgid * num_simd_groups + simdgroup_id;
  if (row >= M) {
    return;
  }
  const uint sK = strides.x;
  const uint row_base = row * strides.y;

  TA best_val = Op::identity();
  uint32_t best_idx = 0;
  for (uint i = simd_lane_id; i < N; i += simdgroup_size) {
    const TA val = static_cast<TA>(input[row_base + i * sK]);
    if (Op::replace(val, best_val)) {
      best_val = val;
      best_idx = i;
    }
  }

  auto rc = simd_arg_reduce<OpFn>(best_val, best_idx);
  if (simd_lane_id == 0) {
    output[row] = static_cast<long>(rc.second);
  }
}

template <template <typename> class OpFn, typename TI>
kernel void arg_reduction_inner_p1(
    constant TI* input [[buffer(0)]],
    device opmath_t<TI>* val_out [[buffer(1)]],
    device int* idx_out [[buffer(2)]],
    constant uint4& sizes [[buffer(3)]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = opmath_t<TI>;
  using Op = OpFn<TA>;
  const uint MG = sizes.x;
  const uint C = sizes.y;
  const uint G = sizes.z;
  const uint num_simd_groups = tptg / simdgroup_size;
  const uint r = tgid * num_simd_groups + simdgroup_id;
  if (r >= MG) {
    return;
  }
  const uint chunk_off = (r % G) * C;
  constant TI* row_ptr = input + r * C;
  TA best_val = Op::identity();
  uint32_t best_idx = 0;
  for (uint i = simd_lane_id; i < C; i += simdgroup_size) {
    const TA val = static_cast<TA>(row_ptr[i]);
    if (Op::replace(val, best_val)) {
      best_val = val;
      best_idx = i;
    }
  }
  auto rc = simd_arg_reduce<OpFn>(best_val, best_idx);
  if (simd_lane_id == 0) {
    val_out[r] = rc.first;
    idx_out[r] = static_cast<int>(chunk_off + rc.second);
  }
}

template <template <typename> class OpFn, typename VT>
kernel void arg_combine_inner(
    constant VT* val_in [[buffer(0)]],
    constant int* idx_in [[buffer(1)]],
    device long* output [[buffer(2)]],
    constant uint2& sizes [[buffer(3)]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = VT;
  using Op = OpFn<TA>;
  const uint M = sizes.x;
  const uint G = sizes.y;
  const uint num_simd_groups = tptg / simdgroup_size;
  const uint row = tgid * num_simd_groups + simdgroup_id;
  if (row >= M) {
    return;
  }
  constant VT* vp = val_in + row * G;
  constant int* ip = idx_in + row * G;
  TA best_val = Op::identity();
  uint32_t best_gidx = ::metal::numeric_limits<uint32_t>::max();
  for (uint g = simd_lane_id; g < G; g += simdgroup_size) {
    const TA v = vp[g];
    const uint32_t gi = static_cast<uint32_t>(ip[g]);
    if (arg_replace<OpFn>(v, gi, best_val, best_gidx)) {
      best_val = v;
      best_gidx = gi;
    }
  }
  auto rc = simd_arg_reduce<OpFn>(best_val, best_gidx);
  if (simd_lane_id == 0) {
    output[row] = static_cast<long>(rc.second);
  }
}

// Outer-dim arg-reduction: input is logically [M, N] contiguous, reduce M
// down so output is [N]. TG_X threads cover adjacent output columns
// (coalesced reads), TG_Y threads split the M rows. Per-thread scan keeps
// the lowest row with the winning value; cross-worker tree reduction uses
// arg_replace (strictly-better OR equal-with-lower-idx).
template <
    template <typename> class OpFn,
    typename TI,
    uint TG_X = 32,
    uint TG_Y = 32>
[[max_total_threads_per_threadgroup(TG_X * TG_Y)]]
kernel void arg_reduction_outer(
    constant TI* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]], // [M, N, output_stride]
    constant uint4& strides [[buffer(3)]],
    uint3 tid_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = opmath_t<TI>;
  using Op = OpFn<TA>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint out_stride = sizes.z;
  const uint sK = strides.x;
  const uint sB = strides.y;
  const uint batch_in = tg_pos.z * strides.z;

  const uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= N) {
    return;
  }

  const uint rows_per_y = ceil_div(M, TG_Y);
  const uint row_start = tid_tg.y * rows_per_y;
  const uint row_end = min(row_start + rows_per_y, M);
  const uint col_off = batch_in + col * sB;

  TA best_val = Op::identity();
  uint32_t best_idx = 0;
  for (uint row = row_start; row < row_end; row++) {
    const TA val = static_cast<TA>(input[col_off + row * sK]);
    if (Op::replace(val, best_val)) {
      best_val = val;
      best_idx = row;
    }
  }

  threadgroup TA shared_vals[TG_Y][TG_X];
  threadgroup uint32_t shared_idxs[TG_Y][TG_X];
  shared_vals[tid_tg.y][tid_tg.x] = best_val;
  shared_idxs[tid_tg.y][tid_tg.x] = best_idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = TG_Y / 2; stride > 0; stride >>= 1) {
    if (tid_tg.y < stride) {
      const TA other_val = shared_vals[tid_tg.y + stride][tid_tg.x];
      const uint32_t other_idx = shared_idxs[tid_tg.y + stride][tid_tg.x];
      const TA self_val = shared_vals[tid_tg.y][tid_tg.x];
      const uint32_t self_idx = shared_idxs[tid_tg.y][tid_tg.x];
      if (arg_replace<OpFn>(other_val, other_idx, self_val, self_idx)) {
        shared_vals[tid_tg.y][tid_tg.x] = other_val;
        shared_idxs[tid_tg.y][tid_tg.x] = other_idx;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid_tg.y == 0) {
    output[tg_pos.z * N + col * out_stride] =
        static_cast<long>(shared_idxs[0][tid_tg.x]);
  }
}

template <
    template <typename> class OpFn,
    typename TI,
    uint TG_X = 32,
    uint TG_Y = 32>
[[max_total_threads_per_threadgroup(TG_X * TG_Y)]]
kernel void arg_reduction_outer_p1(
    constant TI* input [[buffer(0)]],
    device opmath_t<TI>* val_out [[buffer(1)]],
    device int* idx_out [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant uint2& strides [[buffer(4)]],
    uint3 tid_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = opmath_t<TI>;
  using Op = OpFn<TA>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint G = sizes.z;
  const uint sK = strides.x;
  const uint sB = strides.y;

  const uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= N) {
    return;
  }
  const uint seg = tg_pos.y;
  const uint seg_rows = ceil_div(M, G);
  const uint seg_start = seg * seg_rows;
  const uint seg_end = min(seg_start + seg_rows, M);

  const uint rows_per_y = ceil_div(seg_rows, TG_Y);
  const uint row_start = seg_start + tid_tg.y * rows_per_y;
  const uint row_end = min(row_start + rows_per_y, seg_end);
  const uint col_off = col * sB;

  TA best_val = Op::identity();
  uint32_t best_idx = ::metal::numeric_limits<uint32_t>::max();
  for (uint row = row_start; row < row_end; row++) {
    const TA val = static_cast<TA>(input[col_off + row * sK]);
    if (Op::replace(val, best_val)) {
      best_val = val;
      best_idx = row;
    }
  }

  threadgroup TA shared_vals[TG_Y][TG_X];
  threadgroup uint32_t shared_idxs[TG_Y][TG_X];
  shared_vals[tid_tg.y][tid_tg.x] = best_val;
  shared_idxs[tid_tg.y][tid_tg.x] = best_idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = TG_Y / 2; stride > 0; stride >>= 1) {
    if (tid_tg.y < stride) {
      const TA other_val = shared_vals[tid_tg.y + stride][tid_tg.x];
      const uint32_t other_idx = shared_idxs[tid_tg.y + stride][tid_tg.x];
      const TA self_val = shared_vals[tid_tg.y][tid_tg.x];
      const uint32_t self_idx = shared_idxs[tid_tg.y][tid_tg.x];
      if (arg_replace<OpFn>(other_val, other_idx, self_val, self_idx)) {
        shared_vals[tid_tg.y][tid_tg.x] = other_val;
        shared_idxs[tid_tg.y][tid_tg.x] = other_idx;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid_tg.y == 0) {
    val_out[col * G + seg] = shared_vals[0][tid_tg.x];
    idx_out[col * G + seg] = static_cast<int>(shared_idxs[0][tid_tg.x]);
  }
}

#define REGISTER_ARG_REDUCTION_IMPL(TI, NAME, OP)              \
  template [[host_name(NAME "_reduction_" #TI "_long")]]       \
  kernel void arg_reduction<OP, TI>(                           \
      constant TI * input [[buffer(0)]],                       \
      device long* output [[buffer(1)]],                       \
      constant NormParams<>& params [[buffer(2)]],             \
      uint tid [[thread_position_in_threadgroup]],             \
      uint tptg [[threads_per_threadgroup]],                   \
      uint tgid [[threadgroup_position_in_grid]]);             \
  template [[host_name(NAME "_reduction_inner_" #TI "_long")]] \
  kernel void arg_reduction_inner<OP, TI>(                     \
      constant TI * input [[buffer(0)]],                       \
      device long* output [[buffer(1)]],                       \
      constant uint2& sizes [[buffer(2)]],                     \
      constant uint2& strides [[buffer(3)]],                   \
      uint tptg [[threads_per_threadgroup]],                   \
      uint tgid [[threadgroup_position_in_grid]],              \
      uint simd_lane_id [[thread_index_in_simdgroup]],         \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);   \
  template [[host_name(NAME "_reduction_outer_" #TI "_long")]] \
  kernel void arg_reduction_outer<OP, TI, 32, 32>(             \
      constant TI * input [[buffer(0)]],                       \
      device long* output [[buffer(1)]],                       \
      constant uint3& sizes [[buffer(2)]],                     \
      constant uint4& strides [[buffer(3)]],                   \
      uint3 tid_tg [[thread_position_in_threadgroup]],         \
      uint3 tg_pos [[threadgroup_position_in_grid]]);          \
  template [[host_name(NAME "_inner_p1_" #TI)]]                \
  kernel void arg_reduction_inner_p1<OP, TI>(                  \
      constant TI * input [[buffer(0)]],                       \
      device opmath_t<TI> * val_out [[buffer(1)]],             \
      device int* idx_out [[buffer(2)]],                       \
      constant uint4& sizes [[buffer(3)]],                     \
      uint tptg [[threads_per_threadgroup]],                   \
      uint tgid [[threadgroup_position_in_grid]],              \
      uint simd_lane_id [[thread_index_in_simdgroup]],         \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);   \
  template [[host_name(NAME "_outer_p1_" #TI)]]                \
  kernel void arg_reduction_outer_p1<OP, TI, 32, 32>(          \
      constant TI * input [[buffer(0)]],                       \
      device opmath_t<TI> * val_out [[buffer(1)]],             \
      device int* idx_out [[buffer(2)]],                       \
      constant uint3& sizes [[buffer(3)]],                     \
      constant uint2& strides [[buffer(4)]],                   \
      uint3 tid_tg [[thread_position_in_threadgroup]],         \
      uint3 tg_pos [[threadgroup_position_in_grid]]);

#define REGISTER_ARG_COMBINE_IMPL(VT, NAME, OP)        \
  template [[host_name(NAME "_combine_inner_" #VT)]]   \
  kernel void arg_combine_inner<OP, VT>(               \
      constant VT * val_in [[buffer(0)]],              \
      constant int* idx_in [[buffer(1)]],              \
      device long* output [[buffer(2)]],               \
      constant uint2& sizes [[buffer(3)]],             \
      uint tptg [[threads_per_threadgroup]],           \
      uint tgid [[threadgroup_position_in_grid]],      \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define REGISTER_ARG_COMBINE_FOR_VT(VT)          \
  REGISTER_ARG_COMBINE_IMPL(VT, "argmax", MaxOp) \
  REGISTER_ARG_COMBINE_IMPL(VT, "argmin", MinOp)

REGISTER_ARG_COMBINE_FOR_VT(float);
REGISTER_ARG_COMBINE_FOR_VT(int);
REGISTER_ARG_COMBINE_FOR_VT(long);

#define REGISTER_ARG_REDUCTIONS_FOR_TYPE(T)       \
  REGISTER_ARG_REDUCTION_IMPL(T, "argmax", MaxOp) \
  REGISTER_ARG_REDUCTION_IMPL(T, "argmin", MinOp)

REGISTER_ARG_REDUCTIONS_FOR_TYPE(float);
REGISTER_ARG_REDUCTIONS_FOR_TYPE(half);
REGISTER_ARG_REDUCTIONS_FOR_TYPE(bfloat);
REGISTER_ARG_REDUCTIONS_FOR_TYPE(long);
REGISTER_ARG_REDUCTIONS_FOR_TYPE(int);
REGISTER_ARG_REDUCTIONS_FOR_TYPE(short);
REGISTER_ARG_REDUCTIONS_FOR_TYPE(char);
REGISTER_ARG_REDUCTIONS_FOR_TYPE(uchar);
