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

// Load modes for SumOp: identity (sum), nan-to-zero (nansum),
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

// MODE: LOAD_IDENTITY (sum), LOAD_NAN_TO_ZERO (nansum),
// LOAD_NONZERO (count_nonzero -- contributes 1 per nonzero element).
// The compiler eliminates dead branches per instantiation.
template <
    typename TO,
    LoadMode MODE = LOAD_IDENTITY,
    FinalizeOp FINAL = FINAL_NONE>
struct SumOp {
  using acc_t =
      ::metal::conditional_t<MODE == LOAD_NONZERO, uint, opmath_t<TO>>;
  static inline acc_t identity() {
    return acc_t(0);
  }
  template <typename TI>
  static inline acc_t load(TI v) {
    return static_cast<acc_t>(load_val<MODE>(v));
  }
  static inline acc_t combine(acc_t a, acc_t b) {
    return a + b;
  }
  static inline acc_t simd_reduce(acc_t v) {
    return c10::metal::simd_sum(v);
  }
  static inline acc_t threadgroup_reduce(
      threadgroup acc_t* shared,
      acc_t v,
      uint tid,
      uint tptg) {
    return c10::metal::threadgroup_sum(shared, v, tid, tptg);
  }
  static inline TO finalize(acc_t v, float divisor) {
    if (divisor > 0) {
      v /= static_cast<acc_t>(divisor);
    }
    return static_cast<TO>(finalize_val<FINAL>(v));
  }
};

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

// For min/max, TI == TO and Load = IdentityLoad. For all/any, TO = uchar (a
// 1-byte alias for the bool output buffer) and Load = PredicateLoad.
template <template <typename> class OpFn, typename Load, typename TO>
struct ValueOp {
  using acc_t = TO;
  static inline acc_t identity() {
    return OpFn<acc_t>::identity();
  }
  template <typename TI>
  static inline acc_t load(TI v) {
    return Load::template load<acc_t>(v);
  }
  static inline acc_t combine(acc_t a, acc_t b) {
    return OpFn<acc_t>::combine(a, b);
  }
  static inline acc_t simd_reduce(acc_t v) {
    return OpFn<acc_t>::simd_reduce(v);
  }
  static inline acc_t threadgroup_reduce(
      threadgroup acc_t* shared,
      acc_t v,
      uint tid,
      uint tptg) {
    return OpFn<acc_t>::threadgroup_reduce(shared, v, tid, tptg);
  }
  static inline TO finalize(acc_t v, float) {
    return v;
  }
};

template <typename TO, bool TAKE_SQRT>
struct WelfordOp {
  using acc_t = float3;
  static inline acc_t identity() {
    return acc_t(0);
  }
  template <typename TI>
  static inline acc_t load(TI v) {
    return acc_t(static_cast<float>(v), 0, 1);
  }
  static inline acc_t load(float3 partial) {
    return partial;
  }
  static inline acc_t combine(acc_t a, acc_t b) {
    if (a.z == 0) {
      return b;
    }
    if (b.z == 0) {
      return a;
    }
    const float n = a.z + b.z;
    const float delta = b.x - a.x;
    const float w = b.z / n;
    return acc_t(a.x + delta * w, a.y + b.y + delta * delta * a.z * w, n);
  }
  static inline acc_t simd_reduce(acc_t v) {
    for (ushort off = simdgroup_size / 2; off > 0; off >>= 1) {
      v = combine(v, ::metal::simd_shuffle_down(v, off));
    }
    return v;
  }
  static inline acc_t threadgroup_reduce(
      threadgroup acc_t* shared,
      acc_t v,
      uint tid,
      uint tptg) {
    v = simd_reduce(v);
    if (tid % simdgroup_size == 0) {
      shared[tid / simdgroup_size] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < simdgroup_size) {
      v = simd_reduce(tid < tptg / simdgroup_size ? shared[tid] : identity());
    }
    return v;
  }
  template <
      typename T = TO,
      ::metal::enable_if_t<::metal::is_same_v<T, float3>, bool> = true>
  static inline T finalize(acc_t v, float) {
    return v;
  }
  template <
      typename T = TO,
      ::metal::enable_if_t<!::metal::is_same_v<T, float3>, bool> = true>
  static inline T finalize(acc_t v, float correction) {
    const float divisor = v.z > correction ? v.z - correction : 0.0f;
    const float var = v.y / divisor;
    return static_cast<T>(TAKE_SQRT ? ::precise::sqrt(var) : var);
  }
};

// Reduction kernel with multiple independent accumulation chains (ILP).
// Each thread maintains NCHAINS independent accumulators to hide ALU latency
// and keep the memory pipeline saturated.
//
// Two internal paths selected per-threadgroup (not per-element):
//   - Single reduced dim (or full reduction): compute input_base + k * stride
//     once per TG, then direct indexing — no per-element dim loop.
//   - Multiple reduced dims: fall back to get_input_offset per element.
// The max_total_threads_per_threadgroup hint lets the compiler bound the
// runtime tptg value, which in turn lets c10::metal::threadgroup_min/max
// constant-fold its size-vs-simdgroup_size branch.
template <typename OP, typename TI, typename TO, uint NCHAINS = SUM_NCHAINS>
[[max_total_threads_per_threadgroup(MAX_THREADGROUP_SIZE)]]
kernel void reduction_generic(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant NormParams<>& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  using TA = typename OP::acc_t;

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

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = OP::identity();
  }

  const uint32_t rsize = params.reduction_size;
  const uint32_t stride = tptg * NCHAINS;
  uint32_t base = tid * NCHAINS;

  if (num_reduced_dims <= 1) {
    // Fast path: direct indexing with base + k * reduction_stride
    for (; base + NCHAINS <= rsize; base += stride) {
      for (uint j = 0; j < NCHAINS; j++) {
        acc[j] = OP::combine(
            acc[j],
            OP::load(input[input_base + (base + j) * reduction_stride]));
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] = OP::combine(
          acc[idx % NCHAINS],
          OP::load(input[input_base + idx * reduction_stride]));
    }
  } else {
    // Generic path: per-element strided offset for multi-dim reductions
    for (; base + NCHAINS <= rsize; base += stride) {
      for (uint j = 0; j < NCHAINS; j++) {
        acc[j] = OP::combine(
            acc[j], OP::load(input[get_input_offset(base + j, tgid, params)]));
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] = OP::combine(
          acc[idx % NCHAINS],
          OP::load(input[get_input_offset(idx, tgid, params)]));
    }
  }

  // Collapse chains into a single value
  TA output_val = acc[0];
  for (uint j = 1; j < NCHAINS; j++) {
    output_val = OP::combine(output_val, acc[j]);
  }

  // SIMD + threadgroup tree reduction
  threadgroup TA shared_outputs[MAX_THREADGROUP_SIZE / simdgroup_size];
  output_val = OP::threadgroup_reduce(shared_outputs, output_val, tid, tptg);

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
    output[output_offset] = OP::finalize(output_val, params.p);
  }
}

// Specialized kernel for reducing a non-innermost dim. The input is viewed
// as [outer_size, dim_size, inner_size] with dim_size reduced (the same
// decomposition the CUDA spatial softmax / scan_outer_dim kernels use);
// explicit strides let the same kernel serve collapsible non-contiguous
// inputs. TG_X threads cover adjacent inner columns (coalesced), TG_Y
// row-workers split dim_size and combine via shared memory. Grid: x tiles
// inner_size, y carries the split-K segments, z the outer batches.
// Also registered with TG_Y == 1 as the "outer_small_dim" variant, where
// one thread serially reduces the whole (short) reduced dim.
template <
    typename OP,
    typename TI,
    typename TO,
    uint TG_X = OUTER_TG_WIDTH,
    uint TG_Y = OUTER_TG_HEIGHT,
    uint NCHAINS = SUM_NCHAINS>
[[max_total_threads_per_threadgroup(TG_X * TG_Y)]]
kernel void reduction_outer(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    // [dim_size, inner_size, num_segs, unused]
    constant uint4& sizes [[buffer(2)]],
    constant float& param [[buffer(3)]], // >0 divides accumulator before cast
    // [dim_stride, inner_stride, outer_stride, unused]
    constant uint4& strides [[buffer(4)]],
    uint3 tid_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = typename OP::acc_t;
  const uint dim_size = sizes.x;
  const uint inner_size = sizes.y;
  const uint num_segs = max(sizes.z, 1u);
  const uint dim_stride = strides.x;
  const uint inner_stride = strides.y;
  const uint outer_offset = tg_pos.z * strides.z;

  uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= inner_size)
    return;

  const uint seg_rows = ceil_div(dim_size, num_segs);
  const uint seg_start = tg_pos.y * seg_rows;
  const uint seg_end = min(seg_start + seg_rows, dim_size);
  // Split the segment rows among TG_Y workers
  uint rows_per_y = ceil_div(seg_rows, TG_Y);
  uint row_start = seg_start + tid_tg.y * rows_per_y;
  uint row_end = min(row_start + rows_per_y, seg_end);

  // Multiple accumulation chains for ILP
  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++)
    acc[j] = OP::identity();
  const uint col_off = outer_offset + col * inner_stride;

  uint row = row_start;
  for (; row + NCHAINS <= row_end; row += NCHAINS) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] = OP::combine(
          acc[j], OP::load(input[col_off + (row + j) * dim_stride]));
    }
  }
  for (; row < row_end; row++) {
    acc[row % NCHAINS] = OP::combine(
        acc[row % NCHAINS], OP::load(input[col_off + row * dim_stride]));
  }

  TA val = acc[0];
  for (uint j = 1; j < NCHAINS; j++)
    val = OP::combine(val, acc[j]);

  // Reduce across TG_Y row-workers via shared memory
  threadgroup TA shmem[TG_Y][TG_X];
  shmem[tid_tg.y][tid_tg.x] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = TG_Y / 2; stride > 0; stride >>= 1) {
    if (tid_tg.y < stride)
      shmem[tid_tg.y][tid_tg.x] = OP::combine(
          shmem[tid_tg.y][tid_tg.x], shmem[tid_tg.y + stride][tid_tg.x]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid_tg.y == 0) {
    const uint out_idx =
        (num_segs > 1 ? tg_pos.y : tg_pos.z) * inner_size + col;
    output[out_idx] = OP::finalize(shmem[0][tid_tg.x], param);
  }
}

// Narrow layout for inner_size < OUTER_TG_WIDTH: flat coalesced walk; the
// thread count is a multiple of inner_size, pinning each thread to one column.
template <
    typename OP,
    typename TI,
    typename TO,
    uint TG_SIZE = NARROW_TG_SIZE,
    uint NCHAINS = SUM_NCHAINS>
[[max_total_threads_per_threadgroup(TG_SIZE)]]
kernel void reduction_narrow(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    // [dim_size, inner_size, num_segs, unused]
    constant uint4& sizes [[buffer(2)]],
    constant float& param [[buffer(3)]],
    uint3 tid_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = typename OP::acc_t;
  const uint tid = tid_tg.x;
  const uint dim_size = sizes.x;
  const uint inner_size = sizes.y;
  const uint num_segs = max(sizes.z, 1u);
  const uint active = (TG_SIZE / inner_size) * inner_size;

  const uint seg_rows = ceil_div(dim_size, num_segs);
  const uint r0 = tg_pos.y * seg_rows;
  const uint r1 = min(r0 + seg_rows, dim_size);
  const uint base = tg_pos.z * dim_size * inner_size + r0 * inner_size;
  const uint count = (r0 < dim_size) ? (r1 - r0) * inner_size : 0u;

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = OP::identity();
  }
  const uint stride = active * NCHAINS;
  uint k = tid;
  for (; k + (NCHAINS - 1) * active < count; k += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] = OP::combine(acc[j], OP::load(input[base + k + j * active]));
    }
  }
  for (; k < count; k += active) {
    acc[0] = OP::combine(acc[0], OP::load(input[base + k]));
  }
  TA val = acc[0];
  for (uint j = 1; j < NCHAINS; j++) {
    val = OP::combine(val, acc[j]);
  }

  threadgroup TA shmem[TG_SIZE];
  shmem[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < inner_size) {
    TA s = OP::identity();
    for (uint t = tid; t < active; t += inner_size) {
      s = OP::combine(s, shmem[t]);
    }
    const uint out_idx = (num_segs > 1) ? (tg_pos.y * inner_size + tid)
                                        : (tg_pos.z * inner_size + tid);
    output[out_idx] = OP::finalize(s, param);
  }
}

// Specialized kernel for reducing the innermost dim of a contiguous tensor.
// Input [M, N] -> output [M], each SIMD group reduces one row of N elements.
// Multiple SIMD groups per TG handle different rows for occupancy.
// No shared memory needed since simd_reduce suffices for intra-row collapse.
template <typename OP, typename TI, typename TO, uint NCHAINS = SUM_NCHAINS>
kernel void reduction_innermost(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint2& sizes [[buffer(2)]], // [M, N]
    constant float& param [[buffer(3)]], // >0 divides accumulator before cast
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = typename OP::acc_t;
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
    acc[j] = OP::identity();

  // Each of 32 lanes reads elements at stride 32, NCHAINS at a time.
  // Align down to full blocks of stride = 32 * NCHAINS elements.
  const uint stride = 32 * NCHAINS;
  const uint aligned_N = (N / stride) * stride;
  uint base = simd_lane_id * NCHAINS;
  for (; base < aligned_N; base += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] = OP::combine(acc[j], OP::load(row_ptr[base + j]));
    }
  }
  // Tail: remaining elements after last full block, one per lane
  for (uint i = aligned_N + simd_lane_id; i < N; i += 32) {
    acc[0] = OP::combine(acc[0], OP::load(row_ptr[i]));
  }

  TA val = acc[0];
  for (uint j = 1; j < NCHAINS; j++)
    val = OP::combine(val, acc[j]);

  val = OP::simd_reduce(val);

  if (simd_lane_id == 0) {
    output[row] = OP::finalize(val, param);
  }
}

// uchar/char/bool have no vec4type_t; their chunk loads stay scalar.
template <typename...>
using chunk_void_t = void;
template <typename T, typename = void>
constexpr constant bool chunk_has_vec4_v = false;
template <typename T>
constexpr constant bool chunk_has_vec4_v<T, chunk_void_t<vec4type_t<T>>> = true;

template <typename T, ::metal::enable_if_t<sizeof(T) == 8, bool> = true>
inline T chunk_shuffle_down(T val, ushort delta) {
  return as_type<T>(::metal::simd_shuffle_down(as_type<int2>(val), delta));
}
template <
    typename T,
    ::metal::enable_if_t<sizeof(T) == 4 || sizeof(T) == 16, bool> = true>
inline T chunk_shuffle_down(T val, ushort delta) {
  return ::metal::simd_shuffle_down(val, delta);
}
template <
    typename T,
    ::metal::enable_if_t<sizeof(T) < 4 && ::metal::is_integral_v<T>, bool> =
        true>
inline T chunk_shuffle_down(T val, ushort delta) {
  return static_cast<T>(
      ::metal::simd_shuffle_down(static_cast<int>(val), delta));
}
template <
    typename T,
    ::metal::enable_if_t<sizeof(T) < 4 && !::metal::is_integral_v<T>, bool> =
        true>
inline T chunk_shuffle_down(T val, ushort delta) {
  return static_cast<T>(
      ::metal::simd_shuffle_down(static_cast<float>(val), delta));
}

// Consume the vec4-loadable prefix of a lane's segment: lane row_lane reads
// quads row_lane, row_lane + lanes_per_row, ... of [seg_begin, seg_end) and
// returns where its scalar loop should resume. The caller's `ok` must be
// uniform across the simdgroup (rows that share a simdgroup would otherwise
// diverge; measured 15% worse than all-scalar on bf16 rows of 33) and must
// guarantee element-4 alignment of the segment start. Types with no vec4
// (uchar/char/bool) always take the scalar loop.
template <
    typename OPS,
    typename TA,
    typename TI,
    ::metal::enable_if_t<chunk_has_vec4_v<TI>, bool> = true>
inline uint chunk_quads(
    constant TI* input,
    uint row_base,
    uint seg_begin,
    uint seg_end,
    uint lanes_per_row,
    uint row_lane,
    bool ok,
    thread TA& acc) {
  const uint num_quads = (seg_end - seg_begin) / 4;
  if (!ok || num_quads < 2) {
    return seg_begin + row_lane;
  }
  using V = vec4type_t<TI>;
  constant V* vin = reinterpret_cast<constant V*>(input + row_base + seg_begin);
  for (uint q = row_lane; q < num_quads; q += lanes_per_row) {
    const V v = vin[q];
#pragma unroll
    for (uint c = 0; c < 4; c++) {
      acc = OPS::combine(acc, OPS::load(v[c]));
    }
  }
  return seg_begin + num_quads * 4 + row_lane;
}

template <
    typename OPS,
    typename TA,
    typename TI,
    ::metal::enable_if_t<!chunk_has_vec4_v<TI>, bool> = true>
inline uint chunk_quads(
    constant TI*,
    uint,
    uint seg_begin,
    uint,
    uint,
    uint row_lane,
    bool,
    thread TA&) {
  return seg_begin + row_lane;
}

// Shared body of the *_innermost_chunk kernels: each row of length row_len is
// split across lanes_per_row lanes (simdgroup_size / lanes_per_row rows share
// one simdgroup, a shuffle tree folds the per-lane partials) and, for split-K,
// into segs_per_row segments per row producing [num_rows, segs_per_row]
// partials the host combines with a second dispatch. Lanes read their rows
// interleaved, as vec4 quads when chunk_quads allows and as chained scalar
// loads otherwise.
template <typename OP, typename TI, typename TO>
kernel void reduction_innermost_chunk(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint4& sizes [[buffer(2)]], // [rows, row_len, lanes, segs]
    constant float& param [[buffer(3)]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = typename OP::acc_t;
  const uint num_rows = sizes.x;
  const uint row_len = sizes.y;
  const uint lanes_per_row = sizes.z;
  const uint segs_per_row = sizes.w;
  const uint rows_per_simd = c10::metal::simdgroup_size / lanes_per_row;
  const uint simd_idx =
      tgid * (tptg / c10::metal::simdgroup_size) + simdgroup_id;
  const uint out_idx = simd_idx * rows_per_simd + simd_lane_id / lanes_per_row;
  if (out_idx >= num_rows * segs_per_row) {
    return;
  }
  const uint row_lane = simd_lane_id % lanes_per_row;
  const uint row = out_idx / segs_per_row;
  const uint seg = out_idx % segs_per_row;
  const uint seg_len = ceil_div(row_len, segs_per_row);
  const uint seg_begin = seg * seg_len;
  const uint seg_end = min(seg_begin + seg_len, row_len);
  const uint row_base = row * row_len;
  metal::array<TA, 4> acc;
  for (uint j = 0; j < 4; j++) {
    acc[j] = OP::identity();
  }
  // Quads only pay off for the short-row lane counts (up to 10% for 2-byte
  // dtypes); at 32 lanes per row (split-K, combine passes) chained scalar
  // loads measure equal or ahead. A single segment per row with an
  // element-4 aligned row length makes every row start aligned, so the
  // decision is simdgroup-uniform.
  const bool quads_ok = lanes_per_row < c10::metal::simdgroup_size &&
      segs_per_row == 1 && (row_len & 3u) == 0;
  uint k = chunk_quads<OP>(
      input,
      row_base,
      seg_begin,
      seg_end,
      lanes_per_row,
      row_lane,
      quads_ok,
      acc[0]);
  for (; k + 3 * lanes_per_row < seg_end; k += 4 * lanes_per_row) {
#pragma unroll
    for (uint j = 0; j < 4; j++) {
      const uint idx = row_base + k + j * lanes_per_row;
      acc[j] = OP::combine(acc[j], OP::load(input[idx]));
    }
  }
  for (; k < seg_end; k += lanes_per_row) {
    acc[0] = OP::combine(acc[0], OP::load(input[row_base + k]));
  }
  TA val =
      OP::combine(OP::combine(acc[0], acc[1]), OP::combine(acc[2], acc[3]));
  for (uint off = lanes_per_row >> 1; off > 0; off >>= 1) {
    val = OP::combine(val, chunk_shuffle_down(val, static_cast<ushort>(off)));
  }
  if (row_lane == 0) {
    output[out_idx] = OP::finalize(val, param);
  }
}

// Pass-1 kernel for two-pass full reductions over a contiguous buffer:
// threadgroup tgid reduces the slice input[tgid * E .. (tgid + 1) * E) with
// flat indexing (params.y = E), no NormParams offset math. Each chain step
// consumes 4 adjacent elements to cut loop overhead; loads stay scalar so
// the slice base needs no vector alignment.
template <typename OP, typename TI, typename TO, uint NCHAINS = SUM_NCHAINS>
kernel void reduction_flat(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint2& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  using TA = typename OP::acc_t;
  const uint E = params.y;
  constant TI* in = input + tgid * E;
  const uint n_quads = E / 4;

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = OP::identity();
  }
  const uint stride = tptg * NCHAINS;
  uint i = tid;
  for (; i + (NCHAINS - 1) * tptg < n_quads; i += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      const uint base = (i + j * tptg) * 4;
#pragma unroll
      for (uint k = 0; k < 4; k++) {
        acc[j] = OP::combine(acc[j], OP::load(in[base + k]));
      }
    }
  }
  for (; i < n_quads; i += tptg) {
#pragma unroll
    for (uint k = 0; k < 4; k++) {
      acc[0] = OP::combine(acc[0], OP::load(in[i * 4 + k]));
    }
  }
  for (uint k = n_quads * 4 + tid; k < E; k += tptg) {
    acc[0] = OP::combine(acc[0], OP::load(in[k]));
  }
  TA val = acc[0];
  for (uint j = 1; j < NCHAINS; j++) {
    val = OP::combine(val, acc[j]);
  }
  threadgroup TA shared[MAX_THREADGROUP_SIZE / 32];
  val = OP::threadgroup_reduce(shared, val, tid, tptg);
  if (tid == 0) {
    output[tgid] = OP::finalize(val, 0);
  }
}

#define INSTANTIATE_KERNEL(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>

#define REGISTER_REDUCTION(PREFIX, TI, TO, ...)        \
  INSTANTIATE_KERNEL(                                  \
      PREFIX "reduction_" #TI "_" #TO,                 \
      reduction_generic,                               \
      __VA_ARGS__,                                     \
      TI,                                              \
      TO);                                             \
  INSTANTIATE_KERNEL(                                  \
      PREFIX "reduction_flat_" #TI "_" #TO,            \
      reduction_flat,                                  \
      __VA_ARGS__,                                     \
      TI,                                              \
      TO);                                             \
  INSTANTIATE_KERNEL(                                  \
      PREFIX "reduction_outer_" #TI "_" #TO,           \
      reduction_outer,                                 \
      __VA_ARGS__,                                     \
      TI,                                              \
      TO);                                             \
  INSTANTIATE_KERNEL(                                  \
      PREFIX "reduction_outer_small_dim_" #TI "_" #TO, \
      reduction_outer,                                 \
      __VA_ARGS__,                                     \
      TI,                                              \
      TO,                                              \
      OUTER_TG_WIDTH,                                  \
      1);                                              \
  INSTANTIATE_KERNEL(                                  \
      PREFIX "reduction_narrow_" #TI "_" #TO,          \
      reduction_narrow,                                \
      __VA_ARGS__,                                     \
      TI,                                              \
      TO);                                             \
  INSTANTIATE_KERNEL(                                  \
      PREFIX "reduction_innermost_" #TI "_" #TO,       \
      reduction_innermost,                             \
      __VA_ARGS__,                                     \
      TI,                                              \
      TO);                                             \
  INSTANTIATE_KERNEL(                                  \
      PREFIX "reduction_innermost_chunk_" #TI "_" #TO, \
      reduction_innermost_chunk,                       \
      __VA_ARGS__,                                     \
      TI,                                              \
      TO)

#define REGISTER_SUM_IMPL(TI, TO, PREFIX, MODE) \
  REGISTER_REDUCTION(PREFIX, TI, TO, SumOp<TO, MODE>)

#define REGISTER_SUM(TI, TO) REGISTER_SUM_IMPL(TI, TO, "sum_", LOAD_IDENTITY)
#define REGISTER_NANSUM(TI, TO) \
  REGISTER_SUM_IMPL(TI, TO, "nansum_", LOAD_NAN_TO_ZERO)
#define REGISTER_COUNT_NONZERO(TI) \
  REGISTER_SUM_IMPL(TI, long, "count_nonzero_", LOAD_NONZERO)
#define REGISTER_NORM_INNERMOST(TI, TO)           \
  INSTANTIATE_KERNEL(                             \
      "norm_l1_reduction_innermost_" #TI "_" #TO, \
      reduction_innermost,                        \
      SumOp<TO, LOAD_ABS>,                        \
      TI,                                         \
      TO);                                        \
  INSTANTIATE_KERNEL(                             \
      "norm_l2_reduction_innermost_" #TI "_" #TO, \
      reduction_innermost,                        \
      SumOp<TO, LOAD_SQUARE, FINAL_SQRT>,         \
      TI,                                         \
      TO)

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
REGISTER_SUM(float2, half2);
REGISTER_SUM(half2, half2);
REGISTER_SUM(half2, float2);

// nansum variants (floating-point only — integers can't have NaN)
REGISTER_NANSUM(float, float);
REGISTER_NANSUM(half, half);
REGISTER_NANSUM(half, float);
REGISTER_NANSUM(bfloat, bfloat);
REGISTER_NANSUM(bfloat, float);

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

REGISTER_NORM_INNERMOST(float, float);
REGISTER_NORM_INNERMOST(half, half);
REGISTER_NORM_INNERMOST(bfloat, bfloat);

#define REGISTER_VALUE_REDUCTION_IMPL(TI, TO, NAME, OP, LOAD) \
  REGISTER_REDUCTION(NAME "_", TI, TO, ValueOp<OP, LOAD, TO>)

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
  REGISTER_MAX(T);                          \
  REGISTER_MIN(T);                          \
  REGISTER_ANY(T);                          \
  REGISTER_ALL(T)

// Types that only participate in all/any (bool: no simd_min/max; complex:
// no ordering, but predicate-reduce on its real/imag pair is well-defined).
#define REGISTER_PRED_REDUCTIONS_FOR_TYPE(T) \
  REGISTER_ANY(T);                           \
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

#define REGISTER_WELFORD(PREFIX, SQRT, T)                         \
  REGISTER_REDUCTION(PREFIX, T, T, WelfordOp<T, SQRT>);           \
  REGISTER_REDUCTION(PREFIX, T, float3, WelfordOp<float3, SQRT>); \
  REGISTER_REDUCTION(PREFIX, float3, T, WelfordOp<T, SQRT>)

#define REGISTER_STD_VAR(T)           \
  REGISTER_WELFORD("var_", false, T); \
  REGISTER_WELFORD("std_", true, T)

REGISTER_STD_VAR(float);
REGISTER_STD_VAR(half);
REGISTER_STD_VAR(bfloat);

// =============================================================================
// argmax/argmin: per output element find the (linear) index of the max/min
// input element along the reduced dim(s). Output is always int64. NaN
// propagates (first NaN in source order wins); on ties the lowest index wins.
// Mirrors the reduction_generic layout but tracks a (value, index) pair instead
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

  // Compute input_base and detect reduction pattern (mirrors
  // reduction_generic).
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

// Role of the inner arg kernel: one pass over a whole row, split-K pass 1
// (one simdgroup per segment, writes a (value, index) partial), or pass 2
// (indices come from idx_in, so ties break on the lowest source index).
enum ArgMode : uint { ARG_PLAIN = 0, ARG_SPLIT_P1 = 1, ARG_COMBINE = 2 };

// Inner-dim arg-reduction: input is logically [M, N] contiguous, reduce N
// (innermost). One SIMD group (32 lanes) per row, multiple SIMD groups per
// TG for occupancy. Lane L scans positions {L, L+32, L+64, ...} of its row
// with strict-improvement updates (so the lane's stored idx is the lowest
// of its scanned positions matching the winning value). The cross-lane
// collapse uses simd_arg_reduce which ties on lowest IDX, not lowest LANE.
template <template <typename> class OpFn, typename TI, ArgMode MODE = ARG_PLAIN>
kernel void arg_reduction_innermost(
    constant TI* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    // ARG_SPLIT_P1: [num_partials, seg_len, num_segs, row_len]; else [M, N]
    constant uint4& sizes [[buffer(2)]],
    constant int* idx_in [[buffer(3)]],
    device TI* val_out [[buffer(4)]],
    device int* idx_out [[buffer(5)]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = opmath_t<TI>;
  using Op = OpFn<TA>;
  const uint num_simd_groups = tptg / simdgroup_size;

  const uint row = tgid * num_simd_groups + simdgroup_id;
  if (row >= sizes.x) {
    return;
  }

  constexpr bool split = MODE == ARG_SPLIT_P1;
  const uint num_segs = split ? sizes.z : 1;
  const uint seg_off = split ? (row % num_segs) * sizes.y : 0;
  const uint span = split ? min(seg_off + sizes.y, sizes.w) - seg_off : sizes.y;
  const uint row_base = split ? (row / num_segs) * sizes.w : row * sizes.y;
  constant TI* row_ptr = input + row_base + seg_off;

  TA best_val = Op::identity();
  uint32_t best_idx =
      MODE == ARG_COMBINE ? ::metal::numeric_limits<uint32_t>::max() : 0;
  for (uint i = simd_lane_id; i < span; i += simdgroup_size) {
    const TA val = static_cast<TA>(row_ptr[i]);
    if IF_CONSTEXPR (MODE == ARG_COMBINE) {
      const uint32_t idx = static_cast<uint32_t>(idx_in[row * span + i]);
      if (arg_replace<OpFn>(val, idx, best_val, best_idx)) {
        best_val = val;
        best_idx = idx;
      }
    } else if (Op::replace(val, best_val)) {
      best_val = val;
      best_idx = i;
    }
  }

  auto rc = simd_arg_reduce<OpFn>(best_val, best_idx);
  if (simd_lane_id == 0) {
    if IF_CONSTEXPR (split) {
      val_out[row] = static_cast<TI>(rc.first);
      idx_out[row] = static_cast<int>(seg_off + rc.second);
    } else {
      output[row] = static_cast<long>(rc.second);
    }
  }
}

// Outer-dim arg-reduction: input viewed as [outer_size, dim_size, inner_size]
// through explicit strides, reducing dim. TG_X threads cover adjacent inner
// columns (coalesced when inner_stride == 1), TG_Y threads split the dim
// rows, grid z walks the outer batches. Per-thread scan keeps the lowest row
// with the winning value; the cross-worker tree reduction uses arg_replace
// (strictly-better OR equal-with-lower-idx).
// SPLIT selects split-K pass 1 (outer_size == 1): grid y cuts the dim rows
// into num_segs segments and each threadgroup writes its segment's
// (value, index) partial to [inner_size, num_segs]. Winning values are input
// elements, so partials keep the input dtype and pass 2 upcasts them exactly
// like pass 1 upcast the input.
template <
    template <typename> class OpFn,
    typename TI,
    uint TG_X = OUTER_TG_WIDTH,
    uint TG_Y = OUTER_TG_HEIGHT,
    bool SPLIT = false>
[[max_total_threads_per_threadgroup(TG_X * TG_Y)]]
kernel void arg_reduction_outer(
    constant TI* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    // [dim_size, inner_size, num_segs, unused]
    constant uint4& sizes [[buffer(2)]],
    // [dim_stride, inner_stride, outer_stride, unused]
    constant uint4& strides [[buffer(3)]],
    device TI* val_out [[buffer(4)]],
    device int* idx_out [[buffer(5)]],
    uint3 tid_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = opmath_t<TI>;
  using Op = OpFn<TA>;
  const uint dim_size = sizes.x;
  const uint inner_size = sizes.y;
  const uint num_segs = SPLIT ? sizes.z : 1;
  const uint dim_stride = strides.x;
  const uint inner_stride = strides.y;
  const uint outer_offset = SPLIT ? 0 : tg_pos.z * strides.z;

  const uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= inner_size) {
    return;
  }

  // Split the segment rows among the TG_Y workers.
  const uint seg_rows = SPLIT ? ceil_div(dim_size, num_segs) : dim_size;
  const uint seg_start = SPLIT ? tg_pos.y * seg_rows : 0;
  const uint seg_end = SPLIT ? min(seg_start + seg_rows, dim_size) : dim_size;
  const uint rows_per_y = ceil_div(seg_rows, TG_Y);
  const uint row_start = seg_start + tid_tg.y * rows_per_y;
  const uint row_end = min(row_start + rows_per_y, seg_end);
  const uint col_off = outer_offset + col * inner_stride;

  TA best_val = Op::identity();
  // When no element strictly beats the identity (uniform-identity input) the
  // claimed index must still be a row this worker scans first, so the tree
  // tie-break resolves to row 0.
  uint32_t best_idx = row_start;
  for (uint row = row_start; row < row_end; row++) {
    const TA val = static_cast<TA>(input[col_off + row * dim_stride]);
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
    if IF_CONSTEXPR (SPLIT) {
      val_out[col * num_segs + tg_pos.y] =
          static_cast<TI>(shared_vals[0][tid_tg.x]);
      idx_out[col * num_segs + tg_pos.y] =
          static_cast<int>(shared_idxs[0][tid_tg.x]);
    } else {
      output[tg_pos.z * inner_size + col] =
          static_cast<long>(shared_idxs[0][tid_tg.x]);
    }
  }
}

// Split-K pass 1 for inner_size below a threadgroup row (outer_size == 1):
// the threadgroup folds row_step dim rows at a time, combining across rows
// through threadgroup memory. The host dispatches row_step * inner_size
// threads so each thread stays pinned to one inner column. Partials laid
// out [inner_size, num_segs], values in the input dtype.
template <
    template <typename> class OpFn,
    typename TI,
    uint TG_SIZE = NARROW_TG_SIZE>
[[max_total_threads_per_threadgroup(TG_SIZE)]]
kernel void arg_reduction_narrow_p1(
    constant TI* input [[buffer(0)]],
    device TI* val_out [[buffer(1)]],
    device int* idx_out [[buffer(2)]],
    // [dim_size, inner_size, num_segs, unused]
    constant uint4& sizes [[buffer(3)]],
    uint3 tid_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = opmath_t<TI>;
  using Op = OpFn<TA>;
  const uint tid = tid_tg.x;
  const uint dim_size = sizes.x;
  const uint inner_size = sizes.y;
  const uint num_segs = sizes.z;

  const uint seg_rows = ceil_div(dim_size, num_segs);
  const uint r0 = tg_pos.y * seg_rows;
  const uint r1 = min(r0 + seg_rows, dim_size);
  const uint col = tid % inner_size;
  const uint row_step = TG_SIZE / inner_size;
  const uint active = row_step * inner_size;

  TA best_val = Op::identity();
  uint32_t best_idx = r0 + tid / inner_size;
  for (uint row = r0 + tid / inner_size; row < r1; row += row_step) {
    const TA val = static_cast<TA>(input[row * inner_size + col]);
    if (Op::replace(val, best_val)) {
      best_val = val;
      best_idx = row;
    }
  }

  threadgroup TA shared_vals[TG_SIZE];
  threadgroup uint32_t shared_idxs[TG_SIZE];
  shared_vals[tid] = best_val;
  shared_idxs[tid] = best_idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < inner_size) {
    for (uint t = tid + inner_size; t < active; t += inner_size) {
      if (arg_replace<OpFn>(
              shared_vals[t], shared_idxs[t], best_val, best_idx)) {
        best_val = shared_vals[t];
        best_idx = shared_idxs[t];
      }
    }
    val_out[tid * num_segs + tg_pos.y] = static_cast<TI>(best_val);
    idx_out[tid * num_segs + tg_pos.y] = static_cast<int>(best_idx);
  }
}

// Both split-K passes and the combine pass of the inner and outer layouts
// are the single-pass kernels under a different mode, so 4 kernel templates
// cover 7 dispatches.
#define REGISTER_ARG_REDUCTION_IMPL(TI, NAME, OP)                            \
  INSTANTIATE_KERNEL(NAME "_reduction_" #TI "_long", arg_reduction, OP, TI); \
  INSTANTIATE_KERNEL(                                                        \
      NAME "_reduction_innermost_" #TI "_long",                              \
      arg_reduction_innermost,                                               \
      OP,                                                                    \
      TI);                                                                   \
  INSTANTIATE_KERNEL(                                                        \
      NAME "_reduction_innermost_p1_" #TI,                                   \
      arg_reduction_innermost,                                               \
      OP,                                                                    \
      TI,                                                                    \
      ARG_SPLIT_P1);                                                         \
  INSTANTIATE_KERNEL(                                                        \
      NAME "_reduction_combine_" #TI,                                        \
      arg_reduction_innermost,                                               \
      OP,                                                                    \
      TI,                                                                    \
      ARG_COMBINE);                                                          \
  INSTANTIATE_KERNEL(                                                        \
      NAME "_reduction_outer_" #TI "_long", arg_reduction_outer, OP, TI);    \
  INSTANTIATE_KERNEL(                                                        \
      NAME "_reduction_outer_p1_" #TI,                                       \
      arg_reduction_outer,                                                   \
      OP,                                                                    \
      TI,                                                                    \
      OUTER_TG_WIDTH,                                                        \
      OUTER_TG_HEIGHT,                                                       \
      true);                                                                 \
  INSTANTIATE_KERNEL(                                                        \
      NAME "_reduction_narrow_p1_" #TI, arg_reduction_narrow_p1, OP, TI);

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
