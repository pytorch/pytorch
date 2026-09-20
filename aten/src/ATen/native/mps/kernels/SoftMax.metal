#include <ATen/native/mps/kernels/SoftMax.h>
#include <c10/metal/reduction_utils.h>
#include <metal_stdlib>
using namespace metal;
using c10::metal::simdgroup_size;

template <bool log_softmax>
struct SoftmaxEpilogue {
  float max_val;
  float normalizer;

  SoftmaxEpilogue(float max_val, float sum) : max_val(max_val) {
    if IF_CONSTEXPR (log_softmax) {
      normalizer = precise::log(sum);
    } else {
      normalizer = 1.0f / sum;
    }
  }

  float operator()(float value) const {
    if IF_CONSTEXPR (log_softmax) {
      return value - max_val - normalizer;
    } else {
      return precise::exp(value - max_val) * normalizer;
    }
  }
};

template <typename idx_t>
inline idx_t out_row_offset(idx_t row, idx_t dim_size, idx_t out_inner) {
  return (row / out_inner) * dim_size * out_inner + row % out_inner;
}

template <typename idx_t>
inline idx_t input_row_offset(
    idx_t row,
    constant SoftmaxParams<idx_t>& params) {
  idx_t offset = 0;
  for (uint d = params.ndim; d > 0; --d) {
    const uint dim = d - 1;
    if (dim != params.dim) {
      const idx_t coordinate = row % params.sizes[dim];
      row /= params.sizes[dim];
      offset += coordinate * params.strides[dim];
    }
  }
  return offset;
}

// fast path for contiguous case (e.g. dim=-1), when reduction dim is up to 1024
// or when reduction dim is between 1025 and 2048 but there are enough rows
// to fill the GPU with one simdgroup per row (only if there are at least 128
// rows)
template <typename T, typename idx_t, bool log_softmax>
[[max_total_threads_per_threadgroup(kSoftmaxThreads)]] [[kernel]] void
softmax_row(
    constant T* input,
    device T* output,
    constant SoftmaxParams<idx_t>& params,
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const auto dim_size = params.dim_size;
  const idx_t row = idx_t(tgid) * (tptg / simdgroup_size) + sg;
  if (row >= params.num_rows) {
    return;
  }
  const idx_t in_base = row * dim_size;
  float max_val = -INFINITY;
  for (idx_t col = lane; col < dim_size; col += simdgroup_size) {
    max_val = c10::metal::max(max_val, float(input[in_base + col]));
  }
  max_val = c10::metal::simd_max(max_val);
  float sum = 0;
  for (idx_t col = lane; col < dim_size; col += simdgroup_size) {
    sum += precise::exp(float(input[in_base + col]) - max_val);
  }
  const SoftmaxEpilogue<log_softmax> epilogue(
      max_val, c10::metal::simd_sum(sum));
  for (idx_t col = lane; col < dim_size; col += simdgroup_size) {
    output[in_base + col] =
        static_cast<T>(epilogue(float(input[in_base + col])));
  }
}

// First pass of the split path for long contiguous rows. Used when the
// reduction width exceeds 2048 and there are at most 256 logical rows
// and the input consists of contiguous rows.
template <typename T, typename idx_t>
[[max_total_threads_per_threadgroup(kSoftmaxThreads)]] [[kernel]] void
softmax_partial(
    constant T* input,
    device float2* partials,
    constant SoftmaxParams<idx_t>& params,
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]]) {
  const auto dim_size = params.dim_size;
  const auto chunk_size = params.chunk_size;
  const uint ltid = tid.x;
  const uint lsize = tptg.x;
  threadgroup float shared_max[simdgroup_size];
  threadgroup float shared_sum[simdgroup_size];
  const idx_t in_base = idx_t(tgid.y) * dim_size;
  const idx_t chunk_begin = idx_t(tgid.x) * chunk_size;
  const idx_t chunk_end = min(chunk_begin + chunk_size, dim_size);
  float max_val = -INFINITY;
  for (idx_t col = chunk_begin + ltid; col < chunk_end; col += lsize) {
    max_val = c10::metal::max(max_val, float(input[in_base + col]));
  }
  max_val = c10::metal::threadgroup_max(shared_max, max_val, ltid, lsize);
  float sum = 0;
  for (idx_t col = chunk_begin + ltid; col < chunk_end; col += lsize) {
    sum += precise::exp(float(input[in_base + col]) - max_val);
  }
  sum = c10::metal::threadgroup_sum(shared_sum, sum, ltid, lsize);
  if (ltid == 0) {
    partials[idx_t(tgid.y) * params.n_chunks + tgid.x] =
        float2(max_val, max_val == -INFINITY ? 0.f : sum);
  }
}

// Second pass of the split path. Combines all chunk partials into row-wide
// statistics, then each threadgroup writes final values for one
// chunk.
template <typename T, typename idx_t, bool log_softmax>
[[max_total_threads_per_threadgroup(kSoftmaxThreads)]] [[kernel]] void
softmax_finalize(
    constant T* input,
    device T* output,
    constant float2* partials,
    constant SoftmaxParams<idx_t>& params,
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]]) {
  const auto dim_size = params.dim_size;
  const auto chunk_size = params.chunk_size;
  const auto n_chunks = params.n_chunks;
  const uint ltid = tid.x;
  const uint lsize = tptg.x;
  threadgroup float shared_max[simdgroup_size];
  threadgroup float shared_sum[simdgroup_size];
  constant float2* row_partials = partials + idx_t(tgid.y) * n_chunks;
  float max_val = -INFINITY;
  for (idx_t i = ltid; i < n_chunks; i += lsize) {
    max_val = c10::metal::max(max_val, row_partials[i].x);
  }
  max_val = c10::metal::threadgroup_max(shared_max, max_val, ltid, lsize);
  float sum = 0;
  for (idx_t i = ltid; i < n_chunks; i += lsize) {
    sum += row_partials[i].y * precise::exp(row_partials[i].x - max_val);
  }
  sum = c10::metal::threadgroup_sum(shared_sum, sum, ltid, lsize);
  const SoftmaxEpilogue<log_softmax> epilogue(max_val, sum);
  const idx_t row_base = idx_t(tgid.y) * dim_size;
  const idx_t chunk_begin = idx_t(tgid.x) * chunk_size;
  const idx_t chunk_end = min(chunk_begin + chunk_size, dim_size);
  for (idx_t col = chunk_begin + ltid; col < chunk_end; col += lsize) {
    output[row_base + col] =
        static_cast<T>(epilogue(float(input[row_base + col])));
  }
}

// General 2D fallback for strided input and contiguous rows that do not use a
// specialized kernel.
template <typename T, typename idx_t, bool log_softmax>
[[max_total_threads_per_threadgroup(kSoftmaxMaxThreads)]] [[kernel]] void
softmax(
    constant T* input,
    device T* output,
    constant SoftmaxParams<idx_t>& params,
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  threadgroup float shared[kSoftmaxMaxThreads];
  const auto dim_size = params.dim_size;
  const auto inner_size = params.inner_size;
  const idx_t col = idx_t(tgid.x) * tptg.x + tid.x;
  const idx_t row = idx_t(tgid.y) * inner_size + col;
  const idx_t base = input_row_offset(row, params);
  const idx_t out_base = out_row_offset(row, dim_size, inner_size);
  const idx_t dim_stride = params.strides[params.dim];
  const uint shared_idx = tid.y * tptg.x + tid.x;
  const bool active = col < inner_size;

  float max_val = -INFINITY;
  for (idx_t r = tid.y; active && r < dim_size; r += tptg.y) {
    max_val = c10::metal::max(max_val, float(input[base + r * dim_stride]));
  }
  shared[shared_idx] = max_val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tptg.y / 2; stride > 0; stride /= 2) {
    if (tid.y < stride) {
      shared[shared_idx] = c10::metal::max(
          shared[shared_idx], shared[shared_idx + stride * tptg.x]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  max_val = shared[tid.x];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float sum = 0;
  for (idx_t r = tid.y; active && r < dim_size; r += tptg.y) {
    sum += precise::exp(float(input[base + r * dim_stride]) - max_val);
  }
  shared[shared_idx] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tptg.y / 2; stride > 0; stride /= 2) {
    if (tid.y < stride) {
      shared[shared_idx] += shared[shared_idx + stride * tptg.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const SoftmaxEpilogue<log_softmax> epilogue(max_val, shared[tid.x]);

  for (idx_t r = tid.y; active && r < dim_size; r += tptg.y) {
    output[out_base + r * inner_size] =
        static_cast<T>(epilogue(float(input[base + r * dim_stride])));
  }
}

#define REGISTER_SOFTMAX_OP(DTYPE, IDX_T, SUFFIX, NAME, LOG_SOFTMAX)       \
  template [[host_name(#NAME "_row_" #DTYPE "_" #SUFFIX)]] [[kernel]] void \
  softmax_row<DTYPE, IDX_T, LOG_SOFTMAX>(                                  \
      constant DTYPE * input,                                              \
      device DTYPE * output,                                               \
      constant SoftmaxParams<IDX_T> & params,                              \
      uint tptg [[threads_per_threadgroup]],                               \
      uint tgid [[threadgroup_position_in_grid]],                          \
      uint lane [[thread_index_in_simdgroup]],                             \
      uint sg [[simdgroup_index_in_threadgroup]]);                         \
  template                                                                 \
      [[host_name(#NAME "_finalize_" #DTYPE "_" #SUFFIX)]] [[kernel]] void \
      softmax_finalize<DTYPE, IDX_T, LOG_SOFTMAX>(                         \
          constant DTYPE * input,                                          \
          device DTYPE * output,                                           \
          constant float2 * partials,                                      \
          constant SoftmaxParams<IDX_T> & params,                          \
          uint2 tgid [[threadgroup_position_in_grid]],                     \
          uint2 tid [[thread_position_in_threadgroup]],                    \
          uint2 tptg [[threads_per_threadgroup]]);                         \
  template [[host_name(#NAME "_" #DTYPE "_" #SUFFIX)]] [[kernel]] void     \
  softmax<DTYPE, IDX_T, LOG_SOFTMAX>(                                      \
      constant DTYPE * input,                                              \
      device DTYPE * output,                                               \
      constant SoftmaxParams<IDX_T> & params,                              \
      uint2 tid [[thread_position_in_threadgroup]],                        \
      uint2 tptg [[threads_per_threadgroup]],                              \
      uint2 tgid [[threadgroup_position_in_grid]]);

#define REGISTER_SOFTMAX_IDX(DTYPE, IDX_T, SUFFIX)                         \
  REGISTER_SOFTMAX_OP(DTYPE, IDX_T, SUFFIX, softmax, false)                \
  REGISTER_SOFTMAX_OP(DTYPE, IDX_T, SUFFIX, log_softmax, true)             \
  template                                                                 \
      [[host_name("softmax_partial_" #DTYPE "_" #SUFFIX)]] [[kernel]] void \
      softmax_partial<DTYPE, IDX_T>(                                       \
          constant DTYPE * input,                                          \
          device float2 * partials,                                        \
          constant SoftmaxParams<IDX_T> & params,                          \
          uint2 tgid [[threadgroup_position_in_grid]],                     \
          uint2 tid [[thread_position_in_threadgroup]],                    \
          uint2 tptg [[threads_per_threadgroup]]);

#define REGISTER_SOFTMAX(DTYPE)          \
  REGISTER_SOFTMAX_IDX(DTYPE, uint, u32) \
  REGISTER_SOFTMAX_IDX(DTYPE, ulong, u64)

REGISTER_SOFTMAX(float);
REGISTER_SOFTMAX(half);
REGISTER_SOFTMAX(bfloat);
