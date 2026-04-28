#include <ATen/native/mps/kernels/TensorCompare.h>
#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using c10::metal::simdgroup_size;

template <typename T>
kernel void isin(
    constant T* elements [[buffer(0)]],
    constant T* test_elements [[buffer(1)]],
    device atomic_uint* out [[buffer(2)]],
    constant IsinParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  uint elem_idx = tgid % params.numel_elements;
  uint chunk = tgid / params.numel_elements;

  T elem = elements[elem_idx];
  uint chunk_size =
      (params.numel_test + params.num_chunks - 1) / params.num_chunks;
  uint start = chunk * chunk_size;
  uint end = min(start + chunk_size, params.numel_test);

  uint found = 0u;
  for (uint j = start + tid; j < end; j += tptg) {
    found |= (elem == test_elements[j]) ? 1u : 0u;
  }

  threadgroup uint shared[ISIN_THREADS_PER_THREADGROUP];
  uint threads_remaining = tptg;
  while (threads_remaining > 1) {
    found = simd_or(found);
    threads_remaining =
        (threads_remaining + simdgroup_size - 1) / simdgroup_size;
    if (threads_remaining > 1) {
      if (simd_lane_id == 0) {
        shared[simdgroup_id] = found;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < threads_remaining) {
        found = shared[tid];
      } else {
        return;
      }
    }
  }

  if (tid == 0 && found != 0u) {
    atomic_fetch_or_explicit(&out[elem_idx], 1u, memory_order_relaxed);
  }
}

// Casts the atomic-OR'd uint counts buffer (one slot per element) into the
// bool output and applies the `invert` flag. Run as a separate kernel because
// chunks may concurrently OR into the same slot, so the XOR with `invert` must
// happen exactly once per output element after all chunks have completed.
kernel void isin_apply_invert(
    device const uint* counts [[buffer(0)]],
    device bool* out [[buffer(1)]],
    constant bool& invert [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = (counts[tid] != 0u) != invert;
}

#define REGISTER_ISIN_OP(T)                               \
  template [[host_name("isin_" #T)]] kernel void isin<T>( \
      constant T * elements [[buffer(0)]],                \
      constant T * test_elements [[buffer(1)]],           \
      device atomic_uint * out [[buffer(2)]],             \
      constant IsinParams & params [[buffer(3)]],         \
      uint tid [[thread_position_in_threadgroup]],        \
      uint tptg [[threads_per_threadgroup]],              \
      uint tgid [[threadgroup_position_in_grid]],         \
      uint simd_lane_id [[thread_index_in_simdgroup]],    \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

REGISTER_ISIN_OP(float);
REGISTER_ISIN_OP(half);
REGISTER_ISIN_OP(bfloat);
REGISTER_ISIN_OP(int);
REGISTER_ISIN_OP(long);
REGISTER_ISIN_OP(short);
REGISTER_ISIN_OP(char);
REGISTER_ISIN_OP(uchar);

struct clamp_functor {
  template <typename T>
  inline T operator()(const T a, const T b_min, const T c_max) {
    return c10::metal::min(c10::metal::max(a, b_min), c_max);
  }
};

struct clamp_scalar_functor {
  template <typename T>
  inline T operator()(const T a, const ClampScalarParams<T> params) {
    return c10::metal::min(c10::metal::max(a, params.min), params.max);
  }
};

struct clamp_min_scalar_functor {
  template <typename T>
  inline T operator()(const T a, const T b_min) {
    return c10::metal::max(a, b_min);
  }
};

struct clamp_max_scalar_functor {
  template <typename T>
  inline T operator()(const T a, const T b_max) {
    return c10::metal::min(a, b_max);
  }
};

#define REGISTER_CLAMP_SCALAR_OP(T)                   \
  typedef ClampScalarParams<T> ClampScalarParams_##T; \
  REGISTER_UNARY_ALPHA_OP(clamp_scalar, T, ClampScalarParams_##T, T);

#define REGISTER_ALL_CLAMP_OPS(T)                     \
  REGISTER_TERNARY_OP(clamp, T, T);                   \
  REGISTER_CLAMP_SCALAR_OP(T);                        \
  REGISTER_UNARY_ALPHA_OP(clamp_min_scalar, T, T, T); \
  REGISTER_UNARY_ALPHA_OP(clamp_max_scalar, T, T, T);

REGISTER_ALL_CLAMP_OPS(long);
REGISTER_ALL_CLAMP_OPS(int);
REGISTER_ALL_CLAMP_OPS(short);
REGISTER_ALL_CLAMP_OPS(uchar);
REGISTER_ALL_CLAMP_OPS(char);
REGISTER_ALL_CLAMP_OPS(bool);

REGISTER_ALL_CLAMP_OPS(float);
REGISTER_ALL_CLAMP_OPS(half);
REGISTER_ALL_CLAMP_OPS(bfloat);
