#include <ATen/native/mps/kernels/SoftMaxKernel.h>
#include <c10/metal/reduction_utils.h>
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
using c10::metal::simdgroup_size;

static inline ulong offset_a(uint row_idx, constant SoftmaxParams& p) {
  ulong offset = 0;
  uint idx = row_idx;
  for (int d = int(p.ndim) - 2; d >= 0; d--) {
    uint coord = idx % p.outer_sizes[d];
    idx /= p.outer_sizes[d];
    offset += ulong(coord) * ulong(p.outer_strides_a[d]);
  }
  return offset;
}

static inline ulong offset_b(uint row_idx, constant SoftmaxParams& p) {
  ulong offset = 0;
  uint idx = row_idx;
  for (int d = int(p.ndim) - 2; d >= 0; d--) {
    uint coord = idx % p.outer_sizes[d];
    idx /= p.outer_sizes[d];
    offset += ulong(coord) * ulong(p.outer_strides_b[d]);
  }
  return offset;
}

static inline ulong offset_c(uint row_idx, constant SoftmaxParams& p) {
  ulong offset = 0;
  uint idx = row_idx;
  for (int d = int(p.ndim) - 2; d >= 0; d--) {
    uint coord = idx % p.outer_sizes[d];
    idx /= p.outer_sizes[d];
    offset += ulong(coord) * ulong(p.outer_strides_c[d]);
  }
  return offset;
}

static inline float4 load_vec4(device const float* p) {
  return *reinterpret_cast<device const packed_float4*>(p);
}
static inline float4 load_vec4(device const half* p) {
  return float4(*reinterpret_cast<device const packed_half4*>(p));
}
static inline float4 load_vec4(device const bfloat* p) {
  return float4(float(p[0]), float(p[1]), float(p[2]), float(p[3]));
}

static inline void store_vec4(device float* p, float4 v) {
  *reinterpret_cast<device packed_float4*>(p) = v;
}
static inline void store_vec4(device half* p, float4 v) {
  *reinterpret_cast<device packed_half4*>(p) = half4(v);
}
static inline void store_vec4(device bfloat* p, float4 v) {
  p[0] = static_cast<bfloat>(v[0]);
  p[1] = static_cast<bfloat>(v[1]);
  p[2] = static_cast<bfloat>(v[2]);
  p[3] = static_cast<bfloat>(v[3]);
}

// Forward single-row: values cached in registers (1 read, 1 write).
// Reads from input using stride_a, writes to output contiguously.

template <typename T, int N_READS = 4>
kernel void softmax_forward_single_row(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  // N_READS elements per thread, loaded as N_READS/4 vec4 chunks. A wider
  // N_READS shrinks the threadgroup (fewer threads -> cheaper TG reduction and
  // more independent loads in flight per thread), which helps the small-byte
  // half-precision rows that are reduction/overhead bound rather than
  // bandwidth bound.
  constexpr int N_VEC = N_READS / 4;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  device const T* x = input + offset_a(tg_id, params);
  device T* out = output + offset_b(tg_id, params);
  uint base = tid * N_READS;

  bool contiguous = (sa == 1);
  float vals[N_READS];
  float local_max = -INFINITY;
  if (base + N_READS <= axis_size) {
    if (contiguous) {
#pragma unroll
      for (int c = 0; c < N_VEC; c++) {
        float4 v = load_vec4(x + base + c * 4);
        vals[c * 4 + 0] = v.x;
        vals[c * 4 + 1] = v.y;
        vals[c * 4 + 2] = v.z;
        vals[c * 4 + 3] = v.w;
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++)
        vals[i] = float(x[(base + i) * sa]);
    }
#pragma unroll
    for (int i = 0; i < N_READS; i++)
      local_max = fmax(local_max, vals[i]);
  } else {
    for (int i = 0; i < N_READS; i++) {
      vals[i] = (base + i < axis_size)
          ? (contiguous ? float(x[base + i]) : float(x[(base + i) * sa]))
          : -INFINITY;
      local_max = fmax(local_max, vals[i]);
    }
  }

  threadgroup float shared[simdgroup_size];
  float row_max = c10::metal::threadgroup_max(shared, local_max, tid, tptg);

  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < N_READS; i++) {
    vals[i] = metal::precise::exp(vals[i] - row_max);
    local_sum += vals[i];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_sum = c10::metal::threadgroup_sum(shared, local_sum, tid, tptg);
  float inv_sum = 1.0f / total_sum;

  if (base + N_READS <= axis_size) {
    if (sb == 1) {
#pragma unroll
      for (int c = 0; c < N_VEC; c++) {
        float4 result = float4(
                            vals[c * 4 + 0],
                            vals[c * 4 + 1],
                            vals[c * 4 + 2],
                            vals[c * 4 + 3]) *
            inv_sum;
        store_vec4(out + base + c * 4, result);
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++)
        out[(base + i) * sb] = static_cast<T>(vals[i] * inv_sum);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size)
        out[(base + i) * sb] = static_cast<T>(vals[i] * inv_sum);
    }
  }
}

// Forward looped: online softmax fuses max+sum into one pass over memory.

template <typename T>
kernel void softmax_forward_looped(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  device const T* x = input + offset_a(tg_id, params);
  device T* out = output + offset_b(tg_id, params);
  bool contiguous = (sa == 1);

  float local_max = -INFINITY;
  float local_sum = 0.0f;
  bool found_nan = false;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 v;
      if (contiguous) {
        v = load_vec4(x + base);
      } else {
        v = float4(
            x[base * sa],
            x[(base + 1) * sa],
            x[(base + 2) * sa],
            x[(base + 3) * sa]);
      }
      found_nan = found_nan || metal::any(metal::isnan(v));
      float chunk_max = fmax(fmax(v.x, v.y), fmax(v.z, v.w));
      float new_max = fmax(local_max, chunk_max);
      local_sum = (new_max > -INFINITY)
          ? local_sum * metal::precise::exp(local_max - new_max) +
              metal::precise::exp(v.x - new_max) +
              metal::precise::exp(v.y - new_max) +
              metal::precise::exp(v.z - new_max) +
              metal::precise::exp(v.w - new_max)
          : 0.0f;
      local_max = new_max;
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++) {
        float val = contiguous ? float(x[i]) : float(x[i * sa]);
        found_nan = found_nan || metal::isnan(val);
        float new_max = fmax(local_max, val);
        local_sum = (new_max > -INFINITY)
            ? local_sum * metal::precise::exp(local_max - new_max) +
                metal::precise::exp(val - new_max)
            : 0.0f;
        local_max = new_max;
      }
    }
  }

  // A NaN input must poison the whole row (CPU semantics): fmax() drops NaNs,
  // so a NaN seen while the running max was still -INFINITY never entered
  // local_sum. Applied after the scan so it is order-independent.
  if (found_nan) {
    local_sum = NAN;
  }

  float sg_max = simd_max(local_max);
  // Only rescale when this thread saw a finite max: if the whole simdgroup
  // saw just -inf, local_max == sg_max == -INFINITY and exp(-inf - -inf) is
  // NaN, turning the correct local_sum (0, or NaN from a NaN input) into
  // 0 * NaN == NaN. local_sum needs no rescale in that case.
  if (local_max > -INFINITY) {
    local_sum *= metal::precise::exp(local_max - sg_max);
  }
  float sg_sum = simd_sum(local_sum);

  threadgroup float shared_max[simdgroup_size];
  threadgroup float shared_sum[simdgroup_size];
  threadgroup float tg_result[2];

  if (simd_lane_id == 0) {
    shared_max[simdgroup_id] = sg_max;
    shared_sum[simdgroup_id] = sg_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint num_simdgroups = (lsize + simdgroup_size - 1) / simdgroup_size;
  if (simdgroup_id == 0) {
    float m =
        (simd_lane_id < num_simdgroups) ? shared_max[simd_lane_id] : -INFINITY;
    float global_max = simd_max(m);
    // Same rescale guard as above: a fully masked simdgroup has
    // m == -INFINITY with a 0 (or NaN-poisoned) sum that must pass through
    // unscaled; exp(m - global_max) is exp(nan) when global_max is also
    // -INFINITY (fully masked chunk/row).
    float s = (simd_lane_id < num_simdgroups) ? shared_sum[simd_lane_id] : 0.0f;
    if (m > -INFINITY) {
      s *= metal::precise::exp(m - global_max);
    }
    float global_sum = simd_sum(s);
    if (simd_lane_id == 0) {
      tg_result[0] = global_max;
      tg_result[1] = global_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float row_max = tg_result[0];
  float inv_sum = 1.0f / tg_result[1];

  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 v;
      if (contiguous) {
        v = metal::precise::exp(load_vec4(x + base) - row_max) * inv_sum;
      } else {
        v = float4(
            metal::precise::exp(float(x[base * sa]) - row_max) * inv_sum,
            metal::precise::exp(float(x[(base + 1) * sa]) - row_max) * inv_sum,
            metal::precise::exp(float(x[(base + 2) * sa]) - row_max) * inv_sum,
            metal::precise::exp(float(x[(base + 3) * sa]) - row_max) * inv_sum);
      }
      if (sb == 1) {
        store_vec4(out + base, v);
      } else {
#pragma unroll
        for (int i = 0; i < N_READS; i++)
          out[(base + i) * sb] = static_cast<T>(v[i]);
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++) {
        float val = contiguous ? float(x[i]) : float(x[i * sa]);
        out[i * sb] =
            static_cast<T>(metal::precise::exp(val - row_max) * inv_sum);
      }
    }
  }
}

// Two-pass forward for low-occupancy cases (few rows, large axis).
// Phase 1: each threadgroup computes (chunk_max, chunk_sum) via online
// algorithm. Phase 2: each threadgroup combines partials, re-reads input,
// writes output.

// Backward: grad_input = output * (grad_output - sum(grad_output * output))
// stride_a = grad_output strides, stride_b = output strides
// Writes grad_input contiguously.

template <typename T, int N_READS = 4>
kernel void softmax_backward_single_row(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device T* grad_input [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  // N_READS elements per thread, loaded as N_READS/4 vec4 chunks. A wider
  // N_READS shrinks the threadgroup (fewer threads -> cheaper TG reduction),
  // which mirrors the forward 8-wide path so half-precision last-dim fwdbwd
  // does not lose the forward speedup to a still-narrow backward pass.
  constexpr int N_VEC = N_READS / 4;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  uint sc = params.stride_c;
  device const T* dy = grad_output + offset_a(tg_id, params);
  device const T* y = output + offset_b(tg_id, params);
  device T* dx = grad_input + offset_c(tg_id, params);
  uint base = tid * N_READS;

  bool contiguous = (sa == 1) && (sb == 1);
  float dy_vals[N_READS];
  float y_vals[N_READS];
  float local_dot = 0.0f;
  if (base + N_READS <= axis_size) {
    if (contiguous) {
#pragma unroll
      for (int c = 0; c < N_VEC; c++) {
        float4 dy_v = load_vec4(dy + base + c * 4);
        float4 y_v = load_vec4(y + base + c * 4);
        dy_vals[c * 4 + 0] = dy_v.x;
        dy_vals[c * 4 + 1] = dy_v.y;
        dy_vals[c * 4 + 2] = dy_v.z;
        dy_vals[c * 4 + 3] = dy_v.w;
        y_vals[c * 4 + 0] = y_v.x;
        y_vals[c * 4 + 1] = y_v.y;
        y_vals[c * 4 + 2] = y_v.z;
        y_vals[c * 4 + 3] = y_v.w;
        local_dot += dot(dy_v, y_v);
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++) {
        dy_vals[i] = float(dy[(base + i) * sa]);
        y_vals[i] = float(y[(base + i) * sb]);
        local_dot += dy_vals[i] * y_vals[i];
      }
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size) {
        dy_vals[i] =
            contiguous ? float(dy[base + i]) : float(dy[(base + i) * sa]);
        y_vals[i] = contiguous ? float(y[base + i]) : float(y[(base + i) * sb]);
        local_dot += dy_vals[i] * y_vals[i];
      }
    }
  }

  threadgroup float shared_dot[simdgroup_size];
  float dot_sum = c10::metal::threadgroup_sum(shared_dot, local_dot, tid, tptg);

  if (base + N_READS <= axis_size) {
    if (sc == 1) {
#pragma unroll
      for (int c = 0; c < N_VEC; c++) {
        float4 result = float4(
                            y_vals[c * 4 + 0],
                            y_vals[c * 4 + 1],
                            y_vals[c * 4 + 2],
                            y_vals[c * 4 + 3]) *
            (float4(
                 dy_vals[c * 4 + 0],
                 dy_vals[c * 4 + 1],
                 dy_vals[c * 4 + 2],
                 dy_vals[c * 4 + 3]) -
             dot_sum);
        store_vec4(dx + base + c * 4, result);
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++)
        dx[(base + i) * sc] =
            static_cast<T>(y_vals[i] * (dy_vals[i] - dot_sum));
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size)
        dx[(base + i) * sc] =
            static_cast<T>(y_vals[i] * (dy_vals[i] - dot_sum));
    }
  }
}

// Backward looped: vectorized dot product with strided or contiguous access.

template <typename T>
kernel void softmax_backward_looped(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device T* grad_input [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  uint sc = params.stride_c;
  device const T* dy = grad_output + offset_a(tg_id, params);
  device const T* y = output + offset_b(tg_id, params);
  device T* dx = grad_input + offset_c(tg_id, params);
  bool contiguous = (sa == 1) && (sb == 1);

  float local_dot = 0.0f;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      if (contiguous) {
        local_dot += dot(load_vec4(dy + base), load_vec4(y + base));
      } else {
        float4 dy_v = float4(
            dy[base * sa],
            dy[(base + 1) * sa],
            dy[(base + 2) * sa],
            dy[(base + 3) * sa]);
        float4 y_v = float4(
            y[base * sb],
            y[(base + 1) * sb],
            y[(base + 2) * sb],
            y[(base + 3) * sb]);
        local_dot += dot(dy_v, y_v);
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++)
        local_dot += (contiguous ? float(dy[i]) : float(dy[i * sa])) *
            (contiguous ? float(y[i]) : float(y[i * sb]));
    }
  }

  threadgroup float shared_dot[simdgroup_size];
  float dot_sum =
      c10::metal::threadgroup_sum(shared_dot, local_dot, tid, lsize);

  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 y_v, dy_v;
      if (contiguous) {
        y_v = load_vec4(y + base);
        dy_v = load_vec4(dy + base);
      } else {
        y_v = float4(
            y[base * sb],
            y[(base + 1) * sb],
            y[(base + 2) * sb],
            y[(base + 3) * sb]);
        dy_v = float4(
            dy[base * sa],
            dy[(base + 1) * sa],
            dy[(base + 2) * sa],
            dy[(base + 3) * sa]);
      }
      float4 result = y_v * (dy_v - dot_sum);
      if (sc == 1) {
        store_vec4(dx + base, result);
      } else {
#pragma unroll
        for (int i = 0; i < N_READS; i++)
          dx[(base + i) * sc] = static_cast<T>(result[i]);
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++) {
        float yi = contiguous ? float(y[i]) : float(y[i * sb]);
        float dyi = contiguous ? float(dy[i]) : float(dy[i * sa]);
        dx[i * sc] = static_cast<T>(yi * (dyi - dot_sum));
      }
    }
  }
}

// Two-pass backward for low-occupancy cases (few rows, large axis).
// Phase 1: each threadgroup computes a partial dot(dy, y) over its chunk.
// Phase 2: each threadgroup sums partial dots, then computes grad_input for its
// chunk.

// Log-softmax forward 2-pass: for low-occupancy (outer_size < 4, large axis)
// Phase 1: each threadgroup computes (chunk_max, chunk_sum) via online
// algorithm Phase 2: combine partials, compute shift = max + log(sum), write
// output = x - shift Log-softmax backward: grad_input = grad_output -
// exp(output) * sum(grad_output)

#define instantiate_softmax_forward_single_row(DTYPE)                     \
  template [[host_name("softmax_forward_single_row_" #DTYPE)]] [[kernel]] \
  void softmax_forward_single_row<DTYPE, 4>(                              \
      device const DTYPE* input [[buffer(0)]],                            \
      device DTYPE* output [[buffer(1)]],                                 \
      constant SoftmaxParams& params [[buffer(2)]],                       \
      uint tg_id [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_position_in_threadgroup]],                        \
      uint tptg [[threads_per_threadgroup]],                              \
      uint simd_lane_id [[thread_index_in_simdgroup]],                    \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

// 8-wide single-row variant for half-precision last-dim rows.
#define instantiate_softmax_forward_single_row8(DTYPE)                     \
  template [[host_name("softmax_forward_single_row8_" #DTYPE)]] [[kernel]] \
  void softmax_forward_single_row<DTYPE, 8>(                               \
      device const DTYPE* input [[buffer(0)]],                             \
      device DTYPE* output [[buffer(1)]],                                  \
      constant SoftmaxParams& params [[buffer(2)]],                        \
      uint tg_id [[threadgroup_position_in_grid]],                         \
      uint tid [[thread_position_in_threadgroup]],                         \
      uint tptg [[threads_per_threadgroup]],                               \
      uint simd_lane_id [[thread_index_in_simdgroup]],                     \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_forward_looped(DTYPE)                          \
  template [[host_name("softmax_forward_looped_" #DTYPE)]] [[kernel]] void \
  softmax_forward_looped<DTYPE>(                                           \
      device const DTYPE* input [[buffer(0)]],                             \
      device DTYPE* output [[buffer(1)]],                                  \
      constant SoftmaxParams& params [[buffer(2)]],                        \
      uint tg_id [[threadgroup_position_in_grid]],                         \
      uint tid [[thread_position_in_threadgroup]],                         \
      uint lsize [[threads_per_threadgroup]],                              \
      uint simd_lane_id [[thread_index_in_simdgroup]],                     \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_backward_single_row(DTYPE)                     \
  template [[host_name("softmax_backward_single_row_" #DTYPE)]] [[kernel]] \
  void softmax_backward_single_row<DTYPE, 4>(                              \
      device const DTYPE* grad_output [[buffer(0)]],                       \
      device const DTYPE* output [[buffer(1)]],                            \
      device DTYPE* grad_input [[buffer(2)]],                              \
      constant SoftmaxParams& params [[buffer(3)]],                        \
      uint tg_id [[threadgroup_position_in_grid]],                         \
      uint tid [[thread_position_in_threadgroup]],                         \
      uint tptg [[threads_per_threadgroup]],                               \
      uint simd_lane_id [[thread_index_in_simdgroup]],                     \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

// 8-wide single-row backward variant for half-precision last-dim rows.
#define instantiate_softmax_backward_single_row8(DTYPE)                     \
  template [[host_name("softmax_backward_single_row8_" #DTYPE)]] [[kernel]] \
  void softmax_backward_single_row<DTYPE, 8>(                               \
      device const DTYPE* grad_output [[buffer(0)]],                        \
      device const DTYPE* output [[buffer(1)]],                             \
      device DTYPE* grad_input [[buffer(2)]],                               \
      constant SoftmaxParams& params [[buffer(3)]],                         \
      uint tg_id [[threadgroup_position_in_grid]],                          \
      uint tid [[thread_position_in_threadgroup]],                          \
      uint tptg [[threads_per_threadgroup]],                                \
      uint simd_lane_id [[thread_index_in_simdgroup]],                      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_backward_looped(DTYPE)                          \
  template [[host_name("softmax_backward_looped_" #DTYPE)]] [[kernel]] void \
  softmax_backward_looped<DTYPE>(                                           \
      device const DTYPE* grad_output [[buffer(0)]],                        \
      device const DTYPE* output [[buffer(1)]],                             \
      device DTYPE* grad_input [[buffer(2)]],                               \
      constant SoftmaxParams& params [[buffer(3)]],                         \
      uint tg_id [[threadgroup_position_in_grid]],                          \
      uint tid [[thread_position_in_threadgroup]],                          \
      uint lsize [[threads_per_threadgroup]],                               \
      uint simd_lane_id [[thread_index_in_simdgroup]],                      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax(DTYPE)                       \
  instantiate_softmax_forward_single_row(DTYPE)          \
      instantiate_softmax_forward_looped(DTYPE)          \
          instantiate_softmax_backward_single_row(DTYPE) \
              instantiate_softmax_backward_looped(DTYPE)

instantiate_softmax(float);
instantiate_softmax(half);
instantiate_softmax(bfloat);

// 8-wide single-row forward, half precision only (helps small-byte last-dim).
instantiate_softmax_forward_single_row8(half);
instantiate_softmax_forward_single_row8(bfloat);

// 8-wide single-row backward, half precision only (matches forward width so
// last-dim half fwdbwd does not regress against the 8-wide forward).
instantiate_softmax_backward_single_row8(half);
instantiate_softmax_backward_single_row8(bfloat);
