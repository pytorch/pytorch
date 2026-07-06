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
//
// IS_LOG templates softmax vs log_softmax: log_softmax writes
// (x - row_max - log(sum_exp)) instead of exp(x - row_max) / sum_exp, sharing
// the same online max/sum reduction. An all-(-inf) row has total_sum == 0, so
// for log_softmax log(0) == -inf gives -inf (matching CPU) rather than NaN.

template <typename T, bool IS_LOG = false, int N_READS = 4>
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

  // shifted[i] = vals[i] - row_max (needed raw by log_softmax); exp_vals[i] is
  // its exponential, summed either way to form sum_exp.
  float shifted[N_READS];
  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < N_READS; i++) {
    shifted[i] = vals[i] - row_max;
    vals[i] = metal::precise::exp(shifted[i]);
    local_sum += vals[i];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_sum = c10::metal::threadgroup_sum(shared, local_sum, tid, tptg);
  // log_softmax: out = shifted - log(sum_exp). An all-(-inf) row has sum 0, so
  // log(0) = -inf yields -inf (matches CPU), not NaN.
  float inv_sum = IS_LOG ? metal::precise::log(total_sum) : (1.0f / total_sum);

  if (base + N_READS <= axis_size) {
    if (sb == 1) {
#pragma unroll
      for (int c = 0; c < N_VEC; c++) {
        float4 result = IS_LOG ? (float4(
                                      shifted[c * 4 + 0],
                                      shifted[c * 4 + 1],
                                      shifted[c * 4 + 2],
                                      shifted[c * 4 + 3]) -
                                  inv_sum)
                               : (float4(
                                      vals[c * 4 + 0],
                                      vals[c * 4 + 1],
                                      vals[c * 4 + 2],
                                      vals[c * 4 + 3]) *
                                  inv_sum);
        store_vec4(out + base + c * 4, result);
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++)
        out[(base + i) * sb] = static_cast<T>(
            IS_LOG ? (shifted[i] - inv_sum) : (vals[i] * inv_sum));
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size)
        out[(base + i) * sb] = static_cast<T>(
            IS_LOG ? (shifted[i] - inv_sum) : (vals[i] * inv_sum));
    }
  }
}

// Forward looped: online softmax fuses max+sum into one pass over memory.
// IS_LOG selects log_softmax (write x - row_max - log(sum_exp)).

template <typename T, bool IS_LOG = false>
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
        float new_max = fmax(local_max, val);
        local_sum = (new_max > -INFINITY)
            ? local_sum * metal::precise::exp(local_max - new_max) +
                metal::precise::exp(val - new_max)
            : 0.0f;
        local_max = new_max;
      }
    }
  }

  float sg_max = simd_max(local_max);
  local_sum *= metal::precise::exp(local_max - sg_max);
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
    float s = (simd_lane_id < num_simdgroups)
        ? shared_sum[simd_lane_id] * metal::precise::exp(m - global_max)
        : 0.0f;
    float global_sum = simd_sum(s);
    if (simd_lane_id == 0) {
      tg_result[0] = global_max;
      tg_result[1] = global_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float row_max = tg_result[0];
  // log_softmax: subtract log(sum_exp); softmax: multiply by 1/sum_exp.
  float norm =
      IS_LOG ? metal::precise::log(tg_result[1]) : (1.0f / tg_result[1]);

  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 v;
      if (contiguous) {
        float4 sh = load_vec4(x + base) - row_max;
        v = IS_LOG ? (sh - norm) : (metal::precise::exp(sh) * norm);
      } else {
        float4 sh = float4(
                        float(x[base * sa]),
                        float(x[(base + 1) * sa]),
                        float(x[(base + 2) * sa]),
                        float(x[(base + 3) * sa])) -
            row_max;
        v = IS_LOG ? (sh - norm) : (metal::precise::exp(sh) * norm);
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
        float sh = val - row_max;
        out[i * sb] = static_cast<T>(
            IS_LOG ? (sh - norm) : (metal::precise::exp(sh) * norm));
      }
    }
  }
}

// Two-pass forward for low-occupancy cases (few rows, large axis).
// Phase 1: each threadgroup computes (chunk_max, chunk_sum) via online
// algorithm. Phase 2: each threadgroup combines partials, re-reads input,
// writes output.

template <typename T>
kernel void softmax_forward_2pass_reduce(
    device const T* input [[buffer(0)]],
    device float* partials [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint num_chunks = params.num_chunks;
  uint chunk_id = tg_id % num_chunks;
  uint row_id = tg_id / num_chunks;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  device const T* x = input + offset_a(row_id, params);
  bool contiguous = (sa == 1);

  uint elems_per_chunk = (axis_size + num_chunks - 1) / num_chunks;
  uint start = chunk_id * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (uint r = start; r < end; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= end) {
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
      for (uint i = base; i < min(base + uint(N_READS), end); i++) {
        float val = contiguous ? float(x[i]) : float(x[i * sa]);
        float new_max = fmax(local_max, val);
        local_sum = (new_max > -INFINITY)
            ? local_sum * metal::precise::exp(local_max - new_max) +
                metal::precise::exp(val - new_max)
            : 0.0f;
        local_max = new_max;
      }
    }
  }

  float sg_max = simd_max(local_max);
  local_sum *= metal::precise::exp(local_max - sg_max);
  float sg_sum = simd_sum(local_sum);

  threadgroup float shared_max[simdgroup_size];
  threadgroup float shared_sum[simdgroup_size];

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
    float s = (simd_lane_id < num_simdgroups)
        ? shared_sum[simd_lane_id] * metal::precise::exp(m - global_max)
        : 0.0f;
    float global_sum = simd_sum(s);
    if (simd_lane_id == 0) {
      partials[(row_id * num_chunks + chunk_id) * 2] = global_max;
      partials[(row_id * num_chunks + chunk_id) * 2 + 1] = global_sum;
    }
  }
}

// IS_LOG selects log_softmax for the write phase. The reduce phase (max, sum)
// is identical for both and is shared (not templated on IS_LOG).
template <typename T, bool IS_LOG = false>
kernel void softmax_forward_2pass_write(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device const float* partials [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  constexpr int N_READS = 4;
  uint num_chunks = params.num_chunks;
  uint chunk_id = tg_id % num_chunks;
  uint row_id = tg_id / num_chunks;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  device const T* x = input + offset_a(row_id, params);
  device T* out = output + offset_b(row_id, params);
  bool contiguous = (sa == 1);

  float global_max = -INFINITY;
  float global_sum = 0.0f;
  for (uint i = 0; i < num_chunks; i++) {
    float chunk_max = partials[(row_id * num_chunks + i) * 2];
    float chunk_sum = partials[(row_id * num_chunks + i) * 2 + 1];
    float new_max = fmax(global_max, chunk_max);
    global_sum = (new_max > -INFINITY)
        ? global_sum * metal::precise::exp(global_max - new_max) +
            chunk_sum * metal::precise::exp(chunk_max - new_max)
        : 0.0f;
    global_max = new_max;
  }
  // log_softmax: subtract log(sum_exp); softmax: multiply by 1/sum_exp.
  float norm = IS_LOG ? metal::precise::log(global_sum) : (1.0f / global_sum);

  uint elems_per_chunk = (axis_size + num_chunks - 1) / num_chunks;
  uint start = chunk_id * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  for (uint r = start; r < end; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= end) {
      float4 v;
      if (contiguous) {
        float4 sh = load_vec4(x + base) - global_max;
        v = IS_LOG ? (sh - norm) : (metal::precise::exp(sh) * norm);
      } else {
        float4 sh = float4(
                        float(x[base * sa]),
                        float(x[(base + 1) * sa]),
                        float(x[(base + 2) * sa]),
                        float(x[(base + 3) * sa])) -
            global_max;
        v = IS_LOG ? (sh - norm) : (metal::precise::exp(sh) * norm);
      }
      if (sb == 1) {
        store_vec4(out + base, v);
      } else {
#pragma unroll
        for (int i = 0; i < N_READS; i++)
          out[(base + i) * sb] = static_cast<T>(v[i]);
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), end); i++) {
        float val = contiguous ? float(x[i]) : float(x[i * sa]);
        float sh = val - global_max;
        out[i * sb] = static_cast<T>(
            IS_LOG ? (sh - norm) : (metal::precise::exp(sh) * norm));
      }
    }
  }
}

// Backward (softmax): grad_input = output * (grad_output - sum(grad_output *
// output)) Backward (log_softmax): grad_input = grad_output - exp(output) *
// sum(grad_output)
//   - here `output` is the log_softmax result, so exp(output) is the softmax.
//   - the threadgroup reduction is sum(grad_output) (not the dot product).
// stride_a = grad_output strides, stride_b = output strides
// Writes grad_input contiguously.

template <typename T, bool IS_LOG = false, int N_READS = 4>
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
  // softmax: reduce sum(dy*y); log_softmax: reduce sum(dy).
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
        local_dot +=
            IS_LOG ? (dy_v.x + dy_v.y + dy_v.z + dy_v.w) : dot(dy_v, y_v);
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++) {
        dy_vals[i] = float(dy[(base + i) * sa]);
        y_vals[i] = float(y[(base + i) * sb]);
        local_dot += IS_LOG ? dy_vals[i] : (dy_vals[i] * y_vals[i]);
      }
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size) {
        dy_vals[i] =
            contiguous ? float(dy[base + i]) : float(dy[(base + i) * sa]);
        y_vals[i] = contiguous ? float(y[base + i]) : float(y[(base + i) * sb]);
        local_dot += IS_LOG ? dy_vals[i] : (dy_vals[i] * y_vals[i]);
      }
    }
  }

  threadgroup float shared_dot[simdgroup_size];
  float dot_sum = c10::metal::threadgroup_sum(shared_dot, local_dot, tid, tptg);

  if (base + N_READS <= axis_size) {
    if (sc == 1) {
#pragma unroll
      for (int c = 0; c < N_VEC; c++) {
        float4 dyv = float4(
            dy_vals[c * 4 + 0],
            dy_vals[c * 4 + 1],
            dy_vals[c * 4 + 2],
            dy_vals[c * 4 + 3]);
        float4 yv = float4(
            y_vals[c * 4 + 0],
            y_vals[c * 4 + 1],
            y_vals[c * 4 + 2],
            y_vals[c * 4 + 3]);
        float4 result = IS_LOG ? (dyv - metal::precise::exp(yv) * dot_sum)
                               : (yv * (dyv - dot_sum));
        store_vec4(dx + base + c * 4, result);
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++)
        dx[(base + i) * sc] = static_cast<T>(
            IS_LOG ? (dy_vals[i] - metal::precise::exp(y_vals[i]) * dot_sum)
                   : (y_vals[i] * (dy_vals[i] - dot_sum)));
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size)
        dx[(base + i) * sc] = static_cast<T>(
            IS_LOG ? (dy_vals[i] - metal::precise::exp(y_vals[i]) * dot_sum)
                   : (y_vals[i] * (dy_vals[i] - dot_sum)));
    }
  }
}

// Backward looped: vectorized dot product with strided or contiguous access.
// IS_LOG selects log_softmax (reduce sum(dy), write dy - exp(out)*sum).

template <typename T, bool IS_LOG = false>
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
        float4 dy_v = load_vec4(dy + base);
        local_dot += IS_LOG ? (dy_v.x + dy_v.y + dy_v.z + dy_v.w)
                            : dot(dy_v, load_vec4(y + base));
      } else {
        float4 dy_v = float4(
            dy[base * sa],
            dy[(base + 1) * sa],
            dy[(base + 2) * sa],
            dy[(base + 3) * sa]);
        if (IS_LOG) {
          local_dot += dy_v.x + dy_v.y + dy_v.z + dy_v.w;
        } else {
          float4 y_v = float4(
              y[base * sb],
              y[(base + 1) * sb],
              y[(base + 2) * sb],
              y[(base + 3) * sb]);
          local_dot += dot(dy_v, y_v);
        }
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++) {
        float dyi = contiguous ? float(dy[i]) : float(dy[i * sa]);
        local_dot +=
            IS_LOG ? dyi : dyi * (contiguous ? float(y[i]) : float(y[i * sb]));
      }
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
      float4 result = IS_LOG ? (dy_v - metal::precise::exp(y_v) * dot_sum)
                             : (y_v * (dy_v - dot_sum));
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
        dx[i * sc] = static_cast<T>(
            IS_LOG ? (dyi - metal::precise::exp(yi) * dot_sum)
                   : (yi * (dyi - dot_sum)));
      }
    }
  }
}

// Two-pass backward for low-occupancy cases (few rows, large axis).
// Phase 1: each threadgroup computes a partial dot(dy, y) over its chunk.
// Phase 2: each threadgroup sums partial dots, then computes grad_input for its
// chunk.

template <typename T, bool IS_LOG = false>
kernel void softmax_backward_2pass_dot(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device float* partial_sums [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint num_chunks = params.num_chunks;
  uint chunk_id = tg_id % num_chunks;
  uint row_id = tg_id / num_chunks;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  device const T* dy = grad_output + offset_a(row_id, params);
  device const T* y = output + offset_b(row_id, params);
  bool contiguous = (sa == 1) && (sb == 1);

  uint elems_per_chunk = (axis_size + num_chunks - 1) / num_chunks;
  uint start = chunk_id * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  // softmax: partial sum(dy*y); log_softmax: partial sum(dy).
  float local_dot = 0.0f;
  for (uint r = start; r < end; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= end) {
      if (contiguous) {
        float4 dy_v = load_vec4(dy + base);
        local_dot += IS_LOG ? (dy_v.x + dy_v.y + dy_v.z + dy_v.w)
                            : dot(dy_v, load_vec4(y + base));
      } else {
        float4 dy_v = float4(
            dy[base * sa],
            dy[(base + 1) * sa],
            dy[(base + 2) * sa],
            dy[(base + 3) * sa]);
        if (IS_LOG) {
          local_dot += dy_v.x + dy_v.y + dy_v.z + dy_v.w;
        } else {
          float4 y_v = float4(
              y[base * sb],
              y[(base + 1) * sb],
              y[(base + 2) * sb],
              y[(base + 3) * sb]);
          local_dot += dot(dy_v, y_v);
        }
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), end); i++) {
        float dyi = contiguous ? float(dy[i]) : float(dy[i * sa]);
        local_dot +=
            IS_LOG ? dyi : dyi * (contiguous ? float(y[i]) : float(y[i * sb]));
      }
    }
  }

  threadgroup float shared_dot[simdgroup_size];
  float d = c10::metal::threadgroup_sum(shared_dot, local_dot, tid, lsize);
  if (tid == 0)
    partial_sums[row_id * num_chunks + chunk_id] = d;
}

template <typename T, bool IS_LOG = false>
kernel void softmax_backward_2pass_grad(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device T* grad_input [[buffer(2)]],
    device const float* partial_sums [[buffer(3)]],
    constant SoftmaxParams& params [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  constexpr int N_READS = 4;
  uint num_chunks = params.num_chunks;
  uint chunk_id = tg_id % num_chunks;
  uint row_id = tg_id / num_chunks;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  uint sc = params.stride_c;
  device const T* dy = grad_output + offset_a(row_id, params);
  device const T* y = output + offset_b(row_id, params);
  device T* dx = grad_input + offset_c(row_id, params);
  bool contiguous = (sa == 1) && (sb == 1);

  float dot_sum = 0.0f;
  for (uint i = 0; i < num_chunks; i++)
    dot_sum += partial_sums[row_id * num_chunks + i];

  uint elems_per_chunk = (axis_size + num_chunks - 1) / num_chunks;
  uint start = chunk_id * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  for (uint r = start; r < end; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= end) {
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
      float4 result = IS_LOG ? (dy_v - metal::precise::exp(y_v) * dot_sum)
                             : (y_v * (dy_v - dot_sum));
      if (sc == 1) {
        store_vec4(dx + base, result);
      } else {
#pragma unroll
        for (int i = 0; i < N_READS; i++)
          dx[(base + i) * sc] = static_cast<T>(result[i]);
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), end); i++) {
        float yi = contiguous ? float(y[i]) : float(y[i * sb]);
        float dyi = contiguous ? float(dy[i]) : float(dy[i * sa]);
        dx[i * sc] = static_cast<T>(
            IS_LOG ? (dyi - metal::precise::exp(yi) * dot_sum)
                   : (yi * (dyi - dot_sum)));
      }
    }
  }
}

// Tiled forward/backward kernels for non-last-dim softmax.
// Each thread computes softmax for all axis elements at one inner position.
// Adjacent threads access adjacent memory - coalesced reads and writes.
// Uses num_chunks to store the number of inner tiles.

template <typename T, bool IS_LOG = false>
kernel void softmax_forward_tiled(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  uint num_tiles = params.num_chunks;
  uint batch_idx = tg_id / num_tiles;
  uint tile_idx = tg_id % num_tiles;
  uint inner_pos = tile_idx * lsize + tid;

  if (inner_pos >= sa)
    return;

  ulong base_a = ulong(batch_idx) * axis_size * sa + inner_pos;
  ulong base_b = ulong(batch_idx) * axis_size * sb + inner_pos;

  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (uint b = 0; b < axis_size; b++) {
    float val = float(input[base_a + ulong(b) * sa]);
    float new_max = fmax(local_max, val);
    local_sum = (new_max > -INFINITY)
        ? local_sum * metal::precise::exp(local_max - new_max) +
            metal::precise::exp(val - new_max)
        : 0.0f;
    local_max = new_max;
  }
  // log_softmax: subtract log(sum_exp); softmax: multiply by 1/sum_exp.
  float norm = IS_LOG ? metal::precise::log(local_sum) : (1.0f / local_sum);

  for (uint b = 0; b < axis_size; b++) {
    float val = float(input[base_a + ulong(b) * sa]);
    float sh = val - local_max;
    output[base_b + ulong(b) * sb] =
        static_cast<T>(IS_LOG ? (sh - norm) : (metal::precise::exp(sh) * norm));
  }
}

template <typename T, bool IS_LOG = false>
kernel void softmax_backward_tiled(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device T* grad_input [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  constexpr uint kTileAxisCap = 32;
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  uint sc = params.stride_c;
  uint num_tiles = params.num_chunks;
  uint batch_idx = tg_id / num_tiles;
  uint tile_idx = tg_id % num_tiles;
  uint inner_pos = tile_idx * lsize + tid;

  if (inner_pos >= sa)
    return;

  ulong base_a = ulong(batch_idx) * axis_size * sa + inner_pos;
  ulong base_b = ulong(batch_idx) * axis_size * sb + inner_pos;
  ulong base_c = ulong(batch_idx) * axis_size * sc + inner_pos;

  // softmax: dot_sum = sum(dy*y); log_softmax: dot_sum = sum(dy).
  float dot_sum = 0.0f;
  if (axis_size <= kTileAxisCap) {
    float dy_cache[kTileAxisCap];
    float y_cache[kTileAxisCap];
    for (uint b = 0; b < axis_size; b++) {
      dy_cache[b] = float(grad_output[base_a + ulong(b) * sa]);
      y_cache[b] = float(output[base_b + ulong(b) * sb]);
      dot_sum += IS_LOG ? dy_cache[b] : (dy_cache[b] * y_cache[b]);
    }
    for (uint b = 0; b < axis_size; b++)
      grad_input[base_c + ulong(b) * sc] = static_cast<T>(
          IS_LOG ? (dy_cache[b] - metal::precise::exp(y_cache[b]) * dot_sum)
                 : (y_cache[b] * (dy_cache[b] - dot_sum)));
  } else {
    for (uint b = 0; b < axis_size; b++) {
      float dyi = float(grad_output[base_a + ulong(b) * sa]);
      dot_sum += IS_LOG ? dyi : (dyi * float(output[base_b + ulong(b) * sb]));
    }
    for (uint b = 0; b < axis_size; b++) {
      float dyi = float(grad_output[base_a + ulong(b) * sa]);
      float yi = float(output[base_b + ulong(b) * sb]);
      grad_input[base_c + ulong(b) * sc] = static_cast<T>(
          IS_LOG ? (dyi - metal::precise::exp(yi) * dot_sum)
                 : (yi * (dyi - dot_sum)));
    }
  }
}

// Blocked non-last-dim kernels (cooperative axis reduction).
//
// For non-last-dim softmax the axis elements of one column are strided by
// inner_size in memory, while adjacent columns are unit-strided. The tiled
// kernel assigns one thread per column and walks the whole axis serially, which
// is coalesced across threads but offers zero per-column parallelism: when the
// axis is large (square dim=0, e.g. 1024x1024, or tall dim=0, e.g. 65536x128)
// each thread does a long serial 2-pass reduction and the kernel runs far below
// memory bandwidth.
//
// These kernels keep the coalesced access pattern (adjacent threads -> adjacent
// columns) but add per-column parallelism: a threadgroup owns COLS_PER_TG
// adjacent columns (num_chunks) and packs num_axis_threads (lsize/COLS_PER_TG)
// threads per column that cooperate over the axis via a shared-memory tree
// reduction. Unlike the coalesced kernel this works for any axis_size (no 16384
// cap) because the per-thread axis slice is axis_size/num_axis_threads, and
// COLS_PER_TG is independent of inner_size so it scales to large inner_size.
//
//   tid layout:  col = tid % cols_per_tg, axis_tid = tid / cols_per_tg
//   col_global = tg_id * cols_per_tg + col
//   params.num_chunks = cols_per_tg

template <typename T, bool IS_LOG = false>
kernel void softmax_forward_blocked(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]) {
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  uint cols_per_tg = params.num_chunks;
  uint num_axis_threads = lsize / cols_per_tg;
  uint num_col_tiles = (sa + cols_per_tg - 1) / cols_per_tg;
  uint batch_idx = tg_id / num_col_tiles;
  uint col_tile = tg_id % num_col_tiles;
  uint col = tid % cols_per_tg;
  uint axis_tid = tid / cols_per_tg;
  uint col_global = col_tile * cols_per_tg + col;
  bool active = (col_global < sa);

  // Each thread reduces a strided slice of the axis for its column. Adjacent
  // tids (adjacent col) read adjacent addresses -> coalesced.
  ulong col_base_a = ulong(batch_idx) * axis_size * sa + ulong(col_global);
  float local_max = -INFINITY;
  float local_sum = 0.0f;
  if (active) {
    for (uint b = axis_tid; b < axis_size; b += num_axis_threads) {
      float val = float(input[col_base_a + ulong(b) * sa]);
      float new_max = fmax(local_max, val);
      local_sum = (new_max > -INFINITY)
          ? local_sum * metal::precise::exp(local_max - new_max) +
              metal::precise::exp(val - new_max)
          : 0.0f;
      local_max = new_max;
    }
  }

  // Tree reduction across the num_axis_threads sharing each column.
  threadgroup float* mx = smem;
  threadgroup float* sm = smem + lsize;
  mx[tid] = local_max;
  sm[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = num_axis_threads / 2; s > 0; s >>= 1) {
    if (axis_tid < s) {
      uint o = tid + s * cols_per_tg;
      float om = mx[o], os = sm[o], mm = mx[tid], ms = sm[tid];
      float nm = fmax(mm, om);
      sm[tid] = (nm > -INFINITY) ? ms * metal::precise::exp(mm - nm) +
              os * metal::precise::exp(om - nm)
                                 : 0.0f;
      mx[tid] = nm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float row_max = mx[col];
  // log_softmax: subtract log(sum_exp); softmax: multiply by 1/sum_exp.
  float norm = IS_LOG ? metal::precise::log(sm[col]) : (1.0f / sm[col]);

  if (active) {
    ulong col_base_b = ulong(batch_idx) * axis_size * sb + ulong(col_global);
    for (uint b = axis_tid; b < axis_size; b += num_axis_threads) {
      float val = float(input[col_base_a + ulong(b) * sa]);
      float sh = val - row_max;
      output[col_base_b + ulong(b) * sb] = static_cast<T>(
          IS_LOG ? (sh - norm) : (metal::precise::exp(sh) * norm));
    }
  }
}

template <typename T, bool IS_LOG = false>
kernel void softmax_backward_blocked(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device T* grad_input [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]) {
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  uint sc = params.stride_c;
  uint cols_per_tg = params.num_chunks;
  uint num_axis_threads = lsize / cols_per_tg;
  uint num_col_tiles = (sa + cols_per_tg - 1) / cols_per_tg;
  uint batch_idx = tg_id / num_col_tiles;
  uint col_tile = tg_id % num_col_tiles;
  uint col = tid % cols_per_tg;
  uint axis_tid = tid / cols_per_tg;
  uint col_global = col_tile * cols_per_tg + col;
  bool active = (col_global < sa);

  ulong col_base_a = ulong(batch_idx) * axis_size * sa + ulong(col_global);
  ulong col_base_b = ulong(batch_idx) * axis_size * sb + ulong(col_global);
  // softmax: sum(dy*y); log_softmax: sum(dy).
  float local_dot = 0.0f;
  if (active) {
    for (uint b = axis_tid; b < axis_size; b += num_axis_threads) {
      float dyi = float(grad_output[col_base_a + ulong(b) * sa]);
      local_dot +=
          IS_LOG ? dyi : (dyi * float(output[col_base_b + ulong(b) * sb]));
    }
  }

  smem[tid] = local_dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = num_axis_threads / 2; s > 0; s >>= 1) {
    if (axis_tid < s)
      smem[tid] += smem[tid + s * cols_per_tg];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float dot_sum = smem[col];

  if (active) {
    ulong col_base_c = ulong(batch_idx) * axis_size * sc + ulong(col_global);
    for (uint b = axis_tid; b < axis_size; b += num_axis_threads) {
      float dyi = float(grad_output[col_base_a + ulong(b) * sa]);
      float yi = float(output[col_base_b + ulong(b) * sb]);
      grad_input[col_base_c + ulong(b) * sc] = static_cast<T>(
          IS_LOG ? (dyi - metal::precise::exp(yi) * dot_sum)
                 : (yi * (dyi - dot_sum)));
    }
  }
}

// Two-pass blocked non-last-dim kernels for the LOW-COLUMN, long-axis case
// (tall dim=0, e.g. 65536x128: only 128 columns, axis 65536). The single-pass
// blocked kernel can only spawn inner_size/cols_per_tg threadgroups, which
// starves occupancy when columns are few. These kernels add an axis-split grid
// dimension (num_col_chunks chunks) so the grid is
// outer_before * num_col_tiles * num_col_chunks threadgroups.
//
//   tg_id decode: axis_chunk = tg_id % num_col_chunks
//                 col_tile   = (tg_id / num_col_chunks) % num_col_tiles
//                 batch_idx  =  tg_id / num_col_chunks  / num_col_tiles
//   partials layout (2 floats per (column, axis_chunk)):
//     [(batch * inner + col_global) * num_col_chunks + axis_chunk] * 2 +
//     {0:max,1:sum}

template <typename T>
kernel void softmax_forward_blocked2_reduce(
    device const T* input [[buffer(0)]],
    device float* partials [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]) {
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint cols_per_tg = params.num_chunks;
  uint num_axis_chunks = params.num_col_chunks;
  uint num_axis_threads = lsize / cols_per_tg;
  uint num_col_tiles = (sa + cols_per_tg - 1) / cols_per_tg;

  uint axis_chunk = tg_id % num_axis_chunks;
  uint t2 = tg_id / num_axis_chunks;
  uint col_tile = t2 % num_col_tiles;
  uint batch_idx = t2 / num_col_tiles;
  uint col = tid % cols_per_tg;
  uint axis_tid = tid / cols_per_tg;
  uint col_global = col_tile * cols_per_tg + col;
  bool active = (col_global < sa);

  uint elems_per_chunk = (axis_size + num_axis_chunks - 1) / num_axis_chunks;
  uint start = axis_chunk * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  ulong col_base_a = ulong(batch_idx) * axis_size * sa + ulong(col_global);
  float local_max = -INFINITY;
  float local_sum = 0.0f;
  if (active) {
    for (uint b = start + axis_tid; b < end; b += num_axis_threads) {
      float val = float(input[col_base_a + ulong(b) * sa]);
      float new_max = fmax(local_max, val);
      local_sum = (new_max > -INFINITY)
          ? local_sum * metal::precise::exp(local_max - new_max) +
              metal::precise::exp(val - new_max)
          : 0.0f;
      local_max = new_max;
    }
  }

  threadgroup float* mx = smem;
  threadgroup float* sm = smem + lsize;
  mx[tid] = local_max;
  sm[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = num_axis_threads / 2; s > 0; s >>= 1) {
    if (axis_tid < s) {
      uint o = tid + s * cols_per_tg;
      float om = mx[o], os = sm[o], mm = mx[tid], ms = sm[tid];
      float nm = fmax(mm, om);
      sm[tid] = (nm > -INFINITY) ? ms * metal::precise::exp(mm - nm) +
              os * metal::precise::exp(om - nm)
                                 : 0.0f;
      mx[tid] = nm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (axis_tid == 0 && active) {
    ulong pidx = (ulong(batch_idx) * sa + ulong(col_global)) * num_axis_chunks +
        axis_chunk;
    partials[pidx * 2] = mx[col];
    partials[pidx * 2 + 1] = sm[col];
  }
}

template <typename T, bool IS_LOG = false>
kernel void softmax_forward_blocked2_write(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device const float* partials [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  uint cols_per_tg = params.num_chunks;
  uint num_axis_chunks = params.num_col_chunks;
  uint num_axis_threads = lsize / cols_per_tg;
  uint num_col_tiles = (sa + cols_per_tg - 1) / cols_per_tg;

  uint axis_chunk = tg_id % num_axis_chunks;
  uint t2 = tg_id / num_axis_chunks;
  uint col_tile = t2 % num_col_tiles;
  uint batch_idx = t2 / num_col_tiles;
  uint col = tid % cols_per_tg;
  uint axis_tid = tid / cols_per_tg;
  uint col_global = col_tile * cols_per_tg + col;
  bool active = (col_global < sa);

  // Combine all axis-chunk partials for this column (serial over chunks).
  float global_max = -INFINITY;
  float global_sum = 0.0f;
  if (active) {
    ulong pbase = (ulong(batch_idx) * sa + ulong(col_global)) * num_axis_chunks;
    for (uint i = 0; i < num_axis_chunks; i++) {
      float cm = partials[(pbase + i) * 2];
      float cs = partials[(pbase + i) * 2 + 1];
      float nm = fmax(global_max, cm);
      global_sum = (nm > -INFINITY)
          ? global_sum * metal::precise::exp(global_max - nm) +
              cs * metal::precise::exp(cm - nm)
          : 0.0f;
      global_max = nm;
    }
  }
  // log_softmax: subtract log(sum_exp); softmax: multiply by 1/sum_exp.
  float norm = IS_LOG ? metal::precise::log(global_sum) : (1.0f / global_sum);

  uint elems_per_chunk = (axis_size + num_axis_chunks - 1) / num_axis_chunks;
  uint start = axis_chunk * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  if (active) {
    ulong col_base_a = ulong(batch_idx) * axis_size * sa + ulong(col_global);
    ulong col_base_b = ulong(batch_idx) * axis_size * sb + ulong(col_global);
    for (uint b = start + axis_tid; b < end; b += num_axis_threads) {
      float val = float(input[col_base_a + ulong(b) * sa]);
      float sh = val - global_max;
      output[col_base_b + ulong(b) * sb] = static_cast<T>(
          IS_LOG ? (sh - norm) : (metal::precise::exp(sh) * norm));
    }
  }
}

template <typename T, bool IS_LOG = false>
kernel void softmax_backward_blocked2_dot(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device float* partials [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]) {
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  uint cols_per_tg = params.num_chunks;
  uint num_axis_chunks = params.num_col_chunks;
  uint num_axis_threads = lsize / cols_per_tg;
  uint num_col_tiles = (sa + cols_per_tg - 1) / cols_per_tg;

  uint axis_chunk = tg_id % num_axis_chunks;
  uint t2 = tg_id / num_axis_chunks;
  uint col_tile = t2 % num_col_tiles;
  uint batch_idx = t2 / num_col_tiles;
  uint col = tid % cols_per_tg;
  uint axis_tid = tid / cols_per_tg;
  uint col_global = col_tile * cols_per_tg + col;
  bool active = (col_global < sa);

  uint elems_per_chunk = (axis_size + num_axis_chunks - 1) / num_axis_chunks;
  uint start = axis_chunk * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  ulong col_base_a = ulong(batch_idx) * axis_size * sa + ulong(col_global);
  ulong col_base_b = ulong(batch_idx) * axis_size * sb + ulong(col_global);
  // softmax: sum(dy*y); log_softmax: sum(dy).
  float local_dot = 0.0f;
  if (active) {
    for (uint b = start + axis_tid; b < end; b += num_axis_threads) {
      float dyi = float(grad_output[col_base_a + ulong(b) * sa]);
      local_dot +=
          IS_LOG ? dyi : (dyi * float(output[col_base_b + ulong(b) * sb]));
    }
  }

  smem[tid] = local_dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = num_axis_threads / 2; s > 0; s >>= 1) {
    if (axis_tid < s)
      smem[tid] += smem[tid + s * cols_per_tg];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (axis_tid == 0 && active) {
    ulong pidx = (ulong(batch_idx) * sa + ulong(col_global)) * num_axis_chunks +
        axis_chunk;
    partials[pidx] = smem[col];
  }
}

template <typename T, bool IS_LOG = false>
kernel void softmax_backward_blocked2_grad(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device T* grad_input [[buffer(2)]],
    device const float* partials [[buffer(3)]],
    constant SoftmaxParams& params [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  uint sc = params.stride_c;
  uint cols_per_tg = params.num_chunks;
  uint num_axis_chunks = params.num_col_chunks;
  uint num_axis_threads = lsize / cols_per_tg;
  uint num_col_tiles = (sa + cols_per_tg - 1) / cols_per_tg;

  uint axis_chunk = tg_id % num_axis_chunks;
  uint t2 = tg_id / num_axis_chunks;
  uint col_tile = t2 % num_col_tiles;
  uint batch_idx = t2 / num_col_tiles;
  uint col = tid % cols_per_tg;
  uint axis_tid = tid / cols_per_tg;
  uint col_global = col_tile * cols_per_tg + col;
  bool active = (col_global < sa);

  float dot_sum = 0.0f;
  if (active) {
    ulong pbase = (ulong(batch_idx) * sa + ulong(col_global)) * num_axis_chunks;
    for (uint i = 0; i < num_axis_chunks; i++)
      dot_sum += partials[pbase + i];
  }

  uint elems_per_chunk = (axis_size + num_axis_chunks - 1) / num_axis_chunks;
  uint start = axis_chunk * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  if (active) {
    ulong col_base_a = ulong(batch_idx) * axis_size * sa + ulong(col_global);
    ulong col_base_b = ulong(batch_idx) * axis_size * sb + ulong(col_global);
    ulong col_base_c = ulong(batch_idx) * axis_size * sc + ulong(col_global);
    for (uint b = start + axis_tid; b < end; b += num_axis_threads) {
      float dyi = float(grad_output[col_base_a + ulong(b) * sa]);
      float yi = float(output[col_base_b + ulong(b) * sb]);
      grad_input[col_base_c + ulong(b) * sc] = static_cast<T>(
          IS_LOG ? (dyi - metal::precise::exp(yi) * dot_sum)
                 : (yi * (dyi - dot_sum)));
    }
  }
}

// Coalesced non-last-dim kernels for inner_size < axis_size.
// Thread t loads input[base + t] (perfectly coalesced).
// inner_pos = tid % stride_a, axis_tid = tid / stride_a.
// Multiple axis_tid threads share one inner_pos; reduced in shared memory.
// num_chunks stores num_axis_threads for the reduction.

template <typename T, bool IS_LOG = false>
kernel void softmax_forward_coalesced(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]) {
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  uint inner_pos = tid % sa;
  uint axis_tid = tid / sa;
  uint num_axis_threads = params.num_chunks;
  uint batch_idx = tg_id;
  ulong base_a = ulong(batch_idx) * axis_size * sa;
  uint total = axis_size * sa;

  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (uint off = tid; off < total; off += lsize) {
    float val = float(input[base_a + ulong(off)]);
    float new_max = fmax(local_max, val);
    local_sum = (new_max > -INFINITY)
        ? local_sum * metal::precise::exp(local_max - new_max) +
            metal::precise::exp(val - new_max)
        : 0.0f;
    local_max = new_max;
  }

  threadgroup float* mx = smem;
  threadgroup float* sm = smem + lsize;
  mx[tid] = local_max;
  sm[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = num_axis_threads / 2; s > 0; s >>= 1) {
    if (axis_tid < s) {
      uint o = tid + s * sa;
      float om = mx[o], os = sm[o], mm = mx[tid], ms = sm[tid];
      float nm = fmax(mm, om);
      sm[tid] = (nm > -INFINITY) ? ms * metal::precise::exp(mm - nm) +
              os * metal::precise::exp(om - nm)
                                 : 0.0f;
      mx[tid] = nm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float fmax_val = mx[inner_pos];
  // log_softmax: subtract log(sum_exp); softmax: multiply by 1/sum_exp.
  float fnorm =
      IS_LOG ? metal::precise::log(sm[inner_pos]) : (1.0f / sm[inner_pos]);

  ulong base_b = ulong(batch_idx) * axis_size * sb;
  for (uint off = tid; off < total; off += lsize) {
    float val = float(input[base_a + ulong(off)]);
    float sh = val - fmax_val;
    output[base_b + ulong(off)] = static_cast<T>(
        IS_LOG ? (sh - fnorm) : (metal::precise::exp(sh) * fnorm));
  }
}

template <typename T, bool IS_LOG = false>
kernel void softmax_backward_coalesced(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device T* grad_input [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]) {
  uint axis_size = params.axis_size;
  uint sa = params.stride_a;
  uint sb = params.stride_b;
  uint sc = params.stride_c;
  uint inner_pos = tid % sa;
  uint axis_tid = tid / sa;
  uint num_axis_threads = params.num_chunks;
  uint batch_idx = tg_id;
  ulong base_a = ulong(batch_idx) * axis_size * sa;
  ulong base_b = ulong(batch_idx) * axis_size * sb;
  uint total = axis_size * sa;

  // softmax: sum(dy*y); log_softmax: sum(dy).
  float local_dot = 0.0f;
  for (uint off = tid; off < total; off += lsize) {
    float dyi = float(grad_output[base_a + ulong(off)]);
    local_dot += IS_LOG ? dyi : (dyi * float(output[base_b + ulong(off)]));
  }

  smem[tid] = local_dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = num_axis_threads / 2; s > 0; s >>= 1) {
    if (axis_tid < s)
      smem[tid] += smem[tid + s * sa];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float dot_sum = smem[inner_pos];

  ulong base_c = ulong(batch_idx) * axis_size * sc;
  for (uint off = tid; off < total; off += lsize) {
    float dyi = float(grad_output[base_a + ulong(off)]);
    float yi = float(output[base_b + ulong(off)]);
    grad_input[base_c + ulong(off)] = static_cast<T>(
        IS_LOG ? (dyi - metal::precise::exp(yi) * dot_sum)
               : (yi * (dyi - dot_sum)));
  }
}

// Template instantiations

// Each kernel family is instantiated twice: ISLOG=false -> "softmax_*",
// ISLOG=true -> "logsoftmax_*". The dispatch picks the name prefix.
#define instantiate_softmax_forward_single_row_op(OP, ISLOG, DTYPE)    \
  template [[host_name(#OP "_forward_single_row_" #DTYPE)]] [[kernel]] \
  void softmax_forward_single_row<DTYPE, ISLOG, 4>(                    \
      device const DTYPE* input [[buffer(0)]],                         \
      device DTYPE* output [[buffer(1)]],                              \
      constant SoftmaxParams& params [[buffer(2)]],                    \
      uint tg_id [[threadgroup_position_in_grid]],                     \
      uint tid [[thread_position_in_threadgroup]],                     \
      uint tptg [[threads_per_threadgroup]],                           \
      uint simd_lane_id [[thread_index_in_simdgroup]],                 \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);
#define instantiate_softmax_forward_single_row(DTYPE)              \
  instantiate_softmax_forward_single_row_op(softmax, false, DTYPE) \
      instantiate_softmax_forward_single_row_op(logsoftmax, true, DTYPE)

// 8-wide single-row variant for half-precision last-dim rows.
#define instantiate_softmax_forward_single_row8_op(OP, ISLOG, DTYPE)    \
  template [[host_name(#OP "_forward_single_row8_" #DTYPE)]] [[kernel]] \
  void softmax_forward_single_row<DTYPE, ISLOG, 8>(                     \
      device const DTYPE* input [[buffer(0)]],                          \
      device DTYPE* output [[buffer(1)]],                               \
      constant SoftmaxParams& params [[buffer(2)]],                     \
      uint tg_id [[threadgroup_position_in_grid]],                      \
      uint tid [[thread_position_in_threadgroup]],                      \
      uint tptg [[threads_per_threadgroup]],                            \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);
#define instantiate_softmax_forward_single_row8(DTYPE)              \
  instantiate_softmax_forward_single_row8_op(softmax, false, DTYPE) \
      instantiate_softmax_forward_single_row8_op(logsoftmax, true, DTYPE)

#define instantiate_softmax_forward_looped_op(OP, ISLOG, DTYPE)         \
  template [[host_name(#OP "_forward_looped_" #DTYPE)]] [[kernel]] void \
  softmax_forward_looped<DTYPE, ISLOG>(                                 \
      device const DTYPE* input [[buffer(0)]],                          \
      device DTYPE* output [[buffer(1)]],                               \
      constant SoftmaxParams& params [[buffer(2)]],                     \
      uint tg_id [[threadgroup_position_in_grid]],                      \
      uint tid [[thread_position_in_threadgroup]],                      \
      uint lsize [[threads_per_threadgroup]],                           \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);
#define instantiate_softmax_forward_looped(DTYPE)              \
  instantiate_softmax_forward_looped_op(softmax, false, DTYPE) \
      instantiate_softmax_forward_looped_op(logsoftmax, true, DTYPE)

#define instantiate_softmax_backward_single_row_op(OP, ISLOG, DTYPE)    \
  template [[host_name(#OP "_backward_single_row_" #DTYPE)]] [[kernel]] \
  void softmax_backward_single_row<DTYPE, ISLOG, 4>(                    \
      device const DTYPE* grad_output [[buffer(0)]],                    \
      device const DTYPE* output [[buffer(1)]],                         \
      device DTYPE* grad_input [[buffer(2)]],                           \
      constant SoftmaxParams& params [[buffer(3)]],                     \
      uint tg_id [[threadgroup_position_in_grid]],                      \
      uint tid [[thread_position_in_threadgroup]],                      \
      uint tptg [[threads_per_threadgroup]],                            \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);
#define instantiate_softmax_backward_single_row(DTYPE)              \
  instantiate_softmax_backward_single_row_op(softmax, false, DTYPE) \
      instantiate_softmax_backward_single_row_op(logsoftmax, true, DTYPE)

// 8-wide single-row backward variant for half-precision last-dim rows.
#define instantiate_softmax_backward_single_row8_op(OP, ISLOG, DTYPE)    \
  template [[host_name(#OP "_backward_single_row8_" #DTYPE)]] [[kernel]] \
  void softmax_backward_single_row<DTYPE, ISLOG, 8>(                     \
      device const DTYPE* grad_output [[buffer(0)]],                     \
      device const DTYPE* output [[buffer(1)]],                          \
      device DTYPE* grad_input [[buffer(2)]],                            \
      constant SoftmaxParams& params [[buffer(3)]],                      \
      uint tg_id [[threadgroup_position_in_grid]],                       \
      uint tid [[thread_position_in_threadgroup]],                       \
      uint tptg [[threads_per_threadgroup]],                             \
      uint simd_lane_id [[thread_index_in_simdgroup]],                   \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);
#define instantiate_softmax_backward_single_row8(DTYPE)              \
  instantiate_softmax_backward_single_row8_op(softmax, false, DTYPE) \
      instantiate_softmax_backward_single_row8_op(logsoftmax, true, DTYPE)

#define instantiate_softmax_backward_looped_op(OP, ISLOG, DTYPE)         \
  template [[host_name(#OP "_backward_looped_" #DTYPE)]] [[kernel]] void \
  softmax_backward_looped<DTYPE, ISLOG>(                                 \
      device const DTYPE* grad_output [[buffer(0)]],                     \
      device const DTYPE* output [[buffer(1)]],                          \
      device DTYPE* grad_input [[buffer(2)]],                            \
      constant SoftmaxParams& params [[buffer(3)]],                      \
      uint tg_id [[threadgroup_position_in_grid]],                       \
      uint tid [[thread_position_in_threadgroup]],                       \
      uint lsize [[threads_per_threadgroup]],                            \
      uint simd_lane_id [[thread_index_in_simdgroup]],                   \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);
#define instantiate_softmax_backward_looped(DTYPE)              \
  instantiate_softmax_backward_looped_op(softmax, false, DTYPE) \
      instantiate_softmax_backward_looped_op(logsoftmax, true, DTYPE)

#define instantiate_softmax_backward_2pass_dot_op(OP, ISLOG, DTYPE)         \
  template [[host_name(#OP "_backward_2pass_dot_" #DTYPE)]] [[kernel]] void \
  softmax_backward_2pass_dot<DTYPE, ISLOG>(                                 \
      device const DTYPE* grad_output [[buffer(0)]],                        \
      device const DTYPE* output [[buffer(1)]],                             \
      device float* partial_sums [[buffer(2)]],                             \
      constant SoftmaxParams& params [[buffer(3)]],                         \
      uint tg_id [[threadgroup_position_in_grid]],                          \
      uint tid [[thread_position_in_threadgroup]],                          \
      uint lsize [[threads_per_threadgroup]],                               \
      uint simd_lane_id [[thread_index_in_simdgroup]],                      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);
#define instantiate_softmax_backward_2pass_dot(DTYPE)              \
  instantiate_softmax_backward_2pass_dot_op(softmax, false, DTYPE) \
      instantiate_softmax_backward_2pass_dot_op(logsoftmax, true, DTYPE)

#define instantiate_softmax_backward_2pass_grad_op(OP, ISLOG, DTYPE)         \
  template [[host_name(#OP "_backward_2pass_grad_" #DTYPE)]] [[kernel]] void \
  softmax_backward_2pass_grad<DTYPE, ISLOG>(                                 \
      device const DTYPE* grad_output [[buffer(0)]],                         \
      device const DTYPE* output [[buffer(1)]],                              \
      device DTYPE* grad_input [[buffer(2)]],                                \
      device const float* partial_sums [[buffer(3)]],                        \
      constant SoftmaxParams& params [[buffer(4)]],                          \
      uint tg_id [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_position_in_threadgroup]],                           \
      uint lsize [[threads_per_threadgroup]]);
#define instantiate_softmax_backward_2pass_grad(DTYPE)              \
  instantiate_softmax_backward_2pass_grad_op(softmax, false, DTYPE) \
      instantiate_softmax_backward_2pass_grad_op(logsoftmax, true, DTYPE)

// The 2pass forward reduce phase (max, sum) is identical for softmax and
// log_softmax, so it is instantiated once under the "softmax_" name and reused.
#define instantiate_softmax_forward_2pass_reduce(DTYPE)                     \
  template                                                                  \
      [[host_name("softmax_forward_2pass_reduce_" #DTYPE)]] [[kernel]] void \
      softmax_forward_2pass_reduce<DTYPE>(                                  \
          device const DTYPE* input [[buffer(0)]],                          \
          device float* partials [[buffer(1)]],                             \
          constant SoftmaxParams& params [[buffer(2)]],                     \
          uint tg_id [[threadgroup_position_in_grid]],                      \
          uint tid [[thread_position_in_threadgroup]],                      \
          uint lsize [[threads_per_threadgroup]],                           \
          uint simd_lane_id [[thread_index_in_simdgroup]],                  \
          uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_forward_2pass_write_op(OP, ISLOG, DTYPE)         \
  template [[host_name(#OP "_forward_2pass_write_" #DTYPE)]] [[kernel]] void \
  softmax_forward_2pass_write<DTYPE, ISLOG>(                                 \
      device const DTYPE* input [[buffer(0)]],                               \
      device DTYPE* output [[buffer(1)]],                                    \
      device const float* partials [[buffer(2)]],                            \
      constant SoftmaxParams& params [[buffer(3)]],                          \
      uint tg_id [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_position_in_threadgroup]],                           \
      uint lsize [[threads_per_threadgroup]]);
#define instantiate_softmax_forward_2pass_write(DTYPE)              \
  instantiate_softmax_forward_2pass_write_op(softmax, false, DTYPE) \
      instantiate_softmax_forward_2pass_write_op(logsoftmax, true, DTYPE)

#define instantiate_softmax_forward_coalesced_op(OP, ISLOG, DTYPE)         \
  template [[host_name(#OP "_forward_coalesced_" #DTYPE)]] [[kernel]] void \
  softmax_forward_coalesced<DTYPE, ISLOG>(                                 \
      device const DTYPE* input [[buffer(0)]],                             \
      device DTYPE* output [[buffer(1)]],                                  \
      constant SoftmaxParams& params [[buffer(2)]],                        \
      uint tg_id [[threadgroup_position_in_grid]],                         \
      uint tid [[thread_position_in_threadgroup]],                         \
      uint lsize [[threads_per_threadgroup]],                              \
      threadgroup float* smem [[threadgroup(0)]]);
#define instantiate_softmax_forward_coalesced(DTYPE)              \
  instantiate_softmax_forward_coalesced_op(softmax, false, DTYPE) \
      instantiate_softmax_forward_coalesced_op(logsoftmax, true, DTYPE)

#define instantiate_softmax_backward_coalesced_op(OP, ISLOG, DTYPE)         \
  template [[host_name(#OP "_backward_coalesced_" #DTYPE)]] [[kernel]] void \
  softmax_backward_coalesced<DTYPE, ISLOG>(                                 \
      device const DTYPE* grad_output [[buffer(0)]],                        \
      device const DTYPE* output [[buffer(1)]],                             \
      device DTYPE* grad_input [[buffer(2)]],                               \
      constant SoftmaxParams& params [[buffer(3)]],                         \
      uint tg_id [[threadgroup_position_in_grid]],                          \
      uint tid [[thread_position_in_threadgroup]],                          \
      uint lsize [[threads_per_threadgroup]],                               \
      threadgroup float* smem [[threadgroup(0)]]);
#define instantiate_softmax_backward_coalesced(DTYPE)              \
  instantiate_softmax_backward_coalesced_op(softmax, false, DTYPE) \
      instantiate_softmax_backward_coalesced_op(logsoftmax, true, DTYPE)

#define instantiate_softmax_forward_tiled_op(OP, ISLOG, DTYPE)         \
  template [[host_name(#OP "_forward_tiled_" #DTYPE)]] [[kernel]] void \
  softmax_forward_tiled<DTYPE, ISLOG>(                                 \
      device const DTYPE* input [[buffer(0)]],                         \
      device DTYPE* output [[buffer(1)]],                              \
      constant SoftmaxParams& params [[buffer(2)]],                    \
      uint tg_id [[threadgroup_position_in_grid]],                     \
      uint tid [[thread_position_in_threadgroup]],                     \
      uint lsize [[threads_per_threadgroup]]);
#define instantiate_softmax_forward_tiled(DTYPE)              \
  instantiate_softmax_forward_tiled_op(softmax, false, DTYPE) \
      instantiate_softmax_forward_tiled_op(logsoftmax, true, DTYPE)

#define instantiate_softmax_forward_blocked_op(OP, ISLOG, DTYPE)         \
  template [[host_name(#OP "_forward_blocked_" #DTYPE)]] [[kernel]] void \
  softmax_forward_blocked<DTYPE, ISLOG>(                                 \
      device const DTYPE* input [[buffer(0)]],                           \
      device DTYPE* output [[buffer(1)]],                                \
      constant SoftmaxParams& params [[buffer(2)]],                      \
      uint tg_id [[threadgroup_position_in_grid]],                       \
      uint tid [[thread_position_in_threadgroup]],                       \
      uint lsize [[threads_per_threadgroup]],                            \
      threadgroup float* smem [[threadgroup(0)]]);
#define instantiate_softmax_forward_blocked(DTYPE)              \
  instantiate_softmax_forward_blocked_op(softmax, false, DTYPE) \
      instantiate_softmax_forward_blocked_op(logsoftmax, true, DTYPE)

#define instantiate_softmax_backward_blocked_op(OP, ISLOG, DTYPE)         \
  template [[host_name(#OP "_backward_blocked_" #DTYPE)]] [[kernel]] void \
  softmax_backward_blocked<DTYPE, ISLOG>(                                 \
      device const DTYPE* grad_output [[buffer(0)]],                      \
      device const DTYPE* output [[buffer(1)]],                           \
      device DTYPE* grad_input [[buffer(2)]],                             \
      constant SoftmaxParams& params [[buffer(3)]],                       \
      uint tg_id [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_position_in_threadgroup]],                        \
      uint lsize [[threads_per_threadgroup]],                             \
      threadgroup float* smem [[threadgroup(0)]]);
#define instantiate_softmax_backward_blocked(DTYPE)              \
  instantiate_softmax_backward_blocked_op(softmax, false, DTYPE) \
      instantiate_softmax_backward_blocked_op(logsoftmax, true, DTYPE)

// The blocked2 forward reduce phase (max, sum) is identical for softmax and
// log_softmax, so it is instantiated once under the "softmax_" name and reused.
#define instantiate_softmax_forward_blocked2_reduce(DTYPE)                     \
  template                                                                     \
      [[host_name("softmax_forward_blocked2_reduce_" #DTYPE)]] [[kernel]] void \
      softmax_forward_blocked2_reduce<DTYPE>(                                  \
          device const DTYPE* input [[buffer(0)]],                             \
          device float* partials [[buffer(1)]],                                \
          constant SoftmaxParams& params [[buffer(2)]],                        \
          uint tg_id [[threadgroup_position_in_grid]],                         \
          uint tid [[thread_position_in_threadgroup]],                         \
          uint lsize [[threads_per_threadgroup]],                              \
          threadgroup float* smem [[threadgroup(0)]]);

#define instantiate_softmax_forward_blocked2_write_op(OP, ISLOG, DTYPE)    \
  template                                                                 \
      [[host_name(#OP "_forward_blocked2_write_" #DTYPE)]] [[kernel]] void \
      softmax_forward_blocked2_write<DTYPE, ISLOG>(                        \
          device const DTYPE* input [[buffer(0)]],                         \
          device DTYPE* output [[buffer(1)]],                              \
          device const float* partials [[buffer(2)]],                      \
          constant SoftmaxParams& params [[buffer(3)]],                    \
          uint tg_id [[threadgroup_position_in_grid]],                     \
          uint tid [[thread_position_in_threadgroup]],                     \
          uint lsize [[threads_per_threadgroup]]);
#define instantiate_softmax_forward_blocked2_write(DTYPE)              \
  instantiate_softmax_forward_blocked2_write_op(softmax, false, DTYPE) \
      instantiate_softmax_forward_blocked2_write_op(logsoftmax, true, DTYPE)

#define instantiate_softmax_backward_blocked2_dot_op(OP, ISLOG, DTYPE)         \
  template [[host_name(#OP "_backward_blocked2_dot_" #DTYPE)]] [[kernel]] void \
  softmax_backward_blocked2_dot<DTYPE, ISLOG>(                                 \
      device const DTYPE* grad_output [[buffer(0)]],                           \
      device const DTYPE* output [[buffer(1)]],                                \
      device float* partials [[buffer(2)]],                                    \
      constant SoftmaxParams& params [[buffer(3)]],                            \
      uint tg_id [[threadgroup_position_in_grid]],                             \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint lsize [[threads_per_threadgroup]],                                  \
      threadgroup float* smem [[threadgroup(0)]]);
#define instantiate_softmax_backward_blocked2_dot(DTYPE)              \
  instantiate_softmax_backward_blocked2_dot_op(softmax, false, DTYPE) \
      instantiate_softmax_backward_blocked2_dot_op(logsoftmax, true, DTYPE)

#define instantiate_softmax_backward_blocked2_grad_op(OP, ISLOG, DTYPE)    \
  template                                                                 \
      [[host_name(#OP "_backward_blocked2_grad_" #DTYPE)]] [[kernel]] void \
      softmax_backward_blocked2_grad<DTYPE, ISLOG>(                        \
          device const DTYPE* grad_output [[buffer(0)]],                   \
          device const DTYPE* output [[buffer(1)]],                        \
          device DTYPE* grad_input [[buffer(2)]],                          \
          device const float* partials [[buffer(3)]],                      \
          constant SoftmaxParams& params [[buffer(4)]],                    \
          uint tg_id [[threadgroup_position_in_grid]],                     \
          uint tid [[thread_position_in_threadgroup]],                     \
          uint lsize [[threads_per_threadgroup]]);
#define instantiate_softmax_backward_blocked2_grad(DTYPE)              \
  instantiate_softmax_backward_blocked2_grad_op(softmax, false, DTYPE) \
      instantiate_softmax_backward_blocked2_grad_op(logsoftmax, true, DTYPE)

#define instantiate_softmax_backward_tiled_op(OP, ISLOG, DTYPE)         \
  template [[host_name(#OP "_backward_tiled_" #DTYPE)]] [[kernel]] void \
  softmax_backward_tiled<DTYPE, ISLOG>(                                 \
      device const DTYPE* grad_output [[buffer(0)]],                    \
      device const DTYPE* output [[buffer(1)]],                         \
      device DTYPE* grad_input [[buffer(2)]],                           \
      constant SoftmaxParams& params [[buffer(3)]],                     \
      uint tg_id [[threadgroup_position_in_grid]],                      \
      uint tid [[thread_position_in_threadgroup]],                      \
      uint lsize [[threads_per_threadgroup]]);
#define instantiate_softmax_backward_tiled(DTYPE)              \
  instantiate_softmax_backward_tiled_op(softmax, false, DTYPE) \
      instantiate_softmax_backward_tiled_op(logsoftmax, true, DTYPE)

#define instantiate_softmax(DTYPE)                                                  \
  instantiate_softmax_forward_single_row(                                           \
      DTYPE) instantiate_softmax_forward_looped(DTYPE)                              \
      instantiate_softmax_forward_tiled(DTYPE) instantiate_softmax_forward_blocked( \
          DTYPE) instantiate_softmax_forward_blocked2_reduce(DTYPE)                 \
          instantiate_softmax_forward_blocked2_write(                               \
              DTYPE) instantiate_softmax_forward_coalesced(DTYPE)                   \
              instantiate_softmax_forward_2pass_reduce(                             \
                  DTYPE) instantiate_softmax_forward_2pass_write(DTYPE)             \
                  instantiate_softmax_backward_single_row(                          \
                      DTYPE) instantiate_softmax_backward_looped(DTYPE)             \
                      instantiate_softmax_backward_tiled(                           \
                          DTYPE) instantiate_softmax_backward_blocked(DTYPE)        \
                          instantiate_softmax_backward_blocked2_dot(DTYPE)          \
                              instantiate_softmax_backward_blocked2_grad(           \
                                  DTYPE)                                            \
                                  instantiate_softmax_backward_coalesced(           \
                                      DTYPE)                                        \
                                      instantiate_softmax_backward_2pass_dot(       \
                                          DTYPE)                                    \
                                          instantiate_softmax_backward_2pass_grad(  \
                                              DTYPE)

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
