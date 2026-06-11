#include <ATen/native/mps/kernels/SoftMaxKernel.h>
#include <c10/metal/reduction_utils.h>
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
using c10::metal::simdgroup_size;

static inline uint offset_a(uint row_idx, constant SoftmaxParams& p) {
  uint offset = 0;
  uint idx = row_idx;
  for (int d = int(p.ndim) - 2; d >= 0; d--) {
    uint coord = idx % p.outer_sizes[d];
    idx /= p.outer_sizes[d];
    offset += coord * p.outer_strides_a[d];
  }
  return offset;
}

static inline uint offset_b(uint row_idx, constant SoftmaxParams& p) {
  uint offset = 0;
  uint idx = row_idx;
  for (int d = int(p.ndim) - 2; d >= 0; d--) {
    uint coord = idx % p.outer_sizes[d];
    idx /= p.outer_sizes[d];
    offset += coord * p.outer_strides_b[d];
  }
  return offset;
}

static inline uint offset_c(uint row_idx, constant SoftmaxParams& p) {
  uint offset = 0;
  uint idx = row_idx;
  for (int d = int(p.ndim) - 2; d >= 0; d--) {
    uint coord = idx % p.outer_sizes[d];
    idx /= p.outer_sizes[d];
    offset += coord * p.outer_strides_c[d];
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

template <typename T>
kernel void softmax_forward_single_row(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
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
      float4 v = load_vec4(x + base);
      vals[0] = v.x;
      vals[1] = v.y;
      vals[2] = v.z;
      vals[3] = v.w;
    } else {
      for (int i = 0; i < N_READS; i++)
        vals[i] = float(x[(base + i) * sa]);
    }
    local_max = fmax(fmax(vals[0], vals[1]), fmax(vals[2], vals[3]));
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

  float4 result = float4(vals[0], vals[1], vals[2], vals[3]) * inv_sum;
  if (base + N_READS <= axis_size) {
    if (sb == 1) {
      store_vec4(out + base, result);
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++)
        out[(base + i) * sb] = static_cast<T>(result[i]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size)
        out[(base + i) * sb] = static_cast<T>(result[i]);
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
      local_sum = local_sum * metal::precise::exp(local_max - new_max) +
          metal::precise::exp(v.x - new_max) +
          metal::precise::exp(v.y - new_max) +
          metal::precise::exp(v.z - new_max) +
          metal::precise::exp(v.w - new_max);
      local_max = new_max;
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++) {
        float val = contiguous ? float(x[i]) : float(x[i * sa]);
        float new_max = fmax(local_max, val);
        local_sum = local_sum * metal::precise::exp(local_max - new_max) +
            metal::precise::exp(val - new_max);
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
  if (simdgroup_id == 0) {
    float m = shared_max[simd_lane_id];
    float global_max = simd_max(m);
    float s = shared_sum[simd_lane_id] * metal::precise::exp(m - global_max);
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
        out[i * sb] = static_cast<T>(metal::precise::exp(val - row_max) * inv_sum);
      }
    }
  }
}


// Two-pass forward for low-occupancy cases (few rows, large axis).
// Phase 1: each threadgroup computes (chunk_max, chunk_sum) via online algorithm.
// Phase 2: each threadgroup combines partials, re-reads input, writes output.

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
        v = float4(x[base * sa], x[(base + 1) * sa],
                    x[(base + 2) * sa], x[(base + 3) * sa]);
      }
      float chunk_max = fmax(fmax(v.x, v.y), fmax(v.z, v.w));
      float new_max = fmax(local_max, chunk_max);
      local_sum = local_sum * metal::precise::exp(local_max - new_max) +
          metal::precise::exp(v.x - new_max) +
          metal::precise::exp(v.y - new_max) +
          metal::precise::exp(v.z - new_max) +
          metal::precise::exp(v.w - new_max);
      local_max = new_max;
    } else {
      for (uint i = base; i < min(base + uint(N_READS), end); i++) {
        float val = contiguous ? float(x[i]) : float(x[i * sa]);
        float new_max = fmax(local_max, val);
        local_sum = local_sum * metal::precise::exp(local_max - new_max) +
            metal::precise::exp(val - new_max);
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

  if (simdgroup_id == 0) {
    float m = shared_max[simd_lane_id];
    float global_max = simd_max(m);
    float s = shared_sum[simd_lane_id] * metal::precise::exp(m - global_max);
    float global_sum = simd_sum(s);
    if (simd_lane_id == 0) {
      partials[(row_id * num_chunks + chunk_id) * 2] = global_max;
      partials[(row_id * num_chunks + chunk_id) * 2 + 1] = global_sum;
    }
  }
}

template <typename T>
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
    global_sum = global_sum * metal::precise::exp(global_max - new_max) +
        chunk_sum * metal::precise::exp(chunk_max - new_max);
    global_max = new_max;
  }
  float inv_sum = 1.0f / global_sum;

  uint elems_per_chunk = (axis_size + num_chunks - 1) / num_chunks;
  uint start = chunk_id * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  for (uint r = start; r < end; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= end) {
      float4 v;
      if (contiguous) {
        v = metal::precise::exp(load_vec4(x + base) - global_max) * inv_sum;
      } else {
        v = float4(
            metal::precise::exp(float(x[base * sa]) - global_max) * inv_sum,
            metal::precise::exp(float(x[(base + 1) * sa]) - global_max) * inv_sum,
            metal::precise::exp(float(x[(base + 2) * sa]) - global_max) * inv_sum,
            metal::precise::exp(float(x[(base + 3) * sa]) - global_max) * inv_sum);
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
        out[i * sb] = static_cast<T>(metal::precise::exp(val - global_max) * inv_sum);
      }
    }
  }
}

// Backward: grad_input = output * (grad_output - sum(grad_output * output))
// stride_a = grad_output strides, stride_b = output strides
// Writes grad_input contiguously.

template <typename T>
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
  constexpr int N_READS = 4;
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
      float4 dy_v = load_vec4(dy + base);
      float4 y_v = load_vec4(y + base);
      dy_vals[0] = dy_v.x;
      dy_vals[1] = dy_v.y;
      dy_vals[2] = dy_v.z;
      dy_vals[3] = dy_v.w;
      y_vals[0] = y_v.x;
      y_vals[1] = y_v.y;
      y_vals[2] = y_v.z;
      y_vals[3] = y_v.w;
      local_dot = dot(dy_v, y_v);
    } else {
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

  float4 result = float4(y_vals[0], y_vals[1], y_vals[2], y_vals[3]) *
      (float4(dy_vals[0], dy_vals[1], dy_vals[2], dy_vals[3]) - dot_sum);
  if (base + N_READS <= axis_size) {
    if (sc == 1) {
      store_vec4(dx + base, result);
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++)
        dx[(base + i) * sc] = static_cast<T>(result[i]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size)
        dx[(base + i) * sc] = static_cast<T>(y_vals[i] * (dy_vals[i] - dot_sum));
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
  float dot_sum = c10::metal::threadgroup_sum(shared_dot, local_dot, tid, lsize);

  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 y_v, dy_v;
      if (contiguous) {
        y_v = load_vec4(y + base);
        dy_v = load_vec4(dy + base);
      } else {
        y_v = float4(y[base * sb], y[(base+1) * sb], y[(base+2) * sb], y[(base+3) * sb]);
        dy_v = float4(dy[base * sa], dy[(base+1) * sa], dy[(base+2) * sa], dy[(base+3) * sa]);
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

template <typename T>
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

  float local_dot = 0.0f;
  for (uint r = start; r < end; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= end) {
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
      for (uint i = base; i < min(base + uint(N_READS), end); i++)
        local_dot += (contiguous ? float(dy[i]) : float(dy[i * sa])) *
            (contiguous ? float(y[i]) : float(y[i * sb]));
    }
  }

  threadgroup float shared_dot[simdgroup_size];
  float d = c10::metal::threadgroup_sum(shared_dot, local_dot, tid, lsize);
  if (tid == 0)
    partial_sums[row_id * num_chunks + chunk_id] = d;
}

template <typename T>
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
        y_v = float4(y[base * sb], y[(base+1) * sb], y[(base+2) * sb], y[(base+3) * sb]);
        dy_v = float4(dy[base * sa], dy[(base+1) * sa], dy[(base+2) * sa], dy[(base+3) * sa]);
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
      for (uint i = base; i < min(base + uint(N_READS), end); i++) {
        float yi = contiguous ? float(y[i]) : float(y[i * sb]);
        float dyi = contiguous ? float(dy[i]) : float(dy[i * sa]);
        dx[i * sc] = static_cast<T>(yi * (dyi - dot_sum));
      }
    }
  }
}


// ============================================================================
// Log-softmax kernels
// output[i] = (x[i] - max) - log(sum(exp(x - max)))
// ============================================================================

template <typename T>
kernel void log_softmax_forward_single_row(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
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
      float4 v = load_vec4(x + base);
      vals[0] = v.x; vals[1] = v.y; vals[2] = v.z; vals[3] = v.w;
    } else {
      for (int i = 0; i < N_READS; i++)
        vals[i] = float(x[(base + i) * sa]);
    }
    local_max = fmax(fmax(vals[0], vals[1]), fmax(vals[2], vals[3]));
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
  for (int i = 0; i < N_READS; i++)
    local_sum += metal::precise::exp(vals[i] - row_max);

  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_sum = c10::metal::threadgroup_sum(shared, local_sum, tid, tptg);
  float shift = row_max + metal::precise::log(total_sum);

  float4 result = float4(vals[0], vals[1], vals[2], vals[3]) - shift;
  if (base + N_READS <= axis_size) {
    if (sb == 1) {
      store_vec4(out + base, result);
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++)
        out[(base + i) * sb] = static_cast<T>(result[i]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size)
        out[(base + i) * sb] = static_cast<T>(result[i]);
    }
  }
}

template <typename T>
kernel void log_softmax_forward_looped(
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
        v = float4(x[base * sa], x[(base+1) * sa],
                    x[(base+2) * sa], x[(base+3) * sa]);
      }
      float chunk_max = fmax(fmax(v.x, v.y), fmax(v.z, v.w));
      float new_max = fmax(local_max, chunk_max);
      local_sum = local_sum * metal::precise::exp(local_max - new_max) +
          metal::precise::exp(v.x - new_max) +
          metal::precise::exp(v.y - new_max) +
          metal::precise::exp(v.z - new_max) +
          metal::precise::exp(v.w - new_max);
      local_max = new_max;
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++) {
        float val = contiguous ? float(x[i]) : float(x[i * sa]);
        float new_max = fmax(local_max, val);
        local_sum = local_sum * metal::precise::exp(local_max - new_max) +
            metal::precise::exp(val - new_max);
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
  if (simdgroup_id == 0) {
    float m = shared_max[simd_lane_id];
    float global_max = simd_max(m);
    float s = shared_sum[simd_lane_id] * metal::precise::exp(m - global_max);
    float global_sum = simd_sum(s);
    if (simd_lane_id == 0) {
      tg_result[0] = global_max;
      tg_result[1] = global_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float shift = tg_result[0] + metal::precise::log(tg_result[1]);

  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 v;
      if (contiguous) {
        v = load_vec4(x + base) - shift;
      } else {
        v = float4(
            float(x[base * sa]) - shift,
            float(x[(base + 1) * sa]) - shift,
            float(x[(base + 2) * sa]) - shift,
            float(x[(base + 3) * sa]) - shift);
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
        out[i * sb] = static_cast<T>(val - shift);
      }
    }
  }
}

// Log-softmax forward 2-pass: for low-occupancy (outer_size < 4, large axis)
// Phase 1: each threadgroup computes (chunk_max, chunk_sum) via online algorithm
template <typename T>
kernel void log_softmax_forward_2pass_reduce(
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
        v = float4(x[base * sa], x[(base+1) * sa],
                    x[(base+2) * sa], x[(base+3) * sa]);
      }
      float chunk_max = fmax(fmax(v.x, v.y), fmax(v.z, v.w));
      float new_max = fmax(local_max, chunk_max);
      local_sum = local_sum * metal::precise::exp(local_max - new_max) +
          metal::precise::exp(v.x - new_max) +
          metal::precise::exp(v.y - new_max) +
          metal::precise::exp(v.z - new_max) +
          metal::precise::exp(v.w - new_max);
      local_max = new_max;
    } else {
      for (uint i = base; i < min(base + uint(N_READS), end); i++) {
        float val = contiguous ? float(x[i]) : float(x[i * sa]);
        float new_max = fmax(local_max, val);
        local_sum = local_sum * metal::precise::exp(local_max - new_max) +
            metal::precise::exp(val - new_max);
        local_max = new_max;
      }
    }
  }

  // Reduce across simdgroup
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

  if (simdgroup_id == 0) {
    float m = shared_max[simd_lane_id];
    float global_max = simd_max(m);
    float s = shared_sum[simd_lane_id] * metal::precise::exp(m - global_max);
    float global_sum = simd_sum(s);
    if (simd_lane_id == 0) {
      // Store (max, sum) pair as float2
      partials[(row_id * num_chunks + chunk_id) * 2] = global_max;
      partials[(row_id * num_chunks + chunk_id) * 2 + 1] = global_sum;
    }
  }
}

// Phase 2: combine partials, compute shift = max + log(sum), write output = x - shift
template <typename T>
kernel void log_softmax_forward_2pass_write(
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

  // Combine all partial (max, sum) pairs for this row
  float global_max = -INFINITY;
  float global_sum = 0.0f;
  for (uint i = 0; i < num_chunks; i++) {
    float chunk_max = partials[(row_id * num_chunks + i) * 2];
    float chunk_sum = partials[(row_id * num_chunks + i) * 2 + 1];
    float new_max = fmax(global_max, chunk_max);
    global_sum = global_sum * metal::precise::exp(global_max - new_max) +
        chunk_sum * metal::precise::exp(chunk_max - new_max);
    global_max = new_max;
  }
  float shift = global_max + metal::precise::log(global_sum);

  uint elems_per_chunk = (axis_size + num_chunks - 1) / num_chunks;
  uint start = chunk_id * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  for (uint r = start; r < end; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= end) {
      float4 v;
      if (contiguous) {
        v = load_vec4(x + base) - shift;
      } else {
        v = float4(
            float(x[base * sa]) - shift,
            float(x[(base + 1) * sa]) - shift,
            float(x[(base + 2) * sa]) - shift,
            float(x[(base + 3) * sa]) - shift);
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
        out[i * sb] = static_cast<T>(val - shift);
      }
    }
  }
}

// Log-softmax backward: grad_input = grad_output - exp(output) * sum(grad_output)

template <typename T>
kernel void log_softmax_backward_single_row(
    device const T* grad_output [[buffer(0)]],
    device const T* output [[buffer(1)]],
    device T* grad_input [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
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
  uint base = tid * N_READS;

  bool contiguous = (sa == 1) && (sb == 1);
  float dy_vals[N_READS];
  float y_vals[N_READS];
  float local_sum = 0.0f;
  if (base + N_READS <= axis_size) {
    if (contiguous) {
      float4 dy_v = load_vec4(dy + base);
      float4 y_v = load_vec4(y + base);
      dy_vals[0] = dy_v.x; dy_vals[1] = dy_v.y;
      dy_vals[2] = dy_v.z; dy_vals[3] = dy_v.w;
      y_vals[0] = y_v.x; y_vals[1] = y_v.y;
      y_vals[2] = y_v.z; y_vals[3] = y_v.w;
      local_sum = dy_v.x + dy_v.y + dy_v.z + dy_v.w;
    } else {
      for (int i = 0; i < N_READS; i++) {
        dy_vals[i] = float(dy[(base + i) * sa]);
        y_vals[i] = float(y[(base + i) * sb]);
        local_sum += dy_vals[i];
      }
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size) {
        dy_vals[i] = contiguous ? float(dy[base + i]) : float(dy[(base + i) * sa]);
        y_vals[i] = contiguous ? float(y[base + i]) : float(y[(base + i) * sb]);
        local_sum += dy_vals[i];
      } else {
        dy_vals[i] = 0;
        y_vals[i] = 0;
      }
    }
  }

  threadgroup float shared[simdgroup_size];
  float grad_sum = c10::metal::threadgroup_sum(shared, local_sum, tid, tptg);

  float4 dy_v = float4(dy_vals[0], dy_vals[1], dy_vals[2], dy_vals[3]);
  float4 exp_y = float4(
      metal::precise::exp(y_vals[0]), metal::precise::exp(y_vals[1]),
      metal::precise::exp(y_vals[2]), metal::precise::exp(y_vals[3]));
  float4 result = dy_v - exp_y * grad_sum;
  if (base + N_READS <= axis_size) {
    if (sc == 1) {
      store_vec4(dx + base, result);
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++)
        dx[(base + i) * sc] = static_cast<T>(result[i]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size)
        dx[(base + i) * sc] = static_cast<T>(
            dy_vals[i] - metal::precise::exp(y_vals[i]) * grad_sum);
    }
  }
}

template <typename T>
kernel void log_softmax_backward_looped(
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

  float local_sum = 0.0f;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      if (contiguous) {
        float4 dy_v = load_vec4(dy + base);
        local_sum += dy_v.x + dy_v.y + dy_v.z + dy_v.w;
      } else {
        for (int i = 0; i < N_READS; i++)
          local_sum += float(dy[(base + i) * sa]);
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++)
        local_sum += contiguous ? float(dy[i]) : float(dy[i * sa]);
    }
  }

  threadgroup float shared[simdgroup_size];
  float grad_sum = c10::metal::threadgroup_sum(shared, local_sum, tid, lsize);

  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 dy_v, y_v;
      if (contiguous) {
        dy_v = load_vec4(dy + base);
        y_v = load_vec4(y + base);
      } else {
        dy_v = float4(dy[base * sa], dy[(base+1) * sa], dy[(base+2) * sa], dy[(base+3) * sa]);
        y_v = float4(y[base * sb], y[(base+1) * sb], y[(base+2) * sb], y[(base+3) * sb]);
      }
      float4 exp_y = float4(
          metal::precise::exp(y_v.x), metal::precise::exp(y_v.y),
          metal::precise::exp(y_v.z), metal::precise::exp(y_v.w));
      float4 result = dy_v - exp_y * grad_sum;
      if (sc == 1) {
        store_vec4(dx + base, result);
      } else {
#pragma unroll
        for (int i = 0; i < N_READS; i++)
          dx[(base + i) * sc] = static_cast<T>(result[i]);
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), axis_size); i++) {
        float dyi = contiguous ? float(dy[i]) : float(dy[i * sa]);
        float yi = contiguous ? float(y[i]) : float(y[i * sb]);
        dx[i * sc] = static_cast<T>(dyi - metal::precise::exp(yi) * grad_sum);
      }
    }
  }
}

template <typename T>
kernel void log_softmax_backward_2pass_sum(
    device const T* grad_output [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
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
  device const T* dy = grad_output + offset_a(row_id, params);
  bool contiguous = (sa == 1);

  uint elems_per_chunk = (axis_size + num_chunks - 1) / num_chunks;
  uint start = chunk_id * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  float local_sum = 0.0f;
  for (uint r = start; r < end; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= end) {
      if (contiguous) {
        float4 v = load_vec4(dy + base);
        local_sum += v.x + v.y + v.z + v.w;
      } else {
        for (int i = 0; i < N_READS; i++)
          local_sum += float(dy[(base + i) * sa]);
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), end); i++)
        local_sum += contiguous ? float(dy[i]) : float(dy[i * sa]);
    }
  }

  threadgroup float shared[simdgroup_size];
  float s = c10::metal::threadgroup_sum(shared, local_sum, tid, lsize);
  if (tid == 0)
    partial_sums[row_id * num_chunks + chunk_id] = s;
}

template <typename T>
kernel void log_softmax_backward_2pass_grad(
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

  float grad_sum = 0.0f;
  for (uint i = 0; i < num_chunks; i++)
    grad_sum += partial_sums[row_id * num_chunks + i];

  uint elems_per_chunk = (axis_size + num_chunks - 1) / num_chunks;
  uint start = chunk_id * elems_per_chunk;
  uint end = min(start + elems_per_chunk, axis_size);

  for (uint r = start; r < end; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= end) {
      float4 dy_v, y_v;
      if (contiguous) {
        dy_v = load_vec4(dy + base);
        y_v = load_vec4(y + base);
      } else {
        dy_v = float4(dy[base * sa], dy[(base+1) * sa], dy[(base+2) * sa], dy[(base+3) * sa]);
        y_v = float4(y[base * sb], y[(base+1) * sb], y[(base+2) * sb], y[(base+3) * sb]);
      }
      float4 exp_y = float4(
          metal::precise::exp(y_v.x), metal::precise::exp(y_v.y),
          metal::precise::exp(y_v.z), metal::precise::exp(y_v.w));
      float4 result = dy_v - exp_y * grad_sum;
      if (sc == 1) {
        store_vec4(dx + base, result);
      } else {
#pragma unroll
        for (int i = 0; i < N_READS; i++)
          dx[(base + i) * sc] = static_cast<T>(result[i]);
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), end); i++) {
        float dyi = contiguous ? float(dy[i]) : float(dy[i * sa]);
        float yi = contiguous ? float(y[i]) : float(y[i * sb]);
        dx[i * sc] = static_cast<T>(dyi - metal::precise::exp(yi) * grad_sum);
      }
    }
  }
}


// Tiled forward/backward kernels for non-last-dim softmax.
// Each thread computes softmax for all axis elements at one inner position.
// Adjacent threads access adjacent memory — coalesced reads and writes.
// Uses num_chunks to store the number of inner tiles.

template <typename T>
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

  if (inner_pos >= sa) return;

  uint base_a = batch_idx * axis_size * sa + inner_pos;
  uint base_b = batch_idx * axis_size * sb + inner_pos;

  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (uint b = 0; b < axis_size; b++) {
    float val = float(input[base_a + b * sa]);
    float new_max = fmax(local_max, val);
    local_sum = local_sum * metal::precise::exp(local_max - new_max) +
        metal::precise::exp(val - new_max);
    local_max = new_max;
  }
  float inv_sum = 1.0f / local_sum;

  for (uint b = 0; b < axis_size; b++) {
    float val = float(input[base_a + b * sa]);
    output[base_b + b * sb] = static_cast<T>(
        metal::precise::exp(val - local_max) * inv_sum);
  }
}

template <typename T>
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

  if (inner_pos >= sa) return;

  uint base_a = batch_idx * axis_size * sa + inner_pos;
  uint base_b = batch_idx * axis_size * sb + inner_pos;
  uint base_c = batch_idx * axis_size * sc + inner_pos;

  float dot_sum = 0.0f;
  if (axis_size <= kTileAxisCap) {
    float dy_cache[kTileAxisCap];
    float y_cache[kTileAxisCap];
    for (uint b = 0; b < axis_size; b++) {
      dy_cache[b] = float(grad_output[base_a + b * sa]);
      y_cache[b] = float(output[base_b + b * sb]);
      dot_sum += dy_cache[b] * y_cache[b];
    }
    for (uint b = 0; b < axis_size; b++)
      grad_input[base_c + b * sc] = static_cast<T>(y_cache[b] * (dy_cache[b] - dot_sum));
  } else {
    for (uint b = 0; b < axis_size; b++)
      dot_sum += float(grad_output[base_a + b * sa]) * float(output[base_b + b * sb]);
    for (uint b = 0; b < axis_size; b++) {
      float dyi = float(grad_output[base_a + b * sa]);
      float yi = float(output[base_b + b * sb]);
      grad_input[base_c + b * sc] = static_cast<T>(yi * (dyi - dot_sum));
    }
  }
}

template <typename T>
kernel void log_softmax_forward_tiled(
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

  if (inner_pos >= sa) return;

  uint base_a = batch_idx * axis_size * sa + inner_pos;
  uint base_b = batch_idx * axis_size * sb + inner_pos;

  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (uint b = 0; b < axis_size; b++) {
    float val = float(input[base_a + b * sa]);
    float new_max = fmax(local_max, val);
    local_sum = local_sum * metal::precise::exp(local_max - new_max) +
        metal::precise::exp(val - new_max);
    local_max = new_max;
  }
  float shift = local_max + metal::precise::log(local_sum);

  for (uint b = 0; b < axis_size; b++) {
    float val = float(input[base_a + b * sa]);
    output[base_b + b * sb] = static_cast<T>(val - shift);
  }
}

template <typename T>
kernel void log_softmax_backward_tiled(
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

  if (inner_pos >= sa) return;

  uint base_a = batch_idx * axis_size * sa + inner_pos;
  uint base_b = batch_idx * axis_size * sb + inner_pos;
  uint base_c = batch_idx * axis_size * sc + inner_pos;

  float grad_sum = 0.0f;
  if (axis_size <= kTileAxisCap) {
    float dy_cache[kTileAxisCap];
    for (uint b = 0; b < axis_size; b++) {
      dy_cache[b] = float(grad_output[base_a + b * sa]);
      grad_sum += dy_cache[b];
    }
    for (uint b = 0; b < axis_size; b++) {
      float yi = float(output[base_b + b * sb]);
      grad_input[base_c + b * sc] = static_cast<T>(dy_cache[b] - metal::precise::exp(yi) * grad_sum);
    }
  } else {
    for (uint b = 0; b < axis_size; b++)
      grad_sum += float(grad_output[base_a + b * sa]);
    for (uint b = 0; b < axis_size; b++) {
      float dyi = float(grad_output[base_a + b * sa]);
      float yi = float(output[base_b + b * sb]);
      grad_input[base_c + b * sc] = static_cast<T>(dyi - metal::precise::exp(yi) * grad_sum);
    }
  }
}


// Coalesced non-last-dim kernels for inner_size < axis_size.
// Thread t loads input[base + t] (perfectly coalesced).
// inner_pos = tid % stride_a, axis_tid = tid / stride_a.
// Multiple axis_tid threads share one inner_pos; reduced in shared memory.
// num_chunks stores num_axis_threads for the reduction.

template <typename T>
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
  uint base_a = batch_idx * axis_size * sa;
  uint total = axis_size * sa;

  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (uint off = tid; off < total; off += lsize) {
    float val = float(input[base_a + off]);
    float new_max = fmax(local_max, val);
    local_sum = local_sum * metal::precise::exp(local_max - new_max) +
        metal::precise::exp(val - new_max);
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
      sm[tid] = (nm > -INFINITY)
          ? ms * metal::precise::exp(mm - nm) + os * metal::precise::exp(om - nm)
          : 0.0f;
      mx[tid] = nm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float fmax_val = mx[inner_pos];
  float finv = 1.0f / sm[inner_pos];

  uint base_b = batch_idx * axis_size * sb;
  for (uint off = tid; off < total; off += lsize) {
    float val = float(input[base_a + off]);
    output[base_b + off] = static_cast<T>(
        metal::precise::exp(val - fmax_val) * finv);
  }
}

template <typename T>
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
  uint base_a = batch_idx * axis_size * sa;
  uint base_b = batch_idx * axis_size * sb;
  uint total = axis_size * sa;

  float local_dot = 0.0f;
  for (uint off = tid; off < total; off += lsize) {
    local_dot += float(grad_output[base_a + off]) * float(output[base_b + off]);
  }

  smem[tid] = local_dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = num_axis_threads / 2; s > 0; s >>= 1) {
    if (axis_tid < s) smem[tid] += smem[tid + s * sa];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float dot_sum = smem[inner_pos];

  uint base_c = batch_idx * axis_size * sc;
  for (uint off = tid; off < total; off += lsize) {
    float dyi = float(grad_output[base_a + off]);
    float yi = float(output[base_b + off]);
    grad_input[base_c + off] = static_cast<T>(yi * (dyi - dot_sum));
  }
}

template <typename T>
kernel void log_softmax_forward_coalesced(
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
  uint base_a = batch_idx * axis_size * sa;
  uint total = axis_size * sa;

  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (uint off = tid; off < total; off += lsize) {
    float val = float(input[base_a + off]);
    float new_max = fmax(local_max, val);
    local_sum = local_sum * metal::precise::exp(local_max - new_max) +
        metal::precise::exp(val - new_max);
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
      sm[tid] = (nm > -INFINITY)
          ? ms * metal::precise::exp(mm - nm) + os * metal::precise::exp(om - nm)
          : 0.0f;
      mx[tid] = nm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float shift = mx[inner_pos] + metal::precise::log(sm[inner_pos]);

  uint base_b = batch_idx * axis_size * sb;
  for (uint off = tid; off < total; off += lsize) {
    float val = float(input[base_a + off]);
    output[base_b + off] = static_cast<T>(val - shift);
  }
}

template <typename T>
kernel void log_softmax_backward_coalesced(
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
  uint base_a = batch_idx * axis_size * sa;
  uint total = axis_size * sa;

  float local_sum = 0.0f;
  for (uint off = tid; off < total; off += lsize) {
    local_sum += float(grad_output[base_a + off]);
  }

  smem[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = num_axis_threads / 2; s > 0; s >>= 1) {
    if (axis_tid < s) smem[tid] += smem[tid + s * sa];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float grad_sum = smem[inner_pos];

  uint base_b = batch_idx * axis_size * sb;
  uint base_c = batch_idx * axis_size * sc;
  for (uint off = tid; off < total; off += lsize) {
    float dyi = float(grad_output[base_a + off]);
    float yi = float(output[base_b + off]);
    grad_input[base_c + off] = static_cast<T>(dyi - metal::precise::exp(yi) * grad_sum);
  }
}

// Template instantiations

#define instantiate_softmax_forward_single_row(DTYPE)                     \
  template [[host_name("softmax_forward_single_row_" #DTYPE)]] [[kernel]] \
  void softmax_forward_single_row<DTYPE>(                                 \
      device const DTYPE* input [[buffer(0)]],                            \
      device DTYPE* output [[buffer(1)]],                                 \
      constant SoftmaxParams& params [[buffer(2)]],                       \
      uint tg_id [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_position_in_threadgroup]],                        \
      uint tptg [[threads_per_threadgroup]],                              \
      uint simd_lane_id [[thread_index_in_simdgroup]],                    \
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
  void softmax_backward_single_row<DTYPE>(                                 \
      device const DTYPE* grad_output [[buffer(0)]],                       \
      device const DTYPE* output [[buffer(1)]],                            \
      device DTYPE* grad_input [[buffer(2)]],                              \
      constant SoftmaxParams& params [[buffer(3)]],                        \
      uint tg_id [[threadgroup_position_in_grid]],                         \
      uint tid [[thread_position_in_threadgroup]],                         \
      uint tptg [[threads_per_threadgroup]],                               \
      uint simd_lane_id [[thread_index_in_simdgroup]],                     \
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

#define instantiate_softmax_backward_2pass_dot(DTYPE)                          \
  template [[host_name("softmax_backward_2pass_dot_" #DTYPE)]] [[kernel]] void \
  softmax_backward_2pass_dot<DTYPE>(                                           \
      device const DTYPE* grad_output [[buffer(0)]],                           \
      device const DTYPE* output [[buffer(1)]],                                \
      device float* partial_sums [[buffer(2)]],                                \
      constant SoftmaxParams& params [[buffer(3)]],                            \
      uint tg_id [[threadgroup_position_in_grid]],                             \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint lsize [[threads_per_threadgroup]],                                  \
      uint simd_lane_id [[thread_index_in_simdgroup]],                         \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_backward_2pass_grad(DTYPE)                     \
  template                                                                 \
      [[host_name("softmax_backward_2pass_grad_" #DTYPE)]] [[kernel]] void \
      softmax_backward_2pass_grad<DTYPE>(                                  \
          device const DTYPE* grad_output [[buffer(0)]],                   \
          device const DTYPE* output [[buffer(1)]],                        \
          device DTYPE* grad_input [[buffer(2)]],                          \
          device const float* partial_sums [[buffer(3)]],                  \
          constant SoftmaxParams& params [[buffer(4)]],                    \
          uint tg_id [[threadgroup_position_in_grid]],                     \
          uint tid [[thread_position_in_threadgroup]],                     \
          uint lsize [[threads_per_threadgroup]]);


#define instantiate_softmax_forward_2pass_reduce(DTYPE)                          \
  template [[host_name("softmax_forward_2pass_reduce_" #DTYPE)]] [[kernel]] void \
  softmax_forward_2pass_reduce<DTYPE>(                                           \
      device const DTYPE* input [[buffer(0)]],                                   \
      device float* partials [[buffer(1)]],                                      \
      constant SoftmaxParams& params [[buffer(2)]],                              \
      uint tg_id [[threadgroup_position_in_grid]],                               \
      uint tid [[thread_position_in_threadgroup]],                               \
      uint lsize [[threads_per_threadgroup]],                                    \
      uint simd_lane_id [[thread_index_in_simdgroup]],                           \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_forward_2pass_write(DTYPE)                          \
  template [[host_name("softmax_forward_2pass_write_" #DTYPE)]] [[kernel]] void \
  softmax_forward_2pass_write<DTYPE>(                                           \
      device const DTYPE* input [[buffer(0)]],                                  \
      device DTYPE* output [[buffer(1)]],                                       \
      device const float* partials [[buffer(2)]],                               \
      constant SoftmaxParams& params [[buffer(3)]],                             \
      uint tg_id [[threadgroup_position_in_grid]],                              \
      uint tid [[thread_position_in_threadgroup]],                              \
      uint lsize [[threads_per_threadgroup]]);

#define instantiate_softmax_forward_coalesced(DTYPE)                          \
  template [[host_name("softmax_forward_coalesced_" #DTYPE)]] [[kernel]] void  \
  softmax_forward_coalesced<DTYPE>(                                            \
      device const DTYPE* input [[buffer(0)]],                                 \
      device DTYPE* output [[buffer(1)]],                                      \
      constant SoftmaxParams& params [[buffer(2)]],                            \
      uint tg_id [[threadgroup_position_in_grid]],                             \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint lsize [[threads_per_threadgroup]],                                  \
      threadgroup float* smem [[threadgroup(0)]]);

#define instantiate_softmax_backward_coalesced(DTYPE)                          \
  template [[host_name("softmax_backward_coalesced_" #DTYPE)]] [[kernel]] void \
  softmax_backward_coalesced<DTYPE>(                                           \
      device const DTYPE* grad_output [[buffer(0)]],                           \
      device const DTYPE* output [[buffer(1)]],                                \
      device DTYPE* grad_input [[buffer(2)]],                                  \
      constant SoftmaxParams& params [[buffer(3)]],                            \
      uint tg_id [[threadgroup_position_in_grid]],                             \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint lsize [[threads_per_threadgroup]],                                  \
      threadgroup float* smem [[threadgroup(0)]]);

#define instantiate_softmax_forward_tiled(DTYPE)                          \
  template [[host_name("softmax_forward_tiled_" #DTYPE)]] [[kernel]] void \
  softmax_forward_tiled<DTYPE>(                                           \
      device const DTYPE* input [[buffer(0)]],                            \
      device DTYPE* output [[buffer(1)]],                                 \
      constant SoftmaxParams& params [[buffer(2)]],                       \
      uint tg_id [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_position_in_threadgroup]],                        \
      uint lsize [[threads_per_threadgroup]]);

#define instantiate_softmax_backward_tiled(DTYPE)                          \
  template [[host_name("softmax_backward_tiled_" #DTYPE)]] [[kernel]] void \
  softmax_backward_tiled<DTYPE>(                                           \
      device const DTYPE* grad_output [[buffer(0)]],                       \
      device const DTYPE* output [[buffer(1)]],                            \
      device DTYPE* grad_input [[buffer(2)]],                              \
      constant SoftmaxParams& params [[buffer(3)]],                        \
      uint tg_id [[threadgroup_position_in_grid]],                         \
      uint tid [[thread_position_in_threadgroup]],                         \
      uint lsize [[threads_per_threadgroup]]);

#define instantiate_softmax(DTYPE)                              \
  instantiate_softmax_forward_single_row(DTYPE)                 \
      instantiate_softmax_forward_looped(DTYPE)                 \
          instantiate_softmax_forward_tiled(DTYPE)              \
              instantiate_softmax_forward_coalesced(DTYPE)  \
              instantiate_softmax_forward_2pass_reduce(DTYPE)   \
                  instantiate_softmax_forward_2pass_write(DTYPE)\
                      instantiate_softmax_backward_single_row(DTYPE)        \
                          instantiate_softmax_backward_looped(DTYPE)        \
                              instantiate_softmax_backward_tiled(DTYPE)     \
                                  instantiate_softmax_backward_coalesced(DTYPE) \
                                  instantiate_softmax_backward_2pass_dot(DTYPE) \
                                      instantiate_softmax_backward_2pass_grad(DTYPE)

instantiate_softmax(float);
instantiate_softmax(half);
instantiate_softmax(bfloat);

#define instantiate_log_softmax_forward_single_row(DTYPE)                         \
  template [[host_name("log_softmax_forward_single_row_" #DTYPE)]] [[kernel]]     \
  void log_softmax_forward_single_row<DTYPE>(                                     \
      device const DTYPE* input [[buffer(0)]],                                    \
      device DTYPE* output [[buffer(1)]],                                         \
      constant SoftmaxParams& params [[buffer(2)]],                               \
      uint tg_id [[threadgroup_position_in_grid]],                                \
      uint tid [[thread_position_in_threadgroup]],                                \
      uint tptg [[threads_per_threadgroup]],                                      \
      uint simd_lane_id [[thread_index_in_simdgroup]],                            \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_log_softmax_forward_looped(DTYPE)                              \
  template [[host_name("log_softmax_forward_looped_" #DTYPE)]] [[kernel]] void     \
  log_softmax_forward_looped<DTYPE>(                                               \
      device const DTYPE* input [[buffer(0)]],                                     \
      device DTYPE* output [[buffer(1)]],                                          \
      constant SoftmaxParams& params [[buffer(2)]],                                \
      uint tg_id [[threadgroup_position_in_grid]],                                 \
      uint tid [[thread_position_in_threadgroup]],                                 \
      uint lsize [[threads_per_threadgroup]],                                      \
      uint simd_lane_id [[thread_index_in_simdgroup]],                             \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_log_softmax_backward_single_row(DTYPE)                         \
  template [[host_name("log_softmax_backward_single_row_" #DTYPE)]] [[kernel]]     \
  void log_softmax_backward_single_row<DTYPE>(                                     \
      device const DTYPE* grad_output [[buffer(0)]],                               \
      device const DTYPE* output [[buffer(1)]],                                    \
      device DTYPE* grad_input [[buffer(2)]],                                      \
      constant SoftmaxParams& params [[buffer(3)]],                                \
      uint tg_id [[threadgroup_position_in_grid]],                                 \
      uint tid [[thread_position_in_threadgroup]],                                 \
      uint tptg [[threads_per_threadgroup]],                                       \
      uint simd_lane_id [[thread_index_in_simdgroup]],                             \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_log_softmax_backward_looped(DTYPE)                              \
  template [[host_name("log_softmax_backward_looped_" #DTYPE)]] [[kernel]] void     \
  log_softmax_backward_looped<DTYPE>(                                               \
      device const DTYPE* grad_output [[buffer(0)]],                                \
      device const DTYPE* output [[buffer(1)]],                                     \
      device DTYPE* grad_input [[buffer(2)]],                                       \
      constant SoftmaxParams& params [[buffer(3)]],                                 \
      uint tg_id [[threadgroup_position_in_grid]],                                  \
      uint tid [[thread_position_in_threadgroup]],                                  \
      uint lsize [[threads_per_threadgroup]],                                       \
      uint simd_lane_id [[thread_index_in_simdgroup]],                              \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_log_softmax_backward_2pass_sum(DTYPE)                              \
  template [[host_name("log_softmax_backward_2pass_sum_" #DTYPE)]] [[kernel]] void     \
  log_softmax_backward_2pass_sum<DTYPE>(                                               \
      device const DTYPE* grad_output [[buffer(0)]],                                   \
      device float* partial_sums [[buffer(1)]],                                        \
      constant SoftmaxParams& params [[buffer(2)]],                                    \
      uint tg_id [[threadgroup_position_in_grid]],                                     \
      uint tid [[thread_position_in_threadgroup]],                                     \
      uint lsize [[threads_per_threadgroup]],                                          \
      uint simd_lane_id [[thread_index_in_simdgroup]],                                 \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_log_softmax_backward_2pass_grad(DTYPE)                         \
  template                                                                         \
      [[host_name("log_softmax_backward_2pass_grad_" #DTYPE)]] [[kernel]] void     \
      log_softmax_backward_2pass_grad<DTYPE>(                                      \
          device const DTYPE* grad_output [[buffer(0)]],                           \
          device const DTYPE* output [[buffer(1)]],                                \
          device DTYPE* grad_input [[buffer(2)]],                                  \
          device const float* partial_sums [[buffer(3)]],                          \
          constant SoftmaxParams& params [[buffer(4)]],                            \
          uint tg_id [[threadgroup_position_in_grid]],                             \
          uint tid [[thread_position_in_threadgroup]],                             \
          uint lsize [[threads_per_threadgroup]]);

#define instantiate_log_softmax_forward_2pass_reduce(DTYPE)                              \
  template [[host_name("log_softmax_forward_2pass_reduce_" #DTYPE)]] [[kernel]] void    \
  log_softmax_forward_2pass_reduce<DTYPE>(                                              \
      device const DTYPE* input [[buffer(0)]],                                          \
      device float* partials [[buffer(1)]],                                             \
      constant SoftmaxParams& params [[buffer(2)]],                                     \
      uint tg_id [[threadgroup_position_in_grid]],                                      \
      uint tid [[thread_position_in_threadgroup]],                                      \
      uint lsize [[threads_per_threadgroup]],                                           \
      uint simd_lane_id [[thread_index_in_simdgroup]],                                  \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_log_softmax_forward_2pass_write(DTYPE)                              \
  template [[host_name("log_softmax_forward_2pass_write_" #DTYPE)]] [[kernel]] void     \
  log_softmax_forward_2pass_write<DTYPE>(                                               \
      device const DTYPE* input [[buffer(0)]],                                          \
      device DTYPE* output [[buffer(1)]],                                               \
      device const float* partials [[buffer(2)]],                                       \
      constant SoftmaxParams& params [[buffer(3)]],                                     \
      uint tg_id [[threadgroup_position_in_grid]],                                      \
      uint tid [[thread_position_in_threadgroup]],                                      \
      uint lsize [[threads_per_threadgroup]]);

#define instantiate_log_softmax_forward_coalesced(DTYPE)                          \
  template [[host_name("log_softmax_forward_coalesced_" #DTYPE)]] [[kernel]] void \
  log_softmax_forward_coalesced<DTYPE>(                                           \
      device const DTYPE* input [[buffer(0)]],                                    \
      device DTYPE* output [[buffer(1)]],                                         \
      constant SoftmaxParams& params [[buffer(2)]],                               \
      uint tg_id [[threadgroup_position_in_grid]],                                \
      uint tid [[thread_position_in_threadgroup]],                                \
      uint lsize [[threads_per_threadgroup]],                                     \
      threadgroup float* smem [[threadgroup(0)]]);

#define instantiate_log_softmax_backward_coalesced(DTYPE)                          \
  template [[host_name("log_softmax_backward_coalesced_" #DTYPE)]] [[kernel]] void \
  log_softmax_backward_coalesced<DTYPE>(                                           \
      device const DTYPE* grad_output [[buffer(0)]],                               \
      device const DTYPE* output [[buffer(1)]],                                    \
      device DTYPE* grad_input [[buffer(2)]],                                      \
      constant SoftmaxParams& params [[buffer(3)]],                                \
      uint tg_id [[threadgroup_position_in_grid]],                                 \
      uint tid [[thread_position_in_threadgroup]],                                 \
      uint lsize [[threads_per_threadgroup]],                                      \
      threadgroup float* smem [[threadgroup(0)]]);

#define instantiate_log_softmax_forward_tiled(DTYPE)                          \
  template [[host_name("log_softmax_forward_tiled_" #DTYPE)]] [[kernel]] void \
  log_softmax_forward_tiled<DTYPE>(                                           \
      device const DTYPE* input [[buffer(0)]],                                \
      device DTYPE* output [[buffer(1)]],                                     \
      constant SoftmaxParams& params [[buffer(2)]],                           \
      uint tg_id [[threadgroup_position_in_grid]],                            \
      uint tid [[thread_position_in_threadgroup]],                            \
      uint lsize [[threads_per_threadgroup]]);

#define instantiate_log_softmax_backward_tiled(DTYPE)                          \
  template [[host_name("log_softmax_backward_tiled_" #DTYPE)]] [[kernel]] void \
  log_softmax_backward_tiled<DTYPE>(                                           \
      device const DTYPE* grad_output [[buffer(0)]],                           \
      device const DTYPE* output [[buffer(1)]],                                \
      device DTYPE* grad_input [[buffer(2)]],                                  \
      constant SoftmaxParams& params [[buffer(3)]],                            \
      uint tg_id [[threadgroup_position_in_grid]],                             \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint lsize [[threads_per_threadgroup]]);

#define instantiate_log_softmax(DTYPE)                                  \
  instantiate_log_softmax_forward_single_row(DTYPE)                     \
      instantiate_log_softmax_forward_looped(DTYPE)                     \
          instantiate_log_softmax_forward_tiled(DTYPE)                  \
              instantiate_log_softmax_forward_coalesced(DTYPE)      \
              instantiate_log_softmax_forward_2pass_reduce(DTYPE)       \
                  instantiate_log_softmax_forward_2pass_write(DTYPE)    \
                      instantiate_log_softmax_backward_single_row(DTYPE)\
                          instantiate_log_softmax_backward_looped(DTYPE)\
                              instantiate_log_softmax_backward_tiled(DTYPE)     \
                                  instantiate_log_softmax_backward_coalesced(DTYPE) \
                                  instantiate_log_softmax_backward_2pass_sum(DTYPE)     \
                                      instantiate_log_softmax_backward_2pass_grad(DTYPE)

instantiate_log_softmax(float);
instantiate_log_softmax(half);
instantiate_log_softmax(bfloat);
