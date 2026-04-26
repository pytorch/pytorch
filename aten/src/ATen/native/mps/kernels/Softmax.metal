#include <c10/metal/common.h>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using c10::metal::simdgroup_size;
constant constexpr uint ROWS_PER_TG = 32;
constant constexpr uint MAX_CHUNKS = 32;

// Returns 0 for x=-inf, avoids NaN from exp(-inf - -inf) poisoning the sum
inline float safe_exp_diff(float x, float m) {
  return (x == -INFINITY) ? 0.0f : fast::exp(x - m);
}

template <typename T, int N_READS>
kernel void softmax_single_pass(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint& dim_size [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  constant T* x = input + tg_id * dim_size + tid * N_READS;
  device T* y = output + tg_id * dim_size + tid * N_READS;

  threadgroup float scratch[simdgroup_size + 1];

  if (simd_gid == 0) {
    scratch[simd_lane] = -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float vals[N_READS];
  float thread_max = -INFINITY;
#pragma unroll
  for (int i = 0; i < N_READS; i++) {
    vals[i] = static_cast<float>(x[i]);
    thread_max = max(thread_max, vals[i]);
  }

  float lane_max = simd_max(thread_max);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float row_max = simd_max(scratch[simd_lane]);
    if (simd_lane == 0) {
      scratch[simdgroup_size] = row_max;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_max = scratch[simdgroup_size];

  if (simd_gid == 0) {
    scratch[simd_lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float evs[N_READS];
  float thread_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < N_READS; i++) {
    evs[i] = safe_exp_diff(vals[i], row_max);
    thread_sum += evs[i];
  }

  float lane_sum = simd_sum(thread_sum);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float row_sum = simd_sum(scratch[simd_lane]);
    if (simd_lane == 0) {
      scratch[simdgroup_size] = row_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_sum = scratch[simdgroup_size];

#pragma unroll
  for (int i = 0; i < N_READS; i++) {
    y[i] = static_cast<T>(evs[i] / row_sum);
  }
}

template <typename T>
kernel void softmax_looped(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint& dim_size [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  constant T* x = input + tg_id * dim_size;
  device T* y = output + tg_id * dim_size;

  threadgroup float scratch[simdgroup_size + 1];
  uint stride = tptg * N_READS;
  uint num_iters = dim_size / stride;

  float thread_max = -INFINITY;
  for (uint i = 0; i < num_iters; i++) {
    uint base = i * stride + tid * N_READS;
#pragma unroll
    for (int j = 0; j < N_READS; j++) {
      thread_max = max(thread_max, static_cast<float>(x[base + j]));
    }
  }
  float lane_max = simd_max(thread_max);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float row_max = simd_max(scratch[simd_lane]);
    if (simd_lane == 0) {
      scratch[simdgroup_size] = row_max;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_max = scratch[simdgroup_size];

  float thread_sum = 0.0f;
  for (uint i = 0; i < num_iters; i++) {
    uint base = i * stride + tid * N_READS;
#pragma unroll
    for (int j = 0; j < N_READS; j++) {
      float ev = safe_exp_diff(static_cast<float>(x[base + j]), row_max);
      y[base + j] = static_cast<T>(ev);
      thread_sum += ev;
    }
  }
  float lane_sum = simd_sum(thread_sum);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float row_sum = simd_sum(scratch[simd_lane]);
    if (simd_lane == 0) {
      scratch[simdgroup_size] = row_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_sum = scratch[simdgroup_size];

  for (uint i = 0; i < num_iters; i++) {
    uint base = i * stride + tid * N_READS;
#pragma unroll
    for (int j = 0; j < N_READS; j++) {
      y[base + j] = static_cast<T>(static_cast<float>(y[base + j]) / row_sum);
    }
  }
}

template <typename T>
kernel void softmax_general(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint& dim_size [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  constant T* x = input + tg_id * dim_size;
  device T* y = output + tg_id * dim_size;

  threadgroup float scratch[simdgroup_size + 1];

  float thread_max = -INFINITY;
  float thread_sum = 0.0f;
  for (uint i = tid; i < dim_size; i += tptg) {
    float xi = static_cast<float>(x[i]);
    float new_max = max(thread_max, xi);
    float correction = safe_exp_diff(thread_max, new_max);
    thread_sum = thread_sum * correction + safe_exp_diff(xi, new_max);
    thread_max = new_max;
  }

  if (simd_gid == 0) {
    scratch[simd_lane] = -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float lane_max = simd_max(thread_max);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float row_max = simd_max(scratch[simd_lane]);
    if (simd_lane == 0) {
      scratch[simdgroup_size] = row_max;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_max = scratch[simdgroup_size];

  float sum_adj = thread_sum * safe_exp_diff(thread_max, row_max);

  if (simd_gid == 0) {
    scratch[simd_lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float lane_s = simd_sum(sum_adj);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float row_sum = simd_sum(scratch[simd_lane]);
    if (simd_lane == 0) {
      scratch[simdgroup_size] = row_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_sum = scratch[simdgroup_size];

  float inv_s = 1.0f / row_sum;
  for (uint i = tid; i < dim_size; i += tptg) {
    y[i] = static_cast<T>(
        safe_exp_diff(static_cast<float>(x[i]), row_max) * inv_s);
  }
}

template <typename T>
kernel void softmax_split_k_pass1(
    constant T* input [[buffer(0)]],
    device float* partials [[buffer(1)]],
    constant uint2& dim_size_num_chunks [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  uint dim_size = dim_size_num_chunks.x;
  uint num_chunks = dim_size_num_chunks.y;
  uint chunk = tg_id % num_chunks;
  uint row = tg_id / num_chunks;
  uint chunk_size = (dim_size + num_chunks - 1) / num_chunks;
  uint chunk_start = chunk * chunk_size;
  uint chunk_end = min(chunk_start + chunk_size, dim_size);

  constant T* x = input + row * dim_size;

  float thread_max = -INFINITY;
  float thread_sum = 0.0f;
  for (uint i = chunk_start + tid; i < chunk_end; i += tptg) {
    float xi = static_cast<float>(x[i]);
    float new_max = max(thread_max, xi);
    float corr = safe_exp_diff(thread_max, new_max);
    thread_sum = thread_sum * corr + safe_exp_diff(xi, new_max);
    thread_max = new_max;
  }

  threadgroup float scratch[simdgroup_size + 1];
  if (simd_gid == 0) {
    scratch[simd_lane] = -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float lane_max = simd_max(thread_max);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float chunk_m_red = simd_max(scratch[simd_lane]);
    if (simd_lane == 0) {
      scratch[simdgroup_size] = chunk_m_red;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float chunk_m = scratch[simdgroup_size];

  float sum_adj = thread_sum * safe_exp_diff(thread_max, chunk_m);

  if (simd_gid == 0) {
    scratch[simd_lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float lane_s = simd_sum(sum_adj);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float chunk_s = simd_sum(scratch[simd_lane]);
    if (simd_lane == 0) {
      partials[(row * num_chunks + chunk) * 2 + 0] = chunk_m;
      partials[(row * num_chunks + chunk) * 2 + 1] = chunk_s;
    }
  }
}

template <typename T>
kernel void softmax_split_k_pass2(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant float* partials [[buffer(2)]],
    constant uint2& dim_size_num_chunks [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  uint dim_size = dim_size_num_chunks.x;
  uint num_chunks = dim_size_num_chunks.y;
  uint chunk = tg_id % num_chunks;
  uint row = tg_id / num_chunks;
  uint chunk_size = (dim_size + num_chunks - 1) / num_chunks;
  uint chunk_start = chunk * chunk_size;
  uint chunk_end = min(chunk_start + chunk_size, dim_size);

  float row_max = -INFINITY;
  float row_sum = 0.0f;
  for (uint c = 0; c < num_chunks; c++) {
    float c_m = partials[(row * num_chunks + c) * 2 + 0];
    float c_s = partials[(row * num_chunks + c) * 2 + 1];
    float new_max = max(row_max, c_m);
    float corr1 = safe_exp_diff(row_max, new_max);
    float corr2 = safe_exp_diff(c_m, new_max);
    row_sum = row_sum * corr1 + c_s * corr2;
    row_max = new_max;
  }
  float inv_s = 1.0f / row_sum;

  constant T* x = input + row * dim_size;
  device T* y = output + row * dim_size;
  for (uint i = chunk_start + tid; i < chunk_end; i += tptg) {
    y[i] = static_cast<T>(
        safe_exp_diff(static_cast<float>(x[i]), row_max) * inv_s);
  }
}

template <typename T, int N_LOCAL, int N_READS>
kernel void softmax_strided_inner_vec(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint2& params [[buffer(2)]],
    constant uint* sizes_other [[buffer(3)]],
    constant uint* input_strides_other [[buffer(4)]],
    constant uint* output_strides_other [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  uint dim_size = params.x;
  uint ndim_other = params.y;
  uint input_row_base = 0;
  uint output_row_base = 0;
  uint r = tg_id;
  for (int j = (int)ndim_other - 1; j >= 0; j--) {
    uint axis_idx = r % sizes_other[j];
    r /= sizes_other[j];
    input_row_base += axis_idx * input_strides_other[j];
    output_row_base += axis_idx * output_strides_other[j];
  }

  threadgroup float scratch[simdgroup_size + 1];

  float cached[N_LOCAL * N_READS];
  float thread_max = -INFINITY;
#pragma unroll
  for (int k = 0; k < N_LOCAL; k++) {
    uint dbase = (tid + k * tptg) * N_READS;
#pragma unroll
    for (int j = 0; j < N_READS; j++) {
      uint d = dbase + j;
      if (d < dim_size) {
        cached[k * N_READS + j] = static_cast<float>(input[input_row_base + d]);
        thread_max = max(thread_max, cached[k * N_READS + j]);
      }
    }
  }

  if (simd_gid == 0) {
    scratch[simd_lane] = -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float lane_max = simd_max(thread_max);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float row_max = simd_max(scratch[simd_lane]);
    if (simd_lane == 0) {
      scratch[simdgroup_size] = row_max;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_max = scratch[simdgroup_size];

  float thread_sum = 0.0f;
#pragma unroll
  for (int k = 0; k < N_LOCAL; k++) {
    uint dbase = (tid + k * tptg) * N_READS;
#pragma unroll
    for (int j = 0; j < N_READS; j++) {
      uint d = dbase + j;
      if (d < dim_size) {
        cached[k * N_READS + j] =
            safe_exp_diff(cached[k * N_READS + j], row_max);
        thread_sum += cached[k * N_READS + j];
      }
    }
  }

  if (simd_gid == 0) {
    scratch[simd_lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float lane_s = simd_sum(thread_sum);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float row_sum = simd_sum(scratch[simd_lane]);
    if (simd_lane == 0) {
      scratch[simdgroup_size] = row_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_sum = scratch[simdgroup_size];

  float inv_s = 1.0f / row_sum;
#pragma unroll
  for (int k = 0; k < N_LOCAL; k++) {
    uint dbase = (tid + k * tptg) * N_READS;
#pragma unroll
    for (int j = 0; j < N_READS; j++) {
      uint d = dbase + j;
      if (d < dim_size) {
        output[output_row_base + d] =
            static_cast<T>(cached[k * N_READS + j] * inv_s);
      }
    }
  }
}

template <typename T, int N_LOCAL>
kernel void softmax_strided_inner(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint4& params [[buffer(2)]],
    constant uint* sizes_other [[buffer(3)]],
    constant uint* input_strides_other [[buffer(4)]],
    constant uint* output_strides_other [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  uint dim_size = params.x;
  uint input_dim_stride = params.y;
  uint output_dim_stride = params.z;
  uint ndim_other = params.w;
  uint input_row_base = 0;
  uint output_row_base = 0;
  uint r = tg_id;
  for (int j = (int)ndim_other - 1; j >= 0; j--) {
    uint axis_idx = r % sizes_other[j];
    r /= sizes_other[j];
    input_row_base += axis_idx * input_strides_other[j];
    output_row_base += axis_idx * output_strides_other[j];
  }

  threadgroup float scratch[simdgroup_size + 1];

  float cached[N_LOCAL];
  float thread_max = -INFINITY;
#pragma unroll
  for (int k = 0; k < N_LOCAL; k++) {
    uint d = tid + k * tptg;
    if (d < dim_size) {
      cached[k] =
          static_cast<float>(input[input_row_base + d * input_dim_stride]);
      thread_max = max(thread_max, cached[k]);
    }
  }

  if (simd_gid == 0) {
    scratch[simd_lane] = -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float lane_max = simd_max(thread_max);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float row_max = simd_max(scratch[simd_lane]);
    if (simd_lane == 0) {
      scratch[simdgroup_size] = row_max;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_max = scratch[simdgroup_size];

  float thread_sum = 0.0f;
#pragma unroll
  for (int k = 0; k < N_LOCAL; k++) {
    uint d = tid + k * tptg;
    if (d < dim_size) {
      cached[k] = safe_exp_diff(cached[k], row_max);
      thread_sum += cached[k];
    }
  }

  if (simd_gid == 0) {
    scratch[simd_lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float lane_s = simd_sum(thread_sum);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float row_sum = simd_sum(scratch[simd_lane]);
    if (simd_lane == 0) {
      scratch[simdgroup_size] = row_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_sum = scratch[simdgroup_size];

  float inv_s = 1.0f / row_sum;
#pragma unroll
  for (int k = 0; k < N_LOCAL; k++) {
    uint d = tid + k * tptg;
    if (d < dim_size) {
      output[output_row_base + d * output_dim_stride] =
          static_cast<T>(cached[k] * inv_s);
    }
  }
}

template <typename T>
kernel void softmax_strided_looped(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint2& params [[buffer(2)]],
    constant uint* sizes_other [[buffer(3)]],
    constant uint* input_strides_other [[buffer(4)]],
    constant uint* output_strides_other [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  uint dim_size = params.x;
  uint ndim_other = params.y;
  uint input_row_base = 0;
  uint output_row_base = 0;
  uint r = tg_id;
  for (int j = (int)ndim_other - 1; j >= 0; j--) {
    uint axis_idx = r % sizes_other[j];
    r /= sizes_other[j];
    input_row_base += axis_idx * input_strides_other[j];
    output_row_base += axis_idx * output_strides_other[j];
  }

  threadgroup float scratch[simdgroup_size + 1];

  float thread_max = -INFINITY;
  float thread_sum = 0.0f;
  for (uint i = tid; i < dim_size; i += tptg) {
    float xi = static_cast<float>(input[input_row_base + i]);
    float new_max = max(thread_max, xi);
    float correction = safe_exp_diff(thread_max, new_max);
    thread_sum = thread_sum * correction + safe_exp_diff(xi, new_max);
    thread_max = new_max;
  }

  if (simd_gid == 0) {
    scratch[simd_lane] = -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float lane_max = simd_max(thread_max);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float row_max = simd_max(scratch[simd_lane]);
    if (simd_lane == 0) {
      scratch[simdgroup_size] = row_max;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_max = scratch[simdgroup_size];

  float sum_adj = thread_sum * safe_exp_diff(thread_max, row_max);

  if (simd_gid == 0) {
    scratch[simd_lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float lane_s = simd_sum(sum_adj);
  if (simd_lane == 0) {
    scratch[simd_gid] = lane_s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float row_sum = simd_sum(scratch[simd_lane]);
    if (simd_lane == 0) {
      scratch[simdgroup_size] = row_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_sum = scratch[simdgroup_size];

  float inv_s = 1.0f / row_sum;
  for (uint i = tid; i < dim_size; i += tptg) {
    output[output_row_base + i] = static_cast<T>(
        safe_exp_diff(static_cast<float>(input[input_row_base + i]), row_max) *
        inv_s);
  }
}

template <typename T, int CHUNK_CAP>
kernel void softmax_strided_chunked(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint4& params [[buffer(2)]],
    constant uint& ndim_other [[buffer(3)]],
    constant uint* sizes_other [[buffer(4)]],
    constant uint* input_strides_other [[buffer(5)]],
    constant uint* output_strides_other [[buffer(6)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  uint dim_size = params.x;
  uint input_dim_stride = params.y;
  uint output_dim_stride = params.z;
  uint num_rows = params.w;
  uint num_chunks = tptg / 32;
  uint chunk_size = (dim_size + num_chunks - 1) / num_chunks;

  uint row_in_tg = simd_lane;
  uint chunk = simd_gid;
  uint global_row = tg_id * ROWS_PER_TG + row_in_tg;
  bool valid_row = global_row < num_rows;

  uint input_row_base = 0;
  uint output_row_base = 0;
  if (valid_row) {
    uint r = global_row;
    for (int j = (int)ndim_other - 1; j >= 0; j--) {
      uint axis_idx = r % sizes_other[j];
      r /= sizes_other[j];
      input_row_base += axis_idx * input_strides_other[j];
      output_row_base += axis_idx * output_strides_other[j];
    }
  }

  uint chunk_start = chunk * chunk_size;

  threadgroup float chunk_ms[ROWS_PER_TG][MAX_CHUNKS];
  threadgroup float chunk_ss[ROWS_PER_TG][MAX_CHUNKS];
  threadgroup float row_ms_tg[ROWS_PER_TG];
  threadgroup float row_ss_tg[ROWS_PER_TG];

  if (CHUNK_CAP > 0) {
    float cached[CHUNK_CAP > 0 ? CHUNK_CAP : 1];
    float chunk_m = -INFINITY;
    float chunk_s = 0.0f;
    if (valid_row) {
#pragma unroll
      for (int i = 0; i < CHUNK_CAP; i++) {
        uint d = chunk_start + i;
        if (i < (int)chunk_size && d < dim_size) {
          cached[i] =
              static_cast<float>(input[input_row_base + d * input_dim_stride]);
          chunk_m = max(chunk_m, cached[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < CHUNK_CAP; i++) {
        uint d = chunk_start + i;
        if (i < (int)chunk_size && d < dim_size) {
          cached[i] = safe_exp_diff(cached[i], chunk_m);
          chunk_s += cached[i];
        }
      }
    }
    chunk_ms[row_in_tg][chunk] = chunk_m;
    chunk_ss[row_in_tg][chunk] = chunk_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint rows_per_warp = (ROWS_PER_TG + num_chunks - 1) / num_chunks;
    for (uint k = 0; k < rows_per_warp; k++) {
      uint r = simd_gid + k * num_chunks;
      if (r < ROWS_PER_TG) {
        bool active = simd_lane < num_chunks;
        float c_m = active ? chunk_ms[r][simd_lane] : -INFINITY;
        float c_s = active ? chunk_ss[r][simd_lane] : 0.0f;
        float row_m_new = simd_max(c_m);
        float adj = active ? c_s * safe_exp_diff(c_m, row_m_new) : 0.0f;
        float row_s_new = simd_sum(adj);
        if (simd_lane == 0) {
          row_ms_tg[r] = row_m_new;
          row_ss_tg[r] = row_s_new;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_row) {
      float row_max = row_ms_tg[row_in_tg];
      float row_sum = row_ss_tg[row_in_tg];
      float scale = safe_exp_diff(chunk_m, row_max) / row_sum;
#pragma unroll
      for (int i = 0; i < CHUNK_CAP; i++) {
        uint d = chunk_start + i;
        if (i < (int)chunk_size && d < dim_size) {
          output[output_row_base + d * output_dim_stride] =
              static_cast<T>(cached[i] * scale);
        }
      }
    }
  } else {
    uint chunk_end = min(chunk_start + chunk_size, dim_size);
    float chunk_m = -INFINITY;
    float chunk_s = 0.0f;
    if (valid_row) {
      for (uint d = chunk_start; d < chunk_end; d++) {
        float xi =
            static_cast<float>(input[input_row_base + d * input_dim_stride]);
        float new_max = max(chunk_m, xi);
        float correction = safe_exp_diff(chunk_m, new_max);
        chunk_s = chunk_s * correction + safe_exp_diff(xi, new_max);
        chunk_m = new_max;
      }
    }

    chunk_ms[row_in_tg][chunk] = chunk_m;
    chunk_ss[row_in_tg][chunk] = chunk_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint rows_per_warp = (ROWS_PER_TG + num_chunks - 1) / num_chunks;
    for (uint k = 0; k < rows_per_warp; k++) {
      uint r = simd_gid + k * num_chunks;
      if (r < ROWS_PER_TG) {
        bool active = simd_lane < num_chunks;
        float c_m = active ? chunk_ms[r][simd_lane] : -INFINITY;
        float c_s = active ? chunk_ss[r][simd_lane] : 0.0f;
        float row_m_new = simd_max(c_m);
        float adj = active ? c_s * safe_exp_diff(c_m, row_m_new) : 0.0f;
        float row_s_new = simd_sum(adj);
        if (simd_lane == 0) {
          row_ms_tg[r] = row_m_new;
          row_ss_tg[r] = row_s_new;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_row) {
      float row_max = row_ms_tg[row_in_tg];
      float row_sum = row_ss_tg[row_in_tg];
      float inv_s = 1.0f / row_sum;
      for (uint d = chunk_start; d < chunk_end; d++) {
        float xi =
            static_cast<float>(input[input_row_base + d * input_dim_stride]);
        output[output_row_base + d * output_dim_stride] =
            static_cast<T>(safe_exp_diff(xi, row_max) * inv_s);
      }
    }
  }
}

template <typename T>
kernel void softmax_strided_outer(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint4& params [[buffer(2)]],
    constant uint& ndim_other [[buffer(3)]],
    constant uint* sizes_other [[buffer(4)]],
    constant uint* input_strides_other [[buffer(5)]],
    constant uint* output_strides_other [[buffer(6)]],
    uint thread_id [[thread_position_in_grid]]) {
  uint dim_size = params.x;
  uint input_dim_stride = params.y;
  uint output_dim_stride = params.z;
  uint num_rows = params.w;
  uint row_id = thread_id;
  if (row_id >= num_rows) {
    return;
  }
  uint input_row_base = 0;
  uint output_row_base = 0;
  uint r = row_id;
  for (int j = (int)ndim_other - 1; j >= 0; j--) {
    uint axis_idx = r % sizes_other[j];
    r /= sizes_other[j];
    input_row_base += axis_idx * input_strides_other[j];
    output_row_base += axis_idx * output_strides_other[j];
  }

  float row_max = -INFINITY;
  float row_sum = 0.0f;
  for (uint d = 0; d < dim_size; d++) {
    float xi = static_cast<float>(input[input_row_base + d * input_dim_stride]);
    float new_max = max(row_max, xi);
    float correction = safe_exp_diff(row_max, new_max);
    row_sum = row_sum * correction + safe_exp_diff(xi, new_max);
    row_max = new_max;
  }
  float inv_s = 1.0f / row_sum;
  for (uint d = 0; d < dim_size; d++) {
    float xi = static_cast<float>(input[input_row_base + d * input_dim_stride]);
    output[output_row_base + d * output_dim_stride] =
        static_cast<T>(safe_exp_diff(xi, row_max) * inv_s);
  }
}

#define INSTANTIATE_SOFTMAX_SINGLE_ROW(DTYPE, N)                               \
  template [[host_name("softmax_single_pass_" #DTYPE "_" #N)]] [[kernel]] void \
  softmax_single_pass<DTYPE, N>(                                               \
      constant DTYPE * input [[buffer(0)]],                                    \
      device DTYPE * output [[buffer(1)]],                                     \
      constant uint & dim_size [[buffer(2)]],                                  \
      uint tg_id [[threadgroup_position_in_grid]],                             \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]]);

#define INSTANTIATE_SOFTMAX_LOOPED(DTYPE)                          \
  template [[host_name("softmax_looped_" #DTYPE)]] [[kernel]] void \
  softmax_looped<DTYPE>(                                           \
      constant DTYPE * input [[buffer(0)]],                        \
      device DTYPE * output [[buffer(1)]],                         \
      constant uint & dim_size [[buffer(2)]],                      \
      uint tg_id [[threadgroup_position_in_grid]],                 \
      uint tid [[thread_position_in_threadgroup]],                 \
      uint tptg [[threads_per_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                \
      uint simd_gid [[simdgroup_index_in_threadgroup]]);

#define INSTANTIATE_SOFTMAX_GENERAL(DTYPE)                          \
  template [[host_name("softmax_general_" #DTYPE)]] [[kernel]] void \
  softmax_general<DTYPE>(                                           \
      constant DTYPE * input [[buffer(0)]],                         \
      device DTYPE * output [[buffer(1)]],                          \
      constant uint & dim_size [[buffer(2)]],                       \
      uint tg_id [[threadgroup_position_in_grid]],                  \
      uint tid [[thread_position_in_threadgroup]],                  \
      uint tptg [[threads_per_threadgroup]],                        \
      uint simd_lane [[thread_index_in_simdgroup]],                 \
      uint simd_gid [[simdgroup_index_in_threadgroup]]);

#define INSTANTIATE_SOFTMAX_SPLIT_K(DTYPE)                                \
  template [[host_name("softmax_split_k_pass1_" #DTYPE)]] [[kernel]] void \
  softmax_split_k_pass1<DTYPE>(                                           \
      constant DTYPE * input [[buffer(0)]],                               \
      device float* partials [[buffer(1)]],                               \
      constant uint2& dim_size_num_chunks [[buffer(2)]],                  \
      uint tg_id [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_position_in_threadgroup]],                        \
      uint tptg [[threads_per_threadgroup]],                              \
      uint simd_lane [[thread_index_in_simdgroup]],                       \
      uint simd_gid [[simdgroup_index_in_threadgroup]]);                  \
  template [[host_name("softmax_split_k_pass2_" #DTYPE)]] [[kernel]] void \
  softmax_split_k_pass2<DTYPE>(                                           \
      constant DTYPE * input [[buffer(0)]],                               \
      device DTYPE * output [[buffer(1)]],                                \
      constant float* partials [[buffer(2)]],                             \
      constant uint2& dim_size_num_chunks [[buffer(3)]],                  \
      uint tg_id [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_position_in_threadgroup]],                        \
      uint tptg [[threads_per_threadgroup]]);

#define INSTANTIATE_SOFTMAX_STRIDED_INNER_VEC(DTYPE, N, R)        \
  template [[host_name("softmax_strided_inner_vec_" #DTYPE "_" #N \
                       "_" #R)]] [[kernel]] void                  \
  softmax_strided_inner_vec<DTYPE, N, R>(                         \
      constant DTYPE * input [[buffer(0)]],                       \
      device DTYPE * output [[buffer(1)]],                        \
      constant uint2 & params [[buffer(2)]],                      \
      constant uint * sizes_other [[buffer(3)]],                  \
      constant uint * input_strides_other [[buffer(4)]],          \
      constant uint * output_strides_other [[buffer(5)]],         \
      uint tg_id [[threadgroup_position_in_grid]],                \
      uint tid [[thread_position_in_threadgroup]],                \
      uint tptg [[threads_per_threadgroup]],                      \
      uint simd_lane [[thread_index_in_simdgroup]],               \
      uint simd_gid [[simdgroup_index_in_threadgroup]]);

#define INSTANTIATE_SOFTMAX_STRIDED_INNER(DTYPE, N)                         \
  template                                                                  \
      [[host_name("softmax_strided_inner_" #DTYPE "_" #N)]] [[kernel]] void \
      softmax_strided_inner<DTYPE, N>(                                      \
          constant DTYPE * input [[buffer(0)]],                             \
          device DTYPE * output [[buffer(1)]],                              \
          constant uint4 & params [[buffer(2)]],                            \
          constant uint * sizes_other [[buffer(3)]],                        \
          constant uint * input_strides_other [[buffer(4)]],                \
          constant uint * output_strides_other [[buffer(5)]],               \
          uint tg_id [[threadgroup_position_in_grid]],                      \
          uint tid [[thread_position_in_threadgroup]],                      \
          uint tptg [[threads_per_threadgroup]],                            \
          uint simd_lane [[thread_index_in_simdgroup]],                     \
          uint simd_gid [[simdgroup_index_in_threadgroup]]);

#define INSTANTIATE_SOFTMAX_STRIDED_LOOPED(DTYPE)                          \
  template [[host_name("softmax_strided_looped_" #DTYPE)]] [[kernel]] void \
  softmax_strided_looped<DTYPE>(                                           \
      constant DTYPE * input [[buffer(0)]],                                \
      device DTYPE * output [[buffer(1)]],                                 \
      constant uint2 & params [[buffer(2)]],                               \
      constant uint * sizes_other [[buffer(3)]],                           \
      constant uint * input_strides_other [[buffer(4)]],                   \
      constant uint * output_strides_other [[buffer(5)]],                  \
      uint tg_id [[threadgroup_position_in_grid]],                         \
      uint tid [[thread_position_in_threadgroup]],                         \
      uint tptg [[threads_per_threadgroup]],                               \
      uint simd_lane [[thread_index_in_simdgroup]],                        \
      uint simd_gid [[simdgroup_index_in_threadgroup]]);

#define INSTANTIATE_SOFTMAX_STRIDED_CHUNKED(DTYPE, CAP)   \
  template [[host_name("softmax_strided_chunked_" #DTYPE  \
                       "_" #CAP)]] [[kernel]] void        \
  softmax_strided_chunked<DTYPE, CAP>(                    \
      constant DTYPE * input [[buffer(0)]],               \
      device DTYPE * output [[buffer(1)]],                \
      constant uint4 & params [[buffer(2)]],              \
      constant uint & ndim_other [[buffer(3)]],           \
      constant uint * sizes_other [[buffer(4)]],          \
      constant uint * input_strides_other [[buffer(5)]],  \
      constant uint * output_strides_other [[buffer(6)]], \
      uint tg_id [[threadgroup_position_in_grid]],        \
      uint tptg [[threads_per_threadgroup]],              \
      uint simd_lane [[thread_index_in_simdgroup]],       \
      uint simd_gid [[simdgroup_index_in_threadgroup]]);

#define INSTANTIATE_SOFTMAX_STRIDED_OUTER(DTYPE)                          \
  template [[host_name("softmax_strided_outer_" #DTYPE)]] [[kernel]] void \
  softmax_strided_outer<DTYPE>(                                           \
      constant DTYPE * input [[buffer(0)]],                               \
      device DTYPE * output [[buffer(1)]],                                \
      constant uint4 & params [[buffer(2)]],                              \
      constant uint & ndim_other [[buffer(3)]],                           \
      constant uint * sizes_other [[buffer(4)]],                          \
      constant uint * input_strides_other [[buffer(5)]],                  \
      constant uint * output_strides_other [[buffer(6)]],                 \
      uint thread_id [[thread_position_in_grid]]);

INSTANTIATE_SOFTMAX_SINGLE_ROW(float, 1);
INSTANTIATE_SOFTMAX_SINGLE_ROW(float, 4);
INSTANTIATE_SOFTMAX_LOOPED(float);
INSTANTIATE_SOFTMAX_SINGLE_ROW(half, 1);
INSTANTIATE_SOFTMAX_SINGLE_ROW(half, 4);
INSTANTIATE_SOFTMAX_LOOPED(half);
INSTANTIATE_SOFTMAX_SINGLE_ROW(bfloat, 1);
INSTANTIATE_SOFTMAX_SINGLE_ROW(bfloat, 4);
INSTANTIATE_SOFTMAX_LOOPED(bfloat);
INSTANTIATE_SOFTMAX_SPLIT_K(float);
INSTANTIATE_SOFTMAX_SPLIT_K(half);
INSTANTIATE_SOFTMAX_SPLIT_K(bfloat);
INSTANTIATE_SOFTMAX_GENERAL(float);
INSTANTIATE_SOFTMAX_GENERAL(half);
INSTANTIATE_SOFTMAX_GENERAL(bfloat);
INSTANTIATE_SOFTMAX_STRIDED_INNER_VEC(float, 1, 4);
INSTANTIATE_SOFTMAX_STRIDED_INNER_VEC(half, 1, 4);
INSTANTIATE_SOFTMAX_STRIDED_INNER_VEC(bfloat, 1, 4);
INSTANTIATE_SOFTMAX_STRIDED_INNER_VEC(float, 2, 4);
INSTANTIATE_SOFTMAX_STRIDED_INNER_VEC(half, 2, 4);
INSTANTIATE_SOFTMAX_STRIDED_INNER_VEC(bfloat, 2, 4);
INSTANTIATE_SOFTMAX_STRIDED_INNER_VEC(float, 4, 4);
INSTANTIATE_SOFTMAX_STRIDED_INNER_VEC(half, 4, 4);
INSTANTIATE_SOFTMAX_STRIDED_INNER_VEC(bfloat, 4, 4);
INSTANTIATE_SOFTMAX_STRIDED_INNER(float, 1);
INSTANTIATE_SOFTMAX_STRIDED_INNER(half, 1);
INSTANTIATE_SOFTMAX_STRIDED_INNER(bfloat, 1);
INSTANTIATE_SOFTMAX_STRIDED_INNER(float, 2);
INSTANTIATE_SOFTMAX_STRIDED_INNER(half, 2);
INSTANTIATE_SOFTMAX_STRIDED_INNER(bfloat, 2);
INSTANTIATE_SOFTMAX_STRIDED_INNER(float, 4);
INSTANTIATE_SOFTMAX_STRIDED_INNER(half, 4);
INSTANTIATE_SOFTMAX_STRIDED_INNER(bfloat, 4);
INSTANTIATE_SOFTMAX_STRIDED_INNER(float, 8);
INSTANTIATE_SOFTMAX_STRIDED_INNER(half, 8);
INSTANTIATE_SOFTMAX_STRIDED_INNER(bfloat, 8);
INSTANTIATE_SOFTMAX_STRIDED_INNER(float, 16);
INSTANTIATE_SOFTMAX_STRIDED_INNER(half, 16);
INSTANTIATE_SOFTMAX_STRIDED_INNER(bfloat, 16);
INSTANTIATE_SOFTMAX_STRIDED_LOOPED(float);
INSTANTIATE_SOFTMAX_STRIDED_LOOPED(half);
INSTANTIATE_SOFTMAX_STRIDED_LOOPED(bfloat);
INSTANTIATE_SOFTMAX_STRIDED_CHUNKED(float, 32);
INSTANTIATE_SOFTMAX_STRIDED_CHUNKED(half, 32);
INSTANTIATE_SOFTMAX_STRIDED_CHUNKED(bfloat, 32);
INSTANTIATE_SOFTMAX_STRIDED_CHUNKED(float, 0);
INSTANTIATE_SOFTMAX_STRIDED_CHUNKED(half, 0);
INSTANTIATE_SOFTMAX_STRIDED_CHUNKED(bfloat, 0);
INSTANTIATE_SOFTMAX_STRIDED_OUTER(float);
INSTANTIATE_SOFTMAX_STRIDED_OUTER(half);
INSTANTIATE_SOFTMAX_STRIDED_OUTER(bfloat);
