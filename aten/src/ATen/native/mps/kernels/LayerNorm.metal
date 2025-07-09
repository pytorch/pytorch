#include <c10/metal/common.h>
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
using c10::metal::simdgroup_size;

template <typename T>
kernel void layer_norm_single_row(
    device T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device T* meanOut [[buffer(2)]],
    device T* rstdTensor [[buffer(3)]],
    constant uint& axis_size [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    constant int& use_weight [[buffer(6)]],
    constant int& use_bias [[buffer(7)]],
    device T* weight [[buffer(8)]],
    device T* bias [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;

  // each threadgroup handles one full “row” of length axis_size
  uint row_offset = tg_id * axis_size;
  device T* x = input + row_offset + tid * N_READS;
  device T* out = output + row_offset + tid * N_READS;

  // partial sums for calculating mean & variance
  float partial_sum = 0.0f;
  float partial_sum_sq = 0.0f;
  uint base_lane = tid * N_READS;
  if (base_lane + N_READS <= axis_size) {
    float4 v4 = float4(x[0], x[1], x[2], x[3]);
    partial_sum = v4.x + v4.y + v4.z + v4.w;
    partial_sum_sq = dot(v4, v4);
  } else {
    int remaining = axis_size - base_lane;
    if (remaining >= 3) {
      float3 v3 = float3(x[0], x[1], x[2]);
      partial_sum = v3.x + v3.y + v3.z;
      partial_sum_sq = dot(v3, v3);
    } else if (remaining >= 2) {
      float2 v2 = float2(x[0], x[1]);
      partial_sum = v2.x + v2.y;
      partial_sum_sq = dot(v2, v2);
    } else if (remaining >= 1) {
      float v = x[0];
      partial_sum = v;
      partial_sum_sq = fma(v, v, partial_sum_sq);
    }
  }

  // threadgroup‐wide reduction
  threadgroup float local_sums[simdgroup_size];
  threadgroup float local_sums_sq[simdgroup_size];
  threadgroup float tg_mean[1];
  threadgroup float tg_inv_std[1];

  if (simdgroup_id == 0) {
    local_sums[simd_lane_id] = 0.0f;
    local_sums_sq[simd_lane_id] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // each simdgroup writes its partial
  float group_partial_sum = simd_sum(partial_sum);
  float group_partial_sum_sq = simd_sum(partial_sum_sq);
  if (simd_lane_id == 0) {
    local_sums[simdgroup_id] = group_partial_sum;
    local_sums_sq[simdgroup_id] = group_partial_sum_sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // warp 0 reduces those 32 values
  if (simdgroup_id == 0) {
    float sum = simd_sum(local_sums[simd_lane_id]);
    float sum_sq = simd_sum(local_sums_sq[simd_lane_id]);
    if (simd_lane_id == 0) {
      float mean = sum / float(axis_size);
      float var = sum_sq / float(axis_size) - mean * mean;
      var = var < 1e-6 ? 0.0f : var; // for rsqrt precision
      float inv_std = metal::precise::rsqrt(var + epsilon);
      tg_mean[0] = mean;
      tg_inv_std[0] = inv_std;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float mean = tg_mean[0];
  float inv_std = tg_inv_std[0];

  // normalize and optional scale & shift
  if (base_lane + N_READS <= axis_size) {
#pragma unroll
    for (int i = 0; i < N_READS; i++) {
      float v = float(x[i]);
      float norm = (v - mean) * inv_std;
      uint lane_idx = base_lane + i;
      if (use_weight)
        norm *= float(weight[lane_idx]);
      if (use_bias)
        norm += float(bias[lane_idx]);
      out[i] = static_cast<T>(norm);
    }
  } else {
#pragma unroll
    for (int i = 0; i < N_READS; i++) {
      uint lane_idx = base_lane + i;
      if (lane_idx < axis_size) {
        float v = float(x[i]);
        float norm = (v - mean) * inv_std;
        if (use_weight)
          norm *= float(weight[lane_idx]);
        if (use_bias)
          norm += float(bias[lane_idx]);
        out[i] = static_cast<T>(norm);
      }
    }
  }

  if (tid == 0 && simd_lane_id == 0) {
    meanOut[tg_id] = static_cast<T>(mean);
    rstdTensor[tg_id] = static_cast<T>(inv_std);
  }
}

template <typename T>
kernel void layer_norm_looped(
    device T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device T* meanOut [[buffer(2)]],
    device T* rstdTensor [[buffer(3)]],
    constant uint& axis_size [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    constant int& use_weight [[buffer(6)]],
    constant int& use_bias [[buffer(7)]],
    device T* weight [[buffer(8)]],
    device T* bias [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;

  uint row_offset = tg_id * axis_size;
  device T* x = input + row_offset;
  device T* out = output + row_offset;

  float partial_sum = 0.0f;
  float partial_sum_sq = 0.0f;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 xi4 = float4(x[base], x[base + 1], x[base + 2], x[base + 3]);
      partial_sum += xi4.x + xi4.y + xi4.z + xi4.w;
      partial_sum_sq += dot(xi4, xi4);
    } else {
      int remaining = axis_size - base;
      if (remaining >= 3) {
        float3 v3 = float3(x[base], x[base + 1], x[base + 2]);
        partial_sum += v3.x + v3.y + v3.z;
        partial_sum_sq += dot(v3, v3);
      } else if (remaining >= 2) {
        float2 v2 = float2(x[base], x[base + 1]);
        partial_sum += v2.x + v2.y;
        partial_sum_sq += dot(v2, v2);
      } else if (remaining >= 1) {
        float v = x[base];
        partial_sum += v;
        partial_sum_sq = fma(v, v, partial_sum_sq);
      }
    }
  }

  partial_sum = simd_sum(partial_sum);
  partial_sum_sq = simd_sum(partial_sum_sq);

  threadgroup float local_sums[simdgroup_size];
  threadgroup float local_sums_sq[simdgroup_size];
  threadgroup float tg_mean[1];
  threadgroup float tg_inv_std[1];

  if (simd_lane_id == 0) {
    local_sums[simdgroup_id] = 0.0f;
    local_sums_sq[simdgroup_id] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lane_id == 0) {
    local_sums[simdgroup_id] = partial_sum;
    local_sums_sq[simdgroup_id] = partial_sum_sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_id == 0) {
    float s = simd_sum(local_sums[simd_lane_id]);
    float ss = simd_sum(local_sums_sq[simd_lane_id]);
    if (simd_lane_id == 0) {
      float mean = s / float(axis_size);
      float var = ss / float(axis_size) - mean * mean;
      var = var < 1e-6 ? 0.0f : var; // for rsqrt precision
      float inv_std = metal::precise::rsqrt(var + epsilon);
      tg_mean[0] = mean;
      tg_inv_std[0] = inv_std;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float mean = tg_mean[0];
  float inv_std = tg_inv_std[0];

  // write back normalized + scale/shift if needed
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
#pragma unroll
      for (int i = 0; i < N_READS; i++) {
        float xi = float(x[base + i]);
        float norm = (xi - mean) * inv_std;
        if (use_weight)
          norm *= float(weight[base + i]);
        if (use_bias)
          norm += float(bias[base + i]);
        out[base + i] = T(norm);
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++) {
        if (base + i < axis_size) {
          float xi = float(x[base + i]);
          float norm = (xi - mean) * inv_std;
          if (use_weight)
            norm *= float(weight[base + i]);
          if (use_bias)
            norm += float(bias[base + i]);
          out[base + i] = T(norm);
        }
      }
    }
  }

  if (tid == 0 && simd_lane_id == 0) {
    meanOut[tg_id] = T(mean);
    rstdTensor[tg_id] = T(inv_std);
  }
}

#define instantiate_layer_norm_single_row(DTYPE)                          \
  template [[host_name("layer_norm_single_row_" #DTYPE)]] [[kernel]] void \
  layer_norm_single_row<DTYPE>(                                           \
      device DTYPE * input [[buffer(0)]],                                 \
      device DTYPE * output [[buffer(1)]],                                \
      device DTYPE * meanOut [[buffer(2)]],                               \
      device DTYPE * rstdTensor [[buffer(3)]],                            \
      constant uint & axis_size [[buffer(4)]],                            \
      constant float& epsilon [[buffer(5)]],                              \
      constant int& use_weight [[buffer(6)]],                             \
      constant int& use_bias [[buffer(7)]],                               \
      device DTYPE* weight [[buffer(8)]],                                 \
      device DTYPE* bias [[buffer(9)]],                                   \
      uint tg_id [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_position_in_threadgroup]],                        \
      uint simd_lane_id [[thread_index_in_simdgroup]],                    \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_layer_norm_looped(DTYPE)                          \
  template [[host_name("layer_norm_looped_" #DTYPE)]] [[kernel]] void \
  layer_norm_looped<DTYPE>(                                           \
      device DTYPE * input [[buffer(0)]],                             \
      device DTYPE * output [[buffer(1)]],                            \
      device DTYPE * meanOut [[buffer(2)]],                           \
      device DTYPE * rstdTensor [[buffer(3)]],                        \
      constant uint & axis_size [[buffer(4)]],                        \
      constant float& epsilon [[buffer(5)]],                          \
      constant int& use_weight [[buffer(6)]],                         \
      constant int& use_bias [[buffer(7)]],                           \
      device DTYPE* weight [[buffer(8)]],                             \
      device DTYPE* bias [[buffer(9)]],                               \
      uint tg_id [[threadgroup_position_in_grid]],                    \
      uint tid [[thread_position_in_threadgroup]],                    \
      uint lsize [[threads_per_threadgroup]],                         \
      uint simd_lane_id [[thread_index_in_simdgroup]],                \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_layer_norm(DTYPE) \
  instantiate_layer_norm_single_row(DTYPE) instantiate_layer_norm_looped(DTYPE)

instantiate_layer_norm(float) instantiate_layer_norm(half)
#if __METAL_VERSION__ >= 310
    instantiate_layer_norm(bfloat)
#endif
