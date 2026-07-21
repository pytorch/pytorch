#include <c10/metal/common.h>
#include <c10/metal/reduction_utils.h>
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

template <typename T, typename idx_t>
kernel void layer_norm_single_row(
    device T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device T* meanOut [[buffer(2)]],
    device T* rstdTensor [[buffer(3)]],
    constant idx_t& axis_size [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    constant int& use_weight [[buffer(6)]],
    constant int& use_bias [[buffer(7)]],
    device T* weight [[buffer(8)]],
    device T* bias [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int N_READS = 4;

  // Each threadgroup handles one full row of length axis_size, and each thread
  // owns exactly its N_READS-element slice. The slice is loaded once and reused
  // for the variance pass and the write-back
  idx_t row_offset = idx_t(tg_id) * axis_size;
  device T* x = input + row_offset + tid * N_READS;
  device T* out = output + row_offset + tid * N_READS;

  uint base_lane = tid * N_READS;
  int count = min((int)N_READS, max(0, (int)axis_size - (int)base_lane));
  float vals[N_READS];
#pragma unroll
  for (int i = 0; i < N_READS; i++) {
    vals[i] = (i < count) ? float(x[i]) : 0.0f;
  }

  threadgroup float local_sum[simdgroup_size];
  threadgroup float local_sum_sq[simdgroup_size];

  float partial_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < N_READS; i++) {
    partial_sum += vals[i];
  }
  float mean =
      threadgroup_sum(local_sum, partial_sum, tid, lsize) / float(axis_size);

  // Two-pass variance from the centered values avoids the catastrophic
  // cancellation of E[x^2]-E[x]^2 for small variances.
  float partial_sum_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < N_READS; i++) {
    float d = (i < count) ? (vals[i] - mean) : 0.0f;
    partial_sum_sq += d * d;
  }
  float var = threadgroup_sum(local_sum_sq, partial_sum_sq, tid, lsize) /
      float(axis_size);
  float inv_std = metal::precise::rsqrt(var + epsilon);

#pragma unroll
  for (int i = 0; i < N_READS; i++) {
    if (i < count) {
      float norm = (vals[i] - mean) * inv_std;
      uint lane_idx = base_lane + i;
      if (use_weight)
        norm *= float(weight[lane_idx]);
      if (use_bias)
        norm += float(bias[lane_idx]);
      out[i] = static_cast<T>(norm);
    }
  }

  if (tid == 0 && simd_lane_id == 0) {
    meanOut[tg_id] = static_cast<T>(mean);
    rstdTensor[tg_id] = static_cast<T>(inv_std);
  }
}

template <typename T, typename idx_t>
kernel void layer_norm_looped(
    device T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device T* meanOut [[buffer(2)]],
    device T* rstdTensor [[buffer(3)]],
    constant idx_t& axis_size [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    constant int& use_weight [[buffer(6)]],
    constant int& use_bias [[buffer(7)]],
    device T* weight [[buffer(8)]],
    device T* bias [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int N_READS = 4;

  idx_t row_offset = idx_t(tg_id) * axis_size;
  device T* x = input + row_offset;
  device T* out = output + row_offset;

  // This kernel is used when axis_size > 1024 * N_READS, so the row cannot be
  // cached in registers. The mean and variance passes each read it from global
  // memory. Two passes keep it accurate for small variances
  threadgroup float local_sum[simdgroup_size];
  threadgroup float local_sum_sq[simdgroup_size];

  float partial_sum = 0.0f;
  for (idx_t r = 0; r < axis_size; r += lsize * N_READS) {
    idx_t base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 v4 = float4(x[base], x[base + 1], x[base + 2], x[base + 3]);
      partial_sum += v4.x + v4.y + v4.z + v4.w;
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++) {
        if (base + i < axis_size) {
          partial_sum += float(x[base + i]);
        }
      }
    }
  }
  float mean =
      threadgroup_sum(local_sum, partial_sum, tid, lsize) / float(axis_size);

  float partial_sum_sq = 0.0f;
  for (idx_t r = 0; r < axis_size; r += lsize * N_READS) {
    idx_t base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 d4 = float4(x[base], x[base + 1], x[base + 2], x[base + 3]) - mean;
      partial_sum_sq += dot(d4, d4);
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++) {
        if (base + i < axis_size) {
          float d = float(x[base + i]) - mean;
          partial_sum_sq += d * d;
        }
      }
    }
  }
  float var = threadgroup_sum(local_sum_sq, partial_sum_sq, tid, lsize) /
      float(axis_size);
  float inv_std = metal::precise::rsqrt(var + epsilon);

  for (idx_t r = 0; r < axis_size; r += lsize * N_READS) {
    idx_t base = r + tid * N_READS;
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

#define instantiate_layer_norm_single_row(DTYPE, IDX_T, IDXNAME) \
  template [[host_name("layer_norm_single_row_" IDXNAME          \
                       "_" #DTYPE)]] [[kernel]] void             \
  layer_norm_single_row<DTYPE, IDX_T>(                           \
      device DTYPE * input [[buffer(0)]],                        \
      device DTYPE * output [[buffer(1)]],                       \
      device DTYPE * meanOut [[buffer(2)]],                      \
      device DTYPE * rstdTensor [[buffer(3)]],                   \
      constant IDX_T & axis_size [[buffer(4)]],                  \
      constant float& epsilon [[buffer(5)]],                     \
      constant int& use_weight [[buffer(6)]],                    \
      constant int& use_bias [[buffer(7)]],                      \
      device DTYPE* weight [[buffer(8)]],                        \
      device DTYPE* bias [[buffer(9)]],                          \
      uint tg_id [[threadgroup_position_in_grid]],               \
      uint tid [[thread_position_in_threadgroup]],               \
      uint lsize [[threads_per_threadgroup]],                    \
      uint simd_lane_id [[thread_index_in_simdgroup]]);

#define instantiate_layer_norm_looped(DTYPE, IDX_T, IDXNAME)                 \
  template                                                                   \
      [[host_name("layer_norm_looped_" IDXNAME "_" #DTYPE)]] [[kernel]] void \
      layer_norm_looped<DTYPE, IDX_T>(                                       \
          device DTYPE * input [[buffer(0)]],                                \
          device DTYPE * output [[buffer(1)]],                               \
          device DTYPE * meanOut [[buffer(2)]],                              \
          device DTYPE * rstdTensor [[buffer(3)]],                           \
          constant IDX_T & axis_size [[buffer(4)]],                          \
          constant float& epsilon [[buffer(5)]],                             \
          constant int& use_weight [[buffer(6)]],                            \
          constant int& use_bias [[buffer(7)]],                              \
          device DTYPE* weight [[buffer(8)]],                                \
          device DTYPE* bias [[buffer(9)]],                                  \
          uint tg_id [[threadgroup_position_in_grid]],                       \
          uint tid [[thread_position_in_threadgroup]],                       \
          uint lsize [[threads_per_threadgroup]],                            \
          uint simd_lane_id [[thread_index_in_simdgroup]]);

#define instantiate_layer_norm(DTYPE)                        \
  instantiate_layer_norm_single_row(DTYPE, uint, "i32")      \
      instantiate_layer_norm_single_row(DTYPE, ulong, "i64") \
          instantiate_layer_norm_looped(DTYPE, uint, "i32")  \
              instantiate_layer_norm_looped(DTYPE, ulong, "i64")

instantiate_layer_norm(float);
instantiate_layer_norm(half);
instantiate_layer_norm(bfloat);
