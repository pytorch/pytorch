// Adapted from
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rms_norm.metal
// Copyright © 2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

template <typename T>
[[kernel]] void rms_single_row(
    constant T* x,
    constant T* w,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  constexpr int N_READS = 4;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      float xi = x[i];
      acc += xi * xi;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        float xi = x[i];
        acc += xi * xi;
      }
    }
  }
  acc = simd_sum(acc);
  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write the outputs
  out += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = w[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = w[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
      }
    }
  }
}

template <typename T>
[[kernel]] void rms_looped(
    constant T* x,
    constant T* w,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  constexpr int N_READS = 4;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        acc += xi * xi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          acc += xi * xi;
        }
      }
    }
  }
  acc = simd_sum(acc);
  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write the outputs
  out += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        out[r + i] = w[w_stride * (i + r)] *
            static_cast<T>(x[r + i] * local_inv_mean[0]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          out[r + i] = w[w_stride * (i + r)] *
              static_cast<T>(x[r + i] * local_inv_mean[0]);
        }
      }
    }
  }
}

// clang-format off
#define instantiate_rms_single_row(itype)                     \
  template [[host_name("rms_norm_" #itype)]] [[kernel]] void  \
  rms_single_row<itype>(                                      \
      constant itype* x,                                      \
      constant itype* w,                                      \
      device itype* out,                                      \
      constant float& eps,                                    \
      constant uint& axis_size,                               \
      constant uint& w_stride,                                \
      uint gid [[thread_position_in_grid]],                   \
      uint lid [[thread_position_in_threadgroup]],            \
      uint simd_lane_id [[thread_index_in_simdgroup]],        \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_rms_looped(itype)                               \
  template [[host_name("rms_norm_looped_" #itype)]] [[kernel]] void \
  rms_looped<itype>(                                                \
      constant itype* x,                                            \
      constant itype* w,                                            \
      device itype* out,                                            \
      constant float& eps,                                          \
      constant uint& axis_size,                                     \
      constant uint& w_stride,                                      \
      uint gid [[thread_position_in_grid]],                         \
      uint lid [[thread_position_in_threadgroup]],                  \
      uint lsize [[threads_per_threadgroup]],                       \
      uint simd_lane_id [[thread_index_in_simdgroup]],              \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_rms(itype)      \
  instantiate_rms_single_row(itype) \
  instantiate_rms_looped(itype)

instantiate_rms(float)
instantiate_rms(half)
#if __METAL_VERSION__ >= 310
instantiate_rms(bfloat)
#endif // clang-format on
