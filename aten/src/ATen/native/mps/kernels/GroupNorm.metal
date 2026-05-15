#include <ATen/native/mps/kernels/GroupNorm.h>
#include <c10/metal/utils.h>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <typename T>
inline float load_affine_scale(constant T* ptr, uint idx) {
  return float(ptr[idx]);
}

template <>
inline float load_affine_scale(constant void* ptr, uint idx) {
  return 1.0f;
}

template <typename T>
inline float load_affine_bias(constant T* ptr, uint idx) {
  return float(ptr[idx]);
}

template <>
inline float load_affine_bias(constant void* ptr, uint idx) {
  return 0.0f;
}

template <typename T, typename affine_T>
kernel void group_norm(
    device T* Y [[buffer(0)]],
    device T* mean [[buffer(1)]],
    device T* rstd [[buffer(2)]],
    constant T* X [[buffer(3)]],
    constant affine_T* gamma [[buffer(4)]],
    constant affine_T* beta [[buffer(5)]],
    constant GroupNormParams& params [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  uint32_t group_offset = tgid * params.elements_per_group;
  constant T* x = X + group_offset;
  device T* y = Y + group_offset;

  // Divide the elements of the group between each thread in the threadgroup.
  // First, a thread reads all of the elements assigned to it and computes a
  // partial sum and sum of squares of those elements.
  float partial_sum = 0;
  float partial_sum_sq = 0;

  for (uint32_t r = 0; r < params.elements_per_group; r += tptg * BLOCK_SIZE) {
    auto base = r + tid * BLOCK_SIZE;
#pragma unroll
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
      if (base + i < params.elements_per_group) {
        auto v = static_cast<float>(x[base + i]);
        partial_sum += v;
        partial_sum_sq = fma(v, v, partial_sum_sq);
      }
    }
  }

  // Second, the threads in each simdgroup sum their partial sums.
  threadgroup float local_sums[simdgroup_size];
  threadgroup float local_sums_sq[simdgroup_size];
  threadgroup float tg_mean_val[1];
  threadgroup float tg_rstd_val[1];

  partial_sum = simd_sum(partial_sum);
  partial_sum_sq = simd_sum(partial_sum_sq);
  if (simd_lane_id == 0) {
    local_sums[simdgroup_id] = partial_sum;
    local_sums_sq[simdgroup_id] = partial_sum_sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Third, the simdgroups within a threadgroup sum their partial sums to obtain
  // the total sum and sum of squares of all the elements in the group. From
  // those sums, one thread in the threadgroup computes the mean and rstd for
  // the group.
  if (simdgroup_id == 0) {
    float sum = simd_sum(local_sums[simd_lane_id]);
    float sum_sq = simd_sum(local_sums_sq[simd_lane_id]);
    if (simd_lane_id == 0) {
      float m = float(params.elements_per_group);
      float mean = sum / m;
      // Variance of an array `a` is `sum((a - mean(a))**2) / m`, which we
      // factor out into `sum(a**2) / m - 2 * mean(a) * sum(a) / m + mean(a)**2`
      // so that both the mean and variance can be calculated in one pass.
      float var = (sum_sq - 2 * mean * sum) / m + mean * mean;
      tg_mean_val[0] = mean;
      tg_rstd_val[0] = metal::precise::rsqrt(var + params.eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float mean_val = tg_mean_val[0];
  float rstd_val = tg_rstd_val[0];

  // Fourth, each thread reads its assigned input elements again, applies the
  // normalization and affine transform, and writes the results to the output.
  uint32_t channel_base =
      (tgid % params.num_groups) * params.channels_per_group;

  for (uint32_t r = 0; r < params.elements_per_group; r += tptg * BLOCK_SIZE) {
    uint32_t base = r + tid * BLOCK_SIZE;
#pragma unroll
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
      uint32_t elem = base + i;
      if (elem < params.elements_per_group) {
        uint32_t channel = channel_base + elem / params.HxW;
        float norm = (float(x[elem]) - mean_val) * rstd_val;
        norm = norm * load_affine_scale(gamma, channel) +
            load_affine_bias(beta, channel);
        y[elem] = T(norm);
      }
    }
  }

  // Finally, one thread in the group writes the mean and rstd outputs.
  if (tid == 0) {
    mean[tgid] = T(mean_val);
    rstd[tgid] = T(rstd_val);
  }
}

#define REGISTER_GROUP_NORM(T, affine_T)                 \
  template [[host_name("group_norm_" #T "_" #affine_T)]] \
  kernel void group_norm<T, affine_T>(                   \
      device T * Y [[buffer(0)]],                        \
      device T * mean [[buffer(1)]],                     \
      device T * rstd [[buffer(2)]],                     \
      constant T * X [[buffer(3)]],                      \
      constant affine_T * gamma [[buffer(4)]],           \
      constant affine_T * beta [[buffer(5)]],            \
      constant GroupNormParams & params [[buffer(6)]],   \
      uint tg_id [[threadgroup_position_in_grid]],       \
      uint tid [[thread_position_in_threadgroup]],       \
      uint tptg [[threads_per_threadgroup]],             \
      uint simd_lane_id [[thread_index_in_simdgroup]],   \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define REGISTER_GROUP_NORM_AFFINE_TYPES(T) \
  REGISTER_GROUP_NORM(T, float);            \
  REGISTER_GROUP_NORM(T, half);             \
  REGISTER_GROUP_NORM(T, bfloat);           \
  REGISTER_GROUP_NORM(T, uchar);            \
  REGISTER_GROUP_NORM(T, char);             \
  REGISTER_GROUP_NORM(T, short);            \
  REGISTER_GROUP_NORM(T, int);              \
  REGISTER_GROUP_NORM(T, void);

REGISTER_GROUP_NORM_AFFINE_TYPES(float);
REGISTER_GROUP_NORM_AFFINE_TYPES(half);
REGISTER_GROUP_NORM_AFFINE_TYPES(bfloat);
REGISTER_GROUP_NORM_AFFINE_TYPES(uchar);
REGISTER_GROUP_NORM_AFFINE_TYPES(char);
REGISTER_GROUP_NORM_AFFINE_TYPES(short);
REGISTER_GROUP_NORM_AFFINE_TYPES(int);
