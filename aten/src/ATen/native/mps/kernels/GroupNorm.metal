#include <ATen/native/mps/kernels/GroupNorm.h>
#include <c10/metal/reduction_utils.h>
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
  return 1;
}

template <typename T>
inline float load_affine_bias(constant T* ptr, uint idx) {
  return float(ptr[idx]);
}

template <>
inline float load_affine_bias(constant void* ptr, uint idx) {
  return 0;
}

template <typename T, typename gamma_T, typename beta_T>
kernel void group_norm(
    device T* Y [[buffer(0)]],
    device T* mean [[buffer(1)]],
    device T* rstd [[buffer(2)]],
    constant T* X [[buffer(3)]],
    constant gamma_T* gamma [[buffer(4)]],
    constant beta_T* beta [[buffer(5)]],
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
  // First, each thread reads all of the elements assigned to it and computes a
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

  // Second, sum all the partial sums of the threads in the threadgroup and
  // calculate the mean and rstd.
  threadgroup float local_sums[simdgroup_size];
  threadgroup float local_sums_sq[simdgroup_size];

  auto reduction_result = threadgroup_sum2(
      local_sums, local_sums_sq, partial_sum, partial_sum_sq, tid, tptg);

  float sum_val = reduction_result[0];
  float sum_sq_val = reduction_result[1];
  float m = float(params.elements_per_group);
  float mean_val = sum_val / m;
  // Variance of an array `a` is `sum((a - mean(a))**2) / m`, which we
  // factor out into `sum(a**2) / m - 2 * mean(a) * sum(a) / m + mean(a)**2`
  // so that both the mean and variance can be calculated in one pass.
  float var_val =
      (sum_sq_val - 2 * mean_val * sum_val) / m + mean_val * mean_val;
  float rstd_val = metal::precise::rsqrt(var_val + params.eps);

  // Third, each thread reads its assigned input elements again, applies the
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

#define REGISTER_GROUP_NORM(T, gamma_T, beta_T)                     \
  template [[host_name("group_norm_" #T "_" #gamma_T "_" #beta_T)]] \
  kernel void group_norm<T, gamma_T, beta_T>(                       \
      device T * Y [[buffer(0)]],                                   \
      device T * mean [[buffer(1)]],                                \
      device T * rstd [[buffer(2)]],                                \
      constant T * X [[buffer(3)]],                                 \
      constant gamma_T * gamma [[buffer(4)]],                       \
      constant beta_T * beta [[buffer(5)]],                         \
      constant GroupNormParams & params [[buffer(6)]],              \
      uint tg_id [[threadgroup_position_in_grid]],                  \
      uint tid [[thread_position_in_threadgroup]],                  \
      uint tptg [[threads_per_threadgroup]],                        \
      uint simd_lane_id [[thread_index_in_simdgroup]],              \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, affine_T) \
  REGISTER_GROUP_NORM(T, affine_T, affine_T);               \
  REGISTER_GROUP_NORM(T, affine_T, void);                   \
  REGISTER_GROUP_NORM(T, void, affine_T);

#define REGISTER_GROUP_NORM_AFFINE_TYPES(T)          \
  REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, float);  \
  REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, half);   \
  REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, bfloat); \
  REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, uchar);  \
  REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, char);   \
  REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, short);  \
  REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, int);    \
  REGISTER_GROUP_NORM(T, void, void);

REGISTER_GROUP_NORM_AFFINE_TYPES(float);
REGISTER_GROUP_NORM_AFFINE_TYPES(half);
REGISTER_GROUP_NORM_AFFINE_TYPES(bfloat);
REGISTER_GROUP_NORM_AFFINE_TYPES(uchar);
REGISTER_GROUP_NORM_AFFINE_TYPES(char);
REGISTER_GROUP_NORM_AFFINE_TYPES(short);
REGISTER_GROUP_NORM_AFFINE_TYPES(int);
