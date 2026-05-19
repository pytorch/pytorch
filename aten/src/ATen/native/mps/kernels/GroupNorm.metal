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
    uint tptg [[threads_per_threadgroup]]) {
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
      uint tptg [[threads_per_threadgroup]]);

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

template <typename T, typename gamma_T>
kernel void group_norm_backward_x(
    device T* dX [[buffer(0)]],
    constant T* dY [[buffer(1)]],
    constant T* X [[buffer(2)]],
    constant T* mean [[buffer(3)]],
    constant T* rstd [[buffer(4)]],
    constant gamma_T* gamma [[buffer(5)]],
    constant GroupNormBackwardXParams& params [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  uint32_t group_offset = tgid * params.elements_per_group;
  constant T* x = X + group_offset;
  constant T* dy = dY + group_offset;
  device T* dx = dX + group_offset;

  auto mean_val = float(mean[tgid]);
  auto rstd_val = float(rstd[tgid]);
  uint32_t channel_base =
      (tgid % params.num_groups) * params.channels_per_group;

  // Accumulate `ds = sum(dY * gamma * X)` and `db = sum(dY * gamma)` over all
  // elements in the group.
  float partial_ds = 0;
  float partial_db = 0;
  for (uint32_t r = 0; r < params.elements_per_group; r += tptg * BLOCK_SIZE) {
    uint32_t base = r + tid * BLOCK_SIZE;
#pragma unroll
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
      auto elem = base + i;
      if (elem < params.elements_per_group) {
        auto gamma_val =
            load_affine_scale(gamma, channel_base + elem / params.HxW);
        auto dy_val = float(dy[elem]);
        auto x_val = float(x[elem]);
        auto dy_gamma = dy_val * gamma_val;
        partial_ds += dy_gamma * x_val;
        partial_db += dy_gamma;
      }
    }
  }

  // Reduce ds and db across the threadgroup.
  threadgroup float local_ds[simdgroup_size];
  threadgroup float local_db[simdgroup_size];
  auto reduction =
      threadgroup_sum2(local_ds, local_db, partial_ds, partial_db, tid, tptg);
  auto ds_val = reduction[0];
  auto db_val = reduction[1];

  // Compute per-group coefficients
  auto m = float(params.elements_per_group);
  auto c2 = (db_val * mean_val - ds_val) * rstd_val * rstd_val * rstd_val / m;
  auto c3 = -c2 * mean_val - db_val * rstd_val / m;

  // Write dX.
  for (uint32_t r = 0; r < params.elements_per_group; r += tptg * BLOCK_SIZE) {
    uint32_t base = r + tid * BLOCK_SIZE;
#pragma unroll
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
      uint32_t elem = base + i;
      if (elem < params.elements_per_group) {
        auto c1 = rstd_val *
            load_affine_scale(gamma, channel_base + elem / params.HxW);
        dx[elem] = T(c1 * float(dy[elem]) + c2 * float(x[elem]) + c3);
      }
    }
  }
}

#define REGISTER_GROUP_NORM_BACKWARD_X(T, gamma_T)                 \
  template [[host_name("group_norm_backward_x_" #T "_" #gamma_T)]] \
  kernel void group_norm_backward_x<T, gamma_T>(                   \
      device T * dX [[buffer(0)]],                                 \
      constant T * dY [[buffer(1)]],                               \
      constant T * X [[buffer(2)]],                                \
      constant T * mean [[buffer(3)]],                             \
      constant T * rstd [[buffer(4)]],                             \
      constant gamma_T * gamma [[buffer(5)]],                      \
      constant GroupNormBackwardXParams & params [[buffer(6)]],    \
      uint tgid [[threadgroup_position_in_grid]],                  \
      uint tid [[thread_position_in_threadgroup]],                 \
      uint tptg [[threads_per_threadgroup]]);

#define REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(T) \
  REGISTER_GROUP_NORM_BACKWARD_X(T, float);         \
  REGISTER_GROUP_NORM_BACKWARD_X(T, half);          \
  REGISTER_GROUP_NORM_BACKWARD_X(T, bfloat);        \
  REGISTER_GROUP_NORM_BACKWARD_X(T, uchar);         \
  REGISTER_GROUP_NORM_BACKWARD_X(T, char);          \
  REGISTER_GROUP_NORM_BACKWARD_X(T, short);         \
  REGISTER_GROUP_NORM_BACKWARD_X(T, int);           \
  REGISTER_GROUP_NORM_BACKWARD_X(T, void);

REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(float);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(half);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(bfloat);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(uchar);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(char);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(short);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(int);

template <typename T, typename affine_T>
kernel void group_norm_backward_affine(
    device affine_T* dgamma [[buffer(0)]],
    device affine_T* dbeta [[buffer(1)]],
    constant T* dY [[buffer(2)]],
    constant T* X [[buffer(3)]],
    constant T* mean [[buffer(4)]],
    constant T* rstd [[buffer(5)]],
    constant GroupNormBackwardAffineParams& params [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  // One threadgroup per channel
  uint32_t channel = tgid;
  uint32_t group = channel / params.channels_per_group;

  // Accumulate `dg = sum(dy * (x - mean) * rstd)` and `db = sum(dy)` over all
  // dimensions except the channel dimension.
  float partial_dg = 0;
  float partial_db = 0;
  for (uint32_t r = 0; r < params.N_times_HxW; r += tptg * BLOCK_SIZE) {
    uint32_t base = r + tid * BLOCK_SIZE;
#pragma unroll
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
      uint32_t elem = base + i;
      if (elem < params.N_times_HxW) {
        uint32_t batch = elem / params.HxW;
        uint32_t ng = batch * params.num_groups + group;
        uint32_t idx = batch * params.C * params.HxW + channel * params.HxW +
            elem % params.HxW;
        auto dy_val = float(dY[idx]);
        partial_dg +=
            dy_val * (float(X[idx]) - float(mean[ng])) * float(rstd[ng]);
        partial_db += dy_val;
      }
    }
  }

  threadgroup float local_dg[simdgroup_size];
  threadgroup float local_db[simdgroup_size];
  auto reduction =
      threadgroup_sum2(local_dg, local_db, partial_dg, partial_db, tid, tptg);

  if (tid == 0) {
    dgamma[channel] = affine_T(reduction[0]);
    dbeta[channel] = affine_T(reduction[1]);
  }
}

#define REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, affine_T)                 \
  template [[host_name("group_norm_backward_affine_" #T "_" #affine_T)]] \
  kernel void group_norm_backward_affine<T, affine_T>(                   \
      device affine_T * dgamma [[buffer(0)]],                            \
      device affine_T * dbeta [[buffer(1)]],                             \
      constant T * dY [[buffer(2)]],                                     \
      constant T * X [[buffer(3)]],                                      \
      constant T * mean [[buffer(4)]],                                   \
      constant T * rstd [[buffer(5)]],                                   \
      constant GroupNormBackwardAffineParams & params [[buffer(6)]],     \
      uint tgid [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_position_in_threadgroup]],                       \
      uint tptg [[threads_per_threadgroup]]);

#define REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(T) \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, float);     \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, half);      \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, bfloat);    \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, uchar);     \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, char);      \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, short);     \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, int);

REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(float);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(half);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(bfloat);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(uchar);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(char);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(short);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(int);
