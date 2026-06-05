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
inline float load_affine_scale(constant void*, uint) {
  return 1;
}

template <typename T>
inline float load_affine_bias(constant T* ptr, uint idx) {
  return float(ptr[idx]);
}

template <>
inline float load_affine_bias(constant void*, uint) {
  return 0;
}

template <
    typename T,
    typename stat_T,
    typename gamma_T,
    typename beta_T,
    typename idx_T>
kernel void group_norm(
    device T* Y [[buffer(0)]],
    device stat_T* mean [[buffer(1)]],
    device stat_T* rstd [[buffer(2)]],
    constant T* X [[buffer(3)]],
    constant gamma_T* gamma [[buffer(4)]],
    constant beta_T* beta [[buffer(5)]],
    constant GroupNormParams<idx_T>& params [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  idx_T group_offset = tgid * params.elements_per_group;
  constant T* x = X + group_offset;
  device T* y = Y + group_offset;

  // Divide the elements of the group between each thread in the threadgroup.
  // First, each thread reads all of the elements assigned to it and computes a
  // partial sum and sum of squares of those elements.
  float partial_sum = 0;
  float partial_sum_sq = 0;

  for (idx_T r = 0; r < params.elements_per_group; r += tptg * BLOCK_SIZE) {
    auto base = r + tid * BLOCK_SIZE;
#pragma unroll
    for (idx_T i = 0; i < BLOCK_SIZE; i++) {
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
  idx_T channel_base = (tgid % params.num_groups) * params.channels_per_group;

  for (idx_T r = 0; r < params.elements_per_group; r += tptg * BLOCK_SIZE) {
    idx_T base = r + tid * BLOCK_SIZE;
#pragma unroll
    for (idx_T i = 0; i < BLOCK_SIZE; i++) {
      idx_T elem = base + i;
      if (elem < params.elements_per_group) {
        idx_T channel = channel_base + elem / params.HxW;
        float norm = (float(x[elem]) - mean_val) * rstd_val;
        norm = norm * load_affine_scale(gamma, channel) +
            load_affine_bias(beta, channel);
        y[elem] = T(norm);
      }
    }
  }

  // Finally, one thread in the group writes the mean and rstd outputs.
  if (tid == 0) {
    mean[tgid] = stat_T(mean_val);
    rstd[tgid] = stat_T(rstd_val);
  }
}

#define REGISTER_GROUP_NORM(T, stat_T, gamma_T, beta_T, idx_T)               \
  template [[host_name("group_norm_" #T "_" #stat_T "_" #gamma_T "_" #beta_T \
                       "_" #idx_T)]]                                         \
  kernel void group_norm<T, stat_T, gamma_T, beta_T, idx_T>(                 \
      device T * Y [[buffer(0)]],                                            \
      device stat_T * mean [[buffer(1)]],                                    \
      device stat_T * rstd [[buffer(2)]],                                    \
      constant T * X [[buffer(3)]],                                          \
      constant gamma_T * gamma [[buffer(4)]],                                \
      constant beta_T * beta [[buffer(5)]],                                  \
      constant GroupNormParams<idx_T> & params [[buffer(6)]],                \
      uint tg_id [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_position_in_threadgroup]],                           \
      uint tptg [[threads_per_threadgroup]]);

#define REGISTER_GROUP_NORM_IDX_TYPES(T, T_stat, gamma_T, beta_T) \
  REGISTER_GROUP_NORM(T, T_stat, gamma_T, beta_T, uint32_t);      \
  REGISTER_GROUP_NORM(T, T_stat, gamma_T, beta_T, uint64_t);

#define REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, T_stat, affine_T) \
  REGISTER_GROUP_NORM_IDX_TYPES(T, T_stat, affine_T, affine_T);     \
  REGISTER_GROUP_NORM_IDX_TYPES(T, T_stat, affine_T, void);         \
  REGISTER_GROUP_NORM_IDX_TYPES(T, T_stat, void, affine_T);

#define REGISTER_GROUP_NORM_AFFINE_TYPES(T, T_stat)          \
  REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, T_stat, float);  \
  REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, T_stat, half);   \
  REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, T_stat, bfloat); \
  REGISTER_GROUP_NORM_IDX_TYPES(T, T_stat, void, void);

REGISTER_GROUP_NORM_AFFINE_TYPES(float, float);
REGISTER_GROUP_NORM_AFFINE_TYPES(half, float);
REGISTER_GROUP_NORM_AFFINE_TYPES(bfloat, float);

REGISTER_GROUP_NORM_AFFINE_TYPES(half, half);
REGISTER_GROUP_NORM_AFFINE_TYPES(bfloat, bfloat);

template <typename T, typename stat_T, typename gamma_T, typename idx_T>
kernel void group_norm_backward_x(
    device T* dX [[buffer(0)]],
    constant T* dY [[buffer(1)]],
    constant T* X [[buffer(2)]],
    constant stat_T* mean [[buffer(3)]],
    constant stat_T* rstd [[buffer(4)]],
    constant gamma_T* gamma [[buffer(5)]],
    constant GroupNormParams<idx_T>& params [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  idx_T group_offset = tgid * params.elements_per_group;
  constant T* x = X + group_offset;
  constant T* dy = dY + group_offset;
  device T* dx = dX + group_offset;

  auto mean_val = float(mean[tgid]);
  auto rstd_val = float(rstd[tgid]);
  idx_T channel_base = (tgid % params.num_groups) * params.channels_per_group;

  // Accumulate `ds = sum(dY * gamma * X)` and `db = sum(dY * gamma)` over all
  // elements in the group.
  float partial_ds = 0;
  float partial_db = 0;
  for (idx_T r = 0; r < params.elements_per_group; r += tptg * BLOCK_SIZE) {
    idx_T base = r + tid * BLOCK_SIZE;
#pragma unroll
    for (idx_T i = 0; i < BLOCK_SIZE; i++) {
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
  for (idx_T r = 0; r < params.elements_per_group; r += tptg * BLOCK_SIZE) {
    idx_T base = r + tid * BLOCK_SIZE;
#pragma unroll
    for (idx_T i = 0; i < BLOCK_SIZE; i++) {
      idx_T elem = base + i;
      if (elem < params.elements_per_group) {
        auto c1 = rstd_val *
            load_affine_scale(gamma, channel_base + elem / params.HxW);
        dx[elem] = T(c1 * float(dy[elem]) + c2 * float(x[elem]) + c3);
      }
    }
  }
}

#define REGISTER_GROUP_NORM_BACKWARD_X(T, stat_T, gamma_T, idx_T)           \
  template [[host_name("group_norm_backward_x_" #T "_" #stat_T "_" #gamma_T \
                       "_" #idx_T)]]                                        \
  kernel void group_norm_backward_x<T, stat_T, gamma_T, idx_T>(             \
      device T * dX [[buffer(0)]],                                          \
      constant T * dY [[buffer(1)]],                                        \
      constant T * X [[buffer(2)]],                                         \
      constant stat_T * mean [[buffer(3)]],                                 \
      constant stat_T * rstd [[buffer(4)]],                                 \
      constant gamma_T * gamma [[buffer(5)]],                               \
      constant GroupNormParams<idx_T> & params [[buffer(6)]],               \
      uint tgid [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_position_in_threadgroup]],                          \
      uint tptg [[threads_per_threadgroup]]);

#define REGISTER_GROUP_NORM_BACKWARD_INDEX_TYPES(T, stat_T, gamma_T) \
  REGISTER_GROUP_NORM_BACKWARD_X(T, stat_T, gamma_T, uint32_t);      \
  REGISTER_GROUP_NORM_BACKWARD_X(T, stat_T, gamma_T, uint64_t);

#define REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(T, stat_T)    \
  REGISTER_GROUP_NORM_BACKWARD_INDEX_TYPES(T, stat_T, float);  \
  REGISTER_GROUP_NORM_BACKWARD_INDEX_TYPES(T, stat_T, half);   \
  REGISTER_GROUP_NORM_BACKWARD_INDEX_TYPES(T, stat_T, bfloat); \
  REGISTER_GROUP_NORM_BACKWARD_INDEX_TYPES(T, stat_T, void);

REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(float, float);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(half, float);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(bfloat, float);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(half, half);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(bfloat, bfloat);

template <typename T, typename stat_T, typename affine_T, typename idx_T>
kernel void group_norm_backward_affine(
    device affine_T* dgamma [[buffer(0)]],
    device affine_T* dbeta [[buffer(1)]],
    constant T* dY [[buffer(2)]],
    constant T* X [[buffer(3)]],
    constant stat_T* mean [[buffer(4)]],
    constant stat_T* rstd [[buffer(5)]],
    constant GroupNormParams<idx_T>& params [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  // One threadgroup per channel
  idx_T channel = tgid;
  idx_T group = channel / params.channels_per_group;

  // Accumulate `dg = sum(dy * (x - mean) * rstd)` and `db = sum(dy)` over all
  // dimensions except the channel dimension.
  float partial_dg = 0;
  float partial_db = 0;
  for (idx_T r = 0; r < params.N_times_HxW;
       r += idx_T(tptg) * idx_T(BLOCK_SIZE)) {
    idx_T base = r + idx_T(tid) * idx_T(BLOCK_SIZE);
#pragma unroll
    for (idx_T i = 0; i < idx_T(BLOCK_SIZE); i++) {
      idx_T elem = base + i;
      if (elem < params.N_times_HxW) {
        idx_T batch = elem / params.HxW;
        idx_T ng = batch * params.num_groups + group;
        idx_T idx = batch * params.C * params.HxW + channel * params.HxW +
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

#define REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, stat_T, affine_T, idx_T) \
  template [[host_name("group_norm_backward_affine_" #T "_" #stat_T     \
                       "_" #affine_T "_" #idx_T)]]                      \
  kernel void group_norm_backward_affine<T, stat_T, affine_T, idx_T>(   \
      device affine_T * dgamma [[buffer(0)]],                           \
      device affine_T * dbeta [[buffer(1)]],                            \
      constant T * dY [[buffer(2)]],                                    \
      constant T * X [[buffer(3)]],                                     \
      constant stat_T * mean [[buffer(4)]],                             \
      constant stat_T * rstd [[buffer(5)]],                             \
      constant GroupNormParams<idx_T> & params [[buffer(6)]],           \
      uint tgid [[threadgroup_position_in_grid]],                       \
      uint tid [[thread_position_in_threadgroup]],                      \
      uint tptg [[threads_per_threadgroup]]);

#define REGISTER_GROUP_NORM_BACKWARD_AFFINE_INDEX_TYPES(T, stat_T, affine_T) \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, stat_T, affine_T, uint32_t);        \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, stat_T, affine_T, uint64_t);

#define REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(T, stat_T)         \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE_INDEX_TYPES(T, stat_T, float); \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE_INDEX_TYPES(T, stat_T, half);  \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE_INDEX_TYPES(T, stat_T, bfloat);

REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(float, float);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(half, float);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(bfloat, float);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(half, half);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(bfloat, bfloat);
