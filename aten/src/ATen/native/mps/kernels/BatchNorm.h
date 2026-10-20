#pragma once
#include <c10/metal/common.h>

C10_METAL_CONSTEXPR uint32_t BN_MAX_THREADS = 1024;
// Per-channel Welford partial (mean, M2, count), stored as packed_float3
C10_METAL_CONSTEXPR uint32_t BN_PARTIAL_SIZE = 3;
enum { BN_MEAN, BN_SCALE, BN_SHIFT, BN_DY_SCALE, BN_NUM_STATS };
enum {
  BN_WEIGHT,
  BN_BIAS,
  BN_RUNNING_MEAN,
  BN_RUNNING_VAR,
  BN_SAVE_MEAN,
  BN_SAVE_INVSTD,
  BN_NUM_PARAMS
};
enum {
  BN_BACKWARD_WEIGHT,
  BN_BACKWARD_MEAN,
  BN_BACKWARD_VAR,
  BN_GRAD_WEIGHT,
  BN_GRAD_BIAS,
  BN_NUM_BACKWARD_PARAMS
};

// Input viewed as [outer, C, inner], each dim as (size, input stride, output
// stride). outer spans outer_ndim dims, innermost first.
template <typename index_t = int64_t>
struct BatchNormLayout {
  ::c10::metal::vec3<index_t> channel;
  ::c10::metal::vec3<index_t> inner;
  ::c10::metal::array<::c10::metal::vec3<index_t>, ::c10::metal::max_ndim>
      outer_dims;
  index_t outer_numel;
  int32_t outer_ndim;
};

template <typename index_t = int64_t>
struct BatchNormStatsParams {
  index_t num_channels;
  index_t num_partials;
  float eps;
  float momentum;
  ::c10::metal::array<index_t, BN_NUM_PARAMS> strides;
};
