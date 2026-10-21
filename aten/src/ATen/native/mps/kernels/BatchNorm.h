#pragma once
#include <c10/metal/common.h>

// Per-channel Welford partial (mean, M2, count), stored as packed_float3
C10_METAL_CONSTEXPR uint32_t BN_PARTIAL_SIZE = 3;
enum class BNStat : uint32_t { Mean, Scale, Shift, NumStats };
enum class BNParam : uint32_t {
  Weight,
  Bias,
  RunningMean,
  RunningVar,
  SaveMean,
  SaveInvstd,
  NumParams
};

template <typename E>
inline constexpr uint32_t bn_idx(E e) {
  return static_cast<uint32_t>(e);
}

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

struct BatchNormStatsParams {
  uint32_t num_channels;
  uint32_t num_partials;
  float eps;
  float momentum;
  ::c10::metal::array<int64_t, bn_idx(BNParam::NumParams)> strides;
};
