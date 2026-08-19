#pragma once

#include <c10/metal/common.h>

// The group_norm kernel operates on blocks of 4 elements at a time in unrolled
// loops. Other values were tested, and 4 gave the best performance.
C10_METAL_CONSTEXPR uint32_t BLOCK_SIZE = 4;

template <typename idx_T = uint32_t>
struct GroupNormParams {
  idx_T HxW;
  idx_T num_groups;
  idx_T channels_per_group;
  idx_T elements_per_group;
  idx_T C;
  idx_T N_times_HxW;
  float eps;
};
