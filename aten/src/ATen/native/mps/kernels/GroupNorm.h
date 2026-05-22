#pragma once

#include <c10/metal/common.h>

// The group_norm kernel operates on blocks of 4 elements at a time in unrolled
// loops. Other values were tested, and 4 gave the best performance.
C10_METAL_CONSTEXPR uint32_t BLOCK_SIZE = 4;

struct GroupNormParams {
  uint32_t HxW;
  uint32_t num_groups;
  uint32_t channels_per_group;
  uint32_t elements_per_group;
  float eps;
};
