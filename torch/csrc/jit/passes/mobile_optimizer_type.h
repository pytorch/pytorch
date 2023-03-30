#pragma once

#include <cstdint>

enum class MobileOptimizerType : int8_t {
  CONV_BN_FUSION,
  INSERT_FOLD_PREPACK_OPS,
  REMOVE_DROPOUT,
  FUSE_ADD_RELU,
  HOIST_CONV_PACKED_PARAMS,
  CONV_1D_TO_2D,
  VULKAN_AUTOMATIC_GPU_TRANSFER,
};
