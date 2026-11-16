#pragma once

#include <ATen/native/mps/kernels/Matmul.h>
#include <c10/metal/common.h>

C10_METAL_CONSTEXPR uint32_t kGroupedMMMediumTileCols = 128;
C10_METAL_CONSTEXPR uint32_t kGroupedMMFewRowsMax = 32;
C10_METAL_CONSTEXPR uint32_t kGroupedMMFewRowsSimdgroups = 16;
C10_METAL_CONSTEXPR uint32_t kGroupedMMFewRowsLaneChunk = 16;

template <typename idx_t>
struct GroupedMMParams {
  MatmulParams<idx_t> mm;
  uint32_t groups;
  idx_t batch_stride;
};
