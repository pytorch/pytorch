#pragma once

#include <ATen/native/mps/kernels/Matmul.h>

C10_METAL_CONSTEXPR uint32_t kGroupedMMFewRowsMax = 32;
C10_METAL_CONSTEXPR uint32_t kGroupedMMFewRowsSimdgroups = 16;
C10_METAL_CONSTEXPR uint32_t kGroupedMMFewRowsLaneChunk = 16;

template <typename idx_t>
struct GroupedMMParams {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t groups;
  idx_t a_stride_m;
  idx_t a_stride_k;
  idx_t b_stride_k;
  idx_t b_stride_n;
  idx_t out_stride_m;
  idx_t out_stride_n;
  idx_t batch_stride;
};
