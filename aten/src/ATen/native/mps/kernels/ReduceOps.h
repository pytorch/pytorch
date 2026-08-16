#pragma once
#include <c10/metal/common.h>

#define MAX_THREADGROUP_SIZE static_cast<uint32_t>(1024)
C10_METAL_CONSTEXPR uint32_t SUM_NCHAINS = 8;

// Threadgroup size the host dispatches the inner / inner_chunk reduction
// kernels with; both carve the threadgroup into whole simdgroups.
C10_METAL_CONSTEXPR uint32_t INNER_TG_SIZE = 256;
static_assert(
    INNER_TG_SIZE % ::c10::metal::simdgroup_size == 0,
    "must be a whole number of simdgroups");
static_assert(
    INNER_TG_SIZE <= MAX_THREADGROUP_SIZE,
    "exceeds the Metal threadgroup size limit");

// Inner-dim routing thresholds, see reduction_dispatch_mps in ReduceOps.mm.
C10_METAL_CONSTEXPR uint32_t CHUNK_MAX_ROW_LEN = 256;
C10_METAL_CONSTEXPR uint32_t CHUNK_ELEMS_PER_LANE = 16;
C10_METAL_CONSTEXPR uint32_t CHUNK_MIN_NUMEL = 65536;
C10_METAL_CONSTEXPR uint32_t SPLIT_MIN_ROW_LEN = 2048;
C10_METAL_CONSTEXPR uint32_t SPLIT_MIN_SEG_LEN = 64;
C10_METAL_CONSTEXPR uint32_t SPLIT_MAX_SEGS = 2048;
C10_METAL_CONSTEXPR uint32_t SPLIT_TARGET_PARTIALS = 8192;
C10_METAL_CONSTEXPR uint32_t SPLIT_MIN_TGS = 64;

template <unsigned N = c10::metal::max_ndim>
struct NormParams {
  float p;
  uint32_t reduction_size;
  uint32_t ndim;

  ::c10::metal::array<uint32_t, N> input_sizes;
  ::c10::metal::array<uint32_t, N> input_strides;

  ::c10::metal::array<uint32_t, N> output_sizes;
  ::c10::metal::array<uint32_t, N> output_strides;
};
