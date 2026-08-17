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

// Threadgroup shapes of the outer / outer_small_dim / narrow reduction
// kernels; the host dispatch and the kernel registrations must agree on
// these.
C10_METAL_CONSTEXPR uint32_t OUTER_TG_WIDTH = 32;
C10_METAL_CONSTEXPR uint32_t OUTER_TG_HEIGHT = 32;
C10_METAL_CONSTEXPR uint32_t NARROW_TG_SIZE = 256;
static_assert(
    OUTER_TG_WIDTH == ::c10::metal::simdgroup_size,
    "one threadgroup row per simdgroup keeps column loads coalesced");
static_assert(
    OUTER_TG_WIDTH * OUTER_TG_HEIGHT <= MAX_THREADGROUP_SIZE &&
        NARROW_TG_SIZE <= MAX_THREADGROUP_SIZE,
    "exceeds the Metal threadgroup size limit");

// Outer-dim (non-innermost) routing thresholds, phrased in the
// [outer_size, dim_size, inner_size] view of the input (dim reduced).
C10_METAL_CONSTEXPR uint32_t OUTER_SMALL_DIM_MAX_SIZE = 256;
C10_METAL_CONSTEXPR uint32_t NARROW_BATCHED_MIN_DIM_SIZE = 128;
C10_METAL_CONSTEXPR uint32_t NARROW_SPLIT_ELEMS_PER_TG = 8192;
C10_METAL_CONSTEXPR uint32_t OUTER_SPLIT_MIN_DIM_SIZE = 4096;
C10_METAL_CONSTEXPR uint32_t OUTER_SPLIT_MIN_TGS = 32;
C10_METAL_CONSTEXPR uint32_t OUTER_SPLIT_MIN_SEG_LEN = 512;
C10_METAL_CONSTEXPR uint32_t OUTER_SPLIT_MAX_TGS = 2048;
C10_METAL_CONSTEXPR uint32_t OUTER_SPLIT_STRIDED_TARGET_TGS = 256;

// argmax/argmin inner split-K thresholds (argmax_argmin_out_mps in
// ReduceOps.mm). The arg inner kernel keeps one compare chain per lane (no
// NCHAINS ILP), so it saturates the GPU later than the value kernels and
// splits at a higher threadgroup count.
C10_METAL_CONSTEXPR uint32_t ARG_SPLIT_MIN_TGS = 512;
C10_METAL_CONSTEXPR uint32_t ARG_SPLIT_MIN_SEG_LEN = 512;
C10_METAL_CONSTEXPR uint32_t ARG_SPLIT_TARGET_PARTIALS = 16384;

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
