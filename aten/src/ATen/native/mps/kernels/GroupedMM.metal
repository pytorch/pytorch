#include <ATen/native/mps/kernels/GroupedMM.h>
#include <c10/metal/common.h>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;
using c10::metal::ceil_div;
using c10::metal::simdgroup_size;

enum class GroupedMMMode { rows, cols, k };

// One SIMD group computes one output element. Find the expert on-device so
// sparse routing needs neither a CPU sync nor threadgroups for empty experts.
template <typename T, typename idx_t>
[[max_total_threads_per_threadgroup(
    kGroupedMMFewRowsSimdgroups * simdgroup_size)]]
kernel void grouped_mm_few_rows(
    device const T* mat_a [[buffer(0)]],
    device const T* mat_b [[buffer(1)]],
    device const int32_t* offsets [[buffer(2)]],
    device T* output [[buffer(3)]],
    constant GroupedMMParams<idx_t>& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  constant MatmulParams<idx_t>& mm = params.mm;
  const uint32_t row = tgid.y;
  const uint32_t col = tgid.x * kGroupedMMFewRowsSimdgroups + simd_group;
  if (col >= mm.n) {
    return;
  }
  uint32_t group = 0;
  if (simd_lane == 0) {
    group = grouped_mm_find_group(offsets, params.groups, row);
  }
  group = simd_broadcast_first(group);
  if (group == params.groups) {
    return;
  }
  mat_a += row * mm.a_stride_m;
  mat_b += group * params.batch_stride + col * mm.b_stride_n;
  float4 accum = 0;
  constexpr auto chunk = kGroupedMMFewRowsLaneChunk;
  for (idx_t base = simd_lane * chunk; base < mm.k;
       base += simdgroup_size * chunk) {
    const idx_t end = min(base + chunk, static_cast<idx_t>(mm.k));
    idx_t k = base;
    for (; k + 4 <= end; k += 4) {
      const float4 a =
          float4(*reinterpret_cast<device const vec<T, 4>*>(mat_a + k));
      const float4 b =
          float4(*reinterpret_cast<device const vec<T, 4>*>(mat_b + k));
      accum = fma(a, b, accum);
    }
    for (; k < end; ++k) {
      accum.x = fma(float(mat_a[k]), float(mat_b[k]), accum.x);
    }
  }
  const float value = simd_sum(dot(accum, float4(1)));
  if (simd_lane == 0) {
    output[row * mm.out_stride_m + col * mm.out_stride_n] = T(value);
  }
}

#define INSTANTIATE_GROUPED_MM_FEW_ROWS(DTYPE, IDX_T, IDX_NAME)              \
  template                                                                   \
      [[host_name("grouped_mm_few_rows_" #DTYPE "_" #IDX_NAME)]] kernel void \
      grouped_mm_few_rows<DTYPE, IDX_T>(                                     \
          device const DTYPE*,                                               \
          device const DTYPE*,                                               \
          device const int32_t*,                                             \
          device DTYPE*,                                                     \
          constant GroupedMMParams<IDX_T>&,                                  \
          uint3,                                                             \
          uint,                                                              \
          uint)

#define INSTANTIATE_GROUPED_MM_FEW_ROWS_DTYPE(DTYPE)     \
  INSTANTIATE_GROUPED_MM_FEW_ROWS(DTYPE, uint32_t, u32); \
  INSTANTIATE_GROUPED_MM_FEW_ROWS(DTYPE, uint64_t, u64)

INSTANTIATE_GROUPED_MM_FEW_ROWS_DTYPE(float);
INSTANTIATE_GROUPED_MM_FEW_ROWS_DTYPE(half);
INSTANTIATE_GROUPED_MM_FEW_ROWS_DTYPE(bfloat);

// Resolves this threadgroup's output tile and its operand views: rows and cols
// modes look the tile up in the jagged offsets and pick the batched operand,
// k mode slices the shared contraction dim and the batched output.
template <int BM, int BN, GroupedMMMode MODE, typename T, typename idx_t>
inline bool grouped_mm_tile(
    device const T* mat_a,
    device const T* mat_b,
    device const int32_t* offsets,
    device T* output,
    constant GroupedMMParams<idx_t>& params,
    uint3 tgid,
    uint32_t tid,
    threadgroup uint32_t* tile_info,
    thread MatmulTile<T>& tile) {
  constant MatmulParams<idx_t>& mm = params.mm;
  uint32_t group = tgid.z;
  uint32_t row_start = static_cast<uint32_t>(tgid.y) * BM;
  uint32_t col_start = static_cast<uint32_t>(tgid.x) * BN;
  tile.rows = min(static_cast<uint32_t>(BM), mm.m - row_start);
  tile.cols = min(static_cast<uint32_t>(BN), mm.n - col_start);
  tile.k = mm.k;
  if IF_CONSTEXPR (MODE == GroupedMMMode::rows) {
    if (!grouped_mm_group_tile_broadcast<BM>(
            offsets,
            params.groups,
            mm.m,
            tgid.y,
            tid,
            tile_info,
            group,
            row_start,
            tile.rows)) {
      return false;
    }
    mat_b += static_cast<idx_t>(group) * params.batch_stride;
  } else if IF_CONSTEXPR (MODE == GroupedMMMode::cols) {
    if (!grouped_mm_group_tile_broadcast<BN>(
            offsets,
            params.groups,
            mm.n,
            tgid.x,
            tid,
            tile_info,
            group,
            col_start,
            tile.cols)) {
      return false;
    }
    mat_a += static_cast<idx_t>(group) * params.batch_stride;
  } else {
    const uint32_t k_end = grouped_mm_offset(offsets[group], mm.k);
    const uint32_t k_begin = min(
        group == 0 ? 0u : grouped_mm_offset(offsets[group - 1], mm.k), k_end);
    mat_a += static_cast<idx_t>(k_begin) * mm.a_stride_k;
    mat_b += static_cast<idx_t>(k_begin) * mm.b_stride_k;
    output += static_cast<idx_t>(group) * params.batch_stride;
    tile.k = k_end - k_begin;
  }
  tile.a = mat_a + static_cast<idx_t>(row_start) * mm.a_stride_m;
  tile.b = mat_b + static_cast<idx_t>(col_start) * mm.b_stride_n;
  tile.out = output + static_cast<idx_t>(row_start) * mm.out_stride_m +
      static_cast<idx_t>(col_start) * mm.out_stride_n;
  return true;
}

template <typename T, int BM, GroupedMMMode MODE, typename idx_t>
[[max_total_threads_per_threadgroup(
    matmul_simdgroups(BM, kMatmulTileN) * simdgroup_size)]]
kernel void grouped_mm(
    device const T* mat_a [[buffer(0)]],
    device const T* mat_b [[buffer(1)]],
    device const int32_t* offsets [[buffer(2)]],
    device T* output [[buffer(3)]],
    constant GroupedMMParams<idx_t>& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  threadgroup uint32_t tile_info[3];
  threadgroup T a_tile[BM * (kMatmulTileK + kMatmulTilePad)];
  threadgroup T b_tile[kMatmulTileK * (kMatmulTileN + kMatmulTilePad)];
  MatmulTile<T> tile;
  // see comment in MPP kernel for the same check, to clarify what this is for
  if (!grouped_mm_tile<BM, kMatmulTileN, MODE>(
          mat_a,
          mat_b,
          offsets,
          output,
          params,
          tgid,
          simd_group * simdgroup_size + simd_lane,
          tile_info,
          tile)) {
    return;
  }
  matmul_tile_gemm<T, BM>(
      tile, params.mm, a_tile, b_tile, simd_group, simd_lane);
}

#if C10_METAL_HAS_MPP

template <typename T, int BM, int BN, bool TA, bool TB, GroupedMMMode MODE>
[[max_total_threads_per_threadgroup(
    matmul_simdgroups(BM, BN) * simdgroup_size)]]
kernel void grouped_mm_mpp(
    device const T* mat_a [[buffer(0)]],
    device const T* mat_b [[buffer(1)]],
    device const int32_t* offsets [[buffer(2)]],
    device T* output [[buffer(3)]],
    constant GroupedMMParams<uint64_t>& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint32_t tile_info[3];
  MatmulTile<T> tile;
  // To explain why if is needed here. Imagine we have 32 num_tokens and 2
  // groups and the way offsets are setup, 17 tokens go to expert 0 and 15
  // tokens go to expert 1. Naively we can say, launch 2 threadgroups, one for
  // each expert, but given that our tile size is 16 we need an extra
  // threadgroup for the 17 tokens. But here is the problem, on CPU we don't
  // know what the offsets are unless we read them which causes GPU -> CPU sync
  // which we want to avoid so because of that, we launch
  // [ceil_div(sum_of_total_tokens / TILE_SIZE) + num_groups] threadgroups. In
  // the above scenario that will be [ceil_div(32 / 16) + 2] = 4. However we see
  // that 4 threadgroups are not needed (inside of the kernel) therefore this if
  // guards that one threadgroup that needs to do no calculations.
  if (!grouped_mm_tile<BM, BN, MODE>(
          mat_a, mat_b, offsets, output, params, tgid, tid, tile_info, tile)) {
    return;
  }
  matmul_mpp_gemm<T, BM, BN, TA, TB>(tile, params.mm);
}

#define INSTANTIATE_GROUPED_MM_MPP_MODE(DTYPE, BM, BN, TA, TB, LAYOUT, MODE) \
  template [[host_name("grouped_mm_" #MODE "_mpp_" #LAYOUT "_" #DTYPE        \
                       "_bm" #BM "_bn" #BN)]] kernel void                    \
  grouped_mm_mpp<DTYPE, BM, BN, TA, TB, GroupedMMMode::MODE>(                \
      device const DTYPE*,                                                   \
      device const DTYPE*,                                                   \
      device const int32_t*,                                                 \
      device DTYPE*,                                                         \
      constant GroupedMMParams<uint64_t>&,                                   \
      uint3,                                                                 \
      uint)

#define INSTANTIATE_GROUPED_MM_MPP(DTYPE, BM, BN, TA, TB, LAYOUT)       \
  INSTANTIATE_GROUPED_MM_MPP_MODE(DTYPE, BM, BN, TA, TB, LAYOUT, rows); \
  INSTANTIATE_GROUPED_MM_MPP_MODE(DTYPE, BM, BN, TA, TB, LAYOUT, k);    \
  INSTANTIATE_GROUPED_MM_MPP_MODE(DTYPE, BM, BN, TA, TB, LAYOUT, cols)

// Layout suffix: one letter per operand (mat_a then mat_b), 'n' = row-major,
// 't' = column-major; the host picks the variant from the operand strides.
#define INSTANTIATE_GROUPED_MM_MPP_BM(DTYPE, BM, BN)           \
  INSTANTIATE_GROUPED_MM_MPP(DTYPE, BM, BN, false, false, nn); \
  INSTANTIATE_GROUPED_MM_MPP(DTYPE, BM, BN, false, true, nt);  \
  INSTANTIATE_GROUPED_MM_MPP(DTYPE, BM, BN, true, false, tn);  \
  INSTANTIATE_GROUPED_MM_MPP(DTYPE, BM, BN, true, true, tt)

#define INSTANTIATE_GROUPED_MM_MPP_DTYPE(DTYPE)  \
  INSTANTIATE_GROUPED_MM_MPP_BM(DTYPE, 16, 64);  \
  INSTANTIATE_GROUPED_MM_MPP_BM(DTYPE, 16, 128); \
  INSTANTIATE_GROUPED_MM_MPP_BM(DTYPE, 32, 64);  \
  INSTANTIATE_GROUPED_MM_MPP_BM(DTYPE, 32, 128); \
  INSTANTIATE_GROUPED_MM_MPP_BM(DTYPE, 64, 64);  \
  INSTANTIATE_GROUPED_MM_MPP_BM(DTYPE, 64, 256)

INSTANTIATE_GROUPED_MM_MPP_DTYPE(float);
INSTANTIATE_GROUPED_MM_MPP_DTYPE(half);
INSTANTIATE_GROUPED_MM_MPP_DTYPE(bfloat);

#endif

#define INSTANTIATE_GROUPED_MM_MODE(DTYPE, BM, IDX_T, IDX_NAME, MODE) \
  template [[host_name("grouped_mm_" #MODE "_" #DTYPE "_bm" #BM       \
                       "_" #IDX_NAME)]] kernel void                   \
  grouped_mm<DTYPE, BM, GroupedMMMode::MODE, IDX_T>(                  \
      device const DTYPE*,                                            \
      device const DTYPE*,                                            \
      device const int32_t*,                                          \
      device DTYPE*,                                                  \
      constant GroupedMMParams<IDX_T>&,                               \
      uint3,                                                          \
      uint,                                                           \
      uint)

#define INSTANTIATE_GROUPED_MM(DTYPE, BM, IDX_T, IDX_NAME)       \
  INSTANTIATE_GROUPED_MM_MODE(DTYPE, BM, IDX_T, IDX_NAME, rows); \
  INSTANTIATE_GROUPED_MM_MODE(DTYPE, BM, IDX_T, IDX_NAME, k)

#define INSTANTIATE_GROUPED_MM_BM(DTYPE, BM)        \
  INSTANTIATE_GROUPED_MM(DTYPE, BM, uint32_t, u32); \
  INSTANTIATE_GROUPED_MM(DTYPE, BM, uint64_t, u64)

#define INSTANTIATE_GROUPED_MM_DTYPE(DTYPE) \
  INSTANTIATE_GROUPED_MM_BM(DTYPE, 16);     \
  INSTANTIATE_GROUPED_MM_BM(DTYPE, 32);     \
  INSTANTIATE_GROUPED_MM_BM(DTYPE, 64)

INSTANTIATE_GROUPED_MM_DTYPE(float);
INSTANTIATE_GROUPED_MM_DTYPE(half);
INSTANTIATE_GROUPED_MM_DTYPE(bfloat);
