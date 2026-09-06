#include <ATen/native/mps/kernels/GroupedMM.h>
#include <c10/metal/common.h>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;
using c10::metal::ceil_div;
using c10::metal::simdgroup_size;

// simdgroup_matrix fragments are fixed 8x8.
constant constexpr int kFragDim = 8;
// Threadgroup tile rows are padded by one fragment width so consecutive
// simdgroup_load rows do not land on the same threadgroup memory banks.
constant constexpr int kTilePad = kFragDim;

enum class GroupedMMMode { rows, cols, k };

inline uint32_t grouped_mm_offset(int32_t offset, uint32_t limit) {
  if (offset <= 0) {
    return 0;
  }
  return min(static_cast<uint32_t>(offset), limit);
}

// Maps a flat tile index along the jagged dimension (rows or columns, capped
// at limit) to its (group, first index, valid count) triple by walking the
// jagged group extents; padding tiles leave the zero-initialized outputs
// unchanged.
template <int TILE>
inline void grouped_mm_group_tile(
    device const int32_t* offsets,
    uint32_t groups,
    uint32_t limit,
    uint32_t tile,
    thread uint32_t& group,
    thread uint32_t& tile_first,
    thread uint32_t& tile_count) {
  uint32_t start = 0;
  uint32_t tile_start = 0;
  for (uint32_t g = 0; g < groups; ++g) {
    const uint32_t end = grouped_mm_offset(offsets[g], limit);
    const uint32_t count = end > start ? end - start : 0;
    const uint32_t tiles = ceil_div(count, static_cast<uint32_t>(TILE));
    if (tile < tile_start + tiles) {
      group = g;
      tile_first = start + (tile - tile_start) * TILE;
      tile_count = min(static_cast<uint32_t>(TILE), end - tile_first);
      return;
    }
    tile_start += tiles;
    start = max(start, end);
  }
}

// One thread searches the offsets, then the tile is broadcast through
// threadgroup memory to the whole threadgroup.
template <int TILE>
inline bool grouped_mm_group_tile_broadcast(
    device const int32_t* offsets,
    uint32_t groups,
    uint32_t limit,
    uint32_t tile,
    uint32_t tid,
    threadgroup uint32_t* tile_info,
    thread uint32_t& group,
    thread uint32_t& tile_first,
    thread uint32_t& tile_count) {
  if (tid == 0) {
    uint32_t g = 0;
    uint32_t first = 0;
    uint32_t count = 0;
    grouped_mm_group_tile<TILE>(offsets, groups, limit, tile, g, first, count);
    tile_info[0] = g;
    tile_info[1] = first;
    tile_info[2] = count;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  group = tile_info[0];
  tile_first = tile_info[1];
  tile_count = tile_info[2];
  return tile_count != 0;
}

template <typename T>
struct GroupedMMTile {
  device const T* a;
  device const T* b;
  device T* out;
  uint32_t rows;
  uint32_t cols;
  uint32_t k;
};

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
    thread GroupedMMTile<T>& tile) {
  uint32_t group = tgid.z;
  uint32_t row_start = static_cast<uint32_t>(tgid.y) * BM;
  uint32_t col_start = static_cast<uint32_t>(tgid.x) * BN;
  tile.rows = min(static_cast<uint32_t>(BM), params.m - row_start);
  tile.cols = min(static_cast<uint32_t>(BN), params.n - col_start);
  tile.k = params.k;
  if IF_CONSTEXPR (MODE == GroupedMMMode::rows) {
    if (!grouped_mm_group_tile_broadcast<BM>(
            offsets,
            params.groups,
            params.m,
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
            params.n,
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
    const uint32_t k_end = grouped_mm_offset(offsets[group], params.k);
    const uint32_t k_begin =
        min(group == 0 ? 0u : grouped_mm_offset(offsets[group - 1], params.k),
            k_end);
    mat_a += static_cast<idx_t>(k_begin) * params.a_stride_k;
    mat_b += static_cast<idx_t>(k_begin) * params.b_stride_k;
    output += static_cast<idx_t>(group) * params.batch_stride;
    tile.k = k_end - k_begin;
  }
  tile.a = mat_a + static_cast<idx_t>(row_start) * params.a_stride_m;
  tile.b = mat_b + static_cast<idx_t>(col_start) * params.b_stride_n;
  tile.out = output + static_cast<idx_t>(row_start) * params.out_stride_m +
      static_cast<idx_t>(col_start) * params.out_stride_n;
  return true;
}

// simdgroup-matrix tile GEMM: accumulates out[:rows, :cols] = A[:rows, :k] *
// B[:k, :cols] in fp32 and writes the tile back.
template <typename T, int BM, typename idx_t>
inline void grouped_mm_tile_gemm(
    thread const GroupedMMTile<T>& tile,
    constant GroupedMMParams<idx_t>& params,
    threadgroup T* a_tile,
    threadgroup T* b_tile,
    uint simd_group,
    uint simd_lane) {
  constexpr int BN = kGroupedMMTileN;
  constexpr int BK = kGroupedMMTileK;
  constexpr int WN = 2;
  constexpr int WM = grouped_mm_simdgroups(BM, BN) / WN;
  constexpr int threads = WM * WN * simdgroup_size;
  constexpr int warp_m_tile = BM / WM;
  constexpr int warp_n_tile = BN / WN;
  constexpr int thread_tiles_m = warp_m_tile / kFragDim;
  constexpr int thread_tiles_n = warp_n_tile / kFragDim;
  constexpr int lda = BK + kTilePad;
  constexpr int ldb = BN + kTilePad;
  static_assert(BM % (kFragDim * WM) == 0, "invalid M tile");
  static_assert(BN % (kFragDim * WN) == 0, "invalid N tile");
  static_assert(BK % kFragDim == 0, "invalid K tile");

  const uint32_t tid = simd_group * simdgroup_size + simd_lane;
  device const T* mat_a = tile.a;
  device const T* mat_b = tile.b;
  const uint32_t tile_rows = tile.rows;
  const uint32_t tile_cols = tile.cols;
  const uint32_t k_end = tile.k;

  simdgroup_matrix<float, kFragDim, kFragDim> accum[thread_tiles_m]
                                                   [thread_tiles_n];
  for (int i = 0; i < thread_tiles_m; ++i) {
    for (int j = 0; j < thread_tiles_n; ++j) {
      accum[i][j] = simdgroup_matrix<float, kFragDim, kFragDim>(0.0f);
    }
  }

  const uint32_t warp_m = (simd_group / WN) * warp_m_tile;
  const uint32_t warp_n = (simd_group % WN) * warp_n_tile;

  for (idx_t k_start = 0; k_start < k_end; k_start += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (params.a_stride_k == 1) {
      const uint32_t tile_k = tid % BK;
      const uint32_t first_row = tid / BK;
      for (uint32_t tile_row = first_row; tile_row < BM;
           tile_row += threads / BK) {
        const idx_t k = k_start + tile_k;
        a_tile[tile_row * lda + tile_k] = tile_row < tile_rows && k < k_end
            ? mat_a[tile_row * params.a_stride_m + k]
            : T(0);
      }
    } else {
      const uint32_t tile_row = tid % BM;
      const uint32_t first_k = tid / BM;
      for (uint32_t tile_k = first_k; tile_k < BK; tile_k += threads / BM) {
        const idx_t k = k_start + tile_k;
        a_tile[tile_row * lda + tile_k] = tile_row < tile_rows && k < k_end
            ? mat_a[tile_row * params.a_stride_m + k * params.a_stride_k]
            : T(0);
      }
    }

    if (params.b_stride_n == 1) {
      const uint32_t tile_col = tid % BN;
      const uint32_t first_k = tid / BN;
      for (uint32_t tile_k = first_k; tile_k < BK; tile_k += threads / BN) {
        const idx_t k = k_start + tile_k;
        b_tile[tile_k * ldb + tile_col] = k < k_end && tile_col < tile_cols
            ? mat_b[k * params.b_stride_k + tile_col]
            : T(0);
      }
    } else {
      const uint32_t tile_k = tid % BK;
      const uint32_t first_col = tid / BK;
      for (uint32_t tile_col = first_col; tile_col < BN;
           tile_col += threads / BK) {
        const idx_t k = k_start + tile_k;
        b_tile[tile_k * ldb + tile_col] = k < k_end && tile_col < tile_cols
            ? mat_b[k * params.b_stride_k + tile_col * params.b_stride_n]
            : T(0);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < BK; k += kFragDim) {
      simdgroup_matrix<T, kFragDim, kFragDim> a_frag[thread_tiles_m];
      simdgroup_matrix<T, kFragDim, kFragDim> b_frag[thread_tiles_n];
      for (int i = 0; i < thread_tiles_m; ++i) {
        simdgroup_load(
            a_frag[i], &a_tile[(warp_m + i * kFragDim) * lda + k], lda);
      }
      for (int j = 0; j < thread_tiles_n; ++j) {
        simdgroup_load(
            b_frag[j], &b_tile[k * ldb + warp_n + j * kFragDim], ldb);
      }
      for (int i = 0; i < thread_tiles_m; ++i) {
        for (int j = 0; j < thread_tiles_n; ++j) {
          simdgroup_multiply_accumulate(
              accum[i][j], a_frag[i], b_frag[j], accum[i][j]);
        }
      }
    }
  }

  // Per-lane (row, col) ownership within an 8x8 fragment; thread_elements()
  // holds (row, col) and (row, col + 1). Same mapping as Convolution.metal.
  const uint32_t quad = simd_lane / 4;
  const uint32_t frag_m = (quad & 4) + ((simd_lane / 2) % 4);
  const uint32_t frag_n = (quad & 2) * 2 + (simd_lane % 2) * 2;
  for (int i = 0; i < thread_tiles_m; ++i) {
    const uint32_t tile_row = warp_m + i * kFragDim + frag_m;
    if (tile_row >= tile_rows) {
      continue;
    }
    for (int j = 0; j < thread_tiles_n; ++j) {
      for (int element = 0; element < 2; ++element) {
        const uint32_t tile_col = warp_n + j * kFragDim + frag_n + element;
        if (tile_col < tile_cols) {
          tile.out
              [tile_row * params.out_stride_m +
               tile_col * params.out_stride_n] =
              static_cast<T>(accum[i][j].thread_elements()[element]);
        }
      }
    }
  }
}

template <typename T, int BM, GroupedMMMode MODE, typename idx_t>
[[max_total_threads_per_threadgroup(
    grouped_mm_simdgroups(BM, kGroupedMMTileN) * simdgroup_size)]]
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
  threadgroup T a_tile[BM * (kGroupedMMTileK + kTilePad)];
  threadgroup T b_tile[kGroupedMMTileK * (kGroupedMMTileN + kTilePad)];
  GroupedMMTile<T> tile;
  // see comment in MPP kernel for the same check, to clarify what this is for
  if (!grouped_mm_tile<BM, kGroupedMMTileN, MODE>(
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
  grouped_mm_tile_gemm<T, BM>(
      tile, params, a_tile, b_tile, simd_group, simd_lane);
}

#if __METAL_VERSION__ >= 400 && \
    __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// matmul2d tile op. TA/TB name the operand layouts ('t' = column-major); every
// tensor is presented to matmul2d with a unit innermost stride and the matching
// transpose flag, which matmul2d requires (it ignores layouts that contradict
// the descriptor). half/bfloat multiply at relaxed precision; accumulation
// stays fp32 via the destination cooperative tensor.
template <typename T, int BM, int BN, bool TA, bool TB>
inline void grouped_mm_mpp_gemm(
    thread const GroupedMMTile<T>& tile,
    constant GroupedMMParams<uint64_t>& params) {
  constexpr int NSG = grouped_mm_simdgroups(BM, BN);
  constexpr bool relaxed = !is_same_v<T, float>;
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BM,
      BN,
      static_cast<int>(dynamic_extent),
      TA,
      TB,
      relaxed,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<NSG>> op;

  const int rows = int(tile.rows);
  const int cols = int(tile.cols);
  const int k = int(tile.k);
  const auto a_ext =
      TA ? dextents<int32_t, 2>(rows, k) : dextents<int32_t, 2>(k, rows);
  const auto a_str = TA
      ? array<int32_t, 2>{int(params.a_stride_m), int(params.a_stride_k)}
      : array<int32_t, 2>{int(params.a_stride_k), int(params.a_stride_m)};
  const auto b_ext =
      TB ? dextents<int32_t, 2>(k, cols) : dextents<int32_t, 2>(cols, k);
  const auto b_str = TB
      ? array<int32_t, 2>{int(params.b_stride_k), int(params.b_stride_n)}
      : array<int32_t, 2>{int(params.b_stride_n), int(params.b_stride_k)};
  tensor<device T, dextents<int32_t, 2>, tensor_inline> a(
      const_cast<device T*>(tile.a), a_ext, a_str);
  tensor<device T, dextents<int32_t, 2>, tensor_inline> b(
      const_cast<device T*>(tile.b), b_ext, b_str);
  tensor<device T, dextents<int32_t, 2>, tensor_inline> out(
      tile.out,
      dextents<int32_t, 2>(cols, rows),
      array<int32_t, 2>{int(params.out_stride_n), int(params.out_stride_m)});

  auto accum = op.template get_destination_cooperative_tensor<
      decltype(a),
      decltype(b),
      float>();
  if (k == 0) {
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < accum.get_capacity(); ++i) {
      accum[i] = 0.0f;
    }
  } else {
    op.run(a, b, accum);
  }

  if constexpr (is_same_v<T, float>) {
    accum.store(out);
  } else {
    auto result = op.template get_destination_cooperative_tensor<
        decltype(a),
        decltype(b),
        T>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < accum.get_capacity(); ++i) {
      result[i] = static_cast<T>(accum[i]);
    }
    result.store(out);
  }
}

template <typename T, int BM, int BN, bool TA, bool TB, GroupedMMMode MODE>
[[max_total_threads_per_threadgroup(
    grouped_mm_simdgroups(BM, BN) * simdgroup_size)]]
kernel void grouped_mm_mpp(
    device const T* mat_a [[buffer(0)]],
    device const T* mat_b [[buffer(1)]],
    device const int32_t* offsets [[buffer(2)]],
    device T* output [[buffer(3)]],
    constant GroupedMMParams<uint64_t>& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint32_t tile_info[3];
  GroupedMMTile<T> tile;
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
  grouped_mm_mpp_gemm<T, BM, BN, TA, TB>(tile, params);
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
