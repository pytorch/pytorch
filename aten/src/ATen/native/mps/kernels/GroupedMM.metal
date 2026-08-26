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

inline uint32_t grouped_mm_offset(int32_t offset, uint32_t limit) {
  if (offset <= 0) {
    return 0;
  }
  return min(static_cast<uint32_t>(offset), limit);
}

// Maps a flat row-tile index to its (group, first row, valid rows) triple by
// walking the jagged group extents; returns false for the padding tiles the
// host over-allocates (grid y is sized for the worst case of one partial tile
// per group).
template <int BM>
inline bool grouped_mm_row_tile(
    device const int32_t* offsets,
    constant GroupedMMParams& params,
    uint32_t tile,
    thread uint32_t& group,
    thread uint32_t& row,
    thread uint32_t& rows) {
  uint32_t start = 0;
  uint32_t tile_start = 0;
  for (uint32_t g = 0; g < params.groups; ++g) {
    const uint32_t end = grouped_mm_offset(offsets[g], params.m);
    const uint32_t count = end > start ? end - start : 0;
    const uint32_t tiles = ceil_div(count, static_cast<uint32_t>(BM));
    if (tile < tile_start + tiles) {
      group = g;
      row = start + (tile - tile_start) * BM;
      rows = min(static_cast<uint32_t>(BM), end - row);
      return true;
    }
    tile_start += tiles;
    start = max(start, end);
  }
  return false;
}

// One thread searches the offsets, then the tile is broadcast through
// threadgroup memory to the whole threadgroup.
template <int BM>
inline bool grouped_mm_row_tile_broadcast(
    device const int32_t* offsets,
    constant GroupedMMParams& params,
    uint32_t tile,
    uint32_t tid,
    threadgroup uint32_t* tile_info,
    thread uint32_t& group,
    thread uint32_t& row_start,
    thread uint32_t& tile_rows) {
  if (tid == 0) {
    uint32_t g = 0;
    uint32_t row = 0;
    uint32_t rows = 0;
    tile_info[3] = grouped_mm_row_tile<BM>(offsets, params, tile, g, row, rows);
    tile_info[0] = g;
    tile_info[1] = row;
    tile_info[2] = rows;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  group = tile_info[0];
  row_start = tile_info[1];
  tile_rows = tile_info[2];
  return tile_info[3] != 0;
}

struct GroupedMMTileCoords {
  uint32_t row_start;
  uint32_t tile_rows;
  uint32_t col_start;
  uint32_t k_begin;
  uint32_t k_end;
  uint64_t b_offset;
  uint64_t out_offset;
};

// Shared simdgroup-matrix tile GEMM used by both jagged modes: accumulates
// output[row_start:+tile_rows, col_start:+BN] += A[., k_begin:k_end] *
// B[k_begin:k_end, .] in fp32 and writes the tile back.
template <typename T, int BM, int BN, int BK, int WM, int WN>
inline void grouped_mm_tile_gemm(
    device const T* mat_a,
    device const T* mat_b,
    device T* output,
    constant GroupedMMParams& params,
    threadgroup T* a_tile,
    threadgroup T* b_tile,
    thread const GroupedMMTileCoords& coords,
    uint simd_group,
    uint simd_lane) {
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
  const uint32_t row_start = coords.row_start;
  const uint32_t tile_rows = coords.tile_rows;
  const uint32_t col_start = coords.col_start;
  const uint32_t k_end = coords.k_end;

  simdgroup_matrix<float, kFragDim, kFragDim> accum[thread_tiles_m]
                                                   [thread_tiles_n];
  for (int i = 0; i < thread_tiles_m; ++i) {
    for (int j = 0; j < thread_tiles_n; ++j) {
      accum[i][j] = simdgroup_matrix<float, kFragDim, kFragDim>(0.0f);
    }
  }

  const uint32_t warp_m = (simd_group / WN) * warp_m_tile;
  const uint32_t warp_n = (simd_group % WN) * warp_n_tile;

  for (uint64_t k_start = coords.k_begin; k_start < k_end; k_start += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (params.a_stride_k == 1) {
      const uint32_t tile_k = tid % BK;
      const uint32_t first_row = tid / BK;
      for (uint32_t tile_row = first_row; tile_row < BM;
           tile_row += threads / BK) {
        const uint64_t k = k_start + tile_k;
        a_tile[tile_row * lda + tile_k] = tile_row < tile_rows && k < k_end
            ? mat_a
                  [(static_cast<uint64_t>(row_start) + tile_row) *
                       params.a_stride_m +
                   k]
            : T(0);
      }
    } else {
      const uint32_t tile_row = tid % BM;
      const uint32_t first_k = tid / BM;
      for (uint32_t tile_k = first_k; tile_k < BK; tile_k += threads / BM) {
        const uint64_t k = k_start + tile_k;
        a_tile[tile_row * lda + tile_k] = tile_row < tile_rows && k < k_end
            ? mat_a
                  [(static_cast<uint64_t>(row_start) + tile_row) *
                       params.a_stride_m +
                   k * params.a_stride_k]
            : T(0);
      }
    }

    if (params.b_stride_n == 1) {
      const uint32_t tile_col = tid % BN;
      const uint32_t first_k = tid / BN;
      for (uint32_t tile_k = first_k; tile_k < BK; tile_k += threads / BN) {
        const uint64_t k = k_start + tile_k;
        const uint32_t col = col_start + tile_col;
        b_tile[tile_k * ldb + tile_col] = k < k_end && col < params.n
            ? mat_b[coords.b_offset + k * params.b_stride_k + col]
            : T(0);
      }
    } else {
      const uint32_t tile_k = tid % BK;
      const uint32_t first_col = tid / BK;
      for (uint32_t tile_col = first_col; tile_col < BN;
           tile_col += threads / BK) {
        const uint64_t k = k_start + tile_k;
        const uint32_t col = col_start + tile_col;
        b_tile[tile_k * ldb + tile_col] = k < k_end && col < params.n
            ? mat_b
                  [coords.b_offset + k * params.b_stride_k +
                   static_cast<uint64_t>(col) * params.b_stride_n]
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
    const uint32_t row = row_start + tile_row;
    for (int j = 0; j < thread_tiles_n; ++j) {
      for (int element = 0; element < 2; ++element) {
        const uint32_t col =
            col_start + warp_n + j * kFragDim + frag_n + element;
        if (col < params.n) {
          output
              [coords.out_offset +
               static_cast<uint64_t>(row) * params.out_stride_m +
               static_cast<uint64_t>(col) * params.out_stride_n] =
                  static_cast<T>(accum[i][j].thread_elements()[element]);
        }
      }
    }
  }
}

// Jagged rows: A[m, k] is split into groups along dim 0, each multiplied with
// its own batch of B[groups, k, n].
template <typename T, int BM, int BN, int BK, int WM, int WN>
kernel void grouped_mm_rows(
    device const T* mat_a [[buffer(0)]],
    device const T* mat_b [[buffer(1)]],
    device const int32_t* offsets [[buffer(2)]],
    device T* output [[buffer(3)]],
    constant GroupedMMParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  const uint32_t tid = simd_group * simdgroup_size + simd_lane;
  threadgroup uint32_t tile_info[4];
  uint32_t group = 0;
  uint32_t row_start = 0;
  uint32_t tile_rows = 0;
  if (!grouped_mm_row_tile_broadcast<BM>(
          offsets,
          params,
          tgid.y,
          tid,
          tile_info,
          group,
          row_start,
          tile_rows)) {
    return;
  }

  threadgroup T a_tile[BM * (BK + kTilePad)];
  threadgroup T b_tile[BK * (BN + kTilePad)];
  const GroupedMMTileCoords coords = {
      row_start,
      tile_rows,
      static_cast<uint32_t>(tgid.x) * BN,
      0,
      params.k,
      static_cast<uint64_t>(group) * params.b_batch_stride,
      0,
  };
  grouped_mm_tile_gemm<T, BM, BN, BK, WM, WN>(
      mat_a,
      mat_b,
      output,
      params,
      a_tile,
      b_tile,
      coords,
      simd_group,
      simd_lane);
}

// Jagged contraction: A[m, k] and B[k, n] contract over per-group slices of
// dim k, producing output[groups, m, n].
template <typename T, int BM, int BN, int BK, int WM, int WN>
kernel void grouped_mm_k(
    device const T* mat_a [[buffer(0)]],
    device const T* mat_b [[buffer(1)]],
    device const int32_t* offsets [[buffer(2)]],
    device T* output [[buffer(3)]],
    constant GroupedMMParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  const uint32_t group = static_cast<uint32_t>(tgid.z);
  const uint32_t k_end = grouped_mm_offset(offsets[group], params.k);
  const uint32_t raw_start =
      group == 0 ? 0 : grouped_mm_offset(offsets[group - 1], params.k);
  const uint32_t row_start = static_cast<uint32_t>(tgid.y) * BM;

  threadgroup T a_tile[BM * (BK + kTilePad)];
  threadgroup T b_tile[BK * (BN + kTilePad)];
  const GroupedMMTileCoords coords = {
      row_start,
      min(static_cast<uint32_t>(BM), params.m - row_start),
      static_cast<uint32_t>(tgid.x) * BN,
      min(raw_start, k_end),
      k_end,
      0,
      static_cast<uint64_t>(group) * params.out_batch_stride,
  };
  grouped_mm_tile_gemm<T, BM, BN, BK, WM, WN>(
      mat_a,
      mat_b,
      output,
      params,
      a_tile,
      b_tile,
      coords,
      simd_group,
      simd_lane);
}

#if __METAL_VERSION__ >= 400 && \
    __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// matmul2d tile op shared by both jagged modes. TA/TB name the operand
// layouts ('t' = column-major); every tensor is presented to matmul2d with a
// unit innermost stride and the matching transpose flag, which matmul2d
// requires (it ignores layouts that contradict the descriptor). RELAXED
// lowers the multiplication precision for half/bfloat like MppAttention.h;
// accumulation stays fp32 via the destination cooperative tensor.
template <typename T, int BM, int NSG, bool RELAXED, bool TA, bool TB>
inline void grouped_mm_mpp_gemm(
    device T* a_ptr,
    device T* b_ptr,
    device T* out_ptr,
    uint32_t tile_rows,
    uint32_t tile_cols,
    uint32_t k,
    constant GroupedMMParams& params) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BM,
      static_cast<int>(kGroupedMMTileN),
      static_cast<int>(dynamic_extent),
      TA,
      TB,
      RELAXED,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<NSG>> op;

  const auto a_ext = TA ? dextents<int32_t, 2>(int(tile_rows), int(k))
                        : dextents<int32_t, 2>(int(k), int(tile_rows));
  const auto a_str = TA
      ? array<int32_t, 2>{int(params.a_stride_m), int(params.a_stride_k)}
      : array<int32_t, 2>{int(params.a_stride_k), int(params.a_stride_m)};
  const auto b_ext = TB ? dextents<int32_t, 2>(int(k), int(tile_cols))
                        : dextents<int32_t, 2>(int(tile_cols), int(k));
  const auto b_str = TB
      ? array<int32_t, 2>{int(params.b_stride_k), int(params.b_stride_n)}
      : array<int32_t, 2>{int(params.b_stride_n), int(params.b_stride_k)};
  tensor<device T, dextents<int32_t, 2>, tensor_inline> a(a_ptr, a_ext, a_str);
  tensor<device T, dextents<int32_t, 2>, tensor_inline> b(b_ptr, b_ext, b_str);
  tensor<device T, dextents<int32_t, 2>, tensor_inline> out(
      out_ptr,
      dextents<int32_t, 2>(int(tile_cols), int(tile_rows)),
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

  if constexpr (metal::is_same_v<T, float>) {
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

template <typename T, int BM, int NSG, bool RELAXED, bool TA, bool TB>
kernel void grouped_mm_rows_mpp(
    device T* mat_a [[buffer(0)]],
    device T* mat_b [[buffer(1)]],
    device const int32_t* offsets [[buffer(2)]],
    device T* output [[buffer(3)]],
    constant GroupedMMParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint32_t tile_info[4];
  uint32_t group = 0;
  uint32_t row_start = 0;
  uint32_t tile_rows = 0;
  if (!grouped_mm_row_tile_broadcast<BM>(
          offsets,
          params,
          tgid.y,
          tid,
          tile_info,
          group,
          row_start,
          tile_rows)) {
    return;
  }

  const uint32_t col_start = static_cast<uint32_t>(tgid.x) * kGroupedMMTileN;
  const uint32_t tile_cols = min(kGroupedMMTileN, params.n - col_start);
  device T* a_ptr =
      mat_a + static_cast<uint64_t>(row_start) * params.a_stride_m;
  device T* b_ptr = mat_b +
      static_cast<uint64_t>(group) * params.b_batch_stride +
      static_cast<uint64_t>(col_start) * params.b_stride_n;
  device T* out_ptr = output +
      static_cast<uint64_t>(row_start) * params.out_stride_m +
      static_cast<uint64_t>(col_start) * params.out_stride_n;
  grouped_mm_mpp_gemm<T, BM, NSG, RELAXED, TA, TB>(
      a_ptr, b_ptr, out_ptr, tile_rows, tile_cols, params.k, params);
}

template <typename T, int BM, int NSG, bool RELAXED, bool TA, bool TB>
kernel void grouped_mm_k_mpp(
    device T* mat_a [[buffer(0)]],
    device T* mat_b [[buffer(1)]],
    device const int32_t* offsets [[buffer(2)]],
    device T* output [[buffer(3)]],
    constant GroupedMMParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const uint32_t group = static_cast<uint32_t>(tgid.z);
  const uint32_t k_end = grouped_mm_offset(offsets[group], params.k);
  const uint32_t raw_start =
      group == 0 ? 0 : grouped_mm_offset(offsets[group - 1], params.k);
  const uint32_t k_begin = min(raw_start, k_end);
  const uint32_t row_start = static_cast<uint32_t>(tgid.y) * BM;
  const uint32_t tile_rows =
      min(static_cast<uint32_t>(BM), params.m - row_start);
  const uint32_t col_start = static_cast<uint32_t>(tgid.x) * kGroupedMMTileN;
  const uint32_t tile_cols = min(kGroupedMMTileN, params.n - col_start);

  device T* a_ptr = mat_a +
      static_cast<uint64_t>(row_start) * params.a_stride_m +
      static_cast<uint64_t>(k_begin) * params.a_stride_k;
  device T* b_ptr = mat_b + static_cast<uint64_t>(k_begin) * params.b_stride_k +
      static_cast<uint64_t>(col_start) * params.b_stride_n;
  device T* out_ptr = output +
      static_cast<uint64_t>(group) * params.out_batch_stride +
      static_cast<uint64_t>(row_start) * params.out_stride_m +
      static_cast<uint64_t>(col_start) * params.out_stride_n;
  grouped_mm_mpp_gemm<T, BM, NSG, RELAXED, TA, TB>(
      a_ptr, b_ptr, out_ptr, tile_rows, tile_cols, k_end - k_begin, params);
}

#define INSTANTIATE_GROUPED_MM_MPP(DTYPE, BM, RELAXED, TA, TB, LAYOUT)        \
  template [[host_name("grouped_mm_rows_mpp_" #LAYOUT "_" #DTYPE "_bm" #BM)]] \
  kernel void                                                                 \
  grouped_mm_rows_mpp<DTYPE, BM, grouped_mm_simdgroups(BM), RELAXED, TA, TB>( \
      device DTYPE*,                                                          \
      device DTYPE*,                                                          \
      device const int32_t*,                                                  \
      device DTYPE*,                                                          \
      constant GroupedMMParams&,                                              \
      uint3,                                                                  \
      uint);                                                                  \
  template [[host_name("grouped_mm_k_mpp_" #LAYOUT "_" #DTYPE "_bm" #BM)]]    \
  kernel void                                                                 \
  grouped_mm_k_mpp<DTYPE, BM, grouped_mm_simdgroups(BM), RELAXED, TA, TB>(    \
      device DTYPE*,                                                          \
      device DTYPE*,                                                          \
      device const int32_t*,                                                  \
      device DTYPE*,                                                          \
      constant GroupedMMParams&,                                              \
      uint3)

// Layout suffix: one letter per operand (mat_a then mat_b), 'n' = row-major,
// 't' = column-major; the host picks the variant from the operand strides.
#define INSTANTIATE_GROUPED_MM_MPP_BM(DTYPE, BM, RELAXED)           \
  INSTANTIATE_GROUPED_MM_MPP(DTYPE, BM, RELAXED, false, false, nn); \
  INSTANTIATE_GROUPED_MM_MPP(DTYPE, BM, RELAXED, false, true, nt);  \
  INSTANTIATE_GROUPED_MM_MPP(DTYPE, BM, RELAXED, true, false, tn);  \
  INSTANTIATE_GROUPED_MM_MPP(DTYPE, BM, RELAXED, true, true, tt)

#define INSTANTIATE_GROUPED_MM_MPP_DTYPE(DTYPE, RELAXED) \
  INSTANTIATE_GROUPED_MM_MPP_BM(DTYPE, 16, RELAXED);     \
  INSTANTIATE_GROUPED_MM_MPP_BM(DTYPE, 32, RELAXED);     \
  INSTANTIATE_GROUPED_MM_MPP_BM(DTYPE, 64, RELAXED)

INSTANTIATE_GROUPED_MM_MPP_DTYPE(float, false);
INSTANTIATE_GROUPED_MM_MPP_DTYPE(half, true);
INSTANTIATE_GROUPED_MM_MPP_DTYPE(bfloat, true);

#endif

#define INSTANTIATE_GROUPED_MM(DTYPE, BM)                                 \
  template [[host_name("grouped_mm_rows_" #DTYPE "_bm" #BM)]] kernel void \
  grouped_mm_rows<                                                        \
      DTYPE,                                                              \
      BM,                                                                 \
      kGroupedMMTileN,                                                    \
      kGroupedMMTileK,                                                    \
      grouped_mm_simdgroups(BM) / 2,                                      \
      2>(                                                                 \
      device const DTYPE*,                                                \
      device const DTYPE*,                                                \
      device const int32_t*,                                              \
      device DTYPE*,                                                      \
      constant GroupedMMParams&,                                          \
      uint3,                                                              \
      uint,                                                               \
      uint);                                                              \
  template [[host_name("grouped_mm_k_" #DTYPE "_bm" #BM)]] kernel void    \
  grouped_mm_k<                                                           \
      DTYPE,                                                              \
      BM,                                                                 \
      kGroupedMMTileN,                                                    \
      kGroupedMMTileK,                                                    \
      grouped_mm_simdgroups(BM) / 2,                                      \
      2>(                                                                 \
      device const DTYPE*,                                                \
      device const DTYPE*,                                                \
      device const int32_t*,                                              \
      device DTYPE*,                                                      \
      constant GroupedMMParams&,                                          \
      uint3,                                                              \
      uint,                                                               \
      uint)

#define INSTANTIATE_GROUPED_MM_DTYPE(DTYPE) \
  INSTANTIATE_GROUPED_MM(DTYPE, 16);        \
  INSTANTIATE_GROUPED_MM(DTYPE, 32);        \
  INSTANTIATE_GROUPED_MM(DTYPE, 64)

INSTANTIATE_GROUPED_MM_DTYPE(float);
INSTANTIATE_GROUPED_MM_DTYPE(half);
INSTANTIATE_GROUPED_MM_DTYPE(bfloat);
