#pragma once

#include <c10/metal/common.h>

C10_METAL_CONSTEXPR uint32_t kMatmulTileN = 64;
C10_METAL_CONSTEXPR uint32_t kMatmulTileK = 16;

// Tuned on M5 Pro: the 256-wide MPP tile wants eight SIMD groups, narrower
// tiles two or four depending on their row count.
inline constexpr uint32_t matmul_simdgroups(
    uint32_t tile_rows,
    uint32_t tile_cols) {
  return tile_cols >= 256 ? 8 : tile_rows >= 64 ? 4 : 2;
}

template <typename idx_t>
struct MatmulParams {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  idx_t a_stride_m;
  idx_t a_stride_k;
  idx_t b_stride_k;
  idx_t b_stride_n;
  idx_t out_stride_m;
  idx_t out_stride_n;
};

#ifdef __METAL__
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;
using c10::metal::simdgroup_size;

// simdgroup_matrix fragments are fixed 8x8.
constant constexpr int kMatmulFragDim = 8;
// Threadgroup tile rows are padded by one fragment width so consecutive
// simdgroup_load rows do not land on the same threadgroup memory banks.
constant constexpr int kMatmulTilePad = kMatmulFragDim;

template <typename T>
struct MatmulTile {
  device const T* a;
  device const T* b;
  device T* out;
  uint32_t rows;
  uint32_t cols;
  uint32_t k;
};

// simdgroup-matrix tile GEMM: accumulates out[:rows, :cols] = A[:rows, :k] *
// B[:k, :cols] in fp32 and writes the tile back.
template <typename T, int BM, typename idx_t>
inline void matmul_tile_gemm(
    thread const MatmulTile<T>& tile,
    constant MatmulParams<idx_t>& params,
    threadgroup T* a_tile,
    threadgroup T* b_tile,
    uint simd_group,
    uint simd_lane) {
  constexpr int BN = kMatmulTileN;
  constexpr int BK = kMatmulTileK;
  constexpr int WN = 2;
  constexpr int WM = matmul_simdgroups(BM, BN) / WN;
  constexpr int threads = WM * WN * simdgroup_size;
  constexpr int warp_m_tile = BM / WM;
  constexpr int warp_n_tile = BN / WN;
  constexpr int thread_tiles_m = warp_m_tile / kMatmulFragDim;
  constexpr int thread_tiles_n = warp_n_tile / kMatmulFragDim;
  constexpr int lda = BK + kMatmulTilePad;
  constexpr int ldb = BN + kMatmulTilePad;
  static_assert(BM % (kMatmulFragDim * WM) == 0, "invalid M tile");
  static_assert(BN % (kMatmulFragDim * WN) == 0, "invalid N tile");
  static_assert(BK % kMatmulFragDim == 0, "invalid K tile");

  const uint32_t tid = simd_group * simdgroup_size + simd_lane;
  device const T* mat_a = tile.a;
  device const T* mat_b = tile.b;
  const uint32_t tile_rows = tile.rows;
  const uint32_t tile_cols = tile.cols;
  const uint32_t k_end = tile.k;

  simdgroup_matrix<float, kMatmulFragDim, kMatmulFragDim> accum[thread_tiles_m]
                                                               [thread_tiles_n];
  for (int i = 0; i < thread_tiles_m; ++i) {
    for (int j = 0; j < thread_tiles_n; ++j) {
      accum[i][j] =
          simdgroup_matrix<float, kMatmulFragDim, kMatmulFragDim>(0.0f);
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
    for (int k = 0; k < BK; k += kMatmulFragDim) {
      simdgroup_matrix<T, kMatmulFragDim, kMatmulFragDim>
          a_frag[thread_tiles_m];
      simdgroup_matrix<T, kMatmulFragDim, kMatmulFragDim>
          b_frag[thread_tiles_n];
      for (int i = 0; i < thread_tiles_m; ++i) {
        simdgroup_load(
            a_frag[i], &a_tile[(warp_m + i * kMatmulFragDim) * lda + k], lda);
      }
      for (int j = 0; j < thread_tiles_n; ++j) {
        simdgroup_load(
            b_frag[j], &b_tile[k * ldb + warp_n + j * kMatmulFragDim], ldb);
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
    const uint32_t tile_row = warp_m + i * kMatmulFragDim + frag_m;
    if (tile_row >= tile_rows) {
      continue;
    }
    for (int j = 0; j < thread_tiles_n; ++j) {
      for (int element = 0; element < 2; ++element) {
        const uint32_t tile_col =
            warp_n + j * kMatmulFragDim + frag_n + element;
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

#if C10_METAL_HAS_MPP

// matmul2d tile op. TA/TB name the operand layouts ('t' = column-major); every
// tensor is presented to matmul2d with a unit innermost stride and the matching
// transpose flag, which matmul2d requires (it ignores layouts that contradict
// the descriptor). half/bfloat multiply at relaxed precision; accumulation
// stays fp32 via the destination cooperative tensor.
template <typename T, int BM, int BN, bool TA, bool TB>
inline void matmul_mpp_gemm(
    thread const MatmulTile<T>& tile,
    constant MatmulParams<uint64_t>& params) {
  constexpr int NSG = matmul_simdgroups(BM, BN);
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

#endif
#endif
