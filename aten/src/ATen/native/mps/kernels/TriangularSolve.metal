#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <c10/metal/reduction_utils.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

// op(A)(i, j): row-major (n x n) A, optionally transposed and/or conjugated.
template <typename T>
inline T tri_opA(
    device const T* Ab,
    uint i,
    uint j,
    uint n,
    bool transpose,
    bool conj) {
  T v = transpose ? Ab[j * n + i] : Ab[i * n + j];
  return conj ? c10::metal::conj(v) : v;
}

// Offset of element (row, col) of an n x k right-hand side or solution.
inline ulong rhs_offset(
    uint row,
    ulong col,
    constant TriangularSolveParams& p,
    bool col_major) {
  return col_major ? row + col * p.n : ulong(row) * p.k + col;
}

// Batched triangular solve by forward/back substitution. One threadgroup owns
// one right-hand side and walks the n substitution steps serially; the dot
// product against the already-solved prefix is split across the group and
// reduced. The caller folds the side into the transpose so only op(A) X = B is
// handled here, and keeps n small enough that the prefix always fits in
// threadgroup memory; op itself stays in the kernel so that a transposed or
// conjugated solve does not have to materialize an n x n copy.
// Complex support comes from the c10::metal mul/div helpers, no-ops for real T.
template <typename T>
kernel void triangular_solve(
    device const T* A [[buffer(0)]],
    device const T* B [[buffer(1)]],
    device T* X [[buffer(2)]],
    constant TriangularSolveParams& p [[buffer(3)]],
    threadgroup T* xs [[threadgroup(0)]],
    threadgroup T* red [[threadgroup(1)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint nsimd [[simdgroups_per_threadgroup]]) {
  const uint n = p.n;
  const uint k = p.k;
  if (tgid >= p.nbatch * k) {
    return;
  }
  const uint batch = tgid / k;
  const uint vec = tgid % k;
  device const T* Ab = A + batch * n * n;
  const bool tr = p.transpose;
  const bool cj = p.conj;
  // A is upper before op; a transpose flips the effective triangle, and a lower
  // one substitutes forward.
  const bool forward = p.upper == p.transpose;
  // Keep layout handling off the serial per-step path below.
  device const T* b = B + batch * n * k + (p.b_col_major ? vec * n : vec);
  device T* x = X + batch * n * k + (p.x_col_major ? vec * n : vec);
  const uint b_stride = p.b_col_major ? 1 : k;
  const uint x_stride = p.x_col_major ? 1 : k;

  for (uint step = 0; step < n; ++step) {
    const uint t = forward ? step : n - 1 - step;
    const uint s_begin = forward ? 0 : t + 1;
    const uint s_end = forward ? t : n;

    T part = T(0);
    for (uint s = s_begin + lid; s < s_end; s += tg_size) {
      part = part + c10::metal::mul(tri_opA(Ab, t, s, n, tr, cj), xs[s]);
    }
    part = c10::metal::simd_sum(part);
    if (sg_lane == 0) {
      red[sg_id] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
      T sum = b[t * b_stride];
      for (uint s = 0; s < nsimd; ++s) {
        sum = sum - red[s];
      }
      const T xt =
          p.unit ? sum : c10::metal::div(sum, tri_opA(Ab, t, t, n, tr, cj));
      xs[t] = xt;
      x[t * x_stride] = xt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define INSTANTIATE_TRIANGULAR_SOLVE(DTYPE)                      \
  template [[host_name("triangular_solve_" #DTYPE)]] kernel void \
  triangular_solve<DTYPE>(                                       \
      device const DTYPE* A [[buffer(0)]],                       \
      device const DTYPE* B [[buffer(1)]],                       \
      device DTYPE* X [[buffer(2)]],                             \
      constant TriangularSolveParams& p [[buffer(3)]],           \
      threadgroup DTYPE* xs [[threadgroup(0)]],                  \
      threadgroup DTYPE* red [[threadgroup(1)]],                 \
      uint tgid [[threadgroup_position_in_grid]],                \
      uint lid [[thread_position_in_threadgroup]],               \
      uint tg_size [[threads_per_threadgroup]],                  \
      uint sg_lane [[thread_index_in_simdgroup]],                \
      uint sg_id [[simdgroup_index_in_threadgroup]],             \
      uint nsimd [[simdgroups_per_threadgroup]]);

INSTANTIATE_TRIANGULAR_SOLVE(float);
INSTANTIATE_TRIANGULAR_SOLVE(float2);

inline uint triangular_tile_offset(uint row, uint col) {
  return (row * (row + 1) / 2 + col) * kTriangularSolveTileSize *
      kTriangularSolveTileSize;
}

inline uint triangular_elem_offset(uint row, uint col) {
  constexpr uint t = kTriangularSolveTileSize;
  return triangular_tile_offset(row / t, col / t) + row % t * t + col % t;
}

// x = inv(L) b for every RHS tile; returns whether this thread saw a
// nonfinite output. The MPP overload is selected for the mpp_ kernel names.
template <uint n, bool general>
METAL_FUNC bool apply_triangular_inverse(
    bool_constant<false>,
    threadgroup const float* inverse,
    device const float* b,
    device float* x,
    constant TriangularSolveParams& p,
    uint lane,
    uint sg) {
  constexpr uint tile_size = kTriangularSolveTileSize;
  constexpr uint groups = n / tile_size;
  const bool reverse = general && (p.upper != p.transpose);
  using tile_index_t = conditional_t<general, ulong, uint>;
  const auto col_tiles =
      c10::metal::ceil_div(tile_index_t(p.k), tile_index_t(tile_size));
  // Same 8x8 fragment lane mapping as GroupedMM.metal.
  const uint quad = lane / 4;
  const uint frag_row = (quad & 4) + ((lane / 2) % 4);
  const uint frag_col = (quad & 2) * 2 + (lane % 2) * 2;
  bool nonfinite = false;
  for (tile_index_t tile = sg; tile < groups * col_tiles; tile += groups) {
    // Each pass of the simdgroups covers one row block of x when it is
    // row-major, or one column block when it is column-major, so that stores
    // walk the contiguous dimension. The column-major order rotates the rows
    // so every simdgroup does the same amount of work.
    const tile_index_t col_tile =
        p.x_col_major ? tile / groups : tile % col_tiles;
    const uint row =
        p.x_col_major ? (tile + col_tile) % groups : tile / col_tiles;
    const uint col = col_tile * tile_size;
    const bool scalar_io = general && (reverse || p.k - col < tile_size);
    simdgroup_float8x8 lhs_tile, rhs_tile, result_tile(0);
    for (uint j = 0; j <= row; ++j) {
      simdgroup_load(
          lhs_tile, inverse + triangular_tile_offset(row, j), tile_size);
      if (scalar_io) {
        const uint r = j * tile_size + frag_row;
#pragma unroll
        for (uint i = 0; i < 2; ++i) {
          const uint c = col + frag_col + i;
          rhs_tile.thread_elements()[i] = c < p.k
              ? b[rhs_offset(reverse ? n - 1 - r : r, c, p, p.b_col_major)]
              : 0;
        }
      } else {
        simdgroup_load(
            rhs_tile,
            b + rhs_offset(j * tile_size, col, p, p.b_col_major),
            p.b_col_major ? n : p.k,
            ulong2(0),
            p.b_col_major);
      }
      simdgroup_multiply_accumulate(
          result_tile, lhs_tile, rhs_tile, result_tile);
    }
#pragma unroll
    for (uint i = 0; i < tile_size * tile_size / c10::metal::simdgroup_size;
         ++i) {
      nonfinite |= !isfinite(result_tile.thread_elements()[i]);
    }
    if (scalar_io) {
      const uint r = row * tile_size + frag_row;
#pragma unroll
      for (uint i = 0; i < 2; ++i) {
        const uint c = col + frag_col + i;
        if (c < p.k) {
          x[rhs_offset(reverse ? n - 1 - r : r, c, p, p.x_col_major)] =
              result_tile.thread_elements()[i];
        }
      }
    } else {
      simdgroup_store(
          result_tile,
          x + rhs_offset(row * tile_size, col, p, p.x_col_major),
          p.x_col_major ? n : p.k,
          ulong2(0),
          p.x_col_major);
    }
  }
  return nonfinite;
}

#if C10_METAL_HAS_MPP
// MPP tensors need a unit stride along their first dimension, so a
// column-major x is produced as x^T = b^T inv(L)^T, whose tiles are row-major.
// The matrix inputs are then (col, row) of x and the operands swap sides.
// b is loaded in its own layout and transposed by the matmul when it differs
// from x's.
template <uint n, bool general, bool x_col_major, bool b_col_major>
METAL_FUNC bool apply_triangular_inverse_mpp(
    threadgroup const float* inverse,
    device const float* b,
    device float* x,
    constant TriangularSolveParams& p,
    uint sg) {
  constexpr uint groups = n / kTriangularSolveTileSize;
  const bool reverse = general && (p.upper != p.transpose);
  bool nonfinite = false;
  constexpr uint mpp_rows = kTriangularSolveMppRows;
  constexpr uint mpp_cols = kTriangularSolveMppCols;
  constexpr bool transpose_b = x_col_major != b_col_major;
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      x_col_major ? mpp_cols : mpp_rows,
      x_col_major ? mpp_rows : mpp_cols,
      mpp_rows,
      x_col_major && transpose_b,
      !x_col_major && transpose_b,
      false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
  const uint col_tiles = c10::metal::ceil_div(p.k, mpp_cols);
  constexpr uint row_tiles = n / mpp_rows;
  const int x_ld = x_col_major ? int(n) : int(p.k);
  const int b_ld = b_col_major ? int(n) : int(p.k);
  for (uint tile = sg; tile < row_tiles * col_tiles; tile += groups) {
    // Same traversal as the simdgroup_matrix overload, but a pass spans
    // groups / row_tiles column tiles, so rotate by the pass, not the column.
    const uint col_tile = x_col_major ? tile / row_tiles : tile % col_tiles;
    const uint row =
        (x_col_major ? (tile + tile / groups) % row_tiles : tile / col_tiles) *
        mpp_rows;
    const uint col = col_tile * mpp_cols;
    const int cols = min(mpp_cols, p.k - col);
    // (unit-stride, outer) extents of a row block by column block tile
    const auto row_inner = dextents<int32_t, 2>(mpp_rows, cols);
    const auto col_inner = dextents<int32_t, 2>(cols, mpp_rows);
    tensor<device float, dextents<int32_t, 2>, tensor_inline> out(
        x + rhs_offset(row, col, p, x_col_major),
        x_col_major ? row_inner : col_inner,
        array<int32_t, 2>{1, x_ld});
    auto lhs =
        op.template get_left_input_cooperative_tensor<float, float, float>();
    auto rhs =
        op.template get_right_input_cooperative_tensor<float, float, float>();
    auto result = op.template get_destination_cooperative_tensor<
        decltype(lhs),
        decltype(rhs),
        float>();
    // inv(L) is the left operand of a row-major x and inv(L)^T the right
    // operand of a column-major one; b (or b^T) is the other.
    thread auto& inv_op = [&]() -> thread auto& {
      if constexpr (x_col_major) {
        return rhs;
      } else {
        return lhs;
      }
    }();
    thread auto& b_op = [&]() -> thread auto& {
      if constexpr (x_col_major) {
        return lhs;
      } else {
        return rhs;
      }
    }();
#pragma unroll
    for (uint16_t i = 0; i < result.get_capacity(); ++i) {
      result[i] = 0;
    }
    for (uint j = 0; j <= row; j += mpp_rows) {
#pragma unroll
      for (uint16_t i = 0; i < inv_op.get_capacity(); ++i) {
        auto idx = inv_op.get_multidimensional_index(i);
        const uint r = row + idx[x_col_major ? 0 : 1];
        const uint c = j + idx[x_col_major ? 1 : 0];
        inv_op[i] = r >= c ? inverse[triangular_elem_offset(r, c)] : 0;
      }
      if (!reverse) {
        tensor<device float, dextents<int32_t, 2>, tensor_inline> in(
            const_cast<device float*>(b + rhs_offset(j, col, p, b_col_major)),
            b_col_major ? row_inner : col_inner,
            array<int32_t, 2>{1, b_ld});
        b_op.load(in);
      } else {
#pragma unroll
        for (uint16_t i = 0; i < b_op.get_capacity(); ++i) {
          // Indices follow the stored layout, which is b's.
          auto idx = b_op.get_multidimensional_index(i);
          const uint r = j + idx[b_col_major ? 0 : 1];
          const uint c = idx[b_col_major ? 1 : 0];
          b_op[i] = int(c) < cols
              ? b[rhs_offset(reverse ? n - 1 - r : r, col + c, p, b_col_major)]
              : 0;
        }
      }
      op.run(lhs, rhs, result);
    }
#pragma unroll
    for (uint16_t i = 0; i < result.get_capacity(); ++i) {
      nonfinite |= !isfinite(result[i]);
    }
    if (reverse) {
#pragma unroll
      for (uint16_t i = 0; i < result.get_capacity(); ++i) {
        auto idx = result.get_multidimensional_index(i);
        const uint r = row + idx[x_col_major ? 0 : 1];
        const uint c = idx[x_col_major ? 1 : 0];
        if (int(c) < cols) {
          x[rhs_offset(n - 1 - r, col + c, p, x_col_major)] = result[i];
        }
      }
    } else {
      result.store(out);
    }
  }
  return nonfinite;
}

template <uint n, bool general>
METAL_FUNC bool apply_triangular_inverse(
    bool_constant<true>,
    threadgroup const float* inverse,
    device const float* b,
    device float* x,
    constant TriangularSolveParams& p,
    uint,
    uint sg) {
  if (p.x_col_major) {
    return p.b_col_major
        ? apply_triangular_inverse_mpp<n, general, true, true>(
              inverse, b, x, p, sg)
        : apply_triangular_inverse_mpp<n, general, true, false>(
              inverse, b, x, p, sg);
  }
  return p.b_col_major ? apply_triangular_inverse_mpp<n, general, false, true>(
                             inverse, b, x, p, sg)
                       : apply_triangular_inverse_mpp<n, general, false, false>(
                             inverse, b, x, p, sg);
}
#endif

// Invert small triangular matrices in shared memory, then reuse the inverse
// across all RHS columns. Each diagonal 8x8 block is inverted independently;
// larger blocks use inv(L) = [[inv(A), 0], [-inv(D) C inv(A), inv(D)]].
// general=false specializes unit lower matrices and aligned RHS columns.
template <uint n, bool use_mpp, bool general>
[[max_total_threads_per_threadgroup(
    n / kTriangularSolveTileSize * c10::metal::simdgroup_size)]]
kernel void triangular_solve_small(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* x [[buffer(2)]],
    constant TriangularSolveParams& p [[buffer(3)]],
    uint batch [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint tile_size = kTriangularSolveTileSize;
  static_assert(tile_size == 8, "simdgroup_float8x8 tiles");
  constexpr uint groups = n / tile_size;
  constexpr uint threads = groups * c10::metal::simdgroup_size;
  threadgroup float inverse[groups * (groups + 1) / 2 * tile_size * tile_size];
  threadgroup float tmp[n * n / 4];
  threadgroup uint nonfinite_groups[groups];
  a += ulong(batch) * n * n;
  b += ulong(batch) * n * p.k;
  x += ulong(batch) * n * p.k;
  // Reversing rows and columns turns an effective upper triangle into lower.
  const bool transpose = general && p.transpose;
  const bool unit = !general || p.unit;
  const bool reverse = general && (p.upper != p.transpose);
  // Keep off-diagonal blocks negated until they are replaced by their inverse.
  // The unused triangle and implicit unit diagonal must not be read.
  for (uint i = tid; i < n * n; i += threads) {
    const uint row = i / n;
    const uint col = i % n;
    if (row / tile_size >= col / tile_size) {
      const uint r = reverse ? n - 1 - row : row;
      const uint c = reverse ? n - 1 - col : col;
      float value = 0;
      if (row > col) {
        value = -tri_opA(a, r, c, n, transpose, p.conj);
      } else if (row == col) {
        value = unit ? 1 : tri_opA(a, r, c, n, transpose, p.conj);
      }
      inverse[triangular_elem_offset(row, col)] = value;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float v[tile_size];
  const uint base = triangular_tile_offset(sg, sg);
#pragma unroll
  for (uint i = 0; i < tile_size; ++i) {
    v[i] = i == lane ? 1 : 0;
#pragma unroll
    for (uint j = 0; j < i; ++j) {
      v[i] = fma(inverse[base + i * tile_size + j], v[j], v[i]);
    }
    if (!unit) {
      v[i] /= inverse[base + i * tile_size + i];
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < tile_size) {
#pragma unroll
    for (uint i = 0; i < tile_size; ++i) {
      inverse[base + i * tile_size + lane] = v[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = tile_size; s <= n / 2; s *= 2) {
    const uint tiles_per_side = s / tile_size;
    const uint tiles_per_block = tiles_per_side * tiles_per_side;
    const uint tiles = (n / (2 * s)) * tiles_per_block;
    for (uint tile = sg; tile < tiles; tile += groups) {
      const uint block = tile / tiles_per_block;
      const uint row = tile % tiles_per_block / tiles_per_side;
      const uint col = tile % tiles_per_side;
      const uint lo = block * 2 * tiles_per_side;
      const uint hi = lo + tiles_per_side;
      threadgroup float* tmp_row = tmp + (block * s + row * tile_size) * s;
      simdgroup_float8x8 lhs_tile, rhs_tile, result_tile(0);
      for (uint j = 0; j <= row; ++j) {
        simdgroup_load(
            lhs_tile,
            inverse + triangular_tile_offset(hi + row, hi + j),
            tile_size);
        simdgroup_load(
            rhs_tile,
            inverse + triangular_tile_offset(hi + j, lo + col),
            tile_size);
        simdgroup_multiply_accumulate(
            result_tile, lhs_tile, rhs_tile, result_tile);
      }
      simdgroup_store(result_tile, tmp_row + col * tile_size, s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tile = sg; tile < tiles; tile += groups) {
      const uint block = tile / tiles_per_block;
      const uint row = tile % tiles_per_block / tiles_per_side;
      const uint col = tile % tiles_per_side;
      const uint lo = block * 2 * tiles_per_side;
      const uint hi = lo + tiles_per_side;
      threadgroup const float* tmp_row =
          tmp + (block * s + row * tile_size) * s;
      simdgroup_float8x8 lhs_tile, rhs_tile, result_tile(0);
      for (uint j = col; j < tiles_per_side; ++j) {
        simdgroup_load(lhs_tile, tmp_row + j * tile_size, s);
        simdgroup_load(
            rhs_tile,
            inverse + triangular_tile_offset(lo + j, lo + col),
            tile_size);
        simdgroup_multiply_accumulate(
            result_tile, lhs_tile, rhs_tile, result_tile);
      }
      simdgroup_store(
          result_tile,
          inverse + triangular_tile_offset(hi + row, lo + col),
          tile_size);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const bool nonfinite = apply_triangular_inverse<n, general>(
      bool_constant<use_mpp>{}, inverse, b, x, p, lane, sg);
  const bool needs_substitution =
      c10::metal::threadgroup_max(
          nonfinite_groups, uint(nonfinite), tid, threads) != 0;
  // An inverse can overflow even when the solution is finite. Substitution
  // also preserves the dependency order for non-finite matrix or RHS entries.
  if (needs_substitution) {
    threadgroup_barrier(mem_flags::mem_device);
    for (ulong col = tid; col < p.k; col += threads) {
      for (uint step = 0; step < n; ++step) {
        const uint row = reverse ? n - 1 - step : step;
        float value = b[rhs_offset(row, col, p, p.b_col_major)];
        for (uint j = 0; j < step; ++j) {
          const uint prior = reverse ? n - 1 - j : j;
          value =
              fma(-tri_opA(a, row, prior, n, transpose, p.conj),
                  x[rhs_offset(prior, col, p, p.x_col_major)],
                  value);
        }
        if (!unit) {
          value /= tri_opA(a, row, row, n, transpose, p.conj);
        }
        x[rhs_offset(row, col, p, p.x_col_major)] = value;
      }
    }
  }
}

#define INSTANTIATE_TRIANGULAR_SOLVE_SMALL(N, SUFFIX, MPP, GENERAL)       \
  template [[host_name("triangular_solve_small_" SUFFIX #N)]] kernel void \
  triangular_solve_small<N, MPP, GENERAL>(                                \
      device const float*,                                                \
      device const float*,                                                \
      device float*,                                                      \
      constant TriangularSolveParams&,                                    \
      uint,                                                               \
      uint,                                                               \
      uint,                                                               \
      uint);

INSTANTIATE_TRIANGULAR_SOLVE_SMALL(16, "", false, false);
INSTANTIATE_TRIANGULAR_SOLVE_SMALL(32, "", false, false);
INSTANTIATE_TRIANGULAR_SOLVE_SMALL(64, "", false, false);
INSTANTIATE_TRIANGULAR_SOLVE_SMALL(16, "general_", false, true);
INSTANTIATE_TRIANGULAR_SOLVE_SMALL(32, "general_", false, true);
INSTANTIATE_TRIANGULAR_SOLVE_SMALL(64, "general_", false, true);

#if C10_METAL_HAS_MPP
INSTANTIATE_TRIANGULAR_SOLVE_SMALL(16, "mpp_", true, false);
INSTANTIATE_TRIANGULAR_SOLVE_SMALL(32, "mpp_", true, false);
INSTANTIATE_TRIANGULAR_SOLVE_SMALL(64, "mpp_", true, false);
INSTANTIATE_TRIANGULAR_SOLVE_SMALL(16, "mpp_general_", true, true);
INSTANTIATE_TRIANGULAR_SOLVE_SMALL(32, "mpp_general_", true, true);
INSTANTIATE_TRIANGULAR_SOLVE_SMALL(64, "mpp_general_", true, true);
#endif
