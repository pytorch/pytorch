#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <cutlass/functional.h>
#include <cutlass/gemm/warp/mma_simt_tile_iterator.h>
#include <cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h>
#include <cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h>
#include <cutlass/matrix_shape.h>

namespace {

static CUTLASS_DEVICE float atomicMaxFloat(float* addr, float value) {
  // source: https://stackoverflow.com/a/51549250
  return (value >= 0)
      ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
      : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
}
} // namespace

/* Iterates on the accumulator and corresponding position on result matrix

(1) Update `mi[r]` to the max value of the row `r`
(2) In a second iteration do the following:
    (a) accum   <- exp(accum - mi)
    (b) m_prime <- exp(m_prime - mi)
    (c) s_prime <- s_prime * m_prime + sum(accum)

All of this is done on registers, before we store all of this
on shared memory for the next matmul with Value.

We have multiple implementations, because each configuration has a different way
of iterating in the accumulators.
*/

template <typename BASE, typename T, typename accum_t, int kWarpSize>
struct RegisterOps {
  template <
      int kQueriesPerBlock,
      bool kFullColumns,
      bool kIsFirst,
      bool kKeepOutputInRF>
  CUTLASS_DEVICE static void update(
      typename T::Fragment& frag_o, // output so far
      typename T::Fragment& frag,
      cutlass::Array<accum_t, kQueriesPerBlock>& mi,
      cutlass::Array<accum_t, kQueriesPerBlock>& m_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& s_prime,
      int8_t lane_id,
      int8_t thread_id,
      int8_t warp_id,
      int16_t max_col,
      typename T::TensorCoord const& tile_offset,
      float scaling) {
    // Convert to `accum_t` (rather than double)
    constexpr float kLog2e = M_LOG2E;
    if (!kIsFirst) {
      if (thread_id < kQueriesPerBlock) {
        m_prime[thread_id] = mi[thread_id];
      }
      __syncthreads();
    }

    auto lane_offset = BASE::get_lane_offset(lane_id, warp_id, tile_offset);

    // First update `mi` to the max per-row
    {
      accum_t max;
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) { max = -std::numeric_limits<accum_t>::infinity(); },
          [&](int accum_m, int accum_n, int idx) {
            if (kFullColumns || accum_n < max_col) {
              max = std::max(max, frag[idx]);
            }
          },
          [&](int accum_m) {
            // Having 4x atomicMax seems faster than reduce within warp
            // first...
            atomicMaxFloat(&mi[accum_m], max * scaling);
          });
    }
    frag = cutlass::multiplies<typename T::Fragment>()(scaling * kLog2e, frag);

    // Make sure we all share the update values for `mi`
    __syncthreads();

    if (thread_id < kQueriesPerBlock) {
      auto m_prime_exp = exp2f(kLog2e * (m_prime[thread_id] - mi[thread_id]));
      m_prime[thread_id] = m_prime_exp;
      s_prime[thread_id] *= m_prime_exp;
    }
    __syncthreads(); // Update output fragments
    if (kKeepOutputInRF && !kIsFirst) {
      accum_t mp;
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) { mp = m_prime[accum_m]; },
          [&](int accum_m, int accum_n, int idx) { frag_o[idx] *= mp; },
          [&](int accum_m) {});
      __syncthreads();
    }
    // Update accum_m, accum_n, ...
    {
      accum_t mi_row, total_row;
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) { mi_row = kLog2e * mi[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag[idx] = (kFullColumns || accum_n < max_col)
                ? exp2f(frag[idx] - mi_row)
                : accum_t(0.0);
          },
          [&](int accum_m) {});
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) { total_row = 0.0; },
          [&](int accum_m, int accum_n, int idx) { total_row += frag[idx]; },
          [&](int accum_m) {
            if (BASE::reduceSameRow(
                    lane_id, total_row, [](accum_t a, accum_t b) {
                      return a + b;
                    })) {
              atomicAdd(&s_prime[accum_m], total_row);
            }
          });
    }
  }
};

template <typename T, typename accum_t, int kWarpSize>
struct AttentionScalingCoefsUpdaterSm80
    : RegisterOps<
          AttentionScalingCoefsUpdaterSm80<T, accum_t, kWarpSize>,
          T,
          accum_t,
          kWarpSize> {
  static_assert(
      std::is_same<typename T::Layout, cutlass::layout::RowMajor>::value, "");

  using Policy = typename T::Policy;
  using InstructionShape = typename T::InstructionShape;
  using OpDelta = typename T::OpDelta;
  using Shape = typename T::Shape;
  static int const kElementsPerAccess = InstructionShape::kN / 4;
  static int const kRowsPerTile = 8;
  static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

  static cutlass::MatrixCoord CUTLASS_DEVICE get_lane_offset(
      int8_t lane_id,
      int8_t warp_id,
      typename T::TensorCoord const& tile_offset) {
    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);
    return cutlass::MatrixCoord(
        quad + tile_offset.row() * Shape::kRow,
        lane_in_quad * kElementsPerAccess +
            tile_offset.column() * Shape::kColumn);
  }

  template <typename FA, typename FB, typename FC>
  CUTLASS_DEVICE static void iterateRows(
      cutlass::MatrixCoord& lane_offset,
      FA beginRow,
      FB op,
      FC endRow) {
    // See cutlass/gemm/warp/mma_tensor_op_tile_iterator.h
    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int row = 0; row < kAccumulatorRows; ++row) {
        int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
            row * kRowsPerTile + lane_offset.row();
        beginRow(accum_m);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
          int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
              (mma_n * Policy::MmaIterations::kRow + mma_m);
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn +
                col + lane_offset.column();
            int idx = mma_accum_start + row * kElementsPerAccess + col;
            op(accum_m, accum_n, idx);
          }
        }

        endRow(accum_m);
      }
    }
  }

  template <typename DT, typename F>
  CUTLASS_DEVICE static bool reduceSameRow(int lane_id, DT& myValue, F fn) {
    // In each warp, 4 threads will work on the same row
    // - the ones with the same `quad`
    auto otherV = __shfl_xor_sync(0xffffffff, myValue, 1);
    myValue = fn(myValue, otherV);
    otherV = __shfl_xor_sync(0xffffffff, myValue, 2);
    myValue = fn(myValue, otherV);
    int lane_in_quad = (lane_id & 3);
    return lane_in_quad == 0;
  };
};

// cutlass::gemm::warp::MmaVoltaTensorOpAccumulatorTileIterator<cutlass::MatrixShape<32,
// 32>, float, cutlass::layout::RowMajor, cutlass::gemm::GemmShape<16, 16, 4>,
// cutlass::MatrixShape<1, 1>> See
// cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h
template <typename T, typename accum_t, int kWarpSize>
struct AttentionScalingCoefsUpdaterVolta
    : RegisterOps<
          AttentionScalingCoefsUpdaterVolta<T, accum_t, kWarpSize>,
          T,
          accum_t,
          kWarpSize> {
  static_assert(
      std::is_same<typename T::Layout, cutlass::layout::RowMajor>::value, "");

  using Policy = typename T::Policy;
  using InstructionShape = typename T::InstructionShape;
  using OpDelta = typename T::OpDelta;
  using Shape = typename T::Shape;
  using Element = accum_t;

  static int const kElementsPerPartial = 4;
  using EleShapePerPatial = typename cutlass::platform::conditional<
      cutlass::platform::is_same<Element, float>::value,
      cutlass::MatrixShape<2, 2>,
      cutlass::MatrixShape<1, 4>>::type;
  static int const kElementsPerMma = 8;
  static int const kAccumulatorPatials = 2;
  using QuadShapePerPatialMma = cutlass::MatrixShape<4, 4>;

  static cutlass::MatrixCoord CUTLASS_DEVICE get_lane_offset(
      int8_t lane_id,
      int8_t warp_id,
      typename T::TensorCoord const& tile_offset) {
    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);
    int accum_m, accum_n;

    if (cutlass::platform::is_same<Element, float>::value) {
      // (quad[2],quad[0])+lane_in_quad[0]
      accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 + (lane_in_quad & 1);
      // (quad[1])+lane_in_quad[1]
      accum_n =
          ((quad >> 1) & 0x1) * kElementsPerPartial * kAccumulatorPatials +
          (lane_in_quad & 2);
    } else {
      accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 +
          lane_in_quad; // (quad[2],quad[0])
      accum_n = ((quad >> 1) & 0x1) * kElementsPerPartial * kAccumulatorPatials;
    }
    return cutlass::MatrixCoord(
        accum_m + tile_offset.row() * Shape::kRow,
        accum_n + tile_offset.column() * Shape::kColumn);
  }

  template <typename DT, typename F>
  CUTLASS_DEVICE static bool reduceSameRow(int lane_id, DT& myValue, F fn) {
    static_assert(
        cutlass::platform::is_same<Element, float>::value,
        "update to support non-float accum");
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16
    // T0 & T2 share same line within a quad
    auto otherV = __shfl_xor_sync(0xffffffff, myValue, 1 << 1);
    myValue = fn(myValue, otherV);
    // quad 0 and quad 2 are on the same lines
    otherV = __shfl_xor_sync(0xffffffff, myValue, 1 << 3);
    myValue = fn(myValue, otherV);
    return (lane_id & ((1 << 1) | (1 << 3))) == 0;
  };

  template <typename FA, typename FB, typename FC>
  CUTLASS_DEVICE static void iterateRows(
      cutlass::MatrixCoord& lane_offset,
      FA beginRow,
      FB op,
      FC endRow) {
    CUTLASS_PRAGMA_UNROLL
    for (int tile_m = 0; tile_m < Policy::TileIterations::kRow; ++tile_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < EleShapePerPatial::kRow; ++m) {
          int accum_m = tile_m * Policy::InterleavedTile::kRow +
              mma_m * QuadShapePerPatialMma::kRow + m * 2 + lane_offset.row();
          beginRow(accum_m);

          CUTLASS_PRAGMA_UNROLL
          for (int tile_n = 0; tile_n < Policy::TileIterations::kColumn;
               ++tile_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn;
                 ++mma_n) {
              CUTLASS_PRAGMA_UNROLL
              for (int p = 0; p < kAccumulatorPatials; ++p) {
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < EleShapePerPatial::kColumn; ++n) {
                  int mma_accum_start =
                      (((tile_n * Policy::TileIterations::kRow + tile_m) *
                            Policy::MmaIterations::kColumn +
                        mma_n) *
                           Policy::MmaIterations::kRow +
                       mma_m) *
                      kElementsPerMma;
                  int accum_n = tile_n * Policy::InterleavedTile::kColumn +
                      mma_n * QuadShapePerPatialMma::kColumn +
                      p * Policy::InterleavedTile::kColumn / 2 + n +
                      lane_offset.column();
                  int idx = mma_accum_start + p * kElementsPerPartial +
                      m * EleShapePerPatial::kColumn + n;
                  op(accum_m, accum_n, idx);
                }
              }
            }
          }
          endRow(accum_m);
        }
      }
    }
  }
};

template <typename T, typename accum_t, int kWarpSize>
struct AttentionScalingCoefsUpdaterSimt
    : RegisterOps<
          AttentionScalingCoefsUpdaterSimt<T, accum_t, kWarpSize>,
          T,
          accum_t,
          kWarpSize> {
  using Policy = typename T::Policy;
  using Iterations = typename T::Iterations;
  using Element = typename T::Element;
  using Delta = typename T::Delta;
  using Shape = typename T::Shape;
  static_assert(
      std::is_same<typename T::Layout, cutlass::layout::RowMajor>::value, "");
  static_assert(
      std::is_same<typename T::Iterations, typename T::Iterations>::value, "");

  template <typename DT, typename F>
  CUTLASS_DEVICE static bool reduceSameRow(int lane_id, DT& myValue, F fn) {
    CUTLASS_PRAGMA_UNROLL
    for (int bit = 1; bit < Policy::WarpShape::kColumn; bit *= 2) {
      auto otherV = __shfl_xor_sync(0xffffffff, myValue, bit);
      myValue = fn(myValue, otherV);
    }
    return (lane_id & (Policy::WarpShape::kColumn - 1)) == 0;
  }

  template <typename FA, typename FB, typename FC>
  CUTLASS_DEVICE static void iterateRows(
      cutlass::MatrixCoord& lane_offset,
      FA beginRow,
      FB op,
      FC endRow) {
    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {
        int accum_m = mma_m * Delta::kRow + m + lane_offset.row();
        beginRow(accum_m);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
          int accum_n =
              mma_n * Policy::WarpShape::kColumn * Policy::LaneMmaShape::kN +
              lane_offset.column();
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < Policy::LaneMmaShape::kN; ++n) {
            int idx = n +
                Policy::LaneMmaShape::kN *
                    (mma_n +
                     Iterations::kColumn *
                         (m + mma_m * Policy::LaneMmaShape::kM));
            op(accum_m, accum_n + n, idx);
          }
        }
        endRow(accum_m);
      }
    }
  }

  static cutlass::MatrixCoord CUTLASS_DEVICE get_lane_offset(
      int8_t lane_id,
      int8_t warp_id,
      typename T::TensorCoord const& tile_offset) {
    static_assert(std::is_same<
                  typename Policy::LaneLayout,
                  cutlass::layout::RowMajorInterleaved<1>>::value, "");
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    cutlass::MatrixCoord lane_offset = lane_layout.inverse(lane_id) *
        cutlass::MatrixCoord(Policy::LaneMmaShape::kM,
                             Policy::LaneMmaShape::kN);
    return lane_offset +
        tile_offset * cutlass::MatrixCoord(Shape::kRow, Shape::kColumn);
  }
};

template <typename T, typename accum_t, int kWarpSize>
struct DefaultAttentionScalingCoefsUpdater;

// Simt
template <typename S, typename P, typename accum_t, int kWarpSize>
struct DefaultAttentionScalingCoefsUpdater<
    cutlass::gemm::warp::MmaSimtTileIterator<
        S,
        cutlass::gemm::Operand::kC,
        accum_t,
        cutlass::layout::RowMajor,
        P,
        1,
        1>,
    accum_t,
    kWarpSize> {
  using Iterator = typename cutlass::gemm::warp::MmaSimtTileIterator<
      S,
      cutlass::gemm::Operand::kC,
      accum_t,
      cutlass::layout::RowMajor,
      P,
      1,
      1>;
  using Updater =
      AttentionScalingCoefsUpdaterSimt<Iterator, accum_t, kWarpSize>;
};

// TensorOp - Volta
template <typename S1, typename S2, typename accum_t, int kWarpSize>
struct DefaultAttentionScalingCoefsUpdater<
    cutlass::gemm::warp::MmaVoltaTensorOpAccumulatorTileIterator<
        S1,
        accum_t,
        cutlass::layout::RowMajor,
        S2,
        cutlass::MatrixShape<1, 1>>,
    accum_t,
    kWarpSize> {
  using Iterator =
      typename cutlass::gemm::warp::MmaVoltaTensorOpAccumulatorTileIterator<
          S1,
          accum_t,
          cutlass::layout::RowMajor,
          S2,
          cutlass::MatrixShape<1, 1>>;
  using Updater =
      AttentionScalingCoefsUpdaterVolta<Iterator, accum_t, kWarpSize>;
};

// TensorOp - Sm75+
template <
    typename S1,
    typename S2,
    typename S3,
    typename accum_t,
    int kWarpSize>
struct DefaultAttentionScalingCoefsUpdater<
    cutlass::gemm::warp::MmaTensorOpAccumulatorTileIterator<
        S1,
        accum_t,
        cutlass::layout::RowMajor,
        S2,
        S3>,
    accum_t,
    kWarpSize> {
  using Iterator =
      typename cutlass::gemm::warp::MmaTensorOpAccumulatorTileIterator<
          S1,
          accum_t,
          cutlass::layout::RowMajor,
          S2,
          S3>;
  using Updater =
      AttentionScalingCoefsUpdaterSm80<Iterator, accum_t, kWarpSize>;
};
