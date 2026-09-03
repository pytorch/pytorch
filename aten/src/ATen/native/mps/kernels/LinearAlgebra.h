#pragma once
#include <c10/metal/common.h>

template <unsigned N = c10::metal::max_ndim>
struct OrgqrParams {
  int32_t num_batch_dims;

  uint32_t m;
  uint32_t m2;
  uint32_t n;
  uint32_t k;

  ::c10::metal::array<uint32_t, N> A_strides;
  ::c10::metal::array<uint32_t, N> tau_strides;
  ::c10::metal::array<uint32_t, N> H_strides;
  ::c10::metal::array<uint32_t, N> H_sizes;
};

struct UnpackPivotsParams {
  uint32_t perm_batch_stride;
  uint32_t pivots_batch_stride;
  uint32_t dim_size;
};

template <unsigned N = c10::metal::max_ndim>
struct GeqrfParams {
  int32_t num_batch_dims;

  ::c10::metal::array<uint32_t, N> A_sizes;
  ::c10::metal::array<uint32_t, N> A_strides;
  ::c10::metal::array<uint32_t, N> tau_strides;
};

enum class GemmEpilogue : int { None = 0, Bias = 1 };

// n - output length
// ld - matrix row stride
// ms - matrix stride along the reduction/output dimension
// xs - vector stride
// bias_r/bias_c - row/col strides of the bias (for addmm)
// 64-bit so huge operands fit; kernels narrow to their IDX template width.
struct GemvDims {
  int64_t n, K, ld, ms, xs;
  int64_t bias_r, bias_c;
};

struct SvdParams {
  uint32_t m; // staged rows = max(orig m,n) >= n
  uint32_t n; // staged cols = k = min(orig m,n)
  uint32_t max_sweeps;
  uint32_t compute_uv;
  float tol;
  uint32_t u_ld;
  uint32_t u_bstride;
  uint32_t v_ld;
  uint32_t v_bstride;
  uint32_t transposed; // 1 if SVD ran on A^T (left/right vectors swap targets)
  uint32_t stage_v; // 1: V accumulator in threadgroup mem (Vtg); 0: device mem
                    // (Vacc)
};

struct EighParams {
  uint32_t n;
  uint32_t max_sweeps;
  uint32_t compute_v;
  uint32_t upper; // UPLO: 1 read upper triangle, 0 read lower
  float tol;
};

// Per-thread small-matrix LU kernels: up to these sizes one thread
// factors/inverts a whole matrix (solves a whole right-hand-side column) from
// registers. LinearAlgebra.metal instantiates NMAX buckets {4, 8} for
// luFactorSmall, exact sizes 1..8 for luInvSmall and {4, 8, 16} for
// luSolveSmall, and static_asserts those lists against these constants.
C10_METAL_CONSTEXPR int64_t kLUSmallFactorMax = 8;
C10_METAL_CONSTEXPR int64_t kLUSmallSolveMax = 16;

// Strides are in elements; for the adjoint solve the host swaps
// LU_rstride/LU_cstride so the kernel walks LU^T and only has to conjugate.
template <typename index_t = int64_t>
struct LUSmallFactorParams {
  index_t A_bstride;
  index_t A_rstride;
  index_t A_cstride;
  index_t LU_bstride;
  index_t LU_rstride;
  index_t LU_cstride;
  uint32_t batch;
  uint32_t m;
  uint32_t n;
};

template <typename index_t = int64_t>
struct LUSmallInvParams {
  index_t A_bstride;
  index_t A_rstride;
  index_t A_cstride;
  index_t X_bstride;
  index_t X_rstride;
  index_t X_cstride;
  uint32_t batch;
};

template <typename index_t = int64_t>
struct LUSmallSolveParams {
  index_t LU_bstride;
  index_t LU_rstride;
  index_t LU_cstride;
  index_t X_bstride;
  index_t X_rstride;
  index_t X_cstride;
  uint32_t batch;
  uint32_t n;
  uint32_t k;
  bool adjoint;
};

// for LU streaming-panel kernels
C10_METAL_CONSTEXPR unsigned kLUStreamNT = 256;
C10_METAL_CONSTEXPR unsigned kLUStreamWarpsPerTG =
    kLUStreamNT / c10::metal::simdgroup_size;

// Per-batch streaming LU scratch: argmax value partials (float magnitudes),
// argmax index partials (uint), then the U row in the element type. Shared
// host/device so the host allocates B * sizeof(LUStreamScratch<T>) bytes and
// binds it untyped; the kernel indexes scratch[batch] and the compiler owns the
// stride. T is float or c10::metal::complex<float> (float2 on Metal).
template <typename T>
struct LUStreamScratch {
  ::c10::metal::array<float, kLUStreamNT> vpart;
  ::c10::metal::array<uint32_t, kLUStreamNT> ipart;
  ::c10::metal::array<T, c10::metal::simdgroup_size> uRow;
};
