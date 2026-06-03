#pragma once
// Shared definitions for the hand-written MPS GEMM kernels (gemm_*.h).
//
// The enums and POD dim structs below are valid in BOTH Metal and host C++, so
// the host dispatcher (operations/GemmMetal.*) and the shaders agree on the
// constant-buffer layout. Metal-only helpers (vector load type, epilogue) are
// guarded by __METAL__.
//
// These kernels are a fully-templated port of the metalBLAS shaders: every
// variant (dtype, tile sizes, transpose, double-buffer, epilogue, batch) is a
// template parameter and every former preprocessor branch is `if constexpr`
// (via IF_CONSTEXPR). The only preprocessor gate is `#if __METAL_VERSION__ >=
// 400`, used by the tensor-unit (matmul2d) kernels.

#include <c10/metal/common.h>

namespace at_gemm {

// Epilogue applied in the GEMM store path.
//   None      : C = A @ B
//   AlphaBeta : C = alpha * (A @ B) + beta * self[r*self_r + c*self_c]
//               The linear-layer bias add is the special case self_r == 0
//               (row broadcast) with alpha == 1, beta == 1.
enum class GemmEpilogue : int {
  None = 0,
  AlphaBeta = 1,
};

// Dims/strides for the strided kernels (simd_gemm, m5_tensor_gemm, int_gemm).
// One constant buffer (one setBytes). All strides are ELEMENT strides; each
// device pointer is pre-offset to its tensor's first element by the host binding.
struct GemmDimsStrided {
  int M, N, K;
  int lda, ldb, ldc;
  int self_r, self_c; // epilogue addend strides (self_r == 0 => row broadcast)
  int swizzle_log; // threadgroup swizzle for L2 locality (0 = none)
  int batch_a, batch_b, batch_c, batch_self; // per-batch element strides
};

// Dims for the GEMV kernels (gemv_t / gemv_nt). `n` is the output length (cols for
// gemv_t, rows for gemv_nt); `ld` is the matrix row stride; `xs` the vector stride;
// self_r/self_c the epilogue addend strides (the kernel indexes self at its output
// position, so one of them is the broadcast step and the other is 0).
struct GemvDims {
  int n, K, ld, xs;
  int self_r, self_c;
};

// Dims for the split-K GEMM (deep-K, few-output-tile shapes). Inputs are packed
// (lda == K, ldb == N, ldc == N); `kchunk` is the per-chunk K length, applied as
// a runtime extent (the AOT build cannot template on the data-dependent K / G).
struct SplitKDims {
  int M, N, K, kchunk;
};

// Dims for the split-K reduction pass: `n` output elements, `planes` fp32 partials.
struct SplitKReduceDims {
  int n, planes;
};

// Dims for the 1x1-conv GEMM (very-thin-N). Packed inputs (lda == K, ldb == N).
struct ConvDims {
  int M, N, K;
};

} // namespace at_gemm

#ifdef __METAL__
#include <c10/metal/utils.h>
#include <metal_stdlib>

namespace at_gemm {

// 16-byte (sizeof(T)*VEC) aligned vector for coalesced device loads/stores.
// Replaces metalBLAS's per-kernel `struct alignas(...) VecF/VecT`.
template <typename T, int VEC>
struct alignas(sizeof(T) * VEC) GemmVec {
  T v[VEC];
};

// Apply the epilogue to one accumulated output element and cast to OUT_T.
// alpha/beta are runtime scalars; when EPI == AlphaBeta and beta == 0 we must
// not read `self` (it may be uninitialized / NaN) - matches addmm semantics.
template <GemmEpilogue EPI, typename OUT_T, typename ACC_T>
inline OUT_T apply_epilogue(
    ACC_T acc,
    int r,
    int c,
    device const OUT_T* self,
    int self_r,
    int self_c,
    ::c10::metal::opmath_t<OUT_T> alpha,
    ::c10::metal::opmath_t<OUT_T> beta) {
  using op_t = ::c10::metal::opmath_t<OUT_T>;
  op_t v = static_cast<op_t>(acc);
  if IF_CONSTEXPR (EPI == GemmEpilogue::AlphaBeta) {
    v = alpha * v;
    if (beta != op_t(0)) {
      v += beta * static_cast<op_t>(self[r * self_r + c * self_c]);
    }
  }
  return static_cast<OUT_T>(v);
}

} // namespace at_gemm
#endif // __METAL__
