#pragma once
// Host-side entry point for the hand-written Metal GEMM kernels (gemm_*.h),
// shared by mm/addmm/bmm/baddbmm/addbmm (LinearAlgebra.mm), addmv (Blas.mm) and
// linear (Linear.mm). Replaces the MPSGraph matmul path for float/half/bfloat.
#include <ATen/core/Tensor.h>
#include <ATen/native/mps/kernels/gemm_common.h>
#include <c10/core/Scalar.h>
#include <optional>

namespace at::native::mps {

// True when the current device + OS support the M5 tensor unit (mpp matmul2d):
// macOS 26.4+ AND Apple10 (M5) GPU family. When false, all float GEMM is served
// by the Metal-3 simd/gemv kernels, so the MPSGraph path can be dropped fully.
bool gemm_use_tensor_unit();

// True for dtypes the native real/integer GEMM kernels handle directly
// (float/half/bfloat + signed/unsigned integers). Complex is handled separately
// by mps_gemm_complex (decomposed into real GEMMs).
bool gemm_supported_dtype(c10::ScalarType dt);

// Computes `out = epilogue(A @ B)` on MPS with the hand-written Metal kernels.
//
//   A:   (M, K) or (B, M, K)
//   B:   (K, N) or (B, K, N)
//   out: (M, N) or (B, M, N)   (preallocated, row-major in its last dim)
//   self: epilogue addend, broadcastable to `out` (ignored when epi == None)
//
// `epi == AlphaBeta` computes `alpha * (A @ B) + beta * self`. A/B may be
// arbitrary strided / transposed views; non-resolvable layouts are made
// contiguous. dtype must satisfy gemm_supported_dtype().
//
// fp32 defaults to TF32-relaxed tensor-unit math; force_precise_fp32 forces full
// fp32 (used by the complex decomposition to preserve complex64 precision).
void mps_gemm(
    const Tensor& A,
    const Tensor& B,
    const Tensor& out,
    const std::optional<Tensor>& self,
    const Scalar& alpha,
    const Scalar& beta,
    at_gemm::GemmEpilogue epi,
    bool force_precise_fp32 = false);

// Complex GEMM (complex64 / complex32): decomposes A @ B into four real GEMMs on
// the real/imag planes, recombines, and applies the alpha/beta epilogue
// host-side. Same operand/shape contract as mps_gemm; handles 2-D and batched
// 3-D. complex64 sub-GEMMs run at full fp32 precision.
void mps_gemm_complex(
    const Tensor& A,
    const Tensor& B,
    const Tensor& out,
    const std::optional<Tensor>& self,
    const Scalar& alpha,
    const Scalar& beta,
    at_gemm::GemmEpilogue epi);

} // namespace at::native::mps
