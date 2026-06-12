#pragma once
// Host-side entry point for the hand-written Metal GEMM kernels (gemm_*.h),
// shared by mm/addmm/bmm/baddbmm/addbmm (LinearAlgebra.mm), addmv (Blas.mm) and
// linear (Linear.mm). Replaces the MPSGraph matmul path for float/half/bfloat.
#include <ATen/core/Tensor.h>
#include <ATen/native/mps/kernels/gemm_common.h>
#include <c10/core/Scalar.h>
#include <optional>

namespace at::native::mps {

// True when matmul2d (MetalPerformancePrimitives) is available: macOS 26.2+, no
// GPU-family gate (it lowers to the NAX matrix unit where present, else simdgroup).
// When false, all float GEMM falls to the Metal-3 simd/gemv kernels.
bool gemm_use_mpp();

// True for dtypes the native real/integer kernels handle directly (float/half/
// bfloat + signed/unsigned integers). Complex goes through mps_gemm_complex
// (decomposed into real GEMMs).
bool gemm_supported_dtype(c10::ScalarType dt);

// out = epilogue(A @ B), shapes (M,K)@(K,N)->(M,N), optionally batched. `out` is
// preallocated row-major; AlphaBeta computes alpha*(A@B)+beta*self. A/B may be
// strided/transposed (else made contiguous). force_precise_fp32 forces full fp32.
void mps_gemm(
    const Tensor& A,
    const Tensor& B,
    const Tensor& out,
    const std::optional<Tensor>& self,
    const Scalar& alpha,
    const Scalar& beta,
    at_gemm::GemmEpilogue epi,
    bool force_precise_fp32 = false);

// Complex GEMM (complex64 / complex32): decomposes A @ B into four real GEMMs on the
// real/imag planes, recombines, and applies the alpha/beta epilogue host-side. Same
// contract as mps_gemm (2-D and batched); complex64 sub-GEMMs run at full fp32.
void mps_gemm_complex(
    const Tensor& A,
    const Tensor& B,
    const Tensor& out,
    const std::optional<Tensor>& self,
    const Scalar& alpha,
    const Scalar& beta,
    at_gemm::GemmEpilogue epi);

} // namespace at::native::mps
