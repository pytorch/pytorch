#pragma once

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>

#if defined(CUDART_VERSION) && defined(CUSOLVER_VERSION)
#define USE_CUSOLVER
#endif

// cusolverDn<T>potrfBatched may have numerical issue before cuda 11.3 release,
// (which is cusolver version 11101 in the header), so we only use cusolver potrf batched
// if cuda version is >= 11.3
#if CUSOLVER_VERSION >= 11101
  constexpr bool use_cusolver_potrf_batched_ = true;
#else
  constexpr bool use_cusolver_potrf_batched_ = false;
#endif

// cusolverDn<T>syevjBatched may have numerical issue before cuda 11.3.1 release,
// (which is cusolver version 11102 in the header), so we only use cusolver syevj batched
// if cuda version is >= 11.3.1
// See https://github.com/pytorch/pytorch/pull/53040#issuecomment-793626268 and https://github.com/cupy/cupy/issues/4847
#if CUSOLVER_VERSION >= 11102
  constexpr bool use_cusolver_syevj_batched_ = true;
#else
  constexpr bool use_cusolver_syevj_batched_ = false;
#endif


namespace at {
namespace native {

void geqrf_batched_cublas(const Tensor& input, const Tensor& tau);
void triangular_solve_cublas(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular);
void triangular_solve_batched_cublas(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular);
void gels_batched_cublas(const Tensor& a, Tensor& b, Tensor& infos);
void lu_solve_batched_cublas(const Tensor& b, const Tensor& lu, const Tensor& pivots, TransposeType transpose);

#ifdef USE_CUSOLVER

// entrance of calculations of `inverse` using cusolver getrf + getrs, cublas getrfBatched + getriBatched
Tensor _inverse_helper_cuda_lib(const Tensor& self);
Tensor& _linalg_inv_out_helper_cuda_lib(Tensor& result, Tensor& infos_getrf, Tensor& infos_getrs);

// entrance of calculations of `svd` using cusolver gesvdj and gesvdjBatched
void svd_cusolver(const Tensor& A, const bool full_matrices, const bool compute_uv, const Tensor& U, const Tensor& S, const Tensor& V, const Tensor& info);

// entrance of calculations of `cholesky` using cusolver potrf and potrfBatched
void cholesky_helper_cusolver(const Tensor& input, bool upper, const Tensor& info);
Tensor _cholesky_solve_helper_cuda_cusolver(const Tensor& self, const Tensor& A, bool upper);
Tensor& cholesky_inverse_kernel_impl_cusolver(Tensor &result, Tensor& infos, bool upper);

void geqrf_cusolver(const Tensor& input, const Tensor& tau);
void ormqr_cusolver(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose);
Tensor& orgqr_helper_cusolver(Tensor& result, const Tensor& tau);

void linalg_eigh_cusolver(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors);
void lu_solve_looped_cusolver(const Tensor& b, const Tensor& lu, const Tensor& pivots, TransposeType transpose);

void lu_factor_looped_cusolver(const Tensor& self, const Tensor& pivots, const Tensor& infos, bool get_pivots, const bool use_magma_);

#endif  // USE_CUSOLVER

}}  // namespace at::native
