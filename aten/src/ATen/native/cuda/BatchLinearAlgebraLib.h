#pragma once

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>

#if defined(CUDART_VERSION) && defined(CUSOLVER_VERSION) && CUSOLVER_VERSION >= 10200
// some cusolver functions don't work well on cuda 9.2 or cuda 10.1.105, cusolver is used on cuda >= 10.1.243
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

namespace at {
namespace native {

void geqrf_batched_cublas(const Tensor& input, const Tensor& tau);

void triangular_solve_cublas(Tensor& A, Tensor& B, Tensor& infos, bool upper, bool transpose, bool conjugate_transpose, bool unitriangular);
void triangular_solve_batched_cublas(Tensor& A, Tensor& B, Tensor& infos, bool upper, bool transpose, bool conjugate_transpose, bool unitriangular);
void gels_batched_cublas(const Tensor& a, Tensor& b, Tensor& infos);

#ifdef USE_CUSOLVER

// entrance of calculations of `inverse` using cusolver getrf + getrs, cublas getrfBatched + getriBatched
Tensor _inverse_helper_cuda_lib(const Tensor& self);
Tensor& _linalg_inv_out_helper_cuda_lib(Tensor& result, Tensor& infos_getrf, Tensor& infos_getrs);

// entrance of calculations of `svd` using cusolver gesvdj and gesvdjBatched
std::tuple<Tensor, Tensor, Tensor> _svd_helper_cuda_lib(const Tensor& self, bool some, bool compute_uv);

// entrance of calculations of `cholesky` using cusolver potrf and potrfBatched
void cholesky_helper_cusolver(const Tensor& input, bool upper, const Tensor& info);
Tensor _cholesky_solve_helper_cuda_cusolver(const Tensor& self, const Tensor& A, bool upper);
Tensor& cholesky_inverse_kernel_impl_cusolver(Tensor &result, Tensor& infos, bool upper);

void geqrf_cusolver(const Tensor& input, const Tensor& tau);
void ormqr_cusolver(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose);
Tensor& orgqr_helper_cusolver(Tensor& result, const Tensor& tau);

void linalg_eigh_cusolver(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, bool upper, bool compute_eigenvectors);

#endif  // USE_CUSOLVER

}}  // namespace at::native
